"""Stage 1.5: Prompt template construction.

Makes a single LLM call per run to produce a PromptTemplate — a base instruction
plus per-axis modifier phrases. Each of the 2^N candidates then calls
template.render(feature_values) to get its actual prompt text without any
additional LLM calls.

This is what gives the QAOA meaningful signal: different axis combinations produce
structurally different prompts that a judge model can rank with real variance.
"""

import json
import logging
import re

import requests

from qpo.models import PromptTemplate
from qpo.pipeline.utils import post_with_retry

logger = logging.getLogger(__name__)

_BUILD_PROMPT = """\
You are a prompt engineering expert. Your job is to design a reusable prompt \
template for the following optimization task.

GOAL: {goal}

FEATURE AXES (binary, each independently on or off): {axes}

Produce a base prompt that accomplishes the goal cleanly, plus a modifier phrase \
for each axis. When an axis is ON (value=1), its modifier is appended to the base \
prompt. Modifiers must be:
- Independently meaningful (each changes something concrete about how the task is done)
- Non-contradictory (any combination of modifiers produces a coherent prompt)
- 1-3 sentences each

Return ONLY valid JSON in this exact format:
{{
  "base": "<core instruction that accomplishes the goal with no extras>",
  "modifiers": {{
    "<axis_name>": "<modifier text appended when this axis is ON>",
    ...one entry per axis...
  }}
}}\
"""

# Pattern-based fallback modifiers — used when Ollama is unavailable.
# Keys are substrings that may appear in axis names.
_PATTERN_MODIFIERS: list[tuple[str, str]] = [
    ("formal",        "Use precise, professional language throughout your response."),
    ("concis",        "Be concise. Omit all unnecessary words and explanation."),
    ("verbose",       "Explain your reasoning in full before giving the final answer."),
    ("example",       "Include at least one concrete worked example to illustrate your answer."),
    ("step",          "Work through the problem step by step before stating your conclusion."),
    ("chain",         "Show your chain of thought explicitly before giving the final answer."),
    ("schema",        "Strictly follow the output schema with no additional fields."),
    ("role",          "You are an expert assistant specialising in this domain."),
    ("error",         "If the input is ambiguous or malformed, state that clearly rather than guessing."),
    ("clarif",        "Ask a single clarifying question if the input is ambiguous before proceeding."),
    ("retry",         "If your first answer seems incorrect, revise it before finalising."),
    ("bullet",        "Present your response as a bulleted list where appropriate."),
    ("json",          "Return your answer as valid JSON only, with no surrounding prose."),
    ("brief",         "Limit your response to three sentences or fewer."),
    ("detail",        "Provide a detailed, comprehensive response covering all edge cases."),
    ("context",       "Consider the broader context and any implicit assumptions in the request."),
    ("verify",        "After answering, briefly verify that your response satisfies the goal."),
    ("structured",    "Use clear headings or sections to organise a complex response."),
    ("simple",        "Use plain language accessible to a non-expert reader."),
    ("technical",     "Use domain-specific technical terminology where appropriate."),
]


_ANTHROPIC_SYSTEM = """\
You are a prompt engineering expert. Your job is to design reusable prompt templates \
for optimization tasks.

When given a GOAL and a list of FEATURE AXES, produce a base prompt that accomplishes the \
goal cleanly, plus a modifier phrase for each axis. When an axis is ON (value=1), its \
modifier is appended to the base prompt. Modifiers must be:
- Independently meaningful (each changes something concrete about how the task is done)
- Non-contradictory (any combination of modifiers produces a coherent prompt)
- 1-3 sentences each

Return ONLY valid JSON in this exact format:
{
  "base": "<core instruction that accomplishes the goal with no extras>",
  "modifiers": {
    "<axis_name>": "<modifier text appended when this axis is ON>",
    ...one entry per axis...
  }
}\
"""


class PromptBuilder:
    """Builds a PromptTemplate from a goal and axis list with a single LLM call."""

    def __init__(
        self,
        ollama_endpoint: str = "http://localhost:11434",
        model: str = "mistral:7b",
        timeout_s: int = 180,
        backend: str = "ollama",
    ) -> None:
        self.ollama_endpoint = ollama_endpoint
        self.model = model
        self.timeout_s = timeout_s
        self.backend = backend

    def build_template(self, goal: str, axes: list[str]) -> PromptTemplate:
        """Build a PromptTemplate for the given goal and axes.

        Attempts an LLM call first; falls back to pattern-based stub on failure.

        Args:
            goal: Plain English optimization goal
            axes: List of axis names from the decomposer

        Returns:
            PromptTemplate with base prompt and per-axis modifiers
        """
        try:
            if self.backend == "bedrock":
                return self._build_via_anthropic(goal, axes)
            return self._build_via_llm(goal, axes)
        except Exception as exc:
            logger.warning("LLM template build failed (%s), using pattern fallback", exc)
            return self._pattern_template(goal, axes)

    def _build_via_anthropic(self, goal: str, axes: list[str]) -> PromptTemplate:
        from qpo.pipeline import anthropic_llm

        axes_str = ", ".join(axes)
        user_prompt = f"GOAL: {goal}\n\nFEATURE AXES: {axes_str}"
        text = anthropic_llm.call(
            model=self.model,
            system_prompt=_ANTHROPIC_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=2048,
            timeout=self.timeout_s,
        )
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON block in Anthropic template response: {text[:200]!r}")

        data = json.loads(match.group())
        base = data.get("base", "").strip()
        modifiers: dict[str, str] = data.get("modifiers", {})

        if not base:
            raise ValueError("Anthropic returned empty base prompt")

        for axis in axes:
            if axis not in modifiers:
                modifiers[axis] = self._pattern_modifier(axis)
                logger.debug("Anthropic missed axis %r — filled with pattern fallback", axis)

        logger.info("Built Anthropic prompt template: base=%d chars, %d modifiers", len(base), len(modifiers))
        return PromptTemplate(base=base, modifiers=modifiers)

    def _build_via_llm(self, goal: str, axes: list[str]) -> PromptTemplate:
        axes_str = ", ".join(axes)
        prompt = _BUILD_PROMPT.format(goal=goal, axes=axes_str)

        response = post_with_retry(
            f"{self.ollama_endpoint}/api/generate",
            json_body={"model": self.model, "prompt": prompt, "stream": False, "think": False},
            timeout=self.timeout_s,
        )
        text = response.json()["response"]

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON block in LLM template response: {text[:200]!r}")

        data = json.loads(match.group())
        base = data.get("base", "").strip()
        modifiers: dict[str, str] = data.get("modifiers", {})

        if not base:
            raise ValueError("LLM returned empty base prompt")

        # Fill in any axes the LLM missed with pattern fallback
        for axis in axes:
            if axis not in modifiers:
                modifiers[axis] = self._pattern_modifier(axis)
                logger.debug("LLM missed axis %r — filled with pattern fallback", axis)

        logger.info("Built LLM prompt template: base=%d chars, %d modifiers", len(base), len(modifiers))
        return PromptTemplate(base=base, modifiers=modifiers)

    def _pattern_template(self, goal: str, axes: list[str]) -> PromptTemplate:
        """Deterministic fallback — no LLM required."""
        base = f"Complete the following task accurately and completely:\n\n{goal}"
        modifiers = {axis: self._pattern_modifier(axis) for axis in axes}
        return PromptTemplate(base=base, modifiers=modifiers)

    def _pattern_modifier(self, axis: str) -> str:
        axis_lower = axis.lower().replace("_", " ")
        for pattern, modifier in _PATTERN_MODIFIERS:
            if pattern in axis_lower:
                return modifier
        # Generic fallback for unrecognised axis names
        label = axis.replace("_", " ").title()
        return f"Apply the {label!r} approach to this task."
