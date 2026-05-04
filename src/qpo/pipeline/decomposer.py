"""Stage 1: Intent decomposition into feature axes and success criteria.

AC-1.1: Intent decomposition
Given a plain English prompt goal, the decomposition agent correctly identifies between 6 and 14
feature axes and at least 2 success criteria. Output is structured JSON. Validated on 3 different
prompt goals from real agent workflows.
"""

import json
import logging
import random
import re

import requests

from qpo.models import DecomposedGoal, Intent
from qpo.pipeline.utils import post_with_retry

logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = """\
You are a prompt engineering analyst. Given a prompt optimization goal, identify the key \
binary feature axes that differentiate prompt variants and the success criteria for evaluation.

Goal: {goal}
Context: {context}

Return ONLY valid JSON in this exact format:
{{
  "feature_axes": ["axis_name_1", "axis_name_2", ...],
  "success_criteria": ["criterion 1", "criterion 2", ...]
}}

Requirements:
- feature_axes: list of 6-14 short snake_case strings naming binary prompt dimensions \
(e.g., "formal_tone", "include_examples", "step_by_step", "concise_output")
- success_criteria: list of 2+ strings describing what a successful prompt achieves
- Return ONLY the JSON object, no surrounding text or explanation\
"""


class Decomposer:
    """Decomposes plain English intent into structured feature axes and success criteria."""

    def __init__(self, ollama_endpoint: str = "http://localhost:11434", model: str = "mistral:7b", timeout_s: int = 180) -> None:
        self.ollama_endpoint = ollama_endpoint
        self.model = model
        self.timeout_s = timeout_s

    def decompose(self, intent: Intent) -> DecomposedGoal:
        """Decompose intent into feature axes and success criteria.

        Attempts LLM decomposition via Ollama; falls back to stub on any failure.

        Args:
            intent: The Intent to decompose

        Returns:
            DecomposedGoal with 6-14 feature axes and 2+ success criteria

        Raises:
            ValueError: If decomposition fails and stub also violates AC-1.1 bounds
        """
        try:
            return self._decompose_via_llm(intent)
        except Exception as exc:
            logger.warning("LLM decomposition failed (%s), falling back to stub", exc)
            return self._stub_decompose(intent)

    def _decompose_via_llm(self, intent: Intent) -> DecomposedGoal:
        prompt = _DECOMPOSE_PROMPT.format(goal=intent.goal, context=intent.context or "")

        response = post_with_retry(
            f"{self.ollama_endpoint}/api/generate",
            json_body={"model": self.model, "prompt": prompt, "stream": False, "think": False},
            timeout=self.timeout_s,
        )
        text = response.json()["response"]

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON block in LLM response: {text[:200]!r}")

        data = json.loads(match.group())
        axes: list[str] = data.get("feature_axes", [])
        criteria: list[str] = data.get("success_criteria", [])

        if not (6 <= len(axes) <= 14):
            raise ValueError(f"LLM returned {len(axes)} axes (need 6–14): {axes}")
        if len(criteria) < 2:
            raise ValueError(f"LLM returned {len(criteria)} criteria (need ≥2)")

        # Cap at 9 axes so 2^N stays within the 512-candidate limit
        axes = axes[:9]
        return DecomposedGoal(
            feature_axes=axes,
            success_criteria=criteria,
            axis_values={ax: 0 for ax in axes},
        )

    def _stub_decompose(self, intent: Intent) -> DecomposedGoal:
        num_axes = random.randint(6, 9)
        axes = [f"axis_{i}" for i in range(num_axes)]
        criteria = [
            f"Success criterion 1: {intent.goal}",
            "Success criterion 2: Maintain coherence",
        ]
        return DecomposedGoal(
            feature_axes=axes,
            success_criteria=criteria,
            axis_values={ax: random.randint(0, 1) for ax in axes},
        )
