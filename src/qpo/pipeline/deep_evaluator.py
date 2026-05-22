"""Stage 5: Deep evaluation with 32B model.

Scores candidates with the 32B deep evaluation model (Ollama on M1 Pro in Phase 2,
Bedrock in Phase 3). Returns a score, reasoning string, and parse_failed flag per candidate.
"""

import logging
import re
import time
from typing import Sequence

import requests

from qpo.models import Candidate, EvalResult
from qpo.pipeline.utils import post_with_retry

logger = logging.getLogger(__name__)

_EVAL_PROMPT = """\
You are an expert prompt quality evaluator. Deeply evaluate this prompt variant.

Prompt text:
{prompt_text}

Feature configuration: {feature_summary}

Provide a thorough evaluation. Respond in exactly this format:
Score: <decimal 0.0-1.0>
Reasoning: <2-3 sentences explaining the score, covering clarity, effectiveness, and fit>\
"""


_ANTHROPIC_SYSTEM = """\
You are an expert prompt quality evaluator. When given a prompt variant and its feature \
configuration, provide a thorough evaluation. Respond in exactly this format:
Score: <decimal 0.0-1.0>
Reasoning: <2-3 sentences explaining the score, covering clarity, effectiveness, and fit>\
"""


class DeepEvaluator:
    """Deep evaluation of candidates using 32B model (Ollama or Anthropic)."""

    def __init__(
        self,
        ollama_endpoint: str = "http://192.168.1.100:11434",
        model: str = "qwen2.5:32b",
        timeout_s: int = 180,
        backend: str = "ollama",
    ) -> None:
        self.ollama_endpoint = ollama_endpoint
        self.model = model
        self.timeout_s = timeout_s
        self.backend = backend

    def evaluate(self, candidates: Sequence[Candidate]) -> list[EvalResult]:
        """Perform deep evaluation of candidates.

        Args:
            candidates: Candidates to evaluate

        Returns:
            List of EvalResult objects with final scores, reasoning, and parse_failed flags

        Raises:
            RuntimeError: If the LAN Ollama server is unreachable
        """
        results = []
        for candidate in candidates:
            start_time = time.time()
            score, reasoning, parse_failed = self._evaluate_candidate(candidate)
            latency_ms = (time.time() - start_time) * 1000
            results.append(
                EvalResult(
                    candidate=candidate,
                    score=score,
                    reasoning=reasoning,
                    parse_failed=parse_failed,
                    model=self.model,
                    latency_ms=latency_ms,
                )
            )
        return results

    def _evaluate_candidate(self, candidate: Candidate) -> tuple[float, str, bool]:
        """Return (score, reasoning, parse_failed)."""
        if self.backend == "bedrock":
            return self._evaluate_via_anthropic(candidate)
        return self._evaluate_via_ollama(candidate)

    def _evaluate_via_anthropic(self, candidate: Candidate) -> tuple[float, str, bool]:
        from qpo.pipeline import anthropic_llm

        feature_summary = ", ".join(f"{k}={v}" for k, v in candidate.feature_values.items())
        user_prompt = _EVAL_PROMPT.format(
            prompt_text=candidate.prompt_text,
            feature_summary=feature_summary,
        )
        try:
            text = anthropic_llm.call(
                model=self.model,
                system_prompt=_ANTHROPIC_SYSTEM,
                user_prompt=user_prompt,
                max_tokens=256,
                timeout=self.timeout_s,
            )
            return self._parse_eval_response(text)
        except Exception as exc:
            raise RuntimeError(f"Anthropic deep evaluator failed: {exc}") from exc

    def _evaluate_via_ollama(self, candidate: Candidate) -> tuple[float, str, bool]:
        feature_summary = ", ".join(
            f"{k}={v}" for k, v in candidate.feature_values.items()
        )
        prompt = _EVAL_PROMPT.format(
            prompt_text=candidate.prompt_text,
            feature_summary=feature_summary,
        )
        try:
            response = post_with_retry(
                f"{self.ollama_endpoint}/api/generate",
                json_body={"model": self.model, "prompt": prompt, "stream": False, "think": False},
                timeout=self.timeout_s,
            )
            text = response.json()["response"]
            return self._parse_eval_response(text)
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Deep evaluator Ollama unreachable at {self.ollama_endpoint}: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Deep evaluator request failed: {exc}") from exc

    def _parse_eval_response(self, text: str) -> tuple[float, str, bool]:
        """Parse score and reasoning from the structured LLM response.

        Returns:
            (score, reasoning, parse_failed) — parse_failed=True when no Score: line
            was found and the score defaulted to 0.5.
        """
        score_match = re.search(r"[Ss]core:\s*(\d+\.?\d*)", text)
        reason_match = re.search(r"[Rr]easoning:\s*(.+)", text, re.DOTALL)

        parse_failed = score_match is None
        if parse_failed:
            logger.warning("Could not parse score from deep-eval response: %r", text[:200])

        score = max(0.0, min(1.0, float(score_match.group(1)))) if score_match else 0.5
        reasoning = reason_match.group(1).strip() if reason_match else text[:300]

        return score, reasoning, parse_failed
