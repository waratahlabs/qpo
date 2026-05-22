"""Stage 3: Pre-scoring with 7B model.

AC-1.3: Pre-scoring throughput
The 7B judge scores 50 candidates against 5 test inputs each in under 15 minutes on the Predator.
Scores are deterministic within ±5% across two consecutive runs on the same candidates.
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

import requests

from qpo.models import Candidate, ScoredCandidate
from qpo.pipeline.utils import post_with_retry

logger = logging.getLogger(__name__)

_SCORE_PROMPT = """\
Rate this prompt variant on a scale from 0.0 to 1.0 based on quality, clarity, and effectiveness.

Prompt variant:
{prompt_text}

Feature settings: {feature_values}

Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else.\
"""


_ANTHROPIC_SYSTEM = """\
You are a prompt quality evaluator. When given a prompt variant and its feature settings, \
rate it on a scale from 0.0 to 1.0 based on quality, clarity, and effectiveness.
Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else.\
"""


class PreScorer:
    """Scores candidates using the 7B pre-scorer model via Ollama or Anthropic."""

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

    def score_batch(
        self,
        candidates: Sequence[Candidate],
        test_inputs: Optional[Sequence[str]] = None,
        max_workers: int = 4,
    ) -> list[ScoredCandidate]:
        """Score a batch of candidates in parallel.

        Uses ThreadPoolExecutor for concurrent I/O-bound LLM calls. The degree of
        parallelism is capped at min(max_workers, len(candidates)) to avoid
        overwhelming the Ollama server.

        Args:
            candidates: Candidates to score
            test_inputs: Unused in current implementation (reserved for Phase 2)
            max_workers: Max concurrent scoring threads (default 4)

        Returns:
            List of ScoredCandidate objects with pre_score and parse_failed populated

        Raises:
            RuntimeError: If Ollama server is unreachable
        """
        if not candidates:
            return []

        def _score_one(candidate: Candidate) -> ScoredCandidate:
            start_time = time.time()
            score, parse_failed = self._score_candidate(candidate)
            latency_ms = (time.time() - start_time) * 1000
            return ScoredCandidate(
                candidate=candidate,
                pre_score=score,
                parse_failed=parse_failed,
                model=self.model,
                latency_ms=latency_ms,
            )

        n_workers = min(max_workers, len(candidates))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            return list(pool.map(_score_one, candidates))

    def _score_candidate(self, candidate: Candidate) -> tuple[float, bool]:
        """Return (score, parse_failed)."""
        if self.backend == "bedrock":
            return self._score_via_anthropic(candidate)
        return self._score_via_ollama(candidate)

    def _score_via_anthropic(self, candidate: Candidate) -> tuple[float, bool]:
        from qpo.pipeline import anthropic_llm

        user_prompt = _SCORE_PROMPT.format(
            prompt_text=candidate.prompt_text,
            feature_values=candidate.feature_values,
        )
        try:
            text = anthropic_llm.call(
                model=self.model,
                system_prompt=_ANTHROPIC_SYSTEM,
                user_prompt=user_prompt,
                max_tokens=16,
                timeout=self.timeout_s,
            )
            return self._parse_score(text)
        except Exception as exc:
            raise RuntimeError(f"Anthropic pre-scorer failed: {exc}") from exc

    def _score_via_ollama(self, candidate: Candidate) -> tuple[float, bool]:
        prompt = _SCORE_PROMPT.format(
            prompt_text=candidate.prompt_text,
            feature_values=candidate.feature_values,
        )
        try:
            response = post_with_retry(
                f"{self.ollama_endpoint}/api/generate",
                json_body={"model": self.model, "prompt": prompt, "stream": False, "think": False},
                timeout=self.timeout_s,
            )
            text = response.json()["response"]
            return self._parse_score(text)
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Ollama pre-scorer unreachable at {self.ollama_endpoint}: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama pre-scorer request failed: {exc}") from exc

    def _parse_score(self, text: str) -> tuple[float, bool]:
        """Parse a score from LLM response text.

        Returns:
            (score, parse_failed) — parse_failed=True means the score defaulted to 0.5
            because no valid number was found in the response.
        """
        match = re.search(r"(\d+\.?\d*)", text.strip())
        if not match:
            logger.warning("Could not parse score from response: %r", text[:100])
            return 0.5, True
        return max(0.0, min(1.0, float(match.group(1)))), False
