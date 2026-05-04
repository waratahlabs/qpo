"""Stage 3: Pre-scoring with 7B model.

AC-1.3: Pre-scoring throughput
The 7B judge scores 50 candidates against 5 test inputs each in under 15 minutes on the Predator.
Scores are deterministic within ±5% across two consecutive runs on the same candidates.
"""

import logging
import re
import time
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


class PreScorer:
    """Scores candidates using the 7B pre-scorer model via Ollama."""

    def __init__(self, ollama_endpoint: str = "http://localhost:11434", model: str = "mistral:7b", timeout_s: int = 180) -> None:
        self.ollama_endpoint = ollama_endpoint
        self.model = model
        self.timeout_s = timeout_s

    def score_batch(
        self,
        candidates: Sequence[Candidate],
        test_inputs: Optional[Sequence[str]] = None,
    ) -> list[ScoredCandidate]:
        """Score a batch of candidates.

        Args:
            candidates: Candidates to score
            test_inputs: Unused in current implementation (reserved for Phase 2)

        Returns:
            List of ScoredCandidate objects with pre_score populated

        Raises:
            RuntimeError: If Ollama server is unreachable
        """
        scored = []
        for candidate in candidates:
            start_time = time.time()
            pre_score = self._score_candidate(candidate)
            latency_ms = (time.time() - start_time) * 1000
            scored.append(
                ScoredCandidate(
                    candidate=candidate,
                    pre_score=pre_score,
                    model=self.model,
                    latency_ms=latency_ms,
                )
            )
        return scored

    def _score_candidate(self, candidate: Candidate) -> float:
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

    def _parse_score(self, text: str) -> float:
        match = re.search(r"(\d+\.?\d*)", text.strip())
        if not match:
            logger.warning("Could not parse score from response: %r", text[:100])
            return 0.5
        return max(0.0, min(1.0, float(match.group(1))))
