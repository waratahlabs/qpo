"""Pytest configuration and fixtures for QPO tests."""

from typing import Sequence
from unittest.mock import MagicMock, patch

import pytest

from qpo.models import Candidate, EvalResult, Intent, ScoredCandidate
from qpo.pipeline.deep_evaluator import DeepEvaluator
from qpo.pipeline.pre_scorer import PreScorer


@pytest.fixture
def mock_intent() -> Intent:
    return Intent(
        goal="Optimize prompts for code generation with better error handling",
        context="Used in code-assist workflows",
        constraints=["Must maintain context window efficiency"],
    )


@pytest.fixture
def mock_candidates() -> list[Candidate]:
    candidates = []
    for i in range(10):
        candidates.append(
            Candidate(
                variant_id=f"variant-{i:03d}",
                feature_values={"clarity": i % 2, "brevity": (i // 2) % 2},
                prompt_text=f"Generated prompt variant {i}",
                metadata={"index": i},
            )
        )
    return candidates


@pytest.fixture
def mock_scored_candidates(mock_candidates) -> list[ScoredCandidate]:
    scored = []
    for i, cand in enumerate(mock_candidates):
        scored.append(
            ScoredCandidate(
                candidate=cand,
                pre_score=0.5 + (i * 0.02),
                model="7b",
                latency_ms=100.0,
            )
        )
    return scored


@pytest.fixture
def mock_eval_results(mock_scored_candidates) -> list[EvalResult]:
    results = []
    for i, scored in enumerate(mock_scored_candidates):
        results.append(
            EvalResult(
                candidate=scored.candidate,
                score=0.6 + (i * 0.02),
                reasoning=f"Evaluated variant {i}",
                model="32b",
                latency_ms=500.0,
            )
        )
    return results


@pytest.fixture
def mock_ollama_response():
    """Patch requests.post to return a canned Ollama response."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"response": "0.75", "done": True}
    with patch("requests.post", return_value=mock_resp) as mock:
        yield mock


class _StubPreScorer(PreScorer):
    """PreScorer that returns deterministic scores without HTTP calls (CI use only)."""

    def _score_candidate(self, candidate: Candidate) -> float:
        return 0.5 + (hash(candidate.variant_id) % 100) / 200.0


class _StubDeepEvaluator(DeepEvaluator):
    """DeepEvaluator that returns deterministic scores without HTTP calls (CI use only)."""

    def _evaluate_candidate(self, candidate: Candidate) -> tuple[float, str]:
        score = 0.6 + (hash(candidate.variant_id) % 100) / 250.0
        return score, f"Stub eval for {candidate.variant_id}"


@pytest.fixture
def stub_pre_scorer() -> PreScorer:
    return _StubPreScorer()


@pytest.fixture
def stub_deep_evaluator() -> DeepEvaluator:
    return _StubDeepEvaluator()
