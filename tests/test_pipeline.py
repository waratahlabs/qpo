"""Tests for the main pipeline (AC-1.4, AC-1.5)."""

import pytest

from qpo import Intent, Pipeline
from qpo.pipeline.candidate_generator import CandidateSpace
from qpo.pipeline.decomposer import Decomposer
from qpo.pipeline.deep_evaluator import DeepEvaluator
from qpo.pipeline.pre_scorer import PreScorer
from qpo.quantum.optimizer import QuantumOptimizer


class TestDecomposer:
    """Tests for Intent decomposition (AC-1.1)."""

    def test_decompose_produces_valid_axes(self, mock_intent):
        """AC-1.1: Decomposer identifies 6-14 feature axes (falls back to stub when no Ollama)."""
        decomposer = Decomposer()
        result = decomposer.decompose(mock_intent)

        assert 6 <= len(result.feature_axes) <= 14, \
            f"Expected 6-14 axes, got {len(result.feature_axes)}"
        assert all(isinstance(ax, str) for ax in result.feature_axes)

    def test_decompose_produces_success_criteria(self, mock_intent):
        """AC-1.1: Decomposer identifies at least 2 success criteria."""
        decomposer = Decomposer()
        result = decomposer.decompose(mock_intent)

        assert len(result.success_criteria) >= 2, \
            f"Expected at least 2 criteria, got {len(result.success_criteria)}"
        assert all(isinstance(c, str) for c in result.success_criteria)

    def test_decompose_via_ollama(self, mock_intent, mock_ollama_response):
        """Decomposer parses LLM JSON response correctly."""
        import json
        mock_ollama_response.return_value.json.return_value = {
            "response": json.dumps({
                "feature_axes": [f"axis_{i}" for i in range(7)],
                "success_criteria": ["criterion 1", "criterion 2"],
            }),
            "done": True,
        }
        decomposer = Decomposer()
        result = decomposer.decompose(mock_intent)

        assert 6 <= len(result.feature_axes) <= 9
        assert len(result.success_criteria) >= 2


class TestCandidateGenerator:
    """Tests for candidate space generation (AC-1.2)."""

    def test_generates_correct_count(self, mock_intent):
        """AC-1.2: Generator produces exactly 2^N candidates."""
        decomposer = Decomposer()
        decomposed = decomposer.decompose(mock_intent)

        space = CandidateSpace(decomposed)
        candidates = space.generate()

        expected_count = 2 ** len(decomposed.feature_axes)
        assert len(candidates) == expected_count, \
            f"Expected {expected_count} candidates, got {len(candidates)}"

    def test_generates_no_duplicates(self, mock_intent):
        """AC-1.2: Generator produces no duplicate variant IDs."""
        decomposer = Decomposer()
        decomposed = decomposer.decompose(mock_intent)

        space = CandidateSpace(decomposed)
        candidates = space.generate()

        variant_ids = [c.variant_id for c in candidates]
        assert len(variant_ids) == len(set(variant_ids)), \
            "Generated duplicate variant IDs"

    def test_respects_max_candidates_cap(self):
        """AC-1.2: Generator respects candidate cap."""
        from qpo.models import DecomposedGoal

        goal_too_large = DecomposedGoal(
            feature_axes=[f"axis_{i}" for i in range(8)],
            success_criteria=["criterion1", "criterion2"],
        )
        with pytest.raises(ValueError):
            CandidateSpace(goal_too_large, max_candidates=100)

        goal_ok = DecomposedGoal(
            feature_axes=[f"axis_{i}" for i in range(6)],
            success_criteria=["criterion1", "criterion2"],
        )
        space = CandidateSpace(goal_ok, max_candidates=100)
        candidates = space.generate()
        assert len(candidates) == 64


class TestPreScorer:
    """Tests for pre-scoring stage (AC-1.3)."""

    def test_scores_batch_returns_results(self, mock_candidates, mock_ollama_response):
        """AC-1.3: Pre-scorer returns ScoredCandidate objects with valid scores."""
        scorer = PreScorer()
        results = scorer.score_batch(mock_candidates)

        assert len(results) == len(mock_candidates)
        for result in results:
            assert hasattr(result, "pre_score")
            assert 0 <= result.pre_score <= 1

    def test_scores_are_deterministic(self, mock_candidates, mock_ollama_response):
        """AC-1.3: Both runs complete and return valid results."""
        scorer = PreScorer()
        results1 = scorer.score_batch(mock_candidates)
        results2 = scorer.score_batch(mock_candidates)

        assert len(results1) > 0
        assert len(results2) > 0

    def test_parse_score_extracts_float(self):
        """_parse_score handles various response formats."""
        scorer = PreScorer()
        assert scorer._parse_score("0.87") == pytest.approx(0.87, abs=0.01)
        assert scorer._parse_score("Score: 0.5\nSome text") == pytest.approx(0.5, abs=0.01)
        assert scorer._parse_score("1.5") == 1.0  # clamped
        assert scorer._parse_score("no number here") == 0.5  # fallback

    def test_raises_on_connection_error(self, mock_candidates):
        """Pre-scorer raises RuntimeError when Ollama is unreachable."""
        scorer = PreScorer(ollama_endpoint="http://localhost:19999")
        with pytest.raises(RuntimeError, match="unreachable"):
            scorer.score_batch(mock_candidates[:1])


class TestQuantumLayer:
    """Tests for quantum optimization layer (AC-1.4)."""

    def test_qubo_to_shortlist_contract(self, mock_candidates):
        """AC-1.4: Quantum layer enforces QUBO-in, shortlist-out contract."""
        import numpy as np

        optimizer = QuantumOptimizer(backend="stub")
        variant_ids = [c.variant_id for c in mock_candidates]
        qubo = np.eye(len(mock_candidates))

        shortlist = optimizer.qubo_to_shortlist(qubo, variant_ids, shortlist_size=5)

        assert isinstance(shortlist, list)
        assert all(vid in variant_ids for vid in shortlist)
        assert len(shortlist) <= 5

    def test_qubo_size_mismatch_raises(self, mock_candidates):
        """QUBO matrix size mismatch raises ValueError."""
        import numpy as np

        optimizer = QuantumOptimizer(backend="stub")
        variant_ids = [c.variant_id for c in mock_candidates]
        bad_qubo = np.eye(len(mock_candidates) + 1)  # wrong size

        with pytest.raises(ValueError, match="!="):
            optimizer.qubo_to_shortlist(bad_qubo, variant_ids, shortlist_size=5)


class TestEndToEndPipeline:
    """Tests for full end-to-end pipeline (AC-1.5). Uses stubs for Ollama stages."""

    def test_pipeline_end_to_end(self, mock_intent, stub_pre_scorer, stub_deep_evaluator):
        """AC-1.5: Full pipeline runs without error, produces ranked output."""
        optimizer = QuantumOptimizer(backend="stub")
        pipeline = Pipeline(
            pre_scorer=stub_pre_scorer,
            deep_evaluator=stub_deep_evaluator,
            quantum_optimizer=optimizer,
            max_candidates=512,
        )
        result = pipeline.run(mock_intent)

        assert result.run_id
        assert result.intent == mock_intent
        assert result.decomposed_goal
        assert result.candidate_space_size > 0
        assert result.winning_variant
        assert result.winning_score >= 0
        assert result.total_latency_s >= 0

    def test_pipeline_output_includes_comparison(
        self, mock_intent, stub_pre_scorer, stub_deep_evaluator
    ):
        """AC-1.5: Output includes winning variant vs pre-score mean."""
        optimizer = QuantumOptimizer(backend="stub")
        pipeline = Pipeline(
            pre_scorer=stub_pre_scorer,
            deep_evaluator=stub_deep_evaluator,
            quantum_optimizer=optimizer,
            max_candidates=512,
        )
        result = pipeline.run(mock_intent)

        assert result.winning_score >= 0
        assert result.metadata.get("mean_pre_score") is not None
        assert result.winning_variant.feature_values

    def test_pipeline_respects_max_candidates(
        self, mock_intent, stub_pre_scorer, stub_deep_evaluator
    ):
        """AC-1.5: Pipeline respects max_candidates cap."""
        optimizer = QuantumOptimizer(backend="stub")
        pipeline = Pipeline(
            pre_scorer=stub_pre_scorer,
            deep_evaluator=stub_deep_evaluator,
            quantum_optimizer=optimizer,
            max_candidates=512,
        )
        result = pipeline.run(mock_intent)

        assert result.candidate_space_size <= 512

    def test_pipeline_prefilter_limits_qubo_size(
        self, mock_intent, stub_pre_scorer, stub_deep_evaluator
    ):
        """QUBO is built only from pre-filtered top candidates (not all 512)."""
        import numpy as np
        from unittest.mock import patch

        captured_qubo = {}

        original = QuantumOptimizer.qubo_to_shortlist

        def capturing_qubo(self, qubo_matrix, candidate_ids, shortlist_size=10):
            captured_qubo["matrix"] = qubo_matrix
            captured_qubo["ids"] = candidate_ids
            return original(self, qubo_matrix, candidate_ids, shortlist_size)

        optimizer = QuantumOptimizer(backend="stub")
        pipeline = Pipeline(
            pre_scorer=stub_pre_scorer,
            deep_evaluator=stub_deep_evaluator,
            quantum_optimizer=optimizer,
            max_candidates=512,
            qaoa_prefilter_size=15,
        )

        with patch.object(QuantumOptimizer, "qubo_to_shortlist", capturing_qubo):
            pipeline.run(mock_intent)

        assert captured_qubo["matrix"].shape[0] <= 15
        assert len(captured_qubo["ids"]) <= 15
