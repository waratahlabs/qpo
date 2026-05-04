"""Five-stage classical pipeline for prompt optimization.

The pipeline consists of:
1. Decomposer: Intent → Feature axes + success criteria
2. CandidateGenerator: Feature axes → 2^N candidate variants
3. PreScorer: Candidates → Pre-filtered shortlist (7B model)
4. QuantumOptimizer: Shortlist → QUBO → Quantum shortlist
5. DeepEvaluator: Quantum shortlist → Final scores (32B model)
"""

from qpo.pipeline.candidate_generator import CandidateSpace
from qpo.pipeline.deep_evaluator import DeepEvaluator
from qpo.pipeline.decomposer import Decomposer
from qpo.pipeline.pipeline import Pipeline
from qpo.pipeline.pre_scorer import PreScorer

__all__ = [
    "Decomposer",
    "CandidateSpace",
    "PreScorer",
    "DeepEvaluator",
    "Pipeline",
]
