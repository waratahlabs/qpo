"""Quantum-assisted prompt optimization (QPO) pipeline.

A three-phase validation platform for quantum-assisted prompt optimization:

Phase 1: Local hardware cluster (Predator + M1 Pro)
- Classical pipeline skeleton with stubbed quantum layer
- Acceptance criteria: End-to-end runs without error

Phase 2: Offline quantum refinement
- Real QAOA circuits on Lightning.qubit CPU sim
- Validate quantum search quality vs classical baseline

Phase 3: AWS handoff
- Migration to Bedrock + Braket
- QPU smoke test and full validation
"""

__version__ = "0.1.0"

from qpo.models import Candidate, DecomposedGoal, Intent, PipelineRun
from qpo.pipeline import (
    CandidateSpace,
    Decomposer,
    DeepEvaluator,
    Pipeline,
    PreScorer,
)
from qpo.quantum import QuantumOptimizer

__all__ = [
    "Intent",
    "DecomposedGoal",
    "Candidate",
    "PipelineRun",
    "Decomposer",
    "CandidateSpace",
    "PreScorer",
    "QuantumOptimizer",
    "DeepEvaluator",
    "Pipeline",
]
