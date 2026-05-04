"""Domain models for quantum-assisted prompt optimization pipeline."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """Assembled prompt template for a run's candidate space.

    Built once per run by PromptBuilder from the decomposed goal and axis names.
    Each candidate calls render() to get its actual prompt text.
    """

    base: str = Field(..., description="Core instruction that achieves the goal without extras")
    modifiers: dict[str, str] = Field(
        default_factory=dict,
        description="axis_name → text appended to base when that axis is 1",
    )

    def render(self, feature_values: dict[str, int]) -> str:
        """Assemble a candidate's prompt from base + active modifiers.

        Args:
            feature_values: dict mapping axis names to 0 or 1

        Returns:
            Full prompt string with active modifiers appended
        """
        parts = [self.base]
        for axis, value in feature_values.items():
            if value == 1 and axis in self.modifiers:
                parts.append(self.modifiers[axis])
        return "\n\n".join(parts)


class Intent(BaseModel):
    """Plain English goal/objective for prompt optimization.

    Attributes:
        goal: Plain English description of the optimization target
        context: Optional additional context about the domain/task
        constraints: Optional list of constraints or requirements
    """
    goal: str = Field(..., description="Plain English optimization goal")
    context: str = Field(default="", description="Additional context")
    constraints: list[str] = Field(default_factory=list, description="Domain constraints")


class DecomposedGoal(BaseModel):
    """Result of decomposing an Intent into feature axes and success criteria.

    Attributes:
        feature_axes: List of binary feature axes (6-14 per AC-1.1)
        success_criteria: List of success criteria for evaluation
        axis_values: Dict mapping axis names to their binary values
    """
    feature_axes: list[str] = Field(..., description="Identified feature axes")
    success_criteria: list[str] = Field(..., description="Success criteria")
    axis_values: dict[str, int] = Field(default_factory=dict)


class Candidate(BaseModel):
    """A single prompt variant in the candidate space.

    Attributes:
        variant_id: Unique identifier for this variant
        feature_values: Dict mapping feature_axis -> binary value
        prompt_text: Generated or modified prompt text
        metadata: Additional metadata about the variant
    """
    variant_id: str = Field(..., description="Unique variant identifier")
    feature_values: dict[str, int] = Field(..., description="Feature axis assignments")
    prompt_text: str = Field(..., description="Generated prompt variant")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredCandidate(BaseModel):
    """Candidate with pre-scoring results from 7B model.

    Attributes:
        candidate: The base Candidate
        pre_score: Score from 7B judge (0-1 or raw)
        model: Which model produced this score
        latency_ms: Wall-clock scoring time
    """
    candidate: Candidate
    pre_score: float = Field(..., description="Pre-score from judge model")
    model: str = Field(default="7b", description="Model identifier")
    latency_ms: float = Field(default=0.0)


class EvalResult(BaseModel):
    """Deep evaluation result from 32B model.

    Attributes:
        candidate: The base Candidate
        score: Final evaluation score (0-1 or raw)
        reasoning: Optional explanation of the score
        model: Which model produced this evaluation
        latency_ms: Wall-clock evaluation time
    """
    candidate: Candidate
    score: float = Field(..., description="Final evaluation score")
    reasoning: str = Field(default="", description="Reasoning for the score")
    model: str = Field(default="32b", description="Model identifier")
    latency_ms: float = Field(default=0.0)


class PipelineRun(BaseModel):
    """Metadata for a complete pipeline run.

    Attributes:
        run_id: Unique run identifier
        intent: The original Intent
        decomposed_goal: The decomposed representation
        candidate_space_size: Total candidates generated (2^N)
        quantum_backend: Which quantum backend was used
        winning_variant: The best-scoring candidate found
        winning_score: Its final evaluation score
        total_evaluations: How many candidates were deep-evaluated
        total_latency_s: Total wall-clock time for the run
    """
    run_id: str
    intent: Intent
    decomposed_goal: DecomposedGoal
    candidate_space_size: int = Field(..., description="Total candidates (2^N)")
    quantum_backend: str = Field(default="stub", description="Quantum backend used")
    winning_variant: Candidate | None = None
    winning_score: float | None = None
    classical_winner_variant: Candidate | None = None
    classical_winner_score: float | None = None
    classical_overlap: int = 0
    total_evaluations: int = 0
    total_latency_s: float = 0.0
    eval_results: list[EvalResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
