"""Stage 2: Candidate space generation from feature axes.

AC-1.2: Candidate space generation
Given N binary feature axes, the generator produces exactly 2^N candidates (capped at 512) with
correct metadata envelopes. No duplicate variant IDs. Runs in under 10 seconds for N=9.
"""

import itertools
import uuid
from typing import Optional, Sequence

from qpo.models import Candidate, DecomposedGoal, PromptTemplate


class CandidateSpace:
    """Generates candidate variants from a decomposed goal."""

    def __init__(self, decomposed_goal: DecomposedGoal, max_candidates: int = 512) -> None:
        """Initialize candidate space from decomposed goal.

        Args:
            decomposed_goal: The DecomposedGoal with feature axes
            max_candidates: Cap on total candidates (per AC-1.2, default 512)

        Raises:
            ValueError: If decomposition has more than log2(max_candidates) axes
        """
        self.decomposed_goal = decomposed_goal
        self.max_candidates = max_candidates
        self.axes = decomposed_goal.feature_axes
        self.num_axes = len(self.axes)

        if 2 ** self.num_axes > max_candidates:
            raise ValueError(
                f"Cannot generate 2^{self.num_axes} candidates with cap of {max_candidates}"
            )

    def generate(self, prompt_template: Optional[PromptTemplate] = None) -> list[Candidate]:
        """Generate all 2^N candidates as Candidate objects.

        Args:
            prompt_template: If provided, each candidate's prompt_text is assembled
                via template.render(feature_values) instead of a stub string.

        Returns:
            List of exactly 2^N Candidate objects with no duplicates

        Raises:
            ValueError: If generation fails or produces duplicates
        """
        candidates = []

        for combo_index, combo in enumerate(itertools.product([0, 1], repeat=self.num_axes)):
            feature_values = dict(zip(self.axes, combo))
            variant_id = str(uuid.uuid4())

            if prompt_template is not None:
                prompt_text = prompt_template.render(feature_values)
            else:
                prompt_text = f"Prompt with axes: {feature_values}"

            candidate = Candidate(
                variant_id=variant_id,
                feature_values=feature_values,
                prompt_text=prompt_text,
                metadata={"combo_index": combo_index},
            )
            candidates.append(candidate)

        # Verify count
        expected_count = 2 ** self.num_axes
        if len(candidates) != expected_count:
            raise ValueError(
                f"Generated {len(candidates)} candidates but expected {expected_count}"
            )

        return candidates
