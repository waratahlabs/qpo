"""Co-occurrence matrix builder for Phase 2b off-diagonal QUBO.

Operates at candidate level — one qubit per candidate, matching the QUBO
shape the optimizer expects. Q_ij encodes how well candidate i and candidate j
complement each other, derived from historical run winners.

For each candidate pair (i, j), Q_ij is the mean score premium of historical
runs whose winning feature vector overlaps maximally with the union of i's and
j's active features, relative to the global baseline. Candidates whose joint
feature pattern resembles historically high-scoring winners get positive Q_ij.
"""

import logging
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30


def fetch_run_history(endpoint: str, limit: int = 500) -> list[dict[str, Any]]:
    url = f"{endpoint}/api/runs"
    resp = requests.get(url, params={"full": "true", "limit": limit}, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _feature_vec(feature_values: dict[str, int], axes: list[str]) -> np.ndarray:
    return np.array([feature_values.get(a, 0) for a in axes], dtype=float)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.dot(a, b)
    union = np.sum(np.clip(a + b, 0, 1))
    return float(intersection / union) if union > 0 else 0.0


def build_candidate_interaction_matrix(
    candidate_feature_vectors: list[dict[str, int]],
    runs: list[dict[str, Any]],
) -> np.ndarray:
    """Build NxN candidate interaction matrix from historical run winners.

    For each candidate pair (i, j), finds historical runs whose winning
    feature vector overlaps with the union of i and j's active features,
    and scores the pair by the mean score premium of those runs.

    Args:
        candidate_feature_vectors: feature_values dict per prefiltered candidate
        runs: Run records from /api/runs?full=true

    Returns:
        NxN float array, off-diagonal only, normalised to [-1, 1]
    """
    n = len(candidate_feature_vectors)

    # Collect all feature axes seen across candidates and history
    all_axes: set[str] = set()
    for fv in candidate_feature_vectors:
        all_axes.update(fv.keys())
    for run in runs:
        wv = run.get("result", {}).get("winning_variant")
        if wv:
            all_axes.update(wv.get("feature_values", {}).keys())
    axes = sorted(all_axes)

    if not axes:
        return np.zeros((n, n))

    # Candidate vectors in common axis space
    cand_vecs = [_feature_vec(fv, axes) for fv in candidate_feature_vectors]

    # Historical winners: (vector, score)
    history: list[tuple[np.ndarray, float]] = []
    scores_all: list[float] = []
    for run in runs:
        result = run.get("result", {})
        wv = result.get("winning_variant")
        score = result.get("winning_score")
        if wv and score is not None:
            history.append((_feature_vec(wv.get("feature_values", {}), axes), float(score)))
            scores_all.append(float(score))

    if not history:
        logger.warning("No usable historical runs for candidate interaction matrix")
        return np.zeros((n, n))

    baseline = float(np.mean(scores_all))
    logger.info(
        "Candidate interaction: %d candidates × %d historical runs, baseline=%.4f",
        n, len(history), baseline,
    )

    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Union of active features for candidate pair (i, j)
            union_vec = np.clip(cand_vecs[i] + cand_vecs[j], 0, 1)

            # Weight historical runs by Jaccard similarity to the union vector
            weighted_scores: list[tuple[float, float]] = []
            for hist_vec, hist_score in history:
                sim = _jaccard(union_vec, hist_vec)
                if sim > 0:
                    weighted_scores.append((sim, hist_score))

            if len(weighted_scores) < 2:
                continue

            weights = np.array([w for w, _ in weighted_scores])
            scores = np.array([s for _, s in weighted_scores])
            weighted_mean = float(np.average(scores, weights=weights))
            Q[i, j] = weighted_mean - baseline
            Q[j, i] = Q[i, j]

    max_abs = np.abs(Q).max()
    if max_abs > 0:
        Q /= max_abs

    non_zero = int(np.count_nonzero(Q)) // 2
    logger.info("Candidate interaction matrix: %d non-zero pairs", non_zero)
    return Q


def build_qubo_matrix(
    pre_scores: list[float],
    candidate_feature_vectors: list[dict[str, int]],
    runs: list[dict[str, Any]],
    cross_term_weight: float = 0.3,
) -> np.ndarray:
    """Build NxN QUBO matrix: diagonal pre-scores + off-diagonal candidate interactions.

    Args:
        pre_scores: Per-candidate pre-scores (diagonal), length N
        candidate_feature_vectors: feature_values dict per candidate, length N
        runs: Run history from /api/runs?full=true
        cross_term_weight: Scale factor for off-diagonal terms

    Returns:
        NxN QUBO matrix
    """
    n = len(pre_scores)
    Q = np.diag(pre_scores).astype(float)

    if len(runs) < 5:
        logger.info("Fewer than 5 historical runs — diagonal QUBO only")
        return Q

    cross = build_candidate_interaction_matrix(candidate_feature_vectors, runs)

    if cross.shape != (n, n):
        logger.warning(
            "Interaction matrix shape %s != expected (%d, %d) — diagonal QUBO only",
            cross.shape, n, n,
        )
        return Q

    Q += cross_term_weight * cross
    logger.info(
        "QUBO: diagonal + candidate interactions (weight=%.2f)", cross_term_weight
    )
    return Q
