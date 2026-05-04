"""Main pipeline orchestration: 5-stage end-to-end flow.

AC-1.5: End-to-end run
A full pipeline run completes without manual intervention from intent input to ranked shortlist
output. Output includes: winning variant, its feature axis settings, and its eval score vs
the pre-score mean.
"""

import logging
import time
import uuid
from typing import Callable, Optional

import numpy as np

from qpo.config import get_config
from qpo.models import Candidate, EvalResult, Intent, PipelineRun, ScoredCandidate
from qpo.pipeline.candidate_generator import CandidateSpace
from qpo.pipeline.decomposer import Decomposer
from qpo.pipeline.deep_evaluator import DeepEvaluator
from qpo.pipeline.pre_scorer import PreScorer
from qpo.pipeline.prompt_builder import PromptBuilder
from qpo.quantum.optimizer import QuantumOptimizer

logger = logging.getLogger(__name__)


class Pipeline:
    """Five-stage pipeline orchestration.

    1. Decomposer: Intent → Feature axes + success criteria
    2. CandidateGenerator: Feature axes → 2^N variants
    3. PreScorer: Candidates → Pre-filtered shortlist (7B)
    4. QuantumOptimizer: Shortlist → QUBO → Quantum shortlist
    5. DeepEvaluator: Quantum shortlist → Final scores (32B)
    """

    def __init__(
        self,
        decomposer: Optional[Decomposer] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        pre_scorer: Optional[PreScorer] = None,
        quantum_optimizer: Optional[QuantumOptimizer] = None,
        deep_evaluator: Optional[DeepEvaluator] = None,
        max_candidates: int = 512,
        qaoa_prefilter_size: int = 20,
    ) -> None:
        """Initialize pipeline with components.

        Args:
            decomposer: Intent decomposer (default: new Decomposer)
            prompt_builder: Builds PromptTemplate from goal + axes (default: new PromptBuilder)
            pre_scorer: Pre-scoring model (default: new PreScorer)
            quantum_optimizer: Quantum optimization layer (default: new QuantumOptimizer)
            deep_evaluator: Deep evaluation model (default: new DeepEvaluator)
            max_candidates: Cap on candidate space (default 512)
            qaoa_prefilter_size: Top-N pre-scored candidates to pass to QAOA (limits qubits)
        """
        cfg = get_config()
        ep = cfg.ollama.local_7b_endpoint
        ep32 = cfg.ollama.remote_32b_endpoint
        m7b = cfg.ollama.local_7b_model
        m32b = cfg.ollama.remote_32b_model
        t = cfg.ollama.timeout_s
        self.decomposer = decomposer or Decomposer(ollama_endpoint=ep, model=m7b, timeout_s=t)
        self.prompt_builder = prompt_builder or PromptBuilder(ollama_endpoint=ep, model=m7b, timeout_s=t)
        self.pre_scorer = pre_scorer or PreScorer(ollama_endpoint=ep, model=m7b, timeout_s=t)
        self.quantum_optimizer = quantum_optimizer or QuantumOptimizer()
        self.deep_evaluator = deep_evaluator or DeepEvaluator(ollama_endpoint=ep32, model=m32b, timeout_s=t)
        self.max_candidates = max_candidates or cfg.pipeline.max_candidates
        self.qaoa_prefilter_size = qaoa_prefilter_size or cfg.pipeline.qaoa_prefilter_size

    def run(
        self,
        intent: Intent,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> PipelineRun:
        """Execute full pipeline end-to-end.

        Args:
            intent: The optimization goal
            on_event: Optional callback invoked with each pipeline event dict

        Returns:
            PipelineRun with results

        Raises:
            ValueError: If any stage fails
            RuntimeError: If model servers are unavailable
        """
        def emit(event: dict) -> None:
            if on_event:
                on_event(event)

        run_id = str(uuid.uuid4())
        start_time = time.time()

        def ts() -> str:
            from datetime import datetime
            return datetime.now().strftime("%H:%M:%S")

        logger.info(f"[{run_id}] Starting pipeline run: {intent.goal}")

        # Stage 1: Decompose intent
        emit({"type": "stage", "stage": 1, "name": "Decompose", "status": "active"})
        emit({"type": "log", "time": ts(), "level": "info", "message": "Decomposing intent..."})
        logger.info(f"[{run_id}] Stage 1: Decomposing intent...")
        decomposed_goal = self.decomposer.decompose(intent)
        n_axes = len(decomposed_goal.feature_axes)
        n_criteria = len(decomposed_goal.success_criteria)
        logger.info(f"[{run_id}] → Identified {n_axes} axes, {n_criteria} criteria")
        emit({"type": "stage", "stage": 1, "name": "Decompose", "status": "done"})
        emit({"type": "log", "time": ts(), "level": "ok",
              "message": f"Decomposed into {n_axes} feature axes, {n_criteria} criteria"})

        # Stage 1.5: Build prompt template (one LLM call, then deterministic assembly)
        emit({"type": "stage_extra", "name": "Build template", "status": "active"})
        emit({"type": "log", "time": ts(), "level": "info",
              "message": f"Building prompt template for {n_axes} axes..."})
        logger.info(f"[{run_id}] Stage 1.5: Building prompt template...")
        prompt_template = self.prompt_builder.build_template(
            intent.goal, decomposed_goal.feature_axes
        )
        logger.info(f"[{run_id}] → Template: base={len(prompt_template.base)} chars, "
                    f"{len(prompt_template.modifiers)} modifiers")
        emit({"type": "stage_extra", "name": "Build template", "status": "done"})
        emit({"type": "log", "time": ts(), "level": "ok",
              "message": f"Prompt template built — {len(prompt_template.modifiers)} axis modifiers"})

        # Stage 2: Generate candidate space
        emit({"type": "stage", "stage": 2, "name": "Generate", "status": "active"})
        emit({"type": "log", "time": ts(), "level": "info", "message": "Generating candidate space..."})
        logger.info(f"[{run_id}] Stage 2: Generating candidate space...")
        candidate_space = CandidateSpace(decomposed_goal, max_candidates=self.max_candidates)
        candidates = candidate_space.generate(prompt_template=prompt_template)
        logger.info(f"[{run_id}] → Generated {len(candidates)} candidates")
        emit({"type": "stage", "stage": 2, "name": "Generate", "status": "done"})
        emit({"type": "metric", "key": "candidates", "value": len(candidates)})
        emit({"type": "log", "time": ts(), "level": "ok",
              "message": f"Generated {len(candidates)} candidates — no duplicates"})

        # Stage 3: Pre-score with 7B
        emit({"type": "stage", "stage": 3, "name": "Pre-score", "status": "active"})
        emit({"type": "log", "time": ts(), "level": "info",
              "message": f"Pre-scoring {len(candidates)} candidates via {self.pre_scorer.model}..."})
        logger.info(f"[{run_id}] Stage 3: Pre-scoring with {self.pre_scorer.model}...")
        scored_candidates = self.pre_scorer.score_batch(candidates)
        pre_scores = [sc.pre_score for sc in scored_candidates]
        mean_pre_score = np.mean(pre_scores)
        logger.info(f"[{run_id}] → Pre-scores: mean={mean_pre_score:.3f}, "
                    f"min={min(pre_scores):.3f}, max={max(pre_scores):.3f}")
        emit({"type": "stage", "stage": 3, "name": "Pre-score", "status": "done"})
        emit({"type": "log", "time": ts(), "level": "ok",
              "message": f"Pre-scored {len(candidates)} candidates (mean={mean_pre_score:.3f})"})

        # Stage 4: Quantum optimization (QUBO → shortlist)
        emit({"type": "stage", "stage": 4, "name": "QAOA", "status": "active"})
        emit({"type": "log", "time": ts(), "level": "info",
              "message": "Building QUBO and running QAOA..."})
        logger.info(f"[{run_id}] Stage 4: Quantum optimization...")

        # Pre-filter: take top-N by pre-score to limit qubit count
        prefilter_n = min(self.qaoa_prefilter_size, len(scored_candidates))
        prefiltered = sorted(scored_candidates, key=lambda sc: sc.pre_score, reverse=True)[:prefilter_n]
        logger.info(
            f"[{run_id}] → Pre-filtered to top {len(prefiltered)} candidates "
            f"(from {len(scored_candidates)}) before QAOA"
        )
        emit({"type": "log", "time": ts(), "level": "info",
              "message": f"Pre-filtered to top {len(prefiltered)} candidates for QAOA"})

        # Wrap on_event to intercept QAOA iteration metrics
        qaoa_iteration = [0]

        def qaoa_aware_optimizer_run(qm, cids, ss):
            result = self.quantum_optimizer.qubo_to_shortlist(qm, cids, ss)
            return result

        # Build QUBO diagonal from actual pre-scores
        qubo_diag = np.array([sc.pre_score for sc in prefiltered])
        qubo_matrix = np.diag(qubo_diag)
        candidate_ids = [sc.candidate.variant_id for sc in prefiltered]

        # Patch logger temporarily to intercept QAOA iteration logs
        class _QAOALogHandler(logging.Handler):
            def emit(self_h, record: logging.LogRecord) -> None:
                msg = record.getMessage()
                if "iteration" in msg and "cost=" in msg:
                    import re
                    m_iter = re.search(r"iteration (\d+)", msg)
                    m_cost = re.search(r"cost=([\d.]+)", msg)
                    if m_iter and m_cost:
                        it = int(m_iter.group(1))
                        cost = float(m_cost.group(1))
                        emit({"type": "metric", "key": "iteration", "value": it})
                        emit({"type": "metric", "key": "cost", "value": round(cost, 4)})
                        emit({"type": "log", "time": ts(), "level": "info",
                              "message": f"QAOA iter {it} — cost {cost:.4f}"})

        qaoa_logger = logging.getLogger("qpo.quantum.optimizer")
        _handler = _QAOALogHandler()
        _handler.setLevel(logging.DEBUG)
        qaoa_logger.addHandler(_handler)
        qaoa_logger.setLevel(logging.DEBUG)

        try:
            quantum_shortlist_ids = self.quantum_optimizer.qubo_to_shortlist(
                qubo_matrix, candidate_ids, shortlist_size=10
            )
        finally:
            qaoa_logger.removeHandler(_handler)

        quantum_shortlist_ids_set = set(quantum_shortlist_ids)
        quantum_shortlist = [
            sc for sc in prefiltered if sc.candidate.variant_id in quantum_shortlist_ids_set
        ]
        logger.info(f"[{run_id}] → Quantum shortlist: {len(quantum_shortlist)} candidates")
        emit({"type": "stage", "stage": 4, "name": "QAOA", "status": "done"})
        emit({"type": "log", "time": ts(), "level": "ok",
              "message": f"QAOA shortlisted {len(quantum_shortlist)} candidates"})

        # Classical baseline: top-K by pre-score
        shortlist_size = 10
        classical_shortlist = sorted(prefiltered, key=lambda sc: -sc.pre_score)[:shortlist_size]
        classical_ids_set = {sc.candidate.variant_id for sc in classical_shortlist}
        overlap = len(quantum_shortlist_ids_set & classical_ids_set)

        # Stage 5: Deep evaluation — union of QAOA + classical shortlists (eval once)
        all_to_eval_ids = quantum_shortlist_ids_set | classical_ids_set
        all_to_eval = [
            sc for sc in prefiltered if sc.candidate.variant_id in all_to_eval_ids
        ]
        emit({"type": "stage", "stage": 5, "name": "Deep eval", "status": "active"})
        emit({"type": "log", "time": ts(), "level": "info",
              "message": f"Deep-evaluating {len(all_to_eval)} candidates via {self.deep_evaluator.model} (QAOA + classical union)..."})
        logger.info(f"[{run_id}] Stage 5: Deep evaluation with {self.deep_evaluator.model} ({len(all_to_eval)} candidates, {overlap} overlap)...")
        eval_results = self.deep_evaluator.evaluate([sc.candidate for sc in all_to_eval])
        scores_by_id = {er.candidate.variant_id: er for er in eval_results}

        qaoa_evals = [scores_by_id[vid] for vid in quantum_shortlist_ids_set if vid in scores_by_id]
        classical_evals = [scores_by_id[vid] for vid in classical_ids_set if vid in scores_by_id]

        final_scores = [er.score for er in qaoa_evals]
        logger.info(
            f"[{run_id}] → QAOA scores: mean={np.mean(final_scores):.3f}, "
            f"min={min(final_scores):.3f}, max={max(final_scores):.3f}"
        )
        emit({"type": "stage", "stage": 5, "name": "Deep eval", "status": "done"})
        emit({"type": "log", "time": ts(), "level": "ok",
              "message": f"Deep-evaluated {len(eval_results)} candidates ({overlap} shared between QAOA and classical)"})

        # Find QAOA winner
        best_result = max(qaoa_evals, key=lambda er: er.score)
        winning_variant = best_result.candidate
        winning_score = best_result.score

        # Find classical winner
        classical_best = max(classical_evals, key=lambda er: er.score)

        total_latency = time.time() - start_time
        logger.info(
            f"[{run_id}] Pipeline complete! "
            f"QAOA winner: {winning_variant.variant_id} (score={winning_score:.3f}) | "
            f"Classical winner: {classical_best.candidate.variant_id} (score={classical_best.score:.3f})"
        )

        run = PipelineRun(
            run_id=run_id,
            intent=intent,
            decomposed_goal=decomposed_goal,
            candidate_space_size=len(candidates),
            quantum_backend=self.quantum_optimizer.backend,
            winning_variant=winning_variant,
            winning_score=winning_score,
            total_evaluations=len(eval_results),
            total_latency_s=total_latency,
            eval_results=qaoa_evals,
            classical_winner_variant=classical_best.candidate,
            classical_winner_score=classical_best.score,
            classical_overlap=overlap,
            metadata={
                "mean_pre_score": float(mean_pre_score),
                "shortlist_size": len(quantum_shortlist),
                "prescorer_model": self.pre_scorer.model,
                "deepeval_model": self.deep_evaluator.model,
            },
        )
        emit({
            "type": "done",
            "run_id": run_id,
            "winning_score": winning_score,
            "total_latency_s": round(total_latency, 2),
        })
        return run
