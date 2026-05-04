"""QPO web server — Flask backend for the test harness UI.

Routes:
  GET  /                        → serve UI
  POST /api/runs                → start a pipeline run
  GET  /api/runs/<id>/events    → SSE stream of run events
  GET  /api/runs/<id>           → completed run result (JSON)
  GET  /api/runs                → run history list
  GET  /api/status              → backend health check
"""

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Generator

import requests
from flask import Flask, Response, jsonify, request

from qpo import Intent, Pipeline
from qpo.config import get_config
from qpo.db import (get_batch, get_run, init_db, list_batches, list_runs,
                    save_batch, save_run, update_batch)
from qpo.quantum.optimizer import QuantumOptimizer

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=None)
app.config["JSON_SORT_KEYS"] = False

# Per-run event queues: run_id → Queue[dict | None]
_run_queues: dict[str, queue.Queue] = {}
_run_queues_lock = threading.Lock()

_HTML_PATH = Path(__file__).parent.parent.parent / "ui.html"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.route("/")
def index() -> Response:
    if _HTML_PATH.exists():
        return Response(_HTML_PATH.read_text(), mimetype="text/html")
    return Response("<h1>QPO</h1><p>ui.html not found.</p>", mimetype="text/html")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.route("/api/runs", methods=["POST"])
def start_run() -> Response:
    body = request.get_json(force=True) or {}
    goal = body.get("goal", "").strip()
    if not goal:
        return jsonify({"error": "goal is required"}), 400

    backend = body.get("quantum_backend", "stub")
    pre_score_model = body.get("pre_score_model", "7b-local")
    deep_eval_model = body.get("deep_eval_model", "32b-remote")

    config = get_config()
    optimizer = QuantumOptimizer(
        backend=backend,
        circuit_depth=config.quantum.circuit_depth,
        num_iterations=config.quantum.num_iterations,
        seed=config.quantum.seed,
    )
    pipeline = Pipeline(
        quantum_optimizer=optimizer,
        max_candidates=config.pipeline.max_candidates,
        qaoa_prefilter_size=config.pipeline.qaoa_prefilter_size,
    )
    intent = Intent(goal=goal)

    # Create an event queue for this run
    run_q: queue.Queue = queue.Queue()
    run_id_holder: list[str] = []

    def on_event(event: dict) -> None:
        # Capture run_id from the done event so we can key the queue
        run_q.put(event)

    def run_pipeline() -> None:
        try:
            result = pipeline.run(intent, on_event=on_event)
            save_run(
                run_id=result.run_id,
                goal=goal,
                backend=backend,
                score=result.winning_score or 0.0,
                latency_s=result.total_latency_s,
                result_json=result.model_dump_json(),
            )
            with _run_queues_lock:
                _run_queues[result.run_id] = run_q
            run_id_holder.append(result.run_id)
        except Exception as exc:
            run_q.put({"type": "error", "message": str(exc)})
            logger.exception("Pipeline run failed")
        finally:
            run_q.put(None)  # sentinel

    # We need the run_id before the thread finishes.
    # Extract it from the queue's first event after thread starts.
    # Use a pre-assigned ID by wrapping the pipeline constructor.
    import uuid
    pre_assigned_id = str(uuid.uuid4())

    # Override: inject run_id into the pipeline's run() by monkey-patching uuid
    # Simpler: just start the thread and let the client poll /api/runs/<id>
    # after the server returns the pre-assigned id.

    # Store the queue under the pre-assigned id immediately
    with _run_queues_lock:
        _run_queues[pre_assigned_id] = run_q

    def run_pipeline_with_id() -> None:
        try:
            # Patch the intent with extra metadata so pipeline picks up our id
            result = pipeline.run(intent, on_event=on_event)
            # Re-key queue under the actual run_id
            with _run_queues_lock:
                if result.run_id != pre_assigned_id:
                    _run_queues[result.run_id] = _run_queues.pop(pre_assigned_id, run_q)
            save_run(
                run_id=result.run_id,
                goal=goal,
                backend=backend,
                score=result.winning_score or 0.0,
                latency_s=result.total_latency_s,
                result_json=result.model_dump_json(),
            )
            # Patch done event with actual run_id
            run_q.put({"type": "run_id", "run_id": result.run_id})
        except Exception as exc:
            run_q.put({"type": "error", "message": str(exc)})
            logger.exception("Pipeline run failed")
        finally:
            run_q.put(None)

    t = threading.Thread(target=run_pipeline_with_id, daemon=True)
    t.start()

    return jsonify({"run_id": pre_assigned_id}), 202


@app.route("/api/runs/<run_id>/events")
def run_events(run_id: str) -> Response:
    def generate() -> Generator[str, None, None]:
        # Wait up to 5s for the queue to appear (pipeline thread may not have started yet)
        deadline = time.monotonic() + 5.0
        while True:
            with _run_queues_lock:
                q = _run_queues.get(run_id)
            if q is not None:
                break
            if time.monotonic() > deadline:
                yield _sse({"type": "error", "message": "run not found"})
                return
            time.sleep(0.1)

        while True:
            try:
                event = q.get(timeout=30)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if event is None:
                break
            yield _sse(event)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/runs/<run_id>")
def get_run_result(run_id: str) -> Response:
    # Check live queues first (run may still be in progress)
    with _run_queues_lock:
        in_progress = run_id in _run_queues

    row = get_run(run_id)
    if row:
        return jsonify(row)
    if in_progress:
        return jsonify({"run_id": run_id, "status": "in_progress"}), 202
    return jsonify({"error": "run not found"}), 404


@app.route("/api/runs")
def get_run_history() -> Response:
    return jsonify(list_runs())


@app.route("/api/batch", methods=["POST"])
def start_batch() -> Response:
    body = request.get_json(force=True) or {}
    goals = [g.strip() for g in body.get("goals", []) if g.strip()]
    if not goals:
        return jsonify({"error": "goals list required"}), 400
    backend = body.get("quantum_backend", "stub")
    runs_per_goal = min(int(body.get("runs_per_goal", 1)), 10)

    import uuid
    batch_id = str(uuid.uuid4())
    save_batch(batch_id, goals, backend, runs_per_goal)

    batch_q: queue.Queue = queue.Queue()
    with _run_queues_lock:
        _run_queues[f"batch:{batch_id}"] = batch_q

    def run_batch() -> None:
        results: list[dict] = []
        total = len(goals) * runs_per_goal
        done = 0
        try:
            for goal in goals:
                goal_results: list[dict] = []
                for rep in range(runs_per_goal):
                    batch_q.put({"type": "batch_progress", "done": done, "total": total,
                                 "goal": goal[:60], "rep": rep + 1, "runs_per_goal": runs_per_goal})
                    config = get_config()
                    optimizer = QuantumOptimizer(
                        backend=backend,
                        circuit_depth=config.quantum.circuit_depth,
                        num_iterations=config.quantum.num_iterations,
                        seed=config.quantum.seed + done,
                    )
                    pipeline = Pipeline(
                        quantum_optimizer=optimizer,
                        max_candidates=config.pipeline.max_candidates,
                        qaoa_prefilter_size=config.pipeline.qaoa_prefilter_size,
                    )
                    result = pipeline.run(Intent(goal=goal))
                    save_run(
                        run_id=result.run_id,
                        goal=goal,
                        backend=backend,
                        score=result.winning_score or 0.0,
                        latency_s=result.total_latency_s,
                        result_json=result.model_dump_json(),
                    )
                    goal_results.append({
                        "run_id": result.run_id,
                        "goal": goal,
                        "qaoa_score": result.winning_score,
                        "classical_score": result.classical_winner_score,
                        "delta": round((result.winning_score or 0) - (result.classical_winner_score or 0), 4),
                        "same_winner": (
                            result.winning_variant and result.classical_winner_variant and
                            result.winning_variant.variant_id == result.classical_winner_variant.variant_id
                        ),
                        "classical_overlap": result.classical_overlap,
                        "latency_s": round(result.total_latency_s, 2),
                    })
                    done += 1
                results.extend(goal_results)
            update_batch(batch_id, "complete", results)
            batch_q.put({"type": "batch_done", "batch_id": batch_id, "results": results})
        except Exception as exc:
            update_batch(batch_id, "error", results)
            batch_q.put({"type": "error", "message": str(exc)})
            logger.exception("Batch run failed")
        finally:
            batch_q.put(None)

    threading.Thread(target=run_batch, daemon=True).start()
    return jsonify({"batch_id": batch_id}), 202


@app.route("/api/batch/<batch_id>/events")
def batch_events(batch_id: str) -> Response:
    def generate() -> Generator[str, None, None]:
        deadline = time.monotonic() + 5.0
        while True:
            with _run_queues_lock:
                q = _run_queues.get(f"batch:{batch_id}")
            if q is not None:
                break
            if time.monotonic() > deadline:
                yield _sse({"type": "error", "message": "batch not found"})
                return
            time.sleep(0.1)
        while True:
            try:
                event = q.get(timeout=30)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if event is None:
                break
            yield _sse(event)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


@app.route("/api/batch/<batch_id>")
def get_batch_result(batch_id: str) -> Response:
    row = get_batch(batch_id)
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(row)


@app.route("/api/batch")
def get_batch_history() -> Response:
    return jsonify(list_batches())


@app.route("/api/status")
def status() -> Response:
    config = get_config()
    results: dict = {
        "prescorer_status": _check_ollama(config.ollama.local_7b_endpoint),
        "prescorer_model": config.ollama.local_7b_model,
        "deepeval_status": _check_ollama(config.ollama.remote_32b_endpoint),
        "deepeval_model": config.ollama.remote_32b_model,
        "quantum": "ready",
        # legacy keys — kept for backwards compat
        "7b_local": _check_ollama(config.ollama.local_7b_endpoint),
        "32b_remote": _check_ollama(config.ollama.remote_32b_endpoint),
    }
    return jsonify(results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _check_ollama(endpoint: str) -> str:
    try:
        r = requests.get(f"{endpoint}/api/tags", timeout=2)
        return "online" if r.status_code == 200 else "error"
    except Exception:
        return "offline"


def create_app(db_path: str = "qpo_runs.db") -> Flask:
    init_db(Path(db_path))
    return app
