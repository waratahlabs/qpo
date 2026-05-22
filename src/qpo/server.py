"""QPO web server — Flask backend for the test harness UI.

Routes:
  GET  /                            → serve UI
  POST /api/runs                    → start a pipeline run
  GET  /api/runs/<id>/events        → SSE stream of run events
  GET  /api/runs/<id>               → completed run result (JSON)
  GET  /api/runs                    → run history list
  POST /api/batch                   → start a pipeline batch (job-pull model)
  GET  /api/batch/<id>/events       → SSE stream of batch events
  GET  /api/batch/<id>              → batch result + job progress
  GET  /api/batch                   → batch history list
  POST /api/batch/<id>/resume       → requeue failed jobs and resume
  GET  /api/status                  → backend health check
"""

import json
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Generator

import requests
from flask import Flask, Response, jsonify, request

from qpo import Intent, Pipeline
from qpo.config import get_config
from qpo.db import (claim_next_job, complete_job, create_jobs, fail_job,
                    get_batch, get_batch_jobs, get_batch_progress, get_run,
                    init_db, list_batches, list_runs, requeue_failed_jobs,
                    save_batch, save_run, update_batch)
from qpo.quantum.optimizer import QuantumOptimizer

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=None)
app.config["JSON_SORT_KEYS"] = False

# Per-run event queues: run_id → Queue[dict | None]
_run_queues: dict[str, queue.Queue] = {}
# Per-run replay buffer for SSE reconnection. Bounded to MAX_EVENT_BUFFER per
# run; oldest events are dropped when full. Cleaned up alongside _run_queues.
_run_event_buffers: dict[str, list[dict]] = {}
_run_queues_lock = threading.Lock()

MAX_EVENT_BUFFER = 200

_HTML_PATH = Path(__file__).parent.parent.parent / "ui.html"

MAX_JOB_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.route("/")
def index() -> Response:
    if _HTML_PATH.exists():
        return Response(_HTML_PATH.read_text(), mimetype="text/html")
    return Response("<h1>QPO</h1><p>ui.html not found.</p>", mimetype="text/html")


# ---------------------------------------------------------------------------
# API — single runs
# ---------------------------------------------------------------------------

@app.route("/api/runs", methods=["POST"])
def start_run() -> Response:
    body = request.get_json(force=True) or {}
    goal = body.get("goal", "").strip()
    if not goal:
        return jsonify({"error": "goal is required"}), 400

    backend = body.get("quantum_backend", body.get("backend", "stub"))

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
        history_endpoint=os.environ.get("QPO_SERVER_URL", "http://localhost:5001"),
    )
    intent = Intent(goal=goal)

    # Create an event queue + bounded replay buffer for this run
    run_q: queue.Queue = queue.Queue()
    event_buffer: list[dict] = []

    def on_event(event: dict) -> None:
        with _run_queues_lock:
            event_buffer.append(event)
            if len(event_buffer) > MAX_EVENT_BUFFER:
                del event_buffer[0:len(event_buffer) - MAX_EVENT_BUFFER]
        run_q.put(event)

    pre_assigned_id = str(uuid.uuid4())

    with _run_queues_lock:
        _run_queues[pre_assigned_id] = run_q
        _run_event_buffers[pre_assigned_id] = event_buffer

    def run_pipeline_with_id() -> None:
        try:
            result = pipeline.run(intent, on_event=on_event, run_id=pre_assigned_id)
            try:
                save_run(
                    run_id=result.run_id,
                    goal=goal,
                    backend=backend,
                    score=result.winning_score or 0.0,
                    latency_s=result.total_latency_s,
                    result_json=result.model_dump_json(),
                    qaoa_status=result.metadata.get("qaoa_status", "ok"),
                )
            except Exception as db_exc:
                logger.exception("save_run failed for %s", result.run_id)
                on_event({"type": "db_error", "message": str(db_exc), "run_id": result.run_id})
            on_event({"type": "run_id", "run_id": result.run_id})
        except Exception as exc:
            on_event({"type": "error", "message": str(exc)})
            logger.exception("Pipeline run failed")
        finally:
            run_q.put(None)
            with _run_queues_lock:
                _run_queues.pop(pre_assigned_id, None)
                _run_event_buffers.pop(pre_assigned_id, None)

    t = threading.Thread(target=run_pipeline_with_id, daemon=True)
    t.start()

    return jsonify({"run_id": pre_assigned_id}), 202


@app.route("/api/runs/<run_id>/events")
def run_events(run_id: str) -> Response:
    def generate() -> Generator[str, None, None]:
        deadline = time.monotonic() + 5.0
        while True:
            with _run_queues_lock:
                q = _run_queues.get(run_id)
                buf_snapshot = list(_run_event_buffers.get(run_id, []))
            if q is not None:
                break
            if time.monotonic() > deadline:
                yield _sse({"type": "error", "message": "run not found"})
                return
            time.sleep(0.1)

        # Replay buffered events first so reconnecting clients see history.
        seen = len(buf_snapshot)
        for ev in buf_snapshot:
            yield _sse(ev)

        while True:
            try:
                event = q.get(timeout=30)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if event is None:
                break
            # Skip events the buffer replay already delivered. The on_event
            # handler appends to the buffer before put() on the queue, so the
            # first `seen` queue events are duplicates of the replay.
            if seen > 0:
                seen -= 1
                continue
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
    full = request.args.get("full", "").lower() in ("1", "true")
    limit = int(request.args.get("limit", 200))
    return jsonify(list_runs(limit=limit, full=full))


# ---------------------------------------------------------------------------
# API — batches (job-pull model)
# ---------------------------------------------------------------------------


def _run_batch_worker(
    batch_id: str,
    batch_q: queue.Queue,
    batch_buffer: list,
    total: int,
    runs_per_goal: int,
) -> None:
    """Pull-model batch worker. Loops until no pending or failed-retryable
    jobs remain. Safe to invoke either fresh or as a resume — claim_next_job
    is atomic and requeue_failed_jobs only acts on attempts < MAX."""

    def emit_batch(event: dict) -> None:
        with _run_queues_lock:
            batch_buffer.append(event)
            if len(batch_buffer) > MAX_EVENT_BUFFER:
                del batch_buffer[0:len(batch_buffer) - MAX_EVENT_BUFFER]
        batch_q.put(event)

    try:
        # Requeue once before the claim loop. Calling it inside the loop turns
        # a fast-failing job into an instant attempts-burner.
        requeue_failed_jobs(batch_id, MAX_JOB_ATTEMPTS)
        while True:
            job = claim_next_job(batch_id)
            if job is None:
                break

            progress = get_batch_progress(batch_id)
            emit_batch({
                "type": "batch_progress",
                "done": progress["complete"],
                "total": total,
                "goal": job["goal"][:60],
                "rep": job["rep"] + 1,
                "runs_per_goal": runs_per_goal,
            })
            try:
                config = get_config()
                # Deterministic per-job seed: stable across restarts/resumes
                # because it depends only on (config seed, goal, rep) — not on
                # how many other jobs happened to complete first.
                job_seed = (
                    config.quantum.seed
                    + (hash(job["goal"]) & 0xFFFF)
                    + job["rep"]
                ) & 0x7FFFFFFF
                optimizer = QuantumOptimizer(
                    backend=job["backend"],
                    circuit_depth=config.quantum.circuit_depth,
                    num_iterations=config.quantum.num_iterations,
                    seed=job_seed,
                )
                pipeline = Pipeline(
                    quantum_optimizer=optimizer,
                    max_candidates=config.pipeline.max_candidates,
                    qaoa_prefilter_size=config.pipeline.qaoa_prefilter_size,
                    history_endpoint=os.environ.get("QPO_SERVER_URL", "http://localhost:5001"),
                )
                result = pipeline.run(Intent(goal=job["goal"]))
                result_dict = {
                    "run_id": result.run_id,
                    "goal": job["goal"],
                    "qaoa_score": result.winning_score,
                    "classical_score": result.classical_winner_score,
                    "delta": round(
                        (result.winning_score or 0)
                        - (result.classical_winner_score or 0),
                        4,
                    ),
                    "same_winner": bool(
                        result.winning_variant
                        and result.classical_winner_variant
                        and result.winning_variant.variant_id
                        == result.classical_winner_variant.variant_id
                    ),
                    "classical_overlap": result.classical_overlap,
                    "latency_s": round(result.total_latency_s, 2),
                }
                # save_run BEFORE complete_job: if persistence fails, the job
                # stays in 'running' (claim already incremented attempts) and
                # the exception falls through to fail_job below. Avoids the
                # split-brain where job=complete but run row is missing.
                save_run(
                    run_id=result.run_id,
                    goal=job["goal"],
                    backend=job["backend"],
                    score=result.winning_score or 0.0,
                    latency_s=result.total_latency_s,
                    result_json=result.model_dump_json(),
                    qaoa_status=result.metadata.get("qaoa_status", "ok"),
                )
                complete_job(job["job_id"], result_dict)
            except Exception as exc:
                fail_job(job["job_id"], str(exc))
                logger.exception("Job %s failed", job["job_id"])

        jobs = get_batch_jobs(batch_id)
        results = [
            json.loads(j["result_json"])
            for j in jobs
            if j["status"] == "complete" and j["result_json"]
        ]
        failed_count = sum(1 for j in jobs if j["status"] == "failed")
        final_status = "complete" if failed_count == 0 else "partial"
        update_batch(batch_id, final_status, results)
        emit_batch({
            "type": "batch_done",
            "batch_id": batch_id,
            "results": results,
            "failed": failed_count,
        })
    except Exception as exc:
        emit_batch({"type": "error", "message": str(exc)})
        logger.exception("Batch worker crashed for %s", batch_id)
    finally:
        batch_q.put(None)
        with _run_queues_lock:
            _run_queues.pop(f"batch:{batch_id}", None)
            _run_event_buffers.pop(f"batch:{batch_id}", None)


@app.route("/api/batch", methods=["POST"])
def start_batch() -> Response:
    body = request.get_json(force=True) or {}
    goals = [g.strip() for g in body.get("goals", []) if g.strip()]
    if not goals:
        return jsonify({"error": "goals list required"}), 400
    backend = body.get("quantum_backend", body.get("backend", "stub"))
    runs_per_goal = min(int(body.get("runs_per_goal", 1)), 10)

    batch_id = str(uuid.uuid4())
    save_batch(batch_id, goals, backend, runs_per_goal)
    create_jobs(batch_id, goals, backend, runs_per_goal)
    total = len(goals) * runs_per_goal

    batch_q: queue.Queue = queue.Queue()
    batch_buffer: list = []
    with _run_queues_lock:
        _run_queues[f"batch:{batch_id}"] = batch_q
        _run_event_buffers[f"batch:{batch_id}"] = batch_buffer

    threading.Thread(
        target=_run_batch_worker,
        args=(batch_id, batch_q, batch_buffer, total, runs_per_goal),
        daemon=True,
    ).start()
    return jsonify({"batch_id": batch_id}), 202


@app.route("/api/batch/<batch_id>/events")
def batch_events(batch_id: str) -> Response:
    def generate() -> Generator[str, None, None]:
        deadline = time.monotonic() + 5.0
        while True:
            with _run_queues_lock:
                q = _run_queues.get(f"batch:{batch_id}")
                buf_snapshot = list(_run_event_buffers.get(f"batch:{batch_id}", []))
            if q is not None:
                break
            if time.monotonic() > deadline:
                yield _sse({"type": "error", "message": "batch not found"})
                return
            time.sleep(0.1)

        # Replay buffered events so reconnecting clients see history.
        seen = len(buf_snapshot)
        for ev in buf_snapshot:
            yield _sse(ev)

        while True:
            try:
                event = q.get(timeout=30)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if event is None:
                break
            if seen > 0:
                seen -= 1
                continue
            yield _sse(event)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


@app.route("/api/batch/<batch_id>")
def get_batch_result(batch_id: str) -> Response:
    row = get_batch(batch_id)
    if not row:
        return jsonify({"error": "not found"}), 404
    row.update(get_batch_progress(batch_id))
    return jsonify(row)


@app.route("/api/batch")
def get_batch_history() -> Response:
    return jsonify(list_batches())


@app.route("/api/batch/<batch_id>/resume", methods=["POST"])
def resume_batch(batch_id: str) -> Response:
    row = get_batch(batch_id)
    if not row:
        return jsonify({"error": "not found"}), 404

    requeued = requeue_failed_jobs(batch_id, MAX_JOB_ATTEMPTS)
    runs_per_goal = int(row.get("runs_per_goal", 1))
    total = len(row.get("goals", [])) * runs_per_goal

    if total == 0:
        logger.warning("Resume requested for batch %s with total=0 — nothing to do", batch_id)
        return jsonify({"batch_id": batch_id, "requeued": requeued, "warning": "no jobs to run"}), 200

    # Spawn (or replace) the batch event queue and worker thread.
    batch_q: queue.Queue = queue.Queue()
    batch_buffer: list = []
    with _run_queues_lock:
        _run_queues[f"batch:{batch_id}"] = batch_q
        _run_event_buffers[f"batch:{batch_id}"] = batch_buffer

    threading.Thread(
        target=_run_batch_worker,
        args=(batch_id, batch_q, batch_buffer, total, runs_per_goal),
        daemon=True,
    ).start()
    return jsonify({"batch_id": batch_id, "requeued": requeued}), 202


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.route("/api/status")
def status() -> Response:
    config = get_config()
    # Cache one probe per unique endpoint — when 7b and 32b point at the same
    # host (common dev setup) we'd otherwise hit it 4 times for one /status.
    endpoint_status: dict[str, str] = {}

    def probe(endpoint: str) -> str:
        if endpoint not in endpoint_status:
            endpoint_status[endpoint] = _check_ollama(endpoint)
        return endpoint_status[endpoint]

    results: dict = {
        "prescorer_status": probe(config.ollama.local_7b_endpoint),
        "prescorer_model": config.ollama.local_7b_model,
        "deepeval_status": probe(config.ollama.remote_32b_endpoint),
        "deepeval_model": config.ollama.remote_32b_model,
        "quantum": "ready",
        # legacy keys — kept for backwards compat
        "7b_local": probe(config.ollama.local_7b_endpoint),
        "32b_remote": probe(config.ollama.remote_32b_endpoint),
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
