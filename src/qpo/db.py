"""SQLite persistence for QPO pipeline runs."""

import json
import os
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

# Default to repo root (two levels above this file's package directory).
# Overridable via QPO_DB_PATH env var to prevent CWD-relative silently wrong DB.
_DEFAULT_DB_PATH = Path(
    os.getenv("QPO_DB_PATH", str(Path(__file__).parent.parent.parent / "qpo_runs.db"))
)
_DB_PATH = _DEFAULT_DB_PATH
_DB_PATH_LOCK = threading.Lock()
_DB_INITIALISED = False

# SQLite 3.35.0 (2021-03-12) introduced the RETURNING clause used in
# claim_next_job(). Older system SQLite versions will raise an OperationalError.
_MIN_SQLITE_VERSION = (3, 35, 0)


def _connect() -> sqlite3.Connection:
    # Read _DB_PATH inside the lock so we see the current value consistently.
    with _DB_PATH_LOCK:
        path = _DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = _DEFAULT_DB_PATH) -> None:
    """Initialise the schema at db_path.

    Thread-safe: the global path mutation is guarded by a lock. If init_db has
    already been called with a different path, raises RuntimeError — re-init
    to a different DB inside one process is a programming error.

    Raises:
        RuntimeError: If SQLite version < 3.35.0 (RETURNING clause required).
        RuntimeError: If reinitialised with a different path.
    """
    global _DB_PATH, _DB_INITIALISED

    sqlite_ver = tuple(int(x) for x in sqlite3.sqlite_version.split("."))
    if sqlite_ver < _MIN_SQLITE_VERSION:
        raise RuntimeError(
            f"SQLite {sqlite3.sqlite_version} is too old — QPO requires ≥3.35.0 "
            "(RETURNING clause support). Upgrade SQLite or use a newer Python build."
        )

    with _DB_PATH_LOCK:
        if _DB_INITIALISED and Path(_DB_PATH) != Path(db_path):
            raise RuntimeError(
                f"init_db already initialised at {_DB_PATH}; refusing to switch to {db_path}"
            )
        _DB_PATH = db_path
        _DB_INITIALISED = True
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id      TEXT PRIMARY KEY,
                goal        TEXT NOT NULL,
                backend     TEXT NOT NULL,
                score       REAL,
                latency_s   REAL,
                qaoa_status TEXT NOT NULL DEFAULT 'ok',
                result_json TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                batch_id    TEXT PRIMARY KEY,
                goals_json  TEXT NOT NULL,
                backend     TEXT NOT NULL,
                runs_per_goal INTEGER NOT NULL DEFAULT 1,
                status      TEXT NOT NULL DEFAULT 'running',
                results_json TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id      TEXT PRIMARY KEY,
                batch_id    TEXT NOT NULL,
                goal        TEXT NOT NULL,
                rep         INTEGER NOT NULL,
                backend     TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                -- attempts = number of times this job has been claimed
                -- (incremented in claim_next_job), NOT the number of times it
                -- has failed. A job that succeeds on its first claim has attempts=1.
                attempts    INTEGER NOT NULL DEFAULT 0,
                result_json TEXT,
                error       TEXT,
                created_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_batch_status "
            "ON jobs (batch_id, status)"
        )
        # Safe migrations — ADD COLUMN is idempotent via the try/except pattern.
        for migration in [
            "ALTER TABLE jobs ADD COLUMN metadata_json TEXT",
            "ALTER TABLE runs ADD COLUMN qaoa_status TEXT NOT NULL DEFAULT 'ok'",
        ]:
            try:
                conn.execute(migration)
            except Exception:
                pass  # column already exists


def save_run(
    run_id: str,
    goal: str,
    backend: str,
    score: float,
    latency_s: float,
    result_json: str,
    qaoa_status: str = "ok",
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs
                (run_id, goal, backend, score, latency_s, qaoa_status, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, goal, backend, round(score, 4), round(latency_s, 2), qaoa_status, result_json),
        )


def get_run(run_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    if d.get("result_json"):
        d["result"] = json.loads(d.pop("result_json"))
    return d


def list_runs(limit: int = 500, full: bool = False) -> list[dict[str, Any]]:
    """List recent runs. When full=False, the result_json blob is omitted from
    the SELECT entirely — no wasted I/O on payloads we discard."""
    with _connect() as conn:
        if full:
            rows = conn.execute(
                "SELECT run_id, goal, backend, score, latency_s, qaoa_status, created_at, result_json "
                "FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT run_id, goal, backend, score, latency_s, qaoa_status, created_at "
                "FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        if full and d.get("result_json"):
            d["result"] = json.loads(d.pop("result_json"))
        results.append(d)
    return results


def save_batch(batch_id: str, goals: list[str], backend: str, runs_per_goal: int) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO batches (batch_id, goals_json, backend, runs_per_goal) VALUES (?, ?, ?, ?)",
            (batch_id, json.dumps(goals), backend, runs_per_goal),
        )


def update_batch(batch_id: str, status: str, results: list[dict[str, Any]]) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE batches SET status=?, results_json=? WHERE batch_id=?",
            (status, json.dumps(results), batch_id),
        )


def get_batch(batch_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["goals"] = json.loads(d.pop("goals_json"))
    if d.get("results_json"):
        d["results"] = json.loads(d.pop("results_json"))
    else:
        d.pop("results_json", None)
        d["results"] = []
    return d


def list_batches(limit: int = 20) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT batch_id, goals_json, backend, runs_per_goal, status, created_at "
            "FROM batches ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["goals"] = json.loads(d.pop("goals_json"))
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Job queue (per-batch pull-model work units)
# ---------------------------------------------------------------------------


def create_jobs(
    batch_id: str,
    goals: list[str],
    backend: str,
    runs_per_goal: int,
    metadata: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Insert one job per (goal, rep) for the batch. Returns the job_ids in
    insertion order.

    When `metadata` is provided, it must be a list aligned with `goals` —
    each goal's metadata dict is JSON-serialised and stored on every rep
    of that goal. When None, metadata_json is stored as NULL.
    """
    job_ids: list[str] = []
    rows: list[tuple[str, str, str, int, str, str | None]] = []
    for idx, goal in enumerate(goals):
        meta_json: str | None
        if metadata is not None:
            meta_json = json.dumps(metadata[idx])
        else:
            meta_json = None
        for rep in range(runs_per_goal):
            jid = str(uuid.uuid4())
            job_ids.append(jid)
            rows.append((jid, batch_id, goal, rep, backend, meta_json))
    with _connect() as conn:
        conn.executemany(
            "INSERT INTO jobs (job_id, batch_id, goal, rep, backend, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    return job_ids


def claim_next_job(batch_id: str) -> dict[str, Any] | None:
    """Atomically claim one pending job for the batch. Returns None if none
    pending. SQLite single-writer semantics make the UPDATE atomic; the
    nested SELECT picks an arbitrary pending job_id which we then mark
    running and return.

    Requires SQLite ≥3.35.0 (RETURNING clause). Checked at init_db().
    """
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE jobs
               SET status='running',
                   attempts=attempts+1,
                   updated_at=datetime('now')
             WHERE job_id = (
                 SELECT job_id FROM jobs
                  WHERE batch_id=? AND status='pending'
                  LIMIT 1
             )
            RETURNING job_id, batch_id, goal, rep, backend, status, attempts
            """,
            (batch_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return dict(row)


def complete_job(job_id: str, result: dict[str, Any]) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE jobs
               SET status='complete',
                   result_json=?,
                   error=NULL,
                   updated_at=datetime('now')
             WHERE job_id=?
            """,
            (json.dumps(result), job_id),
        )


def fail_job(job_id: str, error: str) -> None:
    """Mark a job failed. Note: attempts is incremented on claim, so we do
    not double-count here — we just record the error and flip status."""
    with _connect() as conn:
        conn.execute(
            """
            UPDATE jobs
               SET status='failed',
                   error=?,
                   updated_at=datetime('now')
             WHERE job_id=?
            """,
            (error, job_id),
        )


def requeue_failed_jobs(batch_id: str, max_attempts: int = 3) -> int:
    """Move failed jobs with attempts < max_attempts back to pending.
    Returns count requeued."""
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE jobs
               SET status='pending',
                   updated_at=datetime('now')
             WHERE batch_id=?
               AND status='failed'
               AND attempts < ?
            """,
            (batch_id, max_attempts),
        )
        return cur.rowcount or 0


def get_batch_jobs(batch_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE batch_id=? ORDER BY created_at ASC",
            (batch_id,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        meta_raw = d.pop("metadata_json", None)
        d["metadata"] = json.loads(meta_raw) if meta_raw else None
        out.append(d)
    return out


def get_batch_progress(batch_id: str) -> dict[str, int]:
    """Return per-status counts for the batch. Always includes all keys."""
    out = {"total": 0, "pending": 0, "running": 0, "complete": 0, "failed": 0}
    with _connect() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs WHERE batch_id=? GROUP BY status",
            (batch_id,),
        ).fetchall()
    for r in rows:
        status = r["status"]
        n = int(r["n"])
        out["total"] += n
        if status in out:
            out[status] = n
    return out
