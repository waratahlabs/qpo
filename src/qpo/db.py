"""SQLite persistence for QPO pipeline runs."""

import json
import sqlite3
from pathlib import Path
from typing import Any

_DB_PATH = Path("qpo_runs.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = _DB_PATH) -> None:
    global _DB_PATH
    _DB_PATH = db_path
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id      TEXT PRIMARY KEY,
                goal        TEXT NOT NULL,
                backend     TEXT NOT NULL,
                score       REAL,
                latency_s   REAL,
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


def save_run(
    run_id: str,
    goal: str,
    backend: str,
    score: float,
    latency_s: float,
    result_json: str,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs
                (run_id, goal, backend, score, latency_s, result_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, goal, backend, round(score, 4), round(latency_s, 2), result_json),
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


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT run_id, goal, backend, score, latency_s, created_at "
            "FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


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
