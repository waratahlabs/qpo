"""CLI entry point for QPO pipeline.

Runnable via: python -m qpo [options]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from qpo import Intent, Pipeline
from qpo.config import get_config, set_config
from qpo.db import init_db, list_runs, save_run


def setup_logging(log_level: str) -> None:
    """Configure structured logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="qpo",
        description="Quantum-assisted prompt optimization pipeline",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the web UI server instead of running a single pipeline",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for --serve mode (default 5000, env: QPO_PORT)",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Plain English optimization goal (required without --serve)",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="Optional context for the optimization",
    )
    parser.add_argument(
        "--quantum-backend",
        type=str,
        default="stub",
        choices=["stub", "lightning", "braket-sim", "braket-qpu"],
        help="Quantum backend to use",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--batch-file",
        type=str,
        default=None,
        help="Path to a JSON batch file describing multiple goals to run",
    )
    parser.add_argument(
        "--runs-per-goal",
        type=int,
        default=None,
        help="Override runs_per_goal from the batch file (default: use file value, fallback 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print goals without executing; only valid with --batch-file",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.batch_file and args.goal:
        parser.error("--batch-file and --goal are mutually exclusive")
    if args.dry_run and not args.batch_file:
        parser.error("--dry-run requires --batch-file")

    if args.batch_file:
        return _run_batch(args)

    if args.serve:
        return _serve(args.port)

    if not args.goal:
        parser.error("--goal is required (or use --serve for web UI)")

    logger = logging.getLogger(__name__)
    logger.info("QPO Pipeline starting...")

    # Create intent
    intent = Intent(goal=args.goal, context=args.context)

    # Configure quantum backend
    config = get_config()
    config.quantum.backend = args.quantum_backend
    set_config(config)

    # Run pipeline
    try:
        pipeline = Pipeline(max_candidates=config.pipeline.max_candidates)
        result = pipeline.run(intent)

        logger.info(f"Pipeline complete! Result: {result.run_id}")
        logger.info(f"  Winner: {result.winning_variant.variant_id}")
        logger.info(f"  Score: {result.winning_score:.3f}")
        logger.info(f"  Total time: {result.total_latency_s:.2f}s")

        # Output results
        result_json = result.model_dump_json(indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(result_json)
            logger.info(f"Results written to {args.output}")
        else:
            print("\n" + "=" * 60)
            print(result_json)
            print("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


def _run_batch(args: argparse.Namespace) -> int:
    """Run a batch of goals from a JSON batch file."""
    logger = logging.getLogger(__name__)

    # 1. Load batch file
    try:
        with open(args.batch_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"batch file not found: {args.batch_file}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"batch file is not valid JSON: {e}")
        return 1

    # 2. Extract raw goals
    raw_goals = data.get("goals", [])
    if not raw_goals:
        logger.error("batch file has no goals")
        return 1

    # 3. Normalise goals
    goals: list[tuple[str, dict | None]] = []
    for item in raw_goals:
        if isinstance(item, str):
            goals.append((item, None))
        elif isinstance(item, dict) and "goal" in item:
            goal_text = item["goal"]
            metadata = {k: v for k, v in item.items() if k != "goal"}
            goals.append((goal_text, metadata or None))
        else:
            logger.error(f"invalid goal entry in batch file: {item!r}")
            return 1

    # 4. Determine backend (CLI arg wins unless it's the default "stub")
    backend = args.quantum_backend if args.quantum_backend != "stub" else data.get("backend", "stub")

    # 5. Determine reps
    reps = args.runs_per_goal if args.runs_per_goal is not None else data.get("runs_per_goal", 1)

    # 6. Dry-run: print and return
    if args.dry_run:
        for idx, (goal_text, meta) in enumerate(goals):
            if meta:
                print(f"[{idx + 1}/{len(goals)}] {goal_text}  metadata={meta}")
            else:
                print(f"[{idx + 1}/{len(goals)}] {goal_text}")
        print(f"backend={backend} runs_per_goal={reps} total_runs={len(goals) * reps}")
        return 0

    # 7. Configure quantum backend
    config = get_config()
    config.quantum.backend = backend
    set_config(config)

    # 8. Initialise DB
    init_db()

    # 9. Build pipeline — load run history from local DB for off-diagonal QUBO
    run_history = list_runs(full=True)
    pipeline = Pipeline(max_candidates=config.pipeline.max_candidates, run_history=run_history)

    # 10. Per-goal accumulators
    accumulators: list[dict] = []
    for idx, (goal_text, meta) in enumerate(goals):
        goal_id = meta.get("id") if meta else f"goal-{idx + 1}"
        accumulators.append(
            {
                "goal_text": goal_text,
                "id": goal_id,
                "metadata": meta,
                "reps": reps,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "delta_sum": 0.0,
                "completed": 0,
            }
        )

    # 11. Run loop
    total = len(goals) * reps
    completed = 0
    interrupted = False
    try:
        for goal_idx, (goal_text, _meta) in enumerate(goals):
            acc = accumulators[goal_idx]
            for rep in range(reps):
                job_num = goal_idx * reps + rep + 1
                print(f"[{job_num}/{total}] {goal_text[:70]}")
                try:
                    result = pipeline.run(Intent(goal=goal_text))
                    save_run(
                        result.run_id,
                        goal_text,
                        backend,
                        result.winning_score,
                        result.total_latency_s,
                        result.model_dump_json(),
                    )
                    delta = result.winning_score - result.classical_winner_score
                    acc["delta_sum"] += delta
                    if result.winning_score > result.classical_winner_score:
                        acc["wins"] += 1
                    elif result.winning_score < result.classical_winner_score:
                        acc["losses"] += 1
                    else:
                        acc["ties"] += 1
                    acc["completed"] += 1
                    completed += 1
                except Exception as e:
                    logger.error(f"job {job_num} failed: {e}", exc_info=True)
                    acc["losses"] += 1
                    acc["completed"] += 1
                    completed += 1
                if args.output and completed % 5 == 0:
                    _write_results(args.output, data, backend, reps, completed, accumulators)
                    logger.info(f"Checkpoint written at {completed}/{total}")
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted — writing partial results...")

    # 13. Write output JSON if requested
    if args.output:
        _write_results(args.output, data, backend, reps, completed, accumulators)
        logger.info(f"Batch results written to {args.output}")

    # 14. Summary table
    _print_summary_table(accumulators)

    if interrupted:
        return 130
    return 0


def _write_results(
    path: str,
    data: dict,
    backend: str,
    reps: int,
    completed: int,
    accumulators: list[dict],
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_goals = []
    for acc in accumulators:
        done = acc["completed"]
        mean_delta = (acc["delta_sum"] / done) if done > 0 else 0.0
        out_goals.append(
            {
                "id": acc["id"],
                "goal": acc["goal_text"],
                "metadata": acc["metadata"],
                "reps": done,
                "wins": acc["wins"],
                "losses": acc["losses"],
                "ties": acc["ties"],
                "mean_delta": mean_delta,
            }
        )
    out_data = {
        "experiment_id": data.get("experiment_id", "batch"),
        "backend": backend,
        "runs_per_goal": reps,
        "total_runs": completed,
        "goals": out_goals,
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)


def _print_summary_table(accumulators: list[dict]) -> None:
    """Print a plain-text box-drawing summary table."""
    col_goal = 31
    col_num = 4
    col_delta = 7

    def fmt_row(goal: str, wins: str, loss: str, ties: str, delta: str) -> str:
        return (
            f"│ {goal:<{col_goal}} "
            f"│ {wins:>{col_num}} "
            f"│ {loss:>{col_num}} "
            f"│ {ties:>{col_num}} "
            f"│ {delta:>{col_delta}} │"
        )

    top = (
        f"┌{'─' * (col_goal + 2)}"
        f"┬{'─' * (col_num + 2)}"
        f"┬{'─' * (col_num + 2)}"
        f"┬{'─' * (col_num + 2)}"
        f"┬{'─' * (col_delta + 2)}┐"
    )
    mid = (
        f"├{'─' * (col_goal + 2)}"
        f"┼{'─' * (col_num + 2)}"
        f"┼{'─' * (col_num + 2)}"
        f"┼{'─' * (col_num + 2)}"
        f"┼{'─' * (col_delta + 2)}┤"
    )
    bot = (
        f"└{'─' * (col_goal + 2)}"
        f"┴{'─' * (col_num + 2)}"
        f"┴{'─' * (col_num + 2)}"
        f"┴{'─' * (col_num + 2)}"
        f"┴{'─' * (col_delta + 2)}┘"
    )

    print(top)
    print(fmt_row("Goal", "Wins", "Loss", "Ties", "Δ mean"))
    print(mid)

    total_wins = 0
    total_losses = 0
    total_ties = 0
    total_delta_sum = 0.0
    total_completed = 0
    for acc in accumulators:
        label = f"{acc['id']}: {acc['goal_text']}"
        if len(label) > col_goal:
            label = label[: col_goal - 3] + "..."
        done = acc["completed"]
        mean_delta = (acc["delta_sum"] / done) if done > 0 else 0.0
        delta_str = f"{mean_delta:+.3f}"
        print(
            fmt_row(
                label,
                str(acc["wins"]),
                str(acc["losses"]),
                str(acc["ties"]),
                delta_str,
            )
        )
        total_wins += acc["wins"]
        total_losses += acc["losses"]
        total_ties += acc["ties"]
        total_delta_sum += acc["delta_sum"]
        total_completed += done

    print(mid)
    total_mean = (total_delta_sum / total_completed) if total_completed > 0 else 0.0
    print(
        fmt_row(
            "TOTAL",
            str(total_wins),
            str(total_losses),
            str(total_ties),
            f"{total_mean:+.3f}",
        )
    )
    print(bot)


def _serve(port: int) -> int:
    from qpo.server import create_app
    app = create_app()
    logger = logging.getLogger(__name__)
    logger.info("QPO web UI starting on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
