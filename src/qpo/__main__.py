"""CLI entry point for QPO pipeline.

Runnable via: python -m qpo [options]
"""

import argparse
import json
import logging
import sys

from qpo import Intent, Pipeline
from qpo.config import get_config, set_config


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

    args = parser.parse_args()
    setup_logging(args.log_level)

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


def _serve(port: int) -> int:
    from qpo.server import create_app
    app = create_app()
    logger = logging.getLogger(__name__)
    logger.info("QPO web UI starting on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
