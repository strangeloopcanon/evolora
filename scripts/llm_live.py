#!/usr/bin/env python3
"""Dry-run LLM live test harness.

In baseline mode we validate wiring without incurring external cost.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

from rich.console import Console

from symbiont_ecology.metrics.telemetry import LiveEvalSummary

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM live goldens")
    parser.add_argument("--dry-run", action="store_true", help="Skip actual provider calls")
    parser.add_argument(
        "--goldens",
        type=Path,
        default=Path("tests_llm_live/goldens.json"),
        help="Golden scenarios file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time()
    summary = LiveEvalSummary(
        total_cases=0,
        passed=0,
        failed=0,
        cost_usd=0.0,
        latency_p95_ms=0.0,
        mode="baseline",
        notes="Dry run executed; no live calls performed.",
    )
    if not args.dry_run and args.goldens.exists():
        summary.notes = "Live evaluations are not yet implemented."
    console.print("[bold cyan]LLM-Live Summary[/bold cyan]")
    console.print(json.dumps(summary.model_dump(), indent=2))
    console.print(f"Elapsed: {time() - t0:.2f}s")


if __name__ == "__main__":
    main()
