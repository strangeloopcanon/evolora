#!/usr/bin/env python3
"""Run a small benchmark suite and write a JSON report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from symbiont_ecology.benchmarks import BenchmarkCase, BenchmarkSuite, run_benchmark_suite


def _default_suite(mode: str) -> BenchmarkSuite:
    repo_root = Path(__file__).resolve().parents[1]
    if mode == "paper_qwen3":
        configs = [
            ("frozen", repo_root / "config" / "experiments" / "paper_qwen3_frozen.yaml"),
            ("single", repo_root / "config" / "experiments" / "paper_qwen3_single.yaml"),
            ("ecology", repo_root / "config" / "experiments" / "paper_qwen3_ecology.yaml"),
        ]
        return BenchmarkSuite(
            cases=[
                BenchmarkCase(
                    name=name, config_path=path, generations=3, batch_size=4, backend="hf"
                )
                for name, path in configs
            ]
        )
    configs = [
        ("ci_frozen", repo_root / "config" / "benchmarks" / "ci_frozen.yaml"),
        ("ci_single", repo_root / "config" / "benchmarks" / "ci_single.yaml"),
        ("ci_ecology", repo_root / "config" / "benchmarks" / "ci_ecology.yaml"),
    ]
    return BenchmarkSuite(
        cases=[
            BenchmarkCase(name=name, config_path=path, generations=2, batch_size=4, backend="stub")
            for name, path in configs
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Evolora benchmark suite.")
    parser.add_argument(
        "--mode",
        choices=["ci", "paper_qwen3"],
        default="ci",
        help="Which suite to run (ci uses stub backend).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory root (default: artifacts_bench_<timestamp>).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = _default_suite(args.mode)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = args.output or Path(f"artifacts_bench_{args.mode}_{ts}")
    report = run_benchmark_suite(suite=suite, output_root=output_root)
    report_path = output_root / "benchmark_report.json"
    output_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    print("Wrote benchmark report to", report_path)
    for result in report.results:
        m = result.metrics
        print(
            f"- {result.name}: episodes={m.episodes} success_rate={m.success_rate:.3f} "
            f"avg_total={m.avg_total_reward:.3f} avg_cost={m.avg_energy_spent:.3f}"
        )


if __name__ == "__main__":
    main()
