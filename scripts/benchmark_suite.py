#!/usr/bin/env python3
"""Run a small benchmark suite and write a JSON report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

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


def _load_suite(path: Path) -> BenchmarkSuite:
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid suite file: {path}")
    suite = BenchmarkSuite.model_validate(data)
    repo_root = Path(__file__).resolve().parents[1]
    for case in suite.cases:
        if not case.config_path.is_absolute():
            case.config_path = (repo_root / case.config_path).resolve()
    return suite


def _format_pm(mean: float, std: float, precision: int = 3) -> str:
    if abs(std) <= 1e-12:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def _write_markdown_report(report, output_root: Path) -> Path:
    lines: list[str] = []
    lines.append("# Evolora Benchmark Report")
    lines.append("")
    lines.append(f"- started_at: `{report.started_at.isoformat()}`")
    lines.append(f"- finished_at: `{report.finished_at.isoformat()}`")
    if report.git_commit:
        lines.append(f"- git_commit: `{report.git_commit}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| case | backend | seeds | gens | episodes | success_rate | avg_roi | merges | qd_cov_max | colonies_max |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in report.results:
        seeds = ",".join(str(r.seed) for r in result.replicates) if result.replicates else "-"
        gens = next((c.generations for c in report.suite.cases if c.name == result.name), None)
        gens_str = str(gens) if gens is not None else "-"
        m = result.metrics
        ms = result.metrics_std
        oe = result.open_endedness
        oes = result.open_endedness_std
        lines.append(
            "| "
            + " | ".join(
                [
                    result.name,
                    result.backend,
                    seeds,
                    gens_str,
                    _format_pm(float(m.episodes), float(ms.episodes), precision=1),
                    _format_pm(m.success_rate, ms.success_rate),
                    _format_pm(m.avg_roi, ms.avg_roi),
                    _format_pm(float(oe.merges_total), float(oes.merges_total), precision=1),
                    _format_pm(oe.qd_archive_coverage_max, oes.qd_archive_coverage_max),
                    _format_pm(float(oe.colonies_max), float(oes.colonies_max), precision=1),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Replicates")
    lines.append("")
    for result in report.results:
        if not result.replicates:
            continue
        lines.append(f"### {result.name}")
        lines.append("")
        lines.append(
            "| seed | episodes | success_rate | avg_roi | merges | qd_cov_max | colonies_max |"
        )
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for rep in result.replicates:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(rep.seed),
                        str(rep.metrics.episodes),
                        f"{rep.metrics.success_rate:.3f}",
                        f"{rep.metrics.avg_roi:.3f}",
                        str(rep.open_endedness.merges_total),
                        f"{rep.open_endedness.qd_archive_coverage_max:.3f}",
                        str(rep.open_endedness.colonies_max),
                    ]
                )
                + " |"
            )
        lines.append("")

    report_path = output_root / "benchmark_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Evolora benchmark suite.")
    parser.add_argument(
        "--mode",
        choices=["ci", "paper_qwen3"],
        default="ci",
        help="Which suite to run (ci uses stub backend).",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=None,
        help="Optional YAML suite file (overrides --mode).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory root (default: artifacts_bench_<timestamp>).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override generations for all cases in suite.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Override comma-separated seeds for all cases (e.g. 1,2,3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = _load_suite(args.suite) if args.suite is not None else _default_suite(args.mode)
    if args.generations is not None:
        for case in suite.cases:
            case.generations = int(args.generations)
    if args.seeds is not None:
        seeds = [int(tok) for tok in args.seeds.split(",") if tok.strip()]
        for case in suite.cases:
            case.seeds = list(seeds)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    label = args.mode if args.suite is None else args.suite.stem
    output_root = args.output or Path(f"artifacts_bench_{label}_{ts}")
    report = run_benchmark_suite(suite=suite, output_root=output_root)
    report_path = output_root / "benchmark_report.json"
    output_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    md_path = _write_markdown_report(report, output_root=output_root)
    print("Wrote benchmark report to", report_path)
    print("Wrote benchmark summary to", md_path)
    for result in report.results:
        m = result.metrics
        oe = result.open_endedness
        print(
            f"- {result.name}: episodes={m.episodes} success_rate={m.success_rate:.3f} "
            f"avg_total={m.avg_total_reward:.3f} avg_cost={m.avg_energy_spent:.3f} "
            f"merges={oe.merges_total} qd_cov_max={oe.qd_archive_coverage_max:.3f}"
        )


if __name__ == "__main__":
    main()
