#!/usr/bin/env python3
"""Package paper-candidate artifacts (tables + curated plots) into a tracked folder.

This keeps long-run `artifacts_*` directories gitignored while preserving the
high-signal outputs needed for writeups and talks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No records found in {path}")
    return records


def _git_commit(repo_root: Path) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or None
        )
    except Exception:
        return None


def _summarize_run(run_dir: Path) -> dict[str, object]:
    records = _load_jsonl(run_dir / "gen_summaries.jsonl")
    roi_series = [float(rec.get("avg_roi", 0.0) or 0.0) for rec in records]
    merges_series = [int(rec.get("merges", 0) or 0) for rec in records]
    colonies_series = [int(rec.get("colonies", 0) or 0) for rec in records]
    qd_series = [float(rec.get("qd_archive_coverage", 0.0) or 0.0) for rec in records]
    episodes_total = int(sum(int(rec.get("episodes", 0) or 0) for rec in records))
    holdout: dict[str, object] | None = None
    holdout_path = run_dir / "final_holdout.json"
    if holdout_path.exists():
        try:
            holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
        except Exception:
            holdout = None
    return {
        "generations": len(records),
        "episodes_total": episodes_total,
        "avg_roi_mean": float(mean(roi_series)) if roi_series else 0.0,
        "avg_roi_min": float(min(roi_series)) if roi_series else 0.0,
        "avg_roi_max": float(max(roi_series)) if roi_series else 0.0,
        "merges_total": int(sum(merges_series)),
        "colonies_max": int(max(colonies_series)) if colonies_series else 0,
        "qd_coverage_last": float(qd_series[-1]) if qd_series else 0.0,
        "qd_coverage_max": float(max(qd_series)) if qd_series else 0.0,
        "holdout_accuracy": (
            float(holdout.get("accuracy", 0.0)) if isinstance(holdout, dict) else None
        ),
        "holdout_avg_cost": (
            float(holdout.get("avg_cost", 0.0)) if isinstance(holdout, dict) else None
        ),
        "holdout_cost_per_correct": (
            float(holdout.get("cost_per_correct", 0.0)) if isinstance(holdout, dict) else None
        ),
        "holdout_sample_size": (
            int(holdout.get("sample_size", 0)) if isinstance(holdout, dict) else None
        ),
        "roi_series": roi_series,
    }


def _maybe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return True


def _format_pct(value: float) -> str:
    if value <= 1.0:
        return f"{value * 100.0:.1f}%"
    return f"{value:.1f}%"


def _format_optional(value: object, fmt: str) -> str:
    if value is None:
        return "-"
    try:
        return format(float(value), fmt)
    except Exception:
        return "-"


def _pack_date_label(out_dir: Path) -> str:
    # Try to infer YYYY-MM-DD from folder names like paper_qwen3_20251216.
    m = re.search(r"(\d{8})", out_dir.name)
    if m:
        stamp = m.group(1)
        try:
            parsed = dt.datetime.strptime(stamp, "%Y%m%d").date()
            return parsed.isoformat()
        except Exception:
            pass
    return dt.datetime.now(tz=dt.timezone.utc).date().isoformat()


def _write_readme(
    out_dir: Path,
    *,
    commit: str | None,
    runs: dict[str, Path],
    summaries: dict[str, dict[str, object]],
    comparison_plot: Path | None,
) -> None:
    lines: list[str] = []
    lines.append(f"# Paper Candidate Pack — Qwen3-0.6B ({_pack_date_label(out_dir)})")
    lines.append("")
    if commit:
        lines.append(f"- git_commit: `{commit[:7]}`")
    lines.append("- source_runs:")
    for name, path in runs.items():
        lines.append(f"  - {name}: `{path}`")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append(
        "| condition | gens | episodes | avg_roi (mean) | merges_total | colonies_max | qd_cov_max | holdout_acc | holdout_avg_cost |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in ["frozen", "single", "ecology"]:
        s = summaries[name]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(s["generations"]),
                    str(s["episodes_total"]),
                    f"{float(s['avg_roi_mean']):.3f}",
                    str(s["merges_total"]),
                    str(s["colonies_max"]),
                    _format_pct(float(s["qd_coverage_max"])),
                    _format_optional(s.get("holdout_accuracy"), ".3f"),
                    _format_optional(s.get("holdout_avg_cost"), ".3f"),
                ]
            )
            + " |"
        )
    lines.append("")
    if comparison_plot is not None and comparison_plot.exists():
        rel = comparison_plot.relative_to(out_dir)
        lines.append("## ROI Comparison")
        lines.append("")
        lines.append(f"![ROI comparison]({rel.as_posix()})")
        lines.append("")
    ecology_highlights = [
        ("Ecology ROI", out_dir / "plots" / "ecology" / "avg_roi.png"),
        ("Ecology merges", out_dir / "plots" / "ecology" / "merges.png"),
        ("Ecology colonies", out_dir / "plots" / "ecology" / "colonies_count.png"),
        ("Ecology QD coverage", out_dir / "plots" / "ecology" / "qd_archive_coverage.png"),
    ]
    existing = [(title, path) for title, path in ecology_highlights if path.exists()]
    if existing:
        lines.append("## Ecology Highlights")
        lines.append("")
        for title, path in existing:
            rel = path.relative_to(out_dir)
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{title}]({rel.as_posix()})")
            lines.append("")
    lines.append("## Included Files")
    lines.append("")
    lines.append("- `reports/`: copied `report.md` from each run")
    lines.append("- `reports/`: copied `final_holdout.md` when present")
    lines.append("- `plots/`: curated plots copied from each run + an aggregate comparison plot")
    lines.append("- `summary.json`: machine-readable summary extracted from `gen_summaries.jsonl`")
    lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_roi_comparison_plot(
    out_dir: Path, summaries: dict[str, dict[str, object]]
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    plt.figure(figsize=(10, 5))
    colors = {"frozen": "#6b7280", "single": "#ef4444", "ecology": "#22c55e"}
    for name in ["frozen", "single", "ecology"]:
        series = summaries[name].get("roi_series") or []
        if not isinstance(series, list) or not series:
            continue
        xs = list(range(1, len(series) + 1))
        ys = [float(v) for v in series]
        plt.plot(xs, ys, label=name, linewidth=2, color=colors.get(name))
    plt.axhline(1.0, linestyle="--", linewidth=1, color="#111827", alpha=0.5)
    plt.xlabel("Generation")
    plt.ylabel("avg_roi")
    plt.title("Average ROI by Generation")
    plt.legend()
    plt.tight_layout()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / "roi_comparison.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True, help="Frozen run directory")
    parser.add_argument("--single", type=Path, required=True, help="Single-adapter run directory")
    parser.add_argument("--ecology", type=Path, required=True, help="Ecology run directory")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory (tracked, e.g. results/paper_qwen3_20251215)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir: Path = args.output
    runs = {"frozen": args.frozen, "single": args.single, "ecology": args.ecology}

    summaries = {name: _summarize_run(path) for name, path in runs.items()}
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    for name, run_dir in runs.items():
        _maybe_copy(run_dir / "report.md", out_dir / "reports" / f"{name}_report.md")
        _maybe_copy(
            run_dir / "final_holdout.md",
            out_dir / "reports" / f"{name}_final_holdout.md",
        )

    curated = [
        "avg_roi.png",
        "mean_energy_balance.png",
        "merges.png",
        "colonies_count.png",
        "colonies_membership_stack.png",
        "qd_archive_coverage.png",
        "team_routes.png",
    ]
    holdout_curated = [
        "final_holdout_accuracy_by_family.png",
        "final_holdout_cost_by_family.png",
    ]
    for name, run_dir in runs.items():
        for filename in curated:
            _maybe_copy(
                run_dir / "visuals" / filename,
                out_dir / "plots" / name / filename,
            )
        for filename in holdout_curated:
            _maybe_copy(
                run_dir / "visuals" / filename,
                out_dir / "plots" / name / filename,
            )

    comparison_plot = _make_roi_comparison_plot(out_dir, summaries)

    commit = _git_commit(repo_root)
    payload = {
        "git_commit": (commit[:7] if commit else None),
        "runs": {name: str(path) for name, path in runs.items()},
        "summaries": {
            name: {k: v for k, v in s.items() if k != "roi_series"} for name, s in summaries.items()
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_readme(
        out_dir, commit=commit, runs=runs, summaries=summaries, comparison_plot=comparison_plot
    )
    print("Wrote paper pack to", out_dir)


if __name__ == "__main__":
    main()
