#!/usr/bin/env python3
"""Inspect merge audit telemetry for an ecology run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from analyze_ecology_run import load_jsonl, summarise_generations


def _format_cell(cell: Dict[str, Any] | None) -> str:
    if not isinstance(cell, dict):
        return "?"
    family = cell.get("family", "?")
    depth = cell.get("depth", "?")
    return f"{family}:{depth}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise merge audit results for a run directory")
    parser.add_argument("run_dir", type=Path, help="Run directory containing gen_summaries.jsonl")
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Display the top-N regressions (sorted by delta ascending).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Only show audits with delta <= threshold (default: 0.0).",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Optional path to export all audit records as JSON.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    gen_path = run_dir / "gen_summaries.jsonl"
    if not gen_path.exists():
        raise SystemExit(f"Missing gen_summaries.jsonl in {run_dir}")

    gen_records = load_jsonl(gen_path)
    summary = summarise_generations(gen_records)
    audits: List[Dict[str, Any]] = list(summary.get("merge_audit_records", []))

    if not audits:
        print("No merge audits recorded for this run.")
        return

    total = int(summary.get("merge_audit_total", len(audits)) or 0)
    delta_mean = float(summary.get("merge_audit_delta_mean", 0.0) or 0.0)
    print(f"Merge audits recorded: {total}")
    print(f"Mean ΔROI (post - pre): {delta_mean:+.4f}")

    if args.export:
        export_path = args.export
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(audits, indent=2))
        print(f"Wrote {len(audits)} audit records to {export_path}")

    filtered = [record for record in audits if isinstance(record.get("delta"), (int, float))]
    if args.threshold is not None:
        filtered = [record for record in filtered if float(record["delta"]) <= args.threshold]

    filtered.sort(key=lambda rec: float(rec.get("delta", 0.0)))
    top_n = filtered[: max(1, args.top)]

    if not top_n:
        print("No audits matched the provided threshold.")
        return

    print(f"\nWorst {len(top_n)} audits (delta <= {args.threshold}):")
    header = "{:>4} {:>4} {:>12} {:>12} {:>12} {:>8}".format("Gen", "Tasks", "Pre ROI", "Post ROI", "Delta", "Org")
    print(header)
    print("-" * len(header))
    for record in top_n:
        gen = int(record.get("generation", 0))
        tasks = int(record.get("tasks", 0))
        pre_roi = record.get("pre_roi")
        post_roi = record.get("post_roi")
        delta = record.get("delta")
        organelle = record.get("organelle_id", "?")
        cell = _format_cell(record.get("cell"))
        print(
            "{:>4} {:>4} {:>12} {:>12} {:>12} {:>8} {:>10}".format(
                gen,
                tasks,
                f"{pre_roi:.4f}" if isinstance(pre_roi, (int, float)) else str(pre_roi),
                f"{post_roi:.4f}" if isinstance(post_roi, (int, float)) else str(post_roi),
                f"{float(delta):+.4f}" if isinstance(delta, (int, float)) else str(delta),
                organelle,
                cell,
            )
        )

    deltas = [float(rec["delta"]) for rec in filtered if isinstance(rec.get("delta"), (int, float))]
    if deltas:
        print(f"\nFiltered mean ΔROI: {mean(deltas):+.4f}")


if __name__ == "__main__":
    main()

