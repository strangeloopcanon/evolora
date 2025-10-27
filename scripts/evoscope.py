#!/usr/bin/env python3
"""EvoScope: generate a simple HTML dashboard for a run directory.

Reads gen_summaries.jsonl and assimilation.jsonl, ensures plots exist via
scripts/analyze_ecology_run.py and writes an index.html summarizing KPIs with
links to the generated PNGs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from analyze_ecology_run import (
    load_jsonl,
    summarise_generations,
    summarise_assimilation,
    ensure_plots,
)


def build_html(summary: Dict[str, Any], records: List[Dict[str, Any]], out_dir: Path) -> str:
    visuals = out_dir / "visuals"
    imgs = [
        "avg_roi.png",
        "avg_total.png",
        "avg_energy_cost.png",
        "mean_energy_balance.png",
        "active.png",
        "bankrupt.png",
        "merges.png",
        "culled_bankrupt.png",
    ]
    rows = []
    for rec in records[-20:]:
        rows.append(
            f"<tr><td>{rec.get('generation')}</td><td>{rec.get('avg_roi', 0):.3f}</td>"
            f"<td>{rec.get('merges', 0)}</td><td>{rec.get('trials_created', 0)}</td>"
            f"<td>{rec.get('promotions', 0)}</td><td>{rec.get('qd_coverage', '')}</td></tr>"
        )
    gating = summary.get("assimilation_gating_total", {}) or {}
    gating_lines = "".join(f"<li>{k}: {v}</li>" for k, v in gating.items())
    img_tags = "".join(
        f"<div style='display:inline-block;margin:6px'><img src='visuals/{name}' alt='{name}' height='180'></div>"
        for name in imgs
        if (visuals / name).exists()
    )
    return f"""
<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>EvoScope</title>
<style>body{{font-family:system-ui,Arial,sans-serif;margin:16px}} th,td{{padding:4px 8px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc}}</style>
</head><body>
<h2>EvoScope: {out_dir.name}</h2>
<p>Generations: {summary.get('generations')} | Avg ROI mean: {summary.get('avg_roi_mean'):.3f} | Total merges: {summary.get('total_merges')} | Trials: {summary.get('trials_total')} | Promotions: {summary.get('promotions_total')}</p>
<p>Energy mean: {summary.get('avg_energy_mean'):.3f} | Balance mean: {summary.get('energy_balance_mean'):.3f} | Diversity samples: {summary.get('diversity_samples')}</p>
<p>QD coverage: {summary.get('qd_coverage')} | QD archive size: {records[-1].get('qd_archive_size', 0)} | ROI volatility: {records[-1].get('roi_volatility', 0.0):.3f}</p>
<p>LP mix active (last): {summary.get('lp_mix_active_last', 0.0):.3f} | Base mean: {summary.get('lp_mix_base_mean', 0.0):.3f} | Colonies: {records[-1].get('colonies', 0)}</p>
<h3>Assimilation gating totals</h3>
<ul>{gating_lines}</ul>
<h3>Recent generations</h3>
<table><thead><tr><th>Gen</th><th>ROI</th><th>Merges</th><th>Trials</th><th>Promotions</th><th>QD Coverage</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h3>Curves</h3>
{img_tags}
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EvoScope HTML for a run directory")
    parser.add_argument("run_dir", type=Path, help="Path to the run directory (contains gen_summaries.jsonl)")
    args = parser.parse_args()
    root = args.run_dir
    gen_path = root / "gen_summaries.jsonl"
    assim_path = root / "assimilation.jsonl"
    records = load_jsonl(gen_path)
    summary = summarise_generations(records)
    if assim_path.exists():
        _ = summarise_assimilation(assim_path)
    ensure_plots(records, root / "visuals")
    html = build_html(summary, records, root)
    (root / "index.html").write_text(html)
    print(f"Wrote {root / 'index.html'}")


if __name__ == "__main__":
    main()
