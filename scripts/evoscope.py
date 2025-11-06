#!/usr/bin/env python3
"""EvoScope: render a richer HTML dashboard for a run directory.

Reads gen_summaries.jsonl and assimilation.jsonl, ensures plots exist via
scripts/analyze_ecology_run.py and writes an index.html that embeds Chart.js
visualisations (ROI, merges, colonies, gating) alongside key summary tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from analyze_ecology_run import (
    load_jsonl,
    summarise_generations,
    summarise_assimilation,
    ensure_plots,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sorted_dict_items(mapping: Mapping[str, Any], *, limit: int | None = None) -> list[tuple[str, Any]]:
    items = sorted(mapping.items(), key=lambda kv: kv[1], reverse=True)
    return items if limit is None else items[:limit]


def build_html(
    summary: Dict[str, Any],
    records: List[Dict[str, Any]],
    assimilation_summary: Dict[str, Any],
    out_dir: Path,
) -> str:
    visuals = out_dir / "visuals"
    generations = [int(rec.get("generation", idx + 1)) for idx, rec in enumerate(records)]
    avg_roi = [_safe_float(rec.get("avg_roi")) for rec in records]
    avg_energy_cost = [_safe_float(rec.get("avg_energy_cost")) for rec in records]
    mean_energy_balance = [_safe_float(rec.get("mean_energy_balance")) for rec in records]
    merges = [_safe_int(rec.get("merges")) for rec in records]
    trials = [_safe_int(rec.get("trials_created")) for rec in records]
    promotions = [_safe_int(rec.get("promotions")) for rec in records]
    team_routes = [_safe_int(rec.get("team_routes")) for rec in records]
    team_promotions = [_safe_int(rec.get("team_promotions")) for rec in records]
    colonies = [_safe_int(rec.get("colonies")) for rec in records]
    lp_mix_active = [_safe_float(rec.get("lp_mix_active")) for rec in records]
    qd_archive = [_safe_int(rec.get("qd_archive_size", 0)) for rec in records]
    qd_archive_cov = [_safe_float(rec.get("qd_archive_coverage", 0.0)) * 100.0 for rec in records]
    latest_record = records[-1] if records else {}
    qd_top = summary.get("qd_archive_top") or latest_record.get("qd_archive_top") or []
    qd_top_html = "".join(
        f"<li>{entry['cell']} (bin {entry['bin']}): ROI {entry['roi']:.3f}, novelty {entry['novelty']:.2f}</li>"
        for entry in qd_top
    ) or "<li>N/A</li>"
    board_latest = summary.get("comms_board_latest") or latest_record.get("comms_board") or []
    board_html = "".join(
        f"<li>{entry['organelle_id']} → {entry.get('topic') or 'general'} · {entry['text']}</li>"
        for entry in board_latest
    ) or "<li>N/A</li>"
    gating_totals = summary.get("assimilation_gating_total", {}) or {}
    gating_sorted = _sorted_dict_items(gating_totals, limit=12)
    gating_labels = [label for label, _ in gating_sorted]
    gating_values = [int(value) for _, value in gating_sorted]
    gating_samples = summary.get("assimilation_gating_samples", []) or []
    gating_sample_rows = "".join(
        f"<tr><td>{item.get('generation','')}</td><td>{item.get('organelle','')}</td>"
        f"<td>{item.get('reason','')}</td><td>{item.get('details','')}</td></tr>"
        for item in gating_samples[-10:]
    )
    co_routing_top = summary.get("co_routing_totals") or {}
    co_routing_lines = "".join(
        f"<li>{pair}: {count}</li>" for pair, count in _sorted_dict_items(co_routing_top, limit=10)
    )
    history_latest = summary.get("assimilation_history_latest") or {}
    history_html = "".join(
        (
            f"<li>{key}: gen {rec.get('generation')} · uplift {float(rec.get('uplift')):+.3f}</li>"
            if isinstance(rec.get("uplift"), (int, float))
            else f"<li>{key}: gen {rec.get('generation')}</li>"
        )
        for key, rec in list(history_latest.items())[:8]
    ) or "<li>N/A</li>"
    tier_totals = summary.get("colony_tier_counts_total") or {}
    tier_lines = "".join(f"<li>Tier {tier}: {count}</li>" for tier, count in _sorted_dict_items(tier_totals))

    recent_rows = []
    for rec in records[-25:]:
        recent_rows.append(
            "<tr>"
            f"<td>{rec.get('generation')}</td>"
            f"<td>{_safe_float(rec.get('avg_roi')):.3f}</td>"
            f"<td>{_safe_int(rec.get('merges'))}</td>"
            f"<td>{_safe_int(rec.get('trials_created'))}</td>"
            f"<td>{_safe_int(rec.get('promotions'))}</td>"
            f"<td>{rec.get('qd_coverage', '')}</td>"
            f"<td>{_safe_int(rec.get('colonies'))}</td>"
            "</tr>"
        )
    img_tags = []
    for name in sorted(visuals.glob("*.png")):
        rel = name.relative_to(out_dir)
        img_tags.append(
            f"<div class='thumb'><img src='{rel.as_posix()}' alt='{name.name}' loading='lazy'></div>"
        )
    evo_data = {
        "generations": generations,
        "avg_roi": avg_roi,
        "avg_energy_cost": avg_energy_cost,
        "mean_energy_balance": mean_energy_balance,
        "merges": merges,
        "trials": trials,
        "promotions": promotions,
        "team_routes": team_routes,
        "team_promotions": team_promotions,
        "colonies": colonies,
        "lp_mix_active": lp_mix_active,
        "qd_archive": qd_archive,
        "qd_archive_cov": qd_archive_cov,
        "gating": {"labels": gating_labels, "values": gating_values},
    }
    assimilation_events = assimilation_summary.get("events", 0)
    assimilation_passes = assimilation_summary.get("passes", 0)
    assimilation_failures = assimilation_summary.get("failures", 0)
    assimilation_extra = []
    if "sample_size_mean" in assimilation_summary:
        assimilation_extra.append(f"Mean sample size: {assimilation_summary['sample_size_mean']:.1f}")
    if "ci_excludes_zero_rate" in assimilation_summary:
        assimilation_extra.append(
            f"CI excludes zero: {assimilation_summary['ci_excludes_zero_rate'] * 100:.1f}%"
        )
    if "power_mean" in assimilation_summary:
        assimilation_extra.append(f"Power proxy mean: {assimilation_summary['power_mean']:.2f}")
    if "dr_used" in assimilation_summary:
        assimilation_extra.append(f"DR uplift events: {assimilation_summary['dr_used']}")
    method_counts = assimilation_summary.get("methods") or {}
    method_lines = "".join(f"<li>{name}: {count}</li>" for name, count in _sorted_dict_items(method_counts))
    strata_lines = ""
    if assimilation_summary.get("dr_strata_top"):
        strata_lines = "".join(
            f"<li>{name}: {count}</li>"
            for name, count in assimilation_summary["dr_strata_top"]
        )
    # Top-line stats for easy scanning
    qd_last_cov = qd_archive_cov[-1] if qd_archive_cov else 0.0
    headline = (
        f"Generations: {summary.get('generations')} · "
        f"Avg ROI (mean): {_safe_float(summary.get('avg_roi_mean')):.3f} · "
        f"Merges: {summary.get('total_merges')} · "
        f"Trials: {summary.get('trials_total')} · "
        f"Promotions: {summary.get('promotions_total')} · "
        f"Colonies (last): {_safe_int(latest_record.get('colonies'))} · "
        f"QD coverage (last): {qd_last_cov:.1f}%"
    )

    assimilation_details = "; ".join(assimilation_extra)
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>EvoScope · {out_dir.name}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f7f7fb; color: #1c1c28; }}
    h1, h2, h3 {{ color: #111; }}
    section {{ margin-bottom: 32px; background: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 2px 12px rgba(15, 23, 42, 0.08); }}
    canvas {{ max-width: 100%; margin: 20px 0; }}
    ul {{ margin: 8px 0 0 20px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 0.95rem; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
    tr:hover {{ background: #f1f5f9; }}
    .thumbs {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .thumb img {{ border-radius: 10px; box-shadow: 0 1px 8px rgba(15, 23, 42, 0.12); max-height: 160px; }}
    .pill {{ display: inline-block; margin: 4px 8px 0 0; padding: 4px 10px; border-radius: 999px; background: #e2e8f0; }}
    .grid {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }}
    .metric-card {{ background: #0f172a; color: #f8fafc; border-radius: 14px; padding: 16px; }}
    .metric-card h4 {{ margin: 0 0 8px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; font-size: 0.8rem; color: rgba(248,250,252,0.7); }}
    .metric-card span {{ font-size: 1.8rem; font-weight: 600; }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.6/dist/chart.umd.min.js"></script>
</head>
<body>
  <section>
    <h1>EvoScope · {out_dir.name}</h1>
    <p>{headline}</p>
    <div class="grid">
      <div class="metric-card">
        <h4>Energy</h4>
        <span>{_safe_float(summary.get('avg_energy_mean')):.3f}</span>
        <div>Balance mean {_safe_float(summary.get('energy_balance_mean')):.3f}</div>
      </div>
      <div class="metric-card">
        <h4>Diversity</h4>
        <span>{summary.get('diversity_samples', 0)}</span>
        <div>Energy Gini {_safe_float(summary.get('diversity_energy_gini_mean')):.3f}</div>
      </div>
      <div class="metric-card">
        <h4>Assimilation events</h4>
        <span>{assimilation_events}</span>
        <div>Pass {assimilation_passes} · Fail {assimilation_failures}</div>
      </div>
      <div class="metric-card">
        <h4>Learning progress</h4>
        <span>{_safe_float(summary.get('lp_mix_active_last', 0.0)):.3f}</span>
        <div>Base mean {_safe_float(summary.get('lp_mix_base_mean', 0.0)):.3f}</div>
      </div>
    </div>
  </section>

  <section>
    <h2>Charts</h2>
    <canvas id="roiChart" height="160"></canvas>
    <canvas id="mergeChart" height="160"></canvas>
    <canvas id="colonyChart" height="160"></canvas>
    <canvas id="qdChart" height="160"></canvas>
    <canvas id="gatingChart" height="160"></canvas>
  </section>

  <section>
    <h2>Assimilation Summary</h2>
    <p>{assimilation_details if assimilation_details else "No additional assimilation metrics captured."}</p>
    <h3>Methods</h3>
    <ul>{method_lines or "<li>No assimilation methods recorded.</li>"}</ul>
    {"<h3>Top DR strata</h3><ul>" + strata_lines + "</ul>" if strata_lines else ""}
    <h3>Top gating reasons</h3>
    <ul>{"".join(f"<li>{label}: {value}</li>" for label, value in gating_sorted) or "<li>No gating recorded.</li>"}</ul>
    <h3>Recent gating samples</h3>
    <table>
      <thead><tr><th>Generation</th><th>Organelle</th><th>Reason</th><th>Details</th></tr></thead>
      <tbody>{gating_sample_rows or "<tr><td colspan='4'>No recent samples.</td></tr>"}</tbody>
    </table>
    <h3>Assimilation history (latest)</h3>
    <ul>{history_html}</ul>
  </section>

    <section>
        <h2>Team & Colony Signals</h2>
        <div class="grid">
          <div>
            <h3>Top co-routing pairs</h3>
            <ul>{co_routing_lines or "<li>N/A</li>"}</ul>
          </div>
          <div>
            <h3>Colony tiers</h3>
            <ul>{tier_lines or "<li>N/A</li>"}</ul>
          </div>
          <div>
            <h3>QD archive top bins</h3>
            <ul>{qd_top_html}</ul>
          </div>
          <div>
            <h3>Message board (latest)</h3>
            <ul>{board_html}</ul>
          </div>
        </div>
      </section>

  <section>
    <h2>Recent Generations</h2>
    <table>
      <thead><tr><th>Gen</th><th>ROI</th><th>Merges</th><th>Trials</th><th>Promotions</th><th>QD Coverage</th><th>Colonies</th></tr></thead>
      <tbody>{''.join(recent_rows)}</tbody>
    </table>
  </section>

  <section>
    <h2>Generated Plots</h2>
    <div class="thumbs">{''.join(img_tags) if img_tags else "<p>No PNG plots were found under visuals/</p>"}</div>
  </section>

  <script>
    const evoData = {json.dumps(evo_data, separators=(',', ':'), ensure_ascii=False)};
    const evoSummary = {json.dumps(summary, ensure_ascii=False)};
    const evoAssim = {json.dumps(assimilation_summary, ensure_ascii=False)};

    function makeLineChart(ctxId, labelData) {{
      const ctx = document.getElementById(ctxId);
      if (!ctx) return;
      const {{ labels, datasets, options }} = labelData;
      new Chart(ctx, {{
        type: 'line',
        data: {{ labels, datasets }},
        options: Object.assign({{ responsive: true, maintainAspectRatio: false, scales: {{ x: {{ ticks: {{ autoSkip: true, maxTicksLimit: 12 }} }} }} }}, options || {{}})
      }});
    }}

    function makeBarChart(ctxId, labelData) {{
      const ctx = document.getElementById(ctxId);
      if (!ctx) return;
      const {{ labels, datasets, options }} = labelData;
      new Chart(ctx, {{
        type: 'bar',
        data: {{ labels, datasets }},
        options: Object.assign({{ responsive: true, maintainAspectRatio: false }}, options || {{}})
      }});
    }}

    makeLineChart('roiChart', {{
      labels: evoData.generations,
      datasets: [
        {{
          label: 'Average ROI',
          data: evoData.avg_roi,
          fill: false,
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37, 99, 235, 0.2)',
          tension: 0.15,
        }},
        {{
          label: 'LP mix (active)',
          data: evoData.lp_mix_active,
          fill: false,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.2)',
          tension: 0.15,
        }}
      ],
      options: {{
        plugins: {{
          legend: {{ display: true }},
          tooltip: {{ mode: 'index', intersect: false }}
        }},
        scales: {{
          y: {{ beginAtZero: false }}
        }}
      }}
    }});

    makeBarChart('mergeChart', {{
      labels: evoData.generations,
      datasets: [
        {{
          type: 'bar',
          label: 'Merges',
          data: evoData.merges,
          backgroundColor: 'rgba(59, 130, 246, 0.6)',
        }},
        {{
          type: 'bar',
          label: 'Promotions',
          data: evoData.promotions,
          backgroundColor: 'rgba(168, 85, 247, 0.6)',
        }},
        {{
          type: 'line',
          label: 'Trials',
          data: evoData.trials,
          borderColor: '#f97316',
          backgroundColor: 'rgba(249, 115, 22, 0.2)',
          tension: 0.1,
          fill: false,
        }}
      ]
    }});

    makeLineChart('colonyChart', {{
      labels: evoData.generations,
      datasets: [
        {{
          label: 'Colonies',
          data: evoData.colonies,
          borderColor: '#0ea5e9',
          backgroundColor: 'rgba(14,165,233,0.2)',
          tension: 0.1,
        }},
        {{
          label: 'Team routes',
          data: evoData.team_routes,
          borderColor: '#f43f5e',
          backgroundColor: 'rgba(244,63,94,0.2)',
          tension: 0.1,
        }},
        {{
          label: 'Team promotions',
          data: evoData.team_promotions,
          borderColor: '#22c55e',
          backgroundColor: 'rgba(34,197,94,0.2)',
          tension: 0.1,
        }}
      ]
    }});

    makeLineChart('qdChart', {{
      labels: evoData.generations,
      datasets: [
        {{
          label: 'QD archive size',
          data: evoData.qd_archive,
          borderColor: '#8b5cf6',
          backgroundColor: 'rgba(139,92,246,0.2)',
          tension: 0.1,
          yAxisID: 'y',
        }},
        {{
          label: 'QD coverage (%)',
          data: evoData.qd_archive_cov,
          borderColor: '#0ea5e9',
          backgroundColor: 'rgba(14,165,233,0.15)',
          tension: 0.1,
          fill: false,
          yAxisID: 'y1',
        }}
      ],
      options: {{
        scales: {{
          y: {{ beginAtZero: true, title: {{ display: true, text: 'Archive size' }} }},
          y1: {{ beginAtZero: true, position: 'right', grid: {{ drawOnChartArea: false }}, title: {{ display: true, text: 'Coverage %' }} }}
        }}
      }}
    }});

    const gatingCtx = document.getElementById('gatingChart');
    if (gatingCtx && evoData.gating.labels.length > 0) {{
      new Chart(gatingCtx, {{
        type: 'bar',
        data: {{
          labels: evoData.gating.labels,
          datasets: [{{
            label: 'Assimilation gating counts',
            data: evoData.gating.values,
            backgroundColor: '#475569'
          }}]
        }},
        options: {{
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          scales: {{
            x: {{ beginAtZero: true }}
          }}
        }}
      }});
    }}
  </script>
</body>
</html>
"""
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EvoScope HTML for a run directory")
    parser.add_argument("run_dir", type=Path, help="Path to the run directory (contains gen_summaries.jsonl)")
    args = parser.parse_args()
    root = args.run_dir
    gen_path = root / "gen_summaries.jsonl"
    assim_path = root / "assimilation.jsonl"
    records = load_jsonl(gen_path)
    summary = summarise_generations(records)
    assim_summary = summarise_assimilation(assim_path)
    ensure_plots(records, root / "visuals")
    html = build_html(summary, records, assim_summary, root)
    (root / "index.html").write_text(html)
    print(f"Wrote {root / 'index.html'}")


if __name__ == "__main__":
    main()
