#!/usr/bin/env python3
"""Quick comparison of two ecology runs.

Usage:
  scripts/compare_runs.py <run_a_dir> <run_b_dir>

Reports diffs on ROI, merges/promotions, assimilation attempts, CI/power, and eval accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def summarise_run(run_dir: Path) -> dict:
    gens = load_jsonl(run_dir / "gen_summaries.jsonl")
    assim = load_jsonl(run_dir / "assimilation.jsonl")
    roi = [rec.get("avg_roi", 0.0) for rec in gens]
    merges = sum(int(rec.get("merges", 0) or 0) for rec in gens)
    trials = sum(int(rec.get("trials_created", 0) or 0) for rec in gens)
    promos = sum(int(rec.get("promotions", 0) or 0) for rec in gens)
    evals = [rec.get("evaluation") for rec in gens if rec.get("evaluation")]
    eval_correct = sum(int(rec.get("correct", 0) or 0) for rec in evals)
    eval_total = sum(int(rec.get("total", 0) or 0) for rec in evals)
    events = [rec for rec in assim if rec.get("type") == "assimilation"]
    passes = sum(1 for r in events if r.get("decision"))
    failures = sum(1 for r in events if not r.get("decision"))
    cis = [r for r in events if isinstance(r.get("ci_low"), (int, float)) and isinstance(r.get("ci_high"), (int, float))]
    ci_excl_zero = sum(1 for r in cis if (r["ci_low"] > 0) or (r["ci_high"] < 0))
    powers = [float(r.get("power", 0.0)) for r in events if isinstance(r.get("power"), (int, float))]
    return {
        "generations": len(gens),
        "roi_mean": float(mean(roi)) if roi else 0.0,
        "merges": merges,
        "trials": trials,
        "promotions": promos,
        "assim_events": len(events),
        "assim_passes": passes,
        "assim_failures": failures,
        "ci_excl_zero_rate": (ci_excl_zero / max(len(cis), 1)) if cis else 0.0,
        "power_mean": float(mean(powers)) if powers else 0.0,
        "eval_correct": eval_correct,
        "eval_total": eval_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two run directories")
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    args = ap.parse_args()
    a = summarise_run(args.run_a)
    b = summarise_run(args.run_b)
    def pct(x):
        return f"{x*100:.1f}%"
    print("A gens:", a["generations"], "B gens:", b["generations"]) 
    print("ROI mean:", f"{a['roi_mean']:.3f}", "vs", f"{b['roi_mean']:.3f}")
    print("Merges:", a["merges"], "vs", b["merges"], "/ Promotions:", a["promotions"], "vs", b["promotions"]) 
    print("Trials:", a["trials"], "vs", b["trials"]) 
    print("Assim events (pass/fail):", f"{a['assim_events']} ({a['assim_passes']}/{a['assim_failures']})", "vs", f"{b['assim_events']} ({b['assim_passes']}/{b['assim_failures']})")
    print("CI excl zero:", pct(a["ci_excl_zero_rate"]), "vs", pct(b["ci_excl_zero_rate"]))
    print("Power mean:", f"{a['power_mean']:.2f}", "vs", f"{b['power_mean']:.2f}")
    print("Eval:", f"{a['eval_correct']}/{a['eval_total']}", "vs", f"{b['eval_correct']}/{b['eval_total']}")


if __name__ == "__main__":
    main()

