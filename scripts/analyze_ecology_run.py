#!/usr/bin/env python3
"""Generate analysis report and plots for an ecology run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    records: List[Dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No records found in {path}")
    return records


def summarise_generations(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    roi = [rec.get("avg_roi", 0.0) for rec in records]
    reward = [rec.get("avg_total", 0.0) for rec in records]
    energy = [rec.get("avg_energy_cost", 0.0) for rec in records]
    active = [rec.get("active", 0) for rec in records]
    bankrupt = [rec.get("bankrupt", 0) for rec in records]
    merges = [rec.get("merges", 0) for rec in records]
    culled = [rec.get("culled_bankrupt", 0) for rec in records]
    energy_balance_means = [rec.get("mean_energy_balance", 0.0) for rec in records]
    eval_records = [rec.get("evaluation") for rec in records if rec.get("evaluation")]
    eval_accuracy = [rec["accuracy"] for rec in eval_records]
    eval_correct = sum(rec.get("correct", 0) for rec in eval_records)
    eval_total = sum(rec.get("total", 0) for rec in eval_records)

    gating_totals: Dict[str, int] = {}
    for rec in records:
        gating = rec.get("assimilation_gating")
        if not gating:
            continue
        for key, value in gating.items():
            gating_totals[key] = gating_totals.get(key, 0) + int(value)

    tuning_records = [rec.get("assimilation_energy_tuning") for rec in records if rec.get("assimilation_energy_tuning")]
    energy_floor_mean = 0.0
    energy_floor_latest: Dict[str, Any] | None = None
    if tuning_records:
        floors = [float(item.get("energy_floor", 0.0)) for item in tuning_records]
        energy_floor_mean = mean(floors)
        energy_floor_latest = tuning_records[-1]

    diversity_records = [rec.get("diversity") for rec in records if rec.get("diversity")]
    diversity_gini_mean = 0.0
    diversity_effective_mean = 0.0
    diversity_max_share_mean = 0.0
    diversity_enforced_rate = 0.0
    if diversity_records:
        ginis = [float(item.get("energy_gini", 0.0)) for item in diversity_records]
        effective = [float(item.get("effective_population", 0.0)) for item in diversity_records]
        max_shares = [float(item.get("max_species_share", 0.0)) for item in diversity_records]
        enforced = [float(item.get("enforced", 0.0)) for item in diversity_records]
        diversity_gini_mean = mean(ginis)
        diversity_effective_mean = mean(effective)
        diversity_max_share_mean = mean(max_shares)
        diversity_enforced_rate = mean(enforced)

    eval_weight = None
    if eval_records:
        last = eval_records[-1]
        eval_weight = last.get("reward_weight")

    latest_record = records[-1]
    latest_gating_samples = latest_record.get("assimilation_gating_samples", [])
    latest_attempts = latest_record.get("assimilation_attempts", [])

    qd_coverage = latest_record.get("qd_coverage")
    roi_volatility = latest_record.get("roi_volatility")
    # Trials/promotions totals
    total_trials = sum(int(rec.get("trials_created", 0) or 0) for rec in records)
    total_promotions = sum(int(rec.get("promotions", 0) or 0) for rec in records)
    return {
        "generations": len(records),
        "avg_roi_mean": mean(roi),
        "avg_roi_median": median(roi),
        "avg_roi_min": min(roi),
        "avg_roi_max": max(roi),
        "avg_roi_std": pstdev(roi) if len(roi) > 1 else 0.0,
        "avg_reward_mean": mean(reward),
        "avg_reward_min": min(reward),
        "avg_reward_max": max(reward),
        "avg_energy_mean": mean(energy),
        "energy_balance_mean": mean(energy_balance_means),
        "energy_balance_min": min(energy_balance_means),
        "energy_balance_max": max(energy_balance_means),
        "active_min": min(active),
        "active_max": max(active),
        "bankrupt_min": min(bankrupt),
        "bankrupt_max": max(bankrupt),
        "culled_total": sum(culled),
        "culled_max": max(culled),
        "total_merges": sum(merges),
        "episodes_total": sum(rec.get("episodes", 0) for rec in records),
        "eval_events": len(eval_records),
        "eval_accuracy_mean": mean(eval_accuracy) if eval_accuracy else 0.0,
        "eval_correct": eval_correct,
        "eval_total": eval_total,
        "eval_reward_weight": eval_weight,
        "assimilation_gating_total": gating_totals,
        "assimilation_energy_floor_mean": energy_floor_mean,
        "assimilation_energy_floor_latest": energy_floor_latest,
        "diversity_samples": len(diversity_records),
        "diversity_energy_gini_mean": diversity_gini_mean,
        "diversity_effective_population_mean": diversity_effective_mean,
        "diversity_max_species_share_mean": diversity_max_share_mean,
        "diversity_enforced_rate": diversity_enforced_rate,
        "assimilation_gating_samples": latest_gating_samples[-5:],
        "assimilation_attempts": latest_attempts[-5:],
        "qd_coverage": qd_coverage,
        "roi_volatility": roi_volatility,
        "trials_total": total_trials,
        "promotions_total": total_promotions,
    }


def summarise_assimilation(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"events": 0, "passes": 0, "failures": 0}
    passes = 0
    failures = 0
    ci_excludes_zero = 0
    sample_sizes: list[int] = []
    powers: list[float] = []
    for_ci = 0
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") != "assimilation":
                continue
            if record.get("decision"):
                passes += 1
            else:
                failures += 1
            event = record
            ss = event.get("sample_size")
            if isinstance(ss, int):
                sample_sizes.append(ss)
            lo = event.get("ci_low")
            hi = event.get("ci_high")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                for_ci += 1
                if lo > 0 or hi < 0:
                    ci_excludes_zero += 1
            pw = event.get("power")
            if isinstance(pw, (int, float)):
                powers.append(float(pw))
    out: Dict[str, Any] = {
        "events": passes + failures,
        "passes": passes,
        "failures": failures,
    }
    if sample_sizes:
        out["sample_size_mean"] = float(mean(sample_sizes))
    if for_ci:
        out["ci_excludes_zero_rate"] = ci_excludes_zero / max(for_ci, 1)
    if powers:
        out["power_mean"] = float(mean(powers))
    return out


def ensure_plots(records: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generations = [rec["generation"] for rec in records]

    def plot_line(metric: str, ylabel: str) -> None:
        values = [rec.get(metric, 0.0) for rec in records]
        plt.figure(figsize=(8, 4))
        plt.plot(generations, values, marker="o", linewidth=1)
        plt.xlabel("Generation")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} over generations")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png")
        plt.close()

    for metric, ylabel in [
        ("avg_roi", "Average ROI"),
        ("avg_total", "Average total reward"),
        ("avg_energy_cost", "Average energy cost"),
        ("mean_energy_balance", "Mean energy balance"),
        ("active", "Active organelles"),
        ("bankrupt", "Bankrupt organelles"),
        ("merges", "Assimilation merges"),
        ("culled_bankrupt", "Culled organelles"),
    ]:
        plot_line(metric, ylabel)

    # Heatmaps for per-cell metrics if available
    if records[0].get("cells"):
        cells = sorted(records[0]["cells"].keys())
        metrics = ["difficulty", "success_ema", "price"]
        for metric in metrics:
            matrix = []
            for rec in records:
                cell_data = rec.get("cells", {})
                matrix.append([cell_data.get(cell, {}).get(metric, 0.0) for cell in cells])
            plt.figure(figsize=(max(6, len(cells)), 4))
            plt.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
            plt.colorbar(label=metric)
            plt.yticks(range(len(records)), generations)
            plt.xticks(range(len(cells)), cells, rotation=45, ha="right")
            plt.xlabel("Cell")
            plt.ylabel("Generation")
            plt.title(f"{metric} heatmap")
            plt.tight_layout()
            plt.savefig(output_dir / f"cells_{metric}.png")
            plt.close()


def write_report(summary: Dict[str, Any], assimilation: Dict[str, int], output_path: Path) -> None:
    lines = ["# Ecology Run Analysis", ""]
    lines.append(f"- Generations: {summary['generations']}")
    lines.append(f"- Total episodes: {summary['episodes_total']}")
    lines.append(
        f"- Average ROI: {summary['avg_roi_mean']:.3f} (median {summary['avg_roi_median']:.3f}, range {summary['avg_roi_min']:.3f} – {summary['avg_roi_max']:.3f}, σ {summary['avg_roi_std']:.3f})"
    )
    lines.append(
        f"- Average total reward: {summary['avg_reward_mean']:.3f} (range {summary['avg_reward_min']:.3f} – {summary['avg_reward_max']:.3f})"
    )
    lines.append(f"- Average energy cost: {summary['avg_energy_mean']:.3f}")
    lines.append(
        f"- Energy balance mean: {summary['energy_balance_mean']:.3f} (range {summary['energy_balance_min']:.3f} – {summary['energy_balance_max']:.3f})"
    )
    lines.append(
        f"- Active organelles per generation: {summary['active_min']} – {summary['active_max']} (bankrupt: {summary['bankrupt_min']} – {summary['bankrupt_max']})"
    )
    lines.append(f"- Bankruptcy culls: total {summary['culled_total']} (max per generation {summary['culled_max']})")
    lines.append(f"- Assimilation merges (per summary): {summary['total_merges']}")
    if summary["assimilation_gating_total"]:
        lines.append("- Assimilation gating totals:")
        for key, value in summary["assimilation_gating_total"].items():
            lines.append(f"  - {key}: {value}")
    latest = summary.get("assimilation_energy_floor_latest")
    if latest:
        floor = float(latest.get("energy_floor", 0.0))
        roi_thr = float(latest.get("energy_floor_roi", 0.0))
        lines.append(
            f"- Assimilation energy tuning: floor {floor:.3f}, ROI threshold {roi_thr:.3f}"
        )
    if assimilation["events"]:
        lines.append(
            f"- Assimilation tests: {assimilation['events']} (passes {assimilation['passes']}, failures {assimilation['failures']})"
        )
    gating_samples = summary.get("assimilation_gating_samples") or []
    if gating_samples:
        lines.append("- Recent gating snapshots:")
        for sample in gating_samples:
            reason = sample.get("reason")
            organelle = sample.get("organelle_id")
            generation = sample.get("generation")
            details = json.dumps(sample.get("details", {}), sort_keys=True)
            lines.append(f"  - gen {generation:03d} {organelle}: {reason} | {details}")
    attempt_samples = summary.get("assimilation_attempts") or []
    if attempt_samples:
        lines.append("- Recent assimilation attempts:")
        for attempt in attempt_samples:
            generation = int(attempt.get("generation", 0))
            organelle = attempt.get("organelle_id")
            cell = attempt.get("cell", {})
            uplift = float(attempt.get("uplift", 0.0) or 0.0)
            p_value = float(attempt.get("p_value", 0.0) or 0.0)
            passed = bool(attempt.get("passes_stat_test"))
            holdout_passed = bool(attempt.get("holdout_passed"))
            global_passed = bool(attempt.get("global_probe_passed"))
            top_up = attempt.get("top_up", {})
            top_up_status = top_up.get("status")
            lines.append(
                f"  - gen {generation:03d} {organelle}@{cell.get('family')}:{cell.get('depth')} "
                f"uplift={uplift:.3f} p={p_value:.3f} stat_pass={passed} holdout={holdout_passed} global={global_passed} topup={top_up_status}"
            )
    else:
        lines.append("- Assimilation tests: none recorded")
    if summary.get("diversity_samples", 0):
        lines.append(
            f"- Diversity: mean energy Gini {summary['diversity_energy_gini_mean']:.3f}, effective pop {summary['diversity_effective_population_mean']:.2f}, max species share {summary['diversity_max_species_share_mean']:.3f}, enforcement rate {summary['diversity_enforced_rate']*100:.1f}%"
        )
    if summary.get("qd_coverage") is not None:
        lines.append(f"- QD coverage: {summary['qd_coverage']}")
    if summary.get("roi_volatility") is not None:
        lines.append(f"- ROI volatility (std across organelles): {summary['roi_volatility']:.3f}")
    if summary.get("trials_total") or summary.get("promotions_total"):
        lines.append(f"- Trials/promotions: {summary.get('trials_total', 0)} / {summary.get('promotions_total', 0)}")
    if summary["eval_events"]:
        accuracy_pct = summary["eval_accuracy_mean"] * 100
        lines.append(
            f"- Evaluation accuracy: {accuracy_pct:.2f}% ({summary['eval_correct']}/{summary['eval_total']})"
        )
        if summary["eval_reward_weight"] is not None:
            lines.append(f"  (reward weight {summary['eval_reward_weight']})")
    else:
        lines.append("- Evaluation accuracy: n/a")

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse an ecology run directory")
    parser.add_argument("run_dir", type=Path, help="Directory containing gen_summaries.jsonl")
    parser.add_argument(
        "--plots", action="store_true", help="Generate plots (saved under <run_dir>/visuals)", default=False
    )
    parser.add_argument(
        "--report", action="store_true", help="Write report.md in the run directory", default=False
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    gen_records = load_jsonl(run_dir / "gen_summaries.jsonl")
    summary = summarise_generations(gen_records)
    assimilation_summary = summarise_assimilation(run_dir / "assimilation.jsonl")

    print("Generations:", summary["generations"])
    print("Total episodes:", summary["episodes_total"])
    print(
        "Average ROI (mean/median/min/max):",
        f"{summary['avg_roi_mean']:.3f}",
        f"{summary['avg_roi_median']:.3f}",
        f"{summary['avg_roi_min']:.3f}",
        f"{summary['avg_roi_max']:.3f}",
    )
    print(
        "Average total reward (mean/min/max):",
        f"{summary['avg_reward_mean']:.3f}",
        f"{summary['avg_reward_min']:.3f}",
        f"{summary['avg_reward_max']:.3f}",
    )
    print("Average energy cost:", f"{summary['avg_energy_mean']:.3f}")
    print(
        "Average energy balance (mean/min/max):",
        f"{summary['energy_balance_mean']:.3f}",
        f"{summary['energy_balance_min']:.3f}",
        f"{summary['energy_balance_max']:.3f}",
    )
    print(
        "Active organelles range:", summary["active_min"], "-", summary["active_max"],
        ", bankrupt:", summary["bankrupt_min"], "-", summary["bankrupt_max"],
    )
    print("Total merges:", summary["total_merges"])
    print("Bankruptcy culls (total/max):", summary["culled_total"], summary["culled_max"])
    if summary["assimilation_gating_total"]:
        print("Assimilation gating totals:")
        for key, value in summary["assimilation_gating_total"].items():
            print(f"  {key}: {value}")
    if summary["eval_events"]:
        accuracy_pct = summary["eval_accuracy_mean"] * 100
        print(
            "Evaluation accuracy (avg | total):",
            f"{accuracy_pct:.2f}%",
            f"{summary['eval_correct']}/{summary['eval_total']}"
        )
    else:
        print("Evaluation accuracy: n/a")
    if assimilation_summary["events"]:
        print(
            "Assimilation events:",
            assimilation_summary["events"],
            "passes:", assimilation_summary["passes"],
            "failures:", assimilation_summary["failures"],
        )
        if "sample_size_mean" in assimilation_summary:
            print("Assimilation mean sample size:", f"{assimilation_summary['sample_size_mean']:.1f}")
        if "ci_excludes_zero_rate" in assimilation_summary:
            print("Uplift CI excludes zero (rate):", f"{assimilation_summary['ci_excludes_zero_rate']*100:.1f}%")
        if "power_mean" in assimilation_summary:
            print("Power (proxy) mean:", f"{assimilation_summary['power_mean']:.2f}")
    else:
        print("No assimilation events recorded")
    if summary.get("trials_total") or summary.get("promotions_total"):
        print("Trials created:", summary.get("trials_total", 0), "Promotions:", summary.get("promotions_total", 0))

    if args.plots:
        ensure_plots(gen_records, run_dir / "visuals")
        print("Plots saved under", run_dir / "visuals")

    if args.report:
        write_report(summary, assimilation_summary, run_dir / "report.md")
        print("Report written to", run_dir / "report.md")


if __name__ == "__main__":
    main()
