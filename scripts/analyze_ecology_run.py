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
    lp_mix_base = [rec.get("lp_mix_base", 0.0) for rec in records]
    lp_mix_active = [rec.get("lp_mix_active", 0.0) for rec in records]
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
    # Aggregate gating reasons seen across snapshots (best-effort; snapshots are bounded)
    gating_reason_counts: Dict[str, int] = {}
    for rec in records:
        for sample in rec.get("assimilation_gating_samples", []) or []:
            reason = sample.get("reason")
            if isinstance(reason, str):
                gating_reason_counts[reason] = gating_reason_counts.get(reason, 0) + 1
    latest_attempts = latest_record.get("assimilation_attempts", [])
    colonies_meta = latest_record.get("colonies_meta")
    # Colonies timeline
    colonies_count_series = [int(rec.get("colonies", 0) or 0) for rec in records]
    colonies_avg_size_series: List[float] = []
    for rec in records:
        meta = rec.get("colonies_meta") or {}
        if isinstance(meta, dict) and meta:
            sizes = []
            for v in meta.values():
                try:
                    sizes.append(len(v.get("members", [])))
                except Exception:
                    continue
            colonies_avg_size_series.append(sum(sizes) / max(1, len(sizes)))
        else:
            colonies_avg_size_series.append(0.0)

    qd_coverage = latest_record.get("qd_coverage")
    roi_volatility = latest_record.get("roi_volatility")
    policy_applied = sum(1 for rec in records if rec.get("policy_applied"))
    policy_attempts_total = sum(int(rec.get("policy_attempts", 0) or 0) for rec in records)
    policy_parsed_total = sum(int(rec.get("policy_parsed", 0) or 0) for rec in records)
    # Policy-conditioned ROI
    roi_when_policy: List[float] = []
    roi_when_no_policy: List[float] = []
    fields_agg: Dict[str, int] = {}
    budget_vals: List[float] = []
    reserve_vals: List[float] = []
    for rec, r in zip(records, roi):
        if rec.get("policy_applied"):
            roi_when_policy.append(float(r))
            fields = rec.get("policy_fields_used") or {}
            if isinstance(fields, dict):
                for k, v in fields.items():
                    try:
                        fields_agg[k] = fields_agg.get(k, 0) + int(v)
                    except Exception:
                        continue
            bf = rec.get("policy_budget_frac_avg")
            rr = rec.get("policy_reserve_ratio_avg")
            if isinstance(bf, (int, float)):
                budget_vals.append(float(bf))
            if isinstance(rr, (int, float)):
                reserve_vals.append(float(rr))
        else:
            roi_when_no_policy.append(float(r))
    # Trials/promotions totals
    total_trials = sum(int(rec.get("trials_created", 0) or 0) for rec in records)
    total_promotions = sum(int(rec.get("promotions", 0) or 0) for rec in records)
    # Team metrics
    team_routes_series: List[int] = [int(rec.get("team_routes", 0) or 0) for rec in records]
    team_promotions_series: List[int] = [int(rec.get("team_promotions", 0) or 0) for rec in records]
    team_routes_total = sum(team_routes_series)
    team_promotions_total = sum(team_promotions_series)
    # Co-routing totals across generations
    co_routing_totals: Dict[str, int] = {}
    for rec in records:
        top = rec.get("co_routing_top") or {}
        if isinstance(top, dict):
            for pair, cnt in top.items():
                try:
                    co_routing_totals[str(pair)] = co_routing_totals.get(str(pair), 0) + int(cnt)
                except Exception:
                    continue
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
        "lp_mix_base_mean": mean(lp_mix_base) if lp_mix_base else 0.0,
        "lp_mix_active_mean": mean(lp_mix_active) if lp_mix_active else 0.0,
        "lp_mix_active_last": lp_mix_active[-1] if lp_mix_active else 0.0,
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
        "assimilation_gating_reasons_samples": gating_reason_counts,
        "assimilation_attempts": latest_attempts[-5:],
        "qd_coverage": qd_coverage,
        "roi_volatility": roi_volatility,
        "trials_total": total_trials,
        "promotions_total": total_promotions,
        "team_routes_series": team_routes_series,
        "team_promotions_series": team_promotions_series,
        "team_routes_total": team_routes_total,
        "team_promotions_total": team_promotions_total,
        "co_routing_totals": dict(sorted(co_routing_totals.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "colonies_meta": colonies_meta,
        "colonies_count_series": colonies_count_series,
        "colonies_avg_size_mean": (sum(colonies_avg_size_series) / max(1, len(colonies_avg_size_series))) if colonies_avg_size_series else 0.0,
        "policy_applied_total": policy_applied,
        "policy_parse_attempts_total": int(policy_attempts_total),
        "policy_parse_parsed_total": int(policy_parsed_total),
        "policy_roi_mean_when_applied": (sum(roi_when_policy) / max(1, len(roi_when_policy))) if roi_when_policy else 0.0,
        "policy_roi_mean_when_not": (sum(roi_when_no_policy) / max(1, len(roi_when_no_policy))) if roi_when_no_policy else 0.0,
        "policy_fields_used_total": fields_agg,
        "policy_budget_frac_mean": (sum(budget_vals) / max(1, len(budget_vals))) if budget_vals else 0.0,
        "policy_reserve_ratio_mean": (sum(reserve_vals) / max(1, len(reserve_vals))) if reserve_vals else 0.0,
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
    method_counts: Dict[str, int] = {}
    dr_used = 0
    strata_agg: Dict[str, int] = {}
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
            method = event.get("method")
            if isinstance(method, str):
                method_counts[method] = method_counts.get(method, 0) + 1
            if bool(event.get("dr_used")):
                dr_used += 1
            strata = event.get("strata")
            if isinstance(strata, dict):
                for name, sizes in strata.items():
                    try:
                        paired = int(sizes.get("paired", 0))
                        strata_agg[name] = strata_agg.get(name, 0) + paired
                    except Exception:
                        continue
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
    if method_counts:
        out["methods"] = method_counts
    if dr_used:
        out["dr_used"] = dr_used
    if strata_agg:
        # Keep top 6 contributing strata by total paired count
        top = sorted(strata_agg.items(), key=lambda kv: kv[1], reverse=True)[:6]
        out["dr_strata_top"] = top
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
        ("lp_mix_active", "LP mix active"),
    ]:
        plot_line(metric, ylabel)

    # Policy parsed plot if available
    if any("policy_parsed" in rec for rec in records):
        values = [int(rec.get("policy_parsed", 0) or 0) for rec in records]
        plt.figure(figsize=(8, 4))
        plt.plot(generations, values, marker="o", linewidth=1)
        plt.xlabel("Generation"); plt.ylabel("Policy parsed count"); plt.title("Policy parsed per generation")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(output_dir / "policy_parsed.png"); plt.close()

    # Colonies plots
    colonies = [int(rec.get("colonies", 0) or 0) for rec in records]
    plt.figure(figsize=(8, 4))
    plt.plot(generations, colonies, marker="o", linewidth=1)
    plt.xlabel("Generation"); plt.ylabel("Colonies count"); plt.title("Colonies over generations")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(output_dir / "colonies_count.png"); plt.close()

    avg_sizes: List[float] = []
    for rec in records:
        meta = rec.get("colonies_meta") or {}
        if isinstance(meta, dict) and meta:
            sizes: List[int] = []
            for v in meta.values():
                try:
                    sizes.append(len(v.get("members", [])))
                except Exception:
                    pass
            avg_sizes.append(sum(sizes) / max(1, len(sizes)))
        else:
            avg_sizes.append(0.0)
    plt.figure(figsize=(8, 4))
    plt.plot(generations, avg_sizes, marker="o", linewidth=1)
    plt.xlabel("Generation"); plt.ylabel("Avg colony size"); plt.title("Colony size over generations")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(output_dir / "colonies_avg_size.png"); plt.close()

    # Colony pots over time (total and per top colonies)
    # Collect all colony ids
    all_ids = set()
    per_gen_meta: List[Dict[str, Any]] = []
    for rec in records:
        meta = rec.get("colonies_meta") or {}
        if isinstance(meta, dict):
            per_gen_meta.append(meta)
            all_ids.update(meta.keys())
        else:
            per_gen_meta.append({})
    if all_ids:
        # Build pot and membership series aligned to generations
        pot_series: Dict[str, List[float]] = {cid: [] for cid in all_ids}
        mem_series: Dict[str, List[int]] = {cid: [] for cid in all_ids}
        for meta in per_gen_meta:
            for cid in all_ids:
                entry = meta.get(cid) or {}
                try:
                    pot_series[cid].append(float(entry.get("pot", 0.0)))
                except Exception:
                    pot_series[cid].append(0.0)
                try:
                    mem_series[cid].append(int(len(entry.get("members", []))))
                except Exception:
                    mem_series[cid].append(0)
        # Total pot
        total_pot = [sum(pot_series[cid][i] for cid in all_ids) for i in range(len(records))]
        plt.figure(figsize=(8, 4))
        plt.plot(generations, total_pot, linewidth=1.5)
        plt.xlabel("Generation"); plt.ylabel("Total pot"); plt.title("Colony pot (total)")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(output_dir / "colonies_pot_total.png"); plt.close()
        # Per-colony (top 5 by max pot)
        top_ids = sorted(all_ids, key=lambda cid: max(pot_series[cid]) if pot_series[cid] else 0.0, reverse=True)[:5]
        if top_ids:
            plt.figure(figsize=(10, 5))
            for cid in top_ids:
                plt.plot(generations, pot_series[cid], label=cid, linewidth=1)
            plt.xlabel("Generation"); plt.ylabel("Pot"); plt.title("Colony pot by colony (top 5)")
            plt.legend(fontsize=8)
            plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(output_dir / "colonies_pot_per_colony.png"); plt.close()
            # Membership stacked area for top 5
            try:
                plt.figure(figsize=(10, 5))
                ys = [mem_series[cid] for cid in top_ids]
                plt.stackplot(generations, *ys, labels=top_ids)
                plt.xlabel("Generation"); plt.ylabel("Members"); plt.title("Colony membership (stacked, top 5)")
                plt.legend(fontsize=8, loc="upper left")
                plt.tight_layout(); plt.savefig(output_dir / "colonies_membership_stack.png"); plt.close()
            except Exception:
                pass

    # Team routes/promotions per generation
    if any("team_routes" in rec for rec in records):
        values = [int(rec.get("team_routes", 0) or 0) for rec in records]
        plt.figure(figsize=(8, 4))
        plt.plot(generations, values, marker="o", linewidth=1)
        plt.xlabel("Generation"); plt.ylabel("Team routes"); plt.title("Team routes per generation")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(output_dir / "team_routes.png"); plt.close()
    if any("team_promotions" in rec for rec in records):
        values = [int(rec.get("team_promotions", 0) or 0) for rec in records]
        plt.figure(figsize=(8, 4))
        plt.plot(generations, values, marker="o", linewidth=1)
        plt.xlabel("Generation"); plt.ylabel("Team promotions"); plt.title("Team promotions per generation")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(output_dir / "team_promotions.png"); plt.close()

    # Co-routing heatmap for top pairs across the run
    pairs_set = set()
    for rec in records:
        top = rec.get("co_routing_top") or {}
        if isinstance(top, dict):
            pairs_set.update(str(k) for k in top.keys())
    pairs = sorted(list(pairs_set))[:10]
    if pairs:
        matrix: List[List[int]] = []
        for rec in records:
            row: List[int] = []
            top = rec.get("co_routing_top") or {}
            for p in pairs:
                try:
                    row.append(int((top or {}).get(p, 0)))
                except Exception:
                    row.append(0)
            matrix.append(row)
        plt.figure(figsize=(max(6, len(pairs)), 4))
        plt.imshow(matrix, aspect="auto", origin="lower", cmap="cividis")
        plt.colorbar(label="Co-route count")
        plt.yticks(range(len(records)), generations)
        plt.xticks(range(len(pairs)), pairs, rotation=45, ha="right")
        plt.xlabel("Pair"); plt.ylabel("Generation"); plt.title("Co-routing (top pairs)")
        plt.tight_layout(); plt.savefig(output_dir / "co_routing_heatmap.png"); plt.close()

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

    # LP progress heatmap if available
    if records[0].get("lp_progress"):
        keys = sorted(records[0]["lp_progress"].keys())
        matrix = []
        for rec in records:
            row = []
            lp = rec.get("lp_progress", {})
            for k in keys:
                try:
                    row.append(float(lp.get(k, 0.0)))
                except Exception:
                    row.append(0.0)
            matrix.append(row)
        plt.figure(figsize=(max(6, len(keys)), 4))
        plt.imshow(matrix, aspect="auto", origin="lower", cmap="magma")
        plt.colorbar(label="LP")
        plt.yticks(range(len(records)), generations)
        plt.xticks(range(len(keys)), keys, rotation=45, ha="right")
        plt.xlabel("Cell")
        plt.ylabel("Generation")
        plt.title("Learning Progress (LP) heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / "lp_progress.png")
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
        f"- Curriculum lp_mix active: mean {summary['lp_mix_active_mean']:.3f} | last {summary['lp_mix_active_last']:.3f} (base mean {summary['lp_mix_base_mean']:.3f})"
    )
    lines.append(
        f"- Active organelles per generation: {summary['active_min']} – {summary['active_max']} (bankrupt: {summary['bankrupt_min']} – {summary['bankrupt_max']})"
    )
    lines.append(f"- Bankruptcy culls: total {summary['culled_total']} (max per generation {summary['culled_max']})")
    lines.append(f"- Assimilation merges (per summary): {summary['total_merges']}")
    if summary.get("team_routes_total") is not None:
        lines.append(
            f"- Team routes / promotions: {summary.get('team_routes_total', 0)} / {summary.get('team_promotions_total', 0)}"
        )
    if summary.get("assimilation_attempt_total"):
        lines.append(f"- Assimilation attempts (all stages): {summary['assimilation_attempt_total']}")
    if summary["assimilation_gating_total"]:
        lines.append("- Assimilation gating totals:")
        for key, value in summary["assimilation_gating_total"].items():
            lines.append(f"  - {key}: {value}")
    if summary.get("policy_applied_total"):
        lines.append(
            f"- Policy usage: {summary['policy_applied_total']}/{summary['generations']} generations"
        )
        lines.append(
            f"  - ROI when policy on/off: {summary['policy_roi_mean_when_applied']:.3f} / {summary['policy_roi_mean_when_not']:.3f}"
        )
        fields_total = summary.get("policy_fields_used_total") or {}
        if fields_total:
            items = ", ".join(f"{k}:{v}" for k, v in fields_total.items())
            lines.append(f"  - Fields used (total): {items}")
        if summary.get("policy_budget_frac_mean"):
            lines.append(f"  - budget_frac mean: {summary['policy_budget_frac_mean']:.2f}")
        if summary.get("policy_reserve_ratio_mean"):
            lines.append(f"  - reserve_ratio mean: {summary['policy_reserve_ratio_mean']:.2f}")
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
        if "sample_size_mean" in assimilation:
            lines.append(f"  - Mean sample size: {assimilation['sample_size_mean']:.1f}")
        if "ci_excludes_zero_rate" in assimilation:
            lines.append(f"  - CI excludes zero: {assimilation['ci_excludes_zero_rate']*100:.1f}%")
        if "power_mean" in assimilation:
            lines.append(f"  - Power (proxy) mean: {assimilation['power_mean']:.2f}")
        if "methods" in assimilation:
            method_items = ", ".join(f"{name}: {count}" for name, count in assimilation["methods"].items())
            lines.append(f"  - Methods: {method_items}")
        if assimilation.get("dr_used"):
            lines.append(f"  - DR uplift applied in {assimilation['dr_used']} events")
    # Co-routing aggregate
    crt = summary.get("co_routing_totals") or {}
    if crt:
        lines.append("- Top co-routing pairs (aggregate):")
        for pair, cnt in crt.items():
            lines.append(f"  - {pair}: {cnt}")
    else:
        lines.append("- Assimilation tests: none recorded")
    if summary.get("tau_relief_active"):
        lines.append("- Tau relief (per cell):")
        for cell, value in summary["tau_relief_active"].items():
            lines.append(f"  - {cell}: {value}")
    if summary.get("roi_relief_active"):
        lines.append("- ROI relief (per organelle):")
        for oid, value in summary["roi_relief_active"].items():
            lines.append(f"  - {oid}: {value}")
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
        "Curriculum lp_mix active (mean/base/last):",
        f"{summary['lp_mix_active_mean']:.3f}",
        f"{summary['lp_mix_base_mean']:.3f}",
        f"{summary['lp_mix_active_last']:.3f}",
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
    if summary.get("assimilation_gating_reasons_samples"):
        print("Gating reasons (samples across gens):")
        for key, value in summary["assimilation_gating_reasons_samples"].items():
            print(f"  {key}: {value}")
    if summary.get("colonies_count_series"):
        series = summary["colonies_count_series"]
        print("Colonies count (min-max):", min(series), "-", max(series))
        print("Avg colony size (mean):", f"{summary['colonies_avg_size_mean']:.2f}")
    if summary.get("policy_applied_total"):
        print("Policy usage: gens with policy:", summary["policy_applied_total"], "/", summary["generations"])
        print(
            "ROI when policy on/off:",
            f"{summary['policy_roi_mean_when_applied']:.3f}",
            f"/ {summary['policy_roi_mean_when_not']:.3f}",
        )
        fields_total = summary.get("policy_fields_used_total") or {}
        if fields_total:
            items = ", ".join(f"{k}:{v}" for k, v in fields_total.items())
            print("Policy fields used (total):", items)
        if summary.get("policy_budget_frac_mean"):
            print("Policy budget_frac mean:", f"{summary['policy_budget_frac_mean']:.2f}")
        if summary.get("policy_reserve_ratio_mean"):
            print("Policy reserve_ratio mean:", f"{summary['policy_reserve_ratio_mean']:.2f}")
    if summary.get("policy_parse_attempts_total") or summary.get("policy_parse_parsed_total"):
        a = summary.get("policy_parse_attempts_total", 0) or 0
        p = summary.get("policy_parse_parsed_total", 0) or 0
        rate = (p / a) * 100 if a else 0.0
        print("Policy parse:", p, "/", a, f"({rate:.1f}% )")
    # Team totals to console
    if summary.get("team_routes_total") is not None:
        print("Team routes / promotions:", summary.get("team_routes_total", 0), "/", summary.get("team_promotions_total", 0))
    if summary.get("co_routing_totals"):
        tops = ", ".join(f"{k}:{v}" for k, v in summary["co_routing_totals"].items())
        print("Top co-routing pairs:", tops)
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
        if "methods" in assimilation_summary:
            method_items = ", ".join(f"{name}: {count}" for name, count in assimilation_summary["methods"].items())
            print("Methods:", method_items)
        if assimilation_summary.get("dr_used"):
            print("DR uplift events:", assimilation_summary["dr_used"])
        if assimilation_summary.get("dr_strata_top"):
            top = assimilation_summary["dr_strata_top"]
            items = ", ".join(f"{k}:{v}" for k, v in top)
            print("DR contributing strata (top):", items)
    else:
        print("No assimilation events recorded")
    if summary.get("trials_total") or summary.get("promotions_total"):
        print("Trials created:", summary.get("trials_total", 0), "Promotions:", summary.get("promotions_total", 0))
    if summary.get("assimilation_attempt_total"):
        print("Assimilation attempts (all stages):", summary["assimilation_attempt_total"])
    if summary.get("tau_relief_active"):
        print("Tau relief active:", summary["tau_relief_active"])
    if summary.get("roi_relief_active"):
        print("ROI relief active:", summary["roi_relief_active"])

    if args.plots:
        ensure_plots(gen_records, run_dir / "visuals")
        print("Plots saved under", run_dir / "visuals")

    if args.report:
        write_report(summary, assimilation_summary, run_dir / "report.md")
        print("Report written to", run_dir / "report.md")


if __name__ == "__main__":
    main()
