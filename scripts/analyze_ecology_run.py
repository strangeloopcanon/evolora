#!/usr/bin/env python3
"""Generate analysis report and plots for an ecology run."""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import Counter, defaultdict
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
    evaluation_family_totals: Dict[str, Dict[str, float]] = {}
    for ev in eval_records:
        family_breakdown = ev.get("family_breakdown") or {}
        if not isinstance(family_breakdown, dict):
            continue
        for family, stats in family_breakdown.items():
            if not isinstance(stats, dict):
                continue
            acc = evaluation_family_totals.setdefault(
                family,
                {
                    "correct": 0.0,
                    "total": 0.0,
                    "roi_sum": 0.0,
                    "delta_sum": 0.0,
                    "cost_sum": 0.0,
                    "count": 0.0,
                },
            )
            correct = float(stats.get("correct", 0.0) or 0.0)
            total = float(stats.get("total", 0.0) or 0.0)
            avg_roi = float(stats.get("avg_roi", 0.0) or 0.0)
            avg_delta = float(stats.get("avg_delta", 0.0) or 0.0)
            avg_cost = float(stats.get("avg_cost", 0.0) or 0.0)
            count = float(stats.get("count", total) or total)
            acc["correct"] += correct
            acc["total"] += total
            acc["roi_sum"] += avg_roi * count
            acc["delta_sum"] += avg_delta * count
            acc["cost_sum"] += avg_cost * count
            acc["count"] += count
    eval_accuracy = [rec["accuracy"] for rec in eval_records]
    eval_correct = sum(rec.get("correct", 0) for rec in eval_records)
    eval_total = sum(rec.get("total", 0) for rec in eval_records)
    qd_archive_size_series = [int(rec.get("qd_archive_size", 0) or 0) for rec in records]
    qd_archive_coverage_series = [
        float(rec.get("qd_archive_coverage", 0.0) or 0.0) for rec in records
    ]
    qd_archive_top_latest = records[-1].get("qd_archive_top") if records else []

    gating_totals: Dict[str, int] = {}
    for rec in records:
        gating = rec.get("assimilation_gating")
        if not gating:
            continue
        for key, value in gating.items():
            gating_totals[key] = gating_totals.get(key, 0) + int(value)

    tuning_records = [
        rec.get("assimilation_energy_tuning")
        for rec in records
        if rec.get("assimilation_energy_tuning")
    ]
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
    prompt_scaffold_counts = latest_record.get("prompt_scaffolds") or {}
    comms_board_latest = latest_record.get("comms_board") or []
    # Aggregate gating reasons seen across snapshots (best-effort; snapshots are bounded)
    gating_reason_counts: Dict[str, int] = {}
    for rec in records:
        for sample in rec.get("assimilation_gating_samples", []) or []:
            reason = sample.get("reason")
            if isinstance(reason, str):
                gating_reason_counts[reason] = gating_reason_counts.get(reason, 0) + 1
    latest_attempts = latest_record.get("assimilation_attempts", [])
    colonies_meta = latest_record.get("colonies_meta")
    policy_failures_total = sum(int(rec.get("policy_failures", 0) or 0) for rec in records)
    policy_failure_latest = latest_record.get("policy_failure_samples") or []
    team_gate_totals: Dict[str, int] = {}
    for rec in records:
        team_counts = rec.get("team_gate_counts") or {}
        if not isinstance(team_counts, dict):
            continue
        for reason, value in team_counts.items():
            try:
                team_gate_totals[reason] = team_gate_totals.get(reason, 0) + int(value)
            except Exception:
                continue
    latest_team_gate_counts = latest_record.get("team_gate_counts") or {}
    latest_team_gate_samples = latest_record.get("team_gate_samples") or []
    population_refresh_events = [
        rec.get("population_refresh") for rec in records if rec.get("population_refresh")
    ]
    population_refresh_total = sum(
        int(evt.get("count", 0)) for evt in population_refresh_events if isinstance(evt, dict)
    )
    population_refresh_latest = population_refresh_events[-1] if population_refresh_events else None
    # Colonies timeline
    colonies_count_series = [int(rec.get("colonies", 0) or 0) for rec in records]
    colonies_avg_size_series: List[float] = []
    colony_events: List[Dict[str, Any]] = []
    colony_event_counter: Counter[str] = Counter()
    colony_event_seen: set[str] = set()
    colony_selection_dissolved_series: List[int] = []
    colony_selection_replicated_series: List[int] = []
    colony_selection_pool_pot_series: List[float] = []
    colony_selection_pool_members_series: List[int] = []
    colony_selection_events: List[Dict[str, Any]] = []
    colony_tier_mean_series: List[float] = []
    colony_tier_counts_series: List[Dict[str, int]] = []
    comms_posts_series: List[int] = []
    comms_reads_series: List[int] = []
    comms_credits_series: List[int] = []
    comms_events_all: List[Dict[str, Any]] = []
    comms_event_counter: Counter[str] = Counter()
    knowledge_totals: Dict[str, float] = {}
    knowledge_entries_series: List[int] = []
    assimilation_history_series: Dict[str, List[Dict[str, Any]]] = {}
    assimilation_history_seen: Dict[str, set[str]] = defaultdict(set)
    survival_reserve_counts: List[int] = []
    survival_hazard_counts: List[int] = []
    survival_events: List[Dict[str, Any]] = []
    survival_event_counter: Counter[str] = Counter()
    survival_price_bias_counts: List[int] = []
    budget_total_series: List[int] = []
    budget_raw_series: List[int] = []
    budget_cap_series: List[int] = []
    budget_zero_counts: List[int] = []
    budget_cap_hits = 0
    budget_energy_samples: List[float] = []
    budget_trait_samples: List[float] = []
    budget_policy_samples: List[float] = []
    colony_bandwidth_series: List[float] = []
    colony_size_total_series: List[int] = []
    colony_delta_mean_series: List[float] = []
    colony_variance_ratio_series: List[float] = []
    colony_hazard_members_series: List[int] = []
    colony_reserve_active_series: List[int] = []
    colony_winter_active_series: List[int] = []
    colony_hazard_z_series: List[float] = []
    foraging_beta_series: List[float | None] = []
    foraging_decay_series: List[float | None] = []
    foraging_ucb_series: List[float | None] = []
    foraging_budget_series: List[float | None] = []
    mutation_rank_noise_series: List[int] = []
    mutation_dropout_series: List[int] = []
    mutation_duplication_series: List[int] = []
    merge_audit_records: List[Dict[str, Any]] = []
    merge_audit_count_series: List[int] = []
    merge_audit_delta_series: List[float | None] = []
    winter_active_series: List[int] = []
    winter_price_series: List[float] = []
    winter_ticket_series: List[float] = []
    winter_events: List[Dict[str, Any]] = []
    winter_event_counter: Counter[str] = Counter()
    winter_roi_deltas: List[float] = []
    winter_assim_deltas: List[float] = []
    winter_cull_counts: List[int] = []
    power_need_weighted = 0.0
    price_multiplier_weighted = 0.0
    power_episodes_total = 0
    tokens_minted_total = 0
    tokens_used_total = 0
    info_topups_total = 0
    for idx, rec in enumerate(records):
        gen = rec.get("generation", idx + 1)
        meta = rec.get("colonies_meta") or {}
        if isinstance(meta, dict) and meta:
            sizes = []
            for cid, v in meta.items():
                try:
                    sizes.append(len(v.get("members", [])))
                except Exception:
                    continue
                events = v.get("events", [])
                if isinstance(events, list):
                    for ev in events:
                        if not isinstance(ev, dict):
                            continue
                        record = {"generation": gen, "colony": cid}
                        record.update(ev)
                        sig = json.dumps(record, sort_keys=True)
                        if sig in colony_event_seen:
                            continue
                        colony_event_seen.add(sig)
                        colony_events.append(record)
                        etype = record.get("type")
                        if isinstance(etype, str):
                            colony_event_counter[etype] += 1
            colonies_avg_size_series.append(sum(sizes) / max(1, len(sizes)))
        else:
            colonies_avg_size_series.append(0.0)
        selection_block = rec.get("colony_selection") or {}
        if isinstance(selection_block, dict) and selection_block:
            colony_selection_dissolved_series.append(int(selection_block.get("dissolved", 0) or 0))
            colony_selection_replicated_series.append(
                int(selection_block.get("replicated", 0) or 0)
            )
            colony_selection_pool_members_series.append(
                int(selection_block.get("pool_members", 0) or 0)
            )
            colony_selection_pool_pot_series.append(
                float(selection_block.get("pool_pot", 0.0) or 0.0)
            )
        else:
            colony_selection_dissolved_series.append(0)
            colony_selection_replicated_series.append(0)
            colony_selection_pool_members_series.append(0)
            colony_selection_pool_pot_series.append(0.0)
        sel_events = rec.get("colony_selection_events") or []
        if isinstance(sel_events, list):
            for ev in sel_events:
                if not isinstance(ev, dict):
                    continue
                record = {"generation": gen}
                record.update(ev)
                colony_selection_events.append(record)
        pool_log = rec.get("colony_selection_pool") or []
        if isinstance(pool_log, list):
            for ev in pool_log:
                if not isinstance(ev, dict):
                    continue
                record = {"generation": gen}
                record.update(ev)
                colony_selection_events.append(record)
        survival = rec.get("survival") or {}
        if isinstance(survival, dict):
            reserve_count = survival.get("reserve_active_count")
            hazard_count = survival.get("hazard_active_count")
            if reserve_count is None:
                reserve_ids = survival.get("reserve_active_ids") or []
                reserve_count = len(reserve_ids)
            if hazard_count is None:
                hazard_ids = survival.get("hazard_active_ids") or []
                hazard_count = len(hazard_ids)
            survival_reserve_counts.append(int(reserve_count or 0))
            survival_hazard_counts.append(int(hazard_count or 0))
            price_bias_count = survival.get("price_bias_active_count")
            if price_bias_count is None:
                bias_ids = survival.get("price_bias_active_ids") or []
                price_bias_count = len(bias_ids)
            survival_price_bias_counts.append(int(price_bias_count or 0))
            for ev in survival.get("events", []) or []:
                if not isinstance(ev, dict):
                    continue
                record = {"generation": gen}
                record.update(ev)
                survival_events.append(record)
                etype = ev.get("type")
                if isinstance(etype, str):
                    survival_event_counter[etype] += 1
        else:
            survival_reserve_counts.append(0)
            survival_hazard_counts.append(0)
            survival_price_bias_counts.append(0)
        econ = rec.get("power_economics") or {}
        if isinstance(econ, dict) and econ:
            episodes = int(econ.get("episodes", 0) or 0)
            if episodes > 0:
                power_episodes_total += episodes
                power_need_weighted += float(econ.get("avg_power_need", 0.0) or 0.0) * episodes
                price_multiplier_weighted += (
                    float(econ.get("avg_price_multiplier", 0.0) or 0.0) * episodes
                )
            tokens_minted_total += int(econ.get("tokens_minted", 0) or 0)
            tokens_used_total += int(econ.get("tokens_used", 0) or 0)
            info_topups_total += int(econ.get("info_topups", 0) or 0)
        comms = rec.get("comms") or {}
        if isinstance(comms, dict):
            posts = int(comms.get("posts", 0) or 0)
            reads = int(comms.get("reads", 0) or 0)
            credits = int(comms.get("credits", 0) or 0)
            comms_posts_series.append(posts)
            comms_reads_series.append(reads)
            comms_credits_series.append(credits)
            events = comms.get("events", []) or []
            if isinstance(events, list):
                for ev in events:
                    if not isinstance(ev, dict):
                        continue
                    record = {"generation": gen}
                    record.update(ev)
                    comms_events_all.append(record)
                    etype = ev.get("type")
                    if isinstance(etype, str):
                        comms_event_counter[etype] += 1
        else:
            comms_posts_series.append(0)
            comms_reads_series.append(0)
            comms_credits_series.append(0)
        knowledge = rec.get("knowledge") or {}
        if isinstance(knowledge, dict) and knowledge:
            entries_val = knowledge.get("entries")
            if isinstance(entries_val, (int, float)):
                knowledge_entries_series.append(int(entries_val))
            for key in ("writes", "write_denied", "reads", "read_denied", "hits", "expired"):
                val = knowledge.get(key)
                if isinstance(val, (int, float)):
                    knowledge_totals[key] = knowledge_totals.get(key, 0.0) + float(val)
        else:
            knowledge_entries_series.append(0)
        history_block = rec.get("assimilation_history") or {}
        if isinstance(history_block, dict):
            for key, entries in history_block.items():
                if isinstance(entries, dict):
                    iter_entries = [entries]
                elif isinstance(entries, list):
                    iter_entries = entries
                else:
                    continue
                dest = assimilation_history_series.setdefault(key, [])
                seen = assimilation_history_seen[key]
                for entry in iter_entries:
                    if not isinstance(entry, dict):
                        continue
                    sig = json.dumps(entry, sort_keys=True)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    dest.append(entry)
        audits = rec.get("merge_audits") or []
        if isinstance(audits, list) and audits:
            count = 0
            delta_samples: List[float] = []
            for audit in audits:
                if not isinstance(audit, dict):
                    continue
                record = dict(audit)
                record.setdefault("generation", int(gen))
                merge_audit_records.append(record)
                delta_val = record.get("delta")
                if isinstance(delta_val, (int, float)):
                    delta_samples.append(float(delta_val))
                count += 1
            merge_audit_count_series.append(count)
            merge_audit_delta_series.append(mean(delta_samples) if delta_samples else None)
        else:
            merge_audit_count_series.append(0)
            merge_audit_delta_series.append(None)
        budget_block = rec.get("budget") or {}
        if isinstance(budget_block, dict) and budget_block:
            final_total = int(
                budget_block.get("final_total", budget_block.get("capped_total", 0)) or 0
            )
            raw_total = int(budget_block.get("raw_total", 0) or 0)
            budget_total_series.append(final_total)
            budget_raw_series.append(raw_total)
            cap_val = int(budget_block.get("global_cap", 0) or 0)
            if cap_val > 0:
                budget_cap_series.append(cap_val)
            if budget_block.get("cap_hit"):
                budget_cap_hits += 1
            per_org_final = budget_block.get("final") or {}
            zero_count = sum(1 for v in per_org_final.values() if int(v or 0) <= 0)
            budget_zero_counts.append(zero_count)
            per_org_meta = budget_block.get("per_org") or {}
            if isinstance(per_org_meta, dict):
                for data in per_org_meta.values():
                    try:
                        budget_energy_samples.append(float(data.get("energy", 0.0)))
                    except Exception:
                        pass
                    try:
                        budget_trait_samples.append(float(data.get("trait", 0.0)))
                    except Exception:
                        pass
                    try:
                        budget_policy_samples.append(float(data.get("policy", 0.0)))
                    except Exception:
                        pass
        else:
            budget_total_series.append(0)
            budget_raw_series.append(0)
            budget_zero_counts.append(0)
        foraging_block = rec.get("foraging") or {}
        trait_map = foraging_block.get("traits", {})
        if isinstance(trait_map, dict) and trait_map:
            betas = [float(data.get("beta", 0.0)) for data in trait_map.values()]
            decays = [float(data.get("decay", 0.0)) for data in trait_map.values()]
            ucbs = [float(data.get("ucb", 0.0)) for data in trait_map.values()]
            budgets = [float(data.get("budget", 0.0)) for data in trait_map.values()]
            foraging_beta_series.append(mean(betas) if betas else None)
            foraging_decay_series.append(mean(decays) if decays else None)
            foraging_ucb_series.append(mean(ucbs) if ucbs else None)
            foraging_budget_series.append(mean(budgets) if budgets else None)
        else:
            foraging_beta_series.append(None)
            foraging_decay_series.append(None)
            foraging_ucb_series.append(None)
            foraging_budget_series.append(None)
        mutation_stats_block = rec.get("mutation_stats") or {}
        if isinstance(mutation_stats_block, dict):
            mutation_rank_noise_series.append(int(mutation_stats_block.get("rank_noise", 0) or 0))
            mutation_dropout_series.append(int(mutation_stats_block.get("dropout", 0) or 0))
            mutation_duplication_series.append(int(mutation_stats_block.get("duplication", 0) or 0))
        else:
            mutation_rank_noise_series.append(0)
            mutation_dropout_series.append(0)
            mutation_duplication_series.append(0)
        if not isinstance(colony_metrics := rec.get("colony_metrics") or {}, dict):
            colony_metrics = {}
        tier_mean = 0.0
        tier_counts_map: Dict[str, int] = {}
        if colony_metrics:
            total_bandwidth = sum(
                float(info.get("bandwidth_budget", 0.0)) for info in colony_metrics.values()
            )
            total_size = sum(int(info.get("size", 0) or 0) for info in colony_metrics.values())
            avg_delta = sum(
                float(info.get("last_delta", 0.0)) for info in colony_metrics.values()
            ) / max(len(colony_metrics), 1)
            avg_variance_ratio = sum(
                float(info.get("variance_ratio", 1.0)) for info in colony_metrics.values()
            ) / max(len(colony_metrics), 1)
            hazard_total = sum(
                int(info.get("hazard_members", 0) or 0) for info in colony_metrics.values()
            )
            reserve_count = sum(1 for info in colony_metrics.values() if info.get("reserve_active"))
            winter_count = sum(1 for info in colony_metrics.values() if info.get("winter_mode"))
            hazard_z_mean = sum(
                float(info.get("hazard_z", 0.0)) for info in colony_metrics.values()
            ) / max(len(colony_metrics), 1)
            tiers = [int(info.get("tier", 0) or 0) for info in colony_metrics.values()]
            if tiers:
                tier_mean = sum(tiers) / len(tiers)
                tier_counter = Counter(tiers)
                tier_counts_map = {str(k): v for k, v in sorted(tier_counter.items())}
        else:
            total_bandwidth = 0.0
            total_size = 0
            avg_delta = 0.0
            avg_variance_ratio = 1.0
            hazard_total = 0
            reserve_count = 0
            winter_count = 0
            hazard_z_mean = 0.0
        colony_bandwidth_series.append(float(total_bandwidth))
        colony_size_total_series.append(int(total_size))
        colony_delta_mean_series.append(float(avg_delta))
        colony_variance_ratio_series.append(float(avg_variance_ratio))
        colony_hazard_members_series.append(int(hazard_total))
        colony_reserve_active_series.append(int(reserve_count))
        colony_winter_active_series.append(int(winter_count))
        colony_hazard_z_series.append(float(hazard_z_mean))
        colony_tier_mean_series.append(float(tier_mean))
        colony_tier_counts_series.append(tier_counts_map)
        extra_colony_events = rec.get("colony_events") or []
        if isinstance(extra_colony_events, list):
            for ev in extra_colony_events:
                if not isinstance(ev, dict):
                    continue
                record = dict(ev)
                record.setdefault("generation", gen)
                sig = json.dumps(record, sort_keys=True)
                if sig in colony_event_seen:
                    continue
                colony_event_seen.add(sig)
                colony_events.append(record)
                etype = record.get("type")
                if isinstance(etype, str):
                    colony_event_counter[etype] += 1
        winter_block = rec.get("winter") or {}
        if isinstance(winter_block, dict):
            winter_active_series.append(1 if winter_block.get("active") else 0)
            winter_price_series.append(float(winter_block.get("price_multiplier", 1.0)))
            winter_ticket_series.append(float(winter_block.get("ticket_multiplier", 1.0)))
            events = winter_block.get("events", []) or []
            if isinstance(events, list):
                for ev in events:
                    if not isinstance(ev, dict):
                        continue
                    record = {"generation": gen}
                    record.update(ev)
                    winter_events.append(record)
                    etype = record.get("type")
                    if isinstance(etype, str):
                        winter_event_counter[etype] += 1
                        if etype == "winter_end":
                            try:
                                winter_roi_deltas.append(float(record.get("delta_roi", 0.0)))
                            except Exception:
                                pass
                            try:
                                winter_assim_deltas.append(float(record.get("delta_assim", 0.0)))
                            except Exception:
                                pass
                        elif etype == "winter_cull":
                            try:
                                winter_cull_counts.append(int(record.get("count", 0)))
                            except Exception:
                                pass
        else:
            winter_active_series.append(0)
            winter_price_series.append(1.0)
            winter_ticket_series.append(1.0)

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
    team_probe_hits_total = 0
    team_probe_pairs: Dict[str, int] = {}
    team_probe_latest: List[Dict[str, Any]] = []
    # Co-routing totals across generations
    co_routing_totals: Dict[str, int] = {}
    for rec in records:
        top = rec.get("co_routing_top") or {}
        if isinstance(top, dict):
            for pair, cnt in top.items():
                try:
                    key = str(pair)
                    co_routing_totals[key] = co_routing_totals.get(key, 0) + int(cnt)
                except Exception:
                    continue
        candidates = rec.get("team_probe_candidates") or []
        if isinstance(candidates, list):
            if candidates:
                team_probe_latest = candidates
            team_probe_hits_total += len(candidates)
            for cand in candidates:
                pair = cand.get("pair")
                key = str(pair)
                team_probe_pairs[key] = team_probe_pairs.get(key, 0) + 1
    power_need_mean = power_need_weighted / max(1, power_episodes_total)
    price_multiplier_mean = price_multiplier_weighted / max(1, power_episodes_total)
    budget_totals_mean = mean(budget_total_series) if budget_total_series else 0.0
    budget_totals_median = median(budget_total_series) if budget_total_series else 0.0
    budget_raw_mean = mean(budget_raw_series) if budget_raw_series else 0.0
    budget_cap_max = max(budget_cap_series) if budget_cap_series else 0
    budget_cap_rate = (budget_cap_hits / len(budget_total_series)) if budget_total_series else 0.0
    budget_zero_mean = mean(budget_zero_counts) if budget_zero_counts else 0.0
    budget_energy_mean = mean(budget_energy_samples) if budget_energy_samples else 0.0
    budget_trait_mean = mean(budget_trait_samples) if budget_trait_samples else 0.0
    budget_policy_mean = mean(budget_policy_samples) if budget_policy_samples else 0.0
    budget_final_last = budget_total_series[-1] if budget_total_series else 0

    def _mean_filtered(values: List[float | None]) -> float:
        filtered = [
            float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))
        ]
        return mean(filtered) if filtered else 0.0

    foraging_beta_mean = _mean_filtered(foraging_beta_series)
    foraging_decay_mean = _mean_filtered(foraging_decay_series)
    foraging_ucb_mean = _mean_filtered(foraging_ucb_series)
    foraging_budget_mean = _mean_filtered(foraging_budget_series)
    winter_active_mean = mean(winter_active_series) if winter_active_series else 0.0
    winter_price_mean = mean(winter_price_series) if winter_price_series else 0.0
    winter_ticket_mean = mean(winter_ticket_series) if winter_ticket_series else 0.0
    winter_roi_delta_mean = mean(winter_roi_deltas) if winter_roi_deltas else 0.0
    winter_assim_delta_mean = mean(winter_assim_deltas) if winter_assim_deltas else 0.0
    winter_cull_total = sum(winter_cull_counts) if winter_cull_counts else 0
    colony_tier_counts_total: Dict[str, int] = {}
    for counts in colony_tier_counts_series:
        for tier, value in counts.items():
            colony_tier_counts_total[tier] = colony_tier_counts_total.get(tier, 0) + int(value)
    colony_tier_mean = mean(colony_tier_mean_series) if colony_tier_mean_series else 0.0
    mutation_totals = {
        "rank_noise": sum(mutation_rank_noise_series),
        "dropout": sum(mutation_dropout_series),
        "duplication": sum(mutation_duplication_series),
    }
    audit_delta_filtered = [
        float(val) for val in merge_audit_delta_series if isinstance(val, (int, float))
    ]
    merge_audit_delta_mean = mean(audit_delta_filtered) if audit_delta_filtered else 0.0
    merge_audit_total = sum(merge_audit_count_series)
    merge_audit_family: Dict[str, Dict[str, float]] = {}
    for record in merge_audit_records:
        cell = record.get("cell") or {}
        family = str(cell.get("family", "unknown"))
        stats = merge_audit_family.setdefault(
            family, {"count": 0, "delta_sum": 0.0, "delta_min": None}
        )
        delta = record.get("delta")
        if isinstance(delta, (int, float)):
            stats["delta_sum"] += float(delta)
            stats["delta_min"] = (
                float(delta)
                if stats["delta_min"] is None
                else min(stats["delta_min"], float(delta))
            )
        stats["count"] += 1
    for _family, stats in merge_audit_family.items():
        count = max(1, stats["count"])
        stats["delta_mean"] = stats["delta_sum"] / count
        if stats["delta_min"] is None:
            stats["delta_min"] = 0.0

    assimilation_family_summary: Dict[str, Dict[str, float]] = {}
    latest_attempts = latest_record.get("assimilation_attempts", []) or []
    for attempt in latest_attempts:
        if not isinstance(attempt, dict):
            continue
        cell = attempt.get("cell") or {}
        family = str(cell.get("family", "unknown"))
        stats = assimilation_family_summary.setdefault(
            family,
            {
                "attempts": 0,
                "passes": 0,
                "uplift_sum": 0.0,
                "uplift_count": 0.0,
                "candidate_roi_sum": 0.0,
                "candidate_roi_count": 0.0,
                "audit_delta_sum": 0.0,
                "audit_delta_count": 0.0,
            },
        )
        stats["attempts"] += 1
        if bool(attempt.get("passes_stat_test")):
            stats["passes"] += 1
        uplift = attempt.get("uplift")
        if isinstance(uplift, (int, float)):
            stats["uplift_sum"] += float(uplift)
            stats["uplift_count"] += 1.0
        holdout = attempt.get("holdout") or {}
        candidate_roi = holdout.get("candidate_roi")
        if isinstance(candidate_roi, (int, float)):
            stats["candidate_roi_sum"] += float(candidate_roi)
            stats["candidate_roi_count"] += 1.0
        audit = attempt.get("audit") or {}
        audit_delta = audit.get("delta")
        if isinstance(audit_delta, (int, float)):
            stats["audit_delta_sum"] += float(audit_delta)
            stats["audit_delta_count"] += 1.0
    for _family, stats in assimilation_family_summary.items():
        attempts = max(1, stats["attempts"])
        stats["pass_rate"] = stats["passes"] / attempts
        stats["uplift_mean"] = (
            stats["uplift_sum"] / stats["uplift_count"] if stats["uplift_count"] else 0.0
        )
        stats["candidate_roi_mean"] = (
            stats["candidate_roi_sum"] / stats["candidate_roi_count"]
            if stats["candidate_roi_count"]
            else 0.0
        )
        stats["audit_delta_mean"] = (
            stats["audit_delta_sum"] / stats["audit_delta_count"]
            if stats["audit_delta_count"]
            else 0.0
        )

    evaluation_family_stats: Dict[str, Dict[str, float]] = {}
    for family, totals in evaluation_family_totals.items():
        total = max(1.0, totals["total"])
        count = max(1.0, totals["count"] or totals["total"])
        evaluation_family_stats[family] = {
            "accuracy": totals["correct"] / total,
            "correct": int(totals["correct"]),
            "total": int(totals["total"]),
            "avg_roi": totals["roi_sum"] / count if count else 0.0,
            "avg_delta": totals["delta_sum"] / count if count else 0.0,
            "avg_cost": totals["cost_sum"] / count if count else 0.0,
        }

    generation_numbers = [
        int(rec.get("generation", 0) or 0) for rec in records if rec.get("generation") is not None
    ]
    generation_numbers = [g for g in generation_numbers if g > 0]
    generation_max = max(generation_numbers) if generation_numbers else len(records)
    generation_set = set(generation_numbers)
    missing_generations = 0
    if generation_set and generation_max > 0:
        missing_generations = sum(
            1 for g in range(1, generation_max + 1) if g not in generation_set
        )
    sparse_records = bool(missing_generations or len(records) != generation_max)

    return {
        "generations": generation_max,
        "records": len(records),
        "missing_generations": missing_generations,
        "sparse_records": sparse_records,
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
        "qd_archive_size_series": qd_archive_size_series,
        "qd_archive_size_mean": mean(qd_archive_size_series) if qd_archive_size_series else 0.0,
        "qd_archive_coverage_series": qd_archive_coverage_series,
        "qd_archive_coverage_mean": (
            mean(qd_archive_coverage_series) if qd_archive_coverage_series else 0.0
        ),
        "qd_archive_top_latest": qd_archive_top_latest or [],
        "roi_volatility": roi_volatility,
        "trials_total": total_trials,
        "promotions_total": total_promotions,
        "team_routes_series": team_routes_series,
        "team_promotions_series": team_promotions_series,
        "team_routes_total": team_routes_total,
        "team_promotions_total": team_promotions_total,
        "prompt_scaffolds": prompt_scaffold_counts,
        "foraging_beta_series": [None if v is None else float(v) for v in foraging_beta_series],
        "foraging_decay_series": [None if v is None else float(v) for v in foraging_decay_series],
        "foraging_ucb_series": [None if v is None else float(v) for v in foraging_ucb_series],
        "foraging_budget_series": [None if v is None else float(v) for v in foraging_budget_series],
        "foraging_beta_mean": foraging_beta_mean,
        "foraging_decay_mean": foraging_decay_mean,
        "foraging_ucb_mean": foraging_ucb_mean,
        "foraging_budget_mean": foraging_budget_mean,
        "foraging_traits_latest": (latest_record.get("foraging") or {}).get("traits", {}),
        "foraging_top_cells_latest": (latest_record.get("foraging") or {}).get("top_cells", {}),
        "team_probe_hits_total": team_probe_hits_total,
        "team_probe_pairs": dict(
            sorted(team_probe_pairs.items(), key=lambda kv: kv[1], reverse=True)[:10]
        ),
        "team_probe_candidates_latest": team_probe_latest,
        "co_routing_totals": dict(
            sorted(co_routing_totals.items(), key=lambda kv: kv[1], reverse=True)[:10]
        ),
        "colonies_meta": colonies_meta,
        "colonies_count_series": colonies_count_series,
        "colonies_avg_size_mean": (
            (sum(colonies_avg_size_series) / max(1, len(colonies_avg_size_series)))
            if colonies_avg_size_series
            else 0.0
        ),
        "colony_events": colony_events,
        "colony_event_counts": dict(colony_event_counter),
        "colony_selection_dissolved_series": colony_selection_dissolved_series,
        "colony_selection_replicated_series": colony_selection_replicated_series,
        "colony_selection_pool_members_series": colony_selection_pool_members_series,
        "colony_selection_pool_pot_series": colony_selection_pool_pot_series,
        "colony_selection_dissolved_total": sum(colony_selection_dissolved_series),
        "colony_selection_replicated_total": sum(colony_selection_replicated_series),
        "colony_selection_pool_members_mean": (
            mean(colony_selection_pool_members_series)
            if colony_selection_pool_members_series
            else 0.0
        ),
        "colony_selection_pool_pot_mean": (
            mean(colony_selection_pool_pot_series) if colony_selection_pool_pot_series else 0.0
        ),
        "colony_selection_events": colony_selection_events,
        "survival_reserve_counts": survival_reserve_counts,
        "survival_hazard_counts": survival_hazard_counts,
        "survival_price_bias_counts": survival_price_bias_counts,
        "survival_events": survival_events,
        "survival_event_counts": dict(survival_event_counter),
        "survival_latest": latest_record.get("survival") or {},
        "comms_posts_series": comms_posts_series,
        "comms_reads_series": comms_reads_series,
        "comms_credits_series": comms_credits_series,
        "comms_events": comms_events_all,
        "comms_event_counts": dict(comms_event_counter),
        "comms_board_latest": comms_board_latest,
        "knowledge_totals": {k: int(v) for k, v in knowledge_totals.items()},
        "knowledge_entries_mean": (
            mean(knowledge_entries_series) if knowledge_entries_series else 0.0
        ),
        "knowledge_entries_latest": knowledge_entries_series[-1] if knowledge_entries_series else 0,
        "policy_failures_total": int(policy_failures_total),
        "policy_failure_latest": policy_failure_latest,
        "team_gate_totals": team_gate_totals,
        "team_gate_latest": latest_team_gate_counts,
        "team_gate_samples_latest": latest_team_gate_samples,
        "population_refresh_total": int(population_refresh_total),
        "population_refresh_latest": population_refresh_latest,
        "assimilation_history_series": assimilation_history_series,
        "assimilation_history_latest": {
            key: records[-1] for key, records in assimilation_history_series.items() if records
        },
        "mutation_rank_noise_series": mutation_rank_noise_series,
        "mutation_dropout_series": mutation_dropout_series,
        "mutation_duplication_series": mutation_duplication_series,
        "mutation_totals": mutation_totals,
        "merge_audit_records": merge_audit_records,
        "merge_audit_count_series": merge_audit_count_series,
        "merge_audit_delta_series": merge_audit_delta_series,
        "merge_audit_total": merge_audit_total,
        "merge_audit_delta_mean": merge_audit_delta_mean,
        "merge_audit_family": merge_audit_family,
        "assimilation_family_summary": assimilation_family_summary,
        "evaluation_family_stats": evaluation_family_stats,
        "colony_bandwidth_series": colony_bandwidth_series,
        "colony_bandwidth_mean": mean(colony_bandwidth_series) if colony_bandwidth_series else 0.0,
        "colony_size_total_series": colony_size_total_series,
        "colony_size_mean": mean(colony_size_total_series) if colony_size_total_series else 0.0,
        "colony_delta_mean_series": colony_delta_mean_series,
        "colony_delta_overall_mean": (
            mean(colony_delta_mean_series) if colony_delta_mean_series else 0.0
        ),
        "colony_variance_ratio_series": colony_variance_ratio_series,
        "colony_variance_ratio_mean": (
            mean(colony_variance_ratio_series) if colony_variance_ratio_series else 0.0
        ),
        "colony_hazard_members_series": colony_hazard_members_series,
        "colony_reserve_active_series": colony_reserve_active_series,
        "colony_reserve_active_mean": (
            mean(colony_reserve_active_series) if colony_reserve_active_series else 0.0
        ),
        "colony_winter_active_series": colony_winter_active_series,
        "colony_winter_active_mean": (
            mean(colony_winter_active_series) if colony_winter_active_series else 0.0
        ),
        "colony_hazard_z_series": colony_hazard_z_series,
        "colony_hazard_z_mean": mean(colony_hazard_z_series) if colony_hazard_z_series else 0.0,
        "colony_tier_mean_series": colony_tier_mean_series,
        "colony_tier_mean": colony_tier_mean,
        "colony_tier_counts_series": colony_tier_counts_series,
        "colony_tier_counts_total": colony_tier_counts_total,
        "winter_active_series": winter_active_series,
        "winter_active_mean": winter_active_mean,
        "winter_price_series": winter_price_series,
        "winter_price_mean": winter_price_mean,
        "winter_ticket_series": winter_ticket_series,
        "winter_ticket_mean": winter_ticket_mean,
        "winter_events": winter_events,
        "winter_event_counts": dict(winter_event_counter),
        "winter_roi_delta_mean": winter_roi_delta_mean,
        "winter_assim_delta_mean": winter_assim_delta_mean,
        "winter_cull_total": winter_cull_total,
        "power_episodes_total": power_episodes_total,
        "power_need_mean": power_need_mean,
        "power_price_multiplier_mean": price_multiplier_mean,
        "power_tokens_minted_total": tokens_minted_total,
        "power_tokens_used_total": tokens_used_total,
        "power_info_topups_total": info_topups_total,
        "budget_totals_series": budget_total_series,
        "budget_totals_mean": budget_totals_mean,
        "budget_totals_median": budget_totals_median,
        "budget_final_last": budget_final_last,
        "budget_raw_mean": budget_raw_mean,
        "budget_cap_max": budget_cap_max,
        "budget_cap_hit_rate": budget_cap_rate,
        "budget_zero_mean": budget_zero_mean,
        "budget_energy_mean": budget_energy_mean,
        "budget_trait_mean": budget_trait_mean,
        "budget_policy_mean": budget_policy_mean,
        "policy_applied_total": policy_applied,
        "policy_parse_attempts_total": int(policy_attempts_total),
        "policy_parse_parsed_total": int(policy_parsed_total),
        "policy_roi_mean_when_applied": (
            (sum(roi_when_policy) / max(1, len(roi_when_policy))) if roi_when_policy else 0.0
        ),
        "policy_roi_mean_when_not": (
            (sum(roi_when_no_policy) / max(1, len(roi_when_no_policy)))
            if roi_when_no_policy
            else 0.0
        ),
        "policy_fields_used_total": fields_agg,
        "policy_budget_frac_mean": (
            (sum(budget_vals) / max(1, len(budget_vals))) if budget_vals else 0.0
        ),
        "policy_reserve_ratio_mean": (
            (sum(reserve_vals) / max(1, len(reserve_vals))) if reserve_vals else 0.0
        ),
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
    fisher_importances: list[float] = []
    fisher_weights: list[float] = []
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
            soup_members = record.get("soup")
            if isinstance(soup_members, list):
                for member in soup_members:
                    if not isinstance(member, dict):
                        continue
                    imp = member.get("importance")
                    if isinstance(imp, (int, float)):
                        fisher_importances.append(float(imp))
                    wt = member.get("weight")
                    if isinstance(wt, (int, float)):
                        fisher_weights.append(float(wt))
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
    if fisher_importances:
        out["fisher_importance_mean"] = float(mean(fisher_importances))
        out["fisher_importance_max"] = float(max(fisher_importances))
    if fisher_weights:
        out["merge_weight_mean"] = float(mean(fisher_weights))
    return out


def ensure_plots(records: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generations = [rec["generation"] for rec in records]
    summary = summarise_generations(records)

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
        plt.xlabel("Generation")
        plt.ylabel("Policy parsed count")
        plt.title("Policy parsed per generation")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "policy_parsed.png")
        plt.close()

    # Colonies plots
    colonies = [int(rec.get("colonies", 0) or 0) for rec in records]
    plt.figure(figsize=(8, 4))
    plt.plot(generations, colonies, marker="o", linewidth=1)
    plt.xlabel("Generation")
    plt.ylabel("Colonies count")
    plt.title("Colonies over generations")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "colonies_count.png")
    plt.close()

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
    plt.xlabel("Generation")
    plt.ylabel("Avg colony size")
    plt.title("Colony size over generations")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "colonies_avg_size.png")
    plt.close()

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
        plt.xlabel("Generation")
        plt.ylabel("Total pot")
        plt.title("Colony pot (total)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "colonies_pot_total.png")
        plt.close()
        # Per-colony (top 5 by max pot)
        top_ids = sorted(
            all_ids, key=lambda cid: max(pot_series[cid]) if pot_series[cid] else 0.0, reverse=True
        )[:5]
        if top_ids:
            plt.figure(figsize=(10, 5))
            for cid in top_ids:
                plt.plot(generations, pot_series[cid], label=cid, linewidth=1)
            plt.xlabel("Generation")
            plt.ylabel("Pot")
            plt.title("Colony pot by colony (top 5)")
            plt.legend(fontsize=8)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "colonies_pot_per_colony.png")
            plt.close()
            # Membership stacked area for top 5
            try:
                plt.figure(figsize=(10, 5))
                ys = [mem_series[cid] for cid in top_ids]
                plt.stackplot(generations, *ys, labels=top_ids)
                plt.xlabel("Generation")
                plt.ylabel("Members")
                plt.title("Colony membership (stacked, top 5)")
                plt.legend(fontsize=8, loc="upper left")
                plt.tight_layout()
                plt.savefig(output_dir / "colonies_membership_stack.png")
                plt.close()
            except Exception:
                pass

    events = summary.get("colony_events") or []
    if events:
        events_path = output_dir / "colony_events.jsonl"
        events_path.write_text("\n".join(json.dumps(ev) for ev in events))
        counts = Counter(ev.get("type") for ev in events if isinstance(ev.get("type"), str))
        if counts:
            plt.figure(figsize=(6, 3))
            labels = list(counts.keys())
            values = [counts[label] for label in labels]
            plt.bar(labels, values, color="#5b8def")
            plt.xlabel("Event type")
            plt.ylabel("Count")
            plt.title("Colony event counts")
            plt.tight_layout()
            plt.savefig(output_dir / "colony_events.png")
            plt.close()
    selection_events = summary.get("colony_selection_events") or []
    if selection_events:
        events_path = output_dir / "colony_selection_events.jsonl"
        events_path.write_text("\n".join(json.dumps(ev) for ev in selection_events))
        counts = Counter(
            ev.get("type") for ev in selection_events if isinstance(ev.get("type"), str)
        )
        if counts:
            plt.figure(figsize=(6, 3))
            labels = list(counts.keys())
            values = [counts[label] for label in labels]
            plt.bar(labels, values, color="#ff7f0e")
            plt.xlabel("Event type")
            plt.ylabel("Count")
            plt.title("Colony selection events")
            plt.tight_layout()
            plt.savefig(output_dir / "colony_selection_events.png")
            plt.close()

    beta_series = summary.get("foraging_beta_series") or []
    decay_series = summary.get("foraging_decay_series") or []
    ucb_series = summary.get("foraging_ucb_series") or []
    budget_series = summary.get("foraging_budget_series") or []
    if beta_series:
        count = min(len(generations), len(beta_series))
        if count > 0:
            xs = generations[:count]

            def _cast(series: List[Any]) -> List[float]:
                values: List[float] = []
                for idx in range(count):
                    val = series[idx]
                    if isinstance(val, (int, float)) and not math.isnan(float(val)):
                        values.append(float(val))
                    else:
                        values.append(float("nan"))
                return values

            plt.figure(figsize=(8, 4))
            plt.plot(xs, _cast(beta_series), label="beta", linewidth=1.2)
            if decay_series:
                plt.plot(xs, _cast(decay_series), label="decay", linewidth=1.0)
            if ucb_series:
                plt.plot(xs, _cast(ucb_series), label="ucb", linewidth=1.0)
            if budget_series:
                plt.plot(xs, _cast(budget_series), label="budget", linewidth=1.0)
            plt.xlabel("Generation")
            plt.ylabel("Value")
            plt.title("Foraging traits (mean per gen)")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "foraging_traits.png")
            plt.close()
    reserve_counts = summary.get("survival_reserve_counts") or []
    if reserve_counts:
        plt.figure(figsize=(8, 4))
        plt.plot(generations, reserve_counts, marker="o", linewidth=1)
        plt.xlabel("Generation")
        plt.ylabel("Reserve guards")
        plt.title("Reserve-active organelles")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "survival_reserve.png")
        plt.close()
    hazard_counts = summary.get("survival_hazard_counts") or []
    if hazard_counts:
        plt.figure(figsize=(8, 4))
        plt.plot(generations, hazard_counts, marker="o", linewidth=1)
        plt.xlabel("Generation")
        plt.ylabel("Hazard-active")
        plt.title("Hazard states")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "survival_hazard.png")
        plt.close()
    winter_price_series = summary.get("winter_price_series") or []
    winter_ticket_series = summary.get("winter_ticket_series") or []
    winter_active_series = summary.get("winter_active_series") or []
    if winter_price_series and winter_ticket_series:
        count = min(len(generations), len(winter_price_series), len(winter_ticket_series))
        if count > 0:
            xs = generations[:count]
            price_vals = [float(winter_price_series[i]) for i in range(count)]
            ticket_vals = [float(winter_ticket_series[i]) for i in range(count)]
            active_vals = [
                int(winter_active_series[i]) if i < len(winter_active_series) else 0
                for i in range(count)
            ]
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
            ax1.plot(xs, price_vals, label="price multiplier", color="#ff7f0e", linewidth=1.5)
            ax1.plot(xs, ticket_vals, label="ticket multiplier", color="#1f77b4", linewidth=1.5)
            ax1.set_ylabel("Multiplier")
            ax1.set_title("Winter multipliers")
            ax1.grid(alpha=0.3)
            ax1.legend()
            ax2.step(xs, active_vals, where="mid", color="#d62728")
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_ylabel("Active")
            ax2.set_xlabel("Generation")
            ax2.set_title("Winter active window")
            ax2.grid(alpha=0.3)
            plt.tight_layout()
            fig.savefig(output_dir / "winter_cycle.png")
            plt.close(fig)
    winter_events = summary.get("winter_events") or []
    if winter_events:
        events_path = output_dir / "winter_events.jsonl"
        events_path.write_text("\n".join(json.dumps(ev) for ev in winter_events))
    winter_event_counts = summary.get("winter_event_counts") or {}
    if winter_event_counts:
        plt.figure(figsize=(6, 3))
        labels = list(winter_event_counts.keys())
        values = [winter_event_counts[label] for label in labels]
        plt.bar(labels, values, color="#9467bd")
        plt.xlabel("Event type")
        plt.ylabel("Count")
        plt.title("Winter events")
        plt.tight_layout()
        plt.savefig(output_dir / "winter_events.png")
        plt.close()
    price_bias_counts = summary.get("survival_price_bias_counts") or []
    if price_bias_counts:
        plt.figure(figsize=(8, 4))
        plt.plot(generations, price_bias_counts, marker="o", linewidth=1)
        plt.xlabel("Generation")
        plt.ylabel("Price-bias active")
        plt.title("Price-aware routing guards")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "survival_price_bias.png")
        plt.close()
    survival_events = summary.get("survival_events") or []
    if survival_events:
        events_path = output_dir / "survival_events.jsonl"
        events_path.write_text("\n".join(json.dumps(ev) for ev in survival_events))
        counts = Counter(
            ev.get("type") for ev in survival_events if isinstance(ev.get("type"), str)
        )
        if counts:
            plt.figure(figsize=(6, 3))
            labels = list(counts.keys())
            values = [counts[label] for label in labels]
            plt.bar(labels, values, color="#c96dfd")
            plt.xlabel("Event type")
            plt.ylabel("Count")
            plt.title("Survival events")
            plt.tight_layout()
            plt.savefig(output_dir / "survival_events.png")
            plt.close()

    comms_posts = summary.get("comms_posts_series") or []
    comms_reads = summary.get("comms_reads_series") or []
    comms_credits = summary.get("comms_credits_series") or []
    if comms_posts:
        plt.figure(figsize=(8, 4))
        plt.plot(generations, comms_posts, marker="o", linewidth=1)
        plt.xlabel("Generation")
        plt.ylabel("Posts")
        plt.title("Comms posts per generation")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "comms_posts.png")
        plt.close()
    if comms_reads:
        plt.figure(figsize=(8, 4))
        plt.plot(generations, comms_reads, marker="o", linewidth=1)
        plt.xlabel("Generation")
        plt.ylabel("Reads")
        plt.title("Comms reads per generation")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "comms_reads.png")
        plt.close()
    if comms_credits:
        plt.figure(figsize=(8, 4))
        plt.plot(generations, comms_credits, marker="o", linewidth=1)
        plt.xlabel("Generation")
        plt.ylabel("Credits")
        plt.title("Comms credits per generation")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "comms_credits.png")
        plt.close()
    mutation_rank = summary.get("mutation_rank_noise_series") or []
    mutation_dropout = summary.get("mutation_dropout_series") or []
    mutation_duplication = summary.get("mutation_duplication_series") or []
    if mutation_rank or mutation_dropout or mutation_duplication:
        plt.figure(figsize=(8, 4))
        if mutation_rank:
            plt.plot(generations, mutation_rank, marker="o", linewidth=1, label="rank noise")
        if mutation_dropout:
            plt.plot(generations, mutation_dropout, marker="o", linewidth=1, label="dropout")
        if mutation_duplication:
            plt.plot(
                generations, mutation_duplication, marker="o", linewidth=1, label="duplication"
            )
        plt.xlabel("Generation")
        plt.ylabel("Count")
        plt.title("Mutation operators per generation")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "mutation_operators.png")
        plt.close()
    audit_counts = summary.get("merge_audit_count_series") or []
    audit_deltas = summary.get("merge_audit_delta_series") or []
    if audit_counts and any(audit_counts):
        plt.figure(figsize=(8, 4))
        plt.plot(generations, audit_counts, marker="o", linewidth=1, color="#ff7f0e")
        plt.xlabel("Generation")
        plt.ylabel("Audits")
        plt.title("Merge audits per generation")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "merge_audits_counts.png")
        plt.close()
    if audit_deltas and any(isinstance(val, (int, float)) for val in audit_deltas):
        plt.figure(figsize=(8, 4))
        deltas_numeric = [val if isinstance(val, (int, float)) else None for val in audit_deltas]
        plt.plot(generations, deltas_numeric, marker="o", linewidth=1, color="#2ca02c")
        plt.xlabel("Generation")
        plt.ylabel("ΔROI post-pre")
        plt.title("Merge audit ΔROI (mean per gen)")
        plt.axhline(0.0, color="#444", linewidth=0.8, linestyle="--")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "merge_audits_delta.png")
        plt.close()
    comms_events = summary.get("comms_events") or []
    if comms_events:
        events_path = output_dir / "comms_events.jsonl"
        events_path.write_text("\n".join(json.dumps(ev) for ev in comms_events))
        counts = Counter(ev.get("type") for ev in comms_events if isinstance(ev.get("type"), str))
        if counts:
            plt.figure(figsize=(6, 3))
            labels = list(counts.keys())
            values = [counts[label] for label in labels]
            plt.bar(labels, values, color="#2d96ff")
            plt.xlabel("Event type")
            plt.ylabel("Count")
            plt.title("Comms events")
            plt.tight_layout()
            plt.savefig(output_dir / "comms_events.png")
            plt.close()
    colony_band_series = summary.get("colony_bandwidth_series") or []
    if colony_band_series:
        plt.figure(figsize=(8, 4))
        plt.plot(
            generations[: len(colony_band_series)], colony_band_series, marker="o", linewidth=1
        )
        plt.xlabel("Generation")
        plt.ylabel("Bandwidth (energy)")
        plt.title("Colony bandwidth per generation")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "colony_bandwidth.png")
        plt.close()
    colony_size_series = summary.get("colony_size_total_series") or []
    if colony_size_series:
        plt.figure(figsize=(8, 4))
        plt.plot(
            generations[: len(colony_size_series)], colony_size_series, marker="o", linewidth=1
        )
        plt.xlabel("Generation")
        plt.ylabel("Total members")
        plt.title("Colony size (total members)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "colony_size_total.png")
        plt.close()
    colony_delta_series = summary.get("colony_delta_mean_series") or []
    if colony_delta_series:
        plt.figure(figsize=(8, 4))
        plt.plot(
            generations[: len(colony_delta_series)], colony_delta_series, marker="o", linewidth=1
        )
        plt.xlabel("Generation")
        plt.ylabel("ΔROI (mean)")
        plt.title("Colony holdout ΔROI")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "colony_delta_mean.png")
        plt.close()
    colony_var_series = summary.get("colony_variance_ratio_series") or []
    if colony_var_series:
        plt.figure(figsize=(8, 4))
        plt.plot(generations[: len(colony_var_series)], colony_var_series, marker="o", linewidth=1)
        plt.xlabel("Generation")
        plt.ylabel("Variance ratio")
        plt.title("Colony variance ratio")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "colony_variance_ratio.png")
        plt.close()
    tier_mean_series = summary.get("colony_tier_mean_series") or []
    if tier_mean_series:
        plt.figure(figsize=(8, 4))
        plt.plot(generations[: len(tier_mean_series)], tier_mean_series, marker="o", linewidth=1)
        plt.xlabel("Generation")
        plt.ylabel("Tier mean")
        plt.title("Colony tier mean")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "colony_tier_mean.png")
        plt.close()
    tier_counts_series = summary.get("colony_tier_counts_series") or []
    if tier_counts_series:
        tier_keys = sorted({int(k) for counts in tier_counts_series for k in counts.keys()})
        if tier_keys:
            xs = generations[: len(tier_counts_series)]
            ys = []
            for key in tier_keys:
                ys.append([int(counts.get(str(key), 0)) for counts in tier_counts_series])
            plt.figure(figsize=(8, 4))
            plt.stackplot(xs, *ys, labels=[str(key) for key in tier_keys])
            plt.xlabel("Generation")
            plt.ylabel("Colonies")
            plt.title("Colony tier counts")
            plt.legend(loc="upper left", fontsize=8)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "colony_tier_counts.png")
            plt.close()
    selection_pot_series = summary.get("colony_selection_pool_pot_series") or []
    if selection_pot_series:
        plt.figure(figsize=(8, 4))
        plt.plot(
            generations[: len(selection_pot_series)], selection_pot_series, marker="o", linewidth=1
        )
        plt.xlabel("Generation")
        plt.ylabel("Pool pot")
        plt.title("Colony selection pool (pot)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "colony_selection_pool_pot.png")
        plt.close()
    selection_members_series = summary.get("colony_selection_pool_members_series") or []
    if selection_members_series:
        plt.figure(figsize=(8, 4))
        plt.plot(
            generations[: len(selection_members_series)],
            selection_members_series,
            marker="o",
            linewidth=1,
        )
        plt.xlabel("Generation")
        plt.ylabel("Pool members")
        plt.title("Colony selection pool (members)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "colony_selection_pool_members.png")
        plt.close()
    selection_rep_series = summary.get("colony_selection_replicated_series") or []
    if selection_rep_series:
        plt.figure(figsize=(8, 4))
        plt.plot(
            generations[: len(selection_rep_series)], selection_rep_series, marker="o", linewidth=1
        )
        plt.xlabel("Generation")
        plt.ylabel("Replications")
        plt.title("Colony selection replications")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "colony_selection_replications.png")
        plt.close()
    qd_size_series = summary.get("qd_archive_size_series") or []
    if qd_size_series:
        plt.figure(figsize=(8, 4))
        plt.plot(generations[: len(qd_size_series)], qd_size_series, marker="o", linewidth=1)
        plt.xlabel("Generation")
        plt.ylabel("Archive size")
        plt.title("QD archive size")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "qd_archive_size.png")
        plt.close()
    qd_cov_series = summary.get("qd_archive_coverage_series") or []
    if qd_cov_series:
        plt.figure(figsize=(8, 4))
        plt.plot(
            generations[: len(qd_cov_series)],
            [value * 100.0 for value in qd_cov_series],
            marker="o",
            linewidth=1,
        )
        plt.xlabel("Generation")
        plt.ylabel("Coverage (%)")
        plt.title("QD archive coverage")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "qd_archive_coverage.png")
        plt.close()
    budget_series = summary.get("budget_totals_series") or []
    if budget_series:
        plt.figure(figsize=(8, 4))
        plt.plot(generations[: len(budget_series)], budget_series, marker="o", linewidth=1)
        plt.xlabel("Generation")
        plt.ylabel("Episodes")
        plt.title("Budgeted episodes per generation")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "budget_totals.png")
        plt.close()

    # Team routes/promotions per generation
    if any("team_routes" in rec for rec in records):
        values = [int(rec.get("team_routes", 0) or 0) for rec in records]
        plt.figure(figsize=(8, 4))
        plt.plot(generations, values, marker="o", linewidth=1)
        plt.xlabel("Generation")
        plt.ylabel("Team routes")
        plt.title("Team routes per generation")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "team_routes.png")
        plt.close()
    if any("team_promotions" in rec for rec in records):
        values = [int(rec.get("team_promotions", 0) or 0) for rec in records]
        plt.figure(figsize=(8, 4))
        plt.plot(generations, values, marker="o", linewidth=1)
        plt.xlabel("Generation")
        plt.ylabel("Team promotions")
        plt.title("Team promotions per generation")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "team_promotions.png")
        plt.close()

    # Promotion controller thresholds over generations (if present)
    if any("promotion_controller" in rec for rec in records):
        margin_vals, power_vals = [], []
        for rec in records:
            pc = rec.get("promotion_controller") or {}
            if isinstance(pc, dict):
                try:
                    margin_vals.append(float(pc.get("team_holdout_margin", float("nan"))))
                except Exception:
                    margin_vals.append(float("nan"))
                try:
                    power_vals.append(float(pc.get("team_min_power", float("nan"))))
                except Exception:
                    power_vals.append(float("nan"))
            else:
                margin_vals.append(float("nan"))
                power_vals.append(float("nan"))
        plt.figure(figsize=(8, 4))
        plt.plot(generations, margin_vals, label="team_holdout_margin", color="#1f77b4")
        plt.plot(generations, power_vals, label="team_min_power", color="#2ca02c")
        plt.xlabel("Generation")
        plt.ylabel("value")
        plt.title("Promotion controller thresholds")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "promotion_controller.png")
        plt.close()

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
        plt.xlabel("Pair")
        plt.ylabel("Generation")
        plt.title("Co-routing (top pairs)")
        plt.tight_layout()
        plt.savefig(output_dir / "co_routing_heatmap.png")
        plt.close()

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
    generation_line = f"- Generations: {summary['generations']}"
    records_count = summary.get("records")
    missing = summary.get("missing_generations")
    if isinstance(records_count, int):
        generation_line += f" (records: {records_count}"
        if isinstance(missing, int) and missing:
            generation_line += f", missing: {missing}"
        generation_line += ")"
    lines.append(generation_line)
    if summary.get("sparse_records"):
        lines.append(
            "- NOTE: `gen_summaries.jsonl` is sparse; aggregates below use recorded generations only."
        )
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
    lines.append(
        f"- Bankruptcy culls: total {summary['culled_total']} (max per generation {summary['culled_max']})"
    )
    lines.append(f"- Assimilation merges (per summary): {summary['total_merges']}")
    if summary.get("team_routes_total") is not None:
        lines.append(
            f"- Team routes / promotions: {summary.get('team_routes_total', 0)} / {summary.get('team_promotions_total', 0)}"
        )
        latest = summary.get("team_probe_candidates_latest") or []
        if summary.get("team_probe_hits_total") and latest:
            formatted = ", ".join(
                f"{tuple(cand.get('pair', []))}:S{cand.get('sustain', 0)}" for cand in latest[:5]
            )
            lines.append(
                f"  - Team probe sustained hits: {summary['team_probe_hits_total']} (latest: {formatted})"
            )
        probe_pairs = summary.get("team_probe_pairs") or {}
        if probe_pairs:
            items = ", ".join(f"{pair}:{cnt}" for pair, cnt in probe_pairs.items())
            lines.append(f"  - Team probe pairs (aggregate): {items}")
        team_gate_totals = summary.get("team_gate_totals") or {}
        if team_gate_totals:
            gate_str = ", ".join(f"{reason}:{count}" for reason, count in team_gate_totals.items())
            lines.append(f"  - Team gate reasons (sum): {gate_str}")
        latest_team_gate = summary.get("team_gate_latest") or {}
        if latest_team_gate:
            latest_gate_str = ", ".join(f"{k}:{v}" for k, v in latest_team_gate.items())
            lines.append(f"  - Latest team gate snapshot: {latest_gate_str}")
        team_gate_samples = summary.get("team_gate_samples_latest") or []
        if team_gate_samples:
            sample = team_gate_samples[-1]
            reason = sample.get("reason")
            same_answers = sample.get("same_answers")
            lines.append(
                f"  - Team gate sample: reason={reason}, tasks={sample.get('tasks')}, same_answers={same_answers}"
            )
    prompt_scaffolds = summary.get("prompt_scaffolds") or {}
    if prompt_scaffolds:
        scaffold_items = ", ".join(f"{fam}:{cnt}" for fam, cnt in prompt_scaffolds.items())
        lines.append(f"- Prompt scaffolds applied (latest generation): {scaffold_items}")
    if summary.get("assimilation_attempt_total"):
        lines.append(
            f"- Assimilation attempts (all stages): {summary['assimilation_attempt_total']}"
        )
    if summary["assimilation_gating_total"]:
        lines.append("- Assimilation gating totals:")
        for key, value in summary["assimilation_gating_total"].items():
            lines.append(f"  - {key}: {value}")
    if summary.get("population_refresh_total"):
        latest_refresh = summary.get("population_refresh_latest") or {}
        lines.append(
            f"- Population refreshes: {summary['population_refresh_total']} (latest: {latest_refresh})"
        )
    if summary.get("policy_applied_total"):
        lines.append(
            f"- Policy usage: {summary['policy_applied_total']}/{summary.get('records', summary['generations'])} records"
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
    if summary.get("policy_failures_total"):
        latest_failure = summary.get("policy_failure_latest") or []
        preview = ""
        if latest_failure:
            preview = latest_failure[-1].get("sample", "")
            if preview:
                preview = textwrap.shorten(preview, width=120, placeholder="…")
        suffix = f" (latest: {preview})" if preview else ""
        lines.append(f"- Policy parse failures: {summary['policy_failures_total']}{suffix}")
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
        if "fisher_importance_mean" in assimilation:
            lines.append(
                "  - Fisher importance mean/max: "
                f"{assimilation['fisher_importance_mean']:.3f} / {assimilation.get('fisher_importance_max', 0.0):.3f}"
            )
        if "merge_weight_mean" in assimilation:
            lines.append(f"  - Merge weight mean: {assimilation['merge_weight_mean']:.3f}")
        if "methods" in assimilation:
            method_items = ", ".join(
                f"{name}: {count}" for name, count in assimilation["methods"].items()
            )
            lines.append(f"  - Methods: {method_items}")
        if assimilation.get("dr_used"):
            lines.append(f"  - DR uplift applied in {assimilation['dr_used']} events")
    history_latest = summary.get("assimilation_history_latest") or {}
    if history_latest:
        lines.append("- Assimilation history (latest per cell):")
        for key, record in list(history_latest.items())[:5]:
            uplift = record.get("uplift")
            gen = record.get("generation")
            lines.append(
                f"  - {key}: gen {gen}, uplift {uplift:+.3f}"
                if isinstance(uplift, (int, float))
                else f"  - {key}: gen {gen}"
            )
    knowledge_totals = summary.get("knowledge_totals") or {}
    if knowledge_totals:
        lines.append(
            "- Knowledge cache: "
            f"writes {knowledge_totals.get('writes', 0)} (denied {knowledge_totals.get('write_denied', 0)}); "
            f"reads {knowledge_totals.get('reads', 0)} (denied {knowledge_totals.get('read_denied', 0)}, hits {knowledge_totals.get('hits', 0)})"
        )
        lines.append(
            f"  - Entries mean {summary.get('knowledge_entries_mean', 0.0):.2f}, "
            f"latest {summary.get('knowledge_entries_latest', 0)}; expired {knowledge_totals.get('expired', 0)}"
        )
    if summary.get("power_episodes_total"):
        lines.append("- Power economics:")
        lines.append(
            f"  - Episodes tracked: {summary['power_episodes_total']}; "
            f"avg power need {summary['power_need_mean']:.3f}; "
            f"avg price multiplier {summary['power_price_multiplier_mean']:.3f}"
        )
        lines.append(
            "  - Evidence tokens minted/used: "
            f"{summary['power_tokens_minted_total']} / {summary['power_tokens_used_total']}"
        )
        if summary.get("power_info_topups_total"):
            lines.append(f"  - Info-aware top-ups granted: {summary['power_info_topups_total']}")
    # Co-routing aggregate
    crt = summary.get("co_routing_totals") or {}
    if crt:
        lines.append("- Top co-routing pairs (aggregate):")
        for pair, cnt in crt.items():
            lines.append(f"  - {pair}: {cnt}")
    else:
        lines.append("- Assimilation tests: none recorded")
    event_counts = summary.get("colony_event_counts") or {}
    if event_counts:
        lines.append("- Colony events:")
        for etype, count in event_counts.items():
            lines.append(f"  - {etype}: {count}")
    if summary.get("colony_selection_dissolved_series"):
        lines.append(
            "- Colony selection: "
            f"dissolved {summary.get('colony_selection_dissolved_total', 0)} / "
            f"replicated {summary.get('colony_selection_replicated_total', 0)}"
        )
        lines.append(
            f"  - Pool mean members {summary.get('colony_selection_pool_members_mean', 0.0):.2f}; "
            f"pool mean pot {summary.get('colony_selection_pool_pot_mean', 0.0):.2f}"
        )
    if summary.get("colony_tier_mean_series"):
        lines.append(
            f"- Colony tier mean: avg {summary.get('colony_tier_mean', 0.0):.2f}, "
            f"last {summary['colony_tier_mean_series'][-1]:.2f}"
        )
        totals = summary.get("colony_tier_counts_total") or {}
        if totals:
            totals_str = ", ".join(f"{tier}:{count}" for tier, count in totals.items())
            lines.append(f"  - Tier totals: {totals_str}")
    if summary.get("colony_reserve_active_series") is not None:
        lines.append(
            f"- Colony reserve guard (mean active colonies): {summary.get('colony_reserve_active_mean', 0.0):.2f}"
        )
    if summary.get("colony_winter_active_series") is not None:
        lines.append(
            f"- Colony winter mode (mean active colonies): {summary.get('colony_winter_active_mean', 0.0):.2f}"
        )
    if summary.get("colony_hazard_z_mean") is not None:
        lines.append(
            f"- Colony hazard z-score (mean): {summary.get('colony_hazard_z_mean', 0.0):.3f}"
        )
    if summary.get("foraging_traits_latest"):
        lines.append(
            "- Foraging traits (mean): "
            f"beta {summary.get('foraging_beta_mean', 0.0):.2f}, "
            f"decay {summary.get('foraging_decay_mean', 0.0):.2f}, "
            f"ucb {summary.get('foraging_ucb_mean', 0.0):.2f}, "
            f"budget {summary.get('foraging_budget_mean', 0.0):.2f}"
        )
        top_cells = summary.get("foraging_top_cells_latest") or {}
        if isinstance(top_cells, dict) and top_cells:
            sample_org, cells = next(iter(top_cells.items()))
            if cells:
                headline = ", ".join(
                    f"{c['family']}:{c['depth']} ({c['q']:.2f})" for c in cells[:3]
                )
                lines.append(f"  - {sample_org} top cells: {headline}")
    winter_counts = summary.get("winter_event_counts") or {}
    if winter_counts:
        lines.append("- Winter cycle:")
        lines.append(
            f"  - Active ratio: {summary.get('winter_active_mean', 0.0):.2f}; "
            f"avg multipliers price {summary.get('winter_price_mean', 0.0):.2f}, ticket {summary.get('winter_ticket_mean', 0.0):.2f}"
        )
        lines.append(
            f"  - Mean ΔROI after winter: {summary.get('winter_roi_delta_mean', 0.0):+.3f}; "
            f"mean Δassim attempts: {summary.get('winter_assim_delta_mean', 0.0):+.2f}"
        )
        if summary.get("winter_cull_total"):
            lines.append(f"  - Winter bankrupt culls: {int(summary['winter_cull_total'])}")
        for etype, count in winter_counts.items():
            lines.append(f"  - {etype}: {count}")
    survival_counts = summary.get("survival_event_counts") or {}
    if survival_counts:
        lines.append("- Survival events:")
        for etype, count in survival_counts.items():
            lines.append(f"  - {etype}: {count}")
    latest_survival = summary.get("survival_latest") or {}
    if latest_survival:
        reserve_count = latest_survival.get("reserve_active_count")
        hazard_count = latest_survival.get("hazard_active_count")
        if reserve_count is None:
            reserve_ids = latest_survival.get("reserve_active_ids") or []
            reserve_count = len(reserve_ids)
        if hazard_count is None:
            hazard_ids = latest_survival.get("hazard_active_ids") or []
            hazard_count = len(hazard_ids)
        price_count = latest_survival.get("price_bias_active_count")
        if price_count is None:
            price_ids = latest_survival.get("price_bias_active_ids") or []
            price_count = len(price_ids)
        lines.append(
            f"- Survival snapshot (last gen): reserve {int(reserve_count or 0)}, hazard {int(hazard_count or 0)}, price-bias {int(price_count or 0)}"
        )
    colony_band_series = summary.get("colony_bandwidth_series") or []
    if colony_band_series:
        lines.append(
            f"- Colony bandwidth: mean {summary['colony_bandwidth_mean']:.3f}, last {colony_band_series[-1]:.3f}"
        )
        colony_size_series = summary.get("colony_size_total_series") or []
        if colony_size_series:
            lines.append(
                f"  total members mean {summary['colony_size_mean']:.2f}, last {colony_size_series[-1]}"
            )
        lines.append(
            f"  ΔROI mean {summary['colony_delta_overall_mean']:.4f}; variance ratio mean {summary['colony_variance_ratio_mean']:.3f}"
        )
        hazard_series = summary.get("colony_hazard_members_series") or []
        if hazard_series:
            lines.append(f"  hazard members (max) {max(hazard_series)}")
    if summary.get("budget_totals_series"):
        lines.append(
            f"- Budget totals: mean {summary['budget_totals_mean']:.2f}, median {summary['budget_totals_median']:.2f}, last {summary['budget_final_last']}"
        )
        if summary.get("budget_cap_max", 0):
            lines.append(
                f"  cap max {summary['budget_cap_max']} (hit-rate {summary['budget_cap_hit_rate']*100:.1f}%)"
            )
        lines.append(
            f"  zero-alloc mean {summary['budget_zero_mean']:.2f}; energy ratio mean {summary['budget_energy_mean']:.2f}; trait mean {summary['budget_trait_mean']:.2f}; policy mean {summary['budget_policy_mean']:.2f}"
        )
    comms_posts = summary.get("comms_posts_series") or []
    comms_reads = summary.get("comms_reads_series") or []
    comms_credits = summary.get("comms_credits_series") or []
    if comms_posts or comms_reads or comms_credits:
        lines.append(
            f"- Comms totals: posts {sum(comms_posts)} / reads {sum(comms_reads)} / credits {sum(comms_credits)}"
        )
    mutation_totals = summary.get("mutation_totals") or {}
    if mutation_totals:
        lines.append("- Mutation operators invoked:")
        lines.append(
            "  - rank noise: {rank_noise}, dropout masks: {dropout}, duplications: {duplication}".format(
                rank_noise=int(mutation_totals.get("rank_noise", 0)),
                dropout=int(mutation_totals.get("dropout", 0)),
                duplication=int(mutation_totals.get("duplication", 0)),
            )
        )
    audit_total = int(summary.get("merge_audit_total", 0) or 0)
    if audit_total:
        lines.append(
            f"- Merge audits: {audit_total} total; mean ΔROI {summary.get('merge_audit_delta_mean', 0.0):+.3f}"
        )
        audit_family = summary.get("merge_audit_family") or {}
        if audit_family:
            lines.append("  - Merge audits by family:")
            for family, stats in audit_family.items():
                lines.append(
                    "    * {family}: count {count}, Δμ {mean:+.3f}, worst {worst:+.3f}".format(
                        family=family,
                        count=int(stats.get("count", 0)),
                        mean=float(stats.get("delta_mean", 0.0)),
                        worst=float(stats.get("delta_min", 0.0)),
                    )
                )
    comms_counts = summary.get("comms_event_counts") or {}
    if comms_counts:
        lines.append("- Comms events:")
        for etype, count in comms_counts.items():
            lines.append(f"  - {etype}: {count}")
    board_latest = summary.get("comms_board_latest") or []
    if board_latest:
        lines.append("- Comms board (latest posts):")
        for entry in board_latest:
            lines.append(
                f"  - {entry['organelle_id']}→{entry.get('topic') or 'general'}: {entry['text']}"
            )
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
            audit_block = attempt.get("audit") or {}
            if isinstance(audit_block, dict) and audit_block:
                lines.append(
                    "    audit tasks={tasks} pre={pre} post={post} Δ={delta}".format(
                        tasks=audit_block.get("tasks"),
                        pre=audit_block.get("pre_roi"),
                        post=audit_block.get("post_roi"),
                        delta=audit_block.get("delta"),
                    )
                )
    else:
        lines.append("- Assimilation tests: none recorded")
    family_summary = summary.get("assimilation_family_summary") or {}
    if family_summary:
        lines.append("- Assimilation (recent window) by family:")
        for family, stats in family_summary.items():
            lines.append(
                "  - {family}: {passes}/{attempts} passes ({rate:.1f}%), uplift μ {uplift:+.3f}, holdout ROI μ {roi:+.3f}, audit Δμ {delta:+.3f}".format(
                    family=family,
                    passes=int(stats.get("passes", 0)),
                    attempts=int(stats.get("attempts", 0)),
                    rate=float(stats.get("pass_rate", 0.0)) * 100.0,
                    uplift=float(stats.get("uplift_mean", 0.0)),
                    roi=float(stats.get("candidate_roi_mean", 0.0)),
                    delta=float(stats.get("audit_delta_mean", 0.0)),
                )
            )
    if summary.get("diversity_samples", 0):
        lines.append(
            f"- Diversity: mean energy Gini {summary['diversity_energy_gini_mean']:.3f}, effective pop {summary['diversity_effective_population_mean']:.2f}, max species share {summary['diversity_max_species_share_mean']:.3f}, enforcement rate {summary['diversity_enforced_rate']*100:.1f}%"
        )
    if summary.get("qd_coverage") is not None:
        lines.append(f"- QD coverage: {summary['qd_coverage']}")
        if summary.get("qd_archive_size_series"):
            size_series = summary.get("qd_archive_size_series") or []
            coverage_series = summary.get("qd_archive_coverage_series") or []
            last_size = size_series[-1] if size_series else 0
            last_cov = coverage_series[-1] if coverage_series else 0.0
            lines.append(
                f"  archive size mean {summary.get('qd_archive_size_mean', 0.0):.1f}, last {last_size}; coverage {summary.get('qd_archive_coverage_mean', 0.0)*100:.1f}% (last {last_cov*100:.1f}%)"
            )
            top_bins = summary.get("qd_archive_top_latest") or []
            if top_bins:
                formatted = ", ".join(
                    f"{entry['cell']}[b{entry['bin']}]:{entry['roi']:.3f}" for entry in top_bins
                )
                lines.append(f"  top bins: {formatted}")
    if summary.get("roi_volatility") is not None:
        lines.append(f"- ROI volatility (std across organelles): {summary['roi_volatility']:.3f}")
    if summary.get("trials_total") or summary.get("promotions_total"):
        lines.append(
            f"- Trials/promotions: {summary.get('trials_total', 0)} / {summary.get('promotions_total', 0)}"
        )
    if summary["eval_events"]:
        accuracy_pct = summary["eval_accuracy_mean"] * 100
        lines.append(
            f"- Evaluation accuracy: {accuracy_pct:.2f}% ({summary['eval_correct']}/{summary['eval_total']})"
        )
        if summary["eval_reward_weight"] is not None:
            lines.append(f"  (reward weight {summary['eval_reward_weight']})")
        eval_family = summary.get("evaluation_family_stats") or {}
        if eval_family:
            lines.append("  - Evaluation by family:")
            for family, stats in eval_family.items():
                lines.append(
                    "    * {family}: {accuracy:.2f}% ({correct}/{total}), ΔROI μ {delta:+.3f}, cost μ {cost:.3f}".format(
                        family=family,
                        accuracy=float(stats.get("accuracy", 0.0)) * 100.0,
                        correct=int(stats.get("correct", 0)),
                        total=int(stats.get("total", 0)),
                        delta=float(stats.get("avg_delta", 0.0)),
                        cost=float(stats.get("avg_cost", 0.0)),
                    )
                )
    else:
        lines.append("- Evaluation accuracy: n/a")

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse an ecology run directory")
    parser.add_argument("run_dir", type=Path, help="Directory containing gen_summaries.jsonl")
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate plots (saved under <run_dir>/visuals)",
        default=False,
    )
    parser.add_argument(
        "--report", action="store_true", help="Write report.md in the run directory", default=False
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    gen_records = load_jsonl(run_dir / "gen_summaries.jsonl")
    summary = summarise_generations(gen_records)
    assimilation_summary = summarise_assimilation(run_dir / "assimilation.jsonl")

    episodes_path = run_dir / "episodes.jsonl"
    if episodes_path.exists():
        try:
            with episodes_path.open("rb") as handle:
                summary["episodes_total"] = sum(1 for _ in handle)
        except Exception:
            pass

    gen_line = f"{summary['generations']}"
    if isinstance(summary.get("records"), int):
        gen_line += f" (records {summary['records']}"
        if int(summary.get("missing_generations", 0) or 0):
            gen_line += f", missing {summary['missing_generations']}"
        gen_line += ")"
    print("Generations:", gen_line)
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
        "Active organelles range:",
        summary["active_min"],
        "-",
        summary["active_max"],
        ", bankrupt:",
        summary["bankrupt_min"],
        "-",
        summary["bankrupt_max"],
    )
    print("Total merges:", summary["total_merges"])
    print("Bankruptcy culls (total/max):", summary["culled_total"], summary["culled_max"])
    if summary["assimilation_gating_total"]:
        print("Assimilation gating totals:")
        for key, value in summary["assimilation_gating_total"].items():
            print(f"  {key}: {value}")
    if summary.get("population_refresh_total"):
        print(
            "Population refreshes:",
            summary["population_refresh_total"],
            summary.get("population_refresh_latest"),
        )
    if summary.get("assimilation_gating_reasons_samples"):
        print("Gating reasons (samples across gens):")
        for key, value in summary["assimilation_gating_reasons_samples"].items():
            print(f"  {key}: {value}")
    if summary.get("colonies_count_series"):
        series = summary["colonies_count_series"]
        print("Colonies count (min-max):", min(series), "-", max(series))
        print("Avg colony size (mean):", f"{summary['colonies_avg_size_mean']:.2f}")
    if summary.get("colony_event_counts"):
        events_str = ", ".join(f"{k}:{v}" for k, v in summary["colony_event_counts"].items())
        print("Colony events:", events_str)
    if summary.get("qd_archive_size_series"):
        size_series = summary.get("qd_archive_size_series") or []
        coverage_series = summary.get("qd_archive_coverage_series") or []
        if size_series:
            print(
                "QD archive size (mean/last):",
                f"{summary.get('qd_archive_size_mean', 0.0):.1f}",
                "|",
                size_series[-1],
            )
        if coverage_series:
            print(
                "QD archive coverage (mean/last):",
                f"{summary.get('qd_archive_coverage_mean', 0.0)*100:.1f}%",
                "|",
                f"{coverage_series[-1]*100:.1f}%",
            )
        top_bins = summary.get("qd_archive_top_latest") or []
        if top_bins:
            top_fmt = ", ".join(
                f"{entry['cell']}[b{entry['bin']}]:{entry['roi']:.3f}" for entry in top_bins
            )
            print("QD archive top bins:", top_fmt)
    if summary.get("colony_selection_dissolved_series"):
        print(
            "Colony selection (dissolved/replicated):",
            summary.get("colony_selection_dissolved_total", 0),
            "/",
            summary.get("colony_selection_replicated_total", 0),
        )
        print(
            "  Selection pool (members mean / pot mean):",
            f"{summary.get('colony_selection_pool_members_mean', 0.0):.2f}",
            "/",
            f"{summary.get('colony_selection_pool_pot_mean', 0.0):.2f}",
        )
    if summary.get("survival_reserve_counts"):
        reserve_counts = summary["survival_reserve_counts"]
        print("Survival reserve-active (min-max):", min(reserve_counts), "-", max(reserve_counts))
    if summary.get("survival_hazard_counts"):
        hazard_counts = summary["survival_hazard_counts"]
        print("Survival hazard-active (min-max):", min(hazard_counts), "-", max(hazard_counts))
    if summary.get("survival_price_bias_counts"):
        price_counts = summary["survival_price_bias_counts"]
        print("Survival price-bias active (min-max):", min(price_counts), "-", max(price_counts))
    if summary.get("survival_event_counts"):
        events_str = ", ".join(f"{k}:{v}" for k, v in summary["survival_event_counts"].items())
        print("Survival events:", events_str)
    if summary.get("colony_bandwidth_series"):
        band_series = summary["colony_bandwidth_series"]
        print(
            "Colony bandwidth total (mean/last):",
            f"{summary['colony_bandwidth_mean']:.3f}",
            "|",
            f"{band_series[-1]:.3f}",
        )
        size_series = summary.get("colony_size_total_series") or []
        if size_series:
            print(
                "Colony total members (mean/last):",
                f"{summary['colony_size_mean']:.2f}",
                "|",
                size_series[-1],
            )
        print(
            "Colony ΔROI mean (overall):",
            f"{summary['colony_delta_overall_mean']:.4f}",
        )
    tier_series = summary.get("colony_tier_mean_series") or []
    if tier_series:
        print(
            "Colony tier mean (mean/last):",
            f"{summary.get('colony_tier_mean', 0.0):.2f}",
            "|",
            f"{tier_series[-1]:.2f}",
        )
        totals = summary.get("colony_tier_counts_total") or {}
        if totals:
            totals_str = ", ".join(f"{tier}:{count}" for tier, count in totals.items())
            print("Colony tier totals:", totals_str)
    print(
        "Colony variance ratio mean:",
        f"{summary['colony_variance_ratio_mean']:.3f}",
    )
    hazard_series = summary.get("colony_hazard_members_series") or []
    if hazard_series:
        print(
            "Colony members in hazard (max):",
            max(hazard_series),
        )
    if summary.get("colony_reserve_active_series") is not None:
        print(
            "Colony reserve guard mean active:",
            f"{summary.get('colony_reserve_active_mean', 0.0):.2f}",
        )
    if summary.get("colony_winter_active_series") is not None:
        print(
            "Colony winter mode mean active:",
            f"{summary.get('colony_winter_active_mean', 0.0):.2f}",
        )
    if summary.get("colony_hazard_z_series") is not None:
        print(
            "Colony hazard z-score mean:",
            f"{summary.get('colony_hazard_z_mean', 0.0):.3f}",
        )
    if summary.get("winter_event_counts"):
        print(
            "Winter active ratio / multipliers:",
            f"{summary.get('winter_active_mean', 0.0):.2f}",
            "|",
            f"{summary.get('winter_price_mean', 0.0):.2f}",
            "/",
            f"{summary.get('winter_ticket_mean', 0.0):.2f}",
        )
        print(
            "Winter ΔROI / Δassim (mean):",
            f"{summary.get('winter_roi_delta_mean', 0.0):+.3f}",
            "/",
            f"{summary.get('winter_assim_delta_mean', 0.0):+.2f}",
        )
        if summary.get("winter_cull_total"):
            print("Winter bankrupt culls:", int(summary["winter_cull_total"]))
        events_str = ", ".join(f"{k}:{v}" for k, v in summary["winter_event_counts"].items())
        print("Winter events:", events_str)
    if summary.get("budget_totals_series"):
        print(
            "Budget totals (mean/median/last):",
            f"{summary['budget_totals_mean']:.2f}",
            f"{summary['budget_totals_median']:.2f}",
            summary["budget_final_last"],
        )
        if summary.get("budget_cap_max", 0):
            print(
                "Budget cap (max/hit-rate):",
                summary["budget_cap_max"],
                f"{summary['budget_cap_hit_rate']*100:.1f}%",
            )
        print(
            "Budget zero-alloc mean:",
            f"{summary['budget_zero_mean']:.2f}",
            "| energy mean:",
            f"{summary['budget_energy_mean']:.2f}",
            "| trait mean:",
            f"{summary['budget_trait_mean']:.2f}",
            "| policy mean:",
            f"{summary['budget_policy_mean']:.2f}",
        )
    if summary.get("comms_posts_series"):
        posts = summary["comms_posts_series"]
        reads = summary.get("comms_reads_series") or []
        credits = summary.get("comms_credits_series") or []
        print(
            "Comms activity (posts/reads/credits totals):",
            sum(posts),
            "/",
            sum(reads) if reads else 0,
            "/",
            sum(credits) if credits else 0,
        )
    if summary.get("comms_event_counts"):
        events_str = ", ".join(f"{k}:{v}" for k, v in summary["comms_event_counts"].items())
        print("Comms events:", events_str)
    board_latest = summary.get("comms_board_latest") or []
    if board_latest:
        formatted = ", ".join(
            f"{entry['organelle_id']}→{entry.get('topic') or 'general'}:{entry['text']}"
            for entry in board_latest
        )
        print("Comms board (latest):", formatted)
    if summary.get("policy_applied_total"):
        print(
            "Policy usage: gens with policy:",
            summary["policy_applied_total"],
            "/",
            summary.get("records", summary["generations"]),
        )
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
    if summary.get("policy_failures_total"):
        latest_failure = summary.get("policy_failure_latest") or []
        preview = ""
        if latest_failure:
            preview = latest_failure[-1].get("sample", "")
            if preview:
                preview = textwrap.shorten(preview, width=80, placeholder="…")
        print("Policy parse failures:", summary["policy_failures_total"], preview)
    # Team totals to console
    if summary.get("team_routes_total") is not None:
        print(
            "Team routes / promotions:",
            summary.get("team_routes_total", 0),
            "/",
            summary.get("team_promotions_total", 0),
        )
        latest = summary.get("team_probe_candidates_latest") or []
        if summary.get("team_probe_hits_total") and latest:
            formatted = ", ".join(
                f"{tuple(cand.get('pair', []))}:S{cand.get('sustain', 0)}" for cand in latest[:5]
            )
            print(
                "Team probe sustained hits:",
                summary["team_probe_hits_total"],
                "| latest:",
                formatted,
            )
        probe_pairs = summary.get("team_probe_pairs") or {}
        if probe_pairs:
            pairs_fmt = ", ".join(f"{pair}:{cnt}" for pair, cnt in probe_pairs.items())
            print("Team probe pairs (aggregate):", pairs_fmt)
        team_gate_totals = summary.get("team_gate_totals") or {}
        if team_gate_totals:
            gate_str = ", ".join(f"{reason}:{count}" for reason, count in team_gate_totals.items())
            print("Team gate reasons (sum):", gate_str)
        latest_team_gate = summary.get("team_gate_latest") or {}
        if latest_team_gate:
            latest_gate_str = ", ".join(f"{k}:{v}" for k, v in latest_team_gate.items())
            print("Team gate snapshot (latest):", latest_gate_str)
    prompt_scaffolds = summary.get("prompt_scaffolds") or {}
    if prompt_scaffolds:
        scaffold_fmt = ", ".join(f"{fam}:{cnt}" for fam, cnt in prompt_scaffolds.items())
        print("Prompt scaffolds applied (latest gen):", scaffold_fmt)
    if summary.get("co_routing_totals"):
        tops = ", ".join(f"{k}:{v}" for k, v in summary["co_routing_totals"].items())
        print("Top co-routing pairs:", tops)
    if summary["eval_events"]:
        accuracy_pct = summary["eval_accuracy_mean"] * 100
        print(
            "Evaluation accuracy (avg | total):",
            f"{accuracy_pct:.2f}%",
            f"{summary['eval_correct']}/{summary['eval_total']}",
        )
    else:
        print("Evaluation accuracy: n/a")
    if assimilation_summary["events"]:
        print(
            "Assimilation events:",
            assimilation_summary["events"],
            "passes:",
            assimilation_summary["passes"],
            "failures:",
            assimilation_summary["failures"],
        )
        if "sample_size_mean" in assimilation_summary:
            print(
                "Assimilation mean sample size:", f"{assimilation_summary['sample_size_mean']:.1f}"
            )
        if "ci_excludes_zero_rate" in assimilation_summary:
            print(
                "Uplift CI excludes zero (rate):",
                f"{assimilation_summary['ci_excludes_zero_rate']*100:.1f}%",
            )
        if "power_mean" in assimilation_summary:
            print("Power (proxy) mean:", f"{assimilation_summary['power_mean']:.2f}")
        if "fisher_importance_mean" in assimilation_summary:
            print(
                "Fisher importance mean/max:",
                f"{assimilation_summary['fisher_importance_mean']:.3f}",
                "/",
                f"{assimilation_summary.get('fisher_importance_max', 0.0):.3f}",
            )
        if "merge_weight_mean" in assimilation_summary:
            print("Merge weight mean:", f"{assimilation_summary['merge_weight_mean']:.3f}")
        if "methods" in assimilation_summary:
            method_items = ", ".join(
                f"{name}: {count}" for name, count in assimilation_summary["methods"].items()
            )
            print("Methods:", method_items)
        if assimilation_summary.get("dr_used"):
            print("DR uplift events:", assimilation_summary["dr_used"])
        if assimilation_summary.get("dr_strata_top"):
            top = assimilation_summary["dr_strata_top"]
            items = ", ".join(f"{k}:{v}" for k, v in top)
            print("DR contributing strata (top):", items)
    else:
        print("No assimilation events recorded")
    history_latest = summary.get("assimilation_history_latest") or {}
    if history_latest:
        preview = []
        for key, record in list(history_latest.items())[:5]:
            uplift = record.get("uplift")
            gen = record.get("generation")
            if isinstance(uplift, (int, float)):
                preview.append(f"{key}@{gen}:{uplift:+.3f}")
            else:
                preview.append(f"{key}@{gen}")
        print("Assimilation history (latest):", ", ".join(preview))
    knowledge_totals = summary.get("knowledge_totals") or {}
    if knowledge_totals:
        print(
            "Knowledge cache:",
            "writes",
            knowledge_totals.get("writes", 0),
            "| reads",
            knowledge_totals.get("reads", 0),
            "| hits",
            knowledge_totals.get("hits", 0),
            "| entries latest",
            summary.get("knowledge_entries_latest", 0),
        )
    if summary.get("power_episodes_total"):
        print(
            "Power economics:",
            "episodes",
            summary["power_episodes_total"],
            "| avg need",
            f"{summary['power_need_mean']:.2f}",
            "| avg price x",
            f"{summary['power_price_multiplier_mean']:.2f}",
        )
        print(
            "  Evidence tokens minted/used:",
            summary["power_tokens_minted_total"],
            "/",
            summary["power_tokens_used_total"],
            "| info top-ups:",
            summary.get("power_info_topups_total", 0),
        )
    if summary.get("trials_total") or summary.get("promotions_total"):
        print(
            "Trials created:",
            summary.get("trials_total", 0),
            "Promotions:",
            summary.get("promotions_total", 0),
        )
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
