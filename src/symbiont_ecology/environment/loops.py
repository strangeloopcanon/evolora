"""Outer training loops for the ecology."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev, quantiles
from typing import Dict, Optional, Tuple
import math
import random

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.human import HumanBandit, HumanFeedbackResult
from symbiont_ecology.environment.grid import GridEnvironment, GridKey, GridTask
from symbiont_ecology.evolution.assimilation import AssimilationTester
from symbiont_ecology.evolution.meta import MetaEvolver
from symbiont_ecology.evolution.morphogenesis import MorphogenesisController
from symbiont_ecology.evolution.population import Genome, PopulationManager
from symbiont_ecology.evaluation import EvaluationManager
from symbiont_ecology.evaluation.manager import EvaluationTask
from symbiont_ecology.host.kernel import HostKernel, RouteMetrics
from symbiont_ecology.metrics.persistence import TelemetrySink
from symbiont_ecology.metrics.telemetry import EpisodeLog, RewardBreakdown


@dataclass
class EnergyGuardBanditArm:
    weight: float
    floor: float = 0.0
    roi: float = 0.0
    pulls: int = 0
    reward: float = 0.0


class EnergyGuardBandit:
    def __init__(self, weights: list[float], seed: int = 4242) -> None:
        self.arms: list[EnergyGuardBanditArm] = [EnergyGuardBanditArm(weight=w) for w in weights]
        self.last_choice: int | None = None
        self.total_pulls = 0
        self.rng = random.Random(seed)
        self.last_reward: float = 0.0

    def record_reward(self, reward: float) -> None:
        if self.last_choice is None:
            return
        arm = self.arms[self.last_choice]
        arm.pulls += 1
        arm.reward += reward
        self.total_pulls += 1
        self.last_reward = reward
        self.last_choice = None

    def select(self, epsilon: float = 0.1) -> int:
        # ensure each arm is tried at least once
        for idx, arm in enumerate(self.arms):
            if arm.pulls == 0 and self.last_choice is None:
                self.last_choice = idx
                return idx
        if self.rng.random() < epsilon:
            choice = self.rng.randint(0, len(self.arms) - 1)
            self.last_choice = choice
            return choice
        total = max(1, self.total_pulls)
        best_idx = 0
        best_score = float("-inf")
        for idx, arm in enumerate(self.arms):
            avg = arm.reward / arm.pulls if arm.pulls > 0 else 0.0
            bonus = math.sqrt(2.0 * math.log(total + 1) / arm.pulls) if arm.pulls > 0 else float("inf")
            score = avg + bonus
            if score > best_score:
                best_score = score
                best_idx = idx
        self.last_choice = best_idx
        return best_idx


@dataclass
class EcologyLoop:
    config: EcologyConfig
    host: HostKernel
    environment: GridEnvironment
    population: PopulationManager
    assimilation: AssimilationTester
    human_bandit: Optional[HumanBandit] = None
    sink: Optional[TelemetrySink] = None
    logs: list[EpisodeLog] = field(default_factory=list)
    generation_index: int = 0
    assimilation_cooldown: Dict[Tuple[str, GridKey], int] = field(default_factory=dict)
    bankruptcy_strikes: Dict[str, int] = field(default_factory=dict)
    morphogenesis: MorphogenesisController | None = None
    meta_evolver: MetaEvolver | None = None
    evaluation_manager: EvaluationManager | None = None
    bandit_counter: int = 0
    holdout_rng: random.Random = field(default_factory=lambda: random.Random(7331))
    _holdout_tasks_cache: list[EvaluationTask] | None = field(default=None, init=False, repr=False)
    _diversity_snapshot: dict[str, float] | None = field(default=None, init=False, repr=False)
    no_merge_counter: int = 0
    assim_fail_streak: int = 0
    assim_gating_samples: list[dict[str, object]] = field(default_factory=list, init=False, repr=False)
    assim_attempt_samples: list[dict[str, object]] = field(default_factory=list, init=False, repr=False)
    _pending_hints: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    trial_offspring: dict[str, dict[str, object]] = field(default_factory=dict, init=False, repr=False)
    promotions_this_gen: int = 0
    trial_creations_this_gen: int = 0
    _qd_archive: dict[tuple[str, str, int], dict[str, object]] = field(default_factory=dict, init=False, repr=False)
    _synergy_window: list[dict[str, object]] = field(default_factory=list, init=False, repr=False)
    colonies: dict[str, dict[str, object]] = field(default_factory=dict, init=False, repr=False)
    _lp_mix_history: list[float] = field(default_factory=list, init=False, repr=False)
    _last_lp_mix: float = field(default=0.0, init=False, repr=False)
    _base_lp_mix: float = field(default=0.0, init=False, repr=False)
    _tau_relief: dict[GridKey, float] = field(default_factory=dict, init=False, repr=False)
    _tau_fail_counts: dict[GridKey, int] = field(default_factory=dict, init=False, repr=False)
    _roi_relief: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _topup_fail_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _active_policies: dict[str, dict[str, object]] = field(default_factory=dict, init=False, repr=False)
    _assim_attempt_total: int = field(default=0, init=False, repr=False)
    _policy_cost_total: float = field(default=0.0, init=False, repr=False)
    _policy_attempts_gen: int = field(default=0, init=False, repr=False)
    _policy_parsed_gen: int = field(default=0, init=False, repr=False)
    _co_routing_counts: dict[tuple[str, str], int] = field(default_factory=dict, init=False, repr=False)

    def run_generation(self, batch_size: int) -> None:
        self.generation_index += 1
        self.promotions_this_gen = 0
        self.trial_creations_this_gen = 0
        self._assim_attempt_total = 0
        self._policy_attempts_gen = 0
        self._policy_parsed_gen = 0
        if self.morphogenesis is None:
            self.morphogenesis = MorphogenesisController(config=self.config, host=self.host)
        if self.config.evaluation.enabled and self.evaluation_manager is None:
            runtime = EvaluationManager.from_file(
                enabled=self.config.evaluation.enabled,
                cadence=self.config.evaluation.cadence,
                tasks_path=self.config.evaluation.tasks_path,
                sample_size=self.config.evaluation.sample_size,
                reward_weight=self.config.evaluation.reward_weight,
            )
            self.evaluation_manager = EvaluationManager(runtime)
        organelle_ids = self.host.list_organelle_ids()
        if not organelle_ids:
            return

        ticket = self.config.energy.m
        active: list[str] = []
        bankrupt: list[str] = []
        grace = max(1, self.config.energy.bankruptcy_grace)
        for organelle_id in organelle_ids:
            balance = self.host.ledger.energy_balance(organelle_id)
            if balance < ticket:
                bankrupt.append(organelle_id)
                continue
            self.host.ledger.consume_energy(organelle_id, ticket)
            active.append(organelle_id)

        culled_bankrupt: list[str] = []
        for organelle_id in organelle_ids:
            if organelle_id in bankrupt:
                strikes = self.bankruptcy_strikes.get(organelle_id, 0) + 1
                self.bankruptcy_strikes[organelle_id] = strikes
            else:
                self.bankruptcy_strikes[organelle_id] = 0
        for organelle_id, strikes in list(self.bankruptcy_strikes.items()):
            if strikes >= grace:
                culled_bankrupt.append(organelle_id)
                self.bankruptcy_strikes.pop(organelle_id, None)
                if organelle_id in active:
                    active.remove(organelle_id)
                if organelle_id in bankrupt:
                    bankrupt.remove(organelle_id)
                self.environment.organism_stats.pop(organelle_id, None)
                for key in list(self.assimilation_cooldown.keys()):
                    if key[0] == organelle_id:
                        self.assimilation_cooldown.pop(key, None)
                self.host.retire_organelle(organelle_id)
                self.population.remove(organelle_id)

        for organelle_id in bankrupt:
            self.population.record_score(organelle_id, 0.0)
            self.population.record_energy(organelle_id, 0.0)
            self.population.record_roi(organelle_id, 0.0)

        # Prepare colony comms budgets (bandwidth + per-gen caps)
        if self.colonies:
            bw_frac = float(getattr(self.config.assimilation_tuning, "colony_bandwidth_frac", 0.02))
            post_cap = int(getattr(self.config.assimilation_tuning, "colony_post_cap", 2))
            read_cap = int(getattr(self.config.assimilation_tuning, "colony_read_cap", 3))
            for cid, meta in self.colonies.items():
                pot = float(meta.get("pot", 0.0))
                meta["bandwidth_left"] = max(0.0, min(pot, bw_frac * pot))
                meta["posts_left"] = post_cap
                meta["reads_left"] = read_cap

        # Optional: organism policy step (JSON intents)
        if bool(getattr(self.config.policy, "enabled", False)):
            for organelle_id in active:
                try:
                    self._request_and_apply_policy(organelle_id)
                except Exception:
                    pass

        # Lightweight co‑routing probes to inform team pairing
        try:
            self._probe_co_routing(active)
        except Exception:
            pass

        # Optional: communication read step once per active organelle
        comms_enabled = getattr(self.config.comms, "enabled", False)
        post_cost = float(getattr(self.config.comms, "post_cost", 0.2))
        read_cost = float(getattr(self.config.comms, "read_cost", 0.1))
        credit_frac = float(getattr(self.config.comms, "credit_frac", 0.2))
        if comms_enabled:
            for organelle_id in active:
                # attempt to read messages and pay read cost per read; scaled by trait
                read_attempts = 2
                genome = self.population.population.get(organelle_id)
                if genome is not None and isinstance(getattr(genome, "read_rate", 0.0), float):
                    read_attempts = max(0, min(2, int(round(2 * max(0.0, min(1.0, genome.read_rate))))))
                # colony gating for reads
                max_reads = read_attempts
                member_colony = None
                if self.colonies:
                    for cid, meta in self.colonies.items():
                        if organelle_id in meta.get("members", []):
                            member_colony = (cid, meta)
                            break
                if member_colony is not None:
                    _cid, _meta = member_colony
                    max_reads = min(max_reads, int(_meta.get("reads_left", read_attempts)))
                messages = self.environment.read_messages(max_items=max(0, max_reads))
                for msg in messages:
                    # charge reader if enough energy
                    bal = self.host.ledger.energy_balance(organelle_id)
                    colony_ok = True
                    if member_colony is not None:
                        _cid, _meta = member_colony
                        colony_ok = float(_meta.get("bandwidth_left", 0.0)) >= read_cost and int(_meta.get("reads_left", 1)) > 0
                    if bal >= read_cost and colony_ok:
                        try:
                            self.host.ledger.consume_energy(organelle_id, read_cost)
                        except Exception:
                            pass
                        if member_colony is not None:
                            _cid, _meta = member_colony
                            _meta["bandwidth_left"] = max(0.0, float(_meta.get("bandwidth_left", 0.0)) - read_cost)
                            _meta["reads_left"] = max(0, int(_meta.get("reads_left", 1)) - 1)
                        # credit poster
                        poster = msg.get("organelle_id")
                        if isinstance(poster, str):
                            try:
                                self.host.ledger.credit_energy(poster, credit_frac * read_cost)
                            except Exception:
                                pass
                        # small gate bias nudge for reader
                        genome = self.population.population.get(organelle_id)
                        if genome is not None:
                            genome.gate_bias += 0.01
                        # stash hint for next prompt
                        hint_text = str(msg.get("text", "")).strip()
                        if hint_text:
                            bucket = self._pending_hints.setdefault(organelle_id, [])
                            if len(bucket) < 3:
                                bucket.append(hint_text)

        # endogenous batch size option
        bs = self._compute_batch_size(batch_size)
        base_lp_mix = float(getattr(self.config.curriculum, "lp_mix", 0.0))
        self._base_lp_mix = base_lp_mix
        lp_mix_value = self._resolve_lp_mix(base_lp_mix)
        team_enabled = bool(getattr(self.config.assimilation_tuning, "team_router_enabled", False))
        team_max = int(getattr(self.config.assimilation_tuning, "team_max_routes_per_gen", 0))
        team_used = 0
        used_pairs: set[tuple[str, str]] = set()
        # Evidence boost budget
        boost_cap = int(getattr(self.config.assimilation_tuning, "evidence_boost_cap", 0))
        boost_left = boost_cap
        for organelle_id in active:
            # policy-driven per-org batch multiplier
            per_org_bs = bs
            pol = self._active_policies.get(organelle_id, {})
            if isinstance(pol.get("budget_frac"), (int, float)):
                frac = max(0.25, min(2.0, float(pol["budget_frac"])) )
                per_org_bs = max(1, int(round(bs * frac)))
            # Evidence boost for near-threshold candidates
            try:
                if boost_left > 0 and self._should_boost(organelle_id):
                    inc = int(getattr(self.config.assimilation_tuning, "evidence_boost_factor", 1))
                    per_org_bs += inc
                    boost_left = max(0, boost_left - inc)
            except Exception:
                pass
            for _ in range(per_org_bs):
                task = (self._sample_task_with_policy(lp_mix_value, organelle_id) if lp_mix_value > 0.0 else self.environment.sample_task())
                # Optional team routing: pick a synergy pair and run team episode
                if team_enabled and team_used < team_max and len(self.population.population) >= 2:
                    pair = self._select_synergy_pair(fallback_organelle=organelle_id, excluded=used_pairs)
                    if pair is not None and pair not in used_pairs:
                        if self._run_team_episode(pair, task):
                            team_used += 1
                            used_pairs.add(pair)
                            continue
                # apply any pending hints to the prompt
                prompt_text = task.prompt
                hints = self._pending_hints.get(organelle_id, [])
                if hints:
                    joined = "; ".join(hints)
                    prompt_text = f"Hints: {joined}\n\n{task.prompt}"
                    # clear after use
                    self._pending_hints[organelle_id] = []
                result = self.host.step(
                    prompt=prompt_text,
                    intent="solve task",
                    max_routes=1,
                    allowed_organelle_ids=[organelle_id],
                )
                metrics = result.responses.get(organelle_id)
                if metrics is None:
                    continue
                success, reward = task.evaluate(metrics.answer)
                responses = {organelle_id: (metrics.answer, float(metrics.tokens))}
                human_feedback = self._collect_human_feedback(task.prompt, responses)
                if human_feedback:
                    blended = self._blend_rewards({organelle_id: reward}, human_feedback)
                    reward = blended.get(organelle_id, reward)
                self.environment.register_result(organelle_id, task, success)
                settlement = self._settle_episode(organelle_id, task, reward, metrics)
                reward = reward.model_copy(update={"cost_penalty": settlement["cost"]})
                task_cell_family, task_cell_depth = task.cell
                meta = {
                    "family": task.family,
                    "depth": task.depth,
                    "task_id": getattr(task, "task_id", None),
                    "cell_family": task_cell_family,
                    "cell_depth": task_cell_depth,
                    "price": float(getattr(task, "price", 0.0)),
                    "difficulty": float(getattr(task, "difficulty", 0.0)),
                    "success": bool(success),
                    "generation": self.generation_index,
                }
                self.population.record_score(organelle_id, reward.total, meta=meta)
                self.population.record_energy(organelle_id, settlement["cost"])
                self.population.record_roi(organelle_id, settlement["roi"])
                self.population.record_adapter_usage(organelle_id, metrics.active_adapters, metrics.tokens)
                utilisation_snapshot = {
                    module: self.population.average_adapter_usage(organelle_id, module)
                    for module in metrics.active_adapters
                    if module not in {"rank", "total"}
                }
                self._record_episode(
                    task, organelle_id, reward, metrics, settlement, success, utilisation_snapshot
                )
                self.host.apply_reward(result.envelope, {organelle_id: reward})

        self._enforce_diversity()
        # Optional: communication post step (one hint per generation by top ROI organelle)
        if comms_enabled and active:
            try:
                top_org = max(active, key=lambda oid: self.population.average_roi(oid, limit=5))
                genome = self.population.population.get(top_org)
                post_ok = True
                if genome is not None and isinstance(getattr(genome, "post_rate", 0.0), float):
                    post_ok = max(0.0, min(1.0, genome.post_rate)) > 0.0
                # Only post if enough energy and trait allows
                colony_ok = True
                member_colony = None
                if self.colonies:
                    for cid, meta in self.colonies.items():
                        if top_org in meta.get("members", []):
                            member_colony = (cid, meta)
                            break
                if member_colony is not None:
                    _cid, _meta = member_colony
                    colony_ok = float(_meta.get("bandwidth_left", 0.0)) >= post_cost and int(_meta.get("posts_left", 1)) > 0
                if post_ok and colony_ok and self.host.ledger.energy_balance(top_org) >= post_cost:
                    self.host.ledger.consume_energy(top_org, post_cost)
                    hint = "Hint: count words ignoring punctuation and double spaces."
                    self.environment.post_message(top_org, hint, cost=post_cost, ttl=int(getattr(self.config.comms, "ttl", 10)))
                    if member_colony is not None:
                        _cid, _meta = member_colony
                        _meta["bandwidth_left"] = max(0.0, float(_meta.get("bandwidth_left", 0.0)) - post_cost)
                        _meta["posts_left"] = max(0, int(_meta.get("posts_left", 1)) - 1)
            except Exception:
                pass
        merges = self._attempt_assimilation(capped=self.config.evolution.max_merges_per_gen)
        # review trial offspring for potential promotion or cull
        self._review_trial_offspring()
        # Optional: team probes to promote colonies via CI gate
        try:
            team_promotions = self._maybe_team_probes()
        except Exception:
            team_promotions = 0
        # Colonies: promotion and upkeep (pooled pot, reserve top-ups)
        try:
            if bool(getattr(self.config.assimilation_tuning, "colonies_enabled", False)):
                self._maybe_promote_colonies()
                self._tick_colonies()
        except Exception:
            pass
        if merges > 0:
            self.no_merge_counter = 0
        else:
            self.no_merge_counter += 1
        viability_map = self._compute_viability_map()
        survivors = self._mu_lambda_selection(viability_map)
        self._apply_morphogenesis(survivors)
        self._spawn_offspring(survivors)
        self.population.increment_ages()
        summary = {
            "active": len(active),
            "bankrupt": len(bankrupt),
            "culled_bankrupt": len(culled_bankrupt),
            "merges": merges + self.promotions_this_gen,
            "team_promotions": int(team_promotions),
            "population": len(self.population.population),
            "team_routes": int(team_used),
            "avg_roi": self.population.aggregate_roi(),
            "avg_energy_cost": self.population.aggregate_energy(),
            "cells": {
                f"{family}:{depth}": {
                    "difficulty": state.difficulty,
                    "success_ema": state.success_ema,
                    "price": state.price,
                }
                for (family, depth), state in self.environment.controller.cells.items()
            },
        }
        # Promotion acceptance governor (adjust team thresholds toward target)
        try:
            target = float(getattr(self.config.assimilation_tuning, "promotion_target_rate", 0.0))
            step = float(getattr(self.config.assimilation_tuning, "promotion_adjust_step", 0.0))
            if step > 0.0 and target > 0.0:
                observed = float(team_promotions)
                # If 0..N per gen, use raw count; normalize against 1.0 (aim ~target per gen)
                err = target - min(observed, 1.0)
                # Adjust team_holdout_margin and team_min_power gently
                margin = float(getattr(self.config.assimilation_tuning, "team_holdout_margin", getattr(self.config.assimilation_tuning, "holdout_margin", 0.02)))
                power = float(getattr(self.config.assimilation_tuning, "team_min_power", 0.2))
                margin_min = float(getattr(self.config.assimilation_tuning, "team_margin_min", 0.0))
                margin_max = float(getattr(self.config.assimilation_tuning, "team_margin_max", 0.1))
                power_min = float(getattr(self.config.assimilation_tuning, "team_power_min", 0.05))
                power_max = float(getattr(self.config.assimilation_tuning, "team_power_max", 0.5))
                # If under target, reduce margin and power; if over, increase
                margin = max(margin_min, min(margin_max, margin - step * (1.0 if err > 0 else -1.0)))
                power = max(power_min, min(power_max, power - step * (1.0 if err > 0 else -1.0)))
                # Write back
                setattr(self.config.assimilation_tuning, "team_holdout_margin", margin)
                setattr(self.config.assimilation_tuning, "team_min_power", power)
                summary["promotion_controller"] = {"team_holdout_margin": round(margin, 4), "team_min_power": round(power, 4), "target": target, "observed": observed}
        except Exception:
            pass
        # Top co-routing pairs this gen (best-effort)
        try:
            if self._co_routing_counts:
                top_pairs = sorted(self._co_routing_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
                summary["co_routing_top"] = {f"{a}:{b}": int(c) for (a, b), c in top_pairs}
                self._co_routing_counts = {}
        except Exception:
            pass
        # Learning progress snapshot per cell (for analyzer LP heatmaps)
        try:
            lp_map = {
                f"{family}:{depth}": float(self.environment.controller.lp_progress.get((family, depth), 0.0))
                for (family, depth) in self.environment.controller.cells.keys()
            }
            summary["lp_progress"] = lp_map
        except Exception:
            pass
        if bool(getattr(self.config.policy, "enabled", False)):
            summary["policy_applied"] = int(len(self._active_policies) > 0)
            if self._active_policies:
                # Aggregate simple field usage and a couple of numeric knobs
                field_counts: dict[str, int] = {}
                budgets: list[float] = []
                reserves: list[float] = []
                for pol in self._active_policies.values():
                    if isinstance(pol, dict):
                        for k in pol.keys():
                            field_counts[k] = field_counts.get(k, 0) + 1
                        bf = pol.get("budget_frac")
                        rr = pol.get("reserve_ratio")
                        if isinstance(bf, (int, float)):
                            budgets.append(float(bf))
                        if isinstance(rr, (int, float)):
                            reserves.append(float(rr))
                if field_counts:
                    summary["policy_fields_used"] = field_counts
                if budgets:
                    summary["policy_budget_frac_avg"] = float(sum(budgets) / max(1, len(budgets)))
                if reserves:
                    summary["policy_reserve_ratio_avg"] = float(sum(reserves) / max(1, len(reserves)))
                if getattr(self, "_policy_cost_total", 0.0) > 0.0:
                    summary["policy_cost_total"] = float(self._policy_cost_total)
            summary["policy_attempts"] = int(self._policy_attempts_gen)
            summary["policy_parsed"] = int(self._policy_parsed_gen)
        # QD coverage (family-depth cells with observed stats)
        try:
            if getattr(self.config.qd, "enabled", False):
                total_bins = len(self.environment.controller.cells)
                populated = 0
                seen = set()
                for stats in self.environment.organism_stats.values():
                    for cell in stats.keys():
                        seen.add(cell)
                populated = len(seen)
                summary["qd_coverage"] = f"{populated}/{total_bins}"
        except Exception:
            pass
        if hasattr(self, "assim_gating_counts"):
            summary["assimilation_gating"] = getattr(self, "assim_gating_counts")
        samples = getattr(self, "assim_gating_samples_snapshot", None)
        if samples:
            summary["assimilation_gating_samples"] = samples
        attempt_samples = getattr(self, "assim_attempt_samples_snapshot", None)
        if attempt_samples:
            summary["assimilation_attempts"] = attempt_samples
        if self.population.assimilation_history:
            history_snapshot: dict[str, dict[str, object]] = {}
            for (organelle_id, family, depth), records in self.population.assimilation_history.items():
                if not records:
                    continue
                history_snapshot[f"{organelle_id}:{family}:{depth}"] = records[-1]
            summary["assimilation_history"] = history_snapshot
        summary["roi_by_organelle"] = {
            organelle_id: float(self.population.average_roi(organelle_id, limit=5))
            for organelle_id in self.population.population
        }
        summary["lp_mix_base"] = float(self._base_lp_mix)
        summary["lp_mix_active"] = float(self._last_lp_mix)
        if self.colonies:
            summary["colonies_meta"] = {
                cid: {
                    "members": list(meta.get("members", [])),
                    "pot": float(meta.get("pot", 0.0)),
                    "holdout_passes": int(meta.get("holdout_passes", 0)),
                    "holdout_failures": int(meta.get("holdout_failures", 0)),
                    "last_delta": float(meta.get("last_delta", 0.0)),
                }
                for cid, meta in self.colonies.items()
            }
        # Learning KPI: ROI volatility (rolling std across organelles)
        try:
            roi_vals = list(summary["roi_by_organelle"].values())
            summary["roi_volatility"] = float(pstdev(roi_vals)) if len(roi_vals) >= 2 else 0.0
        except Exception:
            summary["roi_volatility"] = 0.0
        summary["energy_balance"] = {
            organelle_id: float(self.host.ledger.energy_balance(organelle_id))
            for organelle_id in self.population.population
        }
        summary["mean_energy_balance"] = (
            float(
                sum(summary["energy_balance"].values()) / max(len(summary["energy_balance"]), 1)
            )
            if summary["energy_balance"]
            else 0.0
        )
        if self._diversity_snapshot is not None:
            summary["diversity"] = dict(self._diversity_snapshot)
        summary["assimilation_fail_streak"] = self.assim_fail_streak
        summary["trial_offspring_active"] = len(self.trial_offspring)
        summary["trials_created"] = int(getattr(self, "trial_creations_this_gen", 0))
        summary["promotions"] = int(getattr(self, "promotions_this_gen", 0))
        # Update QD archive and expose size
        try:
            summary["qd_archive_size"] = int(self._update_qd_archive())
        except Exception:
            summary["qd_archive_size"] = 0
        self._auto_tune_assimilation_energy(summary)
        # Auto‑nudge evidence settings when assimilation stalls (no merges, low power)
        try:
            self._auto_nudge_evidence(summary)
        except Exception:
            pass
        # Ephemeral team synergy sampling (log only)
        try:
            self._sample_team_synergy()
            if self._synergy_window:
                summary["synergy_samples"] = list(self._synergy_window[-5:])
        except Exception:
            pass
        # Colonies: ensure count is present even if disabled
        summary["colonies"] = int(len(self.colonies))
        summary["assimilation_attempt_total"] = int(getattr(self, "_assim_attempt_total", 0))
        if self._tau_relief:
            relief_snapshot = {
                f"{cell[0]}:{cell[1]}": round(float(relief), 3)
                for cell, relief in list(self._tau_relief.items())[:8]
                if relief > 0.0
            }
            if relief_snapshot:
                summary["tau_relief_active"] = relief_snapshot
        if self._roi_relief:
            roi_snapshot = {
                organelle: round(float(relief), 3)
                for organelle, relief in list(self._roi_relief.items())[:8]
                if relief > 0.0
            }
            if roi_snapshot:
                summary["roi_relief_active"] = roi_snapshot
        if self.evaluation_manager and self.generation_index % self.evaluation_manager.config.cadence == 0:
            evaluation_result = self.evaluation_manager.evaluate(self.host, self.environment)
            summary["evaluation"] = evaluation_result
            summary["evaluation"]["reward_weight"] = self.evaluation_manager.config.reward_weight
        if self.config.meta.enabled:
            if self.meta_evolver is None:
                self.meta_evolver = MetaEvolver(
                    config=self.config,
                    environment=self.environment,
                    assimilation=self.assimilation,
                )
            meta_info = self.meta_evolver.step(self.generation_index, summary["avg_roi"])
            summary.update(meta_info)
        return summary

    def _compute_batch_size(self, default_bs: int) -> int:
        bs = default_bs
        try:
            if bool(getattr(self.config.environment, "auto_batch", False)):
                roi_sample = self.population.aggregate_roi(limit=5)
                bmin = int(max(1, getattr(self.config.environment, "batch_min", 1)))
                bmax = int(max(bmin, getattr(self.config.environment, "batch_max", 4)))
                if roi_sample <= 0.5:
                    bs = bmin
                elif roi_sample >= 1.5:
                    bs = bmax
                else:
                    frac = (roi_sample - 0.5) / 1.0
                    bs = int(max(bmin, min(bmax, round(bmin + frac * (bmax - bmin)))))
        except Exception:
            bs = default_bs
        return bs

    def _resolve_lp_mix(self, base_mix: float) -> float:
        cfg = self.config.curriculum
        lp_min = float(getattr(cfg, "lp_mix_min", base_mix))
        lp_max = float(getattr(cfg, "lp_mix_max", max(lp_min, base_mix)))
        lp_min = max(0.0, min(1.0, lp_min))
        lp_max = max(lp_min, min(1.0, lp_max))
        mix = max(lp_min, min(lp_max, base_mix))
        if bool(getattr(cfg, "alp_auto_mix", False)):
            progress_values = list(self.environment.controller.lp_progress.values())
            if progress_values:
                spread = max(progress_values) - min(progress_values)
                denom = max(max(progress_values), 1e-6)
                ratio = max(0.0, min(1.0, spread / denom))
                mix = lp_min + (lp_max - lp_min) * ratio
        window = max(1, int(getattr(cfg, "lp_window", 5)))
        self._lp_mix_history.append(mix)
        if len(self._lp_mix_history) > window:
            self._lp_mix_history = self._lp_mix_history[-window:]
        smoothed = sum(self._lp_mix_history) / len(self._lp_mix_history)
        self._last_lp_mix = max(lp_min, min(lp_max, smoothed))
        return self._last_lp_mix

    def _settle_episode(
        self,
        organelle_id: str,
        task: GridTask,
        reward: RewardBreakdown,
        metrics: RouteMetrics,
    ) -> Dict[str, float]:
        energy_before = self.host.ledger.energy_balance(organelle_id)
        price = task.price
        config = self.config.energy
        revenue = price * reward.total
        cost = (
            config.alpha * metrics.flops_estimate
            + config.beta * metrics.memory_gb
            + config.gamma * metrics.latency_ms
            + config.lambda_p * metrics.trainable_params
        )
        cost *= max(0.0, min(self.config.energy.cost_scale, 1.0))
        roi = revenue / max(cost, 1e-6) if cost > 0 else (float("inf") if revenue > 0 else 0.0)
        if not math.isfinite(roi):
            roi = 0.0
        else:
            roi = max(-10.0, min(roi, 10.0))
        delta = revenue - cost
        energy_after = max(0.0, min(self.host.ledger.energy_cap, energy_before + delta))
        self.host.ledger.set_energy(organelle_id, energy_after)
        self.population.record_energy_delta(organelle_id, delta)
        return {
            "energy_before": energy_before,
            "energy_after": energy_after,
            "revenue": revenue,
            "cost": cost,
            "roi": roi,
            "delta": delta,
        }

    def _record_episode(
        self,
        task: GridTask,
        organelle_id: str,
        reward: RewardBreakdown,
        metrics: RouteMetrics,
        settlement: Dict[str, float],
        success: bool,
        adapter_utilisation: dict[str, float],
    ) -> None:
        episode = EpisodeLog(
            episode_id=f"epi_{len(self.logs)}",
            task_id=task.task_id,
            organelles=[organelle_id],
            rewards=reward,
            energy_spent=settlement["cost"],
            observations={
                "prompt": task.prompt,
                "answer": metrics.answer,
                "cell": {
                    "family": task.family,
                    "depth": task.depth,
                },
                "price": task.price,
                "success": success,
                "energy_before": settlement["energy_before"],
                "energy_after": settlement["energy_after"],
                "roi": settlement["roi"],
                "metrics": {
                    "tokens": metrics.tokens,
                    "latency_ms": metrics.latency_ms,
                    "flops_estimate": metrics.flops_estimate,
                    "memory_gb": metrics.memory_gb,
                    "trainable_params": metrics.trainable_params,
                    "active_adapters": metrics.active_adapters,
                    "adapter_utilisation": adapter_utilisation,
                },
            },
        )
        self.logs.append(episode)
        # Keep a bounded in-memory cache to avoid RAM growth on long runs
        try:
            limit = int(getattr(self.config.metrics, "in_memory_log_limit", 256))
        except Exception:
            limit = 256
        if limit <= 0:
            self.logs = []
        elif len(self.logs) > limit:
            self.logs = self.logs[-limit:]
        if self.sink:
            self.sink.log_episode(episode)

    def _select_synergy_pair(
        self,
        fallback_organelle: str | None = None,
        excluded: set[tuple[str, str]] | None = None,
    ) -> tuple[str, str] | None:
        # Prefer top co-routing pairs; else use top-ROI fallback including the provided organelle.
        # Skips any pairs present in `excluded`.
        excluded = excluded or set()
        try:
            if hasattr(self, "_co_routing_counts") and self._co_routing_counts:
                pairs_sorted = sorted(self._co_routing_counts.items(), key=lambda kv: kv[1], reverse=True)
                for (a, b), _c in pairs_sorted:
                    pair = (min(a, b), max(a, b))
                    if pair in excluded:
                        continue
                    if a in self.population.population and b in self.population.population:
                        return pair
        except Exception:
            pass
        # Fallback by ROI
        try:
            scored = [(oid, float(self.population.average_roi(oid, limit=5))) for oid in self.population.population.keys()]
            scored.sort(key=lambda x: x[1], reverse=True)
            top = [oid for oid, _ in scored[: min(6, len(scored))]]
            if fallback_organelle and fallback_organelle in top and len(top) >= 2:
                # Try each candidate paired with fallback until one is not excluded
                for other in top:
                    if other == fallback_organelle:
                        continue
                    pair = tuple(sorted((fallback_organelle, other)))  # type: ignore[assignment]
                    if pair not in excluded:
                        return pair  # type: ignore[return-value]
            if len(top) >= 2:
                # Return the first non-excluded pair in rank order
                for i in range(len(top)):
                    for j in range(i + 1, len(top)):
                        pair = tuple(sorted((top[i], top[j])))  # type: ignore[assignment]
                        if pair not in excluded:
                            return pair  # type: ignore[return-value]
        except Exception:
            pass
        return None

    def _run_team_episode(self, pair: tuple[str, str], task: GridTask) -> bool:
        # Best-of-two answer (vote) with individual energy settlement
        if len(pair) != 2:
            return False
        a_id, b_id = pair
        results: list[tuple[str, RouteMetrics, bool, RewardBreakdown, dict[str, float]]] = []
        for oid in (a_id, b_id):
            result = self.host.step(
                prompt=task.prompt,
                intent="team episode",
                max_routes=1,
                allowed_organelle_ids=[oid],
            )
            metrics = result.responses.get(oid)
            if metrics is None:
                continue
            success, reward = task.evaluate(metrics.answer)
            settlement = self._settle_episode(oid, task, reward, metrics)
            results.append((oid, metrics, success, reward, settlement))
            # record episode per member
            adapter_utilisation = {k: float(v) for k, v in metrics.active_adapters.items()} if isinstance(metrics.active_adapters, dict) else {}
            self._record_episode(task, oid, reward, metrics, settlement, success, adapter_utilisation)
        if not results:
            return False
        # Optional: single handoff (solver→checker)
        try:
            if bool(getattr(self.config.assimilation_tuning, "team_handoff_enabled", False)) and len(results) == 2:
                # pick current winner and let the other revise
                best_pair = max(results, key=lambda tup: float(tup[4].get("roi", 0.0)))
                winner_id = best_pair[0]
                checker_id = b_id if winner_id == a_id else a_id
                winner_answer = str(best_pair[1].answer)
                handoff_prompt = f"Review and improve this answer:\n{winner_answer}\n\nTask:\n{task.prompt}"
                result_rev = self.host.step(prompt=handoff_prompt, intent="team handoff", max_routes=1, allowed_organelle_ids=[checker_id])
                m_rev = result_rev.responses.get(checker_id)
                if m_rev is not None:
                    success_rev, reward_rev = task.evaluate(m_rev.answer)
                    st_rev = self._settle_episode(checker_id, task, reward_rev, m_rev)
                    # Do not record a separate episode for the revision to keep accounting simple
                    results.append((checker_id, m_rev, success_rev, reward_rev, st_rev))
        except Exception:
            pass
        # Choose winner by highest ROI
        best = None
        best_roi = float("-inf")
        for oid, metrics, success, reward, settlement in results:
            roi = float(settlement.get("roi", 0.0))
            if roi > best_roi:
                best_roi = roi
                best = (oid, metrics, success, reward, settlement)
        # post a small hint if the winner succeeded
        if best is not None and bool(best[2]):
            try:
                hint = "Team tip: cross-check units and counts."
                self.environment.post_message(best[0], hint, cost=float(getattr(self.config.comms, "post_cost", 0.2)), ttl=int(getattr(self.config.comms, "ttl", 10)))
            except Exception:
                pass
        return True

    def _sample_task_lp(self, lp_mix: float) -> GridTask:
        try:
            cell = self.environment.controller.sample_cell(lp_mix=lp_mix)
            state = self.environment.controller.get_state(cell)
            use_canary = state.success_ema > self.environment.canary_q_min and self.environment.rng.random() < 0.1
            return self.environment.sample_task_from_cell(cell, canary=use_canary)
        except Exception:
            return self.environment.sample_task()

    def _sample_task_with_policy(self, lp_mix: float, organelle_id: str) -> GridTask:
        pol = self._active_policies.get(organelle_id)
        if pol and isinstance(pol.get("cell_pref"), dict):
            try:
                fam = str(pol["cell_pref"].get("family"))
                dep = str(pol["cell_pref"].get("depth"))
                bias = float(getattr(self.config.policy, "bias_strength", 0.3))
                if 0.0 < bias <= 1.0 and self.environment.rng.random() < bias:
                    return self.environment.sample_task_from_cell((fam, dep))
            except Exception:
                pass
        return self._sample_task_lp(lp_mix)

    @staticmethod
    def _parse_policy_json(text: str, allowed: list[str]) -> dict[str, object]:
        """Extract and repair a small JSON object with allowed fields.

        Strategy (best-effort, no heavy deps):
        1) Prefer fenced ```json blocks; else any fenced block; else the outermost {...} span.
        2) Try strict json.loads; on failure, apply light repairs:
           - strip code fences
           - remove trailing commas before } or ]
           - normalize Python literals to JSON (True/False/None -> true/false/null)
           - convert simple single-quoted keys/values to double-quoted
        3) Return only keys in `allowed`.
        """
        import json as _json
        import re as _re

        def _find_fenced(block: str) -> list[str]:
            candidates: list[str] = []
            # ```json ... ``` preferred
            for m in _re.finditer(r"```json\s*([\s\S]*?)\s*```", block, _re.IGNORECASE):
                candidates.append(m.group(1))
            # any ``` ... ```
            for m in _re.finditer(r"```\s*([\s\S]*?)\s*```", block):
                candidates.append(m.group(1))
            return candidates

        def _outer_object(block: str) -> str | None:
            start = block.find("{")
            if start == -1:
                return None
            depth = 0
            for i in range(start, len(block)):
                ch = block[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return block[start : i + 1]
            return None

        def _repair(s: str) -> str:
            # strip code fences accidentally included
            s = s.strip().strip('`')
            # normalize common Python literals to JSON
            s = _re.sub(r"\bTrue\b", "true", s)
            s = _re.sub(r"\bFalse\b", "false", s)
            s = _re.sub(r"\bNone\b", "null", s)
            # remove trailing commas before closing } or ]
            s = _re.sub(r",\s*([}\]])", r"\1", s)
            # single-quoted keys -> double quotes
            s = _re.sub(r"([{,]\s*)'([^'\s]+)'\s*:", r'\1"\2":', s)
            # single-quoted string values -> double quotes (conservative, no nested quotes)
            s = _re.sub(r":\s*'([^']*)'\s*([,}])", r': "\1"\2', s)
            return s

        def _try_load(s: str) -> dict[str, object] | None:
            try:
                data = _json.loads(s)
            except Exception:
                try:
                    data = _json.loads(_repair(s))
                except Exception:
                    return None
            if not isinstance(data, dict):
                return None
            return {k: v for k, v in data.items() if k in allowed}

        # Collect candidates
        candidates = _find_fenced(text)
        outer = _outer_object(text)
        if outer:
            candidates.append(outer)
        for cand in candidates:
            parsed = _try_load(cand)
            if parsed:
                return parsed
        # Fallback: simple key:value extraction (very tolerant)
        try:
            kvs: dict[str, object] = {}
            # match patterns like key: 0.5 or key=0.5 or key: true/false
            for m in _re.finditer(r"([A-Za-z_][A-Za-z0-9_\-]*)\s*[:=]\s*([\-+]?[0-9]+(?:\.[0-9]+)?|true|false)", text, _re.IGNORECASE):
                k = m.group(1)
                vraw = m.group(2)
                if k not in allowed:
                    continue
                if vraw.lower() in ("true", "false"):
                    kvs[k] = (vraw.lower() == "true")
                else:
                    try:
                        kvs[k] = float(vraw) if "." in vraw else int(vraw)
                    except Exception:
                        continue
            if kvs:
                return kvs
        except Exception:
            pass
        return {}

    def run_colony_inference(self, member_ids: list[str], prompt: str, strategy: str = "best_of_two") -> dict[str, object]:
        """Run an ad-hoc inference over a set of members and return the selected answer.

        - strategy="best_of_two" chooses the answer with the larger number of non-whitespace characters
          (heuristic when no ground-truth reward is available).
        - Returns a dict: {"selected_id", "selected_answer", "answers": {id: answer}, "tokens": {id: int}}
        """
        answers: dict[str, str] = {}
        tokens: dict[str, int] = {}
        for oid in member_ids:
            result = self.host.step(prompt=prompt, intent="colony infer", max_routes=1, allowed_organelle_ids=[oid])
            metrics = result.responses.get(oid)
            ans = ""
            tok = 0
            if metrics is not None:
                ans = str(metrics.answer)
                tok = int(getattr(metrics, "tokens", 0) or 0)
            answers[oid] = ans
            tokens[oid] = tok
        selected_id = member_ids[0] if member_ids else ""
        if strategy == "best_of_two" and len(member_ids) >= 2:
            def _score(a: str) -> int:
                return len(str(a).strip())
            selected_id = max(member_ids, key=lambda oid: _score(answers.get(oid, "")))
        selected_answer = answers.get(selected_id, "")
        return {"selected_id": selected_id, "selected_answer": selected_answer, "answers": answers, "tokens": tokens}

    def _probe_co_routing(self, active_ids: list[str]) -> None:
        """Run a few light routing probes to populate co‑routing counts per generation.

        Uses host routing with k=2 to observe which organelles tend to co‑route on sampled tasks.
        Keeps the count in `_co_routing_counts` and avoids energy settlement/logging.
        """
        try:
            per_gen = int(getattr(self.config.assimilation_tuning, "team_routing_probe_per_gen", 0))
        except Exception:
            per_gen = 0
        if per_gen <= 0 or len(active_ids) < 2:
            return
        # Ensure counter exists
        try:
            _ = self._co_routing_counts
        except Exception:
            self._co_routing_counts = {}
        for _ in range(per_gen):
            try:
                task = self.environment.sample_task()
                # Ask router to pick two candidates via normal step (no settlement here)
                result = self.host.step(
                    prompt=task.prompt,
                    intent="routing probe",
                    max_routes=2,
                    allowed_organelle_ids=active_ids,
                )
                # Collect first two unique routed organelles
                picked: list[str] = []
                for evt in result.routes:
                    oid = getattr(evt, "organelle_id", None)
                    if isinstance(oid, str) and oid not in picked:
                        picked.append(oid)
                    if len(picked) >= 2:
                        break
                if len(picked) == 2:
                    a, b = sorted(picked)
                    key = (a, b)
                    self._co_routing_counts[key] = int(self._co_routing_counts.get(key, 0)) + 1
            except Exception:
                # Best‑effort only; continue
                continue

    def _should_boost(self, organelle_id: str) -> bool:
        """Heuristic: boost evidence for near-threshold or promising organelles.

        Uses recent ROI vs aggregate ROI as a proxy. Best-effort and safe.
        """
        try:
            recent = float(self.population.average_roi(organelle_id, limit=5))
        except Exception:
            return False
        try:
            aggregate = float(self.population.aggregate_roi())
        except Exception:
            aggregate = 0.0
        # Boost when recent ROI is close to or above 90% of aggregate
        threshold = max(0.0, 0.9 * aggregate)
        return recent >= threshold

    def _request_and_apply_policy(self, organelle_id: str) -> None:
        # Count attempt
        try:
            self._policy_attempts_gen += 1
        except Exception:
            self._policy_attempts_gen = 1
        allowed = list(getattr(self.config.policy, "allowed_fields", []))
        prompt = (
            "Emit ONLY a single minified JSON object with these keys: "
            + ", ".join(allowed)
            + ". No prose, no code fences, no trailing commas. If unsure, emit {}."
        )
        result = self.host.step(
            prompt=prompt,
            intent="choose policy",
            max_routes=1,
            allowed_organelle_ids=[organelle_id],
        )
        metrics = result.responses.get(organelle_id)
        if metrics is None:
            return
        answer = result.envelope.observation.state.get("answer", "")
        allowed = list(getattr(self.config.policy, "allowed_fields", []))
        pol = self._parse_policy_json(str(answer), allowed)
        if not pol:
            return
        # Count successful parse
        try:
            self._policy_parsed_gen += 1
        except Exception:
            self._policy_parsed_gen = 1
        # Energy charging for policy request (micro-cost)
        try:
            micro = float(getattr(self.config.policy, "energy_cost", 0.0))
        except Exception:
            micro = 0.0
        if micro > 0.0:
            # Optional scaling by token usage
            try:
                if bool(getattr(self.config.policy, "charge_tokens", False)):
                    cap = max(1, int(getattr(self.config.policy, "token_cap", 64)))
                    scale = min(1.0, float(metrics.tokens) / float(cap))
                    micro = micro * max(0.0, min(1.0, scale))
            except Exception:
                pass
            try:
                if self.host.ledger.energy_balance(organelle_id) >= micro:
                    self.host.ledger.consume_energy(organelle_id, micro)
                    self._policy_cost_total += float(micro)
            except Exception:
                pass
        self._active_policies[organelle_id] = pol
        # Apply immediate effects
        # gate_bias_delta
        try:
            if isinstance(pol.get("gate_bias_delta"), (int, float)):
                genome = self.population.population.get(organelle_id)
                if genome is not None:
                    genome.gate_bias += float(pol["gate_bias_delta"]) * 0.1
        except Exception:
            pass
        # comms preferences
        try:
            genome = self.population.population.get(organelle_id)
            if genome is not None:
                if isinstance(pol.get("read"), bool):
                    genome.read_rate = 1.0 if pol["read"] else 0.0
                if isinstance(pol.get("post"), bool):
                    genome.post_rate = 1.0 if pol["post"] else 0.0
        except Exception:
            pass

    def _attempt_assimilation(self, capped: int | None = None) -> int:
        removable: list[str] = []
        merges = 0
        gating: Dict[str, int] = {
            "canary_failed": 0,
            "low_energy": 0,
            "low_power": 0,
            "no_best_cell": 0,
            "cooldown": 0,
            "uplift_below_threshold": 0,
            "cell_merges_exceeded": 0,
            "insufficient_scores": 0,
            "global_probe_failed": 0,
            "holdout_failed": 0,
            "topup_success": 0,
            "topup_roi_blocked": 0,
            "topup_cap_blocked": 0,
            "topup_already_sufficient": 0,
            "topup_disabled": 0,
        }
        merges_per_cell: Dict[GridKey, int] = {}
        per_cell_interval = self.config.assimilation_tuning.per_cell_interval
        max_cell_merges = self.config.assimilation_tuning.max_merges_per_cell
        for genome in list(self.population.population.values()):
            # Policy reserve gate: skip assimilation if reserve active
            pol = self._active_policies.get(genome.organelle_id)
            if pol and isinstance(pol.get("reserve_ratio"), (int, float)):
                rr = float(pol["reserve_ratio"]) if pol["reserve_ratio"] is not None else 0.0
                rr = max(getattr(self.config.policy, "reserve_min", 0.0), min(getattr(self.config.policy, "reserve_max", 0.75), rr))
                reserve = rr * 4.0 * self.config.energy.m
                if self.host.ledger.energy_balance(genome.organelle_id) < reserve:
                    self._record_assimilation_gate(
                        reason="reserve_active",
                        organelle_id=genome.organelle_id,
                        details={"reserve": reserve, "balance": self.host.ledger.energy_balance(genome.organelle_id)},
                    )
                    continue
            self._assim_attempt_total += 1
            if self.environment.canary_failed(genome.organelle_id):
                gating["canary_failed"] += 1
                self._record_assimilation_gate(
                    reason="canary_failed",
                    organelle_id=genome.organelle_id,
                    details={"generation": self.generation_index},
                )
                continue
            balance = self.host.ledger.energy_balance(genome.organelle_id)
            balance, topup_info = self._maybe_top_up_energy(genome, balance)
            status = topup_info.get("status", "unknown")
            if status == "credited":
                gating["topup_success"] += 1
            elif status == "skip_low_roi":
                gating["topup_roi_blocked"] += 1
            elif status == "skip_no_capacity":
                gating["topup_cap_blocked"] += 1
            elif status == "already_sufficient":
                gating["topup_already_sufficient"] += 1
            elif status == "disabled":
                gating["topup_disabled"] += 1
            if balance < self.config.energy.m:
                gating["low_energy"] += 1
                self._record_assimilation_gate(
                    reason="low_energy",
                    organelle_id=genome.organelle_id,
                    details={
                        "generation": self.generation_index,
                        "balance": balance,
                        "ticket": self.config.energy.m,
                        "top_up": topup_info,
                    },
                )
                continue
            best_cell = self.environment.best_cell_score(genome.organelle_id)
            if best_cell is None:
                gating["no_best_cell"] += 1
                self._record_assimilation_gate(
                    reason="no_best_cell",
                    organelle_id=genome.organelle_id,
                    details={"generation": self.generation_index},
                )
                continue
            cell, ema = best_cell
            family, depth = cell
            key = (genome.organelle_id, cell)
            last_attempt = self.assimilation_cooldown.get(key, -per_cell_interval)
            if self.generation_index - last_attempt < per_cell_interval:
                gating["cooldown"] += 1
                self._record_assimilation_gate(
                    reason="cooldown",
                    organelle_id=genome.organelle_id,
                    details={
                        "generation": self.generation_index,
                        "cooldown_remaining": per_cell_interval - (self.generation_index - last_attempt),
                    },
                )
                continue
            tau = self.config.controller.tau
            if family in {"math", "math.sequence"}:
                tau = min(0.6, tau + 0.08)
            elif family in {"logic.bool", "word.count"}:
                tau = min(0.58, tau + 0.06)
            elif family == "json_repair":
                tau = min(0.57, tau + 0.05)
            relief = float(self._tau_relief.get(cell, 0.0))
            if relief > 0.0:
                tau = max(0.2, tau - relief)
            uplift_gate = ema - tau
            if uplift_gate < self.config.evolution.assimilation_threshold:
                self.assimilation_cooldown[key] = self.generation_index
                gating["uplift_below_threshold"] += 1
                self._record_assimilation_gate(
                    reason="uplift_below_threshold",
                    organelle_id=genome.organelle_id,
                    details={
                        "generation": self.generation_index,
                        "ema": ema,
                        "tau": self.config.controller.tau,
                        "uplift_gate": uplift_gate,
                        "threshold": self.config.evolution.assimilation_threshold,
                        "relief": relief,
                    },
                )
                self._register_tau_failure(cell)
                continue
            if merges_per_cell.get(cell, 0) >= max_cell_merges:
                gating["cell_merges_exceeded"] += 1
                self._record_assimilation_gate(
                    reason="cell_merges_exceeded",
                    organelle_id=genome.organelle_id,
                    details={
                        "generation": self.generation_index,
                        "cell": {"family": cell[0], "depth": cell[1]},
                        "max_cell_merges": max_cell_merges,
                    },
                )
                continue
            score_records = self.population.recent_score_records(genome.organelle_id, limit=16)
            if not score_records:
                continue
            scores = [float(record.get("score", 0.0) or 0.0) for record in score_records]
            base_min = max(4, self.config.assimilation_tuning.min_window)
            desired_min = base_min
            if family in {"math", "math.sequence"}:
                desired_min = max(desired_min, 8)
            elif family in {"logic.bool", "word.count"}:
                desired_min = max(desired_min, 6)
            available = len(scores) - (len(scores) % 2)
            if available < base_min:
                min_window = base_min
            else:
                min_window = min(desired_min, available)
            step = max(2, self.config.assimilation_tuning.window_step)
            if family in {"math", "math.sequence"}:
                step = max(step, 4)
            if available < min_window:
                self.assimilation_cooldown[key] = self.generation_index
                gating["insufficient_scores"] += 1
                self._record_assimilation_gate(
                    reason="insufficient_scores",
                    organelle_id=genome.organelle_id,
                    details={
                        "generation": self.generation_index,
                        "scores_available": len(scores),
                        "min_window": min_window,
                    },
                )
                continue
            window_len = available
            while window_len > min_window and window_len - step >= min_window:
                window_len_candidate = window_len - step
                if window_len_candidate % 2 != 0:
                    window_len_candidate -= 1
                if window_len_candidate < min_window:
                    break
                window_len = window_len_candidate
                break
            start_idx = len(scores) - window_len
            window = scores[start_idx:]
            window_records = score_records[start_idx:]
            split = window_len // 2
            control = window[:split]
            treatment = window[split:]
            control_records = window_records[:split]
            treatment_records = window_records[split:]
            if len(control) < 2 or len(treatment) < 2:
                self.assimilation_cooldown[key] = self.generation_index
                gating["insufficient_scores"] += 1
                self._record_assimilation_gate(
                    reason="insufficient_scores_window",
                    organelle_id=genome.organelle_id,
                    details={
                        "generation": self.generation_index,
                        "window_len": window_len,
                    },
                )
                continue
            avg_energy = self.population.average_energy(genome.organelle_id)
            result = self.assimilation.evaluate(
                organelle_id=genome.organelle_id,
                control_scores=control,
                treatment_scores=treatment,
                safety_hits=0,
                energy_cost=avg_energy * len(treatment),
                energy_balance=balance,
                energy_top_up=topup_info,
                control_meta=[dict(record) for record in control_records],
                treatment_meta=[dict(record) for record in treatment_records],
            )
            self.assimilation_cooldown[key] = self.generation_index
            # If statistical power is too low, defer and expand evidence window next time
            try:
                min_power = float(getattr(self.config.assimilation_tuning, "trial_min_power", 0.1))
            except Exception:
                min_power = 0.1
            try:
                dr_min_power = float(getattr(self.config.assimilation_tuning, "dr_min_power", min_power))
                min_power = max(min_power, dr_min_power)
            except Exception:
                pass
            if result.event.power is not None and result.event.power < min_power:
                gating["low_power"] += 1
                self._record_assimilation_gate(
                    reason="low_power",
                    organelle_id=genome.organelle_id,
                    details={
                        "generation": self.generation_index,
                        "power": float(result.event.power),
                        "min_power": float(min_power),
                        "sample_size": int(result.event.sample_size or 0),
                    },
                )
                try:
                    self.population.grant_evidence(genome.organelle_id, 1)
                except Exception:
                    pass
                continue
            # DR small-n refinement: if DR used but effective sample is too small, defer
            try:
                if bool(result.event.dr_used):
                    dr_sizes = result.event.dr_sample_sizes or {}
                    min_stratum = int(getattr(self.config.assimilation_tuning, "dr_min_stratum_size", 2))
                    contributing = 0
                    for _k, s in dr_sizes.items():
                        try:
                            if int(s.get("paired", 0)) >= min_stratum:
                                contributing += 1
                        except Exception:
                            continue
                    sample_n = int(result.event.sample_size or 0)
                    if contributing < 1 or sample_n < max(2 * min_stratum, 6):
                        gating["low_power"] += 1
                        self._record_assimilation_gate(
                            reason="low_power_dr",
                            organelle_id=genome.organelle_id,
                            details={
                                "generation": self.generation_index,
                                "contributing_strata": int(contributing),
                                "min_stratum": int(min_stratum),
                                "sample_size": sample_n,
                            },
                        )
                        try:
                            self.population.grant_evidence(genome.organelle_id, 1)
                        except Exception:
                            pass
                        continue
            except Exception:
                pass
            probe_records: list[dict[str, object]] = []
            soup_records: list[dict[str, float]] = []
            holdout_info: dict[str, object] | None = None
            audit_info: dict[str, object] | None = None
            decision_final = False
            attempt_detail: dict[str, object] = {
                "generation": self.generation_index,
                "organelle_id": genome.organelle_id,
                "cell": {"family": cell[0], "depth": cell[1]},
                "uplift": float(result.event.uplift),
                "p_value": float(result.event.p_value),
                "sample_size": result.event.sample_size,
                "threshold": self.config.evolution.assimilation_threshold,
                "uplift_threshold": result.event.uplift_threshold,
                "p_value_threshold": result.event.p_value_threshold,
                "control_mean": result.event.control_mean,
                "treatment_mean": result.event.treatment_mean,
                "control_std": result.event.control_std,
                "treatment_std": result.event.treatment_std,
                "energy_balance": balance,
                "top_up": dict(topup_info),
                "tau_relief": float(relief),
                "roi_relief": float(self._roi_relief.get(genome.organelle_id, 0.0)),
                "method": result.event.method,
                "dr_used": bool(result.event.dr_used),
                "strata": dict(result.event.dr_sample_sizes),
                "passes_stat_test": bool(result.decision),
            }
            if not result.decision:
                self.assim_fail_streak += 1
                self._maybe_decay_assimilation_thresholds()
                self._register_tau_failure(cell)
            if result.decision and (capped is None or merges < capped):
                soup_ids, stats_map = self._select_soup_members(cell, genome.organelle_id)
                holdout_ok, holdout_info = self._holdout_accepts(
                    genome.organelle_id,
                    [oid for oid in soup_ids if oid != genome.organelle_id],
                )
                attempt_detail["holdout"] = holdout_info or {}
                attempt_detail["holdout_passed"] = bool(holdout_ok)
                if not holdout_ok:
                    gating["holdout_failed"] += 1
                    self.assim_fail_streak += 1
                    self._maybe_decay_assimilation_thresholds()
                    self._record_assimilation_gate(
                        reason="holdout_failed",
                        organelle_id=genome.organelle_id,
                        details={
                            "generation": self.generation_index,
                            "holdout": holdout_info or {},
                        },
                    )
                    self._register_tau_failure(cell)
                elif self._global_probe(genome.organelle_id, cell, gating):
                    probe_records = self._run_hf_probes(genome.organelle_id)
                    soup_records = self._apply_lora_soup_merge(
                        cell,
                        genome.organelle_id,
                        soup_ids,
                        stats_map,
                        probe_records,
                    )
                    audit_info = None
                    if getattr(self.config.assimilation_tuning, "merge_audit_enabled", False):
                        try:
                            tasks = self._sample_holdout_tasks()
                            post_roi = self._evaluate_holdout_roi(genome.organelle_id, tasks)
                            pre_roi = None
                            if holdout_info:
                                pre_roi = holdout_info.get("candidate_roi")
                            audit_info = {"post_roi": float(post_roi), "pre_roi": float(pre_roi) if pre_roi is not None else None, "tasks": len(tasks)}
                        except Exception:
                            audit_info = {"post_roi": None, "pre_roi": None, "tasks": 0}
                    self._spawn_replacement_from(genome)
                    removable.append(genome.organelle_id)
                    merges += 1
                    merges_per_cell[cell] = merges_per_cell.get(cell, 0) + 1
                    decision_final = True
                    self.assim_fail_streak = 0
                    self._register_tau_success(cell)
                    attempt_detail["global_probe_passed"] = True
                else:
                    attempt_detail["global_probe_passed"] = False
                    self._register_tau_failure(cell)
            else:
                attempt_detail["holdout_passed"] = False
                attempt_detail["global_probe_passed"] = False
                # Offspring merge path: allow energy-eligible trial child
                mode = getattr(self.config.assimilation_tuning, "merge_mode", "strict")
                enable_trials = bool(getattr(self.config.assimilation_tuning, "trial_offspring_enabled", False))
                cap = int(getattr(self.config.assimilation_tuning, "trial_per_gen_cap", 0))
                if enable_trials and mode in {"offspring", "hybrid"} and self.trial_creations_this_gen < cap:
                    soup_ids, stats_map = self._select_soup_members(cell, genome.organelle_id)
                    child_id = self._create_trial_offspring(cell, genome.organelle_id, soup_ids, stats_map)
                    if child_id:
                        self.trial_creations_this_gen += 1
                        attempt_detail["trial_offspring_created"] = child_id
                # Router-team acceptance gate: try a routed pair as a colony candidate
                try:
                    soup_ids, _stats_map = self._select_soup_members(cell, genome.organelle_id)
                    partner = None
                    for oid in soup_ids:
                        if oid != genome.organelle_id:
                            partner = oid
                            break
                    if partner is not None:
                        tasks = self._sample_holdout_tasks()
                        if tasks:
                            # Collect per-task ROI series for team (best-of-two) and bases
                            energy_cfg = self.config.energy
                            team_series: list[float] = []
                            a_series: list[float] = []
                            b_series: list[float] = []
                            for index, task in enumerate(tasks, start=1):
                                grid_task = task.to_grid_task(self.environment, task_id=f"team_{index:04d}")
                                best_roi = 0.0
                                for oid in (genome.organelle_id, partner):
                                    result_i = self.host.step(
                                        prompt=grid_task.prompt,
                                        intent="team holdout",
                                        max_routes=1,
                                        allowed_organelle_ids=[oid],
                                    )
                                    metrics_i = result_i.responses.get(oid)
                                    if metrics_i is None:
                                        continue
                                    success_i, reward_i = grid_task.evaluate(metrics_i.answer)
                                    revenue_i = grid_task.price * reward_i.total
                                    cost_i = (
                                        energy_cfg.alpha * metrics_i.flops_estimate
                                        + energy_cfg.beta * metrics_i.memory_gb
                                        + energy_cfg.gamma * metrics_i.latency_ms
                                        + energy_cfg.lambda_p * metrics_i.trainable_params
                                    )
                                    roi_i = (float("inf") if revenue_i > 0 else 0.0) if cost_i <= 0.0 else (revenue_i / cost_i)
                                    roi_i = 0.0 if not math.isfinite(roi_i) else float(max(0.0, min(roi_i, 10.0)))
                                    if oid == genome.organelle_id:
                                        a_series.append(roi_i)
                                    else:
                                        b_series.append(roi_i)
                                    best_roi = max(best_roi, roi_i)
                                team_series.append(best_roi)
                            # Compute acceptance using CI on team mean vs baseline mean
                            base_a = sum(a_series) / max(len(a_series), 1)
                            base_b = sum(b_series) / max(len(b_series), 1)
                            baseline = max(base_a, base_b)
                            ci_low, ci_high, team_mu, team_se = self._compute_mean_ci(team_series)
                            margin = float(self.config.assimilation_tuning.holdout_margin)
                            min_power_tasks = 8
                            accept = len(team_series) >= min_power_tasks and (ci_low > baseline + margin)
                            if accept:
                                cid = f"col_{genome.organelle_id[:4]}_{partner[:4]}"
                                self.colonies.setdefault(cid, {"members": [genome.organelle_id, partner], "pot": 0.0, "reserve_ratio": 0.25, "created_gen": self.generation_index})
                                self.promotions_this_gen += 1
                                attempt_detail["team_promoted_colony"] = cid
                            # log team acceptance stats
                            attempt_detail["team_holdout"] = {
                                "team_roi": float(team_mu),
                                "base_a": float(base_a),
                                "base_b": float(base_b),
                                "delta": float(team_mu - baseline),
                                "ci_low": float(ci_low),
                                "ci_high": float(ci_high),
                                "tasks": int(len(team_series)),
                                "min_tasks": int(min_power_tasks),
                            }
                except Exception:
                    pass
            self._record_assimilation_attempt(attempt_detail)
            if self.sink:
                event = result.event.model_copy(
                    update={
                        "cell": {"family": cell[0], "depth": cell[1]},
                        "probes": probe_records,
                        "soup": soup_records,
                        "holdout": holdout_info,
                        "audit": audit_info,
                    }
                )
                self.sink.log_assimilation(event, decision_final)
            self.population.record_assimilation(
                genome.organelle_id,
                cell,
                {
                    "generation": self.generation_index,
                    "uplift": float(result.event.uplift),
                    "p_value": float(result.event.p_value),
                    "passed": bool(decision_final),
                    "ema": float(self.environment.organism_stats.get(genome.organelle_id, {}).get(cell, 0.0)),
                    "roi": float(self.population.average_roi(genome.organelle_id)),
                    "probes": probe_records,
                    "holdout": holdout_info,
                    "sample_size": result.event.sample_size,
                    "control_mean": result.event.control_mean,
                    "control_std": result.event.control_std,
                    "treatment_mean": result.event.treatment_mean,
                    "treatment_std": result.event.treatment_std,
                    "energy_balance": result.event.energy_balance,
                    "energy_top_up": (
                        result.event.energy_top_up.model_dump(mode="json")
                        if result.event.energy_top_up is not None
                        else None
                    ),
                },
            )
        for organelle_id in removable:
            self.host.retire_organelle(organelle_id)
            self.population.remove(organelle_id)
        # expose gating counts for diagnostics in generation summary
        try:
            self.assim_gating_counts = gating  # type: ignore[attr-defined]
            self.assim_gating_samples_snapshot = list(self.assim_gating_samples[-24:])  # type: ignore[attr-defined]
            self.assim_attempt_samples_snapshot = list(self.assim_attempt_samples[-24:])  # type: ignore[attr-defined]
        except Exception:
            pass
        return merges

    def _create_trial_offspring(
        self,
        cell: GridKey,
        candidate_id: str,
        soup_ids: list[str],
        stats_map: dict[str, dict[str, float]],
    ) -> str | None:
        try:
            organelle = self.host.get_organelle(candidate_id)
            target_rank = int(getattr(organelle, "rank", self.config.host.max_lora_rank)) if organelle is not None else self.config.host.max_lora_rank
            target_rank = max(1, min(target_rank, self.config.host.max_lora_rank))
            # weights based on roi*ema, same as merge
            weights: list[float] = []
            for oid in soup_ids:
                stats = stats_map.get(oid, {"roi": 0.0, "ema": 0.0})
                w = (max(stats.get("roi", 0.0), 0.0) + 1e-3) * (stats.get("ema", 0.0) + 1e-3)
                weights.append(w)
            total = sum(weights) or 1.0
            soup = {oid: w / total for oid, w in zip(soup_ids, weights)}
            soup_state, _alpha = self.host.build_lora_soup_state(soup, target_rank)
            child_id = self.host.spawn_organelle(rank=target_rank)
            child = self.host.get_organelle(child_id)
            if child is None:
                return None
            child.import_adapter_state(soup_state, alpha=1.0)
            # stipend energy
            stipend = float(getattr(self.config.assimilation_tuning, "trial_stipend", 0.5))
            self.host.ledger.set_energy(child_id, stipend)
            # register in population
            self.population.register(Genome(organelle_id=child_id, drive_weights={"novelty": 0.1}, gate_bias=0.0, rank=target_rank))
            # track probation
            self.trial_offspring[child_id] = {
                "parents": list(soup_ids),
                "cell": {"family": cell[0], "depth": cell[1]},
                "created_gen": self.generation_index,
                "probation_left": int(getattr(self.config.assimilation_tuning, "trial_probation_gens", 5)),
                "promoted": False,
            }
            return child_id
        except Exception:
            return None

    def _review_trial_offspring(self) -> None:
        if not self.trial_offspring:
            return
        margin = float(getattr(self.config.assimilation_tuning, "trial_promote_margin", 0.02))
        min_power = float(getattr(self.config.assimilation_tuning, "trial_min_power", 0.1))
        to_remove: list[str] = []
        for child_id, meta in list(self.trial_offspring.items()):
            if bool(meta.get("promoted", False)):
                to_remove.append(child_id)
                continue
            # sample holdout and compare to baseline mates
            tasks = self._sample_holdout_tasks()
            child_roi = self._evaluate_holdout_roi(str(child_id), tasks)
            parents: list[str] = [str(x) for x in meta.get("parents", [])]
            baselines = [self._evaluate_holdout_roi(pid, tasks) for pid in parents]
            baselines = [b for b in baselines if math.isfinite(b)]
            best_base = max(baselines, default=float("-inf"))
            target = (best_base if math.isfinite(best_base) else 0.0) + margin
            # accumulate evidence across generations
            ev = meta.setdefault("evidence", {"deltas": []})
            try:
                deltas: list[float] = list(ev.get("deltas", []))
            except Exception:
                deltas = []
            uplift = float(child_roi - (best_base if math.isfinite(best_base) else 0.0))
            deltas.append(uplift)
            if len(deltas) > 24:
                deltas = deltas[-24:]
            ev["deltas"] = deltas
            # simple power proxy for mean uplift against margin
            n = len(deltas)
            if n >= 2:
                mean_u = float(sum(deltas) / n)
                var_u = float(sum((x - mean_u) ** 2 for x in deltas) / max(n - 1, 1))
                se = math.sqrt(max(var_u, 1e-12) / n)
                delta = mean_u - margin
                z = delta / max(se, 1e-12)
                z_alpha = 1.64
                from math import erf, sqrt
                Phi = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
                power = max(0.0, min(1.0, 1.0 - float(Phi(z_alpha - z))))
            else:
                mean_u = uplift
                power = 0.0
            if child_roi >= target or (power >= min_power and mean_u >= 0.0):
                # promote
                self.promotions_this_gen += 1
                meta["promoted"] = True
                to_remove.append(child_id)
                # record as assimilation success in history/logs for transparency
                if self.sink:
                    event = AssimilationEvent(
                        organelle_id=str(child_id),
                        uplift=child_roi - (best_base if math.isfinite(best_base) else 0.0),
                        p_value=1.0,
                        passed=True,
                        energy_cost=0.0,
                        safety_hits=0,
                        sample_size=len(tasks),
                        control_mean=best_base if math.isfinite(best_base) else 0.0,
                        treatment_mean=child_roi,
                        control_std=0.0,
                        treatment_std=0.0,
                        ci_low=None,
                        ci_high=None,
                        power=power,
                        uplift_threshold=self.config.evolution.assimilation_threshold,
                        p_value_threshold=self.config.evolution.assimilation_p_value,
                        energy_balance=self.host.ledger.energy_balance(str(child_id)),
                        energy_top_up=None,
                        cell=meta.get("cell"),
                    )
                    self.sink.log_assimilation(event, True)
                continue
            # probation countdown
            meta["probation_left"] = int(meta.get("probation_left", 0)) - 1
            if int(meta["probation_left"]) <= 0:
                # cull child
                to_remove.append(child_id)
        for cid in to_remove:
            # if still present, retire
            self.host.retire_organelle(cid)
            self.population.remove(cid)
            self.trial_offspring.pop(cid, None)

    @staticmethod
    def _sanitize_telemetry(value: object) -> object:
        if isinstance(value, float):
            if math.isfinite(value):
                return float(value)
            return 0.0
        if isinstance(value, (int, str, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {key: EcologyLoop._sanitize_telemetry(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [EcologyLoop._sanitize_telemetry(item) for item in value]
        return str(value)

    def _record_assimilation_gate(self, reason: str, organelle_id: str, details: dict[str, object]) -> None:
        limit = max(1, int(getattr(self.config.assimilation_tuning, "gating_snapshot_limit", 48)))
        sample = {
            "generation": self.generation_index,
            "organelle_id": organelle_id,
            "reason": reason,
            "details": self._sanitize_telemetry(details),
        }
        self.assim_gating_samples.append(sample)
        if len(self.assim_gating_samples) > limit:
            self.assim_gating_samples = self.assim_gating_samples[-limit:]

    def _record_assimilation_attempt(self, details: dict[str, object]) -> None:
        limit = max(1, int(getattr(self.config.assimilation_tuning, "gating_snapshot_limit", 48)))
        sample = self._sanitize_telemetry(details)
        self.assim_attempt_samples.append(sample)
        if len(self.assim_attempt_samples) > limit:
            self.assim_attempt_samples = self.assim_attempt_samples[-limit:]

    def _register_tau_failure(self, cell: GridKey) -> None:
        window = int(getattr(self.config.assimilation_tuning, "tau_relief_window", 12))
        step = float(getattr(self.config.assimilation_tuning, "tau_relief_step", 0.01))
        max_relief = float(getattr(self.config.assimilation_tuning, "tau_relief_max", 0.15))
        if window <= 0 or step <= 0.0 or max_relief <= 0.0:
            return
        count = self._tau_fail_counts.get(cell, 0) + 1
        self._tau_fail_counts[cell] = count
        if count >= window:
            relief = min(max_relief, self._tau_relief.get(cell, 0.0) + step)
            self._tau_relief[cell] = relief
            self._tau_fail_counts[cell] = 0

    def _register_tau_success(self, cell: GridKey) -> None:
        if cell not in self._tau_relief:
            self._tau_fail_counts.pop(cell, None)
            return
        step = float(getattr(self.config.assimilation_tuning, "tau_relief_step", 0.01))
        relief = max(0.0, self._tau_relief.get(cell, 0.0) - step)
        if relief <= 1e-6:
            self._tau_relief.pop(cell, None)
        else:
            self._tau_relief[cell] = relief
        self._tau_fail_counts.pop(cell, None)

    def _register_roi_skip(self, organelle_id: str) -> None:
        window = int(getattr(self.config.assimilation_tuning, "roi_relief_window", 8))
        step = float(getattr(self.config.assimilation_tuning, "roi_relief_step", 0.05))
        max_relief = float(getattr(self.config.assimilation_tuning, "roi_relief_max", 0.5))
        if window <= 0 or step <= 0.0 or max_relief <= 0.0:
            return
        count = self._topup_fail_counts.get(organelle_id, 0) + 1
        self._topup_fail_counts[organelle_id] = count
        if count >= window:
            relief = min(max_relief, self._roi_relief.get(organelle_id, 0.0) + step)
            self._roi_relief[organelle_id] = relief
            self._topup_fail_counts[organelle_id] = 0

    def _register_roi_success(self, organelle_id: str) -> None:
        if organelle_id in self._roi_relief:
            step = float(getattr(self.config.assimilation_tuning, "roi_relief_step", 0.05))
            relief = max(0.0, self._roi_relief.get(organelle_id, 0.0) - step)
            if relief <= 1e-6:
                self._roi_relief.pop(organelle_id, None)
            else:
                self._roi_relief[organelle_id] = relief
        self._topup_fail_counts.pop(organelle_id, None)

    def _update_qd_archive(self) -> int:
        if not getattr(self.config.qd, "enabled", False):
            return len(self._qd_archive)
        # cost bins as in _select_soup_members
        energies = {oid: float(self.population.average_energy(oid)) for oid in self.population.population.keys()}
        vals = sorted(v for v in energies.values() if math.isfinite(v))
        bins: list[float] = []
        if vals:
            try:
                qs = quantiles(vals, n=max(2, getattr(self.config.qd, "cost_bins", 3)), method="inclusive")
                bins = [float(x) for x in qs]
            except Exception:
                bins = [vals[len(vals) // 2]]
        def cost_bin(val: float) -> int:
            if not bins:
                return 0
            for i, edge in enumerate(bins):
                if val <= edge:
                    return i
            return len(bins)
        for oid in self.population.population.keys():
            best = self.environment.best_cell_score(oid)
            if not best:
                continue
            (family, depth), ema = best
            key = (str(family), str(depth), cost_bin(energies.get(oid, 0.0)))
            roi = float(self.population.average_roi(oid, limit=5))
            prev = self._qd_archive.get(key)
            if prev is None or roi > float(prev.get("roi", 0.0)):
                self._qd_archive[key] = {"organelle_id": oid, "roi": roi, "ema": float(ema), "energy": energies.get(oid, 0.0)}
        return len(self._qd_archive)

    def _sample_team_synergy(self) -> None:
        # Sample up to 2 top ROI pairs and log simple synergy stats on matched holdout
        ids = list(self.population.population.keys())
        if len(ids) < 2:
            return
        ranked = sorted(ids, key=lambda oid: self.population.average_roi(oid, limit=5), reverse=True)
        pairs = [(ranked[i], ranked[i+1]) for i in range(0, min(len(ranked)-1, 3), 2)]
        tasks = self._sample_holdout_tasks()
        if not tasks:
            return
        for a, b in pairs:
            solo_a = self._evaluate_holdout_roi(a, tasks)
            solo_b = self._evaluate_holdout_roi(b, tasks)
            # Cheap team ROI proxy: best-of-two per task (approximates routing)
            team_roi = max(solo_a, solo_b)
            synergy = team_roi - (solo_a + solo_b)
            sample = {
                "generation": self.generation_index,
                "a": a,
                "b": b,
                "solo_a": float(solo_a),
                "solo_b": float(solo_b),
                "team": float(team_roi),
                "synergy": float(synergy),
                "tasks": int(len(tasks)),
            }
            self._synergy_window.append(self._sanitize_telemetry(sample))
            if len(self._synergy_window) > 24:
                self._synergy_window = self._synergy_window[-24:]

    def _maybe_promote_colonies(self) -> None:
        if not bool(getattr(self.config.assimilation_tuning, "colonies_enabled", False)):
            return
        windows = int(getattr(self.config.assimilation_tuning, "colony_windows", 3))
        delta = float(getattr(self.config.assimilation_tuning, "colony_synergy_delta", 0.1))
        recent = self._synergy_window[-(windows * 4) :]
        by_pair: dict[tuple[str, str], list[dict[str, object]]] = {}
        for rec in recent:
            a = str(rec.get("a")); b = str(rec.get("b")); s = float(rec.get("synergy", 0.0))
            pair = (min(a, b), max(a, b))
            by_pair.setdefault(pair, []).append(rec)
        variance_improve = float(getattr(self.config.assimilation_tuning, "colony_variance_improve", 0.2))
        margin = float(getattr(self.config.assimilation_tuning, "holdout_margin", 0.03))
        review_interval = int(getattr(self.config.assimilation_tuning, "colony_review_interval", max(3, windows)))
        required_passes = int(getattr(self.config.assimilation_tuning, "colony_required_passes", 2))
        for (a, b), records in by_pair.items():
            if len(records) < windows:
                continue
            window_records = records[-windows:]
            synergies = [float(rec.get("synergy", 0.0)) for rec in window_records]
            mean_s = sum(synergies) / windows if synergies else 0.0
            if mean_s >= delta and (a in self.population.population) and (b in self.population.population):
                cid = f"col_{a[:4]}_{b[:4]}"
                if cid in self.colonies:
                    continue
                team_vals = [float(rec.get("team", 0.0)) for rec in window_records]
                solo_a_vals = [float(rec.get("solo_a", 0.0)) for rec in window_records]
                solo_b_vals = [float(rec.get("solo_b", 0.0)) for rec in window_records]
                try:
                    team_var = float(pstdev(team_vals)) if len(team_vals) >= 2 else 0.0
                except Exception:
                    team_var = 0.0
                solo_var_candidates: list[float] = []
                try:
                    if len(solo_a_vals) >= 2:
                        solo_var_candidates.append(float(pstdev(solo_a_vals)))
                    if len(solo_b_vals) >= 2:
                        solo_var_candidates.append(float(pstdev(solo_b_vals)))
                except Exception:
                    pass
                min_variance = min((v for v in solo_var_candidates if math.isfinite(v)), default=0.0)
                if min_variance > 0.0 and team_var > (1.0 - variance_improve) * min_variance:
                    continue
                colony_meta = {
                    "members": [a, b],
                    "pot": 0.0,
                    "reserve_ratio": 0.25,
                    "created_gen": self.generation_index,
                    "last_review": self.generation_index,
                    "holdout_passes": 0,
                    "holdout_failures": 0,
                    "review_interval": review_interval,
                    "required_passes": required_passes,
                    "last_delta": mean_s,
                    "margin": margin,
                }
                self.colonies[cid] = colony_meta

    def _tick_colonies(self) -> None:
        if not self.colonies:
            return
        cfg = self.config.assimilation_tuning
        max_failures = int(getattr(cfg, "colony_max_failures", 2))
        for cid, meta in list(self.colonies.items()):
            members: list[str] = [str(x) for x in meta.get("members", [])]
            pot = float(meta.get("pot", 0.0))
            earn = 0.0
            for m in members:
                deltas = self.population.recent_energy_deltas(m, limit=4)
                earn += sum(d for d in deltas if isinstance(d, (int, float))) * 0.10
            pot = max(0.0, pot + earn)
            review_interval = int(meta.get("review_interval", 5))
            last_review = int(meta.get("last_review", meta.get("created_gen", self.generation_index)))
            required_passes = int(meta.get("required_passes", 2))
            margin = float(meta.get("margin", getattr(cfg, "holdout_margin", 0.03)))
            if len(members) >= 2 and self.generation_index - last_review >= review_interval:
                tasks = self._sample_holdout_tasks()
                deltas: list[float] = []
                if tasks:
                    try:
                        team_roi = self._evaluate_team_holdout_roi(members[0], members[1], tasks)
                        baselines = [self._evaluate_holdout_roi(mem, tasks) for mem in members]
                        baseline = max(baselines)
                        delta = team_roi - baseline
                    except Exception:
                        delta = float("-inf")
                else:
                    delta = float("-inf")
                meta["last_review"] = self.generation_index
                meta["last_delta"] = float(delta)
                if delta >= margin:
                    meta["holdout_passes"] = int(meta.get("holdout_passes", 0)) + 1
                else:
                    meta["holdout_failures"] = int(meta.get("holdout_failures", 0)) + 1
                # Shrink instead of dissolve when feasible
                try:  # pragma: no cover - exercised in long runs
                    min_size = int(getattr(cfg, "colony_min_size", 2))
                except Exception:  # pragma: no cover
                    min_size = 2
                if (
                    len(members) > min_size
                    and float(meta.get("last_delta", 0.0)) < 0.0
                    and int(meta.get("holdout_failures", 0)) >= 1
                ):
                    try:  # pragma: no cover
                        worst = min(members, key=lambda oid: self.population.average_roi(oid, limit=5))
                        members.remove(worst)
                        meta["members"] = members
                        meta["holdout_failures"] = max(0, int(meta.get("holdout_failures", 0)) - 1)
                    except Exception:  # pragma: no cover
                        pass
                if int(meta.get("holdout_failures", 0)) >= max_failures:
                    self.colonies.pop(cid, None)
                    continue
            reserve_ratio = float(meta.get("reserve_ratio", 0.25))
            privileges_unlocked = int(meta.get("holdout_passes", 0)) >= required_passes
            if privileges_unlocked:
                for m in members:
                    bal = self.host.ledger.energy_balance(m)
                    ticket = self.config.energy.m
                    reserve = reserve_ratio * 4.0 * ticket
                    if pot > reserve and bal < max(ticket, self.config.assimilation_tuning.energy_floor or ticket):
                        amount = min(pot - reserve, max(ticket - bal, 0.0))
                        if amount > 0:
                            self.host.ledger.credit_energy(m, amount)
                            pot -= amount
            meta["pot"] = pot
            # Attempt cautious expansion to a 3rd member if synergy persists
            try:
                max_size = int(getattr(cfg, "colony_max_size", 3))
                if len(members) < max_size and int(meta.get("holdout_passes", 0)) >= required_passes:
                    # Pick a candidate outside the colony with highest recent ROI
                    candidates = [oid for oid in self.population.population.keys() if oid not in members]
                    if candidates:
                        best_cand = max(candidates, key=lambda oid: self.population.average_roi(oid, limit=5))
                        tasks = self._sample_holdout_tasks()
                        if tasks:
                            current = self._evaluate_multi_team_holdout_roi(members, tasks)
                            expanded = self._evaluate_multi_team_holdout_roi(members + [best_cand], tasks)
                            delta = expanded - current
                            margin = float(getattr(cfg, "holdout_margin", 0.03))
                            if delta >= margin:
                                members.append(best_cand)
                                meta["members"] = members
                                meta["last_review"] = self.generation_index
                                meta["last_delta"] = float(delta)
            except Exception:
                pass

    def _auto_tune_assimilation_energy(self, summary: dict[str, object]) -> None:
        tuning = self.config.assimilation_tuning
        window = 12
        costs: list[float] = []
        rois: list[float] = []
        deltas: list[float] = []
        for organelle_id in self.population.population.keys():
            costs.extend(
                value
                for value in self.population.energy.get(organelle_id, [])[-window:]
                if math.isfinite(value)
            )
            rois.extend(
                value
                for value in self.population.roi.get(organelle_id, [])[-window:]
                if math.isfinite(value)
            )
            deltas.extend(
                value
                for value in self.population.recent_energy_deltas(organelle_id, limit=window)
                if math.isfinite(value)
            )
        if not costs or not rois:
            return
        avg_cost = mean(costs)
        n_episodes = max(1, self.config.environment.synthetic_batch_size)
        ticket = max(self.config.energy.m, 1e-6)
        roi_required = 1.0 + ticket / max(n_episodes * avg_cost, 1e-6)
        roi_required = max(1.0, roi_required)
        roi_cap = max(roi_required * 3.0, roi_required + 1.5, 5.0)
        roi_samples = [max(0.0, value) for value in rois if value >= 0.0]
        if not roi_samples:
            percentile = roi_required
        elif len(roi_samples) >= 4:
            try:
                percentile = quantiles(roi_samples, n=4, method="inclusive")[2]
            except Exception:
                percentile = max(roi_samples)
        else:
            percentile = max(roi_samples)
        percentile = max(roi_required, min(percentile, roi_cap))
        std_delta = pstdev(deltas) if len(deltas) >= 2 else 0.0
        target_floor = max(ticket, ticket + std_delta)
        target_floor = min(self.host.ledger.energy_cap, target_floor)

        if not hasattr(self, "_energy_guard_bandit"):
            self._energy_guard_bandit = EnergyGuardBandit(weights=[-1.0, -0.5, 0.0, 0.5])
            if tuning.energy_floor <= 0.0:
                tuning.energy_floor = max(ticket, tuning.energy_floor_base)
            if tuning.energy_floor_roi <= 0.0:
                tuning.energy_floor_roi = max(1.0, tuning.energy_floor_roi_base)

        bandit: EnergyGuardBandit = self._energy_guard_bandit
        bandit.record_reward(self._energy_guard_reward(summary))

        base_floor = max(ticket, tuning.energy_floor_base)
        base_threshold = max(1.0, tuning.energy_floor_roi_base)
        stat_floor = target_floor
        stat_threshold = percentile

        for arm in bandit.arms:
            weight = arm.weight
            floor_candidate = base_floor + weight * (stat_floor - base_floor)
            threshold_candidate = base_threshold + weight * (stat_threshold - base_threshold)
            floor_candidate = max(ticket, min(self.host.ledger.energy_cap, floor_candidate))
            threshold_candidate = max(1.0, min(roi_cap, threshold_candidate))
            arm.floor = floor_candidate
            arm.roi = threshold_candidate

        choice = bandit.select()
        selected = bandit.arms[choice]
        smoothing = 0.12
        current_floor = tuning.energy_floor if tuning.energy_floor > 0.0 else base_floor
        current_threshold = tuning.energy_floor_roi if tuning.energy_floor_roi > 0.0 else base_threshold
        new_floor = (1.0 - smoothing) * current_floor + smoothing * selected.floor
        new_threshold = (1.0 - smoothing) * current_threshold + smoothing * selected.roi
        new_floor = max(ticket, min(self.host.ledger.energy_cap, new_floor))
        new_threshold = max(1.0, min(roi_cap, new_threshold, base_threshold + 0.4))
        decay_applied = False
        stall_generations = getattr(self, "no_merge_counter", 0)
        stall_window = max(4, self.config.assimilation_tuning.per_cell_interval * 2)
        if stall_generations >= stall_window:
            factor = 0.7 ** max(1, stall_generations // stall_window)
            new_threshold = max(1.0, new_threshold * factor)
            new_floor = max(ticket, new_floor * 0.85)
            decay_applied = True
        tuning.energy_floor = new_floor
        tuning.energy_floor_roi = new_threshold
        summary["assimilation_energy_tuning"] = {
            "avg_cost": avg_cost,
            "roi_required": roi_required,
            "roi_percentile": stat_threshold,
            "roi_cap": roi_cap,
            "energy_floor": new_floor,
            "energy_floor_roi": new_threshold,
            "delta_std": std_delta,
            "bandit_choice": choice,
            "bandit_weight": selected.weight,
            "bandit_reward": getattr(bandit, "last_reward", 0.0),
            "base_floor": base_floor,
            "base_roi": base_threshold,
            "stall_generations": stall_generations,
            "decay_applied": decay_applied,
        }

    def _maybe_top_up_energy(self, genome: Genome, balance: float) -> tuple[float, dict[str, float | str]]:
        tuning = self.config.assimilation_tuning
        floor = getattr(tuning, "energy_floor", 0.0)
        roi_threshold = getattr(tuning, "energy_floor_roi", 0.0)
        roi_bonus = getattr(tuning, "energy_topup_roi_bonus", 0.0)
        relief = float(self._roi_relief.get(genome.organelle_id, 0.0))
        # dynamic easing: variance and fail streak make top-ups slightly easier to build evidence
        try:
            recent = self.population.roi.get(genome.organelle_id, [])[-8:]
            roi_std = pstdev([r for r in recent if math.isfinite(r)]) if len(recent) >= 2 else 0.0
        except Exception:
            roi_std = 0.0
        streak = max(0, int(getattr(self, "assim_fail_streak", 0)))
        dynamic_bonus = min(1.5, float(roi_bonus) + 0.15 * float(roi_std) + 0.02 * float(streak))
        effective_threshold = max(0.0, float(roi_threshold) - dynamic_bonus - relief)
        info: dict[str, float | str] = {
            "status": "disabled" if floor <= 0.0 else "pending",
            "before": float(balance),
            "after": float(balance),
            "floor": float(floor),
            "roi_threshold": float(roi_threshold),
            "roi_threshold_effective": float(effective_threshold),
            "credited": 0.0,
            "roi_std": float(roi_std),
            "fail_streak": float(streak),
            "relief": relief,
        }
        if floor <= 0.0:
            return balance, info
        if balance >= floor:
            info["status"] = "already_sufficient"
            info["after"] = float(balance)
            return balance, info
        roi = self.population.average_roi(genome.organelle_id, limit=5)
        info["roi"] = float(roi)
        if roi < effective_threshold:
            # attempt evidence-bypass top-up (only if credit present)
            try:
                pre_credit = 0
                credits = getattr(self.population, "evidence_credit", {})
                if isinstance(credits, dict):
                    pre_credit = int(credits.get(genome.organelle_id, 0) or 0)
                if pre_credit > 0 and bool(self.population.consume_evidence(genome.organelle_id, 1)):
                    ledger = self.host.ledger
                    available = max(0.0, min(floor - balance, ledger.energy_cap - balance))
                    if available > 0.0:
                        ledger.credit_energy(genome.organelle_id, available)
                        new_balance = ledger.energy_balance(genome.organelle_id)
                        info["status"] = "credited"
                        info["credited"] = float(new_balance - balance)
                        info["after"] = float(new_balance)
                        info["floor"] = float(floor)
                        info["roi_threshold"] = float(roi_threshold)
                        info["evidence_bypass"] = True
                        self._register_roi_success(genome.organelle_id)
                        return new_balance, info
            except Exception:
                pass
            info["status"] = "skip_low_roi"
            self._register_roi_skip(genome.organelle_id)
            return balance, info
        ledger = self.host.ledger
        available = max(0.0, min(floor - balance, ledger.energy_cap - balance))
        if available <= 0.0:
            info["status"] = "skip_no_capacity"
            return balance, info
        ledger.credit_energy(genome.organelle_id, available)
        new_balance = ledger.energy_balance(genome.organelle_id)
        info["status"] = "credited"
        info["credited"] = float(new_balance - balance)
        info["after"] = float(new_balance)
        info["floor"] = float(floor)
        info["roi_threshold"] = float(roi_threshold)
        self._register_roi_success(genome.organelle_id)
        return new_balance, info

    def _load_holdout_tasks(self) -> list[EvaluationTask]:
        if self._holdout_tasks_cache is not None:
            return self._holdout_tasks_cache
        cfg = self.config.assimilation_tuning
        path = cfg.holdout_tasks_path or self.config.evaluation.tasks_path
        if path is None:
            self._holdout_tasks_cache = []
            return self._holdout_tasks_cache
        try:
            runtime = EvaluationManager.from_file(
                enabled=True,
                cadence=1,
                tasks_path=path,
                sample_size=None,
                reward_weight=self.config.evaluation.reward_weight,
            )
            self._holdout_tasks_cache = list(runtime.tasks)
        except Exception:
            self._holdout_tasks_cache = []
        return self._holdout_tasks_cache

    def _sample_holdout_tasks(self) -> list[EvaluationTask]:
        tasks = self._load_holdout_tasks()
        if not tasks:
            return []
        sample_size = max(1, min(len(tasks), self.config.assimilation_tuning.holdout_sample_size))
        if len(tasks) <= sample_size:
            return list(tasks)
        return self.holdout_rng.sample(tasks, sample_size)

    def _evaluate_holdout_roi(self, organelle_id: str, tasks: list[EvaluationTask]) -> float:
        if not tasks:
            return 0.0
        energy_cfg = self.config.energy
        rois: list[float] = []
        for index, task in enumerate(tasks, start=1):
            grid_task = task.to_grid_task(self.environment, task_id=f"holdout_{index:04d}")
            result = self.host.step(
                prompt=grid_task.prompt,
                intent="assimilation holdout",
                max_routes=1,
                allowed_organelle_ids=[organelle_id],
            )
            metrics = result.responses.get(organelle_id)
            if metrics is None:
                continue
            success, reward = grid_task.evaluate(metrics.answer)
            revenue = grid_task.price * reward.total
            cost = (
                energy_cfg.alpha * metrics.flops_estimate
                + energy_cfg.beta * metrics.memory_gb
                + energy_cfg.gamma * metrics.latency_ms
                + energy_cfg.lambda_p * metrics.trainable_params
            )
            if cost <= 0.0:
                roi_value = float("inf") if revenue > 0 else 0.0
            else:
                roi_value = revenue / cost
            if not math.isfinite(roi_value):
                roi_value = max(0.0, min(roi_value, 10.0)) if revenue > 0 else 0.0
            if not math.isfinite(roi_value):
                continue
            rois.append(roi_value)
        if not rois:
            return float("-inf")
        return sum(rois) / len(rois)

    def _evaluate_team_holdout_roi(self, a_id: str, b_id: str, tasks: list[EvaluationTask]) -> float:
        if not tasks:
            return 0.0
        energy_cfg = self.config.energy
        rois: list[float] = []
        for index, task in enumerate(tasks, start=1):
            grid_task = task.to_grid_task(self.environment, task_id=f"team_{index:04d}")
            best_roi = 0.0
            for oid in (a_id, b_id):
                result = self.host.step(
                    prompt=grid_task.prompt,
                    intent="team holdout",
                    max_routes=1,
                    allowed_organelle_ids=[oid],
                )
                metrics = result.responses.get(oid)
                if metrics is None:
                    continue
                success, reward = grid_task.evaluate(metrics.answer)
                revenue = grid_task.price * reward.total
                cost = (
                    energy_cfg.alpha * metrics.flops_estimate
                    + energy_cfg.beta * metrics.memory_gb
                    + energy_cfg.gamma * metrics.latency_ms
                    + energy_cfg.lambda_p * metrics.trainable_params
                )
                if cost <= 0.0:
                    roi_value = float("inf") if revenue > 0 else 0.0
                else:
                    roi_value = revenue / cost
                if not math.isfinite(roi_value):
                    roi_value = max(0.0, min(roi_value, 10.0)) if revenue > 0 else 0.0
                best_roi = max(best_roi, float(roi_value))
            rois.append(best_roi)
        return float(sum(rois) / max(len(rois), 1)) if rois else 0.0

    def _evaluate_multi_team_holdout_roi(self, member_ids: list[str], tasks: list[EvaluationTask]) -> float:
        if not tasks or not member_ids:
            return 0.0
        energy_cfg = self.config.energy
        rois: list[float] = []
        for index, task in enumerate(tasks, start=1):
            grid_task = task.to_grid_task(self.environment, task_id=f"team_{index:04d}")
            best_roi = 0.0
            for oid in member_ids:
                result = self.host.step(
                    prompt=grid_task.prompt,
                    intent="team holdout",
                    max_routes=1,
                    allowed_organelle_ids=[oid],
                )
                metrics = result.responses.get(oid)
                if metrics is None:
                    continue
                success, reward = grid_task.evaluate(metrics.answer)
                revenue = grid_task.price * reward.total
                cost = (
                    energy_cfg.alpha * metrics.flops_estimate
                    + energy_cfg.beta * metrics.memory_gb
                    + energy_cfg.gamma * metrics.latency_ms
                    + energy_cfg.lambda_p * metrics.trainable_params
                )
                if cost <= 0.0:
                    roi_value = float("inf") if revenue > 0 else 0.0
                else:
                    roi_value = revenue / cost
                if not math.isfinite(roi_value):
                    roi_value = max(0.0, min(roi_value, 10.0)) if revenue > 0 else 0.0
                best_roi = max(best_roi, float(roi_value))
            rois.append(best_roi)
        return float(sum(rois) / max(len(rois), 1)) if rois else 0.0

    @staticmethod
    def _compute_mean_ci(series: list[float], alpha: float = 0.05) -> tuple[float, float, float, float]:
        """Compute a normal-approximation CI for the mean of a series.

        Returns (ci_low, ci_high, mean, se). Uses sample std (Bessel corrected) when n>1.
        """
        n = len(series)
        if n == 0:
            return (0.0, 0.0, 0.0, float("inf"))
        mu = sum(series) / n
        if n > 1:
            var = sum((x - mu) ** 2 for x in series) / (n - 1)
        else:
            var = 0.0
        se = math.sqrt(max(var, 1e-12)) / math.sqrt(n)
        z = 1.96 if abs(alpha - 0.05) < 1e-6 else 1.0
        return (mu - z * se, mu + z * se, mu, se)

    @staticmethod
    def _power_proxy(mu: float, baseline: float, margin: float, se: float, alpha: float = 0.05) -> float:
        """Approximate one-sided power using a normal approximation.

        Computes z_eff = (mu - (baseline + margin)) / se and returns Phi(z_eff - z_alpha).
        This is a heuristic proxy in lieu of an exact test; bounded to [0,1].
        """
        import math as _m
        if se <= 0.0:
            return 1.0 if mu > (baseline + margin) else 0.0
        z_alpha = 1.645 if abs(alpha - 0.05) < 1e-6 else 1.0
        z_eff = (mu - (baseline + margin)) / max(se, 1e-6)
        # standard normal CDF via erf
        cdf = 0.5 * (1.0 + _m.erf((z_eff - z_alpha) / _m.sqrt(2.0)))
        return max(0.0, min(1.0, float(cdf)))

    @staticmethod
    def _team_accept(ci_low: float, baseline: float, margin: float, n: int, min_tasks: int) -> bool:
        if n < min_tasks:
            return False
        return ci_low > (baseline + margin)

    def _maybe_team_probes(self) -> int:
        """Probe a few high-ROI pairs per generation and promote colonies when CI gate passes.

        Returns number of promotions.
        """
        per_gen = int(getattr(self.config.assimilation_tuning, "team_probe_per_gen", 0))
        if per_gen <= 0 or len(self.population.population) < 2:
            return 0
        # Pick top-K candidates by recent ROI
        scored = [
            (oid, float(self.population.average_roi(oid, limit=5)))
            for oid in self.population.population.keys()
        ]
        if not scored:
            return 0
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [oid for oid, _ in scored[: min(6, len(scored))]]
        # Build candidate pairs; prefer co-routing synergy if available
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        # 1) From co-routing counts
        if hasattr(self, "_co_routing_counts") and self._co_routing_counts:
            pairs_sorted = sorted(self._co_routing_counts.items(), key=lambda kv: kv[1], reverse=True)
            for (a, b), _c in pairs_sorted:
                if a in top and b in top and (a, b) not in seen:
                    pairs.append((a, b))
                    seen.add((a, b))
        # 2) Fill with top-ROI unique pairs
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                key = (top[i], top[j])
                if key not in seen:
                    pairs.append(key)
                    seen.add(key)
        promotions = 0
        rng = self.environment.rng
        rng.shuffle(pairs)
        for (a_id, b_id) in pairs[:per_gen]:
            tasks = self._sample_holdout_tasks()
            if not tasks:
                continue
            # Compute best-of-two ROI series and baseline means
            energy_cfg = self.config.energy
            team_series: list[float] = []
            a_series: list[float] = []
            b_series: list[float] = []
            for index, task in enumerate(tasks, start=1):
                grid_task = task.to_grid_task(self.environment, task_id=f"team_probe_{index:04d}")
                best_roi = 0.0
                for oid, series in ((a_id, a_series), (b_id, b_series)):
                    result_i = self.host.step(
                        prompt=grid_task.prompt,
                        intent="team probe",
                        max_routes=1,
                        allowed_organelle_ids=[oid],
                    )
                    metrics_i = result_i.responses.get(oid)
                    if metrics_i is None:
                        continue
                    success_i, reward_i = grid_task.evaluate(metrics_i.answer)
                    revenue_i = grid_task.price * reward_i.total
                    cost_i = (
                        energy_cfg.alpha * metrics_i.flops_estimate
                        + energy_cfg.beta * metrics_i.memory_gb
                        + energy_cfg.gamma * metrics_i.latency_ms
                        + energy_cfg.lambda_p * metrics_i.trainable_params
                    )
                    roi_i = (float("inf") if revenue_i > 0 else 0.0) if cost_i <= 0.0 else (revenue_i / cost_i)
                    roi_i = 0.0 if not math.isfinite(roi_i) else float(max(0.0, min(roi_i, 10.0)))
                    series.append(roi_i)
                    best_roi = max(best_roi, roi_i)
                team_series.append(best_roi)
            base_a = sum(a_series) / max(len(a_series), 1)
            base_b = sum(b_series) / max(len(b_series), 1)
            baseline = max(base_a, base_b)
            ci_low, ci_high, team_mu, team_se = self._compute_mean_ci(team_series)
            min_tasks = int(getattr(self.config.assimilation_tuning, "team_min_tasks", 8))
            # team_holdout_margin can be present but None; fall back to generic holdout_margin
            margin_val = getattr(self.config.assimilation_tuning, "team_holdout_margin", None)
            if margin_val is None:
                margin_val = getattr(self.config.assimilation_tuning, "holdout_margin", 0.02)
            try:
                margin = float(margin_val)
            except Exception:
                margin = 0.02
            # Power proxy gate
            min_power = float(getattr(self.config.assimilation_tuning, "team_min_power", 0.2))
            power = self._power_proxy(team_mu, baseline, margin, team_se)
            if power >= min_power and self._team_accept(ci_low, baseline, margin, len(team_series), min_tasks):
                cid = f"col_{a_id[:4]}_{b_id[:4]}"
                self.colonies.setdefault(cid, {"members": [a_id, b_id], "pot": 0.0, "reserve_ratio": 0.25, "created_gen": self.generation_index})
                promotions += 1
        if promotions > 0:
            self.promotions_this_gen += promotions
        return promotions

    def _holdout_accepts(self, candidate_id: str, mate_ids: list[str]) -> tuple[bool, dict[str, object] | None]:
        cfg = self.config.assimilation_tuning
        retries = 0
        margin = cfg.holdout_margin
        info: dict[str, object] | None = None
        while retries <= cfg.holdout_max_retries:
            tasks = self._sample_holdout_tasks()
            if not tasks:
                return True, info
            candidate_roi = self._evaluate_holdout_roi(candidate_id, tasks)
            baseline_rois = [self._evaluate_holdout_roi(mid, tasks) for mid in mate_ids]
            baseline_rois = [roi for roi in baseline_rois if math.isfinite(roi)]
            best_baseline = max(baseline_rois, default=float("-inf"))
            target = (best_baseline if math.isfinite(best_baseline) else 0.0) + margin
            accepted = candidate_roi >= target
            info = {
                "candidate_roi": candidate_roi,
                "baseline_roi": best_baseline if math.isfinite(best_baseline) else None,
                "margin": margin,
                "tasks": len(tasks),
                "retries": retries,
            }
            if accepted:
                return True, info
            retries += 1
            margin = max(0.0, margin - cfg.holdout_margin_step)
        return False, info

    def _species_partition(self) -> dict[tuple[tuple[str, int], ...], list[str]]:
        mapping: dict[tuple[tuple[str, int], ...], list[str]] = {}
        for organelle_id in self.host.list_organelle_ids():
            organelle = self.host.get_organelle(organelle_id)
            if organelle is None:
                continue
            adapters = self.host._active_adapters(organelle)
            key = tuple(sorted((module, int(count)) for module, count in adapters.items() if module not in {"rank", "total"}))
            mapping.setdefault(key, []).append(organelle_id)
        return mapping

    @staticmethod
    def _compute_diversity_metrics(
        balances: dict[str, float],
        species_map: dict[tuple[tuple[str, int], ...], list[str]],
    ) -> dict[str, float]:
        values = [max(0.0, val) for val in balances.values()]
        total = sum(values)
        metrics = {
            "energy_gini": 0.0,
            "effective_population": 0.0,
            "species_count": float(len(species_map) if species_map else 0),
            "max_species_share": 0.0,
        }
        if not values or total <= 0.0:
            return metrics
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        cumulative = sum((i + 1) * val for i, val in enumerate(sorted_vals))
        gini = (2.0 * cumulative) / (n * total) - (n + 1) / n
        shares = [val / total for val in values if val > 0.0]
        effective = 1.0 / sum(share * share for share in shares) if shares else 0.0
        species_shares = [sum(balances[oid] for oid in ids) / total for ids in species_map.values() if ids]
        metrics.update(
            {
                "energy_gini": float(max(0.0, min(gini, 1.0))),
                "effective_population": float(effective),
                "max_species_share": float(max(species_shares) if species_shares else 0.0),
            }
        )
        return metrics

    def _enforce_diversity(self) -> None:
        cfg = getattr(self.config, "diversity", None)
        if cfg is None or not cfg.enabled:
            self._diversity_snapshot = None
            return
        organelle_ids = self.host.list_organelle_ids()
        balances = {org_id: max(0.0, self.host.ledger.energy_balance(org_id)) for org_id in organelle_ids}
        species_map = self._species_partition()
        metrics_before = self._compute_diversity_metrics(balances, species_map)
        total_energy = sum(balances.values())
        enforced = False
        if total_energy > 0 and cfg.energy_gini_cap < 1.0 and metrics_before["energy_gini"] > cfg.energy_gini_cap:
            mean_energy = total_energy / max(len(organelle_ids), 1)
            blended = {org_id: 0.5 * balances[org_id] + 0.5 * mean_energy for org_id in organelle_ids}
            scale = total_energy / (sum(blended.values()) or 1.0)
            for org_id, value in blended.items():
                new_energy = value * scale
                self.host.ledger.set_energy(org_id, new_energy)
                balances[org_id] = new_energy
            total_energy = sum(balances.values())
            enforced = True
        if total_energy > 0 and cfg.max_species_energy_share < 1.0:
            cap = max(0.0, min(cfg.max_species_energy_share, 1.0))
            species_map = self._species_partition()
            for ids in species_map.values():
                species_energy = sum(balances.get(org_id, 0.0) for org_id in ids)
                if species_energy <= 0.0:
                    continue
                share = species_energy / total_energy
                if share > cap:
                    enforced = True
                    scale = (cap * total_energy) / species_energy
                    for org_id in ids:
                        new_energy = balances.get(org_id, 0.0) * scale
                        self.host.ledger.set_energy(org_id, new_energy)
                        balances[org_id] = new_energy
            total_energy = sum(balances.values())
        species_map = self._species_partition()
        metrics_after = self._compute_diversity_metrics(balances, species_map)
        metrics_after["enforced"] = float(enforced)
        self._diversity_snapshot = metrics_after

    def _energy_guard_reward(self, summary: dict[str, object]) -> float:
        merges = float(summary.get("merges", 0) or 0)
        avg_roi = float(summary.get("avg_roi", 0.0) or 0.0)
        gating = summary.get("assimilation_gating") or {}
        low_energy = float(gating.get("low_energy", 0) or 0)
        cooldown = float(gating.get("cooldown", 0) or 0)
        insufficient = float(gating.get("insufficient_scores", 0) or 0)
        reward = merges * 2.0 + max(0.0, avg_roi)
        reward -= 0.002 * low_energy
        reward -= 0.01 * cooldown
        reward -= 0.005 * insufficient
        return reward

    def _maybe_decay_assimilation_thresholds(self) -> None:
        cfg = self.config.assimilation_tuning
        if self.assim_fail_streak <= 0:
            return
        if self.assim_fail_streak % 10 != 0:
            return
        decay = max(0.5, min(cfg.adaptive_decay, 1.0))
        new_threshold = max(cfg.adaptive_floor, self.config.evolution.assimilation_threshold * decay)
        self.config.evolution.assimilation_threshold = new_threshold
        self.assimilation.update_thresholds(uplift_threshold=new_threshold)
        # relax p-value marginally when decay triggers
        self.config.evolution.assimilation_p_value = min(0.5, self.config.evolution.assimilation_p_value * (1.0 + (1.0 - decay)))

    def _auto_nudge_evidence(self, summary: dict[str, object]) -> None:
        """Adapt assimilation evidence knobs in‑run when progress stalls.

        Nudges are incremental and bounded, and revert softly after a success.
        """
        tuning = self.config.assimilation_tuning
        gating = summary.get("assimilation_gating") or {}
        if not isinstance(gating, dict):
            gating = {}
        low_power = int(gating.get("low_power", 0) or 0)
        uplift_below = int(gating.get("uplift_below_threshold", 0) or 0)
        topup_blocked = int(gating.get("topup_roi_blocked", 0) or 0)
        insufficient = int(gating.get("insufficient_scores", 0) or 0)
        promotions = int(summary.get("promotions", 0) or 0)
        merges = int(summary.get("merges", 0) or 0)
        # Initialize baselines once
        if not hasattr(self, "_nudge_baseline"):
            self._nudge_baseline = {
                "min_window": int(getattr(tuning, "min_window", 4)),
                "holdout": int(getattr(tuning, "holdout_sample_size", 4)),
                "cap": int(getattr(tuning, "trial_per_gen_cap", 2)),
                "prob": int(getattr(tuning, "trial_probation_gens", 5)),
                "stipend": float(getattr(tuning, "trial_stipend", 0.5)),
                "bonus": float(getattr(tuning, "energy_topup_roi_bonus", 0.0)),
                "tau": float(self.config.controller.tau),
            }
        base = self._nudge_baseline  # type: ignore[attr-defined]
        # Decide whether to nudge up evidence or relax back
        stall = (self.assim_fail_streak >= 8) or (low_power >= 2 and promotions == 0 and merges == 0) or (topup_blocked >= 5)
        changed: dict[str, float] = {}
        # Optional: auto-tune min_window downward when insufficient_scores dominates
        if bool(getattr(tuning, "window_autotune", False)) and insufficient >= 50:
            mw = int(getattr(tuning, "min_window", 4))
            lower = int(getattr(tuning, "min_window_min", 6))
            if mw > lower:
                new_mw = max(lower, mw - 2)
                if new_mw % 2 == 1:
                    new_mw -= 1
                tuning.min_window = max(lower, new_mw)
        if stall:
            # Increase evidence and budget within bounds
            mw = int(getattr(tuning, "min_window", 4))
            ho = int(getattr(tuning, "holdout_sample_size", 4))
            cap = int(getattr(tuning, "trial_per_gen_cap", 2))
            prob = int(getattr(tuning, "trial_probation_gens", 5))
            stipend = float(getattr(tuning, "trial_stipend", 0.5))
            bonus = float(getattr(tuning, "energy_topup_roi_bonus", 0.0))
            tau = float(self.config.controller.tau)
            new_mw = min(12, mw + 2)
            if new_mw % 2 == 1:
                new_mw += 1
            new_ho = min(24, ho + 2)
            new_cap = min(4, cap + 1)
            new_prob = min(12, prob + 2)
            new_stipend = min(1.2, stipend + 0.1)
            new_bonus = min(1.5, bonus + 0.1)
            new_tau = max(0.35, tau - 0.01)
            if new_mw != mw:
                tuning.min_window = new_mw
                changed["min_window"] = new_mw
            if new_ho != ho:
                tuning.holdout_sample_size = new_ho
                changed["holdout_sample_size"] = new_ho
            if new_cap != cap:
                tuning.trial_per_gen_cap = new_cap
                changed["trial_per_gen_cap"] = new_cap
            if new_prob != prob:
                tuning.trial_probation_gens = new_prob
                changed["trial_probation_gens"] = new_prob
            if new_stipend != stipend:
                tuning.trial_stipend = new_stipend
                changed["trial_stipend"] = new_stipend
            if new_bonus != bonus:
                tuning.energy_topup_roi_bonus = new_bonus
                changed["energy_topup_roi_bonus"] = new_bonus
            if new_tau != tau:
                self.config.controller.tau = new_tau
                changed["tau"] = new_tau
        elif promotions > 0 or merges > 0:
            # Softly revert towards baselines after successes
            mw = int(getattr(tuning, "min_window", 4))
            ho = int(getattr(tuning, "holdout_sample_size", 4))
            cap = int(getattr(tuning, "trial_per_gen_cap", 2))
            prob = int(getattr(tuning, "trial_probation_gens", 5))
            stipend = float(getattr(tuning, "trial_stipend", 0.5))
            bonus = float(getattr(tuning, "energy_topup_roi_bonus", 0.0))
            tau = float(self.config.controller.tau)
            def step_towards(cur: float, tgt: float, step: float) -> float:
                if cur > tgt:
                    return max(tgt, cur - step)
                if cur < tgt:
                    return min(tgt, cur + step)
                return cur
            new_mw = int(step_towards(mw, base["min_window"], 2.0))
            if new_mw % 2 == 1:
                new_mw -= 1
            new_ho = int(step_towards(ho, base["holdout"], 2.0))
            new_cap = int(step_towards(cap, base["cap"], 1.0))
            new_prob = int(step_towards(prob, base["prob"], 2.0))
            new_stipend = step_towards(stipend, base["stipend"], 0.1)
            new_bonus = step_towards(bonus, base["bonus"], 0.1)
            new_tau = step_towards(tau, base["tau"], 0.01)
            if new_mw != mw:
                tuning.min_window = new_mw
                changed["min_window"] = new_mw
            if new_ho != ho:
                tuning.holdout_sample_size = new_ho
                changed["holdout_sample_size"] = new_ho
            if new_cap != cap:
                tuning.trial_per_gen_cap = new_cap
                changed["trial_per_gen_cap"] = new_cap
            if new_prob != prob:
                tuning.trial_probation_gens = new_prob
                changed["trial_probation_gens"] = new_prob
            if new_stipend != stipend:
                tuning.trial_stipend = float(new_stipend)
                changed["trial_stipend"] = float(new_stipend)
            if new_bonus != bonus:
                tuning.energy_topup_roi_bonus = float(new_bonus)
                changed["energy_topup_roi_bonus"] = float(new_bonus)
            if new_tau != tau:
                self.config.controller.tau = float(new_tau)
                changed["tau"] = float(new_tau)
        if changed:
            summary["adaptive_nudges"] = {k: float(v) for k, v in changed.items()}

    def _collect_human_feedback(
        self, prompt: str, responses: dict[str, tuple[str, float]]
    ) -> Optional[dict[str, HumanFeedbackResult]]:
        if self.human_bandit is None or not self.config.human_bandit.enabled:
            return None
        freq = getattr(self.human_bandit, "frequency", self.config.human_bandit.frequency)
        freq = max(0.0, min(1.0, freq))
        if freq <= 0.0:
            return None
        interval = 1 if freq >= 1.0 else max(int(round(1.0 / freq)), 1)
        if self.bandit_counter % interval != 0:
            self.bandit_counter += 1
            return None
        self.bandit_counter += 1
        feedback: dict[str, HumanFeedbackResult] = {}
        for organelle_id, (answer, _energy) in responses.items():
            feedback[organelle_id] = self.human_bandit.solicit(prompt, answer)
        return feedback

    def _blend_rewards(
        self,
        base_rewards: dict[str, RewardBreakdown],
        human_feedback: dict[str, HumanFeedbackResult],
    ) -> dict[str, RewardBreakdown]:
        combined: dict[str, RewardBreakdown] = {}
        for organelle_id, reward in base_rewards.items():
            human = human_feedback.get(organelle_id)
            if human is None:
                combined[organelle_id] = reward
                continue
            hf = human.reward
            combined[organelle_id] = RewardBreakdown(
                task_reward=reward.task_reward + hf.task_reward,
                novelty_bonus=reward.novelty_bonus + hf.novelty_bonus,
                competence_bonus=reward.competence_bonus + hf.competence_bonus,
                helper_bonus=reward.helper_bonus + hf.helper_bonus,
                risk_penalty=reward.risk_penalty + hf.risk_penalty,
                cost_penalty=reward.cost_penalty + hf.cost_penalty,
            )
        return combined

    def _spawn_replacement_from(self, genome: Genome) -> None:
        mutant_template = self.population.mutate(genome)
        new_id = self.host.spawn_organelle(
            rank=mutant_template.rank,
            hebbian_config=self.config.hebbian,
            activation_bias=mutant_template.gate_bias,
        )
        child_genome = Genome(
            organelle_id=new_id,
            drive_weights=mutant_template.drive_weights,
            gate_bias=mutant_template.gate_bias,
            rank=mutant_template.rank,
        )
        self.population.register(child_genome)

    def _compute_viability_map(self) -> Dict[str, bool]:
        threshold = self.config.energy.m
        viability: Dict[str, bool] = {}
        canary_failed = getattr(self.environment, "canary_failed", None)
        for organelle_id in self.host.list_organelle_ids():
            has_energy = self.host.ledger.energy_balance(organelle_id) >= threshold
            canary_ok = True
            if callable(canary_failed):
                try:
                    canary_ok = not canary_failed(organelle_id)
                except Exception:
                    canary_ok = True
            viability[organelle_id] = has_energy and canary_ok
        return viability

    def _mu_lambda_selection(self, viability: Dict[str, bool]) -> list[Genome]:
        strategy = self.config.population_strategy
        ranked = self.population.rank_for_selection(viability)
        if not ranked:
            return []
        mu = max(1, min(strategy.mu, len(ranked)))
        survivors = ranked[:mu]
        survivor_ids = {genome.organelle_id for genome in survivors}
        for genome in list(self.population.population.values()):
            if genome.organelle_id not in survivor_ids:
                self.host.retire_organelle(genome.organelle_id)
                self.population.remove(genome.organelle_id)
        return survivors

    def _spawn_offspring(self, survivors: list[Genome]) -> None:
        strategy = self.config.population_strategy
        if not survivors:
            return
        current_population = len(self.population.population)
        lambda_quota = max(0, min(strategy.lambda_, strategy.max_population - current_population))
        if lambda_quota <= 0:
            return
        for idx in range(lambda_quota):
            parent = survivors[idx % len(survivors)]
            self._spawn_replacement_from(parent)

    def _apply_morphogenesis(self, survivors: list[Genome]) -> None:
        if not survivors or self.morphogenesis is None:
            return
        self.morphogenesis.apply(survivors, self.population)

    def _global_probe(
        self,
        organelle_id: str,
        cell: GridKey,
        gating: Dict[str, int],
        per_cell: int = 1,
    ) -> bool:
        passes = 0
        total_required: int

        # Always probe the focus cell first
        focus_tasks = [self.environment.sample_task_from_cell(cell, canary=False) for _ in range(max(1, per_cell))]
        for task in focus_tasks:
            if not self._run_probe_task(organelle_id, task):
                gating["global_probe_failed"] += 1
                return False
        passes += len(focus_tasks)

        other_cells = [other for other in self.environment.iter_cells() if other != cell]
        max_others = self.config.assimilation_tuning.probe_max_other_cells
        if max_others is not None:
            max_others = max(0, max_others)
            other_cells = other_cells[:max_others]

        other_tasks: list[GridTask] = []
        for other in other_cells:
            other_tasks.append(self.environment.sample_task_from_cell(other, canary=False))

        required_passes = self.config.assimilation_tuning.probe_required_passes
        total_checks = len(focus_tasks) + len(other_tasks)
        if required_passes is None:
            total_required = total_checks
        else:
            total_required = min(max(1, required_passes), total_checks)

        for task in other_tasks:
            if self._run_probe_task(organelle_id, task):
                passes += 1
            # Early exit when meeting requirement
            if passes >= total_required:
                return True

        if passes >= total_required:
            return True
        gating["global_probe_failed"] += 1
        return False

    def _run_probe_task(self, organelle_id: str, task: GridTask) -> bool:
        result = self.host.step(
            prompt=task.prompt,
            intent="probe assimilation",
            max_routes=1,
            allowed_organelle_ids=[organelle_id],
        )
        metrics = result.responses.get(organelle_id)
        if metrics is None:
            return False
        success, _reward = task.evaluate(metrics.answer)
        return success

    def _select_soup_members(self, cell: GridKey, candidate_id: str) -> tuple[list[str], dict[str, dict[str, float]]]:
        soup_size = max(2, self.config.assimilation_tuning.soup_size)
        stats_map: dict[str, dict[str, float]] = {}
        candidate_roi = max(self.population.average_roi(candidate_id, limit=5), 0.0)
        candidate_ema = float(self.environment.organism_stats.get(candidate_id, {}).get(cell, 0.0))
        stats_map[candidate_id] = {"roi": float(candidate_roi), "ema": candidate_ema}
        # Compute simple cost bins (QD) from recent average energy to prefer similar-cost merges
        energies: dict[str, float] = {}
        for oid in self.population.population.keys():
            energies[oid] = float(self.population.average_energy(oid))
        energy_values = sorted(v for v in energies.values() if math.isfinite(v))
        bins: list[float] = []
        if energy_values:
            try:
                qs = quantiles(energy_values, n=max(2, getattr(self.config.qd, "cost_bins", 3)), method="inclusive")
                bins = [float(x) for x in qs]
            except Exception:
                bins = [energy_values[len(energy_values) // 2]]
        def cost_bin(val: float) -> int:
            if not bins:
                return 0
            for i, edge in enumerate(bins):
                if val <= edge:
                    return i
            return len(bins)
        cand_bin = cost_bin(energies.get(candidate_id, 0.0))

        mates: list[tuple[str, float, float, int]] = []
        for organelle_id, per_cell in self.environment.organism_stats.items():
            if organelle_id == candidate_id:
                continue
            ema = per_cell.get(cell)
            if ema is None:
                continue
            roi = self.population.average_roi(organelle_id, limit=5)
            mbin = cost_bin(energies.get(organelle_id, 0.0))
            mates.append((organelle_id, float(ema), float(roi), mbin))
        # Prefer same-bin mates; fallback to global best if insufficient
        mates.sort(key=lambda item: (item[3] == cand_bin, item[1], item[2]), reverse=True)
        selected = mates[: max(0, soup_size - 1)]
        for organelle_id, ema, roi, _mbin in selected:
            stats_map[organelle_id] = {"roi": max(roi, 0.0), "ema": float(ema)}
        soup_ids = [candidate_id] + [entry[0] for entry in selected]
        return soup_ids, stats_map

    def _apply_lora_soup_merge(
        self,
        cell: GridKey,
        candidate_id: str,
        soup_ids: list[str],
        stats_map: dict[str, dict[str, float]],
        probe_records: list[dict[str, object]],
    ) -> list[dict[str, float]]:
        summary: list[dict[str, float]] = []
        weights: list[float] = []
        probe_boost = 1.0
        if probe_records:
            scores = [float(record.get("reward", 0.0)) for record in probe_records]
            positives = [score for score in scores if score > 0.0]
            if positives:
                probe_boost += sum(positives) / max(len(positives), 1)
        method = str(getattr(self.config.assimilation_tuning, "merge_method", "naive")).lower()
        # optional fisher-style importance weighting
        importances: dict[str, float] = {}
        if method == "fisher_svd":
            for oid in soup_ids:
                org = self.host.get_organelle(oid)
                imp = 1.0
                try:
                    adapter = getattr(org, "adapter", None)
                    if adapter is not None and hasattr(adapter, "lora_A") and hasattr(adapter, "lora_B"):
                        a = adapter.lora_A.detach()
                        b = adapter.lora_B.detach()
                        imp = float(a.norm().item() * b.norm().item() + 1e-6)
                except Exception:
                    imp = 1.0
                importances[oid] = imp
            # normalize importances
            s = sum(importances.values()) or 1.0
            importances = {k: (v / s) for k, v in importances.items()}
        for oid in soup_ids:
            stats = stats_map.get(oid, {"roi": 0.0, "ema": 0.0})
            roi = max(stats.get("roi", 0.0), 0.0)
            ema = stats.get("ema", 0.0)
            weight = (roi + 1e-3) * (ema + 1e-3)
            if method == "fisher_svd":
                weight *= importances.get(oid, 1.0)
            if oid == candidate_id:
                weight *= probe_boost
            summary.append({"organelle_id": oid, "weight": float(weight), "roi": float(roi), "ema": float(ema)})
            weights.append(weight)
        weight_sum = sum(weights) or 1.0
        soup_map = {oid: (weight / weight_sum) for oid, weight in zip(soup_ids, weights)}
        organelle = self.host.get_organelle(candidate_id)
        target_rank = int(getattr(organelle, "rank", self.config.host.max_lora_rank)) if organelle is not None else self.config.host.max_lora_rank
        target_rank = max(1, min(target_rank, self.config.host.max_lora_rank))
        self.host.merge_lora_soup(soup_map, target_rank)
        return summary

    def _run_hf_probes(self, organelle_id: str) -> list[dict[str, object]]:
        prompts = list(self.config.assimilation_tuning.hf_prompts)
        if not prompts or self.human_bandit is None:
            return []
        records: list[dict[str, object]] = []
        for prompt in prompts:
            result = self.host.step(
                prompt=prompt,
                intent="human probe",
                max_routes=1,
                allowed_organelle_ids=[organelle_id],
            )
            metrics = result.responses.get(organelle_id)
            if metrics is None:
                records.append(
                    {
                        "prompt": prompt,
                        "passed": False,
                        "reward": 0.0,
                        "notes": "no-metrics",
                    }
                )
                continue
            answer = metrics.answer
            feedback = self.human_bandit.solicit(prompt, answer)
            reward_total = feedback.reward.total
            records.append(
                {
                    "prompt": prompt,
                    "passed": reward_total > 0.0,
                    "reward": reward_total,
                    "notes": feedback.notes,
                }
            )
        return records


__all__ = ["EcologyLoop"]
