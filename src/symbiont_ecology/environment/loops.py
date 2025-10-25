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

    def run_generation(self, batch_size: int) -> None:
        self.generation_index += 1
        self.promotions_this_gen = 0
        self.trial_creations_this_gen = 0
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

        # Optional: communication read step once per active organelle
        comms_enabled = getattr(self.config.comms, "enabled", False)
        post_cost = float(getattr(self.config.comms, "post_cost", 0.2))
        read_cost = float(getattr(self.config.comms, "read_cost", 0.1))
        credit_frac = float(getattr(self.config.comms, "credit_frac", 0.2))
        if comms_enabled:
            for organelle_id in active:
                # attempt to read up to 2 messages and pay read cost per read
                messages = self.environment.read_messages(max_items=2)
                for msg in messages:
                    # charge reader if enough energy
                    bal = self.host.ledger.energy_balance(organelle_id)
                    if bal >= read_cost:
                        try:
                            self.host.ledger.consume_energy(organelle_id, read_cost)
                        except Exception:
                            pass
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

        for organelle_id in active:
            for _ in range(batch_size):
                lp_mix = getattr(self.config.curriculum, "lp_mix", 0.0)
                task = (
                    self._sample_task_lp(lp_mix)
                    if lp_mix > 0.0
                    else self.environment.sample_task()
                )
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
                self.population.record_score(organelle_id, reward.total)
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
                top_org = max(
                    active,
                    key=lambda oid: self.population.average_roi(oid, limit=5),
                )
                # Only post if enough energy
                if self.host.ledger.energy_balance(top_org) >= post_cost:
                    self.host.ledger.consume_energy(top_org, post_cost)
                    hint = "Hint: count words ignoring punctuation and double spaces."
                    self.environment.post_message(top_org, hint, cost=post_cost, ttl=int(getattr(self.config.comms, "ttl", 10)))
            except Exception:
                pass
        merges = self._attempt_assimilation(capped=self.config.evolution.max_merges_per_gen)
        # review trial offspring for potential promotion or cull
        self._review_trial_offspring()
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
            "population": len(self.population.population),
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
        self._auto_tune_assimilation_energy(summary)
        # Auto‑nudge evidence settings when assimilation stalls (no merges, low power)
        try:
            self._auto_nudge_evidence(summary)
        except Exception:
            pass
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
        if self.sink:
            self.sink.log_episode(episode)

    def _sample_task_lp(self, lp_mix: float) -> GridTask:
        try:
            cell = self.environment.controller.sample_cell(lp_mix=lp_mix)
            state = self.environment.controller.get_state(cell)
            use_canary = state.success_ema > self.environment.canary_q_min and self.environment.rng.random() < 0.1
            return self.environment.sample_task_from_cell(cell, canary=use_canary)
        except Exception:
            return self.environment.sample_task()

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
                    },
                )
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
            scores = self.population.recent_scores(genome.organelle_id, limit=16)
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
            split = window_len // 2
            control = window[:split]
            treatment = window[split:]
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
            )
            self.assimilation_cooldown[key] = self.generation_index
            # If statistical power is too low, defer and expand evidence window next time
            try:
                min_power = float(getattr(self.config.assimilation_tuning, "trial_min_power", 0.1))
            except Exception:
                min_power = 0.1
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
                continue
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
                "passes_stat_test": bool(result.decision),
            }
            if not result.decision:
                self.assim_fail_streak += 1
                self._maybe_decay_assimilation_thresholds()
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
                    attempt_detail["global_probe_passed"] = True
                else:
                    attempt_detail["global_probe_passed"] = False
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
        # dynamic easing: variance and fail streak make top-ups slightly easier to build evidence
        try:
            recent = self.population.roi.get(genome.organelle_id, [])[-8:]
            roi_std = pstdev([r for r in recent if math.isfinite(r)]) if len(recent) >= 2 else 0.0
        except Exception:
            roi_std = 0.0
        streak = max(0, int(getattr(self, "assim_fail_streak", 0)))
        dynamic_bonus = min(1.5, float(roi_bonus) + 0.15 * float(roi_std) + 0.02 * float(streak))
        effective_threshold = max(0.0, float(roi_threshold) - dynamic_bonus)
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
            info["status"] = "skip_low_roi"
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
        for oid in soup_ids:
            stats = stats_map.get(oid, {"roi": 0.0, "ema": 0.0})
            roi = max(stats.get("roi", 0.0), 0.0)
            ema = stats.get("ema", 0.0)
            weight = (roi + 1e-3) * (ema + 1e-3)
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
