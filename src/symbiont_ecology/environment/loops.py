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

    def run_generation(self, batch_size: int) -> None:
        self.generation_index += 1
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

        for organelle_id in active:
            for _ in range(batch_size):
                task = self.environment.sample_task()
                result = self.host.step(
                    prompt=task.prompt,
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
        merges = self._attempt_assimilation(capped=self.config.evolution.max_merges_per_gen)
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
            "merges": merges,
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
        if hasattr(self, "assim_gating_counts"):
            summary["assimilation_gating"] = getattr(self, "assim_gating_counts")
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
        self._auto_tune_assimilation_energy(summary)
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

    def _attempt_assimilation(self, capped: int | None = None) -> int:
        removable: list[str] = []
        merges = 0
        gating: Dict[str, int] = {
            "canary_failed": 0,
            "low_energy": 0,
            "no_best_cell": 0,
            "cooldown": 0,
            "uplift_below_threshold": 0,
            "cell_merges_exceeded": 0,
            "insufficient_scores": 0,
            "global_probe_failed": 0,
            "holdout_failed": 0,
        }
        merges_per_cell: Dict[GridKey, int] = {}
        per_cell_interval = self.config.assimilation_tuning.per_cell_interval
        max_cell_merges = self.config.assimilation_tuning.max_merges_per_cell
        for genome in list(self.population.population.values()):
            if self.environment.canary_failed(genome.organelle_id):
                gating["canary_failed"] += 1
                continue
            balance = self.host.ledger.energy_balance(genome.organelle_id)
            balance = self._maybe_top_up_energy(genome, balance)
            if balance < self.config.energy.m:
                gating["low_energy"] += 1
                continue
            best_cell = self.environment.best_cell_score(genome.organelle_id)
            if best_cell is None:
                gating["no_best_cell"] += 1
                continue
            cell, ema = best_cell
            key = (genome.organelle_id, cell)
            last_attempt = self.assimilation_cooldown.get(key, -per_cell_interval)
            if self.generation_index - last_attempt < per_cell_interval:
                gating["cooldown"] += 1
                continue
            uplift_gate = ema - self.config.controller.tau
            if uplift_gate < self.config.evolution.assimilation_threshold:
                self.assimilation_cooldown[key] = self.generation_index
                gating["uplift_below_threshold"] += 1
                continue
            if merges_per_cell.get(cell, 0) >= max_cell_merges:
                gating["cell_merges_exceeded"] += 1
                continue
            scores = self.population.recent_scores(genome.organelle_id, limit=8)
            if len(scores) < 4:
                self.assimilation_cooldown[key] = self.generation_index
                gating["insufficient_scores"] += 1
                continue
            split = len(scores) // 2
            control = scores[:split]
            treatment = scores[split:]
            if len(control) < 2 or len(treatment) < 2:
                self.assimilation_cooldown[key] = self.generation_index
                gating["insufficient_scores"] += 1
                continue
            avg_energy = self.population.average_energy(genome.organelle_id)
            result = self.assimilation.evaluate(
                organelle_id=genome.organelle_id,
                control_scores=control,
                treatment_scores=treatment,
                safety_hits=0,
                energy_cost=avg_energy * len(treatment),
            )
            self.assimilation_cooldown[key] = self.generation_index
            probe_records: list[dict[str, object]] = []
            soup_records: list[dict[str, float]] = []
            holdout_info: dict[str, object] | None = None
            decision_final = False
            if result.decision and (capped is None or merges < capped):
                soup_ids, stats_map = self._select_soup_members(cell, genome.organelle_id)
                holdout_ok, holdout_info = self._holdout_accepts(
                    genome.organelle_id,
                    [oid for oid in soup_ids if oid != genome.organelle_id],
                )
                if not holdout_ok:
                    gating["holdout_failed"] += 1
                elif self._global_probe(genome.organelle_id, cell, gating):
                    probe_records = self._run_hf_probes(genome.organelle_id)
                    soup_records = self._apply_lora_soup_merge(
                        cell,
                        genome.organelle_id,
                        soup_ids,
                        stats_map,
                        probe_records,
                    )
                    self._spawn_replacement_from(genome)
                    removable.append(genome.organelle_id)
                    merges += 1
                    merges_per_cell[cell] = merges_per_cell.get(cell, 0) + 1
                    decision_final = True
            if self.sink:
                event = result.event.model_copy(
                    update={
                        "cell": {"family": cell[0], "depth": cell[1]},
                        "probes": probe_records,
                        "soup": soup_records,
                        "holdout": holdout_info,
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
                },
            )
        for organelle_id in removable:
            self.host.retire_organelle(organelle_id)
            self.population.remove(organelle_id)
        # expose gating counts for diagnostics in generation summary
        try:
            self.assim_gating_counts = gating  # type: ignore[attr-defined]
        except Exception:
            pass
        return merges

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

    def _maybe_top_up_energy(self, genome: Genome, balance: float) -> float:
        tuning = self.config.assimilation_tuning
        floor = getattr(tuning, "energy_floor", 0.0)
        if floor <= 0.0 or balance >= floor:
            return balance
        roi_threshold = getattr(tuning, "energy_floor_roi", 0.0)
        roi = self.population.average_roi(genome.organelle_id, limit=5)
        if roi < roi_threshold:
            return balance
        ledger = self.host.ledger
        available = max(0.0, min(floor - balance, ledger.energy_cap - balance))
        if available <= 0.0:
            return balance
        ledger.credit_energy(genome.organelle_id, available)
        return ledger.energy_balance(genome.organelle_id)

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
        tasks = self._sample_holdout_tasks()
        if not tasks:
            return True, None
        candidate_roi = self._evaluate_holdout_roi(candidate_id, tasks)
        baseline_rois = [self._evaluate_holdout_roi(mid, tasks) for mid in mate_ids]
        baseline_rois = [roi for roi in baseline_rois if math.isfinite(roi)]
        best_baseline = max(baseline_rois, default=float("-inf"))
        margin = self.config.assimilation_tuning.holdout_margin
        accepted = candidate_roi >= (best_baseline if math.isfinite(best_baseline) else 0.0) + margin
        info: dict[str, object] = {
            "candidate_roi": candidate_roi,
            "baseline_roi": best_baseline if math.isfinite(best_baseline) else None,
            "margin": margin,
            "tasks": len(tasks),
        }
        return accepted, info

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

        mates: list[tuple[str, float, float]] = []
        for organelle_id, per_cell in self.environment.organism_stats.items():
            if organelle_id == candidate_id:
                continue
            ema = per_cell.get(cell)
            if ema is None:
                continue
            roi = self.population.average_roi(organelle_id, limit=5)
            mates.append((organelle_id, float(ema), float(roi)))
        mates.sort(key=lambda item: (item[1], item[2]), reverse=True)
        selected = mates[: max(0, soup_size - 1)]
        for organelle_id, ema, roi in selected:
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
