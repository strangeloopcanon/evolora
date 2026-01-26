"""Evaluation manager for periodic holdout assessments."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from symbiont_ecology.config import EnergyConfig
from symbiont_ecology.environment.grid import GridEnvironment, GridTask
from symbiont_ecology.host.kernel import HostKernel
from symbiont_ecology.metrics.telemetry import RewardBreakdown


@dataclass
class EvaluationTask:
    prompt: str
    target: Any
    family: str
    depth: str

    def to_grid_task(self, environment: GridEnvironment, task_id: str) -> GridTask:
        family = self.family
        cell = (family, self.depth)
        # Backwards/forwards compatibility for renames like regex ↔ regex.synthesis.
        if cell not in environment.controller.cells:
            if (
                family == "regex"
                and ("regex.synthesis", self.depth) in environment.controller.cells
            ):
                family = "regex.synthesis"
                cell = (family, self.depth)
            elif (
                family == "regex.synthesis"
                and ("regex", self.depth) in environment.controller.cells
            ):
                family = "regex"
                cell = (family, self.depth)
        state = environment.controller.get_state(cell)
        return GridTask(
            task_id=task_id,
            cell=cell,
            prompt=self.prompt,
            price=state.price,
            target=self.target,
            family=family,
            depth=self.depth,
            difficulty=state.difficulty,
            canary=False,
            reward_bonus=getattr(environment, "reward_bonus", 0.0),
            failure_cost_scale=getattr(environment, "failure_cost_multiplier", 1.0),
        )


@dataclass
class EvaluationConfigRuntime:
    enabled: bool
    cadence: int
    tasks: List[EvaluationTask]
    sample_size: int
    reward_weight: float


class EvaluationManager:
    """Runs holdout evaluations and applies rewards."""

    def __init__(self, config: EvaluationConfigRuntime, seed: int = 9602) -> None:
        self.config = config
        self.rng = random.Random(seed)

    @staticmethod
    def from_file(
        enabled: bool,
        cadence: int,
        tasks_path: Path | None,
        sample_size: int | None,
        reward_weight: float,
    ) -> EvaluationConfigRuntime:
        tasks: List[EvaluationTask] = []
        if enabled:
            if tasks_path is None:
                raise ValueError("evaluation.tasks_path must be set when evaluation is enabled")
            with tasks_path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    tasks.append(
                        EvaluationTask(
                            prompt=data["prompt"],
                            target=data["target"],
                            family=data["family"],
                            depth=data["depth"],
                        )
                    )
            if not tasks:
                raise ValueError("evaluation tasks file is empty")
        if sample_size is None or sample_size <= 0:
            sample_size = len(tasks)
        return EvaluationConfigRuntime(
            enabled=enabled,
            cadence=cadence,
            tasks=tasks,
            sample_size=sample_size,
            reward_weight=reward_weight,
        )

    def sample_tasks(self) -> List[EvaluationTask]:
        if len(self.config.tasks) <= self.config.sample_size:
            return list(self.config.tasks)
        return self.rng.sample(self.config.tasks, self.config.sample_size)

    def evaluate(
        self,
        host: HostKernel,
        environment: GridEnvironment,
    ) -> Dict[str, Any]:
        organelle_ids: Sequence[str] = host.list_organelle_ids()
        if not organelle_ids:
            return {"accuracy": 0.0, "correct": 0, "total": 0}

        tasks = self.sample_tasks()
        correct = 0
        total = 0
        reward_weight = self.config.reward_weight
        energy_cfg: EnergyConfig = host.config.energy
        stats_cost = 0.0
        stats_roi = 0.0
        stats_delta = 0.0
        stats_count = 0
        roi_cap = 3.0
        family_stats: Dict[str, Dict[str, float]] = {}

        for idx, eval_task in enumerate(tasks, start=1):
            grid_task = eval_task.to_grid_task(environment, task_id=f"eval_{idx:04d}")
            result = host.step(
                prompt=grid_task.prompt,
                intent="evaluation",
                max_routes=len(organelle_ids),
            )
            answer = result.envelope.observation.state.get("answer", "")
            success, _task_reward = grid_task.evaluate(answer)
            total += 1
            if success:
                correct += 1
            fam_entry = family_stats.setdefault(
                eval_task.family,
                {
                    "correct": 0.0,
                    "total": 0.0,
                    "roi_sum": 0.0,
                    "delta_sum": 0.0,
                    "cost_sum": 0.0,
                    "count": 0.0,
                },
            )
            fam_entry["total"] += 1.0
            if success:
                fam_entry["correct"] += 1.0

            rewards: Dict[str, RewardBreakdown] = {}
            for route in result.routes:
                metrics = result.responses.get(route.organelle_id)
                if metrics is None:
                    continue
                base_reward = reward_weight if success else 0.0
                revenue = grid_task.price * base_reward
                cost = (
                    energy_cfg.alpha * metrics.flops_estimate
                    + energy_cfg.beta * metrics.memory_gb
                    + energy_cfg.gamma * metrics.latency_ms
                    + energy_cfg.lambda_p * metrics.trainable_params
                )
                if cost <= 0.0:
                    if revenue > 0:
                        roi_value = roi_cap
                    else:
                        roi_value = 0.0
                else:
                    roi_value = revenue / cost
                roi_factor = min(roi_value, roi_cap)
                task_reward = base_reward * roi_factor
                risk_penalty = 0.0 if success else reward_weight
                cost_penalty = max(0.0, cost - revenue)
                rewards[route.organelle_id] = RewardBreakdown(
                    task_reward=task_reward,
                    novelty_bonus=0.0,
                    competence_bonus=0.0,
                    helper_bonus=0.0,
                    risk_penalty=risk_penalty,
                    cost_penalty=cost_penalty,
                )
                energy_before = host.ledger.energy_balance(route.organelle_id)
                delta = revenue - cost
                energy_after = max(0.0, min(host.ledger.energy_cap, energy_before + delta))
                host.ledger.set_energy(route.organelle_id, energy_after)
                stats_cost += cost
                stats_roi += roi_value
                stats_delta += delta
                stats_count += 1
                fam_entry["cost_sum"] += cost
                fam_entry["roi_sum"] += roi_value
                fam_entry["delta_sum"] += delta
                fam_entry["count"] += 1.0
            if rewards:
                host.apply_reward(result.envelope, rewards)

        accuracy = (correct / total) if total else 0.0
        avg_cost = stats_cost / stats_count if stats_count else 0.0
        avg_roi = stats_roi / stats_count if stats_count else 0.0
        avg_delta = stats_delta / stats_count if stats_count else 0.0
        family_breakdown: Dict[str, Dict[str, float]] = {}
        for family, stats in family_stats.items():
            fam_total = max(1.0, stats["total"])
            fam_count = max(1.0, stats["count"] or stats["total"])
            family_breakdown[family] = {
                "accuracy": stats["correct"] / fam_total,
                "correct": int(stats["correct"]),
                "total": int(stats["total"]),
                "avg_cost": stats["cost_sum"] / fam_count,
                "avg_roi": stats["roi_sum"] / fam_count,
                "avg_delta": stats["delta_sum"] / fam_count,
                "count": int(stats["count"]) if stats["count"] else int(stats["total"]),
            }
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_cost": avg_cost,
            "avg_roi": avg_roi,
            "avg_delta": avg_delta,
            "evaluated_routes": stats_count,
            "family_breakdown": family_breakdown,
        }
