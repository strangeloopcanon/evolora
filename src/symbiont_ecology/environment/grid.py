"""Grid-based environment with local niches and controller."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from copy import deepcopy

from symbiont_ecology.config import (
    CanaryConfig,
    ControllerConfig,
    GridConfig,
    PricingConfig,
)
from symbiont_ecology.metrics.telemetry import RewardBreakdown

GridKey = Tuple[str, str]


@dataclass
class GridTask:
    task_id: str
    cell: GridKey
    prompt: str
    price: float
    target: Any
    family: str
    depth: str
    difficulty: float
    canary: bool = False
    reward_bonus: float = 0.0
    failure_cost_scale: float = 1.0

    def evaluate(self, answer: str) -> Tuple[bool, RewardBreakdown]:
        clean = answer.strip()
        success = False
        task_reward = 0.0

        if self.family == "math":
            try:
                predicted = float(clean.split()[0]) if " " in clean else float(clean)
            except ValueError:
                predicted = None
            success = predicted is not None and math.isclose(predicted, float(self.target), rel_tol=1e-3)
            task_reward = 1.0 if success else 0.0

        elif self.family == "string.sort":
            expected = "".join(sorted(str(self.target)))
            success = expected in clean.replace(" ", "")
            task_reward = 1.0 if success else 0.0

        elif self.family == "word.count":
            try:
                predicted = int("".join(ch for ch in clean if ch.isdigit()))
            except ValueError:
                predicted = None
            success = predicted == int(self.target)
            task_reward = 1.0 if success else 0.0

        elif self.family == "json_repair":
            try:
                parsed = json.loads(clean)
                success = isinstance(parsed, list) and parsed == self.target
            except json.JSONDecodeError:
                success = False
            task_reward = 1.0 if success else 0.0

        novelty = min(0.1, self.difficulty * 0.1)
        competence = 0.05 if success else 0.0
        helper = 0.0
        risk_penalty = 0.0 if success else 0.1 * max(self.failure_cost_scale, 0.0)
        if success:
            task_reward = 1.0 + max(self.reward_bonus, 0.0)
        else:
            task_reward = 0.0
        reward = RewardBreakdown(
            task_reward=task_reward,
            novelty_bonus=novelty,
            competence_bonus=competence,
            helper_bonus=helper,
            risk_penalty=risk_penalty,
            cost_penalty=0.05 * max(self.failure_cost_scale, 0.0) if not success else 0.0,
        )
        return success, reward


@dataclass
class GridCellState:
    difficulty: float = 0.5
    success_ema: float = 0.5
    price: float = 1.0
    canary_index: int = 0
    canaries: List[GridTask] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.canaries is None:
            self.canaries = []


class EnvironmentController:
    def __init__(
        self,
        grid_config: GridConfig,
        controller_cfg: ControllerConfig,
        pricing_cfg: PricingConfig,
        canary_cfg: CanaryConfig,
        seed: int = 1234,
    ) -> None:
        self.grid_config = grid_config
        self.ctrl = controller_cfg
        self.pricing = pricing_cfg
        self.canary_cfg = canary_cfg
        self.rng = random.Random(seed)
        self.cells: Dict[GridKey, GridCellState] = {}
        for family in grid_config.families:
            for depth in grid_config.depths:
                self.cells[(family, depth)] = GridCellState(price=pricing_cfg.base)

    def sample_cell(self) -> GridKey:
        keys = list(self.cells.keys())
        return self.rng.choice(keys)

    def update(self, cell: GridKey, success: bool) -> None:
        state = self.cells[cell]
        beta = self.ctrl.beta
        tau = self.ctrl.tau
        eta = self.ctrl.eta
        state.success_ema = (1 - beta) * state.success_ema + beta * (1.0 if success else 0.0)
        state.difficulty = min(0.95, max(0.05, state.difficulty + eta * (state.success_ema - tau)))
        state.price = min(
            self.pricing.max,
            max(self.pricing.min, self.pricing.base + self.pricing.k * (tau - state.success_ema)),
        )

    def get_state(self, cell: GridKey) -> GridCellState:
        return deepcopy(self.cells[cell])

    def queue_canary(self, cell: GridKey, task: GridTask) -> None:
        state = self.cells[cell]
        state.canaries.append(task)

    def next_canary(self, cell: GridKey) -> Optional[GridTask]:
        state = self.cells[cell]
        if not state.canaries:
            return None
        task = state.canaries[state.canary_index % len(state.canaries)]
        state.canary_index += 1
        return task

    def apply_parameters(
        self,
        *,
        tau: float | None = None,
        beta: float | None = None,
        eta: float | None = None,
        price_base: float | None = None,
        price_k: float | None = None,
    ) -> None:
        if tau is not None:
            self.ctrl.tau = tau
        if beta is not None:
            self.ctrl.beta = beta
        if eta is not None:
            self.ctrl.eta = eta
        if price_base is not None:
            self.pricing.base = price_base
        if price_k is not None:
            self.pricing.k = price_k
        if price_base is not None or price_k is not None:
            for state in self.cells.values():
                state.price = min(
                    self.pricing.max,
                    max(self.pricing.min, self.pricing.base + self.pricing.k * (self.ctrl.tau - state.success_ema)),
                )


class GridEnvironment:
    def __init__(
        self,
        grid_cfg: GridConfig,
        controller_cfg: ControllerConfig,
        pricing_cfg: PricingConfig,
        canary_cfg: CanaryConfig,
        seed: int = 2024,
        reward_bonus: float = 0.0,
        failure_cost_multiplier: float = 1.0,
    ) -> None:
        self.controller = EnvironmentController(grid_cfg, controller_cfg, pricing_cfg, canary_cfg, seed=seed)
        self.rng = random.Random(seed + 7)
        self.task_counter = 0
        self.organism_stats: Dict[str, Dict[GridKey, float]] = {}
        self.organism_canary_fail: Dict[str, bool] = {}
        self.beta = controller_cfg.beta
        self.tau = controller_cfg.tau
        self.canary_q_min = canary_cfg.q_min
        self.reward_bonus = max(0.0, reward_bonus)
        self.failure_cost_multiplier = max(0.0, min(failure_cost_multiplier, 1.0))
        self._bootstrap_canaries()

    def _bootstrap_canaries(self) -> None:
        for cell in self.controller.cells.keys():
            canaries = [self._make_task(cell, canary=True) for _ in range(5)]
            for task in canaries:
                self.controller.queue_canary(cell, task)

    def _task_id(self) -> str:
        self.task_counter += 1
        return f"task_{self.task_counter:06d}"

    def sample_task(self) -> GridTask:
        cell = self.controller.sample_cell()
        # 10% chance of canary when EMA high
        state = self.controller.get_state(cell)
        use_canary = state.success_ema > self.canary_q_min and self.rng.random() < 0.1
        if use_canary:
            canary = self.controller.next_canary(cell)
            if canary is not None:
                return canary
        return self._make_task(cell, canary=False)

    def sample_task_from_cell(self, cell: GridKey, *, canary: bool = False) -> GridTask:
        if canary:
            queued = self.controller.next_canary(cell)
            if queued is not None:
                return queued
        return self._make_task(cell, canary=canary)

    def catastrophic_shift(self, *, scale: float = 0.5, rng: random.Random | None = None) -> None:
        local_rng = rng or self.rng
        for cell, state in self.controller.cells.items():
            state.difficulty = max(0.05, min(0.95, local_rng.uniform(0.05, 0.95)))
            state.success_ema = max(0.05, min(0.95, self.controller.ctrl.tau * (1.0 - scale)))
            state.price = min(
                self.controller.pricing.max,
                max(
                    self.controller.pricing.min,
                    self.controller.pricing.base + self.controller.pricing.k * (self.controller.ctrl.tau - state.success_ema),
                ),
            )
        self.organism_stats.clear()
        self.organism_canary_fail.clear()

    def _make_task(self, cell: GridKey, canary: bool) -> GridTask:
        family, depth = cell
        state = self.controller.get_state(cell)
        difficulty = state.difficulty
        price = state.price
        task_id = self._task_id()

        if family == "math":
            # Map depth to operand range
            base = {"short": 10, "medium": 100, "long": 1000}.get(depth, 10)
            upper = max(2, int(base * (0.3 + difficulty)))
            a = self.rng.randint(1, upper)
            b = self.rng.randint(1, upper)
            if self.rng.random() < 0.5:
                prompt = f"Add {a} and {b}. Respond with the number only."
                target = a + b
            else:
                prompt = f"Multiply {a} by {b}. Respond with the number only."
                target = a * b
            return GridTask(
                task_id=task_id,
                cell=cell,
                prompt=prompt,
                price=price,
                target=float(target),
                family="math",
                depth=depth,
                difficulty=difficulty,
                canary=canary,
                reward_bonus=self.reward_bonus,
                failure_cost_scale=self.failure_cost_multiplier,
            )

        if family == "string.sort":
            alphabet = "abcdefghijklmnopqrstuvwxyz"
            length_map = {"short": 5, "medium": 8, "long": 12}
            length = length_map.get(depth, 5)
            letters = [self.rng.choice(alphabet) for _ in range(length)]
            prompt = (
                "Sort the following letters alphabetically and respond with the sorted string: "
                + " ".join(letters)
            )
            target = "".join(letters)
            return GridTask(
                task_id=task_id,
                cell=cell,
                prompt=prompt,
                price=price,
                target=target,
                family="string.sort",
                depth=depth,
                difficulty=difficulty,
                canary=canary,
                reward_bonus=self.reward_bonus,
                failure_cost_scale=self.failure_cost_multiplier,
            )

        if family == "word.count":
            base_sentences = {
                "short": [
                    "symbiotic agents cooperate",
                    "loRA adapters evolve rapidly",
                ],
                "medium": [
                    "autonomous organelles compete within the host",
                    "energy tickets enforce survival constraints",
                ],
                "long": [
                    "dynamic niches encourage diversity throughout the colony",
                    "assimilation requires consistent improvement across checkpoints",
                ],
            }
            sentence = self.rng.choice(base_sentences.get(depth, base_sentences["short"]))
            prompt = f"Count the number of words in the sentence: '{sentence}'. Respond with an integer."
            target = len(sentence.split())
            return GridTask(
                task_id=task_id,
                cell=cell,
                prompt=prompt,
                price=price,
                target=target,
                family="word.count",
                depth=depth,
                difficulty=difficulty,
                canary=canary,
                reward_bonus=self.reward_bonus,
                failure_cost_scale=self.failure_cost_multiplier,
            )

        if family == "json_repair":
            length_map = {"short": 4, "medium": 6, "long": 8}
            length = length_map.get(depth, 4)
            numbers = [self.rng.randint(0, int(50 + 100 * difficulty)) for _ in range(length)]
            shuffled = numbers[:]
            self.rng.shuffle(shuffled)
            prompt = (
                "Given the numbers "
                + ", ".join(str(n) for n in shuffled)
                + ", produce a valid JSON array containing them sorted ascending."
            )
            target = sorted(numbers)
            return GridTask(
                task_id=task_id,
                cell=cell,
                prompt=prompt,
                price=price,
                target=target,
                family="json_repair",
                depth=depth,
                difficulty=difficulty,
                canary=canary,
                reward_bonus=self.reward_bonus,
                failure_cost_scale=self.failure_cost_multiplier,
            )

        # fallback: math task
        return self._make_task(("math", depth), canary)

    # ------------------------------------------------------------------
    def register_result(self, organelle_id: str, task: GridTask, success: bool) -> None:
        self.controller.update(task.cell, success)
        stats = self.organism_stats.setdefault(organelle_id, {})
        ema = stats.get(task.cell, self.tau)
        stats[task.cell] = (1 - self.beta) * ema + self.beta * (1.0 if success else 0.0)
        if task.canary and not success:
            self.organism_canary_fail[organelle_id] = True
        elif task.canary and success:
            self.organism_canary_fail.pop(organelle_id, None)

    def best_cell_score(self, organelle_id: str) -> Optional[Tuple[GridKey, float]]:
        stats = self.organism_stats.get(organelle_id)
        if not stats:
            return None
        cell, value = max(stats.items(), key=lambda item: item[1])
        return cell, value

    def canary_failed(self, organelle_id: str) -> bool:
        return self.organism_canary_fail.get(organelle_id, False)

    def iter_cells(self) -> Iterable[GridKey]:
        return self.controller.cells.keys()

    def cell_state(self, cell: GridKey) -> GridCellState:
        return self.controller.get_state(cell)
