"""Grid-based environment with local niches and controller."""

from __future__ import annotations

import json
import math
import random
import re
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

        if self.family in {"math", "math.sequence", "math.multi_step"}:
            # Be tolerant: extract the first numeric token anywhere in the string
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", clean)
            predicted = float(match.group(0)) if match else None
            success = predicted is not None and math.isclose(predicted, float(self.target), rel_tol=1e-3)
            task_reward = 1.0 if success else 0.0

        elif self.family == "string.sort":
            expected = "".join(sorted(str(self.target)))
            success = expected in clean.replace(" ", "")
            task_reward = 1.0 if success else 0.0

        elif self.family == "word.count":
            # Try digits first, then spelled numbers (zero..twenty)
            predicted: int | None = None
            digits = "".join(ch for ch in clean if ch.isdigit())
            if digits:
                try:
                    predicted = int(digits)
                except Exception:
                    predicted = None
            if predicted is None:
                tokens = clean.strip().lower().split()
                words_map = {
                    "zero": 0,
                    "one": 1,
                    "two": 2,
                    "three": 3,
                    "four": 4,
                    "five": 5,
                    "six": 6,
                    "seven": 7,
                    "eight": 8,
                    "nine": 9,
                    "ten": 10,
                    "eleven": 10 + 1,
                    "twelve": 10 + 2,
                    "thirteen": 10 + 3,
                    "fourteen": 10 + 4,
                    "fifteen": 10 + 5,
                    "sixteen": 10 + 6,
                    "seventeen": 10 + 7,
                    "eighteen": 10 + 8,
                    "nineteen": 10 + 9,
                    "twenty": 20,
                }
                for tok in tokens:
                    if tok in words_map:
                        predicted = words_map[tok]
                        break
            success = (predicted is not None) and (predicted == int(self.target))
            task_reward = 1.0 if success else 0.0

        elif self.family == "json_repair":
            # Try direct parse, else attempt to extract the first bracketed array
            parsed_ok = False
            try:
                parsed = json.loads(clean)
                parsed_ok = isinstance(parsed, list)
                success = parsed_ok and parsed == self.target
            except json.JSONDecodeError:
                success = False
            if not success:
                try:
                    start = clean.find("[")
                    end = clean.rfind("]")
                    if start != -1 and end != -1 and end > start:
                        candidate = clean[start : end + 1]
                        parsed = json.loads(candidate)
                        parsed_ok = isinstance(parsed, list)
                        success = parsed_ok and parsed == self.target
                except Exception:
                    success = False
            task_reward = 1.0 if success else 0.0

        elif self.family == "logic.bool":
            # Extract first boolean token ignoring markdown or prose
            text = clean.strip().lower()
            m = re.search(r"\b(true|false|yes|no)\b", text)
            if m:
                token = m.group(1)
                if token in {"true", "false"}:
                    predicted_bool = token == "true"
                else:
                    predicted_bool = token == "yes"
            else:
                predicted_bool = None
            success = predicted_bool is not None and predicted_bool == bool(self.target)
            task_reward = 1.0 if success else 0.0

        elif self.family == "code.format":
            normalized = clean.strip()
            if normalized.startswith("```"):
                # Strip code fences, keep last non-empty line
                stripped = normalized.strip("`")
                lines = [line.strip() for line in stripped.splitlines() if line.strip()]
                if lines and lines[0].lower().startswith("python"):
                    lines = lines[1:]
                normalized = lines[-1] if lines else ""
            normalized = normalized.strip("'\"")
            success = normalized == str(self.target)
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
        *,
        lp_alpha: float = 0.5,
        seed: int = 1234,
    ) -> None:
        self.grid_config = grid_config
        self.ctrl = controller_cfg
        self.pricing = pricing_cfg
        self.canary_cfg = canary_cfg
        self.rng = random.Random(seed)
        self.lp_alpha = max(0.0, min(1.0, lp_alpha))
        self.cells: Dict[GridKey, GridCellState] = {}
        self.bandit_counts: Dict[GridKey, int] = {}
        self.bandit_success: Dict[GridKey, float] = {}
        self.bandit_c = 1.2
        self.lp_progress: Dict[GridKey, float] = {}
        self.lp_prev_ema: Dict[GridKey, float] = {}
        for family in grid_config.families:
            for depth in grid_config.depths:
                key = (family, depth)
                self.cells[key] = GridCellState(price=pricing_cfg.base)
                self.bandit_counts[key] = 0
                self.bandit_success[key] = 0.0
                self.lp_progress[key] = 0.0
                self.lp_prev_ema[key] = 0.5

    def sample_cell(self, *, lp_mix: float = 0.0) -> GridKey:
        keys = list(self.cells.keys())
        if not keys:
            raise RuntimeError("No cells configured for sampling")
        if lp_mix > 0.0 and self.rng.random() < lp_mix:
            best_cell = max(keys, key=lambda k: self.lp_progress.get(k, 0.0))
            self.bandit_counts[best_cell] += 1
            return best_cell
        # Ensure each cell is explored at least once
        for key in keys:
            if self.bandit_counts.get(key, 0) == 0:
                self.bandit_counts[key] = 1
                return key
        # Occasional random exploration
        if self.rng.random() < 0.05:
            choice = self.rng.choice(keys)
            self.bandit_counts[choice] += 1
            return choice
        total_pulls = sum(self.bandit_counts.values())
        best_cell = max(keys, key=lambda key: self._bandit_score(key, total_pulls))
        self.bandit_counts[best_cell] += 1
        return best_cell

    def update(self, cell: GridKey, success: bool) -> None:
        state = self.cells[cell]
        beta = self.ctrl.beta
        tau = self.ctrl.tau
        eta = self.ctrl.eta
        prev = state.success_ema
        state.success_ema = (1 - beta) * prev + beta * (1.0 if success else 0.0)
        difficulty_delta = eta * (state.success_ema - tau)
        state.difficulty = min(0.85, max(0.15, state.difficulty + difficulty_delta))
        if not success:
            state.difficulty = max(0.15, state.difficulty - eta * 0.5)
        state.price = min(
            self.pricing.max,
            max(self.pricing.min, self.pricing.base + self.pricing.k * (tau - state.success_ema)),
        )
        self.bandit_success[cell] = self.bandit_success.get(cell, 0.0) + (1.0 if success else 0.0)
        # learning progress EMA
        delta = abs(state.success_ema - self.lp_prev_ema.get(cell, prev))
        self.lp_prev_ema[cell] = state.success_ema
        lp_alpha = self.lp_alpha
        self.lp_progress[cell] = (1 - lp_alpha) * self.lp_progress.get(cell, 0.0) + lp_alpha * delta

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

    def _bandit_score(self, cell: GridKey, total_pulls: int) -> float:
        pulls = max(1, self.bandit_counts.get(cell, 0))
        reward_total = self.bandit_success.get(cell, 0.0)
        empirical_success = reward_total / pulls
        price = self.cells[cell].price
        roi_estimate = price * empirical_success
        exploration_bonus = self.bandit_c * math.sqrt(math.log(total_pulls + 1) / pulls)
        diversity_bonus = max(0.0, 1.0 - self.cells[cell].success_ema) * 0.1
        return roi_estimate + exploration_bonus + diversity_bonus


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
        lp_alpha: float = 0.5,
    ) -> None:
        self.controller = EnvironmentController(
            grid_cfg,
            controller_cfg,
            pricing_cfg,
            canary_cfg,
            lp_alpha=lp_alpha,
            seed=seed,
        )
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
        # simple message board (off by default unless enabled in config via loop)
        self.message_board: list[dict[str, object]] = []
        self._message_seq: int = 0
        # simple latent cache bus for C2C comms
        self.cache_bus: list[dict[str, object]] = []

    def post_message(
        self,
        organelle_id: str,
        text: str,
        *,
        cost: float = 0.2,
        ttl: int = 10,
        priority: float | None = None,
        topic: str | None = None,
        cell: tuple[str, str] | None = None,
        meta: dict[str, object] | None = None,
    ) -> bool:
        ttl_int = max(int(ttl), 0)
        if ttl_int <= 0:
            return False
        default_priority = getattr(self, "default_comm_priority", 0.0)
        history_cap = max(1, int(getattr(self, "comms_history_cap", 64)))
        try:
            self._message_seq += 1
            entry: dict[str, object] = {
                "id": self._message_seq,
                "organelle_id": str(organelle_id),
                "text": str(text),
                "ttl": ttl_int,
                "priority": float(priority) if priority is not None else float(default_priority),
                "topic": str(topic) if topic else None,
                "cell": list(cell) if cell else None,
                "meta": dict(meta) if meta else {},
                "reads": 0,
                "posted_at": self._message_seq,
                "seen_by": set(),
            }
            self.message_board.append(entry)
            if len(self.message_board) > history_cap:
                overflow = len(self.message_board) - history_cap
                del self.message_board[0:overflow]
            return True
        except Exception:
            return False

    def read_messages(
        self,
        max_items: int = 3,
        *,
        topics: list[str] | None = None,
        exclude: set[str] | None = None,
        reader: str | None = None,
    ) -> list[dict[str, object]]:
        if max_items <= 0:
            return []
        topics_set = {topic for topic in (topics or []) if topic}
        exclude_set = {str(item) for item in (exclude or set())}
        reader_id = str(reader) if reader is not None else None
        ordered = sorted(
            self.message_board,
            key=lambda entry: (float(entry.get("priority", 0.0)), int(entry.get("posted_at", 0))),
            reverse=True,
        )
        primary: list[dict[str, object]] = []
        secondary: list[dict[str, object]] = []
        for entry in ordered:
            ttl = int(entry.get("ttl", 0))
            if ttl <= 0:
                continue
            poster = str(entry.get("organelle_id"))
            if poster in exclude_set:
                continue
            seen_by: set[str] = entry.get("seen_by", set())  # type: ignore[assignment]
            if reader_id and reader_id in seen_by:
                continue
            topic = entry.get("topic")
            target_list = secondary
            if topics_set and topic in topics_set:
                target_list = primary
            elif not topics_set:
                target_list = primary
            target_list.append(entry)
        combined = primary + secondary
        results: list[dict[str, object]] = []
        for entry in combined:
            if len(results) >= max_items:
                break
            ttl = int(entry.get("ttl", 0))
            if ttl <= 0:
                continue
            entry["ttl"] = ttl - 1
            entry["reads"] = int(entry.get("reads", 0)) + 1
            seen_by = entry.get("seen_by", set())
            if isinstance(seen_by, set) and reader_id:
                seen_by.add(reader_id)
            cleaned = {
                "organelle_id": str(entry.get("organelle_id")),
                "text": str(entry.get("text")),
                "topic": entry.get("topic"),
                "priority": float(entry.get("priority", 0.0)),
                "cell": entry.get("cell"),
                "meta": dict(entry.get("meta", {})),
                "ttl": int(entry.get("ttl", 0)),
                "reads": int(entry.get("reads", 0)),
            }
            results.append(cleaned)
        # purge expired messages
        self.message_board = [entry for entry in self.message_board if int(entry.get("ttl", 0)) > 0]
        return results

    def peek_messages(self, limit: int = 5) -> list[dict[str, object]]:
        ordered = sorted(
            self.message_board,
            key=lambda entry: (float(entry.get("priority", 0.0)), int(entry.get("posted_at", 0))),
            reverse=True,
        )
        snapshot: list[dict[str, object]] = []
        for entry in ordered[: max(1, limit)]:
            snapshot.append(
                {
                    "organelle_id": str(entry.get("organelle_id")),
                    "text": str(entry.get("text")),
                    "topic": entry.get("topic"),
                    "priority": float(entry.get("priority", 0.0)),
                    "ttl": int(entry.get("ttl", 0)),
                    "reads": int(entry.get("reads", 0)),
                    "cell": entry.get("cell"),
                }
            )
        return snapshot

    # C2C latent cache bus -------------------------------------------------
    def post_cache(self, organelle_id: str, latent: list[float], *, ttl: int = 5) -> bool:
        try:
            entry = {"organelle_id": organelle_id, "latent": list(latent), "ttl": int(ttl)}
            self.cache_bus.append(entry)
            return True
        except Exception:
            return False

    def read_caches(self, max_items: int = 2) -> list[dict[str, object]]:
        cleaned: list[dict[str, object]] = []
        for entry in list(self.cache_bus):
            ttl = int(entry.get("ttl", 0))
            if ttl <= 0:
                self.cache_bus.remove(entry)
                continue
            entry["ttl"] = ttl - 1
            cleaned.append({
                "organelle_id": str(entry.get("organelle_id")),
                "latent": list(entry.get("latent", [])),
            })
            if len(cleaned) >= max_items:
                break
        return cleaned

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
            prompt_text, target_count = self._make_word_count_task(depth)
            return GridTask(
                task_id=task_id,
                cell=cell,
                prompt=prompt_text,
                price=price,
                target=int(target_count),
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

        if family == "logic.bool":
            literal_count = {"short": 2, "medium": 3, "long": 4}.get(depth, 3)
            values = [self.rng.choice([True, False]) for _ in range(literal_count)]
            operators = [self.rng.choice(["and", "or"]) for _ in range(max(literal_count - 1, 0))]
            parts: List[str] = []
            result_value: Optional[bool] = None
            for idx, value in enumerate(values):
                literal = "TRUE" if value else "FALSE"
                actual = value
                if self.rng.random() < 0.3:
                    literal = f"NOT {literal}"
                    actual = not value
                if idx == 0:
                    parts.append(literal)
                    result_value = actual
                    continue
                op = operators[idx - 1]
                parts.append(op.upper())
                parts.append(literal)
                if op == "and":
                    result_value = bool(result_value and actual)
                else:
                    result_value = bool(result_value or actual)
            expression = " ".join(parts)
            prompt = (
                "Evaluate the logical expression and respond with 'True' or 'False': "
                f"{expression}"
            )
            return GridTask(
                task_id=task_id,
                cell=cell,
                prompt=prompt,
                price=price,
                target=bool(result_value),
                family="logic.bool",
                depth=depth,
                difficulty=difficulty,
                canary=canary,
                reward_bonus=self.reward_bonus,
                failure_cost_scale=self.failure_cost_multiplier,
            )

        if family == "math.sequence":
            length_map = {"short": 3, "medium": 4, "long": 5}
            length = length_map.get(depth, 4)
            pattern = self.rng.choice(["arithmetic", "geometric"])
            base_start = max(2, int(2 + difficulty * 10))
            start = self.rng.randint(2, base_start)
            if pattern == "arithmetic":
                step = self.rng.randint(1, max(2, int(3 + difficulty * 8)))
                sequence = [start + i * step for i in range(length)]
                next_value = start + length * step
            else:
                ratio_candidates = [2, 3]
                ratio = self.rng.choice(ratio_candidates)
                sequence = [start * (ratio**i) for i in range(length)]
                next_value = start * (ratio**length)
            prompt = (
                "Given the sequence "
                + ", ".join(str(n) for n in sequence)
                + ", what is the next number? Respond with the number only."
            )
            return GridTask(
                task_id=task_id,
                cell=cell,
                prompt=prompt,
                price=price,
                target=float(next_value),
                family="math.sequence",
                depth=depth,
                difficulty=difficulty,
                canary=canary,
                reward_bonus=self.reward_bonus,
                failure_cost_scale=self.failure_cost_multiplier,
            )

        if family == "math.multi_step":
            prompt_multi, target_multi = self._make_multi_step_math(depth)
            return GridTask(
                task_id=task_id,
                cell=cell,
                prompt=prompt_multi,
                price=price,
                target=float(target_multi),
                family="math.multi_step",
                depth=depth,
                difficulty=difficulty,
                canary=canary,
                reward_bonus=self.reward_bonus,
                failure_cost_scale=self.failure_cost_multiplier,
            )

        if family == "code.format":
            prompt_code, target_code = self._make_code_format_task(depth)
            return GridTask(
                task_id=task_id,
                cell=cell,
                prompt=prompt_code,
                price=price,
                target=target_code,
                family="code.format",
                depth=depth,
                difficulty=difficulty,
                canary=canary,
                reward_bonus=self.reward_bonus,
                failure_cost_scale=self.failure_cost_multiplier,
            )

        # fallback: math task
        return self._make_task(("math", depth), canary)

    def _count_alpha_words(self, text: str) -> int:
        return len(re.findall(r"[A-Za-z]+", text))

    def _make_word_count_task(self, depth: str) -> tuple[str, int]:
        base_sentences = {
            "short": [
                "Symbiotic agents cooperate",
                "LoRA adapters evolve rapidly",
                "Energy tickets enforce scarcity",
            ],
            "medium": [
                "Autonomous organelles coordinate under fluctuating budgets",
                "Bankruptcy culls reset the colony when energy collapses",
                "Meta mutations jitter the controller and ticket price together",
            ],
            "long": [
                "Dynamic niches encourage exploration throughout the Cambrian soup colony",
                "Assimilation requires consistent improvement across checkpoints and probes",
                "Telemetry snapshots surface ROI variance, energy gini, and uplift deltas per cell",
            ],
        }
        if self.rng.random() < 0.45:
            sentence = self.rng.choice(base_sentences.get(depth, base_sentences["short"]))
            prompt = f"Count the number of words in the sentence: '{sentence}'. Respond with an integer."
            return prompt, self._count_alpha_words(sentence)

        mode = self.rng.choice(["html", "punctuation", "digits"])
        word_pool = [
            "adaptive",
            "agents",
            "align",
            "biosphere",
            "clusters",
            "coordinate",
            "feedback",
            "gradients",
            "harvest",
            "iterate",
            "learning",
            "morphology",
            "organelles",
            "pipeline",
            "quanta",
            "receptors",
            "signals",
            "symbiotic",
            "tensors",
            "uplift",
            "vectors",
            "workflow",
            "yield",
            "zenith",
        ]

        if mode == "html":
            count = {"short": 3, "medium": 4, "long": 5}.get(depth, 3)
            chosen = self.rng.sample(word_pool, k=count)
            decorated = []
            for idx, word in enumerate(chosen):
                if idx == 0:
                    decorated.append(word)
                elif idx == len(chosen) - 1:
                    decorated.append(f"<em>{word}</em>")
                else:
                    decorated.append(f"<strong>{word}</strong>")
            snippet = "<div>" + " ".join(decorated) + "</div>"
            prompt = (
                "Count the number of words (ignore HTML tags) in the snippet: "
                f"`{snippet}` Respond with an integer."
            )
            return prompt, len(chosen)

        if mode == "punctuation":
            length = {"short": 5, "medium": 9, "long": 14}.get(depth, 9)
            punctuators = ["", ",", ";", ":", "!", "?"]
            tokens: List[str] = []
            for _ in range(length):
                token = self.rng.choice(word_pool)
                punct = self.rng.choice(punctuators)
                if punct:
                    token = f"{token}{punct}"
                if self.rng.random() < 0.15:
                    token = token.upper()
                tokens.append(token)
            sentence = " ".join(tokens)
            sentence = re.sub(r"\s+", " ", sentence).strip()
            prompt = (
                "Count the number of words in the sentence (ignore punctuation): "
                f"'{sentence}'. Respond with an integer."
            )
            return prompt, self._count_alpha_words(sentence)

        templates = [
            "Stage 3 executes 2 batches before the final freeze window",
            "Pipeline run 7 completes after 4 retries and 1 fallback",
            "Phase 2 allocates 5 tickets to tier three modules",
        ]
        sentence = self.rng.choice(templates)
        prompt = (
            "Count the number of alphabetic words (ignore digits) in the sentence: "
            f"'{sentence}'. Respond with an integer."
        )
        return prompt, self._count_alpha_words(sentence)

    def _make_multi_step_math(self, depth: str) -> tuple[str, float]:
        a = self.rng.randint(3, 12)
        b = self.rng.randint(2, 9)
        c = self.rng.randint(2, 6)
        d = self.rng.randint(1, 6)
        e = self.rng.randint(1, 5)

        if depth == "short":
            expression = f"{a} + {b} * {c}"
            target = a + b * c
        elif depth == "medium":
            expression = f"{a} + {b} * {c} - {d}"
            target = a + b * c - d
        else:
            expression = f"({a} + {b}) * {c} - ({d} + {e})"
            target = (a + b) * c - (d + e)
        prompt = f"Compute {expression}. Respond with the number only."
        return prompt, target

    def _make_code_format_task(self, depth: str) -> tuple[str, str]:
        parts_pool = [
            "token",
            "adapter",
            "manager",
            "gamma",
            "delta",
            "energy",
            "reserve",
            "budget",
            "router",
            "policy",
            "reward",
            "merge",
            "audit",
            "canvas",
        ]
        length_map = {"short": 2, "medium": 3, "long": 4}
        parts = self.rng.sample(parts_pool, k=length_map.get(depth, 3))
        camel = "".join(part.title() for part in parts)
        if self.rng.random() < 0.25:
            camel = camel + "HTTP"
            parts.append("http")
        snake = "_".join(part.lower() for part in parts)
        prompt = (
            "Convert the variable name `{}` to snake_case. Respond with the new name only."
        ).format(camel)
        return prompt, snake

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
            try:
                baseline = float(getattr(self.controller.ctrl, "tau", 0.5))
            except Exception:
                baseline = 0.5
            seeded = {cell: baseline for cell in self.controller.cells.keys()}
            if seeded:
                self.organism_stats[organelle_id] = seeded
                stats = seeded
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
