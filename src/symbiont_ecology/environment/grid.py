"""Grid-based environment with local niches and controller."""

from __future__ import annotations

import json
import math
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from symbiont_ecology.config import (
    CanaryConfig,
    ControllerConfig,
    GridConfig,
    PricingConfig,
)
from symbiont_ecology.metrics.telemetry import RewardBreakdown
from symbiont_ecology.utils.regex_extract import pick_best_regex_candidate

GridKey = Tuple[str, str]


def _regex_complexity_score(pattern: str) -> float:
    """Compute a lightweight regex complexity score (lower is simpler).

    Kept local to avoid circular imports between environment and evaluation modules.
    """
    char_length = len(pattern)
    alternation_count = pattern.count("|")
    group_count = len(re.findall(r"\((?!\?:)", pattern))
    group_count += len(re.findall(r"\(\?:", pattern))
    quantifier_count = len(re.findall(r"[+*?]|\{\d+(?:,\d*)?\}", pattern))
    node_patterns = [
        r"\[(?:\^)?[^\]]+\]",
        r"\\[dDwWsS]",
        r"\\.",
        r"[+*?]",
        r"\{\d+(?:,\d*)?\}",
        r"\|",
        r"\((?:\?:)?",
        r"\)",
        r"\^|\$",
        r"[^\\[\]{}()+*?|^$]",
    ]
    ast_node_count = sum(len(re.findall(p, pattern)) for p in node_patterns)
    max_depth = 0
    current_depth = 0
    for char in pattern:
        if char == "(":
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ")":
            current_depth = max(0, current_depth - 1)
    has_backtracking_risk = bool(
        re.search(r"\.\*.*\.\*", pattern)
        or re.search(r"\([^)]*[+*][^)]*\)[+*]", pattern)
        or re.search(r"(?<!\?)\*\*", pattern)
    )
    return float(
        char_length * 0.1
        + ast_node_count * 1.0
        + max_depth * 2.0
        + alternation_count * 1.5
        + group_count * 0.5
        + quantifier_count * 0.3
        + (5.0 if has_backtracking_risk else 0.0)
    )


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
            success = predicted is not None and math.isclose(
                predicted, float(self.target), rel_tol=1e-3
            )
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

        elif self.family in {"regex", "regex.debugging", "regex.synthesis"}:
            # The target is a dict containing the correct pattern and test strings
            if isinstance(self.target, dict):
                test_strings = self.target.get("test_strings", [])
            else:
                test_strings = []

            picked, details = pick_best_regex_candidate(clean, test_cases=test_strings)
            if (
                picked
                and isinstance(details.get("passed"), int)
                and isinstance(details.get("total"), int)
            ):
                passed = int(details["passed"])
                total = int(details["total"])
                if total > 0:
                    task_reward = float(passed) / float(total)
                    success = passed >= total
                else:
                    task_reward = 0.0
                    success = False
            elif picked:
                try:
                    re.compile(picked)
                    success = True
                except re.error:
                    success = False
            else:
                success = False

            if task_reward <= 0.0:
                task_reward = 1.0 if success else 0.0

        elif self.family == "regex.recognition":
            expected_bool: bool | None = None
            if isinstance(self.target, dict):
                expected = self.target.get("expected")
                if isinstance(expected, bool):
                    expected_bool = expected
                elif expected is not None:
                    expected_bool = bool(expected)
            elif isinstance(self.target, bool):
                expected_bool = self.target
            else:
                expected_bool = None

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

            success = (
                predicted_bool is not None
                and expected_bool is not None
                and predicted_bool == expected_bool
            )
            task_reward = 1.0 if success else 0.0

        elif self.family in {"regex.explanation", "regex.mutation_effect"}:
            keywords: list[str] = []
            if isinstance(self.target, dict):
                raw = self.target.get("required_keywords") or self.target.get("keywords") or []
                if isinstance(raw, list):
                    keywords = [str(kw) for kw in raw if kw]
            elif isinstance(self.target, list):
                keywords = [str(kw) for kw in self.target if kw]

            response_lower = clean.lower()
            found_keywords = [kw for kw in keywords if kw.lower() in response_lower]
            score = len(found_keywords) / len(keywords) if keywords else 1.0
            success = score >= 0.7
            task_reward = float(score)

        elif self.family == "regex.refactoring":
            test_strings: list[dict[str, object]] = []
            original_pattern = ""
            if isinstance(self.target, dict):
                test_strings = self.target.get("test_strings", []) or []
                original_pattern = str(self.target.get("original_pattern") or "")
            picked, _pick_details = pick_best_regex_candidate(clean, test_cases=test_strings)
            if not picked:
                success = False
                task_reward = 0.0
            else:
                try:
                    compiled = re.compile(picked)
                except re.error:
                    compiled = None
                if compiled is None:
                    success = False
                    task_reward = 0.0
                else:
                    passed = 0
                    total = 0
                    ok = True
                    for tc in test_strings:
                        test_str = str(tc.get("string", ""))
                        should_match = bool(tc.get("should_match", False))
                        matched = bool(compiled.fullmatch(test_str))
                        if matched == should_match:
                            passed += 1
                        else:
                            ok = False
                        total += 1
                    if total <= 0:
                        success = False
                        task_reward = 0.0
                    else:
                        case_ratio = float(passed) / float(total)
                        if original_pattern:
                            try:
                                complexity_ok = _regex_complexity_score(
                                    picked
                                ) <= _regex_complexity_score(original_pattern)
                            except Exception:
                                complexity_ok = True
                        else:
                            complexity_ok = True
                        success = bool(ok and complexity_ok)
                        if case_ratio < 1.0:
                            task_reward = case_ratio * 0.5
                        else:
                            task_reward = 1.0 if complexity_ok else 0.5

        novelty = min(0.1, self.difficulty * 0.1)
        helper = 0.0
        if success:
            task_reward = float(task_reward) + max(self.reward_bonus, 0.0)
        progress = max(0.0, min(1.0, float(task_reward)))
        competence = 0.05 * progress
        risk_penalty = (1.0 - progress) * 0.1 * max(self.failure_cost_scale, 0.0)
        cost_penalty = (1.0 - progress) * 0.05 * max(self.failure_cost_scale, 0.0)
        reward = RewardBreakdown(
            task_reward=task_reward,
            novelty_bonus=novelty,
            competence_bonus=competence,
            helper_bonus=helper,
            risk_penalty=risk_penalty,
            cost_penalty=cost_penalty,
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
                    max(
                        self.pricing.min,
                        self.pricing.base + self.pricing.k * (self.ctrl.tau - state.success_ema),
                    ),
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
            cleaned.append(
                {
                    "organelle_id": str(entry.get("organelle_id")),
                    "latent": list(entry.get("latent", [])),
                }
            )
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
        for _cell, state in self.controller.cells.items():
            state.difficulty = max(0.05, min(0.95, local_rng.uniform(0.05, 0.95)))
            state.success_ema = max(0.05, min(0.95, self.controller.ctrl.tau * (1.0 - scale)))
            state.price = min(
                self.controller.pricing.max,
                max(
                    self.controller.pricing.min,
                    self.controller.pricing.base
                    + self.controller.pricing.k * (self.controller.ctrl.tau - state.success_ema),
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
                f"Evaluate the logical expression and respond with 'True' or 'False': {expression}"
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

        if family == "regex":
            prompt_regex, target_pattern, test_strings = self._make_regex_task(depth)
            # Store both pattern and test strings in target for evaluation
            target_dict = {
                "pattern": target_pattern,
                "test_strings": test_strings,
            }
            return GridTask(
                task_id=task_id,
                cell=cell,
                prompt=prompt_regex,
                price=price,
                target=target_dict,
                family="regex",
                depth=depth,
                difficulty=difficulty,
                canary=canary,
                reward_bonus=self.reward_bonus,
                failure_cost_scale=self.failure_cost_multiplier,
            )

        if family == "regex.synthesis":
            prompt_regex, target_pattern, test_strings = self._make_regex_task(depth)
            target_dict = {"pattern": target_pattern, "test_strings": test_strings}
            return GridTask(
                task_id=task_id,
                cell=cell,
                prompt=prompt_regex,
                price=price,
                target=target_dict,
                family="regex.synthesis",
                depth=depth,
                difficulty=difficulty,
                canary=canary,
                reward_bonus=self.reward_bonus,
                failure_cost_scale=self.failure_cost_multiplier,
            )

        if family == "regex.recognition":
            prompt_recog, recog_target = self._make_regex_recognition_task(depth)
            return GridTask(
                task_id=task_id,
                cell=cell,
                prompt=prompt_recog,
                price=price,
                target=recog_target,
                family="regex.recognition",
                depth=depth,
                difficulty=difficulty,
                canary=canary,
                reward_bonus=self.reward_bonus,
                failure_cost_scale=self.failure_cost_multiplier,
            )

        if family == "regex.explanation":
            prompt_explain, explain_target = self._make_regex_explanation_task(depth)
            return GridTask(
                task_id=task_id,
                cell=cell,
                prompt=prompt_explain,
                price=price,
                target=explain_target,
                family="regex.explanation",
                depth=depth,
                difficulty=difficulty,
                canary=canary,
                reward_bonus=self.reward_bonus,
                failure_cost_scale=self.failure_cost_multiplier,
            )

        if family == "regex.debugging":
            prompt_debug, debug_target = self._make_regex_debugging_task(depth)
            return GridTask(
                task_id=task_id,
                cell=cell,
                prompt=prompt_debug,
                price=price,
                target=debug_target,
                family="regex.debugging",
                depth=depth,
                difficulty=difficulty,
                canary=canary,
                reward_bonus=self.reward_bonus,
                failure_cost_scale=self.failure_cost_multiplier,
            )

        if family == "regex.refactoring":
            prompt_refactor, refactor_target = self._make_regex_refactoring_task(depth)
            return GridTask(
                task_id=task_id,
                cell=cell,
                prompt=prompt_refactor,
                price=price,
                target=refactor_target,
                family="regex.refactoring",
                depth=depth,
                difficulty=difficulty,
                canary=canary,
                reward_bonus=self.reward_bonus,
                failure_cost_scale=self.failure_cost_multiplier,
            )

        if family == "regex.mutation_effect":
            prompt_mut, mut_target = self._make_regex_mutation_effect_task(depth)
            return GridTask(
                task_id=task_id,
                cell=cell,
                prompt=prompt_mut,
                price=price,
                target=mut_target,
                family="regex.mutation_effect",
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
            prompt = (
                f"Count the number of words in the sentence: '{sentence}'. Respond with an integer."
            )
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

    def _make_regex_task(self, depth: str) -> tuple[str, str, list[dict[str, object]]]:
        """Generate a regex pattern task with matching and non-matching examples."""
        rng = self.rng
        alphabet = "abcdefghijklmnopqrstuvwxyz"

        def random_word(min_len: int = 3, max_len: int = 8) -> str:
            length = rng.randint(int(min_len), int(max_len))
            return "".join(rng.choice(alphabet) for _ in range(max(1, length)))

        def random_identifier(
            min_len: int = 3,
            max_len: int = 12,
            *,
            allow_leading_underscore: bool = True,
        ) -> str:
            first_pool = alphabet + alphabet.upper()
            if allow_leading_underscore:
                first_pool += "_"
            first = rng.choice(first_pool)
            rest_chars = alphabet + alphabet.upper() + "0123456789_"
            length = rng.randint(max(1, int(min_len)), max(1, int(max_len)))
            rest = "".join(rng.choice(rest_chars) for _ in range(max(0, length - 1)))
            return first + rest

        def random_domain(tlds: list[str] | None = None) -> str:
            host = random_word(4, 10)
            tld = rng.choice(list(tlds) if tlds else ["com", "org", "net", "io", "dev"])
            if rng.random() < 0.25:
                return f"{random_word(3, 6)}.{host}.{tld}"
            return f"{host}.{tld}"

        def _mmdd(value: int) -> str:
            return f"{int(value):02d}"

        def _yyyy(value: int) -> str:
            return f"{int(value):04d}"

        def task_day_01_31() -> dict[str, object]:
            pattern = r"^(0[1-9]|[12]\d|3[01])$"
            matches_set: set[str] = set()
            while len(matches_set) < 4:
                matches_set.add(_mmdd(rng.randint(1, 31)))
            matches = sorted(matches_set)
            non_matches = [
                "00",
                "32",
                str(rng.randint(1, 9)),
                str(rng.randint(32, 99)),
            ]
            return {
                "desc": "match day numbers 01-31 where days 1-9 must have leading zero",
                "pattern": pattern,
                "matches": matches,
                "non_matches": non_matches,
            }

        def task_date_ymd() -> dict[str, object]:
            pattern = r"^(19\d{2}|20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"
            matches_set: set[str] = set()
            while len(matches_set) < 3:
                y = _yyyy(rng.randint(1900, 2099))
                m = _mmdd(rng.randint(1, 12))
                d = _mmdd(rng.randint(1, 31))
                matches_set.add(f"{y}-{m}-{d}")
            matches = sorted(matches_set)

            y_bad = _yyyy(rng.choice([1899, 2100]))
            y_ok = _yyyy(rng.randint(1900, 2099))
            m_bad = _mmdd(rng.choice([0, 13]))
            d_bad = _mmdd(rng.choice([0, 32]))
            m_ok_int = rng.randint(1, 9)
            d_ok_int = rng.randint(1, 9)
            non_matches = [
                f"{y_bad}-{_mmdd(rng.randint(1, 12))}-{_mmdd(rng.randint(1, 31))}",
                f"{y_ok}-{m_bad}-{_mmdd(rng.randint(1, 31))}",
                f"{y_ok}-{_mmdd(rng.randint(1, 12))}-{d_bad}",
                f"{y_ok}/{_mmdd(rng.randint(1, 12))}/{_mmdd(rng.randint(1, 31))}",
                f"{y_ok}-{m_ok_int}-{d_ok_int}",
            ]
            return {
                "desc": "match dates in YYYY-MM-DD format (years 1900-2099, months 01-12, days 01-31) with no extra characters",
                "pattern": pattern,
                "matches": matches,
                "non_matches": non_matches,
            }

        def task_date_dot() -> dict[str, object]:
            pattern = r"^(20\d{2})\.(0[1-9]|1[0-2])\.(0[1-9]|[12]\d|3[01])$"
            matches_set: set[str] = set()
            while len(matches_set) < 2:
                y = _yyyy(rng.randint(2000, 2099))
                m = _mmdd(rng.randint(1, 12))
                d = _mmdd(rng.randint(1, 31))
                matches_set.add(f"{y}.{m}.{d}")
            matches = sorted(matches_set)
            non_matches = [
                f"{_yyyy(rng.randint(2000, 2099))}-{_mmdd(rng.randint(1, 12))}-{_mmdd(rng.randint(1, 31))}",
                f"{_yyyy(rng.randint(2000, 2099))}.{_mmdd(rng.choice([0, 13]))}.{_mmdd(rng.randint(1, 31))}",
                f"{_yyyy(rng.choice([1999, 2100]))}.{_mmdd(rng.randint(1, 12))}.{_mmdd(rng.randint(1, 31))}",
            ]
            return {
                "desc": "match dates in YYYY.MM.DD format (using dots as separators) with no extra characters",
                "pattern": pattern,
                "matches": matches,
                "non_matches": non_matches,
            }

        def _hh(value: int) -> str:
            return f"{int(value):02d}"

        def _time_hms(*, valid: bool) -> str:
            if valid:
                hh = _hh(rng.randint(0, 23))
                mm = _hh(rng.randint(0, 59))
                ss = _hh(rng.randint(0, 59))
            else:
                hh = _hh(rng.choice([24, 25, 29]))
                mm = _hh(rng.choice([60, 61, 99]))
                ss = _hh(rng.choice([60, 61, 99]))
                # Randomly pick which field to break (avoid always breaking all of them).
                pick = rng.choice(["hh", "mm", "ss"])
                if pick == "hh":
                    mm = _hh(rng.randint(0, 59))
                    ss = _hh(rng.randint(0, 59))
                elif pick == "mm":
                    hh = _hh(rng.randint(0, 23))
                    ss = _hh(rng.randint(0, 59))
                else:
                    hh = _hh(rng.randint(0, 23))
                    mm = _hh(rng.randint(0, 59))
            return f"{hh}:{mm}:{ss}"

        def task_datetime_iso_t() -> dict[str, object]:
            pattern = (
                r"^(19\d{2}|20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])T"
                r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$"
            )
            matches_set: set[str] = set()
            while len(matches_set) < 2:
                y = _yyyy(rng.randint(1900, 2099))
                m = _mmdd(rng.randint(1, 12))
                d = _mmdd(rng.randint(1, 31))
                matches_set.add(f"{y}-{m}-{d}T{_time_hms(valid=True)}")
            matches = sorted(matches_set)
            y = _yyyy(rng.randint(1900, 2099))
            m = _mmdd(rng.randint(1, 12))
            d = _mmdd(rng.randint(1, 31))
            non_matches = [
                f"{y}-{m}-{d} {_time_hms(valid=True)}",
                f"{y}-{m}-{d}T{_time_hms(valid=False)}",
                f"{y}-{m}-{d}T{_hh(rng.randint(0, 23))}:{_hh(rng.randint(0, 59))}",
            ]
            return {
                "desc": "match ISO 8601 datetimes in YYYY-MM-DDTHH:MM:SS format (years 1900-2099) with no extra characters",
                "pattern": pattern,
                "matches": matches,
                "non_matches": non_matches,
            }

        def task_timestamp_ymd_hms() -> dict[str, object]:
            pattern = (
                r"^(19\d{2}|20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]) "
                r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$"
            )
            matches_set: set[str] = set()
            while len(matches_set) < 3:
                y = _yyyy(rng.randint(1900, 2099))
                m = _mmdd(rng.randint(1, 12))
                d = _mmdd(rng.randint(1, 31))
                matches_set.add(f"{y}-{m}-{d} {_time_hms(valid=True)}")
            matches = sorted(matches_set)
            y = _yyyy(rng.randint(1900, 2099))
            non_matches = [
                f"{y}-{_mmdd(rng.randint(1, 12))}-{_mmdd(rng.randint(1, 31))} {_time_hms(valid=False)}",
                f"{y}-{_mmdd(rng.choice([0, 13]))}-{_mmdd(rng.randint(1, 31))} {_time_hms(valid=True)}",
                f"{y}-{_mmdd(rng.randint(1, 12))}-{_mmdd(rng.choice([0, 32]))} {_time_hms(valid=True)}",
                f"{y}-{_mmdd(rng.randint(1, 12))}-{_mmdd(rng.randint(1, 31))}T{_time_hms(valid=True)}",
                f"{y}-{_mmdd(rng.randint(1, 12))}-{_mmdd(rng.randint(1, 31))}{_time_hms(valid=True)}",
            ]
            return {
                "desc": "match timestamps in YYYY-MM-DD HH:MM:SS format (years 1900-2099, 24-hour time) with exactly one space between date and time",
                "pattern": pattern,
                "matches": matches,
                "non_matches": non_matches,
            }

        def pick_task() -> dict[str, object]:
            if depth == "short":
                kind = rng.choice(
                    [
                        "email_prefix",
                        "n_digit",
                        "word_prefix",
                        "literal",
                        "hex_color",
                        "alpha_len",
                    ]
                )
                if kind == "email_prefix":
                    prefix = rng.choice(["admin", "root", "support", "sales", "team"]) + rng.choice(
                        ["", "1", "42", "_bot"]
                    )
                    pattern = rf"{re.escape(prefix)}\w*@\w+(?:\.\w+)+"
                    matches = [
                        f"{prefix}@example.com",
                        f"{prefix}123@{random_domain()}",
                    ]
                    non_matches = [
                        f"user@{random_domain()}",
                        f"{prefix}@",
                        f"{prefix}.user@{random_domain()}",
                    ]
                    return {
                        "desc": f"match emails starting with '{prefix}'",
                        "pattern": pattern,
                        "matches": matches,
                        "non_matches": non_matches,
                    }

                if kind == "n_digit":
                    digits = int(rng.randint(2, 6))
                    pattern = rf"\b\d{{{digits}}}\b"
                    matches = [
                        "".join(str(rng.randint(0, 9)) for _ in range(digits)),
                        "".join(str(rng.randint(0, 9)) for _ in range(digits)),
                        "".join(str(rng.randint(0, 9)) for _ in range(digits)),
                    ]
                    non_matches = [
                        "".join(str(rng.randint(0, 9)) for _ in range(max(1, digits - 1))),
                        "".join(str(rng.randint(0, 9)) for _ in range(digits + 1)),
                        random_word(3, 6),
                    ]
                    return {
                        "desc": f"match exactly {digits}-digit numbers",
                        "pattern": pattern,
                        "matches": matches,
                        "non_matches": non_matches,
                    }

                if kind == "word_prefix":
                    prefix = rng.choice(["test", "pre", "bio", "micro", "sym", "eco"])
                    pattern = rf"\b{re.escape(prefix)}\w*\b"
                    matches = [
                        prefix,
                        prefix + random_word(2, 5),
                        prefix + random_word(2, 5),
                    ]
                    non_matches = [
                        random_word(2, 5) + prefix,
                        random_word(2, 5) + prefix + random_word(2, 5),
                        random_word(4, 8),
                    ]
                    return {
                        "desc": f"match words starting with '{prefix}'",
                        "pattern": pattern,
                        "matches": matches,
                        "non_matches": non_matches,
                    }

                if kind == "literal":
                    literal = rng.choice(
                        [
                            "C++",
                            "file.txt",
                            "[URGENT]",
                            "a|b",
                            "x^2",
                            "hello.world",
                            "(test)",
                        ]
                    )
                    pattern = re.escape(literal)
                    matches = [literal]
                    non_matches = [literal + random_word(1, 2), random_word(1, 2) + literal]
                    return {
                        "desc": f"match the literal string '{literal}' exactly",
                        "pattern": pattern,
                        "matches": matches,
                        "non_matches": non_matches,
                    }

                if kind == "hex_color":
                    digits = int(rng.choice([3, 6]))
                    pattern = rf"#[0-9a-fA-F]{{{digits}}}"
                    matches = ["#" + "".join(rng.choice("0123456789ABCDEF") for _ in range(digits))]
                    non_matches = [
                        "#"
                        + "".join(
                            rng.choice("0123456789ABCDEF") for _ in range(max(1, digits - 1))
                        ),
                        "#GGG" if digits == 3 else "#GGGGGG",
                        random_word(3, 8),
                    ]
                    return {
                        "desc": f"match hex colors with exactly {digits} hex digits",
                        "pattern": pattern,
                        "matches": matches,
                        "non_matches": non_matches,
                    }

                # alpha_len
                length = int(rng.randint(4, 8))
                pattern = rf"[A-Za-z]{{{length}}}"
                matches = ["".join(rng.choice(alphabet) for _ in range(length))]
                non_matches = [
                    "".join(rng.choice(alphabet) for _ in range(length - 1)),
                    "".join(rng.choice(alphabet) for _ in range(length + 1)),
                    "".join(rng.choice("0123456789") for _ in range(length)),
                ]
                return {
                    "desc": f"match exactly {length}-letter words (letters only)",
                    "pattern": pattern,
                    "matches": matches,
                    "non_matches": non_matches,
                }

            if depth == "medium":
                kind = rng.choice(
                    [
                        "phone",
                        "url",
                        "identifier",
                        "date_mdy",
                        "time_hm",
                        "day_01_31",
                        "date_ymd",
                        "date_dot",
                        "datetime_iso_t",
                        "timestamp_ymd_hms",
                        "github",
                    ]
                )
                if kind == "phone":
                    pattern = r"(\d{3}-\d{3}-\d{4}|\(\d{3}\) \d{3}-\d{4})"
                    matches = ["123-456-7890", "(123) 456-7890"]
                    non_matches = ["123.456.7890", "12-345-6789", "123-45-6789"]
                    return {
                        "desc": "match phone numbers in format XXX-XXX-XXXX or (XXX) XXX-XXXX",
                        "pattern": pattern,
                        "matches": matches,
                        "non_matches": non_matches,
                    }

                if kind == "url":
                    domain = random_domain()
                    pattern = r"https?://[\w\.-]+\.\w+(?:/[^\s]*)?"
                    matches = [f"http://{domain}", f"https://{domain}/path/to/file"]
                    non_matches = [f"ftp://{domain}", domain, "http://"]
                    return {
                        "desc": "match URLs starting with http:// or https://",
                        "pattern": pattern,
                        "matches": matches,
                        "non_matches": non_matches,
                    }

                if kind == "identifier":
                    pattern = r"\b[a-zA-Z][a-zA-Z0-9_]*\b"
                    matches = [
                        random_identifier(allow_leading_underscore=False),
                        random_identifier(allow_leading_underscore=False),
                        random_identifier(allow_leading_underscore=False),
                    ]
                    non_matches = [
                        "1" + random_word(2, 4),
                        "_" + random_word(2, 4),
                        "123",
                    ]
                    return {
                        "desc": "match variable names (letters, numbers, underscore, must start with letter)",
                        "pattern": pattern,
                        "matches": matches,
                        "non_matches": non_matches,
                    }

                if kind == "date_mdy":
                    pattern = r"\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}\b"
                    matches = ["01/15/2024", "12/31/2023"]
                    non_matches = ["13/01/2024", "1/5/2024", "01-15-2024"]
                    return {
                        "desc": "match dates in MM/DD/YYYY format",
                        "pattern": pattern,
                        "matches": matches,
                        "non_matches": non_matches,
                    }

                if kind == "time_hm":
                    pattern = r"\b([01]\d|2[0-3]):[0-5]\d\b"
                    matches = ["00:00", "14:30", "23:59"]
                    non_matches = ["24:00", "9:30", "14:60"]
                    return {
                        "desc": "match time in HH:MM format (24-hour clock)",
                        "pattern": pattern,
                        "matches": matches,
                        "non_matches": non_matches,
                    }

                if kind == "day_01_31":
                    return task_day_01_31()

                if kind == "date_ymd":
                    return task_date_ymd()

                if kind == "date_dot":
                    return task_date_dot()

                if kind == "datetime_iso_t":
                    return task_datetime_iso_t()

                if kind == "timestamp_ymd_hms":
                    return task_timestamp_ymd_hms()

                # github
                pattern = r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*"
                matches = ["john-doe", "user123", "a-b-c"]
                non_matches = ["-invalid", "user-", "bad--name"]
                return {
                    "desc": "match GitHub-style usernames (alphanumeric + hyphens, no leading/trailing hyphen)",
                    "pattern": pattern,
                    "matches": matches,
                    "non_matches": non_matches,
                }

            # long
            kind = rng.choice(
                [
                    "ipv4",
                    "json_kv",
                    "py_def",
                    "email_tld",
                    "mac",
                    "semver",
                    "date_iso",
                    "day_01_31",
                    "date_ymd",
                    "date_dot",
                    "datetime_iso_t",
                    "timestamp_ymd_hms",
                    "uuid",
                    "log_level",
                    "file_path",
                ]
            )
            if kind == "day_01_31":
                return task_day_01_31()

            if kind == "date_ymd":
                return task_date_ymd()

            if kind == "date_dot":
                return task_date_dot()

            if kind == "datetime_iso_t":
                return task_datetime_iso_t()

            if kind == "timestamp_ymd_hms":
                return task_timestamp_ymd_hms()

            if kind == "date_iso":
                pattern = r"\b\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"
                matches = ["2024-01-15", "1999-12-31"]
                non_matches = ["2024/01/15", "15-01-2024", "2024-13-01"]
                return {
                    "desc": "match ISO 8601 dates (YYYY-MM-DD)",
                    "pattern": pattern,
                    "matches": matches,
                    "non_matches": non_matches,
                }

            if kind == "uuid":
                pattern = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
                matches = [
                    "123e4567-e89b-12d3-a456-426614174000",
                    "00000000-0000-0000-0000-000000000000",
                ]
                non_matches = ["123e4567-e89b-12d3-a456", "123e4567-e89b-12d3-a456-4266141740000"]
                return {
                    "desc": "match UUIDs (standard 8-4-4-4-12 hex format)",
                    "pattern": pattern,
                    "matches": matches,
                    "non_matches": non_matches,
                }

            if kind == "log_level":
                pattern = r"^\[(INFO|WARN|ERROR|DEBUG)\]\s+.*$"
                matches = ["[INFO] System started", "[ERROR] Connection failed"]
                non_matches = ["INFO System started", "[TRACE] Details...", "[info] lower case"]
                return {
                    "desc": "match log lines starting with [INFO], [WARN], [ERROR], or [DEBUG] at the start of the string",
                    "pattern": pattern,
                    "matches": matches,
                    "non_matches": non_matches,
                }

            if kind == "file_path":
                # Unix-style absolute paths
                pattern = r"^/[a-zA-Z0-9_./-]+$"
                matches = ["/var/log/syslog", "/home/user/.bashrc", "/usr/local/share/data.txt"]
                non_matches = ["var/log", "C:\\Windows", "/var/log/invalid char!"]
                return {
                    "desc": "match Unix-style absolute file paths (start with /, alphanumeric/dot/dash/underscore)",
                    "pattern": pattern,
                    "matches": matches,
                    "non_matches": non_matches,
                }

            if kind == "ipv4":
                pattern = r"\b((25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\b"
                matches = ["192.168.1.1", "10.0.0.255", "8.8.8.8"]
                non_matches = ["256.1.1.1", "192.168.1", "192.168.1.1.1"]
                return {
                    "desc": "match valid IPv4 addresses",
                    "pattern": pattern,
                    "matches": matches,
                    "non_matches": non_matches,
                }

            if kind == "json_kv":
                key = rng.choice(["name", "id", "type", "value", "status"])
                pattern = r'"\w+"\s*:\s*"[^"]*"'
                matches = [f'"{key}": "{random_word(3, 8)}"', '"id": "123"']
                non_matches = [f"{key}: {random_word(3, 8)}", '"name":"', 'name: "John"']
                return {
                    "desc": "match JSON-like key-value pairs with quoted strings",
                    "pattern": pattern,
                    "matches": matches,
                    "non_matches": non_matches,
                }

            if kind == "py_def":
                func = random_identifier(3, 10)
                pattern = r"def\s+[a-zA-Z_]\w*\s*\([^)]*\)\s*:"
                matches = ["def foo():", f"def {func}(x, y):", "def _helper(data):"]
                non_matches = ["def 123():", "def foo()", "function foo():"]
                return {
                    "desc": "match Python function definitions",
                    "pattern": pattern,
                    "matches": matches,
                    "non_matches": non_matches,
                }

            if kind == "email_tld":
                pattern = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.(com|org|net|edu|gov)\b"
                matches = [
                    f"{random_word(3, 8)}@example.com",
                    f"test.user@{random_domain(tlds=['com', 'org', 'net', 'edu', 'gov'])}",
                ]
                non_matches = ["user@example", "@example.com", "user@.com"]
                return {
                    "desc": "match email addresses with common TLDs",
                    "pattern": pattern,
                    "matches": matches,
                    "non_matches": non_matches,
                }

            if kind == "mac":
                pattern = r"\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"
                matches = ["AA:BB:CC:DD:EE:FF", "00-11-22-33-44-55"]
                non_matches = ["ZZ:BB:CC:DD:EE:FF", "AA:BB:CC:DD:EE", "AABBCCDDEEFF"]
                return {
                    "desc": "match MAC addresses (6 groups of 2 hex digits, colon or hyphen separated)",
                    "pattern": pattern,
                    "matches": matches,
                    "non_matches": non_matches,
                }

            # semver
            major = rng.randint(0, 20)
            minor = rng.randint(0, 20)
            patch = rng.randint(0, 20)
            pattern = r"\b\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?\b"
            matches = [
                f"{major}.{minor}.{patch}",
                f"{major}.{minor}.{patch}-alpha.1",
                f"{major}.{minor}.{patch}+build.5",
            ]
            non_matches = [
                f"{major}.{minor}",
                f"{major}.{minor}.{patch}.{rng.randint(0, 9)}",
                f"v{major}.{minor}.{patch}",
            ]
            return {
                "desc": "match semantic version numbers (e.g., 1.2.3, 1.2.3-alpha.1, 1.2.3+build.5)",
                "pattern": pattern,
                "matches": matches,
                "non_matches": non_matches,
            }

        choice = pick_task()
        desc = str(choice["desc"])
        pattern = str(choice["pattern"])
        matches = list(choice["matches"])  # type: ignore[arg-type]
        non_matches = list(choice["non_matches"])  # type: ignore[arg-type]

        # Create test examples string
        test_examples = []
        for s in matches[:2]:
            test_examples.append(f'"{s}" (should match)')
        for s in non_matches[:2]:
            test_examples.append(f'"{s}" (should NOT match)')

        prompt = (
            f"Write a regex pattern to {desc}. "
            f"Test cases: {', '.join(test_examples)}. "
            "Respond with only the regex pattern, no delimiters or explanation."
        )

        test_strings = [{"string": s, "should_match": True} for s in matches] + [
            {"string": s, "should_match": False} for s in non_matches
        ]

        return prompt, pattern, test_strings

    def _regex_cases_pass(self, pattern: str, test_strings: list[dict[str, object]]) -> bool:
        try:
            compiled = re.compile(pattern)
        except re.error:
            return False
        if not test_strings:
            return False
        for tc in test_strings:
            test_str = str(tc.get("string", ""))
            should_match = bool(tc.get("should_match", False))
            matched = bool(compiled.fullmatch(test_str))
            if matched != should_match:
                return False
        return True

    def _extract_regex_desc(self, prompt: str) -> str:
        match = re.search(r"Write a regex pattern to (.+?)\\. Test cases:", prompt)
        return match.group(1) if match else "match the desired strings"

    def _make_regex_recognition_task(self, depth: str) -> tuple[str, dict[str, object]]:
        for _ in range(25):
            synth_prompt, pattern, test_strings = self._make_regex_task(depth)
            if not self._regex_cases_pass(pattern, test_strings):
                continue
            tc = self.rng.choice(list(test_strings))
            test_str = str(tc.get("string", ""))
            expected = bool(tc.get("should_match", False))
            prompt = (
                "Given this regex:\n"
                f"{pattern}\n\n"
                f'Does it match "{test_str}"? Answer yes or no, and briefly explain why.'
            )
            return prompt, {"expected": expected, "pattern": pattern, "test_string": test_str}
        # Fallback: a trivial recognition task
        prompt = 'Does the regex "^a+$" match "aaa"? Answer yes or no.'
        return prompt, {"expected": True, "pattern": "^a+$", "test_string": "aaa"}

    def _make_regex_explanation_task(self, depth: str) -> tuple[str, dict[str, object]]:
        stop = {
            "match",
            "matches",
            "exactly",
            "string",
            "strings",
            "with",
            "the",
            "a",
            "an",
            "of",
            "to",
            "in",
            "format",
            "starting",
            "style",
            "names",
            "name",
            "numbers",
            "number",
            "words",
            "word",
        }
        for _ in range(25):
            synth_prompt, pattern, test_strings = self._make_regex_task(depth)
            if not self._regex_cases_pass(pattern, test_strings):
                continue
            desc = self._extract_regex_desc(synth_prompt)
            tokens = re.findall(r"[A-Za-z0-9_]+", desc)
            keywords: list[str] = []
            for tok in tokens:
                t = tok.strip().lower()
                if not t or t in stop:
                    continue
                if t not in keywords:
                    keywords.append(t)
            if not keywords:
                keywords = ["regex", "match"]
            keywords = keywords[:8]
            explanation = f"This regex matches strings that {desc}."
            prompt = f"Explain what this regex matches in plain English:\n{pattern}"
            return prompt, {
                "required_keywords": keywords,
                "pattern": pattern,
                "reference_answer": explanation,
            }
        prompt = "Explain what this regex matches in plain English:\n^a+$"
        return prompt, {
            "required_keywords": ["a"],
            "pattern": "^a+$",
            "reference_answer": "Matches one or more 'a' characters.",
        }

    def _mutate_regex_pattern(self, pattern: str) -> list[tuple[str, str]]:
        """Return candidate (mutated_pattern, mutation_note) pairs."""
        mutations: list[tuple[str, str]] = []

        if pattern.startswith("^") and pattern.endswith("$") and len(pattern) > 2:
            core = pattern[1:-1]
            mutations.append((core, "removed anchors"))
            mutations.append((pattern[1:], "removed start anchor"))
            mutations.append((pattern[:-1], "removed end anchor"))

        if " " in pattern:
            no_space = pattern.replace(" ", "", 1)
            if no_space != pattern:
                mutations.append((no_space, "removed required whitespace"))

        if "2[0-3]" in pattern:
            mutations.append((pattern.replace("2[0-3]", "2\\d", 1), "widened a bounded range"))
            mutations.append((pattern.replace("2[0-3]", "2[0-4]", 1), "widened a bounded range"))

        # Prefer loosening fixed-length digit runs.
        widened = re.sub(r"\\d\{\d+\}", lambda _m: r"\d+", pattern, count=1)
        if widened != pattern:
            mutations.append((widened, "loosened a fixed-length quantifier"))

        # Loosen a simple range quantifier.
        widened_range = re.sub(r"\{\d+(?:,\d*)?\}", "{1,}", pattern, count=1)
        if widened_range != pattern:
            mutations.append((widened_range, "loosened a range quantifier"))

        # Loosen a plus to star.
        plus_to_star = pattern.replace("+", "*", 1)
        if plus_to_star != pattern:
            mutations.append((plus_to_star, "changed + to *"))

        # Expand month/day constraints to allow invalid numbers (common bug template).
        widened_month = pattern.replace("(0[1-9]|1[0-2])", "(\\d{2})", 1)
        if widened_month != pattern:
            mutations.append((widened_month, "removed month range constraint"))

        widened_day = pattern.replace("(0[1-9]|[12]\\d|3[01])", "(\\d{2})", 1)
        if widened_day != pattern:
            mutations.append((widened_day, "removed day range constraint"))

        return mutations

    def _make_regex_debugging_task(self, depth: str) -> tuple[str, dict[str, object]]:
        for _ in range(40):
            synth_prompt, pattern, test_strings = self._make_regex_task(depth)
            if not self._regex_cases_pass(pattern, test_strings):
                continue
            desc = self._extract_regex_desc(synth_prompt)
            candidates = self._mutate_regex_pattern(pattern)
            self.rng.shuffle(candidates)
            for mutated, note in candidates:
                if mutated == pattern:
                    continue
                if not self._regex_cases_pass(pattern, test_strings):
                    continue
                # Mutated must compile and fail at least one case.
                try:
                    re.compile(mutated)
                except re.error:
                    continue
                if self._regex_cases_pass(mutated, test_strings):
                    continue
                prompt = (
                    f"This regex is supposed to {desc} but has a bug:\n"
                    f"{mutated}\n\n"
                    "Fix the regex. Respond with only the corrected pattern."
                )
                return prompt, {
                    "pattern": pattern,
                    "broken_pattern": mutated,
                    "bug_description": note,
                    "test_strings": test_strings,
                }
        # Fallback: mutate by loosening digits in a simple pattern.
        prompt = (
            "This regex is supposed to match exactly three digits but has a bug:\n"
            "^\\d+$\n\nFix the regex. Respond with only the corrected pattern."
        )
        return prompt, {
            "pattern": "^\\d{3}$",
            "broken_pattern": "^\\d+$",
            "bug_description": "loosened a fixed-length quantifier",
            "test_strings": [
                {"string": "123", "should_match": True},
                {"string": "12", "should_match": False},
                {"string": "1234", "should_match": False},
            ],
        }

    def _inflate_regex_pattern(self, pattern: str) -> str:
        inflated = pattern

        def expand_digits(match: re.Match[str]) -> str:
            n = int(match.group(1))
            if n <= 0 or n > 6:
                return match.group(0)
            return "[0-9]" * n

        inflated = re.sub(r"\\d\{(\d+)\}", expand_digits, inflated)
        inflated = inflated.replace("\\d", "[0-9]")
        inflated = inflated.replace("\\w", "[A-Za-z0-9_]")
        if inflated == pattern:
            inflated = f"(?:{pattern})"
        return inflated

    def _make_regex_refactoring_task(self, depth: str) -> tuple[str, dict[str, object]]:
        for _ in range(25):
            synth_prompt, pattern, test_strings = self._make_regex_task(depth)
            if not self._regex_cases_pass(pattern, test_strings):
                continue
            inflated = self._inflate_regex_pattern(pattern)
            if not self._regex_cases_pass(inflated, test_strings):
                continue
            try:
                if _regex_complexity_score(inflated) <= _regex_complexity_score(pattern):
                    inflated = f"(?:{inflated})"
            except Exception:
                pass
            prompt = (
                "Simplify this regex without changing its matching behavior:\n"
                f"{inflated}\n\nRespond with only the simplified pattern."
            )
            return prompt, {
                "pattern": pattern,
                "original_pattern": inflated,
                "test_strings": test_strings,
            }
        prompt = (
            "Simplify this regex without changing its matching behavior:\n"
            "^(?:[0-9][0-9][0-9])$\n\nRespond with only the simplified pattern."
        )
        return prompt, {
            "pattern": "^\\d{3}$",
            "original_pattern": "^(?:[0-9][0-9][0-9])$",
            "test_strings": [
                {"string": "123", "should_match": True},
                {"string": "12", "should_match": False},
                {"string": "1234", "should_match": False},
            ],
        }

    def _make_regex_mutation_effect_task(self, depth: str) -> tuple[str, dict[str, object]]:
        for _ in range(40):
            synth_prompt, pattern, test_strings = self._make_regex_task(depth)
            if not self._regex_cases_pass(pattern, test_strings):
                continue
            candidates = self._mutate_regex_pattern(pattern)
            self.rng.shuffle(candidates)
            for mutated, note in candidates:
                if mutated == pattern:
                    continue
                try:
                    compiled = re.compile(mutated)
                    compiled_orig = re.compile(pattern)
                except re.error:
                    continue
                # Keep positives matching.
                positives_ok = True
                new_matches: list[str] = []
                for tc in test_strings:
                    test_str = str(tc.get("string", ""))
                    should_match = bool(tc.get("should_match", False))
                    orig_match = bool(compiled_orig.fullmatch(test_str))
                    mut_match = bool(compiled.fullmatch(test_str))
                    if should_match and not mut_match:
                        positives_ok = False
                        break
                    if (not should_match) and (not orig_match) and mut_match:
                        new_matches.append(test_str)
                if not positives_ok or not new_matches:
                    continue
                required = list(dict.fromkeys(new_matches))[:6]
                reference = ", ".join(required)
                prompt = (
                    f"Original regex: {pattern}\n"
                    f"Mutated regex: {mutated}\n\n"
                    f"The regex was changed ({note}). What new strings will now match that didn't before?"
                )
                return prompt, {
                    "required_keywords": required,
                    "pattern": pattern,
                    "mutated_pattern": mutated,
                    "reference_answer": reference,
                }
        prompt = (
            "Original regex: ^\\d{3}$\nMutated regex: ^\\d+$\n\n"
            "The quantifier was changed from {3} to +. What new strings will now match?"
        )
        return prompt, {
            "required_keywords": ["1", "12", "1234"],
            "pattern": "^\\d{3}$",
            "mutated_pattern": "^\\d+$",
            "reference_answer": "1, 12, 1234",
        }

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
