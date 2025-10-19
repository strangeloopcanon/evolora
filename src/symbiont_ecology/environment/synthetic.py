"""Synthetic environment with deterministic programmatic tasks."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Dict, Union

from symbiont_ecology.metrics.telemetry import RewardBreakdown

Payload = Mapping[str, Union[float, str]]


@dataclass
class SyntheticTask:
    task_id: str
    kind: str
    prompt: str
    payload: Payload
    difficulty: float

    def expected_answer(self) -> str:
        if self.kind == "math.add":
            first = int(float(self.payload["a"]))
            second = int(float(self.payload["b"]))
            return str(first + second)
        if self.kind == "math.mul":
            first = int(float(self.payload["a"]))
            second = int(float(self.payload["b"]))
            return str(first * second)
        if self.kind == "string.reverse":
            return str(self.payload["text"])[::-1]
        if self.kind == "string.sort":
            letters_str = str(self.payload["letters"])
            letters = letters_str.split()
            return "".join(sorted(letters))
        if self.kind == "word.count":
            sentence = str(self.payload["sentence"])
            return str(len(sentence.split()))
        if self.kind == "math.prime":
            upper = int(float(self.payload["upper"]))
            primes = [str(value) for value in range(2, upper + 1) if _is_prime(value)]
            return ",".join(primes)
        raise ValueError(f"Unknown task kind: {self.kind}")

    def score(self, response: str, energy_spent: float = 0.0) -> RewardBreakdown:
        gold = self.expected_answer()
        response_text = response.strip()
        task_reward = 1.0 if gold in response_text else 0.0
        novelty_bonus = min(0.2, max(0.0, len(set(response_text)) / 50.0))
        competence_bonus = task_reward * (1 - self.difficulty)
        helper_bonus = 0.0
        risk_penalty = 0.1 if task_reward == 0 else 0.0
        cost_penalty = min(max(energy_spent, 0.0), 2.0)
        return RewardBreakdown(
            task_reward=task_reward,
            novelty_bonus=novelty_bonus,
            competence_bonus=competence_bonus,
            helper_bonus=helper_bonus,
            risk_penalty=risk_penalty,
            cost_penalty=cost_penalty,
        )


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    for factor in range(2, int(math.sqrt(value)) + 1):
        if value % factor == 0:
            return False
    return True


class TaskFactory:
    """Produces deterministic tasks with reproducible seeds and evolving phases."""

    def __init__(self, seed: int = 17) -> None:
        self._base_seed = seed
        self.phase = 0
        self._rng = random.Random(seed)  # nosec B311 - deterministic curriculum

    def advance_phase(self, delta: int = 1) -> None:
        self.phase += delta
        self._rng = random.Random(self._base_seed + self.phase)  # nosec B311

    def _next_id(self) -> str:
        return f"task_{self._rng.randint(0, 9999)}"

    def sample(self, k: int) -> list[SyntheticTask]:
        tasks: list[SyntheticTask] = []
        for _ in range(k):
            choice = self._choose_task_kind()
            if choice == "math.add":
                a, b = self._rng.randint(1, 50), self._rng.randint(1, 50)
                prompt = f"Add {a} and {b}."
                payload: Dict[str, Union[float, str]] = {"a": float(a), "b": float(b)}
                difficulty = 0.2
            elif choice == "math.mul":
                a, b = self._rng.randint(2, 12), self._rng.randint(2, 12)
                prompt = f"Multiply {a} by {b}."
                payload = {"a": float(a), "b": float(b)}
                difficulty = 0.4
            elif choice == "string.reverse":
                text = "symbiosis"
                prompt = "Reverse the string 'symbiosis'."
                payload = {"text": text}
                difficulty = 0.3
            elif choice == "string.sort":
                letters = [self._rng.choice("symbiotic") for _ in range(6)]
                prompt = f"Sort the letters {' '.join(letters)} alphabetically."
                payload = {"letters": " ".join(letters)}
                difficulty = 0.4 + 0.05 * self.phase
            elif choice == "word.count":
                sentence = "endosymbiotic agents cooperate"
                prompt = f"Count the words in '{sentence}'."
                payload = {"sentence": sentence}
                difficulty = 0.3 + 0.05 * self.phase
            else:
                upper = self._rng.randint(5, 15)
                prompt = f"List all primes up to {upper}."
                payload = {"upper": float(upper)}
                difficulty = 0.6
            tasks.append(
                SyntheticTask(
                    task_id=self._next_id(),
                    kind=choice,
                    prompt=prompt,
                    payload=payload,
                    difficulty=difficulty,
                )
            )
        return tasks

    def _choose_task_kind(self) -> str:
        catalog = [
            "math.add",
            "math.mul",
            "string.reverse",
            "math.prime",
            "string.sort",
            "word.count",
        ]
        weights = [
            3,
            2,
            1,
            2 + max(self.phase - 1, 0),
            1 + self.phase,
            1 + max(self.phase - 1, 0),
        ]
        return self._rng.choices(catalog, weights=weights, k=1)[0]


def evaluate_population_responses(
    tasks: Iterable[SyntheticTask],
    responses: dict[str, tuple[str, float]],
) -> dict[str, RewardBreakdown]:
    rewards: dict[str, RewardBreakdown] = {}
    task_cycle = list(tasks)
    if not task_cycle:
        return rewards
    for index, (org_id, (response_text, energy_spent)) in enumerate(responses.items()):
        task = task_cycle[index % len(task_cycle)]
        rewards[org_id] = task.score(response_text, energy_spent=energy_spent)
    return rewards


__all__ = ["SyntheticTask", "TaskFactory", "evaluate_population_responses"]
