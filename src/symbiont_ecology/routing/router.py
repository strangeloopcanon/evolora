"""Routing policies for selecting organelles."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from symbiont_ecology.interfaces.messages import Intent, MessageEnvelope
from symbiont_ecology.organelles.base import Organelle


@dataclass
class BanditArm:
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    successes: float = 0.0
    failures: float = 0.0

    def sample(self) -> float:
        alpha = self.prior_alpha + self.successes
        beta = self.prior_beta + self.failures
        return random.betavariate(alpha, beta)

    def update(self, reward: float) -> None:
        if reward >= 0:
            self.successes += reward
        else:
            self.failures += abs(reward)


@dataclass
class BanditRouter:
    arms: dict[str, BanditArm] = field(default_factory=dict)

    def update_bandit(self, organelle_id: str, prior: float = 0.5) -> None:
        if organelle_id not in self.arms:
            self.arms[organelle_id] = BanditArm(
                prior_alpha=max(prior, 1e-2), prior_beta=max(1 - prior, 1e-2)
            )

    def select(
        self, organelles: dict[str, Organelle], envelope: MessageEnvelope, k: int
    ) -> list[Organelle]:
        scored: list[tuple[str, float]] = []
        for organelle_id, organelle in organelles.items():
            activation = organelle.route_probability(envelope)
            bandit_sample = self.arms.get(organelle_id, BanditArm()).sample()
            score = 0.5 * activation + 0.5 * bandit_sample
            scored.append((organelle_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        top_ids = [organelle_id for organelle_id, _ in scored[:k]]
        return [organelles[organelle_id] for organelle_id in top_ids]

    def observe(self, organelle_id: str, reward: float) -> None:
        self.update_bandit(organelle_id)
        self.arms[organelle_id].update(reward)

    def intent_factory(self, goal: str, constraints: Sequence[str]) -> Intent:
        budget = max(0.0, 1.0 - math.tanh(len(constraints)))
        return Intent(goal=goal, constraints=list(constraints), energy_budget=budget)


__all__ = ["BanditRouter"]
