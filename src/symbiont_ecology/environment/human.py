"""Human-in-the-loop bandit interface placeholders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from symbiont_ecology.metrics.telemetry import RewardBreakdown


@dataclass
class HumanFeedback:
    pairwise_choice: int
    rubric_score: float
    notes: str


@dataclass
class HumanFeedbackResult:
    reward: RewardBreakdown
    notes: str


class HumanBandit:
    def __init__(
        self,
        preference_weight: float = 0.2,
        helper_weight: float = 0.1,
        frequency: float = 1.0,
    ) -> None:
        self.history: list[HumanFeedback] = []
        self.preference_weight = preference_weight
        self.helper_weight = helper_weight
        self.frequency = max(0.0, min(1.0, frequency))

    def solicit(self, prompt: str, response: str) -> HumanFeedbackResult:
        if not response:
            penalty = RewardBreakdown(
                task_reward=0.0,
                novelty_bonus=0.0,
                competence_bonus=0.0,
                helper_bonus=0.0,
                risk_penalty=0.1,
                cost_penalty=0.0,
            )
            return HumanFeedbackResult(reward=penalty, notes="no-response")

        length_score = min(len(response) / 50.0, 1.0)
        preference_reward = self.preference_weight * length_score
        helper_credit = self.helper_weight if "thanks" in response.lower() else 0.0
        novelty = 0.05 if any(char.isupper() for char in response) else 0.0
        risk_penalty = 0.0 if "error" not in response.lower() else 0.1

        feedback = HumanFeedback(
            pairwise_choice=0,
            rubric_score=preference_reward,
            notes="auto-length",
        )
        self.history.append(feedback)
        reward = RewardBreakdown(
            task_reward=preference_reward,
            novelty_bonus=novelty,
            competence_bonus=0.0,
            helper_bonus=helper_credit,
            risk_penalty=risk_penalty,
            cost_penalty=0.0,
        )
        return HumanFeedbackResult(reward=reward, notes=feedback.notes)

    def batch(self, prompt: str, responses: List[str]) -> list[HumanFeedbackResult]:
        return [self.solicit(prompt, response) for response in responses]


__all__ = ["HumanBandit", "HumanFeedback", "HumanFeedbackResult"]
