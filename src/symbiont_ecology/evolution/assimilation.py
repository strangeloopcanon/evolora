"""Assimilation testing utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from symbiont_ecology.metrics.telemetry import AssimilationEvent


@dataclass
class AssimilationTestResult:
    event: AssimilationEvent
    decision: bool


class AssimilationTester:
    def __init__(
        self, uplift_threshold: float, p_value_threshold: float, safety_budget: int
    ) -> None:
        self.uplift_threshold = uplift_threshold
        self.p_value_threshold = p_value_threshold
        self.safety_budget = safety_budget

    def update_thresholds(self, *, uplift_threshold: float | None = None, p_value_threshold: float | None = None) -> None:
        if uplift_threshold is not None:
            self.uplift_threshold = uplift_threshold
        if p_value_threshold is not None:
            self.p_value_threshold = p_value_threshold

    def evaluate(
        self,
        organelle_id: str,
        control_scores: Iterable[float],
        treatment_scores: Iterable[float],
        safety_hits: int,
        energy_cost: float,
    ) -> AssimilationTestResult:
        control = np.array(list(control_scores), dtype=float)
        treatment = np.array(list(treatment_scores), dtype=float)
        min_len = min(len(control), len(treatment))
        if min_len < 2:
            event = AssimilationEvent(
                organelle_id=organelle_id,
                uplift=0.0,
                p_value=1.0,
                passed=False,
                energy_cost=energy_cost,
                safety_hits=safety_hits,
            )
            return AssimilationTestResult(event=event, decision=False)
        if len(control) != len(treatment):
            control = control[:min_len]
            treatment = treatment[:min_len]
        uplift = float(treatment.mean() - control.mean())
        variance = float(np.var(treatment - control, ddof=1)) + 1e-8
        z_score = uplift / np.sqrt(variance / max(len(treatment), 1))
        p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
        passed = (
            uplift >= self.uplift_threshold
            and p_value <= self.p_value_threshold
            and safety_hits <= self.safety_budget
        )
        event = AssimilationEvent(
            organelle_id=organelle_id,
            uplift=uplift,
            p_value=p_value,
            passed=passed,
            energy_cost=energy_cost,
            safety_hits=safety_hits,
        )
        return AssimilationTestResult(event=event, decision=passed)


__all__ = ["AssimilationTestResult", "AssimilationTester"]
