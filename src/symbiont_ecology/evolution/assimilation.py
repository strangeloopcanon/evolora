"""Assimilation testing utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
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
        *,
        energy_balance: float | None = None,
        energy_top_up: Mapping[str, float | str] | None = None,
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
                sample_size=min_len if min_len > 0 else None,
                control_mean=float(control.mean()) if len(control) else None,
                treatment_mean=float(treatment.mean()) if len(treatment) else None,
                control_std=float(control.std(ddof=0)) if len(control) else None,
                treatment_std=float(treatment.std(ddof=0)) if len(treatment) else None,
                uplift_threshold=self.uplift_threshold,
                p_value_threshold=self.p_value_threshold,
                energy_balance=energy_balance,
                energy_top_up=energy_top_up,
            )
            return AssimilationTestResult(event=event, decision=False)
        if len(control) != len(treatment):
            control = control[:min_len]
            treatment = treatment[:min_len]
        sample_size = len(control)
        control_mean = float(control.mean())
        treatment_mean = float(treatment.mean())
        uplift = treatment_mean - control_mean
        diff = treatment - control
        variance = float(np.var(diff, ddof=1)) + 1e-8
        z_score = uplift / np.sqrt(variance / max(sample_size, 1))
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
            sample_size=sample_size,
            control_mean=control_mean,
            control_std=float(control.std(ddof=1)) if sample_size >= 2 else 0.0,
            treatment_mean=treatment_mean,
            treatment_std=float(treatment.std(ddof=1)) if sample_size >= 2 else 0.0,
            uplift_threshold=self.uplift_threshold,
            p_value_threshold=self.p_value_threshold,
            energy_balance=energy_balance,
            energy_top_up=energy_top_up,
        )
        return AssimilationTestResult(event=event, decision=passed)


__all__ = ["AssimilationTestResult", "AssimilationTester"]
