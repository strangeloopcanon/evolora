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
        self.bootstrap_enabled = False
        self.bootstrap_n = 0
        self.permutation_n = 0
        self.min_samples = 2

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
        if min_len < max(2, self.min_samples):
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
        diff = treatment - control
        uplift = float(diff.mean())
        ci_low: float | None = None
        ci_high: float | None = None
        power: float | None = None
        if self.bootstrap_enabled and self.bootstrap_n > 0 and self.permutation_n > 0:
            rng = np.random.default_rng(1337)
            n = sample_size
            # bootstrap CI on mean(diff)
            boot = []
            for _ in range(self.bootstrap_n):
                idx = rng.integers(0, n, size=n)
                boot.append(float(diff[idx].mean()))
            boot.sort()
            ci_low = boot[int(0.025 * len(boot))]
            ci_high = boot[int(0.975 * len(boot))]
            # permutation p-value via sign flips
            obs = abs(uplift)
            count = 0
            for _ in range(self.permutation_n):
                signs = rng.choice([1, -1], size=n)
                perm = abs(float((diff * signs).mean()))
                if perm >= obs:
                    count += 1
            p_value = (count + 1) / (self.permutation_n + 1)
            # simple power proxy from bootstrap SE
            try:
                boot_std = float(np.std(boot, ddof=1))
                se = max(boot_std, 1e-12)
                # one-sided test against uplift_threshold
                delta = uplift - self.uplift_threshold
                # z_alpha from p_value_threshold
                # For one-sided alpha, inverse CDF approximation via erf^-1 not available; use erfc inverse approx
                # Use a rough mapping: z_alpha ~ sqrt(2)*erfc_inv(2*alpha), but we can grid common values
                alpha = max(min(self.p_value_threshold, 0.499), 1e-6)
                # approximate inverse using numpy for robustness
                from math import sqrt
                z_effect = float(delta / se)
                # Normal CDF using erf: Phi(x) = 0.5*(1+erf(x/sqrt(2)))
                def Phi(x: float) -> float:
                    from math import erf
                    return 0.5 * (1.0 + erf(x / sqrt(2.0)))
                # Use z_alpha ~ 1.28 for 0.10, 1.64 for 0.05, 1.96 for 0.025, 2.33 for 0.01, 3.09 for 0.001
                if alpha >= 0.10:
                    z_alpha = 1.28
                elif alpha >= 0.05:
                    z_alpha = 1.64
                elif alpha >= 0.025:
                    z_alpha = 1.96
                elif alpha >= 0.01:
                    z_alpha = 2.33
                else:
                    z_alpha = 3.09
                power = max(0.0, min(1.0, 1.0 - float(Phi(z_alpha - z_effect))))
            except Exception:
                power = None
        else:
            variance = float(np.var(diff, ddof=1)) + 1e-8
            se = math.sqrt(variance / max(sample_size, 1))
            z_score = uplift / max(se, 1e-12)
            p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
            # 95% CI via normal approx
            ci_width = 1.96 * se
            ci_low = uplift - ci_width
            ci_high = uplift + ci_width
            # power proxy for one-sided test
            try:
                alpha = max(min(self.p_value_threshold, 0.499), 1e-6)
                if alpha >= 0.10:
                    z_alpha = 1.28
                elif alpha >= 0.05:
                    z_alpha = 1.64
                elif alpha >= 0.025:
                    z_alpha = 1.96
                elif alpha >= 0.01:
                    z_alpha = 2.33
                else:
                    z_alpha = 3.09
                delta = uplift - self.uplift_threshold
                from math import erf, sqrt
                def Phi(x: float) -> float:
                    return 0.5 * (1.0 + erf(x / sqrt(2.0)))
                power = max(0.0, min(1.0, 1.0 - float(Phi(z_alpha - (delta / max(se, 1e-12))))))
            except Exception:
                power = None
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
            ci_low=ci_low,
            ci_high=ci_high,
            power=power,
            uplift_threshold=self.uplift_threshold,
            p_value_threshold=self.p_value_threshold,
            energy_balance=energy_balance,
            energy_top_up=energy_top_up,
        )
        return AssimilationTestResult(event=event, decision=passed)


__all__ = ["AssimilationTestResult", "AssimilationTester"]
