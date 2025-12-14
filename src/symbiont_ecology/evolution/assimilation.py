"""Assimilation testing utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from symbiont_ecology.metrics.telemetry import AssimilationEvent, EnergyTopUpEvent


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
        self.dr_enabled = False
        self.dr_strata: list[str] = ["family", "depth"]
        self.dr_min_stratum = 2
        self.dr_min_power = 0.2

    def update_thresholds(
        self, *, uplift_threshold: float | None = None, p_value_threshold: float | None = None
    ) -> None:
        if uplift_threshold is not None:
            self.uplift_threshold = uplift_threshold
        if p_value_threshold is not None:
            self.p_value_threshold = p_value_threshold

    def _align_dr(
        self,
        control: list[float],
        treatment: list[float],
        control_meta: list[Mapping[str, object]],
        treatment_meta: list[Mapping[str, object]],
    ) -> tuple[list[float], list[float], dict[str, dict[str, int]]]:
        if not control_meta or not treatment_meta:
            return [], [], {}

        def key_from_meta(meta: Mapping[str, object]) -> tuple[str, ...]:
            if not self.dr_strata:
                return ("*",)
            return tuple(str(meta.get(field, "unknown")) for field in self.dr_strata)

        strata: dict[tuple[str, ...], dict[str, list[float]]] = {}
        for score, meta in zip(control, control_meta):
            key = key_from_meta(meta)
            bucket = strata.setdefault(key, {"control": [], "treatment": []})
            bucket["control"].append(float(score))
        for score, meta in zip(treatment, treatment_meta):
            key = key_from_meta(meta)
            bucket = strata.setdefault(key, {"control": [], "treatment": []})
            bucket["treatment"].append(float(score))

        aligned_control: list[float] = []
        aligned_treatment: list[float] = []
        summary: dict[str, dict[str, int]] = {}
        for key, groups in strata.items():
            control_vals = groups.get("control", [])
            treatment_vals = groups.get("treatment", [])
            paired = min(len(control_vals), len(treatment_vals))
            if paired < self.dr_min_stratum:
                continue
            aligned_control.extend(control_vals[-paired:])
            aligned_treatment.extend(treatment_vals[-paired:])
            summary["|".join(key)] = {
                "control": len(control_vals),
                "treatment": len(treatment_vals),
                "paired": paired,
            }
        return aligned_control, aligned_treatment, summary

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
        control_meta: Iterable[Mapping[str, object]] | None = None,
        treatment_meta: Iterable[Mapping[str, object]] | None = None,
    ) -> AssimilationTestResult:
        control_list = list(control_scores)
        treatment_list = list(treatment_scores)
        control_meta_list = list(control_meta) if control_meta is not None else []
        treatment_meta_list = list(treatment_meta) if treatment_meta is not None else []
        energy_top_up_event: EnergyTopUpEvent | None = None
        if energy_top_up is not None:
            try:
                energy_top_up_event = EnergyTopUpEvent.model_validate(dict(energy_top_up))
            except Exception:
                energy_top_up_event = None

        control_arr = np.array(control_list, dtype=float)
        treatment_arr = np.array(treatment_list, dtype=float)
        min_len = min(len(control_arr), len(treatment_arr))
        min_required = max(1, self.min_samples)
        if min_len < min_required:
            event = AssimilationEvent(
                organelle_id=organelle_id,
                uplift=0.0,
                p_value=1.0,
                passed=False,
                energy_cost=energy_cost,
                safety_hits=safety_hits,
                sample_size=min_len if min_len > 0 else None,
                control_mean=float(control_arr.mean()) if len(control_arr) else None,
                treatment_mean=float(treatment_arr.mean()) if len(treatment_arr) else None,
                control_std=float(control_arr.std(ddof=0)) if len(control_arr) else None,
                treatment_std=float(treatment_arr.std(ddof=0)) if len(treatment_arr) else None,
                uplift_threshold=self.uplift_threshold,
                p_value_threshold=self.p_value_threshold,
                energy_balance=energy_balance,
                energy_top_up=energy_top_up_event,
                method="z_test",
            )
            return AssimilationTestResult(event=event, decision=False)

        if len(control_arr) != len(treatment_arr):
            control_arr = control_arr[:min_len]
            treatment_arr = treatment_arr[:min_len]
            if control_meta_list:
                control_meta_list = control_meta_list[:min_len]
            if treatment_meta_list:
                treatment_meta_list = treatment_meta_list[:min_len]

        control_list = control_arr.tolist()
        treatment_list = treatment_arr.tolist()
        dr_used = False
        strata_summary: dict[str, dict[str, int]] = {}
        if self.dr_enabled and control_meta_list and treatment_meta_list:
            aligned_control, aligned_treatment, strata_summary = self._align_dr(
                control_list,
                treatment_list,
                control_meta_list,
                treatment_meta_list,
            )
            if len(aligned_control) >= min_required and len(aligned_control) == len(
                aligned_treatment
            ):
                control_arr = np.array(aligned_control, dtype=float)
                treatment_arr = np.array(aligned_treatment, dtype=float)
                dr_used = True
            else:
                strata_summary = {}

        sample_size = len(control_arr)
        if sample_size < min_required:
            method = "dr" if dr_used else "z_test"
            event = AssimilationEvent(
                organelle_id=organelle_id,
                uplift=0.0,
                p_value=1.0,
                passed=False,
                energy_cost=energy_cost,
                safety_hits=safety_hits,
                sample_size=sample_size if sample_size > 0 else None,
                control_mean=float(control_arr.mean()) if sample_size else None,
                treatment_mean=float(treatment_arr.mean()) if sample_size else None,
                control_std=float(control_arr.std(ddof=0)) if sample_size else None,
                treatment_std=float(treatment_arr.std(ddof=0)) if sample_size else None,
                uplift_threshold=self.uplift_threshold,
                p_value_threshold=self.p_value_threshold,
                energy_balance=energy_balance,
                energy_top_up=energy_top_up_event,
                method=method,
                dr_used=dr_used,
                dr_strata=self.dr_strata if dr_used else [],
                dr_sample_sizes=strata_summary if dr_used else {},
            )
            return AssimilationTestResult(event=event, decision=False)

        control_mean = float(control_arr.mean())
        treatment_mean = float(treatment_arr.mean())
        diff = treatment_arr - control_arr
        uplift = float(diff.mean())
        ci_low: float | None = None
        ci_high: float | None = None
        power: float | None = None
        method = "dr" if dr_used else "z_test"

        if self.bootstrap_enabled and self.bootstrap_n > 0 and self.permutation_n > 0:
            rng = np.random.default_rng(1337)
            n = sample_size
            boot = []
            for _ in range(self.bootstrap_n):
                idx = rng.integers(0, n, size=n)
                boot.append(float(diff[idx].mean()))
            boot.sort()
            ci_low = boot[int(0.025 * len(boot))]
            ci_high = boot[int(0.975 * len(boot))]
            obs = abs(uplift)
            count = 0
            for _ in range(self.permutation_n):
                signs = rng.choice([1, -1], size=n)
                perm = abs(float((diff * signs).mean()))
                if perm >= obs:
                    count += 1
            p_value = (count + 1) / (self.permutation_n + 1)
            try:
                boot_std = float(np.std(boot, ddof=1))
                se = max(boot_std, 1e-12)
                delta = uplift - self.uplift_threshold
                alpha = max(min(self.p_value_threshold, 0.499), 1e-6)
                from math import sqrt

                def phi(x: float) -> float:
                    from math import erf

                    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

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
                z_effect = float(delta / se)
                power = max(0.0, min(1.0, 1.0 - float(phi(z_alpha - z_effect))))
            except Exception:
                power = None
            method = f"{method}+bootstrap"
        else:
            variance = float(np.var(diff, ddof=1)) + 1e-8
            se = math.sqrt(variance / max(sample_size, 1))
            z_score = uplift / max(se, 1e-12)
            p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
            ci_width = 1.96 * se
            ci_low = uplift - ci_width
            ci_high = uplift + ci_width
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

                def phi(x: float) -> float:
                    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

                power = max(
                    0.0,
                    min(
                        1.0,
                        1.0 - float(phi(z_alpha - (delta / max(se, 1e-12)))),
                    ),
                )
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
            control_std=float(control_arr.std(ddof=1)) if sample_size >= 2 else 0.0,
            treatment_mean=treatment_mean,
            treatment_std=float(treatment_arr.std(ddof=1)) if sample_size >= 2 else 0.0,
            ci_low=ci_low,
            ci_high=ci_high,
            power=power,
            uplift_threshold=self.uplift_threshold,
            p_value_threshold=self.p_value_threshold,
            energy_balance=energy_balance,
            energy_top_up=energy_top_up_event,
            method=method,
            dr_used=dr_used,
            dr_strata=self.dr_strata if dr_used else [],
            dr_sample_sizes=strata_summary if dr_used else {},
        )
        return AssimilationTestResult(event=event, decision=passed)


__all__ = ["AssimilationTestResult", "AssimilationTester"]
