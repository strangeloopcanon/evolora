"""Telemetry models for logging and reporting."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class RouteEvent(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    organelle_id: str
    reward: float
    novelty: float
    competence_gain: float
    helper_credit: float
    risk_penalty: float
    cost_penalty: float
    atp_delta: float
    notes: Optional[str] = None
    tokens: int = 0
    latency_ms: float = 0.0


class EnergyTopUpEvent(BaseModel):
    status: str
    before: float
    after: float
    credited: float = 0.0
    roi: float = 0.0
    floor: float = 0.0
    roi_threshold: float = 0.0
    roi_threshold_effective: float = 0.0


class AssimilationEvent(BaseModel):
    organelle_id: str
    uplift: float
    p_value: float
    passed: bool
    energy_cost: float
    safety_hits: int
    sample_size: int | None = None
    control_mean: float | None = None
    control_std: float | None = None
    treatment_mean: float | None = None
    treatment_std: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    power: float | None = None
    uplift_threshold: float | None = None
    p_value_threshold: float | None = None
    energy_balance: float | None = None
    energy_top_up: EnergyTopUpEvent | None = None
    cell: dict[str, str] | None = None
    soup: list[dict[str, float]] = Field(default_factory=list)
    probes: list[dict[str, object]] = Field(default_factory=list)
    method: str = "z_test"
    dr_used: bool = False
    dr_strata: list[str] = Field(default_factory=list)
    dr_sample_sizes: dict[str, dict[str, int]] = Field(default_factory=dict)


class LiveEvalSummary(BaseModel):
    total_cases: int
    passed: int
    failed: int
    cost_usd: float
    latency_p95_ms: float
    mode: str
    notes: str = ""


class LedgerSnapshot(BaseModel):
    accounts: dict[str, float]
    total_atp: float
    energy: dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RewardBreakdown(BaseModel):
    task_reward: float
    novelty_bonus: float
    competence_bonus: float
    helper_bonus: float
    risk_penalty: float
    cost_penalty: float

    @property
    def total(self) -> float:
        return (
            self.task_reward
            + self.novelty_bonus
            + self.competence_bonus
            + self.helper_bonus
            - self.risk_penalty
            - self.cost_penalty
        )


class EpisodeLog(BaseModel):
    episode_id: str
    task_id: str
    organelles: list[str]
    rewards: RewardBreakdown
    energy_spent: float
    observations: dict[str, Any]


__all__ = [
    "AssimilationEvent",
    "EpisodeLog",
    "EnergyTopUpEvent",
    "LedgerSnapshot",
    "LiveEvalSummary",
    "RewardBreakdown",
    "RouteEvent",
]
