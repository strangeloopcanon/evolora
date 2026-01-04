"""Telemetry models for logging and reporting."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

SoupEntry = Union[float, str, dict[str, Any], list[Any]]


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
    soup: list[SoupEntry] = Field(default_factory=list)
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


class ComputeBudget(BaseModel):
    """Tracks cumulative compute for fair comparison with SFT."""

    total_tokens: int = 0
    total_forward_passes: int = 0
    total_hebbian_updates: int = 0
    total_flops: float = 0.0
    total_generated_tokens: int = 0
    wall_clock_seconds: float = 0.0

    def record_forward(self, prompt_tokens: int, generated_tokens: int, flops: float) -> None:
        """Record a forward pass through the model."""
        self.total_tokens += prompt_tokens + generated_tokens
        self.total_generated_tokens += generated_tokens
        self.total_forward_passes += 1
        self.total_flops += flops

    def record_hebbian_update(self, estimated_flops: float = 0.0) -> None:
        """Record a Hebbian weight update (analogous to a gradient step)."""
        self.total_hebbian_updates += 1
        self.total_flops += estimated_flops

    def add_wall_clock(self, seconds: float) -> None:
        """Accumulate wall-clock training time."""
        self.wall_clock_seconds += seconds

    def summary(self) -> dict[str, object]:
        """Return a dict suitable for logging/checkpointing."""
        return {
            "total_tokens": self.total_tokens,
            "total_forward_passes": self.total_forward_passes,
            "total_hebbian_updates": self.total_hebbian_updates,
            "total_flops": self.total_flops,
            "total_generated_tokens": self.total_generated_tokens,
            "wall_clock_seconds": self.wall_clock_seconds,
        }


__all__ = [
    "AssimilationEvent",
    "ComputeBudget",
    "EpisodeLog",
    "EnergyTopUpEvent",
    "LedgerSnapshot",
    "LiveEvalSummary",
    "RewardBreakdown",
    "RouteEvent",
]
