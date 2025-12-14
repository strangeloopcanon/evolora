"""Public economics interfaces for cross-module use."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from symbiont_ecology.config import EnergyConfig


class CostMetrics(Protocol):
    flops_estimate: float
    memory_gb: float
    latency_ms: float
    trainable_params: int


class RouteCostBreakdown(BaseModel):
    flops_cost: float
    memory_cost: float
    latency_cost: float
    params_cost: float
    base_cost: float
    price_multiplier: float
    cost_scale: float
    total_cost: float


def compute_route_cost(
    config: EnergyConfig,
    metrics: CostMetrics,
    *,
    price_multiplier: float = 1.0,
    cost_scale: float | None = None,
) -> RouteCostBreakdown:
    multiplier = float(price_multiplier)
    scale = float(config.cost_scale if cost_scale is None else cost_scale)
    scale = max(0.0, min(scale, 1.0))

    flops_cost = float(config.alpha) * float(metrics.flops_estimate)
    memory_cost = float(config.beta) * float(metrics.memory_gb)
    latency_cost = float(config.gamma) * float(metrics.latency_ms)
    params_cost = float(config.lambda_p) * float(metrics.trainable_params)
    base_cost = flops_cost + memory_cost + latency_cost + params_cost
    total_cost = base_cost * multiplier * scale
    return RouteCostBreakdown(
        flops_cost=flops_cost,
        memory_cost=memory_cost,
        latency_cost=latency_cost,
        params_cost=params_cost,
        base_cost=base_cost,
        price_multiplier=multiplier,
        cost_scale=scale,
        total_cost=total_cost,
    )
