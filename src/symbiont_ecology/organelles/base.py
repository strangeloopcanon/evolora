"""Base organelle abstractions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from symbiont_ecology.config import HebbianConfig
from symbiont_ecology.interfaces.messages import MessageEnvelope
from symbiont_ecology.metrics.telemetry import RewardBreakdown


@dataclass
class OrganelleContext:
    organelle_id: str
    hebbian: HebbianConfig
    reward_baseline: float = 0.0
    traces: dict[str, torch.Tensor] | None = None


class Organelle:
    """Abstract organelle with local drives and plasticity."""

    def __init__(self, organelle_id: str, context: OrganelleContext) -> None:
        self.organelle_id = organelle_id
        self.context = context
        self._steps = 0

    def route_probability(self, observation: MessageEnvelope) -> float:
        """Return probability of being routed for current observation."""
        raise NotImplementedError  # pragma: no cover

    def forward(self, envelope: MessageEnvelope) -> MessageEnvelope:
        """Produce actions/plan updates."""
        raise NotImplementedError  # pragma: no cover

    def update(self, envelope: MessageEnvelope, reward: RewardBreakdown) -> None:
        """Apply plastic update based on reward."""
        raise NotImplementedError  # pragma: no cover

    # ------------------------------------------------------------------
    def export_adapter_state(self) -> dict[str, torch.Tensor]:
        """Return a lightweight snapshot of adapter weights for assimilation.

        Defaults to empty dict when the organelle does not expose trainable
        adapters. Sub-classes overriding this method should return CPU tensors
        so the host may persist snapshots without retaining device memory.
        """

        return {}

    def import_adapter_state(self, state: dict[str, torch.Tensor], alpha: float = 1.0) -> None:
        """Blend an adapter snapshot into the organelle.

        Args:
            state: Mapping produced by :meth:`export_adapter_state`.
            alpha: Mixing factor applied to the incoming snapshot.
        """

        raise NotImplementedError  # pragma: no cover

    def fisher_importance(self) -> float:
        """Return an approximate Fisher importance score for the organelle.

        Sub-classes with access to activation statistics should override this
        to expose a positive value used when weighting LoRA soups. The base
        implementation returns ``0.0`` to signal that no estimate is
        available.
        """

        return 0.0

    @property
    def steps(self) -> int:
        return self._steps

    def step(self) -> None:
        self._steps += 1


__all__ = ["Organelle", "OrganelleContext"]
