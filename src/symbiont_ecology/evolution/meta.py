"""Meta-evolution of environment/controller parameters."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Tuple

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.evolution.assimilation import AssimilationTester


@dataclass
class MetaEvolver:
    config: EcologyConfig
    environment: GridEnvironment
    assimilation: AssimilationTester
    rng: random.Random = field(default_factory=lambda: random.Random(90210))
    best_roi: float = float("-inf")
    awaiting_eval: bool = False
    last_roi: float = 0.0
    pending_params: Dict[str, float] | None = None
    baseline_params: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.baseline_params = self._capture_params()
        self.best_roi = float("-inf")

    def step(self, generation: int, avg_roi: float) -> dict:
        meta_cfg = self.config.meta
        if not meta_cfg.enabled:
            return {}

        info: dict = {}
        if meta_cfg.catastrophe_interval and generation % meta_cfg.catastrophe_interval == 0 and generation > 0:
            self.environment.catastrophic_shift(scale=meta_cfg.catastrophe_scale, rng=self.rng)
            info["catastrophe"] = True

        if self.awaiting_eval:
            if avg_roi >= self.last_roi:
                self.best_roi = max(self.best_roi, avg_roi)
                self.baseline_params = self._capture_params()
                info["meta_accept"] = True
            else:
                self._apply_params(self.baseline_params)
                info["meta_revert"] = True
            self.awaiting_eval = False
            self.pending_params = None

        if generation % meta_cfg.interval == 0 and not self.awaiting_eval:
            proposal = self._mutate_params(scale=meta_cfg.mutation_scale)
            self._apply_params(proposal)
            self.pending_params = proposal
            self.awaiting_eval = True
            self.last_roi = avg_roi if avg_roi > float("-inf") else 0.0
            info["meta_mutation"] = proposal
        return info

    # ------------------------------------------------------------------
    def _capture_params(self) -> Dict[str, float]:
        ctrl = self.environment.controller
        evo = self.config.evolution
        energy = self.config.energy
        pricing = self.config.pricing
        guard = self.config.assimilation_tuning
        return {
            "tau": ctrl.ctrl.tau,
            "beta": ctrl.ctrl.beta,
            "eta": ctrl.ctrl.eta,
            "price_base": pricing.base,
            "price_k": pricing.k,
            "ticket_m": energy.m,
            "assim_threshold": evo.assimilation_threshold,
            "assim_interval": float(self.config.assimilation_tuning.per_cell_interval),
            "energy_floor_base": guard.energy_floor_base,
            "energy_floor_roi_base": guard.energy_floor_roi_base,
        }

    def _apply_params(self, params: Dict[str, float]) -> None:
        ctrl = self.environment.controller
        ctrl.apply_parameters(
            tau=params["tau"],
            beta=params["beta"],
            eta=params["eta"],
            price_base=params["price_base"],
            price_k=params["price_k"],
        )
        self.config.pricing.base = ctrl.pricing.base
        self.config.pricing.k = ctrl.pricing.k
        self.config.controller.tau = ctrl.ctrl.tau
        self.config.controller.beta = ctrl.ctrl.beta
        self.config.controller.eta = ctrl.ctrl.eta

        self.config.energy.m = params["ticket_m"]
        self.config.evolution.assimilation_threshold = params["assim_threshold"]
        interval = int(round(params["assim_interval"]))
        interval = max(1, interval)
        self.config.assimilation_tuning.per_cell_interval = interval
        self.assimilation.update_thresholds(
            uplift_threshold=self.config.evolution.assimilation_threshold,
            p_value_threshold=self.assimilation.p_value_threshold,
        )
        guard = self.config.assimilation_tuning
        guard.energy_floor_base = max(0.0, params["energy_floor_base"])
        guard.energy_floor_roi_base = max(1.0, params["energy_floor_roi_base"])
        # seed actual guard values with new baseline to let dynamic tuner adapt from there
        guard.energy_floor = max(guard.energy_floor, guard.energy_floor_base)
        guard.energy_floor_roi = max(guard.energy_floor_roi, guard.energy_floor_roi_base)

    def _mutate_params(self, scale: float) -> Dict[str, float]:
        current = self._capture_params()
        bounds: Dict[str, Tuple[float, float]] = {
            "tau": (0.2, 0.8),
            "beta": (0.05, 0.6),
            "eta": (0.1, 1.5),
            "price_base": (0.2, 2.0),
            "price_k": (0.5, 3.5),
            "ticket_m": (0.3, 3.0),
            "assim_threshold": (0.0, 0.2),
            "assim_interval": (1.0, 10.0),
            "energy_floor_base": (0.2, 4.0),
            "energy_floor_roi_base": (1.0, 6.0),
        }
        proposal: Dict[str, float] = {}
        for key, value in current.items():
            low, high = bounds[key]
            perturb = 1.0 + self.rng.uniform(-scale, scale)
            mutated = value * perturb
            if key == "assim_interval":
                mutated = float(int(round(mutated)))
            proposal[key] = max(low, min(high, mutated))
        return proposal
