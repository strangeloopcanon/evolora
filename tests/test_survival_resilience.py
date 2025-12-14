from __future__ import annotations

from statistics import mean

from symbiont_ecology import (
    AssimilationTester,
    ATPLedger,
    BanditRouter,
    EcologyConfig,
    HostKernel,
    PopulationManager,
)
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome


def _make_loop(seed: int = 2025, ticket: float = 0.6) -> tuple[EcologyConfig, EcologyLoop, str]:
    config = EcologyConfig()
    config.energy.m = ticket
    config.environment.synthetic_batch_size = 1
    config.assimilation_tuning.holdout_tasks_path = None
    host = HostKernel(config=config, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    population = PopulationManager(config.evolution, config.foraging)
    organelle_id = host.spawn_organelle(rank=2)
    population.register(
        Genome(
            organelle_id=organelle_id,
            drive_weights={"novelty": 0.5},
            gate_bias=0.0,
            rank=2,
        )
    )
    assimilation = AssimilationTester(
        uplift_threshold=config.evolution.assimilation_threshold,
        p_value_threshold=config.evolution.assimilation_p_value,
        safety_budget=0,
    )
    environment = GridEnvironment(
        grid_cfg=config.grid,
        controller_cfg=config.controller,
        pricing_cfg=config.pricing,
        canary_cfg=config.canary,
        seed=seed,
    )
    loop = EcologyLoop(
        config=config,
        host=host,
        environment=environment,
        population=population,
        assimilation=assimilation,
        human_bandit=None,
        sink=None,
    )
    return config, loop, organelle_id


def test_survival_recovers_after_ticket_spike() -> None:
    config, loop, _ = _make_loop()
    roi_window: list[float] = []
    for _ in range(3):
        summary = loop.run_generation(batch_size=1)
        roi_window.append(summary.get("avg_roi", 0.0))
    baseline_roi = mean(roi_window)
    # Perturb ticket and controller parameters
    config.energy.m = 1.2
    loop.host.ledger.configure_energy_cap(config.energy.Emax)
    loop.environment.controller.apply_parameters(tau=0.6)
    perturbed_roi: list[float] = []
    for _ in range(3):
        summary = loop.run_generation(batch_size=1)
        perturbed_roi.append(summary.get("avg_roi", 0.0))
    # Expect ROI to stabilise near or above baseline despite the perturbation.
    assert perturbed_roi[-1] > 0.0
    assert mean(perturbed_roi) >= baseline_roi * 0.5


def test_survival_handles_task_mix_shift() -> None:
    config, loop, _ = _make_loop(seed=77)
    baseline_roi: list[float] = []
    for _ in range(4):
        summary = loop.run_generation(batch_size=1)
        baseline_roi.append(summary.get("avg_roi", 0.0))
    # Shift pricing to make tasks harder and modify controller tau
    loop.environment.controller.apply_parameters(price_base=1.6, price_k=2.0, tau=0.4)
    config.pricing.base = 1.6
    config.pricing.k = 2.0
    config.controller.tau = 0.4
    shifted_roi: list[float] = []
    for _ in range(4):
        summary = loop.run_generation(batch_size=1)
        shifted_roi.append(summary.get("avg_roi", 0.0))
    assert shifted_roi[-1] > -0.2
    # ROI should not collapse entirely.
    assert mean(shifted_roi[-2:]) >= -0.05
