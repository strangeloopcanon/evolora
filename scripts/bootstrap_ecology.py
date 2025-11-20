#!/usr/bin/env python3
"""Entry point to bootstrap a tiny colony."""
from __future__ import annotations

from pathlib import Path

from symbiont_ecology import (
    ATPLedger,
    AssimilationTester,
    BanditRouter,
    HostKernel,
    PopulationManager,
    TelemetrySink,
    load_ecology_config,
)
from symbiont_ecology.environment.human import HumanBandit
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "ecology.yaml"
    config = load_ecology_config(config_path)
    ledger = ATPLedger()
    router = BanditRouter()
    host = HostKernel(config=config, router=router, ledger=ledger)
    host.freeze_host()
    population = PopulationManager(config.evolution, config.foraging)
    for _ in range(3):
        organelle_id = host.spawn_organelle(rank=4)
        population.register(
            Genome(
                organelle_id=organelle_id,
                drive_weights={"novelty": 0.5},
                gate_bias=0.0,
                rank=4,
            )
        )
    assimilation = AssimilationTester(
        uplift_threshold=config.evolution.assimilation_threshold,
        p_value_threshold=config.evolution.assimilation_p_value,
        safety_budget=0,
    )
    sink = TelemetrySink(
        root=config.metrics.root,
        episodes_file=config.metrics.episodes_file,
        assimilation_file=config.metrics.assimilation_file,
    )
    human_bandit = None
    if config.human_bandit.enabled:
        human_bandit = HumanBandit(
            preference_weight=config.human_bandit.preference_weight,
            helper_weight=config.human_bandit.helper_weight,
            frequency=config.human_bandit.frequency,
        )
    environment = GridEnvironment(
        grid_cfg=config.grid,
        controller_cfg=config.controller,
        pricing_cfg=config.pricing,
        canary_cfg=config.canary,
        reward_bonus=config.environment.success_reward_bonus,
        failure_cost_multiplier=config.environment.failure_cost_multiplier,
    )
    loop = EcologyLoop(
        config=config,
        host=host,
        environment=environment,
        population=population,
        assimilation=assimilation,
        human_bandit=human_bandit,
        sink=sink,
    )
    summary = loop.run_generation(batch_size=config.environment.synthetic_batch_size)
    print(
        "Completed generation with",
        len(loop.logs),
        "episodes logged.",
        "Summary:",
        summary,
    )


if __name__ == "__main__":
    main()
