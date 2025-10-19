#!/usr/bin/env python3
"""Evaluation run using the Gemma 270M host backbone."""

from __future__ import annotations

from pathlib import Path

from symbiont_ecology import (
    ATPLedger,
    AssimilationTester,
    BanditRouter,
    HostKernel,
    HumanBandit,
    PopulationManager,
    TelemetrySink,
    load_ecology_config,
)
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome

GENERATIONS = 5


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "ecology.yaml"
    config = load_ecology_config(config_path)
    config.host.backbone_type = "gemma"
    config.host.backbone_model = "google/gemma-3-270m-it"
    config.host.tokenizer = "google/gemma-3-270m-it"
    config.host.revision = None  # pin a revision when running in production
    config.host.device = "auto"
    config.metrics.root = Path("artifacts_gemma_eval")

    ledger = ATPLedger()
    router = BanditRouter()
    host = HostKernel(config=config, router=router, ledger=ledger)
    host.freeze_host()

    population = PopulationManager(config.evolution)
    for _ in range(3):
        organelle_id = host.spawn_organelle(rank=config.host.max_lora_rank)
        population.register(
            Genome(
                organelle_id=organelle_id,
                drive_weights={"novelty": 0.5},
                gate_bias=0.0,
                rank=config.host.max_lora_rank,
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

    environment = GridEnvironment(
        grid_cfg=config.grid,
        controller_cfg=config.controller,
        pricing_cfg=config.pricing,
        canary_cfg=config.canary,
        seed=42,
        reward_bonus=config.environment.success_reward_bonus,
        failure_cost_multiplier=config.environment.failure_cost_multiplier,
    )

    human_bandit = None
    if config.human_bandit.enabled:
        human_bandit = HumanBandit(
            preference_weight=config.human_bandit.preference_weight,
            helper_weight=config.human_bandit.helper_weight,
            frequency=config.human_bandit.frequency,
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

    for generation in range(GENERATIONS):
        summary = loop.run_generation(batch_size=config.environment.synthetic_batch_size)
        print(
            f"Generation {generation + 1}: population={len(population.population)} episodes={len(loop.logs)}",
            summary,
        )

    print("Telemetry written to", config.metrics.root)


if __name__ == "__main__":
    main()
