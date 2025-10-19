from symbiont_ecology import (
    AssimilationTester,
    ATPLedger,
    BanditRouter,
    EcologyConfig,
    HostKernel,
    PopulationManager,
    TelemetrySink,
)
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.environment.human import HumanBandit
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome


def test_ecology_loop_runs_generation(tmp_path) -> None:
    config = EcologyConfig()
    host = HostKernel(config=config, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    population = PopulationManager(config.evolution)
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
    sink = TelemetrySink(
        root=tmp_path,
        episodes_file="episodes.jsonl",
        assimilation_file="assimilation.jsonl",
    )
    environment = GridEnvironment(
        grid_cfg=config.grid,
        controller_cfg=config.controller,
        pricing_cfg=config.pricing,
        canary_cfg=config.canary,
        seed=3,
    )
    loop = EcologyLoop(
        config=config,
        host=host,
        environment=environment,
        population=population,
        assimilation=assimilation,
        human_bandit=HumanBandit(),
        sink=sink,
    )
    summary = loop.run_generation(batch_size=2)
    assert loop.logs, "Expected logs to be recorded"
    assert (tmp_path / "episodes.jsonl").exists()
    first_log = loop.logs[0]
    metrics = first_log.observations.get("metrics", {})
    assert "active_adapters" in metrics
    assert "adapter_utilisation" in metrics
    assert summary["population"] == len(population.population)


def test_human_bandit_frequency_gating() -> None:
    config = EcologyConfig()
    config.human_bandit.frequency = 0.5
    host = HostKernel(config=config, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    population = PopulationManager(config.evolution)
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
    loop = EcologyLoop(
        config=config,
        host=host,
        environment=GridEnvironment(
            grid_cfg=config.grid,
            controller_cfg=config.controller,
            pricing_cfg=config.pricing,
            canary_cfg=config.canary,
            seed=9,
        ),
        population=population,
        assimilation=assimilation,
        human_bandit=HumanBandit(
            preference_weight=config.human_bandit.preference_weight,
            helper_weight=config.human_bandit.helper_weight,
            frequency=config.human_bandit.frequency,
        ),
        sink=None,
    )
    responses = {organelle_id: ("hello", 0.1)}
    first = loop._collect_human_feedback("test", responses)
    second = loop._collect_human_feedback("test", responses)
    assert first is not None
    assert second is None
