from symbiont_ecology import ATPLedger, AssimilationTester, BanditRouter, EcologyConfig, HostKernel, PopulationManager
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome


def test_energy_bankruptcy_skips_organelle(tmp_path) -> None:
    config = EcologyConfig()
    config.metrics.root = tmp_path
    ledger = ATPLedger()
    router = BanditRouter()
    host = HostKernel(config=config, router=router, ledger=ledger)
    host.freeze_host()

    population = PopulationManager(config.evolution, config.foraging)
    active = host.spawn_organelle(rank=2)
    bankrupt = host.spawn_organelle(rank=2)
    population.register(Genome(organelle_id=active, drive_weights={}, gate_bias=0.0, rank=2))
    population.register(Genome(organelle_id=bankrupt, drive_weights={}, gate_bias=0.0, rank=2))

    host.ledger.set_energy(bankrupt, 0.0)

    environment = GridEnvironment(
        grid_cfg=config.grid,
        controller_cfg=config.controller,
        pricing_cfg=config.pricing,
        canary_cfg=config.canary,
        seed=11,
    )
    loop = EcologyLoop(
        config=config,
        host=host,
        environment=environment,
        population=population,
        assimilation=AssimilationTester(
            uplift_threshold=config.evolution.assimilation_threshold,
            p_value_threshold=config.evolution.assimilation_p_value,
            safety_budget=0,
        ),
        human_bandit=None,
        sink=None,
    )

    summary = loop.run_generation(batch_size=1)
    assert population.history[bankrupt] == [0.0]
    assert population.energy[bankrupt] == [0.0]
    assert population.history[active]
    assert summary["bankrupt"] >= 1
    assert summary["culled_bankrupt"] == 0


def test_bankruptcy_culling_removes_organelle(tmp_path) -> None:
    config = EcologyConfig()
    config.metrics.root = tmp_path
    config.energy.bankruptcy_grace = 1
    ledger = ATPLedger()
    router = BanditRouter()
    host = HostKernel(config=config, router=router, ledger=ledger)
    host.freeze_host()

    population = PopulationManager(config.evolution, config.foraging)
    bankrupt = host.spawn_organelle(rank=2)
    population.register(Genome(organelle_id=bankrupt, drive_weights={}, gate_bias=0.0, rank=2))
    host.ledger.set_energy(bankrupt, 0.0)

    environment = GridEnvironment(
        grid_cfg=config.grid,
        controller_cfg=config.controller,
        pricing_cfg=config.pricing,
        canary_cfg=config.canary,
        seed=13,
    )
    loop = EcologyLoop(
        config=config,
        host=host,
        environment=environment,
        population=population,
        assimilation=AssimilationTester(
            uplift_threshold=config.evolution.assimilation_threshold,
            p_value_threshold=config.evolution.assimilation_p_value,
            safety_budget=0,
        ),
        human_bandit=None,
        sink=None,
    )

    summary = loop.run_generation(batch_size=1)
    assert summary["culled_bankrupt"] == 1
    assert bankrupt not in population.population
