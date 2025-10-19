from symbiont_ecology import ATPLedger, BanditRouter, EcologyConfig, HostKernel, PopulationManager
from symbiont_ecology.evolution.morphogenesis import MorphogenesisController
from symbiont_ecology.evolution.population import Genome


def test_morphogenesis_grow_and_shrink() -> None:
    config = EcologyConfig()
    config.limits.lora_budget_frac = 1.0
    host = HostKernel(config=config, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    population = PopulationManager(config.evolution)

    organelle_id = host.spawn_organelle(rank=2)
    genome = Genome(
        organelle_id=organelle_id,
        drive_weights={"novelty": 0.5},
        gate_bias=0.0,
        rank=2,
    )
    population.register(genome)
    population.record_roi(organelle_id, 1.5)

    controller = MorphogenesisController(config=config, host=host)
    controller.apply([genome], population)
    grown_organelle = host.get_organelle(organelle_id)
    assert grown_organelle is not None
    grown_rank = getattr(grown_organelle, "get_rank")()  # type: ignore[misc]
    assert grown_rank >= 3
    assert genome.rank == grown_rank

    population.roi[organelle_id] = [0.4]
    controller.apply([genome], population)
    shrunk_organelle = host.get_organelle(organelle_id)
    assert shrunk_organelle is not None
    shrunk_rank = getattr(shrunk_organelle, "get_rank")()  # type: ignore[misc]
    assert shrunk_rank == max(1, grown_rank - 1)


def test_morphogenesis_enforces_layer_caps() -> None:
    config = EcologyConfig()
    config.limits.lora_budget_frac = 1.0
    config.limits.max_active_adapters_per_layer = 1
    host = HostKernel(config=config, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    population = PopulationManager(config.evolution)

    first_id = host.spawn_organelle(rank=3)
    second_id = host.spawn_organelle(rank=3)
    first_genome = Genome(organelle_id=first_id, drive_weights={}, gate_bias=0.0, rank=3)
    second_genome = Genome(organelle_id=second_id, drive_weights={}, gate_bias=0.0, rank=3)
    population.register(first_genome)
    population.register(second_genome)
    population.record_roi(first_id, 1.0)
    population.record_roi(second_id, 1.0)

    first_organelle = host.get_organelle(first_id)
    second_organelle = host.get_organelle(second_id)
    assert first_organelle is not None
    assert second_organelle is not None

    population.record_adapter_usage(first_id, host._active_adapters(first_organelle), tokens=128)
    population.record_adapter_usage(second_id, host._active_adapters(second_organelle), tokens=4)

    controller = MorphogenesisController(config=config, host=host)
    controller.apply([first_genome, second_genome], population)

    updated_first = host.get_organelle(first_id)
    updated_second = host.get_organelle(second_id)
    assert updated_first is not None
    assert updated_second is not None
    assert getattr(updated_first, "get_rank")() >= 3  # type: ignore[misc]
    assert getattr(updated_second, "get_rank")() < 3  # type: ignore[misc]
