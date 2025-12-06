from types import SimpleNamespace

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome, PopulationManager


def test_population_refresh_retires_lowest_roi():
    cfg = EcologyConfig()
    cfg.population_strategy.refresh_interval = 1
    cfg.population_strategy.refresh_count = 1
    host_calls = {"retired": [], "spawned": []}

    def _spawn_organelle(rank: int, hebbian_config=None, activation_bias: float = 0.0):  # noqa: ARG001
        new_id = f"child_{len(host_calls['spawned'])}"
        host_calls["spawned"].append(new_id)
        return new_id

    host = SimpleNamespace(
        ledger=SimpleNamespace(),
        spawn_organelle=_spawn_organelle,
        retire_organelle=lambda oid: host_calls["retired"].append(oid),
        list_organelle_ids=lambda: ["low", "high"],
    )
    environment = SimpleNamespace(controller=SimpleNamespace(cells={}), rng=SimpleNamespace())
    pop = PopulationManager(cfg.evolution, cfg.foraging)
    pop.register(Genome(organelle_id="low", drive_weights={}, gate_bias=0.0, rank=2, explore_rate=0.5))
    pop.register(Genome(organelle_id="high", drive_weights={}, gate_bias=0.0, rank=2, explore_rate=0.5))
    loop = EcologyLoop(cfg, host, environment, pop, assimilation=SimpleNamespace())
    loop.population.record_roi("low", 0.1)
    loop.population.record_roi("high", 2.0)
    loop.no_merge_counter = 2
    survivors = [loop.population.population["high"]]

    loop._maybe_refresh_population(survivors)

    assert host_calls["retired"] == ["low"]
    assert any(child.startswith("child_") for child in host_calls["spawned"])
    assert any(gen.organelle_id.startswith("child_") for gen in loop.population.population.values())
