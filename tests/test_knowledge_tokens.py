from types import SimpleNamespace

from symbiont_ecology import (
    AssimilationTester,
    ATPLedger,
    BanditRouter,
    EcologyConfig,
    HostKernel,
    PopulationManager,
)
from symbiont_ecology.environment.grid import GridEnvironment, GridTask
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome


def _make_loop(knowledge_enabled: bool = True) -> tuple[EcologyLoop, str]:
    config = EcologyConfig()
    config.knowledge.enabled = knowledge_enabled
    config.knowledge.write_cost = 0.0
    config.knowledge.read_cost = 0.0
    config.knowledge.ttl = 5
    host = HostKernel(config=config, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    population = PopulationManager(config.evolution, config.foraging)
    organelle_id = host.spawn_organelle(rank=2)
    population.register(
        Genome(
            organelle_id=organelle_id,
            drive_weights={},
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
        seed=3,
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
    loop._knowledge_stats_gen = {
        "writes": 0,
        "write_denied": 0,
        "reads": 0,
        "read_denied": 0,
        "hits": 0,
        "expired": 0,
    }
    loop.generation_index = 10
    host.ledger.set_energy(organelle_id, 5.0)
    return loop, organelle_id


def test_knowledge_write_and_read_flow():
    loop, organelle_id = _make_loop()
    task = GridTask(
        task_id="mem-1",
        cell=("math", "short"),
        prompt="What is 2+2?",
        price=1.0,
        target=4,
        family="math",
        depth="short",
        difficulty=0.1,
    )
    metrics = SimpleNamespace(answer="4")
    loop._record_knowledge_entry(organelle_id, task, metrics)
    assert organelle_id in loop._knowledge_store
    prompt = loop._prepare_knowledge_prompt(organelle_id, task)
    assert "Memory cache" in prompt
    assert loop._knowledge_stats_gen["writes"] == 1
    assert loop._knowledge_stats_gen["reads"] == 1
    assert loop._knowledge_stats_gen["hits"] == 1


def test_knowledge_prunes_expired_entries():
    loop, organelle_id = _make_loop()
    loop.config.knowledge.ttl = 1
    loop._knowledge_store[organelle_id] = [
        {"cell": ("math", "short"), "note": "stale", "gen": loop.generation_index - 5}
    ]
    loop._prune_knowledge_cache()
    assert organelle_id not in loop._knowledge_store
    assert loop._knowledge_stats_gen["expired"] >= 1


def test_knowledge_read_denied_without_energy():
    loop, organelle_id = _make_loop()
    loop.config.knowledge.read_cost = 1.0
    loop.host.ledger.set_energy(organelle_id, 0.1)
    loop._knowledge_store[organelle_id] = [
        {"cell": ("math", "short"), "note": "cached fact", "gen": loop.generation_index}
    ]
    task = GridTask(
        task_id="mem-2",
        cell=("math", "short"),
        prompt="What is 3+3?",
        price=1.0,
        target=6,
        family="math",
        depth="short",
        difficulty=0.1,
    )
    prompt = loop._prepare_knowledge_prompt(organelle_id, task)
    assert prompt == ""
    assert loop._knowledge_stats_gen["read_denied"] == 1


def test_knowledge_write_denied_without_energy():
    loop, organelle_id = _make_loop()
    loop.config.knowledge.write_cost = 1.5
    loop.host.ledger.set_energy(organelle_id, 0.5)
    task = GridTask(
        task_id="mem-3",
        cell=("math", "short"),
        prompt="What is 5+5?",
        price=1.0,
        target=10,
        family="math",
        depth="short",
        difficulty=0.1,
    )
    metrics = SimpleNamespace(answer="10")
    loop._record_knowledge_entry(organelle_id, task, metrics)
    assert organelle_id not in loop._knowledge_store
    assert loop._knowledge_stats_gen["write_denied"] == 1


def test_knowledge_respects_max_items():
    loop, organelle_id = _make_loop()
    loop.config.knowledge.max_items = 1
    task = GridTask(
        task_id="mem-4",
        cell=("math", "short"),
        prompt="What is 1+1?",
        price=1.0,
        target=2,
        family="math",
        depth="short",
        difficulty=0.1,
    )
    loop._record_knowledge_entry(organelle_id, task, SimpleNamespace(answer="2"))
    loop._record_knowledge_entry(organelle_id, task, SimpleNamespace(answer="2 again"))
    store = loop._knowledge_store.get(organelle_id, [])
    assert len(store) == 1
    assert "2 again" in store[0]["note"]
