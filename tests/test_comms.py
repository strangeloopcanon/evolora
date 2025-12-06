import random
from types import SimpleNamespace

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome, PopulationManager


class DummyLedger:
    def __init__(self, balances: dict[str, float], cap: float = 5.0) -> None:
        self._balances = dict(balances)
        self.energy_cap = cap

    def energy_balance(self, organelle_id: str) -> float:
        return self._balances.get(organelle_id, 0.0)

    def credit_energy(self, organelle_id: str, amount: float) -> float:
        updated = min(self.energy_cap, self._balances.get(organelle_id, 0.0) + amount)
        self._balances[organelle_id] = updated
        return updated

    def consume_energy(self, organelle_id: str, amount: float) -> bool:
        if self._balances.get(organelle_id, 0.0) < amount:
            return False
        self._balances[organelle_id] -= amount
        return True

    def set_energy(self, organelle_id: str, value: float) -> None:
        self._balances[organelle_id] = min(self.energy_cap, value)


class DummyHost:
    def __init__(self, balances: dict[str, float]) -> None:
        self.ledger = DummyLedger(balances)

    def list_organelle_ids(self) -> list[str]:
        return list(self.ledger._balances.keys())

    def get_organelle(self, organelle_id: str):
        return SimpleNamespace(rank=2)

    def resize_organelle_rank(self, organelle_id: str, new_rank: int) -> bool:
        return True

    def retire_organelle(self, organelle_id: str) -> None:
        self.ledger._balances.pop(organelle_id, None)


class SpawningHost(DummyHost):
    def __init__(self, balances: dict[str, float]) -> None:
        super().__init__(balances)
        self._next_idx = 0

    def spawn_organelle(self, rank: int, hebbian_config=None, activation_bias: float = 0.0) -> str:
        self._next_idx += 1
        organelle_id = f"child_{self._next_idx}"
        self.ledger.set_energy(organelle_id, 0.0)
        return organelle_id


class TrialHost(SpawningHost):
    def __init__(self, balances: dict[str, float], ranks: dict[str, int] | None = None) -> None:
        super().__init__(balances)
        self._orgs: dict[str, SimpleNamespace] = {}
        for oid, balance in balances.items():
            rank = 2 if ranks is None else ranks.get(oid, 2)
            self._orgs[oid] = SimpleNamespace(rank=rank, import_adapter_state=lambda *args, **kwargs: None)
            self.ledger.set_energy(oid, balance)

    def get_organelle(self, organelle_id: str):
        if organelle_id not in self._orgs:
            self._orgs[organelle_id] = SimpleNamespace(rank=2, import_adapter_state=lambda *args, **kwargs: None)
        return self._orgs[organelle_id]

    def build_lora_soup_state(self, soup: dict[str, float], target_rank: int):
        return {"mix": soup, "rank": target_rank}, 1.0

    def spawn_organelle(self, rank: int, hebbian_config=None, activation_bias: float = 0.0) -> str:
        organelle_id = super().spawn_organelle(rank, hebbian_config=hebbian_config, activation_bias=activation_bias)
        self._orgs[organelle_id] = SimpleNamespace(rank=rank, import_adapter_state=lambda *args, **kwargs: None)
        return organelle_id


def _make_comms_loop() -> EcologyLoop:
    cfg = EcologyConfig()
    cfg.comms.enabled = True
    cfg.comms.credit_power_window = 3
    cfg.comms.credit_power_min_delta = 0.05
    cfg.meta.enabled = False
    host = DummyHost({"poster": 2.0, "reader": 2.0})
    environment = SimpleNamespace(
        rng=random.Random(42),
        controller=SimpleNamespace(cells={}, lp_progress={}),
        organism_stats={},
        read_messages=lambda max_items=1, **kwargs: [],
        read_caches=lambda max_items=1: [],
        post_message=lambda *args, **kwargs: True,
        canary_failed=lambda oid: False,
        best_cell_score=lambda oid: (("math", "short"), 0.5),
    )
    pop = PopulationManager(cfg.evolution, cfg.foraging)
    pop.register(Genome(organelle_id="poster", drive_weights={}, gate_bias=0.0, rank=2))
    pop.register(Genome(organelle_id="reader", drive_weights={}, gate_bias=0.0, rank=2))
    pop.roi["poster"] = [0.2, 0.25]
    pop.roi["reader"] = [0.1, 0.12, 0.15]
    loop = EcologyLoop(cfg, host, environment, pop, assimilation=SimpleNamespace())
    loop.generation_index = 7
    loop._comms_stats_gen = {"posts": 0, "reads": 0, "credits": 0}
    loop._comms_events_gen = []
    return loop


def _make_grid_environment() -> GridEnvironment:
    cfg = EcologyConfig()
    env = GridEnvironment(cfg.grid, cfg.controller, cfg.pricing, cfg.canary, seed=123)
    env.comms_history_cap = 3
    env.default_comm_priority = 0.05
    return env


def test_build_comms_hint_uses_stats_and_traits():
    loop = _make_comms_loop()
    loop.environment.organism_stats = {
        "poster": {
            ("math", "short"): 0.72,
            ("word.count", "all"): 0.41,
        }
    }
    loop.population.roi["poster"].append(0.66)
    genome = loop.population.population["poster"]
    genome.explore_rate = 0.4  # type: ignore[attr-defined]
    genome.hint_weight = 0.25  # type: ignore[attr-defined]

    hint = loop._build_comms_hint("poster", ("math", "short"))
    assert "best=math:short@0.72" in hint
    assert "roi=0.66" in hint
    assert "explore=0.40" in hint
    assert "hint=0.25" in hint


def test_build_comms_hint_falls_back_to_topic():
    loop = _make_comms_loop()
    hint = loop._build_comms_hint("poster", ("word.count", "all"))
    assert hint.startswith("focus=word.count:all")
    assert "roi=" in hint  # recent ROI still included


def test_grid_message_board_priority_and_topics():
    env = _make_grid_environment()
    env.comms_history_cap = 4
    env.post_message("low", "low priority", ttl=3, priority=0.1, topic="word.count:all")
    env.post_message("mid", "mid priority", ttl=3, priority=0.6, topic="math:short")
    env.post_message("high", "still relevant", ttl=3, priority=0.4, topic="math:short")

    batch = env.read_messages(max_items=2, topics=["math:short"], reader="alpha")
    assert [msg["organelle_id"] for msg in batch] == ["mid", "high"]
    assert batch[0]["ttl"] == 2
    assert batch[0]["reads"] == 1

    second = env.read_messages(max_items=2, topics=["math:short"], reader="alpha")
    assert [msg["organelle_id"] for msg in second] == ["low"]
    assert second[0]["ttl"] == 2

    snapshot = env.peek_messages(limit=2)
    assert snapshot
    assert "priority" in snapshot[0]


def test_grid_message_board_respects_history_cap():
    env = _make_grid_environment()
    env.comms_history_cap = 2
    env.post_message("a", "first", ttl=3, priority=0.1)
    env.post_message("b", "second", ttl=3, priority=0.2)
    env.post_message("c", "third", ttl=3, priority=0.3)
    board_ids = {entry["organelle_id"] for entry in env.message_board}
    assert len(env.message_board) == 2
    assert "a" not in board_ids


def test_comms_credit_awarded_on_power_gain():
    loop = _make_comms_loop()
    loop._queue_comms_credit("poster", "reader", baseline=0.1, credit_amount=0.2)
    # simulate improvement
    loop.population.roi["reader"].extend([0.3, 0.35])
    loop._process_comms_credit()
    assert loop._comms_stats_gen["credits"] == 1
    assert loop.host.ledger.energy_balance("poster") > 1.0
    assert any(ev["type"] == "credit" for ev in loop._comms_events_gen)


def test_comms_credit_waits_for_improvement():
    loop = _make_comms_loop()
    loop._queue_comms_credit("poster", "reader", baseline=0.14, credit_amount=0.2)
    loop.population.roi["reader"].append(0.15)
    loop._process_comms_credit()
    assert loop._comms_stats_gen["credits"] == 0
    assert len(loop._comms_credit_queue) == 1


def test_comms_read_post_in_run_generation():
    loop = _make_comms_loop()
    loop.population.population["reader"].read_rate = 1.0
    loop.population.population["poster"].post_rate = 1.0

    messages = [{"organelle_id": "poster", "text": "clue"}]

    def read_messages(max_items: int = 1, **kwargs) -> list[dict[str, str]]:
        nonlocal messages
        if not messages or max_items <= 0:
            return []
        batch = messages[:max_items]
        messages = []
        return batch

    loop.environment.read_messages = read_messages
    loop.environment.read_caches = lambda max_items=1: []
    loop._attempt_assimilation = lambda capped=None: 0
    loop._review_trial_offspring = lambda: None
    loop._maybe_team_probes = lambda: 0
    loop._maybe_promote_colonies = lambda: None
    loop._tick_colonies = lambda: None
    loop._enforce_diversity = lambda: None
    loop._compute_batch_size = lambda default: 0
    loop._resolve_per_org_batch = lambda oid, base: 0
    loop._compute_viability_map = lambda: {oid: True for oid in loop.population.population}
    loop._mu_lambda_selection = lambda viability: list(loop.population.population.values())
    loop._apply_morphogenesis = lambda survivors: None
    loop._spawn_offspring = lambda survivors: None
    loop._compute_budget_map = lambda active, bs: (
        {oid: 0 for oid in active},
        {"global_cap": 0, "capped_total": 0},
    )
    loop._should_boost = lambda oid: False
    loop.run_generation(batch_size=0)
    assert loop._comms_stats_gen["reads"] >= 1
    assert loop._comms_stats_gen["posts"] >= 1


def test_spawn_offspring_preserves_comms_traits():
    cfg = EcologyConfig()
    host = SpawningHost({"parent": 1.0})
    environment = SimpleNamespace()
    pop = PopulationManager(cfg.evolution, cfg.foraging)
    parent = Genome(
        organelle_id="parent",
        drive_weights={},
        gate_bias=0.0,
        rank=2,
        explore_rate=0.3,
        post_rate=0.7,
        read_rate=0.4,
        hint_weight=0.2,
    )
    pop.register(parent)
    loop = EcologyLoop(cfg, host, environment, pop, assimilation=SimpleNamespace())

    mutant = Genome(
        organelle_id="parent",
        drive_weights={"logic_focus": 0.5},
        gate_bias=0.1,
        rank=3,
        explore_rate=0.5,
        post_rate=0.9,
        read_rate=0.6,
        hint_weight=0.45,
    )
    pop.mutate = lambda genome: mutant  # type: ignore[assignment]

    loop._spawn_replacement_from(parent)
    child_ids = [oid for oid in pop.population if oid != "parent"]
    assert child_ids, "expected new child genome to be registered"
    child = pop.population[child_ids[0]]
    assert child.post_rate == mutant.post_rate
    assert child.read_rate == mutant.read_rate
    assert child.explore_rate == mutant.explore_rate
    assert child.hint_weight == mutant.hint_weight


def test_trial_offspring_inherit_average_comms_traits():
    cfg = EcologyConfig()
    host = TrialHost({"org_a": 1.0, "org_b": 1.0})
    environment = SimpleNamespace()
    pop = PopulationManager(cfg.evolution, cfg.foraging)
    pop.register(
        Genome(
            organelle_id="org_a",
            drive_weights={},
            gate_bias=0.0,
            rank=2,
            explore_rate=0.2,
            post_rate=0.9,
            read_rate=0.6,
            hint_weight=0.4,
        )
    )
    pop.register(
        Genome(
            organelle_id="org_b",
            drive_weights={},
            gate_bias=0.0,
            rank=2,
            explore_rate=0.4,
            post_rate=0.3,
            read_rate=0.2,
            hint_weight=0.1,
        )
    )
    loop = EcologyLoop(cfg, host, environment, pop, assimilation=SimpleNamespace())
    stats_map = {
        "org_a": {"roi": 0.4, "ema": 0.5},
        "org_b": {"roi": 0.5, "ema": 0.6},
    }
    child_id = loop._create_trial_offspring(("math", "short"), "org_a", ["org_a", "org_b"], stats_map)
    assert child_id is not None
    child = pop.population[child_id]
    assert abs(child.post_rate - 0.6) < 1e-6
    assert abs(child.read_rate - 0.4) < 1e-6
    assert abs(child.explore_rate - 0.3) < 1e-6
    assert abs(child.hint_weight - 0.25) < 1e-6
