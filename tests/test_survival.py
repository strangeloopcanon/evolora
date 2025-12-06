from types import SimpleNamespace

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.assimilation import AssimilationTester
from symbiont_ecology.evolution.population import Genome, PopulationManager


class DummyLedger:
    def __init__(self, balances: dict[str, float], cap: float = 5.0) -> None:
        self._balances = dict(balances)
        self.energy_cap = cap

    def energy_balance(self, organelle_id: str) -> float:
        return self._balances.get(organelle_id, 0.0)

    def credit_energy(self, organelle_id: str, amount: float) -> float:
        new_val = min(self.energy_cap, self._balances.get(organelle_id, 0.0) + amount)
        self._balances[organelle_id] = new_val
        return new_val

    def consume_energy(self, organelle_id: str, amount: float) -> bool:
        if self._balances.get(organelle_id, 0.0) < amount:
            return False
        self._balances[organelle_id] -= amount
        return True

    def set_energy(self, organelle_id: str, value: float) -> None:
        self._balances[organelle_id] = min(self.energy_cap, value)


class DummyHost:
    def __init__(self, balances: dict[str, float], ranks: dict[str, int] | None = None) -> None:
        self.ledger = DummyLedger(balances)
        self._ranks = {oid: ranks.get(oid, 2) for oid in balances} if ranks else {oid: 2 for oid in balances}

    def list_organelle_ids(self) -> list[str]:
        return list(self._ranks.keys())

    def get_organelle(self, organelle_id: str):
        if organelle_id not in self._ranks:
            return None
        return SimpleNamespace(rank=self._ranks[organelle_id])

    def resize_organelle_rank(self, organelle_id: str, new_rank: int) -> bool:
        if organelle_id not in self._ranks:
            return False
        self._ranks[organelle_id] = new_rank
        return True

    def retire_organelle(self, organelle_id: str) -> None:
        self._ranks.pop(organelle_id, None)
        self.ledger._balances.pop(organelle_id, None)


def _make_loop(balances: dict[str, float]) -> tuple[EcologyLoop, PopulationManager]:
    cfg = EcologyConfig()
    cfg.survival.enabled = True
    cfg.survival.reserve_ratio = 0.7
    cfg.survival.reserve_cost_beta = 1.0
    cfg.survival.reserve_cost_window = 4
    cfg.survival.hazard_threshold = -0.5
    cfg.survival.hazard_exit_threshold = -0.1
    cfg.survival.hazard_roi_relief_boost = 0.05
    cfg.survival.hazard_topup_bonus = 0.4
    cfg.survival.min_power_recovery = 0.2
    cfg.assimilation_tuning.energy_floor = 1.0
    cfg.assimilation_tuning.energy_floor_roi = 1.2
    cfg.assimilation_tuning.energy_topup_roi_bonus = 0.0

    host = DummyHost(balances)
    environment = SimpleNamespace(
        controller=SimpleNamespace(cells={}),
        rng=SimpleNamespace(random=lambda: 0.0),
    )
    pop = PopulationManager(cfg.evolution, cfg.foraging)
    for oid in balances:
        pop.register(Genome(organelle_id=oid, drive_weights={}, gate_bias=0.0, rank=3))
        pop.energy[oid] = [0.6, 0.7, 0.5]
        pop.roi[oid] = [0.2, -0.2, -0.6, -0.8]

    loop = EcologyLoop(
        config=cfg,
        host=host,
        environment=environment,
        population=pop,
        assimilation=SimpleNamespace(),
    )
    loop.generation_index = 5
    return loop, pop


def test_update_survival_states_flags_reserve():
    loop, _pop = _make_loop({"org_a": 0.2})
    loop.population.roi["org_a"] = [0.1, 0.0, 0.05, 0.02]

    loop._update_survival_states(["org_a"])

    reserve_state = loop._reserve_state.get("org_a")
    assert reserve_state is not None and reserve_state["active"] is True
    assert any(ev["type"] == "reserve_enter" for ev in loop._survival_events)


def test_hazard_activation_adjusts_relief_and_rank():
    loop, pop = _make_loop({"org_h": 0.6})
    pop.roi["org_h"] = [-0.6, -0.7, -0.8, -0.9]

    loop._update_survival_states(["org_h"])

    hazard_state = loop._hazard_state.get("org_h")
    assert hazard_state is not None and hazard_state["active"] is True
    assert loop._roi_relief.get("org_h", 0.0) > 0.0
    assert any(ev["type"] == "hazard_enter" for ev in loop._survival_events)


def test_reserve_blocks_assimilation_attempt():
    loop, _pop = _make_loop({"org_r": 0.3})
    loop.population.roi["org_r"] = [0.2, 0.1, 0.05, 0.0]

    loop._update_survival_states(["org_r"])

    details = loop._should_block_assimilation("org_r")
    assert details is not None
    assert "reserve_threshold" in details


def test_topup_threshold_adjusted_by_survival_bonus():
    loop, pop = _make_loop({"org_t": 0.1})
    pop.roi["org_t"] = [0.95, 0.92, 0.9, 0.88]
    genome = pop.population["org_t"]
    loop._reserve_state["org_t"] = {"active": True, "threshold": 0.6, "balance": 0.1}
    loop._hazard_state["org_t"] = {"active": True, "z": -0.8, "cooldown": 0, "roi": -0.8}

    new_balance, info = loop._maybe_top_up_energy(genome, balance=0.1)

    assert info["status"] == "credited"
    assert info.get("survival_bonus") and new_balance > 0.1


def test_resolve_per_org_batch_scales_under_reserve():
    loop, _ = _make_loop({"org_batch": 0.2})
    loop._reserve_state["org_batch"] = {"active": True, "threshold": 0.6, "balance": 0.2}
    scaled = loop._resolve_per_org_batch("org_batch", 5)
    assert 0 <= scaled <= 5


def test_sample_task_for_org_prefers_cheap_cell():
    loop, _ = _make_loop({"org_cell": 0.3})
    cheap_cell = ("word.count", "short")
    pricey_cell = ("math", "short")
    loop.environment.controller.cells = {
        pricey_cell: SimpleNamespace(price=1.5),
        cheap_cell: SimpleNamespace(price=0.25),
    }

    def _sample_cell(cell, *, canary=False):
        return SimpleNamespace(
            cell=cell,
            prompt="solve",
            price=0.5,
            target=0,
            family=cell[0],
            depth=cell[1],
            difficulty=0.1,
        )

    loop.environment.sample_task_from_cell = lambda cell, canary=False: _sample_cell(cell, canary=canary)
    loop.environment.sample_task = lambda: _sample_cell(pricey_cell, canary=False)

    loop._reserve_state["org_cell"] = {"active": True, "threshold": 0.6, "balance": 0.3}
    task = loop._sample_task_for_org("org_cell", lp_mix=0.0)
    assert task.cell == cheap_cell


def test_snapshot_filters_current_generation_events():
    loop, _ = _make_loop({"org_snap": 0.3})
    loop._reserve_state["org_snap"] = {"active": True, "threshold": 0.6, "balance": 0.3}
    loop._hazard_state["org_snap"] = {"active": False, "z": 0.0, "cooldown": 0, "roi": 0.0}
    loop._survival_events = [
        {"gen": loop.generation_index - 1, "type": "reserve_enter", "org": "org_snap"},
        {"gen": loop.generation_index, "type": "reserve_exit", "org": "org_snap"},
    ]
    snapshot = loop._snapshot_survival_state()
    assert snapshot is not None
    assert snapshot["events"] == [{"gen": loop.generation_index, "type": "reserve_exit", "org": "org_snap"}]
    assert snapshot["price_bias_active_count"] == 1


def _make_loop_with_assim(balances: dict[str, float]) -> EcologyLoop:
    cfg = EcologyConfig()
    cfg.survival.enabled = True
    cfg.survival.reserve_ratio = 0.7
    cfg.survival.reserve_cost_beta = 1.0
    cfg.survival.reserve_cost_window = 4
    cfg.survival.hazard_threshold = -0.5
    cfg.survival.hazard_exit_threshold = -0.1
    cfg.survival.hazard_roi_relief_boost = 0.05
    cfg.survival.hazard_topup_bonus = 0.4
    cfg.survival.min_power_recovery = 0.3
    cfg.survival.price_bias_low_energy = True
    cfg.assimilation_tuning.energy_floor = 0.0
    cfg.assimilation_tuning.energy_floor_roi = 0.0

    host = DummyHost(balances)
    environment = SimpleNamespace(
        best_cell_score=lambda oid: (("math", "short"), 0.5),
        canary_failed=lambda oid: False,
        sample_task=lambda: None,
        sample_task_from_cell=lambda cell, canary=False: SimpleNamespace(
            cell=cell,
            prompt="task prompt",
            price=1.0,
            target=0,
            family=cell[0],
            depth=cell[1],
            difficulty=0.5,
        ),
        controller=SimpleNamespace(cells={}),
        rng=__import__("random").Random(0),
    )
    pop = PopulationManager(cfg.evolution, cfg.foraging)
    for oid in balances:
        pop.register(Genome(organelle_id=oid, drive_weights={}, gate_bias=0.0, rank=3))
        pop.energy[oid] = [0.3, 0.4, 0.35]
        pop.roi[oid] = [0.1, 0.12, 0.14, 0.13]

    assimilation = AssimilationTester(0.0, 0.5, 0)
    loop = EcologyLoop(
        config=cfg,
        host=host,
        environment=environment,
        population=pop,
        assimilation=assimilation,
    )
    loop.generation_index = 4
    return loop


def test_cautious_skip_when_hazard_active():
    loop = _make_loop_with_assim({"org_h": 2.0})
    loop._hazard_state["org_h"] = {"active": True, "z": -0.7, "cooldown": 0, "roi": -0.7}
    loop._reserve_state["org_h"] = {"active": False, "threshold": 0.5, "balance": 0.8}

    merges = loop._attempt_assimilation()
    assert merges == 0
    assert loop.assim_gating_counts.get("cautious_skip", 0) >= 1
    assert any(ev.get("type") == "cautious_skip" for ev in loop._survival_events)


def test_cautious_skip_during_recovery_requires_roi():
    loop = _make_loop_with_assim({"org_r": 2.0})
    loop._hazard_state["org_r"] = {"active": False, "z": -0.2, "cooldown": 3, "roi": 0.1}
    loop._reserve_state["org_r"] = {"active": False, "threshold": 0.5, "balance": 0.8}
    loop.config.survival.min_power_recovery = 0.4
    loop.population.roi["org_r"] = [0.1, 0.12, 0.15, 0.18]

    merges = loop._attempt_assimilation()
    assert merges == 0
    assert loop.assim_gating_counts.get("cautious_skip", 0) >= 1
    assert any(ev.get("type") == "cautious_skip" and ev.get("mode") == "recovery" for ev in loop._survival_events)
