from __future__ import annotations

from types import SimpleNamespace

import pytest

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome, PopulationManager


class DummyLedger:
    def __init__(self, balances: dict[str, float]) -> None:
        self._balances = dict(balances)
        self.energy_cap = 10.0

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
    def __init__(self, balances: dict[str, float]) -> None:
        self.ledger = DummyLedger(balances)

    def list_organelle_ids(self) -> list[str]:
        return list(self.ledger._balances.keys())

    def get_organelle(self, organelle_id: str) -> SimpleNamespace:
        return SimpleNamespace(rank=2, family="math")

    def _active_adapters(self, organelle: SimpleNamespace) -> dict[str, int]:
        return {"core": 1}

    def retire_organelle(self, organelle_id: str) -> None:
        self.ledger._balances.pop(organelle_id, None)


def _make_loop(balances: dict[str, float]) -> EcologyLoop:
    cfg = EcologyConfig()
    cfg.evaluation.enabled = False
    cfg.meta.enabled = False
    cfg.winter.enabled = True
    cfg.winter.winter_interval = 2
    cfg.winter.winter_duration = 1
    cfg.winter.price_multiplier = 1.5
    cfg.winter.ticket_multiplier = 0.5
    cfg.winter.post_winter_bonus = 0.1
    host = DummyHost(balances)
    env = SimpleNamespace(
        controller=SimpleNamespace(cells={}, lp_progress={}),
        rng=SimpleNamespace(random=lambda: 0.5),
        organism_stats={},
    )
    pop = PopulationManager(cfg.evolution, cfg.foraging)
    assimilation = SimpleNamespace(
        evaluate=lambda *args, **kwargs: SimpleNamespace(passed=False, uplift=0.0, details={}),
        update_thresholds=lambda **kwargs: None,
    )
    loop = EcologyLoop(cfg, host, env, pop, assimilation=assimilation)
    return loop


def _register_organelle(loop: EcologyLoop, organelle_id: str) -> None:
    genome = Genome(organelle_id=organelle_id, drive_weights={}, gate_bias=0.0, rank=2)
    loop.population.register(genome)
    loop.population.roi[organelle_id] = []
    loop.population.energy[organelle_id] = []


def test_winter_cycle_records_events_and_deltas():
    loop = _make_loop({"org_a": 5.0, "org_b": 5.5})
    for oid in ("org_a", "org_b"):
        _register_organelle(loop, oid)
        loop.population.roi[oid].extend([1.0, 1.1])

    baseline_roi = loop.population.aggregate_roi(limit=6)
    loop._last_assim_attempts = 10
    loop._winter_counter = loop.config.winter.winter_interval - 1
    loop._winter_events_gen = []
    loop.generation_index = 3

    loop._update_winter_cycle()

    assert loop._winter_active is True
    assert loop._winter_timer == loop.config.winter.winter_duration
    assert len(loop._winter_events_gen) == 1
    start_event = loop._winter_events_gen[0]
    assert start_event["type"] == "winter_start"
    assert start_event["pre_assim"] == 10
    assert start_event["pre_roi"] == pytest.approx(baseline_roi)

    # Simulate one winter generation with improved ROI and more assimilation attempts
    loop.population.roi["org_a"].append(1.6)
    loop.population.roi["org_b"].append(1.5)
    loop._last_assim_attempts = 14
    loop._winter_events_gen = []
    loop.generation_index += 1

    loop._update_winter_cycle()

    assert loop._winter_active is False
    assert len(loop._winter_events_gen) >= 1
    end_event = loop._winter_events_gen[0]
    assert end_event["type"] == "winter_end"
    assert end_event["pre_roi"] == pytest.approx(baseline_roi)
    assert end_event["post_assim"] == 14
    assert end_event["pre_assim"] == 10
    assert end_event["delta_assim"] == pytest.approx(4.0)
    assert end_event["delta_roi"] == pytest.approx(
        loop.population.aggregate_roi(limit=6) - baseline_roi
    )
    bonus_events = [ev for ev in loop._winter_events_gen if ev["type"] == "winter_bonus"]
    assert bonus_events, "post-winter bonus should be recorded when bonus > 0"


def test_winter_cull_emits_event():
    loop = _make_loop({"cold_a": 0.2, "cold_b": 0.1})
    loop.config.energy.bankruptcy_grace = 1
    loop._winter_active = True
    loop._winter_timer = 2
    loop._winter_events_gen = []
    loop.generation_index = 5

    loop.run_generation(batch_size=1)

    events = [ev for ev in loop._winter_events_gen if ev["type"] == "winter_cull"]
    assert events, "winter_cull event should be recorded when colonies are culled during winter"
    assert events[0]["count"] == 2
    assert set(events[0]["preview"]) <= {"cold_a", "cold_b"}
