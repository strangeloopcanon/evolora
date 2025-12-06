from __future__ import annotations

import random
from collections import Counter
from types import SimpleNamespace

from symbiont_ecology import EcologyConfig
from symbiont_ecology.environment.grid import GridTask
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome, PopulationManager
from symbiont_ecology.metrics.telemetry import RewardBreakdown


class _DummyLedger:
    def __init__(self, balances: dict[str, float]) -> None:
        self._balances = dict(balances)
        self.energy_cap = 10.0

    def energy_balance(self, organelle_id: str) -> float:
        return self._balances.get(organelle_id, 0.0)

    def credit_energy(self, organelle_id: str, amount: float) -> float:
        self._balances[organelle_id] = self._balances.get(organelle_id, 0.0) + amount
        return self._balances[organelle_id]

    def consume_energy(self, organelle_id: str, amount: float) -> bool:
        bal = self._balances.get(organelle_id, 0.0)
        if bal < amount:
            return False
        self._balances[organelle_id] = bal - amount
        return True


class _DummyHost:
    def __init__(self, balances: dict[str, float]) -> None:
        self.ledger = _DummyLedger(balances)

    def list_organelle_ids(self) -> list[str]:
        return list(self.ledger._balances.keys())

    def get_organelle(self, organelle_id: str) -> SimpleNamespace:
        return SimpleNamespace(rank=2)

    def retire_organelle(self, organelle_id: str) -> None:
        self.ledger._balances.pop(organelle_id, None)

    def spawn_organelle(self, rank: int, hebbian_config=None, activation_bias: float = 0.0) -> str:
        oid = f"org_{len(self.ledger._balances)}"
        self.ledger._balances[oid] = 1.0
        return oid


class _DummyController:
    def __init__(self) -> None:
        self.cells = {
            ("math", "short"): SimpleNamespace(),
            ("word.count", "short"): SimpleNamespace(),
        }
        self.lp_progress = {cell: 0.0 for cell in self.cells}

    def sample_cell(self, lp_mix: float = 0.0):
        return ("math", "short")

    def get_state(self, cell):
        return SimpleNamespace(success_ema=0.5)


class _DummyEnvironment:
    def __init__(self) -> None:
        self.controller = _DummyController()
        self.rng = random.Random(1337)
        self.organism_stats = {}

    def sample_task_from_cell(self, cell, canary: bool = False) -> GridTask:
        return GridTask(
            task_id="task",
            cell=cell,
            prompt="2 + 2",
            price=1.0,
            target=4,
            family=cell[0],
            depth=cell[1],
            difficulty=0.5,
            canary=canary,
        )

    def sample_task(self) -> GridTask:
        return self.sample_task_from_cell(("math", "short"))

    def post_message(self, *args, **kwargs) -> None:  # pragma: no cover - unused but present
        return None

    def read_messages(self, *args, **kwargs):  # pragma: no cover - unused but present
        return []

    def read_caches(self, *args, **kwargs):  # pragma: no cover - unused but present
        return []

    def post_cache(self, *args, **kwargs):  # pragma: no cover - unused but present
        return None


def _make_loop() -> EcologyLoop:
    cfg = EcologyConfig()
    cfg.foraging.enabled = True
    host = _DummyHost({"org": 1.0})
    environment = _DummyEnvironment()
    population = PopulationManager(cfg.evolution, cfg.foraging)
    genome = Genome(
        organelle_id="org",
        drive_weights={},
        gate_bias=0.0,
        rank=2,
        q_decay=0.5,
        beta_exploit=2.5,
        ucb_bonus=0.0,
        budget_aggressiveness=0.6,
    )
    population.register(genome)
    loop = EcologyLoop(cfg, host, environment, population, assimilation=SimpleNamespace())
    return loop


def test_foraging_updates_cell_values() -> None:
    loop = _make_loop()
    task = GridTask(
        task_id="t-1",
        cell=("math", "short"),
        prompt="2 + 2",
        price=1.0,
        target=4,
        family="math",
        depth="short",
        difficulty=0.5,
    )
    reward = RewardBreakdown(
        task_reward=1.0,
        novelty_bonus=0.0,
        competence_bonus=0.0,
        helper_bonus=0.0,
        risk_penalty=0.0,
        cost_penalty=0.0,
    )
    metrics = SimpleNamespace(
        answer="4",
        tokens=12,
        latency_ms=1.0,
        flops_estimate=0.1,
        memory_gb=0.01,
        trainable_params=0,
        active_adapters=[],
        adapter_utilisation={},
    )
    settlement = {"cost": 0.5, "energy_before": 1.0, "energy_after": 1.4, "roi": 2.0}
    loop._record_episode(task, "org", reward, metrics, settlement, True, {})
    values = loop.population.cell_values["org"]
    assert ("math", "short") in values
    assert values[("math", "short")] > 0.9  # EMA should move towards observed ROI


def test_foraging_prefers_high_q_cells() -> None:
    loop = _make_loop()
    # Seed distinct Q-values
    loop.population.update_cell_value("org", ("math", "short"), roi=2.0, decay=0.5, q_init=0.0)
    loop.population.update_cell_value("org", ("word.count", "short"), roi=0.1, decay=0.5, q_init=0.0)
    loop.environment.rng.seed(7)
    picks = Counter(loop._foraging_select_cell("org") for _ in range(20))
    assert picks[("math", "short")] > picks[("word.count", "short")]
