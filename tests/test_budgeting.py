from types import SimpleNamespace

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import Genome, PopulationManager


class _Ledger:
    def __init__(self, balances: dict[str, float]) -> None:
        self._balances = dict(balances)

    def energy_balance(self, organelle_id: str) -> float:
        return self._balances.get(organelle_id, 0.0)

    def consume_energy(self, organelle_id: str, amount: float) -> bool:
        bal = self._balances.get(organelle_id, 0.0)
        if bal < amount:
            return False
        self._balances[organelle_id] = bal - amount
        return True

    def credit_energy(self, organelle_id: str, amount: float) -> float:
        self._balances[organelle_id] = self._balances.get(organelle_id, 0.0) + amount
        return self._balances[organelle_id]


class _Host:
    def __init__(self, balances: dict[str, float]) -> None:
        self.ledger = _Ledger(balances)

    def list_organelle_ids(self) -> list[str]:
        return list(self.ledger._balances.keys())

    def get_organelle(self, organelle_id: str):
        return SimpleNamespace(rank=2)

    def retire_organelle(self, organelle_id: str) -> None:
        self.ledger._balances.pop(organelle_id, None)


def _make_loop(balances: dict[str, float]) -> EcologyLoop:
    cfg = EcologyConfig()
    cfg.environment.budget_enabled = True
    cfg.environment.budget_energy_floor = 0.4
    cfg.environment.budget_energy_ceiling = 2.5
    cfg.environment.budget_trait_bonus = 1.0
    cfg.environment.budget_policy_floor = 0.4
    cfg.environment.budget_policy_ceiling = 1.8
    host = _Host(balances)
    environment = SimpleNamespace(controller=SimpleNamespace(cells={}), rng=SimpleNamespace())
    pop = PopulationManager(cfg.evolution, cfg.foraging)
    loop = EcologyLoop(cfg, host, environment, pop, assimilation=SimpleNamespace())
    return loop


def test_budget_increases_with_energy_and_trait():
    loop = _make_loop({"strong": 3.0, "weak": 0.4})
    loop.population.register(
        Genome(organelle_id="strong", drive_weights={}, gate_bias=0.0, rank=2, explore_rate=0.8)
    )
    loop.population.register(
        Genome(organelle_id="weak", drive_weights={}, gate_bias=0.0, rank=2, explore_rate=0.1)
    )
    budgets, meta = loop._compute_budget_map(["strong", "weak"], base_bs=2)
    assert budgets["strong"] > budgets["weak"]
    assert meta["per_org"]["strong"]["raw"] == budgets["strong"]
    assert meta["cap_hit"] is False


def test_budget_respects_policy_ceiling():
    loop = _make_loop({"policy": 1.5})
    loop.population.register(
        Genome(organelle_id="policy", drive_weights={}, gate_bias=0.0, rank=2, explore_rate=0.5)
    )
    loop._active_policies["policy"] = {"budget_frac": 5.0}  # should clamp to ceiling
    budgets, _meta = loop._compute_budget_map(["policy"], base_bs=2)
    assert 3 <= budgets["policy"] <= 5


def test_budget_global_cap_enforced():
    loop = _make_loop({"a": 1.5, "b": 1.2, "c": 0.8})
    loop.config.environment.global_episode_cap = 3
    loop.population.register(
        Genome(organelle_id="a", drive_weights={}, gate_bias=0.0, rank=2, explore_rate=0.9)
    )
    loop.population.register(
        Genome(organelle_id="b", drive_weights={}, gate_bias=0.0, rank=2, explore_rate=0.6)
    )
    loop.population.register(
        Genome(organelle_id="c", drive_weights={}, gate_bias=0.0, rank=2, explore_rate=0.2)
    )
    budgets, meta = loop._compute_budget_map(["a", "b", "c"], base_bs=2)
    assert sum(budgets.values()) <= loop.config.environment.global_episode_cap
    assert meta["cap_hit"] is True
    assert any(value == 0 for value in budgets.values())


def test_budget_requires_policy_parse_uses_floor():
    loop = _make_loop({"org": 1.0})
    loop.config.environment.budget_policy_requires_parse = True
    loop.config.environment.budget_policy_floor = 0.25
    loop.population.register(
        Genome(organelle_id="org", drive_weights={}, gate_bias=0.0, rank=2, explore_rate=0.5)
    )
    budgets, _meta = loop._compute_budget_map(["org"], base_bs=4)
    assert budgets["org"] < 4
    # Once a policy is parsed the budget should rise again
    loop._active_policies["org"] = {"budget_frac": 1.2}
    budgets_after, _ = loop._compute_budget_map(["org"], base_bs=4)
    assert budgets_after["org"] > budgets["org"]
