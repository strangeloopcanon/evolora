from types import SimpleNamespace

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import PopulationManager


class DummyLedger:
    def __init__(self) -> None:
        self._balances: dict[str, float] = {"org": 10.0}
        self.energy_cap = 10.0

    def energy_balance(self, organelle_id: str) -> float:
        return self._balances.get(organelle_id, 0.0)

    def consume_energy(self, organelle_id: str, amount: float) -> bool:  # pragma: no cover - unused
        bal = self._balances.get(organelle_id, 0.0)
        if bal < amount:
            return False
        self._balances[organelle_id] = bal - amount
        return True

    def credit_energy(self, organelle_id: str, amount: float) -> float:  # pragma: no cover - unused
        self._balances[organelle_id] = self._balances.get(organelle_id, 0.0) + amount
        return self._balances[organelle_id]


class DummyHost:
    def __init__(self) -> None:
        self.ledger = DummyLedger()

    def list_organelle_ids(self) -> list[str]:  # pragma: no cover - unused
        return ["org"]


def _make_loop() -> EcologyLoop:
    cfg = EcologyConfig()
    cfg.prompting.few_shot_enabled = True
    host = DummyHost()
    env = SimpleNamespace(controller=SimpleNamespace(cells={}), rng=SimpleNamespace())
    population = PopulationManager(cfg.evolution, cfg.foraging)
    loop = EcologyLoop(cfg, host, env, population, assimilation=SimpleNamespace())
    return loop


def test_few_shot_scaffold_applies_examples():
    loop = _make_loop()
    task = SimpleNamespace(
        family="word.count",
        prompt="Count the number of words in 'Agents evolve together.'",
        depth="short",
    )

    scaffolded, applied = loop._apply_prompt_scaffold(task, task.prompt)
    assert applied is True
    assert "Example Input" in scaffolded
    assert "Example Output" in scaffolded
    assert scaffolded.endswith(f"Task: {task.prompt}")


def test_few_shot_scaffold_noop_when_disabled():
    loop = _make_loop()
    loop.config.prompting.few_shot_enabled = False
    task = SimpleNamespace(family="math", prompt="Add 1 and 2.", depth="short")

    scaffolded, applied = loop._apply_prompt_scaffold(task, task.prompt)
    assert applied is False
    assert scaffolded == task.prompt
