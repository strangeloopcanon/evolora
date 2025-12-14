from types import SimpleNamespace

from symbiont_ecology.config import (
    CanaryConfig,
    ControllerConfig,
    EnergyConfig,
    GridConfig,
    PricingConfig,
)
from symbiont_ecology.environment.grid import GridEnvironment
from symbiont_ecology.evaluation.manager import (
    EvaluationConfigRuntime,
    EvaluationManager,
    EvaluationTask,
)


class _FakeLedger:
    def __init__(self):
        self._e = {}
        self.energy_cap = 10.0

    def energy_balance(self, organelle_id: str) -> float:
        return float(self._e.get(organelle_id, 1.0))

    def set_energy(self, organelle_id: str, value: float) -> None:
        self._e[organelle_id] = float(value)


class _FakeHost:
    def __init__(self):
        self._ids = ["org_a"]
        self.ledger = _FakeLedger()
        self.config = SimpleNamespace(energy=EnergyConfig())

    def list_organelle_ids(self):
        return list(self._ids)

    def apply_reward(self, envelope, rewards):
        return None

    def step(self, prompt: str, intent: str, max_routes: int, allowed_organelle_ids=None):
        # Always answer correctly with a small, fixed cost footprint
        envelope = SimpleNamespace(observation=SimpleNamespace(state={"answer": "2"}))
        routes = [SimpleNamespace(organelle_id="org_a")]
        metrics = SimpleNamespace(
            flops_estimate=1000.0, memory_gb=0.001, latency_ms=1.0, trainable_params=10, tokens=5
        )
        responses = {"org_a": metrics}
        return SimpleNamespace(
            envelope=envelope, routes=routes, responses=responses, latency_ms=1.0
        )


def test_evaluation_manager_evaluate_smoke():
    # Environment with a single word.count cell
    grid_cfg = GridConfig(families=["word.count"], depths=["short"])
    ctrl_cfg = ControllerConfig(tau=0.5, beta=0.2, eta=0.1)
    price_cfg = PricingConfig(base=1.0, k=1.0, min=0.3, max=2.0)
    canary_cfg = CanaryConfig(q_min=0.8)
    env = GridEnvironment(grid_cfg, ctrl_cfg, price_cfg, canary_cfg, seed=123)
    # One simple task whose correct answer is "2"
    tasks = [
        EvaluationTask(
            prompt="Count words in 'one two'", target=2, family="word.count", depth="short"
        )
    ]
    runtime = EvaluationConfigRuntime(
        enabled=True, cadence=5, tasks=tasks, sample_size=1, reward_weight=0.5
    )
    mgr = EvaluationManager(runtime)
    host = _FakeHost()
    out = mgr.evaluate(host, env)
    assert isinstance(out, dict) and "accuracy" in out and out["total"] == 1
    assert out["accuracy"] >= 0.0 and out["evaluated_routes"] >= 1
    assert "family_breakdown" in out and "word.count" in out["family_breakdown"]
    assert out["family_breakdown"]["word.count"]["total"] == 1


class _ZeroCostHost(_FakeHost):
    def step(self, prompt: str, intent: str, max_routes: int, allowed_organelle_ids=None):
        envelope = SimpleNamespace(observation=SimpleNamespace(state={"answer": "2"}))
        routes = [SimpleNamespace(organelle_id="org_a")]
        # All zeros to trigger cost <= 0.0 branch
        metrics = SimpleNamespace(
            flops_estimate=0.0, memory_gb=0.0, latency_ms=0.0, trainable_params=0, tokens=2
        )
        responses = {"org_a": metrics}
        return SimpleNamespace(
            envelope=envelope, routes=routes, responses=responses, latency_ms=0.1
        )


def test_evaluation_manager_zero_cost_branch():
    grid_cfg = GridConfig(families=["word.count"], depths=["short"])
    ctrl_cfg = ControllerConfig(tau=0.5, beta=0.2, eta=0.1)
    price_cfg = PricingConfig(base=1.0, k=1.0, min=0.3, max=2.0)
    canary_cfg = CanaryConfig(q_min=0.8)
    env = GridEnvironment(grid_cfg, ctrl_cfg, price_cfg, canary_cfg, seed=321)
    tasks = [
        EvaluationTask(
            prompt="Count words in 'one two'", target=2, family="word.count", depth="short"
        )
    ]
    runtime = EvaluationConfigRuntime(
        enabled=True, cadence=5, tasks=tasks, sample_size=1, reward_weight=0.5
    )
    mgr = EvaluationManager(runtime)
    host = _ZeroCostHost()
    out = mgr.evaluate(host, env)
    assert out["total"] == 1 and out["evaluated_routes"] >= 1
    assert out["family_breakdown"]["word.count"]["total"] == 1
