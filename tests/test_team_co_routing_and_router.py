from types import SimpleNamespace

from symbiont_ecology import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.metrics.telemetry import RewardBreakdown


def test_select_synergy_pair_prefers_corouting() -> None:
    cfg = EcologyConfig()
    pop = SimpleNamespace(population={"A": object(), "B": object(), "C": object()})
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=SimpleNamespace(),
        population=pop,
        assimilation=SimpleNamespace(),
    )
    loop._co_routing_counts = {("A", "B"): 3, ("A", "C"): 1}
    pair = loop._select_synergy_pair()
    assert pair == ("A", "B")


def test_run_team_episode_best_of_two_and_records(monkeypatch) -> None:
    cfg = EcologyConfig()
    env = SimpleNamespace(
        post_message=lambda *a, **k: None,
        register_result=lambda *a, **k: None,
    )
    pop = SimpleNamespace(
        population={"A": object(), "B": object()},
        record_score=lambda *a, **k: None,
        record_energy=lambda *a, **k: None,
        record_roi=lambda *a, **k: None,
        record_adapter_usage=lambda *a, **k: None,
    )
    # Minimal loop
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=env,
        population=pop,
        assimilation=SimpleNamespace(),
    )
    # Enable handoff
    loop.config.assimilation_tuning.team_handoff_enabled = True
    loop._record_knowledge_entry = lambda *a, **k: None  # type: ignore[method-assign]
    loop._collect_human_feedback = lambda *a, **k: {}  # type: ignore[method-assign]

    class DummyMetrics:
        def __init__(self):
            self.answer = "ok"
            self.tokens = 1
            self.latency_ms = 0.0
            self.prompt_tokens = 1
            self.trainable_params = 0
            self.flops_estimate = 1.0
            self.memory_gb = 0.001
            self.active_adapters = {}

    # Stub host.step to return metrics for the requested organelle
    def step(prompt: str, intent: str, max_routes: int, allowed_organelle_ids):  # noqa: ARG002
        oid = allowed_organelle_ids[0]
        env = SimpleNamespace(observation=SimpleNamespace(state={"answer": "ok"}))
        return SimpleNamespace(
            envelope=env,
            routes=[SimpleNamespace(organelle_id=oid)],
            responses={oid: DummyMetrics()},
            latency_ms=0.1,
        )

    loop.host = SimpleNamespace(step=step, apply_reward=lambda *a, **k: None)

    # Make A have higher ROI than B via settlement stub
    def settle(oid, task, reward, metrics):  # noqa: ARG002
        return {"roi": 2.0 if oid == "A" else 1.0, "revenue": 1.0, "cost": 1.0}

    recorded = []
    loop._settle_episode = settle  # type: ignore[method-assign]
    loop._record_episode = (
        lambda task, oid, reward, metrics, settlement, success, utilisation: recorded.append(
            (oid, success)
        )
    )  # type: ignore[method-assign]

    # Minimal task: evaluate returns success True
    task = SimpleNamespace(
        task_id="t",
        prompt="p",
        family="math",
        depth="short",
        cell=("math", "short"),
        price=1.0,
        difficulty=0.0,
        supervised_completion=lambda: "1",
        evaluate=lambda answer: (
            True,
            RewardBreakdown(
                task_reward=1.0,
                novelty_bonus=0.0,
                competence_bonus=0.0,
                helper_bonus=0.0,
                risk_penalty=0.0,
                cost_penalty=0.0,
            ),
        ),
    )
    ok = loop._run_team_episode(("A", "B"), task)
    assert ok is True
    # Should record an episode for each member
    assert len(recorded) == 2


def test_select_synergy_pair_fallback_by_roi() -> None:
    cfg = EcologyConfig()

    class Pop:
        def __init__(self):
            self.population = {"A": object(), "B": object(), "C": object()}

        def average_roi(self, oid: str, limit: int = 5) -> float:  # noqa: ARG002
            return {"A": 0.9, "B": 1.1, "C": 0.5}[oid]

    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=SimpleNamespace(),
        population=Pop(),
        assimilation=SimpleNamespace(),
    )
    pair = loop._select_synergy_pair()
    assert pair == ("A", "B") or pair == ("B", "A")


def test_select_synergy_pair_respects_excluded() -> None:
    cfg = EcologyConfig()
    pop = SimpleNamespace(population={"A": object(), "B": object(), "C": object()})
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=SimpleNamespace(),
        population=pop,
        assimilation=SimpleNamespace(),
    )
    loop._co_routing_counts = {("A", "B"): 3, ("A", "C"): 2}
    # Exclude the top pair; expect the next best pair
    excluded = {("A", "B")}
    pair = loop._select_synergy_pair(excluded=excluded)
    assert pair == ("A", "C")


def test_select_synergy_pair_roi_with_excluded_and_fallback() -> None:
    cfg = EcologyConfig()

    class Pop:
        def __init__(self):
            self.population = {"A": object(), "B": object(), "C": object()}

        def average_roi(self, oid: str, limit: int = 5) -> float:  # noqa: ARG002
            return {"A": 1.0, "B": 1.2, "C": 1.1}[oid]

    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=SimpleNamespace(),
        population=Pop(),
        assimilation=SimpleNamespace(),
    )
    # No co-routing counts; use ROI, but exclude (A,B), so expect (A,C)
    pair = loop._select_synergy_pair(fallback_organelle="A", excluded={("A", "B")})
    assert pair == ("A", "C")
