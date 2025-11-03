from types import SimpleNamespace

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop


def test_probe_co_routing_increments_counts():
    cfg = EcologyConfig()
    cfg.assimilation_tuning.team_routing_probe_per_gen = 2
    # Environment that can sample a simple task
    environment = SimpleNamespace(
        rng=__import__("random").Random(0),
        sample_task=lambda: SimpleNamespace(prompt="p"),
    )
    # Host that returns two route events with distinct organelles
    def step(prompt: str, intent: str, max_routes: int, allowed_organelle_ids):  # noqa: ARG002
        a, b = allowed_organelle_ids[:2]
        env = SimpleNamespace(observation=SimpleNamespace(state={"answer": "ok"}))
        return SimpleNamespace(
            envelope=env,
            routes=[SimpleNamespace(organelle_id=a), SimpleNamespace(organelle_id=b)],
            responses={},
            latency_ms=0.0,
        )

    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(step=step),
        environment=environment,
        population=SimpleNamespace(population={"A": object(), "B": object()}),
        assimilation=SimpleNamespace(),
    )
    loop._probe_co_routing(["A", "B"])
    assert hasattr(loop, "_co_routing_counts")
    # Either (A,B) or (B,A) key should be counted as 1 or 2 depending on RNG
    counts = getattr(loop, "_co_routing_counts")
    assert any(k[0] in ("A", "B") and k[1] in ("A", "B") for k in counts.keys())

