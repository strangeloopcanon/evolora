from types import SimpleNamespace

from symbiont_ecology import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.assimilation import AssimilationTester


def test_co_routing_probe_counts_pairs() -> None:
    cfg = EcologyConfig()
    # Enable a couple of probes per generation
    cfg.assimilation_tuning.team_routing_probe_per_gen = 2
    # Minimal loop with stubbed host/environment for routing probe
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(
            step=lambda prompt, intent, max_routes, allowed_organelle_ids: SimpleNamespace(
                routes=[SimpleNamespace(organelle_id="A"), SimpleNamespace(organelle_id="B")]
            )
        ),
        environment=SimpleNamespace(sample_task=lambda: SimpleNamespace(prompt="p"), rng=SimpleNamespace(random=lambda: 0.0)),
        population=SimpleNamespace(),
        assimilation=AssimilationTester(0.0, 0.5, 0),
    )
    loop._co_routing_counts = {}
    loop._probe_co_routing(["A", "B", "C"])
    # Expect at least one count for the (A,B) pair, repeated by probes per gen
    key = ("A", "B")
    assert key in loop._co_routing_counts
    assert loop._co_routing_counts[key] >= 1


def test_co_routing_probe_early_return() -> None:
    cfg = EcologyConfig()
    cfg.assimilation_tuning.team_routing_probe_per_gen = 0  # disabled
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(step=lambda *a, **k: None),
        environment=SimpleNamespace(sample_task=lambda: SimpleNamespace(prompt="p")),
        population=SimpleNamespace(),
        assimilation=AssimilationTester(0.0, 0.5, 0),
    )
    loop._co_routing_counts = {}
    # active_ids < 2 also triggers early return
    loop._probe_co_routing(["A"])  # should not error
    assert loop._co_routing_counts == {}
