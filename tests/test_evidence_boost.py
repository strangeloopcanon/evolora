from types import SimpleNamespace

from symbiont_ecology.environment.loops import EcologyLoop


def test_should_boost_thresholds():
    loop = object.__new__(EcologyLoop)
    loop.population = SimpleNamespace(
        average_roi=lambda oid, limit=5: {"hi": 1.2, "lo": 0.4}.get(oid, 0.0),
        aggregate_roi=lambda limit=None: 1.0,
    )
    assert loop._should_boost("hi") is True
    assert loop._should_boost("lo") is False
