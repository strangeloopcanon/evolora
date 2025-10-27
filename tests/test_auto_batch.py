from types import SimpleNamespace

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.assimilation import AssimilationTester


def test_compute_batch_size_endogenous() -> None:
    cfg = EcologyConfig()
    cfg.environment.auto_batch = True
    cfg.environment.batch_min = 1
    cfg.environment.batch_max = 4
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=SimpleNamespace(),
        population=SimpleNamespace(aggregate_roi=lambda limit=5: 0.4),
        assimilation=AssimilationTester(0.0, 0.5, 0),
    )
    assert loop._compute_batch_size(2) == 1
    loop.population = SimpleNamespace(aggregate_roi=lambda limit=5: 1.6)  # type: ignore
    assert loop._compute_batch_size(2) == 4
    loop.population = SimpleNamespace(aggregate_roi=lambda limit=5: 1.0)  # type: ignore
    mid = loop._compute_batch_size(2)
    assert 1 <= mid <= 4
