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


def test_resolve_lp_mix_auto_tunes() -> None:
    cfg = EcologyConfig()
    cfg.curriculum.alp_auto_mix = True
    cfg.curriculum.lp_mix_min = 0.1
    cfg.curriculum.lp_mix_max = 0.6
    cfg.curriculum.lp_window = 3
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=SimpleNamespace(controller=SimpleNamespace(lp_progress={("math", "short"): 0.1, ("logic", "short"): 0.2})),
        population=SimpleNamespace(),
        assimilation=AssimilationTester(0.0, 0.5, 0),
    )
    mix1 = loop._resolve_lp_mix(0.2)
    assert 0.1 <= mix1 <= 0.6
    # widen spread to increase mix
    loop.environment.controller.lp_progress = {("math", "short"): 0.8, ("logic", "short"): 0.1}  # type: ignore[attr-defined]
    mix2 = loop._resolve_lp_mix(0.2)
    assert mix2 >= mix1
    # Ensure smoothing window trims history
    assert len(loop._lp_mix_history) <= cfg.curriculum.lp_window


def test_resolve_lp_mix_manual_returns_base() -> None:
    cfg = EcologyConfig()
    cfg.curriculum.alp_auto_mix = False
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=SimpleNamespace(controller=SimpleNamespace(lp_progress={})),
        population=SimpleNamespace(),
        assimilation=AssimilationTester(0.0, 0.5, 0),
    )
    mix = loop._resolve_lp_mix(0.3)
    assert mix == 0.3
