import random
from types import SimpleNamespace

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.population import PopulationManager


def _make_loop() -> EcologyLoop:
    cfg = EcologyConfig()
    cfg.curriculum.warmup_generations = 3
    cfg.curriculum.warmup_families = ["math"]
    cfg.curriculum.warmup_depths = ["short"]
    controller = SimpleNamespace(
        cells={
            ("math", "short"): SimpleNamespace(success_ema=0.6, price=1.0),
            ("word.count", "medium"): SimpleNamespace(success_ema=0.4, price=1.0),
        }
    )
    environment = SimpleNamespace(controller=controller, rng=random.Random(0), canary_q_min=0.9)
    host = SimpleNamespace(list_organelle_ids=lambda: [], ledger=None)
    population = PopulationManager(cfg.evolution, cfg.foraging)
    loop = EcologyLoop(cfg, host, environment, population, assimilation=SimpleNamespace())
    return loop


def test_curriculum_warmup_restricts_cells():
    loop = _make_loop()
    loop.generation_index = 1
    allowed = loop._curriculum_allowed_cells()
    assert allowed == [("math", "short")]
    # After warmup window expires all cells become eligible again
    loop.generation_index = 5
    assert loop._curriculum_allowed_cells() is None
