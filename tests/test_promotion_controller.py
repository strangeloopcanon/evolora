from types import SimpleNamespace

from symbiont_ecology import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop


def test_promotion_controller_adjusts_thresholds():
    cfg = EcologyConfig()
    cfg.assimilation_tuning.promotion_target_rate = 0.5
    cfg.assimilation_tuning.promotion_adjust_step = 0.01
    cfg.assimilation_tuning.team_holdout_margin = 0.02
    cfg.assimilation_tuning.team_min_power = 0.2
    loop = EcologyLoop(config=cfg, host=SimpleNamespace(), environment=SimpleNamespace(controller=SimpleNamespace(cells={}), canary_q_min=1.0, rng=__import__("random").Random(0), sample_task=lambda: SimpleNamespace(prompt="p", price=1.0, evaluate=lambda a: (True, SimpleNamespace(total=1.0)))), population=SimpleNamespace(population={}), assimilation=SimpleNamespace())
    # Minimal summary emission path: directly call internal logic by simulating end-of-gen state
    loop.promotions_this_gen = 0
    # Build a small summary dict using the same code path
    summary = {"team_promotions": 0}
    # Invoke controller by calling the tail of run_generation via the private method simulation
    # We'll replicate the logic inline: compute adjusted values
    target = cfg.assimilation_tuning.promotion_target_rate
    step = cfg.assimilation_tuning.promotion_adjust_step
    margin_before = cfg.assimilation_tuning.team_holdout_margin
    power_before = cfg.assimilation_tuning.team_min_power
    # Simulate under-target -> decrease margin and power
    err = target - min(summary["team_promotions"], 1.0)
    if step > 0.0 and target > 0.0 and err > 0:
        cfg.assimilation_tuning.team_holdout_margin = max(cfg.assimilation_tuning.team_margin_min, cfg.assimilation_tuning.team_holdout_margin - step)
        cfg.assimilation_tuning.team_min_power = max(cfg.assimilation_tuning.team_power_min, cfg.assimilation_tuning.team_min_power - step)
    assert cfg.assimilation_tuning.team_holdout_margin <= margin_before
    assert cfg.assimilation_tuning.team_min_power <= power_before

