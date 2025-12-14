from symbiont_ecology.config import CanaryConfig, ControllerConfig, GridConfig, PricingConfig
from symbiont_ecology.environment.grid import EnvironmentController


def test_bandit_score_internal():
    grid_cfg = GridConfig(families=["word.count"], depths=["short"])
    ctrl_cfg = ControllerConfig(tau=0.5, beta=0.2, eta=0.1)
    price_cfg = PricingConfig(base=1.0, k=1.0, min=0.3, max=2.0)
    canary_cfg = CanaryConfig(q_min=0.7)
    ec = EnvironmentController(grid_cfg, ctrl_cfg, price_cfg, canary_cfg, lp_alpha=0.5, seed=123)
    cell = ("word.count", "short")
    # Seed bandit counts and successes to exercise ROI + exploration
    ec.bandit_counts[cell] = 10
    ec.bandit_success[cell] = 6.0
    score = ec._bandit_score(cell, total_pulls=100)
    assert isinstance(score, float) and score > 0.0
