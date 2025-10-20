from symbiont_ecology.config import CanaryConfig, ControllerConfig, GridConfig, PricingConfig
from symbiont_ecology.environment.grid import EnvironmentController


def test_environment_controller_equilibrium_adjustments() -> None:
    grid_cfg = GridConfig(families=["math"], depths=["short"])
    controller_cfg = ControllerConfig(tau=0.5, beta=0.4, eta=0.1)
    pricing_cfg = PricingConfig(base=1.0, k=1.0, min=0.2, max=2.0)
    canary_cfg = CanaryConfig(q_min=0.9)
    controller = EnvironmentController(grid_cfg, controller_cfg, pricing_cfg, canary_cfg, seed=21)
    cell = ("math", "short")

    baseline_state = controller.get_state(cell)
    baseline_success = baseline_state.success_ema
    baseline_price = baseline_state.price

    for _ in range(6):
        controller.update(cell, success=True)

    after_success = controller.get_state(cell)
    assert after_success.success_ema > baseline_success
    assert after_success.price < baseline_price

    for _ in range(6):
        controller.update(cell, success=False)

    after_failure = controller.get_state(cell)
    assert after_failure.success_ema < after_success.success_ema
    assert after_failure.price > after_success.price


def test_bandit_sampling_prefers_successful_cells() -> None:
    grid_cfg = GridConfig(families=["math", "logic.bool"], depths=["short"])
    controller_cfg = ControllerConfig(tau=0.5, beta=0.4, eta=0.1)
    pricing_cfg = PricingConfig(base=1.0, k=1.0, min=0.2, max=2.0)
    canary_cfg = CanaryConfig(q_min=0.8)
    controller = EnvironmentController(grid_cfg, controller_cfg, pricing_cfg, canary_cfg, seed=11)

    math_cell = ("math", "short")
    logic_cell = ("logic.bool", "short")

    first = controller.sample_cell()
    controller.update(first, success=first == math_cell)
    second = controller.sample_cell()
    controller.update(second, success=second == math_cell)
    assert {first, second} == {math_cell, logic_cell}

    for _ in range(20):
        cell = controller.sample_cell()
        success = cell == math_cell
        controller.update(cell, success=success)

    math_picks = 0
    logic_picks = 0
    for _ in range(10):
        cell = controller.sample_cell()
        success = cell == math_cell
        controller.update(cell, success=success)
        if cell == math_cell:
            math_picks += 1
        else:
            logic_picks += 1

    assert math_picks > logic_picks
