from symbiont_ecology.config import CanaryConfig, ControllerConfig, GridConfig, PricingConfig
from symbiont_ecology.environment.grid import GridEnvironment, GridKey


def test_grid_environment_updates_controller() -> None:
    env = GridEnvironment(
        grid_cfg=GridConfig(families=["math"], depths=["short"]),
        controller_cfg=ControllerConfig(tau=0.5, beta=0.5, eta=0.5),
        pricing_cfg=PricingConfig(base=1.0, k=1.5, min=0.5, max=2.0),
        canary_cfg=CanaryConfig(q_min=0.9),
        seed=7,
    )
    task = env.sample_task()
    assert task.family == "math"
    env.register_result("org-1", task, success=True)
    state = env.cell_state(task.cell)
    assert 0.05 <= state.difficulty <= 0.95
    assert abs(state.success_ema - 0.75) < 0.5
    assert 0.5 <= state.price <= 2.0

    # ensure canary rotation works
    key: GridKey = task.cell
    canary = env.controller.next_canary(key)
    assert canary is not None
    assert canary.cell == key


def test_grid_environment_supports_new_families() -> None:
    env = GridEnvironment(
        grid_cfg=GridConfig(families=["logic.bool", "math.sequence"], depths=["short"]),
        controller_cfg=ControllerConfig(tau=0.5, beta=0.5, eta=0.5),
        pricing_cfg=PricingConfig(base=1.0, k=1.5, min=0.5, max=2.0),
        canary_cfg=CanaryConfig(q_min=0.7),
        seed=13,
    )
    seen = set()
    for _ in range(6):
        task = env.sample_task()
        seen.add(task.family)
        env.register_result("org-seed", task, success=True)
    assert {"logic.bool", "math.sequence"}.issubset(seen)
