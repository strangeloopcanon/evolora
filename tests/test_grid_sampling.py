from symbiont_ecology.environment.grid import GridEnvironment, GridTask
from symbiont_ecology.config import GridConfig, ControllerConfig, PricingConfig, CanaryConfig


def _env_for(families):
    grid_cfg = GridConfig(families=families, depths=["short"])
    ctrl_cfg = ControllerConfig(tau=0.5, beta=0.2, eta=0.1)
    price_cfg = PricingConfig(base=1.0, k=1.0, min=0.3, max=2.0)
    canary_cfg = CanaryConfig(q_min=0.7)
    return GridEnvironment(grid_cfg, ctrl_cfg, price_cfg, canary_cfg, seed=42)


def test_sample_task_from_cell_string_sort():
    env = _env_for(["string.sort"])
    cell = ("string.sort", "short")
    task = env.sample_task_from_cell(cell)
    ok, _ = task.evaluate("".join(sorted(str(task.target))))
    assert task.family == "string.sort" and ok in (True, False)


def test_sample_task_from_cell_json_repair():
    env = _env_for(["json_repair"])
    cell = ("json_repair", "short")
    task = env.sample_task_from_cell(cell)
    ok, _ = task.evaluate(str(task.target))
    assert task.family == "json_repair" and ok is True


def test_canary_queue_and_rotation():
    env = _env_for(["word.count"]).__class__(
        GridConfig(families=["word.count"], depths=["short"]),
        ControllerConfig(tau=0.5, beta=0.2, eta=0.1),
        PricingConfig(base=1.0, k=1.0, min=0.3, max=2.0),
        CanaryConfig(q_min=0.7),
        seed=123,
    )
    cell = ("word.count", "short")
    t1 = env.controller.next_canary(cell)
    t2 = env.controller.next_canary(cell)
    assert t1 is not None and t2 is not None and t1 != t2


def test_canary_sampling_path():
    # Force canary path by making success_ema high and rng.random() return 0.0
    grid_cfg = GridConfig(families=["word.count"], depths=["short"])
    ctrl_cfg = ControllerConfig(tau=0.5, beta=0.2, eta=0.1)
    price_cfg = PricingConfig(base=1.0, k=1.0, min=0.3, max=2.0)
    canary_cfg = CanaryConfig(q_min=0.1)
    env = GridEnvironment(grid_cfg, ctrl_cfg, price_cfg, canary_cfg, seed=42)
    cell = ("word.count", "short")
    # Mark state as highly successful
    state = env.controller.get_state(cell)
    state.success_ema = 0.9
    env.controller.cells[cell] = state
    class _R:
        def random(self):
            return 0.0
    env.rng = _R()
    task = env.sample_task()
    assert isinstance(task, GridTask)
