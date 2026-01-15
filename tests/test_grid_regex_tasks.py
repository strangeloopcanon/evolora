import re

from symbiont_ecology.config import CanaryConfig, ControllerConfig, GridConfig, PricingConfig
from symbiont_ecology.environment.grid import GridEnvironment


def _regex_env(*, seed: int = 123) -> GridEnvironment:
    grid_cfg = GridConfig(families=["regex"], depths=["short", "medium", "long"])
    ctrl_cfg = ControllerConfig(tau=0.5, beta=0.2, eta=0.1)
    price_cfg = PricingConfig(base=1.0, k=1.0, min=0.3, max=2.0)
    canary_cfg = CanaryConfig(q_min=0.7)
    return GridEnvironment(grid_cfg, ctrl_cfg, price_cfg, canary_cfg, seed=seed)


def test_make_regex_task_cases_are_self_consistent():
    env = _regex_env(seed=42)
    for depth in ["short", "medium", "long"]:
        for _ in range(50):
            prompt, pattern, test_strings = env._make_regex_task(depth)  # noqa: SLF001
            assert prompt.strip()
            assert pattern.strip()
            assert test_strings
            compiled = re.compile(pattern)
            for tc in test_strings:
                text = str(tc.get("string", ""))
                should_match = bool(tc.get("should_match", False))
                assert bool(compiled.fullmatch(text)) == should_match


def test_regex_gridtask_evaluate_accepts_correct_pattern():
    env = _regex_env(seed=7)
    task = env.sample_task_from_cell(("regex", "short"))
    target = task.target
    assert isinstance(target, dict)
    pattern = str(target.get("pattern", ""))
    ok, _ = task.evaluate(pattern)
    assert ok is True


def test_regex_gridtask_evaluate_rejects_obviously_wrong_pattern():
    env = _regex_env(seed=7)
    task = env.sample_task_from_cell(("regex", "short"))
    ok, _ = task.evaluate(r"^$")
    assert ok is False
