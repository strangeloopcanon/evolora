from types import SimpleNamespace

from symbiont_ecology import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop


def test_sample_task_lp_with_canary_path() -> None:
    cfg = EcologyConfig()
    # Canary threshold small to trigger canary path
    env = SimpleNamespace(
        controller=SimpleNamespace(
            sample_cell=lambda lp_mix: ("math", "short"),
            get_state=lambda cell: SimpleNamespace(success_ema=1.0),
        ),
        canary_q_min=0.5,
        rng=SimpleNamespace(random=lambda: 0.0),
        sample_task_from_cell=lambda cell, canary=False: SimpleNamespace(
            prompt="ok", price=1.0, evaluate=lambda a: (True, SimpleNamespace(total=1.0))
        ),
        sample_task=lambda: SimpleNamespace(
            prompt="nope", price=1.0, evaluate=lambda a: (True, SimpleNamespace(total=1.0))
        ),
    )
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=env,
        population=SimpleNamespace(),
        assimilation=SimpleNamespace(),
    )
    task = loop._sample_task_lp(lp_mix=0.3)
    assert getattr(task, "prompt", "") == "ok"


def test_sample_task_with_policy_bias() -> None:
    cfg = EcologyConfig()
    cfg.policy.bias_strength = 1.0  # always honor
    env = SimpleNamespace(
        controller=SimpleNamespace(),
        sample_task_from_cell=lambda cell, canary=False: SimpleNamespace(
            prompt=f"{cell}", price=1.0, evaluate=lambda a: (True, SimpleNamespace(total=1.0))
        ),
        sample_task=lambda: SimpleNamespace(
            prompt="fallback", price=1.0, evaluate=lambda a: (True, SimpleNamespace(total=1.0))
        ),
        rng=SimpleNamespace(random=lambda: 0.0),
    )
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=env,
        population=SimpleNamespace(),
        assimilation=SimpleNamespace(),
    )
    # Install a policy for org X favoring a specific cell
    loop._active_policies = {
        "X": {"cell_pref": {"family": "math", "depth": "short"}, "budget_frac": 1.0}
    }
    task = loop._sample_task_with_policy(lp_mix=0.0, organelle_id="X")
    # Should return a task object with a prompt string (policy or fallback)
    assert isinstance(getattr(task, "prompt", None), str)
