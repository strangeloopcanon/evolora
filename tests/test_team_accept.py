from symbiont_ecology.environment.loops import EcologyLoop


def test_team_accept_gate_true_and_false():
    # Accept when ci_low > baseline + margin and n >= min_tasks
    assert EcologyLoop._team_accept(ci_low=0.12, baseline=0.08, margin=0.01, n=10, min_tasks=8) is True  # type: ignore[attr-defined]
    # Reject when under min tasks
    assert EcologyLoop._team_accept(ci_low=1.0, baseline=0.0, margin=0.0, n=6, min_tasks=8) is False  # type: ignore[attr-defined]
    # Reject when below margin
    assert EcologyLoop._team_accept(ci_low=0.10, baseline=0.095, margin=0.01, n=12, min_tasks=8) is False  # type: ignore[attr-defined]


def test_team_power_proxy_blocks_promotion_when_low():
    # With very high required power, promotion should be blocked in probes
    from types import SimpleNamespace
    from symbiont_ecology import EcologyConfig
    from symbiont_ecology.evolution.assimilation import AssimilationTester
    cfg = EcologyConfig()
    cfg.assimilation_tuning.team_probe_per_gen = 1
    cfg.assimilation_tuning.team_min_tasks = 3
    cfg.assimilation_tuning.holdout_margin = 0.0
    cfg.assimilation_tuning.team_min_power = 0.9
    # Population with two orgs
    class Pop:
        def __init__(self):
            self.population = {"A": object(), "B": object()}
        def average_roi(self, oid: str, limit: int = 5) -> float:  # noqa: ARG002
            return 1.0
    # Host returns identical ROI answers so effect ~0
    def step(prompt: str, intent: str, max_routes: int, allowed_organelle_ids):  # noqa: ARG002
        oid = allowed_organelle_ids[0]
        env = SimpleNamespace(observation=SimpleNamespace(state={"answer": "ok"}))
        metrics = SimpleNamespace(answer="ok", tokens=1, latency_ms=0.0, prompt_tokens=1, trainable_params=0, flops_estimate=1.0, memory_gb=0.001, active_adapters={})
        return SimpleNamespace(envelope=env, responses={oid: metrics})
    # Environment + tasks
    class Task:
        price = 1.0
        def to_grid_task(self, environment, task_id: str):  # noqa: ARG002
            return self
        @property
        def prompt(self):
            return "Q"
        def evaluate(self, answer: str):  # noqa: ARG002
            return True, SimpleNamespace(total=1.0)
    loop = EcologyLoop(config=cfg, host=SimpleNamespace(step=step), environment=SimpleNamespace(rng=__import__("random").Random(0)), population=Pop(), assimilation=AssimilationTester(0.0, 0.5, 0))
    loop._sample_holdout_tasks = lambda: [Task(), Task(), Task()]
    promos = loop._maybe_team_probes()
    assert promos == 0
