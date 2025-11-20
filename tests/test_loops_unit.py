from types import SimpleNamespace

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.environment.grid import GridTask


def _mk_loop(roi_value: float = 1.0, lp_vals: dict | None = None) -> EcologyLoop:
    cfg = EcologyConfig()
    # enable auto-batch knobs for testing
    cfg.environment.auto_batch = True
    cfg.environment.batch_min = 1
    cfg.environment.batch_max = 4
    host = SimpleNamespace()
    # environment with controller.lp_progress
    controller = SimpleNamespace(lp_progress=lp_vals or {("word.count", "short"): 0.1, ("math", "short"): 0.5})
    environment = SimpleNamespace(controller=controller)
    # population stub with aggregate_roi
    population = SimpleNamespace(aggregate_roi=lambda limit=5: roi_value)
    assimilation = SimpleNamespace()
    loop = EcologyLoop(config=cfg, host=host, environment=environment, population=population, assimilation=assimilation)
    return loop


def test_compute_batch_size_auto_min_max():
    # low ROI -> min
    loop = _mk_loop(roi_value=0.4)
    assert loop._compute_batch_size(3) == 1
    # high ROI -> max
    loop2 = _mk_loop(roi_value=1.6)
    assert loop2._compute_batch_size(2) == 4


def test_resolve_lp_mix_respects_bounds_and_smoothing():
    lp = {("word.count", "short"): 0.2, ("math", "short"): 0.6}
    loop = _mk_loop(lp_vals=lp)
    cfg = loop.config.curriculum
    cfg.lp_mix = 0.2
    cfg.lp_mix_min = 0.1
    cfg.lp_mix_max = 0.6
    cfg.alp_auto_mix = True
    mix = loop._resolve_lp_mix(cfg.lp_mix)
    assert 0.1 <= mix <= 0.6


def test_team_probe_can_promote_colony():
    cfg = EcologyConfig()
    cfg.assimilation_tuning.team_probe_per_gen = 1
    cfg.assimilation_tuning.team_min_tasks = 6
    cfg.assimilation_tuning.holdout_margin = 0.001
    cfg.assimilation_tuning.team_min_power = 0.0
    # Simple population with two orgs
    class Pop:
        def __init__(self):
            self.population = {"org_a": object(), "org_b": object()}

        def average_roi(self, oid: str, limit: int = 5) -> float:  # noqa: ARG002
            return 1.0

    # Host that returns metrics with cost depending on task index and org id
    class Host:
        def __init__(self):
            self.calls = []

        def step(self, prompt: str, intent: str, max_routes: int, allowed_organelle_ids):  # noqa: ARG002
            oid = allowed_organelle_ids[0]
            try:
                idx = int(prompt.split(":")[1])
            except Exception:
                idx = 0
            # Lower cost alternates by org and idx parity to create complementarity
            base = SimpleNamespace(answer="2", tokens=5, latency_ms=1.0, prompt_tokens=5, trainable_params=10, flops_estimate=1000.0, memory_gb=0.001, active_adapters={})
            # Make cost smaller (higher ROI) when parity matches
            if (oid.endswith("a") and idx % 2 == 0) or (oid.endswith("b") and idx % 2 == 1):
                base.flops_estimate = 100.0
            resp = {oid: base}
            env = SimpleNamespace(observation=SimpleNamespace(state={"answer": "2"}))
            return SimpleNamespace(envelope=env, routes=[SimpleNamespace(organelle_id=oid)], responses=resp, latency_ms=1.0)

    # Environment providing RNG and a simple controller (unused here)
    environment = SimpleNamespace(rng=__import__("random").Random(0), controller=SimpleNamespace())

    loop = EcologyLoop(config=cfg, host=Host(), environment=environment, population=Pop(), assimilation=SimpleNamespace())

    # Provide a custom holdout sampler returning tasks with explicit indices in prompt
    class FakeTask:
        def __init__(self, idx: int):
            self.idx = idx

        def to_grid_task(self, environment, task_id: str):  # noqa: ARG002
            return GridTask(
                task_id=task_id,
                cell=("word.count", "short"),
                prompt=f"idx:{self.idx}",
                price=1.0,
                target=2,
                family="word.count",
                depth="short",
                difficulty=0.5,
            )

    loop._sample_holdout_tasks = lambda: [FakeTask(i) for i in range(1, 7)]  # type: ignore[method-assign]
    promoted = loop._maybe_team_probes()
    assert isinstance(promoted, int)
    samples = getattr(loop, "_team_gate_samples", [])
    assert samples, "team gate telemetry should record attempts"
    assert any("answer_samples" in sample for sample in samples)
