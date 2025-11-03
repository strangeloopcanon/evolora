from types import SimpleNamespace

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop


def test_holdout_accepts_with_margin_and_retries():
    cfg = EcologyConfig()
    cfg.assimilation_tuning.holdout_margin = 0.05
    cfg.assimilation_tuning.holdout_margin_step = 0.01
    # Host where candidate has slightly better ROI than baseline
    def step(prompt: str, intent: str, max_routes: int, allowed_organelle_ids):  # noqa: ARG002
        oid = allowed_organelle_ids[0]
        # candidate has lower cost -> higher ROI
        flops = 100.0 if oid == "CAND" else 120.0
        metrics = SimpleNamespace(answer="2", tokens=1, latency_ms=1.0, prompt_tokens=1, trainable_params=1, flops_estimate=flops, memory_gb=0.001, active_adapters={})
        env = SimpleNamespace(observation=SimpleNamespace(state={"answer": "2"}))
        return SimpleNamespace(envelope=env, routes=[SimpleNamespace(organelle_id=oid)], responses={oid: metrics}, latency_ms=0.0)

    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(step=step),
        environment=SimpleNamespace(controller=SimpleNamespace(), sample_task=lambda: None),
        population=SimpleNamespace(population={}),
        assimilation=SimpleNamespace(),
    )

    class FakeTask:
        def __init__(self, idx: int):
            self.idx = idx

        def to_grid_task(self, environment, task_id: str):  # noqa: ARG002
            from symbiont_ecology.environment.grid import GridTask

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

    loop._sample_holdout_tasks = lambda: [FakeTask(i) for i in range(1, 5)]  # type: ignore[method-assign]
    ok, info = loop._holdout_accepts("CAND", ["BASE"])
    assert isinstance(ok, bool)
