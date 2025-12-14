from types import SimpleNamespace

from symbiont_ecology import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop


def test_run_colony_inference_best_of_two():
    cfg = EcologyConfig()

    # Host that returns different length answers for A and B
    class Host:
        def step(
            self, prompt: str, intent: str, max_routes: int, allowed_organelle_ids
        ):  # noqa: ARG002
            oid = allowed_organelle_ids[0]
            ans = "short" if oid == "A" else "much longer answer"
            metrics = SimpleNamespace(answer=ans, tokens=len(ans.split()))
            return SimpleNamespace(responses={oid: metrics})

    loop = EcologyLoop(
        config=cfg,
        host=Host(),
        environment=SimpleNamespace(),
        population=SimpleNamespace(),
        assimilation=SimpleNamespace(),
    )
    res = loop.run_colony_inference(["A", "B"], "Q", strategy="best_of_two")
    assert res["selected_id"] == "B"
    assert "much longer" in res["selected_answer"]
