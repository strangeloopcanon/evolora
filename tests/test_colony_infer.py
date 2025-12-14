from types import SimpleNamespace

from symbiont_ecology.config import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop


def test_colony_inference_best_of_two():
    cfg = EcologyConfig()

    # Host that returns different length answers for A and B
    def step(prompt: str, intent: str, max_routes: int, allowed_organelle_ids):  # noqa: ARG002
        oid = allowed_organelle_ids[0]
        ans = "short" if oid == "A" else "a little bit longer"
        metrics = SimpleNamespace(answer=ans, tokens=1)
        env = SimpleNamespace(observation=SimpleNamespace(state={"answer": ans}))
        return SimpleNamespace(
            envelope=env,
            routes=[SimpleNamespace(organelle_id=oid)],
            responses={oid: metrics},
            latency_ms=0.0,
        )

    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(step=step),
        environment=SimpleNamespace(),
        population=SimpleNamespace(population={}),
        assimilation=SimpleNamespace(),
    )
    result = loop.run_colony_inference(["A", "B"], prompt="p")
    assert result["selected_id"] == "B"
    assert isinstance(result["answers"], dict) and set(result["answers"].keys()) == {"A", "B"}
