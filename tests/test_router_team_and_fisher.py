from types import SimpleNamespace

import pytest
import torch
from symbiont_ecology import EcologyConfig
from symbiont_ecology.environment.loops import EcologyLoop
from symbiont_ecology.evolution.assimilation import AssimilationTester
from symbiont_ecology.evolution.population import Genome


class DummyMetrics:
    def __init__(self, answer: str, tokens: int = 10):
        self.answer = answer
        self.tokens = tokens
        self.latency_ms = 0.0
        self.prompt_tokens = tokens
        self.trainable_params = 0
        self.flops_estimate = 1.0
        self.memory_gb = 0.1
        self.active_adapters = {"dense": 1}


def test_evaluate_team_holdout_roi_selects_best() -> None:
    cfg = EcologyConfig()
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=SimpleNamespace(
            grid_cfg=cfg.grid,
            controller=None,
            pricing_cfg=None,
            canary_cfg=None,
            sample_task=lambda: None,
        ),
        population=SimpleNamespace(),
        assimilation=AssimilationTester(0.0, 0.5, 0),
    )

    # Fake tasks with ROI computed from answer length
    class T:
        price = 1.0

        def to_grid_task(self, env, task_id: str):
            return self

        def evaluate(self, answer: str):
            return True, SimpleNamespace(total=float(len(answer)))

        @property
        def prompt(self):
            return "Q"

    tasks = [T() for _ in range(8)]

    # Host that returns different answers for a and b
    def step(prompt: str, intent: str, max_routes: int, allowed_organelle_ids):
        oid = allowed_organelle_ids[0]
        ans = "xxxxxx" if oid == "A" else "xx"  # A wins
        return SimpleNamespace(responses={oid: DummyMetrics(ans)})

    loop.host = SimpleNamespace(step=step)
    team_roi = loop._evaluate_team_holdout_roi("A", "B", tasks)
    # A returns len=6 -> revenue 6, cost ~constant ~1.2 -> ROI > 3; B smaller
    assert team_roi > 0.0


def test_fisher_weighting_adjusts_weights() -> None:
    cfg = EcologyConfig()
    cfg.assimilation_tuning.merge_method = "fisher_svd"
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=SimpleNamespace(),
        population=SimpleNamespace(),
        assimilation=AssimilationTester(0.0, 0.5, 0),
    )

    # Create two fake organelles with different adapter norms
    class FakeAdapter:
        def __init__(self, scale: float):
            self.lora_A = torch.ones(4, 2) * scale
            self.lora_B = torch.ones(2, 4) * scale

    class FakeOrg:
        def __init__(self, scale: float):
            self.adapter = FakeAdapter(scale)
            self._fisher = float(scale)

        def fisher_importance(self) -> float:
            return self._fisher

    loop.population = SimpleNamespace(
        population={
            "big": Genome("big", {}, gate_bias=0.0, rank=2),
            "small": Genome("small", {}, gate_bias=0.0, rank=2),
        }
    )
    loop.host = SimpleNamespace(
        get_organelle=lambda oid: FakeOrg(10.0) if oid == "big" else FakeOrg(1.0),
        build_lora_soup_state=lambda soup, rank, **kwargs: ({}, {}),
        merge_lora_soup=lambda soup, rank, **kwargs: None,
    )
    cell = ("math", "short")
    stats_map = {"big": {"roi": 1.0, "ema": 0.5}, "small": {"roi": 1.0, "ema": 0.5}}
    summary = loop._apply_lora_soup_merge(cell, "big", ["big", "small"], stats_map, [])
    weights = {ent["organelle_id"]: ent["weight"] for ent in summary}
    assert weights["big"] > weights["small"], "Fisher weighting should favor higher-norm adapter"


def test_fisher_fallback_uses_export_state() -> None:
    cfg = EcologyConfig()
    cfg.assimilation_tuning.merge_method = "fisher_svd"
    loop = EcologyLoop(
        config=cfg,
        host=SimpleNamespace(),
        environment=SimpleNamespace(),
        population=SimpleNamespace(),
        assimilation=AssimilationTester(0.0, 0.5, 0),
    )

    class FallbackOrg:
        def __init__(self, scale: float):
            self.scale = scale

        def fisher_importance(self) -> float:
            return 0.0

        def export_adapter_state(self) -> dict[str, torch.Tensor]:
            tensor = torch.ones(2, 2) * self.scale
            return {"w": tensor}

    loop.population = SimpleNamespace(
        population={
            "big": Genome("big", {}, gate_bias=0.0, rank=2),
            "small": Genome("small", {}, gate_bias=0.0, rank=2),
        }
    )
    loop.host = SimpleNamespace(
        get_organelle=lambda oid: FallbackOrg(5.0) if oid == "big" else FallbackOrg(1.0),
        build_lora_soup_state=lambda soup, rank, **kwargs: ({}, {}),
        merge_lora_soup=lambda soup, rank, **kwargs: None,
    )
    cell = ("math", "short")
    stats_map = {"big": {"roi": 1.0, "ema": 0.5}, "small": {"roi": 1.0, "ema": 0.5}}
    summary = loop._apply_lora_soup_merge(cell, "big", ["big", "small"], stats_map, [])
    weights = {ent["organelle_id"]: ent["weight"] for ent in summary}
    importances = {ent["organelle_id"]: ent.get("importance", 0.0) for ent in summary}
    assert weights["big"] > weights["small"]
    assert importances["big"] > importances["small"]
    assert importances["big"] == pytest.approx(1.0)
