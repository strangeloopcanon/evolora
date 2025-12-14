from pathlib import Path

import pytest
import torch
from symbiont_ecology import EcologyConfig
from symbiont_ecology.evolution.ledger import ATPLedger
from symbiont_ecology.host.kernel import HostKernel
from symbiont_ecology.interfaces.messages import MessageEnvelope, Observation, Plan
from symbiont_ecology.metrics.telemetry import RewardBreakdown
from symbiont_ecology.routing.router import BanditRouter


def test_freeze_host_freezes_backbone_model_parameters() -> None:
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    host.backbone.model = torch.nn.Linear(4, 4)
    for param in host.backbone.model.parameters():
        assert param.requires_grad is True
    host.freeze_host()
    for param in host.backbone.model.parameters():
        assert param.requires_grad is False


def test_spawn_organelle_enforces_capacity() -> None:
    cfg = EcologyConfig()
    cfg.organism.max_organelles = 0
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    with pytest.raises(RuntimeError):
        host.spawn_organelle(rank=1)


def test_resize_organelle_rank_updates_rank_attribute() -> None:
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())

    class FakeResizable:
        def __init__(self) -> None:
            self.rank = 1

        def resize_rank(self, new_rank: int) -> bool:
            return new_rank != self.rank

    org = FakeResizable()
    host.organelles["x"] = org
    assert host.resize_organelle_rank("x", 2) is True
    assert org.rank == 2


def test_export_organelle_adapter_validates_state(tmp_path: Path) -> None:
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())

    class EmptyExport:
        def export_adapter_state(self):
            return {}

    host.organelles["empty"] = EmptyExport()
    with pytest.raises(ValueError):
        host.export_organelle_adapter("empty", tmp_path / "empty.safetensors")

    class BadExport:
        def export_adapter_state(self):
            return {"lora_A": "not-a-tensor"}

    host.organelles["bad"] = BadExport()
    with pytest.raises(TypeError):
        host.export_organelle_adapter("bad", tmp_path / "bad.safetensors")


def test_import_organelle_adapter_missing_organelle_raises(tmp_path: Path) -> None:
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    with pytest.raises(KeyError):
        host.import_organelle_adapter("missing", tmp_path / "missing.safetensors")


def test_step_recurrence_includes_history_in_prompt() -> None:
    cfg = EcologyConfig()
    cfg.host.recurrence_history_template = "Previous:\n{history}\n(pass {pass_idx}/{total_passes})"
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    host.freeze_host()
    organelle_id = host.spawn_organelle(rank=1)
    result = host.step(prompt="Add 2 and 3.", intent="solve", max_routes=1, recurrent_passes=2)
    assert result.responses[organelle_id].recurrent_passes == 2
    assert result.envelope.observation.state.get("recurrent_pass") == 2
    assert result.envelope.observation.state.get("recurrent_history")


def test_apply_reward_skips_unknown_organelle_id() -> None:
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    envelope = MessageEnvelope(
        observation=Observation(state={"text": "hi"}),
        intent=host.router.intent_factory("solve", []),
        plan=Plan(steps=[], confidence=0.1),
    )
    host.apply_reward(
        envelope,
        rewards={
            "missing": RewardBreakdown(
                task_reward=1.0,
                novelty_bonus=0.0,
                competence_bonus=0.0,
                helper_bonus=0.0,
                risk_penalty=0.0,
                cost_penalty=0.0,
            )
        },
    )


def test_kernel_helpers_cover_counts_and_estimates() -> None:
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())

    class Tokenizer:
        def __call__(self, text):
            return {"input_ids": [1, 2, 3, 4]}

    host.backbone.tokenizer = Tokenizer()

    class Adapter:
        def __init__(self) -> None:
            self.rank = 3
            self.layer = torch.nn.Linear(2, 2)

        def parameters(self):
            return self.layer.parameters()

    class Org:
        organelle_id = "o"

        def __init__(self) -> None:
            self.adapter = Adapter()
            self.rank = 2

        def trainable_parameters(self) -> int:
            return 7

        def active_adapters(self) -> dict[str, int]:
            return {"adapter": 1}

        def estimate_trainable(self, new_rank: int) -> int:
            return new_rank * 10

    org = Org()
    assert host._count_tokens("hello world") == 4
    assert host._compute_energy_cost("hello world", org) > 0.0
    assert host._trainable_params(org) == 7
    assert host._active_adapters(org) == {"adapter": 1}
    assert host.estimate_trainable(org, 4) == 40

    host.organelles["o"] = org
    assert host.total_trainable_parameters() == 7

    host.backbone.model = torch.nn.Linear(2, 2)
    assert host.total_backbone_params() > 0


def test_default_recurrence_passes_respects_intent_markers() -> None:
    cfg = EcologyConfig()
    cfg.host.recurrence_enabled = True
    cfg.host.recurrence_train_passes = 2
    cfg.host.recurrence_eval_passes = 3
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    assert host._default_recurrence_passes("solve") == 2
    assert host._default_recurrence_passes("evaluation") == 3


def test_count_tokens_falls_back_on_tokenizer_error() -> None:
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())

    class BadTokenizer:
        def __call__(self, text):
            raise RuntimeError("boom")

    host.backbone.tokenizer = BadTokenizer()
    assert host._count_tokens("one two") == 2


def test_estimate_trainable_fallback_scales_by_rank() -> None:
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())

    class Org:
        def __init__(self) -> None:
            self.rank = 2
            self.adapter = torch.nn.Linear(2, 2)

    org = Org()
    estimate = host.estimate_trainable(org, new_rank=4)
    assert estimate == 12


def test_import_organelle_adapter_swallow_import_errors(tmp_path: Path) -> None:
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())

    class Exporter:
        def export_adapter_state(self):
            return {"lora_A": torch.ones(2, 2)}

    class FailingImporter:
        def export_adapter_state(self):
            return {"lora_A": torch.zeros(2, 2)}

        def import_adapter_state(self, state, alpha=1.0):
            raise RuntimeError("boom")

    host.organelles["a"] = Exporter()
    host.organelles["b"] = FailingImporter()
    out = tmp_path / "snap.safetensors"
    host.export_organelle_adapter("a", out)
    host.import_organelle_adapter("b", out, alpha=0.5)


def test_recurrence_template_format_error_falls_back() -> None:
    cfg = EcologyConfig()
    cfg.host.recurrence_history_template = "{missing}"
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    prompt = host._format_recurrence_prompt(
        "base",
        history=["answer"],
        pass_idx=2,
        total_passes=3,
    )
    assert "Previous passes" in prompt


def test_active_adapters_error_returns_empty_dict() -> None:
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())

    class Org:
        def active_adapters(self):
            raise RuntimeError("boom")

    assert host._active_adapters(Org()) == {}


def test_resize_organelle_rank_missing_id_returns_false() -> None:
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    assert host.resize_organelle_rank("missing", 2) is False


def test_export_organelle_adapter_rejects_non_string_keys(tmp_path: Path) -> None:
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())

    class BadKeys:
        def export_adapter_state(self):
            return {1: torch.ones(2, 2)}

    host.organelles["bad"] = BadKeys()
    with pytest.raises(TypeError):
        host.export_organelle_adapter("bad", tmp_path / "bad.safetensors")
