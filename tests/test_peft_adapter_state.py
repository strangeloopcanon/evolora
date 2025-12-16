import pytest
import torch

from symbiont_ecology.config import HebbianConfig
from symbiont_ecology.organelles.base import OrganelleContext
from symbiont_ecology.organelles.peft_hebbian import HebbianPEFTOrganelle, TraceStore


class _DummyPeftLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.lora_A = torch.nn.ModuleDict(
            {
                "default": torch.nn.Linear(in_features, rank, bias=False),
                "org": torch.nn.Linear(in_features, rank, bias=False),
            }
        )
        self.lora_B = torch.nn.ModuleDict(
            {
                "default": torch.nn.Linear(rank, out_features, bias=False),
                "org": torch.nn.Linear(rank, out_features, bias=False),
            }
        )


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = _DummyPeftLayer(in_features=4, out_features=5, rank=2)
        self.active_adapter: str | None = None

    def set_adapter(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name


def _make_organelle(
    model: torch.nn.Module, hebbian: HebbianConfig | None = None
) -> HebbianPEFTOrganelle:
    context = OrganelleContext(organelle_id="org", hebbian=hebbian or HebbianConfig())
    organelle = HebbianPEFTOrganelle.__new__(HebbianPEFTOrganelle)
    super(HebbianPEFTOrganelle, organelle).__init__(organelle_id="org", context=context)
    organelle.model = model
    organelle.rank = 2
    organelle.device = torch.device("cpu")
    organelle.traces = TraceStore(
        pre=torch.ones(4, dtype=torch.float32),
        post=torch.ones(5, dtype=torch.float32),
    )
    organelle._fisher_importance = 0.0
    organelle._last_activation_scale = 0.0
    return organelle


def test_export_import_load_adapter_state_handles_moduledict() -> None:
    model = _DummyModel()
    organelle = _make_organelle(model)

    exported = organelle.export_adapter_state()
    assert "layer.lora_A" in exported
    assert "layer.lora_B" in exported
    assert exported["layer.lora_A"].shape == model.layer.lora_A["org"].weight.shape
    assert exported["layer.lora_B"].shape == model.layer.lora_B["org"].weight.shape

    with torch.no_grad():
        model.layer.lora_A["org"].weight.zero_()
        model.layer.lora_B["org"].weight.zero_()

    organelle.load_adapter_state(exported)
    assert torch.allclose(model.layer.lora_A["org"].weight, exported["layer.lora_A"])
    assert torch.allclose(model.layer.lora_B["org"].weight, exported["layer.lora_B"])

    with torch.no_grad():
        model.layer.lora_A["org"].weight.zero_()
        model.layer.lora_B["org"].weight.zero_()

    organelle.import_adapter_state(exported, alpha=1.0)
    assert torch.allclose(model.layer.lora_A["org"].weight, exported["layer.lora_A"])
    assert torch.allclose(model.layer.lora_B["org"].weight, exported["layer.lora_B"])


def test_update_targets_organelle_adapter_not_default() -> None:
    model = _DummyModel()
    organelle = _make_organelle(model)

    with torch.no_grad():
        model.layer.lora_A["default"].weight.zero_()
        model.layer.lora_B["default"].weight.zero_()
        model.layer.lora_A["org"].weight.zero_()
        model.layer.lora_B["org"].weight.zero_()

    organelle._update_lora_pair(model.layer, scale=1.0)

    assert torch.count_nonzero(model.layer.lora_A["org"].weight) > 0
    assert torch.count_nonzero(model.layer.lora_B["org"].weight) > 0
    assert torch.count_nonzero(model.layer.lora_A["default"].weight) == 0
    assert torch.count_nonzero(model.layer.lora_B["default"].weight) == 0


def test_update_lora_pair_respects_hebbian_learning_rate() -> None:
    model_slow = _DummyModel()
    model_fast = _DummyModel()
    organelle_slow = _make_organelle(model_slow, HebbianConfig(learning_rate=1e-3))
    organelle_fast = _make_organelle(model_fast, HebbianConfig(learning_rate=2e-3))

    with torch.no_grad():
        model_slow.layer.lora_A["org"].weight.zero_()
        model_slow.layer.lora_B["org"].weight.zero_()
        model_fast.layer.lora_A["org"].weight.zero_()
        model_fast.layer.lora_B["org"].weight.zero_()

    organelle_slow._update_lora_pair(model_slow.layer, scale=1.0)
    organelle_fast._update_lora_pair(model_fast.layer, scale=1.0)

    slow_norm = float(model_slow.layer.lora_A["org"].weight.norm().item())
    fast_norm = float(model_fast.layer.lora_A["org"].weight.norm().item())

    assert slow_norm > 0.0
    assert fast_norm == pytest.approx(2.0 * slow_norm, rel=1e-3)
