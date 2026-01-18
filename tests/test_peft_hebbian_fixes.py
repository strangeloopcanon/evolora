import torch

from symbiont_ecology.config import HebbianConfig
from symbiont_ecology.organelles.base import OrganelleContext
from symbiont_ecology.organelles.peft_hebbian import HebbianPEFTOrganelle, TraceStore


class _DummyPeftLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, adapter: str) -> None:
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({adapter: torch.nn.Linear(in_features, rank, bias=False)})
        self.lora_B = torch.nn.ModuleDict(
            {adapter: torch.nn.Linear(rank, out_features, bias=False)}
        )


class _DummyModel(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.layer = layer
        self.active_adapter: str | None = None

    def set_adapter(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name


def _make_organelle(
    organelle_id: str, model: torch.nn.Module, *, rank: int, hebbian: HebbianConfig | None = None
) -> HebbianPEFTOrganelle:
    context = OrganelleContext(organelle_id=organelle_id, hebbian=hebbian or HebbianConfig())
    organelle = HebbianPEFTOrganelle.__new__(HebbianPEFTOrganelle)
    super(HebbianPEFTOrganelle, organelle).__init__(organelle_id=organelle_id, context=context)
    organelle.model = model
    organelle.rank = rank
    organelle.device = torch.device("cpu")
    organelle.traces = TraceStore(
        pre=torch.ones(4, dtype=torch.float32),
        post=torch.ones(4, dtype=torch.float32),
    )
    organelle._fisher_importance = 0.0
    organelle._last_activation_scale = 0.0
    return organelle


def test_inherit_from_projects_rank_and_prefers_strong_components() -> None:
    parent_layer = _DummyPeftLayer(in_features=4, out_features=6, rank=4, adapter="parent")
    child_layer = _DummyPeftLayer(in_features=4, out_features=6, rank=2, adapter="child")
    parent_model = _DummyModel(parent_layer)
    child_model = _DummyModel(child_layer)

    parent = _make_organelle("parent", parent_model, rank=4)
    child = _make_organelle("child", child_model, rank=2)

    with torch.no_grad():
        parent_layer.lora_A["parent"].weight.zero_()
        parent_layer.lora_B["parent"].weight.zero_()
        # Make rank components have distinct norms so selection is deterministic.
        for idx in range(4):
            parent_layer.lora_A["parent"].weight[idx, :] = float(idx + 1)
            parent_layer.lora_B["parent"].weight[:, idx] = float(idx + 1)
        child_layer.lora_A["child"].weight.zero_()
        child_layer.lora_B["child"].weight.zero_()

    child.inherit_from(parent)

    # Should pick the two strongest components (idx=3,2) and project them into rank=2.
    assert torch.allclose(child_layer.lora_A["child"].weight[0], torch.full((4,), 4.0))
    assert torch.allclose(child_layer.lora_A["child"].weight[1], torch.full((4,), 3.0))
    assert torch.allclose(child_layer.lora_B["child"].weight[:, 0], torch.full((6,), 4.0))
    assert torch.allclose(child_layer.lora_B["child"].weight[:, 1], torch.full((6,), 3.0))


def test_update_lora_pair_handles_dim_mismatch() -> None:
    layer = _DummyPeftLayer(in_features=4, out_features=9, rank=3, adapter="org")
    model = _DummyModel(layer)
    organelle = _make_organelle("org", model, rank=3, hebbian=HebbianConfig(learning_rate=1e-2))
    organelle.traces = TraceStore(
        pre=torch.ones(4, dtype=torch.float32),
        post=torch.ones(4, dtype=torch.float32),  # smaller than out_features -> projection path
    )

    with torch.no_grad():
        layer.lora_A["org"].weight.zero_()
        layer.lora_B["org"].weight.zero_()

    organelle._update_lora_pair(layer, scale=1.0, module_name="layer")
    assert torch.count_nonzero(layer.lora_A["org"].weight) > 0
    assert torch.count_nonzero(layer.lora_B["org"].weight) > 0
