import torch

from symbiont_ecology import EcologyConfig
from symbiont_ecology.evolution.ledger import ATPLedger
from symbiont_ecology.host.kernel import HostKernel
from symbiont_ecology.routing.router import BanditRouter


class FakeOrg2:
    def __init__(self):
        self._state = {
            "lora_A": torch.ones(2, 2),
            "lora_B": torch.ones(2, 2) * 2,
        }

    def export_adapter_state(self):
        return self._state


def test_merge_organelle_and_soup_updates_state():
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    host.organelles["x"] = FakeOrg2()
    # Ensure ledger accounts exist for organelle
    host.ledger.ensure("x", 1.0)
    host.ledger.ensure_energy("x", 1.0)
    # Merge organelle into host assimilation state
    host.merge_organelle_into_host("x", alpha=0.5)
    assert "lora_A" in host.assimilation_state
    # Merge a soup back into host
    host.merge_lora_soup({"x": 0.25}, target_rank=2)
    assert "lora_B" in host.assimilation_state
    assert host.assimilation_weights.get("lora_A", 0.0) > 0.0


class FakeOrgRole:
    def __init__(self, base: float) -> None:
        self._state = {
            "lora_A": torch.ones(2, 3) * base,
            "lora_B": torch.ones(4, 2) * (base + 1.0),
        }

    def export_adapter_state(self):
        return self._state


def test_build_lora_soup_state_block_diagonal():
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    host.organelles["a"] = FakeOrgRole(1.0)
    host.organelles["b"] = FakeOrgRole(3.0)
    host.ledger.ensure("a", 1.0)
    host.ledger.ensure("b", 1.0)
    host.ledger.ensure_energy("a", 1.0)
    host.ledger.ensure_energy("b", 1.0)
    roles = {"a": 0, "b": 1}
    soup_state, _ = host.build_lora_soup_state(
        {"a": 1.0, "b": 1.0}, target_rank=4, roles=roles, mode="block"
    )
    assert "lora_A" in soup_state and "lora_B" in soup_state
    lora_a = soup_state["lora_A"]
    lora_b = soup_state["lora_B"]
    assert lora_a.shape == (4, 3)
    assert lora_b.shape == (4, 4)
    assert torch.allclose(lora_a[:2], torch.ones(2, 3))
    assert torch.allclose(lora_a[2:], torch.ones(2, 3) * 3.0)
    assert torch.allclose(lora_b[:, :2], torch.ones(4, 2) * 2.0)
    assert torch.allclose(lora_b[:, 2:], torch.ones(4, 2) * 4.0)


class FakeOrgMutation:
    def export_adapter_state(self):
        return {
            "layer.attn.q": torch.ones(2, 2),
            "layer.mlp.up": torch.ones(2, 2) * 2.0,
        }


def test_build_lora_soup_state_respects_mutation_meta():
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    host.organelles["mut"] = FakeOrgMutation()
    host.ledger.ensure("mut", 1.0)
    host.ledger.ensure_energy("mut", 1.0)
    soup_state, alpha_sum = host.build_lora_soup_state(
        {"mut": 1.0},
        target_rank=2,
        mutation_meta={
            "mut": {"dropout": ["attn"], "duplication": {"mlp": 1.0}, "rank_noise": {"mlp": 1.0}}
        },
    )
    assert "layer.attn.q" not in soup_state
    assert "layer.mlp.up" in soup_state
    assert alpha_sum["layer.mlp.up"] == 2.0
