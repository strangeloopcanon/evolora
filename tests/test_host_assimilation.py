import torch

from symbiont_ecology import EcologyConfig
from symbiont_ecology.host.kernel import HostKernel
from symbiont_ecology.evolution.ledger import ATPLedger
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
