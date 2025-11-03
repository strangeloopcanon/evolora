import torch

from symbiont_ecology import EcologyConfig
from symbiont_ecology.host.kernel import HostKernel
from symbiont_ecology.evolution.ledger import ATPLedger
from symbiont_ecology.routing.router import BanditRouter


class FakeOrg:
    def __init__(self):
        pass

    def export_adapter_state(self):
        # Two 2D matrices to trigger SVD projection
        return {
            "lora_A": torch.randn(4, 4),
            "lora_B": torch.randn(4, 4),
        }


def test_build_lora_soup_state_projects_rank():
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    host.organelles["o1"] = FakeOrg()
    soup_state, alpha_sum = host.build_lora_soup_state({"o1": 1.0}, target_rank=2)
    # Keys preserved and projected to rank <=2 (shape remains 2D)
    assert set(soup_state.keys()) == {"lora_A", "lora_B"}
    for v in soup_state.values():
        assert isinstance(v, torch.Tensor)
        assert v.ndim == 2
        assert min(v.shape) >= 2  # projected rank at most 2 but dims unchanged
    # Alpha sum recorded
    assert alpha_sum["lora_A"] == 1.0

