from pathlib import Path
import torch

from symbiont_ecology import EcologyConfig
from symbiont_ecology.host.kernel import HostKernel
from symbiont_ecology.evolution.ledger import ATPLedger
from symbiont_ecology.routing.router import BanditRouter


class FakeOrgExport:
    def __init__(self):
        pass
    def export_adapter_state(self):
        return {"lora_A": torch.ones(2, 2), "lora_B": torch.zeros(2, 2)}


class FakeOrgImport:
    def __init__(self):
        self.loaded = None
    def export_adapter_state(self):
        return {"lora_A": torch.zeros(2, 2)}
    def import_adapter_state(self, state, alpha=1.0):  # noqa: ARG002
        self.loaded = state


def test_export_import_adapter(tmp_path: Path):
    cfg = EcologyConfig()
    host = HostKernel(config=cfg, router=BanditRouter(), ledger=ATPLedger())
    host.organelles["a"] = FakeOrgExport()
    host.organelles["b"] = FakeOrgImport()
    out = tmp_path / "snap.pt"
    host.export_organelle_adapter("a", out)
    assert out.exists()
    host.import_organelle_adapter("b", out, alpha=0.5)
    assert isinstance(host.organelles["b"].loaded, dict)
