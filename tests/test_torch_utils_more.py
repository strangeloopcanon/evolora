import types
import torch

from symbiont_ecology.utils.torch_utils import resolve_device


def test_resolve_device_prefers_cuda_then_mps(monkeypatch) -> None:
    # Simulate CUDA available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    dev = resolve_device("auto")
    assert isinstance(dev, torch.device) and dev.type == "cuda"
    # Now simulate no CUDA but MPS available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)
    monkeypatch.setattr(torch.backends, "mps", types.SimpleNamespace(is_available=lambda: True))
    dev2 = resolve_device("auto")
    assert isinstance(dev2, torch.device) and dev2.type == "mps"

