import torch

from symbiont_ecology.utils.torch_utils import clamp_norm, ensure_dtype, no_grad, resolve_device


def test_torch_utils_helpers() -> None:
    device = resolve_device("cpu")
    assert isinstance(device, torch.device)
    tensor = torch.tensor([3.0, 4.0])
    clamped = clamp_norm(tensor, max_norm=5.0)
    assert torch.allclose(clamped, tensor)
    clamped_small = clamp_norm(tensor, max_norm=1.0)
    assert torch.linalg.norm(clamped_small) <= 1.0 + 1e-5
    converted = ensure_dtype(tensor, torch.float64)
    assert converted.dtype == torch.float64
    with no_grad():
        tensor.add_(1.0)
