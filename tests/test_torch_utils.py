import torch

from symbiont_ecology.utils.torch_utils import clamp_norm, ensure_dtype, resolve_device, no_grad


def test_clamp_norm_scales_tensor_down():
    t = torch.ones(4)
    out = clamp_norm(t, max_norm=1.0)
    assert torch.linalg.norm(out) <= 1.0000001


def test_ensure_dtype_converts_dtype():
    t = torch.ones(2, dtype=torch.float32)
    out = ensure_dtype(t, torch.float16)
    assert out.dtype == torch.float16
    # idempotent when already correct
    out2 = ensure_dtype(out, torch.float16)
    assert out2.dtype == torch.float16


def test_resolve_device_string_cpu():
    dev = resolve_device("cpu")
    assert str(dev) == "cpu"


def test_no_grad_context_manager():
    with no_grad():
        t = torch.ones(2, requires_grad=True) * 2
        # Autograd disabled; result should not require grad
        assert t.requires_grad is False
