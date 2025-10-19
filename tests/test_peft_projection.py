import torch

from symbiont_ecology.organelles.peft_hebbian import HebbianPEFTOrganelle as ActualPEFTOrganelle


def test_recompose_preserves_delta_when_expanding() -> None:
    original = ActualPEFTOrganelle
    old_a = torch.randn(4, 6, dtype=torch.float32)
    old_b = torch.randn(5, 4, dtype=torch.float32)
    new_rank = 6
    new_a = torch.zeros(new_rank, old_a.shape[1], dtype=torch.float32)
    new_b = torch.zeros(old_b.shape[0], new_rank, dtype=torch.float32)

    projected_a, projected_b = original._recompose_from_history(  # type: ignore[attr-defined]
        new_rank=new_rank,
        new_a=new_a,
        new_b=new_b,
        old_a=old_a,
        old_b=old_b,
        module_name="test.module",
    )

    delta_original = old_b @ old_a
    delta_projected = projected_b @ projected_a
    assert torch.allclose(delta_original, delta_projected, atol=1e-4, rtol=1e-4)


def test_recompose_truncates_sensibly_when_shrinking() -> None:
    original = ActualPEFTOrganelle
    old_a = torch.randn(6, 8, dtype=torch.float32)
    old_b = torch.randn(7, 6, dtype=torch.float32)
    new_rank = 3
    new_a = torch.zeros(new_rank, old_a.shape[1], dtype=torch.float32)
    new_b = torch.zeros(old_b.shape[0], new_rank, dtype=torch.float32)

    projected_a, projected_b = original._recompose_from_history(  # type: ignore[attr-defined]
        new_rank=new_rank,
        new_a=new_a,
        new_b=new_b,
        old_a=old_a,
        old_b=old_b,
        module_name="test.module",
    )

    delta_original = old_b @ old_a
    delta_projected = projected_b @ projected_a
    rel_error = torch.linalg.norm(delta_projected - delta_original) / (
        torch.linalg.norm(delta_original) + 1e-6
    )
    assert rel_error < 0.5
