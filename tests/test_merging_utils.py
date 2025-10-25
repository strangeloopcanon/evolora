import torch

from symbiont_ecology.evolution.merging import MergeComponent, ModelMerger


def test_model_merger_merge_with_shape_mismatch_and_alpha():
    base = torch.ones(3, 3)
    delta_small = 2 * torch.ones(2, 2)
    delta_same = 3 * torch.ones(3, 3)
    comps = [
        MergeComponent("base", base),
        MergeComponent("delta_small", delta_small, alpha=0.5),
        MergeComponent("delta_same", delta_same, alpha=1.0),
    ]
    merged = ModelMerger(comps).merge()
    # delta_small should be padded into top-left corner and scaled by 0.5
    expected = base.clone()
    expected[:2, :2] += 0.5 * delta_small
    expected += delta_same
    assert torch.allclose(merged, expected)


def test_model_merger_chain():
    a = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
    b = torch.tensor([[2.0, 0.0], [1.0, 2.0]])
    c = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    chained = ModelMerger.chain([a, b, c])
    # compute explicitly for validation
    expected = a @ b @ c
    assert torch.allclose(chained, expected)


def test_model_merger_errors():
    # empty chain
    import pytest

    with pytest.raises(ValueError):
        _ = ModelMerger.chain([])
    with pytest.raises(ValueError):
        _ = ModelMerger([]).merge()
