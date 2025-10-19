import torch

from symbiont_ecology.evolution.merging import MergeComponent, ModelMerger


def test_model_merger_combines_components() -> None:
    base = torch.ones(4, 4)
    delta = torch.eye(4)
    merger = ModelMerger(
        [
            MergeComponent(name="base", weight=base, alpha=1.0),
            MergeComponent(name="delta", weight=delta, alpha=0.5),
        ]
    )
    merged = merger.merge()
    assert torch.allclose(merged, base + 0.5 * delta)
