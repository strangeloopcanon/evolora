"""CI-safe tests for mask derivation and serialization."""

from __future__ import annotations

import tempfile

import torch

from plasticity.masks import (
    MaskSet,
    derive_global_masks,
    derive_masks,
    derive_random_masks,
)


def _fake_importance(shape: tuple = (16, 8)) -> dict[str, torch.Tensor]:
    """Create deterministic importance tensors for two modules."""
    torch.manual_seed(42)
    return {
        "layer.0.linear": torch.rand(shape),
        "layer.1.linear": torch.rand(shape),
    }


class TestDeriveMasks:
    def test_sparsity_50_prunes_half(self) -> None:
        imp = _fake_importance()
        mask = derive_masks(imp, sparsity=0.5)
        for m in mask.masks.values():
            frac_pruned = (~m).sum().item() / m.numel()
            assert abs(frac_pruned - 0.5) < 0.02

    def test_sparsity_0_keeps_all(self) -> None:
        imp = _fake_importance()
        mask = derive_masks(imp, sparsity=0.0)
        for m in mask.masks.values():
            assert m.all()

    def test_sparsity_1_prunes_all(self) -> None:
        imp = _fake_importance()
        mask = derive_masks(imp, sparsity=1.0)
        for m in mask.masks.values():
            assert not m.any()

    def test_invalid_sparsity_raises(self) -> None:
        imp = _fake_importance()
        import pytest

        with pytest.raises(ValueError):
            derive_masks(imp, sparsity=-0.1)
        with pytest.raises(ValueError):
            derive_masks(imp, sparsity=1.1)

    def test_mask_shape_matches_importance(self) -> None:
        imp = _fake_importance(shape=(32, 16))
        mask = derive_masks(imp, sparsity=0.3)
        for name in imp:
            assert mask.masks[name].shape == imp[name].shape

    def test_high_importance_weights_survive(self) -> None:
        imp = {"mod": torch.tensor([[10.0, 1.0], [0.5, 0.1]])}
        mask = derive_masks(imp, sparsity=0.5)
        assert mask.masks["mod"][0, 0].item()  # highest should survive


class TestDeriveRandomMasks:
    def test_sparsity_level_correct(self) -> None:
        imp = _fake_importance()
        mask = derive_random_masks(imp, sparsity=0.5, seed=42)
        sp = mask.sparsity()
        assert abs(sp - 0.5) < 0.02

    def test_deterministic_with_same_seed(self) -> None:
        imp = _fake_importance()
        a = derive_random_masks(imp, 0.5, seed=42)
        b = derive_random_masks(imp, 0.5, seed=42)
        for name in a.masks:
            assert (a.masks[name] == b.masks[name]).all()

    def test_different_seed_gives_different_mask(self) -> None:
        imp = _fake_importance()
        a = derive_random_masks(imp, 0.5, seed=1)
        b = derive_random_masks(imp, 0.5, seed=2)
        any_diff = any(not (a.masks[n] == b.masks[n]).all() for n in a.masks)
        assert any_diff


class TestDeriveGlobalMasks:
    def test_global_averages_families(self) -> None:
        torch.manual_seed(1)
        per_fam = {
            "regex": {"mod": torch.tensor([[1.0, 0.2], [0.1, 0.8]])},
            "math": {"mod": torch.tensor([[0.3, 0.9], [0.7, 0.05]])},
        }
        mask = derive_global_masks(per_fam, sparsity=0.5)
        assert "mod" in mask.masks
        sp = mask.sparsity()
        assert abs(sp - 0.5) < 0.15


class TestMaskSetIO:
    def test_save_and_load_roundtrip(self) -> None:
        imp = _fake_importance()
        mask = derive_masks(imp, sparsity=0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            mask.save(tmpdir)
            loaded = MaskSet.load(tmpdir)

        for name in mask.masks:
            assert (loaded.masks[name] == mask.masks[name]).all()

    def test_per_module_sparsity(self) -> None:
        imp = _fake_importance()
        mask = derive_masks(imp, sparsity=0.3)
        pms = mask.per_module_sparsity()
        assert len(pms) == 2
        for sp in pms.values():
            assert 0.0 <= sp <= 1.0
