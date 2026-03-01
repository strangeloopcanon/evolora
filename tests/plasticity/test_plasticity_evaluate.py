"""CI-safe tests for masked evaluation (weight backup/restore)."""

from __future__ import annotations

import torch
import torch.nn as nn

from plasticity.evaluate import masked_weights
from plasticity.masks import MaskSet


def _make_model_and_mask() -> tuple[nn.Module, MaskSet]:
    """Create a tiny model and a mask that zeros the bottom half of weights."""
    model = nn.Sequential()
    layer = nn.Linear(4, 4, bias=False)
    nn.init.ones_(layer.weight)
    model.add_module("linear", layer)

    mask_tensor = torch.ones(4, 4, dtype=torch.bool)
    mask_tensor[2:, :] = False  # prune bottom two rows
    masks = {"linear": mask_tensor}
    return model, MaskSet(masks)


class TestMaskedWeights:
    def test_weights_are_zeroed_inside_context(self) -> None:
        model, mask = _make_model_and_mask()

        with masked_weights(model, mask):
            w = model.linear.weight.data  # type: ignore[union-attr]
            assert (w[2:, :] == 0).all(), "Bottom rows should be zeroed"
            assert (w[:2, :] != 0).all(), "Top rows should be untouched"

    def test_weights_are_restored_after_context(self) -> None:
        model, mask = _make_model_and_mask()
        original = model.linear.weight.data.clone()  # type: ignore[union-attr]

        with masked_weights(model, mask):
            pass

        restored = model.linear.weight.data  # type: ignore[union-attr]
        torch.testing.assert_close(restored, original)

    def test_weights_restored_on_exception(self) -> None:
        model, mask = _make_model_and_mask()
        original = model.linear.weight.data.clone()  # type: ignore[union-attr]

        try:
            with masked_weights(model, mask):
                raise ValueError("test error")
        except ValueError:
            pass

        restored = model.linear.weight.data  # type: ignore[union-attr]
        torch.testing.assert_close(restored, original)

    def test_empty_mask_is_noop(self) -> None:
        model, _ = _make_model_and_mask()
        original = model.linear.weight.data.clone()  # type: ignore[union-attr]

        empty = MaskSet({})
        with masked_weights(model, empty):
            current = model.linear.weight.data  # type: ignore[union-attr]
            torch.testing.assert_close(current, original)

    def test_forward_pass_uses_masked_weights(self) -> None:
        model, mask = _make_model_and_mask()
        x = torch.ones(1, 4)

        out_dense = model(x)

        with masked_weights(model, mask):
            out_masked = model(x)

        assert not torch.equal(out_dense, out_masked)
        out_after = model(x)
        torch.testing.assert_close(out_after, out_dense)
