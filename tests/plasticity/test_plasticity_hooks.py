"""CI-safe tests for importance recording hooks."""

from __future__ import annotations

import torch
import torch.nn as nn

from plasticity.hooks import ImportanceRecorder


def _make_tiny_model() -> nn.Module:
    """Two-layer linear model for testing."""
    model = nn.Sequential(
        nn.Linear(8, 16, bias=False),
        nn.ReLU(),
        nn.Linear(16, 4, bias=False),
    )
    model.eval()
    return model


class TestImportanceRecorder:
    def test_attach_registers_hooks(self) -> None:
        model = _make_tiny_model()
        rec = ImportanceRecorder()
        rec.attach(model)
        assert rec._attached
        assert len(rec._hooks) == 2  # two nn.Linear layers
        rec.detach()

    def test_detach_removes_hooks(self) -> None:
        model = _make_tiny_model()
        rec = ImportanceRecorder()
        rec.attach(model)
        rec.detach()
        assert not rec._attached
        assert len(rec._hooks) == 0

    def test_double_attach_raises(self) -> None:
        model = _make_tiny_model()
        rec = ImportanceRecorder()
        rec.attach(model)
        try:
            import pytest

            with pytest.raises(RuntimeError, match="Already attached"):
                rec.attach(model)
        finally:
            rec.detach()

    def test_collect_shapes_match_weights(self) -> None:
        model = _make_tiny_model()
        rec = ImportanceRecorder()
        rec.attach(model)

        x = torch.randn(2, 8)
        model(x)

        scores = rec.collect()
        rec.detach()

        assert len(scores) == 2
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                assert name in scores
                assert scores[name].shape == module.weight.shape

    def test_scores_are_nonnegative(self) -> None:
        model = _make_tiny_model()
        rec = ImportanceRecorder()
        rec.attach(model)

        for _ in range(3):
            model(torch.randn(4, 8))

        scores = rec.collect()
        rec.detach()

        for tensor in scores.values():
            assert (tensor >= 0).all()

    def test_accumulation_increases_with_batches(self) -> None:
        model = _make_tiny_model()
        rec = ImportanceRecorder()
        rec.attach(model)

        model(torch.randn(2, 8))
        scores_1 = {k: v.clone() for k, v in rec.collect(normalize=False).items()}

        model(torch.randn(2, 8))
        scores_2 = rec.collect(normalize=False)

        rec.detach()

        for name in scores_1:
            assert (scores_2[name] >= scores_1[name]).all()

    def test_normalization_divides_by_count(self) -> None:
        model = _make_tiny_model()
        rec = ImportanceRecorder()
        rec.attach(model)

        x = torch.randn(2, 8)
        model(x)
        model(x)

        raw = rec.collect(normalize=False)
        normed = rec.collect(normalize=True)
        rec.detach()

        for name in raw:
            expected = raw[name] / 2
            torch.testing.assert_close(normed[name], expected)

    def test_reset_clears_accumulator(self) -> None:
        model = _make_tiny_model()
        rec = ImportanceRecorder()
        rec.attach(model)

        model(torch.randn(2, 8))
        rec.reset()

        scores = rec.collect()
        rec.detach()

        assert len(scores) == 0

    def test_3d_input_handled(self) -> None:
        """Simulates transformer-style (batch, seq_len, hidden) input."""
        model = _make_tiny_model()
        rec = ImportanceRecorder()
        rec.attach(model)

        x = torch.randn(2, 5, 8)  # batch=2, seq_len=5, features=8
        model(x)

        scores = rec.collect()
        rec.detach()

        assert len(scores) == 2
        for tensor in scores.values():
            assert (tensor >= 0).all()
