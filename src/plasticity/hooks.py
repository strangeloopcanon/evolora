"""Forward hooks for per-weight importance recording (Wanda metric).

Registers removable hooks on ``nn.Linear`` layers to accumulate
``importance[i,j] = |W[i,j]| * ||X[:,j]||_2`` across calibration samples.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class ImportanceRecorder:
    """Accumulates Wanda importance scores per ``nn.Linear`` in a model.

    Usage::

        recorder = ImportanceRecorder()
        recorder.attach(model)              # register hooks
        for batch in calibration_data:
            model(batch)                     # hooks fire automatically
        scores = recorder.collect()          # {module_name: importance_tensor}
        recorder.detach()                    # remove hooks
    """

    def __init__(self) -> None:
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._accum: Dict[str, torch.Tensor] = {}
        self._cpu_accum: Dict[str, torch.Tensor] = {}
        self._counts: Dict[str, int] = {}
        self._attached = False

    def attach(self, model: nn.Module, *, prefix: str = "") -> None:
        """Register forward hooks on every ``nn.Linear`` in *model*."""
        if self._attached:
            raise RuntimeError("Already attached; call detach() first.")
        self._accum.clear()
        self._cpu_accum.clear()
        self._counts.clear()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"{prefix}.{name}" if prefix else name
                hook = module.register_forward_hook(self._make_hook(full_name, module))
                self._hooks.append(hook)
        self._attached = True

    def detach(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._attached = False

    def collect(self, *, normalize: bool = True) -> Dict[str, torch.Tensor]:
        """Return accumulated importance scores (merges device + CPU buffers).

        If *normalize* is True, divides by the number of forward passes
        to get the average importance per sample.
        """
        result: Dict[str, torch.Tensor] = {}
        all_names = set(self._accum) | set(self._cpu_accum)
        for name in all_names:
            total = self._cpu_accum.get(name, None)
            device_part = self._accum.get(name, None)
            if device_part is not None:
                device_cpu = device_part.cpu()
                total = (total + device_cpu) if total is not None else device_cpu
            if total is None:
                continue
            if normalize and self._counts.get(name, 0) > 0:
                result[name] = total / self._counts[name]
            else:
                result[name] = total.clone()
        return result

    def flush_to_cpu(self) -> None:
        """Add device accumulators into CPU buffer and zero device tensors."""
        for name in list(self._accum.keys()):
            chunk = self._accum[name].cpu()
            if name not in self._cpu_accum:
                self._cpu_accum[name] = chunk
            else:
                self._cpu_accum[name] += chunk
            self._accum[name].zero_()

    def reset(self) -> None:
        """Clear accumulated scores without removing hooks."""
        self._accum.clear()
        self._cpu_accum.clear()
        self._counts.clear()

    @property
    def module_names(self) -> List[str]:
        return list(self._accum.keys())

    def _make_hook(self, name: str, module: nn.Linear):
        """Build a hook closure that accumulates the Wanda importance metric."""

        def _hook(
            mod: nn.Module,
            inputs: tuple,
            output: torch.Tensor,
        ) -> None:
            x = inputs[0]  # (batch, seq_len, in_features) or (batch, in_features)
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])  # flatten batch and seq dims

            col_norms = x.float().norm(dim=0)  # (in_features,)

            w = mod.weight.detach().float()  # (out_features, in_features)
            importance = w.abs() * col_norms.unsqueeze(0)  # (out, in)

            if name not in self._accum:
                self._accum[name] = torch.zeros_like(importance)
                self._counts[name] = 0
            self._accum[name] += importance
            self._counts[name] += 1

        return _hook


def record_importance(
    model: nn.Module,
    dataloader,
    *,
    device: Optional[torch.device] = None,
    prefix: str = "",
) -> Dict[str, torch.Tensor]:
    """Convenience wrapper: attach hooks, run all batches, collect, detach.

    *dataloader* should yield dicts suitable for ``model(**batch)``
    (e.g. with ``input_ids`` and ``attention_mask``).
    """
    recorder = ImportanceRecorder()
    recorder.attach(model, prefix=prefix)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if device is not None:
                batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            model(**batch)
    scores = recorder.collect(normalize=True)
    recorder.detach()
    return scores
