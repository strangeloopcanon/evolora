"""Backward-compatible alias for older Gemma naming."""

from __future__ import annotations

from symbiont_ecology.host.backbone import HFBackbone, HFOutputs

GemmaBackbone = HFBackbone
GemmaOutputs = HFOutputs

__all__ = ["GemmaBackbone", "GemmaOutputs", "HFBackbone", "HFOutputs"]
