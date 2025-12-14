"""Telemetry serialization helpers."""

from __future__ import annotations

import math


def sanitize_telemetry(value: object) -> object:
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return 0.0
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {key: sanitize_telemetry(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_telemetry(item) for item in value]
    return str(value)


__all__ = ["sanitize_telemetry"]
