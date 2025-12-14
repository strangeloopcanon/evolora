"""Colony-related utilities with stable typing."""

from __future__ import annotations


def _coerce_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Unsupported float-like value: {type(value)!r}")


def _coerce_int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Unsupported int-like value: {type(value)!r}")


def colony_c2c_debit(meta: dict[str, object], amount: float, counter_key: str) -> bool:
    """Attempt to pay a C2C cost from colony pot & bandwidth.

    Returns True when the debit succeeds and updates pot/bandwidth/counter.
    """
    try:
        bandwidth = _coerce_float(meta.get("c2c_bandwidth_left", meta.get("bandwidth_left", 0.0)))
        counter = _coerce_int(meta.get(counter_key, 0))
        pot = _coerce_float(meta.get("pot", 0.0))
    except Exception:
        return False
    if bandwidth < amount or counter <= 0 or pot < amount:
        return False
    meta["pot"] = pot - amount
    meta["c2c_bandwidth_left"] = max(0.0, bandwidth - amount)
    meta[counter_key] = counter - 1
    return True


__all__ = ["colony_c2c_debit"]
