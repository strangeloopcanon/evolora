"""Identifier helpers."""

from __future__ import annotations

import secrets


def short_uid(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(4)}"


__all__ = ["short_uid"]
