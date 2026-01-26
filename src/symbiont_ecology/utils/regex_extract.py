"""Helpers for extracting regex patterns from model responses.

These utilities are shared by the grid environment and evaluation tooling so that
success metrics are not overly sensitive to formatting (Markdown, prefixes, etc.).
"""

from __future__ import annotations

import re
from typing import Sequence

_CODE_BLOCK_RE = re.compile(r"```(?:regex|re|regexp)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_PREFIX_ONLY_RE = re.compile(r"^(?:regex|regexp|re|pattern)\s*:?\s*$", re.IGNORECASE)
_PREFIX_VALUE_RE = re.compile(r"^(?:regex|pattern)\s*:\s*(.+)$", re.IGNORECASE)

_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")
_LIKELY_REGEX_CHARS = set("^$[](){}|?+*\\")


def _looks_like_regex(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if lower.startswith(("here", "the regex", "this regex", "explanation")):
        return False
    # Heuristic: regex patterns typically contain at least one metacharacter/escape.
    if any(ch in _LIKELY_REGEX_CHARS for ch in stripped):
        return True
    # Accept simple literal patterns (common in small tasks), but avoid prose.
    if any(ch.isspace() for ch in stripped):
        return False
    if len(stripped) > 256:
        return False
    return True


def _strip_wrappers(text: str) -> str:
    value = text.strip()
    if not value:
        return ""

    if value.startswith("```") and value.endswith("```"):
        inner = value.strip("`").strip()
        parts = [line.strip() for line in inner.splitlines() if line.strip()]
        if parts and parts[0].lower() in {"regex", "regexp", "re"}:
            parts = parts[1:]
        value = parts[0] if parts else ""

    value = value.strip()
    if value.lower().startswith(('r"', "r'")) and len(value) >= 3:
        quote = value[1]
        if value.endswith(quote):
            value = value[2:-1].strip()

    # Common delimiters from other ecosystems: /.../
    if len(value) >= 2 and value[0] == "/" and value[-1] == "/":
        value = value[1:-1].strip()

    # Trim common quoting/backtick wrappers.
    value = value.strip().strip("`").strip().strip("'\"").strip()

    # Some models add a trailing period or fenced-language marker.
    value = value.rstrip(".").strip()
    return value


def extract_regex_candidates(response: str) -> list[str]:
    """Return ordered, de-duplicated regex candidate strings from `response`."""

    candidates: list[str] = []

    def add(candidate: str) -> None:
        cleaned = _strip_wrappers(candidate)
        if not cleaned:
            return
        if cleaned in candidates:
            return
        candidates.append(cleaned)

    text = response or ""

    for match in _CODE_BLOCK_RE.finditer(text):
        block = match.group(1).strip()
        if not block:
            continue
        add(block)
        for line in block.splitlines():
            line = line.strip()
            if line:
                add(line)
                break

    for match in _INLINE_CODE_RE.finditer(text):
        add(match.group(1))

    for raw in text.splitlines():
        line = _BULLET_PREFIX_RE.sub("", raw.strip())
        if not line:
            continue
        if line.startswith("```"):
            continue
        if _PREFIX_ONLY_RE.match(line):
            continue
        prefix_value = _PREFIX_VALUE_RE.match(line)
        if prefix_value:
            add(prefix_value.group(1))
            continue
        lower = line.lower()
        if ":" in line and any(key in lower for key in ("regex", "pattern", "answer")):
            before, after = line.split(":", 1)
            if after.strip():
                add(after)
        add(line)

    return candidates


def _normalize_test_cases(test_cases: Sequence[object]) -> list[tuple[str, bool]]:
    cases: list[tuple[str, bool]] = []
    for tc in test_cases:
        if isinstance(tc, dict):
            raw = tc.get("string", "")
            should = bool(tc.get("should_match", False))
        else:
            raw = getattr(tc, "string", "")
            should = bool(getattr(tc, "should_match", False))
        if not isinstance(raw, str):
            continue
        cases.append((raw, should))
    return cases


def pick_best_regex_candidate(
    response: str,
    test_cases: Sequence[object] | None = None,
) -> tuple[str | None, dict[str, object]]:
    """Pick the candidate regex that best satisfies the provided test cases.

    Returns (pattern, details). If no candidate can be compiled, pattern is None.
    """

    candidates = extract_regex_candidates(response)
    if not candidates:
        return None, {"candidates": [], "picked": None, "passed": 0, "total": 0}

    if not test_cases:
        picked = next((cand for cand in candidates if _looks_like_regex(cand)), candidates[0])
        return picked, {"candidates": candidates[:5], "picked": picked, "passed": 0, "total": 0}

    cases = _normalize_test_cases(test_cases)
    total = len(cases)
    if total == 0:
        picked = next((cand for cand in candidates if _looks_like_regex(cand)), candidates[0])
        return picked, {"candidates": candidates[:5], "picked": picked, "passed": 0, "total": 0}

    best: str | None = None
    best_passed = -1
    best_len = 10**9
    best_error: str | None = None

    for candidate in candidates:
        try:
            compiled = re.compile(candidate)
        except re.error as exc:
            best_error = best_error or str(exc)
            continue
        passed = 0
        for s, should_match in cases:
            does_match = bool(compiled.fullmatch(s))
            if does_match == should_match:
                passed += 1
        cand_len = len(candidate)
        if passed > best_passed or (passed == best_passed and cand_len < best_len):
            best = candidate
            best_passed = passed
            best_len = cand_len

    details: dict[str, object] = {
        "candidates": candidates[:5],
        "picked": best,
        "passed": max(0, best_passed),
        "total": total,
    }
    if best is None and best_error:
        details["compile_error"] = best_error
    return best, details


__all__ = ["extract_regex_candidates", "pick_best_regex_candidate"]
