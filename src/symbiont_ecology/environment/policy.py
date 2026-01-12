"""Policy parsing utilities."""

from __future__ import annotations


def parse_policy_json(text: str, allowed: list[str]) -> dict[str, object]:
    """Extract a tiny policy payload.

    Strategy (best-effort, no heavy deps):
    1) Prefer fenced ```json blocks; else any fenced block; else the outermost {...} span.
    2) Try strict json.loads; on failure, apply light repairs (strip fences, drop trailing commas,
       normalize Python literals, convert single quotes).
    3) If JSON parsing fails, fall back to `key=value` pairs and coerce numbers/bools.
    4) Return only keys in `allowed`.
    """
    import json
    import re

    allowed_set = set(allowed)

    def find_fenced(block: str) -> list[str]:
        candidates: list[str] = []
        # ```json ... ``` preferred
        for match in re.finditer(r"```json\s*([\s\S]*?)\s*```", block, re.IGNORECASE):
            candidates.append(match.group(1))
        # any ``` ... ```
        for match in re.finditer(r"```\s*([\s\S]*?)\s*```", block):
            candidates.append(match.group(1))
        return candidates

    def outer_object(block: str) -> str | None:
        start = block.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(block)):
            ch = block[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return block[start : i + 1]
        return None

    def repair(candidate: str) -> str:
        # strip code fences accidentally included
        candidate = candidate.strip().strip("`")
        # normalize common Python literals to JSON
        candidate = re.sub(r"\bTrue\b", "true", candidate)
        candidate = re.sub(r"\bFalse\b", "false", candidate)
        candidate = re.sub(r"\bNone\b", "null", candidate)
        # remove trailing commas before closing } or ]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        # single-quoted keys -> double quotes
        candidate = re.sub(r"([{,]\s*)'([^'\s]+)'\s*:", r'\1"\2":', candidate)
        # single-quoted string values -> double quotes (conservative, no nested quotes)
        candidate = re.sub(r":\s*'([^']*)'\s*([,}])", r': "\1"\2', candidate)
        return candidate

    def try_load(candidate: str) -> dict[str, object] | None:
        try:
            data = json.loads(candidate)
        except Exception:
            try:
                data = json.loads(repair(candidate))
            except Exception:
                return None
        if not isinstance(data, dict):
            return None
        return {key: value for key, value in data.items() if key in allowed_set}

    def coerce_value(token: str) -> object:
        raw = token.strip()
        percent = raw.endswith("%")
        if percent:
            raw = raw[:-1].strip()
        lower = raw.lower()
        if lower in {"true", "false"}:
            return lower == "true"
        if lower == "null":
            return None
        if (raw.startswith("'") and raw.endswith("'")) or (
            raw.startswith('"') and raw.endswith('"')
        ):
            raw = raw[1:-1]
        try:
            if "." in raw or "e" in lower:
                value = float(raw)
            else:
                value = float(int(raw))
            if percent:
                value = value / 100.0
            return value
        except Exception:
            return raw

    def parse_kv_pairs(block: str) -> dict[str, object]:
        kvs: dict[str, object] = {}
        equals_matches = re.findall(r"([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*([^\s;,]+)", block)
        colon_matches = re.findall(r"([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*([^\s;,{}]+)", block)
        for key, value in equals_matches + colon_matches:
            if allowed_set and key not in allowed_set:
                continue
            kvs.setdefault(key, coerce_value(value))
        return kvs

    candidates = find_fenced(text)
    outer = outer_object(text)
    if outer:
        candidates.append(outer)
    candidates.append(text)
    for candidate in candidates:
        parsed = try_load(candidate)
        if parsed:
            return parsed
    kvs = parse_kv_pairs(text)
    if kvs:
        return kvs
    return {}


__all__ = ["parse_policy_json"]
