"""Persistent telemetry sinks."""

from __future__ import annotations

import json
from pathlib import Path

from symbiont_ecology.metrics.telemetry import AssimilationEvent, EpisodeLog


class TelemetrySink:
    """Append-only JSONL writer for telemetry events."""

    def __init__(self, root: Path, episodes_file: str, assimilation_file: str) -> None:
        self.root = root
        self.episodes_path = root / episodes_file
        self.assimilation_path = root / assimilation_file
        self.root.mkdir(parents=True, exist_ok=True)

    def log_episode(self, log: EpisodeLog) -> None:
        self._append(self.episodes_path, {"type": "episode", **log.model_dump(mode="json")})

    def log_assimilation(self, event: AssimilationEvent, decision: bool) -> None:
        payload = {
            "type": "assimilation",
            "decision": decision,
            **event.model_dump(mode="json"),
        }
        self._append(self.assimilation_path, payload)

    def _append(self, path: Path, payload: dict[str, object]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["TelemetrySink"]
