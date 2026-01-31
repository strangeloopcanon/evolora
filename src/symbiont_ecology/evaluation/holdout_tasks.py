from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_unique_task_ids(tasks: list[dict[str, Any]]) -> None:
    """Ensure every task has a stable, unique task_id.

    Some holdout JSONL files omit explicit task IDs. That's fine for overall accuracy,
    but routed evaluation can split tasks into buckets; if each bucket re-enumerates
    tasks from 0, missing IDs can lead to duplicate task_0/task_1/... across buckets.
    That makes per-task comparison and debugging painful.
    """
    seen: set[str] = set()
    for idx, task in enumerate(tasks):
        raw = task.get("task_id")
        task_id = str(raw).strip() if raw is not None else ""
        if not task_id:
            task_id = f"task_{idx}"
            task["task_id"] = task_id
        if task_id in seen:
            suffix = 2
            candidate = f"{task_id}__{suffix}"
            while candidate in seen:
                suffix += 1
                candidate = f"{task_id}__{suffix}"
            task_id = candidate
            task["task_id"] = task_id
        seen.add(task_id)


def load_holdout_tasks_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load holdout tasks from a JSONL file and guarantee task_id uniqueness."""
    tasks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    ensure_unique_task_ids(tasks)
    return tasks
