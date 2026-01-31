from __future__ import annotations

import json
from pathlib import Path

from symbiont_ecology.evaluation.holdout_tasks import load_holdout_tasks_jsonl


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def test_load_holdout_tasks_assigns_stable_task_ids(tmp_path: Path) -> None:
    holdout_path = tmp_path / "holdout.jsonl"
    _write_jsonl(
        holdout_path,
        [
            {"prompt": "Add 1 and 1. Respond with the number only.", "target": 2, "family": "math"},
            {"prompt": "Add 2 and 2. Respond with the number only.", "target": 4, "family": "math"},
        ],
    )

    tasks = load_holdout_tasks_jsonl(holdout_path)
    assert [task.get("task_id") for task in tasks] == ["task_0", "task_1"]


def test_load_holdout_tasks_disambiguates_duplicate_task_ids(tmp_path: Path) -> None:
    holdout_path = tmp_path / "holdout.jsonl"
    _write_jsonl(
        holdout_path,
        [
            {"task_id": "dup", "prompt": "A", "target": 1, "family": "math"},
            {"task_id": "dup", "prompt": "B", "target": 2, "family": "math"},
            {"task_id": "dup", "prompt": "C", "target": 3, "family": "math"},
        ],
    )

    tasks = load_holdout_tasks_jsonl(holdout_path)
    ids = [str(task.get("task_id")) for task in tasks]

    assert ids[0] == "dup"
    assert ids[1] == "dup__2"
    assert ids[2] == "dup__3"
    assert len(ids) == len(set(ids))
