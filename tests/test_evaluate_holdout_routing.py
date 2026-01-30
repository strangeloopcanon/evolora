from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _load_eval_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "evaluate_holdout.py"
    spec = importlib.util.spec_from_file_location("evaluate_holdout", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_infer_bucket_family_and_cell():
    m = _load_eval_module()

    task = {"family": "math", "depth": "short"}
    assert m._infer_bucket(task, mode="family") == "math"
    assert m._infer_bucket(task, mode="cell") == "math:short"

    task2 = {"capability": "recognition", "depth": "medium"}
    assert m._infer_bucket(task2, mode="family") == "recognition"
    assert m._infer_bucket(task2, mode="cell") == "recognition:medium"

    task3 = {"family": "regex.synthesis", "depth": "long"}
    assert m._infer_bucket(task3, mode="family") == "regex.synthesis"
    assert m._infer_bucket(task3, mode="cell") == "regex.synthesis:long"


def test_select_bucket_routing_from_candidate_metrics():
    m = _load_eval_module()

    candidate_metrics = {
        "org_a": {
            "selection_abs_max_lora_b": 1.0,
            "selection_metrics": {
                "accuracy": 0.5,
                "geometric_mean_accuracy": 0.4,
                "by_bucket": {
                    "math": {"correct": 3, "total": 4},
                    "json_repair": {"correct": 1, "total": 4},
                },
            },
        },
        "org_b": {
            "selection_abs_max_lora_b": 0.5,
            "selection_metrics": {
                "accuracy": 0.5,
                "geometric_mean_accuracy": 0.45,
                "by_bucket": {
                    "math": {"correct": 1, "total": 4},
                    "json_repair": {"correct": 4, "total": 4},
                },
            },
        },
    }

    routing, details = m._select_bucket_routing_from_candidate_metrics(
        candidate_metrics, ["math", "json_repair"]
    )
    assert routing["math"] == "org_a"
    assert routing["json_repair"] == "org_b"
    assert details["math"]["organelle_id"] == "org_a"
    assert details["json_repair"]["organelle_id"] == "org_b"


def test_select_global_best_from_candidate_metrics_prefers_gm():
    m = _load_eval_module()

    candidate_metrics = {
        "org_a": {
            "selection_abs_max_lora_b": 0.0,
            "selection_metrics": {
                "accuracy": 0.6,
                "geometric_mean_accuracy": 0.3,
                "case_accuracy": 0.0,
            },
        },
        "org_b": {
            "selection_abs_max_lora_b": 0.0,
            "selection_metrics": {
                "accuracy": 0.55,
                "geometric_mean_accuracy": 0.4,
                "case_accuracy": 0.0,
            },
        },
    }
    best_id, details = m._select_global_best_from_candidate_metrics(candidate_metrics)
    assert best_id == "org_b"
    assert details["selection_geometric_mean_accuracy"] == 0.4


def test_eval_result_merge_from():
    m = _load_eval_module()

    a = m.EvalResult(model_name="a", correct=1, total=2, task_results=[{"task_id": "t1"}])
    a.bucket_breakdown = {"math": {"correct": 1, "total": 2}}
    b = m.EvalResult(model_name="b", correct=2, total=3, task_results=[{"task_id": "t2"}])
    b.bucket_breakdown = {
        "math": {"correct": 2, "total": 3},
        "json_repair": {"correct": 0, "total": 1},
    }
    a.merge_from(b)
    assert a.correct == 3
    assert a.total == 5
    assert len(a.task_results) == 2
    assert a.bucket_breakdown["math"] == {"correct": 3, "total": 5}
    assert a.bucket_breakdown["json_repair"] == {"correct": 0, "total": 1}


def test_augment_grid_holdout_tasks_includes_clean_variant():
    m = _load_eval_module()

    tasks = [
        {"task_id": "t1", "prompt": "p", "target": 1, "family": "math.multi_step", "depth": "short"}
    ]
    augmented, augmentations = m._augment_grid_holdout_tasks(tasks, ["ws_prefix"])  # type: ignore[attr-defined]
    assert augmentations[0] == "clean"
    assert "ws_prefix" in augmentations
    assert len(augmented) == 2
    prompts = {t["augmentation"]: t["prompt"] for t in augmented}
    assert prompts["clean"] == "p"
    assert prompts["ws_prefix"].endswith("p")


def test_prompt_robustness_summary_tracks_brittleness():
    m = _load_eval_module()

    base_tasks = [
        {"task_id": "t1", "prompt": "p", "target": 1, "family": "math.multi_step", "depth": "short"}
    ]
    tasks, augmentations = m._augment_grid_holdout_tasks(base_tasks, ["clean", "ws_prefix"])  # type: ignore[attr-defined]

    brittle = m.EvalResult(model_name="brittle")
    brittle.task_results = [
        {"task_id": t["task_id"], "correct": t["augmentation"] == "clean"} for t in tasks
    ]
    robust = m.EvalResult(model_name="robust")
    robust.task_results = [{"task_id": t["task_id"], "correct": True} for t in tasks]

    summary = m._compute_prompt_robustness_summary(tasks, [brittle, robust], augmentations)  # type: ignore[attr-defined]
    assert summary is not None
    stats = summary["prompt_robustness"]
    assert stats["brittle"]["clean_accuracy"] == 1.0
    assert stats["brittle"]["all_variants_accuracy"] == 0.0
    assert stats["brittle"]["any_variant_accuracy"] == 1.0
    assert stats["brittle"]["brittleness_rate"] == 1.0
    assert stats["robust"]["all_variants_accuracy"] == 1.0


def test_sample_holdout_tasks_stratified_family():
    m = _load_eval_module()

    tasks = [
        {"task_id": "a1", "prompt": "p", "target": 1, "family": "a", "depth": "short"},
        {"task_id": "a2", "prompt": "p", "target": 1, "family": "a", "depth": "short"},
        {"task_id": "a3", "prompt": "p", "target": 1, "family": "a", "depth": "short"},
        {"task_id": "b1", "prompt": "p", "target": 1, "family": "b", "depth": "short"},
        {"task_id": "b2", "prompt": "p", "target": 1, "family": "b", "depth": "short"},
        {"task_id": "b3", "prompt": "p", "target": 1, "family": "b", "depth": "short"},
    ]
    sampled = m._sample_holdout_tasks(tasks, max_samples=4, sampling="stratified_family", seed=1)  # type: ignore[attr-defined]
    assert len(sampled) == 4
    families = [t["family"] for t in sampled]
    assert families.count("a") == 2
    assert families.count("b") == 2
