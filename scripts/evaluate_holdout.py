#!/usr/bin/env python3
"""Evaluate models on holdout tasks for generalization comparison.

Usage:
    # Compare base model vs SFT adapter
    python scripts/evaluate_holdout.py \
        --holdout config/evaluation/regex_generalization.jsonl \
        --sft-adapter artifacts_sft_e2e_test/peft_adapter

    # Compare base model vs evolution checkpoint
    python scripts/evaluate_holdout.py \
        --holdout config/evaluation/regex_generalization.jsonl \
        --evo-checkpoint artifacts_evo_e2e_test/checkpoint.pt

    # Compare all three
    python scripts/evaluate_holdout.py \
        --holdout config/evaluation/regex_generalization.jsonl \
        --sft-adapter artifacts_sft_e2e_test/peft_adapter \
        --evo-checkpoint artifacts_evo_e2e_test/checkpoint.pt

    # Specify model and sample size
    python scripts/evaluate_holdout.py \
        --holdout config/evaluation/regex_generalization.jsonl \
        --model Qwen/Qwen2.5-0.5B \
        --max-samples 20 \
        --verbose
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from symbiont_ecology.utils.checkpoint_io import load_checkpoint

_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class EvalResult:
    """Results from evaluating a model on holdout tasks."""

    model_name: str
    correct: int = 0
    total: int = 0
    task_results: list[dict[str, Any]] = field(default_factory=list)
    bucket_breakdown: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        bucket_stats: dict[str, dict[str, float]] = {}
        bucket_accs: list[float] = []
        for bucket, stats in self.bucket_breakdown.items():
            total = int(stats.get("total", 0) or 0)
            correct = int(stats.get("correct", 0) or 0)
            acc = float(correct / total) if total else 0.0
            bucket_stats[bucket] = {"correct": correct, "total": total, "accuracy": acc}
            bucket_accs.append(acc)
        macro_acc = float(sum(bucket_accs) / len(bucket_accs)) if bucket_accs else None
        worst_acc = float(min(bucket_accs)) if bucket_accs else None
        gm_acc = None
        if bucket_accs:
            import math

            eps = 1e-6
            gm_acc = float(
                math.exp(sum(math.log(max(a, eps)) for a in bucket_accs) / len(bucket_accs))
            )
        return {
            "model": self.model_name,
            "correct": self.correct,
            "total": self.total,
            "accuracy": self.accuracy,
            "accuracy_pct": f"{100 * self.accuracy:.1f}%",
            "bucket_breakdown": bucket_stats,
            "bucket_macro_accuracy": macro_acc,
            "bucket_worst_accuracy": worst_acc,
            "bucket_geometric_mean_accuracy": gm_acc,
        }

    def merge_from(self, other: "EvalResult") -> None:
        self.correct += int(other.correct)
        self.total += int(other.total)
        self.task_results.extend(list(other.task_results))
        for bucket, stats in (other.bucket_breakdown or {}).items():
            dst = self.bucket_breakdown.setdefault(str(bucket), {"correct": 0, "total": 0})
            dst["correct"] = int(dst.get("correct", 0) or 0) + int(stats.get("correct", 0) or 0)
            dst["total"] = int(dst.get("total", 0) or 0) + int(stats.get("total", 0) or 0)


def load_holdout_tasks(path: Path) -> list[dict[str, Any]]:
    """Load holdout tasks from JSONL file."""
    from symbiont_ecology.evaluation.holdout_tasks import load_holdout_tasks_jsonl

    return load_holdout_tasks_jsonl(path)


_DEFAULT_GRID_PROMPT_AUGMENTATIONS: tuple[str, ...] = (
    "clean",
    "ws_prefix",
    "ws_suffix",
    "markdown_sections",
    "roleplay",
    "distractor_preamble",
)


def _parse_comma_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _augment_grid_prompt(prompt: str, augmentation: str) -> str:
    augmentation = str(augmentation).strip().lower()
    if augmentation in {"clean", "none"}:
        return prompt
    if augmentation == "ws_prefix":
        return "\n\n" + prompt
    if augmentation == "ws_suffix":
        return prompt + "\n\n"
    if augmentation == "markdown_sections":
        return f"### Task\n{prompt}\n\n### Answer\n"
    if augmentation == "roleplay":
        return f"User: {prompt}\nAssistant:"
    if augmentation == "distractor_preamble":
        return f"(Ignore this note; it's unrelated.)\n\n{prompt}"
    raise ValueError(f"Unknown grid prompt augmentation: {augmentation}")


def _augment_grid_holdout_tasks(
    tasks: list[dict[str, Any]], augmentations: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Expand each GridTask-like holdout item into multiple prompt variants.

    Returns (augmented_tasks, augmentations_used).
    """
    if not tasks:
        return [], []
    if not augmentations:
        return tasks, []

    normalized: list[str] = []
    for aug in augmentations:
        aug_norm = str(aug).strip().lower()
        if not aug_norm:
            continue
        if aug_norm == "default":
            normalized.extend(_DEFAULT_GRID_PROMPT_AUGMENTATIONS)
        else:
            normalized.append(aug_norm)
    if not normalized:
        return tasks, []
    if "clean" not in normalized:
        normalized.insert(0, "clean")
    seen: set[str] = set()
    augmentations_used: list[str] = []
    for aug in normalized:
        if aug in seen:
            continue
        seen.add(aug)
        augmentations_used.append(aug)

    augmented: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        prompt = str(task.get("prompt", "") or "")
        family = task.get("family")
        target = task.get("target")
        if family is None or target is None:
            raise ValueError(
                "Prompt augmentations are only supported for GridTask-style holdouts "
                "(must include 'family' and 'target')."
            )
        base_task_id = str(task.get("task_id", "") or f"task_{idx}")
        for aug in augmentations_used:
            variant = dict(task)
            variant["base_task_id"] = base_task_id
            variant["augmentation"] = aug
            variant["task_id"] = f"{base_task_id}::{aug}"
            variant["prompt"] = _augment_grid_prompt(prompt, aug)
            augmented.append(variant)
    return augmented, augmentations_used


def _compute_prompt_robustness_summary(
    tasks: list[dict[str, Any]], results: list[EvalResult], augmentations: list[str]
) -> dict[str, Any] | None:
    if not tasks or not results or not augmentations:
        return None
    if len(augmentations) <= 1:
        return None

    base_ids: list[str] = []
    base_id_to_bucket: dict[str, str] = {}
    for task in tasks:
        base_id = task.get("base_task_id")
        aug = task.get("augmentation")
        if not base_id or not aug:
            continue
        base_id_str = str(base_id)
        if base_id_str not in base_id_to_bucket:
            base_id_to_bucket[base_id_str] = str(
                task.get("family") or task.get("capability") or "unknown"
            )
        base_ids.append(base_id_str)

    base_ids = sorted(set(base_ids))
    if not base_ids:
        return None

    task_id_to_meta: dict[str, tuple[str, str]] = {}
    for task in tasks:
        task_id = task.get("task_id")
        base_id = task.get("base_task_id")
        aug = task.get("augmentation")
        if not task_id or not base_id or not aug:
            continue
        task_id_to_meta[str(task_id)] = (str(base_id), str(aug))

    per_model: dict[str, Any] = {}
    for result in results:
        by_base: dict[str, dict[str, bool]] = {}
        for tr in result.task_results:
            task_id = str(tr.get("task_id", "") or "")
            meta = task_id_to_meta.get(task_id)
            if not meta:
                continue
            base_id, aug = meta
            by_base.setdefault(base_id, {})[aug] = bool(tr.get("correct"))

        def _acc(
            ids: list[str], predicate, *, _by_base: dict[str, dict[str, bool]] = by_base
        ) -> float:
            if not ids:
                return 0.0
            return float(
                sum(1 for base_id in ids if predicate(_by_base.get(base_id, {}))) / len(ids)
            )

        clean_acc = _acc(base_ids, lambda row: bool(row.get("clean")))
        all_variants_acc = _acc(
            base_ids, lambda row: all(bool(row.get(aug)) for aug in augmentations)
        )
        any_variant_acc = _acc(
            base_ids, lambda row: any(bool(row.get(aug)) for aug in augmentations)
        )
        brittle_count = sum(
            1
            for base_id in base_ids
            if bool(by_base.get(base_id, {}).get("clean"))
            and not all(bool(by_base.get(base_id, {}).get(aug)) for aug in augmentations)
        )
        clean_correct_count = sum(
            1 for base_id in base_ids if bool(by_base.get(base_id, {}).get("clean"))
        )
        brittleness_rate = (
            float(brittle_count / clean_correct_count) if clean_correct_count else None
        )

        per_aug_acc: dict[str, float] = {}
        for aug in augmentations:
            per_aug_acc[aug] = _acc(base_ids, lambda row, aug=aug: bool(row.get(aug)))

        family_rows: dict[str, list[str]] = {}
        for base_id in base_ids:
            family_rows.setdefault(base_id_to_bucket.get(base_id, "unknown"), []).append(base_id)
        family_all: dict[str, float] = {}
        for bucket, ids in sorted(family_rows.items()):
            family_all[bucket] = _acc(
                ids, lambda row: all(bool(row.get(aug)) for aug in augmentations)
            )

        per_model[result.model_name] = {
            "base_task_count": len(base_ids),
            "augmentation_count": len(augmentations),
            "augmentations": list(augmentations),
            "clean_accuracy": clean_acc,
            "all_variants_accuracy": all_variants_acc,
            "any_variant_accuracy": any_variant_acc,
            "brittleness_rate": brittleness_rate,
            "per_augmentation_accuracy": per_aug_acc,
            "all_variants_accuracy_by_bucket": family_all,
        }

    return {"prompt_robustness": per_model}


def _sample_holdout_tasks(
    tasks: list[dict[str, Any]],
    *,
    max_samples: int | None,
    sampling: str,
    seed: int,
) -> list[dict[str, Any]]:
    if max_samples is None or max_samples <= 0 or len(tasks) <= max_samples:
        return tasks

    sampling = str(sampling or "head").strip().lower()
    rng = random.Random(int(seed))

    if sampling in {"head", "first"}:
        return tasks[: int(max_samples)]
    if sampling == "random":
        return rng.sample(tasks, int(max_samples))

    if sampling in {"stratified_family", "stratified_cell"}:
        mode = "family" if sampling == "stratified_family" else "cell"
        by_bucket: dict[str, list[dict[str, Any]]] = {}
        for task in tasks:
            by_bucket.setdefault(_infer_bucket(task, mode=mode), []).append(task)
        buckets = sorted(by_bucket.keys())
        if not buckets:
            return tasks[: int(max_samples)]

        for bucket in buckets:
            rng.shuffle(by_bucket[bucket])

        per_bucket = int(max_samples) // len(buckets)
        remainder = int(max_samples) % len(buckets)
        selected: list[dict[str, Any]] = []
        leftover: list[dict[str, Any]] = []
        for idx, bucket in enumerate(buckets):
            take = per_bucket + (1 if idx < remainder else 0)
            selected.extend(by_bucket[bucket][:take])
            leftover.extend(by_bucket[bucket][take:])

        if len(selected) < int(max_samples) and leftover:
            rng.shuffle(leftover)
            selected.extend(leftover[: int(max_samples) - len(selected)])
        return selected[: int(max_samples)]

    raise ValueError(f"Unknown holdout sampling mode: {sampling}")


def check_answer(response: str, task: dict[str, Any]) -> tuple[bool, str]:
    """Check if the model response is correct for a task.

    Returns (is_correct, reason).
    """
    if "capability" in task:
        try:
            from symbiont_ecology.evaluation.regex_generalization import (
                RegexGeneralizationEvaluator,
                RegexTask,
            )

            regex_task = RegexTask.from_dict(task)
            evaluator = RegexGeneralizationEvaluator([regex_task])
            result = evaluator.evaluate_single(regex_task, response)
            return bool(result.success), f"capability={regex_task.capability.value}"
        except Exception:
            # Fall back to the lightweight evaluator below.
            pass

    # GridEnvironment-style tasks: {"prompt": ..., "target": ..., "family": ..., "depth": ...}
    if "family" in task and "target" in task:
        try:
            from symbiont_ecology.environment.grid import GridTask

            family = str(task.get("family", "") or "")
            depth = str(task.get("depth", "") or "short")
            prompt = str(task.get("prompt", "") or "")
            grid_task = GridTask(
                task_id=str(task.get("task_id", "") or "holdout"),
                cell=(family, depth),
                prompt=prompt,
                price=0.0,
                target=task.get("target"),
                family=family,
                depth=depth,
                difficulty=0.0,
            )
            ok, _reward = grid_task.evaluate(response)
            return bool(ok), f"family={family}"
        except Exception:
            pass

    expected = task.get("expected_answer")
    test_cases = task.get("test_cases", []) or []
    if not test_cases:
        target = task.get("target", {}) or {}
        if isinstance(target, dict):
            test_cases = target.get("test_strings", []) or []
    required_keywords = task.get("metadata", {}).get("required_keywords", [])

    # For recognition tasks with expected yes/no answer
    if expected:
        response_lower = response.lower().strip()
        expected_lower = expected.lower().strip()

        # Check for yes/no answers
        if expected_lower in ("yes", "no"):
            # Look for yes/no at the start of the response
            if response_lower.startswith("yes"):
                is_correct = expected_lower == "yes"
            elif response_lower.startswith("no"):
                is_correct = expected_lower == "no"
            else:
                # Check if the answer appears anywhere
                is_correct = expected_lower in response_lower
            return is_correct, f"expected '{expected}'"

        # For other expected answers, check containment
        is_correct = expected_lower in response_lower
        return is_correct, f"expected '{expected}'"

    # For explanation tasks with required keywords
    if required_keywords:
        response_lower = response.lower()
        found = sum(1 for kw in required_keywords if kw.lower() in response_lower)
        threshold = (len(required_keywords) * 7 + 9) // 10  # ceil(0.7 * n)
        is_correct = found >= threshold
        return is_correct, f"keywords: {found}/{len(required_keywords)}"

    # For synthesis tasks with test cases, try to validate the regex
    if test_cases:
        # Extract the regex pattern from the response using shared heuristics
        from symbiont_ecology.utils.regex_extract import pick_best_regex_candidate

        pattern, _pick_details = pick_best_regex_candidate(response, test_cases=test_cases)
        if not pattern:
            return False, "no pattern extracted"

        try:
            compiled = re.compile(pattern)
            passed = 0
            for tc in test_cases:
                test_str = tc.get("string", "")
                should_match = tc.get("should_match", True)
                matches = bool(compiled.fullmatch(test_str))
                if matches == should_match:
                    passed += 1

            # Strict: require all test cases to pass for correctness.
            is_correct = passed == len(test_cases)
            return is_correct, f"test cases: {passed}/{len(test_cases)}"
        except re.error as e:
            return False, f"invalid regex: {e}"

    # Default: check if response is non-empty and looks reasonable
    is_correct = len(response.strip()) > 0 and not response.strip().startswith("I don't")
    return is_correct, "non-empty response"


def evaluate_model(
    model,
    tokenizer,
    tasks: list[dict[str, Any]],
    model_name: str,
    max_samples: int | None = None,
    verbose: bool = False,
) -> EvalResult:
    """Evaluate a model on holdout tasks."""
    result = EvalResult(model_name=model_name)

    eval_tasks = tasks[:max_samples] if max_samples else tasks

    for i, task in enumerate(eval_tasks):
        prompt = task["prompt"]
        task_id = task.get("task_id", f"task_{i}")
        bucket = str(task.get("family") or task.get("capability") or "unknown")

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        # Check correctness
        is_correct, reason = check_answer(response, task)

        if is_correct:
            result.correct += 1
        result.total += 1
        bucket_stats = result.bucket_breakdown.setdefault(bucket, {"correct": 0, "total": 0})
        bucket_stats["total"] += 1
        if is_correct:
            bucket_stats["correct"] += 1

        task_result = {
            "task_id": task_id,
            "bucket": bucket,
            "correct": is_correct,
            "reason": reason,
            "response_preview": response[:100],
        }
        if "base_task_id" in task:
            task_result["base_task_id"] = str(task.get("base_task_id") or "")
        if "augmentation" in task:
            task_result["augmentation"] = str(task.get("augmentation") or "")
        result.task_results.append(task_result)

        if verbose:
            status = "PASS" if is_correct else "FAIL"
            print(f"  [{status}] {task_id}: {reason}")
            if not is_correct:
                print(f"       Response: {response[:80]}...")

    return result


def load_sft_model(base_model, adapter_path: Path):
    """Load SFT adapter onto base model."""
    from peft import PeftModel

    return PeftModel.from_pretrained(base_model, adapter_path)


def _load_selection_tasks(
    path: Path,
    *,
    max_samples: int | None = None,
    seed: int = 9403,
    family: str | None = "any",
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Selection tasks file not found: {path}")
    tasks: list[dict[str, Any]] = []
    family_filter = family
    if isinstance(family_filter, str) and family_filter.lower() in {"any", "all", "*"}:
        family_filter = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if family_filter is not None:
                if str(obj.get("family", "")).lower() != str(family_filter).lower():
                    continue
            if "prompt" not in obj:
                continue
            has_test_cases = bool(obj.get("test_cases"))
            has_expected = bool(obj.get("expected_answer"))
            has_keywords = bool(obj.get("metadata", {}).get("required_keywords", []))
            has_target = bool(obj.get("target"))
            has_capability = bool(obj.get("capability"))
            if not (has_target or has_test_cases or has_expected or has_keywords or has_capability):
                continue
            tasks.append(obj)
    if not tasks:
        return []
    if max_samples is None or max_samples <= 0 or len(tasks) <= max_samples:
        return tasks
    rng = random.Random(seed)
    return rng.sample(tasks, int(max_samples))


def _reset_lora_weights(peft_model) -> None:
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                param.zero_()


def _apply_adapter_state_to_peft_model(peft_model, adapter_state: dict[str, torch.Tensor]) -> int:
    model_state = peft_model.state_dict()
    updates: dict[str, torch.Tensor] = {}
    for ckpt_key, tensor in adapter_state.items():
        peft_key = ckpt_key
        if ".lora_A" in peft_key and ".weight" not in peft_key:
            peft_key = peft_key.replace(".lora_A", ".lora_A.default.weight")
        if ".lora_B" in peft_key and ".weight" not in peft_key:
            peft_key = peft_key.replace(".lora_B", ".lora_B.default.weight")
        if peft_key not in model_state:
            continue
        expected = model_state[peft_key]
        candidate = tensor
        if expected.shape != candidate.shape:
            # Allow rank-mismatched organelles by padding smaller LoRA ranks into a larger PEFT model.
            # This enables selection across a checkpoint where morphogenesis changed LoRA rank.
            if (
                candidate.ndim == 2
                and expected.ndim == 2
                and "lora_A" in peft_key
                and candidate.shape[1] == expected.shape[1]
                and candidate.shape[0] < expected.shape[0]
            ):
                padded = torch.zeros(expected.shape, dtype=candidate.dtype)
                padded[: candidate.shape[0], :] = candidate
                candidate = padded
            elif (
                candidate.ndim == 2
                and expected.ndim == 2
                and "lora_B" in peft_key
                and candidate.shape[0] == expected.shape[0]
                and candidate.shape[1] < expected.shape[1]
            ):
                padded = torch.zeros(expected.shape, dtype=candidate.dtype)
                padded[:, : candidate.shape[1]] = candidate
                candidate = padded
            else:
                continue
        updates[peft_key] = candidate.to(dtype=expected.dtype)
    if not updates:
        return 0
    _reset_lora_weights(peft_model)
    peft_model.load_state_dict(updates, strict=False)
    return len(updates)


def _infer_capability(task: dict[str, Any]) -> str:
    capability = task.get("capability")
    if capability:
        return str(capability).strip().lower()
    family = str(task.get("family", "")).strip().lower()
    if family in {"regex", "regex.synthesis"}:
        return "synthesis"
    if family == "regex.debugging":
        return "debugging"
    if family == "regex.recognition":
        return "recognition"
    if family in {"regex.explanation", "regex.mutation_effect"}:
        return "explanation"
    if family == "regex.refactoring":
        return "refactoring"
    return family or "unknown"


def _infer_bucket(task: dict[str, Any], *, mode: str) -> str:
    family = task.get("family")
    if family:
        base = str(family).strip().lower()
    else:
        capability = task.get("capability")
        if capability:
            base = str(capability).strip().lower()
        else:
            base = "unknown"
    mode_norm = str(mode or "").strip().lower()
    if mode_norm == "cell":
        depth = str(task.get("depth", "") or "").strip().lower()
        if depth:
            return f"{base}:{depth}"
    return base


def _unique_buckets(tasks: list[dict[str, Any]], *, mode: str) -> list[str]:
    buckets: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        bucket = _infer_bucket(task, mode=mode)
        if bucket in seen:
            continue
        seen.add(bucket)
        buckets.append(bucket)
    return buckets


def _extract_test_cases(task: dict[str, Any]) -> list[dict[str, object]]:
    test_cases = task.get("test_cases", []) or []
    if test_cases:
        return list(test_cases)
    target = task.get("target", {}) or {}
    if isinstance(target, dict):
        return list(target.get("test_strings", []) or [])
    return []


def _evaluate_selection_tasks(
    model,
    tokenizer,
    tasks: list[dict[str, Any]],
    *,
    max_new_tokens: int = 96,
    bucket_mode: str = "family",
) -> dict[str, Any]:
    from symbiont_ecology.utils.regex_extract import pick_best_regex_candidate

    correct = 0
    total = 0
    passed_cases = 0
    total_cases = 0
    by_capability: dict[str, dict[str, int]] = {}
    by_bucket: dict[str, dict[str, int]] = {}
    mode_norm = str(bucket_mode or "").strip().lower()

    for task in tasks:
        prompt = str(task.get("prompt", ""))
        if not prompt:
            continue
        bucket_key = _infer_bucket(task, mode="cell" if mode_norm == "cell" else "family")
        cap_key = _infer_capability(task)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        is_correct, _reason = check_answer(response, task)
        total += 1
        bucket = by_bucket.setdefault(bucket_key, {"correct": 0, "total": 0})
        bucket["total"] = int(bucket.get("total", 0) or 0) + 1
        if is_correct:
            correct += 1
            bucket["correct"] = int(bucket.get("correct", 0) or 0) + 1
        cap_bucket = by_capability.setdefault(cap_key, {"correct": 0, "total": 0})
        cap_bucket["total"] = int(cap_bucket.get("total", 0) or 0) + 1
        if is_correct:
            cap_bucket["correct"] = int(cap_bucket.get("correct", 0) or 0) + 1

        test_cases = _extract_test_cases(task)
        if not test_cases:
            continue
        pattern, _pick = pick_best_regex_candidate(response, test_cases=test_cases)
        if not pattern:
            total_cases += len(test_cases)
            continue
        try:
            compiled = re.compile(pattern)
        except re.error:
            total_cases += len(test_cases)
            continue
        for tc in test_cases:
            test_str = str(tc.get("string", ""))
            should_match = bool(tc.get("should_match", False))
            match = bool(compiled.fullmatch(test_str))
            if match == should_match:
                passed_cases += 1
            total_cases += 1

    acc = float(correct / total) if total else 0.0
    case_acc = float(passed_cases / total_cases) if total_cases else 0.0

    gm = acc
    bucket_accs: list[float] = []
    for stats in by_bucket.values():
        if not stats.get("total"):
            continue
        bucket_accs.append(float(stats["correct"] / stats["total"]))
    if bucket_accs:
        import math

        eps = 1e-6
        gm = float(math.exp(sum(math.log(max(a, eps)) for a in bucket_accs) / len(bucket_accs)))

    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "case_accuracy": case_acc,
        "cases_passed": passed_cases,
        "cases_total": total_cases,
        "geometric_mean_accuracy": gm,
        "by_capability": by_capability,
        "by_bucket": by_bucket,
    }


def _abs_max_lora_b(state: dict[str, torch.Tensor]) -> float:
    abs_max = 0.0
    for key, tensor in state.items():
        if "lora_B" not in key:
            continue
        if not isinstance(tensor, torch.Tensor) or tensor.numel() <= 0:
            continue
        try:
            value = float(tensor.detach().abs().max().item())
        except Exception:
            continue
        if value > abs_max:
            abs_max = value
    return abs_max


def _select_best_organelle_by_selection_tasks(
    peft_model,
    tokenizer,
    adapter_states: dict[str, dict[str, torch.Tensor]],
    tasks: list[dict[str, Any]],
    *,
    candidate_ids: list[str] | None = None,
    max_new_tokens: int = 96,
) -> tuple[str | None, dict[str, Any]]:
    best_id: str | None = None
    best_score: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0)
    best_details: dict[str, Any] = {}
    if not tasks:
        return None, {"error": "no selection tasks"}
    if candidate_ids is None:
        candidate_ids = list(adapter_states.keys())
    candidate_ids = [cid for cid in candidate_ids if cid in adapter_states]
    if not candidate_ids:
        return None, {"error": "no candidate organelles"}

    abs_max_by_id: dict[str, float] = {}
    for oid in candidate_ids:
        state = adapter_states.get(oid)
        if not isinstance(state, dict) or not state:
            continue
        abs_max_by_id[oid] = _abs_max_lora_b(state)

    # If any organelles actually diverged from the base adapter (non-zero LoRA-B), restrict
    # selection to those. This avoids picking a no-op adapter just because it preserves a
    # slightly higher geometric mean on an unrelated selection set.
    nonzero_ids = [oid for oid in candidate_ids if abs_max_by_id.get(oid, 0.0) > 0.0]
    if nonzero_ids:
        skipped = len(candidate_ids) - len(nonzero_ids)
        if skipped:
            print(f"  [select] Skipping {skipped} zero-magnitude adapters")
        candidate_ids = nonzero_ids

    total = len(candidate_ids)
    for idx, oid in enumerate(candidate_ids, 1):
        state = adapter_states.get(oid)
        if not isinstance(state, dict) or not state:
            continue
        abs_max_b = float(abs_max_by_id.get(oid, 0.0))
        if _apply_adapter_state_to_peft_model(peft_model, state) <= 0:
            continue
        metrics = _evaluate_selection_tasks(
            peft_model, tokenizer, tasks, max_new_tokens=max_new_tokens
        )
        score = (
            float(metrics.get("geometric_mean_accuracy", 0.0) or 0.0),
            float(metrics.get("accuracy", 0.0) or 0.0),
            float(metrics.get("case_accuracy", 0.0) or 0.0),
            float(abs_max_b),
        )
        if score > best_score:
            best_score = score
            best_id = oid
            best_details = {
                "selection_geometric_mean_accuracy": score[0],
                "selection_accuracy": score[1],
                "selection_case_accuracy": score[2],
                "selection_abs_max_lora_b": score[3],
                "selection_correct": metrics.get("correct", 0),
                "selection_total": metrics.get("total", 0),
                "selection_cases_passed": metrics.get("cases_passed", 0),
                "selection_cases_total": metrics.get("cases_total", 0),
                "selection_by_capability": metrics.get("by_capability", {}),
            }
        if total > 1:
            print(
                f"  [select] {idx}/{total} organelle={oid} gm_acc={score[0]:.3f} "
                f"acc={score[1]:.3f} cases={score[2]:.3f} abs_max_B={score[3]:.3g}"
            )
    return best_id, best_details


def _compute_selection_metrics_for_candidates(
    peft_model,
    tokenizer,
    adapter_states: dict[str, dict[str, torch.Tensor]],
    tasks: list[dict[str, Any]],
    *,
    candidate_ids: list[str] | None = None,
    max_new_tokens: int = 96,
    bucket_mode: str = "family",
    skip_zero_magnitude: bool = False,
) -> dict[str, dict[str, Any]]:
    if not tasks:
        return {}
    if candidate_ids is None:
        candidate_ids = list(adapter_states.keys())
    candidate_ids = [cid for cid in candidate_ids if cid in adapter_states]
    if not candidate_ids:
        return {}

    abs_max_by_id: dict[str, float] = {}
    for oid in candidate_ids:
        state = adapter_states.get(oid)
        if not isinstance(state, dict) or not state:
            continue
        abs_max_by_id[oid] = _abs_max_lora_b(state)

    if skip_zero_magnitude:
        nonzero_ids = [oid for oid in candidate_ids if abs_max_by_id.get(oid, 0.0) > 0.0]
        if nonzero_ids:
            skipped = len(candidate_ids) - len(nonzero_ids)
            if skipped:
                print(f"  [select] Skipping {skipped} zero-magnitude adapters")
            candidate_ids = nonzero_ids

    results: dict[str, dict[str, Any]] = {}
    total = len(candidate_ids)
    for idx, oid in enumerate(candidate_ids, 1):
        state = adapter_states.get(oid)
        if not isinstance(state, dict) or not state:
            continue
        abs_max_b = float(abs_max_by_id.get(oid, 0.0))
        loaded = _apply_adapter_state_to_peft_model(peft_model, state)
        if loaded <= 0:
            continue
        metrics = _evaluate_selection_tasks(
            peft_model,
            tokenizer,
            tasks,
            max_new_tokens=max_new_tokens,
            bucket_mode=bucket_mode,
        )
        results[oid] = {
            "selection_metrics": metrics,
            "selection_abs_max_lora_b": abs_max_b,
            "loaded_tensors": int(loaded),
        }
        if total > 1:
            print(
                f"  [select] {idx}/{total} organelle={oid} gm_acc={float(metrics.get('geometric_mean_accuracy', 0.0) or 0.0):.3f} "
                f"acc={float(metrics.get('accuracy', 0.0) or 0.0):.3f} abs_max_B={abs_max_b:.3g}"
            )
    return results


def _select_bucket_routing_from_candidate_metrics(
    candidate_metrics: dict[str, dict[str, Any]], buckets: list[str]
) -> tuple[dict[str, str], dict[str, Any]]:
    routing: dict[str, str] = {}
    details: dict[str, Any] = {}
    for bucket in buckets:
        best_oid: str | None = None
        best_score: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0)
        best_bucket_acc = 0.0
        for oid, info in candidate_metrics.items():
            metrics = info.get("selection_metrics") or {}
            by_bucket = metrics.get("by_bucket") or {}
            bucket_stats = by_bucket.get(bucket) or {}
            total = int(bucket_stats.get("total", 0) or 0)
            correct = int(bucket_stats.get("correct", 0) or 0)
            if total <= 0:
                continue
            bucket_acc = float(correct / total)
            score = (
                bucket_acc,
                float(metrics.get("geometric_mean_accuracy", 0.0) or 0.0),
                float(metrics.get("accuracy", 0.0) or 0.0),
                float(info.get("selection_abs_max_lora_b", 0.0) or 0.0),
            )
            if score > best_score:
                best_score = score
                best_oid = oid
                best_bucket_acc = bucket_acc
        if best_oid is None:
            continue
        routing[bucket] = best_oid
        details[bucket] = {
            "organelle_id": best_oid,
            "bucket_accuracy": best_bucket_acc,
            "score": {
                "bucket_accuracy": best_score[0],
                "candidate_gm_accuracy": best_score[1],
                "candidate_accuracy": best_score[2],
                "candidate_abs_max_lora_b": best_score[3],
            },
        }
    return routing, details


def _select_global_best_from_candidate_metrics(
    candidate_metrics: dict[str, dict[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    best_id: str | None = None
    best_score: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0)
    best_details: dict[str, Any] = {}
    for oid, info in candidate_metrics.items():
        metrics = info.get("selection_metrics") or {}
        score = (
            float(metrics.get("geometric_mean_accuracy", 0.0) or 0.0),
            float(metrics.get("accuracy", 0.0) or 0.0),
            float(metrics.get("case_accuracy", 0.0) or 0.0),
            float(info.get("selection_abs_max_lora_b", 0.0) or 0.0),
        )
        if score > best_score:
            best_score = score
            best_id = oid
            best_details = {
                "selection_geometric_mean_accuracy": score[0],
                "selection_accuracy": score[1],
                "selection_case_accuracy": score[2],
                "selection_abs_max_lora_b": score[3],
                "selection_by_capability": metrics.get("by_capability", {}),
                "selection_correct": metrics.get("correct", 0),
                "selection_total": metrics.get("total", 0),
                "selection_cases_passed": metrics.get("cases_passed", 0),
                "selection_cases_total": metrics.get("cases_total", 0),
            }
    return best_id, best_details


def _evaluate_model_with_evo_routing(
    *,
    peft_model,
    tokenizer,
    tasks: list[dict[str, Any]],
    adapter_states: dict[str, dict[str, torch.Tensor]],
    routing: dict[str, str],
    routing_mode: str,
    fallback_organelle_id: str,
    model_name: str,
    max_samples: int | None = None,
    verbose: bool = False,
) -> EvalResult:
    eval_tasks = tasks[:max_samples] if max_samples else tasks
    routed_groups: dict[str, list[dict[str, Any]]] = {}
    for task in eval_tasks:
        bucket = _infer_bucket(task, mode=routing_mode)
        oid = routing.get(bucket) or fallback_organelle_id
        routed_groups.setdefault(str(oid), []).append(task)

    result = EvalResult(model_name=model_name)
    for oid, group in routed_groups.items():
        state = adapter_states.get(oid)
        if not isinstance(state, dict) or not state:
            continue
        loaded = _apply_adapter_state_to_peft_model(peft_model, state)
        if loaded <= 0:
            continue
        partial = evaluate_model(
            peft_model,
            tokenizer,
            group,
            model_name=model_name,
            max_samples=None,
            verbose=verbose,
        )
        result.merge_from(partial)
    return result


def _load_evo_peft_model_and_states(
    base_model, checkpoint_path: Path, *, allow_unsafe_pickle: bool = False
) -> tuple[object, dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    from peft import LoraConfig, get_peft_model

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, allow_unsafe_pickle=allow_unsafe_pickle)
    if not isinstance(checkpoint, dict):
        raise ValueError("Evolution checkpoint did not deserialize into a dict")
    adapter_states = checkpoint.get("adapter_states", {})
    if not isinstance(adapter_states, dict) or not adapter_states:
        raise ValueError("No adapter_states found in checkpoint")

    # Infer a PEFT config that can load *all* organelles (some may differ in rank due to morphogenesis).
    # We build a max-rank adapter and pad smaller ranks when loading weights.
    lora_rank = 0
    target_modules: set[str] = set()
    for state in adapter_states.values():
        if not isinstance(state, dict) or not state:
            continue
        for key, tensor in state.items():
            if "lora_A" in key:
                try:
                    lora_rank = max(lora_rank, int(tensor.shape[0]))
                except Exception:
                    continue
            if any(module in key for module in _LORA_TARGET_MODULES):
                for part in key.split("."):
                    if part in _LORA_TARGET_MODULES:
                        target_modules.add(part)
    if lora_rank <= 0:
        raise ValueError("Could not infer LoRA rank from checkpoint adapter_states")
    if not target_modules:
        raise ValueError("Could not infer LoRA target_modules from checkpoint adapter_states")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=list(target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.eval()

    meta = {"lora_rank": int(lora_rank), "target_modules": sorted(target_modules)}
    return peft_model, adapter_states, meta


def load_evo_model(
    base_model,
    checkpoint_path: Path,
    tokenizer,
    organelle_id: str | None = None,
    *,
    selection_tasks_path: Path | None = None,
    selection_max_samples: int | None = None,
    selection_seed: int = 9403,
    selection_family: str | None = "any",
    selection_top_k_by_roi: int | None = None,
    selection_max_new_tokens: int = 96,
    allow_unsafe_pickle: bool = False,
):
    """Load evolution checkpoint and apply the best organelle.

    Args:
        base_model: The base HuggingFace model
        checkpoint_path: Path to evolution checkpoint.pt
        tokenizer: Tokenizer (unused but kept for API consistency)
        organelle_id: Specific organelle to load, or None to auto-select best by ROI
        selection_tasks_path: Optional validation set used to select the organelle (recommended).
        selection_max_samples: Optional cap on the number of selection tasks evaluated.
        selection_seed: RNG seed for selection task sampling.
        selection_family: Optional family filter for selection tasks.
        selection_max_new_tokens: Generation cap during selection (keep low to reduce cost).

    Returns:
        (PEFT model with the selected organelle applied, organelle_id)
        Returns (base_model, None) if loading fails
    """
    try:
        peft_model, adapter_states, meta = _load_evo_peft_model_and_states(
            base_model, checkpoint_path, allow_unsafe_pickle=allow_unsafe_pickle
        )
    except Exception as exc:
        print(f"  [warn] Failed to load evolution checkpoint: {exc}")
        return base_model, None

    lora_rank = int(meta.get("lora_rank", 0) or 0)
    target_modules = list(meta.get("target_modules", []) or [])
    print(f"  [info] Evolution checkpoint has {len(adapter_states)} organelles")
    print(f"  [info] LoRA config: rank={lora_rank}, targets={sorted(target_modules)}")

    # Prefer organelle selection by a separate validation/selection set (avoid training-ROI bias).
    selection_details: dict[str, Any] = {}
    if organelle_id is None:
        try:
            if selection_tasks_path is not None:
                candidate_ids: list[str] | None = None
                shortlist_k = (
                    int(selection_top_k_by_roi)
                    if selection_top_k_by_roi is not None and int(selection_top_k_by_roi) > 0
                    else None
                )
                if shortlist_k is not None and shortlist_k < len(adapter_states):
                    summaries_path = checkpoint_path.parent / "gen_summaries.jsonl"
                    if summaries_path.exists():
                        try:
                            summaries = [
                                json.loads(line)
                                for line in summaries_path.read_text().splitlines()
                                if line.strip()
                            ]
                            if summaries:
                                roi_by_org = summaries[-1].get("roi_by_organelle", {}) or {}
                                valid_rois: list[tuple[str, float]] = []
                                for oid, roi in roi_by_org.items():
                                    if oid in adapter_states:
                                        try:
                                            valid_rois.append((str(oid), float(roi)))
                                        except Exception:
                                            continue
                                valid_rois.sort(key=lambda item: item[1], reverse=True)
                                if valid_rois:
                                    candidate_ids = [oid for oid, _ in valid_rois[:shortlist_k]]
                                    print(
                                        f"  [info] Selection shortlist by ROI: {len(candidate_ids)} candidates"
                                    )
                        except Exception:
                            candidate_ids = None
                tasks = _load_selection_tasks(
                    selection_tasks_path,
                    max_samples=selection_max_samples,
                    seed=selection_seed,
                    family=selection_family,
                )
                choice, details = _select_best_organelle_by_selection_tasks(
                    peft_model,
                    tokenizer,
                    adapter_states,
                    tasks,
                    candidate_ids=candidate_ids,
                    max_new_tokens=selection_max_new_tokens,
                )
                if choice is not None:
                    organelle_id = choice
                    selection_details = details
                    print(
                        f"  [info] Selected organelle by validation tasks: {organelle_id} "
                        f"(acc={details.get('selection_accuracy', 0.0):.3f}, "
                        f"cases={details.get('selection_case_accuracy', 0.0):.3f})"
                    )
        except Exception as exc:
            print(f"  [warn] Failed to select organelle by validation tasks: {exc}")
            selection_details = {}

    # Fallback: best organelle by overall ROI from gen_summaries (cheap, but selection-biased).
    if organelle_id is None:
        summaries_path = checkpoint_path.parent / "gen_summaries.jsonl"
        if summaries_path.exists():
            summaries = [
                json.loads(line) for line in summaries_path.read_text().splitlines() if line.strip()
            ]
            if summaries:
                final = summaries[-1]
                roi_by_org = final.get("roi_by_organelle", {})
                # Filter to organelles that exist in adapter_states
                valid_rois = {k: v for k, v in roi_by_org.items() if k in adapter_states}
                if valid_rois:
                    organelle_id = max(valid_rois.items(), key=lambda x: x[1])[0]
                    print(
                        f"  [info] Selected best organelle: {organelle_id} (ROI: {valid_rois[organelle_id]:.4f})"
                    )

    if organelle_id is None:
        # Fallback: just pick the first one
        organelle_id = list(adapter_states.keys())[0]
        print(f"  [info] Using first organelle: {organelle_id}")

    if organelle_id not in adapter_states:
        print(f"  [error] Organelle {organelle_id} not found in checkpoint")
        return base_model, None

    # Get the adapter state dict
    adapter_state = adapter_states[organelle_id]
    if not isinstance(adapter_state, dict) or not adapter_state:
        print("  [warn] Selected organelle has no adapter tensors; using base model")
        _reset_lora_weights(peft_model)
        return peft_model, None
    loaded_count = _apply_adapter_state_to_peft_model(peft_model, adapter_state)
    if loaded_count <= 0:
        print("  [warn] No adapter tensors loaded; using base model")
        _reset_lora_weights(peft_model)
        return peft_model, None
    if selection_details:
        peft_model._selection_details = selection_details  # type: ignore[attr-defined]
    print(f"  [info] Loaded {loaded_count} tensors from organelle {organelle_id}")
    return peft_model, organelle_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate models on holdout tasks for generalization comparison."
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        required=True,
        help="Path to holdout tasks JSONL file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Base model name (default: Qwen/Qwen2.5-0.5B).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code when loading HF model/tokenizer (default: disabled).",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
        help=(
            "Attention implementation passed to the HF model loader (default: sdpa). "
            "Use eager if you need a numerically safer attention kernel on your device."
        ),
    )
    parser.add_argument(
        "--sft-adapter",
        type=Path,
        default=None,
        help="Path to SFT PEFT adapter directory.",
    )
    parser.add_argument(
        "--evo-checkpoint",
        type=Path,
        default=None,
        help="Path to evolution checkpoint.pt file.",
    )
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help=(
            "Allow trusted legacy pickle checkpoints for --evo-checkpoint. "
            "Unsafe: untrusted pickle files can execute arbitrary code."
        ),
    )
    parser.add_argument(
        "--evo-organelle-id",
        type=str,
        default=None,
        help="Optional organelle_id to evaluate from the evolution checkpoint (overrides auto-selection).",
    )
    default_selection_tasks = (
        Path(__file__).resolve().parents[1] / "config" / "evaluation" / "holdout_regex.jsonl"
    )
    parser.add_argument(
        "--evo-selection-tasks",
        type=Path,
        default=default_selection_tasks if default_selection_tasks.exists() else None,
        help=(
            "Optional selection/validation tasks JSONL used to pick the best evolved organelle "
            "(default: config/evaluation/holdout_regex.jsonl). "
            "Must be different from --holdout to avoid test leakage."
        ),
    )
    parser.add_argument(
        "--evo-selection-max-samples",
        type=int,
        default=None,
        help="Optional cap on selection task count (default: all).",
    )
    parser.add_argument(
        "--evo-selection-seed",
        type=int,
        default=9403,
        help="Seed for selection task sampling (default: 9403).",
    )
    parser.add_argument(
        "--evo-selection-family",
        type=str,
        default="any",
        help="Optional family filter applied to selection tasks (default: any).",
    )
    parser.add_argument(
        "--evo-routing-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing a precomputed routing map for evo routed evaluation "
            "(skips selection-task evaluation). Accepts either a direct mapping of bucket->organelle_id "
            "or a run_evolution final_holdout.json payload (uses its 'selection' field)."
        ),
    )
    parser.add_argument(
        "--evo-eval-routing",
        type=str,
        choices=["single", "family", "cell"],
        default="single",
        help=(
            "How to evaluate evolution: a single selected organelle (single), or a routed portfolio "
            "picked by selection tasks per family/cell."
        ),
    )
    parser.add_argument(
        "--evo-selection-max-new-tokens",
        type=int,
        default=96,
        help="Max new tokens during selection generations (default: 96).",
    )
    parser.add_argument(
        "--evo-selection-top-k-by-roi",
        type=int,
        default=None,
        help=(
            "Optional shortlist of candidate organelles (by ROI from gen_summaries.jsonl) before "
            "running selection tasks. This trades a small selection-bias risk for much faster "
            "selection (default: all organelles)."
        ),
    )
    parser.add_argument(
        "--no-evo-selection",
        action="store_true",
        help="Disable selection-task organelle picking (falls back to ROI-based selection).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of holdout tasks to evaluate (default: all).",
    )
    parser.add_argument(
        "--holdout-sampling",
        type=str,
        choices=["head", "random", "stratified_family", "stratified_cell"],
        default="head",
        help=(
            "How to sample holdout tasks when --max-samples is set (default: head). "
            "Use stratified_family/stratified_cell for small smoke tests on mixed buckets."
        ),
    )
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=9403,
        help="RNG seed for holdout sampling (default: 9403).",
    )
    parser.add_argument(
        "--grid-prompt-augmentations",
        type=str,
        default="",
        help=(
            "Optional comma-separated grid prompt augmentations for robustness evaluation. "
            "Example: default or clean,ws_prefix,markdown_sections. "
            "When set, each GridTask holdout item spawns multiple prompt variants."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each task.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save results as JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu).",
    )

    args = parser.parse_args()

    # Load holdout tasks
    print(f"Loading holdout tasks from {args.holdout}")
    tasks = load_holdout_tasks(args.holdout)
    print(f"  Loaded {len(tasks)} tasks")

    if args.max_samples:
        tasks = _sample_holdout_tasks(
            tasks,
            max_samples=args.max_samples,
            sampling=args.holdout_sampling,
            seed=args.holdout_seed,
        )
        print(
            f"  Evaluating on {len(tasks)} samples (sampling={args.holdout_sampling}, seed={args.holdout_seed})"
        )

    augmentations_raw = _parse_comma_list(args.grid_prompt_augmentations)
    tasks, augmentations_used = _augment_grid_holdout_tasks(tasks, augmentations_raw)
    if augmentations_used:
        print(
            f"  Prompt robustness enabled: {len(augmentations_used)} augmentations "
            f"({', '.join(augmentations_used)})"
        )
        print(f"  Expanded to {len(tasks)} total prompt variants")

    # Load base model and tokenizer
    print(f"\nLoading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=bool(args.trust_remote_code)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_device = args.device
    if resolved_device == "auto":
        if torch.backends.mps.is_available():
            resolved_device = "mps"
        elif torch.cuda.is_available():
            resolved_device = "cuda"
        else:
            resolved_device = "cpu"
    dtype = torch.float16 if resolved_device in {"mps", "cuda"} else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=args.attn_implementation,
    )
    base_model.to(resolved_device)
    base_model.eval()

    results = []
    evo_selection_details: dict[str, Any] | None = None
    evo_selection_tasks: str | None = None
    evo_selected_organelle: str | None = None

    # Evaluate base model
    print("\n" + "=" * 50)
    print("Evaluating BASE model")
    print("=" * 50)
    base_result = evaluate_model(
        base_model, tokenizer, tasks, "base", max_samples=None, verbose=args.verbose
    )
    results.append(base_result)
    print(
        f"\nBase model: {base_result.correct}/{base_result.total} = {100*base_result.accuracy:.1f}%"
    )

    # Evaluate SFT model if provided
    if args.sft_adapter:
        print("\n" + "=" * 50)
        print(f"Evaluating SFT model from {args.sft_adapter}")
        print("=" * 50)
        if args.sft_adapter.exists():
            # Load fresh base model to avoid state pollution
            sft_base = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=dtype,
                trust_remote_code=bool(args.trust_remote_code),
                attn_implementation=args.attn_implementation,
            )
            sft_base.to(resolved_device)
            sft_base.eval()
            sft_model = load_sft_model(sft_base, args.sft_adapter)
            sft_result = evaluate_model(
                sft_model,
                tokenizer,
                tasks,
                "sft",
                max_samples=None,
                verbose=args.verbose,
            )
            results.append(sft_result)
            print(
                f"\nSFT model: {sft_result.correct}/{sft_result.total} = {100*sft_result.accuracy:.1f}%"
            )
            del sft_model, sft_base  # Free memory
        else:
            print(f"  [error] SFT adapter not found: {args.sft_adapter}")

    # Evaluate evolution model if provided
    if args.evo_checkpoint:
        print("\n" + "=" * 50)
        print(f"Evaluating EVOLUTION model from {args.evo_checkpoint}")
        print("=" * 50)
        if args.evo_checkpoint.exists():
            selection_tasks_path = None if args.no_evo_selection else args.evo_selection_tasks
            if args.evo_organelle_id is None and selection_tasks_path is not None:
                same_set = False
                try:
                    same_set = (
                        selection_tasks_path.exists()
                        and selection_tasks_path.resolve() == args.holdout.resolve()
                    )
                except Exception:
                    same_set = False
                if same_set:
                    raise ValueError(
                        "--evo-selection-tasks must be different from --holdout to avoid test leakage"
                    )
            # Load fresh base model to avoid state pollution
            evo_base = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=dtype,
                trust_remote_code=bool(args.trust_remote_code),
                attn_implementation=args.attn_implementation,
            )
            evo_base.to(resolved_device)
            evo_base.eval()
            routing_mode = (
                str(getattr(args, "evo_eval_routing", "single") or "single").strip().lower()
            )
            if routing_mode in {"family", "cell"} and args.evo_routing_json is not None:
                try:
                    evo_model, adapter_states, _meta = _load_evo_peft_model_and_states(
                        evo_base,
                        args.evo_checkpoint,
                        allow_unsafe_pickle=args.allow_unsafe_pickle,
                    )
                except Exception as exc:
                    print(f"  [error] Failed to load evolution checkpoint: {exc}")
                    evo_model = None
                    adapter_states = {}

                if evo_model is None or not adapter_states:
                    print(
                        "  [warn] Routed evaluation requires a checkpoint with adapters; falling back to single"
                    )
                    routing_mode = "single"
                else:
                    routing_payload = json.loads(args.evo_routing_json.read_text(encoding="utf-8"))
                    routing: dict[str, str] | None = None
                    if isinstance(routing_payload, dict) and isinstance(
                        routing_payload.get("selection"), dict
                    ):
                        routing = {}
                        for key, value in routing_payload["selection"].items():
                            if isinstance(value, str) and value:
                                routing[str(key)] = str(value)
                                continue
                            if isinstance(value, dict):
                                organelle_id = value.get("organelle_id")
                                if isinstance(organelle_id, str) and organelle_id:
                                    routing[str(key)] = str(organelle_id)
                    elif isinstance(routing_payload, dict):
                        routing = {}
                        for key, value in routing_payload.items():
                            if isinstance(value, str) and value:
                                routing[str(key)] = str(value)
                                continue
                            if isinstance(value, dict):
                                organelle_id = value.get("organelle_id")
                                if isinstance(organelle_id, str) and organelle_id:
                                    routing[str(key)] = str(organelle_id)
                    if not routing:
                        raise ValueError(f"No routing map found in {args.evo_routing_json}")

                    bucket_set = sorted({_infer_bucket(t, mode=routing_mode) for t in tasks})
                    missing = [b for b in bucket_set if b not in routing]
                    if missing:
                        print(
                            f"  [warn] Routing map missing {len(missing)}/{len(bucket_set)} buckets; "
                            f"will fall back for: {missing[:6]}"
                        )

                    # Prefer the most common routed organelle as fallback.
                    counts: dict[str, int] = {}
                    for oid in routing.values():
                        counts[oid] = counts.get(oid, 0) + 1
                    fallback_oid = max(counts.items(), key=lambda kv: kv[1])[0] if counts else ""
                    if not fallback_oid and adapter_states:
                        fallback_oid = next(iter(adapter_states.keys()))
                    if (
                        args.evo_organelle_id is not None
                        and args.evo_organelle_id in adapter_states
                    ):
                        fallback_oid = str(args.evo_organelle_id)

                    model_name = f"evolution routed ({routing_mode})"
                    evo_result = _evaluate_model_with_evo_routing(
                        peft_model=evo_model,
                        tokenizer=tokenizer,
                        tasks=tasks,
                        adapter_states=adapter_states,
                        routing=routing,
                        routing_mode=routing_mode,
                        fallback_organelle_id=fallback_oid,
                        model_name=model_name,
                        max_samples=None,
                        verbose=args.verbose,
                    )
                    results.append(evo_result)
                    evo_selected_organelle = fallback_oid
                    evo_selection_tasks = None
                    evo_selection_details = {
                        "selection_mode": "routing_json",
                        "routing_mode": routing_mode,
                        "routing_json": str(args.evo_routing_json),
                        "bucket_count": len(bucket_set),
                        "fallback_organelle_id": fallback_oid,
                        "routing": routing,
                    }
                    print(
                        f"\nEvolution model: {evo_result.correct}/{evo_result.total} = {100*evo_result.accuracy:.1f}%"
                    )
                    del evo_model, evo_base  # Free memory
            elif routing_mode in {"family", "cell"} and selection_tasks_path is not None:
                try:
                    evo_model, adapter_states, _meta = _load_evo_peft_model_and_states(
                        evo_base,
                        args.evo_checkpoint,
                        allow_unsafe_pickle=args.allow_unsafe_pickle,
                    )
                except Exception as exc:
                    print(f"  [error] Failed to load evolution checkpoint: {exc}")
                    evo_model = None
                    adapter_states = {}

                selection_tasks: list[dict[str, Any]] = []
                if evo_model is not None and selection_tasks_path is not None:
                    selection_tasks = _load_selection_tasks(
                        selection_tasks_path,
                        max_samples=args.evo_selection_max_samples,
                        seed=args.evo_selection_seed,
                        family=args.evo_selection_family,
                    )

                if evo_model is None or not selection_tasks:
                    print(
                        "  [warn] Routed evaluation requires non-empty selection tasks; falling back to single"
                    )
                    if evo_model is not None:
                        del evo_model
                    routing_mode = "single"
                else:
                    buckets = _unique_buckets(selection_tasks, mode=routing_mode)
                    candidate_ids: list[str] | None = None
                    shortlist_k = (
                        int(args.evo_selection_top_k_by_roi)
                        if args.evo_selection_top_k_by_roi is not None
                        and int(args.evo_selection_top_k_by_roi) > 0
                        else None
                    )
                    if shortlist_k is not None and shortlist_k < len(adapter_states):
                        summaries_path = args.evo_checkpoint.parent / "gen_summaries.jsonl"
                        if summaries_path.exists():
                            try:
                                summaries = [
                                    json.loads(line)
                                    for line in summaries_path.read_text().splitlines()
                                    if line.strip()
                                ]
                                if summaries:
                                    roi_by_org = summaries[-1].get("roi_by_organelle", {}) or {}
                                    valid_rois: list[tuple[str, float]] = []
                                    for oid, roi in roi_by_org.items():
                                        if oid in adapter_states:
                                            try:
                                                valid_rois.append((str(oid), float(roi)))
                                            except Exception:
                                                continue
                                    valid_rois.sort(key=lambda item: item[1], reverse=True)
                                    if valid_rois:
                                        candidate_ids = [oid for oid, _ in valid_rois[:shortlist_k]]
                                        print(
                                            f"  [info] Selection shortlist by ROI: {len(candidate_ids)} candidates"
                                        )
                            except Exception:
                                candidate_ids = None

                    candidate_metrics = _compute_selection_metrics_for_candidates(
                        evo_model,
                        tokenizer,
                        adapter_states,
                        selection_tasks,
                        candidate_ids=candidate_ids,
                        max_new_tokens=args.evo_selection_max_new_tokens,
                        bucket_mode=routing_mode,
                        skip_zero_magnitude=False,
                    )
                    routing, routing_details = _select_bucket_routing_from_candidate_metrics(
                        candidate_metrics, buckets
                    )
                    global_best, global_details = _select_global_best_from_candidate_metrics(
                        candidate_metrics
                    )
                    fallback_oid = global_best or (
                        list(adapter_states.keys())[0] if adapter_states else ""
                    )
                    if (
                        args.evo_organelle_id is not None
                        and args.evo_organelle_id in adapter_states
                    ):
                        fallback_oid = str(args.evo_organelle_id)

                    model_name = f"evolution routed ({routing_mode})"
                    evo_result = _evaluate_model_with_evo_routing(
                        peft_model=evo_model,
                        tokenizer=tokenizer,
                        tasks=tasks,
                        adapter_states=adapter_states,
                        routing=routing,
                        routing_mode=routing_mode,
                        fallback_organelle_id=fallback_oid,
                        model_name=model_name,
                        max_samples=None,
                        verbose=args.verbose,
                    )
                    results.append(evo_result)
                    evo_selected_organelle = fallback_oid
                    evo_selection_tasks = (
                        str(selection_tasks_path) if selection_tasks_path else None
                    )
                    evo_selection_details = {
                        "selection_mode": f"routed_{routing_mode}",
                        "routing_mode": routing_mode,
                        "bucket_count": len(buckets),
                        "candidate_count": len(candidate_metrics),
                        "fallback_organelle_id": fallback_oid,
                        "global_best_organelle_id": global_best,
                        "global_best_details": global_details,
                        "routing": routing,
                        "routing_details": routing_details,
                    }
                    print(
                        f"\nEvolution model: {evo_result.correct}/{evo_result.total} = {100*evo_result.accuracy:.1f}%"
                    )
                    del evo_model, evo_base  # Free memory

            if routing_mode == "single":
                evo_model, organelle_id = load_evo_model(
                    evo_base,
                    args.evo_checkpoint,
                    tokenizer,
                    organelle_id=args.evo_organelle_id,
                    selection_tasks_path=selection_tasks_path,
                    selection_max_samples=args.evo_selection_max_samples,
                    selection_seed=args.evo_selection_seed,
                    selection_family=args.evo_selection_family,
                    selection_top_k_by_roi=args.evo_selection_top_k_by_roi,
                    selection_max_new_tokens=args.evo_selection_max_new_tokens,
                    allow_unsafe_pickle=args.allow_unsafe_pickle,
                )
                evo_selection_details = getattr(evo_model, "_selection_details", None)
                evo_selection_tasks = str(selection_tasks_path) if selection_tasks_path else None
                evo_selected_organelle = organelle_id
                model_name = f"evolution ({organelle_id})" if organelle_id else "evolution"
                evo_result = evaluate_model(
                    evo_model,
                    tokenizer,
                    tasks,
                    model_name,
                    max_samples=None,
                    verbose=args.verbose,
                )
                results.append(evo_result)
                print(
                    f"\nEvolution model: {evo_result.correct}/{evo_result.total} = {100*evo_result.accuracy:.1f}%"
                )
                del evo_model, evo_base  # Free memory
        else:
            print(f"  [error] Evolution checkpoint not found: {args.evo_checkpoint}")

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'Model':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 45)
    for r in results:
        print(f"{r.model_name:<15} {r.correct:<10} {r.total:<10} {100*r.accuracy:.1f}%")

    robustness_summary = _compute_prompt_robustness_summary(tasks, results, augmentations_used)
    if robustness_summary is not None:
        print("\n" + "=" * 50)
        print("PROMPT ROBUSTNESS (ALL VARIANTS MUST PASS)")
        print("=" * 50)
        print(f"{'Model':<22} {'Clean':>8} {'All':>8} {'Any':>8} {'Brittle':>10}")
        print("-" * 60)
        per_model = robustness_summary.get("prompt_robustness", {}) or {}
        for model_name, stats in per_model.items():
            clean = float(stats.get("clean_accuracy", 0.0) or 0.0)
            all_acc = float(stats.get("all_variants_accuracy", 0.0) or 0.0)
            any_acc = float(stats.get("any_variant_accuracy", 0.0) or 0.0)
            brittle = stats.get("brittleness_rate")
            brittle_str = "n/a" if brittle is None else f"{100*float(brittle):.1f}%"
            print(
                f"{model_name:<22} {100*clean:7.1f}% {100*all_acc:7.1f}% {100*any_acc:7.1f}% {brittle_str:>10}"
            )

    # Save results if requested
    if args.output:
        output_data = {
            "holdout_file": str(args.holdout),
            "base_model": args.model,
            "max_samples": args.max_samples,
            "results": [r.summary() for r in results],
            "task_details": {r.model_name: r.task_results for r in results},
        }
        if robustness_summary is not None:
            output_data.update(robustness_summary)
        if args.evo_checkpoint:
            output_data["evolution"] = {
                "checkpoint": str(args.evo_checkpoint),
                "selected_organelle_id": evo_selected_organelle,
                "selection_tasks": evo_selection_tasks,
                "selection_details": evo_selection_details,
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
