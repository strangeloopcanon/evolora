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
import pickle
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "correct": self.correct,
            "total": self.total,
            "accuracy": self.accuracy,
            "accuracy_pct": f"{100 * self.accuracy:.1f}%",
        }


def load_holdout_tasks(path: Path) -> list[dict[str, Any]]:
    """Load holdout tasks from JSONL file."""
    tasks = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


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

        task_result = {
            "task_id": task_id,
            "correct": is_correct,
            "reason": reason,
            "response_preview": response[:100],
        }
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
    family: str | None = "regex",
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
) -> dict[str, Any]:
    from symbiont_ecology.utils.regex_extract import pick_best_regex_candidate

    correct = 0
    total = 0
    passed_cases = 0
    total_cases = 0
    by_capability: dict[str, dict[str, int]] = {}

    for task in tasks:
        prompt = str(task.get("prompt", ""))
        if not prompt:
            continue
        cap = _infer_capability(task)

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
        bucket = by_capability.setdefault(cap, {"correct": 0, "total": 0})
        bucket["total"] += 1
        if is_correct:
            correct += 1
            bucket["correct"] += 1

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

    # Geometric mean across major capability buckets to avoid selecting a specialist that collapses
    # on one axis (e.g., synthesis-only).
    major_caps = ["recognition", "synthesis", "explanation", "debugging", "refactoring"]
    cap_accs: list[float] = []
    for c in major_caps:
        stats = by_capability.get(c)
        if not stats or not stats.get("total"):
            continue
        cap_accs.append(float(stats["correct"] / stats["total"]))
    if cap_accs:
        import math

        eps = 1e-6
        gm = float(math.exp(sum(math.log(max(a, eps)) for a in cap_accs) / len(cap_accs)))
    else:
        gm = acc

    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "case_accuracy": case_acc,
        "cases_passed": passed_cases,
        "cases_total": total_cases,
        "geometric_mean_accuracy": gm,
        "by_capability": by_capability,
    }


def _select_best_organelle_by_selection_tasks(
    peft_model,
    tokenizer,
    adapter_states: dict[str, dict[str, torch.Tensor]],
    tasks: list[dict[str, Any]],
    *,
    candidate_ids: list[str] | None = None,
    max_new_tokens: int = 96,
) -> tuple[str | None, dict[str, Any]]:
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


def load_evo_model(
    base_model,
    checkpoint_path: Path,
    tokenizer,
    organelle_id: str | None = None,
    *,
    selection_tasks_path: Path | None = None,
    selection_max_samples: int | None = None,
    selection_seed: int = 9403,
    selection_family: str | None = "regex",
    selection_top_k_by_roi: int | None = None,
    selection_max_new_tokens: int = 96,
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
    from peft import LoraConfig, get_peft_model

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = pickle.loads(checkpoint_path.read_bytes())

    # Try to load adapter states from checkpoint
    adapter_states = checkpoint.get("adapter_states", {})
    if not adapter_states:
        print("  [warn] No adapter states found in checkpoint, using base model")
        return base_model, None

    print(f"  [info] Evolution checkpoint has {len(adapter_states)} organelles")

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
        print("  [error] Could not infer LoRA rank from checkpoint")
        return base_model, None
    if not target_modules:
        print("  [error] Could not infer LoRA target_modules from checkpoint")
        return base_model, None

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
    print(f"  [info] LoRA config: rank={lora_rank}, targets={sorted(target_modules)}")
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
        default="regex",
        help="Optional family filter applied to selection tasks (default: regex).",
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
        print(f"  Evaluating on {min(args.max_samples, len(tasks))} samples")

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
        base_model, tokenizer, tasks, "base", max_samples=args.max_samples, verbose=args.verbose
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
                max_samples=args.max_samples,
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
                max_samples=args.max_samples,
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

    # Save results if requested
    if args.output:
        output_data = {
            "holdout_file": str(args.holdout),
            "base_model": args.model,
            "max_samples": args.max_samples,
            "results": [r.summary() for r in results],
            "task_details": {r.model_name: r.task_results for r in results},
        }
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
