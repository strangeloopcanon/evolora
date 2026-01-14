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
    expected = task.get("expected_answer")
    test_cases = task.get("test_cases", [])
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
        is_correct = found >= len(required_keywords) // 2  # At least half
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


def _select_best_organelle_by_training_roi(
    episodes_path: Path, adapter_ids: set[str], *, family: str = "regex"
) -> str | None:
    if not episodes_path.exists():
        return None
    roi_sum: dict[str, float] = {}
    roi_count: dict[str, int] = {}
    with episodes_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") != "episode":
                continue
            organelles = obj.get("organelles") or []
            if not organelles:
                continue
            oid = str(organelles[0])
            if oid not in adapter_ids:
                continue
            obs = obj.get("observations") or {}
            cell = obs.get("cell") or {}
            if str(cell.get("family", "")).lower() != str(family).lower():
                continue
            try:
                roi = float(obs.get("roi", 0.0))
            except Exception:
                continue
            roi_sum[oid] = float(roi_sum.get(oid, 0.0)) + roi
            roi_count[oid] = int(roi_count.get(oid, 0)) + 1
    if not roi_sum:
        return None
    scored = [(oid, roi_sum[oid] / max(1, roi_count.get(oid, 1))) for oid in roi_sum]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[0][0] if scored else None


def load_evo_model(
    base_model,
    checkpoint_path: Path,
    tokenizer,
    organelle_id: str | None = None,
    *,
    training_family: str | None = "regex",
):
    """Load evolution checkpoint and apply the best organelle.

    Args:
        base_model: The base HuggingFace model
        checkpoint_path: Path to evolution checkpoint.pt
        tokenizer: Tokenizer (unused but kept for API consistency)
        organelle_id: Specific organelle to load, or None to auto-select best by ROI
        training_family: When set, prefers the organelle with best training ROI on that family.

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

    # Prefer organelles that performed well on the relevant training family (e.g., regex tasks).
    if organelle_id is None:
        try:
            if training_family:
                episodes_path = checkpoint_path.parent / "episodes.jsonl"
                choice = _select_best_organelle_by_training_roi(
                    episodes_path, set(adapter_states.keys()), family=training_family
                )
                if choice is not None:
                    organelle_id = choice
                    print(
                        f"  [info] Selected organelle by training ROI on {training_family}: {organelle_id}"
                    )
        except Exception:
            pass

    # Fallback: best organelle by overall ROI from gen_summaries.
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

    # Infer LoRA config from the state dict
    # Find all lora_A tensors to get the rank and target modules
    lora_rank = None
    target_modules = set()
    for key, tensor in adapter_state.items():
        if "lora_A" in key:
            if lora_rank is None:
                lora_rank = tensor.shape[0]  # rank is first dimension of lora_A
            # Extract module name (e.g., "q_proj" from "...self_attn.q_proj.lora_A")
            parts = key.replace("base_model.model.model.", "").split(".")
            for part in parts:
                if part in _LORA_TARGET_MODULES:
                    target_modules.add(part)

    if lora_rank is None:
        print("  [error] Could not infer LoRA rank from checkpoint")
        return base_model, None

    print(f"  [info] LoRA config: rank={lora_rank}, targets={sorted(target_modules)}")

    # Create PEFT model with matching config
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,  # Common default
        target_modules=list(target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(base_model, lora_config)

    # Load the weights
    # Convert checkpoint keys to PEFT model keys
    model_state = peft_model.state_dict()
    loaded_count = 0
    shape_mismatches = []

    for ckpt_key, tensor in adapter_state.items():
        # Build expected PEFT key
        peft_key = ckpt_key
        if ".lora_A" in peft_key and ".weight" not in peft_key:
            peft_key = peft_key.replace(".lora_A", ".lora_A.default.weight")
        if ".lora_B" in peft_key and ".weight" not in peft_key:
            peft_key = peft_key.replace(".lora_B", ".lora_B.default.weight")

        if peft_key in model_state:
            if model_state[peft_key].shape == tensor.shape:
                model_state[peft_key] = tensor.to(model_state[peft_key].dtype)
                loaded_count += 1
            else:
                shape_mismatches.append((ckpt_key, tensor.shape, model_state[peft_key].shape))

    if shape_mismatches:
        print(f"  [warn] Shape mismatches detected ({len(shape_mismatches)} tensors):")
        for ckpt_key, ckpt_shape, model_shape in shape_mismatches[:3]:
            print(f"         {ckpt_key}: checkpoint {ckpt_shape} vs model {model_shape}")
        if len(shape_mismatches) > 3:
            print(f"         ... and {len(shape_mismatches) - 3} more")
        print("  [warn] This usually means the checkpoint was created with a different model.")
        print("         The evaluation will use the base model without the organelle.")
        return base_model, None

    if loaded_count > 0:
        peft_model.load_state_dict(model_state, strict=False)
        print(f"  [info] Loaded {loaded_count} tensors from organelle {organelle_id}")
        return peft_model, organelle_id
    else:
        print("  [warn] No tensors loaded - key format may not match")
        return base_model, None


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
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = args.device if args.device != "auto" else "auto"
    dtype = torch.float16 if args.device == "mps" else torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    results = []

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
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
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
            # Load fresh base model to avoid state pollution
            evo_base = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            evo_model, organelle_id = load_evo_model(
                evo_base,
                args.evo_checkpoint,
                tokenizer,
                organelle_id=args.evo_organelle_id,
            )
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
