#!/usr/bin/env python3
"""Evaluate multi-task transfer performance to quantify robustness premium.

This script evaluates how well models trained on a source task family transfer
to target task families. It compares evolution vs SFT approaches to quantify
the "robustness premium" of evolutionary adaptation.

Usage:
    # Evaluate transfer from regex to other families
    python scripts/evaluate_transfer.py \
        --source-family regex \
        --target-families math.multi_step code.format logic.bool \
        --holdout config/evaluation/holdout_grid_multiobjective.jsonl \
        --evo-checkpoint artifacts/evo/checkpoint.pt \
        --sft-adapter artifacts/sft/peft_adapter

    # With sampling and verbose output
    python scripts/evaluate_transfer.py \
        --source-family regex \
        --target-families math.multi_step code.format \
        --holdout config/evaluation/holdout_grid_multiobjective.jsonl \
        --evo-checkpoint artifacts/evo/checkpoint.pt \
        --samples-per-family 50 \
        --verbose

Transfer Metrics:
    - Source accuracy: In-distribution performance (training family)
    - Target accuracy: Out-of-distribution performance (each target family)
    - Transfer ratio: target_acc / source_acc (measures retention)
    - Transfer gap: source_acc - target_acc (measures degradation)
    - Geometric mean: Balanced accuracy across all families
    - Robustness premium: (evo_transfer - sft_transfer) for each metric
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
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
class FamilyResult:
    """Results for a single task family."""

    family: str
    correct: int = 0
    total: int = 0
    is_source: bool = False

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


@dataclass
class TransferResult:
    """Complete transfer evaluation results for a model."""

    model_name: str
    source_family: str
    family_results: dict[str, FamilyResult] = field(default_factory=dict)

    @property
    def source_accuracy(self) -> float:
        """Accuracy on the source (training) family."""
        if self.source_family in self.family_results:
            return self.family_results[self.source_family].accuracy
        return 0.0

    @property
    def target_accuracies(self) -> dict[str, float]:
        """Accuracy on each target family."""
        return {k: v.accuracy for k, v in self.family_results.items() if not v.is_source}

    @property
    def mean_target_accuracy(self) -> float:
        """Arithmetic mean of target family accuracies."""
        accs = list(self.target_accuracies.values())
        return sum(accs) / len(accs) if accs else 0.0

    @property
    def geometric_mean_accuracy(self) -> float:
        """Geometric mean across all families (source + targets)."""
        accs = [r.accuracy for r in self.family_results.values()]
        if not accs:
            return 0.0
        eps = 1e-6
        return math.exp(sum(math.log(max(a, eps)) for a in accs) / len(accs))

    @property
    def transfer_ratios(self) -> dict[str, float]:
        """Transfer ratio for each target family: target_acc / source_acc."""
        src_acc = self.source_accuracy
        if src_acc <= 0:
            return {k: 0.0 for k in self.target_accuracies}
        return {k: v / src_acc for k, v in self.target_accuracies.items()}

    @property
    def mean_transfer_ratio(self) -> float:
        """Mean transfer ratio across target families."""
        ratios = list(self.transfer_ratios.values())
        return sum(ratios) / len(ratios) if ratios else 0.0

    @property
    def transfer_gaps(self) -> dict[str, float]:
        """Transfer gap for each target family: source_acc - target_acc."""
        src_acc = self.source_accuracy
        return {k: src_acc - v for k, v in self.target_accuracies.items()}

    @property
    def mean_transfer_gap(self) -> float:
        """Mean transfer gap across target families."""
        gaps = list(self.transfer_gaps.values())
        return sum(gaps) / len(gaps) if gaps else 0.0

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of all transfer metrics."""
        family_stats = {
            k: {
                "accuracy": r.accuracy,
                "correct": r.correct,
                "total": r.total,
                "is_source": r.is_source,
            }
            for k, r in self.family_results.items()
        }
        return {
            "model": self.model_name,
            "source_family": self.source_family,
            "source_accuracy": self.source_accuracy,
            "mean_target_accuracy": self.mean_target_accuracy,
            "geometric_mean_accuracy": self.geometric_mean_accuracy,
            "mean_transfer_ratio": self.mean_transfer_ratio,
            "mean_transfer_gap": self.mean_transfer_gap,
            "target_accuracies": self.target_accuracies,
            "transfer_ratios": self.transfer_ratios,
            "transfer_gaps": self.transfer_gaps,
            "family_results": family_stats,
        }


def compute_robustness_premium(
    evo_result: TransferResult, sft_result: TransferResult
) -> dict[str, Any]:
    """Compute the robustness premium of evolution over SFT.

    Robustness premium measures how much better evolution transfers to new
    task families compared to SFT. Positive values indicate evolution is
    more robust; negative values indicate SFT is more robust.
    """
    premium: dict[str, Any] = {
        "source_accuracy_delta": evo_result.source_accuracy - sft_result.source_accuracy,
        "mean_target_accuracy_delta": (
            evo_result.mean_target_accuracy - sft_result.mean_target_accuracy
        ),
        "geometric_mean_delta": (
            evo_result.geometric_mean_accuracy - sft_result.geometric_mean_accuracy
        ),
        "mean_transfer_ratio_delta": (
            evo_result.mean_transfer_ratio - sft_result.mean_transfer_ratio
        ),
        "per_family_deltas": {},
    }

    # Per-family accuracy deltas
    all_families = set(evo_result.family_results.keys()) | set(sft_result.family_results.keys())
    for family in all_families:
        evo_acc = evo_result.family_results.get(family, FamilyResult(family)).accuracy
        sft_acc = sft_result.family_results.get(family, FamilyResult(family)).accuracy
        premium["per_family_deltas"][family] = evo_acc - sft_acc

    # Summary interpretation
    if premium["mean_transfer_ratio_delta"] > 0.05:
        premium["interpretation"] = "evolution_more_robust"
    elif premium["mean_transfer_ratio_delta"] < -0.05:
        premium["interpretation"] = "sft_more_robust"
    else:
        premium["interpretation"] = "roughly_equivalent"

    return premium


def load_holdout_tasks(
    path: str | Path,
    source_family: str,
    target_families: list[str],
    samples_per_family: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load holdout tasks, filtering to source and target families."""
    import random

    path = Path(path)
    tasks: list[dict[str, Any]] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            family = task.get("family", "")
            # Match exact family or family prefix (e.g., "regex" matches "regex.synthesis")
            if family == source_family or family.startswith(f"{source_family}."):
                task["_is_source"] = True
                tasks.append(task)
            elif any(family == t or family.startswith(f"{t}.") for t in target_families):
                task["_is_source"] = False
                tasks.append(task)

    if samples_per_family:
        # Stratified sampling by family
        random.seed(seed)
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for t in tasks:
            by_family[t.get("family", "unknown")].append(t)
        sampled: list[dict[str, Any]] = []
        for _fam, fam_tasks in by_family.items():
            if len(fam_tasks) <= samples_per_family:
                sampled.extend(fam_tasks)
            else:
                sampled.extend(random.sample(fam_tasks, samples_per_family))
        tasks = sampled

    return tasks


def evaluate_model_transfer(
    model: Any,
    tokenizer: Any,
    tasks: list[dict[str, Any]],
    source_family: str,
    model_name: str,
    verbose: bool = False,
) -> TransferResult:
    """Evaluate a model's transfer performance on holdout tasks."""
    from symbiont_ecology.environment.grid import GridTask

    result = TransferResult(model_name=model_name, source_family=source_family)

    for idx, task in enumerate(tasks):
        family = task.get("family", "unknown")
        is_source = task.get("_is_source", False)

        # Initialize family result if needed
        if family not in result.family_results:
            result.family_results[family] = FamilyResult(family=family, is_source=is_source)

        # Generate response
        prompt = task.get("prompt", "")
        target = task.get("target")

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            # Create task object for scoring
            depth = str(task.get("depth", "short") or "short")
            grid_task = GridTask(
                task_id=str(task.get("task_id") or f"transfer_{idx}"),
                cell=(str(family), depth),
                prompt=prompt,
                price=0.0,
                target=target,
                family=str(family),
                depth=depth,
                difficulty=float(task.get("difficulty", 0.0) or 0.0),
            )
            success, _reward = grid_task.evaluate(response)

            result.family_results[family].total += 1
            if success:
                result.family_results[family].correct += 1

            if verbose:
                status = "PASS" if success else "FAIL"
                print(f"[{status}] {family}: {prompt[:60]}...")

        except Exception as e:
            if verbose:
                print(f"[ERROR] {family}: {e}")
            result.family_results[family].total += 1

    return result


def load_evolution_model(
    checkpoint_path: str | Path,
    model_id: str,
    device: str = "auto",
    *,
    allow_unsafe_pickle: bool = False,
) -> tuple[Any, Any]:
    """Load a model with evolution checkpoint adapters."""
    from peft import LoraConfig, get_peft_model

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load checkpoint
    checkpoint = load_checkpoint(Path(checkpoint_path), allow_unsafe_pickle=allow_unsafe_pickle)

    # Get adapter states
    adapter_states = checkpoint.get("adapter_states", {})
    if not adapter_states:
        print("Warning: No adapter states found in checkpoint")
        return model, tokenizer

    # Apply first adapter (or could implement routing)
    first_adapter_id = next(iter(adapter_states.keys()))
    state = adapter_states[first_adapter_id]

    # Infer rank from state
    rank = 8
    for k, v in state.items():
        if "lora_A" in k and hasattr(v, "shape"):
            rank = v.shape[0]
            break

    # Create and apply PEFT model
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=list(_LORA_TARGET_MODULES),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)  # type: ignore[assignment]

    # Load weights
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: Missing keys when loading adapter: {len(missing)}")

    return model, tokenizer


def load_sft_model(
    adapter_path: str | Path,
    model_id: str,
    device: str = "auto",
) -> tuple[Any, Any]:
    """Load a model with SFT adapter."""
    from peft import PeftModel

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load PEFT adapter
    model = PeftModel.from_pretrained(model, adapter_path)  # type: ignore[assignment]

    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-task transfer performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source-family",
        required=True,
        help="Source (training) task family, e.g., regex",
    )
    parser.add_argument(
        "--target-families",
        nargs="+",
        required=True,
        help="Target task families to evaluate transfer, e.g., math.multi_step code.format",
    )
    parser.add_argument(
        "--holdout",
        required=True,
        help="Path to holdout tasks JSONL file",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B",
        help="Base model ID (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--evo-checkpoint",
        help="Path to evolution checkpoint (.pt file)",
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
        "--sft-adapter",
        help="Path to SFT PEFT adapter directory",
    )
    parser.add_argument(
        "--samples-per-family",
        type=int,
        help="Maximum samples per family (stratified sampling)",
    )
    parser.add_argument(
        "--output",
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-task results",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)",
    )

    args = parser.parse_args()

    # Load tasks
    print(f"Loading holdout tasks from {args.holdout}...")
    tasks = load_holdout_tasks(
        args.holdout,
        args.source_family,
        args.target_families,
        args.samples_per_family,
    )
    print(f"Loaded {len(tasks)} tasks")

    # Count by family
    family_counts: dict[str, int] = defaultdict(int)
    for t in tasks:
        family_counts[t.get("family", "unknown")] += 1
    print("Tasks by family:")
    for fam, count in sorted(family_counts.items()):
        src = (
            "(source)"
            if any(t.get("_is_source") for t in tasks if t.get("family") == fam)
            else "(target)"
        )
        print(f"  {fam}: {count} {src}")

    results: dict[str, TransferResult] = {}

    # Evaluate evolution checkpoint
    if args.evo_checkpoint:
        print(f"\nLoading evolution checkpoint from {args.evo_checkpoint}...")
        model, tokenizer = load_evolution_model(
            args.evo_checkpoint,
            args.model,
            args.device,
            allow_unsafe_pickle=args.allow_unsafe_pickle,
        )
        print("Evaluating evolution model...")
        evo_result = evaluate_model_transfer(
            model, tokenizer, tasks, args.source_family, "evolution", args.verbose
        )
        results["evolution"] = evo_result
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Evaluate SFT adapter
    if args.sft_adapter:
        print(f"\nLoading SFT adapter from {args.sft_adapter}...")
        model, tokenizer = load_sft_model(args.sft_adapter, args.model, args.device)
        print("Evaluating SFT model...")
        sft_result = evaluate_model_transfer(
            model, tokenizer, tasks, args.source_family, "sft", args.verbose
        )
        results["sft"] = sft_result
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print results
    print("\n" + "=" * 60)
    print("TRANSFER EVALUATION RESULTS")
    print("=" * 60)

    for name, result in results.items():
        print(f"\n{name.upper()} Model:")
        print(f"  Source family ({args.source_family}): {result.source_accuracy:.1%}")
        print(f"  Mean target accuracy: {result.mean_target_accuracy:.1%}")
        print(f"  Geometric mean (all): {result.geometric_mean_accuracy:.1%}")
        print(f"  Mean transfer ratio: {result.mean_transfer_ratio:.2f}")
        print(f"  Mean transfer gap: {result.mean_transfer_gap:.1%}")
        print("\n  Per-family breakdown:")
        for fam, fam_result in sorted(result.family_results.items()):
            src_tag = " (source)" if fam_result.is_source else ""
            print(
                f"    {fam}{src_tag}: {fam_result.accuracy:.1%} "
                f"({fam_result.correct}/{fam_result.total})"
            )

    # Compute robustness premium if both models evaluated
    if "evolution" in results and "sft" in results:
        print("\n" + "-" * 60)
        print("ROBUSTNESS PREMIUM (Evolution - SFT)")
        print("-" * 60)
        premium = compute_robustness_premium(results["evolution"], results["sft"])
        print(f"  Source accuracy delta: {premium['source_accuracy_delta']:+.1%}")
        print(f"  Mean target accuracy delta: {premium['mean_target_accuracy_delta']:+.1%}")
        print(f"  Geometric mean delta: {premium['geometric_mean_delta']:+.1%}")
        print(f"  Mean transfer ratio delta: {premium['mean_transfer_ratio_delta']:+.2f}")
        print(f"  Interpretation: {premium['interpretation']}")
        print("\n  Per-family accuracy deltas:")
        for fam, delta in sorted(premium["per_family_deltas"].items()):
            print(f"    {fam}: {delta:+.1%}")

    # Save results
    if args.output:
        output = {
            "source_family": args.source_family,
            "target_families": args.target_families,
            "results": {name: r.summary() for name, r in results.items()},
        }
        if "evolution" in results and "sft" in results:
            output["robustness_premium"] = compute_robustness_premium(
                results["evolution"], results["sft"]
            )
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
