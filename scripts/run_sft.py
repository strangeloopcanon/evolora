#!/usr/bin/env python3
"""SFT baseline trainer with compute-matched token budget.

This script trains a LoRA adapter using standard supervised fine-tuning,
with a token budget matched to an evolved LoRA checkpoint for fair comparison.

Usage:
    # Match compute to an evolution checkpoint
    python scripts/run_sft.py \
        --checkpoint artifacts_xxx/checkpoint.pt \
        --data training_data.jsonl \
        --output sft_baseline

    # Or specify token budget directly
    python scripts/run_sft.py \
        --token-budget 100000 \
        --data training_data.jsonl \
        --output sft_baseline
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=".*PEFT.*")

BudgetKind = Literal["tokens", "forward_flops", "wall_clock_seconds"]


# ---------------------------------------------------------------------------
# Compute budget utilities
# ---------------------------------------------------------------------------


def load_compute_budget_from_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load compute budget from an evolution checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint_state = pickle.loads(checkpoint_path.read_bytes())
    compute = checkpoint_state.get("compute_budget")
    if compute is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain compute_budget. "
            "Was it created with an older version of run_evolution.py?"
        )
    return compute


@dataclass(frozen=True)
class SFTBudget:
    kind: BudgetKind
    value: float
    details: dict[str, object] | None = None

    def exhausted(self, *, tokens: int, forward_flops: float, wall_clock_seconds: float) -> bool:
        if self.kind == "tokens":
            return tokens >= int(self.value)
        if self.kind == "forward_flops":
            return forward_flops >= float(self.value)
        if self.kind == "wall_clock_seconds":
            return wall_clock_seconds >= float(self.value)
        raise ValueError(f"Unknown budget kind: {self.kind}")


def estimate_sft_budget_from_checkpoint(
    checkpoint_path: Path,
    *,
    match_budget_field: str = "total_tokens",
    backprop_multiplier: float = 2.0,
) -> SFTBudget:
    """Estimate an SFT compute budget from an evolution checkpoint.

    The evolution runner tracks *forward-only* token counts (prompt + generated) in
    `compute_budget.total_tokens` (and subset `train_tokens`).

    SFT tokens are more expensive than forward-only inference because they include
    backprop. We approximate this with a multiplier and convert the evolution
    compute budget into an SFT token budget:

        sft_token_budget ≈ evolution_tokens / backprop_multiplier
    """
    budget = load_compute_budget_from_checkpoint(checkpoint_path)
    field = str(match_budget_field)
    if field not in (
        "total_tokens",
        "train_tokens",
        "total_flops",
        "train_flops",
        "wall_clock_seconds",
    ):
        raise ValueError(f"Unsupported match_budget_field: {field}")

    raw = budget.get(field)
    if raw is None or (isinstance(raw, (int, float)) and float(raw) <= 0):
        raise ValueError(f"Checkpoint compute_budget.{field} is missing/zero; cannot match budget")

    multiplier = float(backprop_multiplier)
    if multiplier <= 0 and field != "wall_clock_seconds":
        raise ValueError("--backprop-multiplier must be positive")

    if field in ("total_tokens", "train_tokens"):
        evolution_tokens = int(raw)
        sft_tokens = max(1, int(evolution_tokens / multiplier))
        details: dict[str, object] = {
            "match_budget_field": field,
            "evolution_tokens": evolution_tokens,
            "backprop_multiplier": multiplier,
            "sft_token_budget": sft_tokens,
        }
        return SFTBudget(kind="tokens", value=float(sft_tokens), details=details)

    if field in ("total_flops", "train_flops"):
        evolution_flops = float(raw)
        sft_forward_flops = max(1.0, evolution_flops / multiplier)
        details = {
            "match_budget_field": field,
            "evolution_flops": evolution_flops,
            "backprop_multiplier": multiplier,
            "sft_forward_flops_budget": sft_forward_flops,
        }
        return SFTBudget(kind="forward_flops", value=float(sft_forward_flops), details=details)

    if field == "wall_clock_seconds":
        seconds = float(raw)
        details = {
            "match_budget_field": field,
            "evolution_wall_clock_seconds": seconds,
            "sft_wall_clock_budget_seconds": seconds,
        }
        return SFTBudget(kind="wall_clock_seconds", value=float(seconds), details=details)

    raise ValueError(f"Unhandled match_budget_field: {field}")


# ---------------------------------------------------------------------------
# Token budget stopping callback
# ---------------------------------------------------------------------------


@dataclass
class TokenBudgetState:
    """Tracks cumulative tokens processed during training."""

    total_tokens: int = 0
    budget: int = 0

    def add_tokens(self, count: int) -> None:
        self.total_tokens += count

    def budget_exhausted(self) -> bool:
        return self.budget > 0 and self.total_tokens >= self.budget


class TokenBudgetCallback(TrainerCallback):
    """Stops training when token budget is exhausted."""

    def __init__(self, state: TokenBudgetState):
        self.state = state

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        # Estimate tokens from batch size and sequence length
        # The actual count comes from the training loop
        if self.state.budget_exhausted():
            print(
                f"[sft] Token budget exhausted: {self.state.total_tokens:,} >= {self.state.budget:,}"
            )
            control.should_training_stop = True
        return control


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl_data(path: Path) -> list[dict[str, str]]:
    """Load training data from JSONL file.

    Expected format per line:
        {"prompt": "...", "completion": "..."}
    or:
        {"prompt": "...", "target": "..."}
    or:
        {"text": "..."}  (full sequence, no separation)
    """
    records: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[sft] Warning: skipping malformed line {line_num}: {e}")
                continue

            # Normalize to {"text": full_sequence}
            if "text" in obj:
                records.append({"text": obj["text"]})
            elif "prompt" in obj:
                completion = obj.get("completion") or obj.get("target") or ""
                # Format as prompt + completion for causal LM training
                text = f"{obj['prompt']}\n{completion}".strip()
                records.append({"text": text})
            else:
                print(f"[sft] Warning: skipping line {line_num}, no 'prompt' or 'text' field")
                continue

    return records


def prepare_dataset(
    data: list[dict[str, str]],
    tokenizer: AutoTokenizer,
    max_length: int,
    token_state: TokenBudgetState,
) -> Dataset:
    """Tokenize data and track total tokens."""
    dataset = Dataset.from_list(data)

    def tokenize_and_count(examples: dict[str, list[str]]) -> dict[str, list]:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        # Count tokens for budget tracking
        for ids in tokenized["input_ids"]:
            token_state.add_tokens(len(ids))
        return tokenized

    tokenized = dataset.map(
        tokenize_and_count,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    # Reset counter - we'll count again during actual training
    # The map count was just for preprocessing visibility
    token_state.total_tokens = 0

    return tokenized


# ---------------------------------------------------------------------------
# LoRA export
# ---------------------------------------------------------------------------


def export_lora_to_safetensors(model, output_path: Path) -> None:
    """Export LoRA adapter weights to safetensors format.

    Compatible with HostKernel.load_organelle_adapter().
    """
    from torch import nn

    state: dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue

        # PEFT stores adapters in dicts keyed by adapter name
        lora_a = module.lora_A
        lora_b = module.lora_B

        # Handle nn.ModuleDict (modern PEFT)
        if isinstance(lora_a, nn.ModuleDict):
            for adapter_name in lora_a.keys():
                if adapter_name in lora_b:
                    key_a = f"{name}.lora_A"
                    key_b = f"{name}.lora_B"
                    state[key_a] = lora_a[adapter_name].weight.detach().cpu().contiguous()
                    state[key_b] = lora_b[adapter_name].weight.detach().cpu().contiguous()
                    break  # Just export the first/default adapter
        elif isinstance(lora_a, dict):
            for adapter_name in lora_a.keys():
                if adapter_name in lora_b:
                    key_a = f"{name}.lora_A"
                    key_b = f"{name}.lora_B"
                    state[key_a] = lora_a[adapter_name].weight.detach().cpu().contiguous()
                    state[key_b] = lora_b[adapter_name].weight.detach().cpu().contiguous()
                    break  # Just export the first/default adapter
        elif hasattr(lora_a, "weight"):
            # Direct weight tensor
            state[f"{name}.lora_A"] = lora_a.weight.detach().cpu().contiguous()
            state[f"{name}.lora_B"] = lora_b.weight.detach().cpu().contiguous()

    if not state:
        # Fallback: use PEFT's built-in save method
        adapter_dir = output_path.parent / "adapter"
        model.save_pretrained(adapter_dir)
        print(f"[sft] Exported LoRA via PEFT to {adapter_dir}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output_path))
    print(f"[sft] Exported LoRA to {output_path} ({len(state)} tensors)")


# ---------------------------------------------------------------------------
# Custom Trainer to track tokens
# ---------------------------------------------------------------------------


class TokenTrackingTrainer(Trainer):
    """Trainer that tracks tokens processed for budget enforcement."""

    def __init__(self, token_state: TokenBudgetState, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_state = token_state

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Count non-padding tokens for fair comparison with evolution token counts.
        if "attention_mask" in inputs:
            try:
                batch_tokens = int(inputs["attention_mask"].sum().item())
            except Exception:
                batch_tokens = 0
            self.token_state.add_tokens(batch_tokens)
        elif "input_ids" in inputs:
            self.token_state.add_tokens(int(inputs["input_ids"].numel()))

        return super().training_step(model, inputs, num_items_in_batch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT baseline trainer with compute-matched token budget."
    )

    # Compute budget source (one required)
    budget_group = parser.add_mutually_exclusive_group(required=True)
    budget_group.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to evolution checkpoint.pt to match compute budget from.",
    )
    budget_group.add_argument(
        "--token-budget",
        type=int,
        help="Explicit token budget (counts non-padding tokens; alternative to --checkpoint).",
    )
    budget_group.add_argument(
        "--flops-budget",
        type=float,
        help="Explicit forward-flops budget (proxy: tokens * hidden_size * 2; alternative to --checkpoint).",
    )
    budget_group.add_argument(
        "--wall-clock-budget-seconds",
        type=float,
        help="Explicit wall-clock budget in seconds (alternative to --checkpoint).",
    )
    parser.add_argument(
        "--match-budget-field",
        type=str,
        choices=[
            "total_tokens",
            "train_tokens",
            "total_flops",
            "train_flops",
            "wall_clock_seconds",
        ],
        default="total_tokens",
        help=(
            "When using --checkpoint, which evolution compute_budget field to match before applying "
            "--backprop-multiplier (default: total_tokens)."
        ),
    )
    parser.add_argument(
        "--backprop-multiplier",
        type=float,
        default=2.0,
        help=(
            "Approximate relative compute of SFT per token vs forward-only inference "
            "(default: 2.0). SFT token budget = evolution_tokens / backprop_multiplier."
        ),
    )
    parser.add_argument(
        "--budget-scale",
        type=float,
        default=1.0,
        help="Scale the matched budget (default: 1.0). Useful for quick smoke runs (e.g., 0.1).",
    )

    # Data
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training data JSONL file.",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model to fine-tune (default: Qwen/Qwen3-0.6B).",
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
            "If you hit NaNs during MPS training, try --attn-implementation eager."
        ),
    )

    # LoRA config
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8, matching evolution default).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (default: 16 = 2 * rank).",
    )

    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01).",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for LR schedule (default: 0.1).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max sequence length (default: 256).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max epochs (will stop early if token budget exhausted).",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["auto", "trainer", "manual"],
        default="auto",
        help=(
            "Training engine. 'manual' adds non-finite guards and frequent checkpointing; "
            "'trainer' uses HF Trainer. 'auto' selects manual on MPS (default)."
        ),
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.3,
        help="Gradient clipping max norm (default: 0.3).",
    )
    parser.add_argument(
        "--adam-epsilon",
        type=float,
        default=1e-6,
        help="AdamW epsilon for numerical stability (default: 1e-6).",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        help="LR scheduler (default: cosine).",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=20,
        help="Log progress every N steps (default: 20).",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=200,
        help="Save a resumable training checkpoint every N steps (manual engine only; default: 200).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume manual training from output/sft_train_state.pt if present.",
    )
    parser.add_argument(
        "--max-consecutive-nonfinite",
        type=int,
        default=5,
        help="Stop training after this many consecutive non-finite steps (default: 5).",
    )
    parser.add_argument(
        "--max-nonfinite-restarts",
        type=int,
        default=3,
        help=(
            "Maximum automatic recoveries after repeated non-finite steps. "
            "A recovery restores last-good weights, clears optimizer state, and reduces LR (default: 3)."
        ),
    )
    parser.add_argument(
        "--nonfinite-lr-reduce-factor",
        type=float,
        default=0.5,
        help="On non-finite recovery, multiply LR by this factor (default: 0.5).",
    )
    parser.add_argument(
        "--nonfinite-lr-min",
        type=float,
        default=1e-6,
        help="Minimum LR after non-finite recoveries (default: 1e-6).",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sft_baseline"),
        help="Output directory for checkpoints and final LoRA.",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cuda, mps, cpu).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--optim",
        type=str,
        choices=["adamw_torch", "adamw_torch_fused"],
        default="adamw_torch",
        help=(
            "Optimizer implementation (default: adamw_torch). "
            "On MPS, the fused variant can be unstable for some models."
        ),
    )

    return parser.parse_args()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _count_nonpad_tokens(batch: dict[str, torch.Tensor], *, pad_token_id: int) -> int:
    if "attention_mask" in batch:
        return int(batch["attention_mask"].sum().item())
    if "input_ids" in batch:
        return int((batch["input_ids"] != int(pad_token_id)).sum().item())
    return 0


def _estimate_forward_flops(tokens: int, *, hidden_size: int) -> float:
    return float(int(tokens) * int(hidden_size) * 2)


def _atomic_torch_save(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _snapshot_trainable_params(model) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.detach().cpu().contiguous()
    return state


def _restore_trainable_params(model, state: dict[str, torch.Tensor], device: torch.device) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            tensor = state.get(name)
            if tensor is None or tensor.shape != param.shape:
                continue
            param.copy_(tensor.to(device=device, dtype=param.dtype))


def train_manual(
    *,
    model,
    tokenizer,
    dataset: Dataset,
    data_collator,
    budget: SFTBudget,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    """Manual training loop with non-finite guards and resumable checkpoints."""
    from transformers.optimization import get_scheduler

    model.to(device)
    model.train()
    try:
        model.config.use_cache = False
    except Exception:
        pass

    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        raise ValueError("Could not infer model hidden_size for flops accounting")

    # Deterministic dataset order across restarts: shuffle once, then iterate sequentially.
    dataset = dataset.shuffle(seed=int(args.seed))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found; LoRA did not attach correctly")

    use_fused = args.optim == "adamw_torch_fused"
    optimizer_kwargs: dict[str, object] = {
        "lr": float(args.learning_rate),
        "betas": (0.9, 0.999),
        "eps": float(args.adam_epsilon),
        "weight_decay": float(args.weight_decay),
    }
    try:
        optimizer = torch.optim.AdamW(trainable_params, fused=use_fused, **optimizer_kwargs)  # type: ignore[arg-type]
    except TypeError:
        optimizer = torch.optim.AdamW(trainable_params, **optimizer_kwargs)  # type: ignore[arg-type]

    steps_per_epoch = max(1, math.ceil(len(dataset) / int(args.batch_size)))
    max_steps = steps_per_epoch * int(args.epochs)
    warmup_steps = int(max_steps * float(args.warmup_ratio))
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=data_collator,
    )

    # Progress / resume state
    global_step = 0
    epoch = 0
    step_in_epoch = 0
    tokens_processed = 0
    forward_flops_processed = 0.0
    elapsed_prior = 0.0
    nonfinite_events = 0
    consecutive_nonfinite = 0
    nonfinite_restarts = 0
    stop_reason = "max_epochs"
    last_good = _snapshot_trainable_params(model)

    ckpt_path = args.output / "sft_train_state.pt"
    if bool(args.resume) and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        global_step = int(ckpt.get("global_step", 0) or 0)
        epoch = int(ckpt.get("epoch", 0) or 0)
        step_in_epoch = int(ckpt.get("step_in_epoch", 0) or 0)
        tokens_processed = int(ckpt.get("tokens_processed", 0) or 0)
        forward_flops_processed = float(ckpt.get("forward_flops_processed", 0.0) or 0.0)
        elapsed_prior = float(ckpt.get("elapsed_seconds", 0.0) or 0.0)
        nonfinite_restarts = int(ckpt.get("nonfinite_restarts", 0) or 0)

        saved_state = ckpt.get("trainable_state")
        if isinstance(saved_state, dict):
            _restore_trainable_params(model, saved_state, device)
            last_good = {k: v for k, v in saved_state.items() if isinstance(v, torch.Tensor)}
        opt_state = ckpt.get("optimizer_state")
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
            except Exception:
                pass
        sched_state = ckpt.get("scheduler_state")
        if sched_state is not None:
            try:
                scheduler.load_state_dict(sched_state)
            except Exception:
                pass
        py_state = ckpt.get("python_random_state")
        if py_state is not None:
            try:
                random.setstate(py_state)
            except Exception:
                pass
        torch_state = ckpt.get("torch_rng_state")
        if isinstance(torch_state, torch.Tensor):
            try:
                torch.set_rng_state(torch_state)
            except Exception:
                pass

        print(
            "[sft] Resumed from checkpoint: "
            f"epoch={epoch} step_in_epoch={step_in_epoch} global_step={global_step} "
            f"tokens={tokens_processed:,} forward_flops={forward_flops_processed:,.0f} "
            f"nonfinite_restarts={nonfinite_restarts}"
        )

    start_time = time.perf_counter() - elapsed_prior

    def _scale_lr(factor: float) -> float:
        min_lr = float(args.nonfinite_lr_min)
        new_lrs: list[float] = []
        for group in optimizer.param_groups:
            current = float(group.get("lr", 0.0))
            updated = max(min_lr, current * float(factor))
            group["lr"] = updated
            new_lrs.append(updated)
        if hasattr(scheduler, "base_lrs"):
            try:
                scheduler.base_lrs = list(new_lrs)  # type: ignore[attr-defined]
            except Exception:
                pass
        return float(new_lrs[0]) if new_lrs else 0.0

    def _recover(reason: str) -> bool:
        nonlocal consecutive_nonfinite, nonfinite_restarts, stop_reason
        if nonfinite_restarts >= int(args.max_nonfinite_restarts):
            stop_reason = reason
            return False
        nonfinite_restarts += 1
        _restore_trainable_params(model, last_good, device)
        try:
            optimizer.state.clear()
        except Exception:
            pass
        new_lr = _scale_lr(float(args.nonfinite_lr_reduce_factor))
        consecutive_nonfinite = 0
        print(
            f"[sft] nonfinite_recovery={nonfinite_restarts}/{int(args.max_nonfinite_restarts)} "
            f"reason={reason} new_lr={new_lr:.3g}"
        )
        return True

    for epoch_idx in range(epoch, int(args.epochs)):
        epoch = epoch_idx
        step_in_epoch_local = 0
        loader_iter = iter(train_loader)
        if epoch_idx == epoch and step_in_epoch > 0:
            for _ in range(step_in_epoch):
                next(loader_iter, None)
            step_in_epoch_local = step_in_epoch
        for batch in loader_iter:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            wall = time.perf_counter() - start_time
            if budget.exhausted(
                tokens=tokens_processed,
                forward_flops=forward_flops_processed,
                wall_clock_seconds=wall,
            ):
                stop_reason = "budget_exhausted"
                break

            global_step += 1
            step_in_epoch_local += 1
            step_in_epoch = step_in_epoch_local

            batch = {k: v.to(device) for k, v in batch.items()}
            batch_tokens = _count_nonpad_tokens(batch, pad_token_id=int(tokenizer.pad_token_id))
            tokens_processed += batch_tokens
            forward_flops_processed += _estimate_forward_flops(
                batch_tokens, hidden_size=hidden_size
            )

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss).all():
                nonfinite_events += 1
                consecutive_nonfinite += 1
                if consecutive_nonfinite >= int(args.max_consecutive_nonfinite):
                    if not _recover("nonfinite_loss"):
                        break
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float(args.max_grad_norm))
            if not math.isfinite(float(grad_norm)):
                nonfinite_events += 1
                consecutive_nonfinite += 1
                if consecutive_nonfinite >= int(args.max_consecutive_nonfinite):
                    if not _recover("nonfinite_grad"):
                        break
                continue

            optimizer.step()
            scheduler.step()
            consecutive_nonfinite = 0

            # Guard against parameters becoming NaN/Inf; restore last-good snapshot.
            params_ok = True
            for p in trainable_params:
                if not torch.isfinite(p).all():
                    params_ok = False
                    break
            if not params_ok:
                nonfinite_events += 1
                consecutive_nonfinite += 1
                # Parameter corruption is severe: restore immediately (and count as a recovery).
                if not _recover("nonfinite_params"):
                    break
                continue

            last_good = _snapshot_trainable_params(model)

            if int(args.log_every_steps) > 0 and global_step % int(args.log_every_steps) == 0:
                lr = float(optimizer.param_groups[0].get("lr", 0.0))
                print(
                    f"[sft] step={global_step} epoch={epoch_idx} "
                    f"loss={float(loss.detach().item()):.4f} grad_norm={float(grad_norm):.4f} "
                    f"lr={lr:.3g} tokens={tokens_processed:,} flops={forward_flops_processed:,.0f}"
                )

            if int(args.save_every_steps) > 0 and global_step % int(args.save_every_steps) == 0:
                wall = time.perf_counter() - start_time
                _atomic_torch_save(
                    {
                        "global_step": global_step,
                        "epoch": epoch_idx,
                        "step_in_epoch": step_in_epoch_local,
                        "tokens_processed": tokens_processed,
                        "forward_flops_processed": forward_flops_processed,
                        "elapsed_seconds": wall,
                        "trainable_state": last_good,
                        "nonfinite_restarts": nonfinite_restarts,
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "python_random_state": random.getstate(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                    ckpt_path,
                )

        if stop_reason != "max_epochs":
            break
        step_in_epoch = 0

    wall = time.perf_counter() - start_time
    # Always save a final resumable checkpoint (cheap for LoRA).
    _atomic_torch_save(
        {
            "global_step": global_step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "tokens_processed": tokens_processed,
            "forward_flops_processed": forward_flops_processed,
            "elapsed_seconds": wall,
            "trainable_state": last_good,
            "nonfinite_restarts": nonfinite_restarts,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        },
        ckpt_path,
    )
    return {
        "stop_reason": stop_reason,
        "global_step": global_step,
        "epoch": epoch,
        "tokens_processed": tokens_processed,
        "forward_flops_processed": forward_flops_processed,
        "wall_clock_seconds": wall,
        "nonfinite_events": nonfinite_events,
        "nonfinite_restarts": nonfinite_restarts,
        "checkpoint_path": str(ckpt_path),
    }


def main() -> None:
    args = parse_args()

    # Determine token budget
    budget_details: dict[str, object] | None = None
    if args.checkpoint:
        budget = estimate_sft_budget_from_checkpoint(
            args.checkpoint,
            match_budget_field=args.match_budget_field,
            backprop_multiplier=args.backprop_multiplier,
        )
        if budget.details is not None:
            budget_details = dict(budget.details)
        scale = float(args.budget_scale)
        if scale <= 0:
            raise ValueError("--budget-scale must be positive")
        if scale != 1.0:
            budget = SFTBudget(
                kind=budget.kind,
                value=float(budget.value) * scale,
                details=budget.details,
            )
            if budget_details is not None:
                budget_details["budget_scale"] = scale

        if budget.kind == "tokens" and budget_details is not None:
            print(
                "[sft] Loaded evolution compute budget: "
                f"{int(budget_details['evolution_tokens']):,} {budget_details['match_budget_field']} tokens"
            )
            print(
                f"[sft] Backprop multiplier: {float(budget_details['backprop_multiplier']):.2f} "
                f"→ SFT token budget: {int(budget.value):,} tokens"
            )
        elif budget.kind == "forward_flops" and budget_details is not None:
            print(
                "[sft] Loaded evolution compute budget: "
                f"{float(budget_details['evolution_flops']):,.0f} {budget_details['match_budget_field']}"
            )
            print(
                f"[sft] Backprop multiplier: {float(budget_details['backprop_multiplier']):.2f} "
                f"→ SFT forward-flops budget: {float(budget.value):,.0f}"
            )
        elif budget.kind == "wall_clock_seconds" and budget_details is not None:
            evo_seconds = float(budget_details.get("evolution_wall_clock_seconds", 0.0))
            print(
                f"[sft] Matching wall-clock budget: {float(budget.value):,.2f}s "
                f"(evolution: {evo_seconds:,.2f}s)"
            )
    else:
        if args.token_budget is not None:
            budget = SFTBudget(kind="tokens", value=float(args.token_budget))
            print(f"[sft] Using explicit token budget: {int(budget.value):,} tokens")
        elif args.flops_budget is not None:
            budget = SFTBudget(kind="forward_flops", value=float(args.flops_budget))
            print(f"[sft] Using explicit forward-flops budget: {float(budget.value):,.0f}")
        elif args.wall_clock_budget_seconds is not None:
            budget = SFTBudget(
                kind="wall_clock_seconds", value=float(args.wall_clock_budget_seconds)
            )
            print(f"[sft] Using explicit wall-clock budget: {float(budget.value):,.2f}s")
        else:
            raise ValueError("A budget source is required")

    if budget.value <= 0:
        raise ValueError("Budget must be positive")

    # Load training data
    if not args.data.exists():
        raise FileNotFoundError(f"Training data not found: {args.data}")

    print(f"[sft] Loading training data from {args.data}")
    raw_data = load_jsonl_data(args.data)
    print(f"[sft] Loaded {len(raw_data)} training examples")

    if not raw_data:
        raise ValueError("No valid training examples found")

    # Initialize token tracking
    token_state = TokenBudgetState(budget=int(budget.value) if budget.kind == "tokens" else 0)

    # Load tokenizer and model
    print(f"[sft] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=bool(args.trust_remote_code)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # Use float32 for stability on CPU/MPS
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=args.attn_implementation,
    )

    # Apply LoRA - same target modules as evolution
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    print(f"[sft] Tokenizing dataset (max_length={args.max_length})")
    dataset = prepare_dataset(raw_data, tokenizer, args.max_length, token_state)

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Output directory
    args.output.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    engine = str(args.engine)
    if engine == "auto":
        engine = "manual" if device.type == "mps" else "trainer"
    if budget.kind != "tokens" and engine != "manual":
        raise ValueError("Non-token budgets require --engine manual")

    training_summary: dict[str, object] = {}
    if engine == "manual":
        print(f"[sft] Starting manual training on {device} (budget kind={budget.kind})")
        training_summary = train_manual(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            data_collator=data_collator,
            budget=budget,
            args=args,
            device=device,
        )
        token_state.total_tokens = int(training_summary.get("tokens_processed", 0) or 0)
        print(
            "[sft] Training complete "
            f"(stop_reason={training_summary.get('stop_reason')}, "
            f"tokens={token_state.total_tokens:,})"
        )
    else:
        # Training arguments (HF Trainer)
        training_args = TrainingArguments(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optim=args.optim,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.max_grad_norm,
            adam_epsilon=args.adam_epsilon,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=max(1, int(args.log_every_steps)),
            save_strategy="epoch",
            save_total_limit=2,
            seed=args.seed,
            report_to="none",  # Disable wandb etc.
            remove_unused_columns=False,
            # Disable evaluation during training
            eval_strategy="no",
        )

        # Token budget callback
        budget_callback = TokenBudgetCallback(token_state)

        # Create trainer
        trainer = TokenTrackingTrainer(
            token_state=token_state,
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=[budget_callback],
        )

        # Train
        print(f"[sft] Starting Trainer training (budget: {int(budget.value):,} tokens)")
        trainer.train()
        print("[sft] Training complete")
        print(f"[sft] Tokens processed: {token_state.total_tokens:,} / {int(budget.value):,}")

    # Export LoRA weights in safetensors format (compatible with evolution loader)
    lora_output = args.output / "lora_adapter.safetensors"
    export_lora_to_safetensors(model, lora_output)

    # Also save via PEFT's native format for HF compatibility
    peft_output = args.output / "peft_adapter"
    model.save_pretrained(str(peft_output))
    print(f"[sft] Saved PEFT adapter to {peft_output}")

    # Save training metadata
    metadata = {
        "model": args.model,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "budget_kind": budget.kind,
        "budget_value": budget.value,
        "tokens_processed": token_state.total_tokens,
        "training_examples": len(raw_data),
        "source_checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "budget_details": budget_details,
        "training_summary": training_summary,
        "engine": engine,
        "training_hparams": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "max_grad_norm": args.max_grad_norm,
            "adam_epsilon": args.adam_epsilon,
            "lr_scheduler_type": args.lr_scheduler_type,
            "optim": args.optim,
        },
    }
    metadata_path = args.output / "sft_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[sft] Saved metadata to {metadata_path}")

    print(f"[sft] Done. Output directory: {args.output}")


if __name__ == "__main__":
    main()
