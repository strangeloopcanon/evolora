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
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
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


# ---------------------------------------------------------------------------
# Compute budget utilities
# ---------------------------------------------------------------------------


def load_compute_budget_from_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load compute budget from an evolution checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = pickle.loads(checkpoint_path.read_bytes())
    compute = state.get("compute_budget")
    if compute is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain compute_budget. "
            "Was it created with an older version of run_evolution.py?"
        )
    return compute


def estimate_tokens_from_checkpoint(checkpoint_path: Path) -> int:
    """Extract total_tokens from a checkpoint's compute budget."""
    budget = load_compute_budget_from_checkpoint(checkpoint_path)
    return int(budget.get("total_tokens", 0))


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
    records = []
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
    state: dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue

        # PEFT stores adapters in dicts keyed by adapter name
        lora_a = module.lora_A
        lora_b = module.lora_B

        if isinstance(lora_a, dict):
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
        raise ValueError("No LoRA weights found to export")

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
        # Count tokens in this batch
        if "input_ids" in inputs:
            batch_tokens = inputs["input_ids"].numel()
            self.token_state.add_tokens(batch_tokens)

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
        help="Explicit token budget (alternative to --checkpoint).",
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine token budget
    if args.checkpoint:
        token_budget = estimate_tokens_from_checkpoint(args.checkpoint)
        print(f"[sft] Loaded token budget from checkpoint: {token_budget:,} tokens")
    else:
        token_budget = args.token_budget
        print(f"[sft] Using explicit token budget: {token_budget:,} tokens")

    if token_budget <= 0:
        raise ValueError("Token budget must be positive")

    # Load training data
    if not args.data.exists():
        raise FileNotFoundError(f"Training data not found: {args.data}")

    print(f"[sft] Loading training data from {args.data}")
    raw_data = load_jsonl_data(args.data)
    print(f"[sft] Loaded {len(raw_data)} training examples")

    if not raw_data:
        raise ValueError("No valid training examples found")

    # Initialize token tracking
    token_state = TokenBudgetState(budget=token_budget)

    # Load tokenizer and model
    print(f"[sft] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # Use float32 for stability on CPU/MPS
        trust_remote_code=True,
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

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
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
    print(f"[sft] Starting training (budget: {token_budget:,} tokens)")
    trainer.train()

    # Report final stats
    print("[sft] Training complete")
    print(f"[sft] Tokens processed: {token_state.total_tokens:,} / {token_budget:,}")

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
        "token_budget": token_budget,
        "tokens_processed": token_state.total_tokens,
        "training_examples": len(raw_data),
        "source_checkpoint": str(args.checkpoint) if args.checkpoint else None,
    }
    metadata_path = args.output / "sft_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[sft] Saved metadata to {metadata_path}")

    print(f"[sft] Done. Output directory: {args.output}")


if __name__ == "__main__":
    main()
