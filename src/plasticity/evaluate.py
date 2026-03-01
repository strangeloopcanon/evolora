"""Holdout evaluation under different mask conditions.

Applies masks by zeroing model weights (with backup/restore) and runs
inference on holdout tasks to measure per-family accuracy or perplexity.
"""

from __future__ import annotations

import json
import logging
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy

from plasticity.masks import MaskSet
from plasticity.tasks import Task

logger = logging.getLogger(__name__)


@contextmanager
def masked_weights(model: nn.Module, mask: MaskSet) -> Generator[None, None, None]:
    """Context manager that zeros masked weights and restores them on exit."""
    backups: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if name in mask.masks and isinstance(module, nn.Linear):
            backups[name] = module.weight.data.clone()
            module.weight.data.mul_(mask.masks[name].to(module.weight.device))
    try:
        yield
    finally:
        for name, module in model.named_modules():
            if name in backups and isinstance(module, nn.Linear):
                module.weight.data.copy_(backups[name])


def run_inference(
    model: nn.Module,
    tokenizer: Any,
    prompts: List[str],
    *,
    max_new_tokens: int = 64,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> List[str]:
    """Generate responses for a list of prompts."""
    model.eval()
    target_device = device or next(model.parameters()).device
    answers: List[str] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        encoded = {k: v.to(target_device) for k, v in encoded.items()}
        prompt_len = encoded["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        for seq in outputs:
            generated = seq[prompt_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            answers.append(text)

    return answers


def evaluate_tasks(
    model: nn.Module,
    tokenizer: Any,
    tasks: List[Task],
    *,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate a list of tasks and return per-family accuracy."""
    prompts = [t.prompt for t in tasks]
    answers = run_inference(model, tokenizer, prompts, batch_size=batch_size, device=device)

    family_correct: Dict[str, int] = {}
    family_total: Dict[str, int] = {}
    results: List[Dict[str, Any]] = []

    for task, answer in zip(tasks, answers):
        correct = task.evaluate(answer)
        family_correct.setdefault(task.family, 0)
        family_total.setdefault(task.family, 0)
        family_total[task.family] += 1
        if correct:
            family_correct[task.family] += 1
        results.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "correct": correct,
                "answer": answer[:200],
            }
        )

    accuracy: Dict[str, float] = {
        fam: family_correct.get(fam, 0) / family_total[fam] for fam in family_total
    }
    overall = sum(family_correct.values()) / max(sum(family_total.values()), 1)

    return {
        "accuracy_per_family": accuracy,
        "accuracy_overall": overall,
        "n_per_family": family_total,
        "results": results,
    }


def compute_perplexity(
    model: nn.Module,
    tokenizer: Any,
    tasks: List[Task],
    *,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Compute cross-entropy loss on target tokens only.

    For each task, tokenises ``prompt + " " + str(target)`` as one sequence,
    identifies the boundary where the prompt ends, and computes CE loss
    only on the target portion.  Returns per-family mean perplexity.
    """
    model.eval()
    target_device = device or next(model.parameters()).device

    family_losses: Dict[str, List[float]] = {}

    for start in range(0, len(tasks), batch_size):
        batch_tasks = tasks[start : start + batch_size]

        prompts = [t.prompt for t in batch_tasks]
        full_texts = [f"{t.prompt} {t.target}" for t in batch_tasks]

        prompt_enc = tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=256,
        )
        prompt_lengths = [len(ids) for ids in prompt_enc["input_ids"]]

        full_enc = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=320,
        )
        input_ids = full_enc["input_ids"].to(target_device)
        attention_mask = full_enc["attention_mask"].to(target_device)

        labels = input_ids.clone()
        for bi, plen in enumerate(prompt_lengths):
            labels[bi, :plen] = -100
        labels[attention_mask == 0] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # (B, T, V)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        for bi, task in enumerate(batch_tasks):
            mask_1d = shift_labels[bi] != -100
            if mask_1d.sum() == 0:
                continue
            token_logits = shift_logits[bi][mask_1d]
            token_labels = shift_labels[bi][mask_1d]
            loss = cross_entropy(token_logits, token_labels).item()
            family_losses.setdefault(task.family, []).append(loss)

    ppl_per_family: Dict[str, float] = {}
    for fam, losses in family_losses.items():
        mean_loss = sum(losses) / len(losses)
        ppl_per_family[fam] = math.exp(min(mean_loss, 100.0))

    all_losses = [v for losses in family_losses.values() for v in losses]
    overall_loss = sum(all_losses) / max(len(all_losses), 1)

    return {
        "perplexity_per_family": ppl_per_family,
        "perplexity_overall": math.exp(min(overall_loss, 100.0)),
        "loss_per_family": {
            fam: sum(losses) / len(losses) for fam, losses in family_losses.items()
        },
        "loss_overall": overall_loss,
        "n_per_family": {fam: len(losses) for fam, losses in family_losses.items()},
    }


def run_condition(
    condition: str,
    model: nn.Module,
    tokenizer: Any,
    tasks: List[Task],
    *,
    mask: Optional[MaskSet] = None,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    eval_mode: str = "accuracy",
) -> Dict[str, Any]:
    """Run evaluation under a single condition (dense or masked).

    *eval_mode* is ``"accuracy"`` (generate + exact-match) or
    ``"perplexity"`` (CE loss on reference targets).
    """
    logger.info("evaluating_condition", extra={"condition": condition})

    eval_fn = evaluate_tasks if eval_mode == "accuracy" else compute_perplexity

    if condition == "dense" or mask is None:
        result = eval_fn(model, tokenizer, tasks, batch_size=batch_size, device=device)
    else:
        with masked_weights(model, mask):
            result = eval_fn(model, tokenizer, tasks, batch_size=batch_size, device=device)

    result["condition"] = condition
    if mask is not None and condition != "dense":
        result["sparsity"] = mask.sparsity()
    return result


def save_results(results: List[Dict[str, Any]], path: str | Path) -> None:
    """Save evaluation results to a JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(obj: Any) -> Any:
        if isinstance(obj, (torch.Tensor,)):
            return obj.tolist()
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=_serialize)


def load_results(path: str | Path) -> List[Dict[str, Any]]:
    """Load evaluation results from a JSON file."""
    with open(path) as f:
        return json.load(f)
