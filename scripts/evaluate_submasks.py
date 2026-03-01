#!/usr/bin/env python3
"""Phase 3: Evaluate model under 5 mask conditions per sparsity level.

Conditions: dense, task_matched, cross_task, global, random.

Usage:
    python scripts/evaluate_submasks.py --config config/experiments/submask_discovery.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from plasticity.config import EvalConfig
from plasticity.evaluate import run_condition, save_results
from plasticity.masks import MaskSet
from plasticity.tasks import generate_tasks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(name, torch.float32)


def _load_model(cfg: EvalConfig):
    device = _resolve_device(cfg.model.device)
    dtype = _resolve_dtype(cfg.model.dtype)
    logger.info("Loading model %s on %s", cfg.model.model_id, device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_id, dtype=dtype, device_map=None)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, tokenizer, device


def _cross_task_family(family: str, families: list[str]) -> str:
    """Pick a different family for cross-task evaluation."""
    others = [f for f in families if f != family]
    return others[0] if others else family


def _merge_per_family(
    per_family_results: List[Dict[str, Any]],
    condition: str,
    eval_mode: str,
) -> Dict[str, Any]:
    """Merge per-family sub-results into a single condition result."""
    sparsity = per_family_results[0].get("sparsity") if per_family_results else None

    if eval_mode == "perplexity":
        combined_ppl: Dict[str, float] = {}
        combined_loss: Dict[str, float] = {}
        combined_n: Dict[str, int] = {}
        for r in per_family_results:
            combined_ppl.update(r.get("perplexity_per_family", {}))
            combined_loss.update(r.get("loss_per_family", {}))
            combined_n.update(r.get("n_per_family", {}))
        total_n = sum(combined_n.values())
        weighted_loss = sum(
            combined_loss.get(f, 0) * combined_n.get(f, 0) for f in combined_n
        ) / max(total_n, 1)
        import math as _math

        return {
            "condition": condition,
            "perplexity_per_family": combined_ppl,
            "perplexity_overall": _math.exp(min(weighted_loss, 100.0)),
            "loss_per_family": combined_loss,
            "loss_overall": weighted_loss,
            "n_per_family": combined_n,
            "sparsity": sparsity,
        }

    combined_acc: Dict[str, float] = {}
    for r in per_family_results:
        combined_acc.update(r.get("accuracy_per_family", {}))
    n_total = sum(
        r.get("n_per_family", {}).get(f, 0)
        for r in per_family_results
        for f in r.get("n_per_family", {})
    )
    n_correct = sum(
        r.get("accuracy_per_family", {}).get(f, 0) * r.get("n_per_family", {}).get(f, 0)
        for r in per_family_results
        for f in r.get("accuracy_per_family", {})
    )
    return {
        "condition": condition,
        "accuracy_per_family": combined_acc,
        "accuracy_overall": n_correct / max(n_total, 1),
        "sparsity": sparsity,
    }


def run_evaluation(cfg: EvalConfig, eval_mode: str = "accuracy") -> None:
    """Run all conditions across all sparsity levels."""
    model, tokenizer, device = _load_model(cfg)
    families = cfg.tasks.families
    masks_dir = Path(cfg.masks_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluation mode: %s", eval_mode)

    holdout_tasks = {}
    for family in families:
        holdout_tasks[family] = generate_tasks(
            family, cfg.tasks.holdout_per_family, seed=cfg.tasks.holdout_seed
        )
    all_tasks = [t for fam_tasks in holdout_tasks.values() for t in fam_tasks]

    sparsity_dirs = sorted(masks_dir.glob("sparsity_*"))
    if not sparsity_dirs:
        logger.error("No sparsity directories found in %s", masks_dir)
        return

    all_results: List[Dict[str, Any]] = []
    rc_kwargs: Dict[str, Any] = {
        "batch_size": cfg.model.batch_size,
        "device": device,
        "eval_mode": eval_mode,
    }

    for sp_dir in sparsity_dirs:
        sparsity_label = sp_dir.name
        logger.info("=== %s ===", sparsity_label)

        family_masks = {}
        for family in families:
            mask_path = sp_dir / f"task_{family}"
            if mask_path.exists():
                family_masks[family] = MaskSet.load(mask_path)

        global_mask = None
        global_path = sp_dir / "global"
        if global_path.exists():
            global_mask = MaskSet.load(global_path)

        random_mask = None
        random_path = sp_dir / "random"
        if random_path.exists():
            random_mask = MaskSet.load(random_path)

        for condition in cfg.conditions:
            t0 = time.time()

            if condition == "dense":
                result = run_condition(
                    "dense",
                    model,
                    tokenizer,
                    all_tasks,
                    **rc_kwargs,
                )
            elif condition == "task_matched":
                per_family_results: List[Dict[str, Any]] = []
                for family in families:
                    mask = family_masks.get(family)
                    fam_tasks = holdout_tasks[family]
                    r = run_condition(
                        f"task_matched_{family}",
                        model,
                        tokenizer,
                        fam_tasks,
                        mask=mask,
                        **rc_kwargs,
                    )
                    per_family_results.append(r)
                result = _merge_per_family(per_family_results, "task_matched", eval_mode)
            elif condition == "cross_task":
                per_family_results = []
                for family in families:
                    cross_fam = _cross_task_family(family, families)
                    mask = family_masks.get(cross_fam)
                    fam_tasks = holdout_tasks[family]
                    r = run_condition(
                        f"cross_task_{family}_with_{cross_fam}_mask",
                        model,
                        tokenizer,
                        fam_tasks,
                        mask=mask,
                        **rc_kwargs,
                    )
                    per_family_results.append(r)
                result = _merge_per_family(per_family_results, "cross_task", eval_mode)
            elif condition == "global":
                result = run_condition(
                    "global",
                    model,
                    tokenizer,
                    all_tasks,
                    mask=global_mask,
                    **rc_kwargs,
                )
            elif condition == "random":
                result = run_condition(
                    "random",
                    model,
                    tokenizer,
                    all_tasks,
                    mask=random_mask,
                    **rc_kwargs,
                )
            else:
                logger.warning("Unknown condition: %s", condition)
                continue

            elapsed = time.time() - t0
            result["sparsity_level"] = sparsity_label
            result["elapsed_s"] = round(elapsed, 1)
            result["eval_mode"] = eval_mode
            all_results.append(result)

            if eval_mode == "perplexity":
                metric = result.get("perplexity_overall", 0)
                logger.info(
                    "  %s | %s | ppl=%.2f | %.1fs",
                    sparsity_label,
                    condition,
                    metric,
                    elapsed,
                )
            else:
                metric = result.get("accuracy_overall", 0)
                logger.info(
                    "  %s | %s | acc=%.3f | %.1fs",
                    sparsity_label,
                    condition,
                    metric,
                    elapsed,
                )

    results_path = output_dir / "results.json"
    save_results(all_results, results_path)
    logger.info("Results saved to %s", results_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: evaluate submasks")
    parser.add_argument("--config", type=str, help="YAML config file path")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--masks-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--n-holdout", type=int, default=None, help="Override holdout tasks per family"
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="perplexity",
        choices=["accuracy", "perplexity"],
        help="Evaluation metric (default: perplexity)",
    )
    args = parser.parse_args()

    if args.config:
        from omegaconf import OmegaConf

        raw = OmegaConf.load(args.config)
        cfg = EvalConfig(**OmegaConf.to_container(raw.get("evaluation", raw), resolve=True))
    else:
        cfg = EvalConfig()

    if args.model:
        cfg.model.model_id = args.model
    if args.masks_dir:
        cfg.masks_dir = args.masks_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.families:
        cfg.tasks.families = args.families
    if args.device:
        cfg.model.device = args.device
    if args.batch_size:
        cfg.model.batch_size = args.batch_size
    if args.n_holdout:
        cfg.tasks.holdout_per_family = args.n_holdout

    run_evaluation(cfg, eval_mode=args.eval_mode)


if __name__ == "__main__":
    main()
