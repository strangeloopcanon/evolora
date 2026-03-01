#!/usr/bin/env python3
"""Phase 4: Structural analysis -- overlap, correlation, and plots.

Loads masks and evaluation results, computes Jaccard overlap and importance
correlation, generates heatmaps and accuracy-vs-sparsity charts.

Usage:
    python scripts/analyze_submasks.py --config config/experiments/submask_discovery.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plasticity.analysis import (
    build_summary,
    importance_correlation,
    mask_overlap,
    per_layer_overlap,
    per_module_type_sparsity,
    plot_accuracy_vs_sparsity,
    plot_overlap_heatmap,
    save_summary,
)
from plasticity.config import AnalysisConfig
from plasticity.evaluate import load_results
from plasticity.masks import MaskSet, load_importance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_analysis(cfg: AnalysisConfig) -> None:
    """Run full structural analysis pipeline."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    families = cfg.families

    # ── Load importance tensors ──
    logger.info("Loading importance tensors from %s", cfg.importance_dir)
    per_family_importance: Dict[str, Dict[str, Any]] = {}
    for family in families:
        try:
            per_family_importance[family] = load_importance(cfg.importance_dir, family)
        except FileNotFoundError:
            logger.warning("Importance not found for %s, skipping", family)

    # ── Importance correlation ──
    corr: Dict[Tuple[str, str], float] = {}
    if len(per_family_importance) >= 2:
        corr = importance_correlation(per_family_importance)
        logger.info("Importance correlations:")
        for pair, r in corr.items():
            logger.info("  %s vs %s: r=%.4f", pair[0], pair[1], r)

    # ── Per-sparsity mask analysis ──
    masks_dir = Path(cfg.masks_dir)
    all_overlaps: Dict[float, Dict[Tuple[str, str], float]] = {}
    all_layer_overlaps: Dict[float, Dict[str, Dict[Tuple[str, str], float]]] = {}

    for sparsity in cfg.sparsity_levels:
        sp_dir = masks_dir / f"sparsity_{sparsity:.2f}"
        if not sp_dir.exists():
            logger.warning("Sparsity dir not found: %s", sp_dir)
            continue

        logger.info("=== Sparsity %.0f%% ===", sparsity * 100)

        family_masks: Dict[str, MaskSet] = {}
        for family in families:
            mask_path = sp_dir / f"task_{family}"
            if mask_path.exists():
                family_masks[family] = MaskSet.load(mask_path)

        if len(family_masks) >= 2:
            overlap = mask_overlap(family_masks)
            all_overlaps[sparsity] = overlap
            logger.info("  Mask overlap (Jaccard):")
            for pair, j in overlap.items():
                logger.info("    %s vs %s: %.4f", pair[0], pair[1], j)

            layer_ov = per_layer_overlap(family_masks)
            all_layer_overlaps[sparsity] = layer_ov

            plot_overlap_heatmap(
                layer_ov,
                families,
                output_dir / f"heatmap_sp{sparsity:.2f}",
                dpi=cfg.dpi,
            )

        for family, mask in family_masks.items():
            mod_sp = per_module_type_sparsity(mask)
            logger.info("  %s module-type sparsity: %s", family, mod_sp)

    # ── Load and plot evaluation results ──
    eval_path = Path(cfg.eval_dir) / "results.json"
    eval_results: List[Dict[str, Any]] = []
    if eval_path.exists():
        eval_results = load_results(eval_path)
        logger.info("Loaded %d evaluation results", len(eval_results))

        results_by_sparsity: Dict[float, List[Dict[str, Any]]] = {}
        for r in eval_results:
            sp_label = r.get("sparsity_level", "")
            try:
                sp = float(sp_label.replace("sparsity_", ""))
            except (ValueError, AttributeError):
                continue
            results_by_sparsity.setdefault(sp, []).append(r)

        eval_mode = eval_results[0].get("eval_mode", "accuracy") if eval_results else "accuracy"

        if results_by_sparsity:
            plot_accuracy_vs_sparsity(
                results_by_sparsity,
                families,
                output_dir,
                dpi=cfg.dpi,
            )
            logger.info("Plots saved to %s", output_dir)

        _print_results_table(eval_results, families, eval_mode=eval_mode)
    else:
        logger.info("No evaluation results at %s, skipping accuracy plots", eval_path)

    # ── Save summary ──
    summary = build_summary(eval_results, all_overlaps.get(0.5, {}), corr)
    save_summary(summary, output_dir / "summary.json")
    logger.info("Summary saved to %s/summary.json", output_dir)

    # Save detailed overlap data
    overlap_data = {
        f"sparsity_{sp:.2f}": {f"{p[0]}_vs_{p[1]}": v for p, v in ov.items()}
        for sp, ov in all_overlaps.items()
    }
    with open(output_dir / "overlap_detail.json", "w") as f:
        json.dump(overlap_data, f, indent=2)
    logger.info("Analysis complete. All outputs in %s", output_dir)


def _print_results_table(
    results: List[Dict[str, Any]],
    families: List[str],
    *,
    eval_mode: str = "accuracy",
) -> None:
    """Print a formatted results table to the log."""
    metric_key = "perplexity_per_family" if eval_mode == "perplexity" else "accuracy_per_family"
    fmt = ".2f" if eval_mode == "perplexity" else ".3f"
    label = "ppl" if eval_mode == "perplexity" else "acc"

    header_cols = "  ".join(f"{f:>12s}" for f in families)
    logger.info("\n%-20s %-15s %s (%s)", "Sparsity", "Condition", header_cols, label)
    logger.info("-" * (40 + 14 * len(families)))
    for r in sorted(results, key=lambda x: (x.get("sparsity_level", ""), x.get("condition", ""))):
        vals = r.get(metric_key, {})
        val_str = "  ".join(f"{vals.get(f, 0.0):>12{fmt}}" for f in families)
        logger.info(
            "%-20s %-15s %s",
            r.get("sparsity_level", ""),
            r.get("condition", ""),
            val_str,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: structural analysis")
    parser.add_argument("--config", type=str, help="YAML config file path")
    parser.add_argument("--masks-dir", type=str, default=None)
    parser.add_argument("--importance-dir", type=str, default=None)
    parser.add_argument("--eval-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--families", nargs="+", default=None)
    args = parser.parse_args()

    if args.config:
        from omegaconf import OmegaConf

        raw = OmegaConf.load(args.config)
        cfg = AnalysisConfig(**OmegaConf.to_container(raw.get("analysis", raw), resolve=True))
    else:
        cfg = AnalysisConfig()

    if args.masks_dir:
        cfg.masks_dir = args.masks_dir
    if args.importance_dir:
        cfg.importance_dir = args.importance_dir
    if args.eval_dir:
        cfg.eval_dir = args.eval_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.families:
        cfg.families = args.families

    run_analysis(cfg)


if __name__ == "__main__":
    main()
