#!/usr/bin/env python3
"""Phase 2: Derive binary masks from importance tensors.

Loads per-family importance tensors from Phase 1, derives task-specific,
global, and random masks at each configured sparsity level, and saves them.

Usage:
    python scripts/derive_masks.py --config config/experiments/submask_discovery.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plasticity.config import MaskConfig
from plasticity.masks import (
    derive_global_masks,
    derive_masks,
    derive_random_masks,
    load_importance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_derivation(cfg: MaskConfig, families: list[str]) -> None:
    """Derive all masks and save to disk."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_family_importance = {}
    for family in families:
        logger.info("Loading importance for family: %s", family)
        per_family_importance[family] = load_importance(cfg.importance_dir, family)

    any_family = next(iter(per_family_importance.values()))
    n_modules = len(any_family)
    logger.info("Loaded %d families, %d modules each", len(families), n_modules)

    for sparsity in cfg.sparsity_levels:
        sp_dir = output_dir / f"sparsity_{sparsity:.2f}"
        logger.info("=== Sparsity %.0f%% ===", sparsity * 100)

        for family in families:
            mask = derive_masks(per_family_importance[family], sparsity)
            save_path = sp_dir / f"task_{family}"
            mask.save(save_path)
            logger.info("  %s task mask: %.1f%% sparse", family, mask.sparsity() * 100)

        global_mask = derive_global_masks(per_family_importance, sparsity)
        global_mask.save(sp_dir / "global")
        logger.info("  global mask: %.1f%% sparse", global_mask.sparsity() * 100)

        random_mask = derive_random_masks(any_family, sparsity, seed=cfg.random_seed)
        random_mask.save(sp_dir / "random")
        logger.info("  random mask: %.1f%% sparse", random_mask.sparsity() * 100)

    logger.info("All masks saved to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: derive binary masks")
    parser.add_argument("--config", type=str, help="YAML config file path")
    parser.add_argument("--importance-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--sparsity-levels", nargs="+", type=float, default=None)
    args = parser.parse_args()

    families = args.families or ["regex", "math", "word.count"]

    if args.config:
        from omegaconf import OmegaConf

        raw = OmegaConf.load(args.config)
        cfg = MaskConfig(**OmegaConf.to_container(raw.get("masks", raw), resolve=True))
        if "calibration" in raw and not args.families:
            cal = raw["calibration"]
            if "tasks" in cal and "families" in cal["tasks"]:
                families = list(cal["tasks"]["families"])
    else:
        cfg = MaskConfig()

    if args.importance_dir:
        cfg.importance_dir = args.importance_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.sparsity_levels:
        cfg.sparsity_levels = args.sparsity_levels

    run_derivation(cfg, families)


if __name__ == "__main__":
    main()
