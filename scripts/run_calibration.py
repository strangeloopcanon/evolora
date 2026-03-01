#!/usr/bin/env python3
"""Phase 1: Per-family importance recording.

Loads a pretrained model, generates calibration tasks for each family,
runs inference with importance hooks, and saves per-family importance tensors.

Usage:
    python scripts/run_calibration.py --config config/experiments/submask_discovery.yaml
    python scripts/run_calibration.py --model Qwen/Qwen3-1.7B --families regex math word.count
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plasticity.config import CalibrationConfig
from plasticity.hooks import ImportanceRecorder
from plasticity.masks import save_importance
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


def load_model(cfg: CalibrationConfig):
    """Load pretrained model and tokenizer."""
    device = _resolve_device(cfg.model.device)
    dtype = _resolve_dtype(cfg.model.dtype)
    logger.info("Loading model %s on %s (%s)", cfg.model.model_id, device, dtype)

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


def run_calibration(cfg: CalibrationConfig) -> None:
    """Run calibration for all configured families."""
    model, tokenizer, device = load_model(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for family in cfg.tasks.families:
        logger.info("=== Calibrating family: %s ===", family)
        tasks = generate_tasks(
            family,
            cfg.tasks.calibration_per_family,
            seed=cfg.tasks.calibration_seed,
        )
        prompts = [t.prompt for t in tasks]

        recorder = ImportanceRecorder()
        recorder.attach(model)

        t0 = time.time()
        for start in range(0, len(prompts), cfg.model.batch_size):
            batch_prompts = prompts[start : start + cfg.model.batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.tasks.max_sequence_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                model(**encoded)

            done = min(start + cfg.model.batch_size, len(prompts))
            if done % 20 == 0 or done == len(prompts):
                logger.info("  %s: %d/%d prompts processed", family, done, len(prompts))

        elapsed = time.time() - t0
        recorder.flush_to_cpu()
        importance = recorder.collect(normalize=True)
        recorder.detach()

        logger.info(
            "  %s: %d modules, %.1fs elapsed",
            family,
            len(importance),
            elapsed,
        )
        save_importance(importance, str(output_dir), family)
        logger.info("  Saved importance to %s/%s/", output_dir, family)

    logger.info("Calibration complete. Output: %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: per-family importance recording")
    parser.add_argument("--config", type=str, help="YAML config file path")
    parser.add_argument("--model", type=str, default=None, help="Override model ID")
    parser.add_argument("--families", nargs="+", default=None, help="Override task families")
    parser.add_argument(
        "--n-calibration", type=int, default=None, help="Override calibration count"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    if args.config:
        from omegaconf import OmegaConf

        raw = OmegaConf.load(args.config)
        cfg = CalibrationConfig(**OmegaConf.to_container(raw.get("calibration", raw), resolve=True))
    else:
        cfg = CalibrationConfig()

    if args.model:
        cfg.model.model_id = args.model
    if args.families:
        cfg.tasks.families = args.families
    if args.n_calibration:
        cfg.tasks.calibration_per_family = args.n_calibration
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.device:
        cfg.model.device = args.device
    if args.batch_size:
        cfg.model.batch_size = args.batch_size

    run_calibration(cfg)


if __name__ == "__main__":
    main()
