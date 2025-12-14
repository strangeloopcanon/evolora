#!/usr/bin/env python3
"""Load a saved adapter snapshot and answer prompts.

Example:
  python scripts/infer_saved_adapter.py --config config/experiments/paper_qwen3_single.yaml \\
      --adapter artifacts_run/adapters/org_abc.safetensors \\
      --prompt "Add 2 and 3. Respond with the number only."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from symbiont_ecology import ATPLedger, BanditRouter, HostKernel, load_ecology_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with a persisted adapter")
    parser.add_argument("--config", type=Path, required=True, help="Experiment config YAML")
    parser.add_argument("--adapter", type=Path, default=None, help="Adapter safetensors file")
    parser.add_argument(
        "--run", type=Path, default=None, help="Run directory with adapters/index.json"
    )
    parser.add_argument(
        "--organelle-id", type=str, default=None, help="Organelle to load (default: best)"
    )
    parser.add_argument(
        "--rank", type=int, default=None, help="LoRA rank override (default: from index/config)"
    )
    parser.add_argument(
        "--prompt", type=str, action="append", required=True, help="Prompt (repeatable)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Torch device override (cpu/mps/cuda)"
    )
    return parser.parse_args()


def _resolve_from_run(run_dir: Path, organelle_id: str | None) -> tuple[str, Path, int | None]:
    index_path = run_dir / "adapters" / "index.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    exported = data.get("exported", []) or []
    if not isinstance(exported, list) or not exported:
        raise SystemExit(f"No exported adapters listed in {index_path}")
    if organelle_id is None:
        organelle_id = str(exported[0].get("organelle_id"))
    for entry in exported:
        if str(entry.get("organelle_id")) == organelle_id:
            filename = str(entry.get("file"))
            rank = entry.get("rank")
            rank_val = int(rank) if isinstance(rank, int) else None
            return organelle_id, run_dir / "adapters" / filename, rank_val
    raise SystemExit(f"Organelle {organelle_id} not found in {index_path}")


def main() -> None:
    args = parse_args()
    cfg = load_ecology_config(args.config)
    if args.device is not None:
        cfg.host.device = args.device
    ledger = ATPLedger()
    router = BanditRouter()
    host = HostKernel(config=cfg, router=router, ledger=ledger)
    host.freeze_host()

    organelle_id = args.organelle_id
    adapter_path = args.adapter
    rank_from_index = None
    if args.run is not None:
        organelle_id, adapter_path, rank_from_index = _resolve_from_run(args.run, organelle_id)
    if adapter_path is None:
        raise SystemExit("Provide --adapter or --run")
    if organelle_id is None:
        organelle_id = adapter_path.stem
    rank = args.rank or rank_from_index or cfg.host.max_lora_rank
    host.spawn_organelle(rank=int(rank), organelle_id=organelle_id)
    host.load_organelle_adapter(organelle_id, adapter_path)

    for prompt in args.prompt:
        result = host.step(
            prompt=prompt,
            intent="adapter infer",
            max_routes=1,
            allowed_organelle_ids=[organelle_id],
        )
        metrics = result.responses.get(organelle_id)
        answer = metrics.answer if metrics is not None else ""
        print(answer)  # noqa: T201


if __name__ == "__main__":
    main()
