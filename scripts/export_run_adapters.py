#!/usr/bin/env python3
"""Export top-N organelle adapters from a checkpointed run into safetensors files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from symbiont_ecology.utils.checkpoint_io import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export adapters from a run checkpoint")
    parser.add_argument(
        "--run", type=Path, required=True, help="Run directory containing checkpoint.pt"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file path (default: <run>/checkpoint.pt)",
    )
    parser.add_argument("--top-n", type=int, default=4, help="How many adapters to export")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <run>/adapters)",
    )
    parser.add_argument(
        "--sort-by",
        choices=["roi", "score"],
        default="roi",
        help="Metric used to rank organelles for export",
    )
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help=(
            "Allow trusted legacy pickle checkpoints. "
            "Unsafe: untrusted pickle files can execute arbitrary code."
        ),
    )
    return parser.parse_args()


def _as_tensor_state(state: object) -> dict[str, torch.Tensor] | None:
    if not isinstance(state, dict):
        return None
    tensor_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue
        tensor_state[key] = value.detach().cpu().contiguous()
    return tensor_state or None


def _average_tail(values: object, limit: int = 10) -> float:
    if not isinstance(values, list) or not values:
        return 0.0
    tail = values[-max(1, limit) :]
    nums: list[float] = []
    for value in tail:
        try:
            nums.append(float(value))
        except Exception:
            continue
    if not nums:
        return 0.0
    return float(sum(nums) / len(nums))


def _metrics_from_population_state(
    state: dict[str, object], organelle_id: str, *, limit: int = 10
) -> tuple[float, float, int | None]:
    avg_roi = 0.0
    avg_score = 0.0
    rank: int | None = None
    population_state = state.get("population_state")
    if isinstance(population_state, dict):
        roi_map = population_state.get("roi")
        history_map = population_state.get("history")
        genomes_map = population_state.get("genomes")
        if isinstance(roi_map, dict):
            avg_roi = _average_tail(roi_map.get(organelle_id), limit=limit)
        if isinstance(history_map, dict):
            avg_score = _average_tail(history_map.get(organelle_id), limit=limit)
        if isinstance(genomes_map, dict):
            genome_obj = genomes_map.get(organelle_id)
            if isinstance(genome_obj, dict):
                try:
                    rank = int(genome_obj.get("rank", 0))
                except Exception:
                    rank = None
        return avg_roi, avg_score, rank

    # Legacy checkpoint compatibility
    population = state.get("population")
    try:
        avg_roi = float(population.average_roi(organelle_id, limit=limit))  # type: ignore[attr-defined]
    except Exception:
        avg_roi = 0.0
    try:
        avg_score = float(population.average_score(organelle_id, limit=limit))  # type: ignore[attr-defined]
    except Exception:
        avg_score = 0.0
    try:
        genome = getattr(population, "population", {}).get(organelle_id)  # type: ignore[attr-defined]
        if genome is not None:
            rank = int(getattr(genome, "rank", 0))
    except Exception:
        rank = None
    return avg_roi, avg_score, rank


def main() -> None:
    args = parse_args()
    run_dir = args.run
    checkpoint_path = args.checkpoint or (run_dir / "checkpoint.pt")
    output_dir = args.output or (run_dir / "adapters")
    if args.top_n <= 0:
        raise SystemExit("--top-n must be > 0")
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    state = load_checkpoint(checkpoint_path, allow_unsafe_pickle=args.allow_unsafe_pickle)
    adapter_states = state.get("adapter_states", {}) or {}

    rows: list[dict[str, object]] = []
    for organelle_id, adapter_state in adapter_states.items():
        if not isinstance(organelle_id, str):
            continue
        avg_roi, avg_score, rank = _metrics_from_population_state(state, organelle_id, limit=10)
        rows.append(
            {
                "organelle_id": organelle_id,
                "avg_roi": avg_roi,
                "avg_score": avg_score,
                "rank": rank,
                "state": adapter_state,
            }
        )

    sort_key = "avg_roi" if args.sort_by == "roi" else "avg_score"
    rows.sort(key=lambda row: float(row.get(sort_key, 0.0)), reverse=True)
    selected = rows[: args.top_n]

    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, object]] = []
    for row in selected:
        organelle_id = str(row["organelle_id"])
        state_obj = row.get("state")
        tensor_state = _as_tensor_state(state_obj)
        if tensor_state is None:
            continue
        filename = f"{organelle_id}.safetensors"
        save_file(tensor_state, str(output_dir / filename))
        exported.append(
            {
                "organelle_id": organelle_id,
                "file": filename,
                "rank": row.get("rank"),
                "avg_roi": row.get("avg_roi", 0.0),
                "avg_score": row.get("avg_score", 0.0),
            }
        )

    index = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "top_n": args.top_n,
        "sort_by": args.sort_by,
        "exported": exported,
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Exported {len(exported)} adapters to {output_dir}")  # noqa: T201


if __name__ == "__main__":
    main()
