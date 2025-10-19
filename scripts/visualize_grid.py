#!/usr/bin/env python3
"""Visualise grid ecology metrics over generations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render heatmaps of grid cell metrics.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to gen_summaries.jsonl produced by eval_gemma_long.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts_visuals"),
        help="Directory to write visualisations.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["difficulty", "success_ema", "price", "avg_roi", "avg_energy_cost"],
        help="Cell metrics to visualise. Non-cell metrics (avg_roi, avg_energy_cost, merges, population) will be plotted as line charts.",
    )
    return parser.parse_args()


def load_generations(path: Path) -> List[Dict]:
    generations: List[Dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            generations.append(json.loads(line))
    return generations


def ensure_cells_matrix(generations: List[Dict], metric: str) -> np.ndarray:
    cells = sorted(generations[0]["cells"].keys())
    matrix = np.zeros((len(generations), len(cells)))
    for gen_idx, record in enumerate(generations):
        for cell_idx, cell_name in enumerate(cells):
            matrix[gen_idx, cell_idx] = record["cells"][cell_name].get(metric, 0.0)
    return matrix, cells


def plot_heatmap(matrix: np.ndarray, cells: List[str], metric: str, output: Path) -> None:
    plt.figure(figsize=(max(6, len(cells)), 4))
    plt.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label=metric)
    plt.yticks(range(matrix.shape[0]), range(1, matrix.shape[0] + 1))
    plt.xticks(range(len(cells)), cells, rotation=45, ha="right")
    plt.xlabel("Cell")
    plt.ylabel("Generation")
    plt.title(f"{metric} over generations")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_line(data: List[float], metric: str, output: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(data) + 1), data, marker="o")
    plt.xlabel("Generation")
    plt.ylabel(metric)
    plt.title(f"{metric} over generations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main() -> None:
    args = parse_args()
    generations = load_generations(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cell_metrics = {"difficulty", "success_ema", "price"}
    scalar_metrics = {"avg_roi", "avg_energy_cost", "merges", "population", "active", "bankrupt"}

    for metric in args.metrics:
        if metric in cell_metrics:
            matrix, cells = ensure_cells_matrix(generations, metric)
            plot_heatmap(matrix, cells, metric, args.output_dir / f"{metric}_heatmap.png")
        elif metric in scalar_metrics:
            data = [record.get(metric, 0.0) for record in generations]
            plot_line(data, metric, args.output_dir / f"{metric}_line.png")
        else:
            # default to scalar line if metric not recognised
            data = [record.get(metric, 0.0) for record in generations]
            plot_line(data, metric, args.output_dir / f"{metric}_line.png")

    print("Visualisations written to", args.output_dir)


if __name__ == "__main__":
    main()
