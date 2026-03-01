"""Structural analysis of submasks: overlap, correlation, and visualisation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from plasticity.masks import MaskSet


def jaccard_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Jaccard similarity between two boolean masks."""
    a_bool = a.bool()
    b_bool = b.bool()
    intersection = (a_bool & b_bool).sum().item()
    union = (a_bool | b_bool).sum().item()
    return intersection / union if union > 0 else 1.0


def mask_overlap(
    masks: Dict[str, MaskSet],
) -> Dict[Tuple[str, str], float]:
    """Compute pairwise Jaccard similarity across families.

    Returns a dict keyed by ``(family_a, family_b) -> jaccard``.
    """
    families = sorted(masks.keys())
    result: Dict[Tuple[str, str], float] = {}
    for i, fa in enumerate(families):
        for fb in families[i + 1 :]:
            all_a = torch.cat([masks[fa].masks[m].flatten() for m in masks[fa].module_names])
            all_b = torch.cat([masks[fb].masks[m].flatten() for m in masks[fb].module_names])
            result[(fa, fb)] = jaccard_similarity(all_a, all_b)
    return result


def per_layer_overlap(
    masks: Dict[str, MaskSet],
) -> Dict[str, Dict[Tuple[str, str], float]]:
    """Compute pairwise Jaccard per module (layer) across families."""
    families = sorted(masks.keys())
    ref = masks[families[0]]
    result: Dict[str, Dict[Tuple[str, str], float]] = {}
    for mod_name in ref.module_names:
        result[mod_name] = {}
        for i, fa in enumerate(families):
            for fb in families[i + 1 :]:
                ma = masks[fa].masks[mod_name]
                mb = masks[fb].masks[mod_name]
                result[mod_name][(fa, fb)] = jaccard_similarity(ma, mb)
    return result


def importance_correlation(
    per_family_importance: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[Tuple[str, str], float]:
    """Pearson correlation of importance scores across families.

    Flattens all per-module importance tensors into a single vector per family,
    then computes pairwise Pearson r.
    """
    families = sorted(per_family_importance.keys())
    flat: Dict[str, torch.Tensor] = {}
    for fam in families:
        vecs = [per_family_importance[fam][m].flatten() for m in sorted(per_family_importance[fam])]
        flat[fam] = torch.cat(vecs).float()

    result: Dict[Tuple[str, str], float] = {}
    for i, fa in enumerate(families):
        for fb in families[i + 1 :]:
            va = flat[fa]
            vb = flat[fb]
            va_centered = va - va.mean()
            vb_centered = vb - vb.mean()
            num = (va_centered * vb_centered).sum()
            denom = va_centered.norm() * vb_centered.norm()
            r = (num / denom).item() if denom > 0 else 0.0
            result[(fa, fb)] = r
    return result


def per_module_type_sparsity(
    mask: MaskSet,
    attn_keywords: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    mlp_keywords: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj"),
) -> Dict[str, float]:
    """Aggregate sparsity by module type (attention vs MLP vs other)."""
    groups: Dict[str, List[float]] = {"attention": [], "mlp": [], "other": []}
    for name, m in mask.masks.items():
        sp = float((~m).sum().item()) / m.numel() if m.numel() > 0 else 0.0
        if any(kw in name for kw in attn_keywords):
            groups["attention"].append(sp)
        elif any(kw in name for kw in mlp_keywords):
            groups["mlp"].append(sp)
        else:
            groups["other"].append(sp)
    return {k: sum(v) / len(v) if v else 0.0 for k, v in groups.items()}


def build_summary(
    eval_results: List[Dict[str, Any]],
    overlap: Dict[Tuple[str, str], float],
    correlation: Dict[Tuple[str, str], float],
) -> Dict[str, Any]:
    """Build a combined summary dict for saving."""

    def _key(pair: Tuple[str, str]) -> str:
        return f"{pair[0]}_vs_{pair[1]}"

    eval_entries = []
    for r in eval_results:
        entry: Dict[str, Any] = {
            "condition": r["condition"],
            "sparsity": r.get("sparsity"),
            "sparsity_level": r.get("sparsity_level"),
            "eval_mode": r.get("eval_mode", "accuracy"),
        }
        if "accuracy_per_family" in r:
            entry["accuracy_per_family"] = r["accuracy_per_family"]
            entry["accuracy_overall"] = r["accuracy_overall"]
        if "perplexity_per_family" in r:
            entry["perplexity_per_family"] = r["perplexity_per_family"]
            entry["perplexity_overall"] = r["perplexity_overall"]
            entry["loss_per_family"] = r.get("loss_per_family")
            entry["loss_overall"] = r.get("loss_overall")
        eval_entries.append(entry)

    return {
        "evaluation": eval_entries,
        "mask_overlap": {_key(k): v for k, v in overlap.items()},
        "importance_correlation": {_key(k): v for k, v in correlation.items()},
    }


# ── Plotting ────────────────────────────────────────────────────────────


def plot_accuracy_vs_sparsity(
    results_by_sparsity: Dict[float, List[Dict[str, Any]]],
    families: List[str],
    output_path: str | Path,
    *,
    dpi: int = 150,
) -> None:
    """Bar chart of metric by condition and sparsity level, per family.

    Automatically detects whether results use accuracy or perplexity mode.
    """
    if not HAS_MPL:
        return

    first_result = next(
        (r for sp_results in results_by_sparsity.values() for r in sp_results), None
    )
    is_ppl = first_result is not None and "perplexity_per_family" in first_result
    metric_key = "perplexity_per_family" if is_ppl else "accuracy_per_family"
    y_label = "Perplexity" if is_ppl else "Accuracy"
    file_prefix = "perplexity" if is_ppl else "accuracy"

    sparsities = sorted(results_by_sparsity.keys())
    conditions = ["dense", "task_matched", "cross_task", "global", "random"]

    for family in families:
        fig, ax = plt.subplots(figsize=(10, 5))
        x_positions = range(len(sparsities))
        width = 0.15

        for ci, cond in enumerate(conditions):
            vals = []
            for sp in sparsities:
                matching = [r for r in results_by_sparsity[sp] if r["condition"] == cond]
                if matching:
                    vals.append(matching[0].get(metric_key, {}).get(family, 0.0))
                else:
                    vals.append(0.0)
            offsets = [x + ci * width for x in x_positions]
            ax.bar(offsets, vals, width, label=cond)

        ax.set_xlabel("Sparsity")
        ax.set_ylabel(y_label)
        ax.set_title(f"{family}: {y_label} by Condition and Sparsity")
        center_offset = (len(conditions) - 1) * width / 2
        ax.set_xticks([x + center_offset for x in x_positions])
        ax.set_xticklabels([f"{s:.0%}" for s in sparsities])
        ax.legend(fontsize=8)
        if not is_ppl:
            ax.set_ylim(0, 1.05)
        fig.tight_layout()

        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{file_prefix}_{family}.png", dpi=dpi)
        plt.close(fig)


def plot_overlap_heatmap(
    per_layer: Dict[str, Dict[Tuple[str, str], float]],
    families: List[str],
    output_path: str | Path,
    *,
    dpi: int = 150,
) -> None:
    """Heatmap of per-layer Jaccard overlap between family pairs."""
    if not HAS_MPL:
        return

    pairs = []
    fam_sorted = sorted(families)
    for i, fa in enumerate(fam_sorted):
        for fb in fam_sorted[i + 1 :]:
            pairs.append((fa, fb))

    if not pairs:
        return

    module_names = sorted(per_layer.keys())
    n_layers = len(module_names)
    n_pairs = len(pairs)

    data = torch.zeros(n_layers, n_pairs)
    for li, mod in enumerate(module_names):
        for pi, pair in enumerate(pairs):
            data[li, pi] = per_layer[mod].get(pair, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, n_pairs * 2), max(8, n_layers * 0.3)))
    im = ax.imshow(data.numpy(), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_yticks(range(n_layers))
    short_names = [_short_module_name(m) for m in module_names]
    ax.set_yticklabels(short_names, fontsize=6)
    ax.set_xticks(range(n_pairs))
    ax.set_xticklabels([f"{a} vs {b}" for a, b in pairs], rotation=30, ha="right", fontsize=8)
    ax.set_title("Per-Layer Jaccard Overlap Between Task Families")
    fig.colorbar(im, ax=ax, label="Jaccard Similarity")
    fig.tight_layout()

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "layer_overlap_heatmap.png", dpi=dpi)
    plt.close(fig)


def _short_module_name(name: str) -> str:
    """Shorten 'model.layers.5.self_attn.q_proj' to 'L5.q_proj'."""
    parts = name.split(".")
    layer_num = ""
    suffix = parts[-1] if parts else name
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            layer_num = parts[i + 1]
            break
    if layer_num:
        return f"L{layer_num}.{suffix}"
    return name


def save_summary(summary: Dict[str, Any], path: str | Path) -> None:
    """Write summary dict to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
