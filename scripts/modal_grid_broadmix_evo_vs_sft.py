"""Modal runner for the broad-mix grid evo-vs-SFT experiment.

This runs the full pipeline on Modal:
  1) evolution (resumable checkpoints)
  2) compute-matched SFT
  3) OOD suite evaluation

Example:
  .venv311/bin/modal run scripts/modal_grid_broadmix_evo_vs_sft.py --seed 777 --calib-gens 5 --full-gens 50
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "evolora-grid-broadmix-evo-vs-sft"

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/evolora")

RUNS_VOLUME = modal.Volume.from_name("evolora-runs", create_if_missing=True)
HF_VOLUME = modal.Volume.from_name("evolora-hf-cache", create_if_missing=True)

# Torch 2.2.1 is the oldest version in our stack; keep CUDA 12.1 wheels for GPU runs.
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.1",
        index_url="https://download.pytorch.org/whl/cu121",
        extra_index_url="https://pypi.org/simple",
    )
    .pip_install(
        "transformers==4.57.5",
        "peft==0.18.1",
        "accelerate==1.12.0",
        "datasets==4.4.2",
        "numpy==1.26.4",
        "scipy==1.17.0",
        "rich==14.2.0",
        "pydantic==2.12.5",
        "omegaconf==2.3.0",
        "python-dotenv==1.2.1",
    )
    .add_local_dir(
        ROOT,
        remote_path=str(REMOTE_ROOT),
        ignore=[
            ".git",
            ".venv",
            ".venv311",
            ".beads",
            ".codex",
            "artifacts*",
            "plots",
            "results",
            "visuals",
            "__pycache__",
        ],
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=IMAGE,
    timeout=12 * 60 * 60,
    gpu="A10G",
    cpu=8.0,
    memory=32_000,
    volumes={
        "/vol": RUNS_VOLUME,
        "/root/.cache/huggingface": HF_VOLUME,
    },
)
def run_broadmix(
    *,
    seed: int = 777,
    calib_gens: int = 5,
    full_gens: int = 50,
    checkpoint_every: int = 5,
    train_size: int = 20_000,
    selection_size: int = 256,
    id_holdout_size: int = 512,
    ood_max_samples: int = 192,
    evo_selection_max_samples: int = 64,
    evo_selection_top_k_by_roi: int = 32,
    evo_selection_max_new_tokens: int = 64,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REMOTE_ROOT / "src")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("EVOLORA_PLASTICITY", "backprop")
    env.setdefault("AGENT_MODE", "baseline")

    output_dir = Path("/vol") / f"grid_broadmix_{seed}_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = REMOTE_ROOT / "config" / "experiments" / "qwen25_grid_broadmix_full_ecology.yaml"

    # Use the bash runner for resume logic + compute-matched SFT.
    subprocess.run(
        [
            "bash",
            str(REMOTE_ROOT / "scripts" / "run_grid_multiobjective_evo_vs_sft.sh"),
            "--config",
            str(config),
            "--output",
            str(output_dir),
            "--train-size",
            str(int(train_size)),
            "--selection-size",
            str(int(selection_size)),
            "--id-holdout-size",
            str(int(id_holdout_size)),
            "--calib-gens",
            str(calib_gens),
            "--full-gens",
            str(full_gens),
            "--checkpoint-every",
            str(checkpoint_every),
            "--seed",
            str(seed),
            "--device",
            "cuda",
            "--disable-human",
            "--evo-eval-routing",
            "family",
            "--evo-selection-max-samples",
            str(int(evo_selection_max_samples)),
            "--evo-selection-top-k-by-roi",
            str(int(evo_selection_top_k_by_roi)),
            "--evo-selection-max-new-tokens",
            str(int(evo_selection_max_new_tokens)),
            "--sft-match-budget-field",
            "train_flops",
            "--backprop-multiplier",
            "3.0",
        ],
        check=True,
        cwd=str(REMOTE_ROOT),
        env={**env, "PY": "python"},
    )

    # OOD suite: paper-family shift, plus ID if datasets exist.
    subprocess.run(
        [
            "python",
            str(REMOTE_ROOT / "scripts" / "run_ood_suite.py"),
            "--run-dir",
            str(output_dir),
            "--model",
            "Qwen/Qwen3-0.6B",
            "--device",
            "cuda",
            "--max-samples",
            str(int(ood_max_samples)),
            "--holdout-sampling",
            "stratified_family",
            "--evo-selection-max-samples",
            str(int(evo_selection_max_samples)),
            "--evo-selection-max-new-tokens",
            str(int(evo_selection_max_new_tokens)),
        ],
        check=True,
        cwd=str(REMOTE_ROOT),
        env=env,
    )

    RUNS_VOLUME.commit()
    HF_VOLUME.commit()

    return {"output_dir": str(output_dir)}


@app.local_entrypoint()
def main(
    seed: int = 777,
    calib_gens: int = 5,
    full_gens: int = 50,
    checkpoint_every: int = 5,
    train_size: int = 20_000,
    selection_size: int = 256,
    id_holdout_size: int = 512,
    ood_max_samples: int = 192,
    evo_selection_max_samples: int = 64,
    evo_selection_top_k_by_roi: int = 32,
    evo_selection_max_new_tokens: int = 64,
) -> None:
    result = run_broadmix.remote(
        seed=seed,
        calib_gens=calib_gens,
        full_gens=full_gens,
        checkpoint_every=checkpoint_every,
        train_size=train_size,
        selection_size=selection_size,
        id_holdout_size=id_holdout_size,
        ood_max_samples=ood_max_samples,
        evo_selection_max_samples=evo_selection_max_samples,
        evo_selection_top_k_by_roi=evo_selection_top_k_by_roi,
        evo_selection_max_new_tokens=evo_selection_max_new_tokens,
    )
    print(result)
