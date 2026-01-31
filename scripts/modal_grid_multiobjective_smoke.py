"""Modal smoke test for Evolora grid multiobjective runs.

This is intentionally small: it only checks that we can run a tiny evolution job
and a tiny base-vs-evo holdout evaluation in a Linux (Modal) environment.

Example:
  .venv311/bin/modal run scripts/modal_grid_multiobjective_smoke.py
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "evolora-grid-multiobjective-smoke"

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/evolora")

RUNS_VOLUME = modal.Volume.from_name("evolora-runs", create_if_missing=True)
HF_VOLUME = modal.Volume.from_name("evolora-hf-cache", create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    # Pin to versions we've already exercised locally (keeps Modal runs stable).
    # Install torch from the CPU wheel index to avoid pulling in CUDA runtime deps for the smoke test.
    .pip_install(
        "torch==2.2.1",
        index_url="https://download.pytorch.org/whl/cpu",
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
    timeout=60 * 60,
    cpu=4.0,
    memory=16_000,
    volumes={
        "/vol": RUNS_VOLUME,
        "/root/.cache/huggingface": HF_VOLUME,
    },
)
def run_smoke(*, seed: int = 777, generations: int = 1, device: str = "cpu") -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REMOTE_ROOT / "src")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("EVOLORA_PLASTICITY", "backprop")

    output_dir = Path("/vol") / f"grid_multiobj_smoke_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = REMOTE_ROOT / "config" / "experiments" / "qwen25_grid_multiobjective_full_ecology.yaml"
    model = "Qwen/Qwen2.5-0.5B"

    subprocess.run(
        [
            "python",
            str(REMOTE_ROOT / "scripts" / "run_evolution.py"),
            "--config",
            str(config),
            "--generations",
            str(generations),
            "--output",
            str(output_dir),
            "--checkpoint-every",
            "1",
            "--seed",
            str(seed),
            "--device",
            device,
            "--batch-size",
            "1",
            "--disable-human",
        ],
        check=True,
        env=env,
    )

    subprocess.run(
        [
            "python",
            str(REMOTE_ROOT / "scripts" / "evaluate_holdout.py"),
            "--holdout",
            str(REMOTE_ROOT / "config" / "evaluation" / "holdout_grid_multiobjective.jsonl"),
            "--model",
            model,
            "--evo-checkpoint",
            str(output_dir / "checkpoint.pt"),
            "--no-evo-selection",
            "--max-samples",
            "6",
            "--device",
            device,
            "--output",
            str(output_dir / "eval_modal_smoke.json"),
        ],
        check=True,
        env=env,
    )

    RUNS_VOLUME.commit()
    HF_VOLUME.commit()

    return {"output_dir": str(output_dir)}


@app.local_entrypoint()
def main(seed: int = 777, generations: int = 1, device: str = "cpu") -> None:
    result = run_smoke.remote(seed=seed, generations=generations, device=device)
    print(result)
