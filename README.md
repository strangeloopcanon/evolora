# Evolora — Symbiotic LLM Ecology
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/strangeloopcanon/evolora)

**What if we started *evolving* language models?**

Evolora treats a small LLM as a computational substrate—like the "physics" of a world—and evolves a population of tiny LoRA adapters ("organelles") that compete, learn, merge, and die based on their performance. It's artificial life meets language models: adapters pay energy to attempt tasks, earn rewards when they succeed, go bankrupt when they fail, and occasionally merge into new offspring when they prove their worth.

---

## Core Idea

```
┌─────────────────────────────────────────────────────────────────┐
│                     FROZEN HOST (Qwen3-0.6B)                    │
│                    (weights never change)                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       ┌──────────┐   │
│  │Organelle │  │Organelle │  │Organelle │  ...  │Organelle │   │
│  │  (LoRA)  │  │  (LoRA)  │  │  (LoRA)  │       │  (LoRA)  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       └────┬─────┘   │
│       │             │             │                  │         │
│       └─────────────┴──────┬──────┴──────────────────┘         │
│                            ▼                                    │
│              ┌─────────────────────────┐                       │
│              │    ENERGY ECONOMY       │                       │
│              │  • Pay to participate   │                       │
│              │  • Earn via rewards     │                       │
│              │  • Bankrupt → retire    │                       │
│              │  • Merge on uplift      │                       │
│              └─────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

Three things are being optimised simultaneously:
1. **Per-call competence** — each organelle tries to maximise reward on tasks
2. **Energy efficiency** — ROI = reward − cost; cheap *and* good wins
3. **Population diversity** — a quality-diversity archive nudges the ecology toward covering different niches

### Three Levels of Learning

1. **Local plasticity** — Hebbian reward-modulated updates within each organelle (per episode)
2. **Population selection** — assimilation + merging of successful adapters (per generation)
3. **Meta-evolution** — tuning ecology parameters (assimilation threshold, energy floor, etc.)

---

## Key Results (Qwen3-0.6B, 150 generations)

| Condition | Episodes | Mean ROI | Merges | QD Coverage | Holdout Acc | Holdout Cost |
|-----------|----------|----------|--------|-------------|-------------|--------------|
| Frozen base | 1,727 | 0.94 | 0 | 0% | 13.3% | 1.91 |
| Single adapter | 63 | −0.06 | 0 | 0% | 13.3% | 1.94 |
| **Full ecology** | 7,413 | **1.39** | 52 | 16.7% | **14.2%** | **1.14** |

![ROI comparison across conditions](docs/paper_packs/paper_qwen3_20251220_ecology_long150/plots/roi_comparison.png)

The ecology achieves **higher ROI**, **better holdout accuracy**, and **40% lower inference cost** compared to baselines—while the frozen host and single-adapter setups remain structurally inert.

See `docs/paper_packs/` for detailed run reports and plots.

---

## Comparing Evolution vs SFT (Compute-Matched)

A key question: does evolutionary adaptation actually outperform standard supervised fine-tuning given the same compute budget? This section explains how to run a fair comparison.

### Quick E2E Example

```bash
# 1. Run evolution (e.g., 10 generations with regex tasks)
python scripts/run_evolution.py \
    --config config/experiments/qwen3_regex_simple.yaml \
    --generations 10 \
    --output artifacts_evo_run \
    --checkpoint-every 1 \
    --disable-human

# 2. Run SFT with matched token budget from the evolution checkpoint
python scripts/run_sft.py \
    --checkpoint artifacts_evo_run/checkpoint.pt \
    --data config/training/regex_sft_data.jsonl \
    --output artifacts_sft_run

# 3. Evaluate both on holdout tasks
python scripts/evaluate_holdout.py \
    --holdout config/evaluation/regex_generalization.jsonl \
    --sft-adapter artifacts_sft_run/peft_adapter \
    --evo-checkpoint artifacts_evo_run/checkpoint.pt \
    --output comparison_results.json \
    --verbose
```

### How It Works

1. **Run evolution** with compute tracking enabled:
   - `ComputeBudget` tracks total tokens, forward passes, and Hebbian updates
   - Metrics saved to checkpoint and `gen_summaries.jsonl`

2. **Run SFT with matched budget**:
   ```bash
   # Match token budget from evolution checkpoint
   python scripts/run_sft.py \
       --checkpoint artifacts_evo/checkpoint.pt \
       --data config/training/regex_sft_data.jsonl \
       --output artifacts_sft

   # Or specify explicit budget
   python scripts/run_sft.py \
       --token-budget 500000 \
       --data config/training/regex_sft_data.jsonl \
       --output artifacts_sft
   ```
   - `TokenBudgetCallback` stops training when budget exhausted
   - Exports LoRA compatible with `HostKernel.load_organelle_adapter()`

3. **Evaluate both** on the same holdout tasks:
   ```bash
   python scripts/evaluate_holdout.py \
       --holdout config/evaluation/regex_generalization.jsonl \
       --sft-adapter artifacts_sft/peft_adapter \
       --evo-checkpoint artifacts_evo/checkpoint.pt \
       --verbose
   ```
   - Compares base model, SFT, and best evolution organelle
   - Auto-selects the best organelle by ROI from `gen_summaries.jsonl`
   - Reports accuracy on held-out tasks

### Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_evolution.py` | Run evolutionary LoRA with compute tracking |
| `scripts/run_sft.py` | Train SFT baseline with matched token budget |
| `scripts/evaluate_holdout.py` | Compare models on holdout tasks |

### Important Notes

- **Model compatibility**: The evolution checkpoint and evaluation must use the same base model (check tensor shapes match)
- **Token tracking**: Tokens are tracked in `observations.metrics.tokens` in episodes.jsonl
- **Best organelle selection**: `evaluate_holdout.py` automatically picks the organelle with highest ROI

This enables fair comparison: evolutionary adaptation (many small adapters + population dynamics) vs traditional gradient-based fine-tuning (single adapter, supervised loss).

---

## Project Structure

```
src/symbiont_ecology/
├── host/           # Frozen backbone wrapper, LoRA slot management
├── organelles/     # Hebbian-PEFT adapters with eligibility traces
├── routing/        # Bandit router for adapter selection
├── evolution/      # Population manager, model merger, morphogenesis
├── environment/    # Task factory, ecology loop, grid controller
├── economics/      # ATP ledger, energy settlement
├── metrics/        # Telemetry sink, QD archive, ComputeBudget tracking
└── config.py       # Pydantic config models

config/
├── experiments/    # YAML configs: frozen, single, ecology variants
├── evaluation/     # Holdout task sets (e.g., regex_generalization.jsonl)
├── training/       # SFT training data (e.g., regex_sft_data.jsonl)
└── ecology.yaml    # Base ecology parameters

scripts/
├── run_evolution.py        # Main experiment runner (resumable)
├── run_sft.py              # SFT baseline trainer (compute-matched)
├── evaluate_holdout.py     # Compare models on holdout tasks
├── analyze_ecology_run.py  # Generate reports + plots from a run
├── evoscope.py             # Interactive run visualisation
├── paper_pack.py           # Bundle runs into tracked summaries
└── benchmark_suite.py      # CI-safe benchmark harness
```

---

## What you get
- YAML-configured experiments under `config/experiments/`
- CI-safe benchmark harness (stubbed backend; no model downloads)
- Run artifacts under `artifacts_*` (telemetry + plots)
- Analysis + visualization: `scripts/analyze_ecology_run.py`, `scripts/evoscope.py`
- Paper packs (tracked summaries): `docs/paper_packs/`

## Quickstart (CI-safe)
```bash
make all
AGENT_MODE=baseline .venv/bin/python scripts/benchmark_suite.py --mode ci
```

## Reproduce the paper-style Qwen3 suite
```bash
scripts/run_paper_ecology_suite.sh all
```

Optional: add a fixed, measurement-only holdout suite (paper packs will include holdout metrics):
```bash
FINAL_HOLDOUT_TASKS=config/evaluation/paper_qwen3_holdout_v1.jsonl \
FINAL_HOLDOUT_SAMPLE_SIZE=120 \
scripts/run_paper_ecology_suite.sh all
```

## Analyze a run
```bash
MPLCONFIGDIR="$(mktemp -d)" .venv/bin/python scripts/analyze_ecology_run.py <run_dir> --plots --report
MPLCONFIGDIR="$(mktemp -d)" .venv/bin/python scripts/evoscope.py <run_dir>
```

## Docs
- `docs/running.md` — how to run longer experiments (resume/holdout)
- `docs/ecology_overview.md` — conceptual overview + baselines framing
- `docs/architecture.md` — module-level architecture sketch
- `docs/paper_packs/README.md` — paper packs + examples

## Notes
- Full experiments download model weights from Hugging Face; use the CI benchmark mode above for a fast smoke test without downloads.
- On macOS, `MPLCONFIGDIR="$(mktemp -d)"` avoids matplotlib font-cache issues in long runs/plots.
