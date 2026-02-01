# Multi-Task Transfer Evaluation

Transfer evaluation measures how well models generalize from a training task family to unseen task families. This is central to understanding the "robustness premium" that evolutionary adaptation provides over standard supervised fine-tuning.

## Why Transfer Matters

Training models on a single task family often leads to overfitting on that family's specific patterns. Evolution, with its population-based exploration, may develop more generalizable representations. Transfer evaluation quantifies this by:

1. Training on a **source family** (e.g., regex tasks)
2. Evaluating on **target families** (e.g., math, code formatting)
3. Comparing how well each approach (evolution vs SFT) retains performance

## Key Metrics

| Metric | Description |
|--------|-------------|
| **Source accuracy** | Performance on the training family (in-distribution) |
| **Target accuracy** | Performance on each held-out family (out-of-distribution) |
| **Transfer ratio** | target_acc / source_acc — values near 1.0 indicate good transfer |
| **Transfer gap** | source_acc - target_acc — smaller is better |
| **Geometric mean** | Balanced accuracy across all families |
| **Robustness premium** | Difference between evolution and SFT on each metric |

<details>
<summary>Interpreting Transfer Metrics</summary>

- **Transfer ratio > 0.8**: Strong transfer; the model generalizes well
- **Transfer ratio 0.5–0.8**: Moderate transfer; some family-specific learning
- **Transfer ratio < 0.5**: Weak transfer; likely overfit to source family

The **robustness premium** measures whether evolution or SFT transfers better:
- Positive premium: Evolution is more robust
- Negative premium: SFT is more robust
- Near zero: Roughly equivalent
</details>

## Running Transfer Evaluation

### Basic Usage

```bash
python scripts/evaluate_transfer.py \
    --source-family regex \
    --target-families math.multi_step code.format logic.bool \
    --holdout config/evaluation/holdout_grid_multiobjective.jsonl \
    --evo-checkpoint artifacts/evo/checkpoint.pt \
    --sft-adapter artifacts/sft/peft_adapter
```

### Options

| Flag | Description |
|------|-------------|
| `--source-family` | Task family used for training |
| `--target-families` | Space-separated list of evaluation families |
| `--holdout` | Path to JSONL file with holdout tasks |
| `--evo-checkpoint` | Evolution checkpoint (.pt file) |
| `--sft-adapter` | SFT PEFT adapter directory |
| `--samples-per-family` | Limit samples per family (stratified) |
| `--output` | Save JSON results to this path |
| `--verbose` | Print per-task results |

### Example Workflow

1. **Train evolution model on regex tasks:**
   ```bash
   python scripts/run_evolution.py --config config/experiments/qwen3_regex.yaml
   ```

2. **Train compute-matched SFT:**
   ```bash
   python scripts/run_sft.py \
       --checkpoint artifacts/evo_regex/checkpoint.pt \
       --data config/training/regex_sft_data.jsonl
   ```

3. **Evaluate transfer:**
   ```bash
   python scripts/evaluate_transfer.py \
       --source-family regex \
       --target-families math.multi_step code.format json_repair \
       --holdout config/evaluation/holdout_grid_multiobjective.jsonl \
       --evo-checkpoint artifacts/evo_regex/checkpoint.pt \
       --sft-adapter artifacts/sft_regex/peft_adapter \
       --output results/transfer_regex_to_others.json
   ```

## Output Format

The script produces a JSON output with the following structure:

```json
{
  "source_family": "regex",
  "target_families": ["math.multi_step", "code.format"],
  "results": {
    "evolution": {
      "source_accuracy": 0.85,
      "mean_target_accuracy": 0.72,
      "geometric_mean_accuracy": 0.78,
      "mean_transfer_ratio": 0.85,
      "target_accuracies": {
        "math.multi_step": 0.70,
        "code.format": 0.74
      }
    },
    "sft": {
      "source_accuracy": 0.88,
      "mean_target_accuracy": 0.58,
      "mean_transfer_ratio": 0.66
    }
  },
  "robustness_premium": {
    "mean_transfer_ratio_delta": 0.19,
    "interpretation": "evolution_more_robust"
  }
}
```

## Creating Holdout Datasets

Holdout datasets should include tasks from multiple families in JSONL format:

```json
{"prompt": "...", "target": "...", "family": "regex", "depth": "short"}
{"prompt": "...", "target": "...", "family": "math.multi_step", "depth": "medium"}
```

Use `--samples-per-family` to ensure balanced evaluation across families with different task counts.

## Related Scripts

- `evaluate_holdout.py` — General holdout evaluation with bucket breakdown
- `run_evolution.py` — Train evolution models
- `run_sft.py` — Train SFT baseline models
- `compare_runs.py` — Compare multiple experiment runs
