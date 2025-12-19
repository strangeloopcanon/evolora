# Final Holdout Evaluation

- git_commit: `a099f94`
- tasks: `config/evaluation/paper_qwen3_holdout_v1.jsonl` (n=120)
- selection_mode: `best_per_cell`

## Summary

- accuracy: 0.133 (16/120)
- avg_cost: 1.935
- avg_latency_ms: 1910.8
- avg_tokens: 23.2
- cost_per_correct: 14.509

## By Family

| family | accuracy | correct | total | avg_cost | avg_latency_ms | avg_tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code.format | 0.000 | 0 | 40 | 2.122 | 2098.7 | 22.9 |
| math | 0.300 | 12 | 40 | 1.834 | 1810.7 | 17.4 |
| word.count | 0.100 | 4 | 40 | 1.847 | 1822.9 | 29.1 |
