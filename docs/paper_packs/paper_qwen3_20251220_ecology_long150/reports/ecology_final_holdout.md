# Final Holdout Evaluation

- git_commit: `c80aaa7`
- tasks: `config/evaluation/paper_qwen3_holdout_v1.jsonl` (n=120)
- selection_mode: `best_per_cell`

## Summary

- accuracy: 0.142 (17/120)
- avg_cost: 1.138
- avg_latency_ms: 3312.6
- avg_tokens: 22.7
- cost_per_correct: 8.030

## By Family

| family | accuracy | correct | total | avg_cost | avg_latency_ms | avg_tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code.format | 0.050 | 2 | 40 | 1.161 | 3303.6 | 21.2 |
| math | 0.275 | 11 | 40 | 1.076 | 3284.7 | 16.5 |
| word.count | 0.100 | 4 | 40 | 1.175 | 3349.5 | 30.5 |
