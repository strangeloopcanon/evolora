# Docx Inserts (Copy/Paste)

This file is meant to be copied into `evolora.docx` (or adapted into prose).

## Results Snapshot (Numbers from Tracked Runs)

### Long-Run Ecology (Qwen3-0.6B, paper pack)

From `docs/paper_packs/paper_qwen3_20251220_ecology_long150/summary.json`:

| condition | generations | episodes | merges | QD coverage (last) | holdout accuracy | holdout avg cost | ROI (last) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen base | 50 | 1,727 | 0 | 0.0% | 13.3% | 1.91 | 0.94 |
| single adapter | 50 | 63 | 0 | 0.0% | 13.3% | 1.93 | -0.06 |
| full ecology | 150 | 7,413 | 52 | 16.7% | 14.2% | 1.14 | 1.39 |

What this is (and isn’t):
- It’s mainly evidence of **economic / structural activity** (merges + niche coverage) and **cost efficiency** under an energy budget.
- The holdout accuracy lift here is real but small; the more striking change is the **cost-per-correct** improvement via the ecology.

### Compute-Matched SFT vs Evolution (Qwen/Qwen2.5-0.5B)

From `artifacts_grid_multiobj_full_ecology_full_20260129_131032/eval_id.json` and `artifacts_grid_multiobj_full_ecology_full_20260129_131032/ood_suite/summary.md`:

| setting | base | compute-matched SFT | evo (best single) | evo (routed, reselected) |
| --- | ---: | ---: | ---: | ---: |
| ID (6-family grid mix; 512 tasks) | 70/512 (13.7%) | 318/512 (62.1%) | – | 157/512 (30.7%) |
| OOD (paper holdout; 120 tasks) | 5/120 (4.2%) | 54/120 (45.0%) | 58/120 (48.3%) | 68/120 (56.7%) |

Fairness note:
- “routed, reselected” uses a small selection set to pick specialists (extra eval-time compute). “best single” is a strict single-pass baseline.

OOD per-family breakdown (paper holdout is 40 tasks each):

| family | base | SFT | evo (routed, reselected) |
| --- | ---: | ---: | ---: |
| `math` | 0/40 | 21/40 | 21/40 |
| `code.format` | 0/40 | 32/40 | 40/40 |
| `word.count` | 5/40 | 1/40 | 7/40 |

## Three Concrete Vignettes (Paper Holdout)

These are from `artifacts_grid_multiobj_full_ecology_full_20260129_131032/ood_suite/ood_paper_routed_reselect.json`.

In this run’s OOD reselection routing:
- `math` → `org_64383491`
- `code.format` → `org_83f255eb`
- `word.count` → `org_83f255eb`

### 1) Negative transfer example (`word.count`)

Prompt:
> Count the number of words in the sentence: 'scarcity routes learn'. Respond with an integer.

Target: `3`

Outputs:
- base: `The number of words in the sentence 'scarcity routes learn' is 3.` ✅
- SFT: `4` ❌
- evo (routed): `3` ✅

### 2) Output-constraint example (`code.format`)

Prompt:
> Convert the variable name `BudgetEnergyCeiling32` to snake_case. Respond with the new name only.

Target: `budget_energy_ceiling32`

Outputs:
- base: `The new name of the variable `BudgetEnergyCeiling32` is `budget_energy_ceiling_32`.` ❌
- SFT: `budget_energy_ceiling_32` ❌
- evo (routed): `budget_energy_ceiling32` ✅

### 3) “Knows it, but formats it wrong” (`math`)

Prompt:
> Compute 63 / 7. Respond with the number only.

Target: `9`

Outputs:
- base: `To compute 63 / 7, we divide 63 by 7. When we divide 63 by 7, we get 9. Therefore, the answer is 9.` ❌
- SFT: `9.` ✅
- evo (routed): `9\n63 / 7 = 9` ✅

(The base model “knows” the answer, but our evaluator extracts the *first* numeric token; if you restate the operands first, you fail.)

## Are the “paper ecology” results and “compute-matched SFT” results at odds?

Not really—they’re answering different questions:

- The paper pack run is mostly about: “does the ecology become a living economy (merges, niches), and can it lower cost under pressure?”
- The compute-matched SFT comparisons are about: “given a fixed training compute budget, which learning signal is more sample-efficient?”

In-distribution, dense supervised gradients tend to win (SFT). Under a family shift, routing/specialization can look less brittle than a single adapter (and sometimes beats SFT).
