# Evolora — Symbiotic LLM Ecology

TL;DR: Many tiny LoRA adapters live on a frozen small host model (Gemma‑270M or Qwen3‑0.6B, depending on the experiment) and compete for energy by solving small tasks. We measure reward‑on‑cost (ROI), gate merges by uplift and holdout checks, and adapt the energy floor automatically. The goal is to see if useful behaviours emerge via evolutionary pressure rather than full fine‑tuning.

## Why We’re Doing This (Human-Sized Version)
We want to see whether a colony of tiny LoRA adapters can learn useful behaviour *without* back-propagating through the whole model. Think of the base Gemma-270M host as an ecosystem and each adapter as an organism. They earn “energy” by solving small tasks, spend it on inference, and occasionally merge if they consistently outperform their peers. Our goal is to nudge this colony from random noise to adaptive cooperation, the way life crept out of the Hadean soup into the Cambrian explosion.

## How It Works — in Plain English
1. **Spawn organelles**: we start with a handful of LoRA adapters attached to a frozen Gemma host.
2. **Run tasks**: each generation, organelles tackle math, logic, sorting, sequences, and JSON repair puzzles. Good answers earn credits; bad ones lose energy.
3. **Survive or starve**: energy tickets enforce scarcity. Low-performing organelles run out of fuel and are culled; promising ones accumulate reserves.
4. **Assimilate**: when an organelle keeps winning, we run probes, compare its scores to recent history, and (optionally) merge it with high-ROI neighbours using an SVD-aware LoRA soup.
5. **Log everything**: ROI, energy, assimilation gating reasons, the most recent gating snapshots, diversity metrics, and holdout verdicts are written to `artifacts_*` directories for later analysis.

If the colony keeps earning more than it spends, the population should drift toward better answers all on its own.

## Current Status
- Latest ecology run on Qwen3‑0.6B (`artifacts_paper_qwen3_ecology_20251202_120102`, 50 gens): ROI **~0.78** avg (up to ~1.50), **28 merges**, **82 assimilation events / 13 passes**, power mean **~0.84**, uplift CIs basically never include zero, eval **2/3**. Evidence tokens minted/used **1533 / 160**, knowledge cache writes/reads/hits **42 / 4 / 4**, colonies size **2** and stable, QD archive coverage **~21 %**.
- Frozen‑base baseline (`artifacts_paper_qwen3_frozen_20251201_205728`): same host/tasks, no merges or colonies, ROI **~0.82** with solid but static performance; no structural change, no QD coverage, no knowledge usage.
- Single‑adapter baseline (`artifacts_paper_qwen3_single_20251203_120617`): one organelle with extra energy but no ecology. It stays alive, logs a trickle of episodes (22 across 50 gens), and hovers near zero ROI with no merges, no QD coverage, and almost no cache use. Good sanity check that “just one LoRA” doesn’t spontaneously turn into an economy.
- Earlier long run (`artifacts_qwen3_endogenous_colonies_memory_a_20251124_150138`, gens 71–100) shows the same story at 100‑gen scale: ROI **~1.26**, **33 merges**, **459 assimilation tests / 67 passes**, power **0.85**, eval **1/1**, policies parse 100 %, budgets ~45, evidence tokens **1191 / 410**.
- Gates are now dominated by `low_power` and `insufficient_scores` rather than `no_activity`/`no_best_cell` starvation. Population refreshes only fire when merges stall for a while, which is what we want.
- Team routing promotes selectively (tens of promotions over hundreds of team routes) with stricter handoff and diversity guards; colonies remain small (size 2) but actually do something useful.
- Long runs are resumable via checkpoints: you can split runs into small chunks and resume the same `run_id` without restarting the population (see commands below). Keep using `MPLCONFIGDIR=$(mktemp -d)` to dodge font-cache crashes on macOS.

### What this is teaching me so far
- A frozen base model plus a single LoRA (even with a bit of energy and a memory cache) mostly just jitters around: you get a few episodes, near‑zero ROI, and no real structural change.
- A frozen base plus a *population* of LoRAs with energy, merges, holdouts, and a small memory+policy+team loop does something qualitatively different: the system keeps running, keeps earning, and keeps rewriting its own “org chart” via merges and colonies.
- The economy matters as much as the learning rule. Ticket prices, energy floors, bankruptcy rules, and evidence tokens decide who even gets to be sampled; the Hebbian/LORA updates then climb whatever hill that economy exposes.
- You can get long‑horizon adaptation on a tiny host without ever touching its base weights, as long as you give the adapters somewhere to live (an ecology), a way to compete (energy), and a cautious, statistics‑aware path to merge.

### Running long evaluations safely (resumable)
```bash
run_id=artifacts_qwen3_endogenous_colonies_memory_a_$(date +%Y%m%d_%H%M%S)

# First chunk (e.g., 20–50 gens)
MPLCONFIGDIR=$(mktemp -d) AGENT_MODE=baseline .venv311/bin/python scripts/eval_gemma_long.py \
  --config config/experiments/qwen3_endogenous.yaml \
  --generations 50 \
  --output "$run_id" \
  --checkpoint-every 5

# Analyzer (only after a successful eval)
MPLCONFIGDIR=$(mktemp -d) .venv311/bin/python scripts/analyze_ecology_run.py "$run_id" --plots --report

# Continue the same population/state
MPLCONFIGDIR=$(mktemp -d) AGENT_MODE=baseline .venv311/bin/python scripts/eval_gemma_long.py \
  --config config/experiments/qwen3_endogenous.yaml \
  --generations 20 \
  --output "$run_id" \
  --resume-from "$run_id" \
  --checkpoint-every 5
```
If a chunk is killed by macOS, rerun the same `--resume-from` command; it will pick up from the last checkpoint in `run_id/checkpoint.pt`.

## What’s New (Oct 2025)
- Policy hook (per‑org JSON policy) with energy micro‑cost. Policies can bias routing (cell_pref), adjust per‑org budget_frac, enable comms, and set a small gate_bias delta. Analyzer shows policy usage and ROI when policy is on vs off.
- Colonies (team mode) with cautious promotion/shrink and priced comms. Per‑colony bandwidth fraction + read/post caps; analyzer shows colonies over time and average size.
- Co-routing probes per generation to surface synergistic pairs; optional team router can route a few best-of-two episodes per generation and reports `team_routes`/`team_promotions`.
- Colony-level selection now pools dissolved members and pot, rewards the top colony, and replicates with inherited funds. Multi-tier migration tracks `tier` promotions/demotions, boosts high-tier bandwidth, and shows up in `colony_tier_{mean,counts}.png` plus analyzer summaries so you can see when groups level up.
- EvoScope dashboard now ships with inline Chart.js plots (ROI, merges/promotions, colony health, gating distribution) and richer assimilation diagnostics. Run `scripts/evoscope.py artifacts_*` to generate an interactive `index.html` for any run directory.
- Mutation operators diversified: genomes now evolve layer-specific rank noise, adapter dropout masks, and duplication boosts. Telemetry tracks per-gen counts and analyzer plots `mutation_operators.png` to show which operators fire.
- Merge audits are first-class: each successful merge records pre/post ROI and ΔROI along with task counts; analyzer emits `merge_audits_counts.png` + `merge_audits_delta.png`, and `scripts/merge_audit.py` surfaces regressions straight from run artifacts.
- Curriculum refresh: word-count tasks now include adversarial HTML/digit formats, plus new `math.multi_step` and `code.format` families to stress multi-operation reasoning and snake_case formatting; evaluation per-family metrics show how each band fares.
- MAP-Elites coverage is now first-class: merge candidate ranking blends ROI/EMA with novelty, the archive keeps a novelty-weighted elite set with configurable caps, analyzer plots archive size & coverage, and EvoScope surfaces top bins so you can spot emerging specialists at a glance.
- Knowledge tokens: each organelle maintains a small energy-priced memory cache; successful answers persist as short hints, the loop prepends them during routing, and analyzer reports memory read/write rates so you can gauge usefulness.
- Assimilation history: every organelle/cell pair retains a configurable trail of uplift + probe stats; summaries/analyzer expose the latest slices so dashboards can track where real learning is happening (and the telemetry survives restarts).
- Promotion guardrails relaxed: holdout windows and statistical-power thresholds now match the small synthetic batch sizes, and reserve/hazard guards no longer block every attempt. Colonies are re-enabled so team routing can actually promote synergistic pairs.
- Retrofitted recurrence: the host can now run an organelle through multiple internal reasoning passes per episode (`host.recurrence_*`). Training episodes default to two passes while holdouts/evals stay at one; scratchpad history is appended automatically and telemetry reports `recurrent_passes` so we can measure the extra compute budget.
- Evaluation/holdout set retargeted to math, word-count, and code-formatting cells (short/medium) with a smaller sample size and reduced cadence, so we measure the same skills we train without draining ROI.
- Policy prompt/parser now demand strict `key=value` pairs (with penalties for malformed replies) and include a KV fallback plus a success bonus, so budget/reserve requests finally register instead of silently failing JSON repairs.
- Team probes share solver→checker handoffs (with configurable prompts) so the second member critiques or improves the first answer, giving the CI gate real variance to work with.
- Curriculum mix now leans more heavily on learning-progress routing, and controller difficulty is clamped to keep cells from running away from the colony.
- Fisher-aware LoRA soups: assimilation weighting now uses activation-derived Fisher importances (with fallback to adapter energy) so high-signal organelles dominate merges; analyzer surfaces mean/max Fisher importance alongside merge weight stats.
- Evidence auto-tune for assimilation windows to reduce “insufficient_scores”. DR small-n deferral (low_power_dr) grants evidence credit instead of forcing bad calls.
- Doubly‑robust uplift with stratified holdout telemetry (method, power, strata), plus snapshots of assimilation gates and attempts.
- Winter stress cycles: configurable price/ticket pulses with reserve bonuses on thaw. Analyzer now emits `winter_cycle.png`, `winter_events.png`, and `winter_events.jsonl`, and reports post‑winter ROI and assimilation recovery deltas so you can see whether the colony bounces back.
- Foraging traits + Q-routing: organelles evolve `beta`, `q_decay`, `ucb_bonus`, and budget aggressiveness; Q-values update per cell, policy bias stays bounded, and analyzer reports `foraging_traits.png` plus top cell tables.
- Learning-progress curriculum heatmap (LP) and smoothed lp_mix. QD coverage readout (optional).
- Adaptive relief: τ/ROI guardrails relax when merges stall; analyzer surfaces relief snapshots.

## Next Hypotheses & Instrumentation (Nov 2025)
1. **Team acceptance trace** — instrument `_maybe_team_probes` / `_team_accept` to log tasks/CI/power gate outcomes so we can explain the current **0/438** team promotions and tune `team_probe_variance_nu`, `team_min_tasks`, and `team_min_power` with data.
2. **Policy parser reality check** — capture raw policy outputs + parser failures, then A/B a KV-only fallback prompt for `budget_frac` / `reserve_ratio`. Hypothesis: Qwen‑0.6B needs structured hints, not strict JSON.
3. **Evaluation alignment** — replicate the short/medium-only change from holdouts into `config/evaluation/*.jsonl` and remeasure. If accuracy jumps above 19%, keep the easier eval; otherwise lighten training tasks instead.
4. **Memory pressure audit** — log RSS + `metrics.in_memory_log_limit` deltas each gen to pinpoint the `malloc` spikes; trim `evaluation.sample_size` or `max_episode_steps` if holdouts/evals drive the peaks.

## Evolution Glossary
| Term | In-code meaning | Evolution analogy |
| --- | --- | --- |
| Relaxed sandbox | Low-cost config (`gemma_relaxed*`) that lets organelles experiment freely | Nutrient-rich tidal pool |
| Energy ticket (`m`) | Baseline energy debit charged each generation | Daily caloric requirement |
| Energy top-up | Auto-mint that credits high-ROI organelles to reach the merge gate | A sunny photosynthesis burst |
| ROI (reward-on-cost) | Profit per unit energy spent on tasks | Fitness per calorie |
| Uplift | Difference between recent control vs. treatment scores during assimilation | Selective advantage over peers |
| Holdout trial | Assimilation check on reserved evaluation prompts | Migration challenge to a new habitat |
| Bandit routing | UCB-based prompt allocator that balances exploration/exploitation | Foraging instinct steering the swarm |
| Diversity guard | Energy Gini cap and species quotas | Predator-prey balance keeping the biome varied |
| Assimilation | LoRA soup/merge when uplift and probes pass | Symbiosis or gene exchange between lineages |

## Architecture Overview (Technical Layer)
- **Host & organelles**: `src/symbiont_ecology/host` wraps the Gemma backbone; `symbiont_ecology/organelles/peft_hebbian.py` implements Hebbian LoRA adapters with reward-modulated updates.
- **Environment**: `symbiont_ecology/environment/grid.py` spins up math, logic, sorting, sequence, and JSON repair tasks and routes prompts with a UCB-style bandit; `environment/loops.py` orchestrates generations, energy settlement, assimilation, and diversity guardrails.
- **Economics & diversity**: `config/experiments/gemma_relaxed.yaml` (long runs) and `config/experiments/gemma_relaxed_debug.yaml` (diagnostics) encode tickets, reward bonuses, assimilation knobs, and species energy caps.
- **Telemetry**: episodes land in `artifacts_*/episodes.jsonl`; generation summaries (ROI, gating stats, diversity snapshot) go to `gen_summaries.jsonl`; assimilation events stream into `assimilation.jsonl`.
- **Analysis toolkit**: `scripts/analyze_ecology_run.py` summarises runs, reports ROI volatility, energy Gini/effective population, assimilation gates + audits, and shows why merges passed or failed. Use `scripts/merge_audit.py <run_dir>` to list the worst merge deltas or export them for deeper inspection.

## Getting Started
1. **Python**: use Python 3.11 (we develop under `.venv311`).
2. **Install dependencies**
   ```bash
   python3.11 -m venv .venv311
   source .venv311/bin/activate
   pip install --upgrade pip
   pip install -r requirements-dev.txt
   ```
3. **Optional tooling**: install [Beads](https://github.com/steveyegge/beads) (`bd`) for local issue tracking, and `make` for shortcuts.

## Running the Ecology
- **Smoke test**
  ```bash
  AGENT_MODE=baseline make test VENV=.venv311
  ```
- **Short baseline run**
  ```bash
  MPLCONFIGDIR=$(mktemp -d) \
    AGENT_MODE=baseline .venv311/bin/python scripts/eval_gemma.py \
      --config config/experiments/gemma_relaxed.yaml \
      --generations 10 --batch-size 2 \
      --output artifacts_gemma_smoke
  ```
- **Diagnostics run (≤20 generations, relaxed energy gate)**  
  Use this before any overnight job to confirm merges can fire and to inspect the new telemetry snapshots.
  ```bash
  MPLCONFIGDIR=$(mktemp -d) \
    AGENT_MODE=baseline .venv311/bin/python scripts/eval_gemma_long.py \
      --config config/experiments/gemma_relaxed_debug.yaml \
      --generations 20 --batch-size 2 \
      --output artifacts_gemma_debug
  ```
- **100-generation study**
  ```bash
  MPLCONFIGDIR=$(mktemp -d) \
    AGENT_MODE=baseline .venv311/bin/python scripts/eval_gemma_long.py \
      --config config/experiments/gemma_relaxed.yaml \
      --generations 100 --batch-size 2 \
      --output artifacts_gemma_relaxed_autotune_v5
  ```

## Analysing Results
- Quick summary (no plots):
  ```bash
  MPLCONFIGDIR=$(mktemp -d) \
    .venv311/bin/python scripts/analyze_ecology_run.py \
      artifacts_gemma_relaxed_autotune_v5 --report
  ```
- Report + visuals:
  ```bash
  MPLCONFIGDIR=$(mktemp -d) \
    .venv311/bin/python scripts/analyze_ecology_run.py \
      artifacts_gemma_relaxed_autotune_v5 --plots --report
  ```
The report now highlights ROI stats, assimilation gating totals, **recent gating snapshots**, and **the last few assimilation attempts** (control/treatment means, uplift vs. threshold, holdout verdict, top-up status), plus the diversity metrics (energy Gini, effective population, species share) and guard decay counters. Colony summaries add size/bandwidth timelines, mean ΔROI, and variance ratios so you can see when teams expand or shrink.

## Key Configuration Knobs
- `energy.m`, `energy.cost_scale`: ticket price and cost damping.
- `environment.success_reward_bonus`, `environment.failure_cost_multiplier`: steer survival pressure.
- `environment.*`: per-org budgeting (`budget_*` knobs) blends energy ratio, evolved traits, and policy `budget_frac`; `global_episode_cap` enforces a hard ceiling per generation and shows up in telemetry when hit.
- `assimilation_tuning.*`: guard baselines, probe requirements, holdout settings, LoRA soup behaviour, **`energy_topup_roi_bonus` for lenient top-ups**, and `gating_snapshot_limit` for telemetry retention.
- **Assimilation sanity checks (why merges can stall):**
  1. **Evidence quota:** if `min_window` × `holdout_sample_size` exceeds the tasks an organelle sees per generation, every attempt will die as “insufficient_scores”. We now keep the quota in line with `environment.synthetic_batch_size`.
  2. **Uplift margin:** noisy math tasks rarely clear a large `holdout_margin`. When we need the pipeline to move, we temporarily lower the margin or feed easier cells, then tighten it once merges start landing.
  3. **Power target:** demanding 80% statistical power on tiny samples is wishful thinking. Dropping `power_target` (and letting evidence tokens bridge bigger gaps) keeps the hypothesis test from vetoing every attempt.
  4. **Reserve/hazard guard:** safety rails are useful, but if they disable probes/assimilation whenever an organelle dips into reserve, nothing can promote. Lowering `reserve_ratio`, loosening the hazard band, or turning off `hazard_probe_disable` gives the population a chance to evolve before the guard kills experimentation.
- `assimilation_tuning.colony_*`: synergy thresholds, expand/shrink windows, and bandwidth budgets (`colony_expand_delta`, `colony_expand_windows`, `colony_bandwidth_base`, `colony_hazard_bandwidth_scale`, etc.) so colonies grow only when holdout ROI keeps improving and comms spend stays within the pot.
- `grid.families`: choose which mix of math, logic, sorting, sequence, and JSON cells make up the curriculum.
- `diversity.*`: energy Gini cap and per-species share limit.
- `policy.*`: turn on per-org policy proposals (`enabled`), cap prompt size (`token_cap`), and charge a micro-cost for policy requests (`energy_cost`, optionally scaled by tokens with `charge_tokens`). Analyzer reports policy usage and ROI when policy is on vs off.
- `foraging.*`: flip on trait-driven routing (`enabled`), set defaults for `beta`, `q_decay`, `ucb_bonus`, and clamp policy bias (`policy_bias_cap`); analyzer surfaces `foraging_traits.png` and top-Q cells when enabled.
- `survival.*`: configure reserve thresholds (`reserve_ratio`, `reserve_cost_beta`), batch throttling, and hazard response (z-score window, rank downshift, probe suppression, top-up bonuses).
- `comms.*`: tune per-gen post/read caps, credit window, and power delta thresholds for teacher credit; `credit_frac` sets the energy share posters receive when a reader improves.
- `winter.*`: schedule price/ticket pulses (`winter_interval`, `winter_duration`, multipliers), grant post-winter energy bonuses, and tune analyzer-friendly metrics to study how ROI and assimilation recover after stress events.

Tweak `config/experiments/gemma_relaxed.yaml`, rerun the long evaluation, and compare `gen_summaries.jsonl` across artifacts.

## Repo Layout
```
src/                 core code (host, environment, evolution, metrics)
scripts/             CLI automation (bootstrap, eval, analysis)
config/              experiment YAMLs and holdout tasks
tests/               pytest suite (economics, morphogenesis, assimilation)
tests_llm_live/      placeholder for live LLM golden tests
artifacts_*/         run outputs (gitignored telemetry & plots)
WORK_LOG.md          running notes for long experiments
```

## Development Workflow
1. Track work with Beads (`bd create`, `bd list`) — issues live under `.beads/` (gitignored).
2. Make changes in a feature branch, keep commits Conventional Commit style (`feat:`, `fix:`, `chore:`...).
3. Run the fast gates locally: `make test` (includes pytest) and optionally lint via `ruff`/`black`.
4. For long experiments, append a short entry to `WORK_LOG.md` with command, config, and summary.
5. Analyse runs with the analysis script and stash artefacts under the gitignored `artifacts_*` directories.

## Current Focus / Next Experiments
- **Assimilation success**: we see high ROI but merges still fail. Use the debug run + new telemetry snapshots to inspect top-up status, uplift deltas, and holdout batches before tweaking thresholds.
- **Skill metrics**: evaluation accuracy remains 0/60. Consider curriculum changes or additional task families.
- **Human bandit calibration**: sweep `preference_weight`, `helper_weight`, and `frequency` once the ecology stabilises; watch the impact on ROI volatility.
- **Telemetry dashboards**: the data already exists in `gen_summaries.jsonl` — wiring a live dashboard is the natural follow-up.

## Contributing
Pull requests welcome! Please:
- open an issue (or `bd create`) before large feature work,
- keep gitignored paths (artifacts, virtualenvs, caches) out of commits,
- run the full test suite (`AGENT_MODE=baseline make test VENV=.venv311`),
- run the survival regression (`tests/test_survival_resilience.py`) after touching economics,
- include analysis notes or updated documentation when behaviour changes.

Happy evolving — let’s see if this Cambrian-era colony learns something genuinely new.

## Quick Start (Qwen3‑0.6B)
- 40‑gen smoke (endogenous config, policy + colonies + auto‑tune on):
  - `MPLCONFIGDIR=$(mktemp -d) AGENT_MODE=baseline .venv311/bin/python scripts/eval_gemma_long.py --config config/experiments/qwen3_endogenous.yaml --generations 40 --output artifacts_qwen3_endogenous_flags_40`
- Simple ladder (short cells; fast):
  - `MPLCONFIGDIR=$(mktemp -d) AGENT_MODE=baseline .venv311/bin/python scripts/eval_gemma_long.py --config config/experiments/qwen3_simple.yaml --generations 40 --output artifacts_qwen3_simple_40`
- Analyze any run dir:
  - `.venv311/bin/python scripts/analyze_ecology_run.py <run_dir> --plots --report`
- Inspect merge audits:
  - `.venv311/bin/python scripts/merge_audit.py <run_dir> --top 10`

### Paper‑style ecology suite (three baselines)

To reproduce the three Qwen3‑0.6B runs used in the baselines comparison:

```bash
scripts/run_paper_ecology_suite.sh all
```

This will sequentially run:
- frozen base: `paper_qwen3_frozen.yaml`
- single adapter: `paper_qwen3_single.yaml`
- full ecology: `paper_qwen3_ecology.yaml`

and write separate `artifacts_paper_qwen3_*_<timestamp>` directories for each, with plots and a `report.md` under each root.

What to watch in the ticker each generation
- `ROI`: keep mean in a healthy band (>1.3 desirable on Qwen3 smoke).
- `merges`: accepted assimilations (can be 0–3 in 40‑gen smoke).
- `energy floor` and `(ROI≥x)`: how strict the gate is and the ROI needed for top‑ups.
- `gating`: which reasons dominated (low_energy, low_power, low_power_dr, insufficient_scores, …).
  Watch for `reserve_guard` and `cautious_skip` now that survival guards can pause merges.
- `trials/promotions`: trial offspring created and team promotions.
- `eval a/b`: holdout accuracy if scheduled.

Analyzer insights (after the run)
- Colonies count (min–max) and average size: ensures team behavior isn’t collapsing to monoculture.
- Policy usage: gens with policy, parse counts, ROI when policy on vs off, and field usage summary.
- Survival safeguards: reserve-active, hazard-active, and price-bias counts plus event breakdown (`reserve_enter`, `hazard_enter`, `hazard_exit`, `cautious_skip`).
- Comms telemetry: posts/reads/credits per generation and `comms_events` (reads, credits) to monitor teacher credit flow.
- Power economics: average power need, price multiplier, evidence tokens minted/used, and info-aware top-ups so we can see whether the economy is rewarding information.
- Foraging snapshot: mean trait drift (beta/decay/ucb/budget) and latest top-Q cells per organelle; plots land in `foraging_traits.png`.
- Mutation telemetry: operator counts per generation with `mutation_operators.png`, plus `mutation_totals` in the summary so you can see how often rank noise, dropout masks, and duplication boosts fire.
- Merge audits: `merge_audits_counts.png` + `merge_audits_delta.png` chart pass/fail impact; raw records live in the summary and are inspectable via `scripts/merge_audit.py <run_dir>`.
- Evaluation breakdown: per-family accuracy/ROI deltas show which curriculum bands regress or improve across the new tasks.
- Reserve & winter telemetry: reserve freezes, colony winter mode counts, hazard z-score traces, plus price/ticket multiplier traces and winter event JSON (`winter_cycle.png`, `winter_events.png`, `winter_events.jsonl`). Post-winter ROI/assim deltas show whether the system rebounds after each cold spell.
- Assimilation: events summary, mean sample size, CI-excludes-zero share, mean power; gating totals and sample reasons (look for `low_power_dr`).
- LP heatmap and cell difficulty/price heatmaps: confirms curriculum pressure and pricing dynamics.

## Baselines vs Ecology (Qwen3‑0.6B, 50 gens)
To understand whether the ecology is actually doing anything and not just adding complexity, it helps to look at three matched setups on the same host and task mix.

### 1. Frozen base (no adaptation)
- Config: `config/experiments/paper_qwen3_frozen.yaml`
- Run: `artifacts_paper_qwen3_frozen_20251201_205728` (50 generations)
- Behaviour:
  - Episodes: ~1.3k total.
  - ROI: mean ≈ **0.82**, fairly stable.
  - Structure: **0 merges**, no colonies, empty QD archive, knowledge cache off.
  - Evaluation: small holdout sample sits around **2/3** accuracy.
- Takeaway: the frozen host plus this curriculum already gives decent reward, but nothing structural happens. It’s a solid control: no emergent structure, no exploration of the archive, no memory or team behaviour.

### 2. Single adapter (one organelle with a bit of learning)
- Config: `config/experiments/paper_qwen3_single.yaml`
- Run: `artifacts_paper_qwen3_single_20251203_120617` (50 generations)
- Behaviour:
  - Episodes: **22** total across 50 gens.
  - ROI: mean ≈ **0.06**, with a couple of early positive spikes and then a long tail near zero.
  - Structure: **0 merges** (disabled), no colonies, empty QD archive.
  - Memory: a handful of cache writes/reads; hits exactly a few times; nothing sustained.
  - Evaluation: also around **2/3** on the tiny holdout.
- Takeaway: one LoRA on a frozen host, even with extra energy and a memory cache, mostly jitters. It learns a bit, then idles. You don’t get anything that looks like long‑horizon, population‑level improvement.

### 3. Full ecology (many organelles + economy)
- Config: `config/experiments/paper_qwen3_ecology.yaml`
- Run: `artifacts_paper_qwen3_ecology_20251202_120102` (50 generations)
- Behaviour:
  - Episodes: ~1.7k total (more than either baseline).
  - ROI: mean ≈ **0.78**, with a band of healthy high‑ROI generations.
  - Structure:
    - **28 merges** plus 66 trials, with **82** assimilation events and **13** accepted.
    - Colonies: one colony of size 2 throughout, with real holdout passes and pot updates.
    - QD archive: average size ≈ 4, coverage ≈ **21 %**, with distinct bins for math and word.count.
  - Memory and policy:
    - Knowledge cache: **42** writes, **4** reads, **4** hits; it actually gets used.
    - Policy channel: active every generation; 100 % parse rate; budgets and routing respond to policy outputs.
  - Evidence tokens and power:
    - Evidence tokens: **1533 / 160** minted/used; uplift tests almost never include zero once tokens accrue.
    - Gating reasons: `low_power`, `insufficient_scores`, and `uplift_below_threshold` dominate, not `no_activity`/`no_best_cell`.
  - Evaluation: again about **2/3** on the small holdout, but now with a much richer internal life.
- Takeaway: with the ecology turned on, the system keeps rewriting itself. Organelles earn and spend energy, merge, and spawn offspring; a small colony maintains shared pot/bandwidth; the archive fills in; the knowledge cache and policy channel are exercised. It’s still a small model on simple tasks, but it’s clearly doing more than “answer questions and stop.”

### What this comparison suggests
- The frozen and single‑adapter baselines are useful sanity checks: they confirm that the base model is competent, and that a lone adapter doesn’t spontaneously generate an ecosystem.
- The ecology run shows that you can get ongoing, inspectable adaptation on top of a frozen host:
  - **Who** gets to act (energy, budgets, and policy decisions),
  - **Which** organelles survive and merge (uplift + holdouts),
  - **Where** the population explores (QD coverage, colonies, co‑routing),
  all change over time even though the backbone weights never move.
- In other words, the “learning rule” is small and shared (a tiny Hebbian update), but the selection pressure and energy flow are not. Most of the interesting behaviour comes from the ecology and economy wrapped around the host, not from making the host bigger.

See also: `docs/ecology_overview.md` for a compact numeric table and conceptual framing of these three runs.

Suggested next nudges (short runs)
- If insufficient_scores remains high: temporarily reduce `assimilation_tuning.min_window` by 2 (e.g., 8→6) to accelerate attempts, then revert after merges increase.
- If macOS memory pressure reappears: lower `host.gen_max_new_tokens` (e.g., 48→32) or reduce `evaluation.sample_size` slightly; keep `metrics.in_memory_log_limit: 0`.
- Improve policy parsing: use a stricter JSON‑only system instruction or a tolerant extractor to increase parse rate before raising `policy.bias_strength` further.

## Run & Observe
- Debug (20 generations, rich telemetry):
  - `MPLCONFIGDIR=$(mktemp -d) AGENT_MODE=baseline .venv311/bin/python scripts/eval_gemma_long.py --config config/experiments/gemma_relaxed_debug.yaml --generations 20 --batch-size 2 --output artifacts_gemma_debug_latest`
- Long (overnight, ~24–30h, 120–150 gens):
  - `MPLCONFIGDIR=$(mktemp -d) AGENT_MODE=baseline .venv311/bin/python scripts/eval_gemma_long.py --config config/experiments/gemma_relaxed_plus.yaml --generations 150 --batch-size 2 --output artifacts_gemma_relaxed_plus_latest`
- Analyse any run dir:
  - `.venv311/bin/python scripts/analyze_ecology_run.py <run_dir> --plots --report`
  - Optional HTML dashboard: `.venv311/bin/python scripts/evoscope.py <run_dir>` then open `<run_dir>/index.html`
  - Optional animation (timeline GIF): `.venv311/bin/python scripts/evoscope_anim.py <run_dir> --gif` (writes `<run_dir>/timeline.gif`)
- macOS tip: set `MPLCONFIGDIR=$(mktemp -d)` when running analysis or long training to avoid font cache writes and occasional malloc pressure. If runs still hit `malloc: Failed to allocate segment`, export `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (for Metal) and raise the file descriptor limit (`ulimit -n 4096`) before launching long jobs.

Inline animation (optional)
- If you want the animation visible directly on GitHub, copy a generated `timeline.gif` into `docs/timeline.gif` and it will render inline here:

<p align="center">
  <img src="docs/timeline.gif" alt="Evolution timeline (ROI and merges over generations)" width="720"/>
</p>

During runs, the progress ticker prints what actually matters each generation:
`Generation 090 | ROI 0.932 | merges 0 | energy floor 0.61 (ROI≥1.00) | gating low-energy 14 cooldown 1 | episodes 28 | eval 0.000 (0/12)`

- ROI: average reward-on-cost this gen
- merges: accepted assimilation merges
- energy floor / ROI≥x: current auto‑tuned energy gate and ROI needed for top‑ups
- gating: most frequent assimilation gates hit this gen
- episodes: episode count processed this gen
- eval: holdout accuracy if scheduled for this gen

Note: noisy Transformer/PEFT warnings and pad‑token notices are suppressed in `scripts/eval_gemma_long.py` to keep the ticker readable.

Analyzer notes: `scripts/analyze_ecology_run.py` reports assimilation power and may show a `low_power` gating total when uplift tests defer. If `low_power` dominates, extend runs slightly or increase evidence windows (config knobs) to accumulate more samples per attempt.

### Colony inference (best‑of‑two)

You can query a small “colony” (2 members) directly and see the combined response.

CLI:

```
.venv311/bin/python scripts/colony_infer.py \
  --config config/experiments/qwen3_endogenous.yaml \
  --members auto \
  --prompt "Sort: pear banana apple"
```

Python API (inside a run):

```
result = loop.run_colony_inference(["org_a", "org_b"], "Sort: pear banana apple")
print(result["selected_id"], result["selected_answer"])  # per‑member answers in result["answers"]
```

Note: For querying previously trained adapters, add persistence for organelle adapter states and reload before calling.

### Latest Snapshot (Qwen endogenous, 40 gens — “_40_b”)

Key numbers (quick glance):
- ROI mean 2.39 (min 0.00, max 3.68)
- Team routes 220, promotions 0
- Merges 0; Eval 22.22% (8/36)

Plots (click to open):
- Team routes: `artifacts_qwen3_endogenous_colonies_40_b/visuals/team_routes.png`
- Team promotions: `artifacts_qwen3_endogenous_colonies_40_b/visuals/team_promotions.png`
- Co‑routing heatmap: `artifacts_qwen3_endogenous_colonies_40_b/visuals/co_routing_heatmap.png`
- Colonies count: `artifacts_qwen3_endogenous_colonies_40_b/visuals/colonies_count.png`
- Colony pot (total): `artifacts_qwen3_endogenous_colonies_40_b/visuals/colonies_pot_total.png`
- Avg ROI: `artifacts_qwen3_endogenous_colonies_40_b/visuals/avg_roi.png`

Reproduce analysis:

```
MPLCONFIGDIR=$(mktemp -d) AGENT_MODE=baseline .venv311/bin/python \
  scripts/analyze_ecology_run.py artifacts_qwen3_endogenous_colonies_40_b --plots --report
```

### Teams & Policy quick run (Qwen‑0.6B)

```bash
MPLCONFIGDIR=$(mktemp -d) AGENT_MODE=baseline .venv311/bin/python scripts/eval_gemma_long.py \
  --config config/experiments/qwen3_endogenous.yaml \
  --generations 40 \
  --output artifacts_qwen3_endogenous_colonies_40

.venv311/bin/python scripts/analyze_ecology_run.py artifacts_qwen3_endogenous_colonies_40 --plots --report
```

## Tuning Cheatsheet
- Faster assimilation cadence (smoke only; revert for long stability):
  - `assimilation_tuning.min_window: 8–10`, `probe_max_other_cells: 1–2`, `holdout_margin: 0.02–0.03`.
- More exploration: lower `controller.tau` slightly; increase `curriculum.lp_mix_max`.
- Reduce stalls from small‑n evidence: keep DR on, accept deferral (`low_power_dr`) and let auto‑tune shrink windows; don’t force merges.
- Strengthen policy influence: `policy.bias_strength: 0.4–0.6` while parse rate is healthy.
- Team router & co‑routing:
  - `assimilation_tuning.team_router_enabled: true`
  - `assimilation_tuning.team_max_routes_per_gen: 4–8`
  - `assimilation_tuning.team_routing_probe_per_gen: 1–3`
