# Evolora — Symbiotic LLM Ecology

TL;DR: Many tiny LoRA adapters live on a frozen Gemma‑270M host and compete for energy by solving small tasks. We measure reward‑on‑cost (ROI), gate merges by uplift and holdout checks, and adapt the energy floor automatically. The goal is to see if useful behaviours emerge via evolutionary pressure rather than full fine‑tuning.

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
- Latest 40‑gen smoke (Nov 10 2025, `artifacts_qwen3_endogenous_colonies_memory_20251110_114241`): avg ROI **1.68 ± 0.27**, 2 666 episodes, **32 merges / 83 trials / 29 promotions**, eval accuracy **19% (7/36, only math succeeds)**, no bankruptcies.
- Assimilation gating is now statistical: low_power 172, uplift_below_threshold 136, insufficient_scores 130, versus low_energy 58. Mean sample size 4.6 after dropping `min_window` to 2 and minting evidence tokens (1 615 minted / 209 spent), but power still misses targets ~60 % of the time.
- Team/colony path still stalled: 438 team routes yielded **0 promotions**, so no colonies formed and colony telemetry stayed at zero. We’ve added CI diagnostics and solver→checker handoffs so the next run can actually change the team mean.
- Policy system still inert (0/573 parses) and evaluation set remains misaligned (ΔROI ≈ ‑22). Documentation now calls this out so we stop assuming policies influence routing, and we’re refocusing the eval set + parser with the next run.
- macOS runner still emits `malloc … out of space` during heavy gens; keep runs to short/medium grids, set `MPLCONFIGDIR=$(mktemp -d)`, and monitor RSS when enabling more eval tasks.

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
- Policy prompt/parser now demand strict `key=value` pairs (with penalties for malformed replies) and include a KV fallback, so budget/reserve requests finally register instead of silently failing JSON repairs.
- Team probes share solver→checker handoffs (with configurable prompts) so the second member critiques or improves the first answer, giving the CI gate real variance to work with.
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

## How Learning Emerges (Core Hypothesis)

We freeze the base model (Gemma‑270M) and let many tiny LoRA adapters (“organelles”) evolve on top of it. They earn energy by solving tasks and spend energy to run. Selection pressure (energy, pricing, curriculum) and telemetry (ROI, holdouts, probes) create a closed loop where useful behaviours persist and poor ones go extinct—without back‑prop through the base.

What actually changes the model’s behaviour
- Reward‑modulated LoRA updates: after each episode, the adapter nudges toward activations that produced higher reward‑on‑cost and away from costly failures.
- Survival economics: energy tickets force a positive ROI; bankrupt organelles are culled and replaced.
- Curriculum pressure: a bandit + learning‑progress (ALP) mix routes tasks to where progress is steepest, steadily increasing difficulty and diversity.
- Assimilation: when an organelle’s uplift is statistically real on holdouts (doubly‑robust test), we merge its LoRA (optionally Fisher‑aware soup) or promote a trial offspring; successful children become first‑class organisms.

Beyond a single LoRA
- Communication: a paid message board lets organelles post/read hints (comms costs are energy‑governed). As comms traits evolve, a shared “language” can emerge if it increases team ROI.
- Colonies: when pairs consistently beat their solo baselines and reduce variance, they can pool energy under cautious reserve policies and reproduce as a team.
- Traits and mutation: inheritable traits (explore/read/post rates, budgets) and lightweight mutation operators (rank tweaks, low‑rank rotations) maintain diversity and exploration alongside Hebbian learning.

Does the environment change?
- Yes by design. Prices adapt to success rates, the curriculum shifts to high‑progress cells, and the economy throttles or boosts experimentation via energy floors/top‑ups. This makes the world a moving target, so “learning” means adapting fast enough to remain profitable and pass holdouts.

Population and diversity
- Population size is bounded (default 4–16) and managed by culling/replication; initial organelles differ via random seeds/ranks and rapidly diverge due to reward‑modulated updates and mutation. Diversity guardrails (energy Gini cap, species share caps, QD archive) prevent monoculture, preserving niches where specialists can evolve.


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

## Latest Results Snapshot — Qwen3 Endogenous (40 gens)
- Run: `artifacts_qwen3_endogenous_colonies_40_c2c_retry`
- ROI: mean 2.63 (median 2.69; min/max 0.044/3.84); avg energy cost 0.75
- Energy balances: mean 2.83 (min/max 2.38/3.07); ticket still 0.5
- Teaming: 438 team routes, 0 promotions; top co-routing pairs repeat across gens (e.g., `org_7f9af5a9:org_a59406d2`)
- Team probes: sustained synergy candidates now reported (none met 3-window threshold in this run)
- Few-shot scaffolds: enabled for word.count, logic.bool, math, and math.sequence task families
- Assimilation: 13 attempts (1 pass, 12 fails); gating led by `insufficient_scores` (406) and `uplift_below_threshold` (129); top-ups credited 115 times (508 already_sufficient)
- Trials/promotions: 12 trials created; 1 promotion via assimilation
- Colonies: none held this run (bandwidth remains available for next trials)
- Policy parse: 0/578 (0.0%) — strict JSON + KV fallbacks still miss on Qwen‑0.6B
- Power: uplift proxy mean 0.30 (target 0.8); evidence tokens mint/usage now tracked in summaries
- Evaluation: 22.22% (8/36) on the 10-gen cadence

Interpretation
- ROI now comfortably >2× cost, so we can afford more aggressive gating or curriculum pressure.
- Assimilation remains evidence-limited; lowering `min_window` or increasing minted evidence may accelerate merges.
- Policy path is still silent; next step is stricter prompting or pruning allowed fields to improve parse rate.
- Colonies did not activate this run — team routing is working, so we can raise colony bandwidth or relax expansion gates once evidence arrives.

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
