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
- Long run (300 gens): avg ROI ≈ 0.94; merges 0; promotions 0; eval 0/360.
- Assimilation blockers: low_energy and low statistical power dominate; uplift windows too short; top‑ups often ROI‑blocked.
- Recent changes: added low‑power gating (defer decisions), larger evidence windows, easier evidence top‑ups, and trial offspring with promotion checks.

## What’s New (Oct 2025)
- Policy hook (per‑org JSON policy) with energy micro‑cost. Policies can bias routing (cell_pref), adjust per‑org budget_frac, enable comms, and set a small gate_bias delta. Analyzer shows policy usage and ROI when policy is on vs off.
- Colonies (team mode) with cautious promotion/shrink and priced comms. Per‑colony bandwidth fraction + read/post caps; analyzer shows colonies over time and average size.
- Co‑routing probes per generation to surface synergistic pairs; optional team router can route a few best‑of‑two episodes per generation and reports `team_routes`/`team_promotions`.
- Evidence auto‑tune for assimilation windows to reduce “insufficient_scores”. DR small‑n deferral (low_power_dr) grants evidence credit instead of forcing bad calls.
- Doubly‑robust uplift with stratified holdout telemetry (method, power, strata), plus snapshots of assimilation gates and attempts.
- Learning‑progress curriculum heatmap (LP) and smoothed lp_mix. QD coverage readout (optional).
- Adaptive relief: τ/ROI guardrails relax when merges stall; analyzer surfaces relief snapshots.

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
- **Analysis toolkit**: `scripts/analyze_ecology_run.py` summarises runs, reports ROI volatility, energy Gini/effective population, and shows why assimilation passed or failed.

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
The report now highlights ROI stats, assimilation gating totals, **recent gating snapshots**, and **the last few assimilation attempts** (control/treatment means, uplift vs. threshold, holdout verdict, top-up status), plus the diversity metrics (energy Gini, effective population, species share) and guard decay counters.

## Key Configuration Knobs
- `energy.m`, `energy.cost_scale`: ticket price and cost damping.
- `environment.success_reward_bonus`, `environment.failure_cost_multiplier`: steer survival pressure.
- `assimilation_tuning.*`: guard baselines, probe requirements, holdout settings, LoRA soup behaviour, **`energy_topup_roi_bonus` for lenient top-ups**, and `gating_snapshot_limit` for telemetry retention.
- `grid.families`: choose which mix of math, logic, sorting, sequence, and JSON cells make up the curriculum.
- `diversity.*`: energy Gini cap and per-species share limit.
- `policy.*`: turn on per-org policy proposals (`enabled`), cap prompt size (`token_cap`), and charge a micro-cost for policy requests (`energy_cost`, optionally scaled by tokens with `charge_tokens`). Analyzer reports policy usage and ROI when policy is on vs off.

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

What to watch in the ticker each generation
- `ROI`: keep mean in a healthy band (>1.3 desirable on Qwen3 smoke).
- `merges`: accepted assimilations (can be 0–3 in 40‑gen smoke).
- `energy floor` and `(ROI≥x)`: how strict the gate is and the ROI needed for top‑ups.
- `gating`: which reasons dominated (low_energy, low_power, low_power_dr, insufficient_scores, …).
- `trials/promotions`: trial offspring created and team promotions.
- `eval a/b`: holdout accuracy if scheduled.

Analyzer insights (after the run)
- Colonies count (min–max) and average size: ensures team behavior isn’t collapsing to monoculture.
- Policy usage: gens with policy, parse counts, ROI when policy on vs off, and field usage summary.
- Assimilation: events summary, mean sample size, CI‑excludes‑zero share, mean power; gating totals and sample reasons (look for `low_power_dr`).
- LP heatmap and cell difficulty/price heatmaps: confirms curriculum pressure and pricing dynamics.

## Latest Results Snapshot — Qwen3 Endogenous (40 gens)
- Run: `artifacts_qwen3_endogenous_flags_40_e`
- ROI: mean 1.872 (median 1.895; min/max 0.063/2.752)
- Energy: avg cost 0.985; mean balance 2.859 (min/max 2.382/3.075)
- Curriculum: lp_mix active mean 0.547 (base 0.200; last 0.600)
- Merges: 1 total; Bankruptcy culls: 0
- Evaluation: 22.22% (8/36); cadence 10
- Colonies: present briefly (count 0–1), average size 1.70
- Trials/promotions: trials 1, promotions 1
- Assimilation gating totals (top items):
  - uplift_below_threshold 238; insufficient_scores 211; cooldown 89; low_power 25; low_energy 39; no_best_cell 24
  - Top‑up status: success 195; ROI‑blocked 8; already_sufficient 421
- Policy parse rate: 0/580 (0.0%) — current JSON extraction is strict; consider more robust parsing or narrower allowed fields

Interpretation
- Healthy ROI and stable energy indicate the economy is viable; assimilation is still evidence‑limited (insufficient_scores) and uplift‑limited (below threshold).
- DR small‑n deferrals appear (low_power/low_power_dr), which is expected when holdout power is low; auto‑tune will keep nudging evidence windows within configured bounds.
- Policy hook influence is likely minimal given 0% parse rate; improving JSON extraction or steering the policy prompt can surface stronger routing/budget effects.
- Colonies form occasionally but remain small and transient under cautious gates — by design at this stage.

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
  - macOS tip: set `MPLCONFIGDIR=$(mktemp -d)` when running analysis or long training to avoid font cache writes and occasional malloc pressure.

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
