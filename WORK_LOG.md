# WORK LOG

## 2024-10-02 — Repository scaffolding
- **Context**: Build symbiotic agent ecology with Hebbian LoRA organelles per AGENT_MODE baseline.
- **Plan**: Establish project skeleton, tooling, and governance; implement host, environment, and evolution scaffolds; add tests and docs.
- **Assumptions**: Python 3.11+, local dev on Linux/macOS; LoRA-compatible backbone accessible via huggingface-style configs.
- **Tests**: Pending.
- **Failures/Notes**: None yet.

## 2024-10-02 — Core implementation
- **Context**: Implemented host kernel, Hebbian LoRA organelles, routing, evolution, and environment scaffolding per symbiotic ecology design.
- **Plan**: Deliver modular Python packages, CLI scripts, tests, and tooling; ensure Make targets succeed (check/test).
- **Changes**: Added configuration models, host/orchestrator, LoRA adapters with Hebbian traces, ATP ledger, morphogenesis, synthetic/human environments, quality-diversity manager, metrics, and documentation (`docs/architecture.md`).
- **Tests**: `make test`, `make check` (includes black, ruff, mypy, bandit, detect-secrets).
- **Failures/Notes**: Addressed mypy type issues via explicit casts and ignores; annotated Bandit B311 usages; updated `.secrets.baseline` after initial scan error.

## 2024-10-03 — ATP + evolution integration
- **Context**: Replaced prototype stubs with functional simulation, ATP economy, and evolutionary loop aligning with design brief.
- **Changes**: Added deterministic synthetic tasks, heuristic answer generation in Hebbian organelles, energy-aware ATP ledger updates, assimilation + mutation loop, and bootstrap wiring for population manager.
- **Tests**: `make format`, `make test`, `make check` (black, ruff, mypy, bandit, detect-secrets).
- **Notes**: Synthetic RNG flagged by Bandit; annotated deterministic usage. Evolution loop now mutates and assimilates organelles per generation.

## 2024-10-03 — Human feedback integration
- **Context**: Incorporate human-bandit reward channel alongside synthetic tasks.
- **Changes**: Added calculator tool, HumanBandit with reward deltas, reward blending in ecology loop, telemetry persistence for human notes, updated bootstrap/tests.
- **Tests**: `make test`, `make check`.
- **Notes**: Bandit warnings acknowledged via documented subprocess usage.

## 2024-10-03 — Gemma backbone optional integration
- **Context**: Enable real LLM backbone option while keeping tests on lightweight surrogate.
- **Changes**: Added Gemma backbone loader (transformers) with frozen weights, config toggle, reward blending tests, population replacement to avoid collapse.
- **Tests**: `make test`, `make check`.
- **Notes**: Long-run simulation still needs tuning for dynamic environments; telemetry stored under `artifacts_long_run/`.

## 2024-10-03 — Dynamic tasks & Gemma eval harness

## 2025-10-03 — Grid niches plan + doc alignment
- Context: Replace global phase rotation with local niche grid; harden ATP via energy tickets and compute-aware pricing; make assimilation niche-aware. Keep existing loop and evolve in place.
- Docs: Added README.md clarifying objective and doc roles; previous planning/status docs (PLAN.md, PROGRESS.md, docs/SITREP.md) kept as historical context at the time, later consolidated into README + docs/ecology_overview.md.
- Next: Implement GridTaskFactory + Teacher, energy settlement (Emax/m/alpha/beta/gamma/lambda_p), μ+λ selection with ROI tracking, niche-aware assimilation with EMA uplift, extended telemetry fields, and focused tests. Reduce human bandit weight until stable.
- **Context**: Add evolving task niches and prepare Gemma-only evaluation run.
- **Changes**: TaskFactory phases with new domains (sorting, word counts), environment loop now advances phase each generation, seeded replacements keep population alive, added Gemma evaluation script (`scripts/eval_gemma.py`).
- **Tests**: `make test`, `make check`, 10-gen dry run with dynamic factory produced population growth.
- **Notes**: Coverage excludes Gemma host file; long-run telemetry stored under `artifacts_long_run/`.

## 2025-10-04 — Grid integration + ROI economics
- **Context**: Replace phase rotation with grid niches, wire energy settlement, update selection/assimilation, refresh docs.
- **Changes**:
  - Added `GridEnvironment` controller wiring into `EcologyLoop`; introduced energy tickets, compute-aware settlement, ROI logging, μ+λ selection, per-cell assimilation cooldown.
  - Exposed `load_ecology_config`, updated CLI scripts to read `config/ecology.yaml`, ensured host step emits compute metrics.
  - Extended telemetry (ROI, energy before/after, cell metadata), optional `omegaconf` import, new pytest smoke tests (`tests/test_task_factory.py`, `tests/test_ecology_loop.py`) and test `conftest` path helper.
  - Updated `AGENTS.md`, `README.md`, and supporting docs to reflect grid world + ROI economy.
- **Tests**: `pytest tests/test_task_factory.py tests/test_ecology_loop.py` (passes but overall coverage 68% < 80% gate — broader suite still required).
- **Notes**: Morphogenesis mutations and LoRA soup assimilation still pending; coverage guard remains red until additional tests land.

## 2025-10-04 — Morphogenesis + assimilation upgrade
- **Context**: Implement ROI-driven morphogenesis, niche-aware assimilation soups/probes, and expand regression coverage for the new grid ecology.
- **Changes**:
  - Added organelle rank resize methods (LoRA + PEFT), host route metrics (active adapters, compute stats), morphogenesis controller applying grow/shrink/gate tweaks under adapter budget caps.
  - Enhanced assimilation with per-cell cooldown tracking, global probe validation, and ROI-weighted LoRA soups; integration updated to log cell metadata.
  - Extended `EcologyLoop` for μ+λ selection + morphogenesis pipeline, added `_global_probe`/`_lora_soup_merge`, and new helper methods in `GridEnvironment`.
  - Expanded test suite (morphogenesis, assimilation, energy bankruptcy, synthetic factory, ledger, telemetry sink, torch utils) and lifted coverage to 83%.
  - Refreshed docs (`PLAN.md`, `PROGRESS.md`, `docs/SITREP.md`, `README.md`) to capture completed milestones and next focus areas.
- **Tests**: `pytest` (full suite, 18 tests, coverage 83.22%).
- **Notes**: PEFT rank resizing copies overlapping weights; schedule Gemma grid eval + human bandit tuning next.

## 2025-10-04 — Meta-evolution + visualisation pipeline
- **Context**: Enable co-evolution of controller/assimilation parameters, support catastrophic shifts, and surface grid visualisations before the 100-generation experiment.
- **Changes**:
  - Added `MetaEvolver` with ROI-driven parameter mutations, per-cell catastrophic shift support, and config knobs (interval, scale) plus summary telemetry of cell states.
  - Extended `EcologyLoop` summaries (cell metrics, meta actions), `GridEnvironment` controller updates, assimilation threshold updates, and new heatmap visualiser (`scripts/visualize_grid.py`).
  - Updated long-run script to accept CLI args (`--generations`, `--backbone`, etc.) and emit JSONL/CSV summaries for plotting.
  - Added regression tests for meta-evolution, ledger, synthetic tasks, telemetry sink, and ensured coverage ≥83% with new dependencies (matplotlib).
- **Tests**: `pytest` (20 tests, coverage 83.92%).
- **Notes**: Ready to launch 100-generation run once approved; visualiser consumes `gen_summaries.jsonl` to produce per-metric heatmaps.

## 2025-10-05 — Gemma long runs + analysis tooling
- **Context**: Execute baseline and low-cost Gemma 100-gen runs; build reusable analysis/report pipeline.
- **Changes**:
  - Ran `scripts/eval_gemma_long.py` (baseline config) to collect reference metrics (`artifacts_gemma_100gen/`): ROI ≈0.23, reward ≈–2.0, 0/5 assimilation passes.
  - Tuned energy/assimilation/meta parameters (`config/experiments/gemma_mps_lowcost.yaml`) and reran 100 gens with improved ROI ≈0.38 and reward ≈–0.95 (`artifacts_gemma_lowcost_100gen/`), though merges remain 0/2.
  - Added `scripts/analyze_ecology_run.py` to generate Markdown reports and plots for any run; fixed CSV export to handle per-cell/meta fields.
  - Introduced evaluation infrastructure (config `evaluation`, holdout tasks, periodic reward injection) and wired summaries to capture accuracy.
- **Tests**: `pytest` (20 tests, coverage ≈83.9%).
- **Notes**: Visual outputs live under each run’s `visuals/`; next step is relaxing assimilation gates and logging meta mutations for heatmaps.

## 2025-10-05 — Utilisation telemetry + assimilation analytics
- **Context**: Close out morphogenesis/assimilation backlog (evolora-1..3), surface ROI/energy KPIs, and add human-bandit calibration knobs.
- **Changes**:
  - Tracked per-layer adapter utilisation, enforced LoRA caps, and projected PEFT rank history via SVD during rank resize.
  - Upgraded assimilation to multi-member soups with ROI/EMA/HF weighting, logged HF probe outcomes, and persisted per-cell uplift history.
  - Added bankruptcy culling (configurable grace), generation-level KPIs (ROI/energy per org, culls, assimilation snapshots), and refreshed analysis tooling/visuals.
  - Introduced config-driven human-bandit weights/frequency; loops/scripts respect deterministic gating.
- **Tests**: `pytest` focus (morphogenesis, assimilation, environment controller, energy economy, bandit gating, telemetry persistence). Full suite pending env setup.
- **Notes**: README and supporting docs updated; next actions centre on Gemma validation runs and dashboard wiring.
# Run 2025-10-18 – survival tweaks
- Config: config/experiments/gemma_relaxed.yaml (energy.m=0.6, success_reward_bonus=0.75, cost_scale=0.7)
- Command: AGENT_MODE=baseline .venv311/bin/python scripts/eval_gemma_long.py --config config/experiments/gemma_relaxed.yaml --generations 100 --batch-size 2 --output artifacts_gemma_relaxed_autotune_v5
- Outcome: ROI mean 0.66 (max 2.99), merges=0, assimilation attempts 14 (failures: uplift 286, insufficient 67, holdout 1)

# Run 2025-10-19 – assimilation tuning pre-run
- Adjusted assimilation windows (even-length up to 16), adaptive uplift decay, holdout retries, and larger soups.
- Added survival resilience regression tests (tests/test_survival_resilience.py).
- Config: config/experiments/gemma_relaxed.yaml updated with adaptive guard knobs.
- Tests: AGENT_MODE=baseline make test VENV=.venv311

# Run 2025-10-19 – assimilation guard loosened
- Adjusted config: assimilation_threshold→0.0, seed_scale 0.8, holdout retries 3 with step 0.02, max merges/gen 6.
- Tests: AGENT_MODE=baseline make test VENV=.venv311

# Run 2025-10-20 – first successful assimilations
- Config: `config/experiments/gemma_relaxed.yaml` (mutation_rate 0.32, per-cell interval 2, holdout sample 4 with margin 0.05).
- Command: `MPLCONFIGDIR=$(mktemp -d) AGENT_MODE=baseline .venv311/bin/python scripts/eval_gemma_long.py --config config/experiments/gemma_relaxed.yaml --generations 100 --batch-size 2 --output artifacts_gemma_relaxed_autotune_v8`
- Outcome: ROI mean 3.49 (max 7.49), 3 assimilation passes on `word.count:medium` cells (sample_size 2–4, uplift 0.18–0.60) with HF probes clean; 86 attempts total, rest failed on sub-threshold uplift. Evaluation accuracy still 0/60.
- Notes: Energy floor decayed to 0.65 with 133 top-ups and no bankruptcies. Next up—harder word-count curricula and broader mutations to push uplift onto external holdouts.
