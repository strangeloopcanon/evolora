# Symbiotic Agent Ecology — SITREP (Grid Niches, Oct 2025)

## Summary
Grid niches remain live for the frozen Gemma‑270M host with Hebbian LoRA organelles. Each cell maintains its own difficulty/price equilibrium and seeded canaries. Energy is still scarce: organelles pre-buy tickets, settle compute-aware costs (FLOPs/GB/ms/params), and ROI is tracked per episode. Population selection uses μ+λ with viability + ROI gating, morphogenesis now adapts LoRA rank based on ROI **and** logs per-layer utilisation while enforcing adapter caps, and assimilation applies EMA gates, global probes, and multi-member LoRA soups weighted by ROI, EMA, and human-feedback probes. Generation summaries surface ROI/energy KPIs, bankruptcy culls, and assimilation history snapshots; scripts default to the grid environment and analysis tooling ingests the richer telemetry.

## Configuration Snapshot
- **Backbone**: Gemma‑3‑270M‑IT (transformers). Host ledger caps energy at `Emax=5` and seeds new organelles at 0.5 * cap.
- **Organelles**: Hebbian PEFT LoRA adapters (rank ≤8) with reward-modulated updates; trainable parameter count and active adapters logged for energy settlement.
- **Environment**: `GridEnvironment` over families {math, json_repair} and depths {short, medium, long}. Controller maintains per-cell success EMA (`tau=0.5`), adjusts difficulty, and prices tasks via dynamic k factor. Each cell has 5 rotating canaries.
- **Economics**: Episode settlement `ΔE = price * reward_total – (α FLOPs + β GB + γ ms + λp params)`, clamped to `[0, Emax]`. Tickets of size `m=1` required before evaluation. ROI logged as `revenue / cost` (cost floor 1e-6).
- **Selection**: μ=4 survivors ranked by viability (energy + canaries) and average ROI, λ=12 offspring mutated from survivors subject to `max_population=16`. Morphogenesis grows/shrinks LoRA rank within budget, and assimilation cools down per `(organelle, cell)` every 5 generations (max 1 merge per cell) with ROI-weighted LoRA soups.
- **Telemetry**: JSONL episodes include prompt, answer, cell, price, success, energy before/after, ROI, compute metrics, adapter utilisation, and active adapters. Assimilation events capture cell metadata, soup composition, HF probe outcomes, and per-cell uplift history while generation summaries expose ROI/energy KPIs and bankruptcy culls.

## Environment Pressure & Signals
- **Local niches**: Keeps diversity; cells drift difficulty toward 50% success. Prices rise in under-served cells, encouraging exploration.
- **Energy scarcity**: Empty energy accounts force culling; compute-heavy answers reduce ROI and energy, shrinking future opportunities.
- **Canaries**: High-EMA cells occasionally inject canaries; failures mark organelles unsafe (viability false) until redeemed.
- **Human bandit**: Still available, currently neutral weight; planned down-weight until grid stabilises.

## Recent Runs (baseline smoke)
- Full `pytest` suite (18 tests) passes on surrogate host with 83% coverage. New cases cover morphogenesis growth/shrink, assimilation soups/probes, energy bankruptcy, synthetic tasks, ledger, telemetry sink, and torch utils.
- Long Gemma grid eval still pending; prior phase-rotation artifacts remain in `artifacts_gemma_long_eval/` but are outdated for the new ROI economy.
- **Gemma 270M (mps) — 100 generations, batch size 2, no human bandit**: completed successfully (`artifacts_gemma_100gen/`). Average ROI hovered around 0.23 (peaks ≈0.86, troughs ≈–0.10); mean episode reward stayed negative (≈–2.0) because energy penalties outweighed correctness gains. Active organelles per generation ranged 4–15; bankrupt count 0–4. No assimilation merges passed current gates. Generated line charts at `artifacts_gemma_100gen/visuals/*.png` summarising ROI, reward, energy cost, and active/bankrupt counts; `scripts/analyze_ecology_run.py artifacts_gemma_100gen --plots --report` writes a Markdown report with the headline metrics.
- **Gemma 270M (mps) low-cost config — 100 generations, batch size 2, no human bandit**: using `config/experiments/gemma_mps_lowcost.yaml`, ROI mean improved to ≈0.38 (max ≈1.37) and average total reward climbed to ≈–0.95 with lower energy cost (≈0.62). Assimilation tests remain rare (2 failures) so merges stay at 0. Artifacts + report: `artifacts_gemma_lowcost_100gen/`.

## What Changed vs Prior SITREP
1. **Morphogenesis**: Rank resize preserves historical LoRA deltas via SVD projection, tracks per-layer utilisation, and enforces adapter caps.
2. **Assimilation**: Soups now blend multiple organelles with ROI/EMA/HF weighting; HF probe outcomes and uplift history persist per cell for dashboards.
3. **Telemetry & Analysis**: Generation summaries add ROI/energy KPIs, bankruptcy culls, and assimilation history snapshots. `analyze_ecology_run.py` and the visualiser consume the new metrics.
4. **Human bandit**: Feedback channel is configurable (weight + frequency) and respected by loops/CLI, enabling deterministic gating in long runs.

## Outstanding Gaps
- Validate the projection-based PEFT rank resizing + utilisation telemetry on Gemma long runs (watch ROI drift / adapter churn).
- Build dashboards on top of per-cell uplift history and HF probe logs; set niche-level assimilation SLOs.
- Run refreshed Gemma evaluations with human-bandit sweeps to calibrate preference/helper weights and frequency.
- Align meta-evolution scheduling with the richer KPIs; add catastrophic trigger tests once telemetry stabilises.

## Next Actions
1. Validate morphogenesis telemetry (utilisation + capping) on Gemma long runs and compare ROI deltas vs. baseline.
2. Instrument dashboards for assimilation uplift history + HF probes; set niche-level pass-rate targets.
3. Sweep human-bandit weights/frequency for stability vs. exploration and fold recommendations into config/docs.
