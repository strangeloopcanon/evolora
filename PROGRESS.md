# Symbiont LLM Ecology — PROGRESS

## Status (2025-10-04)
- Grid ecology running end-to-end with ROI economics, per-cell stats, adapter-budgeted morphogenesis, and per-layer utilisation telemetry.
- Assimilation soups now blend multiple organelles with ROI/EMA/HF-weighted alphas, log HF probe outcomes, and persist per-cell uplift history.
- Generation summaries expose ROI/energy KPIs, bankruptcy culls, and assimilation history snapshots for downstream dashboards.
- Human bandit feedback governed via config (weights + frequency) and respected by scripts/loops for deterministic toggles.
- Test suite expanded (morphogenesis caps, PEFT rank projection, assimilation probes, environment controller equilibrium, bankruptcy culling, bandit gating) — `pytest` (stubbed host) holds ≥83% coverage.
- Analysis tooling (`analyze_ecology_run.py`, visualiser) updated to consume new KPIs; CLI scripts honour human-bandit config knobs.

## Blockers / Risks
- PEFT rank resizing currently copies overlapping LoRA weights only; needs validation on Gemma to ensure no drift.
- LoRA soup/global probe logic unverified on Gemma long runs; requires compute budget and HF weighting calibration.
- Human bandit weight unchanged; risk of noisy rewards until tuning pass lands.

## Decisions / Notes
- `omegaconf` kept optional; YAML loading raises actionable error if dependency absent.
- Human bandit remains hooked but weighting unchanged (needs tuning once stability verified).
- Telemetry includes ROI/energy metrics, active adapters, and assimilation cell metadata; SITREP refreshed with grid/ROI summary.

## Near-term Focus
1. Validate morphogenesis telemetry (utilisation + capping) on Gemma long runs and compare ROI deltas vs. baseline.
2. Instrument dashboards for assimilation uplift history + HF probes; set niche-level pass-rate targets.
3. Sweep human-bandit weights/frequency for stability vs. exploration and fold recommendations into config/docs.
