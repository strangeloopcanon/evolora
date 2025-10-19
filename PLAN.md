# Symbiont LLM Ecology — PLAN (Grid Niches + ROI)

## Mission
Keep the Gemma‑270M + PEFT‑LoRA + Hebbian ecology alive while shifting from global phase rotation to local niches, hardened survival economics, and niche‑aware assimilation that optimises ROI (reward per cost) without losing diversity.

## Completed (Delta → ✅)
- ✅ Config flags/YAML (`env.grid`, `teacher.*`, `pricing.*`, `energy.*`, `canary.q_min`, `population.*`, `assimilation.*`, `limits.*`).
- ✅ Environment layer: grid task factory per `(family, depth)`, per‑cell controller (“Teacher”), seeded canaries, org stats tracked per cell.
- ✅ Survival economics: energy tickets (`m`), compute‑aware settlement (`price*reward` minus FLOPs/GB/ms/params), ledger cap, bankruptcy path.
- ✅ Population selection: ROI logging, μ+λ retention (viability + ROI), per‑cell assimilation cooldown, canary gating.
- ✅ Morphogenesis: ROI-aware grow/shrink of LoRA rank within budget, neutral gate tweaks, metrics for active adapters per episode.
- ✅ Assimilation: per-cell cooldowns + EMA uplift gate, global probe checks, ROI-weighted LoRA soups feeding host merges.
- ✅ Telemetry & tests: extended JSONL (energy before/after, ROI, adapters), regression coverage (grid, morphogenesis, assimilation, ledger, synthetic tasks); overall coverage ≥80%.

## In Flight / Next
1. **Morphogenesis validation**
   - Benchmark utilisation telemetry + per-layer caps on Gemma long runs and monitor ROI drift after rank changes.
   - Extend dashboards to surface adapter occupancy trends from the new telemetry JSONL.

2. **Assimilation analytics**
   - Build dashboards atop per-cell uplift history and HF probe logs; define pass-rate SLOs per niche.
   - Evaluate multi-member soups on staged Gemma runs and compare uplift vs. two-member baseline.

3. **Long-run evaluation workflow**
   - Use updated generation summaries (ROI/energy, bankruptcy culls) in SITREP/analysis outputs.
   - Automate `analyze_ecology_run.py` + visualiser to include energy balance and assimilation history views.

4. **Human bandit experiments**
   - Sweep preference/helper weights & frequency for deterministic vs. exploratory runs; document recommended presets.
   - Capture calibration outcomes in config + README once stable.

## Success Gates
- ROI > 1.0 for ≥3 consecutive generations (baseline grid run).
- ≥1 successful assimilation per 5 generations; QD coverage across all grid cells.
- Net trainable params stable or shrinking while held‑out accuracy rises.

## References
- `docs/SITREP.md` for the latest measurements / narrative.
- `PROGRESS.md` for daily burndown and blockers.
- `WORK_LOG.md` for task-level append-only notes.
