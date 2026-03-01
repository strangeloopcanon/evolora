"""Pretrained submask discovery: task-specific circuits in frozen LLMs.

Probes whether a pretrained Qwen3-1.7B contains task-type-specific
subnetworks discoverable through activation-based importance (Wanda
metric) alone -- no training, no LoRA, no fine-tuning.

Initial results (200 calibration tasks/family, perplexity eval):

* Importance-based masks (task-specific, global) preserve model quality
  3-4 orders of magnitude better than random masks at every sparsity
  level, confirming Wanda importance captures genuine structure.

* Task-matched masks do NOT consistently outperform global masks.
  At 50% sparsity the advantage is mixed (regex: 21.2 vs 21.8 ppl,
  math: 60.3 vs 45.6 ppl).  The importance rankings are ~91% shared
  across task families (Jaccard at 30% sparsity).

* Interpretation: at 1.7B scale with these task families, the model's
  weight importance landscape is largely task-agnostic -- a shared
  "core" dominates.  Task-specific circuits, if they exist, are a
  small fraction of the network.  This is the expected null result
  under the hypothesis that small LLMs lack the capacity for strong
  task specialisation in weight space.

See ``scripts/run_calibration.py`` through ``scripts/analyze_submasks.py``
for the four-phase pipeline, and ``config/experiments/submask_discovery.yaml``
for the reference configuration.
"""
