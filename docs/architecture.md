# Symbiont Ecology Architecture

The system implements a frozen host backbone with evolvable LoRA organelles that adapt via reward-modulated Hebbian plasticity and population-based evolution. Key components include:

- **Host Kernel**: `HostKernel` wraps a frozen HF backbone (Qwen3 by default; Gemma aliases remain supported), manages LoRA slots, orchestrates routing, and handles ATP accounting.
- **Organelles**: `HebbianPEFTOrganelle` maintains rank-limited adapters with eligibility traces and reward baselines (a small Hebbian LoRA organelle remains for tests/mocks).
- **Routing**: `BanditRouter` combines local activation signals and Thompson sampling to choose a thin execution path.
- **Evolution**: `PopulationManager`, `ModelMerger`, and `MorphogenesisController` implement quality-diversity breeding, weight merging, and organism growth.
- **Environments**: `TaskFactory` and `EcologyLoop` provide mixed tasks, while human/tool bridges are stubbed for extension.

Growth is regulated through energy budgets (ATP), assimilation tests, and morphogenesis controls that resize the backbone or merge adapter deltas when sustained uplift is detected.
