# Symbiotic LLM Ecology — Overview, Baselines, and Framing

This document captures the core idea behind the “symbiont ecology” setup, how it behaves in practice on Qwen3‑0.6B, and how it relates to both standard adaptation methods and classic artificial life work.

It is meant to be a single place you can pull from for a paper or blog post.

---

## 1. Objects in the World

At the highest level, the system has three kinds of objects:

- A **frozen host model** `f_θ`, here a small transformer (Qwen3‑0.6B or Gemma‑270M) whose backbone parameters never change.
- A population of **organelles** `O = {o_i}`, each a tiny LoRA adapter plus a bit of state (traces, reward baseline, traits). An organelle defines a small, structured perturbation `Δθ_i` to the host.
- An **environment + economy** that:
  - serves tasks `x` drawn from a curriculum and evaluates answers `y`,
  - charges energy for each call,
  - pays out reward when answers are good,
  - and periodically runs assimilation tests and merges.

A concrete forward pass looks like:

```text
answer = decode( f_θ(x; Δθ_i) )
reward = R(x, answer)  # task reward, possibly shaped
```

where `Δθ_i` is the low‑rank adapter for organelle `i`.

The host is shared; only the adapters (and who is allowed to use them) change over time.

---

## 2. What Is Being Optimised?

There are three coupled objectives:

1. **Per‑call competence**: for each task `x`, we want the selected organelle `i` to maximise reward (accuracy, formatting correctness, etc.).
2. **Energy‑adjusted performance**: the ecology tracks **ROI** ≈ reward minus cost. Organelles are rewarded for being good *and* cheap, because each call burns energy.
3. **Population‑level diversity**: the QD archive uses behaviour descriptors (task family, depth, cost bin) to nudge the population towards covering different niches, not just repeating the same trick.

The measurable signal is a stream of per‑episode tuples:

```text
(organelle_id, cell_metadata, reward, cost, roi, success_flag, novelty_descriptor)
```

The ecology uses this stream to:

- update per‑organelle Hebbian traces and reward baselines,
- adjust energy balances and budgets,
- decide which organelles to merge or retire,
- and update the QD archive of “interesting” behaviours.

No gradients flow into the host backbone; all adaptation happens in the LoRA population, their traits, and the energy landscape they live in.

---

## 3. Where the “Learning” Actually Happens

There are two different learning mechanisms stacked on top of each other:

1. **Local Hebbian plasticity inside each organelle**
   - Each organelle maintains eligibility traces and a simple reward‑modulated update rule for its low‑rank weights.
   - You can think of this as a tiny, per‑adapter RL‑style critic that responds to recent reward and nudges the adapter weights accordingly.
   - This is where “credit assignment” happens along the time axis of a single organelle’s episodes.

2. **Population‑level selection and morphogenesis**
   - The population manager keeps recent ROI and energy stats for each organelle and each `(family, depth)` cell.
   - When an organelle consistently delivers uplift in a cell, assimilation tests run:
     - compute uplift vs recent history,
     - bootstrap or DR‑estimate confidence intervals,
     - run a holdout test on unseen tasks,
     - and, if all checks pass, merge weights into a child.
   - Children can inherit traits (e.g., energy budget preference, exploration parameters), and unsuccessful adapters are retired when energy and bankruptcy rules say so.

The key idea is that the second mechanism doesn’t care about the local update rule; it only needs a scalar ROI signal and enough episodes to estimate uplift. That means we can keep the Hebbian rule tiny and focus most of our design work on the ecology and economy.

---

## 4. Energy and Tickets as a Control Surface

Energy is the main way the system decides **who gets to learn**:

- Every generation, organelles pay a fixed ticket cost to participate.
- Calls to the host (inference, policy, comms, memory reads/writes) also charge energy.
- Successful episodes earn reward, which is converted into energy top‑ups subject to an auto‑tuned floor/cap.

Because energy balances directly determine:

- how often an organelle is sampled,
- whether it survives bankruptcy,
- whether it gets probed in assimilation tests,

the energy rules are as important as the learning rule. A few practical consequences:

- If tickets are too cheap, everyone survives and the population never sharpens.
- If tickets are too expensive or bankruptcy grace is too low, the ecology collapses to zero organelles and stops.
- Evidence tokens and “power” estimates govern whether assimilation tests defer (`low_power`) or fire; they effectively shape the cadence of merges.

From a control‑theory point of view, the host+organelles see tasks and rewards; the ecology sees only coarse ROI and energy stats, but can shape the future task distribution by reallocating energy.

---

## 5. How This Relates to Familiar Methods

It helps to place the ecology next to more standard adaptation tools:

- **Supervised fine‑tuning (SFT)**:
  - Optimises a loss on labelled (x, y) pairs.
  - Typically updates the full parameter set or a large LoRA.
  - Uses gradient descent with clear objectives and strong supervision.

- **Reinforcement learning (RL)**:
  - Optimises expected cumulative reward over trajectories.
  - Updates policy parameters based on credit assignment over time.
  - Often uses baselines, advantage estimates, and explicit exploration strategies.

- **Ecological adaptation (this project)**:
  - Keeps the host frozen; only small adapters mutate and merge.
  - Uses a simple local learning rule inside each adapter *plus* a population and economy around them.
  - Optimises a mix of ROI, survival, and diversity by:
    - deciding which adapters survive,
    - which get more energy,
    - and which get merged into children.

You can think of the ecology as:

- SFT‑like at the per‑call level (it uses the same base model for next‑token prediction), but without explicit labels or backbone gradients.
- RL‑like in that it has rewards, energy, and survival, but the primary “policy” being optimised is *which adapters exist and where they get used* rather than a monolithic network.

The upside is that you can adapt behaviour over long horizons without ever touching the base weights. The downside is that you’re working with noisy, coarse signals (ROI and survival) and you need careful design to avoid collapse or drift.

---

## 6. Three Baselines on Qwen3‑0.6B (50 Gens)

To make the difference between “just a model”, “one adapter”, and “full ecology” concrete, we ran three matched 50‑generation experiments on the same host (Qwen3‑0.6B) and task mix (word.count, math, code.format).

### 6.1 Summary Table

| Setup            | Config                                      | Run ID                                              | Gens | Episodes | Mean ROI | Merges | Assim events (pass / fail) | QD coverage (mean / last) | Eval (correct / total) |
|------------------|---------------------------------------------|-----------------------------------------------------|------|----------|----------|--------|-----------------------------|---------------------------|------------------------|
| Frozen base      | `paper_qwen3_frozen.yaml`                   | `artifacts_paper_qwen3_frozen_20251201_205728`      | 50   | 1364     | ~0.82    | 0      | 0                           | 0 % / 0 %                 | 2 / 3                  |
| Single adapter   | `paper_qwen3_single.yaml`                   | `artifacts_paper_qwen3_single_20251203_120617`      | 50   | 22       | ~0.06    | 0      | 0                           | 0 % / 0 %                 | 2 / 3                  |
| Full ecology     | `paper_qwen3_ecology.yaml`                  | `artifacts_paper_qwen3_ecology_20251202_120102`     | 50   | 1511     | ~0.78    | 28     | 82 (13 / 69)                | ~17.5 % / 20.8 %          | 2 / 3                  |

All numbers come from `scripts/analyze_ecology_run.py` for the respective run directories.

### 6.2 Quick Read of Each Setup

**Frozen base (no adaptation)**

- Behaviour:
  - Episodes: ~1.3k.
  - ROI: mean ≈ 0.82, reasonably stable.
  - Structure: 0 merges, no colonies, empty QD archive, knowledge cache off.
  - Eval: ~2/3 on the small holdout.
- Takeaway: the host+tasks are sensible and competent, but structurally inert. Good control, no emergent behaviour.

**Single adapter (one organelle with a bit of learning)**

- Behaviour:
  - Episodes: 22 over 50 gens.
  - ROI: mean ≈ 0.06; early positive spikes then a long tail near zero.
  - Structure: 0 merges (disabled), no colonies, empty QD archive.
  - Memory: a few cache writes/reads; hits exactly a few times.
  - Eval: also ~2/3 on the small holdout.
- Takeaway: just adding a single LoRA on a frozen host does *not* give you a rich, self‑sustaining adaptation process. It learns a bit, then mostly idles.

**Full ecology (many organelles + economy)**

- Behaviour:
  - Episodes: ~1.7k.
  - ROI: mean ≈ 0.78, with a band of healthy high‑ROI generations.
  - Structure:
    - 28 merges plus 66 trials, with 82 assimilation events (13 passes).
    - Colonies: one colony (size 2) active through the run, with holdout passes and pot updates.
    - QD archive: mean coverage ≈ 17.5 %, last ≈ 20.8 %, with top bins in math and word.count.
  - Memory and policy:
    - Knowledge cache: 42 writes, 4 reads, 4 hits; actually used.
    - Policy channel: active every generation; 100 % parse rate; ROI higher when policy is on.
  - Evidence and power:
    - Evidence tokens: 1533 minted / 160 used; uplift tests almost never include zero once tokens accrue.
    - Gating: dominated by `low_power`, `insufficient_scores`, and `uplift_below_threshold`, not `no_activity`.
  - Eval: again ~2/3 on the small holdout.
- Takeaway: with the ecology switched on, the system keeps restructuring itself—organelles earn and spend energy, merge, and spawn children; colonies form; the archive fills in; memory and policy channels get exercised.

Overall, the baselines show that the ecology is doing something qualitatively different from both the frozen host and the “one adapter” setup, even though all three are running on the same backbone.

---

## 7. Open Questions This Setup Raises

This framing naturally leads to a few questions for future work:

- **Scaling laws**: how does ecological adaptation behave as we scale host size, adapter capacity, and task difficulty? Is there a regime where ecology gives disproportionate gains?
- **Credit assignment**: can we borrow stronger ideas from RL (e.g., advantage estimates, variance reduction) to improve the per‑adapter learning rule without losing the nice “frozen host” property?
- **Diversity vs performance**: how should the QD archive and energy rules trade off novelty and immediate reward? When is it worth keeping a weak but diverse organelle alive?
- **Interfacing with external tools**: can organelles specialise not just on task families but on tool use, memory, or retrieval, so that the ecology evolves a division of labour over a larger system?

The current codebase answers a small but important sub‑question: even on a tiny host, an ecology with a reasonable economy can sustain non‑trivial, inspectable adaptation over many generations using only small adapters.

---

## 8. Relation to Artificial Life

This system has a lot in common with classic artificial‑life work:

- There is a **limited resource** (energy), a **population of agents** (LoRA organelles), and a structured **environment** (task grid with prices and curriculum). Agents compete, reproduce (via merges/offspring), and die (bankruptcy and culls).
- Behaviour, not just parameters, determines survival: organelles live or die based on how they perform in specific “niches” (task family/depth/cost bins), tracked by the QD archive.
- The “physics” of the world is set by the frozen host: like a fixed instruction set or cellular automaton rule, it defines what’s possible, while evolution happens in the small programs (adapters) that run on top of it.

What’s unusual compared to traditional A‑life is the substrate: instead of evolving tiny hand‑rolled programs, the ecology is evolving small, structured perturbations of a powerful language model. That makes the agents’ phenotypes immediately legible—you can literally talk to the colony and read out what evolution has produced.

---

## 9. Rough Positioning vs Related Work

This setup touches several existing areas:

- **Evolutionary neural nets and quality‑diversity**  
  - There is a long history of evolving network weights and architectures (e.g. NEAT, CoDeepNEAT, QD methods like MAP‑Elites).  
  - This project borrows the QD mindset (maintain a diverse archive over behaviour descriptors) but uses a frozen host and tiny LoRA “deltas” instead of evolving full networks.

- **Population‑based RL and open‑ended training**  
  - Population‑based training, league‑based RL, and open‑ended algorithms (e.g. POET‑style environments) also evolve sets of agents over time.  
  - Here, the “environment” is task‑based rather than embodied, and the main thing evolving is which small adapters exist on a fixed language model, not entire agents with their own world models.

- **Adapter and LoRA‑based specialisation**  
  - A lot of recent work uses adapters/LoRAs for multi‑tasking and domain specialisation (one adapter per skill, routing, etc.).  
  - The ecology can be viewed as an automated, continual version of this: instead of hand‑designing which adapters to keep and when to use them, we let energy and uplift tests decide.

- **Artificial life in program spaces**  
  - Classic A‑life systems (Tierra, Avida, many later systems) evolve small programs in a fixed computational substrate with resource constraints.  
  - Evolora is similar in spirit, but the “programs” are low‑rank adapter perturbations on a modern LLM. That makes the behaviours human‑readable and directly useful in language‑centric tasks.

The distinctive combination here is:

- a frozen LLM host,
- many small adapters with a simple local update rule,
- an explicit energy economy and ticketing scheme,
- and a QD‑style archive,

all working together to produce long‑horizon, inspectable adaptation without ever fine‑tuning the base model.

