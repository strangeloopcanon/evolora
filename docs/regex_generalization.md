# Regex Generalizability Evaluation Framework

## So what (practical note)

In real evo runs, it’s possible to end up with a mixed population where many organelles never actually update
their LoRA weights (effectively behaving like the base model). When evaluating “best evo organelle” via a
separate selection/validation set, make sure selection does not accidentally choose a no-op adapter.

Also note that the SFT baseline trains on gold prompt→completion targets, while evolution must discover
those behaviors via rollouts + reward. Under a fixed training compute budget, SFT is expected to be much
more sample-efficient unless the environment provides dense learning signal.

### Empirical snapshot (single run; Qwen/Qwen2.5-0.5B)

- Base OOD holdout: `2/19` on `config/evaluation/regex_generalization.jsonl`
- Compute-matched SFT (LoRA rank=7; matched to evo `train_flops`): `12/19` (63.2%)
- Best evolved organelles (backprop plasticity, 50 generations): `5/19` (26.3%)
- In-distribution holdout (512 generated tasks): base `94/512` (18.4%), SFT `354/512` (69.1%), evo `148/512` (28.9%)
- Failure mode: selection initially picked a zero-magnitude adapter and reported `2/19`; evaluation now skips no-op adapters when any non-zero adapters exist.

## Prologue: What This Is For

This framework is intended to **measure the generalizability of regex-related skills** in language models, rather than raw task performance on a fixed set of examples.

In particular, it aims to distinguish:
- **Conceptual understanding** of regex semantics from
- **Template memorization** or overfitting to small test sets

The techniques below are designed to be:
- **Legible** (humans can understand *why* a model passed or failed),
- **Quantifiable** (produce concrete metrics),
- **Comparable** across training methods (e.g., SFT vs evolutionary LoRA methods).

Each dimension is defined independently and can be applied to the same underlying regex intent.

---

## Common Running Example (Single Intent, Many Probes)

### Intent

> Match a valid log timestamp of the form
> YYYY-MM-DD HH:MM:SS
> with:
> year: 2000–2099
> month: 01–12
> day: 01–31
> 24-hour time
> exactly one space between date and time
> no extra characters before or after

This is nice because:

- deterministic
- familiar
- has edge cases
- can be made fiendish
- supports refactoring and mutation
- has obvious overfitting traps

A reasonably clean “target” regex (Python-ish) might be:
```^(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]) (?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$```

---

## 1) Capability Axes
**What kind of regex skill is being exercised?**

### Definition
Different tasks probe different aspects of regex competence, even when they share the same semantic intent.

### Examples (shared intent: timestamps)

#### Recognition
**Task:**
Does this regex match `2023-01-01 24:00:00`? Why?

**Skill:**
Local semantic reasoning about ranges and constraints.

---

#### Synthesis
**Task:**
Write a regex that matches valid `YYYY-MM-DD HH:MM:SS` timestamps.

**Skill:**
Composing multiple constraints into a single pattern.

---

#### Explanation
**Task:**
Explain what this regex matches in plain English.

**Skill:**
Mapping syntax → semantics.

---

#### Debugging
**Task:**
Fix this regex so it no longer matches invalid hours.

**Skill:**
Identifying and correcting failure modes.

---

#### Refactoring
**Task:**
Simplify this regex without changing its behavior.

**Skill:**
Abstraction and redundancy elimination.

---

### What This Measures
The **breadth and shape of regex competence**, not generalization by itself.

---

## 2) Hold-Out Structure, Not Just Items
**What kind of regex knowledge is novel at test time?**

### Definition
Training and testing differ in **regex structure or concepts**, not merely in which strings appear.

### Examples

#### Operator Hold-Out
- **Train:** regexes without alternation (`|`)
- **Test:** task requires `(0[1-9]|1[0-2])`

---

#### Composition Hold-Out
- **Train:** character classes, alternation, and quantifiers appear independently
- **Test:** combined usage, e.g. `(0[1-9]|[12]\d|3[01])`

---

#### Semantic Coupling Hold-Out
- **Train:** date-only patterns and time-only patterns
- **Test:** full timestamp with joint constraints

---

#### Surface-Form Hold-Out
- **Train:** `YYYY/MM/DDTHH:MM:SS`
- **Test:** `YYYY-MM-DD HH:MM:SS`

---

### What This Measures
Whether the model generalizes across **conceptual structure**, rather than reproducing memorized templates.

---

## 3) Simplicity / Minimality Pressure
**How abstract is the learned solution?**

### Definition
Among regexes that behave correctly, prefer those that are **simpler, more canonical, or more constrained**.

### Examples

Two regexes that may pass the same tests:

```^20\d{2}-\d\d-\d\d \d\d:\d\d:\d\d$```
```^(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]) (?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$```

#### Metrics

- character length
- AST node count
- nesting depth
- redundant subpatterns
 - backtracking risk

---

### What %his Measures
Whether the model learned constraints or just “how to pass tests.”

---

## 4) Mutation / Counterfactual Tests
**Does the model understand local regex semantics?**

### Definition
Apply small, controlled mutations to a regex and evaluate reasoning about the resulting behavioral changes.

### Example
**Original:**
```^(?:[01]\d|2[0-3]):[0-5]\d$```

### Mutations
- Widen range → ```2\d```
- Remove grouping → ```[01]\d|2[0-3]```
- Change quantifier → ```[0-5]\d+```

### Tasks

#### Effect Prediction
Which strings will now match that did not before?

#### Repair
Fix the regex to restore the original behavior.

### Metrics
- Effect prediction accuracy
- Repair success rate
- Edit distance from the original regex

### What This Measures
Whether the model reasons about regex semantics locally, rather than relying on pattern recall or global heuristics.
