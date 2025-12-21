# Regex Generation Task Family

This document describes the `regex` task family added to the Evolora ecology system.

## Overview

The regex task family teaches LoRA adapters to generate regular expression patterns that match specific criteria. Tasks are provided with a natural language description and test cases, and the model must respond with a working regex pattern.

## Task Structure

### Depth Levels

**Short (Simple patterns)**
- Basic character classes (`\d`, `\w`)
- Simple quantifiers (`{3}`, `*`)
- Word boundaries (`\b`)
- Examples: "match 3-digit numbers", "match hex colors", "match words starting with 'test'"

**Medium (Intermediate patterns)**
- Alternation (`|`)
- Grouping with parentheses
- More complex quantifiers
- Character ranges
- Examples: "match phone numbers", "match URLs", "match dates in MM/DD/YYYY format"

**Long (Complex patterns)**
- Precise numeric ranges
- Nested groups
- Multiple alternations
- Complex character classes
- Examples: "match IPv4 addresses", "match Python function definitions", "match semantic versions"

## Evaluation Method

The system evaluates regex patterns by:

1. **Extracting the pattern** from the model's response (strips markdown, quotes, delimiters)
2. **Compiling the regex** to check validity
3. **Testing against provided test cases**:
   - Each test case specifies a string and whether it should match
   - Pattern must correctly match/reject ALL test cases to succeed
4. **Scoring**: Binary success (1.0) or failure (0.0)

This ensures that evolved LoRAs produce **functionally correct** patterns, not just syntactically valid ones.

## File Locations

- **Task generation**: `src/symbiont_ecology/environment/grid.py:807-826` (in `_make_task`)
- **Task templates**: `src/symbiont_ecology/environment/grid.py:925-1035` (in `_make_regex_task`)
- **Evaluation logic**: `src/symbiont_ecology/environment/grid.py:148-201` (in `GridTask.evaluate`)
- **Holdout tasks**: `config/evaluation/holdout_regex.jsonl`
- **Example config**: `config/experiments/qwen3_regex.yaml`

## Running a Regex Experiment

### Quick Start (50 generations)

```bash
MPLCONFIGDIR="$(mktemp -d)" .venv/bin/python scripts/eval_gemma_long.py \
  --config config/experiments/qwen3_regex.yaml \
  --gens 50 \
  --output artifacts_regex_experiment
```

### Longer Run (150 generations with holdout evaluation)

```bash
FINAL_HOLDOUT_TASKS=config/evaluation/holdout_regex.jsonl \
FINAL_HOLDOUT_SAMPLE_SIZE=12 \
MPLCONFIGDIR="$(mktemp -d)" .venv/bin/python scripts/eval_gemma_long.py \
  --config config/experiments/qwen3_regex.yaml \
  --gens 150 \
  --output artifacts_regex_long
```

### Analyze Results

```bash
# Generate plots and reports
MPLCONFIGDIR="$(mktemp -d)" .venv/bin/python scripts/analyze_ecology_run.py \
  artifacts_regex_experiment \
  --plots \
  --report

# Interactive visualization
MPLCONFIGDIR="$(mktemp -d)" .venv/bin/python scripts/evoscope.py \
  artifacts_regex_experiment
```

## What to Expect

As the ecology evolves:

1. **Early generations**: Models struggle with regex syntax, produce invalid patterns
2. **Mid generations**: Success on simple patterns (short depth), basic matching works
3. **Late generations**:
   - Specialization emerges across depth levels
   - Some LoRAs specialize in character classes, others in quantifiers
   - QD archive coverage increases across (regex, short/medium/long) cells
   - Best performers get merged, creating more capable offspring

## Metrics to Track

- **Success rate per depth**: Monitor `cells["regex:short"]`, `cells["regex:medium"]`, `cells["regex:long"]`
- **ROI**: Regex tasks have computational cost; successful patterns should show positive ROI
- **QD coverage**: Diversity of behaviors across the regex grid cells
- **Holdout accuracy**: Performance on unseen regex patterns in `holdout_regex.jsonl`

## Extending the Task Family

To add new regex pattern types:

1. Edit `_make_regex_task()` in `grid.py`
2. Add new template dicts with:
   - `desc`: Natural language description
   - `pattern`: Target regex pattern
   - `matches`: List of strings that should match
   - `non_matches`: List of strings that should NOT match
3. Add to appropriate depth level (short/medium/long)

Example:

```python
{
    "desc": "match social security numbers (XXX-XX-XXXX)",
    "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
    "matches": ["123-45-6789", "987-65-4321"],
    "non_matches": ["12-345-6789", "123456789"],
}
```

## Integration with Other Families

The config `config/experiments/qwen3_regex.yaml` includes regex alongside `math` and `word.count` families. This creates a multi-domain ecology where:

- Some LoRAs may specialize in regex generation
- Others may be generalists across domains
- The energy economy selects for efficiency across all task types

You can adjust the family mix in the config:

```yaml
env:
  grid:
    families: ["regex"]  # Regex only
    # OR
    families: ["regex", "math", "code.format", "json_repair"]  # Multi-domain
    depths: ["short", "medium", "long"]
```

## Troubleshooting

**Low success rates even after many generations:**
- Check if the base model (Qwen3-0.6B) has regex knowledge
- Try increasing `lora_r` in the config for more capacity
- Reduce `bankruptcy_grace` to cull poor performers faster
- Increase `max_merges_per_gen` to accelerate evolution

**Invalid regex patterns from model:**
- The evaluation safely catches `re.error` exceptions
- Failed compilations count as task failure (0.0 reward)
- LoRAs will evolve away from invalid syntax through negative ROI

**All patterns too conservative (e.g., just `.*`):**
- The test cases in `_make_regex_task` include non-matches
- Overly broad patterns will fail on non-match test cases
- This forces specificity through the evaluation function
