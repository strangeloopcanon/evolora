#!/usr/bin/env python3
"""Generate a fixed, deterministically-graded holdout suite for paper runs.

The goal is a small-but-real evaluation set: prompts have exact targets and can be
graded without an LLM judge. This is meant for *measurement* (accuracy/cost), not
for training-time reward shaping.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


def _to_snake_case(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    clean = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", clean)
    clean = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", clean)
    clean = re.sub(r"_+", "_", clean)
    return clean.lower()


def _make_word_count_short(rng: random.Random, count: int) -> list[dict[str, Any]]:
    vocab = [
        "symbiotic",
        "agents",
        "learn",
        "cooperate",
        "adapt",
        "quickly",
        "under",
        "scarcity",
        "energy",
        "tickets",
        "colonies",
        "emerge",
        "from",
        "selection",
        "pressure",
        "policy",
        "routes",
        "tasks",
        "across",
        "niches",
    ]
    tasks: list[dict[str, Any]] = []
    for _ in range(count):
        n = rng.randint(3, 9)
        words = rng.sample(vocab, n)
        sentence = " ".join(words)
        tasks.append(
            {
                "prompt": (
                    "Count the number of words in the sentence: "
                    f"'{sentence}'. Respond with an integer."
                ),
                "target": n,
                "family": "word.count",
                "depth": "short",
            }
        )
    return tasks


def _make_word_count_medium(rng: random.Random, count: int) -> list[dict[str, Any]]:
    templates: list[tuple[str, list[str], list[str]]] = [
        (
            "Count the number of alphabetic words (ignore digits) in the sentence: '{text}'. "
            "Respond with an integer.",
            ["Run", "{a}", "completes", "after", "{b}", "retries", "and", "{c}", "fallbacks"],
            ["Run", "completes", "after", "retries", "and", "fallbacks"],
        ),
        (
            "Count the number of words (ignore HTML tags) in the snippet: `{html}` "
            "Respond with an integer.",
            ["workflow", "vectors", "biosphere", "align", "merge"],
            ["workflow", "vectors", "biosphere", "align", "merge"],
        ),
    ]
    tasks: list[dict[str, Any]] = []
    for _ in range(count):
        which = rng.randrange(len(templates))
        prompt_tmpl, raw_words, count_words = templates[which]
        if which == 0:
            a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(1, 9)
            words = [w.format(a=a, b=b, c=c) for w in raw_words]
            text = " ".join(words)
            target = len(count_words)
            prompt = prompt_tmpl.format(text=text)
        else:
            tokens = rng.sample(raw_words, rng.randint(3, len(raw_words)))
            # Wrap some tokens in tags, but count tokens only (tags ignored)
            wrapped: list[str] = []
            for tok in tokens:
                tag = rng.choice(["strong", "em", "span"])
                if rng.random() < 0.5:
                    wrapped.append(f"<{tag}>{tok}</{tag}>")
                else:
                    wrapped.append(tok)
            html = "<div>" + " ".join(wrapped) + "</div>"
            target = len(tokens)
            prompt = prompt_tmpl.format(html=html)
        tasks.append(
            {
                "prompt": prompt,
                "target": target,
                "family": "word.count",
                "depth": "medium",
            }
        )
    return tasks


def _make_math_short(rng: random.Random, count: int) -> list[dict[str, Any]]:
    ops = ["add", "sub", "mul", "div"]
    tasks: list[dict[str, Any]] = []
    for _ in range(count):
        op = rng.choice(ops)
        if op == "add":
            a, b = rng.randint(1, 99), rng.randint(1, 99)
            prompt = f"Add {a} and {b}. Respond with the number only."
            target = a + b
        elif op == "sub":
            a, b = rng.randint(10, 120), rng.randint(1, 99)
            prompt = f"Subtract {b} from {a}. Respond with the number only."
            target = a - b
        elif op == "mul":
            a, b = rng.randint(2, 14), rng.randint(2, 14)
            prompt = f"Multiply {a} by {b}. Respond with the number only."
            target = a * b
        else:
            b = rng.randint(2, 12)
            q = rng.randint(2, 20)
            a = b * q
            prompt = f"Compute {a} / {b}. Respond with the number only."
            target = q
        tasks.append({"prompt": prompt, "target": target, "family": "math", "depth": "short"})
    return tasks


def _make_math_medium(rng: random.Random, count: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for _ in range(count):
        form = rng.choice(["a", "b", "c"])
        if form == "a":
            a, b, c = rng.randint(2, 30), rng.randint(2, 30), rng.randint(2, 12)
            prompt = f"Compute ({a} + {b}) * {c}. Respond with the number only."
            target = (a + b) * c
        elif form == "b":
            a, b, c = rng.randint(2, 20), rng.randint(2, 20), rng.randint(1, 50)
            prompt = f"Compute ({a} * {b}) + {c}. Respond with the number only."
            target = (a * b) + c
        else:
            a, b, c = rng.randint(10, 99), rng.randint(1, 9), rng.randint(1, 9)
            prompt = f"Compute ({a} - {b}) / {c}. Respond with the number only."
            target = (a - b) / c
        tasks.append({"prompt": prompt, "target": target, "family": "math", "depth": "medium"})
    return tasks


def _make_code_format_tasks(rng: random.Random, count: int) -> list[dict[str, Any]]:
    bases = [
        "DeltaRewardEnergy",
        "maxRoutesPerGen",
        "HoldoutSampleSize",
        "policyBudgetFrac",
        "EvidenceTokenWindow",
        "colonyVarianceRatio",
        "AdapterStateCache",
        "TeamProbeSynergyDelta",
        "assimilationHistoryLimit",
        "EnergyTicketMultiplier",
        "QdArchiveCoverage",
        "MeanEnergyBalance",
        "RouteCostPenalty",
        "RecurrentEvalPasses",
        "bankruptcyGrace",
        "GlobalEpisodeCap",
        "BudgetEnergyCeiling",
        "minWindowMax",
        "promotionTargetRate",
        "mergeAuditEnabled",
    ]
    tasks: list[dict[str, Any]] = []
    for idx in range(count):
        raw = rng.choice(bases) + (str(idx) if rng.random() < 0.25 else "")
        target = _to_snake_case(raw)
        depth = "short" if idx % 2 == 0 else "medium"
        tasks.append(
            {
                "prompt": f"Convert the variable name `{raw}` to snake_case. Respond with the new name only.",
                "target": target,
                "family": "code.format",
                "depth": depth,
            }
        )
    return tasks


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/evaluation/paper_qwen3_holdout_v1.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--per_family", type=int, default=40, help="Tasks per family (total=3×).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    per_family = max(1, int(args.per_family))
    # Split each family evenly across short/medium where applicable.
    half = per_family // 2
    records: list[dict[str, Any]] = []
    records.extend(_make_word_count_short(rng, half))
    records.extend(_make_word_count_medium(rng, per_family - half))
    records.extend(_make_math_short(rng, half))
    records.extend(_make_math_medium(rng, per_family - half))
    records.extend(_make_code_format_tasks(rng, per_family))
    # Stable shuffle so the file ordering is deterministic but mixed.
    rng.shuffle(records)
    _write_jsonl(args.output, records)
    print(f"Wrote {len(records)} tasks to {args.output}")


if __name__ == "__main__":
    main()
