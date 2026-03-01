"""Standalone task generators for submask discovery experiments.

Each generator produces Task objects with a prompt, target, and evaluate method.
No dependency on symbiont_ecology.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, List


@dataclass
class Task:
    """A single evaluation task."""

    task_id: str
    family: str
    prompt: str
    target: Any
    _evaluator: Callable[[str], bool] = field(repr=False)

    def evaluate(self, answer: str) -> bool:
        return self._evaluator(answer)


class RegexTaskGen:
    """Generates regex-pattern tasks at varying difficulty."""

    TEMPLATES = [
        "day_01_31",
        "date_ymd",
        "email_simple",
        "digits_run",
        "hex_color",
        "time_hhmm",
        "ipv4_octet",
        "identifier",
    ]

    def generate(self, n: int, seed: int = 42) -> List[Task]:
        rng = random.Random(seed)
        tasks: List[Task] = []
        for i in range(n):
            template = rng.choice(self.TEMPLATES)
            builder = getattr(self, f"_build_{template}")
            desc, pattern, matches, non_matches = builder(rng)
            prompt = self._format_prompt(desc, matches, non_matches)
            target_pattern = pattern

            def _make_evaluator(
                pat: str,
                match_list: List[str],
                non_match_list: List[str],
            ) -> Callable[[str], bool]:
                def _eval(answer: str) -> bool:
                    answer = answer.strip()
                    for fence in ("```regex", "```"):
                        if answer.startswith(fence):
                            answer = answer[len(fence) :]
                        if answer.endswith("```"):
                            answer = answer[:-3]
                    answer = answer.strip()
                    if not answer:
                        return False
                    try:
                        compiled = re.compile(answer)
                    except re.error:
                        return False
                    for m in match_list:
                        if not compiled.search(m):
                            return False
                    for nm in non_match_list:
                        if compiled.search(nm):
                            return False
                    return True

                return _eval

            evaluator = _make_evaluator(target_pattern, matches, non_matches)
            tasks.append(
                Task(
                    task_id=f"regex_{seed}_{i:04d}",
                    family="regex",
                    prompt=prompt,
                    target=target_pattern,
                    _evaluator=evaluator,
                )
            )
        return tasks

    @staticmethod
    def _format_prompt(desc: str, matches: List[str], non_matches: List[str]) -> str:
        m_str = ", ".join(f'"{s}"' for s in matches)
        nm_str = ", ".join(f'"{s}"' for s in non_matches)
        return (
            f"Write a regex pattern that will {desc}.\n"
            f"It MUST match: {m_str}\n"
            f"It must NOT match: {nm_str}\n"
            "Respond with only the regex pattern, nothing else."
        )

    @staticmethod
    def _build_day_01_31(rng: random.Random) -> tuple:
        pattern = r"^(0[1-9]|[12]\d|3[01])$"
        matches = sorted({f"{rng.randint(1, 31):02d}" for _ in range(4)})
        while len(matches) < 4:
            matches.append(f"{rng.randint(1, 31):02d}")
        non_matches = ["00", "32", str(rng.randint(1, 9)), str(rng.randint(32, 99))]
        return "match day numbers 01-31 with leading zeros", pattern, matches, non_matches

    @staticmethod
    def _build_date_ymd(rng: random.Random) -> tuple:
        pattern = r"^(19\d{2}|20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"
        matches = []
        while len(matches) < 3:
            y = f"{rng.randint(1900, 2099):04d}"
            m = f"{rng.randint(1, 12):02d}"
            d = f"{rng.randint(1, 31):02d}"
            candidate = f"{y}-{m}-{d}"
            if candidate not in matches:
                matches.append(candidate)
        non_matches = [
            f"{rng.choice([1899, 2100]):04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 31):02d}",
            f"{rng.randint(1900, 2099):04d}-{rng.choice([0, 13]):02d}-{rng.randint(1, 31):02d}",
            f"{rng.randint(1900, 2099):04d}/{rng.randint(1, 12):02d}/{rng.randint(1, 31):02d}",
        ]
        return "match dates in YYYY-MM-DD format (1900-2099)", pattern, matches, non_matches

    @staticmethod
    def _build_email_simple(rng: random.Random) -> tuple:
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$"
        alpha = "abcdefghijklmnopqrstuvwxyz"

        def _word(lo: int = 3, hi: int = 8) -> str:
            return "".join(rng.choice(alpha) for _ in range(rng.randint(lo, hi)))

        tld = rng.choice(["com", "org", "net", "io"])
        matches = [f"{_word()}@{_word()}.{tld}" for _ in range(3)]
        non_matches = [
            f"{_word()}@.{tld}",
            f"@{_word()}.{tld}",
            f"{_word()}@{_word()}",
        ]
        return "match simple email addresses", pattern, matches, non_matches

    @staticmethod
    def _build_digits_run(rng: random.Random) -> tuple:
        length = rng.randint(3, 6)
        pattern = rf"^\d{{{length}}}$"
        matches = ["".join(str(rng.randint(0, 9)) for _ in range(length)) for _ in range(3)]
        non_matches = [
            "".join(str(rng.randint(0, 9)) for _ in range(length + 1)),
            "".join(str(rng.randint(0, 9)) for _ in range(max(1, length - 1))),
            "abc" + "".join(str(rng.randint(0, 9)) for _ in range(max(1, length - 2))),
        ]
        return f"match exactly {length}-digit strings", pattern, matches, non_matches

    @staticmethod
    def _build_hex_color(rng: random.Random) -> tuple:
        pattern = r"^#[0-9a-fA-F]{6}$"
        hex_chars = "0123456789abcdef"
        matches = ["#" + "".join(rng.choice(hex_chars) for _ in range(6)) for _ in range(3)]
        non_matches = [
            "#" + "".join(rng.choice(hex_chars) for _ in range(3)),
            "".join(rng.choice(hex_chars) for _ in range(6)),
            "#" + "".join(rng.choice(hex_chars) for _ in range(6)) + "z",
        ]
        return "match hex color codes like #a1b2c3", pattern, matches, non_matches

    @staticmethod
    def _build_time_hhmm(rng: random.Random) -> tuple:
        pattern = r"^([01]\d|2[0-3]):[0-5]\d$"
        matches = [f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}" for _ in range(3)]
        non_matches = [
            f"{rng.choice([24, 25]):02d}:{rng.randint(0, 59):02d}",
            f"{rng.randint(0, 23):02d}:{rng.choice([60, 61]):02d}",
            f"{rng.randint(0, 23)}{rng.randint(0, 59):02d}",
        ]
        return "match times in HH:MM 24-hour format", pattern, matches, non_matches

    @staticmethod
    def _build_ipv4_octet(rng: random.Random) -> tuple:
        pattern = r"^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$"

        def _ip() -> str:
            return ".".join(str(rng.randint(0, 255)) for _ in range(4))

        matches = [_ip() for _ in range(3)]
        non_matches = [
            f"256.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}",
            ".".join(str(rng.randint(0, 255)) for _ in range(3)),
            ".".join(str(rng.randint(0, 255)) for _ in range(5)),
        ]
        return "match valid IPv4 addresses", pattern, matches, non_matches

    @staticmethod
    def _build_identifier(rng: random.Random) -> tuple:
        pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"  # pragma: allowlist secret
        alphanum = alpha + "0123456789"

        def _ident() -> str:
            first = rng.choice(alpha)
            rest = "".join(rng.choice(alphanum) for _ in range(rng.randint(2, 8)))
            return first + rest

        matches = [_ident() for _ in range(3)]
        non_matches = [
            str(rng.randint(1, 9)) + _ident(),
            "",
            " " + _ident(),
        ]
        return "match valid programming identifiers", pattern, matches, non_matches


class MathTaskGen:
    """Generates arithmetic tasks with 2-4 operands."""

    def generate(self, n: int, seed: int = 42) -> List[Task]:
        rng = random.Random(seed)
        tasks: List[Task] = []
        for i in range(n):
            num_operands = rng.randint(2, 4)
            operands = [rng.randint(1, 100) for _ in range(num_operands)]
            ops = [rng.choice(["+", "-", "*"]) for _ in range(num_operands - 1)]

            expr_parts: List[str] = [str(operands[0])]
            for op, operand in zip(ops, operands[1:]):
                expr_parts.append(op)
                expr_parts.append(str(operand))
            expr_str = " ".join(expr_parts)
            target = self._eval_expr(operands, ops)

            prompt = f"Compute: {expr_str}\nRespond with the number only."

            def _make_evaluator(expected: float) -> Callable[[str], bool]:
                def _eval(answer: str) -> bool:
                    clean = answer.strip()
                    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", clean)
                    if match is None:
                        return False
                    try:
                        predicted = float(match.group(0))
                    except ValueError:
                        return False
                    return math.isclose(predicted, expected, rel_tol=1e-3)

                return _eval

            tasks.append(
                Task(
                    task_id=f"math_{seed}_{i:04d}",
                    family="math",
                    prompt=prompt,
                    target=target,
                    _evaluator=_make_evaluator(target),
                )
            )
        return tasks

    @staticmethod
    def _eval_expr(operands: List[int], ops: List[str]) -> float:
        """Evaluate left-to-right (no precedence) to match what the prompt implies."""
        result = float(operands[0])
        for op, operand in zip(ops, operands[1:]):
            if op == "+":
                result += operand
            elif op == "-":
                result -= operand
            elif op == "*":
                result *= operand
        return result


class WordCountTaskGen:
    """Generates word-counting tasks."""

    WORD_POOL = [
        "adaptive",
        "agents",
        "align",
        "biosphere",
        "clusters",
        "coordinate",
        "feedback",
        "gradients",
        "harvest",
        "iterate",
        "learning",
        "morphology",
        "organelles",
        "pipeline",
        "quanta",
        "receptors",
        "signals",
        "symbiotic",
        "tensors",
        "uplift",
        "vectors",
        "workflow",
        "yield",
        "zenith",
        "compute",
    ]

    SENTENCES = [
        "Symbiotic agents cooperate in diverse environments",
        "LoRA adapters evolve rapidly under competitive pressure",
        "Energy tickets enforce scarcity in the ecosystem",
        "Autonomous organelles coordinate under fluctuating budgets",
        "Dynamic niches encourage exploration throughout the colony",
        "Telemetry snapshots surface variance and uplift deltas per cell",
        "Meta mutations jitter the controller and ticket price together",
    ]

    def generate(self, n: int, seed: int = 42) -> List[Task]:
        rng = random.Random(seed)
        tasks: List[Task] = []
        for i in range(n):
            mode = rng.choice(["count_words", "count_target_word"])
            if mode == "count_words":
                sentence = rng.choice(self.SENTENCES)
                word_count = len(sentence.split())
                prompt = (
                    f"Count the number of words in the sentence: '{sentence}'. "
                    "Respond with an integer."
                )
                target = word_count
            else:
                target_word = rng.choice(self.WORD_POOL)
                count = rng.randint(1, 5)
                other_words = [w for w in self.WORD_POOL if w != target_word]
                filler = rng.sample(other_words, k=min(rng.randint(3, 7), len(other_words)))
                sentence_words = filler + [target_word] * count
                rng.shuffle(sentence_words)
                sentence = " ".join(sentence_words)
                prompt = (
                    f'How many times does the word "{target_word}" appear in: '
                    f"'{sentence}'? Respond with an integer."
                )
                target = count

            def _make_evaluator(expected: int) -> Callable[[str], bool]:
                def _eval(answer: str) -> bool:
                    clean = answer.strip()
                    digits = "".join(ch for ch in clean if ch.isdigit())
                    if digits:
                        try:
                            return int(digits) == expected
                        except ValueError:
                            return False
                    return False

                return _eval

            tasks.append(
                Task(
                    task_id=f"wordcount_{seed}_{i:04d}",
                    family="word.count",
                    prompt=prompt,
                    target=target,
                    _evaluator=_make_evaluator(target),
                )
            )
        return tasks


GENERATORS = {
    "regex": RegexTaskGen,
    "math": MathTaskGen,
    "word.count": WordCountTaskGen,
}


def generate_tasks(family: str, n: int, seed: int = 42) -> List[Task]:
    """Generate n tasks for the given family."""
    gen_cls = GENERATORS.get(family)
    if gen_cls is None:
        raise ValueError(f"Unknown task family: {family!r}. Available: {list(GENERATORS)}")
    return gen_cls().generate(n, seed=seed)
