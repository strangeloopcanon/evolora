from __future__ import annotations

from pathlib import Path

import torch

from symbiont_ecology.benchmarks import BenchmarkCase, BenchmarkSuite, run_benchmark_suite
from symbiont_ecology.benchmarks.stubs import BenchmarkStubBackbone, solve_prompt
from symbiont_ecology.config import HostConfig


def test_benchmark_stub_backbone_encode_text_is_deterministic():
    backbone = BenchmarkStubBackbone(
        HostConfig(
            backbone_model="stub",
            tokenizer="stub",
            dtype="float32",
            device="cpu",
            max_sequence_length=16,
        )
    )
    first = backbone.encode_text(["hello world", "hi"])
    second = backbone.encode_text(["hello world", "hi"])
    assert first.shape == (2, backbone.hidden_size)
    assert torch.allclose(first, second)


def test_benchmark_stub_prompt_solver_covers_core_families():
    assert solve_prompt("Add 2 and 7. Respond with the number only.") == "9"
    assert solve_prompt("Multiply 3 by 4. Respond with the number only.") == "12"
    assert (
        solve_prompt(
            "Given the numbers 3, 1, 2, produce a valid JSON array containing them sorted ascending."
        )
        == "[1, 2, 3]"
    )
    assert (
        solve_prompt(
            "Sort the following letters alphabetically and respond with the sorted string: b a c"
        )
        == "abc"
    )
    assert (
        solve_prompt(
            "Count the number of words in the sentence: 'Symbiotic agents cooperate'. Respond with an integer."
        )
        == "3"
    )
    assert (
        solve_prompt(
            "Evaluate the logical expression and respond with 'True' or 'False': TRUE AND NOT FALSE"
        )
        == "True"
    )
    assert (
        solve_prompt(
            "Given the sequence 2, 4, 6, what is the next number? Respond with the number only."
        )
        == "8"
    )
    assert solve_prompt("Compute (3 + 2) * 4 - (1 + 1). Respond with the number only.") == "18"
    assert (
        solve_prompt(
            "Convert the variable name `FooBarHTTP` to snake_case. Respond with the new name only."
        )
        == "foo_bar_http"
    )


def test_benchmark_suite_smoke_run(tmp_path: Path):
    suite = BenchmarkSuite(
        cases=[
            BenchmarkCase(
                name="smoke",
                config_path=Path("config/benchmarks/ci_single.yaml"),
                generations=1,
                batch_size=2,
                seed=123,
                backend="stub",
            )
        ]
    )
    report = run_benchmark_suite(suite=suite, output_root=tmp_path)
    assert len(report.results) == 1
    result = report.results[0]
    assert result.name == "smoke"
    assert result.metrics.episodes > 0
    assert 0.0 <= result.metrics.success_rate <= 1.0
    assert len(result.replicates) == 1
    assert result.replicates[0].seed == 123
    assert result.open_endedness.merges_total >= 0
    assert (tmp_path / "smoke" / "episodes.jsonl").exists()


def test_paper_qwen3_single_baseline_runs_fixed_episode_budget(tmp_path: Path):
    generations = 10
    batch_size = 4
    suite = BenchmarkSuite(
        cases=[
            BenchmarkCase(
                name="paper_single",
                config_path=Path("config/experiments/paper_qwen3_single.yaml"),
                generations=generations,
                batch_size=batch_size,
                seed=123,
                backend="stub",
                disable_human=True,
            )
        ]
    )
    report = run_benchmark_suite(suite=suite, output_root=tmp_path)
    assert report.results[0].metrics.episodes == generations * batch_size
