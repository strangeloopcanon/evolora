from pathlib import Path


def test_evoscope_anim_frames(tmp_path: Path) -> None:
    # create minimal gen_summaries
    run = tmp_path / "run"
    run.mkdir()
    (run / "gen_summaries.jsonl").write_text(
        "\n".join(
            [
                '{"generation": 1, "avg_roi": 0.5, "merges": 0}',
                '{"generation": 2, "avg_roi": 1.2, "merges": 1}',
                '{"generation": 3, "avg_roi": 0.9, "merges": 0}',
            ]
        )
    )
    import sys
    import types

    sys.path.insert(0, str(run.parents[1] / "scripts"))
    # stub imageio so import works without external dep
    dummy = types.SimpleNamespace()
    dummy.imread = lambda buf: b"x"
    dummy.mimsave = lambda *args, **kwargs: None
    # create a module stub with attribute v2
    imageio_mod = types.ModuleType("imageio")
    imageio_mod.v2 = dummy  # type: ignore[attr-defined]
    sys.modules["imageio"] = imageio_mod
    from evoscope_anim import load_jsonl, render_frames  # type: ignore

    records = load_jsonl(run / "gen_summaries.jsonl")
    frames = render_frames(records)
    assert len(frames) == 3
