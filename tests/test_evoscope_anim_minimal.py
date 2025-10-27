from pathlib import Path
import importlib
import sys


def test_evoscope_anim_without_imageio(tmp_path: Path) -> None:
    # Ensure no imageio stub is present
    sys.modules.pop("imageio", None)
    run = tmp_path / "run"
    run.mkdir()
    (run / "gen_summaries.jsonl").write_text(
        "\n".join(
            [
                '{"generation": 1, "avg_roi": 0.2, "merges": 0}',
                '{"generation": 2, "avg_roi": 0.4, "merges": 1}',
            ]
        )
    )
    # import after sys.modules manipulation
    sys.path.insert(0, str((run.parents[1] / "scripts")))
    mod = importlib.import_module("evoscope_anim")
    # Render frames without imageio present
    records = mod.load_jsonl(run / "gen_summaries.jsonl")
    frames = mod.render_frames(records)
    assert isinstance(frames, list) and len(frames) == 2
    # exercise main() path; writing is best-effort but should not crash
    argv = sys.argv
    try:
        sys.argv = ["evoscope_anim.py", str(run), "--gif"]
        mod.main()
    finally:
        sys.argv = argv

