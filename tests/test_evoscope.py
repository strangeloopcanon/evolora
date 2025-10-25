from pathlib import Path

import importlib.util
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))


def write_gen(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                '{"generation": 1, "avg_roi": 0.8, "merges": 0, "trials_created": 0, "promotions": 0, "qd_coverage": "0/6"}',
                '{"generation": 2, "avg_roi": 1.1, "merges": 0, "trials_created": 1, "promotions": 0, "qd_coverage": "1/6"}',
            ]
        )
    )


def test_evoscope_generates_html(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    gen = run / "gen_summaries.jsonl"
    write_gen(gen)
    # import and run evoscope (module name is 'evoscope' after path insert)
    import importlib
    import sys
    evoscope = importlib.import_module("evoscope")
    argv = sys.argv
    try:
        sys.argv = ["evoscope.py", str(run)]
        evoscope.main()
    finally:
        sys.argv = argv
    assert (run / "index.html").exists()
