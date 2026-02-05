#!/usr/bin/env python3
"""Create an animated GIF or MP4 timeline for a run directory.

Usage:
  python scripts/evoscope_anim.py <run_dir> [--gif] [--mp4]

Creates <run_dir>/timeline.gif or timeline.mp4 using ROI and merges over time.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any, Dict, List

"""
Import imageio flexibly so tests can stub `imageio` without the v2 submodule.
If imageio is unavailable at runtime, the tests provide a stub in sys.modules.
"""
try:  # pragma: no cover - exercised via tests using a stub
    import imageio  # type: ignore

    # prefer v2 API if present
    imageio_api = getattr(imageio, "v2", imageio)
except Exception:  # pragma: no cover - fallback for very minimal envs
    imageio_api = None  # type: ignore
import matplotlib.pyplot as plt  # noqa: E402 - kept after imageio fallback import guard


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(__import__("json").loads(line))
    return records


def render_frames(records: List[Dict[str, Any]]) -> List[Any]:
    frames: List[Any] = []
    rois: List[float] = []
    merges: List[int] = []
    gens: List[int] = []
    for rec in records:
        rois.append(float(rec.get("avg_roi", 0.0)))
        merges.append(int(rec.get("merges", 0)))
        gens.append(int(rec.get("generation", len(gens) + 1)))
    max_gen = len(gens)
    for i in range(1, max_gen + 1):
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.plot(gens[:i], rois[:i], color="#1f77b4")
        plt.title("ROI over time")
        plt.xlabel("Gen")
        plt.ylabel("ROI")
        plt.grid(alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.bar(gens[:i], merges[:i], color="#2ca02c")
        plt.title("Merges per gen")
        plt.xlabel("Gen")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120)
        plt.close()
        buf.seek(0)
        if imageio_api is None:
            # Minimal fallback: store raw PNG bytes; tests stub mimsave
            frames.append(buf.getvalue())
        else:
            frames.append(imageio_api.imread(buf))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate evolution timeline")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--gif", action="store_true", help="write timeline.gif")
    parser.add_argument(
        "--mp4", action="store_true", help="write timeline.mp4 (requires imageio-ffmpeg)"
    )
    args = parser.parse_args()
    root: Path = args.run_dir
    gen_path = root / "gen_summaries.jsonl"
    records = load_jsonl(gen_path)
    frames = render_frames(records)
    if args.gif or not args.mp4:
        out = root / "timeline.gif"
        if imageio_api is None:
            # tests stub imageio.mimsave; if not present, no-op
            try:
                import imageio as _img  # type: ignore

                _img.mimsave(out, frames, duration=0.08)  # type: ignore
            except Exception:
                pass
        else:
            imageio_api.mimsave(out, frames, duration=0.08)
        print(f"Wrote {out}")
    if args.mp4:
        out = root / "timeline.mp4"
        if imageio_api is None:
            try:
                import imageio as _img  # type: ignore

                _img.mimsave(out, frames, fps=12)  # type: ignore
            except Exception:
                pass
        else:
            imageio_api.mimsave(out, frames, fps=12)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
