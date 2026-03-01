"""Conftest for plasticity tests -- ensures src/ is on the path."""

import sys
from pathlib import Path

src_dir = str(Path(__file__).resolve().parents[2] / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
