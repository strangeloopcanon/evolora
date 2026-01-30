"""Minimal Modal connectivity sanity check.

Run:
  .venv311/bin/modal run -q scripts/modal_sanity.py
"""

from __future__ import annotations

import modal

app = modal.App("evolora-sanity")


@app.function(image=modal.Image.debian_slim(python_version="3.11"))
def hello() -> str:
    return "ok"


@app.local_entrypoint()
def main() -> None:
    print(hello.remote())
