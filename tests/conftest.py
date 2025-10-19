import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest
import torch

from symbiont_ecology.organelles.hebbian import HebbianLoRAOrganelle


@pytest.fixture(autouse=True)
def _stub_gemma(monkeypatch):
    class _StubTokenizer:
        def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None):
            ids = []
            for text in texts:
                length = max(1, len(text.split()))
                ids.append([1] * length)
            return {"input_ids": ids}

    class _StubBackbone:
        def __init__(self, host_config):
            self.device = torch.device("cpu")
            self.hidden_size = 512
            self.max_length = host_config.max_sequence_length
            self.tokenizer = _StubTokenizer()
            self.model = None

        def encode_text(self, text_batch, device=None):
            batch = list(text_batch)
            return torch.zeros(len(batch), self.hidden_size, dtype=torch.float32)

        def parameters(self):  # pragma: no cover - simple stub for tests
            return []

    class _StubPEFTOrganelle(HebbianLoRAOrganelle):
        def __init__(self, backbone, rank, context, activation_bias=0.0):
            super().__init__(
                input_dim=backbone.hidden_size,
                rank=rank,
                dtype=torch.float32,
                device=torch.device("cpu"),
                context=context,
                activation_bias=activation_bias,
            )
            self.rank = rank

        def trainable_parameters(self) -> int:  # pragma: no cover - simple stub
            return super().trainable_parameters()

    monkeypatch.setattr("symbiont_ecology.host.kernel.GemmaBackbone", _StubBackbone)
    monkeypatch.setattr("symbiont_ecology.organelles.peft_hebbian.HebbianPEFTOrganelle", _StubPEFTOrganelle)
