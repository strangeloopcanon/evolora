import types

import torch

from symbiont_ecology.config import HostConfig
from symbiont_ecology.host.backbone import HFBackbone


class _StubTokenizer:
    eos_token = "<eos>"

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None):
        batch = list(texts)
        lengths = [max(1, len(str(text).split())) for text in batch]
        max_len = max(lengths) if lengths else 1
        input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
        for idx, length in enumerate(lengths):
            input_ids[idx, :length] = 1
        return {"input_ids": input_ids}


class _StubModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 8, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self.dtype = dtype
        self.dummy_param = torch.nn.Parameter(torch.zeros(1, dtype=dtype))

    def forward(self, input_ids=None, output_hidden_states=False, **kwargs):
        del output_hidden_states, kwargs
        batch, seq_len = input_ids.shape
        hidden = torch.zeros((batch, seq_len, self.config.hidden_size), dtype=self.dtype)
        return types.SimpleNamespace(hidden_states=[hidden])


def test_hf_backbone_init_and_encode_text(monkeypatch):
    tokenizer = _StubTokenizer()
    model = _StubModel(hidden_size=4, dtype=torch.float32)

    monkeypatch.setattr(
        "symbiont_ecology.host.backbone.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "symbiont_ecology.host.backbone.AutoModelForCausalLM.from_pretrained",
        lambda *_args, **_kwargs: model,
    )

    host_config = HostConfig(
        backbone_model="dummy-model",
        tokenizer="dummy-tokenizer",
        revision=None,
        dtype="float32",
        device="cpu",
        max_sequence_length=8,
    )
    backbone = HFBackbone(host_config)
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == tokenizer.eos_token
    assert backbone.hidden_size == 4
    assert model.dummy_param.requires_grad is False

    latent = backbone.encode_text(["hello world", "hi"], device=torch.device("cpu"))
    assert latent.shape == (2, 4)
    assert latent.dtype == torch.float32


def test_hf_backbone_resolve_dtype():
    assert HFBackbone._resolve_dtype("float16") is torch.float16
    assert HFBackbone._resolve_dtype("bfloat16") is torch.bfloat16
    assert HFBackbone._resolve_dtype("float32") is torch.float32
