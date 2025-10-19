"""Gemma backbone integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from symbiont_ecology.config import HostConfig
from symbiont_ecology.utils.torch_utils import ensure_dtype, resolve_device


@dataclass
class GemmaOutputs:
    latent: torch.Tensor


class GemmaBackbone:
    """Wrapper around Gemma for frozen-host operation."""

    def __init__(self, host_config: HostConfig) -> None:
        self.host_config = host_config
        self.device = resolve_device(host_config.device)
        model_id = host_config.backbone_model
        tokenizer_id = host_config.tokenizer or model_id
        self.tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 - controlled model id
            tokenizer_id,
            revision=host_config.revision,
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = self._resolve_dtype(host_config.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(  # nosec B615 - controlled model id
            model_id,
            dtype=dtype,
            device_map=None,
            revision=host_config.revision,
        )
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = self.model.config.hidden_size
        self.max_length = host_config.max_sequence_length

    def encode_text(
        self, text_batch: Iterable[str], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        target_device = device or self.device
        encoded = self.tokenizer(
            list(text_batch),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: tensor.to(self.device) for key, tensor in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            latent = hidden[:, -1, :]
        latent = cast(torch.Tensor, ensure_dtype(latent, self.model.dtype)).to(target_device)
        return latent

    @staticmethod
    def _resolve_dtype(name: str) -> torch.dtype:
        if name == "float16":
            return torch.float16
        if name == "bfloat16":
            return torch.bfloat16
        return torch.float32


__all__ = ["GemmaBackbone"]
