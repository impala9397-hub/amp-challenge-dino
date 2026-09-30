"""The generator: a causal transformer over amino-acid characters.

Architecture, verbatim from the trained checkpoint:

    vocabulary 23 · d_model 256 · 6 layers · 8 heads · FFN 1024 · GELU
    pre-norm blocks (norm_first=True) · learned positional embeddings up to 64
    4,767,255 parameters, of which the six blocks are 99.4%

There is no novel component. What makes it autoregressive is the causal mask
plus left-to-right sampling, not the block design.

Only inference lives here. Training is described in ``docs/METHOD.md``; the
released checkpoint is frozen and this repository does not retrain it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from dino_amp import tokenizer

MAXIMUM_POSITIONS = 64


@dataclass(frozen=True)
class GeneratorConfig:
    model_dimension: int = 256
    layers: int = 6
    heads: int = 8
    feed_forward_dimension: int = 1024
    dropout: float = 0.1


class CausalTransformer(nn.Module):
    """Predicts the next token given the tokens to its left."""

    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        self.config = config
        size = len(tokenizer.VOCABULARY)
        self.token_embedding = nn.Embedding(
            size, config.model_dimension, padding_idx=tokenizer.PAD_ID
        )
        self.position_embedding = nn.Embedding(MAXIMUM_POSITIONS, config.model_dimension)
        block = nn.TransformerEncoderLayer(
            config.model_dimension,
            config.heads,
            config.feed_forward_dimension,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.model_dimension)
        self.head = nn.Linear(config.model_dimension, size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        width = tokens.size(1)
        positions = torch.arange(width, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        causal = nn.Transformer.generate_square_subsequent_mask(width, device=tokens.device)
        hidden = self.encoder(hidden, mask=causal, is_causal=True)
        return self.head(self.norm(hidden))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def load(path: str | Path) -> CausalTransformer:
    """Load the frozen checkpoint onto the CPU.

    ``weights_only=True`` so that loading the file cannot execute code — the
    checkpoint is data, and a reviewer running this should not have to trust it
    as a program.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    # The saved config also records training hyper-parameters (learning rate,
    # epochs, ...). Keep only the fields that define the architecture, so an
    # older checkpoint stays loadable and a stray key is not a crash.
    stored = payload.get("config", {})
    fields = GeneratorConfig.__dataclass_fields__
    config = GeneratorConfig(**{k: v for k, v in stored.items() if k in fields})
    model = CausalTransformer(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
