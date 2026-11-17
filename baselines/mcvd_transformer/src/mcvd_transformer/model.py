"""MCvD-Transformer architecture used as a supervised curve-surrogate baseline.

The attention convention deliberately follows the public MCvD-Transformer
implementation: ``key_dim`` is the width of *each* head, rather than the
embedding width divided by the number of heads as in ``nn.MultiheadAttention``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class KerasStyleMultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with the Keras per-head ``key_dim`` rule."""

    def __init__(self, embed_dim: int, num_heads: int, key_dim: int) -> None:
        super().__init__()
        if min(embed_dim, num_heads, key_dim) < 1:
            raise ValueError("Attention dimensions must be positive.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        projection_dim = num_heads * key_dim
        self.query = nn.Linear(embed_dim, projection_dim)
        self.key = nn.Linear(embed_dim, projection_dim)
        self.value = nn.Linear(embed_dim, projection_dim)
        self.output = nn.Linear(projection_dim, embed_dim)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, entities, _ = tensor.shape
        return tensor.reshape(batch, entities, self.num_heads, self.key_dim).transpose(1, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.query(inputs))
        key = self._split_heads(self.key(inputs))
        value = self._split_heads(self.value(inputs))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.key_dim)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, value).transpose(1, 2).contiguous()
        batch, entities, _, _ = attended.shape
        return self.output(attended.reshape(batch, entities, self.num_heads * self.key_dim))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, key_dim: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = KerasStyleMultiHeadSelfAttention(embed_dim, num_heads, key_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1.0e-6)
        self.feed_forward = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim))
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(embed_dim, eps=1.0e-6)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.attention_norm(inputs + self.attention_dropout(self.attention(inputs)))
        return self.feed_forward_norm(hidden + self.feed_forward_dropout(self.feed_forward(hidden)))


class MCvDTransformer(nn.Module):
    """Transformer that maps shell geometry and transmitter position to ``q(t)``."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        cfg = config["model"]
        self.max_entities = int(cfg.get("max_entities", 15))
        self.entity_dim = int(cfg.get("entity_dim", 13))
        self.numerical_dim = int(cfg.get("numerical_dim", 7))
        self.n_time = int(config["time"].get("n_samples", 800))
        self.zero_padding = int(config["time"].get("zero_padding", 10))
        if self.entity_dim != 13 or self.numerical_dim != 7:
            raise ValueError("The MCvD encoders use entity_dim=13 and numerical_dim=7.")
        output_size = self.n_time + self.zero_padding
        self.transformer = nn.Sequential(*[
            TransformerBlock(
                embed_dim=self.entity_dim,
                num_heads=int(cfg.get("num_heads", 16)),
                key_dim=int(cfg.get("key_dim", 13)),
                ff_dim=int(cfg.get("ff_dim", 32)),
                dropout=float(cfg.get("dropout", 0.1)),
            )
            for _ in range(int(cfg.get("num_transformer_layers", 2)))
        ])
        shared_widths = [int(value) for value in cfg.get("shared_widths", [400, 800, 1600])]
        max_widths = [int(value) for value in cfg.get("max_widths", [1600, 800, 400])]
        if not shared_widths or not max_widths:
            raise ValueError("model.shared_widths and model.max_widths must not be empty.")

        shared_layers: list[nn.Module] = []
        width = self.max_entities * self.entity_dim + self.numerical_dim
        for next_width in shared_widths:
            shared_layers.extend([nn.Linear(width, next_width), nn.ReLU()])
            width = next_width
        self.shared_mlp = nn.Sequential(*shared_layers)
        self.shape_head = nn.Sequential(
            nn.Linear(shared_widths[-1], int(cfg.get("shape_hidden", 2 * output_size))),
            nn.ReLU(),
            nn.Linear(int(cfg.get("shape_hidden", 2 * output_size)), output_size),
        )
        max_layers: list[nn.Module] = []
        width = shared_widths[-1]
        for next_width in max_widths:
            max_layers.extend([nn.Linear(width, next_width), nn.ReLU()])
            width = next_width
        max_layers.append(nn.Linear(width, 1))
        self.max_head = nn.Sequential(*max_layers)

    def forward(self, topology: torch.Tensor, numerical: torch.Tensor) -> dict[str, torch.Tensor]:
        if topology.ndim != 3 or tuple(topology.shape[1:]) != (self.max_entities, self.entity_dim):
            raise ValueError(
                f"Expected topology shape (batch, {self.max_entities}, {self.entity_dim}), got {tuple(topology.shape)}"
            )
        if numerical.ndim != 2 or numerical.shape[1] != self.numerical_dim:
            raise ValueError(f"Expected numerical shape (batch, {self.numerical_dim}), got {tuple(numerical.shape)}")
        encoded = self.transformer(topology).flatten(start_dim=1)
        hidden = self.shared_mlp(torch.cat([encoded, numerical], dim=1))
        shape = self.shape_head(hidden)
        maximum = self.max_head(hidden)
        return {"shape": shape, "maximum": maximum, "curve": shape * maximum}


def build_mcvd_transformer(config: dict[str, Any]) -> MCvDTransformer:
    return MCvDTransformer(config)

