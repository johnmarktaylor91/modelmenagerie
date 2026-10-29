"""A small MLX module for the worker's MLX preview smoke (a test fixture, no real name)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

VARIANTS = {"m900031": {}, "m900032": {"model": "dropout"}}


class SmallMLP(nn.Module):
    def __init__(self, dropout: bool) -> None:
        super().__init__()
        self.lin1 = nn.Linear(8, 16)
        self.drop = nn.Dropout(0.5) if dropout else None
        self.lin2 = nn.Linear(16, 4)

    def __call__(self, x: mx.array) -> mx.array:
        y = nn.relu(self.lin1(x))
        return self.lin2(self.drop(y) if self.drop is not None else y)


def build(cfg: dict) -> nn.Module:
    return SmallMLP(dropout=cfg.get("model") == "dropout")
