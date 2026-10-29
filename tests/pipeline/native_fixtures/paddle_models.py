"""A small Paddle dygraph CNN for the worker's Paddle preview tests (a test fixture, no real name)."""

from __future__ import annotations

import paddle
from paddle import nn

VARIANTS = {"m900021": {}, "m900022": {"model": "function"}}


class SmallCNN(nn.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2D(8)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(8, 5)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        y = nn.functional.relu(self.bn(self.conv(x)))
        return self.fc(self.drop(y.mean(axis=[2, 3])))


def build(cfg: dict):
    if cfg.get("model") == "function":  # parameter-free: what the 2.35.3 coverage oracle passes
        return lambda x: nn.functional.relu(x) * 2.0 + paddle.tanh(x)
    return SmallCNN()
