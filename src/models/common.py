from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class DigitProjection(nn.Module):
    """Low-information numeric input: a projection of digits, never a value lookup."""

    def __init__(self, num_digits: int, d_model: int, use_absolute_digits: bool = True) -> None:
        super().__init__()
        self.use_absolute_digits = use_absolute_digits
        self.projection = nn.Linear(num_digits, d_model)

    def forward(self, digits: Tensor) -> Tensor:
        source = digits / 9.0 if self.use_absolute_digits else torch.zeros_like(digits)
        return self.projection(source)


def sinusoidal_positions(length: int, d_model: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    even = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
    scale = torch.exp(even * (-math.log(10_000.0) / d_model))
    result = torch.zeros(length, d_model, device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(position * scale)
    if d_model > 1:
        result[:, 1::2] = torch.cos(position * scale[: result[:, 1::2].shape[1]])
    return result.to(dtype=dtype)

