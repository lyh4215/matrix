from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ZoneMatcher(nn.Module):
    """Match pooled cipher zones to learnable Hangul-zone prototypes."""

    def __init__(self, d_model: int, num_hangul_zones: int = 19, mode: str = "relational") -> None:
        super().__init__()
        if mode not in {"dot", "relational"}:
            raise ValueError("zone matching mode must be dot or relational")
        self.mode = mode
        self.prototypes = nn.Parameter(torch.empty(num_hangul_zones, d_model))
        nn.init.normal_(self.prototypes, std=d_model**-0.5)
        self.relation_mlp = (
            nn.Sequential(
                nn.Linear(4 * d_model, 2 * d_model),
                nn.GELU(),
                nn.Linear(2 * d_model, 1),
            )
            if mode == "relational"
            else None
        )

    def forward(self, cipher_zones: Tensor, zone_mask: Tensor) -> Tensor:
        if self.relation_mlp is None:
            scores = torch.matmul(cipher_zones, self.prototypes.t()) / math.sqrt(cipher_zones.shape[-1])
        else:
            cipher = cipher_zones.unsqueeze(2)
            prototype = self.prototypes.view(1, 1, *self.prototypes.shape)
            difference = cipher - prototype
            relation = torch.cat(
                (
                    cipher.expand(-1, -1, self.prototypes.shape[0], -1),
                    prototype.expand(cipher.shape[0], cipher.shape[1], -1, -1),
                    difference,
                    difference.abs(),
                ),
                dim=-1,
            )
            scores = self.relation_mlp(relation).squeeze(-1)
        # Invalid pooled rows are ignored by their explicit mask. Zeros are safer
        # than all--inf rows for cross entropy and Sinkhorn autograd.
        return scores.masked_fill(~zone_mask.unsqueeze(-1), 0.0)
