from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class PooledZones:
    representations: Tensor
    mask: Tensor
    zone_indices: Tensor


class ZonePooling(nn.Module):
    """Pool the observed members of each episode-local cipher region."""

    def __init__(self, d_model: int, mode: str = "mean") -> None:
        super().__init__()
        if mode not in {"mean", "attention"}:
            raise ValueError("zone pooling mode must be mean or attention")
        self.mode = mode
        self.attention_score = nn.Linear(d_model, 1, bias=False) if mode == "attention" else None

    def forward(self, hidden: Tensor, cipher_zone_ids: Tensor, attention_mask: Tensor) -> PooledZones:
        batch, _length, d_model = hidden.shape
        unique_by_batch = [torch.unique(cipher_zone_ids[row][attention_mask[row]], sorted=True) for row in range(batch)]
        max_zones = max((len(item) for item in unique_by_batch), default=0)
        if max_zones == 0:
            raise ValueError("zone pooling requires at least one valid token")
        pooled = hidden.new_zeros(batch, max_zones, d_model)
        zone_mask = torch.zeros(batch, max_zones, dtype=torch.bool, device=hidden.device)
        zone_indices = torch.full(
            (batch, max_zones), -1, dtype=cipher_zone_ids.dtype, device=hidden.device
        )
        for row, unique_zones in enumerate(unique_by_batch):
            for column, zone in enumerate(unique_zones):
                member_mask = attention_mask[row] & (cipher_zone_ids[row] == zone)
                members = hidden[row, member_mask]
                if self.attention_score is None:
                    representation = members.mean(dim=0)
                else:
                    weights = torch.softmax(self.attention_score(members).squeeze(-1), dim=0)
                    representation = (weights.unsqueeze(-1) * members).sum(dim=0)
                pooled[row, column] = representation
                zone_mask[row, column] = True
                zone_indices[row, column] = zone
        return PooledZones(pooled, zone_mask, zone_indices)


def pool_zone_targets(
    token_labels: Tensor,
    cipher_zone_ids: Tensor,
    pooled: PooledZones,
    attention_mask: Tensor,
) -> Tensor:
    """Build one supervised label per pooled zone and reject inconsistent metadata."""
    targets = torch.full_like(pooled.zone_indices, -100)
    for row in range(token_labels.shape[0]):
        for column in range(pooled.zone_indices.shape[1]):
            if not pooled.mask[row, column]:
                continue
            zone = pooled.zone_indices[row, column]
            labels = torch.unique(
                token_labels[row][attention_mask[row] & (cipher_zone_ids[row] == zone)]
            )
            if labels.numel() != 1:
                raise ValueError("one cipher zone maps to multiple Hangul labels in an episode")
            targets[row, column] = labels.item()
    return targets

