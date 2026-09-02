from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from ..models.decoder import DecoderOutput
from ..models.zone_pooling import pool_zone_targets


@dataclass
class LossBreakdown:
    total: Tensor
    supervised: Tensor
    local: Tensor
    entropy: Tensor

    def detached(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "supervised_loss": float(self.supervised.detach()),
            "local_loss": float(self.local.detach()),
            "entropy_loss": float(self.entropy.detach()),
        }


def local_consistency_loss(
    token_scores: Tensor,
    cipher_values: Tensor,
    attention_mask: Tensor,
    distance_scale: float = 256.0,
    scores_are_probabilities: bool = False,
) -> Tensor:
    probabilities = (
        token_scores.clamp_min(0.0)
        if scores_are_probabilities
        else torch.softmax(token_scores, dim=-1)
    )
    difference = probabilities.unsqueeze(2) - probabilities.unsqueeze(1)
    divergence = difference.square().sum(dim=-1)
    distance = (
        cipher_values.unsqueeze(2).to(token_scores.dtype)
        - cipher_values.unsqueeze(1).to(token_scores.dtype)
    ).abs()
    weight = torch.exp(-distance / distance_scale)
    valid = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)
    diagonal = torch.eye(token_scores.shape[1], dtype=torch.bool, device=token_scores.device)
    valid = valid & ~diagonal.unsqueeze(0)
    denominator = (weight * valid).sum().clamp_min(1.0)
    return (divergence * weight * valid).sum() / denominator


def entropy_regularizer(scores: Tensor, mask: Tensor, scores_are_probabilities: bool) -> Tensor:
    probabilities = scores.clamp_min(1e-8) if scores_are_probabilities else torch.softmax(scores, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    return entropy[mask].mean()


def compute_loss(
    output: DecoderOutput,
    batch: dict,
    uses_sinkhorn: bool = False,
    lambda_local: float = 0.0,
    lambda_entropy: float = 0.0,
    distance_scale: float = 256.0,
) -> LossBreakdown:
    if output.pooled_zones is None:
        supervised = F.cross_entropy(
            output.token_scores.reshape(-1, output.token_scores.shape[-1]),
            batch["zone_labels"].reshape(-1),
            ignore_index=-100,
        )
    else:
        assert output.zone_scores is not None
        targets = pool_zone_targets(
            batch["zone_labels"],
            batch["cipher_zone_ids"],
            output.pooled_zones,
            batch["attention_mask"],
        )
        if uses_sinkhorn:
            supervised = F.nll_loss(
                output.zone_scores.clamp_min(1e-8).log().reshape(-1, output.zone_scores.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
        else:
            supervised = F.cross_entropy(
                output.zone_scores.reshape(-1, output.zone_scores.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )

    zero = supervised.new_zeros(())
    local = (
        local_consistency_loss(
            output.token_scores,
            batch["cipher_values"],
            batch["attention_mask"],
            distance_scale,
            uses_sinkhorn and output.pooled_zones is not None,
        )
        if lambda_local
        else zero
    )
    entropy = (
        entropy_regularizer(
            output.zone_scores,
            output.pooled_zones.mask,
            uses_sinkhorn,
        )
        if lambda_entropy and output.zone_scores is not None and output.pooled_zones is not None
        else zero
    )
    total = supervised + lambda_local * local + lambda_entropy * entropy
    return LossBreakdown(total, supervised, local, entropy)

