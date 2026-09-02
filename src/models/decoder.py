from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config import BASELINES, ModelConfig
from .baseline_transformer import BaselineTransformer
from .relational_transformer import RelationalTransformer
from .sinkhorn import sinkhorn
from .zone_matcher import ZoneMatcher
from .zone_pooling import PooledZones, ZonePooling


@dataclass
class DecoderOutput:
    token_scores: Tensor
    hidden_states: Tensor
    zone_scores: Tensor | None = None
    raw_zone_scores: Tensor | None = None
    pooled_zones: PooledZones | None = None
    attentions: list[Tensor] | None = None


class NeuralCipherDecoder(nn.Module):
    """One public model covering all five requested experimental baselines."""

    def __init__(self, config: ModelConfig, baseline: str = "relational") -> None:
        super().__init__()
        if baseline not in BASELINES:
            raise ValueError(f"unknown baseline {baseline!r}")
        config.validate()
        self.config = config
        self.baseline = baseline
        self.is_relational = baseline != "standard"
        self.uses_pooling = baseline in {"relational_pool", "relational_match", "relational_sinkhorn"}
        self.uses_matching = baseline in {"relational_match", "relational_sinkhorn"}
        self.uses_sinkhorn = baseline == "relational_sinkhorn"

        self.encoder = RelationalTransformer(config) if self.is_relational else BaselineTransformer(config)
        self.token_classifier = nn.Linear(config.d_model, config.num_zones)
        self.zone_pooling = ZonePooling(config.d_model, config.pooling) if self.uses_pooling else None
        self.zone_classifier = (
            nn.Linear(config.d_model, config.num_zones)
            if self.uses_pooling and not self.uses_matching
            else None
        )
        self.zone_matcher = (
            ZoneMatcher(config.d_model, config.num_zones, config.matching)
            if self.uses_matching
            else None
        )

    def forward(
        self,
        digits: Tensor,
        cipher_values: Tensor,
        attention_mask: Tensor,
        cipher_zone_ids: Tensor | None = None,
        return_attention: bool = False,
    ) -> DecoderOutput:
        if self.is_relational:
            hidden, attentions = self.encoder(
                digits, cipher_values, attention_mask, return_attention
            )
        else:
            hidden = self.encoder(digits, attention_mask)
            attentions = []

        if not self.uses_pooling:
            token_scores = self.token_classifier(hidden)
            return DecoderOutput(token_scores, hidden, attentions=attentions)

        if cipher_zone_ids is None:
            raise ValueError(f"baseline {self.baseline} requires cipher_zone_ids")
        assert self.zone_pooling is not None
        pooled = self.zone_pooling(hidden, cipher_zone_ids, attention_mask)
        if self.zone_matcher is not None:
            raw_zone_scores = self.zone_matcher(pooled.representations, pooled.mask)
        else:
            assert self.zone_classifier is not None
            raw_zone_scores = self.zone_classifier(pooled.representations).masked_fill(
                ~pooled.mask.unsqueeze(-1), 0.0
            )
        zone_scores = (
            sinkhorn(
                raw_zone_scores,
                row_mask=pooled.mask,
                iterations=self.config.sinkhorn_iterations,
                temperature=self.config.sinkhorn_temperature,
            )
            if self.uses_sinkhorn
            else raw_zone_scores
        )
        token_scores = self._broadcast_zone_scores(zone_scores, pooled, cipher_zone_ids, attention_mask)
        return DecoderOutput(
            token_scores=token_scores,
            hidden_states=hidden,
            zone_scores=zone_scores,
            raw_zone_scores=raw_zone_scores,
            pooled_zones=pooled,
            attentions=attentions,
        )

    @staticmethod
    def _broadcast_zone_scores(
        zone_scores: Tensor,
        pooled: PooledZones,
        cipher_zone_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        batch, length = cipher_zone_ids.shape
        result = zone_scores.new_zeros(batch, length, zone_scores.shape[-1])
        for row in range(batch):
            matches = cipher_zone_ids[row].unsqueeze(1) == pooled.zone_indices[row].unsqueeze(0)
            safe_scores = zone_scores[row].masked_fill(~pooled.mask[row].unsqueeze(-1), 0.0)
            result[row] = torch.matmul(matches.to(zone_scores.dtype), safe_scores)
        return result * attention_mask.unsqueeze(-1)
