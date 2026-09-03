from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .sinkhorn import sinkhorn


@dataclass(frozen=True)
class RegionZoneMatchOutput:
    scores: Tensor
    assignment_probabilities: Tensor
    region_representations: Tensor


class RegionGraphLayer(nn.Module):
    """Permutation-equivariant directed transition message passing."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.out_projection = nn.Linear(d_model, d_model)
        self.in_projection = nn.Linear(d_model, d_model)
        self.update = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden: Tensor, transition: Tensor) -> Tensor:
        outgoing = torch.matmul(transition, self.out_projection(hidden))
        incoming_weights = transition.transpose(1, 2)
        incoming_weights = incoming_weights / incoming_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        incoming = torch.matmul(incoming_weights, self.in_projection(hidden))
        update = self.update(torch.cat((hidden, outgoing, incoming), dim=-1))
        return self.norm(hidden + update)


class LearnedRegionZoneMatcher(nn.Module):
    """Match anonymous transition-graph nodes to semantic zone prototypes."""

    def __init__(
        self,
        num_zones: int = 19,
        node_feature_dim: int = 5,
        d_model: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        sinkhorn_iterations: int = 30,
        sinkhorn_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_zones < 2 or d_model < 1 or num_layers < 1:
            raise ValueError("matcher dimensions and layer count must be positive")
        self.num_zones = num_zones
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_temperature = sinkhorn_temperature
        self.input_projection = nn.Sequential(
            nn.Linear(node_feature_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.layers = nn.ModuleList(
            RegionGraphLayer(d_model, dropout) for _ in range(num_layers)
        )
        self.semantic_prototypes = nn.Parameter(torch.empty(num_zones, d_model))
        nn.init.normal_(self.semantic_prototypes, std=d_model**-0.5)
        self.compatibility = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, 1),
        )

    def forward(
        self,
        node_features: Tensor,
        transition: Tensor,
        observed_mask: Tensor,
    ) -> RegionZoneMatchOutput:
        if transition.ndim != 3 or transition.shape[1:] != (
            self.num_zones,
            self.num_zones,
        ):
            raise ValueError("transition must have shape [batch, num_zones, num_zones]")
        if node_features.shape[:2] != transition.shape[:2]:
            raise ValueError("node features must align with transition rows")
        if observed_mask.shape != transition.shape[:2]:
            raise ValueError("observed mask must align with transition rows")
        hidden = self.input_projection(node_features)
        for layer in self.layers:
            hidden = layer(hidden, transition)
        region = hidden.unsqueeze(2)
        prototype = self.semantic_prototypes.view(1, 1, self.num_zones, -1)
        difference = region - prototype
        pair = torch.cat(
            (
                region.expand(-1, -1, self.num_zones, -1),
                prototype.expand(hidden.shape[0], self.num_zones, -1, -1),
                difference,
                difference.abs(),
            ),
            dim=-1,
        )
        scores = self.compatibility(pair).squeeze(-1)
        probabilities = sinkhorn(
            scores,
            row_mask=observed_mask,
            iterations=self.sinkhorn_iterations,
            temperature=self.sinkhorn_temperature,
        )
        return RegionZoneMatchOutput(scores, probabilities, hidden)


def observed_permutation_nll(
    assignment_probabilities: Tensor,
    true_assignment: Tensor,
    observed_mask: Tensor,
) -> Tensor:
    if assignment_probabilities.shape[:2] != true_assignment.shape:
        raise ValueError("assignment probabilities and targets do not align")
    selected = assignment_probabilities.gather(
        2, true_assignment.unsqueeze(-1)
    ).squeeze(-1)
    if not bool(observed_mask.any()):
        raise ValueError("learned matcher loss requires an observed region")
    return -selected[observed_mask].clamp_min(1e-8).log().mean()
