from __future__ import annotations

import math

import torch
from torch import Tensor

from .region_zone_matcher import LearnedRegionZoneMatcher, RegionZoneMatchOutput


def structural_sinkhorn(
    scores: Tensor, observed_mask: Tensor, iterations: int, temperature: float,
) -> Tensor:
    """Batch the square neutral-dummy Sinkhorn used by structural refinement.

    Inactive region rows stay in place with zero logits as neutral dummy rows.
    This is equivalent to collecting observed rows and padding with dummies,
    while avoiding a Python loop and separate GPU launches for every example.
    """
    if scores.ndim != 3 or scores.shape[-1] != scores.shape[-2]:
        raise ValueError("structural Sinkhorn requires square batched scores")
    if observed_mask.shape != scores.shape[:2]:
        raise ValueError("observed mask must align with score rows")
    if iterations < 1 or temperature <= 0:
        raise ValueError("iterations and temperature must be positive")
    if not bool(observed_mask.any(dim=-1).all()):
        raise ValueError("each Sinkhorn matrix needs an observed row")
    log_values = scores.masked_fill(~observed_mask.unsqueeze(-1), 0.0) / temperature
    for _ in range(iterations):
        log_values = log_values - torch.logsumexp(log_values, dim=-1, keepdim=True)
        log_values = log_values - torch.logsumexp(log_values, dim=-2, keepdim=True)
    log_values = log_values - torch.logsumexp(log_values, dim=-1, keepdim=True)
    return log_values.exp().masked_fill(~observed_mask.unsqueeze(-1), 0.0)


def transition_structural_score(
    assignment: Tensor, transition_counts: Tensor, log_canonical: Tensor
) -> Tensor:
    """Directed edge agreement: C S (log P)^T + C^T S log P.

    Counts retain their original scale, so beta controls evidence strength per
    observed transition. No hard assignment or gradient detachment is used.
    """
    return (
        transition_counts @ assignment @ log_canonical.transpose(-1, -2)
        + transition_counts.transpose(-1, -2) @ assignment @ log_canonical
    )


def graph_consistency_loss(
    assignment: Tensor,
    transition_counts: Tensor,
    canonical_transition: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    """Mean per-graph count NLL of Q_hat = S P S^T.

    Use unsmoothed counts. Unobserved rows of S may be zero: their counts are
    also zero, so no extra row normalization or dummy-region supervision is
    needed. Graphs without transitions contribute a differentiable zero.
    """
    if assignment.ndim != 3 or transition_counts.shape != assignment.shape:
        raise ValueError("assignment and counts must have shape [batch, zones, zones]")
    if canonical_transition.shape != assignment.shape[1:]:
        raise ValueError("canonical transition must have shape [zones, zones]")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    canonical = canonical_transition.to(assignment)
    counts = transition_counts.to(assignment)
    reconstructed = assignment @ canonical @ assignment.transpose(-1, -2)
    weighted_nll = -(counts * reconstructed.clamp_min(epsilon).log()).sum(dim=(-2, -1))
    return (weighted_nll / counts.sum(dim=(-2, -1)).clamp_min(epsilon)).mean()


class LearnedStructuralRegionZoneMatcher(LearnedRegionZoneMatcher):
    """Existing unary GNN followed by differentiable transition refinement."""

    def __init__(
        self,
        canonical_transition: Tensor,
        num_zones: int = 19,
        node_feature_dim: int = 5,
        d_model: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        sinkhorn_iterations: int = 30,
        sinkhorn_temperature: float = 1.0,
        structural_refinement_steps: int = 4,
        structural_beta: float = 0.1,
    ) -> None:
        if (
            not isinstance(structural_refinement_steps, int)
            or isinstance(structural_refinement_steps, bool)
            or structural_refinement_steps < 0
        ):
            raise ValueError("structural_refinement_steps must be a non-negative integer")
        if not math.isfinite(structural_beta) or structural_beta < 0:
            raise ValueError("structural_beta must be finite and non-negative")
        canonical = torch.as_tensor(canonical_transition).detach().clone().to(torch.float32)
        if canonical.shape != (num_zones, num_zones):
            raise ValueError("canonical transition must have shape [num_zones, num_zones]")
        if not bool(torch.isfinite(canonical).all()) or bool((canonical < 0).any()):
            raise ValueError("canonical transition must be finite and non-negative")
        if not torch.allclose(canonical.sum(-1), torch.ones_like(canonical.sum(-1)), atol=1e-5):
            raise ValueError("canonical transition rows must sum to one")
        super().__init__(
            num_zones=num_zones,
            node_feature_dim=node_feature_dim,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            sinkhorn_iterations=sinkhorn_iterations,
            sinkhorn_temperature=sinkhorn_temperature,
        )
        self.structural_refinement_steps = structural_refinement_steps
        self.structural_beta = structural_beta
        self.register_buffer("canonical_transition", canonical)

    def forward(
        self,
        node_features: Tensor,
        transition: Tensor,
        observed_mask: Tensor,
        transition_counts: Tensor,
    ) -> RegionZoneMatchOutput:
        if transition_counts.shape != transition.shape:
            raise ValueError("transition counts must align with the transition graph")
        unary = super().forward(node_features, transition, observed_mask)
        assignment = unary.assignment_probabilities
        counts = transition_counts.to(assignment)
        # Mask both endpoints; smoothed GNN transitions are never used as counts.
        counts = counts * (observed_mask.unsqueeze(-1) & observed_mask.unsqueeze(-2))
        log_canonical = self.canonical_transition.to(assignment).clamp_min(1e-8).log()
        scores = unary.scores
        for _ in range(self.structural_refinement_steps):
            structural = transition_structural_score(assignment, counts, log_canonical)
            # A constant per row cancels in Sinkhorn and reduces logit magnitude.
            structural = structural - structural.mean(dim=-1, keepdim=True)
            scores = unary.scores + self.structural_beta * structural
            assignment = structural_sinkhorn(
                scores,
                observed_mask=observed_mask,
                iterations=self.sinkhorn_iterations,
                temperature=self.sinkhorn_temperature,
            )
        return RegionZoneMatchOutput(scores, assignment, unary.region_representations)
