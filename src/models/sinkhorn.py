from __future__ import annotations

import torch
from torch import Tensor


def sinkhorn(
    scores: Tensor,
    row_mask: Tensor | None = None,
    column_mask: Tensor | None = None,
    iterations: int = 20,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Masked log-domain rectangular Sinkhorn normalization.

    Valid rows have mass 1. In an R x C matrix valid columns have target mass
    R/C, which is the compatible rectangular analogue of double stochasticity.
    For R == C, both row and column sums approach one.
    """
    if scores.ndim != 3:
        raise ValueError("scores must have shape [batch, rows, columns]")
    if iterations < 1 or temperature <= 0:
        raise ValueError("iterations and temperature must be positive")
    batch, rows, columns = scores.shape
    if row_mask is None:
        row_mask = torch.ones(batch, rows, dtype=torch.bool, device=scores.device)
    if column_mask is None:
        column_mask = torch.ones(batch, columns, dtype=torch.bool, device=scores.device)
    if bool((row_mask.sum(dim=1) == 0).any()) or bool((column_mask.sum(dim=1) == 0).any()):
        raise ValueError("each Sinkhorn matrix needs a valid row and column")

    normalized_batches: list[Tensor] = []
    for batch_index in range(batch):
        row_indices = row_mask[batch_index].nonzero(as_tuple=False).squeeze(1)
        column_indices = column_mask[batch_index].nonzero(as_tuple=False).squeeze(1)
        active = scores[batch_index].index_select(0, row_indices).index_select(1, column_indices)
        log_values = active / temperature
        column_mass = active.shape[0] / active.shape[1]
        log_column_mass = active.new_tensor(column_mass).clamp_min(eps).log()
        for _ in range(iterations):
            log_values = log_values - torch.logsumexp(log_values, dim=1, keepdim=True)
            log_values = (
                log_values - torch.logsumexp(log_values, dim=0, keepdim=True) + log_column_mass
            )
        log_values = log_values - torch.logsumexp(log_values, dim=1, keepdim=True)
        active_probabilities = torch.exp(log_values)
        padded = scores[batch_index].new_zeros(rows, columns)
        row_grid = row_indices.unsqueeze(1).expand(-1, len(column_indices)).reshape(-1)
        column_grid = column_indices.unsqueeze(0).expand(len(row_indices), -1).reshape(-1)
        padded = padded.index_put(
            (row_grid, column_grid), active_probabilities.reshape(-1), accumulate=False
        )
        normalized_batches.append(padded)
    return torch.stack(normalized_batches)
