from __future__ import annotations

import torch
from torch import Tensor


def sinkhorn(
    scores: Tensor,
    row_mask: Tensor | None = None,
    column_mask: Tensor | None = None,
    iterations: int = 20,
    temperature: float = 1.0,
    dummy_mode: str = "neutral",
    eps: float = 1e-8,
) -> Tensor:
    """Partial-observation Sinkhorn using neutral dummy rows.

    An R x C matrix with R <= C is padded with C-R neutral rows, normalized as
    a square C x C matrix, then sliced back to its real rows. Unobserved labels
    are absorbed by dummy rows instead of receiving forced mass from real rows.
    """
    if scores.ndim != 3:
        raise ValueError("scores must have shape [batch, rows, columns]")
    if iterations < 1 or temperature <= 0:
        raise ValueError("iterations and temperature must be positive")
    if dummy_mode != "neutral":
        raise ValueError("only neutral dummy rows are currently supported")
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
        real_rows, active_columns = active.shape
        if real_rows > active_columns:
            raise ValueError("partial Sinkhorn requires real rows <= active columns")
        dummy_rows = active.new_zeros(active_columns - real_rows, active_columns)
        log_values = torch.cat((active, dummy_rows), dim=0) / temperature
        for _ in range(iterations):
            log_values = log_values - torch.logsumexp(log_values, dim=1, keepdim=True)
            log_values = log_values - torch.logsumexp(log_values, dim=0, keepdim=True)
        log_values = log_values - torch.logsumexp(log_values, dim=1, keepdim=True)
        active_probabilities = torch.exp(log_values[:real_rows])
        padded = scores[batch_index].new_zeros(rows, columns)
        row_grid = row_indices.unsqueeze(1).expand(-1, len(column_indices)).reshape(-1)
        column_grid = column_indices.unsqueeze(0).expand(len(row_indices), -1).reshape(-1)
        padded = padded.index_put(
            (row_grid, column_grid), active_probabilities.reshape(-1), accumulate=False
        )
        normalized_batches.append(padded)
    return torch.stack(normalized_batches)
