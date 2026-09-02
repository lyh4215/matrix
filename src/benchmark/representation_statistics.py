from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from ..models.decoder import NeuralCipherDecoder


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def representation_zone_statistics(
    model: NeuralCipherDecoder,
    loader: Iterable[dict],
    device: str | torch.device = "cpu",
    max_batches: int | None = 4,
    max_tokens: int = 2048,
) -> dict[str, float | int]:
    """Compare cosine similarity for same-zone and different-zone encoder states."""
    if max_tokens < 2:
        raise ValueError("max_tokens must be at least two")
    device = torch.device(device)
    model.eval()
    hidden_parts: list[Tensor] = []
    label_parts: list[Tensor] = []
    collected = 0
    for batch_index, original_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _move_batch(original_batch, device)
        output = model(
            digits=batch["digits"],
            cipher_values=batch["cipher_values"],
            attention_mask=batch["attention_mask"],
            cipher_zone_ids=batch["cipher_zone_ids"],
        )
        hidden = output.hidden_states[batch["attention_mask"]]
        labels = batch["zone_labels"][batch["attention_mask"]]
        remaining = max_tokens - collected
        hidden_parts.append(hidden[:remaining])
        label_parts.append(labels[:remaining])
        collected += min(hidden.shape[0], remaining)
        if collected >= max_tokens:
            break
    if collected < 2:
        raise ValueError("representation diagnostic needs at least two valid tokens")

    hidden = torch.cat(hidden_parts)
    labels = torch.cat(label_parts)
    normalized = torch.nn.functional.normalize(hidden, dim=-1)
    similarities = normalized @ normalized.transpose(0, 1)
    diagonal = torch.eye(collected, dtype=torch.bool, device=device)
    same_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~diagonal
    different_mask = labels.unsqueeze(0).ne(labels.unsqueeze(1))
    same = float(similarities[same_mask].mean()) if bool(same_mask.any()) else 0.0
    different = (
        float(similarities[different_mask].mean()) if bool(different_mask.any()) else 0.0
    )
    return {
        "sampled_tokens": collected,
        "same_true_zone_cosine_similarity": same,
        "different_true_zone_cosine_similarity": different,
        "separation": same - different,
    }
