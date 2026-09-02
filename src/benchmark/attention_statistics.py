from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import Tensor

from ..models.decoder import NeuralCipherDecoder

DISTANCE_BUCKETS = (
    (0, 2, "0-2"),
    (3, 5, "3-5"),
    (6, 10, "6-10"),
    (11, 50, "11-50"),
    (51, None, "50+"),
)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _bucket_mask(distances: Tensor, lower: int, upper: int | None) -> Tensor:
    result = distances >= lower
    return result if upper is None else result & (distances <= upper)


@torch.no_grad()
def attention_distance_statistics(
    model: NeuralCipherDecoder,
    loader: Iterable[dict],
    device: str | torch.device = "cpu",
    layer: int = -1,
    max_batches: int | None = None,
) -> dict:
    """Aggregate mean attention weight by cipher and sequence distance bucket."""
    if not model.is_relational:
        raise ValueError("attention distance statistics require a relational model")
    device = torch.device(device)
    model.eval()
    sums = {axis: defaultdict(float) for axis in ("cipher_distance", "sequence_distance")}
    counts = {axis: defaultdict(int) for axis in ("cipher_distance", "sequence_distance")}
    head_sums: dict[str, dict[int, dict[str, float]]] = {
        axis: defaultdict(lambda: defaultdict(float))
        for axis in ("cipher_distance", "sequence_distance")
    }
    head_counts: dict[str, dict[int, dict[str, int]]] = {
        axis: defaultdict(lambda: defaultdict(int))
        for axis in ("cipher_distance", "sequence_distance")
    }

    for batch_index, original_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _move_batch(original_batch, device)
        output = model(
            batch["digits"],
            batch["cipher_values"],
            batch["attention_mask"],
            batch["cipher_zone_ids"],
            return_attention=True,
        )
        if not output.attentions:
            raise RuntimeError("relational encoder returned no attention weights")
        weights = output.attentions[layer]
        batch_size, heads, length, _ = weights.shape
        valid_pairs = batch["attention_mask"].unsqueeze(2) & batch["attention_mask"].unsqueeze(1)
        cipher_distance = (
            batch["cipher_values"].unsqueeze(2) - batch["cipher_values"].unsqueeze(1)
        ).abs()
        positions = torch.arange(length, device=device)
        sequence_distance = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
        distances_by_axis = {
            "cipher_distance": cipher_distance,
            "sequence_distance": sequence_distance.unsqueeze(0).expand(batch_size, -1, -1),
        }
        for axis, distances in distances_by_axis.items():
            for lower, upper, name in DISTANCE_BUCKETS:
                pair_mask = valid_pairs & _bucket_mask(distances, lower, upper)
                if not bool(pair_mask.any()):
                    continue
                expanded = pair_mask.unsqueeze(1).expand(-1, heads, -1, -1)
                sums[axis][name] += float(weights[expanded].sum())
                counts[axis][name] += int(expanded.sum())
                for head in range(heads):
                    selected = weights[:, head][pair_mask]
                    head_sums[axis][head][name] += float(selected.sum())
                    head_counts[axis][head][name] += selected.numel()

    result: dict[str, dict] = {}
    for axis in ("cipher_distance", "sequence_distance"):
        result[axis] = {
            name: {
                "mean_attention": sums[axis][name] / counts[axis][name] if counts[axis][name] else 0.0,
                "pair_count": counts[axis][name],
            }
            for _lower, _upper, name in DISTANCE_BUCKETS
        }
        result[f"{axis}_by_head"] = {
            str(head): {
                name: {
                    "mean_attention": (
                        head_sums[axis][head][name] / head_counts[axis][head][name]
                        if head_counts[axis][head][name]
                        else 0.0
                    ),
                    "pair_count": head_counts[axis][head][name],
                }
                for _lower, _upper, name in DISTANCE_BUCKETS
            }
            for head in sorted(head_sums[axis])
        }
    return result
