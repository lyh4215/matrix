from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from ..models.decoder import NeuralCipherDecoder


@dataclass(frozen=True)
class AttentionNeighbor:
    head: int
    query_index: int
    key_index: int
    query_cipher: int
    key_cipher: int
    cipher_delta: int
    sequence_delta: int
    attention: float


@torch.no_grad()
def inspect_attention(
    model: NeuralCipherDecoder,
    batch: dict,
    query_index: int,
    sample_index: int = 0,
    layer: int = -1,
    top_k: int = 5,
) -> dict[int, list[dict]]:
    """Return the strongest keys per head for one relational-attention query.

    Reported deltas are key minus query, matching the human-readable example in
    the specification. Internal score features use query minus key.
    """
    device = next(model.parameters()).device
    moved = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    output = model(
        moved["digits"],
        moved["cipher_values"],
        moved["attention_mask"],
        moved.get("cipher_zone_ids"),
        return_attention=True,
    )
    if not output.attentions:
        raise ValueError("attention inspection requires a relational baseline")
    weights = output.attentions[layer][sample_index]
    valid_length = int(moved["attention_mask"][sample_index].sum())
    if not 0 <= query_index < valid_length:
        raise IndexError("query_index points outside the unpadded sequence")
    cipher_values = moved["cipher_values"][sample_index]
    report: dict[int, list[dict]] = {}
    for head in range(weights.shape[0]):
        count = min(top_k, valid_length)
        key_indices = weights[head, query_index, :valid_length].topk(count).indices.tolist()
        neighbors = []
        for key_index in key_indices:
            item = AttentionNeighbor(
                head=head,
                query_index=query_index,
                key_index=key_index,
                query_cipher=int(cipher_values[query_index]),
                key_cipher=int(cipher_values[key_index]),
                cipher_delta=int(cipher_values[key_index] - cipher_values[query_index]),
                sequence_delta=key_index - query_index,
                attention=float(weights[head, query_index, key_index]),
            )
            neighbors.append(asdict(item))
        report[head] = neighbors
    return report


def format_attention_report(report: dict[int, list[dict]]) -> str:
    lines: list[str] = []
    for head, neighbors in report.items():
        lines.append(f"head {head}")
        for item in neighbors:
            lines.append(
                f"{item['key_cipher']:>6}  cipher Δ={item['cipher_delta']:+d}  "
                f"seq Δ={item['sequence_delta']:+d}  attn={item['attention']:.4f}"
            )
    return "\n".join(lines)

