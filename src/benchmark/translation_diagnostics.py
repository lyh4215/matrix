from __future__ import annotations

from functools import partial

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..data.controlled_synthetic import translate_cipher_episode
from ..data.dataset import CipherEpisode, CipherEpisodeDataset, collate_episodes
from ..models.decoder import NeuralCipherDecoder


def diagnostic_translation_offsets(
    episode: CipherEpisode, num_digits: int, max_variants: int = 4
) -> list[int]:
    if max_variants < 2:
        raise ValueError("translation diagnostic needs at least two variants")
    lower = -min(episode.cipher_values)
    upper = 10**num_digits - 1 - max(episode.cipher_values)
    candidates = {lower, upper, 0, round((lower + upper) / 2)}
    if len(candidates) < max_variants and upper > lower:
        for index in range(max_variants):
            candidates.add(round(lower + (upper - lower) * index / (max_variants - 1)))
    offsets = sorted(candidates)
    if len(offsets) < 2:
        raise ValueError("episode has no room for distinct valid translations")
    if len(offsets) <= max_variants:
        return offsets
    indices = torch.linspace(0, len(offsets) - 1, max_variants).round().to(torch.long)
    return [offsets[index] for index in indices.tolist()]


@torch.no_grad()
def translation_invariance_statistics(
    model: NeuralCipherDecoder,
    episode: CipherEpisode,
    num_digits: int,
    device: str | torch.device = "cpu",
    max_variants: int = 4,
) -> dict:
    """Measure hidden-state cosine and prediction agreement across translations."""
    device = torch.device(device)
    offsets = diagnostic_translation_offsets(episode, num_digits, max_variants)
    variants = [translate_cipher_episode(episode, offset, num_digits) for offset in offsets]
    loader = DataLoader(
        CipherEpisodeDataset(variants),
        batch_size=len(variants),
        collate_fn=partial(collate_episodes, num_digits=num_digits),
    )
    batch = next(iter(loader))
    batch = {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }
    model.eval()
    output = model(
        digits=batch["digits"],
        cipher_values=batch["cipher_values"],
        attention_mask=batch["attention_mask"],
        cipher_zone_ids=batch["cipher_zone_ids"],
    )
    normalized = torch.nn.functional.normalize(output.hidden_states, dim=-1)
    predictions = output.token_scores.argmax(dim=-1)
    cosine_values: list[Tensor] = []
    agreement_values: list[Tensor] = []
    for left in range(len(offsets)):
        for right in range(left + 1, len(offsets)):
            cosine_values.append((normalized[left] * normalized[right]).sum(dim=-1))
            agreement_values.append((predictions[left] == predictions[right]).to(torch.float32))
    cosine = torch.cat(cosine_values)
    agreement = torch.cat(agreement_values)
    return {
        "offsets": offsets,
        "num_variant_pairs": len(cosine_values),
        "translation_hidden_cosine_similarity_mean": float(cosine.mean()),
        "translation_hidden_cosine_similarity_std": float(cosine.std(unbiased=False)),
        "translation_prediction_consistency": float(agreement.mean()),
    }
