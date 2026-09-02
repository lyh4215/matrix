from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CipherEpisode:
    cipher_values: tuple[int, ...]
    zone_labels: tuple[int, ...]
    cipher_zone_ids: tuple[int, ...]
    table_id: Hashable

    def __post_init__(self) -> None:
        lengths = {len(self.cipher_values), len(self.zone_labels), len(self.cipher_zone_ids)}
        if len(lengths) != 1 or not self.cipher_values:
            raise ValueError("episode fields must have one common, non-zero length")
        if min(self.cipher_values) < 0:
            raise ValueError("cipher values must be non-negative")
        if min(self.zone_labels) < 0 or min(self.cipher_zone_ids) < 0:
            raise ValueError("zone labels and cipher zone ids must be non-negative")

    @classmethod
    def from_mapping(cls, item: dict) -> "CipherEpisode":
        return cls(
            cipher_values=tuple(map(int, item["cipher_values"])),
            zone_labels=tuple(map(int, item["zone_labels"])),
            cipher_zone_ids=tuple(map(int, item["cipher_zone_ids"])),
            table_id=item["table_id"],
        )


class CipherEpisodeDataset(Dataset[CipherEpisode]):
    def __init__(self, episodes: Sequence[CipherEpisode]):
        self.episodes = list(episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> CipherEpisode:
        return self.episodes[index]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "CipherEpisodeDataset":
        episodes: list[CipherEpisode] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        episodes.append(CipherEpisode.from_mapping(json.loads(line)))
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(f"invalid JSONL record at line {line_number}") from exc
        return cls(episodes)


def values_to_digits(values: Tensor, num_digits: int = 4) -> Tensor:
    """Convert integer values to most-significant-first digits without a lookup table."""
    if values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("cipher values must be an integer tensor")
    upper = 10**num_digits
    if bool(((values < 0) | (values >= upper)).any()):
        raise ValueError(f"cipher values must be in [0, {upper}) for {num_digits} digits")
    divisors = torch.tensor(
        [10**power for power in range(num_digits - 1, -1, -1)],
        device=values.device,
        dtype=values.dtype,
    )
    return ((values.unsqueeze(-1) // divisors) % 10).to(torch.float32)


def collate_episodes(episodes: Sequence[CipherEpisode], num_digits: int = 4) -> dict:
    if not episodes:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(episodes)
    max_length = max(len(item.cipher_values) for item in episodes)
    values = torch.zeros(batch_size, max_length, dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    zone_ids = torch.full((batch_size, max_length), -1, dtype=torch.long)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    lengths = torch.empty(batch_size, dtype=torch.long)
    for row, episode in enumerate(episodes):
        length = len(episode.cipher_values)
        values[row, :length] = torch.tensor(episode.cipher_values)
        labels[row, :length] = torch.tensor(episode.zone_labels)
        zone_ids[row, :length] = torch.tensor(episode.cipher_zone_ids)
        mask[row, :length] = True
        lengths[row] = length
    return {
        "cipher_values": values,
        "digits": values_to_digits(values, num_digits),
        "zone_labels": labels,
        "cipher_zone_ids": zone_ids,
        "attention_mask": mask,
        "lengths": lengths,
        "table_ids": [item.table_id for item in episodes],
    }


def split_by_cipher_table(
    episodes: Sequence[CipherEpisode],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[CipherEpisode], list[CipherEpisode], list[CipherEpisode]]:
    """Split exclusively by f/table id, never by ciphertext sentence."""
    if train_fraction <= 0 or validation_fraction < 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("invalid split fractions")
    table_ids = sorted({item.table_id for item in episodes}, key=str)
    if len(table_ids) < 3:
        raise ValueError("at least three cipher tables are required")
    random.Random(seed).shuffle(table_ids)
    train_end = max(1, int(len(table_ids) * train_fraction))
    validation_count = max(1, int(len(table_ids) * validation_fraction))
    validation_end = min(len(table_ids) - 1, train_end + validation_count)
    train_ids = set(table_ids[:train_end])
    validation_ids = set(table_ids[train_end:validation_end])
    test_ids = set(table_ids[validation_end:])
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("cipher-table leakage in split")
    groups = tuple(
        [item for item in episodes if item.table_id in ids]
        for ids in (train_ids, validation_ids, test_ids)
    )
    if any(not group for group in groups):
        raise ValueError("split produced an empty partition; use more cipher tables")
    return groups


def table_ids(episodes: Iterable[CipherEpisode]) -> set[Hashable]:
    return {item.table_id for item in episodes}

