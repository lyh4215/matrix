from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .dataset import CipherEpisode


@dataclass(frozen=True)
class SyntheticCipherTable:
    table_id: str
    zone_bases: tuple[int, ...]
    zone_width: int = 64

    def encrypt_zones(self, zones: Sequence[int], rng: random.Random) -> CipherEpisode:
        values = tuple(self.zone_bases[zone] + rng.randrange(self.zone_width) for zone in zones)
        return CipherEpisode(values, tuple(zones), tuple(zones), self.table_id)


def make_cipher_table(
    table_id: str,
    rng: random.Random,
    num_zones: int = 19,
    num_digits: int = 4,
    zone_width: int = 64,
) -> SyntheticCipherTable:
    """Place zones into independently permuted, non-overlapping local regions."""
    max_value = 10**num_digits
    stride = max(zone_width + 1, max_value // (num_zones + 2))
    candidates = list(range(stride, max_value - zone_width, stride))
    if len(candidates) < num_zones:
        raise ValueError("digit space is too small for the requested zone layout")
    bases = rng.sample(candidates, num_zones)
    return SyntheticCipherTable(table_id, tuple(bases), zone_width)


def generate_synthetic_episodes(
    num_tables: int = 30,
    sentences_per_table: int = 3,
    min_length: int = 8,
    max_length: int = 24,
    num_zones: int = 19,
    num_digits: int = 4,
    seed: int = 42,
) -> list[CipherEpisode]:
    """Generate structural smoke-test data; real experiments should supply Korean episodes."""
    rng = random.Random(seed)
    episodes: list[CipherEpisode] = []
    for table_index in range(num_tables):
        table = make_cipher_table(f"f-{table_index:04d}", rng, num_zones, num_digits)
        for _ in range(sentences_per_table):
            length = rng.randint(min_length, max_length)
            first = rng.randrange(num_zones)
            zones = [first]
            for _position in range(1, length):
                if rng.random() < 0.6:
                    zones.append((zones[-1] + rng.choice((-2, -1, 0, 1, 2))) % num_zones)
                else:
                    zones.append(rng.randrange(num_zones))
            episodes.append(table.encrypt_zones(zones, rng))
    return episodes


class CipherAugmenter:
    """Episode-local anti-shortcut transforms that preserve labels and local offsets."""

    def __init__(
        self,
        num_digits: int = 4,
        random_offset: bool = False,
        max_random_offset: int = 500,
        offset_mode: str = "clamp",
        zone_relocation: bool = False,
        seed: int = 42,
    ) -> None:
        if offset_mode not in {"clamp", "wrap", "error"}:
            raise ValueError("offset_mode must be clamp, wrap, or error")
        self.num_digits = num_digits
        self.random_offset = random_offset
        self.max_random_offset = max_random_offset
        self.offset_mode = offset_mode
        self.zone_relocation = zone_relocation
        self.rng = random.Random(seed)

    def __call__(self, episode: CipherEpisode) -> CipherEpisode:
        values = list(episode.cipher_values)
        upper = 10**self.num_digits
        if self.zone_relocation:
            values = self._relocate(values, episode.cipher_zone_ids, upper)
        if self.random_offset:
            offset = self.rng.randint(-self.max_random_offset, self.max_random_offset)
            values = [self._bound(value + offset, upper) for value in values]
        return CipherEpisode(tuple(values), episode.zone_labels, episode.cipher_zone_ids, episode.table_id)

    def _relocate(self, values: list[int], zone_ids: tuple[int, ...], upper: int) -> list[int]:
        unique = sorted(set(zone_ids))
        widths: dict[int, int] = {}
        minima: dict[int, int] = {}
        for zone in unique:
            members = [value for value, item_zone in zip(values, zone_ids) if item_zone == zone]
            minima[zone] = min(members)
            widths[zone] = max(members) - min(members) + 1
        slot_width = max(max(widths.values()), 2)
        available = list(range(0, upper - slot_width + 1, slot_width))
        if len(available) < len(unique):
            raise ValueError("not enough numeric space for zone relocation")
        new_bases = dict(zip(unique, self.rng.sample(available, len(unique))))
        return [new_bases[zone] + (value - minima[zone]) for value, zone in zip(values, zone_ids)]

    def _bound(self, value: int, upper: int) -> int:
        if self.offset_mode == "wrap":
            return value % upper
        if self.offset_mode == "clamp":
            return min(max(value, 0), upper - 1)
        if not 0 <= value < upper:
            raise ValueError("random offset moved a cipher value outside digit range")
        return value


class AugmentedDataset:
    def __init__(self, episodes: Sequence[CipherEpisode], augmenter: CipherAugmenter):
        self.episodes = list(episodes)
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> CipherEpisode:
        return self.augmenter(self.episodes[index])
