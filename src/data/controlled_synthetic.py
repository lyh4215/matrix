from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from .dataset import CipherEpisode


@dataclass
class ControlledSyntheticConfig:
    """Configuration for the controlled locality/generalization benchmark."""

    num_zones: int = 19
    num_digits: int = 4
    num_tables: int | None = None
    train_tables: int = 1600
    validation_tables: int = 200
    test_tables: int = 200
    ood_test_tables: int = 200
    sequence_lengths: tuple[int, ...] = (8, 16, 32, 64, 128)
    sequences_per_length: int = 1
    plaintext_symbols_per_zone: int = 64
    region_width: int = 128
    locality_noise: float = 0.1
    noise_levels: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
    region_gap_min: int = 32
    region_gap_max: int = 96
    iid_value_min: int = 0
    iid_value_max: int = 4500
    ood_value_min: int = 5500
    ood_value_max: int = 10_000
    preferred_transitions: int = 3
    transition_strength: float = 8.0
    language_seed: int = 1729

    def validate(self) -> None:
        if self.num_zones < 2 or self.num_digits < 1:
            raise ValueError("the benchmark needs at least two zones and one digit")
        if min(
            self.train_tables,
            self.validation_tables,
            self.test_tables,
            self.ood_test_tables,
            self.sequences_per_length,
        ) < 1:
            raise ValueError("all table and sequence counts must be positive")
        split_total = self.train_tables + self.validation_tables + self.test_tables
        if self.num_tables is not None and self.num_tables != split_total:
            raise ValueError("num_tables must equal train + validation + IID test tables")
        if not self.sequence_lengths or min(self.sequence_lengths) < 1:
            raise ValueError("sequence_lengths must contain positive lengths")
        if not 0.0 <= self.locality_noise <= 1.0:
            raise ValueError("locality_noise must be between zero and one")
        if any(not 0.0 <= level <= 1.0 for level in self.noise_levels):
            raise ValueError("all noise levels must be between zero and one")
        if self.region_width < 2 or self.plaintext_symbols_per_zone < 2:
            raise ValueError("region and symbol counts must be at least two")
        if not 0 <= self.region_gap_min <= self.region_gap_max:
            raise ValueError("invalid region gap bounds")
        numeric_limit = 10**self.num_digits
        if not 0 <= self.iid_value_min < self.iid_value_max <= numeric_limit:
            raise ValueError("invalid IID numeric range")
        if not 0 <= self.ood_value_min < self.ood_value_max <= numeric_limit:
            raise ValueError("invalid OOD numeric range")
        maximum_span = self.num_zones * self.region_width + (self.num_zones - 1) * self.region_gap_max
        if self.iid_value_max - self.iid_value_min < maximum_span:
            raise ValueError("IID numeric range cannot fit all local regions")
        if self.ood_value_max - self.ood_value_min < maximum_span:
            raise ValueError("OOD numeric range cannot fit all local regions")
        if not 1 <= self.preferred_transitions <= self.num_zones:
            raise ValueError("preferred_transitions must be in [1, num_zones]")
        if self.transition_strength <= 1.0:
            raise ValueError("transition_strength must be greater than one")


@dataclass(frozen=True)
class PlainZoneSequence:
    zones: tuple[int, ...]
    symbol_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.zones) != len(self.symbol_ids) or not self.zones:
            raise ValueError("plain sequence fields must have one non-zero length")


@dataclass(frozen=True)
class MarkovZoneLanguage:
    """One shared, non-uniform plaintext transition law for every cipher table."""

    transition_matrix: tuple[tuple[float, ...], ...]
    symbols_per_zone: int

    @property
    def num_zones(self) -> int:
        return len(self.transition_matrix)

    @classmethod
    def create(
        cls,
        num_zones: int,
        symbols_per_zone: int,
        preferred_transitions: int,
        transition_strength: float,
        seed: int,
    ) -> "MarkovZoneLanguage":
        rng = random.Random(seed)
        rows: list[tuple[float, ...]] = []
        for source in range(num_zones):
            weights = [rng.uniform(0.25, 1.0) for _ in range(num_zones)]
            # A random row-specific signature makes the common language
            # identifiable without tying it to numeric region order.
            preferred = rng.sample(range(num_zones), preferred_transitions)
            for target in preferred:
                weights[target] *= transition_strength
            # Keep self-transition informative but not universally dominant.
            weights[source] *= 1.0 + 0.25 * transition_strength
            total = sum(weights)
            rows.append(tuple(weight / total for weight in weights))
        return cls(tuple(rows), symbols_per_zone)

    def sample(self, length: int, rng: random.Random) -> PlainZoneSequence:
        if length < 1:
            raise ValueError("sequence length must be positive")
        zones = [rng.randrange(self.num_zones)]
        symbols = [rng.randrange(self.symbols_per_zone)]
        population = range(self.num_zones)
        for _ in range(1, length):
            zones.append(rng.choices(population, weights=self.transition_matrix[zones[-1]], k=1)[0])
            symbols.append(rng.randrange(self.symbols_per_zone))
        return PlainZoneSequence(tuple(zones), tuple(symbols))


@dataclass(frozen=True)
class ControlledCipherTable:
    """A random permutation of plaintext zones over ordered numeric regions."""

    table_id: str
    region_bases: tuple[int, ...]
    zone_to_region: tuple[int, ...]
    region_width: int
    symbols_per_zone: int

    def encrypt(
        self,
        sequence: PlainZoneSequence,
        locality_noise: float,
        rng: random.Random,
    ) -> CipherEpisode:
        if not 0.0 <= locality_noise <= 1.0:
            raise ValueError("locality_noise must be between zero and one")
        values: list[int] = []
        region_ids: list[int] = []
        numeric_min = self.region_bases[0]
        numeric_max = self.region_bases[-1] + self.region_width - 1
        for zone, symbol in zip(sequence.zones, sequence.symbol_ids):
            region = self.zone_to_region[zone]
            ideal = round(symbol * (self.region_width - 1) / (self.symbols_per_zone - 1))
            ideal_value = self.region_bases[region] + ideal
            # Noise is measured in region widths. At stronger settings some
            # observations cross a region boundary, making locality genuinely
            # less reliable instead of merely scrambling order inside a cluster.
            noisy_value = round(ideal_value + rng.gauss(0.0, locality_noise * self.region_width))
            values.append(min(max(noisy_value, numeric_min), numeric_max))
            region_ids.append(region)
        return CipherEpisode(
            tuple(values), sequence.zones, tuple(region_ids), self.table_id
        )


@dataclass(frozen=True)
class PlannedEpisode:
    table: ControlledCipherTable
    plaintext: PlainZoneSequence


@dataclass
class ControlledBenchmarkBundle:
    transition_matrix: tuple[tuple[float, ...], ...]
    train: list[CipherEpisode]
    validation: list[CipherEpisode]
    iid_test: list[CipherEpisode]
    ood_test: list[CipherEpisode]
    iid_by_length: dict[int, list[CipherEpisode]]
    iid_by_noise: dict[float, list[CipherEpisode]]

    @property
    def train_table_ids(self) -> set[str]:
        return {str(item.table_id) for item in self.train}


def _make_table(
    table_id: str,
    config: ControlledSyntheticConfig,
    rng: random.Random,
    value_min: int,
    value_max: int,
) -> ControlledCipherTable:
    gaps = [rng.randint(config.region_gap_min, config.region_gap_max) for _ in range(config.num_zones - 1)]
    occupied = config.num_zones * config.region_width + sum(gaps)
    if occupied > value_max - value_min:
        raise ValueError("sampled region gaps do not fit the configured numeric range")
    start = rng.randint(value_min, value_max - occupied)
    bases = [start]
    for gap in gaps:
        bases.append(bases[-1] + config.region_width + gap)
    permutation = list(range(config.num_zones))
    rng.shuffle(permutation)
    return ControlledCipherTable(
        table_id,
        tuple(bases),
        tuple(permutation),
        config.region_width,
        config.plaintext_symbols_per_zone,
    )


def _make_tables(
    prefix: str,
    count: int,
    config: ControlledSyntheticConfig,
    seed: int,
    ood: bool = False,
) -> list[ControlledCipherTable]:
    rng = random.Random(seed)
    lower = config.ood_value_min if ood else config.iid_value_min
    upper = config.ood_value_max if ood else config.iid_value_max
    return [_make_table(f"{prefix}-{index:05d}", config, rng, lower, upper) for index in range(count)]


def _plan_episodes(
    tables: Sequence[ControlledCipherTable],
    language: MarkovZoneLanguage,
    config: ControlledSyntheticConfig,
    seed: int,
) -> list[PlannedEpisode]:
    rng = random.Random(seed)
    plans: list[PlannedEpisode] = []
    for table in tables:
        for length in config.sequence_lengths:
            for _ in range(config.sequences_per_length):
                plans.append(PlannedEpisode(table, language.sample(length, rng)))
    return plans


def _encrypt_plans(
    plans: Iterable[PlannedEpisode], locality_noise: float, seed: int
) -> list[CipherEpisode]:
    rng = random.Random(seed)
    return [plan.table.encrypt(plan.plaintext, locality_noise, rng) for plan in plans]


def generate_controlled_benchmark(
    config: ControlledSyntheticConfig, seed: int
) -> ControlledBenchmarkBundle:
    """Generate deterministic, table-disjoint IID and numeric-support-shifted OOD data."""
    config.validate()
    language = MarkovZoneLanguage.create(
        config.num_zones,
        config.plaintext_symbols_per_zone,
        config.preferred_transitions,
        config.transition_strength,
        config.language_seed,
    )
    train_tables = _make_tables("train-f", config.train_tables, config, seed + 101)
    validation_tables = _make_tables("validation-f", config.validation_tables, config, seed + 202)
    iid_tables = _make_tables("iid-test-f", config.test_tables, config, seed + 303)
    ood_tables = _make_tables("ood-test-f", config.ood_test_tables, config, seed + 404, ood=True)

    train_plans = _plan_episodes(train_tables, language, config, seed + 1001)
    validation_plans = _plan_episodes(validation_tables, language, config, seed + 1002)
    iid_plans = _plan_episodes(iid_tables, language, config, seed + 1003)
    ood_plans = _plan_episodes(ood_tables, language, config, seed + 1004)
    train = _encrypt_plans(train_plans, config.locality_noise, seed + 2001)
    validation = _encrypt_plans(validation_plans, config.locality_noise, seed + 2002)
    iid_test = _encrypt_plans(iid_plans, config.locality_noise, seed + 2003)
    ood_test = _encrypt_plans(ood_plans, config.locality_noise, seed + 2004)
    iid_by_noise = {
        level: _encrypt_plans(iid_plans, level, seed + 3000 + index)
        for index, level in enumerate(config.noise_levels)
    }
    iid_by_length = {
        length: [episode for episode in iid_test if len(episode.cipher_values) == length]
        for length in config.sequence_lengths
    }

    groups = [
        {item.table_id for item in split}
        for split in (train, validation, iid_test, ood_test)
    ]
    if any(groups[left] & groups[right] for left in range(4) for right in range(left + 1, 4)):
        raise AssertionError("cipher-table leakage in controlled benchmark")
    return ControlledBenchmarkBundle(
        language.transition_matrix,
        train,
        validation,
        iid_test,
        ood_test,
        iid_by_length,
        iid_by_noise,
    )
