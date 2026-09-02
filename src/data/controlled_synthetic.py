from __future__ import annotations

import random
import statistics
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
    allow_cipher_collisions: bool = False
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
        mapping_size = self.num_zones * self.plaintext_symbols_per_zone
        if not self.allow_cipher_collisions and (
            self.iid_value_max - self.iid_value_min < mapping_size
            or self.ood_value_max - self.ood_value_min < mapping_size
        ):
            raise ValueError("numeric ranges cannot fit a collision-free symbol mapping")
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
    symbol_to_cipher: tuple[tuple[int, ...], ...]
    locality_noise: float
    allow_cipher_collisions: bool = False

    @classmethod
    def create(
        cls,
        table_id: str,
        region_bases: Sequence[int],
        zone_to_region: Sequence[int],
        region_width: int,
        symbols_per_zone: int,
        locality_noise: float,
        rng: random.Random,
        allow_cipher_collisions: bool = False,
    ) -> "ControlledCipherTable":
        if not 0.0 <= locality_noise <= 1.0:
            raise ValueError("locality_noise must be between zero and one")
        bases = tuple(region_bases)
        permutation = tuple(zone_to_region)
        if not bases or tuple(sorted(bases)) != bases:
            raise ValueError("region bases must be non-empty and sorted")
        if sorted(permutation) != list(range(len(bases))):
            raise ValueError("zone_to_region must be a permutation of region indices")
        if symbols_per_zone < 2 or region_width < 2:
            raise ValueError("region width and symbols per zone must be at least two")
        numeric_min = bases[0]
        numeric_max = bases[-1] + region_width - 1
        used: set[int] = set()
        mappings: list[tuple[int, ...]] = []
        for zone, region in enumerate(permutation):
            zone_mapping: list[int] = []
            for symbol in range(symbols_per_zone):
                coordinate = round(symbol * (region_width - 1) / (symbols_per_zone - 1))
                ideal_value = bases[region] + coordinate
                target = round(ideal_value + rng.gauss(0.0, locality_noise * region_width))
                target = min(max(target, numeric_min), numeric_max)
                value = (
                    target
                    if allow_cipher_collisions
                    else _nearest_unused(target, used, numeric_min, numeric_max)
                )
                used.add(value)
                zone_mapping.append(value)
            mappings.append(tuple(zone_mapping))
        return cls(
            table_id,
            bases,
            permutation,
            region_width,
            symbols_per_zone,
            tuple(mappings),
            locality_noise,
            allow_cipher_collisions,
        )

    def encrypt(
        self,
        sequence: PlainZoneSequence,
    ) -> CipherEpisode:
        values: list[int] = []
        region_ids: list[int] = []
        for zone, symbol in zip(sequence.zones, sequence.symbol_ids):
            region = self.zone_to_region[zone]
            values.append(self.symbol_to_cipher[zone][symbol])
            region_ids.append(region)
        return CipherEpisode(
            tuple(values), sequence.zones, tuple(region_ids), self.table_id
        )

    def with_locality_noise(
        self,
        locality_noise: float,
        rng: random.Random,
        allow_cipher_collisions: bool | None = None,
    ) -> "ControlledCipherTable":
        return self.create(
            self.table_id,
            self.region_bases,
            self.zone_to_region,
            self.region_width,
            self.symbols_per_zone,
            locality_noise,
            rng,
            self.allow_cipher_collisions if allow_cipher_collisions is None else allow_cipher_collisions,
        )


def _nearest_unused(target: int, used: set[int], lower: int, upper: int) -> int:
    if target not in used:
        return target
    maximum_radius = max(target - lower, upper - target)
    for radius in range(1, maximum_radius + 1):
        left = target - radius
        right = target + radius
        if left >= lower and left not in used:
            return left
        if right <= upper and right not in used:
            return right
    raise ValueError("numeric range is too small for a collision-free cipher mapping")


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


@dataclass
class FixedCipherBenchmarkBundle:
    """Plaintext-disjoint episodes encrypted by one shared deterministic table."""

    transition_matrix: tuple[tuple[float, ...], ...]
    cipher_table: ControlledCipherTable
    train: list[CipherEpisode]
    validation: list[CipherEpisode]
    test: list[CipherEpisode]

    @property
    def table_metadata(self) -> dict:
        return {
            "table_id": self.cipher_table.table_id,
            "region_bases": list(self.cipher_table.region_bases),
            "zone_to_region": list(self.cipher_table.zone_to_region),
            "region_width": self.cipher_table.region_width,
            "symbols_per_zone": self.cipher_table.symbols_per_zone,
            "symbol_to_cipher": [list(row) for row in self.cipher_table.symbol_to_cipher],
            "locality_noise": self.cipher_table.locality_noise,
        }


@dataclass
class TranslatedFixedCipherBenchmark:
    """A fixed-f bundle observed in independently translated episode coordinates."""

    bundle: FixedCipherBenchmarkBundle
    offsets: dict[str, list[int]]

    @property
    def translation_metadata(self) -> dict:
        def summarize(values: Sequence[int]) -> dict[str, float | int | None]:
            return {
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": statistics.mean(values) if values else None,
                "std": statistics.pstdev(values) if values else None,
            }

        all_offsets = [offset for values in self.offsets.values() for offset in values]
        return {
            "overall": summarize(all_offsets),
            "by_split": {
                split: summarize(values) for split, values in self.offsets.items()
            },
        }


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
    return ControlledCipherTable.create(
        table_id,
        bases,
        permutation,
        config.region_width,
        config.plaintext_symbols_per_zone,
        config.locality_noise,
        rng,
        config.allow_cipher_collisions,
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
    plans: Iterable[PlannedEpisode], table_overrides: dict[str, ControlledCipherTable] | None = None
) -> list[CipherEpisode]:
    table_overrides = table_overrides or {}
    return [
        table_overrides.get(plan.table.table_id, plan.table).encrypt(plan.plaintext)
        for plan in plans
    ]


def _remap_tables(
    tables: Sequence[ControlledCipherTable],
    locality_noise: float,
    seed: int,
) -> dict[str, ControlledCipherTable]:
    rng = random.Random(seed)
    return {
        table.table_id: table.with_locality_noise(locality_noise, rng)
        for table in tables
    }


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
    train = _encrypt_plans(train_plans)
    validation = _encrypt_plans(validation_plans)
    ood_test = _encrypt_plans(ood_plans)
    iid_by_noise = {
        level: _encrypt_plans(
            iid_plans,
            # Reusing the RNG seed pairs the per-symbol Gaussian draw across
            # levels; only its scale changes, so noise comparisons are controlled.
            _remap_tables(iid_tables, level, seed + 3000),
        )
        for level in config.noise_levels
    }
    iid_test = iid_by_noise.get(config.locality_noise, _encrypt_plans(iid_plans))
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


def generate_fixed_cipher_benchmark(
    config: ControlledSyntheticConfig,
    train_episodes: int,
    validation_episodes: int,
    test_episodes: int,
    seed: int,
) -> FixedCipherBenchmarkBundle:
    """Generate unique plaintext splits encrypted with exactly one fixed cipher function."""
    config.validate()
    if len(config.sequence_lengths) != 1:
        raise ValueError("fixed-cipher benchmark requires exactly one sequence length")
    if train_episodes < 1 or validation_episodes < 0 or test_episodes < 0:
        raise ValueError("fixed-cipher episode counts must be non-negative with a non-empty train split")
    language = MarkovZoneLanguage.create(
        config.num_zones,
        config.plaintext_symbols_per_zone,
        config.preferred_transitions,
        config.transition_strength,
        config.language_seed,
    )
    cipher_table = _make_table(
        "fixed-f",
        config,
        random.Random(seed + 101),
        config.iid_value_min,
        config.iid_value_max,
    )
    length = config.sequence_lengths[0]
    seen_plaintexts: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    def make_split(count: int, split_seed: int) -> list[CipherEpisode]:
        rng = random.Random(split_seed)
        episodes: list[CipherEpisode] = []
        while len(episodes) < count:
            plaintext = language.sample(length, rng)
            identity = (plaintext.zones, plaintext.symbol_ids)
            if identity in seen_plaintexts:
                continue
            seen_plaintexts.add(identity)
            episodes.append(cipher_table.encrypt(plaintext))
        return episodes

    train = make_split(train_episodes, seed + 1001)
    validation = make_split(validation_episodes, seed + 1002)
    test = make_split(test_episodes, seed + 1003)
    all_episodes = train + validation + test
    if {str(episode.table_id) for episode in all_episodes} != {cipher_table.table_id}:
        raise AssertionError("fixed-cipher splits do not share exactly one table")
    if len({episode.cipher_values for episode in all_episodes}) != len(all_episodes):
        raise AssertionError("fixed-cipher plaintext splits contain duplicate episodes")
    return FixedCipherBenchmarkBundle(
        language.transition_matrix,
        cipher_table,
        train,
        validation,
        test,
    )


def translate_cipher_episode(
    episode: CipherEpisode,
    offset: int,
    num_digits: int,
) -> CipherEpisode:
    """Move one complete episode without changing any pairwise cipher differences."""
    translated = tuple(value + offset for value in episode.cipher_values)
    upper = 10**num_digits
    if min(translated) < 0 or max(translated) >= upper:
        raise ValueError(f"translation moves ciphertext outside [0, {upper})")
    return CipherEpisode(
        translated,
        episode.zone_labels,
        episode.cipher_zone_ids,
        episode.table_id,
    )


def translate_fixed_cipher_benchmark(
    original: FixedCipherBenchmarkBundle,
    num_digits: int,
    seed: int,
) -> TranslatedFixedCipherBenchmark:
    """Apply one broad, independently sampled integer offset to every episode."""
    numeric_max = 10**num_digits - 1

    def translate_split(
        episodes: Sequence[CipherEpisode], split_seed: int
    ) -> tuple[list[CipherEpisode], list[int]]:
        rng = random.Random(split_seed)
        translated: list[CipherEpisode] = []
        offsets: list[int] = []
        for episode in episodes:
            lower = -min(episode.cipher_values)
            upper = numeric_max - max(episode.cipher_values)
            signed_ranges: list[tuple[int, int]] = []
            if lower <= -1:
                signed_ranges.append((lower, -1))
            if upper >= 1:
                signed_ranges.append((1, upper))
            offset = rng.randint(*rng.choice(signed_ranges)) if signed_ranges else 0
            translated.append(translate_cipher_episode(episode, offset, num_digits))
            offsets.append(offset)
        return translated, offsets

    translated_splits: dict[str, list[CipherEpisode]] = {}
    offsets: dict[str, list[int]] = {}
    for split, episodes, split_seed in (
        ("train", original.train, seed + 2001),
        ("validation", original.validation, seed + 2002),
        ("test", original.test, seed + 2003),
    ):
        translated_splits[split], offsets[split] = translate_split(episodes, split_seed)
    bundle = FixedCipherBenchmarkBundle(
        original.transition_matrix,
        original.cipher_table,
        translated_splits["train"],
        translated_splits["validation"],
        translated_splits["test"],
    )
    return TranslatedFixedCipherBenchmark(bundle, offsets)
