from functools import partial
import random

from torch.utils.data import DataLoader

from src.benchmark.attention_statistics import DISTANCE_BUCKETS, attention_distance_statistics
from src.benchmark.config import load_benchmark_config
from src.benchmark.reporting import aggregate_runs, write_results
from src.benchmark.runner import experiment_variants
from src.config import ModelConfig
from src.data.controlled_synthetic import (
    ControlledCipherTable,
    ControlledSyntheticConfig,
    MarkovZoneLanguage,
    PlainZoneSequence,
    generate_controlled_benchmark,
)
from src.data.dataset import CipherEpisodeDataset, collate_episodes
from src.models.decoder import NeuralCipherDecoder


def tiny_synthetic_config() -> ControlledSyntheticConfig:
    return ControlledSyntheticConfig(
        num_zones=4,
        num_digits=3,
        train_tables=2,
        validation_tables=1,
        test_tables=1,
        ood_test_tables=1,
        sequence_lengths=(8, 16),
        plaintext_symbols_per_zone=8,
        region_width=20,
        locality_noise=0.1,
        noise_levels=(0.0, 0.5),
        region_gap_min=5,
        region_gap_max=10,
        iid_value_min=0,
        iid_value_max=400,
        ood_value_min=600,
        ood_value_max=1000,
        preferred_transitions=2,
    )


def test_controlled_bundle_has_shared_language_and_disjoint_numeric_ood() -> None:
    config = tiny_synthetic_config()
    bundle = generate_controlled_benchmark(config, seed=11)
    assert len(bundle.train) == config.train_tables * len(config.sequence_lengths)
    assert {len(item.cipher_values) for item in bundle.iid_test} == {8, 16}
    assert max(value for item in bundle.train for value in item.cipher_values) < 400
    assert min(value for item in bundle.ood_test for value in item.cipher_values) >= 600
    groups = [
        {item.table_id for item in split}
        for split in (bundle.train, bundle.validation, bundle.iid_test, bundle.ood_test)
    ]
    assert all(not groups[left] & groups[right] for left in range(4) for right in range(left + 1, 4))
    assert set(bundle.iid_by_noise) == {0.0, 0.5}
    assert all(abs(sum(row) - 1.0) < 1e-10 for row in bundle.transition_matrix)
    assert all(max(row) > 2 * min(row) for row in bundle.transition_matrix)


def test_zero_noise_preserves_intra_zone_symbol_geometry() -> None:
    table = ControlledCipherTable("f", (100, 300), (1, 0), 50, 10)
    plain = PlainZoneSequence((0, 0, 0), (0, 4, 9))
    episode = table.encrypt(plain, locality_noise=0.0, rng=random.Random(3))
    assert episode.cipher_zone_ids == (1, 1, 1)
    assert episode.cipher_values[0] < episode.cipher_values[1] < episode.cipher_values[2]


def test_markov_language_is_seeded_and_shared() -> None:
    first = MarkovZoneLanguage.create(5, 8, 2, 8.0, seed=99)
    second = MarkovZoneLanguage.create(5, 8, 2, 8.0, seed=99)
    assert first.transition_matrix == second.transition_matrix


def test_benchmark_config_and_ablation_conditions() -> None:
    smoke = load_benchmark_config("configs/synthetic_benchmark_smoke.yaml")
    assert smoke.synthetic.sequence_lengths == (8, 16)
    full = load_benchmark_config("configs/synthetic_benchmark.yaml")
    assert full.synthetic.train_tables == 1600
    full.models = ("relational",)
    variants = experiment_variants(full)
    assert [item.ablation_condition[:1] for item in variants] == ["A", "B", "C", "D"]
    assert dict(variants[1].overrides)["use_absolute_digits"] is False
    assert dict(variants[2].overrides)["use_cipher_delta"] is False


def test_attention_statistics_aggregate_distance_buckets() -> None:
    config = ModelConfig(
        d_model=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        num_zones=4,
        num_digits=3,
        num_cipher_local_heads=1,
        num_sequence_heads=1,
    )
    bundle = generate_controlled_benchmark(tiny_synthetic_config(), seed=2)
    loader = DataLoader(
        CipherEpisodeDataset(bundle.iid_test),
        batch_size=2,
        collate_fn=partial(collate_episodes, num_digits=3),
    )
    stats = attention_distance_statistics(
        NeuralCipherDecoder(config, "relational"), loader, max_batches=1
    )
    names = {name for _lower, _upper, name in DISTANCE_BUCKETS}
    assert set(stats["cipher_distance"]) == names
    assert stats["sequence_distance"]["0-2"]["pair_count"] > 0
    assert set(stats["cipher_distance_by_head"]) == {"0", "1"}


def test_reporting_writes_machine_and_human_readable_results(tmp_path) -> None:
    run = {
        "seed": 1,
        "model": "relational",
        "baseline": "relational",
        "ablation_condition": "A",
        "best_epoch": 1,
        "best_validation_token_accuracy": 0.2,
        "iid": {
            "token_accuracy": 0.3,
            "zone_accuracy": 0.4,
            "exact_mapping_accuracy_per_f": 0.1,
            "token_top_3_accuracy": 0.6,
            "unseen_f_accuracy": 0.3,
        },
        "ood": {"token_accuracy": 0.2},
        "accuracy_by_sequence_length": {"8": 0.2},
        "accuracy_by_locality_noise": {"0.0": 0.4},
    }
    paths = write_results([run], {"seed": 1}, tmp_path)
    assert all((tmp_path / name).exists() for name in ("results.json", "results.csv", "summary.md"))
    assert "Token Accuracy" in paths["table"]
    assert aggregate_runs([run])[0]["token_accuracy_std"] == 0.0
