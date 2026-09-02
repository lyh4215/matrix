from __future__ import annotations

from src.benchmark.config import load_benchmark_config
from src.benchmark.translation_diagnostics import translation_invariance_statistics
from src.benchmark.translation_reporting import (
    translation_comparisons,
    write_translation_results,
)
from src.config import ModelConfig
from src.data.controlled_synthetic import (
    ControlledSyntheticConfig,
    generate_fixed_cipher_benchmark,
    translate_cipher_episode,
    translate_fixed_cipher_benchmark,
)
from src.data.dataset import CipherEpisode
from src.models.decoder import NeuralCipherDecoder


def _synthetic_config() -> ControlledSyntheticConfig:
    return ControlledSyntheticConfig(
        num_zones=4,
        num_digits=3,
        train_tables=1,
        validation_tables=1,
        test_tables=1,
        ood_test_tables=1,
        sequence_lengths=(8,),
        plaintext_symbols_per_zone=8,
        region_width=20,
        locality_noise=0.1,
        noise_levels=(0.1,),
        region_gap_min=5,
        region_gap_max=10,
        iid_value_min=0,
        iid_value_max=400,
        ood_value_min=600,
        ood_value_max=1000,
        preferred_transitions=2,
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
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


def test_translation_config_matches_fixed_f_defaults() -> None:
    config = load_benchmark_config("configs/translation_ablation.yaml")
    assert config.models == ("standard", "relational")
    assert config.seeds == (42,)
    assert config.synthetic.sequence_lengths == (128,)
    assert (config.training.epochs, config.training.batch_size) == (50, 8)


def test_episode_translation_preserves_every_pairwise_cipher_relation() -> None:
    original = CipherEpisode((105, 212, 399, 501), (0, 1, 2, 3), (3, 1, 0, 2), "f")
    translated = translate_cipher_episode(original, -100, num_digits=3)
    assert translated.zone_labels == original.zone_labels
    assert translated.cipher_zone_ids == original.cipher_zone_ids
    assert all(
        translated.cipher_values[right] - translated.cipher_values[left]
        == original.cipher_values[right] - original.cipher_values[left]
        for left in range(len(original.cipher_values))
        for right in range(len(original.cipher_values))
    )


def test_translated_fixed_f_is_deterministic_shared_and_four_digit_safe() -> None:
    config = _synthetic_config()
    original = generate_fixed_cipher_benchmark(config, 6, 3, 3, seed=42)
    first = translate_fixed_cipher_benchmark(original, config.num_digits, seed=42)
    second = translate_fixed_cipher_benchmark(original, config.num_digits, seed=42)
    assert first.offsets == second.offsets
    assert first.bundle.cipher_table is original.cipher_table
    for split in ("train", "validation", "test"):
        original_episodes = getattr(original, split)
        translated_episodes = getattr(first.bundle, split)
        assert len(original_episodes) == len(translated_episodes)
        for source, shifted, offset in zip(
            original_episodes, translated_episodes, first.offsets[split]
        ):
            assert shifted.cipher_values == tuple(value + offset for value in source.cipher_values)
            assert shifted.zone_labels == source.zone_labels
            assert min(shifted.cipher_values) >= 0
            assert max(shifted.cipher_values) < 1000
    metadata = first.translation_metadata
    assert metadata["overall"]["count"] == 12
    assert set(metadata["by_split"]) == {"train", "validation", "test"}


def test_translation_invariance_diagnostic_compares_same_token_positions() -> None:
    config = _synthetic_config()
    original = generate_fixed_cipher_benchmark(config, 1, 1, 1, seed=3)
    model = NeuralCipherDecoder(_model_config(), "relational")
    metrics = translation_invariance_statistics(
        model, original.test[0], num_digits=3, max_variants=4
    )
    assert len(metrics["offsets"]) >= 2
    assert metrics["num_variant_pairs"] >= 1
    assert -1.0 <= metrics["translation_hidden_cosine_similarity_mean"] <= 1.0
    assert 0.0 <= metrics["translation_prediction_consistency"] <= 1.0


def _metric(accuracy: float) -> dict:
    return {
        "token_accuracy": accuracy,
        "zone_accuracy": accuracy,
        "token_top_3_accuracy": min(1.0, accuracy + 0.1),
        "token_top_5_accuracy": min(1.0, accuracy + 0.2),
        "prediction_entropy": 1.0,
        "max_predicted_class_fraction": 0.3,
        "predicted_zone_distribution": {"0": 0.3, "1": 0.7},
    }


def _run(condition: str, model: str, accuracy: float) -> dict:
    return {
        "condition": condition,
        "model": model,
        "train": _metric(accuracy),
        "validation": _metric(accuracy),
        "test": _metric(accuracy),
        "history": [
            {
                "epoch": 1,
                "train_accuracy": accuracy,
                "validation_accuracy": accuracy,
            }
        ],
        "translation_invariance_statistics": {
            "offsets": [-10, 0, 10],
            "num_variant_pairs": 3,
            "translation_hidden_cosine_similarity_mean": 0.8,
            "translation_hidden_cosine_similarity_std": 0.1,
            "translation_prediction_consistency": 0.9,
        },
    }


def test_reporting_calculates_paired_translation_drops(tmp_path) -> None:
    runs = [
        _run("original", "standard", 0.95),
        _run("original", "relational", 0.96),
        _run("translated", "standard", 0.50),
        _run("translated", "relational", 0.90),
    ]
    comparisons = translation_comparisons(runs)
    assert abs(comparisons["standard_translation_drop"] - 0.45) < 1e-12
    assert abs(comparisons["relational_translation_drop"] - 0.06) < 1e-12
    assert abs(comparisons["translated_relational_minus_standard"] - 0.40) < 1e-12
    paths = write_translation_results(
        runs,
        {"seed": 42},
        {"table_id": "fixed-f"},
        {"overall": {"min": -100, "max": 500}},
        majority_baseline=0.15,
        output_dir=tmp_path,
    )
    assert all((tmp_path / name).exists() for name in (
        "raw_results.json",
        "summary.json",
        "summary.md",
        "plots/original_accuracy.png",
        "plots/translated_accuracy.png",
        "histories/original_standard.json",
        "histories/translated_relational.json",
    ))
    assert "Standard translation drop" in (tmp_path / "summary.md").read_text(
        encoding="utf-8"
    )
    assert paths["raw_results"].endswith("raw_results.json")
