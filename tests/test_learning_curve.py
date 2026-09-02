from __future__ import annotations

import json
from functools import partial

from torch.utils.data import DataLoader

from src.benchmark.config import load_benchmark_config
from src.benchmark.learning_curve import majority_accuracy, select_nested_training_episodes
from src.benchmark.learning_curve_reporting import (
    load_completed_runs,
    write_learning_curve_results,
)
from src.benchmark.representation_statistics import representation_zone_statistics
from src.config import ModelConfig
from src.data.controlled_synthetic import ControlledSyntheticConfig, generate_controlled_benchmark
from src.data.dataset import CipherEpisode, CipherEpisodeDataset, collate_episodes
from src.models.decoder import NeuralCipherDecoder
from src.training.evaluate import evaluate_model


def _tiny_bundle():
    config = ControlledSyntheticConfig(
        num_zones=4,
        num_digits=3,
        train_tables=4,
        validation_tables=1,
        test_tables=1,
        ood_test_tables=1,
        sequence_lengths=(8,),
        sequences_per_length=2,
        plaintext_symbols_per_zone=8,
        region_width=20,
        noise_levels=(0.1,),
        region_gap_min=5,
        region_gap_max=10,
        iid_value_min=0,
        iid_value_max=400,
        ood_value_min=600,
        ood_value_max=1000,
        preferred_transitions=2,
    )
    return config, generate_controlled_benchmark(config, seed=5)


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


def test_learning_curve_config_matches_colab_defaults() -> None:
    config = load_benchmark_config("configs/learning_curve.yaml")
    assert config.seeds == (41, 42, 43)
    assert config.models == ("standard", "relational")
    assert config.include_ablations is False
    assert config.synthetic.sequence_lengths == (128,)
    assert config.synthetic.sequences_per_length == 2
    assert (
        config.synthetic.validation_tables,
        config.synthetic.test_tables,
        config.synthetic.ood_test_tables,
    ) == (100, 200, 200)
    assert (config.training.epochs, config.training.batch_size) == (10, 8)


def test_training_table_prefixes_are_nested() -> None:
    _config, bundle = _tiny_bundle()
    first_two = select_nested_training_episodes(bundle.train, 2)
    first_four = select_nested_training_episodes(bundle.train, 4)
    ids_two = {episode.table_id for episode in first_two}
    ids_four = {episode.table_id for episode in first_four}
    assert ids_two < ids_four
    assert len(first_two) == 4
    assert len(first_four) == 8
    assert majority_accuracy(first_two) >= 0.25


def test_evaluation_and_representation_expose_collapse_diagnostics() -> None:
    config, bundle = _tiny_bundle()
    loader = DataLoader(
        CipherEpisodeDataset(bundle.iid_test),
        batch_size=1,
        collate_fn=partial(collate_episodes, num_digits=config.num_digits),
    )
    model = NeuralCipherDecoder(_model_config(), "standard")
    metrics = evaluate_model(model, loader)
    assert abs(sum(metrics["predicted_zone_distribution"].values()) - 1.0) < 1e-8
    assert metrics["supervised_loss"] > 0.0
    assert 0.0 <= metrics["max_predicted_class_fraction"] <= 1.0
    representation = representation_zone_statistics(model, loader, max_batches=1)
    assert representation["sampled_tokens"] == 8
    assert "separation" in representation


def _fake_run(model: str) -> dict:
    split = {
        "token_accuracy": 0.3,
        "zone_accuracy": 0.25,
        "token_top_3_accuracy": 0.6,
        "token_top_5_accuracy": 0.8,
        "supervised_loss": 1.2,
        "prediction_entropy": 1.0,
        "max_predicted_class_fraction": 0.4,
        "predicted_zone_distribution": {"0": 0.4, "1": 0.2, "2": 0.2, "3": 0.2},
    }
    return {
        "train_table_count": 2,
        "seed": 1,
        "model": model,
        "parameter_count": 100,
        "best_epoch": 1,
        "final_epoch": 1,
        "best_training_loss": 1.1,
        "final_training_loss": 1.1,
        "best_validation_loss": 1.2,
        "final_validation_loss": 1.2,
        "baselines": {
            "random_19_way_accuracy": 1 / 19,
            "train_majority_accuracy": 0.2,
            "validation_majority_accuracy": 0.2,
            "iid_majority_accuracy": 0.2,
            "ood_majority_accuracy": 0.2,
        },
        "train": dict(split),
        "validation": dict(split),
        "iid": dict(split),
        "ood": dict(split),
        "iid_representation_statistics": {
            "sampled_tokens": 8,
            "same_true_zone_cosine_similarity": 0.5,
            "different_true_zone_cosine_similarity": 0.2,
            "separation": 0.3,
        },
        "attention_distance_statistics": None,
    }


def test_learning_curve_reporting_is_incremental_and_resumable(tmp_path) -> None:
    config = {"train_table_counts": [2], "benchmark": {"seeds": [1]}}
    runs = [_fake_run("standard"), _fake_run("relational")]
    paths = write_learning_curve_results(runs, config, tmp_path)
    assert set(paths) == {"raw_results", "results_csv", "learning_curve_csv", "summary"}
    assert (tmp_path / "learning_curve_iid.png").exists()
    assert "Rel. − Std. IID" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert len(load_completed_runs(tmp_path / "raw_results.json", config)) == 2
    payload = json.loads((tmp_path / "raw_results.json").read_text(encoding="utf-8"))
    assert payload["runs"][0]["train"]["predicted_zone_distribution"]["0"] == 0.4


def test_majority_accuracy_uses_token_frequency() -> None:
    episodes = [CipherEpisode((1, 2, 3), (0, 0, 1), (0, 0, 1), "f")]
    assert majority_accuracy(episodes) == 2 / 3
