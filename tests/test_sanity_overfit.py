from __future__ import annotations

from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader

from src.benchmark.config import BenchmarkConfig, load_benchmark_config
from src.benchmark.sanity_reporting import write_sanity_results
from src.config import ModelConfig, TrainingConfig
from src.data.controlled_synthetic import (
    ControlledSyntheticConfig,
    generate_fixed_cipher_benchmark,
)
from src.data.dataset import CipherEpisodeDataset, collate_episodes
from src.models.decoder import NeuralCipherDecoder
from src.training.train import train_epoch


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


def test_sanity_config_has_fixed_f_defaults() -> None:
    config = load_benchmark_config("configs/sanity_overfit.yaml")
    assert config.seeds == (42,)
    assert config.models == ("standard", "relational")
    assert config.synthetic.sequence_lengths == (128,)
    assert config.training.epochs == 50
    assert config.training.lambda_local == config.training.lambda_entropy == 0.0


def test_fixed_cipher_splits_share_mapping_but_not_episodes() -> None:
    config = _synthetic_config()
    first = generate_fixed_cipher_benchmark(config, 4, 2, 2, seed=7)
    second = generate_fixed_cipher_benchmark(config, 4, 2, 2, seed=7)
    episodes = first.train + first.validation + first.test
    assert {episode.table_id for episode in episodes} == {"fixed-f"}
    assert len({episode.cipher_values for episode in episodes}) == 8
    assert {len(episode.cipher_values) for episode in episodes} == {8}
    assert first.cipher_table.zone_to_region == second.cipher_table.zone_to_region
    assert first.cipher_table.symbol_to_cipher == second.cipher_table.symbol_to_cipher
    assert first.table_metadata["zone_to_region"] == list(first.cipher_table.zone_to_region)


@pytest.mark.parametrize("model_name", ("standard", "relational"))
def test_first_step_has_finite_gradients_and_parameter_updates(model_name: str) -> None:
    synthetic = _synthetic_config()
    bundle = generate_fixed_cipher_benchmark(synthetic, 2, 0, 0, seed=4)
    config = BenchmarkConfig(
        seeds=(4,),
        models=("standard", "relational"),
        include_ablations=False,
        synthetic=synthetic,
        model=_model_config(),
        training=TrainingConfig(epochs=1, batch_size=2, weight_decay=0.0),
    )
    config.validate()
    loader = DataLoader(
        CipherEpisodeDataset(bundle.train),
        batch_size=2,
        collate_fn=partial(collate_episodes, num_digits=3),
    )
    model = NeuralCipherDecoder(config.model, model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    result = train_epoch(
        model,
        loader,
        optimizer,
        config,
        torch.device("cpu"),
        capture_first_step_diagnostics=True,
    )
    diagnostics = result["first_step_diagnostics"]
    assert diagnostics["all_gradients_finite"] is True
    assert diagnostics["encoder_gradient_norm"] > 0.0
    assert diagnostics["token_classifier_gradient_norm"] > 0.0
    assert diagnostics["encoder_parameter_delta_norm"] > 0.0
    assert diagnostics["token_classifier_parameter_delta_norm"] > 0.0


def _fake_result(model: str, train_accuracy: float, test_accuracy: float) -> dict:
    history = [
        {
            "epoch": 1,
            "train_loss": 2.0,
            "train_evaluation_loss": 1.9,
            "train_accuracy": train_accuracy,
            "validation_loss": 1.8,
            "validation_accuracy": test_accuracy,
            "train_prediction_entropy": 1.0,
            "validation_prediction_entropy": 1.1,
            "train_max_predicted_class_fraction": 0.3,
            "validation_max_predicted_class_fraction": 0.25,
        }
    ]
    metric = lambda accuracy: {
        "token_accuracy": accuracy,
        "predicted_zone_distribution": {"0": 0.5, "1": 0.5},
        "prediction_entropy": 0.69,
        "max_predicted_class_fraction": 0.5,
    }
    return {
        "model": model,
        "parameter_count": 100,
        "best_epoch": 1,
        "final_epoch": 1,
        "selection_accuracy": test_accuracy,
        "first_step_diagnostics": {"encoder_gradient_norm": 1.0},
        "train": metric(train_accuracy),
        "validation": metric(test_accuracy),
        "test": metric(test_accuracy),
        "history": history,
        "checkpoint": f"{model}.pt",
    }


def test_sanity_reporting_writes_histories_curves_and_heuristic(tmp_path) -> None:
    baselines = {
        "random_19_way_accuracy": 1 / 19,
        "train_majority_accuracy": 0.15,
        "validation_majority_accuracy": 0.15,
        "test_majority_accuracy": 0.15,
    }
    results = [
        _fake_result("standard", 0.95, 0.92),
        _fake_result("relational", 0.96, 0.93),
    ]
    paths = write_sanity_results(
        results,
        {"epochs": 1},
        baselines,
        {"table_id": "fixed-f"},
        tmp_path,
        "fixed_f",
        0.9,
    )
    assert all((tmp_path / name).exists() for name in (
        "standard_history.json",
        "relational_history.json",
        "summary.json",
        "summary.md",
        "training_curves.png",
        "loss_curves.png",
        "fixed_cipher_table.json",
    ))
    assert "appears functional" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert paths["summary_json"].endswith("summary.json")
