from __future__ import annotations

from functools import partial

import torch
from torch.utils.data import DataLoader

from src.benchmark.config import load_benchmark_config
from src.benchmark.same_region_probe import (
    PROBE_MODELS,
    SameRegionProbeModel,
    _best_distance_threshold,
    evaluate_probe,
)
from src.config import ModelConfig
from src.data.dataset import CipherEpisode, CipherEpisodeDataset, collate_episodes


def _config() -> ModelConfig:
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
        same_region_gate_hidden_dim=8,
    )


def _loader() -> DataLoader:
    episodes = [
        CipherEpisode((100, 110, 300, 315), (0, 0, 2, 2), (7, 7, 3, 3), "f")
    ]
    return DataLoader(
        CipherEpisodeDataset(episodes),
        batch_size=1,
        collate_fn=partial(collate_episodes, num_digits=3),
    )


def test_probe_config_and_all_model_outputs() -> None:
    config = load_benchmark_config("configs/same_region_probe.yaml")
    assert config.models == PROBE_MODELS
    assert (
        config.synthetic.train_tables,
        config.synthetic.validation_tables,
        config.synthetic.test_tables,
    ) == (50, 20, 50)
    assert config.synthetic.sequence_lengths == (128,)
    batch = next(iter(_loader()))
    for model_name in PROBE_MODELS:
        logits = SameRegionProbeModel(_config(), model_name)(batch)
        assert logits.shape == (1, 4, 4)


def test_probe_metrics_include_distance_and_hard_pair_diagnostics() -> None:
    model = SameRegionProbeModel(_config(), "relational_gated")
    metrics = evaluate_probe(
        model,
        _loader(),
        torch.device("cpu"),
        hard_negative_threshold=220,
        hard_positive_threshold=10,
    )
    assert "0-16" in metrics["distance_stratified"]
    assert metrics["hard_negative_count"] > 0
    assert metrics["hard_positive_count"] > 0
    assert 0.0 <= metrics["same_region_balanced_accuracy"] <= 1.0


def test_distance_threshold_is_selected_by_balanced_accuracy() -> None:
    distances = torch.tensor([2, 4, 100, 200])
    targets = torch.tensor([True, True, False, False])
    threshold, balanced_accuracy = _best_distance_threshold(distances, targets)
    assert threshold == 4
    assert balanced_accuracy == 1.0
