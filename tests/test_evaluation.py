from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import ModelConfig
from src.data.dataset import CipherEpisode, CipherEpisodeDataset, collate_episodes
from src.models.decoder import DecoderOutput, NeuralCipherDecoder
from src.training.assignment import maximum_weight_assignment
from src.training.evaluate import evaluate_model


class FixedScoreModel(nn.Module):
    uses_sinkhorn = False
    is_relational = False

    def forward(
        self,
        digits,
        cipher_values,
        attention_mask,
        cipher_zone_ids=None,
        return_attention=False,
    ) -> DecoderOutput:
        lookup = {
            10: (5.0, 0.0),  # episode 1, region 0: correct
            20: (2.0, 0.0),  # episode 1, region 1: wrong
            11: (0.0, 2.0),  # episode 2, region 0: wrong
            21: (0.0, 5.0),  # episode 2, region 1: correct
        }
        scores = torch.zeros(*cipher_values.shape, 2, device=cipher_values.device)
        for row in range(cipher_values.shape[0]):
            for column in range(cipher_values.shape[1]):
                value = int(cipher_values[row, column])
                if value in lookup:
                    scores[row, column] = torch.tensor(lookup[value], device=scores.device)
        return DecoderOutput(scores, torch.zeros_like(scores))


def test_hungarian_assignment_enforces_distinct_columns() -> None:
    scores = torch.tensor([[5.0, 4.0], [5.0, 1.0]])
    assert scores.argmax(dim=-1).tolist() == [0, 0]
    assert maximum_weight_assignment(scores) == [1, 0]


def test_table_aggregation_recovers_mapping_when_each_episode_is_inexact() -> None:
    episodes = [
        CipherEpisode((10, 20), (0, 1), (0, 1), "f"),
        CipherEpisode((11, 21), (0, 1), (0, 1), "f"),
    ]
    loader = DataLoader(
        CipherEpisodeDataset(episodes),
        batch_size=2,
        collate_fn=partial(collate_episodes, num_digits=2),
    )
    metrics = evaluate_model(FixedScoreModel(), loader)
    assert metrics["exact_mapping_accuracy_per_episode"] == 0.0
    assert metrics["all_episodes_exact_per_f"] == 0.0
    assert metrics["table_zone_accuracy_argmax"] == 1.0
    assert metrics["table_exact_mapping_accuracy_argmax"] == 1.0
    assert metrics["table_zone_accuracy_assignment"] == 1.0
    assert metrics["table_exact_mapping_accuracy_assignment"] == 1.0


def test_sinkhorn_evaluation_exposes_partial_constraint_diagnostics() -> None:
    episodes = [CipherEpisode((10, 12, 80), (0, 0, 2), (4, 4, 7), "f")]
    loader = DataLoader(
        CipherEpisodeDataset(episodes),
        batch_size=1,
        collate_fn=partial(collate_episodes, num_digits=2),
    )
    config = ModelConfig(
        d_model=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        num_zones=4,
        num_digits=2,
        num_cipher_local_heads=1,
        num_sequence_heads=1,
    )
    metrics = evaluate_model(NeuralCipherDecoder(config, "relational_sinkhorn"), loader)
    diagnostics = metrics["sinkhorn_diagnostics"]
    assert abs(diagnostics["real_row_sum_mean"] - 1.0) < 1e-5
    assert diagnostics["real_column_max_mass"] <= 1.0001
    assert diagnostics["dummy_row_mass_distribution"]["mean"] > 0.0
