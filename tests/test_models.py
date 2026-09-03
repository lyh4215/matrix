import pytest
import torch

from src.analysis.attention_debug import format_attention_report, inspect_attention
from src.config import ModelConfig
from src.data.dataset import CipherEpisode, collate_episodes
from src.models.decoder import NeuralCipherDecoder
from src.models.zone_pooling import ZonePooling, pool_zone_targets
from src.training.losses import compute_loss


def small_config() -> ModelConfig:
    return ModelConfig(
        d_model=16,
        num_heads=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        num_cipher_local_heads=1,
        num_sequence_heads=1,
        use_locality_gate=True,
        hard_local_radius=100.0,
        pooling="attention",
        matching="relational",
        sinkhorn_iterations=30,
    )


def sample_batch() -> dict:
    return collate_episodes(
        [
            CipherEpisode((3124, 3128, 7010), (0, 0, 2), (8, 8, 3), "f1"),
            CipherEpisode((2001, 2050), (4, 5), (12, 7), "f2"),
        ]
    )


@pytest.mark.parametrize(
    "baseline",
    [
        "standard",
        "relational",
        "relational_gated",
        "relational_pool",
        "relational_match",
        "relational_sinkhorn",
    ],
)
def test_all_baselines_forward_and_backward(baseline: str) -> None:
    batch = sample_batch()
    model = NeuralCipherDecoder(small_config(), baseline)
    output = model(
        batch["digits"], batch["cipher_values"], batch["attention_mask"], batch["cipher_zone_ids"]
    )
    assert output.token_scores.shape == (2, 3, 19)
    assert torch.isfinite(output.token_scores[batch["attention_mask"]]).all()
    loss = compute_loss(output, batch, uses_sinkhorn=model.uses_sinkhorn)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_relational_attention_is_normalized_and_respects_padding() -> None:
    batch = sample_batch()
    model = NeuralCipherDecoder(small_config(), "relational")
    output = model(
        batch["digits"],
        batch["cipher_values"],
        batch["attention_mask"],
        batch["cipher_zone_ids"],
        return_attention=True,
    )
    weights = output.attentions[0]
    assert weights.shape == (2, 4, 3, 3)
    assert torch.allclose(weights[0].sum(-1), torch.ones(4, 3), atol=1e-6)
    assert torch.all(weights[1, :, :, 2] == 0)
    assert torch.all(weights[1, :, 2, :] == 0)


def test_pooling_supports_sparse_episode_local_zone_ids() -> None:
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [10.0, 12.0]]])
    ids = torch.tensor([[20, 20, 3]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    pooled = ZonePooling(2, "mean")(hidden, ids, mask)
    assert pooled.zone_indices.tolist() == [[3, 20]]
    assert torch.allclose(pooled.representations[0, 0], torch.tensor([10.0, 12.0]))
    assert torch.allclose(pooled.representations[0, 1], torch.tensor([2.0, 4.0]))
    labels = torch.tensor([[1, 1, 7]])
    assert pool_zone_targets(labels, ids, pooled, mask).tolist() == [[7, 1]]


def test_attention_debug_reports_each_head() -> None:
    batch = sample_batch()
    model = NeuralCipherDecoder(small_config(), "relational")
    report = inspect_attention(model, batch, query_index=0, top_k=2)
    assert set(report) == {0, 1, 2, 3}
    assert all(len(items) == 2 for items in report.values())
    assert "cipher Δ=" in format_attention_report(report)
