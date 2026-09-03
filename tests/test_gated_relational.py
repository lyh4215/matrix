from __future__ import annotations

import torch

from src.config import ModelConfig
from src.data.dataset import CipherEpisode, collate_episodes
from src.models.decoder import NeuralCipherDecoder
from src.models.gated_relational_attention import (
    GatedRelationalAttention,
    blend_branch_scores,
    combine_gated_branch_features,
)
from src.training.losses import compute_loss
from src.training.same_region import (
    balanced_same_region_loss,
    same_region_pair_targets,
)


def _config() -> ModelConfig:
    return ModelConfig(
        d_model=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        num_zones=4,
        num_digits=3,
        distance_clip=100.0,
        num_cipher_local_heads=1,
        num_sequence_heads=1,
        same_region_gate_hidden_dim=8,
    )


def _batch() -> dict:
    return collate_episodes(
        [
            CipherEpisode((100, 120, 400), (0, 0, 2), (7, 7, 3), "f1"),
            CipherEpisode((210, 500), (1, 3), (9, 2), "f2"),
        ],
        num_digits=3,
    )


def test_same_region_targets_are_correct_and_permutation_invariant() -> None:
    batch = _batch()
    targets, valid = same_region_pair_targets(
        batch["cipher_zone_ids"], batch["attention_mask"]
    )
    assert targets[0, 0, 1]
    assert not targets[0, 0, 2]
    # Semantic labels are not an input to the target builder.
    permuted_labels = batch["zone_labels"].roll(1, dims=0)
    repeated_targets, _ = same_region_pair_targets(
        batch["cipher_zone_ids"], batch["attention_mask"]
    )
    assert not torch.equal(permuted_labels, batch["zone_labels"])
    assert torch.equal(targets, repeated_targets)
    assert valid[0, 0, 1]


def test_padding_pairs_are_excluded_from_auxiliary_loss() -> None:
    batch = _batch()
    targets, valid = same_region_pair_targets(
        batch["cipher_zone_ids"], batch["attention_mask"]
    )
    assert not valid[1, :, 2].any()
    assert not valid[1, 2, :].any()
    logits = torch.zeros_like(targets, dtype=torch.float32, requires_grad=True)
    baseline = balanced_same_region_loss(
        logits, batch["cipher_zone_ids"], batch["attention_mask"]
    )
    changed = logits.detach().clone()
    changed[1, :, 2] = -1000.0
    changed[1, 2, :] = 1000.0
    masked = balanced_same_region_loss(
        changed, batch["cipher_zone_ids"], batch["attention_mask"]
    )
    assert torch.allclose(baseline, masked)


def test_self_pairs_are_excluded_from_auxiliary_loss() -> None:
    batch = _batch()
    targets, valid = same_region_pair_targets(
        batch["cipher_zone_ids"], batch["attention_mask"]
    )
    assert not torch.diagonal(valid, dim1=1, dim2=2).any()
    logits = torch.zeros_like(targets, dtype=torch.float32)
    baseline = balanced_same_region_loss(
        logits, batch["cipher_zone_ids"], batch["attention_mask"]
    )
    logits[:, torch.arange(3), torch.arange(3)] = 1000.0
    changed = balanced_same_region_loss(
        logits, batch["cipher_zone_ids"], batch["attention_mask"]
    )
    assert torch.allclose(baseline, changed)


def test_auxiliary_loss_balances_positive_and_negative_pair_classes() -> None:
    region_ids = torch.tensor([[0, 0, 1, 2]])
    mask = torch.ones_like(region_ids, dtype=torch.bool)
    targets, valid = same_region_pair_targets(region_ids, mask)
    logits = torch.zeros_like(targets, dtype=torch.float32)
    logits[valid & targets] = -2.0
    logits[valid & ~targets] = 1.0
    loss = balanced_same_region_loss(logits, region_ids, mask)
    expected = 0.5 * (
        torch.nn.functional.softplus(torch.tensor(2.0))
        + torch.nn.functional.softplus(torch.tensor(1.0))
    )
    assert torch.allclose(loss, expected)


def test_local_branch_has_signed_numeric_delta() -> None:
    attention = GatedRelationalAttention(_config())
    hidden = torch.zeros(1, 2, _config().d_model)
    digits = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]])
    cipher_values = torch.tensor([[100, 120]])
    features = attention.build_pair_features(
        hidden, digits, cipher_values, 2, torch.float32
    )
    cipher_signed_index = 2 * _config().num_digits
    assert features.local_numeric[0, 0, 1, cipher_signed_index] < 0
    assert features.local_numeric[0, 1, 0, cipher_signed_index] > 0


def test_cross_branch_does_not_receive_raw_numeric_features() -> None:
    attention = GatedRelationalAttention(_config())
    hidden = torch.zeros(1, 2, _config().d_model)
    digits = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]])
    features = attention.build_pair_features(
        hidden, digits, torch.tensor([[100, 120]]), 2, torch.float32
    )
    hidden_relation = torch.zeros(1, 2, 2, 4)
    local_a, cross_a = combine_gated_branch_features(
        hidden_relation, features.local_numeric, features.sequence
    )
    changed_numeric = features.local_numeric + 7.0
    local_b, cross_b = combine_gated_branch_features(
        hidden_relation, changed_numeric, features.sequence
    )
    assert not torch.equal(local_a, local_b)
    assert torch.equal(cross_a, cross_b)


def test_gate_features_do_not_encode_signed_global_ordering() -> None:
    attention = GatedRelationalAttention(_config())
    hidden = torch.zeros(1, 2, _config().d_model)
    digits = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]])
    forward = attention.build_pair_features(
        hidden, digits, torch.tensor([[100, 120]]), 2, torch.float32
    )
    reverse = attention.build_pair_features(
        hidden.flip(1), digits.flip(1), torch.tensor([[120, 100]]), 2, torch.float32
    )
    assert torch.allclose(forward.gate[0, 0, 1], reverse.gate[0, 0, 1])


def test_same_probability_one_selects_local_branch() -> None:
    local = torch.tensor([[2.0, 3.0]])
    cross = torch.tensor([[-4.0, 9.0]])
    assert torch.equal(blend_branch_scores(local, cross, torch.ones_like(local)), local)


def test_same_probability_zero_selects_cross_branch() -> None:
    local = torch.tensor([[2.0, 3.0]])
    cross = torch.tensor([[-4.0, 9.0]])
    assert torch.equal(blend_branch_scores(local, cross, torch.zeros_like(local)), cross)


def test_legacy_and_gated_decoder_outputs_remain_compatible() -> None:
    batch = _batch()
    legacy = NeuralCipherDecoder(_config(), "relational")
    gated = NeuralCipherDecoder(_config(), "relational_gated")
    legacy_output = legacy(
        batch["digits"],
        batch["cipher_values"],
        batch["attention_mask"],
        batch["cipher_zone_ids"],
    )
    gated_output = gated(
        batch["digits"],
        batch["cipher_values"],
        batch["attention_mask"],
        batch["cipher_zone_ids"],
    )
    assert legacy_output.token_scores.shape == gated_output.token_scores.shape == (2, 3, 4)
    assert legacy_output.same_region_logits is None
    assert gated_output.same_region_logits is not None
    assert gated_output.same_region_logits[-1].shape == (2, 3, 3)


def test_auxiliary_loss_reaches_the_shared_gate() -> None:
    batch = _batch()
    model = NeuralCipherDecoder(_config(), "relational_gated")
    output = model(
        batch["digits"],
        batch["cipher_values"],
        batch["attention_mask"],
        batch["cipher_zone_ids"],
    )
    losses = compute_loss(output, batch, lambda_same_region=1.0)
    assert losses.same_region > 0
    losses.total.backward()
    gate_gradients = [
        parameter.grad
        for parameter in model.encoder.layers[-1].attention.same_region_gate.parameters()
    ]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gate_gradients)
