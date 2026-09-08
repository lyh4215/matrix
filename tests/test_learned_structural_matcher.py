from __future__ import annotations

import itertools
import json
import math
import sys

import pytest
import torch

from src.benchmark import region_zone_match_probe as probe
from src.data.controlled_synthetic import MarkovZoneLanguage
from src.models import learned_structural_matcher as structural
from src.models.region_zone_matcher import LearnedRegionZoneMatcher, observed_permutation_nll
from src.models.sinkhorn import sinkhorn


def _canonical(zones: int = 19) -> torch.Tensor:
    language = MarkovZoneLanguage.create(
        zones, symbols_per_zone=8, preferred_transitions=2,
        transition_strength=8.0, seed=9,
    )
    return torch.tensor(language.transition_matrix, dtype=torch.float32)


def _model(zones: int = 19, **kwargs) -> structural.LearnedStructuralRegionZoneMatcher:
    return structural.LearnedStructuralRegionZoneMatcher(
        _canonical(zones), num_zones=zones, d_model=8, num_layers=1,
        dropout=0.0, sinkhorn_iterations=60, **kwargs,
    ).eval()


def _inputs(partial: bool = False):
    torch.manual_seed(17)
    features = torch.randn(2, 19, 5)
    observed = torch.ones(2, 19, dtype=torch.bool)
    if partial:
        observed[0, 8:] = False
        observed[1, 13:] = False
    counts = torch.randint(0, 3, (2, 19, 19)).float()
    counts *= observed.unsqueeze(-1) & observed.unsqueeze(-2)
    transition = (counts + 1e-3) / (counts + 1e-3).sum(-1, keepdim=True)
    return features, transition, observed, counts


def test_batched_structural_sinkhorn_matches_existing_probabilities_and_gradients() -> None:
    torch.manual_seed(22)
    scores = torch.randn(3, 19, 19, dtype=torch.float64, requires_grad=True)
    observed = torch.ones(3, 19, dtype=torch.bool)
    observed[1, ::2] = False
    observed[2, :18] = False
    existing = sinkhorn(scores, row_mask=observed, iterations=60, temperature=0.7)
    batched = structural.structural_sinkhorn(scores, observed, iterations=60, temperature=0.7)
    assert torch.allclose(existing, batched, atol=1e-12, rtol=1e-12)
    weights = torch.randn_like(scores)
    existing_grad = torch.autograd.grad((existing * weights).sum(), scores)[0]
    batched_grad = torch.autograd.grad((batched * weights).sum(), scores)[0]
    assert torch.allclose(existing_grad, batched_grad, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("partial", [False, True])
def test_shape_sinkhorn_and_anonymous_reindexing(partial: bool) -> None:
    features, transition, observed, counts = _inputs(partial)
    model = _model()
    output = model(features, transition, observed, counts)
    assignment = output.assignment_probabilities
    assert assignment.shape == (2, 19, 19)
    assert torch.isfinite(assignment).all()
    assert torch.allclose(assignment.sum(-1), observed.float(), atol=1e-5)
    assert bool((assignment.sum(-2) <= 1.0001).all())
    if not partial:
        assert torch.allclose(assignment.sum(-2), torch.ones(2, 19), atol=1e-4)
    assert bool((assignment[~observed] == 0).all())
    permutation = torch.randperm(19)
    reindexed = model(
        features[:, permutation], transition[:, permutation][:, :, permutation],
        observed[:, permutation], counts[:, permutation][:, :, permutation],
    )
    assert torch.allclose(
        reindexed.assignment_probabilities, assignment[:, permutation], atol=1e-6,
    )
    assert torch.allclose(
        structural.graph_consistency_loss(assignment, counts, model.canonical_transition),
        structural.graph_consistency_loss(
            reindexed.assignment_probabilities,
            counts[:, permutation][:, :, permutation], model.canonical_transition,
        ), atol=1e-6,
    )


def test_structural_score_matches_directed_edge_sum() -> None:
    torch.manual_seed(4)
    assignment = torch.softmax(torch.randn(1, 4, 4), -1)
    counts = torch.randint(0, 5, (1, 4, 4)).float()
    log_p = _canonical(4).log()
    expected = torch.zeros_like(assignment)
    for i, a, j, b in itertools.product(range(4), repeat=4):
        expected[0, i, a] += assignment[0, j, b] * (
            counts[0, i, j] * log_p[a, b] + counts[0, j, i] * log_p[b, a]
        )
    assert torch.allclose(
        structural.transition_structural_score(assignment, counts, log_p), expected,
        atol=1e-5,
    )


def test_gradient_flows_through_every_refinement_step(monkeypatch) -> None:
    features, transition, observed, counts = _inputs()
    model = _model()
    intermediate = []
    original_score = structural.transition_structural_score

    def tracked_score(assignment, counts, log_p):
        assignment.retain_grad()
        intermediate.append(assignment)
        return original_score(assignment, counts, log_p)

    monkeypatch.setattr(structural, "transition_structural_score", tracked_score)
    output = model(features, transition, observed, counts)
    targets = torch.arange(19).expand(2, -1)
    loss = observed_permutation_nll(output.assignment_probabilities, targets, observed)
    loss += 0.5 * structural.graph_consistency_loss(
        output.assignment_probabilities, counts, model.canonical_transition,
    )
    loss.backward()
    assert len(intermediate) == 4
    for assignment in intermediate:
        assert assignment.grad is not None
        assert torch.isfinite(assignment.grad).all()
        assert assignment.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert model.semantic_prototypes.grad.abs().sum() > 0
    assert model.input_projection[0].weight.grad.abs().sum() > 0
    assert not model.canonical_transition.requires_grad


@pytest.mark.parametrize("kwargs", [
    {"structural_refinement_steps": 0}, {"structural_beta": 0.0},
])
def test_disabling_refinement_reproduces_unchanged_learned(kwargs) -> None:
    inputs = _inputs()
    model = _model(**kwargs)
    baseline = LearnedRegionZoneMatcher(
        d_model=8, num_layers=1, dropout=0.0, sinkhorn_iterations=60,
    ).eval()
    baseline.load_state_dict({
        k: v for k, v in model.state_dict().items() if k != "canonical_transition"
    })
    assert torch.equal(
        model(*inputs).assignment_probabilities,
        baseline(*inputs[:3]).assignment_probabilities,
    )


def test_exact_permutation_minimizes_graph_loss_and_matches_count_likelihood() -> None:
    canonical = _canonical(5).double()
    truth = torch.tensor([3, 0, 4, 1, 2])
    anonymous = canonical[truth][:, truth]
    counts = (anonymous * torch.tensor([10, 20, 30, 40, 50]).unsqueeze(1)).unsqueeze(0)
    correct = torch.eye(5, dtype=torch.float64)[truth].unsqueeze(0)
    correct_loss = structural.graph_consistency_loss(correct, counts, canonical)
    expected = -(counts[0] * anonymous.log()).sum() / counts.sum()
    assert torch.allclose(correct_loss, expected, atol=1e-12)
    for permutation in itertools.permutations(range(5)):
        if permutation == tuple(truth.tolist()):
            continue
        wrong = torch.eye(5, dtype=torch.float64)[list(permutation)].unsqueeze(0)
        assert structural.graph_consistency_loss(wrong, counts, canonical) > correct_loss


def test_graph_loss_handles_zero_probabilities_partial_observation_and_empty_counts() -> None:
    assignment = torch.tensor([[[0.6, 0.4], [0.0, 0.0]]], requires_grad=True)
    counts = torch.tensor([[[2.0, 0.0], [0.0, 0.0]]])
    canonical = torch.eye(2)
    loss = structural.graph_consistency_loss(assignment, counts, canonical)
    assert torch.allclose(loss, -torch.tensor(0.52).log())
    loss.backward()
    assert torch.isfinite(assignment.grad).all()
    empty = structural.graph_consistency_loss(assignment, torch.zeros_like(counts), canonical)
    assert empty.item() == 0.0
    empty.backward()
    impossible_counts = torch.ones(1, 2, 2)
    assert torch.isfinite(structural.graph_consistency_loss(
        torch.eye(2).unsqueeze(0), impossible_counts, canonical,
    ))


@pytest.mark.parametrize("name,value", [
    ("structural_refinement_steps", -1), ("structural_refinement_steps", 1.5),
    ("structural_beta", -1.0), ("structural_beta", float("nan")),
    ("lambda_graph", -0.1), ("lambda_graph", float("inf")),
])
def test_invalid_structural_config(name, value) -> None:
    config = probe.StructuralMatcherConfig(**{name: value})
    with pytest.raises(ValueError):
        config.validate()


def test_cli_trains_both_models_and_keeps_artifacts_separate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "region_zone_match_probe.py", "--smoke", "--matchers", "learned", "learned_structural",
        "--epochs", "1", "--structural-refinement-steps", "3",
        "--structural-beta", "0.2", "--lambda-graph", "0.7",
        "--device", "cpu", "--output-dir", str(tmp_path),
    ])
    probe.main()
    payload = json.loads((tmp_path / "raw_results.json").read_text())
    assert {row["matcher"] for row in payload["results"]} == {"learned", "learned_structural"}
    assert len(payload["learned_history"]) == len(payload["learned_structural_history"]) == 1
    history = payload["learned_structural_history"][0]
    assert math.isfinite(history["train_graph_loss"])
    assert history["train_loss"] == pytest.approx(
        history["train_permutation_loss"] + 0.7 * history["train_graph_loss"], rel=1e-6,
    )
    assert "train_graph_loss" not in payload["learned_history"][0]
    checkpoint = torch.load(tmp_path / "learned_structural/checkpoint.pt", weights_only=True)
    assert checkpoint["structural_config"] == {
        "structural_refinement_steps": 3, "structural_beta": 0.2, "lambda_graph": 0.7,
    }
    assert torch.allclose(
        checkpoint["model_state"]["canonical_transition"],
        torch.tensor(payload["canonical_transition_matrix"]),
    )
    model = _model(4, structural_refinement_steps=3, structural_beta=0.2)
    model.load_state_dict(checkpoint["model_state"])
    assert (tmp_path / "learned/checkpoint.pt").is_file()
    assert (tmp_path / "learned_structural/history.json").is_file()
