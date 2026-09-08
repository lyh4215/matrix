from __future__ import annotations

import json
import sys

import pytest
import torch

from src.benchmark import region_zone_match_probe as probe
from src.benchmark.region_zone_local_search import (
    LocalSearchConfig,
    evaluate_local_search,
    refine_count_assignment,
)
from src.benchmark.region_zone_matching import (
    AnonymousRegionGraph,
    aggregate_assignment_results,
    score_region_assignment,
)
from src.benchmark.region_zone_refine import refine_saved_results
from src.data.controlled_synthetic import MarkovZoneLanguage
from src.models.region_zone_matcher import LearnedRegionZoneMatcher


def _canonical(zones: int) -> torch.Tensor:
    language = MarkovZoneLanguage.create(
        zones, symbols_per_zone=8, preferred_transitions=2,
        transition_strength=8.0, seed=9,
    )
    return torch.tensor(language.transition_matrix, dtype=torch.float64)


@pytest.mark.parametrize("zones", [5, 19])
def test_repairs_swapped_exact_permutation_and_is_equivariant(zones: int) -> None:
    generator = torch.Generator().manual_seed(17)
    canonical = _canonical(zones)
    truth = torch.randperm(zones, generator=generator)
    counts = 100 * canonical[truth][:, truth]
    initial = truth.clone()
    initial[0], initial[1] = initial[1].item(), initial[0].item()
    result = refine_count_assignment(counts, canonical, initial)
    assert result.assignment == tuple(truth.tolist())
    assert result.final_count_nll < result.initial_count_nll
    assert result.accepted_moves == 1
    assert result.converged
    reindex = torch.randperm(zones, generator=generator)
    reindexed = refine_count_assignment(counts[reindex][:, reindex], canonical, initial[reindex])
    assert reindexed.assignment == tuple(truth[reindex].tolist())
    assert reindexed.final_count_nll == pytest.approx(result.final_count_nll)


def test_noisy_partial_counts_never_worsen_nll_and_preserve_permutation() -> None:
    generator = torch.Generator().manual_seed(3)
    canonical = _canonical(19)
    counts = torch.randint(0, 5, (19, 19), generator=generator).double()
    counts[10:] = 0
    counts[:, 10:] = 0
    initial = torch.randperm(19, generator=generator)
    settings = LocalSearchConfig(max_iterations=5, restarts=3)
    result = refine_count_assignment(counts, canonical, initial, settings, seed=4)
    repeated = refine_count_assignment(counts, canonical, initial, settings, seed=4)
    assert sorted(result.assignment) == list(range(19))
    assert result.final_count_nll <= result.initial_count_nll
    assert result.assignment == repeated.assignment
    assert result.winning_restart == repeated.winning_restart
    permutation = torch.tensor(result.assignment)
    expected = -(counts * canonical[permutation][:, permutation].log()).sum()
    assert result.final_count_nll == pytest.approx(float(expected), abs=1e-10)


def test_empty_counts_keep_initial_assignment_even_with_restarts() -> None:
    initial = [2, 0, 3, 1]
    result = refine_count_assignment(
        torch.zeros(4, 4), _canonical(4), initial, LocalSearchConfig(restarts=4),
    )
    assert result.assignment == tuple(initial)
    assert result.initial_count_nll == result.final_count_nll == 0.0
    assert result.accepted_moves == 0


def test_reports_accuracy_regression_even_when_likelihood_improves() -> None:
    canonical = _canonical(4)
    noisy_order = torch.tensor([1, 0, 2, 3])
    counts = 100 * canonical[noisy_order][:, noisy_order]
    graph = AnonymousRegionGraph(
        table_id="noisy", sequence_length=32, anonymous_sequences=(),
        anonymous_to_raw_region=(0, 1, 2, 3), transition_counts=counts,
        transition_matrix=counts / counts.sum(1, keepdim=True),
        token_counts=torch.ones(4), token_frequency=torch.ones(4) / 4,
        observed_mask=torch.ones(4, dtype=torch.bool),
        true_zone_by_anonymous_region=torch.arange(4),
    )
    base_table = score_region_assignment(graph, (0, 1, 2, 3))
    base = {
        "matcher": "learned", "sequence_length": 32,
        **aggregate_assignment_results([base_table], 4), "table_results": [base_table],
    }
    result = evaluate_local_search(base, [graph], canonical, LocalSearchConfig(), seed=2)
    assert result["local_search"]["tables_assignment_worsened"] == 1
    assert result["local_search"]["assignment_accuracy_delta"] == -0.5
    assert result["local_search"]["mean_nll_improvement_per_transition"] > 0
    assert result["table_results"][0]["predicted_assignment"] == noisy_order.tolist()


@pytest.mark.parametrize("kwargs", [
    {"max_iterations": 0}, {"restarts": -1}, {"restarts": 1.5}, {"epsilon": float("nan")},
])
def test_invalid_local_search_config(kwargs) -> None:
    with pytest.raises(ValueError):
        LocalSearchConfig(**kwargs).validate()


def test_rejects_non_permutation_initial_assignment() -> None:
    with pytest.raises(ValueError, match="one-to-one"):
        refine_count_assignment(torch.ones(4, 4), _canonical(4), [0, 0, 2, 3])


def test_probe_and_saved_prediction_replay_match_without_training(tmp_path, monkeypatch) -> None:
    original_dir = tmp_path / "original"
    monkeypatch.setattr(sys, "argv", [
        "region_zone_match_probe.py", "--smoke", "--matchers", "learned", "learned_structural",
        "--local-search", "--local-search-max-iterations", "10", "--local-search-restarts", "2",
        "--device", "cpu", "--output-dir", str(original_dir),
    ])
    probe.main()
    source = original_dir / "raw_results.json"
    original_bytes = source.read_bytes()
    original = json.loads(original_bytes)
    assert {row["matcher"] for row in original["results"]} == {
        "learned", "learned_structural", "learned_local_search", "learned_structural_local_search",
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("saved prediction replay must not train or run a neural model")

    monkeypatch.setattr(probe, "_train_learned_matcher", forbidden)
    monkeypatch.setattr(LearnedRegionZoneMatcher, "forward", forbidden)
    replay = refine_saved_results(
        source, tmp_path / "replay", LocalSearchConfig(max_iterations=10, restarts=2),
    )
    assert source.read_bytes() == original_bytes
    for before, after in zip(original["results"], replay["results"]):
        assert before["matcher"] == after["matcher"]
        assert before["observed_region_assignment_accuracy"] == after["observed_region_assignment_accuracy"]
        assert before["token_accuracy"] == after["token_accuracy"]
        for left, right in zip(before["table_results"], after["table_results"]):
            assert left["predicted_assignment"] == right["predicted_assignment"]
    markdown = (tmp_path / "replay/summary.md").read_text()
    assert "Better / Worse / Same" in markdown
    assert "Seconds / table" in markdown
    with pytest.raises(ValueError, match="separate output"):
        refine_saved_results(source, original_dir)
    # The canonical identity check prevents silently evaluating old predictions
    # against a different synthetic language.
    original["canonical_transition_matrix"][0][0] += 0.1
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="canonical matrix"):
        refine_saved_results(changed, tmp_path / "wrong_language")
