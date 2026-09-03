from __future__ import annotations

import itertools

import torch

from src.benchmark.region_zone_match_probe import load_region_zone_probe_config
from src.benchmark.region_zone_matching import (
    _count_nll_candidate_objectives_from_delta,
    _swap_candidates,
    anonymize_region_sequences,
    build_anonymous_region_graph,
    frequency_assignment,
    graph_node_features,
    matching_objective,
    oracle_transition_assignment,
    row_normalize_transition_counts,
    score_region_assignment,
    stationary_distribution,
    transition_count_matrix,
)
from src.data.controlled_synthetic import MarkovZoneLanguage
from src.data.dataset import CipherEpisode
from src.models.region_zone_matcher import (
    LearnedRegionZoneMatcher,
    observed_permutation_nll,
)


def _canonical_transition(num_zones: int = 5) -> torch.Tensor:
    language = MarkovZoneLanguage.create(
        num_zones, symbols_per_zone=8, preferred_transitions=2, transition_strength=8.0, seed=9
    )
    return torch.tensor(language.transition_matrix, dtype=torch.float64)


def test_anonymous_reindexing_uses_first_occurrence_not_raw_region_identity() -> None:
    original, _mapping = anonymize_region_sequences(
        ((7, 7, 2, 9, 2),), 10, seed=4, table_id="f"
    )
    relabeled, _relabeled_mapping = anonymize_region_sequences(
        ((3, 3, 8, 1, 8),), 10, seed=4, table_id="f"
    )
    assert original == relabeled == ((0, 0, 1, 2, 1),)


def test_transition_counts_do_not_cross_episode_boundaries() -> None:
    counts = transition_count_matrix(((0, 1), (1, 0)), num_zones=3)
    expected = torch.zeros(3, 3, dtype=torch.float64)
    expected[0, 1] = 1
    expected[1, 0] = 1
    assert torch.equal(counts, expected)


def test_row_normalized_transition_matrix_is_correct() -> None:
    counts = torch.tensor([[1.0, 3.0], [0.0, 0.0]])
    normalized = row_normalize_transition_counts(counts, alpha=0.0)
    assert torch.allclose(normalized[0], torch.tensor([0.25, 0.75], dtype=torch.float64))
    assert torch.allclose(normalized[1], torch.tensor([0.5, 0.5], dtype=torch.float64))


def test_oracle_recovers_an_exactly_permuted_canonical_transition() -> None:
    canonical = _canonical_transition(7)
    true_assignment = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    observed = canonical[true_assignment][:, true_assignment]
    result = oracle_transition_assignment(
        observed,
        canonical,
        stationary_distribution(observed),
        torch.ones(7),
        max_iterations=30,
        restarts=3,
        seed=2,
    )
    assert result.assignment == tuple(true_assignment.tolist())
    expected_objective = matching_objective(
        observed,
        canonical,
        true_assignment.tolist(),
        torch.ones(7),
        objective="count_nll",
    )
    assert abs(result.objective - expected_objective) < 1e-12


def test_count_nll_true_permutation_is_exact_case_global_optimum() -> None:
    canonical = _canonical_transition(5)
    true_assignment = torch.tensor([3, 0, 4, 1, 2])
    semantic_counts = (
        10_000
        * stationary_distribution(canonical).unsqueeze(1)
        * canonical
    )
    observed_counts = semantic_counts[true_assignment][:, true_assignment]
    observed = observed_counts / observed_counts.sum(dim=1, keepdim=True)
    objectives = {
        permutation: matching_objective(
            observed,
            canonical,
            permutation,
            observed_counts.sum(dim=1),
            objective="count_nll",
            transition_counts=observed_counts,
        )
        for permutation in itertools.permutations(range(5))
    }
    truth = tuple(true_assignment.tolist())
    assert objectives[truth] == min(objectives.values())
    result = oracle_transition_assignment(
        observed,
        canonical,
        stationary_distribution(observed),
        observed_counts.sum(dim=1),
        max_iterations=30,
        restarts=3,
        seed=5,
        objective="count_nll",
        transition_counts=observed_counts,
    )
    assert result.assignment == truth


def test_vectorized_count_nll_move_delta_matches_full_objective() -> None:
    canonical = _canonical_transition(6)
    counts = torch.tensor(
        [
            [3, 1, 0, 2, 0, 1],
            [0, 4, 2, 0, 1, 0],
            [1, 0, 3, 1, 2, 0],
            [0, 2, 1, 5, 0, 1],
            [2, 0, 0, 1, 4, 1],
            [0, 1, 2, 0, 1, 3],
        ],
        dtype=torch.float64,
    )
    observed = counts / counts.sum(dim=1, keepdim=True)
    current = torch.tensor([2, 5, 1, 4, 0, 3])
    current_objective = matching_objective(
        observed,
        canonical,
        current.tolist(),
        counts.sum(dim=1),
        objective="count_nll",
        transition_counts=counts,
    )
    candidates = _swap_candidates(current)
    delta_values = _count_nll_candidate_objectives_from_delta(
        candidates, current, current_objective, counts, canonical, 1e-12
    )
    full_values = torch.tensor(
        [
            matching_objective(
                observed,
                canonical,
                candidate.tolist(),
                counts.sum(dim=1),
                objective="count_nll",
                transition_counts=counts,
            )
            for candidate in candidates
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(delta_values, full_values, atol=1e-10)


def test_oracle_assignment_is_equivariant_to_anonymous_row_reindexing() -> None:
    canonical = _canonical_transition(6)
    true_assignment = torch.tensor([4, 1, 5, 0, 3, 2])
    observed = canonical[true_assignment][:, true_assignment]
    reindex = torch.tensor([2, 5, 0, 3, 1, 4])
    result = oracle_transition_assignment(
        observed[reindex][:, reindex],
        canonical,
        stationary_distribution(observed)[reindex],
        torch.ones(6),
        max_iterations=30,
        restarts=3,
        seed=3,
    )
    assert result.assignment == tuple(true_assignment[reindex].tolist())


def test_unobserved_regions_are_masked_from_sinkhorn_and_loss() -> None:
    torch.manual_seed(1)
    model = LearnedRegionZoneMatcher(
        num_zones=4, d_model=8, num_layers=1, dropout=0.0, sinkhorn_iterations=60
    )
    features = torch.rand(1, 4, 5)
    transition = torch.softmax(torch.rand(1, 4, 4), dim=-1)
    observed = torch.tensor([[True, True, False, False]])
    targets = torch.tensor([[1, 3, 0, 2]])
    output = model(features, transition, observed)
    assert torch.all(output.assignment_probabilities[0, 2:] == 0)
    first = observed_permutation_nll(output.assignment_probabilities, targets, observed)
    changed = targets.clone()
    changed[0, 2:] = torch.tensor([3, 1])
    second = observed_permutation_nll(output.assignment_probabilities, changed, observed)
    assert torch.allclose(first, second)


def test_token_reconstruction_accuracy_uses_region_frequency() -> None:
    episode = CipherEpisode(
        cipher_values=(20, 21, 10, 22),
        zone_labels=(1, 1, 0, 1),
        cipher_zone_ids=(2, 2, 1, 2),
        table_id="f",
    )
    graph = build_anonymous_region_graph(
        [episode], zone_to_raw_region=(1, 2, 0), num_zones=3, alpha=1e-3, anonymization_seed=5
    )
    result = score_region_assignment(graph, (1, 2, 0))
    assert result["observed_region_assignment_accuracy"] == 0.5
    assert result["token_accuracy"] == 0.75


def test_frequency_baseline_is_a_one_to_one_assignment() -> None:
    assignment = frequency_assignment(
        torch.tensor([0.1, 0.7, 0.2]), torch.tensor([0.6, 0.3, 0.1])
    )
    assert assignment == (2, 0, 1)
    assert sorted(assignment) == [0, 1, 2]


def test_learned_matcher_output_is_permutation_equivariant() -> None:
    torch.manual_seed(3)
    model = LearnedRegionZoneMatcher(
        num_zones=4, d_model=8, num_layers=2, dropout=0.0, sinkhorn_iterations=60
    ).eval()
    transition = torch.softmax(torch.randn(1, 4, 4), dim=-1)
    features = torch.randn(1, 4, 5)
    observed = torch.ones(1, 4, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 3, 1])
    original = model(features, transition, observed)
    reindexed = model(
        features[:, permutation],
        transition[:, permutation][:, :, permutation],
        observed[:, permutation],
    )
    assert original.assignment_probabilities.shape == (1, 4, 4)
    assert torch.allclose(
        reindexed.assignment_probabilities,
        original.assignment_probabilities[:, permutation],
        atol=1e-6,
    )


def test_graph_features_have_no_numeric_cipher_input_dimension() -> None:
    episode = CipherEpisode((999, 101), (0, 1), (1, 0), "f")
    graph = build_anonymous_region_graph(
        [episode], zone_to_raw_region=(1, 0), num_zones=2, alpha=1e-3, anonymization_seed=1
    )
    assert graph_node_features(graph).shape == (2, 5)


def test_region_zone_probe_config_has_requested_defaults() -> None:
    config = load_region_zone_probe_config("configs/region_zone_match_probe.yaml")
    assert config.matchers == ("random", "frequency", "oracle_transition", "learned")
    assert config.oracle.objective == "count_nll"
    assert config.synthetic.sequence_lengths == (128,)
    assert config.synthetic.sequences_per_length == 1
