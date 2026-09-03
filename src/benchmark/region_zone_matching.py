from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from ..data.dataset import CipherEpisode
from ..training.assignment import maximum_weight_assignment


MATCHERS = ("random", "frequency", "oracle_transition", "learned")


@dataclass(frozen=True)
class AnonymousRegionGraph:
    """A table-level transition graph with no numeric coordinate features."""

    table_id: str
    sequence_length: int
    anonymous_sequences: tuple[tuple[int, ...], ...]
    anonymous_to_raw_region: tuple[int, ...]
    transition_counts: Tensor
    transition_matrix: Tensor
    token_counts: Tensor
    token_frequency: Tensor
    observed_mask: Tensor
    true_zone_by_anonymous_region: Tensor

    @property
    def num_zones(self) -> int:
        return int(self.transition_matrix.shape[0])


@dataclass(frozen=True)
class OracleMatchResult:
    assignment: tuple[int, ...]
    objective: float
    iterations: int
    converged: bool


def _stable_seed(seed: int, table_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{table_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def anonymize_region_sequences(
    raw_sequences: Sequence[Sequence[int]],
    num_zones: int,
    seed: int,
    table_id: str,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Reindex observed regions by first occurrence and randomize unseen rows."""
    ordered: list[int] = []
    seen: set[int] = set()
    for sequence in raw_sequences:
        for region in sequence:
            if not 0 <= int(region) < num_zones:
                raise ValueError("raw region index is outside the configured zone range")
            if int(region) not in seen:
                seen.add(int(region))
                ordered.append(int(region))
    missing = [region for region in range(num_zones) if region not in seen]
    random.Random(_stable_seed(seed, table_id)).shuffle(missing)
    anonymous_to_raw = tuple(ordered + missing)
    raw_to_anonymous = {
        raw_region: anonymous for anonymous, raw_region in enumerate(anonymous_to_raw)
    }
    anonymous_sequences = tuple(
        tuple(raw_to_anonymous[int(region)] for region in sequence)
        for sequence in raw_sequences
    )
    return anonymous_sequences, anonymous_to_raw


def transition_count_matrix(
    sequences: Sequence[Sequence[int]], num_zones: int
) -> Tensor:
    """Count directed transitions without adding links across episode boundaries."""
    counts = torch.zeros(num_zones, num_zones, dtype=torch.float64)
    for sequence in sequences:
        for source, target in zip(sequence, sequence[1:]):
            counts[int(source), int(target)] += 1.0
    return counts


def row_normalize_transition_counts(counts: Tensor, alpha: float = 1e-3) -> Tensor:
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("transition counts must be a square matrix")
    if alpha < 0:
        raise ValueError("Laplace smoothing alpha must be non-negative")
    smoothed = counts.to(torch.float64) + alpha
    denominator = smoothed.sum(dim=1, keepdim=True)
    uniform = torch.full_like(smoothed, 1.0 / counts.shape[0])
    return torch.where(denominator > 0, smoothed / denominator.clamp_min(1e-12), uniform)


def build_anonymous_region_graph(
    episodes: Sequence[CipherEpisode],
    zone_to_raw_region: Sequence[int],
    num_zones: int,
    alpha: float,
    anonymization_seed: int,
) -> AnonymousRegionGraph:
    if not episodes:
        raise ValueError("cannot build a region graph from no episodes")
    table_ids = {str(episode.table_id) for episode in episodes}
    lengths = {len(episode.cipher_zone_ids) for episode in episodes}
    if len(table_ids) != 1 or len(lengths) != 1:
        raise ValueError("one graph requires one table and one sequence length")
    if sorted(map(int, zone_to_raw_region)) != list(range(num_zones)):
        raise ValueError("zone_to_raw_region must be a complete permutation")
    table_id = next(iter(table_ids))
    raw_sequences = [episode.cipher_zone_ids for episode in episodes]
    anonymous_sequences, anonymous_to_raw = anonymize_region_sequences(
        raw_sequences, num_zones, anonymization_seed, table_id
    )
    counts = transition_count_matrix(anonymous_sequences, num_zones)
    token_counts = torch.zeros(num_zones, dtype=torch.float64)
    for sequence in anonymous_sequences:
        token_counts += torch.bincount(
            torch.tensor(sequence), minlength=num_zones
        ).to(torch.float64)
    observed_mask = token_counts > 0
    raw_to_zone = [-1] * num_zones
    for zone, raw_region in enumerate(zone_to_raw_region):
        raw_to_zone[int(raw_region)] = zone
    true_assignment = torch.tensor(
        [raw_to_zone[raw_region] for raw_region in anonymous_to_raw], dtype=torch.long
    )
    for episode, anonymous_sequence in zip(episodes, anonymous_sequences):
        expected = true_assignment[torch.tensor(anonymous_sequence)]
        if not torch.equal(expected, torch.tensor(episode.zone_labels)):
            raise ValueError("episode labels disagree with table permutation metadata")
    return AnonymousRegionGraph(
        table_id=table_id,
        sequence_length=next(iter(lengths)),
        anonymous_sequences=anonymous_sequences,
        anonymous_to_raw_region=anonymous_to_raw,
        transition_counts=counts,
        transition_matrix=row_normalize_transition_counts(counts, alpha),
        token_counts=token_counts,
        token_frequency=token_counts / token_counts.sum().clamp_min(1.0),
        observed_mask=observed_mask,
        true_zone_by_anonymous_region=true_assignment,
    )


def build_graphs_by_length(
    episodes: Sequence[CipherEpisode],
    table_zone_to_region: dict[str, tuple[int, ...]],
    num_zones: int,
    alpha: float,
    anonymization_seed: int,
) -> dict[int, list[AnonymousRegionGraph]]:
    grouped: dict[tuple[int, str], list[CipherEpisode]] = {}
    for episode in episodes:
        key = (len(episode.cipher_values), str(episode.table_id))
        grouped.setdefault(key, []).append(episode)
    result: dict[int, list[AnonymousRegionGraph]] = {}
    for (length, table_id), table_episodes in sorted(grouped.items()):
        result.setdefault(length, []).append(
            build_anonymous_region_graph(
                table_episodes,
                table_zone_to_region[table_id],
                num_zones,
                alpha,
                anonymization_seed,
            )
        )
    return result


def stationary_distribution(transition: Tensor, iterations: int = 1000) -> Tensor:
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition matrix must be square")
    probability = torch.full(
        (transition.shape[0],),
        1.0 / transition.shape[0],
        dtype=torch.float64,
        device=transition.device,
    )
    matrix = transition.to(torch.float64)
    for _ in range(iterations):
        updated = probability @ matrix
        if float((updated - probability).abs().max()) < 1e-13:
            probability = updated
            break
        probability = updated
    return probability / probability.sum()


def _entropy(probabilities: Tensor) -> Tensor:
    normalizer = math.log(max(probabilities.shape[-1], 2))
    return -(
        probabilities.clamp_min(1e-12)
        * probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1) / normalizer


def transition_node_signatures(
    transition: Tensor,
    frequency: Tensor | None = None,
) -> Tensor:
    """Permutation-invariant node descriptors used only for oracle initialization."""
    matrix = transition.to(torch.float64)
    frequency = (
        stationary_distribution(matrix)
        if frequency is None
        else frequency.to(torch.float64) / frequency.sum().clamp_min(1e-12)
    )
    incoming_joint = frequency.unsqueeze(1) * matrix
    incoming = incoming_joint.t()
    incoming = incoming / incoming.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.cat(
        (
            frequency.unsqueeze(1),
            matrix.diag().unsqueeze(1),
            _entropy(matrix).unsqueeze(1),
            _entropy(incoming).unsqueeze(1),
            matrix.sort(dim=1).values,
            incoming.sort(dim=1).values,
        ),
        dim=1,
    )


def graph_node_features(graph: AnonymousRegionGraph) -> Tensor:
    counts = graph.transition_counts
    incoming_total = counts.sum(dim=0)
    incoming_distribution = counts.t() / incoming_total.unsqueeze(1).clamp_min(1.0)
    return torch.stack(
        (
            graph.token_frequency,
            graph.transition_matrix.diag(),
            incoming_total / counts.sum().clamp_min(1.0),
            _entropy(graph.transition_matrix),
            _entropy(incoming_distribution),
        ),
        dim=1,
    ).to(torch.float32)


def random_assignment(num_zones: int, seed: int) -> tuple[int, ...]:
    assignment = list(range(num_zones))
    random.Random(seed).shuffle(assignment)
    return tuple(assignment)


def frequency_assignment(
    anonymous_frequency: Tensor, canonical_frequency: Tensor
) -> tuple[int, ...]:
    if anonymous_frequency.shape != canonical_frequency.shape:
        raise ValueError("frequency vectors must have the same shape")
    anonymous_order = sorted(
        range(len(anonymous_frequency)),
        key=lambda index: (-float(anonymous_frequency[index]), index),
    )
    semantic_order = sorted(
        range(len(canonical_frequency)),
        key=lambda index: (-float(canonical_frequency[index]), index),
    )
    assignment = [-1] * len(anonymous_order)
    for anonymous, semantic in zip(anonymous_order, semantic_order):
        assignment[anonymous] = semantic
    return tuple(assignment)


def matching_objective(
    observed_transition: Tensor,
    canonical_transition: Tensor,
    assignment: Sequence[int],
    row_weights: Tensor | None = None,
) -> float:
    permutation = torch.tensor(assignment, dtype=torch.long)
    aligned = canonical_transition.to(torch.float64)[permutation][:, permutation]
    difference = (observed_transition.to(torch.float64) - aligned).square().mean(dim=1)
    weights = (
        torch.ones_like(difference)
        if row_weights is None
        else row_weights.to(torch.float64)
    )
    return float((difference * weights).sum() / weights.sum().clamp_min(1e-12))


def _signature_initialization(
    observed_transition: Tensor,
    canonical_transition: Tensor,
    observed_frequency: Tensor | None,
) -> list[int]:
    observed = transition_node_signatures(observed_transition, observed_frequency)
    canonical = transition_node_signatures(canonical_transition)
    combined = torch.cat((observed, canonical), dim=0)
    scale = combined.std(dim=0, unbiased=False).clamp_min(1e-6)
    cost = ((observed.unsqueeze(1) - canonical.unsqueeze(0)) / scale).square().mean(-1)
    return maximum_weight_assignment(-cost)


def _structural_assignment_cost(
    observed_transition: Tensor,
    canonical_transition: Tensor,
    current: Sequence[int],
    row_weights: Tensor,
) -> Tensor:
    num_zones = observed_transition.shape[0]
    current_index = torch.tensor(current, dtype=torch.long)
    q = observed_transition.to(torch.float64)
    p = canonical_transition.to(torch.float64)
    weights = row_weights.to(torch.float64)
    result = torch.empty(num_zones, num_zones, dtype=torch.float64)
    for anonymous in range(num_zones):
        for semantic in range(num_zones):
            outgoing_expected = p[semantic, current_index].clone()
            incoming_expected = p[current_index, semantic].clone()
            outgoing_expected[anonymous] = p[semantic, semantic]
            incoming_expected[anonymous] = p[semantic, semantic]
            outgoing = weights[anonymous] * (
                q[anonymous] - outgoing_expected
            ).square().mean()
            incoming = (
                weights * (q[:, anonymous] - incoming_expected).square()
            ).sum() / weights.sum().clamp_min(1e-12)
            result[anonymous, semantic] = outgoing + incoming
    return result


def oracle_transition_assignment(
    observed_transition: Tensor,
    canonical_transition: Tensor,
    observed_frequency: Tensor | None = None,
    row_weights: Tensor | None = None,
    max_iterations: int = 50,
    restarts: int = 4,
    seed: int = 0,
) -> OracleMatchResult:
    """Approximate graph matching without enumerating the 19! permutations."""
    if observed_transition.shape != canonical_transition.shape:
        raise ValueError("observed and canonical transitions must have matching shape")
    if max_iterations < 1 or restarts < 1:
        raise ValueError("oracle iterations and restarts must be positive")
    num_zones = observed_transition.shape[0]
    weights = (
        torch.ones(num_zones, dtype=torch.float64)
        if row_weights is None
        else row_weights.to(torch.float64)
    )
    signature_start = _signature_initialization(
        observed_transition, canonical_transition, observed_frequency
    )
    rng = random.Random(seed)
    starts = [signature_start]
    for _ in range(restarts - 1):
        candidate = list(range(num_zones))
        rng.shuffle(candidate)
        starts.append(candidate)

    best_assignment: list[int] | None = None
    best_objective = math.inf
    best_iterations = 0
    best_converged = False
    for start in starts:
        current = list(start)
        objective = matching_objective(
            observed_transition, canonical_transition, current, weights
        )
        iterations = 0
        converged = False
        for _ in range(max_iterations):
            iterations += 1
            proposal = maximum_weight_assignment(
                -_structural_assignment_cost(
                    observed_transition, canonical_transition, current, weights
                )
            )
            proposal_objective = matching_objective(
                observed_transition, canonical_transition, proposal, weights
            )
            if proposal == current or proposal_objective >= objective - 1e-14:
                converged = True
                break
            current = proposal
            objective = proposal_objective

        # Pair-swap descent catches improvements that a simultaneous Hungarian
        # update can miss while remaining tiny for a 19-node graph.
        for _ in range(max_iterations):
            best_swap: tuple[int, int] | None = None
            swap_objective = objective
            for left in range(num_zones):
                for right in range(left + 1, num_zones):
                    proposal = current.copy()
                    proposal[left], proposal[right] = proposal[right], proposal[left]
                    candidate_objective = matching_objective(
                        observed_transition, canonical_transition, proposal, weights
                    )
                    if candidate_objective < swap_objective - 1e-14:
                        best_swap = (left, right)
                        swap_objective = candidate_objective
            if best_swap is None:
                converged = True
                break
            current[best_swap[0]], current[best_swap[1]] = (
                current[best_swap[1]],
                current[best_swap[0]],
            )
            objective = swap_objective
            iterations += 1
        if objective < best_objective:
            best_assignment = current
            best_objective = objective
            best_iterations = iterations
            best_converged = converged
    assert best_assignment is not None
    return OracleMatchResult(
        tuple(best_assignment), best_objective, best_iterations, best_converged
    )


def score_region_assignment(
    graph: AnonymousRegionGraph, assignment: Sequence[int]
) -> dict:
    predicted = torch.tensor(assignment, dtype=torch.long)
    if sorted(predicted.tolist()) != list(range(graph.num_zones)):
        raise ValueError("region matcher must return a one-to-one permutation")
    target = graph.true_zone_by_anonymous_region
    observed = graph.observed_mask
    correct = predicted.eq(target)
    token_correct = graph.token_counts[correct].sum()
    assignment_confusion = torch.zeros(
        graph.num_zones, graph.num_zones, dtype=torch.long
    )
    token_confusion = torch.zeros_like(assignment_confusion)
    for row in range(graph.num_zones):
        if bool(observed[row]):
            assignment_confusion[target[row], predicted[row]] += 1
            token_confusion[target[row], predicted[row]] += int(graph.token_counts[row])
    return {
        "table_id": graph.table_id,
        "sequence_length": graph.sequence_length,
        "observed_region_assignment_accuracy": float(correct[observed].to(torch.float64).mean()),
        "full_assignment_accuracy": float(correct.to(torch.float64).mean()),
        "token_accuracy": float(token_correct / graph.token_counts.sum().clamp_min(1.0)),
        "observed_exact_recovery": bool(correct[observed].all()),
        "full_exact_recovery": bool(correct.all()),
        "num_observed_regions": int(observed.sum()),
        "region_coverage": float(observed.to(torch.float64).mean()),
        "predicted_assignment": predicted.tolist(),
        "true_assignment": target.tolist(),
        "assignment_confusion": assignment_confusion.tolist(),
        "token_confusion": token_confusion.tolist(),
    }


def aggregate_assignment_results(table_results: Sequence[dict], num_zones: int) -> dict:
    if not table_results:
        raise ValueError("cannot aggregate no table results")
    assignment_confusion = torch.zeros(num_zones, num_zones, dtype=torch.long)
    token_confusion = torch.zeros_like(assignment_confusion)
    for result in table_results:
        assignment_confusion += torch.tensor(result["assignment_confusion"])
        token_confusion += torch.tensor(result["token_confusion"])
    per_zone_total = assignment_confusion.sum(dim=1)
    per_zone_accuracy = {
        str(zone): (
            float(assignment_confusion[zone, zone] / per_zone_total[zone])
            if int(per_zone_total[zone])
            else 0.0
        )
        for zone in range(num_zones)
    }

    def mean(key: str) -> float:
        return sum(float(result[key]) for result in table_results) / len(table_results)

    observed_correct = sum(
        result["observed_region_assignment_accuracy"] * result["num_observed_regions"]
        for result in table_results
    )
    observed_total = sum(result["num_observed_regions"] for result in table_results)
    return {
        "num_tables": len(table_results),
        "observed_region_assignment_accuracy": observed_correct / max(observed_total, 1),
        "mean_table_observed_assignment_accuracy": mean(
            "observed_region_assignment_accuracy"
        ),
        "full_assignment_accuracy": mean("full_assignment_accuracy"),
        "token_accuracy": mean("token_accuracy"),
        "observed_exact_recovery_rate": mean("observed_exact_recovery"),
        "full_exact_recovery_rate": mean("full_exact_recovery"),
        "mean_num_observed_regions": mean("num_observed_regions"),
        "min_num_observed_regions": min(
            result["num_observed_regions"] for result in table_results
        ),
        "max_num_observed_regions": max(
            result["num_observed_regions"] for result in table_results
        ),
        "mean_region_coverage": mean("region_coverage"),
        "per_zone_observed_assignment_accuracy": per_zone_accuracy,
        "assignment_confusion_matrix": assignment_confusion.tolist(),
        "token_confusion_matrix": token_confusion.tolist(),
    }


def canonical_identifiability(transition: Tensor, top_k: int = 10) -> dict:
    matrix = transition.to(torch.float64)
    signatures = transition_node_signatures(matrix)
    scale = signatures.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = signatures / scale
    ambiguous: list[dict] = []
    swaps: list[dict] = []
    identity = list(range(matrix.shape[0]))
    for left in range(matrix.shape[0]):
        for right in range(left + 1, matrix.shape[0]):
            ambiguous.append(
                {
                    "zones": [left, right],
                    "signature_distance": float(
                        (normalized[left] - normalized[right]).square().mean().sqrt()
                    ),
                }
            )
            permutation = identity.copy()
            permutation[left], permutation[right] = permutation[right], permutation[left]
            swaps.append(
                {
                    "zones": [left, right],
                    "swap_objective": matching_objective(
                        matrix, matrix, permutation
                    ),
                }
            )
    ambiguous.sort(key=lambda item: item["signature_distance"])
    swaps.sort(key=lambda item: item["swap_objective"])
    return {
        "stationary_frequency": stationary_distribution(matrix).tolist(),
        "most_ambiguous_signature_pairs": ambiguous[:top_k],
        "lowest_cost_pair_swaps": swaps[:top_k],
    }
