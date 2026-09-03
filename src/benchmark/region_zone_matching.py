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
    objective: str = "mse",
    transition_counts: Tensor | None = None,
    epsilon: float = 1e-12,
) -> float:
    permutation = torch.tensor(assignment, dtype=torch.long).unsqueeze(0)
    return float(
        _objective_values(
            permutation,
            observed_transition,
            canonical_transition,
            row_weights,
            objective,
            transition_counts,
            epsilon,
        )[0]
    )


def _objective_values(
    permutations: Tensor,
    observed_transition: Tensor,
    canonical_transition: Tensor,
    row_weights: Tensor | None,
    objective: str,
    transition_counts: Tensor | None,
    epsilon: float,
) -> Tensor:
    if objective not in {"mse", "count_nll"}:
        raise ValueError("oracle objective must be mse or count_nll")
    if epsilon <= 0:
        raise ValueError("objective epsilon must be positive")
    q = observed_transition.to(torch.float64)
    p = canonical_transition.to(torch.float64)
    permutations = permutations.to(device=p.device, dtype=torch.long)
    aligned = p[
        permutations.unsqueeze(2),
        permutations.unsqueeze(1),
    ]
    weights = (
        torch.ones(q.shape[0], dtype=torch.float64, device=q.device)
        if row_weights is None
        else row_weights.to(device=q.device, dtype=torch.float64)
    )
    if objective == "mse":
        difference = (q.unsqueeze(0) - aligned).square().mean(dim=2)
        return (difference * weights.unsqueeze(0)).sum(dim=1) / weights.sum().clamp_min(
            1e-12
        )
    counts = (
        q * weights.unsqueeze(1)
        if transition_counts is None
        else transition_counts.to(device=q.device, dtype=torch.float64)
    )
    return -(
        counts.unsqueeze(0) * aligned.clamp_min(epsilon).log()
    ).sum(dim=(1, 2))


def _signature_cost_matrix(
    observed_transition: Tensor,
    canonical_transition: Tensor,
    observed_frequency: Tensor | None,
) -> Tensor:
    observed = transition_node_signatures(observed_transition, observed_frequency)
    canonical = transition_node_signatures(canonical_transition)
    combined = torch.cat((observed, canonical), dim=0)
    scale = combined.std(dim=0, unbiased=False).clamp_min(1e-6)
    return ((observed.unsqueeze(1) - canonical.unsqueeze(0)) / scale).square().mean(-1)


def _structural_assignment_cost(
    observed_transition: Tensor,
    canonical_transition: Tensor,
    current: Sequence[int],
    row_weights: Tensor,
    objective: str,
    transition_counts: Tensor | None,
    epsilon: float,
) -> Tensor:
    q = observed_transition.to(torch.float64)
    p = canonical_transition.to(torch.float64)
    current_index = torch.tensor(current, dtype=torch.long, device=q.device)
    diagonal_mask = torch.eye(q.shape[0], dtype=torch.bool, device=q.device)
    if objective == "count_nll":
        counts = (
            q * row_weights.to(torch.float64).unsqueeze(1)
            if transition_counts is None
            else transition_counts.to(torch.float64)
        )
        off_diagonal = counts.masked_fill(diagonal_mask, 0.0)
        log_p = p.clamp_min(epsilon).log()
        outgoing = -(off_diagonal @ log_p[:, current_index].t())
        incoming = -(off_diagonal.t() @ log_p[current_index])
        self_cost = -counts.diag().unsqueeze(1) * log_p.diag().unsqueeze(0)
        return 0.5 * (outgoing + incoming) + self_cost
    if objective != "mse":
        raise ValueError("oracle objective must be mse or count_nll")
    weights = row_weights.to(torch.float64)
    off_diagonal = (~diagonal_mask).to(torch.float64)
    outgoing_difference = q.unsqueeze(1) - p[:, current_index].unsqueeze(0)
    outgoing = (
        outgoing_difference.square() * off_diagonal.unsqueeze(1)
    ).sum(dim=2) / q.shape[0]
    outgoing = outgoing * weights.unsqueeze(1)
    incoming_difference = q.t().unsqueeze(1) - p[current_index].t().unsqueeze(0)
    incoming = (
        incoming_difference.square()
        * off_diagonal.unsqueeze(1)
        * weights.view(1, 1, -1)
    ).sum(dim=2) / weights.sum().clamp_min(1e-12)
    self_difference = (q.diag().unsqueeze(1) - p.diag().unsqueeze(0)).square()
    self_scale = weights.unsqueeze(1) * (
        0.5 / q.shape[0] + 0.5 / weights.sum().clamp_min(1e-12)
    )
    return 0.5 * (outgoing + incoming) + self_scale * self_difference


def _initial_assignments(
    observed_transition: Tensor,
    canonical_transition: Tensor,
    observed_frequency: Tensor | None,
    row_weights: Tensor,
    objective: str,
    transition_counts: Tensor | None,
    epsilon: float,
    restarts: int,
    seed: int,
) -> list[list[int]]:
    signature_cost = _signature_cost_matrix(
        observed_transition, canonical_transition, observed_frequency
    )
    frequency = (
        stationary_distribution(observed_transition)
        if observed_frequency is None
        else observed_frequency
    )
    signature = maximum_weight_assignment(-signature_cost)
    frequency_match = list(
        frequency_assignment(
            frequency, stationary_distribution(canonical_transition)
        )
    )
    candidates = [signature, frequency_match]
    if objective == "count_nll":
        counts = (
            observed_transition.to(torch.float64)
            * row_weights.to(torch.float64).unsqueeze(1)
            if transition_counts is None
            else transition_counts.to(torch.float64)
        )
        candidates.extend(
            (
                _faq_initialization(
                    counts, canonical_transition, signature, epsilon
                ),
                _faq_initialization(
                    counts, canonical_transition, frequency_match, epsilon
                ),
            )
        )
    generator = torch.Generator().manual_seed(seed)
    base_noise_scale = max(float(signature_cost.std(unbiased=False)), 1e-6)
    for restart in range(max(restarts - 1, 0)):
        noise = torch.randn(
            signature_cost.shape, generator=generator, dtype=signature_cost.dtype
        )
        candidates.append(
            maximum_weight_assignment(
                -(signature_cost + base_noise_scale * (0.2 + 0.1 * restart) * noise)
            )
        )
    random_candidate = list(range(observed_transition.shape[0]))
    random.Random(seed).shuffle(random_candidate)
    candidates.append(random_candidate)
    unique: list[list[int]] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _faq_initialization(
    transition_counts: Tensor,
    canonical_transition: Tensor,
    initial_assignment: Sequence[int],
    epsilon: float,
    iterations: int = 30,
) -> list[int]:
    """Frank-Wolfe FAQ relaxation used only to seed discrete local search."""
    counts = transition_counts.to(torch.float64)
    pair_cost = -canonical_transition.to(torch.float64).clamp_min(epsilon).log()
    num_zones = counts.shape[0]
    permutation = torch.zeros(num_zones, num_zones, dtype=torch.float64)
    permutation[
        torch.arange(num_zones), torch.tensor(initial_assignment, dtype=torch.long)
    ] = 1.0
    uniform = torch.full_like(permutation, 1.0 / num_zones)
    relaxed = 0.5 * permutation + 0.5 * uniform

    def relaxed_objective(matrix: Tensor) -> Tensor:
        return (counts * (matrix @ pair_cost @ matrix.t())).sum()

    for _ in range(iterations):
        gradient = (
            counts @ relaxed @ pair_cost.t()
            + counts.t() @ relaxed @ pair_cost
        )
        direction_assignment = maximum_weight_assignment(-gradient)
        endpoint = torch.zeros_like(relaxed)
        endpoint[
            torch.arange(num_zones),
            torch.tensor(direction_assignment, dtype=torch.long),
        ] = 1.0
        direction = endpoint - relaxed
        quadratic = (counts * (direction @ pair_cost @ direction.t())).sum()
        linear = (
            counts
            * (
                direction @ pair_cost @ relaxed.t()
                + relaxed @ pair_cost @ direction.t()
            )
        ).sum()
        if float(quadratic) > 0:
            step = float((-linear / (2.0 * quadratic)).clamp(0.0, 1.0))
        else:
            step = (
                1.0
                if float(relaxed_objective(endpoint))
                < float(relaxed_objective(relaxed))
                else 0.0
            )
        if step <= 1e-10:
            break
        relaxed = relaxed + step * direction
    return maximum_weight_assignment(relaxed)


def _swap_candidates(current: Tensor) -> Tensor:
    pairs = torch.triu_indices(
        len(current), len(current), offset=1, device=current.device
    )
    candidates = current.unsqueeze(0).expand(pairs.shape[1], -1).clone()
    rows = torch.arange(pairs.shape[1], device=current.device)
    candidates[rows, pairs[0]] = current[pairs[1]]
    candidates[rows, pairs[1]] = current[pairs[0]]
    return candidates


def _three_cycle_candidates(current: Tensor) -> Tensor:
    triples = torch.combinations(
        torch.arange(len(current), device=current.device), r=3
    )
    candidates = current.unsqueeze(0).expand(2 * len(triples), -1).clone()
    for offset, direction in ((0, (1, 2, 0)), (len(triples), (2, 0, 1))):
        rows = torch.arange(len(triples), device=current.device) + offset
        for target, source in enumerate(direction):
            candidates[rows, triples[:, target]] = current[triples[:, source]]
    return candidates


def _double_swap_candidates(current: Tensor) -> Tensor:
    quadruples = torch.combinations(
        torch.arange(len(current), device=current.device), r=4
    )
    pairings = ((1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0))
    candidates = current.unsqueeze(0).expand(len(pairings) * len(quadruples), -1).clone()
    for pairing_index, sources in enumerate(pairings):
        rows = (
            torch.arange(len(quadruples), device=current.device)
            + pairing_index * len(quadruples)
        )
        for target, source in enumerate(sources):
            candidates[rows, quadruples[:, target]] = current[quadruples[:, source]]
    return candidates


def _best_candidate(
    candidates: Tensor,
    current: Tensor,
    current_objective: float,
    observed_transition: Tensor,
    canonical_transition: Tensor,
    row_weights: Tensor,
    objective: str,
    transition_counts: Tensor | None,
    epsilon: float,
) -> tuple[Tensor, float]:
    if objective == "count_nll":
        counts = (
            observed_transition.to(torch.float64)
            * row_weights.to(torch.float64).unsqueeze(1)
            if transition_counts is None
            else transition_counts.to(torch.float64)
        )
        objectives = _count_nll_candidate_objectives_from_delta(
            candidates,
            current,
            current_objective,
            counts,
            canonical_transition,
            epsilon,
        )
    else:
        objectives = _objective_values(
            candidates,
            observed_transition,
            canonical_transition,
            row_weights,
            objective,
            transition_counts,
            epsilon,
        )
    best = int(objectives.argmin())
    return candidates[best], float(objectives[best])


def _count_nll_candidate_objectives_from_delta(
    candidates: Tensor,
    current: Tensor,
    current_objective: float,
    transition_counts: Tensor,
    canonical_transition: Tensor,
    epsilon: float,
) -> Tensor:
    """Score fixed-size permutation moves from only their changed rows/columns."""
    candidates = candidates.to(torch.long)
    current = current.to(torch.long)
    changed_mask = candidates.ne(current.unsqueeze(0))
    changed_counts = changed_mask.sum(dim=1)
    if not bool((changed_counts == changed_counts[0]).all()) or int(changed_counts[0]) == 0:
        raise ValueError("delta scoring requires equally sized non-empty permutation moves")
    batch = candidates.shape[0]
    changed = changed_mask.nonzero(as_tuple=False)[:, 1].view(batch, -1)
    changed_semantic = candidates.gather(1, changed)
    pair_cost = -canonical_transition.to(torch.float64).clamp_min(epsilon).log()
    counts = transition_counts.to(torch.float64)

    # All edges leaving changed nodes, including edges within the changed set.
    outgoing_counts = counts[changed]
    old_outgoing_cost = pair_cost[
        current[changed].unsqueeze(2), current.view(1, 1, -1)
    ]
    new_outgoing_cost = pair_cost[
        changed_semantic.unsqueeze(2), candidates.unsqueeze(1)
    ]
    outgoing_delta = (
        outgoing_counts * (new_outgoing_cost - old_outgoing_cost)
    ).sum(dim=(1, 2))

    # Incoming edges from unchanged sources. Incoming edges from changed
    # sources were already included above, so the two terms never double count.
    incoming_counts = counts.t()[changed]
    old_incoming_cost = pair_cost[
        current.view(1, 1, -1), current[changed].unsqueeze(2)
    ]
    new_incoming_cost = pair_cost[
        candidates.unsqueeze(1), changed_semantic.unsqueeze(2)
    ]
    unchanged_sources = (~changed_mask).unsqueeze(1)
    incoming_delta = (
        incoming_counts
        * (new_incoming_cost - old_incoming_cost)
        * unchanged_sources
    ).sum(dim=(1, 2))
    return candidates.new_full(
        (batch,), current_objective, dtype=torch.float64
    ) + outgoing_delta + incoming_delta


def _pair_cycle_descent(
    current: Tensor,
    current_objective: float,
    observed_transition: Tensor,
    canonical_transition: Tensor,
    row_weights: Tensor,
    objective_name: str,
    transition_counts: Tensor | None,
    epsilon: float,
    max_iterations: int,
) -> tuple[Tensor, float, int, bool]:
    moves = 0
    for _ in range(max_iterations):
        swap, swap_objective = _best_candidate(
            _swap_candidates(current),
            current,
            current_objective,
            observed_transition,
            canonical_transition,
            row_weights,
            objective_name,
            transition_counts,
            epsilon,
        )
        tolerance = 1e-12 * max(1.0, abs(current_objective))
        if swap_objective < current_objective - tolerance:
            current = swap
            current_objective = swap_objective
            moves += 1
            continue
        if len(current) >= 3:
            cycle, cycle_objective = _best_candidate(
                _three_cycle_candidates(current),
                current,
                current_objective,
                observed_transition,
                canonical_transition,
                row_weights,
                objective_name,
                transition_counts,
                epsilon,
            )
            if cycle_objective < current_objective - tolerance:
                current = cycle
                current_objective = cycle_objective
                moves += 1
                continue
        return current, current_objective, moves, True
    return current, current_objective, moves, False


def oracle_transition_assignment(
    observed_transition: Tensor,
    canonical_transition: Tensor,
    observed_frequency: Tensor | None = None,
    row_weights: Tensor | None = None,
    max_iterations: int = 50,
    restarts: int = 4,
    seed: int = 0,
    objective: str = "count_nll",
    transition_counts: Tensor | None = None,
    epsilon: float = 1e-12,
) -> OracleMatchResult:
    """Approximate graph matching without enumerating the 19! permutations."""
    if observed_transition.shape != canonical_transition.shape:
        raise ValueError("observed and canonical transitions must have matching shape")
    if max_iterations < 1 or restarts < 1:
        raise ValueError("oracle iterations and restarts must be positive")
    if objective not in {"mse", "count_nll"}:
        raise ValueError("oracle objective must be mse or count_nll")
    objective_name = objective
    num_zones = observed_transition.shape[0]
    weights = (
        torch.ones(num_zones, dtype=torch.float64)
        if row_weights is None
        else row_weights.to(torch.float64)
    )
    starts = _initial_assignments(
        observed_transition,
        canonical_transition,
        observed_frequency,
        weights,
        objective_name,
        transition_counts,
        epsilon,
        restarts,
        seed,
    )

    best_assignment: list[int] | None = None
    best_objective = math.inf
    best_iterations = 0
    best_converged = False
    for start in starts:
        current = list(start)
        current_objective = matching_objective(
            observed_transition,
            canonical_transition,
            current,
            weights,
            objective=objective_name,
            transition_counts=transition_counts,
            epsilon=epsilon,
        )
        iterations = 0
        converged = False
        for _ in range(max_iterations):
            iterations += 1
            proposal = maximum_weight_assignment(
                -_structural_assignment_cost(
                    observed_transition,
                    canonical_transition,
                    current,
                    weights,
                    objective_name,
                    transition_counts,
                    epsilon,
                )
            )
            proposal_objective = matching_objective(
                observed_transition,
                canonical_transition,
                proposal,
                weights,
                objective=objective_name,
                transition_counts=transition_counts,
                epsilon=epsilon,
            )
            tolerance = 1e-12 * max(1.0, abs(current_objective))
            if (
                proposal == current
                or proposal_objective >= current_objective - tolerance
            ):
                break
            current = proposal
            current_objective = proposal_objective

        current_tensor, current_objective, local_moves, converged = (
            _pair_cycle_descent(
                torch.tensor(current, dtype=torch.long),
                current_objective,
                observed_transition,
                canonical_transition,
                weights,
                objective_name,
                transition_counts,
                epsilon,
                max_iterations,
            )
        )
        iterations += local_moves

        # Evaluate the larger 4-node neighborhood only after the cheaper
        # pair/3-cycle descent. Each successful escape is locally refined again.
        if num_zones >= 4:
            for _ in range(max_iterations):
                double_swap, double_swap_objective = _best_candidate(
                    _double_swap_candidates(current_tensor),
                    current_tensor,
                    current_objective,
                    observed_transition,
                    canonical_transition,
                    weights,
                    objective_name,
                    transition_counts,
                    epsilon,
                )
                tolerance = 1e-12 * max(1.0, abs(current_objective))
                if double_swap_objective >= current_objective - tolerance:
                    break
                current_tensor = double_swap
                current_objective = double_swap_objective
                iterations += 1
                current_tensor, current_objective, moves, converged = (
                    _pair_cycle_descent(
                        current_tensor,
                        current_objective,
                        observed_transition,
                        canonical_transition,
                        weights,
                        objective_name,
                        transition_counts,
                        epsilon,
                        max_iterations,
                    )
                )
                iterations += moves
            else:
                converged = False
        current = current_tensor.tolist()
        if current_objective < best_objective:
            best_assignment = current
            best_objective = current_objective
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
