from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .region_zone_matching import (
    AnonymousRegionGraph,
    _pair_cycle_descent,
    _stable_seed,
    aggregate_assignment_results,
    score_region_assignment,
)


@dataclass
class LocalSearchConfig:
    enabled: bool = False
    max_iterations: int = 50
    restarts: int = 1
    epsilon: float = 1e-12

    def validate(self) -> None:
        for name in ("max_iterations", "restarts"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"local search {name} must be a positive integer")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("local search epsilon must be finite and positive")


@dataclass(frozen=True)
class LocalSearchResult:
    assignment: tuple[int, ...]
    initial_count_nll: float
    final_count_nll: float
    accepted_moves: int
    winning_restart: int
    starts_evaluated: int
    converged: bool
    seconds: float


def refine_count_assignment(
    transition_counts: Tensor,
    canonical_transition: Tensor,
    initial_assignment: Sequence[int],
    config: LocalSearchConfig | None = None,
    seed: int = 0,
) -> LocalSearchResult:
    """Refine a learned permutation with count-NLL pair swaps and 3-cycles.

    This inference-only CPU search reuses the oracle's delta-scored descent,
    but starts only from the supplied learned permutation (and optional small
    perturbations of it). It never reads labels, numeric values, or oracle
    initializations. The initial permutation is always a fallback candidate.
    """
    started = time.perf_counter()
    config = config or LocalSearchConfig()
    config.validate()
    counts = transition_counts.detach().to(device="cpu", dtype=torch.float64)
    canonical = canonical_transition.detach().to(device="cpu", dtype=torch.float64)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or counts.shape[0] < 2:
        raise ValueError("transition counts must be a square matrix with at least two zones")
    if canonical.shape != counts.shape:
        raise ValueError("canonical transition and counts must have matching shape")
    if not torch.isfinite(counts).all() or bool((counts < 0).any()):
        raise ValueError("transition counts must be finite and non-negative")
    if not torch.isfinite(canonical).all() or bool((canonical < 0).any()):
        raise ValueError("canonical transition must be finite and non-negative")
    if not torch.allclose(canonical.sum(1), torch.ones(counts.shape[0], dtype=counts.dtype)):
        raise ValueError("canonical transition rows must sum to one")
    initial = torch.as_tensor(initial_assignment)
    if initial.ndim != 1 or sorted(initial.tolist()) != list(range(counts.shape[0])):
        raise ValueError("initial assignment must be a complete one-to-one permutation")
    initial = initial.to(device="cpu", dtype=torch.long)
    pair_cost = -canonical.clamp_min(config.epsilon).log()

    def objective(assignment: Tensor) -> float:
        return float((counts * pair_cost[assignment[:, None], assignment[None, :]]).sum())

    initial_objective = objective(initial)
    best, best_objective = initial, initial_objective
    best_moves, best_restart, best_converged = 0, 0, False
    weights = counts.sum(1)
    transition = counts / weights.unsqueeze(1).clamp_min(1.0)
    starts = [initial]
    rng = random.Random(seed)
    for restart in range(1, config.restarts):
        candidate = initial.clone()
        for _ in range(1 + (restart - 1) % 3):
            left, right = rng.sample(range(len(initial)), 2)
            candidate[left], candidate[right] = candidate[right].item(), candidate[left].item()
        if not any(torch.equal(candidate, previous) for previous in starts):
            starts.append(candidate)
    for restart, start in enumerate(starts):
        candidate, _delta_objective, moves, converged = _pair_cycle_descent(
            start, objective(start), transition, canonical, weights,
            "count_nll", counts, config.epsilon, config.max_iterations,
        )
        # Recompute to avoid accepting an apparent gain from accumulated delta
        # roundoff. Strict improvement also preserves the input on empty graphs.
        candidate_objective = objective(candidate)
        if candidate_objective < best_objective:
            best, best_objective = candidate, candidate_objective
            best_moves, best_restart, best_converged = moves, restart, converged
        elif restart == 0 and torch.equal(candidate, initial):
            best_converged = converged
    return LocalSearchResult(
        tuple(best.tolist()), initial_objective, best_objective,
        best_moves, best_restart, len(starts), best_converged,
        time.perf_counter() - started,
    )


def evaluate_local_search(
    base_result: dict,
    graphs: Sequence[AnonymousRegionGraph],
    canonical_transition: Tensor,
    config: LocalSearchConfig,
    seed: int,
) -> dict:
    """Paired comparison on exactly the saved baseline predictions/tables."""
    base_tables = {row["table_id"]: row for row in base_result["table_results"]}
    if len(base_tables) != len(graphs) or set(base_tables) != {g.table_id for g in graphs}:
        raise ValueError("saved predictions do not match the regenerated test tables")
    table_results = []
    for graph in graphs:
        base = base_tables[graph.table_id]
        if base["sequence_length"] != graph.sequence_length:
            raise ValueError("saved prediction sequence length does not match the graph")
        # Identity check only: labels below are used for metrics, never search.
        if base["true_assignment"] != graph.true_zone_by_anonymous_region.tolist():
            raise ValueError("saved results and regenerated benchmark disagree; check config/seed")
        result = refine_count_assignment(
            graph.transition_counts, canonical_transition, base["predicted_assignment"],
            config, _stable_seed(seed, graph.table_id),
        )
        original = score_region_assignment(graph, base["predicted_assignment"])
        scored = score_region_assignment(graph, result.assignment)
        transitions = max(float(graph.transition_counts.sum()), 1.0)
        scored.update({
            "initial_assignment": base["predicted_assignment"],
            "initial_count_nll": result.initial_count_nll,
            "final_count_nll": result.final_count_nll,
            "count_nll_improvement_per_transition": (
                result.initial_count_nll - result.final_count_nll
            ) / transitions,
            "initial_observed_assignment_accuracy": original["observed_region_assignment_accuracy"],
            "assignment_accuracy_delta": (
                scored["observed_region_assignment_accuracy"]
                - original["observed_region_assignment_accuracy"]
            ),
            "token_accuracy_delta": scored["token_accuracy"] - original["token_accuracy"],
            "search_seconds": result.seconds,
            "search_accepted_moves": result.accepted_moves,
            "search_winning_restart": result.winning_restart,
            "search_starts_evaluated": result.starts_evaluated,
            "search_converged": result.converged,
        })
        table_results.append(scored)
    aggregate = aggregate_assignment_results(table_results, graphs[0].num_zones)
    search = {
        "base_matcher": base_result["matcher"],
        "assignment_accuracy_delta": (
            aggregate["observed_region_assignment_accuracy"]
            - base_result["observed_region_assignment_accuracy"]
        ),
        "token_accuracy_delta": aggregate["token_accuracy"] - base_result["token_accuracy"],
        "tables_assignment_improved": sum(row["assignment_accuracy_delta"] > 0 for row in table_results),
        "tables_assignment_worsened": sum(row["assignment_accuracy_delta"] < 0 for row in table_results),
        "tables_assignment_unchanged": sum(row["assignment_accuracy_delta"] == 0 for row in table_results),
        "mean_nll_improvement_per_transition": sum(
            row["count_nll_improvement_per_transition"] for row in table_results
        ) / len(table_results),
        "mean_seconds_per_table": sum(row["search_seconds"] for row in table_results) / len(table_results),
        "total_seconds": sum(row["search_seconds"] for row in table_results),
        "mean_accepted_moves": sum(row["search_accepted_moves"] for row in table_results) / len(table_results),
        "convergence_rate": sum(row["search_converged"] for row in table_results) / len(table_results),
    }
    for row in table_results:
        row.pop("assignment_confusion")
        row.pop("token_confusion")
    return {
        "matcher": base_result["matcher"] + "_local_search",
        "sequence_length": base_result["sequence_length"],
        **aggregate,
        "local_search": search,
        "table_results": table_results,
    }
