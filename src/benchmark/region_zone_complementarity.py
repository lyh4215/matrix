from __future__ import annotations

import statistics
from typing import Sequence

import torch
from torch import Tensor

from .region_zone_matching import (
    AnonymousRegionGraph, aggregate_assignment_results, score_region_assignment,
)


def select_by_count_nll(
    counts: Tensor, canonical: Tensor, oracle: Sequence[int], structural: Sequence[int],
    epsilon: float = 1e-12,
) -> tuple[tuple[int, ...], str, float, float]:
    """Label-free selection on one graph; exact NLL ties retain oracle."""
    if not 0 < epsilon < float("inf"):
        raise ValueError("epsilon must be finite and positive")
    counts = counts.detach().cpu().double()
    canonical = canonical.detach().cpu().double()
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or canonical.shape != counts.shape:
        raise ValueError("counts and canonical must be matching square matrices")
    if not torch.isfinite(counts).all() or bool((counts < 0).any()):
        raise ValueError("counts must be finite and non-negative")
    if not torch.isfinite(canonical).all() or bool((canonical < 0).any()):
        raise ValueError("canonical must be finite and non-negative")
    costs = -canonical.clamp_min(epsilon).log()

    def objective(candidate: Sequence[int]) -> float:
        if sorted(candidate) != list(range(len(counts))):
            raise ValueError("candidate must be a one-to-one permutation")
        index = torch.tensor(candidate, dtype=torch.long)
        return float((counts * costs[index[:, None], index[None, :]]).sum())

    oracle_nll, structural_nll = objective(oracle), objective(structural)
    if structural_nll < oracle_nll:
        return tuple(structural), "learned_structural_local_search", oracle_nll, structural_nll
    return tuple(oracle), "oracle_transition", oracle_nll, structural_nll


def evaluate_complementarity(
    oracle_result: dict, structural_result: dict, graphs: Sequence[AnonymousRegionGraph],
    canonical: Tensor, epsilon: float = 1e-12,
) -> dict:
    """Compare matched tables and score the deployable NLL selector.

    Truth is used only after selection, for accuracy/overlap and objective-gap
    diagnostics. Label-assisted best-of-two metrics are explicitly diagnostic.
    """
    if not graphs or len({g.table_id for g in graphs}) != len(graphs):
        raise ValueError("comparison requires non-empty unique tables")
    indexed = []
    for result in (oracle_result, structural_result):
        rows = result["table_results"]
        by_id = {row["table_id"]: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != {g.table_id for g in graphs}:
            raise ValueError("candidate results must contain exactly the same tables")
        for graph in graphs:
            row = by_id[graph.table_id]
            if row["sequence_length"] != graph.sequence_length or row["true_assignment"] != graph.true_zone_by_anonymous_region.tolist():
                raise ValueError("candidate results disagree with regenerated table identity")
        indexed.append(by_id)
    tables, oracle_scores, structural_scores = [], [], []
    for graph in graphs:
        left, right = [by_id[graph.table_id]["predicted_assignment"] for by_id in indexed]
        chosen, source, left_nll, right_nll = select_by_count_nll(
            graph.transition_counts, canonical, left, right, epsilon,
        )
        # All label-dependent operations are below the selector call.
        truth = graph.true_zone_by_anonymous_region.tolist()
        _, _, true_nll, _ = select_by_count_nll(graph.transition_counts, canonical, truth, truth, epsilon)
        left_score, right_score = score_region_assignment(graph, left), score_region_assignment(graph, right)
        oracle_scores.append(left_score)
        structural_scores.append(right_score)
        scored = score_region_assignment(graph, chosen)
        scored.update({
            "selected_source": source,
            "oracle_assignment": list(left), "structural_assignment": list(right),
            "oracle_count_nll": left_nll, "structural_count_nll": right_nll,
            "selected_count_nll": min(left_nll, right_nll), "true_count_nll": true_nll,
            "oracle_gap_to_true": left_nll - true_nll,
            "structural_gap_to_true": right_nll - true_nll,
            "selected_gap_to_true": min(left_nll, right_nll) - true_nll,
            "oracle_exact": left_score["observed_exact_recovery"],
            "structural_exact": right_score["observed_exact_recovery"],
            "same_observed_assignment": all(left[i] == right[i] for i in range(graph.num_zones) if graph.observed_mask[i]),
            "oracle_assignment_accuracy": left_score["observed_region_assignment_accuracy"],
            "structural_assignment_accuracy": right_score["observed_region_assignment_accuracy"],
            "label_assisted_best_assignment_accuracy": max(
                left_score["observed_region_assignment_accuracy"], right_score["observed_region_assignment_accuracy"],
            ),
            "label_assisted_best_token_accuracy": max(left_score["token_accuracy"], right_score["token_accuracy"]),
        })
        tables.append(scored)
    aggregate = aggregate_assignment_results(tables, graphs[0].num_zones)
    candidate_metrics = {
        "oracle_transition": aggregate_assignment_results(oracle_scores, graphs[0].num_zones),
        "learned_structural_local_search": aggregate_assignment_results(structural_scores, graphs[0].num_zones),
    }
    diagnostics = {
        "tie_policy": "exact NLL ties retain oracle_transition",
        "selected_oracle_tables": sum(t["selected_source"] == "oracle_transition" for t in tables),
        "selected_structural_tables": sum(t["selected_source"] != "oracle_transition" for t in tables),
        "equal_nll_tables": sum(t["oracle_count_nll"] == t["structural_count_nll"] for t in tables),
        "same_observed_assignment_tables": sum(t["same_observed_assignment"] for t in tables),
        "exact_overlap": {
            "both": sum(t["oracle_exact"] and t["structural_exact"] for t in tables),
            "oracle_only": sum(t["oracle_exact"] and not t["structural_exact"] for t in tables),
            "structural_only": sum(t["structural_exact"] and not t["oracle_exact"] for t in tables),
            "neither": sum(not t["oracle_exact"] and not t["structural_exact"] for t in tables),
        },
        "label_assisted_diagnostic_only": {
            "best_of_two_assignment_accuracy": sum(t["label_assisted_best_assignment_accuracy"] * t["num_observed_regions"] for t in tables) / sum(t["num_observed_regions"] for t in tables),
            "best_of_two_token_accuracy": statistics.mean(t["label_assisted_best_token_accuracy"] for t in tables),
            "either_exact_rate": statistics.mean(t["oracle_exact"] or t["structural_exact"] for t in tables),
            "selector_missed_better_assignment_tables": sum(t["observed_region_assignment_accuracy"] < t["label_assisted_best_assignment_accuracy"] for t in tables),
        },
        "objective_diagnostics": {},
        "paired_vs_candidates": {},
    }
    for prefix in ("oracle", "structural", "selected"):
        gaps = [t[f"{prefix}_gap_to_true"] for t in tables]
        diagnostics["objective_diagnostics"][prefix] = {
            "fraction_nll_le_true": statistics.mean(gap <= 1e-9 for gap in gaps),
            "mean_gap_to_true": statistics.mean(gaps),
            "median_gap_to_true": statistics.median(gaps),
        }
    for name, base in candidate_metrics.items():
        key = "oracle_assignment_accuracy" if name == "oracle_transition" else "structural_assignment_accuracy"
        diagnostics["paired_vs_candidates"][name] = {
            "assignment_delta": aggregate["observed_region_assignment_accuracy"] - base["observed_region_assignment_accuracy"],
            "token_delta": aggregate["token_accuracy"] - base["token_accuracy"],
            "tables_improved": sum(t["observed_region_assignment_accuracy"] > t[key] for t in tables),
            "tables_worsened": sum(t["observed_region_assignment_accuracy"] < t[key] for t in tables),
        }
    for table in tables:
        table.pop("assignment_confusion")
        table.pop("token_confusion")
    return {
        "matcher": "oracle_structural_nll_select", "sequence_length": graphs[0].sequence_length,
        **aggregate, "complementarity": diagnostics, "table_results": tables,
    }


def append_complementarity(results: list[dict], graphs: dict, canonical: Tensor, epsilon: float) -> None:
    by_key = {(r["matcher"], r["sequence_length"]): r for r in results}
    for length, batch in sorted(graphs.items()):
        results.append(evaluate_complementarity(
            by_key[("oracle_transition", length)],
            by_key[("learned_structural_local_search", length)], batch, canonical, epsilon,
        ))
