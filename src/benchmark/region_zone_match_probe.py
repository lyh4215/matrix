from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import yaml
from torch import Tensor

from ..data.controlled_synthetic import (
    ControlledSyntheticConfig,
    generate_controlled_benchmark,
)
from ..models.region_zone_matcher import (
    LearnedRegionZoneMatcher,
    observed_permutation_nll,
)
from ..training.assignment import maximum_weight_assignment
from ..training.train import resolve_device
from .region_zone_matching import (
    MATCHERS,
    AnonymousRegionGraph,
    aggregate_assignment_results,
    build_graphs_by_length,
    canonical_identifiability,
    frequency_assignment,
    graph_node_features,
    matching_objective,
    oracle_transition_assignment,
    random_assignment,
    score_region_assignment,
    stationary_distribution,
)
from .region_zone_reporting import write_region_zone_results
from .sanity_overfit import print_environment


@dataclass
class OracleMatcherConfig:
    alpha: float = 1e-3
    objective: str = "count_nll"
    epsilon: float = 1e-12
    max_iterations: int = 50
    restarts: int = 4

    def validate(self) -> None:
        if self.alpha < 0:
            raise ValueError("oracle alpha must be non-negative")
        if self.objective not in {"mse", "count_nll"}:
            raise ValueError("oracle objective must be mse or count_nll")
        if self.epsilon <= 0:
            raise ValueError("oracle epsilon must be positive")
        if self.max_iterations < 1 or self.restarts < 1:
            raise ValueError("oracle iterations and restarts must be positive")


@dataclass
class LearnedMatcherConfig:
    d_model: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    sinkhorn_iterations: int = 30
    sinkhorn_temperature: float = 1.0
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "auto"

    def validate(self) -> None:
        if min(
            self.d_model,
            self.num_layers,
            self.sinkhorn_iterations,
            self.epochs,
            self.batch_size,
        ) < 1:
            raise ValueError("learned matcher dimensions and training counts must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("learned matcher dropout must be in [0, 1)")
        if self.sinkhorn_temperature <= 0 or self.learning_rate <= 0:
            raise ValueError("temperature and learning rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight decay must be non-negative")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")


@dataclass
class RegionZoneProbeConfig:
    seed: int = 42
    matchers: tuple[str, ...] = MATCHERS
    output_dir: str = "artifacts/region_zone_match_probe"
    synthetic: ControlledSyntheticConfig = field(
        default_factory=lambda: ControlledSyntheticConfig(
            train_tables=200,
            validation_tables=50,
            test_tables=100,
            ood_test_tables=1,
            sequence_lengths=(128,),
            sequences_per_length=1,
            noise_levels=(0.1,),
        )
    )
    oracle: OracleMatcherConfig = field(default_factory=OracleMatcherConfig)
    learned: LearnedMatcherConfig = field(default_factory=LearnedMatcherConfig)

    def validate(self) -> None:
        if not self.matchers or set(self.matchers) - set(MATCHERS):
            raise ValueError(f"matchers must be selected from {MATCHERS}")
        if len(self.matchers) != len(set(self.matchers)):
            raise ValueError("matcher names must be unique")
        self.synthetic.validate()
        self.oracle.validate()
        self.learned.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type, values: Mapping[str, Any] | None):
    payload = dict(values or {})
    allowed = {item.name for item in fields(cls)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**payload)


def load_region_zone_probe_config(path: str | Path) -> RegionZoneProbeConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    synthetic_raw = dict(raw.get("synthetic") or {})
    for key in ("sequence_lengths", "noise_levels"):
        if key in synthetic_raw:
            synthetic_raw[key] = tuple(synthetic_raw[key])
    config = RegionZoneProbeConfig(
        seed=int(raw.get("seed", 42)),
        matchers=tuple(raw.get("matchers", MATCHERS)),
        output_dir=str(raw.get("output_dir", "artifacts/region_zone_match_probe")),
        synthetic=_construct(ControlledSyntheticConfig, synthetic_raw),
        oracle=_construct(OracleMatcherConfig, raw.get("oracle")),
        learned=_construct(LearnedMatcherConfig, raw.get("learned")),
    )
    config.validate()
    return config


def _row_weights(graph: AnonymousRegionGraph) -> Tensor:
    weights = graph.transition_counts.sum(dim=1)
    return weights if float(weights.sum()) > 0 else graph.observed_mask.to(torch.float64)


def _finish_assignment(
    observed_rows: Tensor,
    observed_assignment: Sequence[int],
    num_zones: int,
) -> tuple[int, ...]:
    result = [-1] * num_zones
    for row, semantic in zip(observed_rows.tolist(), observed_assignment):
        result[row] = int(semantic)
    remaining = [zone for zone in range(num_zones) if zone not in observed_assignment]
    for row, semantic in zip(
        [index for index, value in enumerate(result) if value < 0], remaining
    ):
        result[row] = semantic
    return tuple(result)


def _evaluate_graphs(
    matcher: str,
    graphs: Sequence[AnonymousRegionGraph],
    predictor: Callable[[AnonymousRegionGraph, int], tuple[Sequence[int], dict]],
    num_zones: int,
) -> dict:
    table_results: list[dict] = []
    for index, graph in enumerate(graphs):
        assignment, diagnostics = predictor(graph, index)
        scored = score_region_assignment(graph, assignment)
        scored.update(diagnostics)
        table_results.append(scored)
    aggregate = aggregate_assignment_results(table_results, num_zones)
    for result in table_results:
        # Confusions are retained once in the aggregate, not repeated for every
        # table in raw_results.json.
        result.pop("assignment_confusion")
        result.pop("token_confusion")
    objective_rows = [
        result for result in table_results if "predicted_objective" in result
    ]
    if objective_rows:
        gaps = [result["optimization_gap_to_true"] for result in objective_rows]
        aggregate.update(
            {
                "oracle_objective": objective_rows[0]["objective_name"],
                "mean_predicted_objective": sum(
                    result["predicted_objective"] for result in objective_rows
                )
                / len(objective_rows),
                "mean_true_permutation_objective": sum(
                    result["true_permutation_objective"] for result in objective_rows
                )
                / len(objective_rows),
                "oracle_convergence_rate": sum(
                    float(result["converged"]) for result in objective_rows
                )
                / len(objective_rows),
                "mean_oracle_iterations": sum(
                    result["iterations"] for result in objective_rows
                )
                / len(objective_rows),
                "fraction_predicted_objective_le_true_objective": sum(
                    result["predicted_objective"]
                    <= result["true_permutation_objective"] + 1e-9
                    for result in objective_rows
                )
                / len(objective_rows),
                "mean_optimization_gap": statistics.mean(gaps),
                "median_optimization_gap": statistics.median(gaps),
            }
        )
    return {
        "matcher": matcher,
        "sequence_length": graphs[0].sequence_length,
        **aggregate,
        "table_results": table_results,
    }


def _run_nonlearned(
    matcher: str,
    graphs_by_length: dict[int, list[AnonymousRegionGraph]],
    canonical_transition: Tensor,
    config: RegionZoneProbeConfig,
) -> list[dict]:
    canonical_frequency = stationary_distribution(canonical_transition)

    def predictor(graph: AnonymousRegionGraph, index: int) -> tuple[Sequence[int], dict]:
        if matcher == "random":
            return random_assignment(
                graph.num_zones,
                config.seed + graph.sequence_length * 100_003 + index,
            ), {}
        if matcher == "frequency":
            return frequency_assignment(
                graph.token_frequency, canonical_frequency
            ), {}
        if matcher == "oracle_transition":
            result = oracle_transition_assignment(
                graph.transition_matrix,
                canonical_transition,
                graph.token_frequency,
                _row_weights(graph),
                config.oracle.max_iterations,
                config.oracle.restarts,
                config.seed + graph.sequence_length * 100_003 + index,
                objective=config.oracle.objective,
                transition_counts=graph.transition_counts,
                epsilon=config.oracle.epsilon,
            )
            true_objective = matching_objective(
                graph.transition_matrix,
                canonical_transition,
                graph.true_zone_by_anonymous_region.tolist(),
                _row_weights(graph),
                objective=config.oracle.objective,
                transition_counts=graph.transition_counts,
                epsilon=config.oracle.epsilon,
            )
            true_aligned_mse = matching_objective(
                graph.transition_matrix,
                canonical_transition,
                graph.true_zone_by_anonymous_region.tolist(),
                _row_weights(graph),
                objective="mse",
            )
            return result.assignment, {
                "objective_name": config.oracle.objective,
                "predicted_objective": result.objective,
                "true_permutation_objective": true_objective,
                "true_aligned_transition_mse": true_aligned_mse,
                "iterations": result.iterations,
                "converged": result.converged,
                "optimization_gap_to_true": result.objective - true_objective,
            }
        raise ValueError(f"unsupported non-learned matcher {matcher!r}")

    return [
        _evaluate_graphs(matcher, graphs, predictor, config.synthetic.num_zones)
        for _length, graphs in sorted(graphs_by_length.items())
    ]


def _stack_graphs(
    graphs: Sequence[AnonymousRegionGraph], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.stack([graph_node_features(graph) for graph in graphs]).to(device),
        torch.stack([graph.transition_matrix for graph in graphs])
        .to(torch.float32)
        .to(device),
        torch.stack([graph.observed_mask for graph in graphs]).to(device),
        torch.stack([graph.true_zone_by_anonymous_region for graph in graphs]).to(device),
    )


def _learned_assignments(
    model: LearnedRegionZoneMatcher,
    graphs: Sequence[AnonymousRegionGraph],
    batch_size: int,
    device: torch.device,
) -> list[tuple[int, ...]]:
    model.eval()
    assignments: list[tuple[int, ...]] = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = graphs[start : start + batch_size]
            features, transition, observed, _targets = _stack_graphs(batch, device)
            probabilities = model(features, transition, observed).assignment_probabilities.cpu()
            for row, graph in enumerate(batch):
                observed_rows = graph.observed_mask.nonzero(as_tuple=False).flatten()
                partial = maximum_weight_assignment(
                    probabilities[row].index_select(0, observed_rows)
                )
                assignments.append(
                    _finish_assignment(observed_rows, partial, graph.num_zones)
                )
    return assignments


def _evaluate_learned_graphs(
    model: LearnedRegionZoneMatcher,
    graphs: Sequence[AnonymousRegionGraph],
    batch_size: int,
    device: torch.device,
) -> dict:
    assignments = _learned_assignments(model, graphs, batch_size, device)

    def predictor(_graph: AnonymousRegionGraph, index: int):
        return assignments[index], {}

    return _evaluate_graphs(
        "learned", graphs, predictor, model.num_zones
    )


def _train_learned_matcher(
    train_graphs_by_length: dict[int, list[AnonymousRegionGraph]],
    validation_graphs_by_length: dict[int, list[AnonymousRegionGraph]],
    test_graphs_by_length: dict[int, list[AnonymousRegionGraph]],
    config: RegionZoneProbeConfig,
    device: torch.device,
) -> tuple[list[dict], list[dict], str]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    train_graphs = [
        graph for _length, graphs in sorted(train_graphs_by_length.items()) for graph in graphs
    ]
    validation_graphs = [
        graph
        for _length, graphs in sorted(validation_graphs_by_length.items())
        for graph in graphs
    ]
    learned = config.learned
    model = LearnedRegionZoneMatcher(
        num_zones=config.synthetic.num_zones,
        d_model=learned.d_model,
        num_layers=learned.num_layers,
        dropout=learned.dropout,
        sinkhorn_iterations=learned.sinkhorn_iterations,
        sinkhorn_temperature=learned.sinkhorn_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learned.learning_rate, weight_decay=learned.weight_decay
    )
    history: list[dict] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    for epoch in range(1, learned.epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(config.seed + epoch)
        order = torch.randperm(len(train_graphs), generator=generator).tolist()
        losses: list[float] = []
        for start in range(0, len(order), learned.batch_size):
            batch = [train_graphs[index] for index in order[start : start + learned.batch_size]]
            features, transition, observed, targets = _stack_graphs(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(features, transition, observed)
            loss = observed_permutation_nll(
                output.assignment_probabilities, targets, observed
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = _evaluate_learned_graphs(
            model, validation_graphs, learned.batch_size, device
        )
        record = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(len(losses), 1),
            "validation_observed_assignment_accuracy": validation[
                "observed_region_assignment_accuracy"
            ],
            "validation_full_assignment_accuracy": validation[
                "full_assignment_accuracy"
            ],
            "validation_token_accuracy": validation["token_accuracy"],
            "validation_observed_exact_recovery_rate": validation[
                "observed_exact_recovery_rate"
            ],
        }
        history.append(record)
        print(json.dumps({"matcher": "learned", **record}), flush=True)
        accuracy = float(validation["observed_region_assignment_accuracy"])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    assert best_state is not None
    model.load_state_dict(best_state)
    checkpoint_path = Path(config.output_dir) / "learned" / "checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "model_config": asdict(config.learned),
            "num_zones": config.synthetic.num_zones,
            "best_epoch": best_epoch,
            "seed": config.seed,
        },
        checkpoint_path,
    )
    results = [
        _evaluate_learned_graphs(model, graphs, learned.batch_size, device)
        for _length, graphs in sorted(test_graphs_by_length.items())
    ]
    for result in results:
        result["best_epoch"] = best_epoch
        result["best_validation_observed_assignment_accuracy"] = best_accuracy
    return results, history, str(checkpoint_path)


def run_region_zone_match_probe(config: RegionZoneProbeConfig) -> dict:
    config.validate()
    device = resolve_device(config.learned.device)
    print_environment(device, "Region-zone matching probe")
    bundle = generate_controlled_benchmark(config.synthetic, config.seed)
    canonical_transition = torch.tensor(bundle.transition_matrix, dtype=torch.float64)
    graph_splits = {
        "train": build_graphs_by_length(
            bundle.train,
            bundle.table_zone_to_region,
            config.synthetic.num_zones,
            config.oracle.alpha,
            config.seed + 5001,
        ),
        "validation": build_graphs_by_length(
            bundle.validation,
            bundle.table_zone_to_region,
            config.synthetic.num_zones,
            config.oracle.alpha,
            config.seed + 5002,
        ),
        "test": build_graphs_by_length(
            bundle.iid_test,
            bundle.table_zone_to_region,
            config.synthetic.num_zones,
            config.oracle.alpha,
            config.seed + 5003,
        ),
    }
    results: list[dict] = []
    for matcher in config.matchers:
        if matcher == "learned":
            continue
        matcher_results = _run_nonlearned(
            matcher, graph_splits["test"], canonical_transition, config
        )
        results.extend(matcher_results)
        for result in matcher_results:
            print(
                json.dumps(
                    {
                        "matcher": matcher,
                        "sequence_length": result["sequence_length"],
                        "observed_assignment_accuracy": result[
                            "observed_region_assignment_accuracy"
                        ],
                        "token_accuracy": result["token_accuracy"],
                        "exact_recovery_rate": result[
                            "observed_exact_recovery_rate"
                        ],
                        "mean_region_coverage": result["mean_region_coverage"],
                    }
                ),
                flush=True,
            )
    learned_history: list[dict] = []
    learned_checkpoint: str | None = None
    if "learned" in config.matchers:
        learned_results, learned_history, learned_checkpoint = _train_learned_matcher(
            graph_splits["train"],
            graph_splits["validation"],
            graph_splits["test"],
            config,
            device,
        )
        results.extend(learned_results)
        for result in learned_results:
            print(
                json.dumps(
                    {
                        "matcher": "learned",
                        "sequence_length": result["sequence_length"],
                        "observed_assignment_accuracy": result[
                            "observed_region_assignment_accuracy"
                        ],
                        "token_accuracy": result["token_accuracy"],
                        "exact_recovery_rate": result[
                            "observed_exact_recovery_rate"
                        ],
                        "mean_region_coverage": result["mean_region_coverage"],
                    }
                ),
                flush=True,
            )
    identifiability = canonical_identifiability(canonical_transition)
    paths = write_region_zone_results(
        results,
        learned_history,
        learned_checkpoint,
        identifiability,
        bundle.transition_matrix,
        config.to_dict(),
        config.output_dir,
    )
    return {
        "results": results,
        "learned_history": learned_history,
        "identifiability": identifiability,
        "paths": paths,
    }


def _apply_smoke_settings(config: RegionZoneProbeConfig) -> None:
    config.matchers = MATCHERS
    config.synthetic.num_zones = 4
    config.synthetic.num_digits = 3
    config.synthetic.train_tables = 4
    config.synthetic.validation_tables = 2
    config.synthetic.test_tables = 2
    config.synthetic.ood_test_tables = 1
    config.synthetic.sequence_lengths = (16,)
    config.synthetic.sequences_per_length = 1
    config.synthetic.plaintext_symbols_per_zone = 8
    config.synthetic.region_width = 20
    config.synthetic.noise_levels = (config.synthetic.locality_noise,)
    config.synthetic.region_gap_min = 5
    config.synthetic.region_gap_max = 10
    config.synthetic.iid_value_min = 0
    config.synthetic.iid_value_max = 400
    config.synthetic.ood_value_min = 600
    config.synthetic.ood_value_max = 1000
    config.synthetic.preferred_transitions = 2
    config.oracle.max_iterations = 20
    config.oracle.restarts = 2
    config.learned.d_model = 8
    config.learned.num_layers = 1
    config.learned.dropout = 0.0
    config.learned.sinkhorn_iterations = 20
    config.learned.epochs = 1
    config.learned.batch_size = 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe anonymous cipher-region to semantic-zone permutation recovery"
    )
    parser.add_argument("--config", default="configs/region_zone_match_probe.yaml")
    parser.add_argument("--matchers", nargs="+", choices=MATCHERS)
    parser.add_argument("--train-tables", type=int)
    parser.add_argument("--validation-tables", type=int)
    parser.add_argument("--test-tables", type=int)
    parser.add_argument("--sequence-lengths", nargs="+", type=int)
    parser.add_argument("--sequences-per-table", type=int)
    parser.add_argument("--oracle-objective", choices=("mse", "count_nll"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_region_zone_probe_config(args.config)
    if args.matchers:
        config.matchers = tuple(args.matchers)
    if args.train_tables is not None:
        config.synthetic.train_tables = args.train_tables
    if args.validation_tables is not None:
        config.synthetic.validation_tables = args.validation_tables
    if args.test_tables is not None:
        config.synthetic.test_tables = args.test_tables
    if args.sequence_lengths:
        config.synthetic.sequence_lengths = tuple(args.sequence_lengths)
    if args.sequences_per_table is not None:
        config.synthetic.sequences_per_length = args.sequences_per_table
    if args.oracle_objective is not None:
        config.oracle.objective = args.oracle_objective
    if args.epochs is not None:
        config.learned.epochs = args.epochs
    if args.batch_size is not None:
        config.learned.batch_size = args.batch_size
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.learned.device = args.device
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.smoke:
        _apply_smoke_settings(config)
    run_region_zone_match_probe(config)


if __name__ == "__main__":
    main()
