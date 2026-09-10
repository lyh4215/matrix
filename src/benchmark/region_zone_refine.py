from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from ..data.controlled_synthetic import generate_controlled_benchmark
from .region_zone_local_search import LocalSearchConfig, evaluate_local_search
from .region_zone_match_probe import region_zone_probe_config_from_dict, _run_nonlearned
from .region_zone_complementarity import append_complementarity
from .region_zone_matching import build_graphs_by_length, canonical_identifiability
from .region_zone_reporting import write_region_zone_results


def refine_saved_results(
    source: str | Path,
    output_dir: str | Path,
    search: LocalSearchConfig | None = None,
    matchers: Sequence[str] = ("learned", "learned_structural"),
    compare_oracle: bool = False,
) -> dict:
    """Re-evaluate saved predictions without loading or training neural models."""
    source = Path(source).resolve()
    output_dir = Path(output_dir).resolve()
    if source.parent == output_dir:
        raise ValueError("use a separate output directory to preserve the original results")
    search = search or LocalSearchConfig(enabled=True)
    search.validate()
    if not matchers or set(matchers) - {"learned", "learned_structural"}:
        raise ValueError("select learned and/or learned_structural for local search")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload["config"].get("data_source", {}).get("kind") == "korean_corpus":
        raise ValueError("corpus results cannot be replayed with the synthetic generator; use korean_corpus_benchmark.py")
    config = region_zone_probe_config_from_dict(payload["config"])
    config.output_dir = str(output_dir)
    config.local_search = LocalSearchConfig(**{**asdict(search), "enabled": True})
    base_results = [row for row in payload["results"] if "local_search" not in row and "complementarity" not in row]
    selected = [row for row in base_results if row["matcher"] in matchers]
    if not selected:
        raise ValueError("source raw_results.json contains no selected learned predictions")
    if compare_oracle and not any(row["matcher"] == "learned_structural" for row in selected):
        raise ValueError("oracle comparison requires saved learned_structural predictions")
    # Reproduce the exact source benchmark, including split sizes and seed.
    # Changing these could change table identities in the generator.
    bundle = generate_controlled_benchmark(config.synthetic, config.seed)
    canonical = torch.tensor(bundle.transition_matrix, dtype=torch.float64)
    saved_canonical = torch.tensor(payload["canonical_transition_matrix"], dtype=torch.float64)
    if canonical.shape != saved_canonical.shape or not torch.allclose(
        canonical, saved_canonical, atol=1e-12, rtol=1e-12,
    ):
        raise ValueError("source canonical matrix differs from the regenerated benchmark")
    graphs = build_graphs_by_length(
        bundle.iid_test, bundle.table_zone_to_region, config.synthetic.num_zones,
        config.oracle.alpha, config.seed + 5003,
    )
    results = list(base_results)
    for base in selected:
        refined = evaluate_local_search(
            base, graphs[base["sequence_length"]], canonical, config.local_search, config.seed,
        )
        results.append(refined)
        print(json.dumps({
            "matcher": refined["matcher"],
            "sequence_length": refined["sequence_length"],
            "observed_assignment_accuracy": refined["observed_region_assignment_accuracy"],
            "token_accuracy": refined["token_accuracy"],
            "local_search": refined["local_search"],
        }), flush=True)
    if compare_oracle:
        lengths = {r["sequence_length"] for r in selected if r["matcher"] == "learned_structural"}
        comparison_graphs = {length: graphs[length] for length in sorted(lengths)}
        # Recompute oracle using exactly the same graphs, raw counts and epsilon.
        # Previously reported oracle objectives may have used MSE or another run.
        config.oracle.objective = "count_nll"
        config.compare_oracle = True
        if "oracle_transition" not in config.matchers:
            config.matchers = ("oracle_transition", *config.matchers)
        results = [r for r in results if not (r["matcher"] == "oracle_transition" and r["sequence_length"] in lengths)]
        print("Running oracle on the saved experiment's test tables (CPU, no training).", flush=True)
        results.extend(_run_nonlearned("oracle_transition", comparison_graphs, canonical, config))
        append_complementarity(results, comparison_graphs, canonical, config.oracle.epsilon)
    else:
        config.compare_oracle = False
    config_payload = {**config.to_dict(), "postprocess_source": str(source)}
    paths = write_region_zone_results(
        results, payload.get("learned_history", []), payload.get("learned_checkpoint"),
        canonical_identifiability(canonical), bundle.transition_matrix, config_payload,
        output_dir, structural_history=payload.get("learned_structural_history", []),
        structural_checkpoint=payload.get("learned_structural_checkpoint"),
    )
    return {"results": results, "paths": paths}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply count-NLL local search to saved learned predictions; no training or GPU needed",
    )
    parser.add_argument("--results", required=True, help="Source probe raw_results.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--matchers", nargs="+", choices=("learned", "learned_structural"),
                        default=("learned", "learned_structural"))
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--restarts", type=int, default=1)
    args = parser.parse_args()
    refine_saved_results(
        args.results, args.output_dir,
        LocalSearchConfig(enabled=True, max_iterations=args.max_iterations, restarts=args.restarts),
        args.matchers,
    )


if __name__ == "__main__":
    main()
