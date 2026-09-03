from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def _without_table_results(result: dict) -> dict:
    return {
        key: value
        for key, value in result.items()
        if key not in {
            "table_results",
            "assignment_confusion_matrix",
            "token_confusion_matrix",
        }
    }


def _heuristic(results: Sequence[dict], num_zones: int) -> list[str]:
    observations: list[str] = []
    random_level = 1.0 / num_zones
    by_key = {
        (result["matcher"], result["sequence_length"]): result for result in results
    }
    for length in sorted({result["sequence_length"] for result in results}):
        oracle = by_key.get(("oracle_transition", length))
        learned = by_key.get(("learned", length))
        if oracle:
            accuracy = oracle["observed_region_assignment_accuracy"]
            if accuracy >= 0.8:
                observations.append(
                    f"At length {length}, oracle transition structure contains strong "
                    "permutation-recovery information."
                )
            elif accuracy <= random_level + 0.05:
                observations.append(
                    f"At length {length}, oracle recovery is near random; context may be "
                    "insufficient or the canonical transition graph ambiguous."
                )
            else:
                observations.append(
                    f"At length {length}, oracle recovery is partial; inspect coverage, "
                    "sampling noise, and signature ambiguity."
                )
            if (
                oracle.get("mean_predicted_objective", 0.0)
                > oracle.get("mean_true_permutation_objective", 0.0) + 1e-8
            ):
                observations.append(
                    f"At length {length}, the predicted objective is worse than the known "
                    "true-permutation objective, indicating oracle search optimization failure."
                )
        if oracle and learned:
            gap = (
                oracle["observed_region_assignment_accuracy"]
                - learned["observed_region_assignment_accuracy"]
            )
            if gap >= 0.1:
                observations.append(
                    f"At length {length}, the learned matcher trails oracle by {gap:.3f}; "
                    "the learned architecture or objective is a likely bottleneck."
                )
            elif min(
                oracle["observed_region_assignment_accuracy"],
                learned["observed_region_assignment_accuracy"],
            ) >= 0.8:
                observations.append(
                    f"At length {length}, both oracle and learned matching are strong; if "
                    "end-to-end unseen-f stays low, grouping-to-matching integration is suspect."
                )
    return observations or ["Not enough matcher results for an automatic interpretation."]


def _summary_markdown(payload: dict) -> str:
    lines = [
        "# Region-to-zone permutation matching probe",
        "",
        "| Matcher | Length | Coverage | Observed Assign | Full Assign | Token Acc | "
        "Observed Exact | Full Exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["results"]:
        lines.append(
            "| {matcher} | {sequence_length} | {mean_region_coverage:.4f} | "
            "{observed_region_assignment_accuracy:.4f} | {full_assignment_accuracy:.4f} | "
            "{token_accuracy:.4f} | {observed_exact_recovery_rate:.4f} | "
            "{full_exact_recovery_rate:.4f} |".format(**row)
        )
    lines.extend(("", "## Heuristic interpretation", ""))
    lines.extend(f"- {observation}" for observation in payload["heuristic_interpretation"])
    ambiguity = payload["canonical_identifiability"]
    lines.extend(("", "## Most ambiguous canonical zone pairs", ""))
    for item in ambiguity["most_ambiguous_signature_pairs"][:5]:
        lines.append(
            f"- zones {item['zones']}: signature distance "
            f"{item['signature_distance']:.6f}"
        )
    lines.extend(("", "These interpretations are heuristic, not causal conclusions.", ""))
    return "\n".join(lines)


def _plot_curves(results: Sequence[dict], output_dir: Path) -> None:
    if not results:
        return
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    colors = {
        "random": "#999999",
        "frequency": "#4C72B0",
        "oracle_transition": "#55A868",
        "learned": "#DD8452",
    }
    for metric, filename, ylabel in (
        (
            "observed_region_assignment_accuracy",
            "sequence_length_assignment_accuracy.png",
            "Observed-region assignment accuracy",
        ),
        ("token_accuracy", "sequence_length_token_accuracy.png", "Token accuracy"),
        ("mean_region_coverage", "coverage_curve.png", "Mean region coverage"),
    ):
        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        for matcher in colors:
            rows = sorted(
                (row for row in results if row["matcher"] == matcher),
                key=lambda row: row["sequence_length"],
            )
            if rows:
                axis.plot(
                    [row["sequence_length"] for row in rows],
                    [row[metric] for row in rows],
                    marker="o",
                    label=matcher,
                    color=colors[matcher],
                )
        axis.set_xlabel("Sequence length")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plots / filename, dpi=160)
        plt.close(figure)

    preferred = next(
        (
            row
            for matcher in ("oracle_transition", "learned", "frequency", "random")
            for row in results
            if row["matcher"] == matcher
            and row["sequence_length"]
            == max(item["sequence_length"] for item in results)
        ),
        results[0],
    )
    matrix = preferred["assignment_confusion_matrix"]
    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    axis.set_xlabel("Predicted semantic zone")
    axis.set_ylabel("True semantic zone")
    axis.set_title(
        f"{preferred['matcher']} assignment confusion, length "
        f"{preferred['sequence_length']}"
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(plots / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_region_zone_results(
    results: Sequence[dict],
    learned_history: Sequence[dict],
    learned_checkpoint: str | None,
    identifiability: dict,
    canonical_transition: Sequence[Sequence[float]],
    config: dict,
    output_dir: str | Path,
) -> dict[str, str | None]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_path = output_path / "raw_results.json"
    _write_json(
        raw_path,
        {
            "config": config,
            "canonical_transition_matrix": canonical_transition,
            "canonical_identifiability": identifiability,
            "results": list(results),
            "learned_history": list(learned_history),
            "learned_checkpoint": learned_checkpoint,
        },
    )
    summary_results = [_without_table_results(result) for result in results]
    summary_payload = {
        "results": summary_results,
        "canonical_identifiability": identifiability,
        "heuristic_interpretation": _heuristic(
            results, int(config["synthetic"]["num_zones"])
        ),
    }
    summary_path = output_path / "summary.json"
    markdown_path = output_path / "summary.md"
    _write_json(summary_path, summary_payload)
    markdown_path.write_text(_summary_markdown(summary_payload), encoding="utf-8")

    oracle_tables = [
        {
            "sequence_length": result["sequence_length"],
            "tables": result["table_results"],
        }
        for result in results
        if result["matcher"] == "oracle_transition"
    ]
    oracle_path = output_path / "oracle" / "table_results.json"
    if oracle_tables:
        _write_json(oracle_path, oracle_tables)
    history_path = output_path / "learned" / "history.json"
    if learned_history:
        _write_json(history_path, list(learned_history))
    _plot_curves(results, output_path)
    return {
        "raw_results": str(raw_path),
        "summary_json": str(summary_path),
        "summary_markdown": str(markdown_path),
        "oracle_table_results": str(oracle_path) if oracle_tables else None,
        "learned_history": str(history_path) if learned_history else None,
        "learned_checkpoint": learned_checkpoint,
        "plots": str(output_path / "plots"),
    }
