from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def _summary_row(run: dict) -> dict:
    metrics = run["test"]
    return {
        "model": run["model"],
        "pair_accuracy": metrics["same_region_pair_accuracy"],
        "balanced_accuracy": metrics["same_region_balanced_accuracy"],
        "precision": metrics["same_region_precision"],
        "recall": metrics["same_region_recall"],
        "f1": metrics["same_region_f1"],
        "same_class_accuracy": metrics["same_region_same_class_accuracy"],
        "different_class_accuracy": metrics["same_region_different_class_accuracy"],
        "hard_negative_accuracy": metrics["hard_negative_accuracy"],
        "hard_positive_accuracy": metrics["hard_positive_accuracy"],
        "true_same_gate_mean": metrics["true_same_gate_mean"],
        "true_different_gate_mean": metrics["true_different_gate_mean"],
    }


def _heuristic(rows: Sequence[dict]) -> str:
    lookup = {row["model"]: row for row in rows}
    gated = lookup.get("relational_gated")
    legacy = lookup.get("relational")
    distance = lookup.get("distance_threshold")
    observations: list[str] = []
    if gated and legacy:
        difference = gated["balanced_accuracy"] - legacy["balanced_accuracy"]
        if difference >= 0.05:
            observations.append(
                "Gated relational improves numeric grouping over legacy relational."
            )
        elif difference <= -0.05:
            observations.append(
                "Gated relational underperforms legacy relational on numeric grouping."
            )
        else:
            observations.append(
                "Gated and legacy relational grouping are within 0.05 balanced accuracy."
            )
    if distance:
        neural = [
            row["balanced_accuracy"]
            for row in rows
            if row["model"] != "distance_threshold"
        ]
        if neural and max(neural) <= distance["balanced_accuracy"] + 0.02:
            observations.append(
                "Neural probes do not clearly exceed the pure distance threshold; locality may "
                "explain most of this probe."
            )
    if gated and gated["balanced_accuracy"] >= 0.8:
        observations.append(
            "Numeric grouping is strong; if 19-way unseen-f remains low, semantic mapping is a "
            "likely bottleneck."
        )
    elif gated:
        observations.append("Front-end numeric region discovery remains unresolved for the gated probe.")
    return " ".join(observations) if observations else "Incomplete neural probe results."


def _summary_markdown(payload: dict) -> str:
    lines = [
        "# Same-region grouping probe",
        "",
        "| Model | Pair Acc | Balanced Acc | F1 | Hard Neg | Hard Pos | True Same p | True Diff p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["results"]:
        lines.append(
            "| {model} | {pair_accuracy:.4f} | {balanced_accuracy:.4f} | {f1:.4f} | "
            "{hard_negative_accuracy:.4f} | {hard_positive_accuracy:.4f} | "
            "{true_same_gate_mean:.4f} | {true_different_gate_mean:.4f} |".format(**row)
        )
    lines.extend(
        (
            "",
            f"Distance threshold selected on validation: `{payload['distance_threshold']}`",
            "",
            "## Heuristic interpretation",
            "",
            payload["heuristic_interpretation"],
            "",
        )
    )
    return "\n".join(lines)


def _plot_histories(runs: Sequence[dict], output_dir: Path) -> None:
    if not runs:
        return
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "standard": "#4C72B0",
        "relational": "#55A868",
        "relational_gated": "#DD8452",
    }
    for metric, filename, ylabel in (
        ("same_region_f1", "f1_curve.png", "Validation F1"),
        (
            "same_region_balanced_accuracy",
            "balanced_accuracy_curve.png",
            "Validation balanced accuracy",
        ),
    ):
        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        for run in runs:
            axis.plot(
                [row["epoch"] for row in run["history"]],
                [row[f"validation_{metric}"] for row in run["history"]],
                label=run["model"],
                color=colors[run["model"]],
            )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plots_dir / filename, dpi=160)
        plt.close(figure)

    gated = next((run for run in runs if run["model"] == "relational_gated"), None)
    if gated:
        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        epochs = [row["epoch"] for row in gated["history"]]
        axis.plot(
            epochs,
            [row["validation_true_same_gate_mean"] for row in gated["history"]],
            label="true same",
        )
        axis.plot(
            epochs,
            [row["validation_true_different_gate_mean"] for row in gated["history"]],
            label="true different",
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean predicted same-region probability")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plots_dir / "gate_separation.png", dpi=160)
        plt.close(figure)


def write_probe_results(
    runs: Sequence[dict],
    distance_baseline: dict,
    config: dict,
    output_dir: str | Path,
) -> dict[str, str]:
    output_path = Path(output_dir)
    histories_dir = output_path / "histories"
    histories_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        (histories_dir / f"{run['model']}.json").write_text(
            json.dumps(run["history"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    raw_payload = {
        "config": config,
        "runs": list(runs),
        "distance_threshold_baseline": distance_baseline,
    }
    raw_path = output_path / "raw_results.json"
    temporary = output_path / ".raw_results.json.tmp"
    temporary.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(raw_path)

    rows = [_summary_row(run) for run in runs]
    rows.append(_summary_row(distance_baseline))
    summary_payload = {
        "results": rows,
        "distance_threshold": distance_baseline["threshold"],
        "heuristic": True,
        "heuristic_interpretation": _heuristic(rows),
    }
    summary_path = output_path / "summary.json"
    summary_temporary = output_path / ".summary.json.tmp"
    summary_temporary.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_temporary.replace(summary_path)
    markdown_path = output_path / "summary.md"
    markdown_path.write_text(_summary_markdown(summary_payload), encoding="utf-8")
    _plot_histories(runs, output_path)
    return {
        "raw_results": str(raw_path),
        "summary_json": str(summary_path),
        "summary_markdown": str(markdown_path),
        "plots": str(output_path / "plots"),
    }
