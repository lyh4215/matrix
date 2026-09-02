from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


SPLITS = ("train", "validation", "iid", "ood")
MODELS = ("standard", "relational")


def run_key(run: dict) -> tuple[int, int, str]:
    return int(run["train_table_count"]), int(run["seed"]), str(run["model"])


def flatten_learning_curve_run(run: dict) -> dict[str, Any]:
    row: dict[str, Any] = {
        "train_table_count": run["train_table_count"],
        "seed": run["seed"],
        "model": run["model"],
        "parameter_count": run["parameter_count"],
        "best_epoch": run["best_epoch"],
        "final_epoch": run["final_epoch"],
        "best_training_loss": run["best_training_loss"],
        "final_training_loss": run["final_training_loss"],
        "best_validation_loss": run["best_validation_loss"],
        "final_validation_loss": run["final_validation_loss"],
    }
    for split in SPLITS:
        metrics = run[split]
        row[f"{split}_token_accuracy"] = metrics["token_accuracy"]
        row[f"{split}_zone_accuracy"] = metrics["zone_accuracy"]
        row[f"{split}_token_top_3_accuracy"] = metrics["token_top_3_accuracy"]
        row[f"{split}_token_top_5_accuracy"] = metrics["token_top_5_accuracy"]
        row[f"{split}_loss"] = metrics["supervised_loss"]
        row[f"{split}_prediction_entropy"] = metrics["prediction_entropy"]
        row[f"{split}_max_predicted_class_fraction"] = metrics[
            "max_predicted_class_fraction"
        ]
        row[f"{split}_majority_accuracy"] = run["baselines"][
            f"{split}_majority_accuracy"
        ]
    representation = run["iid_representation_statistics"]
    row["iid_same_zone_cosine"] = representation["same_true_zone_cosine_similarity"]
    row["iid_different_zone_cosine"] = representation[
        "different_true_zone_cosine_similarity"
    ]
    row["iid_representation_separation"] = representation["separation"]
    attention = run.get("attention_distance_statistics")
    row["cipher_near_far_attention_ratio"] = (
        attention.get("cipher_near_far_attention_ratio") if attention else None
    )
    row["cipher_local_near_far_attention_ratio"] = (
        attention.get("cipher_local_near_far_attention_ratio") if attention else None
    )
    return row


def tidy_accuracy_rows(runs: Sequence[dict]) -> list[dict[str, Any]]:
    return [
        {
            "train_tables": run["train_table_count"],
            "model": run["model"],
            "seed": run["seed"],
            "split": split,
            "accuracy": run[split]["token_accuracy"],
            "majority_accuracy": run["baselines"][f"{split}_majority_accuracy"],
            "random_accuracy": run["baselines"]["random_19_way_accuracy"],
        }
        for run in runs
        for split in SPLITS
    ]


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def _format(values: Sequence[float]) -> str:
    if not values:
        return "—"
    mean, std = _mean_std(values)
    return f"{mean:.4f} ± {std:.4f}"


def learning_curve_summary(runs: Sequence[dict], train_table_counts: Sequence[int]) -> str:
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    paired: dict[tuple[int, int, str], float] = {}
    for run in runs:
        count, seed, model = run_key(run)
        for split in ("iid", "ood", "train"):
            accuracy = float(run[split]["token_accuracy"])
            grouped[(count, model, split)].append(accuracy)
            paired[(count, seed, f"{model}_{split}")] = accuracy

    headers = (
        "Train f",
        "Standard IID",
        "Relational IID",
        "Rel. − Std. IID",
        "Standard OOD",
        "Relational OOD",
        "Rel. − Std. OOD",
    )
    lines = [
        "# Standard vs Relational learning curve",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for count in train_table_counts:
        differences: dict[str, list[float]] = {"iid": [], "ood": []}
        seeds = sorted({seed for table_count, seed, _name in paired if table_count == count})
        for seed in seeds:
            for split in ("iid", "ood"):
                standard = paired.get((count, seed, f"standard_{split}"))
                relational = paired.get((count, seed, f"relational_{split}"))
                if standard is not None and relational is not None:
                    differences[split].append(relational - standard)
        values = [
            str(count),
            _format(grouped[(count, "standard", "iid")]),
            _format(grouped[(count, "relational", "iid")]),
            _format(differences["iid"]),
            _format(grouped[(count, "standard", "ood")]),
            _format(grouped[(count, "relational", "ood")]),
            _format(differences["ood"]),
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(("", "All values are seed mean ± sample standard deviation.", ""))
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_learning_curves(runs: Sequence[dict], output_dir: Path) -> None:
    if not runs:
        return
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    colors = {"standard": "#4C72B0", "relational": "#DD8452"}
    for split in ("iid", "ood", "train"):
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        for model in MODELS:
            points: dict[int, list[float]] = defaultdict(list)
            for run in runs:
                if run["model"] == model:
                    points[int(run["train_table_count"])].append(
                        float(run[split]["token_accuracy"])
                    )
            if not points:
                continue
            x_values = sorted(points)
            means, deviations = zip(*(_mean_std(points[count]) for count in x_values))
            axis.plot(x_values, means, marker="o", label=model, color=colors[model])
            axis.fill_between(
                x_values,
                [mean - deviation for mean, deviation in zip(means, deviations)],
                [mean + deviation for mean, deviation in zip(means, deviations)],
                alpha=0.2,
                color=colors[model],
            )
        majority_values = [
            float(run["baselines"][f"{split}_majority_accuracy"]) for run in runs
        ]
        random_accuracy = float(runs[0]["baselines"]["random_19_way_accuracy"])
        axis.axhline(
            statistics.mean(majority_values),
            color="#777777",
            linestyle="--",
            label="mean majority baseline",
        )
        axis.axhline(
            random_accuracy,
            color="#999999",
            linestyle=":",
            label="random 19-way baseline",
        )
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Number of training cipher tables f")
        axis.set_ylabel(f"{split.upper()} token accuracy")
        axis.set_title(f"{split.upper()} learning curve (sequence length 128)")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"learning_curve_{split}.png", dpi=160)
        plt.close(figure)


def write_learning_curve_results(
    runs: Sequence[dict], config: dict, output_dir: str | Path
) -> dict[str, str]:
    """Persist recoverable raw data plus wide/tidy tables, summary, and plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_path = output_path / "raw_results.json"
    temporary_path = output_path / ".raw_results.json.tmp"
    temporary_path.write_text(
        json.dumps({"config": config, "runs": list(runs)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(raw_path)

    results_path = output_path / "results.csv"
    tidy_path = output_path / "learning_curve.csv"
    summary_path = output_path / "summary.md"
    _write_csv(results_path, [flatten_learning_curve_run(run) for run in runs])
    _write_csv(tidy_path, tidy_accuracy_rows(runs))
    summary_path.write_text(
        learning_curve_summary(runs, config["train_table_counts"]), encoding="utf-8"
    )
    _plot_learning_curves(runs, output_path)
    return {
        "raw_results": str(raw_path),
        "results_csv": str(results_path),
        "learning_curve_csv": str(tidy_path),
        "summary": str(summary_path),
    }


def load_completed_runs(path: str | Path, expected_config: dict) -> list[dict]:
    raw_path = Path(path)
    if not raw_path.exists():
        return []
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    if payload.get("config") != expected_config:
        raise ValueError(
            "resume configuration differs from raw_results.json; use the original options "
            "or choose another --output-dir"
        )
    return list(payload.get("runs", []))
