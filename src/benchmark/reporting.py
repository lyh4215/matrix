from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

SUMMARY_FIELDS = (
    ("token_accuracy", "Token Accuracy"),
    ("zone_accuracy", "Zone Accuracy"),
    ("exact_mapping_accuracy", "Exact Mapping Accuracy"),
    ("top_3_accuracy", "Top-3 Accuracy"),
    ("iid_unseen_f_accuracy", "IID unseen-f Accuracy"),
    ("ood_relocated_accuracy", "OOD relocated Accuracy"),
)


def flatten_run(run: dict) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": run["seed"],
        "model": run["model"],
        "baseline": run["baseline"],
        "ablation_condition": run.get("ablation_condition", ""),
        "best_epoch": run["best_epoch"],
        "best_validation_token_accuracy": run["best_validation_token_accuracy"],
        "token_accuracy": run["iid"]["token_accuracy"],
        "zone_accuracy": run["iid"]["zone_accuracy"],
        "exact_mapping_accuracy": run["iid"]["exact_mapping_accuracy_per_f"],
        "top_3_accuracy": run["iid"]["token_top_3_accuracy"],
        "iid_unseen_f_accuracy": run["iid"]["unseen_f_accuracy"],
        "ood_relocated_accuracy": run["ood"]["token_accuracy"],
    }
    row.update(
        {f"length_{length}_accuracy": accuracy for length, accuracy in run["accuracy_by_sequence_length"].items()}
    )
    row.update(
        {f"noise_{noise}_accuracy": accuracy for noise, accuracy in run["accuracy_by_locality_noise"].items()}
    )
    return row


def aggregate_runs(runs: Sequence[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["model"]].append(flatten_run(run))
    aggregates: list[dict] = []
    for model, rows in grouped.items():
        item: dict[str, Any] = {"model": model, "num_seeds": len(rows)}
        numeric_keys = [
            key
            for key, value in rows[0].items()
            if key not in {"seed", "model", "baseline", "ablation_condition"}
            and isinstance(value, (int, float))
        ]
        for key in numeric_keys:
            values = [float(row[key]) for row in rows]
            item[f"{key}_mean"] = statistics.mean(values)
            item[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregates.append(item)
    return aggregates


def markdown_summary(aggregates: Sequence[dict]) -> str:
    headers = ["Model", *(label for _key, label in SUMMARY_FIELDS)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for item in aggregates:
        values = [item["model"]]
        for key, _label in SUMMARY_FIELDS:
            mean = item[f"{key}_mean"]
            std = item[f"{key}_std"]
            values.append(f"{mean:.4f} ± {std:.4f}")
        lines.append("| " + " | ".join(values) + " |")
    sections = ["\n".join(lines)]
    for prefix, title in (
        ("length_", "Token accuracy by sequence length"),
        ("noise_", "Token accuracy by locality noise"),
    ):
        labels = sorted(
            {
                key.removeprefix(prefix).removesuffix("_accuracy_mean")
                for item in aggregates
                for key in item
                if key.startswith(prefix) and key.endswith("_accuracy_mean")
            },
            key=float,
        )
        if not labels:
            continue
        section = [
            f"## {title}",
            "| Model | " + " | ".join(labels) + " |",
            "| " + " | ".join(["---"] * (len(labels) + 1)) + " |",
        ]
        for item in aggregates:
            values = [item["model"]]
            for label in labels:
                key = f"{prefix}{label}_accuracy"
                values.append(f"{item[f'{key}_mean']:.4f} ± {item[f'{key}_std']:.4f}")
            section.append("| " + " | ".join(values) + " |")
        sections.append("\n".join(section))
    return "\n\n".join(sections) + "\n"


def write_results(runs: Sequence[dict], config: dict, output_dir: str | Path) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    flattened = [flatten_run(run) for run in runs]
    aggregates = aggregate_runs(runs)
    json_path = output_path / "results.json"
    csv_path = output_path / "results.csv"
    summary_path = output_path / "summary.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"config": config, "runs": list(runs), "aggregates": aggregates}, handle, ensure_ascii=False, indent=2)
    fieldnames = list(dict.fromkeys(key for row in flattened for key in row))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)
    summary = markdown_summary(aggregates)
    summary_path.write_text(summary, encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "summary": str(summary_path), "table": summary}
