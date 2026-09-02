from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


CONDITIONS = ("original", "translated")
MODELS = ("standard", "relational")


def _run_lookup(runs: Sequence[dict]) -> dict[tuple[str, str], dict]:
    return {(run["condition"], run["model"]): run for run in runs}


def translation_comparisons(runs: Sequence[dict]) -> dict[str, float | None]:
    lookup = _run_lookup(runs)

    def test_accuracy(condition: str, model: str) -> float | None:
        run = lookup.get((condition, model))
        return float(run["test"]["token_accuracy"]) if run and run.get("test") else None

    standard_original = test_accuracy("original", "standard")
    standard_translated = test_accuracy("translated", "standard")
    relational_original = test_accuracy("original", "relational")
    relational_translated = test_accuracy("translated", "relational")
    return {
        "standard_translation_drop": (
            standard_original - standard_translated
            if standard_original is not None and standard_translated is not None
            else None
        ),
        "relational_translation_drop": (
            relational_original - relational_translated
            if relational_original is not None and relational_translated is not None
            else None
        ),
        "translated_relational_minus_standard": (
            relational_translated - standard_translated
            if relational_translated is not None and standard_translated is not None
            else None
        ),
    }


def _heuristic_conclusion(runs: Sequence[dict], comparisons: dict, majority: float) -> str:
    lookup = _run_lookup(runs)
    if len(lookup) < 4:
        return "Incomplete heuristic: all four condition/model runs are needed for paired deltas."
    standard_translated = float(lookup[("translated", "standard")]["test"]["token_accuracy"])
    relational_translated = float(lookup[("translated", "relational")]["test"]["token_accuracy"])
    standard_drop = float(comparisons["standard_translation_drop"])
    relational_drop = float(comparisons["relational_translation_drop"])
    if standard_translated <= majority and relational_translated <= majority:
        return (
            "Heuristic: both translated models are at or below the test majority baseline; both may "
            "depend strongly on absolute numeric location."
        )
    if standard_drop >= 0.10 and standard_drop - relational_drop >= 0.10:
        return (
            "Heuristic: Standard degrades substantially more under translation, which is evidence "
            "that Relational uses translation-invariant structure more effectively."
        )
    if abs(standard_drop) <= 0.05 and abs(relational_drop) <= 0.05:
        return (
            "Heuristic: both models are largely robust to translation; fixed-f success is unlikely "
            "to come only from memorizing one absolute numeric range."
        )
    return (
        "Heuristic: translation effects are mixed. Use the paired drops, representation cosine, and "
        "prediction consistency as diagnostics rather than treating this as a causal conclusion."
    )


def _summary_rows(runs: Sequence[dict]) -> list[dict]:
    return [
        {
            "condition": run["condition"],
            "model": run["model"],
            "train_token_accuracy": run["train"]["token_accuracy"],
            "validation_token_accuracy": run["validation"]["token_accuracy"],
            "test_token_accuracy": run["test"]["token_accuracy"],
            "test_zone_accuracy": run["test"]["zone_accuracy"],
            "test_top_3_accuracy": run["test"]["token_top_3_accuracy"],
            "test_top_5_accuracy": run["test"]["token_top_5_accuracy"],
            "test_prediction_entropy": run["test"]["prediction_entropy"],
            "test_max_predicted_class_fraction": run["test"][
                "max_predicted_class_fraction"
            ],
            **run["translation_invariance_statistics"],
        }
        for run in runs
    ]


def _summary_markdown(payload: dict) -> str:
    rows = {(row["condition"], row["model"]): row for row in payload["results"]}
    lines = [
        "# Absolute-number translation ablation",
        "",
        "| Condition | Model | Train | Validation | Test |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        for model in MODELS:
            row = rows.get((condition, model))
            lines.append(
                f"| {condition.title()} | {model.title()} | "
                + (
                    f"{row['train_token_accuracy']:.4f} | "
                    f"{row['validation_token_accuracy']:.4f} | "
                    f"{row['test_token_accuracy']:.4f} |"
                    if row
                    else "— | — | — |"
                )
            )
    comparisons = payload["comparisons"]

    def metric(name: str) -> str:
        value = comparisons[name]
        return f"{value:.4f}" if value is not None else "—"

    lines.extend(
        (
            "",
            f"Standard translation drop: `{metric('standard_translation_drop')}`",
            "",
            f"Relational translation drop: `{metric('relational_translation_drop')}`",
            "",
            "Translated Relational − Standard: "
            f"`{metric('translated_relational_minus_standard')}`",
            "",
            "## Heuristic diagnostic",
            "",
            payload["heuristic_conclusion"],
            "",
        )
    )
    return "\n".join(lines)


def _plot_accuracy(runs: Sequence[dict], output_dir: Path) -> None:
    if not runs:
        return
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    colors = {"standard": "#4C72B0", "relational": "#DD8452"}
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        condition_runs = [run for run in runs if run["condition"] == condition]
        if not condition_runs:
            continue
        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        for run in condition_runs:
            history = run["history"]
            axis.plot(
                [row["epoch"] for row in history],
                [row["validation_accuracy"] for row in history],
                color=colors[run["model"]],
                label=run["model"],
            )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation token accuracy")
        axis.set_title(f"{condition.title()} fixed-f")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plots_dir / f"{condition}_accuracy.png", dpi=160)
        plt.close(figure)


def write_translation_results(
    runs: Sequence[dict],
    config: dict,
    table_metadata: dict,
    translation_metadata: dict,
    majority_baseline: float,
    output_dir: str | Path,
) -> dict[str, str]:
    output_path = Path(output_dir)
    histories_path = output_path / "histories"
    histories_path.mkdir(parents=True, exist_ok=True)
    for run in runs:
        (histories_path / f"{run['condition']}_{run['model']}.json").write_text(
            json.dumps(run["history"], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    raw_payload = {
        "config": config,
        "fixed_cipher_table": table_metadata,
        "translation_metadata": translation_metadata,
        "runs": list(runs),
    }
    raw_path = output_path / "raw_results.json"
    raw_temporary = output_path / ".raw_results.json.tmp"
    raw_temporary.write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    raw_temporary.replace(raw_path)

    comparisons = translation_comparisons(runs)
    summary_payload = {
        "results": _summary_rows(runs),
        "comparisons": comparisons,
        "translation_metadata": translation_metadata,
        "heuristic": True,
        "heuristic_conclusion": _heuristic_conclusion(
            runs, comparisons, majority_baseline
        ),
    }
    summary_path = output_path / "summary.json"
    summary_temporary = output_path / ".summary.json.tmp"
    summary_temporary.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_temporary.replace(summary_path)
    markdown_path = output_path / "summary.md"
    markdown_path.write_text(_summary_markdown(summary_payload), encoding="utf-8")
    _plot_accuracy(runs, output_path)
    return {
        "raw_results": str(raw_path),
        "summary_json": str(summary_path),
        "summary_markdown": str(markdown_path),
        "plots": str(output_path / "plots"),
    }
