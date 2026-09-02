from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def _model_judgment(
    result: dict, baselines: dict, mode: str, success_threshold: float
) -> dict:
    train_accuracy = float(result["train"]["token_accuracy"])
    test_metrics = result.get("test")
    test_accuracy = float(test_metrics["token_accuracy"]) if test_metrics else None
    return {
        "train_above_majority": train_accuracy > baselines["train_majority_accuracy"],
        "test_above_majority": (
            test_accuracy > baselines["test_majority_accuracy"]
            if test_accuracy is not None and baselines["test_majority_accuracy"] is not None
            else None
        ),
        "tiny_memorization_success": (
            train_accuracy >= success_threshold if mode == "tiny_memorize" else None
        ),
        "fixed_f_success": (
            train_accuracy >= success_threshold
            and test_accuracy is not None
            and test_accuracy >= success_threshold
            if mode == "fixed_f"
            else None
        ),
    }


def diagnostic_conclusion(
    results: Sequence[dict], baselines: dict, mode: str, success_threshold: float
) -> dict:
    judgments = {
        result["model"]: _model_judgment(result, baselines, mode, success_threshold)
        for result in results
    }
    if len(results) < 2:
        conclusion = "Incomplete: both Standard and Relational runs are required."
    elif mode == "tiny_memorize":
        successes = [bool(item["tiny_memorization_success"]) for item in judgments.values()]
        conclusion = (
            "Heuristic: both pipelines can memorize the tiny fixed examples. If fixed-f succeeds "
            "but unseen-f fails, the failure is likely related to permutation inference/generalization."
            if all(successes)
            else "Heuristic: tiny memorization failed for at least one model; investigate the training "
            "pipeline, optimization, or model implementation before interpreting unseen-f results."
        )
    else:
        successes = [bool(item["fixed_f_success"]) for item in judgments.values()]
        conclusion = (
            "Heuristic: both fixed-f runs succeeded; the training pipeline appears functional, and "
            "unseen-f failure is likely related to permutation inference/generalization."
            if all(successes)
            else "Heuristic: fixed-f did not reach the success threshold for at least one model. Run "
            "--tiny-memorize to distinguish basic memorization/optimization failure from failure to "
            "generalize a fixed mapping to new plaintext sequences."
        )
    return {
        "heuristic": True,
        "success_threshold": success_threshold,
        "model_judgments": judgments,
        "conclusion": conclusion,
    }


def _summary_markdown(payload: dict) -> str:
    baselines = payload["baselines"]

    def format_baseline(value: float | None) -> str:
        return f"`{value:.5f}`" if value is not None else "—"

    lines = [
        "# Fixed-cipher overfit sanity benchmark",
        "",
        f"Mode: `{payload['mode']}`",
        "",
        "| Model | Train accuracy | Train > majority | Test accuracy | Test > majority | Best epoch |",
        "| --- | ---: | :---: | ---: | :---: | ---: |",
    ]
    judgments = payload["diagnostic"]["model_judgments"]
    for result in payload["models"]:
        judgment = judgments[result["model"]]
        test_accuracy = result["test"]["token_accuracy"] if result.get("test") else None
        lines.append(
            "| {model} | {train:.4f} | {train_check} | {test} | {test_check} | {epoch} |".format(
                model=result["model"],
                train=result["train"]["token_accuracy"],
                train_check="yes" if judgment["train_above_majority"] else "no",
                test=f"{test_accuracy:.4f}" if test_accuracy is not None else "—",
                test_check=(
                    "yes" if judgment["test_above_majority"] else "no"
                ) if judgment["test_above_majority"] is not None else "—",
                epoch=result["best_epoch"],
            )
        )
    lines.extend(
        (
            "",
            f"Random 19-way baseline: {format_baseline(baselines['random_19_way_accuracy'])}",
            "",
            f"Train majority baseline: {format_baseline(baselines['train_majority_accuracy'])}",
            "",
            "Validation majority baseline: "
            f"{format_baseline(baselines['validation_majority_accuracy'])}",
            "",
            f"Test majority baseline: {format_baseline(baselines['test_majority_accuracy'])}",
            "",
            "## Heuristic diagnostic",
            "",
            payload["diagnostic"]["conclusion"],
            "",
        )
    )
    return "\n".join(lines)


def _plot_histories(results: Sequence[dict], output_dir: Path) -> None:
    if not results:
        return
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    colors = {"standard": "#4C72B0", "relational": "#DD8452"}
    accuracy_figure, accuracy_axis = plt.subplots(figsize=(8, 5))
    loss_figure, loss_axis = plt.subplots(figsize=(8, 5))
    for result in results:
        model = result["model"]
        history = result["history"]
        epochs = [row["epoch"] for row in history]
        accuracy_axis.plot(
            epochs,
            [row["train_accuracy"] for row in history],
            color=colors[model],
            label=f"{model} train",
        )
        loss_axis.plot(
            epochs,
            [row["train_loss"] for row in history],
            color=colors[model],
            label=f"{model} train",
        )
        if history and history[0]["validation_accuracy"] is not None:
            accuracy_axis.plot(
                epochs,
                [row["validation_accuracy"] for row in history],
                color=colors[model],
                linestyle="--",
                label=f"{model} validation",
            )
            loss_axis.plot(
                epochs,
                [row["validation_loss"] for row in history],
                color=colors[model],
                linestyle="--",
                label=f"{model} validation",
            )
    for axis, ylabel, title in (
        (accuracy_axis, "Token accuracy", "Fixed-f training dynamics"),
        (loss_axis, "Cross-entropy loss", "Fixed-f loss curves"),
    ):
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
    accuracy_figure.tight_layout()
    loss_figure.tight_layout()
    accuracy_figure.savefig(output_dir / "training_curves.png", dpi=160)
    loss_figure.savefig(output_dir / "loss_curves.png", dpi=160)
    plt.close(accuracy_figure)
    plt.close(loss_figure)


def write_sanity_results(
    results: Sequence[dict],
    config: dict,
    baselines: dict,
    table_metadata: dict,
    output_dir: str | Path,
    mode: str,
    success_threshold: float,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for result in results:
        (output_path / f"{result['model']}_history.json").write_text(
            json.dumps(result["history"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output_path / "fixed_cipher_table.json").write_text(
        json.dumps(table_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summarized_models = [
        {key: value for key, value in result.items() if key != "history"}
        for result in results
    ]
    diagnostic = diagnostic_conclusion(results, baselines, mode, success_threshold)
    payload = {
        "mode": mode,
        "config": config,
        "baselines": baselines,
        "models": summarized_models,
        "diagnostic": diagnostic,
    }
    summary_json = output_path / "summary.json"
    temporary = output_path / ".summary.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(summary_json)
    summary_markdown = output_path / "summary.md"
    summary_markdown.write_text(_summary_markdown(payload), encoding="utf-8")
    _plot_histories(results, output_path)
    return {
        "summary_json": str(summary_json),
        "summary_markdown": str(summary_markdown),
        "training_curves": str(output_path / "training_curves.png"),
        "loss_curves": str(output_path / "loss_curves.png"),
    }
