from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from ..data.controlled_synthetic import ControlledBenchmarkBundle, generate_controlled_benchmark
from ..data.dataset import CipherEpisode, CipherEpisodeDataset, collate_episodes
from ..models.decoder import NeuralCipherDecoder
from ..training.evaluate import evaluate_model
from ..training.train import resolve_device, train_epoch
from .attention_statistics import attention_distance_statistics
from .config import BenchmarkConfig, load_benchmark_config
from .learning_curve_reporting import (
    load_completed_runs,
    run_key,
    write_learning_curve_results,
)
from .representation_statistics import representation_zone_statistics


TRAIN_TABLE_COUNTS = (50, 100, 200, 400, 800, 1600)
QUICK_TRAIN_TABLE_COUNTS = (50, 200)
QUICK_SEEDS = (42,)


def _loader(
    episodes: Sequence[CipherEpisode],
    config: BenchmarkConfig,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        CipherEpisodeDataset(episodes),
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=partial(collate_episodes, num_digits=config.synthetic.num_digits),
    )


def select_nested_training_episodes(
    episodes: Sequence[CipherEpisode], train_table_count: int
) -> list[CipherEpisode]:
    """Select a deterministic table-prefix, making every smaller split a subset."""
    table_ids = sorted({str(episode.table_id) for episode in episodes})
    if train_table_count < 1 or train_table_count > len(table_ids):
        raise ValueError(
            f"requested {train_table_count} training tables from a bundle with {len(table_ids)}"
        )
    selected = set(table_ids[:train_table_count])
    return [episode for episode in episodes if str(episode.table_id) in selected]


def majority_accuracy(episodes: Sequence[CipherEpisode]) -> float:
    counts = Counter(label for episode in episodes for label in episode.zone_labels)
    if not counts:
        raise ValueError("cannot compute a majority baseline for an empty split")
    return max(counts.values()) / sum(counts.values())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cpu_state_dict(model: NeuralCipherDecoder) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _run_paths(output_dir: str | Path, train_tables: int, seed: int, model: str) -> dict[str, Path]:
    root = Path(output_dir)
    stem = f"train_{train_tables}_seed_{seed}_{model}"
    paths = {
        "checkpoint": root / "checkpoints" / f"{stem}.pt",
        "history": root / "training_history" / f"{stem}.json",
        "attention": root / "attention_statistics" / f"{stem}.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def _train_one(
    config: BenchmarkConfig,
    bundle: ControlledBenchmarkBundle,
    train_episodes: Sequence[CipherEpisode],
    train_table_count: int,
    model_name: str,
    seed: int,
    device: torch.device,
    max_representation_batches: int | None,
) -> dict:
    _seed_everything(seed)
    model_config = copy.deepcopy(config.model)
    model = NeuralCipherDecoder(model_config, model_name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    train_ids = {str(episode.table_id) for episode in train_episodes}
    train_loader = _loader(train_episodes, config, shuffle=True, seed=seed)
    validation_loader = _loader(bundle.validation, config)
    history: list[dict] = []
    best_accuracy = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.training.epochs + 1):
        training_losses = train_epoch(model, train_loader, optimizer, config, device)
        validation_metrics = evaluate_model(model, validation_loader, device, train_ids)
        record = {
            "epoch": epoch,
            **training_losses,
            "validation_loss": validation_metrics["supervised_loss"],
            "validation_token_accuracy": validation_metrics["token_accuracy"],
            "validation_zone_accuracy": validation_metrics["zone_accuracy"],
            "validation_prediction_entropy": validation_metrics["prediction_entropy"],
            "validation_max_predicted_class_fraction": validation_metrics[
                "max_predicted_class_fraction"
            ],
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "train_tables": train_table_count,
                    "seed": seed,
                    "model": model_name,
                    **record,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        accuracy = float(validation_metrics["token_accuracy"])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = _cpu_state_dict(model)

    assert best_state is not None
    model.load_state_dict(best_state)
    evaluation_loaders = {
        "train": _loader(train_episodes, config),
        "validation": validation_loader,
        "iid": _loader(bundle.iid_test, config),
        "ood": _loader(bundle.ood_test, config),
    }
    metrics = {
        split: evaluate_model(model, loader, device, train_ids)
        for split, loader in evaluation_loaders.items()
    }
    representation = representation_zone_statistics(
        model,
        evaluation_loaders["iid"],
        device,
        max_batches=max_representation_batches,
    )
    attention = (
        attention_distance_statistics(
            model,
            evaluation_loaders["iid"],
            device,
            max_batches=config.max_attention_batches,
        )
        if model_name == "relational"
        else None
    )

    paths = _run_paths(config.output_dir, train_table_count, seed, model_name)
    torch.save(
        {
            "model_state": best_state,
            "model_config": asdict(model_config),
            "model": model_name,
            "seed": seed,
            "train_table_count": train_table_count,
            "best_epoch": best_epoch,
        },
        paths["checkpoint"],
    )
    paths["history"].write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if attention is not None:
        paths["attention"].write_text(
            json.dumps(attention, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    best_record = history[best_epoch - 1]
    return {
        "train_table_count": train_table_count,
        "seed": seed,
        "model": model_name,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_config": asdict(model_config),
        "best_epoch": best_epoch,
        "final_epoch": config.training.epochs,
        "best_validation_token_accuracy": best_accuracy,
        "best_training_loss": best_record["loss"],
        "final_training_loss": history[-1]["loss"],
        "best_validation_loss": best_record["validation_loss"],
        "final_validation_loss": history[-1]["validation_loss"],
        "baselines": {
            "random_19_way_accuracy": 1.0 / config.synthetic.num_zones,
            "train_majority_accuracy": majority_accuracy(train_episodes),
            "validation_majority_accuracy": majority_accuracy(bundle.validation),
            "iid_majority_accuracy": majority_accuracy(bundle.iid_test),
            "ood_majority_accuracy": majority_accuracy(bundle.ood_test),
        },
        **metrics,
        "iid_representation_statistics": representation,
        "attention_distance_statistics": attention,
        "training_history": history,
        "checkpoint": str(paths["checkpoint"]),
        "history_file": str(paths["history"]),
        "attention_statistics_file": str(paths["attention"]) if attention is not None else None,
    }


def _print_environment(device: torch.device) -> None:
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None"
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"GPU: {gpu_name}")
    print(f"Device: {device}")
    if not cuda_available:
        print("WARNING: CUDA is not available. Learning-curve benchmark may be very slow.")


def _normalized_payload(
    config: BenchmarkConfig,
    train_table_counts: Sequence[int],
    max_representation_batches: int | None,
) -> dict:
    payload = {
        "train_table_counts": list(train_table_counts),
        "benchmark": config.to_dict(),
        "max_representation_batches": max_representation_batches,
    }
    return json.loads(json.dumps(payload))


def run_learning_curve(
    config: BenchmarkConfig,
    train_table_counts: Sequence[int] = TRAIN_TABLE_COUNTS,
    resume: bool = False,
    max_representation_batches: int | None = 4,
) -> dict:
    counts = tuple(train_table_counts)
    if not counts or tuple(sorted(set(counts))) != counts or min(counts) < 1:
        raise ValueError("train table counts must be positive, unique, and increasing")
    if tuple(config.models) != ("standard", "relational") or config.include_ablations:
        raise ValueError("learning curve supports exactly standard and relational without ablations")
    config.synthetic.train_tables = max(counts)
    config.validate()
    device = resolve_device(config.training.device)
    _print_environment(device)
    payload = _normalized_payload(config, counts, max_representation_batches)
    raw_path = Path(config.output_dir) / "raw_results.json"
    runs = load_completed_runs(raw_path, payload) if resume else []
    completed = {run_key(run) for run in runs}
    if completed:
        print(f"Resume: skipping {len(completed)} completed run(s).", flush=True)

    for seed in config.seeds:
        expected = {
            (count, int(seed), model) for count in counts for model in config.models
        }
        if expected <= completed:
            continue
        bundle = generate_controlled_benchmark(config.synthetic, int(seed))
        previous_ids: set[str] = set()
        for count in counts:
            train_episodes = select_nested_training_episodes(bundle.train, count)
            current_ids = {str(episode.table_id) for episode in train_episodes}
            if not previous_ids <= current_ids:
                raise AssertionError("training table subsets are not nested")
            previous_ids = current_ids
            for model_name in config.models:
                key = (count, int(seed), model_name)
                if key in completed:
                    print(f"Skip completed run: {key}", flush=True)
                    continue
                run = _train_one(
                    config,
                    bundle,
                    train_episodes,
                    count,
                    model_name,
                    int(seed),
                    device,
                    max_representation_batches,
                )
                runs.append(run)
                completed.add(key)
                paths = write_learning_curve_results(runs, payload, config.output_dir)
                print(
                    json.dumps(
                        {
                            "completed": key,
                            "train_accuracy": run["train"]["token_accuracy"],
                            "iid_accuracy": run["iid"]["token_accuracy"],
                            "ood_accuracy": run["ood"]["token_accuracy"],
                            "saved": paths["raw_results"],
                        }
                    ),
                    flush=True,
                )
    paths = write_learning_curve_results(runs, payload, config.output_dir)
    print(f"Completed {len(runs)} run(s). Results: {paths['raw_results']}")
    return {"runs": runs, "paths": paths, "config": payload}


def _apply_smoke_settings(config: BenchmarkConfig) -> None:
    config.seeds = (42,)
    config.synthetic.validation_tables = 2
    config.synthetic.test_tables = 2
    config.synthetic.ood_test_tables = 2
    config.synthetic.sequence_lengths = (128,)
    config.synthetic.sequences_per_length = 1
    config.synthetic.noise_levels = (config.synthetic.locality_noise,)
    config.model.d_model = 8
    config.model.num_heads = 2
    config.model.num_layers = 1
    config.model.dim_feedforward = 16
    config.model.dropout = 0.0
    config.model.num_cipher_local_heads = 1
    config.model.num_sequence_heads = 1
    config.training.epochs = 1
    config.training.batch_size = 2
    config.max_attention_batches = 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 128-token Standard vs Relational unseen-f learning curve"
    )
    parser.add_argument("--config", default="configs/learning_curve.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir")
    parser.add_argument("--train-table-counts", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-representation-batches", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    config = load_benchmark_config(args.config)
    counts = tuple(
        args.train_table_counts
        or (QUICK_TRAIN_TABLE_COUNTS if args.quick else TRAIN_TABLE_COUNTS)
    )
    if args.quick and args.seeds is None:
        config.seeds = QUICK_SEEDS
    if args.seeds is not None:
        config.seeds = tuple(args.seeds)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.device is not None:
        config.training.device = args.device
    if args.quick and args.output_dir is None:
        config.output_dir = f"{config.output_dir}_quick"
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.smoke:
        counts = (2,)
        _apply_smoke_settings(config)
    run_learning_curve(
        config,
        counts,
        resume=args.resume,
        max_representation_batches=(1 if args.smoke else args.max_representation_batches),
    )


if __name__ == "__main__":
    main()
