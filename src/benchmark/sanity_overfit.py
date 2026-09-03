from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from ..data.controlled_synthetic import (
    FixedCipherBenchmarkBundle,
    generate_fixed_cipher_benchmark,
)
from ..data.dataset import CipherEpisode, CipherEpisodeDataset, collate_episodes
from ..models.decoder import NeuralCipherDecoder
from ..training.evaluate import evaluate_model
from ..training.train import resolve_device, train_epoch
from .config import BenchmarkConfig, load_benchmark_config
from .learning_curve import majority_accuracy
from .sanity_reporting import write_sanity_results


DEFAULT_TRAIN_EPISODES = 256
DEFAULT_VALIDATION_EPISODES = 64
DEFAULT_TEST_EPISODES = 64
TINY_TRAIN_EPISODES = 8
TINY_EPOCHS = 200


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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate_optional(
    model: NeuralCipherDecoder,
    episodes: Sequence[CipherEpisode],
    config: BenchmarkConfig,
    device: torch.device,
    train_ids: set[str],
) -> dict | None:
    return (
        evaluate_model(model, _loader(episodes, config), device, train_ids)
        if episodes
        else None
    )


def train_fixed_cipher_model(
    model_name: str,
    bundle: FixedCipherBenchmarkBundle,
    config: BenchmarkConfig,
    seed: int,
    device: torch.device,
    condition: str | None = None,
) -> dict:
    _seed_everything(seed)
    model_config = copy.deepcopy(config.model)
    model = NeuralCipherDecoder(model_config, model_name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    train_loader = _loader(bundle.train, config, shuffle=True, seed=seed)
    train_eval_loader = _loader(bundle.train, config)
    validation_loader = _loader(bundle.validation, config) if bundle.validation else None
    train_ids = {str(episode.table_id) for episode in bundle.train}
    history: list[dict] = []
    first_step_diagnostics: dict | None = None
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.training.epochs + 1):
        training_output = train_epoch(
            model,
            train_loader,
            optimizer,
            config,
            device,
            capture_first_step_diagnostics=epoch == 1,
        )
        if epoch == 1:
            first_step_diagnostics = training_output["first_step_diagnostics"]
        train_metrics = evaluate_model(model, train_eval_loader, device, train_ids)
        validation_metrics = (
            evaluate_model(model, validation_loader, device, train_ids)
            if validation_loader is not None
            else None
        )
        record = {
            "epoch": epoch,
            "train_loss": training_output["loss"],
            "train_evaluation_loss": train_metrics["supervised_loss"],
            "train_accuracy": train_metrics["token_accuracy"],
            "validation_loss": (
                validation_metrics["supervised_loss"] if validation_metrics else None
            ),
            "validation_accuracy": (
                validation_metrics["token_accuracy"] if validation_metrics else None
            ),
            "train_prediction_entropy": train_metrics["prediction_entropy"],
            "validation_prediction_entropy": (
                validation_metrics["prediction_entropy"] if validation_metrics else None
            ),
            "train_max_predicted_class_fraction": train_metrics[
                "max_predicted_class_fraction"
            ],
            "validation_max_predicted_class_fraction": (
                validation_metrics["max_predicted_class_fraction"]
                if validation_metrics
                else None
            ),
        }
        history.append(record)
        output_record = {"model": model_name, **record}
        if condition is not None:
            output_record = {"condition": condition, **output_record}
        print(json.dumps(output_record, ensure_ascii=False), flush=True)
        selection_score = float(
            validation_metrics["token_accuracy"] if validation_metrics else train_metrics["token_accuracy"]
        )
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    assert first_step_diagnostics is not None and best_state is not None
    model.load_state_dict(best_state)
    final_metrics = {
        "train": evaluate_model(model, train_eval_loader, device, train_ids),
        "validation": _evaluate_optional(
            model, bundle.validation, config, device, train_ids
        ),
        "test": _evaluate_optional(model, bundle.test, config, device, train_ids),
    }
    checkpoint_dir = Path(config.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = f"{condition}_{model_name}.pt" if condition else f"{model_name}.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(
        {
            "model_state": best_state,
            "model_config": asdict(model_config),
            "model": model_name,
            "seed": seed,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )
    result = {
        "model": model_name,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "final_epoch": config.training.epochs,
        "selection_accuracy": best_score,
        "first_step_diagnostics": first_step_diagnostics,
        **final_metrics,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    if condition is not None:
        result["condition"] = condition
    return result


def print_environment(
    device: torch.device,
    benchmark_name: str = "Fixed-f sanity benchmark",
) -> None:
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"GPU: {torch.cuda.get_device_name(0) if cuda_available else 'None'}")
    print(f"Device: {device}")
    if not cuda_available:
        print(f"WARNING: CUDA is not available. {benchmark_name} may be slow.")


def run_sanity_overfit(
    config: BenchmarkConfig,
    train_episodes: int = DEFAULT_TRAIN_EPISODES,
    validation_episodes: int = DEFAULT_VALIDATION_EPISODES,
    test_episodes: int = DEFAULT_TEST_EPISODES,
    tiny_memorize: bool = False,
    success_threshold: float = 0.9,
) -> dict:
    if tuple(config.models) != ("standard", "relational") or config.include_ablations:
        raise ValueError("sanity benchmark supports exactly standard and relational without ablations")
    if config.training.lambda_local != 0.0 or config.training.lambda_entropy != 0.0:
        raise ValueError("sanity benchmark requires pure supervised loss with regularizers disabled")
    if config.synthetic.sequence_lengths != (128,):
        raise ValueError("sanity benchmark requires sequence length 128")
    if len(config.seeds) != 1:
        raise ValueError("sanity benchmark requires exactly one seed")
    if not 0.0 < success_threshold <= 1.0:
        raise ValueError("success threshold must be in (0, 1]")
    config.validate()
    device = resolve_device(config.training.device)
    print_environment(device)
    seed = int(config.seeds[0])
    bundle = generate_fixed_cipher_benchmark(
        config.synthetic,
        train_episodes,
        validation_episodes,
        test_episodes,
        seed,
    )
    baselines = {
        "random_19_way_accuracy": 1.0 / config.synthetic.num_zones,
        "train_majority_accuracy": majority_accuracy(bundle.train),
        "validation_majority_accuracy": (
            majority_accuracy(bundle.validation) if bundle.validation else None
        ),
        "test_majority_accuracy": majority_accuracy(bundle.test) if bundle.test else None,
    }
    print(json.dumps({"baselines": baselines}, ensure_ascii=False), flush=True)
    results: list[dict] = []
    mode = "tiny_memorize" if tiny_memorize else "fixed_f"
    payload_config = {
        "benchmark": config.to_dict(),
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "test_episodes": test_episodes,
        "mode": mode,
    }
    paths: dict[str, str] = {}
    for model_name in config.models:
        result = train_fixed_cipher_model(model_name, bundle, config, seed, device)
        results.append(result)
        paths = write_sanity_results(
            results,
            payload_config,
            baselines,
            bundle.table_metadata,
            config.output_dir,
            mode,
            success_threshold,
        )
        print(
            json.dumps(
                {
                    "completed": model_name,
                    "train_accuracy": result["train"]["token_accuracy"],
                    "test_accuracy": result["test"]["token_accuracy"] if result["test"] else None,
                    "gradient_update_diagnostics": result["first_step_diagnostics"],
                    "summary": paths["summary_json"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return {"models": results, "baselines": baselines, "paths": paths}


def _apply_smoke_settings(config: BenchmarkConfig) -> None:
    config.model.d_model = 8
    config.model.num_heads = 2
    config.model.num_layers = 1
    config.model.dim_feedforward = 16
    config.model.dropout = 0.0
    config.model.num_cipher_local_heads = 1
    config.model.num_sequence_heads = 1
    config.training.epochs = 1
    config.training.batch_size = 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overfit Standard and Relational attention on one fixed cipher function"
    )
    parser.add_argument("--config", default="configs/sanity_overfit.yaml")
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--validation-episodes", type=int, default=DEFAULT_VALIDATION_EPISODES)
    parser.add_argument("--test-episodes", type=int, default=DEFAULT_TEST_EPISODES)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir")
    parser.add_argument("--success-threshold", type=float, default=0.9)
    parser.add_argument("--tiny-memorize", action="store_true")
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    config = load_benchmark_config(args.config)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.seed is not None:
        config.seeds = (args.seed,)
    if args.device is not None:
        config.training.device = args.device
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    train_count = args.train_episodes
    validation_count = args.validation_episodes
    test_count = args.test_episodes
    if args.tiny_memorize:
        train_count = TINY_TRAIN_EPISODES
        validation_count = 0
        test_count = 0
        config.model.dropout = 0.0
        config.training.weight_decay = 0.0
        if args.epochs is None:
            config.training.epochs = TINY_EPOCHS
        if args.batch_size is None:
            config.training.batch_size = TINY_TRAIN_EPISODES
        if args.output_dir is None:
            config.output_dir = "artifacts/sanity_overfit_tiny"
    if args.smoke:
        train_count, validation_count, test_count = 2, 1, 1
        _apply_smoke_settings(config)
    run_sanity_overfit(
        config,
        train_count,
        validation_count,
        test_count,
        tiny_memorize=args.tiny_memorize,
        success_threshold=args.success_threshold,
    )


if __name__ == "__main__":
    main()
