from __future__ import annotations

import argparse
import json

from ..data.controlled_synthetic import (
    generate_fixed_cipher_benchmark,
    translate_fixed_cipher_benchmark,
)
from ..training.train import resolve_device
from .config import BenchmarkConfig, load_benchmark_config
from .learning_curve import majority_accuracy
from .sanity_overfit import print_environment, train_fixed_cipher_model
from .translation_diagnostics import translation_invariance_statistics
from .translation_reporting import write_translation_results


DEFAULT_TRAIN_EPISODES = 256
DEFAULT_VALIDATION_EPISODES = 64
DEFAULT_TEST_EPISODES = 64


def run_translation_ablation(
    config: BenchmarkConfig,
    conditions: tuple[str, ...] = ("original", "translated"),
    train_episodes: int = DEFAULT_TRAIN_EPISODES,
    validation_episodes: int = DEFAULT_VALIDATION_EPISODES,
    test_episodes: int = DEFAULT_TEST_EPISODES,
) -> dict:
    if tuple(config.models) != ("standard", "relational") or config.include_ablations:
        raise ValueError("translation ablation supports exactly standard and relational")
    if not conditions or set(conditions) - {"original", "translated"}:
        raise ValueError("conditions must contain original and/or translated")
    if min(train_episodes, validation_episodes, test_episodes) < 1:
        raise ValueError("translation ablation requires non-empty train, validation, and test splits")
    if config.synthetic.sequence_lengths != (128,):
        raise ValueError("translation ablation requires sequence length 128")
    if len(config.seeds) != 1:
        raise ValueError("translation ablation requires exactly one seed")
    if config.training.lambda_local != 0.0 or config.training.lambda_entropy != 0.0:
        raise ValueError("translation ablation requires unregularized token cross-entropy")
    config.validate()
    device = resolve_device(config.training.device)
    print_environment(device)
    seed = int(config.seeds[0])

    original = generate_fixed_cipher_benchmark(
        config.synthetic,
        train_episodes,
        validation_episodes,
        test_episodes,
        seed,
    )
    translated = translate_fixed_cipher_benchmark(
        original,
        config.synthetic.num_digits,
        seed,
    )
    translation_metadata = translated.translation_metadata
    print(json.dumps({"translation_offsets": translation_metadata}, ensure_ascii=False), flush=True)
    condition_bundles = {
        "original": original,
        "translated": translated.bundle,
    }
    config_payload = {
        "benchmark": config.to_dict(),
        "conditions": list(conditions),
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "test_episodes": test_episodes,
    }
    runs: list[dict] = []
    paths: dict[str, str] = {}
    for condition in conditions:
        bundle = condition_bundles[condition]
        for model_name in config.models:
            run = train_fixed_cipher_model(
                model_name,
                bundle,
                config,
                seed,
                device,
                condition=condition,
            )
            run["translation_invariance_statistics"] = translation_invariance_statistics(
                # The diagnostic always starts from the same unshifted plaintext/cipher structure.
                # It therefore compares representation invariance independently of train condition.
                model=_load_trained_model(run, config, device),
                episode=original.test[0],
                num_digits=config.synthetic.num_digits,
                device=device,
            )
            runs.append(run)
            paths = write_translation_results(
                runs,
                config_payload,
                original.table_metadata,
                translation_metadata,
                majority_accuracy(original.test),
                config.output_dir,
            )
            print(
                json.dumps(
                    {
                        "completed": {"condition": condition, "model": model_name},
                        "train_accuracy": run["train"]["token_accuracy"],
                        "validation_accuracy": run["validation"]["token_accuracy"],
                        "test_accuracy": run["test"]["token_accuracy"],
                        "translation_invariance": run[
                            "translation_invariance_statistics"
                        ],
                        "saved": paths["raw_results"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return {"runs": runs, "translation_metadata": translation_metadata, "paths": paths}


def _load_trained_model(run: dict, config: BenchmarkConfig, device):
    """Load the selected checkpoint without changing the shared training utility contract."""
    import torch

    from ..models.decoder import NeuralCipherDecoder

    model = NeuralCipherDecoder(config.model, run["model"]).to(device)
    checkpoint = torch.load(run["checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return model


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
        description="Compare original and episode-translated fixed-f cipher benchmarks"
    )
    parser.add_argument("--config", default="configs/translation_ablation.yaml")
    parser.add_argument(
        "--condition", choices=("both", "original", "translated"), default="both"
    )
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--validation-episodes", type=int, default=DEFAULT_VALIDATION_EPISODES)
    parser.add_argument("--test-episodes", type=int, default=DEFAULT_TEST_EPISODES)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir")
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
    train_episodes = args.train_episodes
    validation_episodes = args.validation_episodes
    test_episodes = args.test_episodes
    if args.smoke:
        train_episodes, validation_episodes, test_episodes = 2, 1, 1
        _apply_smoke_settings(config)
    conditions = (
        ("original", "translated")
        if args.condition == "both"
        else (args.condition,)
    )
    run_translation_ablation(
        config,
        conditions,
        train_episodes,
        validation_episodes,
        test_episodes,
    )


if __name__ == "__main__":
    main()
