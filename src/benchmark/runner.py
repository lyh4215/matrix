from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from ..config import ModelConfig
from ..data.controlled_synthetic import ControlledBenchmarkBundle, generate_controlled_benchmark
from ..data.dataset import CipherEpisode, CipherEpisodeDataset, collate_episodes
from ..models.decoder import NeuralCipherDecoder
from ..training.evaluate import evaluate_model
from ..training.train import resolve_device, train_epoch
from .attention_statistics import attention_distance_statistics
from .config import BenchmarkConfig, load_benchmark_config
from .reporting import write_results


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    baseline: str
    ablation_condition: str = ""
    overrides: tuple[tuple[str, object], ...] = ()


def experiment_variants(config: BenchmarkConfig) -> list[ExperimentVariant]:
    variants = [
        ExperimentVariant(
            name=baseline,
            baseline=baseline,
            ablation_condition="A: absolute ON, relative ON" if baseline == "relational" else "",
        )
        for baseline in config.models
    ]
    if config.include_ablations:
        variants.extend(
            (
                ExperimentVariant(
                    "relational_abs_off_rel_on",
                    "relational",
                    "B: absolute OFF, relative ON",
                    (("use_absolute_digits", False), ("use_digit_delta", True), ("use_cipher_delta", True)),
                ),
                ExperimentVariant(
                    "relational_abs_on_rel_off",
                    "relational",
                    "C: absolute ON, relative OFF",
                    (
                        ("use_absolute_digits", True),
                        ("use_digit_delta", False),
                        ("use_cipher_delta", False),
                        ("use_locality_gate", False),
                        ("hard_local_radius", None),
                    ),
                ),
                ExperimentVariant(
                    "relational_abs_off_rel_off",
                    "relational",
                    "D: absolute OFF, relative OFF",
                    (
                        ("use_absolute_digits", False),
                        ("use_digit_delta", False),
                        ("use_cipher_delta", False),
                        ("use_locality_gate", False),
                        ("hard_local_radius", None),
                    ),
                ),
            )
        )
    names = [variant.name for variant in variants]
    if len(names) != len(set(names)):
        raise ValueError("benchmark variant names must be unique")
    return variants


def _model_config(base: ModelConfig, variant: ExperimentVariant) -> ModelConfig:
    result = copy.deepcopy(base)
    for key, value in variant.overrides:
        setattr(result, key, value)
    result.validate()
    return result


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


def _evaluate_episodes(
    model: NeuralCipherDecoder,
    episodes: Sequence[CipherEpisode],
    config: BenchmarkConfig,
    device: torch.device,
    train_table_ids: set[str],
) -> dict:
    return evaluate_model(model, _loader(episodes, config), device, train_table_ids)


def _train_variant(
    config: BenchmarkConfig,
    bundle: ControlledBenchmarkBundle,
    variant: ExperimentVariant,
    seed: int,
    device: torch.device,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model_config = _model_config(config.model, variant)
    model = NeuralCipherDecoder(model_config, variant.baseline).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    train_loader = _loader(bundle.train, config, shuffle=True, seed=seed)
    validation_loader = _loader(bundle.validation, config)
    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict] = []
    for epoch in range(1, config.training.epochs + 1):
        losses = train_epoch(model, train_loader, optimizer, config, device)
        validation = evaluate_model(model, validation_loader, device, bundle.train_table_ids)
        history.append({"epoch": epoch, **losses, "validation": validation})
        accuracy = float(validation["token_accuracy"])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)

    iid = _evaluate_episodes(model, bundle.iid_test, config, device, bundle.train_table_ids)
    ood = _evaluate_episodes(model, bundle.ood_test, config, device, bundle.train_table_ids)
    by_length = {
        str(length): float(
            _evaluate_episodes(model, episodes, config, device, bundle.train_table_ids)["token_accuracy"]
        )
        for length, episodes in bundle.iid_by_length.items()
    }
    noise_metrics = {
        str(noise): _evaluate_episodes(model, episodes, config, device, bundle.train_table_ids)
        for noise, episodes in bundle.iid_by_noise.items()
    }
    by_noise = {
        noise: float(metrics["token_accuracy"]) for noise, metrics in noise_metrics.items()
    }
    attention = None
    if model.is_relational:
        attention = attention_distance_statistics(
            model,
            _loader(bundle.iid_test, config),
            device,
            max_batches=config.max_attention_batches,
        )

    checkpoint_dir = Path(config.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{variant.name}_seed_{seed}.pt"
    torch.save(
        {
            "model_state": best_state,
            "model_config": asdict(model_config),
            "baseline": variant.baseline,
            "variant": variant.name,
            "seed": seed,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )
    return {
        "seed": seed,
        "model": variant.name,
        "baseline": variant.baseline,
        "ablation_condition": variant.ablation_condition,
        "model_config": asdict(model_config),
        "best_epoch": best_epoch,
        "best_validation_token_accuracy": best_accuracy,
        "iid": iid,
        "ood": ood,
        "accuracy_by_sequence_length": by_length,
        "accuracy_by_locality_noise": by_noise,
        "locality_noise_metrics": noise_metrics,
        "attention_distance_statistics": attention,
        "language_transition_matrix": bundle.transition_matrix,
        "training_history": history,
        "checkpoint": str(checkpoint_path),
    }


def run_benchmark(config: BenchmarkConfig) -> dict:
    config.validate()
    device = resolve_device(config.training.device)
    variants = experiment_variants(config)
    runs: list[dict] = []
    for seed in config.seeds:
        bundle = generate_controlled_benchmark(config.synthetic, seed)
        for variant in variants:
            run = _train_variant(config, bundle, variant, seed, device)
            runs.append(run)
            print(
                json.dumps(
                    {
                        "completed": variant.name,
                        "seed": seed,
                        "iid_token_accuracy": run["iid"]["token_accuracy"],
                        "ood_token_accuracy": run["ood"]["token_accuracy"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            # Keep long multi-seed studies recoverable after every model.
            write_results(runs, config.to_dict(), config.output_dir)
    paths = write_results(runs, config.to_dict(), config.output_dir)
    print(paths["table"])
    print(json.dumps({key: value for key, value in paths.items() if key != "table"}, indent=2))
    return {"runs": runs, "paths": paths}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled synthetic cipher benchmark")
    parser.add_argument("--config", default="configs/synthetic_benchmark.yaml")
    args = parser.parse_args()
    run_benchmark(load_benchmark_config(args.config))


if __name__ == "__main__":
    main()
