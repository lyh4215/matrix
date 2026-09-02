from __future__ import annotations

import argparse
import json
import math
import random
from functools import partial
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..config import BASELINES, ExperimentConfig, load_config
from ..data.cipher_generator import (
    AugmentedDataset,
    CipherAugmenter,
    generate_synthetic_episodes,
)
from ..data.dataset import (
    CipherEpisodeDataset,
    collate_episodes,
    split_by_cipher_table,
    table_ids,
)
from ..models.decoder import NeuralCipherDecoder
from .evaluate import evaluate_model
from .losses import compute_loss


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def build_episode_splits(config: ExperimentConfig, data_jsonl: str | None = None):
    if data_jsonl:
        episodes = CipherEpisodeDataset.from_jsonl(data_jsonl).episodes
    else:
        episodes = generate_synthetic_episodes(
            num_tables=config.data.num_tables,
            sentences_per_table=config.data.sentences_per_table,
            min_length=config.data.min_length,
            max_length=config.data.max_length,
            num_zones=config.model.num_zones,
            num_digits=config.data.num_digits,
            seed=config.seed,
        )
    return split_by_cipher_table(
        episodes,
        train_fraction=config.data.train_fraction,
        validation_fraction=config.data.validation_fraction,
        seed=config.seed,
    )


def build_loaders(config: ExperimentConfig, data_jsonl: str | None = None):
    train_episodes, validation_episodes, test_episodes = build_episode_splits(config, data_jsonl)
    augmenter = CipherAugmenter(
        num_digits=config.data.num_digits,
        random_offset=config.data.random_offset,
        max_random_offset=config.data.max_random_offset,
        offset_mode=config.data.offset_mode,
        zone_relocation=config.data.zone_relocation,
        seed=config.seed + 1,
    )
    train_dataset = AugmentedDataset(train_episodes, augmenter)
    collate = partial(collate_episodes, num_digits=config.data.num_digits)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        CipherEpisodeDataset(validation_episodes),
        batch_size=config.training.batch_size,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        CipherEpisodeDataset(test_episodes),
        batch_size=config.training.batch_size,
        collate_fn=collate,
    )
    return train_loader, validation_loader, test_loader, table_ids(train_episodes)


def train_epoch(
    model: NeuralCipherDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    device: torch.device,
    capture_first_step_diagnostics: bool = False,
) -> dict:
    model.train()
    totals = {"loss": 0.0, "supervised_loss": 0.0, "local_loss": 0.0, "entropy_loss": 0.0}
    batches = 0
    first_step_diagnostics: dict[str, float | bool | int] | None = None

    def snapshot(module: torch.nn.Module) -> dict[str, Tensor]:
        return {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
        }

    def gradient_norm(module: torch.nn.Module) -> float:
        squared = sum(
            float(parameter.grad.detach().float().square().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        return math.sqrt(squared)

    def gradients_finite(module: torch.nn.Module) -> bool:
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        return bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)

    def parameter_delta(module: torch.nn.Module, before: dict[str, Tensor]) -> float:
        squared = sum(
            float((parameter.detach() - before[name]).float().square().sum())
            for name, parameter in module.named_parameters()
        )
        return math.sqrt(squared)

    for original_batch in loader:
        batch = move_batch(original_batch, device)
        optimizer.zero_grad(set_to_none=True)
        capture = capture_first_step_diagnostics and first_step_diagnostics is None
        encoder_before = snapshot(model.encoder) if capture else None
        classifier_before = snapshot(model.token_classifier) if capture else None
        output = model(
            digits=batch["digits"],
            cipher_values=batch["cipher_values"],
            attention_mask=batch["attention_mask"],
            cipher_zone_ids=batch["cipher_zone_ids"],
        )
        losses = compute_loss(
            output,
            batch,
            uses_sinkhorn=model.uses_sinkhorn,
            lambda_local=config.training.lambda_local,
            lambda_entropy=config.training.lambda_entropy,
            distance_scale=config.model.distance_clip,
        )
        losses.total.backward()
        if capture:
            first_step_diagnostics = {
                "total_gradient_norm": gradient_norm(model),
                "encoder_gradient_norm": gradient_norm(model.encoder),
                "token_classifier_gradient_norm": gradient_norm(model.token_classifier),
                "all_gradients_finite": gradients_finite(model),
                "encoder_gradients_finite": gradients_finite(model.encoder),
                "token_classifier_gradients_finite": gradients_finite(model.token_classifier),
                "encoder_parameters_with_gradient": sum(
                    parameter.grad is not None for parameter in model.encoder.parameters()
                ),
                "token_classifier_parameters_with_gradient": sum(
                    parameter.grad is not None for parameter in model.token_classifier.parameters()
                ),
            }
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if capture:
            assert first_step_diagnostics is not None
            assert encoder_before is not None and classifier_before is not None
            first_step_diagnostics["encoder_parameter_delta_norm"] = parameter_delta(
                model.encoder, encoder_before
            )
            first_step_diagnostics["token_classifier_parameter_delta_norm"] = parameter_delta(
                model.token_classifier, classifier_before
            )
        for key, value in losses.detached().items():
            totals[key] += value
        batches += 1
    result: dict = {key: value / max(batches, 1) for key, value in totals.items()}
    if capture_first_step_diagnostics:
        if first_step_diagnostics is None:
            raise ValueError("cannot capture first-step diagnostics from an empty loader")
        result["first_step_diagnostics"] = first_step_diagnostics
    return result


def run_training(config: ExperimentConfig, data_jsonl: str | None = None) -> dict:
    config.validate()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.training.device)
    train_loader, validation_loader, test_loader, train_ids = build_loaders(config, data_jsonl)
    model = NeuralCipherDecoder(config.model, config.baseline).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    best_accuracy = -1.0
    history: list[dict] = []
    checkpoint_path = Path(config.training.checkpoint_path)
    for epoch in range(1, config.training.epochs + 1):
        losses = train_epoch(model, train_loader, optimizer, config, device)
        validation = evaluate_model(model, validation_loader, device, train_ids)
        record = {"epoch": epoch, **losses, "validation": validation}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if float(validation["token_accuracy"]) > best_accuracy:
            best_accuracy = float(validation["token_accuracy"])
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "baseline": config.baseline,
                    "config": config.to_dict(),
                    "epoch": epoch,
                },
                checkpoint_path,
            )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate_model(model, test_loader, device, train_ids)
    result = {
        "baseline": config.baseline,
        "best_validation_token_accuracy": best_accuracy,
        "test": test_metrics,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, ensure_ascii=False, indent=2))
    return result


def apply_smoke_settings(config: ExperimentConfig) -> None:
    config.data.num_tables = 6
    config.data.sentences_per_table = 1
    config.data.min_length = 6
    config.data.max_length = 10
    config.model.d_model = 32
    config.model.num_layers = 1
    config.model.dim_feedforward = 64
    config.training.epochs = 1
    config.training.batch_size = 2
    config.training.checkpoint_path = "tmp/smoke_model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Hangul cipher-zone decoder")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--baseline", choices=sorted(BASELINES))
    parser.add_argument("--data-jsonl")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.baseline:
        config.baseline = args.baseline
    if args.smoke:
        apply_smoke_settings(config)
    run_training(config, args.data_jsonl)


if __name__ == "__main__":
    main()
