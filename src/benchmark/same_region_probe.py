from __future__ import annotations

import argparse
import json
import random
from functools import partial
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..data.controlled_synthetic import generate_controlled_benchmark
from ..data.dataset import CipherEpisode, CipherEpisodeDataset, collate_episodes
from ..models.decoder import NeuralCipherDecoder
from ..training.same_region import (
    balanced_same_region_loss,
    flatten_same_region_pairs,
    same_region_classification_metrics,
    same_region_pair_targets,
)
from ..training.train import move_batch, resolve_device
from .config import BenchmarkConfig, load_benchmark_config
from .probe_reporting import write_probe_results
from .sanity_overfit import print_environment


PROBE_MODELS = ("standard", "relational", "relational_gated")


class SameRegionProbeModel(nn.Module):
    """End-to-end encoder probe with no 19-way semantic-zone objective."""

    def __init__(self, config, model_name: str) -> None:
        super().__init__()
        if model_name not in PROBE_MODELS:
            raise ValueError(f"unsupported same-region probe model {model_name!r}")
        self.model_name = model_name
        self.decoder = NeuralCipherDecoder(config, model_name)
        for parameter in self.decoder.token_classifier.parameters():
            parameter.requires_grad_(False)
        self.pair_classifier = (
            nn.Sequential(
                nn.Linear(4 * config.d_model, config.same_region_gate_hidden_dim),
                nn.GELU(),
                nn.Linear(config.same_region_gate_hidden_dim, 1),
            )
            if model_name != "relational_gated"
            else None
        )

    def forward(self, batch: dict) -> Tensor:
        output = self.decoder(
            digits=batch["digits"],
            cipher_values=batch["cipher_values"],
            attention_mask=batch["attention_mask"],
            cipher_zone_ids=batch["cipher_zone_ids"],
        )
        if self.model_name == "relational_gated":
            assert output.same_region_logits
            return output.same_region_logits[-1]
        assert self.pair_classifier is not None
        hidden = output.hidden_states
        left = hidden.unsqueeze(2).expand(-1, -1, hidden.shape[1], -1)
        right = hidden.unsqueeze(1).expand(-1, hidden.shape[1], -1, -1)
        pair_features = torch.cat((left, right, left - right, (left - right).abs()), dim=-1)
        return self.pair_classifier(pair_features).squeeze(-1)


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


@torch.no_grad()
def evaluate_probe(
    model: SameRegionProbeModel,
    loader: DataLoader,
    device: torch.device,
    hard_negative_threshold: int,
    hard_positive_threshold: int,
) -> dict:
    model.eval()
    logits_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    distance_parts: list[Tensor] = []
    for original_batch in loader:
        batch = move_batch(original_batch, device)
        logits = model(batch)
        pair_logits, targets, distances = flatten_same_region_pairs(
            logits,
            batch["cipher_zone_ids"],
            batch["attention_mask"],
            batch["cipher_values"],
        )
        assert distances is not None
        logits_parts.append(pair_logits.cpu())
        target_parts.append(targets.cpu())
        distance_parts.append(distances.cpu())
    return same_region_classification_metrics(
        torch.cat(logits_parts),
        torch.cat(target_parts),
        torch.cat(distance_parts),
        hard_negative_threshold,
        hard_positive_threshold,
    )


def _train_probe_model(
    model_name: str,
    train_episodes: Sequence[CipherEpisode],
    validation_episodes: Sequence[CipherEpisode],
    test_episodes: Sequence[CipherEpisode],
    config: BenchmarkConfig,
    seed: int,
    device: torch.device,
    hard_negative_threshold: int,
    hard_positive_threshold: int,
) -> dict:
    _seed_everything(seed)
    model = SameRegionProbeModel(config.model, model_name).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    train_loader = _loader(train_episodes, config, shuffle=True, seed=seed)
    train_evaluation_loader = _loader(train_episodes, config)
    validation_loader = _loader(validation_episodes, config)
    test_loader = _loader(test_episodes, config)
    history: list[dict] = []
    best_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        loss_sum = 0.0
        batches = 0
        for original_batch in train_loader:
            batch = move_batch(original_batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = balanced_same_region_loss(
                logits, batch["cipher_zone_ids"], batch["attention_mask"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            batches += 1
        train_metrics = evaluate_probe(
            model,
            train_evaluation_loader,
            device,
            hard_negative_threshold,
            hard_positive_threshold,
        )
        validation_metrics = evaluate_probe(
            model,
            validation_loader,
            device,
            hard_negative_threshold,
            hard_positive_threshold,
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(batches, 1),
            **{f"train_{key}": value for key, value in train_metrics.items() if key != "distance_stratified"},
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
                if key != "distance_stratified"
            },
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "model": model_name,
                    "epoch": epoch,
                    "train_loss": record["train_loss"],
                    "train_f1": train_metrics["same_region_f1"],
                    "train_balanced_accuracy": train_metrics[
                        "same_region_balanced_accuracy"
                    ],
                    "validation_f1": validation_metrics["same_region_f1"],
                    "validation_balanced_accuracy": validation_metrics[
                        "same_region_balanced_accuracy"
                    ],
                    "validation_true_same_gate_mean": validation_metrics[
                        "true_same_gate_mean"
                    ],
                    "validation_true_different_gate_mean": validation_metrics[
                        "true_different_gate_mean"
                    ],
                }
            ),
            flush=True,
        )
        if validation_metrics["same_region_f1"] > best_f1:
            best_f1 = float(validation_metrics["same_region_f1"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    assert best_state is not None
    model.load_state_dict(best_state)
    final_metrics = {
        "train": evaluate_probe(
            model,
            train_evaluation_loader,
            device,
            hard_negative_threshold,
            hard_positive_threshold,
        ),
        "validation": evaluate_probe(
            model,
            validation_loader,
            device,
            hard_negative_threshold,
            hard_positive_threshold,
        ),
        "test": evaluate_probe(
            model,
            test_loader,
            device,
            hard_negative_threshold,
            hard_positive_threshold,
        ),
    }
    checkpoint_dir = Path(config.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}.pt"
    torch.save(
        {
            "model_state": best_state,
            "model": model_name,
            "model_config": config.to_dict()["model"],
            "seed": seed,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )
    return {
        "model": model_name,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "best_epoch": best_epoch,
        "best_validation_f1": best_f1,
        **final_metrics,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }


def _collect_distance_pairs(episodes: Sequence[CipherEpisode]) -> tuple[Tensor, Tensor]:
    distance_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    for episode in episodes:
        values = torch.tensor(episode.cipher_values)
        regions = torch.tensor(episode.cipher_zone_ids)
        mask = torch.ones(1, len(episode.cipher_values), dtype=torch.bool)
        targets, valid = same_region_pair_targets(regions.unsqueeze(0), mask)
        distances = (values.unsqueeze(1) - values.unsqueeze(0)).abs().unsqueeze(0)
        distance_parts.append(distances[valid])
        target_parts.append(targets[valid])
    return torch.cat(distance_parts), torch.cat(target_parts)


def _best_distance_threshold(distances: Tensor, targets: Tensor) -> tuple[int, float]:
    order = distances.argsort()
    sorted_distances = distances[order]
    sorted_targets = targets[order].to(torch.long)
    cumulative_positive = sorted_targets.cumsum(0)
    cumulative_negative = (~sorted_targets.bool()).to(torch.long).cumsum(0)
    boundary = torch.ones_like(sorted_distances, dtype=torch.bool)
    boundary[:-1] = sorted_distances[:-1] != sorted_distances[1:]
    indices = torch.nonzero(boundary, as_tuple=False).flatten()
    positive_total = int(sorted_targets.sum())
    negative_total = len(sorted_targets) - positive_total
    true_positive_rates = cumulative_positive[indices].to(torch.float64) / max(positive_total, 1)
    false_positive_rates = cumulative_negative[indices].to(torch.float64) / max(negative_total, 1)
    balanced = 0.5 * (true_positive_rates + 1.0 - false_positive_rates)
    best_index = int(indices[int(balanced.argmax())])
    return int(sorted_distances[best_index]), float(balanced.max())


def _distance_baseline(
    train: Sequence[CipherEpisode],
    validation: Sequence[CipherEpisode],
    test: Sequence[CipherEpisode],
    hard_negative_threshold: int,
    hard_positive_threshold: int,
) -> dict:
    pairs = {
        split: _collect_distance_pairs(episodes)
        for split, episodes in (("train", train), ("validation", validation), ("test", test))
    }
    threshold, selection_score = _best_distance_threshold(*pairs["validation"])
    metrics: dict[str, dict] = {}
    for split, (distances, targets) in pairs.items():
        logits = torch.where(distances <= threshold, 20.0, -20.0)
        metrics[split] = same_region_classification_metrics(
            logits,
            targets,
            distances,
            hard_negative_threshold,
            hard_positive_threshold,
        )
    return {
        "model": "distance_threshold",
        "threshold": threshold,
        "best_validation_balanced_accuracy": selection_score,
        **metrics,
    }


def run_same_region_probe(config: BenchmarkConfig) -> dict:
    if not config.models or set(config.models) - set(PROBE_MODELS):
        raise ValueError(f"probe models must be selected from {PROBE_MODELS}")
    if len(config.synthetic.sequence_lengths) != 1:
        raise ValueError("same-region probe requires exactly one sequence length")
    if len(config.seeds) != 1:
        raise ValueError("same-region probe requires exactly one seed")
    config.validate()
    device = resolve_device(config.training.device)
    print_environment(device, "Same-region probe")
    seed = int(config.seeds[0])
    bundle = generate_controlled_benchmark(config.synthetic, seed)
    hard_negative_threshold = config.synthetic.region_width
    hard_positive_threshold = max(config.synthetic.region_width // 2, 1)
    distance_baseline = _distance_baseline(
        bundle.train,
        bundle.validation,
        bundle.iid_test,
        hard_negative_threshold,
        hard_positive_threshold,
    )
    runs: list[dict] = []
    paths: dict[str, str] = {}
    for model_name in config.models:
        run = _train_probe_model(
            model_name,
            bundle.train,
            bundle.validation,
            bundle.iid_test,
            config,
            seed,
            device,
            hard_negative_threshold,
            hard_positive_threshold,
        )
        runs.append(run)
        paths = write_probe_results(
            runs,
            distance_baseline,
            config.to_dict(),
            config.output_dir,
        )
    return {"runs": runs, "distance_baseline": distance_baseline, "paths": paths}


def _apply_smoke_settings(config: BenchmarkConfig) -> None:
    config.synthetic.num_zones = 4
    config.synthetic.train_tables = 2
    config.synthetic.validation_tables = 1
    config.synthetic.test_tables = 1
    config.synthetic.ood_test_tables = 1
    config.synthetic.sequence_lengths = (8,)
    config.synthetic.sequences_per_length = 1
    config.synthetic.noise_levels = (config.synthetic.locality_noise,)
    config.synthetic.preferred_transitions = 2
    config.model.d_model = 8
    config.model.num_heads = 2
    config.model.num_layers = 1
    config.model.dim_feedforward = 16
    config.model.dropout = 0.0
    config.model.num_cipher_local_heads = 1
    config.model.num_sequence_heads = 1
    config.model.same_region_gate_hidden_dim = 8
    config.training.epochs = 1
    config.training.batch_size = 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe permutation-invariant same-region grouping on unseen cipher tables"
    )
    parser.add_argument("--config", default="configs/same_region_probe.yaml")
    parser.add_argument("--models", nargs="+", choices=PROBE_MODELS)
    parser.add_argument("--train-tables", type=int)
    parser.add_argument("--validation-tables", type=int)
    parser.add_argument("--test-tables", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    config = load_benchmark_config(args.config)
    if args.models:
        config.models = tuple(args.models)
    if args.train_tables is not None:
        config.synthetic.train_tables = args.train_tables
    if args.validation_tables is not None:
        config.synthetic.validation_tables = args.validation_tables
    if args.test_tables is not None:
        config.synthetic.test_tables = args.test_tables
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
    if args.smoke:
        _apply_smoke_settings(config)
    run_same_region_probe(config)


if __name__ == "__main__":
    main()
