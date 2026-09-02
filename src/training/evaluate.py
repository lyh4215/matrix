from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import partial
from typing import Hashable, Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..config import load_config
from ..data.dataset import CipherEpisodeDataset, collate_episodes
from ..models.decoder import DecoderOutput, NeuralCipherDecoder
from ..models.zone_pooling import pool_zone_targets

LENGTH_BUCKETS = ((1, 10, "1-10"), (11, 20, "11-20"), (21, 50, "21-50"), (51, None, "50+"))


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _correct_at_k(scores: Tensor, targets: Tensor, k: int) -> int:
    k = min(k, scores.shape[-1])
    return int((scores.topk(k, dim=-1).indices == targets.unsqueeze(-1)).any(dim=-1).sum())


def _length_bucket(length: int) -> str:
    for lower, upper, name in LENGTH_BUCKETS:
        if length >= lower and (upper is None or length <= upper):
            return name
    raise AssertionError("unreachable length bucket")


@torch.no_grad()
def evaluate_model(
    model: NeuralCipherDecoder,
    loader: Iterable[dict],
    device: str | torch.device = "cpu",
    train_table_ids: set[Hashable] | None = None,
) -> dict[str, float | dict[str, float]]:
    """Evaluate token, zone, full-mapping, top-k, unseen-f, and length metrics."""
    device = torch.device(device)
    model.eval()
    token_total = token_correct = 0
    token_top = defaultdict(int)
    zone_total = zone_correct = 0
    zone_top = defaultdict(int)
    episode_exact: list[bool] = []
    table_correct: dict[Hashable, bool] = {}
    unseen_correct = unseen_total = 0
    length_correct = defaultdict(int)
    length_total = defaultdict(int)

    for original_batch in loader:
        batch = _to_device(original_batch, device)
        output: DecoderOutput = model(
            digits=batch["digits"],
            cipher_values=batch["cipher_values"],
            attention_mask=batch["attention_mask"],
            cipher_zone_ids=batch["cipher_zone_ids"],
        )
        mask = batch["attention_mask"]
        valid_scores = output.token_scores[mask]
        valid_targets = batch["zone_labels"][mask]
        predictions = valid_scores.argmax(dim=-1)
        token_correct += int((predictions == valid_targets).sum())
        token_total += valid_targets.numel()
        for k in (1, 3, 5):
            token_top[k] += _correct_at_k(valid_scores, valid_targets, k)

        for row, table_id in enumerate(batch["table_ids"]):
            row_mask = mask[row]
            row_correct = output.token_scores[row, row_mask].argmax(-1) == batch["zone_labels"][row, row_mask]
            bucket = _length_bucket(int(batch["lengths"][row]))
            length_correct[bucket] += int(row_correct.sum())
            length_total[bucket] += row_correct.numel()
            is_unseen = train_table_ids is None or table_id not in train_table_ids
            if is_unseen:
                unseen_correct += int(row_correct.sum())
                unseen_total += row_correct.numel()

        if output.pooled_zones is not None and output.zone_scores is not None:
            zone_targets = pool_zone_targets(
                batch["zone_labels"], batch["cipher_zone_ids"], output.pooled_zones, mask
            )
            for row, table_id in enumerate(batch["table_ids"]):
                row_mask = output.pooled_zones.mask[row]
                scores = output.zone_scores[row, row_mask]
                targets = zone_targets[row, row_mask]
                correct = scores.argmax(-1) == targets
                zone_correct += int(correct.sum())
                zone_total += targets.numel()
                for k in (1, 3, 5):
                    zone_top[k] += _correct_at_k(scores, targets, k)
                exact = bool(correct.all())
                episode_exact.append(exact)
                table_correct[table_id] = table_correct.get(table_id, True) and exact
        else:
            # Derive an episode-local zone decision by averaging member token scores.
            for row, table_id in enumerate(batch["table_ids"]):
                predictions_ok: list[bool] = []
                for zone in torch.unique(batch["cipher_zone_ids"][row, mask[row]]):
                    members = mask[row] & (batch["cipher_zone_ids"][row] == zone)
                    score = output.token_scores[row, members].mean(dim=0, keepdim=True)
                    target_values = torch.unique(batch["zone_labels"][row, members])
                    if target_values.numel() != 1:
                        raise ValueError("inconsistent labels within cipher zone")
                    target = target_values[:1]
                    correct = bool(score.argmax(-1).eq(target).item())
                    predictions_ok.append(correct)
                    zone_correct += int(correct)
                    zone_total += 1
                    for k in (1, 3, 5):
                        zone_top[k] += _correct_at_k(score, target, k)
                exact = all(predictions_ok)
                episode_exact.append(exact)
                table_correct[table_id] = table_correct.get(table_id, True) and exact

    if token_total == 0:
        raise ValueError("evaluation loader contained no valid tokens")
    metrics: dict[str, float | dict[str, float]] = {
        "token_accuracy": token_correct / token_total,
        "zone_accuracy": zone_correct / max(zone_total, 1),
        "exact_mapping_accuracy_per_episode": sum(episode_exact) / max(len(episode_exact), 1),
        "exact_mapping_accuracy_per_f": sum(table_correct.values()) / max(len(table_correct), 1),
        "unseen_f_accuracy": unseen_correct / max(unseen_total, 1),
    }
    for k in (1, 3, 5):
        metrics[f"token_top_{k}_accuracy"] = token_top[k] / token_total
        metrics[f"zone_top_{k}_accuracy"] = zone_top[k] / max(zone_total, 1)
    metrics["accuracy_by_length"] = {
        name: length_correct[name] / length_total[name] if length_total[name] else 0.0
        for _lower, _upper, name in LENGTH_BUCKETS
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained cipher decoder")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--baseline")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.baseline:
        config.baseline = args.baseline
        config.validate()
    device_name = "cuda" if config.training.device == "auto" and torch.cuda.is_available() else (
        "cpu" if config.training.device == "auto" else config.training.device
    )
    device = torch.device(device_name)
    model = NeuralCipherDecoder(config.model, config.baseline).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
    dataset = CipherEpisodeDataset.from_jsonl(args.data_jsonl)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        collate_fn=partial(collate_episodes, num_digits=config.data.num_digits),
    )
    print(json.dumps(evaluate_model(model, loader, device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
