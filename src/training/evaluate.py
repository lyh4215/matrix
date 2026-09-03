from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from functools import partial
from typing import Hashable, Iterable

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..config import load_config
from ..data.dataset import CipherEpisodeDataset, collate_episodes
from ..models.decoder import DecoderOutput, NeuralCipherDecoder
from ..models.zone_pooling import pool_zone_targets
from .assignment import maximum_weight_assignment
from .same_region import SameRegionMetricAccumulator

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
    table_score_sums: dict[Hashable, dict[int, Tensor]] = defaultdict(dict)
    table_score_counts: dict[Hashable, dict[int, int]] = defaultdict(dict)
    table_region_labels: dict[Hashable, dict[int, int]] = defaultdict(dict)
    unseen_correct = unseen_total = 0
    length_correct = defaultdict(int)
    length_total = defaultdict(int)
    sinkhorn_row_sums: list[float] = []
    sinkhorn_column_maxima: list[float] = []
    sinkhorn_duplicate_rates: list[float] = []
    sinkhorn_entropies: list[float] = []
    sinkhorn_dummy_column_mass: list[float] = []
    supervised_loss_sum = 0.0
    supervised_loss_count = 0
    prediction_counts: Tensor | None = None
    same_region_metrics = SameRegionMetricAccumulator()

    for original_batch in loader:
        batch = _to_device(original_batch, device)
        output: DecoderOutput = model(
            digits=batch["digits"],
            cipher_values=batch["cipher_values"],
            attention_mask=batch["attention_mask"],
            cipher_zone_ids=batch["cipher_zone_ids"],
        )
        if output.same_region_logits:
            same_region_metrics.update(
                output.same_region_logits[-1],
                batch["cipher_zone_ids"],
                batch["attention_mask"],
            )
        mask = batch["attention_mask"]
        valid_scores = output.token_scores[mask]
        valid_targets = batch["zone_labels"][mask]
        predictions = valid_scores.argmax(dim=-1)
        if prediction_counts is None:
            prediction_counts = torch.zeros(valid_scores.shape[-1], dtype=torch.long)
        prediction_counts += torch.bincount(
            predictions.detach().cpu(), minlength=valid_scores.shape[-1]
        )
        token_correct += int((predictions == valid_targets).sum())
        token_total += valid_targets.numel()
        if output.pooled_zones is None:
            supervised_loss_sum += float(
                F.cross_entropy(valid_scores, valid_targets, reduction="sum")
            )
            supervised_loss_count += valid_targets.numel()
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
            for region_tensor in torch.unique(batch["cipher_zone_ids"][row, row_mask]):
                region = int(region_tensor)
                members = row_mask & (batch["cipher_zone_ids"][row] == region_tensor)
                score_sum = output.token_scores[row, members].detach().sum(dim=0).cpu()
                count = int(members.sum())
                labels = torch.unique(batch["zone_labels"][row, members])
                if labels.numel() != 1:
                    raise ValueError("inconsistent labels within cipher region")
                label = int(labels.item())
                if region in table_score_sums[table_id]:
                    table_score_sums[table_id][region] += score_sum
                    table_score_counts[table_id][region] += count
                    if table_region_labels[table_id][region] != label:
                        raise ValueError("cipher region maps to inconsistent labels across a table")
                else:
                    table_score_sums[table_id][region] = score_sum
                    table_score_counts[table_id][region] = count
                    table_region_labels[table_id][region] = label

        if output.pooled_zones is not None and output.zone_scores is not None:
            zone_targets = pool_zone_targets(
                batch["zone_labels"], batch["cipher_zone_ids"], output.pooled_zones, mask
            )
            valid_zone_scores = output.zone_scores[output.pooled_zones.mask]
            valid_zone_targets = zone_targets[output.pooled_zones.mask]
            loss_function = F.nll_loss if getattr(model, "uses_sinkhorn", False) else F.cross_entropy
            loss_scores = (
                valid_zone_scores.clamp_min(1e-8).log()
                if getattr(model, "uses_sinkhorn", False)
                else valid_zone_scores
            )
            supervised_loss_sum += float(
                loss_function(loss_scores, valid_zone_targets, reduction="sum")
            )
            supervised_loss_count += valid_zone_targets.numel()
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
                if getattr(model, "uses_sinkhorn", False):
                    probabilities = scores.clamp_min(1e-12)
                    row_sums = probabilities.sum(dim=-1)
                    column_mass = probabilities.sum(dim=0)
                    sinkhorn_row_sums.extend(row_sums.cpu().tolist())
                    sinkhorn_column_maxima.append(float(column_mass.max()))
                    argmax_columns = probabilities.argmax(dim=-1)
                    duplicate_count = len(argmax_columns) - len(torch.unique(argmax_columns))
                    sinkhorn_duplicate_rates.append(duplicate_count / max(len(argmax_columns), 1))
                    entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean()
                    sinkhorn_entropies.append(float(entropy))
                    sinkhorn_dummy_column_mass.extend((1.0 - column_mass).clamp_min(0.0).cpu().tolist())
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
    table_zone_correct_argmax = 0
    table_zone_correct_assignment = 0
    table_zone_total = 0
    table_exact_argmax: list[bool] = []
    table_exact_assignment: list[bool] = []
    for table_id, region_scores in table_score_sums.items():
        regions = sorted(region_scores)
        score_matrix = torch.stack(
            [region_scores[region] / table_score_counts[table_id][region] for region in regions]
        )
        targets = torch.tensor([table_region_labels[table_id][region] for region in regions])
        argmax_predictions = score_matrix.argmax(dim=-1)
        assignment_predictions = torch.tensor(maximum_weight_assignment(score_matrix))
        argmax_correct = argmax_predictions == targets
        assignment_correct = assignment_predictions == targets
        table_zone_correct_argmax += int(argmax_correct.sum())
        table_zone_correct_assignment += int(assignment_correct.sum())
        table_zone_total += len(regions)
        full_mapping_seen = len(regions) == score_matrix.shape[-1]
        table_exact_argmax.append(full_mapping_seen and bool(argmax_correct.all()))
        table_exact_assignment.append(full_mapping_seen and bool(assignment_correct.all()))

    metrics: dict[str, float | dict[str, float]] = {
        "token_accuracy": token_correct / token_total,
        "supervised_loss": supervised_loss_sum / max(supervised_loss_count, 1),
        "zone_accuracy": zone_correct / max(zone_total, 1),
        "exact_mapping_accuracy_per_episode": sum(episode_exact) / max(len(episode_exact), 1),
        "all_episodes_exact_per_f": sum(table_correct.values()) / max(len(table_correct), 1),
        # Deprecated compatibility alias; use all_episodes_exact_per_f for this old metric.
        "exact_mapping_accuracy_per_f": sum(table_correct.values()) / max(len(table_correct), 1),
        "table_zone_accuracy_argmax": table_zone_correct_argmax / max(table_zone_total, 1),
        "table_exact_mapping_accuracy_argmax": sum(table_exact_argmax) / max(len(table_exact_argmax), 1),
        "table_zone_accuracy_assignment": table_zone_correct_assignment / max(table_zone_total, 1),
        "table_exact_mapping_accuracy_assignment": sum(table_exact_assignment) / max(
            len(table_exact_assignment), 1
        ),
        "unseen_f_accuracy": unseen_correct / max(unseen_total, 1),
    }
    assert prediction_counts is not None
    prediction_probabilities = prediction_counts.to(torch.float64) / token_total
    nonzero_probabilities = prediction_probabilities[prediction_probabilities > 0]
    metrics["predicted_zone_distribution"] = {
        str(index): float(probability)
        for index, probability in enumerate(prediction_probabilities)
    }
    metrics["prediction_entropy"] = float(
        -(nonzero_probabilities * nonzero_probabilities.log()).sum()
    )
    metrics["max_predicted_class_fraction"] = float(prediction_probabilities.max())
    if same_region_metrics.count:
        metrics.update(same_region_metrics.compute())
    for k in (1, 3, 5):
        metrics[f"token_top_{k}_accuracy"] = token_top[k] / token_total
        metrics[f"zone_top_{k}_accuracy"] = zone_top[k] / max(zone_total, 1)
    metrics["accuracy_by_length"] = {
        name: length_correct[name] / length_total[name] if length_total[name] else 0.0
        for _lower, _upper, name in LENGTH_BUCKETS
    }
    if sinkhorn_row_sums:
        def mean_std(values: list[float]) -> tuple[float, float]:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            return mean, math.sqrt(variance)

        row_mean, row_std = mean_std(sinkhorn_row_sums)
        dummy_mean, dummy_std = mean_std(sinkhorn_dummy_column_mass)
        metrics["sinkhorn_diagnostics"] = {
            "real_row_sum_mean": row_mean,
            "real_row_sum_std": row_std,
            "real_column_max_mass": sum(sinkhorn_column_maxima) / len(sinkhorn_column_maxima),
            "duplicate_argmax_column_rate": sum(sinkhorn_duplicate_rates) / len(
                sinkhorn_duplicate_rates
            ),
            "assignment_entropy": sum(sinkhorn_entropies) / len(sinkhorn_entropies),
            "dummy_row_mass_distribution": {
                "mean": dummy_mean,
                "std": dummy_std,
                "min": min(sinkhorn_dummy_column_mass),
                "max": max(sinkhorn_dummy_column_mass),
            },
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
