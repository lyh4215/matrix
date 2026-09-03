from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor
from torch.nn import functional as F


DISTANCE_BUCKETS = (
    (0, 16, "0-16"),
    (17, 32, "17-32"),
    (33, 64, "33-64"),
    (65, 128, "65-128"),
    (129, 256, "129-256"),
    (257, None, "256+"),
)


class SameRegionMetricAccumulator:
    """Streaming pair metrics that avoid retaining O(dataset × length²) logits."""

    def __init__(self) -> None:
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0
        self.probability_sum = 0.0
        self.probability_square_sum = 0.0
        self.positive_probability_sum = 0.0
        self.negative_probability_sum = 0.0
        self.positive_loss_sum = 0.0
        self.negative_loss_sum = 0.0

    @property
    def count(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @torch.no_grad()
    def update(self, logits: Tensor, cipher_region_ids: Tensor, attention_mask: Tensor) -> None:
        pair_logits, targets, _distances = flatten_same_region_pairs(
            logits, cipher_region_ids, attention_mask
        )
        pair_logits = pair_logits.detach()
        targets = targets.detach()
        predictions = pair_logits >= 0
        positive = targets
        negative = ~targets
        probabilities = torch.sigmoid(pair_logits).to(torch.float64)
        self.true_positive += int((predictions & positive).sum())
        self.false_positive += int((predictions & negative).sum())
        self.true_negative += int((~predictions & negative).sum())
        self.false_negative += int((~predictions & positive).sum())
        self.probability_sum += float(probabilities.sum())
        self.probability_square_sum += float(probabilities.square().sum())
        self.positive_probability_sum += float(probabilities[positive].sum())
        self.negative_probability_sum += float(probabilities[negative].sum())
        self.positive_loss_sum += float(F.softplus(-pair_logits[positive]).sum())
        self.negative_loss_sum += float(F.softplus(pair_logits[negative]).sum())

    def compute(self) -> dict:
        if self.count == 0:
            raise ValueError("same-region metric accumulator is empty")
        positive_count = self.true_positive + self.false_negative
        negative_count = self.true_negative + self.false_positive
        precision = self.true_positive / max(self.true_positive + self.false_positive, 1)
        recall = self.true_positive / max(positive_count, 1)
        different_accuracy = self.true_negative / max(negative_count, 1)
        probability_mean = self.probability_sum / self.count
        probability_variance = max(
            self.probability_square_sum / self.count - probability_mean**2, 0.0
        )
        class_losses = []
        if positive_count:
            class_losses.append(self.positive_loss_sum / positive_count)
        if negative_count:
            class_losses.append(self.negative_loss_sum / negative_count)
        return {
            "same_region_pair_accuracy": (
                self.true_positive + self.true_negative
            ) / self.count,
            "same_region_precision": precision,
            "same_region_recall": recall,
            "same_region_f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "same_region_balanced_accuracy": 0.5 * (recall + different_accuracy),
            "same_region_same_class_accuracy": recall,
            "same_region_different_class_accuracy": different_accuracy,
            "same_region_positive_rate": positive_count / self.count,
            "same_region_balanced_bce": sum(class_losses) / len(class_losses),
            "predicted_same_region_rate": (
                self.true_positive + self.false_positive
            ) / self.count,
            "same_region_gate_probability_mean": probability_mean,
            "same_region_gate_probability_std": probability_variance**0.5,
            "true_same_gate_mean": self.positive_probability_sum / max(positive_count, 1),
            "true_different_gate_mean": self.negative_probability_sum / max(negative_count, 1),
            "mean_local_gate_weight": probability_mean,
            "mean_cross_gate_weight": 1.0 - probability_mean,
            "pair_count": self.count,
        }


def same_region_pair_targets(
    cipher_region_ids: Tensor,
    attention_mask: Tensor,
    exclude_self: bool = True,
) -> tuple[Tensor, Tensor]:
    """Return permutation-invariant pair labels and a padding-aware valid mask."""
    if cipher_region_ids.shape != attention_mask.shape:
        raise ValueError("cipher region ids and attention mask must have the same shape")
    targets = cipher_region_ids.unsqueeze(2).eq(cipher_region_ids.unsqueeze(1))
    valid = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)
    if exclude_self:
        diagonal = torch.eye(
            cipher_region_ids.shape[1], dtype=torch.bool, device=cipher_region_ids.device
        )
        valid = valid & ~diagonal.unsqueeze(0)
    return targets, valid


def balanced_same_region_loss(
    logits: Tensor,
    cipher_region_ids: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    """Class-balanced BCE so abundant cross-region pairs cannot dominate the gate."""
    targets, valid = same_region_pair_targets(cipher_region_ids, attention_mask)
    if logits.shape != targets.shape:
        raise ValueError("same-region logits must have shape [batch, length, length]")
    positive = valid & targets
    negative = valid & ~targets
    losses: list[Tensor] = []
    if bool(positive.any()):
        losses.append(F.binary_cross_entropy_with_logits(logits[positive], torch.ones_like(logits[positive])))
    if bool(negative.any()):
        losses.append(F.binary_cross_entropy_with_logits(logits[negative], torch.zeros_like(logits[negative])))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def flatten_same_region_pairs(
    logits: Tensor,
    cipher_region_ids: Tensor,
    attention_mask: Tensor,
    cipher_values: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    targets, valid = same_region_pair_targets(cipher_region_ids, attention_mask)
    distances = None
    if cipher_values is not None:
        distances = (
            cipher_values.unsqueeze(2) - cipher_values.unsqueeze(1)
        ).abs()[valid]
    return logits[valid], targets[valid], distances


def same_region_classification_metrics(
    logits: Tensor,
    targets: Tensor,
    distances: Tensor | None = None,
    hard_negative_threshold: int | None = None,
    hard_positive_threshold: int | None = None,
) -> dict:
    """Compute grouping, gate-collapse, distance, and boundary diagnostics."""
    logits = logits.detach().flatten().to(torch.float64)
    targets = targets.detach().flatten().to(torch.bool)
    if logits.numel() == 0 or logits.shape != targets.shape:
        raise ValueError("same-region metrics require non-empty aligned logits and targets")
    probabilities = torch.sigmoid(logits)
    predictions = logits >= 0
    positive = targets
    negative = ~targets
    true_positive = int((predictions & positive).sum())
    false_positive = int((predictions & negative).sum())
    true_negative = int((~predictions & negative).sum())
    false_negative = int((~predictions & positive).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    same_accuracy = recall
    different_accuracy = true_negative / max(true_negative + false_positive, 1)
    result: dict = {
        "same_region_pair_accuracy": (true_positive + true_negative) / logits.numel(),
        "same_region_precision": precision,
        "same_region_recall": recall,
        "same_region_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "same_region_balanced_accuracy": 0.5 * (same_accuracy + different_accuracy),
        "same_region_same_class_accuracy": same_accuracy,
        "same_region_different_class_accuracy": different_accuracy,
        "same_region_positive_rate": float(positive.to(torch.float64).mean()),
        "predicted_same_region_rate": float(predictions.to(torch.float64).mean()),
        "same_region_gate_probability_mean": float(probabilities.mean()),
        "same_region_gate_probability_std": float(probabilities.std(unbiased=False)),
        "true_same_gate_mean": float(probabilities[positive].mean()) if bool(positive.any()) else 0.0,
        "true_different_gate_mean": (
            float(probabilities[negative].mean()) if bool(negative.any()) else 0.0
        ),
        "mean_local_gate_weight": float(probabilities.mean()),
        "mean_cross_gate_weight": float(1.0 - probabilities.mean()),
        "pair_count": logits.numel(),
    }
    if distances is None:
        return result
    distances = distances.detach().flatten().to(torch.long)
    if distances.shape != targets.shape:
        raise ValueError("cipher distances must align with same-region pairs")
    bucket_metrics: OrderedDict[str, dict] = OrderedDict()
    for lower, upper, name in DISTANCE_BUCKETS:
        selected = distances >= lower
        if upper is not None:
            selected &= distances <= upper
        bucket_metrics[name] = {
            "pair_count": int(selected.sum()),
            "accuracy": (
                float((predictions[selected] == targets[selected]).to(torch.float64).mean())
                if bool(selected.any())
                else 0.0
            ),
            "positive_probability_mean": (
                float(probabilities[selected].mean()) if bool(selected.any()) else 0.0
            ),
            "positive_rate": (
                float(targets[selected].to(torch.float64).mean()) if bool(selected.any()) else 0.0
            ),
        }
    result["distance_stratified"] = bucket_metrics
    if hard_negative_threshold is not None:
        hard_negative = negative & (distances <= hard_negative_threshold)
        result["hard_negative_threshold"] = hard_negative_threshold
        result["hard_negative_count"] = int(hard_negative.sum())
        result["hard_negative_accuracy"] = (
            float((~predictions[hard_negative]).to(torch.float64).mean())
            if bool(hard_negative.any())
            else 0.0
        )
    if hard_positive_threshold is not None:
        hard_positive = positive & (distances >= hard_positive_threshold)
        result["hard_positive_threshold"] = hard_positive_threshold
        result["hard_positive_count"] = int(hard_positive.sum())
        result["hard_positive_accuracy"] = (
            float(predictions[hard_positive].to(torch.float64).mean())
            if bool(hard_positive.any())
            else 0.0
        )
    return result
