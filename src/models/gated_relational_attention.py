from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from ..config import ModelConfig


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 1),
    )


@dataclass(frozen=True)
class GatedPairFeatures:
    gate: Tensor
    local_numeric: Tensor
    sequence: Tensor | None


def blend_branch_scores(local_score: Tensor, cross_score: Tensor, same_probability: Tensor) -> Tensor:
    """Differentiably interpolate within-region and cross-region attention scores."""
    return same_probability * local_score + (1.0 - same_probability) * cross_score


def combine_gated_branch_features(
    hidden_relation: Tensor,
    local_numeric: Tensor,
    sequence: Tensor | None,
    role: str = "mixed",
) -> tuple[Tensor, Tensor]:
    """Build branches while making cross-region numeric leakage structurally impossible."""
    numeric = torch.zeros_like(local_numeric) if role == "sequence" else local_numeric
    local_parts = [hidden_relation, numeric]
    cross_parts = [hidden_relation]
    if sequence is not None:
        effective_sequence = torch.zeros_like(sequence) if role == "cipher_local" else sequence
        local_parts.append(effective_sequence)
        cross_parts.append(effective_sequence)
    # Deliberately, local_numeric is never concatenated into cross_parts.
    return torch.cat(local_parts, dim=-1), torch.cat(cross_parts, dim=-1)


class GatedRelationalAttention(nn.Module):
    """Relational attention with a shared soft same-region structural gate."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.q_projection = nn.Linear(config.d_model, config.d_model)
        self.k_projection = nn.Linear(config.d_model, config.d_model)
        self.v_projection = nn.Linear(config.d_model, config.d_model)
        self.out_projection = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # The gate is head-shared because region co-membership is a structural,
        # not head-semantic, relation. Its inputs contain no signed global ordering.
        gate_dim = 2 + 2 * config.num_digits + 2 * config.d_model
        self.same_region_gate = _mlp(gate_dim, config.same_region_gate_hidden_dim)
        local_numeric_dim = (2 * config.num_digits if config.use_digit_delta else 0) + (
            4 if config.use_cipher_delta else 0
        )
        sequence_dim = 4 if config.use_relative_sequence_position else 0
        hidden_dim = max(16, self.head_dim * 2)
        self.local_score_mlps = nn.ModuleList(
            _mlp(4 * self.head_dim + local_numeric_dim + sequence_dim, hidden_dim)
            for _ in range(config.num_heads)
        )
        self.cross_score_mlps = nn.ModuleList(
            _mlp(4 * self.head_dim + sequence_dim, hidden_dim)
            for _ in range(config.num_heads)
        )
        self.head_roles = tuple(self._role(head) for head in range(config.num_heads))

    def _role(self, head: int) -> str:
        if head < self.config.num_cipher_local_heads:
            return "cipher_local"
        if head < self.config.num_cipher_local_heads + self.config.num_sequence_heads:
            return "sequence"
        return "mixed"

    @staticmethod
    def _scalar_encoding(delta: Tensor, clip: float, use_log: bool) -> Tensor:
        absolute = delta.abs()
        clipped = absolute.clamp(max=clip)
        log_value = (
            torch.log1p(absolute) / math.log1p(clip)
            if use_log
            else torch.zeros_like(delta)
        )
        return torch.stack(
            (
                delta.clamp(min=-clip, max=clip) / clip,
                clipped / clip,
                delta.sign(),
                log_value,
            ),
            dim=-1,
        )

    def _split_heads(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def build_pair_features(
        self,
        hidden: Tensor,
        digits: Tensor,
        cipher_values: Tensor,
        length: int,
        dtype: torch.dtype,
    ) -> GatedPairFeatures:
        digit_delta = digits.unsqueeze(2) - digits.unsqueeze(1)
        digit_absolute = digit_delta.abs()
        cipher_delta = (
            cipher_values.unsqueeze(2).to(dtype) - cipher_values.unsqueeze(1).to(dtype)
        )
        cipher_absolute = cipher_delta.abs()
        clipped_absolute = cipher_absolute.clamp(max=self.config.distance_clip)
        hidden_left = hidden.unsqueeze(2).expand(-1, -1, length, -1)
        hidden_right = hidden.unsqueeze(1).expand(-1, length, -1, -1)
        # Sum and absolute difference keep this context contribution symmetric;
        # no signed global numeric ordering is provided to the shared gate.
        hidden_gate_context = torch.cat(
            (hidden_left + hidden_right, (hidden_left - hidden_right).abs()), dim=-1
        )
        gate = torch.cat(
            (
                (clipped_absolute / self.config.distance_clip).unsqueeze(-1),
                (
                    torch.log1p(clipped_absolute)
                    / math.log1p(self.config.distance_clip)
                ).unsqueeze(-1),
                digit_absolute / 9.0,
                digit_delta.eq(0).to(dtype),
                hidden_gate_context,
            ),
            dim=-1,
        )

        local_parts: list[Tensor] = []
        if self.config.use_digit_delta:
            local_parts.append(torch.cat((digit_delta / 9.0, digit_absolute / 9.0), dim=-1))
        if self.config.use_cipher_delta:
            local_parts.append(
                self._scalar_encoding(
                    cipher_delta, self.config.distance_clip, self.config.use_log_distance
                )
            )
        local_numeric = torch.cat(local_parts, dim=-1) if local_parts else gate.new_zeros(*gate.shape[:-1], 0)

        sequence = None
        if self.config.use_relative_sequence_position:
            positions = torch.arange(length, device=digits.device, dtype=dtype)
            sequence_delta = positions.unsqueeze(1) - positions.unsqueeze(0)
            sequence = self._scalar_encoding(
                sequence_delta, float(max(length - 1, 1)), self.config.use_log_distance
            ).unsqueeze(0).expand(digits.shape[0], -1, -1, -1)
        return GatedPairFeatures(gate, local_numeric, sequence)

    def forward(
        self,
        hidden: Tensor,
        digits: Tensor,
        cipher_values: Tensor,
        attention_mask: Tensor,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        _batch, length, _hidden_dim = hidden.shape
        q = self._split_heads(self.q_projection(hidden))
        k = self._split_heads(self.k_projection(hidden))
        v = self._split_heads(self.v_projection(hidden))
        q_pair = q.unsqueeze(3).expand(-1, -1, -1, length, -1)
        k_pair = k.unsqueeze(2).expand(-1, -1, length, -1, -1)
        hidden_relation = torch.cat(
            (q_pair, k_pair, q_pair - k_pair, (q_pair - k_pair).abs()), dim=-1
        )
        features = self.build_pair_features(
            hidden, digits, cipher_values, length, hidden.dtype
        )
        same_region_logits = self.same_region_gate(features.gate).squeeze(-1)
        same_probability = torch.sigmoid(same_region_logits)

        head_scores: list[Tensor] = []
        for head, role in enumerate(self.head_roles):
            local_features, cross_features = combine_gated_branch_features(
                hidden_relation[:, head], features.local_numeric, features.sequence, role
            )
            local_score = self.local_score_mlps[head](local_features).squeeze(-1)
            cross_score = self.cross_score_mlps[head](cross_features).squeeze(-1)
            head_scores.append(
                blend_branch_scores(local_score, cross_score, same_probability)
            )
        scores = torch.stack(head_scores, dim=1)
        scores = scores.masked_fill(~attention_mask[:, None, None, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights) * attention_mask[:, None, :, None]
        messages = torch.matmul(self.dropout(weights), v)
        messages = messages.transpose(1, 2).contiguous().view(hidden.shape[0], length, -1)
        output = self.out_projection(messages) * attention_mask.unsqueeze(-1)
        return output, weights if return_attention else None, same_region_logits
