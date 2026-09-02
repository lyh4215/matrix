from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import ModelConfig


def _mlp(input_dim: int, hidden_dim: int, output_dim: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class RelationalAttention(nn.Module):
    """Multi-head attention scored by an MLP over explicit pairwise relations.

    This deliberately does not use Q @ K.T. Q/K remain contextual routing
    features, while V carries the message aggregated at each query token.
    """

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

        digit_dim = 2 * config.num_digits if config.use_digit_delta else 0
        cipher_dim = 4 if config.use_cipher_delta else 0
        sequence_dim = 4
        pair_dim = 4 * self.head_dim + digit_dim + cipher_dim + sequence_dim
        relative_dim = digit_dim + cipher_dim + sequence_dim
        hidden_dim = max(16, self.head_dim * 2)
        self.score_mlps = nn.ModuleList(
            _mlp(pair_dim, hidden_dim) for _ in range(config.num_heads)
        )
        self.gate_mlps = (
            nn.ModuleList(_mlp(relative_dim, hidden_dim) for _ in range(config.num_heads))
            if config.use_locality_gate
            else None
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
        log_value = torch.log1p(absolute) / math.log1p(clip) if use_log else torch.zeros_like(delta)
        return torch.stack(
            ((delta.clamp(min=-clip, max=clip) / clip), clipped / clip, delta.sign(), log_value),
            dim=-1,
        )

    def _split_heads(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden: Tensor,
        digits: Tensor,
        cipher_values: Tensor,
        attention_mask: Tensor,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        batch, length, _ = hidden.shape
        q = self._split_heads(self.q_projection(hidden))
        k = self._split_heads(self.k_projection(hidden))
        v = self._split_heads(self.v_projection(hidden))

        q_pair = q.unsqueeze(3).expand(-1, -1, -1, length, -1)
        k_pair = k.unsqueeze(2).expand(-1, -1, length, -1, -1)
        hidden_relation = torch.cat((q_pair, k_pair, q_pair - k_pair, (q_pair - k_pair).abs()), dim=-1)

        digit_features: Tensor | None = None
        if self.config.use_digit_delta:
            digit_delta = digits.unsqueeze(2) - digits.unsqueeze(1)
            digit_features = torch.cat((digit_delta / 9.0, digit_delta.abs() / 9.0), dim=-1)

        cipher_delta = cipher_values.unsqueeze(2).to(hidden.dtype) - cipher_values.unsqueeze(1).to(hidden.dtype)
        cipher_features: Tensor | None = None
        if self.config.use_cipher_delta:
            cipher_features = self._scalar_encoding(
                cipher_delta, self.config.distance_clip, self.config.use_log_distance
            )

        positions = torch.arange(length, device=hidden.device, dtype=hidden.dtype)
        sequence_delta = positions.unsqueeze(1) - positions.unsqueeze(0)
        sequence_features = self._scalar_encoding(
            sequence_delta, float(max(length - 1, 1)), self.config.use_log_distance
        ).unsqueeze(0).expand(batch, -1, -1, -1)

        head_scores: list[Tensor] = []
        for head, role in enumerate(self.head_roles):
            role_parts: list[Tensor] = []
            if digit_features is not None:
                role_parts.append(torch.zeros_like(digit_features) if role == "sequence" else digit_features)
            if cipher_features is not None:
                role_parts.append(torch.zeros_like(cipher_features) if role == "sequence" else cipher_features)
            role_parts.append(torch.zeros_like(sequence_features) if role == "cipher_local" else sequence_features)
            relative = torch.cat(role_parts, dim=-1)
            pair = torch.cat((hidden_relation[:, head], relative), dim=-1)
            score = self.score_mlps[head](pair).squeeze(-1)
            if self.gate_mlps is not None and role != "sequence":
                score = score + F.logsigmoid(self.gate_mlps[head](relative).squeeze(-1))
            if role == "cipher_local" and self.config.hard_local_radius is not None:
                too_far = cipher_delta.abs() > self.config.hard_local_radius
                # Padded queries are zeroed after softmax. Keeping their valid keys
                # avoids an all--inf row, whose undefined gradient can contaminate training.
                too_far = too_far & attention_mask.unsqueeze(2)
                score = score.masked_fill(too_far, -torch.inf)
            head_scores.append(score)
        scores = torch.stack(head_scores, dim=1)

        scores = scores.masked_fill(~attention_mask[:, None, None, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights) * attention_mask[:, None, :, None]
        messages = torch.matmul(self.dropout(weights), v)
        messages = messages.transpose(1, 2).contiguous().view(batch, length, -1)
        output = self.out_projection(messages) * attention_mask.unsqueeze(-1)
        return output, weights if return_attention else None
