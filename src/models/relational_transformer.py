from __future__ import annotations

from torch import Tensor, nn

from ..config import ModelConfig
from .common import DigitProjection, sinusoidal_positions
from .relational_attention import RelationalAttention


class RelationalEncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = RelationalAttention(config)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.feedforward_norm = nn.LayerNorm(config.d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden: Tensor,
        digits: Tensor,
        cipher_values: Tensor,
        attention_mask: Tensor,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        message, weights = self.attention(
            self.attention_norm(hidden), digits, cipher_values, attention_mask, return_attention
        )
        hidden = hidden + self.attention_dropout(message)
        hidden = hidden + self.feedforward(self.feedforward_norm(hidden))
        hidden = hidden * attention_mask.unsqueeze(-1)
        return hidden, weights


class RelationalTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = DigitProjection(
            config.num_digits, config.d_model, config.use_absolute_digits
        )
        self.layers = nn.ModuleList(RelationalEncoderLayer(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        digits: Tensor,
        cipher_values: Tensor,
        attention_mask: Tensor,
        return_attention: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        hidden = self.input_projection(digits)
        if self.config.use_sequence_position:
            hidden = hidden + sinusoidal_positions(
                hidden.shape[1], hidden.shape[2], hidden.device, hidden.dtype
            ).unsqueeze(0)
        attentions: list[Tensor] = []
        for layer in self.layers:
            hidden, weights = layer(
                hidden, digits, cipher_values, attention_mask, return_attention
            )
            if weights is not None:
                attentions.append(weights)
        return self.final_norm(hidden) * attention_mask.unsqueeze(-1), attentions
