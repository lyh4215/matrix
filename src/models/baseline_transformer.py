from __future__ import annotations

from torch import Tensor, nn

from ..config import ModelConfig
from .common import DigitProjection, sinusoidal_positions


class BaselineTransformer(nn.Module):
    """Phase-1 dot-product Transformer baseline over projected digit vectors."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = DigitProjection(
            config.num_digits, config.d_model, config.use_absolute_digits
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, config.num_layers, norm=nn.LayerNorm(config.d_model))

    def forward(self, digits: Tensor, attention_mask: Tensor) -> Tensor:
        hidden = self.input_projection(digits)
        if self.config.use_sequence_position:
            hidden = hidden + sinusoidal_positions(
                hidden.shape[1], hidden.shape[2], hidden.device, hidden.dtype
            ).unsqueeze(0)
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask)
        return hidden * attention_mask.unsqueeze(-1)

