"""Region-label-equivariant sequence attention for semantic zone assignment."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .region_zone_matcher import RegionZoneMatchOutput
from .sinkhorn import sinkhorn


class SequenceAttentionLayer(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.same_region_bias = nn.Parameter(torch.ones(heads))
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, same, valid):
        b, length = valid.shape
        bias = same[:, None].to(x.dtype) * self.same_region_bias[None, :, None, None]
        bias = bias.masked_fill(~valid[:, None, None, :], float('-inf'))
        y = self.norm1(x)
        y = self.attention(y, y, y, attn_mask=bias.reshape(b * self.heads, length, length), need_weights=False)[0]
        x = x + self.dropout(y)
        return x + self.dropout(self.ff(self.norm2(x)))


class SequenceRegionZoneMatcher(nn.Module):
    """Only sequence position, frequency, and equality of anonymous regions enter.

    Region IDs are used for equality tests and pooling, never embedding lookup.
    Padding is -1. Position encodings refer to text positions, not numeric regions.
    """
    def __init__(self, num_zones=19, d_model=64, num_layers=3, dropout=.1,
                 sinkhorn_iterations=30, sinkhorn_temperature=1., num_heads=4):
        super().__init__()
        if num_zones < 2 or num_layers < 1 or d_model < 2 or d_model % num_heads:
            raise ValueError('positive layers and d_model divisible by num_heads required')
        self.num_zones = num_zones
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_temperature = sinkhorn_temperature
        self.frequency_projection = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([SequenceAttentionLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.semantic_prototypes = nn.Parameter(torch.randn(num_zones, d_model) / math.sqrt(d_model))
        self.compatibility = nn.Sequential(nn.Linear(4 * d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, 1))

    def forward(self, sequence):
        if sequence.ndim != 2 or sequence.dtype != torch.long:
            raise ValueError('sequence must be a [batch,length] LongTensor')
        valid = sequence >= 0
        if not bool(valid.any(1).all()) or bool((sequence < -1).any()) or bool((sequence >= self.num_zones).any()):
            raise ValueError('each sequence needs valid region IDs; padding must be -1')
        membership = F.one_hot(sequence.clamp_min(0), self.num_zones).to(self.semantic_prototypes.dtype) * valid[..., None]
        counts = membership.sum(1)
        frequency = counts / valid.sum(1, keepdim=True)
        token_frequency = (membership * frequency[:, None]).sum(-1, keepdim=True)
        x = self.frequency_projection(token_frequency)
        length, width = sequence.shape[1], x.shape[-1]
        positions = torch.arange(length, device=x.device, dtype=x.dtype)[:, None]
        scales = torch.exp(torch.arange(0, width, 2, device=x.device, dtype=x.dtype) * (-math.log(10000.) / width))
        encoding = torch.zeros(length, width, device=x.device, dtype=x.dtype)
        encoding[:, 0::2] = torch.sin(positions * scales)
        encoding[:, 1::2] = torch.cos(positions * scales[:encoding[:, 1::2].shape[1]])
        x = x + encoding
        same = sequence[:, :, None] == sequence[:, None, :]
        for layer in self.layers:
            x = layer(x, same, valid)
        hidden = membership.transpose(1, 2) @ self.norm(x) / counts.clamp_min(1)[..., None]
        region = hidden[:, :, None]
        prototype = self.semantic_prototypes[None, None]
        difference = region - prototype
        pair = torch.cat((region.expand(-1, -1, self.num_zones, -1),
                          prototype.expand(len(sequence), self.num_zones, -1, -1),
                          difference, difference.abs()), -1)
        scores = self.compatibility(pair).squeeze(-1)
        assignment = sinkhorn(scores, row_mask=counts > 0, iterations=self.sinkhorn_iterations,
                              temperature=self.sinkhorn_temperature)
        return RegionZoneMatchOutput(scores, assignment, hidden)
