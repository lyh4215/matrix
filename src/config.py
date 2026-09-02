from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


BASELINES = {
    "standard",
    "relational",
    "relational_pool",
    "relational_match",
    "relational_sinkhorn",
}


@dataclass
class DataConfig:
    num_digits: int = 4
    num_tables: int = 30
    sentences_per_table: int = 3
    min_length: int = 8
    max_length: int = 24
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    random_offset: bool = False
    max_random_offset: int = 500
    offset_mode: str = "clamp"
    zone_relocation: bool = False

    def validate(self) -> None:
        if self.num_digits < 1:
            raise ValueError("num_digits must be positive")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train + validation fractions must leave a test split")
        if self.offset_mode not in {"clamp", "wrap", "error"}:
            raise ValueError("offset_mode must be clamp, wrap, or error")


@dataclass
class ModelConfig:
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    num_zones: int = 19
    num_digits: int = 4
    use_absolute_digits: bool = True
    use_digit_delta: bool = True
    use_cipher_delta: bool = True
    # Legacy switch. When set, it controls both new sequence-position switches.
    use_sequence_position: bool | None = None
    use_absolute_sequence_position: bool = True
    use_relative_sequence_position: bool = True
    distance_clip: float = 256.0
    use_log_distance: bool = True
    use_locality_gate: bool = False
    hard_local_radius: float | None = None
    num_cipher_local_heads: int = 0
    num_sequence_heads: int = 0
    pooling: str = "mean"
    matching: str = "dot"
    sinkhorn_iterations: int = 20
    sinkhorn_temperature: float = 1.0
    sinkhorn_dummy_mode: str = "neutral"

    def validate(self) -> None:
        if self.use_sequence_position is not None:
            self.use_absolute_sequence_position = self.use_sequence_position
            self.use_relative_sequence_position = self.use_sequence_position
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_cipher_local_heads + self.num_sequence_heads > self.num_heads:
            raise ValueError("specialized head counts exceed num_heads")
        if self.pooling not in {"mean", "attention"}:
            raise ValueError("pooling must be mean or attention")
        if self.matching not in {"dot", "relational"}:
            raise ValueError("matching must be dot or relational")
        if self.distance_clip <= 0:
            raise ValueError("distance_clip must be positive")
        if self.sinkhorn_dummy_mode != "neutral":
            raise ValueError("only neutral Sinkhorn dummy rows are currently supported")


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    lambda_local: float = 0.0
    lambda_entropy: float = 0.0
    device: str = "auto"
    checkpoint_path: str = "artifacts/model.pt"


@dataclass
class ExperimentConfig:
    seed: int = 42
    baseline: str = "relational_sinkhorn"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if self.baseline not in BASELINES:
            raise ValueError(f"unknown baseline {self.baseline!r}; choose from {sorted(BASELINES)}")
        self.data.validate()
        self.model.num_digits = self.data.num_digits
        self.model.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type, values: Mapping[str, Any] | None):
    values = dict(values or {})
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = ExperimentConfig(
        seed=int(raw.get("seed", 42)),
        baseline=raw.get("baseline", "relational_sinkhorn"),
        data=_construct(DataConfig, raw.get("data")),
        model=_construct(ModelConfig, raw.get("model")),
        training=_construct(TrainingConfig, raw.get("training")),
    )
    config.validate()
    return config
