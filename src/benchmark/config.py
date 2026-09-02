from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..config import BASELINES, ModelConfig, TrainingConfig
from ..data.controlled_synthetic import ControlledSyntheticConfig


@dataclass
class BenchmarkConfig:
    seeds: tuple[int, ...] = (41, 42, 43)
    models: tuple[str, ...] = (
        "standard",
        "relational",
        "relational_pool",
        "relational_match",
        "relational_sinkhorn",
    )
    include_ablations: bool = True
    output_dir: str = "artifacts/synthetic_benchmark"
    max_attention_batches: int | None = 10
    synthetic: ControlledSyntheticConfig = field(default_factory=ControlledSyntheticConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        unknown_models = set(self.models) - BASELINES
        if unknown_models:
            raise ValueError(f"unknown benchmark models: {sorted(unknown_models)}")
        if self.max_attention_batches is not None and self.max_attention_batches < 1:
            raise ValueError("max_attention_batches must be positive or null")
        self.synthetic.validate()
        self.model.num_zones = self.synthetic.num_zones
        self.model.num_digits = self.synthetic.num_digits
        self.model.validate()
        if self.training.epochs < 1 or self.training.batch_size < 1:
            raise ValueError("training epochs and batch size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type, values: Mapping[str, Any] | None):
    values = dict(values or {})
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    synthetic_raw = dict(raw.get("synthetic") or {})
    for key in ("sequence_lengths", "noise_levels"):
        if key in synthetic_raw:
            synthetic_raw[key] = tuple(synthetic_raw[key])
    config = BenchmarkConfig(
        seeds=tuple(raw.get("seeds", (41, 42, 43))),
        models=tuple(raw.get("models", tuple(sorted(BASELINES)))),
        include_ablations=bool(raw.get("include_ablations", True)),
        output_dir=str(raw.get("output_dir", "artifacts/synthetic_benchmark")),
        max_attention_batches=raw.get("max_attention_batches", 10),
        synthetic=_construct(ControlledSyntheticConfig, synthetic_raw),
        model=_construct(ModelConfig, raw.get("model")),
        training=_construct(TrainingConfig, raw.get("training")),
    )
    config.validate()
    return config
