from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass

import yaml

from ..config import ExperimentConfig, load_config


@dataclass(frozen=True)
class AblationVariant:
    name: str
    baseline: str | None = None
    model_changes: tuple[tuple[str, object], ...] = ()
    data_changes: tuple[tuple[str, object], ...] = ()


ABLATIONS = (
    AblationVariant("no_absolute_digits", model_changes=(("use_absolute_digits", False),)),
    AblationVariant("no_cipher_delta", model_changes=(("use_cipher_delta", False),)),
    AblationVariant("no_digit_delta", model_changes=(("use_digit_delta", False),)),
    AblationVariant("no_sequence_position", model_changes=(("use_sequence_position", False),)),
    AblationVariant("standard_attention", baseline="standard"),
    AblationVariant("token_only", baseline="relational"),
    AblationVariant("zone_pooling", baseline="relational_pool"),
    AblationVariant("cross_matching", baseline="relational_match"),
    AblationVariant("sinkhorn", baseline="relational_sinkhorn"),
    AblationVariant("random_offset", data_changes=(("random_offset", True),)),
    AblationVariant("zone_relocation", data_changes=(("zone_relocation", True),)),
)


def make_ablation_configs(config: ExperimentConfig) -> dict[str, ExperimentConfig]:
    result: dict[str, ExperimentConfig] = {}
    for variant in ABLATIONS:
        changed = copy.deepcopy(config)
        if variant.baseline is not None:
            changed.baseline = variant.baseline
        for key, value in variant.model_changes:
            setattr(changed.model, key, value)
        for key, value in variant.data_changes:
            setattr(changed.data, key, value)
        changed.validate()
        result[variant.name] = changed
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Print reproducible ablation configurations")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    variants = make_ablation_configs(load_config(args.config))
    print(yaml.safe_dump({name: item.to_dict() for name, item in variants.items()}, sort_keys=False))


if __name__ == "__main__":
    main()
