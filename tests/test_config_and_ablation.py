from pathlib import Path

from src.analysis.ablation import make_ablation_configs
from src.config import ModelConfig, load_config


def test_default_config_and_ablation_matrix() -> None:
    config = load_config(Path("configs/default.yaml"))
    variants = make_ablation_configs(config)
    assert variants["standard_attention"].baseline == "standard"
    assert variants["no_absolute_digits"].model.use_absolute_digits is False
    assert variants["random_offset"].data.random_offset is True
    assert variants["zone_relocation"].data.zone_relocation is True
    assert "no_absolute_sequence_position" in variants
    assert "no_relative_sequence_position" in variants


def test_legacy_and_independent_sequence_position_switches() -> None:
    legacy = ModelConfig(use_sequence_position=False)
    legacy.validate()
    assert legacy.use_absolute_sequence_position is False
    assert legacy.use_relative_sequence_position is False

    independent = ModelConfig(
        use_sequence_position=None,
        use_absolute_sequence_position=False,
        use_relative_sequence_position=True,
    )
    independent.validate()
    assert independent.use_absolute_sequence_position is False
    assert independent.use_relative_sequence_position is True
