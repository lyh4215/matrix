from pathlib import Path

from src.analysis.ablation import make_ablation_configs
from src.config import load_config


def test_default_config_and_ablation_matrix() -> None:
    config = load_config(Path("configs/default.yaml"))
    variants = make_ablation_configs(config)
    assert variants["standard_attention"].baseline == "standard"
    assert variants["no_absolute_digits"].model.use_absolute_digits is False
    assert variants["random_offset"].data.random_offset is True
    assert variants["zone_relocation"].data.zone_relocation is True
