import random

import pytest
import torch

from src.data.cipher_generator import CipherAugmenter, generate_synthetic_episodes
from src.data.dataset import CipherEpisode, collate_episodes, split_by_cipher_table, table_ids, values_to_digits
from src.data.hangul_zones import CHOSEONG, ChoseongZoneScheme


def test_choseong_zone_scheme() -> None:
    scheme = ChoseongZoneScheme()
    assert scheme.num_zones == 19
    assert scheme.zone_id("가") == 0
    assert scheme.zone_id("까") == 1
    assert scheme.zone_id("힣") == 18
    assert CHOSEONG[scheme.zone_id("나")] == "ㄴ"
    with pytest.raises(ValueError):
        scheme.zone_id("A")


def test_digit_conversion_and_padding() -> None:
    episodes = [
        CipherEpisode((3126, 42), (0, 1), (10, 11), "f1"),
        CipherEpisode((9001,), (2,), (7,), "f2"),
    ]
    batch = collate_episodes(episodes)
    assert batch["digits"][0, 0].tolist() == [3.0, 1.0, 2.0, 6.0]
    assert batch["digits"][0, 1].tolist() == [0.0, 0.0, 4.0, 2.0]
    assert batch["attention_mask"].tolist() == [[True, True], [True, False]]
    assert batch["zone_labels"][1, 1].item() == -100
    with pytest.raises(ValueError):
        values_to_digits(torch.tensor([10_000]), 4)


def test_split_is_strictly_by_cipher_table() -> None:
    episodes = generate_synthetic_episodes(num_tables=10, sentences_per_table=3, seed=7)
    train, validation, test = split_by_cipher_table(episodes, 0.6, 0.2, seed=9)
    assert not (table_ids(train) & table_ids(validation))
    assert not (table_ids(train) & table_ids(test))
    assert not (table_ids(validation) & table_ids(test))
    assert len(train) + len(validation) + len(test) == len(episodes)


def test_zone_relocation_preserves_within_zone_offsets() -> None:
    episode = CipherEpisode((101, 105, 800, 808), (0, 0, 2, 2), (4, 4, 9, 9), "f")
    augmenter = CipherAugmenter(zone_relocation=True, seed=3)
    changed = augmenter(episode)
    assert changed.cipher_values[1] - changed.cipher_values[0] == 4
    assert changed.cipher_values[3] - changed.cipher_values[2] == 8
    assert changed.zone_labels == episode.zone_labels

