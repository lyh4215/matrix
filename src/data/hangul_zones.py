from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

CHOSEONG = ("ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")
HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3
SYLLABLES_PER_INITIAL = 21 * 28


class HangulZoneScheme(Protocol):
    @property
    def num_zones(self) -> int: ...

    @property
    def zone_names(self) -> tuple[str, ...]: ...

    def zone_id(self, syllable: str) -> int: ...


@dataclass(frozen=True)
class ChoseongZoneScheme:
    """The initial 19-zone scheme; replace this object for finer granularity."""

    @property
    def num_zones(self) -> int:
        return len(CHOSEONG)

    @property
    def zone_names(self) -> tuple[str, ...]:
        return CHOSEONG

    def zone_id(self, syllable: str) -> int:
        if len(syllable) != 1 or not HANGUL_BASE <= ord(syllable) <= HANGUL_END:
            raise ValueError(f"expected one precomposed Hangul syllable, got {syllable!r}")
        return (ord(syllable) - HANGUL_BASE) // SYLLABLES_PER_INITIAL

    def zone_name(self, zone_id: int) -> str:
        return self.zone_names[zone_id]


def syllable_to_zone(syllable: str) -> int:
    return ChoseongZoneScheme().zone_id(syllable)

