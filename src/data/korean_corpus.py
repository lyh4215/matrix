"""Real Korean paragraph windows for anonymous choseong-region matching."""
from __future__ import annotations

import hashlib
import json
import math
import random
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import torch

from .hangul_zones import CHOSEONG, syllable_to_zone
from ..benchmark.region_zone_matching import (
    AnonymousRegionGraph, anonymize_region_sequences, row_normalize_transition_counts,
    transition_count_matrix,
)


KORQUAD_URL = "https://korquad.github.io/dataset/KorQuAD_v1.0_train.json"
KORQUAD_SHA256 = "40d5115879a701751781df721d901abfa736d8db5f89000f2619433f39bf2dd2"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hangul_only(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFC", text) if "가" <= c <= "힣")


def load_korquad(cache_dir: str | Path) -> tuple[dict[str, list[str]], dict]:
    path = Path(cache_dir) / "KorQuAD_v1.0_train.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading real Korean corpus: {KORQUAD_URL}", flush=True)
        with urlopen(KORQUAD_URL, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != KORQUAD_SHA256:
            raise ValueError("KorQuAD source checksum mismatch")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != KORQUAD_SHA256:
        raise ValueError("cached KorQuAD checksum mismatch")
    documents: dict[str, list[str]] = {}
    for article in json.loads(content)["data"]:
        # All paragraphs of one Wikipedia title stay in the same split. Read
        # each context once, regardless of how many QA pairs it contains.
        documents.setdefault(article["title"], []).extend(p["context"] for p in article["paragraphs"])
    return documents, {
        "name": "KorQuAD 1.0 train contexts (Wikipedia)", "url": KORQUAD_URL,
        "documentation": "https://korquad.github.io/category/1.0_ENG.html",
        "license": "CC BY-ND 2.0 KR", "sha256": KORQUAD_SHA256,
        "note": "Original train file is repartitioned by article title; QA pairs are not used.",
    }


def load_local_documents(path: str | Path) -> tuple[dict[str, list[str]], dict]:
    """JSONL records {id, text}, or a directory with one .txt per document."""
    path = Path(path)
    documents: dict[str, list[str]] = {}
    if path.is_dir():
        for file in sorted(path.rglob("*.txt")):
            documents[str(file.relative_to(path))] = [file.read_text(encoding="utf-8")]
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                documents.setdefault(str(record["id"]), []).append(record["text"])
    fingerprint = digest(json.dumps(documents, ensure_ascii=False, sort_keys=True))
    return documents, {"name": "local documents", "path": str(path.resolve()), "sha256": fingerprint}


def split_document_ids(documents: dict, seed: int) -> dict[str, list[str]]:
    ids = sorted(documents)
    if len(ids) < 3:
        raise ValueError("at least three documents are needed for document-disjoint splits")
    random.Random(seed).shuffle(ids)
    train_end = min(len(ids) - 2, max(1, int(0.8 * len(ids))))
    validation_end = min(len(ids) - 1, train_end + max(1, int(0.1 * len(ids))))
    return {"train": ids[:train_end], "validation": ids[train_end:validation_end], "test": ids[validation_end:]}


def _window_graph(text: str, table_id: str, seed: int, alpha: float) -> AnonymousRegionGraph:
    zones = tuple(syllable_to_zone(char) for char in text)
    zone_to_region = list(range(19))
    random.Random(seed).shuffle(zone_to_region)
    raw = tuple(zone_to_region[zone] for zone in zones)
    sequences, anonymous_to_raw = anonymize_region_sequences((raw,), 19, seed, table_id)
    counts = transition_count_matrix(sequences, 19)
    frequency = torch.bincount(torch.tensor(sequences[0]), minlength=19).double()
    raw_to_zone = {region: zone for zone, region in enumerate(zone_to_region)}
    return AnonymousRegionGraph(
        table_id, len(text), sequences, anonymous_to_raw, counts,
        row_normalize_transition_counts(counts, alpha), frequency, frequency / frequency.sum(),
        frequency > 0, torch.tensor([raw_to_zone[region] for region in anonymous_to_raw]),
    )


def build_corpus_graphs(
    documents: dict[str, list[str]], counts_by_split: dict[str, int],
    length: int = 256, seed: int = 42, canonical_alpha: float = 0.1,
    graph_alpha: float = 0.001, source: dict | None = None,
) -> tuple[torch.Tensor, dict, dict]:
    """P is estimated only from training paragraphs; no sequence is sampled from P."""
    if length < 2 or not math.isfinite(canonical_alpha) or canonical_alpha <= 0:
        raise ValueError("length must be >= 2 and canonical_alpha finite and positive")
    if set(counts_by_split) != {"train", "validation", "test"} or min(counts_by_split.values()) < 1:
        raise ValueError("positive train, validation and test table counts are required")
    partitions = split_document_ids(documents, seed)
    canonical_counts = torch.zeros(19, 19, dtype=torch.float64)
    seen_paragraphs: set[str] = set()
    seen_windows: set[str] = set()
    graphs, window_manifest, statistics = {}, {}, {}
    for split_index, (split, ids) in enumerate(partitions.items()):
        windows = []
        syllables, paragraphs, duplicated_paragraphs, duplicated_windows = 0, 0, 0, 0
        for document_id in ids:
            for segment_index, segment in enumerate(documents[document_id]):
                text = hangul_only(segment)
                if not text:
                    continue
                fingerprint = digest(text)
                if fingerprint in seen_paragraphs:
                    duplicated_paragraphs += 1
                    continue
                seen_paragraphs.add(fingerprint)
                paragraphs += 1
                syllables += len(text)
                if split == "train":
                    zones = [syllable_to_zone(char) for char in text]
                    for (left, right), count in Counter(zip(zones, zones[1:])).items():
                        canonical_counts[left, right] += count
                # Never join unrelated paragraphs or documents to reach length.
                for start in range(0, len(text) - length + 1, length):
                    window = text[start:start + length]
                    window_hash = digest(window)
                    if window_hash in seen_windows:
                        duplicated_windows += 1
                        continue
                    seen_windows.add(window_hash)
                    windows.append((window, {
                        "document_sha256": digest(document_id), "segment_index": segment_index,
                        "hangul_start": start, "length": length, "window_sha256": window_hash,
                    }))
        random.Random(seed + 1001 + split_index).shuffle(windows)
        requested = counts_by_split[split]
        if len(windows) < requested:
            raise ValueError(f"{split}: only {len(windows)} unique length-{length} windows for {requested} tables; reduce table counts or provide more documents")
        batch, manifests = [], []
        for index, (text, metadata) in enumerate(windows[:requested]):
            table_id = f"corpus-{split}-{index:05d}"
            batch.append(_window_graph(text, table_id, seed + split_index * 100000 + index, graph_alpha))
            manifests.append({"table_id": table_id, **metadata})
        graphs[split] = {length: batch}
        window_manifest[split] = manifests
        statistics[split] = {
            "documents": len(ids), "paragraphs": paragraphs, "hangul_syllables": syllables,
            "eligible_windows": len(windows), "selected_windows": requested,
            "selected_documents": len({row["document_sha256"] for row in manifests}),
            "duplicate_paragraphs_removed": duplicated_paragraphs,
            "duplicate_windows_removed": duplicated_windows,
            "mean_observed_zones": sum(int(g.observed_mask.sum()) for g in batch) / len(batch),
            "min_observed_zones": min(int(g.observed_mask.sum()) for g in batch),
            "semantic_zone_presence_tables": [sum(bool((g.true_zone_by_anonymous_region[g.observed_mask] == zone).any()) for g in batch) for zone in range(19)],
        }
    p = (canonical_counts + canonical_alpha) / (canonical_counts.sum(1, keepdim=True) + 19 * canonical_alpha)
    metadata = {
        "kind": "korean_corpus", "source": source or {}, "seed": seed,
        "token_unit": "NFC Hangul syllable (not word or subword)", "sequence_length": length,
        "preprocessing": "keep U+AC00..U+D7A3; remove spaces/punctuation/other scripts; non-overlapping windows within a paragraph",
        "grouping": "true choseong-region grouping supplied; numeric cipher values are not inputs",
        "canonical_source": "training paragraphs only; row-smoothed empirical counts",
        "canonical_alpha": canonical_alpha, "canonical_counts": canonical_counts.tolist(),
        "zone_names": list(CHOSEONG), "statistics": statistics,
        "document_splits": {key: [digest(i) for i in ids] for key, ids in partitions.items()},
        "windows": window_manifest,
        "limitations": "exact paragraph/window deduplication only, not near-duplicate detection; multiple test windows may share an article; evaluates semantic zones, not full syllable recovery",
    }
    return p, graphs, metadata
