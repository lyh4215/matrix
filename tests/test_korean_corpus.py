from __future__ import annotations

import copy
import hashlib
import json
import sys
import zipfile

import pytest
import torch

from src.data import korean_corpus as corpus
from src.data.hangul_zones import syllable_to_zone
from src.benchmark import korean_corpus_benchmark as runner
from src.benchmark import region_zone_match_probe as probe
from src.benchmark.region_zone_refine import refine_saved_results


def _documents():
    # Deterministic text fixtures exercise document/window bookkeeping; the
    # separately run public-corpus smoke uses genuine Korean paragraphs.
    return {f"doc-{i:02d}": ["".join(chr(0xAC00 + (i * 37 + j * 13) % 11172) for j in range(512))] for i in range(30)}


def test_nfc_hangul_preprocessing():
    assert corpus.hangul_only("가 나! abc 123 ㄱ🙂") == "가나"


def test_windows_preserve_original_order_and_splits_have_no_overlap():
    documents = _documents()
    p, splits, manifest = corpus.build_corpus_graphs(documents, {"train": 4, "validation": 2, "test": 2})
    ids = [set(values) for values in manifest["document_splits"].values()]
    assert all(not ids[i] & ids[j] for i in range(3) for j in range(i))
    seen = set()
    by_hash = {corpus.digest(key): value for key, value in documents.items()}
    for split, by_length in splits.items():
        for graph, info in zip(by_length[256], manifest["windows"][split]):
            text = by_hash[info["document_sha256"]][info["segment_index"]]
            window = text[info["hangul_start"]:info["hangul_start"] + 256]
            assert corpus.digest(window) == info["window_sha256"]
            assert info["window_sha256"] not in seen
            seen.add(info["window_sha256"])
            recovered = graph.true_zone_by_anonymous_region[torch.tensor(graph.anonymous_sequences[0])]
            assert recovered.tolist() == [syllable_to_zone(char) for char in window]
            assert graph.transition_counts.sum() == 255
            assert graph.token_counts.sum() == 256
            assert graph.transition_matrix.shape == (19, 19)
    assert torch.isfinite(p).all() and bool((p > 0).all())
    assert torch.allclose(p.sum(1), torch.ones(19, dtype=torch.float64))


def test_canonical_counts_use_only_train_and_never_cross_paragraphs():
    documents = _documents()
    for key in documents:
        documents[key].append("가나")
    requested = {"train": 4, "validation": 2, "test": 2}
    p, _, metadata = corpus.build_corpus_graphs(documents, requested)
    partition = corpus.split_document_ids(documents, 42)
    changed = copy.deepcopy(documents)
    for key in partition["validation"] + partition["test"]:
        changed[key] = [text[::-1] for text in changed[key]]
    other_p, _, _ = corpus.build_corpus_graphs(changed, requested)
    assert torch.equal(p, other_p)
    expected = torch.zeros(19, 19, dtype=torch.float64)
    seen_paragraphs = set()
    for key in partition["train"]:
        for text in documents[key]:
            if text in seen_paragraphs:
                continue
            seen_paragraphs.add(text)
            zones = [syllable_to_zone(char) for char in text]
            for left, right in zip(zones, zones[1:]):
                expected[left, right] += 1
    assert torch.equal(torch.tensor(metadata["canonical_counts"], dtype=torch.float64), expected)


def test_duplicate_paragraphs_and_windows_are_removed_across_splits():
    documents = _documents()
    ids = corpus.split_document_ids(documents, 42)
    copied = documents[ids["train"][0]][0]
    documents[ids["test"][0]].extend([copied, copied[:256]])
    _, _, metadata = corpus.build_corpus_graphs(documents, {"train": 4, "validation": 2, "test": 2})
    assert metadata["statistics"]["test"]["duplicate_paragraphs_removed"] >= 1
    assert metadata["statistics"]["test"]["duplicate_windows_removed"] >= 1


def test_short_paragraphs_are_not_concatenated_to_create_windows():
    documents = {key: [text[:128], text[128:256]] for key, (text,) in _documents().items()}
    with pytest.raises(ValueError, match="only 0 unique length-256"):
        corpus.build_corpus_graphs(documents, {"train": 1, "validation": 1, "test": 1})


def test_korquad_reads_context_once_and_verifies_checksum(tmp_path, monkeypatch):
    body = json.dumps({"data": [{"title": "article", "paragraphs": [{"context": "한국어 문단", "qas": [1, 2, 3]}]}]}).encode()
    path = tmp_path / "KorQuAD_v1.0_train.json"
    path.write_bytes(body)
    monkeypatch.setattr(corpus, "KORQUAD_SHA256", hashlib.sha256(body).hexdigest())
    docs, _ = corpus.load_korquad(tmp_path)
    assert docs == {"article": ["한국어 문단"]}
    path.write_bytes(body + b" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        corpus.load_korquad(tmp_path)


def test_corpus_cli_smoke_bypasses_markov_generator_and_exports_provenance(tmp_path, monkeypatch):
    documents = _documents()
    source = tmp_path / "documents.jsonl"
    source.write_text("\n".join(json.dumps({"id": key, "text": text[0]}, ensure_ascii=False) for key, text in documents.items()), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("real corpus benchmark must never generate Markov sequences")

    monkeypatch.setattr(probe, "generate_controlled_benchmark", forbidden)
    monkeypatch.setattr(sys, "argv", ["korean_corpus_benchmark.py", "--corpus", str(source), "--smoke", "--output-dir", str(tmp_path / "results")])
    runner.main()
    raw, = (tmp_path / "results").glob("*/raw_results.json")
    payload = json.loads(raw.read_text())
    assert payload["config"]["data_source"]["kind"] == "korean_corpus"
    assert len(payload["learned_history"]) == len(payload["learned_structural_history"]) == 1
    assert len(payload["results"]) == 8
    assert all(row["sequence_length"] == 256 for row in payload["results"])
    archive, = (tmp_path / "results").glob("*.zip")
    with zipfile.ZipFile(archive) as bundle:
        assert "corpus_manifest.json" in bundle.namelist()
        assert "summary.csv" in bundle.namelist()
    with pytest.raises(ValueError, match="synthetic generator"):
        refine_saved_results(raw, tmp_path / "wrong_replay")
