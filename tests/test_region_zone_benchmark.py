from __future__ import annotations

import csv
import json
import sys
import zipfile

from src.benchmark import region_zone_benchmark as benchmark
from src.benchmark.region_zone_match_probe import load_region_zone_probe_config


def test_ready_benchmark_defaults() -> None:
    config = load_region_zone_probe_config(benchmark.DEFAULT_CONFIG)
    assert config.matchers == ("learned", "learned_structural")
    assert config.synthetic.sequence_lengths == (128, 256)
    assert config.synthetic.train_tables == 1600
    assert config.synthetic.validation_tables == 100
    assert config.synthetic.test_tables == 200
    assert config.learned.epochs == 50
    assert config.learned.device == "cuda"
    assert config.local_search.enabled


def test_cli_smoke_trains_lengths_independently_and_bundles_results(tmp_path, monkeypatch) -> None:
    lengths_seen = []
    run_probe = benchmark.run_region_zone_match_probe

    def tracked_probe(config):
        lengths_seen.append(config.synthetic.sequence_lengths)
        return run_probe(config)

    monkeypatch.setattr(benchmark, "run_region_zone_match_probe", tracked_probe)
    monkeypatch.setattr(sys, "argv", [
        "region_zone_benchmark.py", "--smoke", "--sequence-lengths", "8", "16",
        "--output-dir", str(tmp_path),
    ])
    benchmark.main()
    assert lengths_seen == [(8,), (16,)]
    summary_path, = tmp_path.glob("*/summary.json")
    summary = json.loads(summary_path.read_text())
    assert summary["training_mode"] == "independent_per_length"
    assert len(summary["runs"]) == 2
    assert len(summary["results"]) == 8
    assert {row["matcher"] for row in summary["results"]} == {
        "learned", "learned_structural", "learned_local_search", "learned_structural_local_search",
    }
    with summary_path.with_suffix(".csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 8
    archive_path, = tmp_path.glob("*.zip")
    with zipfile.ZipFile(archive_path) as archive:
        for length in (8, 16):
            assert f"length_{length}/raw_results.json" in archive.namelist()
            for matcher in ("learned", "learned_structural"):
                assert f"length_{length}/{matcher}/checkpoint.pt" in archive.namelist()
        assert "summary.csv" in archive.namelist()
        assert json.loads(archive.read("summary.json")) == summary
