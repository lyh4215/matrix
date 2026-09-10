from __future__ import annotations

import json

import pytest
import torch

from src.benchmark import region_zone_match_probe as probe
from src.benchmark.region_zone_complementarity import evaluate_complementarity, select_by_count_nll
from src.benchmark.region_zone_matching import AnonymousRegionGraph, score_region_assignment
from src.benchmark.region_zone_refine import refine_saved_results
from src.data.controlled_synthetic import MarkovZoneLanguage


def _canonical():
    return torch.tensor(MarkovZoneLanguage.create(4, 8, 2, 8.0, 9).transition_matrix).double()


def _graph(name, counts):
    return AnonymousRegionGraph(
        name, 128, (), (0, 1, 2, 3), counts, counts / counts.sum(1, keepdim=True),
        torch.ones(4), torch.ones(4) / 4, torch.ones(4, dtype=torch.bool), torch.arange(4),
    )


def _result(graphs, assignments):
    return {"table_results": [score_region_assignment(g, a) for g, a in zip(graphs, assignments)]}


def test_selector_uses_nll_and_retains_oracle_on_ties():
    p = _canonical()
    truth, wrong = (0, 1, 2, 3), (1, 0, 2, 3)
    chosen, source, left, right = select_by_count_nll(100 * p, p, wrong, truth)
    assert chosen == truth
    assert source == "learned_structural_local_search"
    assert right < left
    chosen, source, left, right = select_by_count_nll(torch.zeros_like(p), p, wrong, truth)
    assert left == right == 0
    assert chosen == wrong and source == "oracle_transition"
    with pytest.raises(ValueError, match="one-to-one"):
        select_by_count_nll(p, p, [0, 0, 2, 3], truth)


def test_complementarity_overlap_alignment_and_minimum_nll():
    p = _canonical()
    truth, wrong, other = (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2)
    graphs = [_graph(str(i), 100 * p) for i in range(4)]
    left = _result(graphs, [truth, truth, wrong, wrong])
    right = _result(graphs, [truth, wrong, truth, other])
    # Input order must not affect pairing. Stale cached objectives are ignored.
    right["table_results"].reverse()
    for row in right["table_results"]:
        row["final_count_nll"] = -999999.0
    result = evaluate_complementarity(left, right, graphs, p)
    d = result["complementarity"]
    assert d["exact_overlap"] == {"both": 1, "oracle_only": 1, "structural_only": 1, "neither": 1}
    assert d["label_assisted_diagnostic_only"]["either_exact_rate"] == 0.75
    assert d["same_observed_assignment_tables"] == 1
    for row in result["table_results"]:
        assert row["selected_count_nll"] == min(row["oracle_count_nll"], row["structural_count_nll"])
        assert row["selected_gap_to_true"] == pytest.approx(row["selected_count_nll"] - row["true_count_nll"])
    right["table_results"].pop()
    with pytest.raises(ValueError, match="same tables"):
        evaluate_complementarity(left, right, graphs, p)


def test_lower_nll_can_choose_less_accurate_candidate_without_label_leakage():
    p = _canonical()
    truth, wrong = (0, 1, 2, 3), (1, 0, 2, 3)
    graph = _graph("noisy", 100 * p[list(wrong)][:, list(wrong)])
    result = evaluate_complementarity(_result([graph], [truth]), _result([graph], [wrong]), [graph], p)
    assert result["table_results"][0]["predicted_assignment"] == list(wrong)
    d = result["complementarity"]
    assert d["label_assisted_diagnostic_only"]["selector_missed_better_assignment_tables"] == 1
    assert d["label_assisted_diagnostic_only"]["best_of_two_assignment_accuracy"] == 1
    assert d["paired_vs_candidates"]["oracle_transition"]["tables_worsened"] == 1


def test_saved_experiment_comparison_has_no_training_and_preserves_source(tmp_path, monkeypatch):
    config = probe.RegionZoneProbeConfig()
    probe._apply_smoke_settings(config)
    config.matchers = ("oracle_transition", "learned_structural")
    config.local_search.enabled = True
    config.compare_oracle = True
    config.learned.device = "cpu"
    config.output_dir = str(tmp_path / "original")
    original = probe.run_region_zone_match_probe(config)
    source = tmp_path / "original/raw_results.json"
    before = source.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("comparison must not train or load a model")

    monkeypatch.setattr(probe, "_train_learned_matcher", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    replay = refine_saved_results(source, tmp_path / "replay", compare_oracle=True)
    assert source.read_bytes() == before
    actual = next(r for r in replay["results"] if "complementarity" in r)
    expected = next(r for r in original["results"] if "complementarity" in r)
    assert actual["complementarity"] == expected["complementarity"]
    assert actual["table_results"] == expected["table_results"]
    markdown = (tmp_path / "replay/summary.md").read_text()
    assert "diagnostic only; not a deployable selector" in markdown
    assert "Exact recovery overlap" in markdown
    assert json.loads((tmp_path / "replay/raw_results.json").read_text())["config"]["compare_oracle"]
