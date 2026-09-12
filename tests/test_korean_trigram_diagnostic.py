import torch
import pytest

from src.benchmark.korean_trigram_diagnostic import (
    fit_trigrams, trigram_probabilities, sequence_nll, diagnose, select_setting,
)
from src.data.korean_corpus import _window_graph, split_document_ids


def test_train_only_counts_and_no_paragraph_bridges():
    docs = {str(i): ['가나다', '라마'] for i in range(10)}
    counts = fit_trigrams(docs, 42)
    assert counts.sum() == 1  # repeated normalized train paragraphs deduplicated
    parts = split_document_ids(docs, 42)
    for key in parts['validation'] + parts['test']:
        docs[key] = ['바바바바바']
    assert torch.equal(counts, fit_trigrams(docs, 42))


def test_backoff_normalization_and_finite_unseen_histories():
    bigram = torch.rand(19, 19, dtype=torch.float64)
    bigram /= bigram.sum(-1, keepdim=True)
    counts = torch.zeros(19, 19, 19, dtype=torch.float64)
    counts[0, 0, 0] = 100
    probability = trigram_probabilities(counts, bigram, 10)
    assert torch.allclose(probability.sum(-1), torch.ones(19, 19, dtype=torch.float64))
    assert torch.allclose(probability[1], bigram, atol=1e-15, rtol=1e-14)
    assert torch.isfinite(probability.log()).all()
    with pytest.raises(ValueError):
        trigram_probabilities(counts, bigram, 0)


def test_zero_weight_matches_original_bigram_nll_and_reindexing():
    p = torch.rand(19, 19, dtype=torch.float64)
    p /= p.sum(-1, keepdim=True)
    t = trigram_probabilities(torch.zeros(19, 19, 19), p, 1)
    sequence = torch.tensor([0, 1, 2, 1, 0])
    a = torch.randperm(19)
    z = a[sequence]
    expected = -p[z[:-1], z[1:]].log().sum()
    assert sequence_nll(sequence, a, p, t, 0) == pytest.approx(float(expected))
    reindex = torch.randperm(19)
    new_a = torch.empty_like(a)
    new_a[reindex] = a
    assert sequence_nll(reindex[sequence], new_a, p, t, .5) == pytest.approx(sequence_nll(sequence, a, p, t, .5))


def test_truth_diagnostic_separate_from_candidate_selection():
    graph = _window_graph('가나다' * 20, 'table', 42, .001)
    truth = graph.true_zone_by_anonymous_region.tolist()
    wrong = [(x + 1) % 19 for x in truth]
    p = torch.full((19, 19), 1 / 19, dtype=torch.float64)
    counts = torch.zeros(19, 19, 19, dtype=torch.float64)
    # Encode the true periodic triples; a shifted permutation is disfavored.
    zones = graph.true_zone_by_anonymous_region[torch.tensor(graph.anonymous_sequences[0])]
    for a, b, c in zip(zones, zones[1:], zones[2:]):
        counts[a, b, c] += 1
    t = trigram_probabilities(counts, p, 1)
    candidates = {'table': [{'matcher': 'only_wrong', 'predicted_assignment': wrong}]}
    result = diagnose([graph], candidates, p, t, 1)
    assert result['truth_win_rate'] == 1
    assert result['assignment_accuracy'] == 0
    assert result['table_results'][0]['selected_source'] == 'only_wrong'
    candidates['table'].append({'matcher': 'correct', 'predicted_assignment': truth})
    assert diagnose([graph], candidates, p, t, 1)['assignment_accuracy'] == 1
    # All-exact candidates are not evidence of distinguishing truth from wrong.
    candidates['table'] = candidates['table'][1:]
    assert diagnose([graph], candidates, p, t, 1)['truth_win_rate'] is None


def test_selection_uses_validation_win_rate_then_token_accuracy():
    results = [{'truth_win_rate': .2, 'token_accuracy': .9},
               {'truth_win_rate': .3, 'token_accuracy': .4},
               {'truth_win_rate': .3, 'token_accuracy': .5}]
    assert select_setting(results) == 2
    assert select_setting([results[0], results[0]]) == 0
