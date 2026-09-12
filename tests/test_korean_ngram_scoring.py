import pytest
import torch

from src.benchmark import korean_ngram_scoring as ng
from src.benchmark.korean_trigram_diagnostic import fit_trigrams, sequence_nll, trigram_probabilities
from src.data.korean_corpus import _window_graph, split_document_ids


def test_counts_train_only_and_no_boundary_ngrams():
    docs = {str(i): ['가나다라마', '바사'] for i in range(10)}
    counts = ng.fit_ngrams(docs, 42)
    assert [int(counts[n].sum()) for n in (3, 4, 5)] == [3, 2, 1]
    assert torch.equal(counts[3], fit_trigrams(docs, 42))
    parts = split_document_ids(docs, 42)
    for key in parts['validation'] + parts['test']:
        docs[key] = ['바바바바바바']
    other = ng.fit_ngrams(docs, 42)
    assert all(torch.equal(counts[n], other[n]) for n in counts)


def test_normalization_backoff_and_trigram_equivalence():
    p = torch.rand(19, 19, dtype=torch.float64)
    p /= p.sum(-1, keepdim=True)
    counts = {n: torch.zeros((19,) * n, dtype=torch.float64) for n in (3, 4, 5)}
    counts[3][0, 1, 2] = 10
    levels = ng.probabilities(counts, p, 10, 5)
    for n in (3, 4, 5):
        assert torch.allclose(levels[n].sum(-1), torch.ones_like(levels[n].sum(-1)))
        assert torch.isfinite(levels[n].log()).all()
    assert torch.allclose(levels[5][0], levels[4])
    for length in (0, 1, 2, 3, 4, 5, 20):
        seq = torch.arange(length) % 19
        assignment = torch.randperm(19)
        for weight in (0., .5, 1.):
            actual = ng.sequence_nll(seq, assignment, p, {k: v for k, v in levels.items() if k <= 3}, weight)
            expected = sequence_nll(seq, assignment, p, trigram_probabilities(counts[3], p, 10), weight)
            assert actual == pytest.approx(expected)
        z = assignment[seq]
        assert ng.sequence_nll(seq, assignment, p, levels, 0) == pytest.approx(float(-p[z[:-1], z[1:]].log().sum()))


def test_longer_order_scoring_and_reindexing():
    p = torch.full((19, 19), 1 / 19, dtype=torch.float64)
    counts = {n: torch.zeros((19,) * n, dtype=torch.float64) for n in (3, 4, 5)}
    counts[5][0, 1, 2, 3, 4] = 100
    levels = ng.probabilities(counts, p, 1, 5)
    seq = torch.tensor([0, 1, 2, 3, 4])
    a = torch.arange(19)
    assert ng.sequence_nll(seq, a, p, levels, 1) < ng.sequence_nll(seq, a.roll(1), p, levels, 1)
    reindex = torch.randperm(19)
    new_a = torch.empty_like(a)
    new_a[reindex] = a
    assert ng.sequence_nll(reindex[seq], new_a, p, levels, 1) == pytest.approx(ng.sequence_nll(seq, a, p, levels, 1))


def test_unseen_history_distinguished_from_unseen_continuation():
    g = _window_graph('가나다라마', 'table', 42, .001)
    counts = {n: torch.zeros((19,) * n, dtype=torch.float64) for n in (3, 4, 5)}
    z = g.true_zone_by_anonymous_region[torch.tensor(g.anonymous_sequences[0])]
    # Observed four-symbol history with a different continuation.
    counts[5][tuple(z[:4]) + ((int(z[4]) + 1) % 19,)] = 10
    candidates = {'table': [{'matcher': name, 'predicted_assignment': g.true_zone_by_anonymous_region.tolist()}
                            for name in ('oracle_transition', 'learned_structural_local_search')]}
    stats = ng.context_coverage([g], candidates, counts, 5, 1)['truth'][5]
    assert stats['positions'] == 1
    assert stats['unseen_history_rate'] == 0
    assert stats['unseen_ngram_rate'] == 1
    assert stats['mean_lower_order_mass'] == pytest.approx(1 / 11)
