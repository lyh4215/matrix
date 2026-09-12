import numpy as np
import pytest
import torch

from src.benchmark import korean_sparse_ngram_scoring as sparse
from src.benchmark import korean_ngram_scoring as dense
from src.data.korean_corpus import split_document_ids, _window_graph


def documents():
    return {str(i): ['가나다라마바사가나다라마바사', '바사'] for i in range(10)}


def test_sparse_dense_count_score_equivalence_and_train_isolation():
    docs = documents()
    a, b = sparse.fit_ngrams(docs, 42, 5), dense.fit_ngrams(docs, 42, 5)
    for n in a:
        assert a[n].sum() == int(b[n].sum())
        for key, value in a[n].patterns.items():
            assert float(b[n].flatten()[key]) == value
    p = torch.rand(19, 19, dtype=torch.float64)
    p /= p.sum(-1, keepdim=True)
    for n in (2, 3, 4, 5):
        for k in (1, 100):
            x, y = sparse.probabilities(a, p, k, n), dense.probabilities(b, p, k, n)
            for length in (0, 1, 2, 4, 7, 20):
                seq = torch.arange(length) % 7
                assignment = torch.arange(19)
                for weight in (0, .5, 1):
                    assert sparse.sequence_nll(seq, assignment, p, x, weight) == pytest.approx(dense.sequence_nll(seq, assignment, p, y, weight))
    parts = split_document_ids(docs, 42)
    for key in parts['validation'] + parts['test']:
        docs[key] = ['하하하하하하하하하']
    assert a == sparse.fit_ngrams(docs, 42, 5)


def test_seven_gram_backoff_normalized_finite_and_reindexed():
    c = sparse.fit_ngrams(documents(), 42, 7)
    assert c[7].sum() == 8  # 14-symbol paragraph, no cross-paragraph windows
    assert len(c[7].patterns) <= 8
    p = torch.full((19, 19), 1 / 19, dtype=torch.float64)
    levels = sparse.probabilities(c, p, 100, 7)
    for history in ([0, 2, 3, 5, 6, 7], [18] * 6):
        probs = [sparse.position_probabilities(history + [next_zone], levels)[-1] for next_zone in range(19)]
        assert sum(probs) == pytest.approx(1)
        assert np.isfinite(np.log(probs)).all()
    seq = np.array([0, 2, 3, 5, 6, 7, 9, 0])
    a = np.arange(19)
    reindex = np.random.default_rng(42).permutation(19)
    new_a = np.empty_like(a)
    new_a[reindex] = a
    assert sparse.sequence_nll(seq, a, p, levels, 1) == pytest.approx(sparse.sequence_nll(reindex[seq], new_a, p, levels, 1))


def test_sparse_coverage_matches_dense_and_reports_full_backoff():
    g = _window_graph('가나다라마바사', 'table', 42, .001)
    candidates = {'table': [{'matcher': name, 'predicted_assignment': g.true_zone_by_anonymous_region.tolist()}
                           for name in ('oracle_transition', 'learned_structural_local_search')]}
    a = sparse.fit_ngrams(documents(), 42, 7)
    b = dense.fit_ngrams(documents(), 42, 5)
    x = sparse.context_coverage([g], candidates, a, 5, 100)
    y = dense.context_coverage([g], candidates, b, 5, 100)
    for name in y:
        for n in y[name]:
            for key, value in y[name][n].items():
                assert x[name][n][key] == pytest.approx(value)
    # Remove all observations of high orders: every position backs off to <=5.
    for n in (6, 7):
        a[n].patterns.clear()
        a[n].histories.clear()
    report = sparse.context_coverage([g], candidates, a, 7, 100)
    assert report['truth'][7]['mean_mass_to_order_5_or_lower'] == 1
    assert report['truth'][7]['unseen_history_rate'] == 1
