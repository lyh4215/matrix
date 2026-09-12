from itertools import permutations

import numpy as np
import pytest
import torch

from src.benchmark.korean_sparse_ngram_scoring import fit_ngrams, probabilities, sequence_nll
from src.benchmark.korean_ngram_search import batch_nll, refine, neighbors


def levels():
    docs = {str(i): ['가나다가나다가나다가나다'] for i in range(10)}
    counts = fit_ngrams(docs, 42, 6)
    p = torch.rand(19, 19, generator=torch.Generator().manual_seed(42), dtype=torch.float64)
    p /= p.sum(-1, keepdim=True)
    return probabilities(counts, p, 100., 6)


def test_batch_matches_scalar_all_orders_and_sequence_boundaries():
    full = levels()
    rng = np.random.default_rng(42)
    assignments = np.array([rng.permutation(19) for _ in range(8)])
    for order in (5, 6):
        model = (*full[:3], order)
        for weight in (0, .5, 1):
            seqs = [[], [1], [0, 2], [0, 2, 3, 0, 2, 3, 0]]
            actual = batch_nll(seqs, assignments, model, weight)
            expected = [sum(sequence_nll(s, a, full[1], model, weight) for s in seqs) for a in assignments]
            assert actual == pytest.approx(expected)


def test_search_preserves_permutation_decreases_nll_and_converges_locally():
    model = levels()
    seqs = [[0, 1, 2, 0, 2, 1] * 5]
    result = refine(seqs, [2, 0, 1], model)
    assert sorted(result['assignment']) == [0, 1, 2]
    assert result['converged']
    assert all(b < a - 1e-8 for a, b in zip(result['nll_trace'], result['nll_trace'][1:]))
    scores = batch_nll(seqs, list(permutations(range(3))), model)
    assert result['final_nll'] == pytest.approx(float(scores.min()))
    assert result['final_nll'] <= result['initial_nll'] + 1e-8
    assert result['final_nll'] == pytest.approx(batch_nll(seqs, [result['assignment']], model)[0])


def test_unobserved_region_exchange_and_both_cycle_directions():
    a = np.arange(4)
    observed = np.array([True, True, False, False])
    pairs = list(neighbors(a, observed))
    assert len(pairs) == 5
    assert any(np.array_equal(x, [2, 1, 0, 3]) for x in pairs)
    cycles = list(neighbors(a, observed, cycles=True))
    assert len(cycles) == 8
    assert any(np.array_equal(x, [2, 0, 1, 3]) for x in cycles)
    assert any(np.array_equal(x, [1, 2, 0, 3]) for x in cycles)


def test_move_cap_and_chunk_size_agree():
    seqs = [[0, 1, 2, 0, 1, 2] * 5]
    model = levels()
    a = refine(seqs, [2, 1, 0], model, max_moves=1, batch_size=1)
    b = refine(seqs, [2, 1, 0], model, max_moves=1, batch_size=128)
    assert a['assignment'] == b['assignment']
    assert a['final_nll'] == pytest.approx(b['final_nll'])
    if a['accepted_moves'] == 1:
        assert not a['converged']
    with pytest.raises(ValueError):
        refine(seqs, [0, 0, 1], model)
