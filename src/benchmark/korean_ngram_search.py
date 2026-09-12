"""Label-free pair-swap and directed 3-cycle descent under a fixed n-gram NLL."""
from itertools import combinations
import time

import numpy as np

from .korean_sparse_ngram_scoring import lookup


def batch_nll(sequences, assignments, levels, weight=1.0):
    counts, bigram, strength, order = levels
    assignments = np.asarray(assignments, dtype=np.int64)
    if assignments.ndim != 2 or not 0 <= weight <= 1:
        raise ValueError('assignments must be a batch and weight in [0,1]')
    total = np.zeros(len(assignments))
    for sequence in sequences:
        z = assignments[:, np.asarray(sequence, dtype=np.int64)]
        base = bigram[z[:, :-1], z[:, 1:]]
        probability = base.copy()
        for n in range(3, min(order, z.shape[1]) + 1):
            size = z.shape[1] - n + 1
            keys = np.zeros((len(z), size), dtype=np.int64)
            for i in range(n):
                keys = keys * 19 + z[:, i:i + size]
            # Shared n-grams across neighboring permutations are queried once.
            unique, inverse = np.unique(keys, return_inverse=True)
            c = lookup(counts[n].patterns, unique)[inverse].reshape(keys.shape)
            h = lookup(counts[n].histories, unique // 19)[inverse].reshape(keys.shape)
            probability[:, n - 2:] = (c + strength * probability[:, n - 2:]) / (h + strength)
        total -= np.log((1 - weight) * base + weight * probability).sum(1)
    return total


def neighbors(assignment, observed, cycles=False):
    for indices in combinations(range(len(assignment)), 3 if cycles else 2):
        if not any(observed[i] for i in indices):
            continue
        for shift in ((1, 2) if cycles else (1,)):
            candidate = assignment.copy()
            candidate[list(indices)] = np.roll(assignment[list(indices)], shift)
            yield candidate


def refine(sequences, initial, levels, *, max_moves=50, batch_size=128, weight=1.0):
    """Best pair move first; if none improves, best directed 3-cycle.

    Missing regions remain swappable with observed ones. No truth enters this API.
    A move cap does not imply convergence. Strict decrease prevents cycling.
    """
    if max_moves < 1 or batch_size < 1:
        raise ValueError('max_moves and batch_size must be positive')
    current = np.asarray(initial, dtype=np.int64)
    if sorted(current.tolist()) != list(range(len(current))):
        raise ValueError('initial assignment must be a permutation')
    sequences = [np.asarray(s, dtype=np.int64) for s in sequences]
    observed = np.zeros(len(current), dtype=bool)
    for s in sequences:
        observed[s] = True
    started = time.perf_counter()
    score = float(batch_nll(sequences, current[None], levels, weight)[0])
    initial_score = score
    trace = [score]
    evaluations, pair_moves, cycle_moves = 1, 0, 0
    converged = False
    for _ in range(max_moves):
        accepted = False
        for cycles in (False, True):
            best, best_score = current, score
            pending = []

            def consider(batch):
                nonlocal best, best_score, evaluations
                values = batch_nll(sequences, batch, levels, weight)
                evaluations += len(batch)
                index = int(values.argmin())
                if values[index] < best_score - 1e-8:
                    best, best_score = batch[index], float(values[index])

            for candidate in neighbors(current, observed, cycles):
                pending.append(candidate)
                if len(pending) == batch_size:
                    consider(pending)
                    pending = []
            if pending:
                consider(pending)
            if best_score < score - 1e-8:
                current, score = best, best_score
                trace.append(score)
                cycle_moves += int(cycles)
                pair_moves += int(not cycles)
                accepted = True
                break
        if not accepted:
            converged = True
            break
    return {'assignment': current.tolist(), 'initial_nll': initial_score, 'final_nll': score,
            'accepted_moves': len(trace) - 1, 'pair_moves': pair_moves, 'cycle_moves': cycle_moves,
            'converged': converged, 'nll_trace': trace, 'evaluations': evaluations,
            'seconds': time.perf_counter() - started}
