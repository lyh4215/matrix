"""Sparse observed n-gram counts with recursive backoff through order seven."""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..data.korean_corpus import digest, hangul_only, split_document_ids
from ..data.hangul_zones import syllable_to_zone


@dataclass
class SparseCounts:
    patterns: Counter
    histories: Counter

    def sum(self):
        return sum(self.patterns.values())


def encode_windows(zones, order):
    z = np.asarray(zones, dtype=np.int64)
    size = max(0, len(z) - order + 1)
    keys = np.zeros(size, dtype=np.int64)
    for i in range(order):
        keys = keys * 19 + z[i:i + size]
    return keys


def fit_ngrams(documents, seed, max_order=7):
    if max_order not in range(3, 8):
        raise ValueError('max_order must be between 3 and 7')
    counts = {n: SparseCounts(Counter(), Counter()) for n in range(3, max_order + 1)}
    seen = set()
    for key in split_document_ids(documents, seed)['train']:
        for paragraph in documents[key]:
            text = hangul_only(paragraph)
            fingerprint = digest(text)
            if not text or fingerprint in seen:
                continue
            seen.add(fingerprint)
            z = [syllable_to_zone(c) for c in text]
            for n, c in counts.items():
                keys, frequencies = np.unique(encode_windows(z, n), return_counts=True)
                c.patterns.update({int(k): int(v) for k, v in zip(keys, frequencies)})
    # History totals exclude paragraph-end histories without a continuation.
    for c in counts.values():
        for key, value in c.patterns.items():
            c.histories[key // 19] += value
    return counts


def lookup(counter, keys):
    return np.fromiter((counter.get(int(k), 0) for k in keys), dtype=np.float64, count=len(keys))


def probabilities(counts, bigram, strength, order):
    if not 0 < strength < float('inf'):
        raise ValueError('strength must be finite and positive')
    return counts, np.asarray(bigram), strength, order


def position_probabilities(z, levels):
    counts, bigram, k, order = levels
    z = np.asarray(z, dtype=np.int64)
    result = bigram[z[:-1], z[1:]].copy()
    for n in range(3, min(order, len(z)) + 1):
        keys = encode_windows(z, n)
        c = counts[n]
        # The same positions currently hold P(n-1), with the earliest history removed.
        result[n - 2:] = (lookup(c.patterns, keys) + k * result[n - 2:]) / (lookup(c.histories, keys // 19) + k)
    return result


def sequence_nll(sequence, assignment, bigram, levels, weight):
    if not 0 <= weight <= 1:
        raise ValueError('weight must be in [0,1]')
    z = np.asarray(assignment, dtype=np.int64)[np.asarray(sequence, dtype=np.int64)]
    base = np.asarray(bigram)[z[:-1], z[1:]]
    return float(-np.log((1 - weight) * base + weight * position_probabilities(z, levels)).sum())


def context_coverage(graphs, candidates, counts, order, strength):
    totals = {name: {n: np.zeros(5) for n in range(3, order + 1)}
              for name in ('truth', 'oracle_transition', 'learned_structural_local_search')}
    for graph in graphs:
        assignments = [('truth', graph.true_zone_by_anonymous_region)] + [
            (r['matcher'], r['predicted_assignment']) for r in candidates[graph.table_id]]
        for name, assignment in assignments:
            for sequence in graph.anonymous_sequences:
                z = np.asarray(assignment)[np.asarray(sequence)]
                masses = {}
                for n in range(3, min(order, len(z)) + 1):
                    keys = encode_windows(z, n)
                    h = lookup(counts[n].histories, keys // 19)
                    mass = strength / (h + strength)
                    masses[n] = mass
                    row = totals[name][n]
                    # For order>5: mass passing through every high-order level to <=5.
                    to_five = np.ones(len(keys))
                    for m in range(6, n + 1):
                        to_five *= masses[m][n - m:]
                    row += [len(keys), (h == 0).sum(), (lookup(counts[n].patterns, keys) == 0).sum(), mass.sum(), to_five.sum()]
    return {name: {n: {
        'positions': int(r[0]),
        'unseen_history_rate': r[1] / r[0] if r[0] else None,
        'unseen_ngram_rate': r[2] / r[0] if r[0] else None,
        'mean_lower_order_mass': r[3] / r[0] if r[0] else None,
        'mean_mass_to_order_5_or_lower': r[4] / r[0] if r[0] else None,
    } for n, r in orders.items()} for name, orders in totals.items()}
