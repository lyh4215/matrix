"""Hierarchically smoothed 3--5 gram models trained on Korean train paragraphs."""
from collections import Counter

import torch

from ..data.korean_corpus import digest, hangul_only, split_document_ids
from ..data.hangul_zones import syllable_to_zone


def fit_ngrams(documents, seed, max_order=5):
    if max_order not in (3, 4, 5):
        raise ValueError('max_order must be 3, 4, or 5')
    counts = {n: torch.zeros((19,) * n, dtype=torch.float64) for n in range(3, max_order + 1)}
    seen = set()
    for key in split_document_ids(documents, seed)['train']:
        for paragraph in documents[key]:
            text = hangul_only(paragraph)
            fingerprint = digest(text)
            if not text or fingerprint in seen:
                continue
            seen.add(fingerprint)
            zones = [syllable_to_zone(c) for c in text]
            for n, tensor in counts.items():
                # Flatten indices to count efficiently without millions of scalar tensor writes.
                patterns = Counter(zip(*(zones[i:] for i in range(n))))
                if patterns:
                    keys = list(patterns)
                    indices = tuple(torch.tensor([key[i] for key in keys]) for i in range(n))
                    tensor[indices] += torch.tensor(list(patterns.values()), dtype=torch.float64)
    return counts


def probabilities(counts, bigram, strength, order):
    if not 0 < strength < float('inf'):
        raise ValueError('strength must be finite and positive')
    levels = {2: bigram}
    for n in range(3, order + 1):
        c = counts[n]
        levels[n] = (c + strength * levels[n - 1].unsqueeze(0)) / (c.sum(-1, keepdim=True) + strength)
    return levels


def sequence_nll(sequence, assignment, bigram, levels, weight):
    """Use available history at sequence start; interpolate top level with bigram."""
    if not 0 <= weight <= 1:
        raise ValueError('weight must be in [0,1]')
    z = torch.as_tensor(assignment, dtype=torch.long)[torch.as_tensor(sequence, dtype=torch.long)]
    order = max(levels)
    total = torch.zeros((), dtype=torch.float64)
    for n in range(2, order + 1):
        start = n - 1
        end = len(z) if n == order else min(len(z), n)
        if start >= end:
            continue
        indices = tuple(z[start - n + 1 + i:end - n + 1 + i] for i in range(n))
        p = levels[n][indices]
        base = bigram[z[start - 1:end - 1], z[start:end]]
        total -= ((1 - weight) * base + weight * p).log().sum()
    return float(total)


def context_coverage(graphs, candidates, counts, order, strength):
    """Occurrence-weighted coverage on truth and both fixed candidates, separately.

    A seen history with unseen continuation is distinct from an unseen history.
    Backoff mass k/(history count+k) measures reliance on the next lower order.
    """
    totals = {name: {n: [0, 0, 0, 0.0] for n in range(3, order + 1)}
              for name in ('truth', 'oracle_transition', 'learned_structural_local_search')}
    histories = {n: c.sum(-1) for n, c in counts.items() if n <= order}
    for graph in graphs:
        assignments = [('truth', graph.true_zone_by_anonymous_region)] + [
            (r['matcher'], r['predicted_assignment']) for r in candidates[graph.table_id]]
        for name, assignment in assignments:
            for sequence in graph.anonymous_sequences:
                z = torch.as_tensor(assignment)[torch.as_tensor(sequence)]
                for n in range(3, order + 1):
                    size = len(z) - n + 1
                    if size <= 0:
                        continue
                    indices = tuple(z[i:i + size] for i in range(n))
                    h = histories[n][indices[:-1]]
                    row = totals[name][n]
                    row[0] += size
                    row[1] += int((h == 0).sum())
                    row[2] += int((counts[n][indices] == 0).sum())
                    row[3] += float((strength / (h + strength)).sum())
    return {name: {n: {
        'positions': r[0], 'unseen_history_rate': r[1] / r[0] if r[0] else None,
        'unseen_ngram_rate': r[2] / r[0] if r[0] else None,
        'mean_lower_order_mass': r[3] / r[0] if r[0] else None,
    } for n, r in orders.items()} for name, orders in totals.items()}
