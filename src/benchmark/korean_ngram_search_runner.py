"""Replay saved corpus candidates with fixed 5/6-gram local search, no training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import zipfile

import torch

from ..data.korean_corpus import build_corpus_graphs, load_korquad, load_local_documents
from .korean_sparse_ngram_scoring import fit_ngrams, probabilities
from .korean_ngram_search import batch_nll, refine
from .region_zone_matching import score_region_assignment, aggregate_assignment_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, help='Original Korean corpus benchmark ZIP')
    parser.add_argument('--cache-dir', default='artifacts/corpus_cache')
    parser.add_argument('--corpus')
    parser.add_argument('--output-dir', default='artifacts/korean_ngram_search')
    parser.add_argument('--max-moves', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128, help='Neighbor scoring batch, not training')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.max_moves < 1 or args.batch_size < 1:
        parser.error('max-moves and batch-size must be positive')
    torch.set_num_threads(1)
    with zipfile.ZipFile(args.results) as archive:
        raw = json.loads(archive.read('raw_results.json'))
    meta = raw['config'].get('data_source', {})
    if meta.get('kind') != 'korean_corpus':
        raise ValueError('original Korean corpus benchmark required')
    documents, source = load_local_documents(args.corpus) if args.corpus else load_korquad(args.cache_dir)
    if source['sha256'] != meta['source']['sha256']:
        raise ValueError('corpus fingerprint mismatch')
    print('Reconstructing original corpus windows and fitting train-only 3–6 gram counts', flush=True)
    p, splits, rebuilt = build_corpus_graphs(documents,
        {s: meta['statistics'][s]['selected_windows'] for s in ('train', 'validation', 'test')},
        meta['sequence_length'], raw['config']['seed'], meta['canonical_alpha'],
        raw['config']['oracle']['alpha'], source)
    if rebuilt['windows'] != meta['windows'] or rebuilt['document_splits'] != meta['document_splits']:
        raise ValueError('saved corpus windows/splits mismatch')
    if not torch.equal(p, torch.tensor(raw['canonical_transition_matrix'], dtype=torch.float64)):
        raise ValueError('canonical P mismatch')
    counts = fit_ngrams(documents, raw['config']['seed'], 6)
    names = ('oracle_transition', 'learned_structural_local_search')
    saved = {name: {t['table_id']: t for t in next(r for r in raw['results'] if r['matcher'] == name)['table_results']} for name in names}
    graphs = splits['test'][meta['sequence_length']]
    for graph in graphs:
        for name in names:
            if saved[name][graph.table_id]['true_assignment'] != graph.true_zone_by_anonymous_region.tolist():
                raise ValueError('saved assignments mismatch')
    if args.smoke:
        graphs = graphs[:2]
    max_moves = min(2, args.max_moves) if args.smoke else args.max_moves
    output = Path(args.output_dir) / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_UTC')
    output.mkdir(parents=True, exist_ok=False)
    summary = []
    for order in (5, 6):
        levels = probabilities(counts, p, 100., order)
        metrics = {name: [] for name in ('before_select', *names, 'after_select')}
        order_details = []
        for index, graph in enumerate(graphs):
            starts = [saved[name][graph.table_id]['predicted_assignment'] for name in names]
            scores = batch_nll(graph.anonymous_sequences, starts, levels)
            before = int(scores.argmin())
            results = [refine(graph.anonymous_sequences, a, levels, max_moves=max_moves, batch_size=args.batch_size) for a in starts]
            winner = min(range(2), key=lambda i: results[i]['final_nll'])  # exact ties retain oracle start
            assignments = [starts[before], results[0]['assignment'], results[1]['assignment'], results[winner]['assignment']]
            for name, a in zip(metrics, assignments):
                metrics[name].append(score_region_assignment(graph, a))
            truth_nll = float(batch_nll(graph.anonymous_sequences, [graph.true_zone_by_anonymous_region.tolist()], levels)[0])
            row = {'table_id': graph.table_id, 'order': order, 'starts': dict(zip(names, results)),
                   'selected_source_before': names[before], 'selected_source_after': names[winner],
                   'true_nll': truth_nll, 'selected_gap_to_true': results[winner]['final_nll'] - truth_nll,
                   'assignment_delta': metrics['after_select'][-1]['observed_region_assignment_accuracy'] - metrics['before_select'][-1]['observed_region_assignment_accuracy'],
                   'token_delta': metrics['after_select'][-1]['token_accuracy'] - metrics['before_select'][-1]['token_accuracy'],
                   'metrics': {name: {k: m[-1][k] for k in ('observed_region_assignment_accuracy', 'token_accuracy', 'observed_exact_recovery')} for name, m in metrics.items()}}
            order_details.append(row)
            with (output / 'table_results.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row) + '\n')
            print(f'order={order} table={index+1}/{len(graphs)} NLL {scores[before]:.3f} -> {results[winner]["final_nll"]:.3f}', flush=True)
        aggregated = {name: aggregate_assignment_results(rows, 19) for name, rows in metrics.items()}
        searches = [s for row in order_details for s in row['starts'].values()]
        summary.append({'order': order, 'strength': 100., 'weight': 1., 'metrics': aggregated,
            'total_search_seconds': sum(s['seconds'] for s in searches),
            'mean_search_seconds_per_table': sum(s['seconds'] for s in searches) / len(graphs),
            'convergence_rate_per_start': sum(s['converged'] for s in searches) / len(searches),
            'mean_accepted_moves_per_start': sum(s['accepted_moves'] for s in searches) / len(searches),
            'tables_assignment_improved': sum(r['assignment_delta'] > 0 for r in order_details),
            'tables_assignment_worsened': sum(r['assignment_delta'] < 0 for r in order_details),
            'fraction_selected_nll_le_true': sum(r['selected_gap_to_true'] <= 1e-8 for r in order_details) / len(graphs)})
    report = {'source_zip': str(Path(args.results).resolve()), 'corpus_source': source,
        'smoke': args.smoke, 'max_moves': max_moves, 'batch_size': args.batch_size,
        'primary_order': 6, 'comparison_order': 5, 'summary': summary,
        'limitations': 'Fixed previously validation-selected settings; previously inspected test; no test tuning. Convergence is only pair/3-cycle local optimality. Scores and timing exclude corpus preparation. No neural training.'}
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Korean n-gram local search', '', f'Smoke: {args.smoke}; primary=6, comparison=5; strength=100, weight=1', '',
             '| Order | Method | Assignment | Token | Exact |', '|---:|---|---:|---:|---:|']
    for row in summary:
        for name, m in row['metrics'].items():
            lines.append(f"| {row['order']} | {name} | {m['observed_region_assignment_accuracy']:.2%} | {m['token_accuracy']:.2%} | {m['observed_exact_recovery_rate']:.2%} |")
    for row in summary:
        lines += ['', f"Order {row['order']}: search {row['mean_search_seconds_per_table']:.2f}s/table, convergence {row['convergence_rate_per_start']:.1%}, assignment improved/worsened {row['tables_assignment_improved']}/{row['tables_assignment_worsened']}", '']
    lines += [report['limitations']]
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print('Results ZIP:', shutil.make_archive(str(output), 'zip', root_dir=output), flush=True)


if __name__ == '__main__':
    main()
