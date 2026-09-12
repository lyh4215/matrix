"""Train-only n-gram scoring of fixed candidates; no neural training or n-gram search."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import zipfile

import torch

from ..data.korean_corpus import (
    build_corpus_graphs, digest, hangul_only, load_korquad, load_local_documents,
    split_document_ids,
)
from ..data.hangul_zones import syllable_to_zone
from ..models.learned_structural_matcher import LearnedStructuralRegionZoneMatcher
from .region_zone_match_probe import (
    _evaluate_learned_graphs, _run_nonlearned, region_zone_probe_config_from_dict,
)
from .region_zone_local_search import evaluate_local_search


def fit_trigrams(documents: dict, seed: int) -> torch.Tensor:
    """Same train partition and normalized paragraph deduplication as corpus P."""
    counts = torch.zeros(19, 19, 19, dtype=torch.float64)
    seen = set()
    for key in split_document_ids(documents, seed)["train"]:
        for paragraph in documents[key]:
            text = hangul_only(paragraph)
            fingerprint = digest(text)
            if not text or fingerprint in seen:
                continue
            seen.add(fingerprint)
            zones = [syllable_to_zone(c) for c in text]
            for triple, n in Counter(zip(zones, zones[1:], zones[2:])).items():
                counts[triple] += n
    return counts


def trigram_probabilities(counts: torch.Tensor, bigram: torch.Tensor, strength: float) -> torch.Tensor:
    if not 0 < strength < float("inf"):
        raise ValueError("backoff strength must be finite and positive")
    # Unseen (a,b) histories back off exactly to P(c|b).
    return (counts + strength * bigram.unsqueeze(0)) / (counts.sum(-1, keepdim=True) + strength)


def sequence_nll(sequence, assignment, bigram, trigram, weight: float) -> float:
    if not 0 <= weight <= 1:
        raise ValueError("trigram weight must be in [0,1]")
    zones = torch.as_tensor(assignment, dtype=torch.long)[torch.as_tensor(sequence, dtype=torch.long)]
    if len(zones) < 2:
        return 0.0
    score = -bigram[zones[0], zones[1]].log()
    if len(zones) > 2:
        probability = ((1 - weight) * bigram[zones[1:-1], zones[2:]]
                       + weight * trigram[zones[:-2], zones[1:-1], zones[2:]])
        score = score - probability.log().sum()
    return float(score)


def diagnose(graphs, candidates, bigram, trigrams, weight, scorer=sequence_nll):
    """Truth is scored separately and never enters candidate selection."""
    rows = []
    for graph in graphs:
        predictions = candidates[graph.table_id]
        assignments = [x["predicted_assignment"] for x in predictions]
        scores = [sum(scorer(s, a, bigram, trigrams, weight)
                      for s in graph.anonymous_sequences) for a in assignments]
        winner = min(range(len(scores)), key=scores.__getitem__)  # exact ties retain oracle
        truth = graph.true_zone_by_anonymous_region
        observed = graph.observed_mask
        wrong = [i for i, a in enumerate(assignments)
                 if not torch.equal(torch.tensor(a)[observed], truth[observed])]
        true_score = sum(scorer(s, truth, bigram, trigrams, weight)
                         for s in graph.anonymous_sequences)
        gap = min((scores[i] - true_score for i in wrong), default=None)
        predicted = torch.tensor(assignments[winner])
        correct = predicted == truth
        rows.append({
            "table_id": graph.table_id, "true_nll": true_score,
            "candidate_nll": scores, "selected_source": predictions[winner]["matcher"],
            "best_wrong_minus_true_nll": gap,
            "correct_regions": int(correct[observed].sum()),
            "observed_regions": int(observed.sum()),
            "correct_tokens": int(graph.token_counts[correct].sum()),
            "tokens": int(graph.token_counts.sum()),
            "observed_exact": bool(correct[observed].all()),
        })
    gaps = [r["best_wrong_minus_true_nll"] for r in rows if r["best_wrong_minus_true_nll"] is not None]
    return {
        "tables": len(rows), "tables_with_wrong_candidate": len(gaps),
        "truth_beats_all_wrong": sum(g > 1e-8 for g in gaps),
        "truth_ties_best_wrong": sum(abs(g) <= 1e-8 for g in gaps),
        "wrong_beats_truth": sum(g < -1e-8 for g in gaps),
        "truth_win_rate": sum(g > 1e-8 for g in gaps) / len(gaps) if gaps else None,
        "mean_wrong_minus_true_nll": sum(gaps) / len(gaps) if gaps else None,
        "assignment_accuracy": sum(r["correct_regions"] for r in rows) / sum(r["observed_regions"] for r in rows),
        "token_accuracy": sum(r["correct_tokens"] for r in rows) / sum(r["tokens"] for r in rows),
        "exact_recovery_rate": sum(r["observed_exact"] for r in rows) / len(rows),
        "table_results": rows,
    }


def select_setting(validation_results):
    # Predeclared ranking; no test values are accepted here. Stable ties prefer
    # the earlier setting (bigram first, then lower strength/weight).
    return max(range(len(validation_results)), key=lambda i: (
        validation_results[i]["truth_win_rate"] if validation_results[i]["truth_win_rate"] is not None else -1,
        validation_results[i]["token_accuracy"],
    ))


def candidate_map(results):
    mapped = {}
    for result in results:
        for row in result["table_results"]:
            mapped.setdefault(row["table_id"], []).append({**row, "matcher": result["matcher"]})
    return mapped


def main(max_order=3):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-order", type=int, choices=(3, 4, 5, 6, 7), default=max_order)
    parser.add_argument("--results", required=True, help="Existing Korean corpus benchmark ZIP")
    parser.add_argument("--cache-dir", default="artifacts/corpus_cache")
    parser.add_argument("--corpus", help="Original local corpus, if the benchmark used one")
    parser.add_argument("--output-dir", default="artifacts/korean_ngram_diagnostic" if max_order > 3 else "artifacts/korean_trigram_diagnostic")
    parser.add_argument("--smoke", action="store_true", help="Only two validation/test tables and reduced oracle search; not a final experiment")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with zipfile.ZipFile(args.results) as archive:
        raw = json.loads(archive.read("raw_results.json"))
        checkpoint = torch.load(io.BytesIO(archive.read("learned_structural/checkpoint.pt")),
                                map_location="cpu", weights_only=True)
    metadata = raw["config"].get("data_source", {})
    if metadata.get("kind") != "korean_corpus":
        raise ValueError("a real Korean corpus benchmark ZIP is required")
    config = region_zone_probe_config_from_dict(raw["config"])
    documents, source = load_local_documents(args.corpus) if args.corpus else load_korquad(args.cache_dir)
    if source["sha256"] != metadata["source"]["sha256"]:
        raise ValueError("corpus fingerprint differs from the original benchmark")
    length = metadata["sequence_length"]
    p, splits, rebuilt = build_corpus_graphs(
        documents, {s: metadata["statistics"][s]["selected_windows"] for s in ("train", "validation", "test")},
        length, config.seed, metadata["canonical_alpha"], config.oracle.alpha, source,
    )
    if rebuilt["windows"] != metadata["windows"] or rebuilt["document_splits"] != metadata["document_splits"]:
        raise ValueError("regenerated windows/splits differ from saved provenance")
    if not torch.equal(p, torch.tensor(raw["canonical_transition_matrix"], dtype=torch.float64)):
        raise ValueError("train canonical P differs from saved benchmark")
    if args.max_order >= 6:
        from . import korean_sparse_ngram_scoring as ngram
    else:
        from . import korean_ngram_scoring as ngram
    print(f"Counting train-only n-grams through order {args.max_order}", flush=True)
    all_counts = ngram.fit_ngrams(documents, config.seed, args.max_order)
    counts = all_counts[3]
    learned = checkpoint["model_config"]
    kwargs = {k: learned[k] for k in ("d_model", "num_layers", "dropout", "sinkhorn_iterations", "sinkhorn_temperature")}
    structural = checkpoint["structural_config"]
    model = LearnedStructuralRegionZoneMatcher(p, num_zones=19, **kwargs,
        structural_refinement_steps=structural["structural_refinement_steps"],
        structural_beta=structural["structural_beta"])
    model.load_state_dict(checkpoint["model_state"])
    validation = splits["validation"][length]
    test = splits["test"][length]
    if args.smoke:
        validation, test = validation[:2], test[:2]
        config.oracle.max_iterations = 2
        config.oracle.restarts = 1
    print(f"Generating fixed validation candidates for {len(validation)} tables; no epoch training", flush=True)
    oracle = _run_nonlearned("oracle_transition", {length: validation}, p, config)[0]
    neural = _evaluate_learned_graphs(model, validation, learned["batch_size"], torch.device("cpu"))
    refined = evaluate_local_search(neural, validation, p, config.local_search, config.seed)
    candidates = candidate_map([oracle, refined])
    settings = [{"order": 2, "strength": 1.0, "weight": 0.0}] + [
        {"order": n, "strength": k, "weight": w} for n in range(3, args.max_order + 1)
        for k in (1.0, 10.0, 100.0) for w in (0.25, 0.5, 0.75, 1.0)]
    def evaluate(graphs, fixed_candidates, setting):
        levels = ngram.probabilities(all_counts, p, setting["strength"], setting["order"])
        result = diagnose(graphs, fixed_candidates, p, levels, setting["weight"], ngram.sequence_nll)
        result["context_coverage"] = ngram.context_coverage(
            graphs, fixed_candidates, all_counts, setting["order"], setting["strength"])
        return result

    validation_results = []
    for setting in settings:
        report = evaluate(validation, candidates, setting)
        validation_results.append({"setting": setting, **report})
        print(f"Scored validation: {setting}", flush=True)
    chosen = select_setting(validation_results)
    # Selection is finalized before any test candidate scores are computed.
    test_candidates = candidate_map([next(r for r in raw["results"] if r["matcher"] == name)
                                    for name in ("oracle_transition", "learned_structural_local_search")])
    for graph in test:
        for row in test_candidates[graph.table_id]:
            if row["true_assignment"] != graph.true_zone_by_anonymous_region.tolist():
                raise ValueError("saved test assignments differ from reconstructed graphs")
    test_results = []
    # Fixed historical trigram reference is predeclared, never chosen from test.
    trigram_reference = settings.index({"order": 3, "strength": 1.0, "weight": 1.0})
    references = [0, trigram_reference]
    if args.max_order >= 5:
        references.append(settings.index({"order": 5, "strength": 100.0, "weight": 1.0}))
    for index in dict.fromkeys([*references, chosen]):
        setting = settings[index]
        test_results.append({"setting": setting, **evaluate(test, test_candidates, setting)})
    report = {
        "source_zip": str(Path(args.results).resolve()), "corpus_source": source, "smoke": args.smoke,
        "selection_rule": "validation truth_win_rate, then validation token_accuracy; stable ties favor earlier setting",
        "selected_setting": settings[chosen], "train_trigram_count": int(counts.sum()),
        "train_ngram_counts": {n: int(c.sum()) for n, c in all_counts.items()},
        "max_order": args.max_order,
        "count_storage": "sparse" if args.max_order >= 6 else "dense",
        "observed_patterns": {n: len(c.patterns) for n, c in all_counts.items()} if args.max_order >= 6 else {},
        "validation": validation_results, "test": test_results,
        "limitations": "fixed oracle/structural-local-search candidate pair; truth scored for diagnostics only, never selected; validation reused from neural checkpoint selection; existing test was previously inspected; no n-gram search or new training",
    }
    output = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    output.mkdir(parents=True, exist_ok=False)
    (output / "diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "validation_candidates.json").write_text(json.dumps(candidates, ensure_ascii=False), encoding="utf-8")
    lines = ["# Korean n-gram diagnostic", "", f"Smoke: {args.smoke}", f"Selected: {settings[chosen]}", "",
             "Truth win = true NLL strictly below every wrong candidate (observed assignment); excludes tables with no wrong candidates.", "",
             "| Split | Order | Strength | Weight | Truth wins / eligible | Assignment | Token |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split, results in (("validation", validation_results), ("test", test_results)):
        for r in results:
            s = r["setting"]
            lines.append(f"| {split} | {s['order']} | {s['strength']} | {s['weight']} | {r['truth_beats_all_wrong']}/{r['tables_with_wrong_candidate']} | {r['assignment_accuracy']:.2%} | {r['token_accuracy']:.2%} |")
    lines += ["", "## Context coverage (validation; weight=1)", "",
              "| Order | Strength | Sequence source | Unseen history | Unseen n-gram | Lower-order mass | Mass to <=5 |",
              "|---:|---:|---|---:|---:|---:|---:|"]
    for r in validation_results:
        s = r["setting"]
        if s["weight"] != 1:
            continue
        for name, levels in r["context_coverage"].items():
            c = levels[s["order"]]
            if c["positions"]:
                lines.append(f"| {s['order']} | {s['strength']} | {name} | {c['unseen_history_rate']:.2%} | {c['unseen_ngram_rate']:.2%} | {c['mean_lower_order_mass']:.2%} | {c.get('mean_mass_to_order_5_or_lower', 1.0):.2%} |")
    lines += ["", report["limitations"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print("Results ZIP:", shutil.make_archive(str(output), "zip", root_dir=output), flush=True)


if __name__ == "__main__":
    main()
