from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..data.korean_corpus import build_corpus_graphs, load_korquad, load_local_documents
from .region_zone_match_probe import load_region_zone_probe_config, run_region_zone_match_probe


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark real Korean text at 256 Hangul syllables; train-only empirical P")
    parser.add_argument("--config", default=str(ROOT / "configs/korean_corpus_benchmark.yaml"))
    parser.add_argument("--corpus", help="Optional JSONL {id,text} or directory of .txt documents; default downloads KorQuAD contexts")
    parser.add_argument("--cache-dir", default=str(ROOT / "artifacts/corpus_cache"))
    parser.add_argument("--output-dir")
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--train-tables", type=int)
    parser.add_argument("--validation-tables", type=int)
    parser.add_argument("--test-tables", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--canonical-alpha", type=float, default=0.1)
    parser.add_argument("--prepare-only", action="store_true", help="Download/split/audit corpus without training or oracle search")
    parser.add_argument("--smoke", action="store_true", help="Real corpus, 19 zones, 256 syllables, 4/2/2 tables, one CPU epoch")
    args = parser.parse_args()
    config = load_region_zone_probe_config(args.config)
    if args.smoke:
        config.synthetic.train_tables = 4
        config.synthetic.validation_tables = config.synthetic.test_tables = 2
        config.learned.epochs = 1
        config.learned.batch_size = 2
        config.learned.d_model = 8
        config.learned.num_layers = 1
        config.learned.dropout = 0.0
        config.learned.sinkhorn_iterations = 20
        config.learned.device = "cpu"
        config.oracle.max_iterations = 10
        config.oracle.restarts = 1
    for name in ("train_tables", "validation_tables", "test_tables"):
        if getattr(args, name) is not None:
            setattr(config.synthetic, name, getattr(args, name))
    for name in ("epochs", "batch_size", "device"):
        if getattr(args, name) is not None:
            setattr(config.learned, name, getattr(args, name))
    if args.seed is not None:
        config.seed = args.seed
    if args.sequence_length is not None:
        config.synthetic.sequence_lengths = (args.sequence_length,)
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    config.validate()
    if config.synthetic.num_zones != 19 or len(config.synthetic.sequence_lengths) != 1 or config.synthetic.sequences_per_length != 1:
        parser.error("corpus benchmark requires 19 choseong zones, one length, and one window per table")
    if not args.prepare_only and config.learned.device == "cuda" and not torch.cuda.is_available():
        parser.error("select a GPU runtime, or use --prepare-only / --smoke / --device cpu")
    torch.set_num_threads(1)
    documents, source = load_local_documents(args.corpus) if args.corpus else load_korquad(args.cache_dir)
    graph_data = build_corpus_graphs(
        documents,
        {"train": config.synthetic.train_tables, "validation": config.synthetic.validation_tables, "test": config.synthetic.test_tables},
        length=config.synthetic.sequence_lengths[0], seed=config.seed,
        canonical_alpha=args.canonical_alpha, graph_alpha=config.oracle.alpha, source=source,
    )
    run_dir = Path(config.output_dir).resolve() / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    run_dir.mkdir(parents=True, exist_ok=False)
    config.output_dir = str(run_dir)
    metadata = graph_data[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    metadata["environment"] = {
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (run_dir / "corpus_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps({"source": source, "statistics": metadata["statistics"]}, ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        print(f"Corpus audit complete, no training: {run_dir}", flush=True)
        return
    result = run_region_zone_match_probe(config, graph_data=graph_data)
    summary = [{
        "matcher": row["matcher"], "sequence_length": row["sequence_length"],
        "observed_assignment_accuracy": row["observed_region_assignment_accuracy"],
        "token_accuracy": row["token_accuracy"], "exact_recovery_rate": row["observed_exact_recovery_rate"],
        "mean_region_coverage": row["mean_region_coverage"],
    } for row in result["results"]]
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    archive = shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
    print("\nReal Korean corpus results (zone recovery, not full syllable decoding):", flush=True)
    for row in summary:
        print(f"{row['matcher']:<40} assignment={row['observed_assignment_accuracy']:.2%} "
              f"token={row['token_accuracy']:.2%} coverage={row['mean_region_coverage']:.2%}", flush=True)
    print(f"Summary: {run_dir / 'summary.md'}\nResults ZIP: {archive}", flush=True)


if __name__ == "__main__":
    main()
