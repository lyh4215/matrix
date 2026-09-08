from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from .region_zone_match_probe import (
    RegionZoneProbeConfig,
    _apply_smoke_settings,
    load_region_zone_probe_config,
    run_region_zone_match_probe,
)


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/region_zone_benchmark.yaml"


def run_benchmark(config: RegionZoneProbeConfig) -> dict:
    """Train each length independently and bundle paired comparison results."""
    config.validate()
    lengths = config.synthetic.sequence_lengths
    if len(set(lengths)) != len(lengths):
        raise ValueError("benchmark sequence lengths must be unique")
    if config.learned.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU required: select a Colab GPU runtime, or use --smoke / --device cpu")
    # Small CPU graph-search tensors suffer from excessive thread overhead.
    torch.set_num_threads(1)
    run_dir = Path(config.output_dir).resolve() / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f_UTC"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=DEFAULT_CONFIG.parent.parent,
        capture_output=True, text=True, check=False,
    )
    manifest = {
        "training_mode": "independent_per_length",
        "config": config.to_dict(),
        "environment": {
            "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
            "torch": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "runs": [],
        "results": [],
    }
    manifest_path = run_dir / "summary.json"

    def save_manifest() -> None:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)

    save_manifest()
    for index, length in enumerate(lengths, start=1):
        current = copy.deepcopy(config)
        current.synthetic.sequence_lengths = (length,)
        current.output_dir = str(run_dir / f"length_{length}")
        print(f"\n[{index}/{len(lengths)}] Independent benchmark: length={length}", flush=True)
        result = run_region_zone_match_probe(current)
        manifest["runs"].append({"sequence_length": length, "paths": result["paths"]})
        for row in result["results"]:
            search = row.get("local_search", {})
            manifest["results"].append({
                "matcher": row["matcher"],
                "sequence_length": length,
                "assignment_accuracy": row["observed_region_assignment_accuracy"],
                "token_accuracy": row["token_accuracy"],
                "exact_recovery_rate": row["observed_exact_recovery_rate"],
                "assignment_delta": search.get("assignment_accuracy_delta"),
                "token_delta": search.get("token_accuracy_delta"),
                "tables_improved": search.get("tables_assignment_improved"),
                "tables_worsened": search.get("tables_assignment_worsened"),
                "nll_gain_per_transition": search.get("mean_nll_improvement_per_transition"),
                "search_seconds_per_table": search.get("mean_seconds_per_table"),
            })
        # Preserve completed lengths if a later run is interrupted.
        save_manifest()
        with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest["results"][0]))
            writer.writeheader()
            writer.writerows(manifest["results"])
    archive = shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
    print("\nMatcher                                  Length  Assignment  Token", flush=True)
    for row in manifest["results"]:
        print(f"{row['matcher']:<40} {row['sequence_length']:>6}  "
              f"{row['assignment_accuracy']:>9.2%}  {row['token_accuracy']:>6.2%}", flush=True)
    print(f"\nSummary: {manifest_path}\nResults ZIP: {archive}", flush=True)
    return {"summary": str(manifest_path), "archive": archive, "results": manifest["results"]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ready-to-run GPU benchmark: independently train learned/structural at 128 and 256, then compare local search",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--sequence-lengths", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-tables", type=int)
    parser.add_argument("--validation-tables", type=int)
    parser.add_argument("--test-tables", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir")
    parser.add_argument("--include-oracle", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny CPU check: 4 zones, length 16, one epoch")
    args = parser.parse_args()
    config = load_region_zone_probe_config(args.config)
    if args.smoke:
        _apply_smoke_settings(config)
        config.matchers = ("learned", "learned_structural")
        config.learned.device = "cpu"
    if args.sequence_lengths is not None:
        config.synthetic.sequence_lengths = tuple(args.sequence_lengths)
    for name in ("train_tables", "validation_tables", "test_tables"):
        if getattr(args, name) is not None:
            setattr(config.synthetic, name, getattr(args, name))
    for name in ("epochs", "batch_size", "device"):
        if getattr(args, name) is not None:
            setattr(config.learned, name, getattr(args, name))
    if args.seed is not None:
        config.seed = args.seed
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.include_oracle and "oracle_transition" not in config.matchers:
        config.matchers = ("oracle_transition", *config.matchers)
    run_benchmark(config)


if __name__ == "__main__":
    main()
