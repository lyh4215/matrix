"""Compare existing Colab predictions with oracle without retraining."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from src.benchmark.region_zone_refine import refine_saved_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Length-specific raw_results.json from the original benchmark")
    parser.add_argument("--output-dir", help="Default: oracle_comparison next to the source results")
    args = parser.parse_args()
    source = Path(args.results).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else source.parent / "oracle_comparison"
    if output.exists() and any(output.iterdir()):
        parser.error("output directory is not empty; choose a new --output-dir")
    torch.set_num_threads(1)
    # Retain the original local-search settings when replaying the experiment.
    from src.benchmark.region_zone_match_probe import region_zone_probe_config_from_dict
    config = region_zone_probe_config_from_dict(json.loads(source.read_text())["config"])
    result = refine_saved_results(source, output, search=config.local_search, compare_oracle=True)
    for row in result["results"]:
        print(f"{row['matcher']}: assignment={row['observed_region_assignment_accuracy']:.2%}, "
              f"token={row['token_accuracy']:.2%}, exact={row['observed_exact_recovery_rate']:.2%}")
    archive = shutil.make_archive(str(output), "zip", root_dir=output)
    print(f"Summary: {result['paths']['summary_markdown']}\nResults ZIP: {archive}")


if __name__ == "__main__":
    main()
