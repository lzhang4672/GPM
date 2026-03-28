#!/usr/bin/env python3
"""
Run GPM baseline experiments for the six main graph-classification datasets
using an 80/20 split and multiple random seeds, then aggregate results.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = [
    ("Ba2Motifs", "ba2motifs"),
    ("BAMultiShapes", "bamultishapes"),
    ("Mutagenicity", "mutagenicity"),
    ("BBBP", "bbbp"),
    ("NCI1", "nci1"),
    ("PROTEINS", "proteins_gc"),
]


def parse_metrics(stdout: str):
    test_match = re.search(r"Test\s+([a-zA-Z0-9@_]+):\s*([0-9.]+)\s*±\s*([0-9.]+)", stdout)
    time_match = re.search(r"Training time per seed \(s\):\s*([0-9.]+)\s*±\s*([0-9.]+)", stdout)

    if test_match is None:
        raise RuntimeError("Could not parse test metric from stdout.")
    if time_match is None:
        raise RuntimeError("Could not parse training time from stdout.")

    metric_name, test_mean, test_std = test_match.groups()
    time_mean, time_std = time_match.groups()
    return {
        "metric": metric_name,
        "test_mean": float(test_mean),
        "test_std": float(test_std),
        "train_time_mean_sec": float(time_mean),
        "train_time_std_sec": float(time_std),
    }


def run_one(dataset_key: str, seeds: int, gpu: int, epochs: int):
    cmd = [
        sys.executable,
        "GPM/main.py",
        "--dataset",
        dataset_key,
        "--split",
        "train80_test20",
        "--split_repeat",
        str(seeds),
        "--gpu",
        str(gpu),
        "--epochs",
        str(epochs),
        "--debug",
    ]

    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
    return parse_metrics(proc.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, choices=[3, 5])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--out_csv", type=Path, default=Path("artifacts/gpm_baseline_results.csv"))
    parser.add_argument("--out_json", type=Path, default=Path("artifacts/gpm_baseline_results.json"))
    args = parser.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for display_name, dataset_key in DEFAULT_DATASETS:
        print(f"[RUN] {display_name} ({dataset_key})")
        metrics = run_one(dataset_key, args.seeds, args.gpu, args.epochs)
        row = {"dataset": display_name, "dataset_key": dataset_key, **metrics}
        rows.append(row)
        print(
            f"  -> {metrics['metric']}: {metrics['test_mean']:.2f} ± {metrics['test_std']:.2f}, "
            f"time: {metrics['train_time_mean_sec']:.2f} ± {metrics['train_time_std_sec']:.2f}s"
        )

    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "dataset_key",
                "metric",
                "test_mean",
                "test_std",
                "train_time_mean_sec",
                "train_time_std_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with args.out_json.open("w") as f:
        json.dump(rows, f, indent=2)

    print(f"[DONE] Wrote {args.out_csv} and {args.out_json}")


if __name__ == "__main__":
    main()
