#!/usr/bin/env python3
"""
Pre-download and preprocess datasets into a persistent directory so SLURM jobs
can run without external network access.
"""

import argparse
import os
import os.path as osp
import sys
import yaml


REVIEW_DATASETS = ["ba2motifs", "bamultishapes", "mutagenicity", "bbbp", "nci1", "proteins_gc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=REVIEW_DATASETS)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--pattern_path", required=True)
    parser.add_argument("--split", default="train80_test20")
    parser.add_argument("--split_repeat", type=int, default=1)
    parser.add_argument("--node_pe", default="rw", choices=["rw", "lap", "none"])
    parser.add_argument("--node_pe_dim", type=int, default=8)
    args = parser.parse_args()

    repo_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    sys.path.append(osp.join(repo_root, "GPM"))

    from data.pyg_data_loader import load_data

    config_path = osp.join(repo_root, "config", "data.yaml")
    with open(config_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.pattern_path, exist_ok=True)

    for dataset in args.datasets:
        if dataset not in data_cfg:
            raise ValueError(f"Unknown dataset key: {dataset}")

        params = {
            "dataset": dataset,
            "task": data_cfg[dataset]["task"],
            "metric": data_cfg[dataset]["metric"],
            "num_tasks": data_cfg[dataset].get("num_tasks", None),
            "data_path": args.data_path,
            "pattern_path": args.pattern_path,
            "split": args.split,
            "split_repeat": args.split_repeat,
            "node_pe": args.node_pe,
            "node_pe_dim": args.node_pe_dim,
        }

        print(f"[CACHE] {dataset} -> {args.data_path}")
        load_data(params)

    print("[DONE] Dataset cache complete.")


if __name__ == "__main__":
    main()
