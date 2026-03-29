#!/usr/bin/env python3
"""
Check runtime Python dependencies needed by GPM and print missing/crashing modules.

This script isolates each import in a subprocess so a native-extension crash
(e.g., mismatched torch/CUDA wheels) does not terminate the whole checker.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Dict, List, Tuple


# Lightweight deps that are usually safe to install via pip.
REQUIRED_CORE = {
    "yaml": "pyyaml",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "networkx": "networkx",
    "fsspec": "fsspec",
    "tqdm": "tqdm",
    "psutil": "psutil",
    "einops": "einops",
    "wandb": "wandb",
    "torchmetrics": "torchmetrics",
    "ogb": "ogb",
    "googledrivedownloader": "googledrivedownloader",
}

# Native-extension stack that commonly breaks if wheel/CUDA/Python ABI mismatch.
REQUIRED_TORCH_STACK = {
    "torch": "torch",
    "torch_geometric": "torch-geometric",
    "pyg_lib": "pyg_lib",
    "torch_cluster": "torch_cluster",
    "torch_scatter": "torch_scatter",
    "torch_sparse": "torch_sparse",
    "torch_spline_conv": "torch_spline_conv",
}


def _probe_import(module_name: str) -> Tuple[str, str]:
    """Return (status, detail) for import status of module_name.

    status in {"ok", "missing", "error", "crash"}
    """
    code = (
        "import importlib, json\n"
        f"name = {module_name!r}\n"
        "try:\n"
        "    importlib.import_module(name)\n"
        "except ModuleNotFoundError:\n"
        "    print(json.dumps({'status': 'missing'}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'status': 'error', 'detail': f'{type(e).__name__}: {e}'}))\n"
        "else:\n"
        "    print(json.dumps({'status': 'ok'}))\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode < 0:
        return "crash", f"terminated by signal {-proc.returncode}"
    if proc.returncode > 0:
        detail = proc.stderr.strip() or f"exit code {proc.returncode}"
        return "error", detail

    try:
        payload = json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError:
        detail = proc.stdout.strip() or proc.stderr.strip() or "invalid probe output"
        return "error", detail

    status = payload.get("status", "error")
    detail = payload.get("detail", "")
    return status, detail


def _check_group(group: Dict[str, str], title: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    missing: List[Tuple[str, str]] = []
    errored: List[Tuple[str, str]] = []
    crashed: List[Tuple[str, str]] = []

    print(f"\n== {title} ==")
    for module_name, package_name in group.items():
        status, detail = _probe_import(module_name)
        if status == "ok":
            print(f"[OK] {module_name}")
        elif status == "missing":
            print(f"[MISSING] {module_name} (pip package: {package_name})")
            missing.append((module_name, package_name))
        elif status == "crash":
            print(f"[CRASH] {module_name} ({detail})")
            crashed.append((module_name, package_name))
        else:
            msg = detail if detail else "import failed"
            print(f"[ERROR] {module_name} ({msg})")
            errored.append((module_name, package_name))

    return missing, errored, crashed


def main() -> int:
    parser = argparse.ArgumentParser(description="Check GPM runtime dependencies with crash isolation.")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Only check core dependencies (skip torch/PyG native extension stack).",
    )
    args = parser.parse_args()

    all_missing: List[Tuple[str, str]] = []
    all_errored: List[Tuple[str, str]] = []
    all_crashed: List[Tuple[str, str]] = []

    missing, errored, crashed = _check_group(REQUIRED_CORE, "Core Python dependencies")
    all_missing.extend(missing)
    all_errored.extend(errored)
    all_crashed.extend(crashed)

    if not args.core_only:
        missing, errored, crashed = _check_group(REQUIRED_TORCH_STACK, "Torch / PyG native stack")
        all_missing.extend(missing)
        all_errored.extend(errored)
        all_crashed.extend(crashed)

    print("\n== Summary ==")
    print(f"missing: {len(all_missing)}, errors: {len(all_errored)}, crashes: {len(all_crashed)}")

    if all_missing:
        packages = " ".join(sorted({pkg for _, pkg in all_missing}))
        print("\nInstall missing (core) packages with, for example:")
        print(f"python -m pip install {packages}")

    if all_crashed:
        print(
            "\nDetected import crashes (native-extension ABI/CUDA mismatch likely).\n"
            "Do NOT blindly 'pip install -r requirements/runtime.txt' for torch/PyG on clusters.\n"
            "Install torch first using your cluster's supported CUDA build, then install matching PyG wheels."
        )

    return 1 if (all_errored or all_crashed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
