#!/usr/bin/env python3
"""
Check runtime Python dependencies needed by GPM and print missing packages.
"""

from importlib import import_module


REQUIRED = {
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
    "torch": "torch",
    "torch_geometric": "torch-geometric",
    "torch_cluster": "torch_cluster",
    "torch_scatter": "torch_scatter",
    "torch_sparse": "torch_sparse",
    "torch_spline_conv": "torch_spline_conv",
}


def main():
    missing = []
    for module_name, package_name in REQUIRED.items():
        try:
            import_module(module_name)
        except Exception:
            missing.append((module_name, package_name))

    if not missing:
        print("All runtime dependencies are available.")
        return

    print("Missing runtime dependencies:")
    for module_name, package_name in missing:
        print(f"  - module '{module_name}' (install package: {package_name})")

    packages = " ".join(sorted({pkg for _, pkg in missing}))
    print("\nInstall command:")
    print(f"python -m pip install {packages}")


if __name__ == "__main__":
    main()
