#!/usr/bin/env python3
"""Diagnose torch import crashes without crashing the parent process."""

from __future__ import annotations

import json
import subprocess
import sys


def run_probe() -> tuple[int, str, str]:
    code = r'''
import json
import sys

payload = {"python": sys.executable}

# torch import is the critical gate.
try:
    import torch
except Exception as e:
    print(json.dumps({"error": f"TORCH_IMPORT: {type(e).__name__}: {e}"}))
    raise SystemExit(3)

payload["torch"] = torch.__version__
payload["cuda_built"] = bool(torch.backends.cuda.is_built())
payload["torch_cuda"] = str(torch.version.cuda)
payload["cuda_available"] = bool(torch.cuda.is_available())
if torch.cuda.is_available():
    payload["device0"] = torch.cuda.get_device_name(0)

# numpy is informative only; do not fail probe on numpy import errors.
try:
    import numpy
    payload["numpy"] = numpy.__version__
except Exception as e:
    payload["numpy_error"] = f"{type(e).__name__}: {e}"

print(json.dumps(payload))
'''
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def main() -> int:
    rc, out, err = run_probe()

    if rc == 0:
        print("Torch probe OK")
        if out:
            try:
                print(json.dumps(json.loads(out), indent=2))
            except json.JSONDecodeError:
                print(out)
        return 0

    print("Torch probe FAILED")
    if rc < 0:
        print(f"- crashed with signal {-rc} (segfault is signal 11)")
    else:
        print(f"- exited with code {rc}")
    if err:
        print(f"- stderr: {err}")
    if out:
        print(f"- stdout: {out}")

    print("\nLikely cause: torch/PyG/CUDA native binary mismatch in this environment.")
    print("Suggested clean reinstall (CUDA 12.1 example used by this repo):")
    print("1) python -m pip uninstall -y torch torchvision torchaudio torch-geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv")
    print("2) python -m pip install --no-cache-dir 'numpy<2'")
    print("3) python -m pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121")
    print("4) python -m pip install --no-cache-dir torch-geometric==2.6.1")
    print("5) python -m pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html")
    print("6) python scripts/diagnose_torch_segfault.py")
    print("\nIf your cluster is not CUDA 12.1, change both torch index URL and PyG wheel URL to your CUDA version.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
