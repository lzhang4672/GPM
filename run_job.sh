#!/bin/bash
#SBATCH --account=aip-six
#SBATCH --job-name=gpm_one
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16G
#SBATCH --time=6:00:00

set -euo pipefail

DATASET="${1:?dataset required (ba2motifs|bamultishapes|mutagenicity|bbbp|nci1|proteins_gc)}"
SEEDS="${2:-5}"
EPOCHS="${3:-1000}"

module --force purge || true
module load StdEnv/2023
module load gcc/12.3

ZE_LIB_DIR="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib64"
ORIG_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

PY_EXEC="$HOME/apps/miniforge3/envs/clean_env/bin/python"
BASE_PY_DEFAULT="$HOME/apps/miniforge3/bin/python"
BASE_PY=""

unset PYTHONPATH || true
export PYTHONNOUSERSITE=1

SOURCE_DIR="/home/lzhang46/projects/aip-six/lzhang46/GPM"
JOBID="${SLURM_JOB_ID:-manual}"
WORK_DIR="${SLURM_TMPDIR:-/tmp/$USER/$JOBID}/GPM"

DATA_PATH="/home/lzhang46/projects/aip-six/lzhang46/GPM/shared/gpm_data"
PATTERN_PATH="/home/lzhang46/projects/aip-six/lzhang46/GPM/shared/gpm_patterns"
SAVE_PATH="/home/lzhang46/projects/aip-six/lzhang46/GPM/shared/gpm_models"

mkdir -p "$WORK_DIR" "$DATA_PATH" "$PATTERN_PATH" "$SAVE_PATH"

rsync -a --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "$SOURCE_DIR/" "$WORK_DIR/"

cd "$WORK_DIR"
chmod +x scripts/diagnose_torch_segfault.py || true
mkdir -p logs artifacts

copy_back() {
  mkdir -p "$SOURCE_DIR/logs" "$SOURCE_DIR/artifacts"
  rsync -a --update "$WORK_DIR/logs/" "$SOURCE_DIR/logs/"
  rsync -a --update "$WORK_DIR/artifacts/" "$SOURCE_DIR/artifacts/"
}
trap copy_back EXIT

pick_bootstrap_python() {
  local candidates=("$BASE_PY_DEFAULT" "$(command -v python3 || true)" "/usr/bin/python3")
  for p in "${candidates[@]}"; do
    [[ -n "$p" && -x "$p" ]] || continue
    if env -u LD_LIBRARY_PATH "$p" - <<'PY' >/dev/null 2>&1
import sys
print(sys.executable)
PY
    then
      BASE_PY="$p"
      return 0
    fi
  done
  return 1
}

ensure_fresh_env() {
  local target_env="$1"
  [[ -n "$BASE_PY" ]] || { echo "CRITICAL: no stable bootstrap python found"; exit 2; }

  echo "[env] creating fresh venv at $target_env with $BASE_PY"
  env -u LD_LIBRARY_PATH "$BASE_PY" -m venv "$target_env"
  local py="$target_env/bin/python"

  export PIP_CONFIG_FILE=/dev/null

  env -u LD_LIBRARY_PATH "$py" -m pip install --upgrade pip setuptools wheel

  # Core deps from local wheelhouse; no internet dependency.
  env -u LD_LIBRARY_PATH "$py" -m pip install --no-index \
    --find-links /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3 \
    --find-links /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic \
    --find-links /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic \
    pyyaml "numpy<2" pandas scipy scikit-learn networkx fsspec tqdm psutil einops

  # Optional deps should not block job bootstrap.
  env -u LD_LIBRARY_PATH "$py" -m pip install --no-cache-dir wandb torchmetrics ogb googledrivedownloader || true

  env -u LD_LIBRARY_PATH "$py" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
  env -u LD_LIBRARY_PATH "$py" -m pip install --no-cache-dir torch-geometric==2.6.1
  env -u LD_LIBRARY_PATH "$py" -m pip install --no-index --find-links https://data.pyg.org/whl/torch-2.4.0+cu121.html pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv

  PY_EXEC="$py"
}

run_probe() {
  local out_file="$WORK_DIR/artifacts/torch_probe_${JOBID}.log"
  local use_ze="${1:-0}"
  local ld="$ORIG_LD_LIBRARY_PATH"
  [[ "$use_ze" == "1" && -d "$ZE_LIB_DIR" ]] && ld="${ld}:$ZE_LIB_DIR"

  set +e
  env LD_LIBRARY_PATH="$ld" "$PY_EXEC" scripts/diagnose_torch_segfault.py >"$out_file" 2>&1
  local rc=$?
  set -e
  cat "$out_file"
  return $rc
}

pick_bootstrap_python || { echo "CRITICAL: no working bootstrap python found."; exit 3; }

if [[ ! -x "$PY_EXEC" ]] || ! env -u LD_LIBRARY_PATH "$PY_EXEC" - <<'PY' >/dev/null 2>&1
import sys
print(sys.executable)
PY
then
  echo "[preflight] configured env python is unavailable/unstable; using fresh job-local env"
  ensure_fresh_env "$WORK_DIR/.gpm_env"
fi

echo "[preflight] probing torch/numpy runtime"
probe_ok=0
if run_probe 0; then
  probe_ok=1
elif grep -q "libze_loader.so.1" "$WORK_DIR/artifacts/torch_probe_${JOBID}.log"; then
  echo "[preflight] missing libze_loader detected; retrying probe with ZE runtime path"
  run_probe 1 && probe_ok=1 || true
fi

if [[ "$probe_ok" -ne 1 ]]; then
  echo "[preflight] torch probe still failed. Rebuilding fresh job-local env and retrying..."
  ensure_fresh_env "$WORK_DIR/.gpm_env"
  echo "[preflight] re-probing torch/numpy runtime"
  run_probe 0 || run_probe 1
fi

env -u LD_LIBRARY_PATH "$PY_EXEC" scripts/check_runtime_deps.py --core-only || true

LOG_FILE="logs/${DATASET}_${JOBID}.log"
CSV_FILE="artifacts/gpm_results_${JOBID}.csv"

echo "Starting dataset=$DATASET seeds=$SEEDS epochs=$EPOCHS"

env LD_LIBRARY_PATH="${ORIG_LD_LIBRARY_PATH}:${ZE_LIB_DIR}" "$PY_EXEC" -u GPM/main.py \
  --dataset "$DATASET" \
  --split train80_test20 \
  --split_repeat "$SEEDS" \
  --epochs "$EPOCHS" \
  --gpu 0 \
  --debug \
  --data_path "$DATA_PATH" \
  --pattern_path "$PATTERN_PATH" \
  --save_path "$SAVE_PATH" | tee "$LOG_FILE"

TEST_LINE=$(grep -E "^Test " "$LOG_FILE" | tail -1 || true)
TIME_LINE=$(grep -E "^Training time per seed \(s\):" "$LOG_FILE" | tail -1 || true)
METRIC=$(echo "$TEST_LINE" | sed -E 's/^Test ([^:]+):.*/\1/')
TEST_MEAN=$(echo "$TEST_LINE" | sed -E 's/^Test [^:]+: ([0-9.]+) ± ([0-9.]+).*/\1/')
TEST_STD=$(echo "$TEST_LINE" | sed -E 's/^Test [^:]+: ([0-9.]+) ± ([0-9.]+).*/\2/')
TIME_MEAN=$(echo "$TIME_LINE" | sed -E 's/^Training time per seed \(s\): ([0-9.]+) ± ([0-9.]+).*/\1/')
TIME_STD=$(echo "$TIME_LINE" | sed -E 's/^Training time per seed \(s\): ([0-9.]+) ± ([0-9.]+).*/\2/')

echo "job_id,dataset,seeds,epochs,metric,test_mean,test_std,time_mean_s,time_std_s,log_file" > "$CSV_FILE"
echo "${JOBID},${DATASET},${SEEDS},${EPOCHS},${METRIC},${TEST_MEAN},${TEST_STD},${TIME_MEAN},${TIME_STD},${LOG_FILE}" >> "$CSV_FILE"

echo "Done."
