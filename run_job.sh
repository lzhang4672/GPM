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

PY_EXEC="$HOME/apps/miniforge3/envs/clean_env/bin/python"
BASE_PY="$HOME/apps/miniforge3/bin/python"

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

ensure_fresh_env() {
  local target_env="$1"
  if [[ ! -x "$BASE_PY" ]]; then
    echo "CRITICAL: base python not found at $BASE_PY"
    exit 2
  fi

  echo "[env] creating fresh venv at $target_env"
  "$BASE_PY" -m venv "$target_env"
  local py="$target_env/bin/python"

  export PIP_CONFIG_FILE=/dev/null

  "$py" -m pip install --upgrade pip setuptools wheel
  "$py" -m pip install --no-index --find-links /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3 --find-links /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic --find-links /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic -r requirements/runtime.txt

  "$py" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
  "$py" -m pip install --no-cache-dir torch-geometric==2.6.1

  # CRITICAL: disable site pip config and indexes to avoid torch29 ComputeCanada wheels.
  "$py" -m pip install --no-index --find-links https://data.pyg.org/whl/torch-2.4.0+cu121.html pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv

  PY_EXEC="$py"
}

run_probe() {
  local out_file="$WORK_DIR/artifacts/torch_probe_${JOBID}.log"
  set +e
  "$PY_EXEC" scripts/diagnose_torch_segfault.py >"$out_file" 2>&1
  local rc=$?
  set -e
  cat "$out_file"
  return $rc
}

if [[ ! -x "$PY_EXEC" ]] || ! "$PY_EXEC" - <<'PY' >/dev/null 2>&1
import sys
print(sys.executable)
PY
then
  echo "[preflight] configured env python is unavailable/unstable; using fresh job-local env"
  ensure_fresh_env "$WORK_DIR/.gpm_env"
fi

echo "[preflight] probing torch/numpy runtime"
if ! run_probe; then
  if [[ -d "$ZE_LIB_DIR" ]] && grep -q "libze_loader.so.1" "$WORK_DIR/artifacts/torch_probe_${JOBID}.log"; then
    echo "[preflight] missing libze_loader detected; enabling ZE runtime path and retrying probe"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$ZE_LIB_DIR"
    run_probe || true
  fi
fi

if ! run_probe; then
  echo "[preflight] torch probe still failed. Rebuilding fresh job-local env and retrying..."
  ensure_fresh_env "$WORK_DIR/.gpm_env"
  echo "[preflight] re-probing torch/numpy runtime"
  run_probe
fi

"$PY_EXEC" scripts/check_runtime_deps.py --core-only || true

LOG_FILE="logs/${DATASET}_${JOBID}.log"
CSV_FILE="artifacts/gpm_results_${JOBID}.csv"

echo "Starting dataset=$DATASET seeds=$SEEDS epochs=$EPOCHS"

"$PY_EXEC" -u GPM/main.py \
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
