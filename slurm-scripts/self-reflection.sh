#!/usr/bin/env bash
#SBATCH --job-name=multi-turn-revision
#SBATCH --output=slurm-logs/multi-turn-revision.%j.out
#SBATCH --error=slurm-logs/multi-turn-revision.%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G

set -euxo pipefail

# ─── Which algorithms to run ────────────────────────────────────────────
ALGORITHMS=(ae lzw rle huffman)

WORKDIR=$(pwd)
CONTAINER=/projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.06.sif
RUN_PY=/py3.10/bin/python

# ─── One-time environment setup ─────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate round-trip-myenv 2>/dev/null \
  || { conda create -y -n round-trip-myenv python=3.10 && conda activate round-trip-myenv; }

LOCKFILE="${WORKDIR}/.env_installed"
if [[ ! -f "$LOCKFILE" ]]; then
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
  touch "$LOCKFILE"
fi

nvidia-smi --list-gpus


# ─── Inference + Verification Loop ─────────────────────────────
# export RAY_TMPDIR=/tmp/ray
# ray start --head --temp-dir /tmp/ray --dashboard-host 0.0.0.0  

# ─── Loop over algorithms, sequentially ─────────────────────────────────
# for ALGO in "${ALGORITHMS[@]}"; do
#   DATADIR="${WORKDIR}/processed_datasets/${ALGO}"
#   LOGDIR="${WORKDIR}/logs/${ALGO}"
#   mkdir -p "$DATADIR" "$LOGDIR"


for ALGO in "${ALGORITHMS[@]}"; do
  LOGDIR="${WORKDIR}/logs/${ALGO}-reflect"
  mkdir -p "$LOGDIR"

  for INPUT in "processed_datasets/${ALGO}"/*.jsonl; do
    BASENAME="$(basename "${INPUT}" .jsonl)"
    OUTPUT="processed_datasets/${ALGO}/${BASENAME}_reflected.jsonl"
    MODEL="$(python - "${INPUT}" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        print(obj.get("model",""))
        break
PY
)"
    TEMP="$(basename "${INPUT}" | sed -n 's/.*_temp_\([0-9.]\+\)_.*/\1/p')"

    # translation step: uses all 8 GPUs (no manual subset math)
    singularity exec --nv \
      --bind /home/u5u/nmaveli.u5u/projects/round-trip-code-compression:/workspace \
      /projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.06.sif \
      bash -lc "cd /workspace && \
        /py3.10/bin/python src/self_reflection.py \
          --input ${INPUT} \
          --output ${OUTPUT} \
          --model ${MODEL} \
          --temperature ${TEMP} \
          --algo ${ALGO} \
          --num_gpus 4 \
          --max_model_len 8192 \
          --gpu_memory_utilization 0.95 \
          --critique_style B \
          --gt_stop_on em \
          --reflection_rounds 2  \
          --subset :5 \
          --gt_allow_leak"
    > "${LOGDIR}/self_reflection.log" 2>&1
  done
done
