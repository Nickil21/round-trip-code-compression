#!/usr/bin/env bash

set -euxo pipefail

# ─── Which algorithms to run ────────────────────────────────────────────
ALGORITHMS=(lzw ae rle huffman)

# ─── Setup Conda ─────────────────────────────────────────────────────────────
CONDA_ROOT="/mnt_path/miniconda3"
ENV_PATH="${CONDA_ROOT}/envs/round-trip-code-compression-env"

# ─── Create env if it doesn't exist ──────────────────────────────────────────
if [ ! -d "${ENV_PATH}" ]; then
  echo "Creating conda env at ${ENV_PATH} (Python 3.11)…"
  conda create --prefix "${ENV_PATH}" python=3.11 -y
else
  echo "Conda env already exists at ${ENV_PATH}, skipping creation."
fi

# ─── Activate it ──────────────────────────────────────────────────────────────
export PATH="${CONDA_ROOT}/bin:${PATH}"
eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"

nvidia-smi --list-gpus

# ─── Inference + Verification Loop ─────────────────────────────
# export RAY_TMPDIR=/tmp/ray
# ray start --head --temp-dir /tmp/ray --dashboard-host 0.0.0.0

# ─── Loop over algorithms, sequentially ─────────────────────────────────
for ALGO in "${ALGORITHMS[@]}"; do
  DATADIR="processed_datasets/${ALGO}"
  LOGDIR="logs/${ALGO}"
  mkdir -p "$DATADIR" "$LOGDIR"

  # data-prep steps
  # python tasks/generate_data.py         --algorithms "$ALGO" --source mixed --count 20
  # python src/build_codeio_msg.py        \
  #        --input_file "${DATADIR}/data.jsonl"         \
  #        --output_file "${DATADIR}/codeio_1k_msg.jsonl" \
  #        --algorithm "$ALGO"
  python src/generate_execution_trace.py --algorithm "$ALGO"
  python src/filter_execution_trace.py   --algorithm "$ALGO"

  # translation step: uses all 8 GPUs (no manual subset math)
  python src/execution_trace_translation.py \
        --algorithm $ALGO \
        --translator_model Qwen/Qwen2.5-Coder-32B-Instruct \
        --num_gpus 4 > "${LOGDIR}/translated.log" 2>&1

  echo "[$ALGO] translation log → ${LOGDIR}/translated.log"

  # data construction for SFT (cluster-specific; OUTPUTDIR not set in local env)
  # python src/data/data_construction_sft.py --algorithm "$ALGO" --output_file "${OUTPUTDIR}/${ALGO}_training_data_sft.jsonl" --trained_model Qwen/QwQ-32B
  # jq -s '.' "${OUTPUTDIR}/${ALGO}_training_data_sft.jsonl" > "${OUTPUTDIR}/${ALGO}_training_data_sft.json"

done

# ray stop
