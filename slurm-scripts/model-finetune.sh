#!/usr/bin/env bash
#SBATCH --job-name=io-pred-inference-qwq32b-ft
#SBATCH --output=slurm-logs/io-pred-inference-qwq32b-ft.%A_%a.out
#SBATCH --error=slurm-logs/io-pred-inference-qwq32b-ft.%A_%a.err
#SBATCH --partition=workq
#SBATCH --array=0-7%8            # 4 algs × 2 temps
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00


# FIRST RUN THIS TO FINETUNE THE MODEL:
# ./LLaMA-Factory/bash-scripts/finetune.sh

set -euo pipefail

#–– Which GPU did I get?
nvidia-smi --list-gpus || true

# Where the finetuned model lives (merged export root)
FINETUNE_ROOT="${FINETUNE_ROOT:-LLaMA-Factory/finetune}"

# ─── ENV SETUP ───────────────────────────────────────────────────
source ~/miniforge3/bin/activate
conda activate round-trip-myenv \
  || (conda create -y -n round-trip-myenv python=3.10 && conda activate round-trip-myenv)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ─── GRID: algorithms & temperatures ─────────────────────────────
ALGS=(lzw ae rle huffman)
TEMPS=(0.2 0.8)
NUM_COMPLETIONS=5

# ─── MAP ARRAY ID → (ALG, TEMP) ─────────────────────────────────
IDX=${SLURM_ARRAY_TASK_ID}
ALG_IDX=$(( IDX / ${#TEMPS[@]} ))
TEMP_IDX=$(( IDX % ${#TEMPS[@]} ))
ALG=${ALGS[$ALG_IDX]}
T=${TEMPS[$TEMP_IDX]}

echo "[$SLURM_JOB_ID:$SLURM_ARRAY_TASK_ID] → ALG=$ALG | T=$T"

# ─── DIRECTORIES (append -finetune to existing) ─────────────────
DATA_DIR="processed_datasets/${ALG}"            # read-only inputs
LOG_DIR="logs-finetune/${ALG}"                  # logs → logs-finetune/ALG
OUT_DIR="${DATA_DIR}-finetune"                  # outputs → processed_datasets/ALG-finetune
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

INPUT_JSON="${DATA_DIR}/data.jsonl"
MSG_FILE="${DATA_DIR}/codeio_1k_msg.jsonl"

# ─── MODEL: base + LoRA (unmerged) ──────────────────────────────
BASE_MODEL="Qwen/QwQ-32B"
MODEL_TAG=$(basename "$BASE_MODEL" | sed -E 's/-([0-9]+)[bB]$/-\1B/')
FT_DIR="${FINETUNE_ROOT}/${ALG}/${MODEL_TAG}/lora/sft"
ADAPTER="${FT_DIR}"

# If you require a merged checkpoint instead, uncomment this block:
# MERGED_DIR="${FT_DIR}/export-merged"
# [[ -f "${MERGED_DIR}/config.json" ]] || { echo "❌ Missing merged checkpoint at: ${MERGED_DIR}"; exit 2; }
# MODEL="${MERGED_DIR}"

# Outputs now live in OUT_DIR (next to dataset dir, with -finetune suffix)
GENS_PREFIX="${OUT_DIR}/codeio_1k_gens"
OUT_FILE="${GENS_PREFIX}_model_${MODEL_TAG}_temp_${T}_n${NUM_COMPLETIONS}.jsonl"
VERIF_FILE="${OUT_FILE%.jsonl}_verified.jsonl"

# ─── INFERENCE WITH DYNAMIC TP_SIZE & MAX_LENGTH ────────────────
# Match TP_SIZES to --gres=gpu:4
TP_SIZES=(1 2 4)
INITIAL_MAX_LENGTH=32768
MIN_MAX_LENGTH=2048

# start with FlashAttention enabled (flip to fallback on error)
export VLLM_USE_V1=1

for TP in "${TP_SIZES[@]}"; do
  ML=$INITIAL_MAX_LENGTH
  while true; do
    LOG_FILE="${LOG_DIR}/${MODEL_TAG}_T${T}_tp${TP}_ml${ML}.log"
    echo "[$SLURM_JOB_ID] Trying tp_size=${TP}, max_length=${ML}, VLLM_USE_V1=${VLLM_USE_V1}" | tee "${LOG_FILE}"

    CURRENT_OUT=${OUT_TMP:-${OUT_FILE}}

    singularity exec --nv \
      --bind "$(pwd)":/workspace \
      /projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.06.sif \
      bash -lc "export VLLM_USE_V1=${VLLM_USE_V1} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && cd /workspace && \
        /py3.10/bin/python src/batched_api_inference.py \
          --model '${BASE_MODEL}' \
          --lora_adapter '${ADAPTER}' \
          --input '${MSG_FILE}' \
          --output '${CURRENT_OUT}' \
          --temperature ${T} \
          --num_completions ${NUM_COMPLETIONS} \
          --tp_size ${TP} \
          --max_tokens ${ML} \
          --hf_offline \
          --cache_dir /home/u5u/nmaveli.u5u/.cache/huggingface/hub \
      " >> "${LOG_FILE}" 2>&1
    EXIT=$?

    if [ $EXIT -eq 0 ]; then
      echo "[$SLURM_JOB_ID] Success @ tp=${TP}, max_length=${ML}" | tee -a "${LOG_FILE}"
      break 2
    fi

    if grep -qi "out of memory" "${LOG_FILE}"; then
      echo "[$SLURM_JOB_ID] OOM @ tp=${TP}; moving to next TP" | tee -a "${LOG_FILE}"
      break
    fi

    if grep -qiE "Head size .* not supported by FlashAttention" "${LOG_FILE}"; then
      if [ "${VLLM_USE_V1}" -eq 0 ]; then
        echo "[$SLURM_JOB_ID] Already using fallback backend; cannot recover" | tee -a "${LOG_FILE}"
        exit 1
      fi
      export VLLM_USE_V1=0
      echo "[$SLURM_JOB_ID] FlashAttention error; setting VLLM_USE_V1=0 and retrying" | tee -a "${LOG_FILE}"
      continue
    fi

    if grep -qiE "max_model_len|max_model_length|User-specified max_model_len" "${LOG_FILE}"; then
      if [ ${ML} -le ${MIN_MAX_LENGTH} ]; then
        echo "[$SLURM_JOB_ID] Reached min max_length=${MIN_MAX_LENGTH}; terminating" | tee -a "${LOG_FILE}"
        exit 1
      fi
      ML=$(( ML / 2 ))
      echo "[$SLURM_JOB_ID] max_length error; retrying with max_length=${ML}" | tee -a "${LOG_FILE}"
      continue
    fi

    echo "[$SLURM_JOB_ID] Unexpected error (exit $EXIT); see ${LOG_FILE}" >&2
    exit $EXIT
  done
done

# ─── MERGE RESUMED OUTPUT (stays in ALG-finetune dir) ───────────
if [ -n "${OUT_TMP:-}" ]; then
  echo "[$SLURM_JOB_ID] Merging ${OUT_TMP} into ${OUT_FILE}..."
  cat "${OUT_TMP}" >> "${OUT_FILE}"
  rm "${OUT_TMP}"
  echo "[$SLURM_JOB_ID] Now have $(wc -l < "${OUT_FILE}")/$(wc -l < "${MSG_FILE}") lines."
fi

# ─── VERIFY OUTPUT (writes alongside gens in ALG-finetune dir) ──
python src/check_io_pred_acc_mp.py \
  --parsed_file_name "${INPUT_JSON}" \
  --pred_file_name   "${OUT_FILE}" \
  --res_file_name    "${VERIF_FILE}" \
  --algo