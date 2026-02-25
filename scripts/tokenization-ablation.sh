#!/usr/bin/env bash
# Tokenization ablation pipeline — QwQ-32B (vLLM via singularity)
#
# Each array task runs one algo:variant pair with the same singularity +
# TP/max-length retry logic used in zero-shot-inference.sh.
#
# Usage:
#   bash scripts/tokenization-ablation.sh [model] [temperature]   # interactive (all variants sequentially)
#   sbatch scripts/tokenization-ablation.sh                        # SLURM — auto-sizes array to VARIANTS
#   sbatch scripts/tokenization-ablation.sh Qwen/QwQ-32B 0.2
#   TASK_FAMILY=both|output|input bash scripts/tokenization-ablation.sh ...
#   INPUT_TOKENIZATIONS=raw,base64,hex,codepoints,unicode_escape bash scripts/tokenization-ablation.sh ...
#
# The --array size is computed at runtime from the VARIANTS list and passed via
# self-resubmission, so adding/removing variants requires no manual header edits.
#
#SBATCH --job-name=tok-ablation
#SBATCH --output=logs/slurm/tok-ablation.%A_%a.out
#SBATCH --error=logs/slurm/tok-ablation.%A_%a.err
# (--array is set dynamically via self-resubmission; see below)
#SBATCH --partition=workq
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=8:00:00       # per-variant limit for QwQ-32B

set -euo pipefail

MODEL="${1:-Qwen/QwQ-32B}"
T="${2:-0.2}"
NUM_COMPLETIONS=5
MAX_CONCURRENT=3   # max array tasks running at once (each uses 4 GPUs)
TASK_FAMILY="${TASK_FAMILY:-both}"   # output | input | both
INPUT_TOKENIZATIONS_CSV="${INPUT_TOKENIZATIONS:-raw}"

VARIANTS=(
  "huffman:base64"
  "huffman:hex"
  "huffman:spaced_decimal"
  "huffman:char_hex"
  "huffman:binary"
  "lzw:csv"
  "lzw:spaced"
  "rle:compact"
  "rle:spaced"
  "ae:fraction"
)

# ── Change to project root ────────────────────────────────────────────────────
# SLURM copies the script to /var/spool/slurmd, so BASH_SOURCE[0] is useless
# there.  SLURM_SUBMIT_DIR is always the directory sbatch was called from.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# ── Dynamic array sizing (self-resubmission) ──────────────────────────────────
# #SBATCH --array cannot reference shell variables, so when this script is
# submitted as a plain (non-array) SLURM job it re-submits itself with the
# correct --array=0-N%MAX_CONCURRENT computed from the VARIANTS list, then exits.
# Array tasks skip this block (SLURM_ARRAY_TASK_ID is already set).
if [[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  N=${#VARIANTS[@]}
  ARRAY_SPEC="0-$((N - 1))%${MAX_CONCURRENT}"
  echo "[${SLURM_JOB_ID}] Re-submitting as array job: --array=${ARRAY_SPEC}"
  sbatch --array="${ARRAY_SPEC}" "$0" "$@"
  exit 0
fi

# ── HF / vLLM environment ─────────────────────────────────────────────────────
export HF_HOME="/lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache"
export HF_CACHE_DIR="${HF_HOME}/models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SIF="/projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.11.sif"
PYTHON_IN_SIF="/opt/python/pkgs/python-3.12.11/bin/python"

TP_SIZES=(1 2 4)
INITIAL_MAX_LENGTH=32768
MIN_MAX_LENGTH=2048

nvidia-smi --list-gpus || true

echo "Working directory: $(pwd)"
echo "Model: ${MODEL}  Temperature: ${T}  N completions: ${NUM_COMPLETIONS}"
echo "Comparison task family: ${TASK_FAMILY}"
echo "Input tokenizations: ${INPUT_TOKENIZATIONS_CSV}"

IFS=',' read -r -a INPUT_TOKENIZATIONS <<< "${INPUT_TOKENIZATIONS_CSV}"

# ── Select variant(s) ─────────────────────────────────────────────────────────
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  RUN_VARIANTS=("${VARIANTS[${SLURM_ARRAY_TASK_ID}]}")
  echo "Array task ${SLURM_ARRAY_TASK_ID} → ${RUN_VARIANTS[0]}"
else
  echo "[INFO] No SLURM_ARRAY_TASK_ID; running all ${#VARIANTS[@]} variants sequentially."
  RUN_VARIANTS=("${VARIANTS[@]}")
fi
echo

# ── Inference helper (singularity + TP/max-length retry) ─────────────────────
run_inference() {
  local MSG_FILE="$1" OUT_FILE="$2" LOG_DIR="$3"
  local VLLM_USE_V1 ML EXIT LOG_FILE

  for TP in "${TP_SIZES[@]}"; do
    VLLM_USE_V1=1
    ML=${INITIAL_MAX_LENGTH}
    while true; do
      LOG_FILE="${LOG_DIR}/tp${TP}_ml${ML}.log"
      echo "[${SLURM_JOB_ID:-local}] Trying tp_size=${TP}, max_length=${ML}, VLLM_USE_V1=${VLLM_USE_V1}" | tee "${LOG_FILE}"

      set +e
      singularity exec --nv \
        --bind "$(pwd)":/workspace \
        --bind "${HF_HOME}:${HF_HOME}" \
        "${SIF}" \
        bash -lc "export VLLM_USE_V1=${VLLM_USE_V1} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
          TORCHINDUCTOR_CACHE_DIR=/tmp/vllm-inductor-${SLURM_JOB_ID:-$$} \
          TRITON_CACHE_DIR=/tmp/vllm-triton-${SLURM_JOB_ID:-$$} && cd /workspace && \
          ${PYTHON_IN_SIF} src/inference/batched_api_inference.py \
            --model '${MODEL}' \
            --input '${MSG_FILE}' \
            --output '${OUT_FILE}' \
            --temperature ${T} \
            --num_completions ${NUM_COMPLETIONS} \
            --tp_size ${TP} \
            --max_tokens ${ML} \
            --hf_offline \
            --cache_dir ${HF_CACHE_DIR} \
        " >> "${LOG_FILE}" 2>&1
      EXIT=$?
      set -e

      if [[ ${EXIT} -eq 0 ]]; then
        echo "[${SLURM_JOB_ID:-local}] Success @ tp=${TP}, max_length=${ML}" | tee -a "${LOG_FILE}"
        return 0
      fi

      if grep -qi "out of memory" "${LOG_FILE}"; then
        echo "[${SLURM_JOB_ID:-local}] OOM @ tp=${TP}; trying next TP" | tee -a "${LOG_FILE}"
        break
      fi

      if grep -qiE "Head size .* not supported by FlashAttention" "${LOG_FILE}"; then
        if [[ "${VLLM_USE_V1}" -eq 0 ]]; then
          echo "[${SLURM_JOB_ID:-local}] Already on fallback backend; cannot recover" | tee -a "${LOG_FILE}"
          exit 1
        fi
        VLLM_USE_V1=0
        echo "[${SLURM_JOB_ID:-local}] FlashAttention error; switching VLLM_USE_V1=0" | tee -a "${LOG_FILE}"
        continue
      fi

      if grep -qiE "User-specified max_model_len|max_model_length|reducing max_model_len" "${LOG_FILE}"; then
        if [[ ${ML} -le ${MIN_MAX_LENGTH} ]]; then
          echo "[${SLURM_JOB_ID:-local}] Reached min max_length=${MIN_MAX_LENGTH}; giving up" | tee -a "${LOG_FILE}"
          exit 1
        fi
        ML=$(( ML / 2 ))
        echo "[${SLURM_JOB_ID:-local}] max_length error; retrying with max_length=${ML}" | tee -a "${LOG_FILE}"
        continue
      fi

      echo "[${SLURM_JOB_ID:-local}] Unexpected error (exit ${EXIT}); see ${LOG_FILE}" >&2
      exit "${EXIT}"
    done
  done

  echo "[${SLURM_JOB_ID:-local}] All TP sizes exhausted; giving up" >&2
  exit 1
}

# ── Per-variant pipeline ──────────────────────────────────────────────────────
run_variant() {
  local ALG_VARIANT="$1"
  local INPUT_TOK="$2"
  local ALG="${ALG_VARIANT%%:*}"
  local VARIANT="${ALG_VARIANT##*:}"

  echo "═══════════════════════════════════════════════════════════════════"
  echo "  Processing: ${ALG}/${VARIANT} | input_tokenization=${INPUT_TOK}"
  echo "═══════════════════════════════════════════════════════════════════"

  local DATA_DIR="data/processed/${ALG}"
  local INPUT_JSON="${DATA_DIR}/data.jsonl"

  if [[ ! -f "${INPUT_JSON}" ]]; then
    echo "[WARN] Input file not found, skipping: ${INPUT_JSON}"
    return 0
  fi

  local MODEL_TAG
  MODEL_TAG=$(echo "${MODEL##*/}" | tr '[:upper:]' '[:lower:]' | tr ' -' '__')

  # Step 1: Build alternative prompts
  local INPUT_TOK_SUFFIX=""
  if [[ "${INPUT_TOK}" != "raw" ]]; then
    INPUT_TOK_SUFFIX="_in_${INPUT_TOK}"
  fi
  local MSG_FILE="${DATA_DIR}/codeio_alt_${VARIANT}${INPUT_TOK_SUFFIX}_msg.jsonl"
  echo "Step 1: Building alternative prompts → ${MSG_FILE}"
  python src/ablation/build_tokenization_ablation.py \
    --algorithm   "${ALG}" \
    --variant     "${VARIANT}" \
    --input_tokenization "${INPUT_TOK}" \
    --input_file  "${INPUT_JSON}" \
    --output_file "${MSG_FILE}" \
    --prompt_type zero_shot

  # Step 2: Inference
  local OUT_FILE="${DATA_DIR}/codeio_alt_${VARIANT}${INPUT_TOK_SUFFIX}_gens_model_${MODEL_TAG}_temp_${T}_n${NUM_COMPLETIONS}.jsonl"
  local VERIF_JSONL="${OUT_FILE%.jsonl}_verified.jsonl"
  local VERIF_CSV="${OUT_FILE%.jsonl}_verified.csv"
  local LOG_DIR="logs/ablation/${ALG}/${VARIANT}"
  mkdir -p "${LOG_DIR}"

  echo "Step 2: Running inference → ${OUT_FILE}"
  if [[ "${MODEL}" != *"/"* ]]; then
    # OpenAI model — direct call, no GPU needed
    python src/inference/batched_api_inference.py \
      --model           "${MODEL}" \
      --input           "${MSG_FILE}" \
      --output          "${OUT_FILE}" \
      --temperature     "${T}" \
      --num_completions "${NUM_COMPLETIONS}" \
      --use_openai
  else
    # Local model — singularity + TP/max-length retry
    run_inference "${MSG_FILE}" "${OUT_FILE}" "${LOG_DIR}"
  fi

  # Step 3: Verify predictions
  echo "Step 3: Verifying predictions → ${VERIF_JSONL}"
  python src/ablation/check_tokenization_ablation.py \
    --parsed_file_name "${INPUT_JSON}" \
    --pred_file_name   "${OUT_FILE}" \
    --res_file_name    "${VERIF_JSONL}" \
    --algo             "${ALG}" \
    --variant          "${VARIANT}"

  # Step 4: Compare with original (skipped if baseline CSV absent)
  local ORIG_CSV="${DATA_DIR}/codeio_1k_gens_model_${MODEL_TAG}_temp_${T}_n${NUM_COMPLETIONS}_verified.csv"
  if [[ -f "${ORIG_CSV}" ]]; then
    echo "Step 4: Comparing with original → ${ORIG_CSV}"
    python src/ablation/compare_ablation_results.py \
      --original_csv "${ORIG_CSV}" \
      --ablation_csv "${VERIF_CSV}" \
      --algo         "${ALG}" \
      --variant      "${VARIANT}" \
      --task_family  "${TASK_FAMILY}"
  else
    echo "Step 4: Skipping comparison (original CSV not found: ${ORIG_CSV})"
  fi

  echo
}

for PAIR in "${RUN_VARIANTS[@]}"; do
  for INPUT_TOK in "${INPUT_TOKENIZATIONS[@]}"; do
    run_variant "${PAIR}" "${INPUT_TOK}"
  done
done

echo "Done: ${RUN_VARIANTS[*]}"


# ─────────────────────────────────────────────────────────────────────────────
# Valid algo/variant combinations
#
#   ┌─────────────┬────────────────┬─────────────────────────────────────────────┐
#   │ --algorithm │   --variant    │            Output format change             │
#   ├─────────────┼────────────────┼─────────────────────────────────────────────┤
#   │ huffman     │ base64         │ [170, 0, ...] → "qg=="                      │
#   ├─────────────┼────────────────┼─────────────────────────────────────────────┤
#   │ huffman     │ hex            │ [170, 0, ...] → "aa00"                      │
#   ├─────────────┼────────────────┼─────────────────────────────────────────────┤
#   │ huffman     │ spaced_decimal │ [170, 0, ...] → "170 0 255"                 │
#   ├─────────────┼────────────────┼─────────────────────────────────────────────┤
#   │ huffman     │ char_hex       │ [170, 0, ...] → "a a 0 0 f f"              │
#   ├─────────────┼────────────────┼─────────────────────────────────────────────┤
#   │ huffman     │ binary         │ [170, 0, ...] → "10101010 00000000"         │
#   ├─────────────┼────────────────┼─────────────────────────────────────────────┤
#   │ lzw         │ csv            │ [85, 256, 257] → "85,256,257"               │
#   ├─────────────┼────────────────┼─────────────────────────────────────────────┤
#   │ lzw         │ spaced         │ [85, 256, 257] → "85 256 257"               │
#   ├─────────────┼────────────────┼─────────────────────────────────────────────┤
#   │ rle         │ compact        │ [["U",8]] → "U8"                            │
#   ├─────────────┼────────────────┼─────────────────────────────────────────────┤
#   │ rle         │ spaced         │ [["U",8],["A",3]] → "U 8 A 3"              │
#   ├─────────────┼────────────────┼─────────────────────────────────────────────┤
#   │ ae          │ fraction       │ 0.6319... → "28/29"                         │
#   └─────────────┴────────────────┴─────────────────────────────────────────────┘
#
# Prerequisites:
#   - For OpenAI models: OPENAI_API_KEY must be set; remove --gres=gpu:4 / --partition flags
#   - For local models: GPU access; HF_HOME path must be correct for your cluster
