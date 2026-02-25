#!/usr/bin/env bash
#SBATCH --job-name=codeio-infer
#SBATCH --output=logs/slurm/%x.%A_%a.out
#SBATCH --error=logs/slurm/%x.%A_%a.err
#SBATCH --partition=workq
#SBATCH --array=0-0
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

INFERENCE_MODE="${INFERENCE_MODE:-zero_shot}"
case "${INFERENCE_MODE}" in
  zero_shot|finetune) ;;
  *)
    echo "Unsupported INFERENCE_MODE='${INFERENCE_MODE}'. Use zero_shot or finetune." >&2
    exit 2
    ;;
esac

ALGS=(lzw ae rle huffman)
TEMPS=(0.2) # (0.2 0.8)
NUM_COMPLETIONS="${NUM_COMPLETIONS:-5}"

MODEL_CONFIG_FILE="${MODEL_CONFIG_FILE:-${ROOT_DIR}/configs/models.yaml}"
FINETUNE_ROOT="${FINETUNE_ROOT:-LLaMA-Factory/finetune}"
BASE_MODEL="${BASE_MODEL:-Qwen/QwQ-32B}"

HF_HOME="${HF_HOME:-/lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME}/models}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/lus/lfs1aip2/projects/u6cg/nmaveli/nmaveli/conda-envs/envs/code-retrieval-llms}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.11.sif}"
OPENAI_WORKERS="${OPENAI_WORKERS:-64}"

setup_env() {
  if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  else
    # shellcheck disable=SC1091
    source "${HOME}/miniforge3/bin/activate"
  fi
  conda activate "${CONDA_ENV_PATH}"

  export HF_HOME
  export HF_CACHE_DIR

  if [[ "${INFERENCE_MODE}" == "finetune" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
  fi
}

load_zero_shot_models() {
  if [[ ! -f "${MODEL_CONFIG_FILE}" ]]; then
    echo "Model config file not found: ${MODEL_CONFIG_FILE}" >&2
    exit 1
  fi

  mapfile -t MODELS < <(
    python - "${MODEL_CONFIG_FILE}" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

for model in data.get("models", []):
    if model.get("active") and model.get("id"):
        print(model["id"])
PY
  )

  if [[ "${#MODELS[@]}" -eq 0 ]]; then
    echo "No active models found in ${MODEL_CONFIG_FILE}" >&2
    exit 1
  fi
}

if [[ "${INFERENCE_MODE}" == "zero_shot" ]]; then
  load_zero_shot_models
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if [[ "${INFERENCE_MODE}" == "zero_shot" ]]; then
    N=$(( ${#ALGS[@]} * ${#MODELS[@]} * ${#TEMPS[@]} ))
    JOB_LABEL="${JOB_LABEL:-codeio-zero-shot}"
    ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-}"
    echo "Mode: zero_shot"
    echo "Grid: ${#ALGS[@]} algs x ${#MODELS[@]} models x ${#TEMPS[@]} temps = ${N} tasks"
  else
    N=$(( ${#ALGS[@]} * ${#TEMPS[@]} ))
    JOB_LABEL="${JOB_LABEL:-codeio-finetune-infer}"
    ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-8}"
    echo "Mode: finetune"
    echo "Grid: ${#ALGS[@]} algs x ${#TEMPS[@]} temps = ${N} tasks"
  fi

  if [[ "${N}" -le 0 ]]; then
    echo "Computed empty task grid." >&2
    exit 1
  fi

  ARRAY_SPEC="0-$((N-1))"
  if [[ -n "${ARRAY_CONCURRENCY}" ]]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${ARRAY_CONCURRENCY}"
  fi

  echo "Submitting array ${ARRAY_SPEC} as ${JOB_LABEL}"
  sbatch --job-name="${JOB_LABEL}" --array="${ARRAY_SPEC}" "$0"
  exit 0
fi

nvidia-smi --list-gpus || true
setup_env

IDX="${SLURM_ARRAY_TASK_ID}"

if [[ "${INFERENCE_MODE}" == "zero_shot" ]]; then
  ALG_IDX=$(( IDX / ( ${#MODELS[@]} * ${#TEMPS[@]} ) ))
  ALG="${ALGS[$ALG_IDX]:-}"

  REST=$(( IDX % ( ${#MODELS[@]} * ${#TEMPS[@]} ) ))
  MODEL_IDX=$(( REST / ${#TEMPS[@]} ))
  TEMP_IDX=$(( REST % ${#TEMPS[@]} ))

  MODEL="${MODELS[$MODEL_IDX]:-}"
  T="${TEMPS[$TEMP_IDX]:-}"
else
  ALG_IDX=$(( IDX / ${#TEMPS[@]} ))
  TEMP_IDX=$(( IDX % ${#TEMPS[@]} ))

  ALG="${ALGS[$ALG_IDX]:-}"
  T="${TEMPS[$TEMP_IDX]:-}"
  MODEL="${BASE_MODEL}"
fi

if [[ -z "${ALG}" || -z "${T}" || -z "${MODEL}" ]]; then
  echo "[${SLURM_JOB_ID:-local}:${SLURM_ARRAY_TASK_ID}] Task ID out of grid range; nothing to do." >&2
  exit 0
fi

echo "[${SLURM_JOB_ID:-local}:${SLURM_ARRAY_TASK_ID}] Mode=${INFERENCE_MODE} ALG=${ALG} MODEL=${MODEL} T=${T}"

DATA_DIR="data/processed/${ALG}"
INPUT_JSON="${DATA_DIR}/data.jsonl"
MSG_FILE="${DATA_DIR}/codeio_1k_msg.jsonl"

if [[ "${INFERENCE_MODE}" == "zero_shot" ]]; then
  LOG_DIR="logs/inference/${ALG}"
  OUT_DIR="${DATA_DIR}"
else
  LOG_DIR="logs/finetune/${ALG}"
  OUT_DIR="${DATA_DIR}-finetune"
fi
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

if [[ "${INFERENCE_MODE}" == "zero_shot" ]]; then
  echo "[${SLURM_JOB_ID:-local}] Preparing data for ${ALG}"
  python scripts/generate_data.py \
    --algorithms "${ALG}" \
    --source mixed \
    --count 50

  echo "[${SLURM_JOB_ID:-local}] Building prompts for ${ALG}"
  python src/data/build_codeio_msg.py \
    --input_file  "${INPUT_JSON}" \
    --output_file "${MSG_FILE}" \
    --algorithm   "${ALG}" \
    --prompt_type zero_shot \
    --blind
fi

if [[ ! -f "${MSG_FILE}" ]]; then
  echo "Missing prompt file: ${MSG_FILE}" >&2
  exit 1
fi

if [[ "${INFERENCE_MODE}" == "zero_shot" ]]; then
  MODEL_TAG="$(echo "${MODEL##*/}" | tr '[:upper:]-' '[:lower:]_')"
else
  MODEL_TAG="$(basename "${MODEL}" | sed -E 's/-([0-9]+)[bB]$/-\1B/')"
fi

GENS_PREFIX="${OUT_DIR}/codeio_1k_gens"
OUT_FILE="${GENS_PREFIX}_model_${MODEL_TAG}_temp_${T}_n${NUM_COMPLETIONS}.jsonl"

if [[ "${INFERENCE_MODE}" == "finetune" ]]; then
  FT_DIR="${FINETUNE_ROOT}/${ALG}/${MODEL_TAG}/lora/sft"
  ADAPTER="${FT_DIR}"

  TP_SIZES=(1 2 4)
  INITIAL_MAX_LENGTH=32768
  MIN_MAX_LENGTH=2048

  for TP in "${TP_SIZES[@]}"; do
    VLLM_USE_V1=1
    ML="${INITIAL_MAX_LENGTH}"

    while true; do
      LOG_FILE="${LOG_DIR}/${MODEL_TAG}_T${T}_tp${TP}_ml${ML}.log"
      echo "[${SLURM_JOB_ID}] Trying tp_size=${TP}, max_length=${ML}, VLLM_USE_V1=${VLLM_USE_V1}" | tee "${LOG_FILE}"

      set +e
      singularity exec --nv \
        --bind "$(pwd)":/workspace \
        --bind "${HF_HOME}:${HF_HOME}" \
        "${CONTAINER_IMAGE}" \
        bash -lc "export VLLM_USE_V1=${VLLM_USE_V1} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
          TORCHINDUCTOR_CACHE_DIR=/tmp/vllm-inductor-${SLURM_JOB_ID} \
          TRITON_CACHE_DIR=/tmp/vllm-triton-${SLURM_JOB_ID} && cd /workspace && \
          /opt/python/pkgs/python-3.12.11/bin/python src/inference/batched_api_inference.py \
            --model '${BASE_MODEL}' \
            --lora_adapter '${ADAPTER}' \
            --input '${MSG_FILE}' \
            --output '${OUT_FILE}' \
            --temperature ${T} \
            --num_completions ${NUM_COMPLETIONS} \
            --tp_size ${TP} \
            --max_tokens ${ML} \
            --hf_offline \
            --cache_dir ${HF_CACHE_DIR:-${HOME}/.cache/huggingface/hub} \
        " >> "${LOG_FILE}" 2>&1
      EXIT=$?
      set -e

      if [[ ${EXIT} -eq 0 ]]; then
        echo "[${SLURM_JOB_ID}] Success @ tp=${TP}, max_length=${ML}" | tee -a "${LOG_FILE}"
        break 2
      fi

      if grep -qi "out of memory" "${LOG_FILE}"; then
        echo "[${SLURM_JOB_ID}] OOM @ tp=${TP}; moving to next TP" | tee -a "${LOG_FILE}"
        break
      fi

      if grep -qiE "Head size .* not supported by FlashAttention" "${LOG_FILE}"; then
        if [[ "${VLLM_USE_V1}" -eq 0 ]]; then
          echo "[${SLURM_JOB_ID}] Already using fallback backend; cannot recover" | tee -a "${LOG_FILE}"
          exit 1
        fi
        VLLM_USE_V1=0
        echo "[${SLURM_JOB_ID}] FlashAttention error; setting VLLM_USE_V1=0 and retrying" | tee -a "${LOG_FILE}"
        continue
      fi

      if grep -qiE "User-specified max_model_len|max_model_length|reducing max_model_len" "${LOG_FILE}"; then
        if [[ ${ML} -le ${MIN_MAX_LENGTH} ]]; then
          echo "[${SLURM_JOB_ID}] Reached min max_length=${MIN_MAX_LENGTH}; terminating" | tee -a "${LOG_FILE}"
          exit 1
        fi
        ML=$(( ML / 2 ))
        echo "[${SLURM_JOB_ID}] max_length error; retrying with max_length=${ML}" | tee -a "${LOG_FILE}"
        continue
      fi

      echo "[${SLURM_JOB_ID}] Unexpected error (exit ${EXIT}); see ${LOG_FILE}" >&2
      exit ${EXIT}
    done
  done

  echo "[${SLURM_JOB_ID}:${SLURM_ARRAY_TASK_ID}] Finetune inference complete."
  echo "Run scripts/evaluation-stats-finetune.sh to verify finetune outputs."
  exit 0
fi

if [[ "${MODEL}" != *"/"* || "${MODEL}" == openai/* ]]; then
  IS_OPENAI=true
else
  IS_OPENAI=false
fi

TP_SIZES=(1 2 4 8)
if echo "${MODEL}" | grep -qiE "QwQ|DeepSeek-R1|Qwen3"; then
  INITIAL_MAX_LENGTH=16384
else
  INITIAL_MAX_LENGTH=32768
fi
MIN_MAX_LENGTH=2048

if [[ "${IS_OPENAI}" == true ]]; then
  echo "[${SLURM_JOB_ID}] Calling OpenAI API: MODEL=${MODEL} | T=${T}"
  python src/inference/batched_api_inference.py \
    --model           "${MODEL}" \
    --input           "${MSG_FILE}" \
    --output          "${OUT_FILE}" \
    --temperature     "${T}" \
    --num_completions "${NUM_COMPLETIONS}" \
    --workers         "${OPENAI_WORKERS}" \
    --use_openai
else
  export VLLM_USE_V1=1

  if echo "${MODEL}" | grep -qiE "70[bB]|72[bB]"; then
    START_TP_IDX=2
  else
    START_TP_IDX=0
  fi

  for (( TPI=START_TP_IDX; TPI<${#TP_SIZES[@]}; TPI++ )); do
    TP=${TP_SIZES[$TPI]}
    ML=${INITIAL_MAX_LENGTH}

    while true; do
      LOG_FILE="${LOG_DIR}/${MODEL_TAG}_T${T}_tp${TP}_ml${ML}.log"
      echo "[${SLURM_JOB_ID}] Trying tp_size=${TP}, max_length=${ML}, VLLM_USE_V1=${VLLM_USE_V1}" | tee "${LOG_FILE}"

      CURRENT_OUT=${OUT_TMP:-${OUT_FILE}}
      LOCAL_CACHE="/tmp/vllm_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
      mkdir -p "${LOCAL_CACHE}"

      singularity exec --nv \
        --writable-tmpfs \
        --bind "$(pwd)":/workspace \
        --bind "${HF_CACHE_DIR}:${HF_CACHE_DIR}" \
        --bind "${LOCAL_CACHE}:${LOCAL_CACHE}" \
        "${CONTAINER_IMAGE}" \
        bash -lc "export VLLM_USE_V1=${VLLM_USE_V1} VLLM_TORCH_COMPILE_LEVEL=0 VLLM_CACHE_ROOT=${LOCAL_CACHE} TORCHINDUCTOR_CACHE_DIR=${LOCAL_CACHE}/inductor && cd /workspace && \
          /opt/python/pkgs/python-3.12.11/bin/python src/inference/batched_api_inference.py \
            --model ${MODEL} \
            --input ${MSG_FILE} \
            --output ${CURRENT_OUT} \
            --temperature ${T} \
            --num_completions ${NUM_COMPLETIONS} \
            --tp_size ${TP} \
            --max_tokens ${ML} \
            --hf_offline \
            --cache_dir ${HF_CACHE_DIR:-${HOME}/.cache/huggingface/hub}" \
        >> "${LOG_FILE}" 2>&1
      EXIT=$?

      if [[ ${EXIT} -eq 0 ]]; then
        echo "[${SLURM_JOB_ID}] Success @ tp=${TP}, max_length=${ML}" | tee -a "${LOG_FILE}"
        break 2
      fi

      if grep -qi "out of memory" "${LOG_FILE}"; then
        echo "[${SLURM_JOB_ID}] OOM @ tp=${TP}; moving to next TP" | tee -a "${LOG_FILE}"
        break
      fi

      if grep -qi "Head size .* not supported by FlashAttention" "${LOG_FILE}"; then
        if [[ "${VLLM_USE_V1}" -eq 0 ]]; then
          echo "[${SLURM_JOB_ID}] Already using fallback backend; cannot recover" | tee -a "${LOG_FILE}"
          exit 1
        fi
        export VLLM_USE_V1=0
        echo "[${SLURM_JOB_ID}] FlashAttention error; setting VLLM_USE_V1=0 and retrying" | tee -a "${LOG_FILE}"
        continue
      fi

      if grep -qi "WorkerProc initialization failed\|Engine core initialization failed" "${LOG_FILE}"; then
        echo "[${SLURM_JOB_ID}] WorkerProc/EngineCore init failed @ tp=${TP}; moving to next TP" | tee -a "${LOG_FILE}"
        break
      fi

      if grep -qi "envs\.VLLM_USE_V1=False\|VLLM_USE_V1=False" "${LOG_FILE}"; then
        export VLLM_USE_V1=1
        echo "[${SLURM_JOB_ID}] V0 backend incompatible with this vLLM build; reverting VLLM_USE_V1=1 and moving to next TP" | tee -a "${LOG_FILE}"
        break
      fi

      if grep -qi "ValueError: User-specified max_model_len" "${LOG_FILE}"; then
        if [[ ${ML} -le ${MIN_MAX_LENGTH} ]]; then
          echo "[${SLURM_JOB_ID}] Reached min max_length=${MIN_MAX_LENGTH}; terminating" | tee -a "${LOG_FILE}"
          exit 1
        fi
        ML=$(( ML / 2 ))
        echo "[${SLURM_JOB_ID}] max_length error; retrying with max_length=${ML}" | tee -a "${LOG_FILE}"
        continue
      fi

      echo "[${SLURM_JOB_ID}] Unexpected error (exit ${EXIT}); see ${LOG_FILE}" >&2
      exit ${EXIT}
    done
  done
fi

if [[ -n "${OUT_TMP:-}" ]]; then
  echo "[${SLURM_JOB_ID}] Merging ${OUT_TMP} into ${OUT_FILE}..."
  cat "${OUT_TMP}" >> "${OUT_FILE}"
  rm "${OUT_TMP}"
  echo "[${SLURM_JOB_ID}] Now have $(wc -l < "${OUT_FILE}")/$(wc -l < "${MSG_FILE}") lines."
fi

echo "[${SLURM_JOB_ID}] Done."
