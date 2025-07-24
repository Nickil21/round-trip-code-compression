#!/usr/bin/env bash
set -euo pipefail

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

# ─── Install Python requirements ──────────────────────────────────────────────
pip install -r requirements.txt

# ─── Load .env if present ────────────────────────────────────────────────────
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi

# ─── Login to Hugging Face CLI ────────────────────────────────────────────────
if command -v huggingface-cli >/dev/null; then
  if [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
    echo "Logging in to Hugging Face CLI…"
    huggingface-cli login --token "${HUGGINGFACE_TOKEN}"
  else
    echo "Warning: HUGGINGFACE_TOKEN not set; skipping hf-cli login."
  fi
else
  echo "Warning: huggingface-cli not installed; skip login."
fi

# ─── Configuration (via ENV VARS) ─────────────────────────────────────────────
ALGOS="${ALGOS:?ERROR: please set ALGOS (e.g. \"lzw ae bf cg\")}"
TEMPERATURES="${TEMPERATURES:-0.2 0.8}"
NUM_COMPLETIONS="${NUM_COMPLETIONS:-5}"


MODELS=(
  'Qwen/Qwen3-4B'
  'Qwen/Qwen3-8B'
  'Qwen/Qwen3-32B'
  'meta-llama/Llama-3.2-1B-Instruct'
  'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
  'meta-llama/Llama-3.2-3B-Instruct'
  'microsoft/Phi-3-mini-128k-instruct'
  'microsoft/Phi-3.5-mini-instruct'
  'meta-llama/Llama-3.1-8B-Instruct'
  'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'
  'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
  'microsoft/phi-4'
  'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
  'bigcode/starcoder2-15b-instruct-v0.1'
  'mistralai/Codestral-22B-v0.1'
  'Qwen/QwQ-32B'
  'Qwen/Qwen2.5-Coder-32B-Instruct'
  'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
  'deepseek-ai/deepseek-coder-33b-instruct'
  'codellama/CodeLlama-34b-Instruct-hf'
  'codellama/CodeLlama-70b-Python-hf'
  'meta-llama/Llama-3.1-70B-Instruct'
  'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
  'WizardLMTeam/WizardLM-70B-V1.0'
  'Qwen/Qwen2.5-72B-Instruct'
)

nvidia-smi --list-gpus

# ─── How many models have we got? ────────────────────────────────────────────
echo "Total models: ${#MODELS[@]}"

# ─── Start Ray once for all algorithms ────────────────────────────────────────
export RAY_TMPDIR=/tmp/ray
ray stop || true
ray start --head --temp-dir /tmp/ray --dashboard-host 0.0.0.0

# ─── Parallelism settings ─────────────────────────────────────────────────────
MAX_JOBS=1   # max concurrent algorithms

# ─── Loop over each algorithm in parallel ────────────────────────────────────
for ALG in ${ALGOS}; do
  (
    echo "[${ALG}] === Starting ==="
    LOG_DIR="logs/${ALG}"; mkdir -p "${LOG_DIR}"

    echo "[${ALG}] Preparing data…"
    python tasks/generate_data.py --algorithms "${ALG}" --source mixed --count 1

    echo "[${ALG}] Building prompt messages…"
    python src/build_codeio_msg.py \
      --input_file "processed_datasets/${ALG}/data.jsonl" \
      --output_file "processed_datasets/${ALG}/codeio_1k_msg.jsonl" \
      --algorithm "${ALG}" \
      --prompt_type "zero_shot"

    DATA_DIR="processed_datasets/${ALG}"
    INPUT_JSON="${DATA_DIR}/data.jsonl"
    MSG_FILE="${DATA_DIR}/codeio_1k_msg.jsonl"
    GENS_PREFIX="${DATA_DIR}/codeio_1k_gens"

    for model in ${MODELS}; do
      MODEL_NAME="${model##*/}"
      MODEL_NAME="${MODEL_NAME,,}"
      MODEL_NAME="${MODEL_NAME//-/_}"

      for T in ${TEMPERATURES}; do
        echo "[${ALG}] → ${MODEL_NAME} @ T=${T}"
        OUT_FILE="${GENS_PREFIX}_model_${MODEL_NAME}_temp_${T}_n${NUM_COMPLETIONS}.jsonl"
        LOG_FILE="${LOG_DIR}/${MODEL_NAME}_temp_${T}_n${NUM_COMPLETIONS}.log"
        RES_FILE="${OUT_FILE%.jsonl}_verified.jsonl"

        python src/batched_api_inference.py \
          --model "${model}" \
          --input "${MSG_FILE}" \
          --output "${OUT_FILE}" \
          --temperature "${T}" \
          --num_completions "${NUM_COMPLETIONS}" \
          > "${LOG_FILE}" 2>&1

        python src/check_io_pred_acc_mp.py \
          --parsed_file_name "${INPUT_JSON}" \
          --pred_file_name   "${OUT_FILE}" \
          --res_file_name    "${RES_FILE}" \
          --algo             "${ALG}"
      done
    done

    echo "[${ALG}] === Done ==="
  ) &

  # throttle: wait if we've hit $MAX_JOBS concurrent jobs
  while [ "$(jobs -rp | wc -l)" -ge "${MAX_JOBS}" ]; do
    sleep 1
  done
done

# ─── Wait for all to finish ───────────────────────────────────────────────────
wait

# ─── Stop Ray when everything’s finished ──────────────────────────────────────
ray stop
