#!/usr/bin/env bash
#SBATCH --job-name=io-pred-inference
#SBATCH --output=slurm-logs/io-pred-inference.%A_%a.out
#SBATCH --error=slurm-logs/io-pred-inference.%A_%a.err
#SBATCH --partition=workq
#SBATCH --array=0-319%32            # 208 tasks, max 64 running at once
#SBATCH --gres=gpu:4                # four GPUs per task
#SBATCH --cpus-per-task=4           # tweak per-GPU CPU as needed
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

#–– Which GPU did I get?
nvidia-smi --list-gpus

# ─── ENV SETUP ────────────────────────────────────────────────────
source ~/miniforge3/bin/activate
conda activate round-trip-myenv \
  || (conda create -y -n round-trip-myenv python=3.10 && conda activate round-trip-myenv)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# ─── GRID DEFINITION ─────────────────────────────────────────────
ALGS=(lzw ae rle huffman)
MODELS=(
  'Qwen/Qwen2.5-7B-Instruct'
  'mistralai/Mistral-7B-Instruct-v0.3'
  '01-ai/Yi-Coder-9B-Chat'
  'google/codegemma-7b-it'
  # 'Qwen/Qwen3-4B'
  # 'Qwen/Qwen3-8B'
  # 'Qwen/Qwen3-32B'
  'meta-llama/Llama-3.2-1B-Instruct'
  'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
  'meta-llama/Llama-3.2-3B-Instruct'
  'microsoft/Phi-3-mini-128k-instruct'
  'microsoft/Phi-3.5-mini-instruct'
  'meta-llama/Llama-3.1-8B-Instruct'
  'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'
  'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
  'microsoft/phi-4'
  'microsoft/phi-2'
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
TEMPS=(0.2 0.8)
NUM_COMPLETIONS=5

# ─── MAP ARRAY ID → (ALG, MODEL, TEMP) ───────────────────────────
IDX=$SLURM_ARRAY_TASK_ID
ALG_IDX=$(( IDX / ( ${#MODELS[@]} * ${#TEMPS[@]} ) ))
ALG=${ALGS[$ALG_IDX]}

REST=$(( IDX % ( ${#MODELS[@]} * ${#TEMPS[@]} ) ))
MODEL_IDX=$(( REST / ${#TEMPS[@]} ))
TEMP_IDX=$(( REST % ${#TEMPS[@]} ))

MODEL=${MODELS[$MODEL_IDX]}
T=${TEMPS[$TEMP_IDX]}

echo "[$SLURM_JOB_ID:$SLURM_ARRAY_TASK_ID] → ALG=$ALG | MODEL=$MODEL | T=$T"

# ─── DIRECTORIES ────────────────────────────────────────────────
LOG_DIR="logs/${ALG}"
mkdir -p "${LOG_DIR}"

DATA_DIR="processed_datasets/${ALG}"
INPUT_JSON="${DATA_DIR}/data.jsonl"
MSG_FILE="${DATA_DIR}/codeio_1k_msg.jsonl"
GENS_PREFIX="${DATA_DIR}/codeio_1k_gens"

# ─── PREPARE DATA & PROMPTS ─────────────────────────────────────
echo "[$SLURM_JOB_ID] Preparing data for $ALG"
python tasks/generate_data.py \
  --algorithms "${ALG}" \
  --source mixed \
  --count 50

echo "[$SLURM_JOB_ID] Building prompts for $ALG"
python src/build_codeio_msg.py \
  --input_file "${INPUT_JSON}" \
  --output_file "${MSG_FILE}" \
  --algorithm "${ALG}" \
  --prompt_type zero_shot

# ─── INFERENCE WITH DYNAMIC TP_SIZE & MAX_LENGTH ───────────────
TP_SIZES=(1 2 4 8)
INITIAL_MAX_LENGTH=32768    # start high
MIN_MAX_LENGTH=2048        # do not go below this

MODEL_TAG=$(echo "${MODEL##*/}" | tr '[:upper:]-' '[:lower:]_')
OUT_FILE="${GENS_PREFIX}_model_${MODEL_TAG}_temp_${T}_n${NUM_COMPLETIONS}.jsonl"
VERIF_FILE="${OUT_FILE%.jsonl}_verified.jsonl"

# start with FlashAttention enabled
export VLLM_USE_V1=1

for TP in "${TP_SIZES[@]}"; do
  ML=$INITIAL_MAX_LENGTH

  while true; do
    LOG_FILE="${LOG_DIR}/${MODEL_TAG}_T${T}_tp${TP}_ml${ML}.log"
    echo "[$SLURM_JOB_ID] Trying tp_size=${TP}, max_length=${ML}, VLLM_USE_V1=${VLLM_USE_V1}" | tee "${LOG_FILE}"

    # choose tmp or final out
    CURRENT_OUT=${OUT_TMP:-${OUT_FILE}}

    singularity exec --nv \
      --bind "$(pwd)":/workspace \
      /projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.06.sif \
      bash -lc "export VLLM_USE_V1=${VLLM_USE_V1} && cd /workspace && \
        /py3.10/bin/python src/batched_api_inference.py \
          --model ${MODEL} \
          --input ${MSG_FILE} \
          --output ${CURRENT_OUT} \
          --temperature ${T} \
          --num_completions ${NUM_COMPLETIONS} \
          --tp_size ${TP} \
          --max_tokens ${ML}" \
      >> "${LOG_FILE}" 2>&1
    EXIT=$?

    # Success
    if [ $EXIT -eq 0 ]; then
      echo "[$SLURM_JOB_ID] Success @ tp=${TP}, max_length=${ML}" | tee -a "${LOG_FILE}"
      break 2
    fi

    # Out of Memory → next TP size
    if grep -qi "out of memory" "${LOG_FILE}"; then
      echo "[$SLURM_JOB_ID] OOM @ tp=${TP}; moving to next TP" | tee -a "${LOG_FILE}"
      break
    fi

    # FlashAttention head-size error → disable V1 backend
    if grep -qi "Head size .* not supported by FlashAttention" "${LOG_FILE}"; then
      if [ "${VLLM_USE_V1}" -eq 0 ]; then
        echo "[$SLURM_JOB_ID] Already using fallback backend; cannot recover" | tee -a "${LOG_FILE}"
        exit 1
      fi
      export VLLM_USE_V1=0
      echo "[$SLURM_JOB_ID] FlashAttention error; setting VLLM_USE_V1=0 and retrying" | tee -a "${LOG_FILE}"
      continue
    fi

    # Generic max_length error → halve ML
    if grep -qi "ValueError: User-specified max_model_len" "${LOG_FILE}"; then
      if [ ${ML} -le ${MIN_MAX_LENGTH} ]; then
        echo "[$SLURM_JOB_ID] Reached min max_length=${MIN_MAX_LENGTH}; terminating" | tee -a "${LOG_FILE}"
        exit 1
      fi
      ML=$(( ML / 2 ))
      echo "[$SLURM_JOB_ID] max_length error; retrying with max_length=${ML}" | tee -a "${LOG_FILE}"
      continue
    fi

    # Any other error → exit
    echo "[$SLURM_JOB_ID] Unexpected error (exit $EXIT); see ${LOG_FILE}" >&2
    exit $EXIT
  done
done

# ─── MERGE RESUMED OUTPUT ───────────────────────────────────────
if [ -n "${OUT_TMP}" ]; then
  echo "[$SLURM_JOB_ID] Merging ${OUT_TMP} into ${OUT_FILE}..."
  cat "${OUT_TMP}" >> "${OUT_FILE}"
  rm "${OUT_TMP}"
  echo "[$SLURM_JOB_ID] Now have $(wc -l < "${OUT_FILE}")/$(wc -l < "${DATA_DIR}/codeio_1k_msg.jsonl") lines."
fi

# ─── VERIFY OUTPUT ───────────────────────────────────────────────
python src/check_io_pred_acc_mp.py \
  --parsed_file_name "${INPUT_JSON}" \
  --pred_file_name   "${OUT_FILE}" \
  --res_file_name    "${VERIF_FILE}" \
  --algo             "${ALG}"

echo "[$SLURM_JOB_ID] Done."