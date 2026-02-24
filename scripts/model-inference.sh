#!/usr/bin/env bash
#SBATCH --job-name=io-pred-inference
#SBATCH --output=logs/slurm/io-pred-inference.%A_%a.out
#SBATCH --error=logs/slurm/io-pred-inference.%A_%a.err
#SBATCH --partition=workq
#SBATCH --array=0-0                 # placeholder — overridden dynamically by Phase 1 below
#SBATCH --gres=gpu:4                # four GPUs per task
#SBATCH --cpus-per-task=16          # vLLM benefits from more CPUs (4 per GPU)
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

# ─── GRID DEFINITION ─────────────────────────────────────────────
# Defined first so Phase 1 can compute the array size before any heavy setup.
ALGS=(lzw ae rle huffman)
MODELS=(
  # 'Qwen/Qwen2.5-7B-Instruct'
  # 'mistralai/Mistral-7B-Instruct-v0.3'
  # '01-ai/Yi-Coder-9B-Chat'
  # 'google/codegemma-7b-it'
  # 'Qwen/Qwen3-4B'
  # 'Qwen/Qwen3-8B'
  # 'Qwen/Qwen3-32B'
  # 'meta-llama/Llama-3.2-1B-Instruct'
  # 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
  # 'meta-llama/Llama-3.2-3B-Instruct'
  # 'microsoft/Phi-3-mini-128k-instruct'
  # 'microsoft/Phi-3.5-mini-instruct'
  # 'meta-llama/Llama-3.1-8B-Instruct'
  # 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'
  # 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
  # 'microsoft/phi-4'
  # 'microsoft/phi-2'
  # 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
  # 'bigcode/starcoder2-15b-instruct-v0.1'
  # 'mistralai/Codestral-22B-v0.1'
  'Qwen/QwQ-32B'
  # 'Qwen/Qwen2.5-Coder-32B-Instruct'
  # 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
  # 'deepseek-ai/deepseek-coder-33b-instruct'
  # 'codellama/CodeLlama-34b-Instruct-hf'
  # 'codellama/CodeLlama-70b-Python-hf'
  # 'meta-llama/Llama-3.1-70B-Instruct'
  # 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
  # 'WizardLMTeam/WizardLM-70B-V1.0'
  # 'Qwen/Qwen2.5-72B-Instruct'
  # 'openai/gpt-oss-20b'
  # ── OpenAI API models (no GPU needed; uses --use_openai path) ──
  # 'gpt-4.1-mini'
  # 'gpt-4o-mini'
)
TEMPS=(0.2) # (0.2 0.8)
NUM_COMPLETIONS=5

# ─── Phase 1: self-resubmit as a correctly-sized array ───────────
# When submitted directly (no SLURM_ARRAY_TASK_ID), compute the grid
# size and resubmit with the exact --array range.  The command-line
# --array overrides the #SBATCH --array placeholder above.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  N=$(( ${#ALGS[@]} * ${#MODELS[@]} * ${#TEMPS[@]} ))
  echo "Grid: ${#ALGS[@]} algs × ${#MODELS[@]} models × ${#TEMPS[@]} temps = ${N} tasks"
  echo "Submitting array of ${N} tasks (indices 0–$((N-1)))..."
  sbatch --array="0-$((N-1))" "$0"
  exit 0
fi

#–– Which GPU did I get?
nvidia-smi --list-gpus

# ─── ENV SETUP ────────────────────────────────────────────────────
source ~/miniforge3/bin/activate
conda activate /lus/lfs1aip2/projects/u6cg/nmaveli/nmaveli/conda-envs/envs/code-retrieval-llms

# Models are stored in a flat layout at this path (not the default ~/.cache structure).
# resolve_local_snapshot_dir() will find e.g. .../models/Qwen2.5-7B-Instruct/
# as a fallback when the standard HF hub cache structure is absent.
export HF_CACHE_DIR="/lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models"

# Skip pip install if requirements are already satisfied (saves ~30s per task)
python -c "import pkg_resources; pkg_resources.require(open('requirements.txt').read().splitlines())" 2>/dev/null \
  || { python -m pip install --upgrade pip && python -m pip install -r requirements.txt; }

# ─── MAP ARRAY ID → (ALG, MODEL, TEMP) ───────────────────────────
IDX=$SLURM_ARRAY_TASK_ID
ALG_IDX=$(( IDX / ( ${#MODELS[@]} * ${#TEMPS[@]} ) ))
ALG=${ALGS[$ALG_IDX]}

REST=$(( IDX % ( ${#MODELS[@]} * ${#TEMPS[@]} ) ))
MODEL_IDX=$(( REST / ${#TEMPS[@]} ))
TEMP_IDX=$(( REST % ${#TEMPS[@]} ))

MODEL=${MODELS[$MODEL_IDX]}
T=${TEMPS[$TEMP_IDX]}

# Guard: if the array was submitted with more task IDs than the active grid,
# (e.g. 256 IDs when only 8 are valid) exit cleanly for the excess tasks.
if [[ -z "$ALG" || -z "$MODEL" || -z "$T" ]]; then
  echo "[$SLURM_JOB_ID:$SLURM_ARRAY_TASK_ID] Task ID out of grid range (ALG='$ALG' MODEL='$MODEL' T='$T'); nothing to do."
  exit 0
fi

echo "[$SLURM_JOB_ID:$SLURM_ARRAY_TASK_ID] → ALG=$ALG | MODEL=$MODEL | T=$T"

# ─── DIRECTORIES ────────────────────────────────────────────────
LOG_DIR="logs/inference/${ALG}"
mkdir -p "${LOG_DIR}"

DATA_DIR="data/processed/${ALG}"
INPUT_JSON="${DATA_DIR}/data.jsonl"
MSG_FILE="${DATA_DIR}/codeio_1k_msg.jsonl"
GENS_PREFIX="${DATA_DIR}/codeio_1k_gens"

echo "[$SLURM_JOB_ID] Preparing data for $ALG"
python scripts/generate_data.py \
  --algorithms "${ALG}" \
  --source mixed \
  --count 50


echo "[$SLURM_JOB_ID] Building prompts for $ALG"
python src/data/build_codeio_msg.py \
  --input_file  "${INPUT_JSON}" \
  --output_file "${MSG_FILE}" \
  --algorithm   "${ALG}" \
  --prompt_type zero_shot \
  --blind



# ─── INFERENCE ───────────────────────────────────────────────────
# Detect OpenAI API models:
#   - no '/' in name (e.g. gpt-4o-mini) → not a HuggingFace repo ID
#   - 'openai/' prefix (e.g. openai/gpt-oss-20b) → OpenAI-compatible API, not a local HF model
if [[ "$MODEL" != *"/"* || "$MODEL" == openai/* ]]; then
  IS_OPENAI=true
else
  IS_OPENAI=false
fi

TP_SIZES=(1 2 4 8)
# Reasoning models (QwQ, DeepSeek-R1, Qwen3) generate long <think> blocks; cap at 16K
# to avoid KV-cache exhaustion and throughput collapse in the tail of a batch.
# Non-reasoning models rarely need more than 16K tokens per answer anyway.
if echo "${MODEL}" | grep -qiE "QwQ|DeepSeek-R1|Qwen3"; then
  INITIAL_MAX_LENGTH=16384
else
  INITIAL_MAX_LENGTH=32768
fi
MIN_MAX_LENGTH=2048         # do not go below this

MODEL_TAG=$(echo "${MODEL##*/}" | tr '[:upper:]-' '[:lower:]_')
OUT_FILE="${GENS_PREFIX}_model_${MODEL_TAG}_temp_${T}_n${NUM_COMPLETIONS}.jsonl"
VERIF_FILE="${OUT_FILE%.jsonl}_verified.jsonl"

if [ "$IS_OPENAI" = true ]; then
  # ── OpenAI API path (no GPU / Singularity required) ────────────
  echo "[$SLURM_JOB_ID] Calling OpenAI API: MODEL=${MODEL} | T=${T}"
  python src/inference/batched_api_inference.py \
    --model           "${MODEL}" \
    --input           "${MSG_FILE}" \
    --output          "${OUT_FILE}" \
    --temperature     "${T}" \
    --num_completions "${NUM_COMPLETIONS}" \
    --workers         "${OPENAI_WORKERS:-64}" \
    --use_openai

else
  # ── vLLM path (GPU / Singularity) ──────────────────────────────
  # start with FlashAttention enabled
  export VLLM_USE_V1=1

  # Choose a sensible starting TP to avoid wasted OOM retries:
  #   ≥70B params → start at TP=4 (must span ≥2 GPUs to fit in VRAM)
  #   30–33B     → start at TP=1 (fits on a single GH200 120GB; TP=1 avoids
  #                inter-GPU SymmetricMemory issues, progression 1→2→4→8)
  #   else       → start at TP=1
  if echo "${MODEL}" | grep -qiE "70[bB]|72[bB]"; then
    START_TP_IDX=2   # TP_SIZES[2] = 4
  else
    START_TP_IDX=0   # TP_SIZES[0] = 1
  fi

  for (( TPI=START_TP_IDX; TPI<${#TP_SIZES[@]}; TPI++ )); do
    TP=${TP_SIZES[$TPI]}
    ML=$INITIAL_MAX_LENGTH

    while true; do
      LOG_FILE="${LOG_DIR}/${MODEL_TAG}_T${T}_tp${TP}_ml${ML}.log"
      echo "[$SLURM_JOB_ID] Trying tp_size=${TP}, max_length=${ML}, VLLM_USE_V1=${VLLM_USE_V1}" | tee "${LOG_FILE}"

      # choose tmp or final out
      CURRENT_OUT=${OUT_TMP:-${OUT_FILE}}

      # Redirect ALL vllm and inductor caches to /tmp to avoid Lustre stale file
      # handle errors ([Errno 116]). vllm uses VLLM_CACHE_ROOT for its torch_compile_cache;
      # PyTorch inductor uses TORCHINDUCTOR_CACHE_DIR. Both must point to local storage.
      LOCAL_CACHE="/tmp/vllm_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
      mkdir -p "${LOCAL_CACHE}"

      singularity exec --nv \
        --writable-tmpfs \
        --bind "$(pwd)":/workspace \
        --bind "${HF_CACHE_DIR}:${HF_CACHE_DIR}" \
        --bind "${LOCAL_CACHE}:${LOCAL_CACHE}" \
        /projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.11.sif \
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

      # vLLM V1 WorkerProc / EngineCore init failure (e.g. SymmetricMemory socket,
      # NCCL hang, GPU comm error) → move to next TP size.
      # Note: VLLM_USE_V1=0 is broken in this vLLM dev build (it selects V1 internally
      # for GH200 then raises ValueError on the mismatch), so we never set it.
      if grep -qi "WorkerProc initialization failed\|Engine core initialization failed" "${LOG_FILE}"; then
        echo "[$SLURM_JOB_ID] WorkerProc/EngineCore init failed @ tp=${TP}; moving to next TP" | tee -a "${LOG_FILE}"
        break
      fi

      # Safety net: if VLLM_USE_V1=0 was ever set and caused the V1/V0 mismatch
      # ValueError, revert to V1 and move on.
      if grep -qi "envs\.VLLM_USE_V1=False\|VLLM_USE_V1=False" "${LOG_FILE}"; then
        export VLLM_USE_V1=1
        echo "[$SLURM_JOB_ID] V0 backend incompatible with this vLLM build; reverting VLLM_USE_V1=1 and moving to next TP" | tee -a "${LOG_FILE}"
        break
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

fi

# ─── MERGE RESUMED OUTPUT ───────────────────────────────────────
if [ -n "${OUT_TMP}" ]; then
  echo "[$SLURM_JOB_ID] Merging ${OUT_TMP} into ${OUT_FILE}..."
  cat "${OUT_TMP}" >> "${OUT_FILE}"
  rm "${OUT_TMP}"
  echo "[$SLURM_JOB_ID] Now have $(wc -l < "${OUT_FILE}")/$(wc -l < "${DATA_DIR}/codeio_1k_msg.jsonl") lines."
fi

# ─── VERIFY OUTPUT ───────────────────────────────────────────────
if [[ ! -f "${OUT_FILE}" ]]; then
  echo "[$SLURM_JOB_ID] Output file missing (inference likely failed); skipping verification." >&2
  exit 1
fi

python src/eval/check_io_pred_acc_mp.py \
  --parsed_file_name "${INPUT_JSON}" \
  --pred_file_name   "${OUT_FILE}" \
  --res_file_name    "${VERIF_FILE}" \
  --algo             "${ALG}"

echo "[$SLURM_JOB_ID] Done."