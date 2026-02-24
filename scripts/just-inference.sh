#!/usr/bin/env bash
#SBATCH --job-name=io-pred-inference
#SBATCH --output=logs/slurm/io-pred-inference.%A_%a.out
#SBATCH --error=logs/slurm/io-pred-inference.%A_%a.err
#SBATCH --partition=workq
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

set -euo pipefail

nvidia-smi --list-gpus

# ─── Environment Setup ────────────────────────────────────────────
source ~/miniforge3/bin/activate
conda activate /lus/lfs1aip2/projects/u6cg/nmaveli/nmaveli/conda-envs/envs/code-retrieval-llms

cd "$(pwd)"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# ─── Verification Targets ───────────────────────────────────────────
ALGORITHMS=(lzw ae rle huffman)
echo "[$SLURM_JOB_ID:${SLURM_ARRAY_TASK_ID:-0}] Running verification for algorithms: ${ALGORITHMS[*]}"

# ─── Configuration ────────────────────────────────────────────────
TEMPERATURES=(0.2) # (0.2 0.8)
NUM_COMPLETIONS=5
models=(
  # 'Qwen/Qwen2.5-7B-Instruct'
  # 'mistralai/Mistral-7B-Instruct-v0.3'
  # '01-ai/Yi-Coder-9B-Chat'
  # 'google/codegemma-7b-it'
  # # 'Qwen/Qwen3-4B'
  # # 'Qwen/Qwen3-8B'
  # # 'Qwen/Qwen3-32B'
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
  # 'gpt-4o-mini'
  # 'openai/gpt-oss-20b'
  # 'gpt-4.1-mini'
)

# ─── Verification ──────────────────────────────────────────────────
for ALG in "${ALGORITHMS[@]}"; do
  LOG_DIR="logs/inference/${ALG}"
  mkdir -p "${LOG_DIR}"

  DATA_DIR="data/processed/${ALG}"
  INPUT_JSON="${DATA_DIR}/data.jsonl"
  GENS_PREFIX="${DATA_DIR}/codeio_1k_gens"

  for model in "${models[@]}"; do
    MODEL_NAME="${model##*/}"
    MODEL_NAME="${MODEL_NAME,,}"
    MODEL_NAME="${MODEL_NAME//-/_}"

    for T in "${TEMPERATURES[@]}"; do
      echo "[$SLURM_JOB_ID:${SLURM_ARRAY_TASK_ID:-0}] → $ALG | $MODEL_NAME @ T=$T"
      OUT_FILE="${GENS_PREFIX}_model_${MODEL_NAME}_temp_${T}_n${NUM_COMPLETIONS}.jsonl"
      LOG_FILE="${LOG_DIR}/${MODEL_NAME}_temp_${T}_n${NUM_COMPLETIONS}.log"
      RES_FILE="${OUT_FILE%.jsonl}_verified.jsonl"

      if [[ ! -f "${OUT_FILE}" ]]; then
        echo "[$SLURM_JOB_ID:${SLURM_ARRAY_TASK_ID:-0}] Missing predictions file: ${OUT_FILE}; skipping."
        continue
      fi

      python src/eval/check_io_pred_acc_mp.py \
        --parsed_file_name "${INPUT_JSON}" \
        --pred_file_name   "${OUT_FILE}" \
        --res_file_name    "${RES_FILE}" \
        --algo             "${ALG}"
    done
  done
done
