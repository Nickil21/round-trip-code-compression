#!/usr/bin/env bash
#SBATCH --job-name=io-pred-inference
#SBATCH --output=slurm-logs/io-pred-inference.%A_%a.out
#SBATCH --error=slurm-logs/io-pred-inference.%A_%a.err
#SBATCH --partition=workq
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

nvidia-smi --list-gpus

# ─── Environment Setup ────────────────────────────────────────────
source ~/miniforge3/bin/activate
conda activate round-trip-myenv \
  || (conda create -y -n round-trip-myenv python=3.10 && conda activate round-trip-myenv)

cd "$(pwd)"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# ─── Pick one algorithm per array task ────────────────────────────
ALGORITHMS=(rle lzw ae huffman)
ALG="${ALGORITHMS[$SLURM_ARRAY_TASK_ID]}"
echo "[$SLURM_JOB_ID:$SLURM_ARRAY_TASK_ID] Running with algorithm: $ALG"

# ─── Configuration ────────────────────────────────────────────────
TEMPERATURES=(0.2 0.8)
NUM_COMPLETIONS=5
models=(
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

LOG_DIR="logs/${ALG}"
mkdir -p "${LOG_DIR}"

# ─── Inference + Verification ─────────────────────────────────────
export RAY_TMPDIR=/tmp/ray
ray start --head --temp-dir /tmp/ray --dashboard-host 0.0.0.0

DATA_DIR="processed_datasets/${ALG}"
INPUT_JSON="${DATA_DIR}/data.jsonl"
MSG_FILE="${DATA_DIR}/codeio_1k_msg.jsonl"
GENS_PREFIX="${DATA_DIR}/codeio_1k_gens"

for model in "${models[@]}"; do
  MODEL_NAME="${model##*/}"
  MODEL_NAME="${MODEL_NAME,,}"
  MODEL_NAME="${MODEL_NAME//-/_}"

  for T in "${TEMPERATURES[@]}"; do
    echo "[$SLURM_JOB_ID:$SLURM_ARRAY_TASK_ID] → $ALG | $MODEL_NAME @ T=$T"
    OUT_FILE="${GENS_PREFIX}_model_${MODEL_NAME}_temp_${T}_n${NUM_COMPLETIONS}.jsonl"
    LOG_FILE="${LOG_DIR}/${MODEL_NAME}_temp_${T}_n${NUM_COMPLETIONS}.log"
    RES_FILE="${OUT_FILE%.jsonl}_verified.jsonl"

    python src/check_io_pred_acc_mp.py \
      --parsed_file_name "${INPUT_JSON}" \
      --pred_file_name   "${OUT_FILE}" \
      --res_file_name    "${RES_FILE}" \
      --algo             "${ALG}"
  done
done

ray stop
