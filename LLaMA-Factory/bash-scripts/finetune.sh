#!/usr/bin/env bash
#SBATCH --job-name=model_finetune_qwq32b
#SBATCH --output=slurm-logs/model_finetune.%j.out
#SBATCH --error=slurm-logs/model_finetune.%j.err
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G

# Example (if your sources are jsonl):
# jq -s '.' projects/codecs-ft/processed_datasets/rle/training_data_sft.jsonl > projects/codecs-ft/processed_datasets/rle/training_data_sft.json

module load cuda/12.6
module load cudatoolkit/24.11_12.6

set -euo pipefail

mkdir -p slurm-logs

ENV_NAME="llama-factory-env"

source ~/miniforge3/bin/activate
# create env if missing
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "🛠  Creating conda env '${ENV_NAME}'..."
  conda create -y -n "${ENV_NAME}" python=3.11
fi

conda activate "${ENV_NAME}"

# CUDA 12.6 wheels
pip install --upgrade \
  torch --index-url https://download.pytorch.org/whl/cu126 \
  torchvision --index-url https://download.pytorch.org/whl/cu126 \
  torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "=== NVIDIA-SMI ==="
nvidia-smi

# Install your project root requirements (if you have a requirements.txt there)
# pip install -r requirements.txt

REPO_DIR="LLaMA-Factory"
cd "${REPO_DIR}"

# Install LLaMA-Factory in editable mode
pip install -e ".[torch,metrics]" --no-build-isolation

# Pin datasets to a compatible range for LLaMA-Factory (avoid the 4.x error)
pip install "datasets>=2.16.0,<=3.6.0"

echo "=== Torch CUDA check ==="
python - <<'EOF'
import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device[0]:", torch.cuda.get_device_name(0))
EOF

# -------- Training config for Qwen/QwQ-32B + multi-algorithm loop -------- #

# 1) Base model
MODEL_NAME="Qwen/QwQ-32B"
TRUST_REMOTE_CODE=true

# 2) Method / fine-tuning setup
STAGE="sft"
DO_TRAIN=true
FINETUNING_TYPE="lora"
LORA_RANK=8
LORA_TARGET="all"

# 3) Dataset / Template / Preprocessing
# Expect dataset names like rle_trace, lzw_trace, ae_trace, huffman_trace
TEMPLATE="qwen"          # ✅ proper template for Qwen/QwQ models
CUTOFF_LEN=2048          # safer than 16384 for a 32B model
MAX_SAMPLES=1000
OVERWRITE_CACHE=true
PREPROCESS_WORKERS=16
DATALOADER_WORKERS=4

# 4) Output & logging
LOGGING_STEPS=10
SAVE_STEPS=500
PLOT_LOSS=true
OVERWRITE_OUTPUT_DIR=true
SAVE_ONLY_MODEL=false
REPORT_TO="none"         # [none, wandb, tensorboard, swanlab, mlflow]

# 5) Training hyperparams
BATCH_SIZE=1
GRAD_ACCUM=8
LR=1.0e-4
EPOCHS=3.0
SCHEDULER="cosine"
WARMUP_RATIO=0.1
FP16=true                # mixed precision for 32B with LoRA
DDP_TIMEOUT=180000000

# 6) Algorithms to run
ALGOS=("lzw" "ae" "rle" "huffman")

echo "🔎 Checking dataset config..."
if [[ ! -f "data/dataset_info.json" ]]; then
  echo "❌ data/dataset_info.json not found. Please create it to map *_trace to your dataset files."
  echo "   Example entry:"
  echo '   {"rle_execution_trace": {"file_name": "projects/codecs-ft/processed_datasets/rle/training_data_sft.json","formatting":"alpaca"}}'
  exit 1
fi

for algo in "${ALGOS[@]}"; do
  echo "🚀 Starting fine-tuning for: ${algo}"

  DATASET="${algo}_execution_trace"
  OUTPUT_DIR="finetune/${algo}/QwQ-32B/lora/sft"
  mkdir -p "${OUTPUT_DIR}"

  llamafactory-cli train \
    --model_name_or_path ${MODEL_NAME} \
    --trust_remote_code ${TRUST_REMOTE_CODE} \
    --stage ${STAGE} \
    --do_train ${DO_TRAIN} \
    --finetuning_type ${FINETUNING_TYPE} \
    --lora_rank ${LORA_RANK} \
    --lora_target ${LORA_TARGET} \
    --dataset ${DATASET} \
    --template ${TEMPLATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --max_samples ${MAX_SAMPLES} \
    --overwrite_cache ${OVERWRITE_CACHE} \
    --preprocessing_num_workers ${PREPROCESS_WORKERS} \
    --dataloader_num_workers ${DATALOADER_WORKERS} \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --plot_loss ${PLOT_LOSS} \
    --overwrite_output_dir ${OVERWRITE_OUTPUT_DIR} \
    --save_only_model ${SAVE_ONLY_MODEL} \
    --report_to ${REPORT_TO} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCHS} \
    --lr_scheduler_type ${SCHEDULER} \
    --warmup_ratio ${WARMUP_RATIO} \
    --fp16 ${FP16} \
    --ddp_timeout ${DDP_TIMEOUT} \
    --ddp_find_unused_parameters false
done
