# jq -s '.' processed_datasets/rle/training_data_sft.jsonl > processed_datasets/rle/training_data_sft.json

#!/usr/bin/env bash
set -euo pipefail

# 1) Base model
MODEL_NAME="01-ai/Yi-Coder-9B-Chat"
TRUST_REMOTE_CODE=true

# 2) Method / fine-tuning setup
STAGE="sft"
DO_TRAIN=true
FINETUNING_TYPE="lora"
LORA_RANK=8
LORA_TARGET="all"

# 3) Dataset
DATASET="rle_trace"
TEMPLATE="yi"
CUTOFF_LEN=2048
MAX_SAMPLES=1000
OVERWRITE_CACHE=true
PREPROCESS_WORKERS=16
DATALOADER_WORKERS=4

# 4) Output & logging
OUTPUT_DIR="finetune/rle/Yi-Coder-9B-Chat/lora/sft"
LOGGING_STEPS=10
SAVE_STEPS=500
PLOT_LOSS=true
OVERWRITE_OUTPUT_DIR=true
SAVE_ONLY_MODEL=false
REPORT_TO="none"    # [none, wandb, tensorboard, swanlab, mlflow]

# 5) Training hyperparams
BATCH_SIZE=1
GRAD_ACCUM=8
LR=1.0e-4
EPOCHS=3.0
SCHEDULER="cosine"
WARMUP_RATIO=0.1
FP16=true
DDP_TIMEOUT=180000000
RESUME_CHECKPOINT="null"

# 6) (Optional) Eval settings are commented out – uncomment to enable
# EVAL_DATASET="alpaca_en_demo"
# VAL_SIZE=0.1
# EVAL_BATCH=1
# EVAL_STRATEGY="steps"
# EVAL_STEPS=500

# Launch the fine-tune
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
  --resume_from_checkpoint ${RESUME_CHECKPOINT}
  # --eval_dataset ${EVAL_DATASET} \
  # --val_size ${VAL_SIZE} \
  # --per_device_eval_batch_size ${EVAL_BATCH} \
  # --eval_strategy ${EVAL_STRATEGY} \
  # --eval_steps ${EVAL_STEPS}
