#!/bin/bash
#SBATCH --job-name=io-pred
#SBATCH --output=slurm-logs/io-pred.%j.out
#SBATCH --error=slurm-logs/io-pred.%j.err
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G

nvidia-smi --list-gpus

# ─── Environment Setup ────────────────────────────────────────────
source ~/miniforge3/bin/activate
conda activate round-trip-myenv || (conda create -y -n round-trip-myenv python=3.10 && conda activate round-trip-myenv)

cd "$(pwd)"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# ─── Configuration ────────────────────────────────────────────────
ALGORITHMS=("lzw" "ae" "rle")
TEMPERATURES=(0.2 0.8)
MAX_TOKENS=4096  # Adjust as needed
models=(
  'bigcode/starcoder2-15b-instruct-v0.1'
  'codellama/CodeLlama-70b-Instruct-hf'
  'codellama/CodeLlama-34b-Instruct-hf'
  'microsoft/Phi-3-mini-4k-instruct'
)
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"


# ─── Prepare input and output data ────────────
for ALGORITHM in "${ALGORITHMS[@]}"; do
  echo "=== Preparing & processing for $ALGORITHM ==="
  python tasks/generate_data.py --algorithms ${ALGORITHM} --source mixed --count 20
done

# ─── Build prompt messages per algorithm ────────────
for ALGORITHM in "${ALGORITHMS[@]}"; do
  python src/build_codeio_msg.py \
    --input_file "processed_datasets/${ALGORITHM}/data.jsonl" \
    --output_file "processed_datasets/${ALGORITHM}/codeio_1k_msg.jsonl" \
    --algorithm "${ALGORITHM}"
done

# ─── Inference + Verification Loop ─────────────────────────────
export RAY_TMPDIR=/tmp/ray
ray start --head --temp-dir /tmp/ray --dashboard-host 0.0.0.0  

for ALGORITHM in "${ALGORITHMS[@]}"; do
  echo "=== Algorithm: $ALGORITHM ==="
  DATA_DIR="processed_datasets/${ALGORITHM}"
  INPUT_JSON="${DATA_DIR}/data.jsonl"
  MSG_FILE="${DATA_DIR}/codeio_1k_msg.jsonl"
  GENS_PREFIX="${DATA_DIR}/codeio_1k_gens"
  ALG_LOG_DIR="${LOG_DIR}/${ALGORITHM}"
  mkdir -p "${ALG_LOG_DIR}"

  for model in "${models[@]}"; do
    echo "  Model: $model"
    MODEL_NAME="${model##*/}"
    MODEL_NAME="${MODEL_NAME,,}"
    MODEL_NAME="${MODEL_NAME//-/_}"

    for T in "${TEMPERATURES[@]}"; do
      echo "    Temp: $T"
      OUT_FILE="${GENS_PREFIX}_model_${MODEL_NAME}_temp_${T}.jsonl"
      LOG_FILE="${ALG_LOG_DIR}/${MODEL_NAME}_temp_${T}.log"
      RES_FILE="${GENS_PREFIX}_model_${MODEL_NAME}_temp_${T}_verified.jsonl"

      # Run inference INSIDE Singularity
      singularity exec --nv \
        --bind "$(pwd)":/workspace \
        /projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.06.sif \
        bash -c "
          cd /workspace
          /py3.10/bin/python src/batched_api_inference.py \
            --temperature ${T} \
            --input ${MSG_FILE} \
            --output ${OUT_FILE} \
            --model ${model} \
            --max_tokens ${MAX_TOKENS}
        " > "${LOG_FILE}" 2>&1

      # Run verification OUTSIDE Singularity
      python src/check_io_pred_acc_mp.py \
        --parsed_file_name "${INPUT_JSON}" \
        --pred_file_name   "${OUT_FILE}" \
        --res_file_name    "${RES_FILE}" \
        --algo             "${ALGORITHM}"
    done
  done
done

ray stop