#!/usr/bin/env bash
#SBATCH --job-name=codeio-eval-stats
#SBATCH --output=logs/slurm/codeio-eval-stats.%A_%a.out
#SBATCH --error=logs/slurm/codeio-eval-stats.%A_%a.err
#SBATCH --partition=workq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

EVAL_MODE="${EVAL_MODE:-zero_shot}"
case "${EVAL_MODE}" in
  zero_shot)
    JOB_LABEL="zero-shot"
    DATA_SUFFIX=""
    ;;
  finetune)
    JOB_LABEL="finetune"
    DATA_SUFFIX="-finetune"
    ;;
  *)
    echo "Unsupported EVAL_MODE='${EVAL_MODE}'. Use zero_shot or finetune." >&2
    exit 2
    ;;
esac

nvidia-smi --list-gpus || true

# Optional env setup. Set EVAL_SETUP_ENV=0 to skip.
EVAL_SETUP_ENV="${EVAL_SETUP_ENV:-1}"
if [[ "${EVAL_SETUP_ENV}" == "1" ]]; then
  # shellcheck disable=SC1091
  source ~/miniforge3/bin/activate
  conda activate /lus/lfs1aip2/projects/u6cg/nmaveli/nmaveli/conda-envs/envs/code-retrieval-llms
fi

# Configuration
ALGORITHMS_CSV="${ALGORITHMS:-lzw,ae,rle,huffman}"
MODEL_FILTER="${MODEL_FILTER:-}"   # e.g. MODEL_FILTER=qwq_32b
TEMP_FILTER="${TEMP_FILTER:-}"     # e.g. TEMP_FILTER=temp_0.2
VERIFY_JOBS="${VERIFY_JOBS:-1}"    # parallel workers
SKIP_IF_VERIFIED="${SKIP_IF_VERIFIED:-1}"

IFS=',' read -r -a ALGO_LIST <<< "${ALGORITHMS_CSV}"
for i in "${!ALGO_LIST[@]}"; do
  ALGO_LIST[$i]="${ALGO_LIST[$i]// /}"
done

log_prefix="[${SLURM_JOB_ID:-local}:${SLURM_ARRAY_TASK_ID:-0}]"
echo "${log_prefix} Running ${JOB_LABEL} evaluation stats"
echo "${log_prefix} Algorithms: ${ALGO_LIST[*]}"
echo "${log_prefix} Verify workers: ${VERIFY_JOBS}"

verify_one() {
  local out_file="$1"
  local alg="$2"
  local input_json="data/processed/${alg}/data.jsonl"
  local res_file="${out_file%.jsonl}_verified.jsonl"

  if [[ ! -f "${input_json}" ]]; then
    echo "${log_prefix} Missing parsed data for ${alg}: ${input_json}; skipping ${out_file}"
    return 0
  fi

  if [[ "${SKIP_IF_VERIFIED}" == "1" && -f "${res_file}" ]]; then
    echo "${log_prefix} Skip existing verified file: ${res_file}"
    return 0
  fi

  echo "${log_prefix} Verify ${JOB_LABEL} ${alg}: $(basename "${out_file}")"
  python src/eval/check_io_pred_acc_mp.py \
    --parsed_file_name "${input_json}" \
    --pred_file_name   "${out_file}" \
    --res_file_name    "${res_file}" \
    --algo             "${alg}"
}

declare -a TASKS=()
for ALG in "${ALGO_LIST[@]}"; do
  DATA_DIR="data/processed/${ALG}${DATA_SUFFIX}"
  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "${log_prefix} Missing ${JOB_LABEL} dir: ${DATA_DIR}; skipping."
    continue
  fi

  while IFS= read -r file; do
    base="$(basename "${file}")"
    [[ "${base}" == *_verified.jsonl ]] && continue
    [[ -n "${MODEL_FILTER}" && "${base}" != *"${MODEL_FILTER}"* ]] && continue
    [[ -n "${TEMP_FILTER}" && "${base}" != *"${TEMP_FILTER}"* ]] && continue
    TASKS+=("${ALG}|${file}")
  done < <(find "${DATA_DIR}" -maxdepth 1 -type f -name "codeio_1k_gens_model_*_temp_*_n*.jsonl" | sort)
done

if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "${log_prefix} No ${JOB_LABEL} prediction files matched filters. Nothing to verify."
  exit 0
fi

echo "${log_prefix} Found ${#TASKS[@]} ${JOB_LABEL} prediction files."

# Run sequentially or in background workers.
if [[ "${VERIFY_JOBS}" -le 1 ]]; then
  for task in "${TASKS[@]}"; do
    ALG="${task%%|*}"
    FILE="${task#*|}"
    verify_one "${FILE}" "${ALG}"
  done
else
  running=0
  for task in "${TASKS[@]}"; do
    ALG="${task%%|*}"
    FILE="${task#*|}"
    verify_one "${FILE}" "${ALG}" &
    ((running+=1))
    if (( running >= VERIFY_JOBS )); then
      wait -n
      ((running-=1))
    fi
  done
  wait
fi

echo "${log_prefix} ${JOB_LABEL} evaluation stats complete."
