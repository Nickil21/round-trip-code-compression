#!/usr/bin/env bash
#SBATCH --job-name=codeio-self-reflect
#SBATCH --output=logs/slurm/codeio-self-reflect.%A_%a.out
#SBATCH --error=logs/slurm/codeio-self-reflect.%A_%a.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G

set -euxo pipefail

# ─── Which algorithms to run ────────────────────────────────────────────
ALGORITHMS=(ae lzw rle huffman)

WORKDIR=$(pwd)
CONTAINER=/projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.11.sif
RUN_PY=/opt/python/pkgs/python-3.12.11/bin/python

# ─── Load secrets from .env (never commit this file) ────────────────────────
if [[ -f "${WORKDIR}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${WORKDIR}/.env"
  set +a
fi

# Skip non-matching globs instead of iterating with the literal pattern string
shopt -s nullglob

TASKLIST="${WORKDIR}/.self-reflect-tasks.txt"

# ── Phase 1: enumerate tasks and re-submit as a job array ────────────────────
# Runs when the script is submitted directly (SLURM_ARRAY_TASK_ID is unset).
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  rm -f "${TASKLIST}"
  for ALGO in "${ALGORITHMS[@]}"; do
    for INPUT in "data/processed/${ALGO}"/*_verified.jsonl; do
      printf '%s\t%s\n' "${ALGO}" "${INPUT}" >> "${TASKLIST}"
    done
  done

  N=$(wc -l < "${TASKLIST}")
  if [[ "${N}" -eq 0 ]]; then
    echo "No *_verified.jsonl files found — nothing to submit." >&2
    exit 1
  fi

  echo "Submitting array of ${N} tasks (indices 0–$((N-1)))..."
  sbatch --array="0-$((N-1))" "$0"
  exit 0
fi

# ── Phase 2: process the file assigned to this array task ────────────────────
# Read line SLURM_ARRAY_TASK_ID+1 (awk is 1-indexed) from the task list.
LINE=$(awk -v n="${SLURM_ARRAY_TASK_ID}" 'NR == n+1 { print; exit }' "${TASKLIST}")
ALGO="${LINE%%$'\t'*}"
INPUT="${LINE##*$'\t'}"

if [[ -z "${ALGO}" || -z "${INPUT}" || ! -f "${INPUT}" ]]; then
  echo "ERROR: could not resolve task ${SLURM_ARRAY_TASK_ID} from ${TASKLIST}" >&2
  exit 1
fi

nvidia-smi --list-gpus

LOGDIR="${WORKDIR}/logs/inference/${ALGO}-reflect"
mkdir -p "$LOGDIR"

BASENAME="$(basename "${INPUT}" .jsonl)"
OUTPUT="data/processed/${ALGO}-reflect/${BASENAME}.jsonl"
STATS="data/processed/${ALGO}-reflect/${BASENAME}_stats.csv"

MODEL="$(python - "${INPUT}" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        print(obj.get("model",""))
        break
PY
)"
TEMP="$(basename "${INPUT}" | sed -n 's/.*_temp_\([0-9.]\+\)_.*/\1/p')"
TEMP="${TEMP:-0.2}"   # default if filename doesn't match _temp_X.X_ pattern

# OpenAI models have no "/" in their name (e.g. gpt-4.1-mini).
# Local HF models always use "org/name" format (e.g. Qwen/QwQ-32B).
if [[ "${MODEL}" != */* ]]; then
  # ── OpenAI API model ──────────────────────────────────────────────
  singularity exec --nv \
    --bind "${WORKDIR}":/workspace \
    "${CONTAINER}" \
    bash -lc "unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE; cd /workspace && \
      ${RUN_PY} src/inference/self_reflection.py \
        --input '${INPUT}' \
        --output '${OUTPUT}' \
        --model '${MODEL}' \
        --temperature '${TEMP}' \
        --algo '${ALGO}' \
        --use_openai \
        --critique_style B \
        --gt_stop_on em \
        --reflection_rounds 2 \
        --on_mismatch annotate \
        --max_tokens 1024 \
        --max_model_len 8192 \
        --model_ctx 8192 \
        --gen_tokens 512 \
        --safety_margin 64 \
        --truncate_hard_chars 16000 \
        --chars_per_token 1.5 \
        --per_item_stats_csv '${STATS}'" \
  > "${LOGDIR}/${BASENAME}.log" 2>&1
else
  # ── Local / HF model (vLLM) ───────────────────────────────────────
  singularity exec --nv \
    --bind "${WORKDIR}":/workspace \
    "${CONTAINER}" \
    bash -lc "cd /workspace && \
      ${RUN_PY} src/inference/self_reflection.py \
        --input '${INPUT}' \
        --output '${OUTPUT}' \
        --model '${MODEL}' \
        --temperature '${TEMP}' \
        --algo '${ALGO}' \
        --num_gpus 1 \
        --max_model_len 8192 \
        --gpu_memory_utilization 0.9 \
        --critique_style B \
        --gt_stop_on em \
        --reflection_rounds 2 \
        --hf_offline \
        --cache_dir '${HF_CACHE_DIR:-/lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache}' \
        --on_mismatch annotate \
        --max_tokens 1024 \
        --model_ctx 8192 \
        --gen_tokens 512 \
        --safety_margin 64 \
        --truncate_hard_chars 16000 \
        --chars_per_token 1.5 \
        --per_item_stats_csv '${STATS}'" \
  > "${LOGDIR}/${BASENAME}.log" 2>&1
fi
