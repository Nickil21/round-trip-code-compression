#!/usr/bin/env bash
#SBATCH --job-name=gen_exec_trace
#SBATCH --output=logs/slurm/gen_exec_trace.%j.out
#SBATCH --error=logs/slurm/gen_exec_trace.%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G

set -euxo pipefail

# ─── Which algorithms to run ────────────────────────────────────────────
ALGORITHMS=(ae)

WORKDIR=$(pwd)
CONTAINER=/projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.11.sif
RUN_PY=/opt/python/pkgs/python-3.12.11/bin/python

# ─── One-time environment setup ─────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /lus/lfs1aip2/projects/u6cg/nmaveli/nmaveli/conda-envs/envs/code-retrieval-llms

LOCKFILE="${WORKDIR}/.env_installed"
if [[ ! -f "$LOCKFILE" ]]; then
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
  touch "$LOCKFILE"
fi

nvidia-smi --list-gpus


# ─── Inference + Verification Loop ─────────────────────────────
# export RAY_TMPDIR=/tmp/ray
# ray start --head --temp-dir /tmp/ray --dashboard-host 0.0.0.0  

# ─── Loop over algorithms, sequentially ─────────────────────────────────
for ALGO in "${ALGORITHMS[@]}"; do
  DATADIR="${WORKDIR}/data/processed/${ALGO}"
  OUTPUTDIR="${WORKDIR}/LLaMA-Factory/data"
  LOGDIR="${WORKDIR}/logs/inference/${ALGO}"
  mkdir -p "$DATADIR" "$LOGDIR"

  # data-prep steps
  python scripts/generate_data.py  --algorithms "$ALGO" --source mixed --count 50
  python src/data/build_codeio_msg.py        \
         --input_file "${DATADIR}/data.jsonl"         \
         --output_file "${DATADIR}/codeio_1k_msg.jsonl" \
         --algorithm "$ALGO" \
         --prompt_type zero_shot
      
  python src/data/generate_execution_trace.py --algorithm "$ALGO"
  python src/data/filter_execution_trace.py   --algorithm "$ALGO"

  # translation step: uses all 8 GPUs (no manual subset math)
  singularity exec --nv \
    --bind "${WORKDIR}":/workspace \
    /projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.11.sif \
    bash -lc "cd /workspace && \
      /opt/python/pkgs/python-3.12.11/bin/python src/data/execution_trace_translation.py \
        --algorithm $ALGO \
        --translator_model mistralai/Mistral-7B-Instruct-v0.3 \
        --num_gpus 1"  \
    > "${LOGDIR}/translated_exec_trace.log" 2>&1

  echo "[$ALGO] translation log → ${LOGDIR}/translated.log"

  # data construction for SFT
  python src/data/data_construction_sft.py --algorithm "$ALGO" --output_file "${OUTPUTDIR}/${ALGO}_training_data_sft.jsonl" --trained_model Qwen/QwQ-32B
  jq -s '.' "${OUTPUTDIR}/${ALGO}_training_data_sft.jsonl" > "${OUTPUTDIR}/${ALGO}_training_data_sft.json"

done

# ray stop
