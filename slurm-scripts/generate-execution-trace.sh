#!/usr/bin/env bash
#SBATCH --job-name=gen_exec_trace
#SBATCH --output=slurm-logs/gen_exec_trace.%j.out
#SBATCH --error=slurm-logs/gen_exec_trace.%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G

set -euxo pipefail

# ─── Which algorithms to run ────────────────────────────────────────────
ALGORITHMS=(rle huffman ae lzw)

WORKDIR=$(pwd)
CONTAINER=/projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.06.sif
RUN_PY=/py3.10/bin/python

# ─── One-time environment setup ─────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate round-trip-myenv 2>/dev/null \
  || { conda create -y -n round-trip-myenv python=3.10 && conda activate round-trip-myenv; }

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
  DATADIR="${WORKDIR}/processed_datasets/${ALGO}"
  OUTPUTDIR="${WORKDIR}/LLaMA-Factory/data"
  LOGDIR="${WORKDIR}/logs/${ALGO}"
  mkdir -p "$DATADIR" "$LOGDIR"

  # data-prep steps
  python tasks/generate_data.py  --algorithms "$ALGO" --source mixed --count 50
  python src/build_codeio_msg.py        \
         --input_file "${DATADIR}/data.jsonl"         \
         --output_file "${DATADIR}/codeio_1k_msg.jsonl" \
         --algorithm "$ALGO" \
         --prompt_type zero_shot
      
  python src/generate_execution_trace.py --algorithm "$ALGO"
  python src/filter_execution_trace.py   --algorithm "$ALGO"

  # translation step: uses all 8 GPUs (no manual subset math)
  singularity exec --nv \
    --bind /home/u5an/nmaveli.u5an/projects/round-trip-code-compression:/workspace \
    /projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.06.sif \
    bash -lc "cd /workspace && \
      /py3.10/bin/python src/execution_trace_translation.py \
        --algorithm $ALGO \
        --translator_model mistralai/Mistral-7B-Instruct-v0.3 \
        --num_gpus 1"  \
    > "${LOGDIR}/translated_exec_trace.log" 2>&1

  echo "[$ALGO] translation log → ${LOGDIR}/translated.log"

  # data construction for SFT
  python src/data_construction_sft.py --algorithm "$ALGO" --output_file "${OUTPUTDIR}/${ALGO}_training_data_sft.jsonl" --trained_model Qwen/QwQ-32B
  jq -s '.' "${OUTPUTDIR}/${ALGO}_training_data_sft.jsonl" > "${OUTPUTDIR}/${ALGO}_training_data_sft.json"

done

# ray stop
