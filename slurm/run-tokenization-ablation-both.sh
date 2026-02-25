#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL="${1:-Qwen/QwQ-32B}"
TEMP="${2:-0.2}"

jid_in=$(TASK_FAMILY=input sbatch scripts/tokenization-ablation.sh "${MODEL}" "${TEMP}" | awk '{print $4}')
jid_out=$(TASK_FAMILY=output sbatch scripts/tokenization-ablation.sh "${MODEL}" "${TEMP}" | awk '{print $4}')

echo "Submitted input array job:  ${jid_in}"
echo "Submitted output array job: ${jid_out}"
echo "Monitor with: squeue -j ${jid_in},${jid_out}"
