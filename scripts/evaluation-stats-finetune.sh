#!/usr/bin/env bash
#SBATCH --job-name=codeio-eval-finetune
#SBATCH --output=logs/slurm/codeio-eval-finetune.%A_%a.out
#SBATCH --error=logs/slurm/codeio-eval-finetune.%A_%a.err
#SBATCH --partition=workq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Compatibility wrapper: finetune evaluation now lives in evaluation-stats.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EVAL_MODE=finetune
exec "${SCRIPT_DIR}/evaluation-stats.sh" "$@"
