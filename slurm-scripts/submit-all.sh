#!/usr/bin/env bash
#SBATCH --job-name=submit-all
#SBATCH --output=slurm-logs/submit-all.out
#SBATCH --error=slurm-logs/submit-all.err
#SBATCH --time=00:05:00        # just needs to run for a few seconds
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

# Submit each real workload script
sbatch slurm-scripts/generate-execution-trace.sh
sbatch slurm-scripts/input-output-prediction-1.sh
sbatch slurm-scripts/input-output-prediction-2.sh
sbatch slurm-scripts/input-output-prediction-3.sh
sbatch slurm-scripts/input-output-prediction-4.sh