#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="/mnt_path/miniconda3"
ENV_NAME="round-trip-code-compression-env"
ENV_PATH="${CONDA_ROOT}/envs/${ENV_NAME}"
PYTHON_VERSION="3.11"

# ─── Init conda ───────────────────────────────────────────────────────────────
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
else
  source "${CONDA_ROOT}/bin/activate"
fi

# ─── Create env if it doesn't exist ──────────────────────────────────────────
if [[ ! -d "${ENV_PATH}" ]]; then
  echo "Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
  conda create -y --prefix "${ENV_PATH}" python="${PYTHON_VERSION}"
else
  echo "Conda env '${ENV_NAME}' already exists, skipping creation."
fi

conda activate "${ENV_PATH}"

# ─── Install packages ─────────────────────────────────────────────────────────
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# vllm is commented out in requirements.txt (too heavy for non-inference envs);
# install it explicitly here for inference use.
echo "Installing vllm..."
pip install vllm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALIASES_LINE="source ${SCRIPT_DIR}/docker/tmux_aliases.sh"
if ! grep -qF "${ALIASES_LINE}" "$HOME/.bashrc" 2>/dev/null; then
  echo "${ALIASES_LINE}" >> "$HOME/.bashrc"
fi

echo ""
echo "Done. To activate:"
echo "  source ${CONDA_ROOT}/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_PATH}"
echo ""
echo "tmux aliases added to ~/.bashrc (tn/ta/tl/tk/ts/tw/trn)."
echo "Run: source ~/.bashrc"
