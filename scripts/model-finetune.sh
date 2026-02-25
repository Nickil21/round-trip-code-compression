#!/usr/bin/env bash
# Compatibility wrapper: finetune inference now lives in zero-shot-inference.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export INFERENCE_MODE=finetune
exec "${SCRIPT_DIR}/zero-shot-inference.sh" "$@"
