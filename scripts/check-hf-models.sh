#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_CONFIG_FILE="${MODEL_CONFIG_FILE:-${ROOT_DIR}/configs/models.yaml}"
HF_CACHE_MODELS="${HF_CACHE_MODELS:-/lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models}"
CHECK_SCOPE="${CHECK_SCOPE:-all_hf}"   # active | all_hf
DOWNLOAD_MISSING="${DOWNLOAD_MISSING:-1}"   # 0 | 1
DOWNLOAD_JOBS="${DOWNLOAD_JOBS:-4}"   # parallel downloads when DOWNLOAD_MISSING=1
MISSING_FILE="$(mktemp /tmp/hf_missing_models.XXXXXX)"
trap 'rm -f "${MISSING_FILE}"' EXIT

echo "[hf-check] model config      : ${MODEL_CONFIG_FILE}"
echo "[hf-check] hf cache (models) : ${HF_CACHE_MODELS}"
echo "[hf-check] check scope       : ${CHECK_SCOPE}"
echo "[hf-check] download missing  : ${DOWNLOAD_MISSING}"
echo "[hf-check] download workers  : ${DOWNLOAD_JOBS}"

if [[ ! -f "${MODEL_CONFIG_FILE}" ]]; then
  echo "[hf-check][error] Missing model config: ${MODEL_CONFIG_FILE}" >&2
  exit 1
fi

if [[ ! -d "${HF_CACHE_MODELS}" ]]; then
  echo "[hf-check][error] Missing HF cache dir: ${HF_CACHE_MODELS}" >&2
  exit 1
fi

run_check() {
python - <<'PY' "${MODEL_CONFIG_FILE}" "${HF_CACHE_MODELS}" "${CHECK_SCOPE}" "${MISSING_FILE}"
import os
import sys
from pathlib import Path

import yaml


def get_hub_root(cache_models: str) -> str:
    # Accept either ".../models" or ".../models/hub"
    if cache_models.endswith("/hub"):
        return cache_models
    with_hub = os.path.join(cache_models, "hub")
    if os.path.isdir(with_hub):
        return with_hub
    return cache_models


def resolve_local_snapshot_dir(repo_id: str, hub_root: str, cache_models: str):
    # Standard HF cache layout: <hub_root>/models--ORG--MODEL/snapshots/<hash>/config.json
    safe_repo = repo_id.replace("/", "--")
    snaps_root = os.path.join(hub_root, f"models--{safe_repo}", "snapshots")
    if os.path.isdir(snaps_root):
        candidates = []
        for d in os.listdir(snaps_root):
            p = os.path.join(snaps_root, d)
            if os.path.isdir(p):
                candidates.append(p)
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for snap in candidates:
            if os.path.exists(os.path.join(snap, "config.json")):
                return snap

    # Flat layout fallback: <cache_models>/<ModelName>/config.json
    model_name = repo_id.split("/")[-1]
    flat_path = os.path.join(cache_models, model_name)
    if os.path.isdir(flat_path) and os.path.exists(os.path.join(flat_path, "config.json")):
        return flat_path

    return None


cfg_path = Path(sys.argv[1])
cache_models = sys.argv[2]
scope = sys.argv[3]
missing_file = Path(sys.argv[4])

if scope not in {"active", "all_hf"}:
    raise SystemExit(f"[hf-check][error] Invalid CHECK_SCOPE='{scope}'. Use 'active' or 'all_hf'.")

with cfg_path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
models = data.get("models", [])

targets = []
for m in models:
    if m.get("provider") != "hf":
        continue
    if scope == "active" and not m.get("active", False):
        continue
    model_id = m.get("id")
    if model_id:
        targets.append(model_id)

if not targets:
    print("[hf-check] No HF models matched this scope.")
    raise SystemExit(0)

hub_root = get_hub_root(cache_models)
print(f"[hf-check] Resolved hub root: {hub_root}")

missing = []
for model_id in targets:
    local = resolve_local_snapshot_dir(model_id, hub_root, cache_models)
    if local:
        print(f"[ok] {model_id} -> {local}")
    else:
        print(f"[missing] {model_id}")
        missing.append(model_id)

print(f"[hf-check] Checked {len(targets)} model(s); missing={len(missing)}")
if missing:
    # Deduplicate while preserving order.
    dedup = []
    seen = set()
    for mid in missing:
        if mid in seen:
            continue
        seen.add(mid)
        dedup.append(mid)
    with missing_file.open("w", encoding="utf-8") as f:
        for model_id in dedup:
            f.write(model_id + "\n")
    print("[hf-check] Suggested downloads:")
    for model_id in dedup:
        print(f"  hf download {model_id} --cache-dir {cache_models}")
    raise SystemExit(2)
PY
}

set +e
run_check
status=$?
set -e
if [[ $status -eq 0 ]]; then
  echo "[hf-check] Done."
  exit 0
fi

if [[ "${DOWNLOAD_MISSING}" != "1" ]]; then
  echo "[hf-check] Missing models found. Re-run with DOWNLOAD_MISSING=1 to fetch them."
  exit "${status}"
fi

if command -v hf >/dev/null 2>&1; then
  HF_DL_CMD="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DL_CMD="huggingface-cli"
else
  echo "[hf-check][error] Neither 'hf' nor 'huggingface-cli' found in PATH." >&2
  exit 1
fi
echo "[hf-check] Using download command: ${HF_DL_CMD} download"

if [[ ! -s "${MISSING_FILE}" ]]; then
  echo "[hf-check][error] Missing list file not found: ${MISSING_FILE}" >&2
  exit 1
fi

echo "[hf-check] Downloading missing HF models..."
if ! [[ "${DOWNLOAD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[hf-check][error] DOWNLOAD_JOBS must be a positive integer, got '${DOWNLOAD_JOBS}'." >&2
  exit 1
fi

echo "[hf-check] Missing model count: $(wc -l < "${MISSING_FILE}")"
cat "${MISSING_FILE}" | xargs -I{} -P "${DOWNLOAD_JOBS}" bash -c '
  model_id="$1"
  cache_dir="$2"
  dl_cmd="$3"
  echo "[hf-check][download] ${model_id}"
  "${dl_cmd}" download "${model_id}" --cache-dir "${cache_dir}"
' _ {} "${HF_CACHE_MODELS}" "${HF_DL_CMD}"

echo "[hf-check] Re-checking after download..."
set +e
run_check
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[hf-check][error] Some models are still missing after download." >&2
  exit "$status"
fi

echo "[hf-check] Done."
