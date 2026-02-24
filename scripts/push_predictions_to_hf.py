"""
Upload large prediction files to a private Hugging Face dataset.

Usage:
    export HF_TOKEN=hf_...
    python scripts/push_predictions_to_hf.py

The dataset will be created as private if it doesn't exist yet.
Files are uploaded with their relative path preserved under data/processed/.
"""

import os
import sys
import glob
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ID    = "nickil-shay-antonio/round-trip-code-compression-predictions"
REPO_TYPE  = "dataset"
PRIVATE    = True
BASE_DIR   = Path(__file__).parent.parent  # project root

# Patterns to upload (the large files excluded from git)
UPLOAD_PATTERNS = [
    "data/processed/**/*_gens_*.jsonl",
    "data/processed/**/*_verified.jsonl",
]
# ──────────────────────────────────────────────────────────────────────────────

token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("HF_TOKEN env var is not set. Run: export HF_TOKEN=hf_...")

api = HfApi(token=token)

# Create dataset repo if it doesn't exist yet
print(f"Ensuring repo exists: {REPO_ID} (private={PRIVATE})")
create_repo(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    private=PRIVATE,
    exist_ok=True,
    token=token,
)

# Collect files to upload
files = []
for pattern in UPLOAD_PATTERNS:
    files.extend(BASE_DIR.glob(pattern))
files = sorted(set(files))

if not files:
    sys.exit(f"No files matched patterns: {UPLOAD_PATTERNS}")

total_bytes = sum(f.stat().st_size for f in files)
print(f"\nFiles to upload: {len(files)}  ({total_bytes / 1e9:.2f} GB total)")
for f in files:
    print(f"  {f.relative_to(BASE_DIR)}  ({f.stat().st_size / 1e6:.0f} MB)")

print()

# Upload each file, preserving its data/processed/... path in the repo
for f in files:
    path_in_repo = str(f.relative_to(BASE_DIR))
    print(f"Uploading {path_in_repo} ...", end=" ", flush=True)
    api.upload_file(
        path_or_fileobj=str(f),
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=token,
        commit_message=f"Add {f.name}",
    )
    print("done")

print(f"\nAll files uploaded to https://huggingface.co/datasets/{REPO_ID}")
