#!/bin/bash

source ~/miniforge3/bin/activate
conda activate round-trip-myenv

# rm -f data/processed/{ae,lzw,rle,huffman,ae-reflect,lzw-reflect,rle-reflect,huffman-reflect}/*_verified.jsonl
# rm -f data/processed/{ae,lzw,rle,huffman,ae-reflect,lzw-reflect,rle-reflect,huffman-reflect}/*_verified.csv

python webapp/merge_data.py

# CONFIGURATION — Edit these before running
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var is not set. Run: export HF_TOKEN=hf_...}"
SPACE_NAME="nickil-shay-antonio/round-trip-code-compression"  # Your Space path on HF
REPO_DIR="webapp/round-trip-code-compression"  # Local clone of your HF Space

# Optional: Only set remote once
cd "$REPO_DIR" || exit 1

# Set authenticated remote if not already set
current_url=$(git remote get-url origin)

if [[ "$current_url" != *"huggingface.co"* ]]; then
    echo "❌ Not a HF repo. Check remote URL."
    exit 1
fi

if [[ "$current_url" != *"$HF_TOKEN"* ]]; then
    echo "🔁 Updating remote URL with HF token..."
    git remote set-url origin "https://$HF_TOKEN@huggingface.co/spaces/$SPACE_NAME"
fi

# Stage changes (only data/ folder)
git add *

# Check for staged changes
if ! git diff --cached --quiet; then
    echo "📦 Committing and pushing changes..."
    git commit -m "Auto-update data folder"
    git push origin main -f
    echo "✅ Data pushed to HF Space."
else
    echo "🚫 No changes to commit."
fi
