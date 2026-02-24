#!/usr/bin/env bash
MODEL_DIR=/lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models

# python paper/rebuttal/compare_tokenizers.py \
#   --models \
#     ${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-14B \
#     ${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-32B \
#     ${MODEL_DIR}/gpt-oss-20b \
#     ${MODEL_DIR}/Llama-3.1-8B-Instruct \
#     ${MODEL_DIR}/Qwen2.5-7B-Instruct \
#     ${MODEL_DIR}/Qwen3-Coder-30B-A3B-Instruct \
#     ${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-1.5B \
#     ${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-7B \
#     ${MODEL_DIR}/Qwen2.5-32B-Instruct \
#     ${MODEL_DIR}/Qwen2.5-Coder-32B-Instruct \
#     ${MODEL_DIR}/QwQ-32B \
#   --algorithms ae huffman lzw rle \
#   --preset mixed \
#   --local-files-only \
#   --markdown-out paper/rebuttal/tokenizer_report_all_models.md \
#   --json-out paper/rebuttal/tokenizer_report_all_models.json

# echo ""
# echo "=== Running full-prompt tokenizer comparison ==="
# echo ""

python paper/rebuttal/tokenize_full_prompts.py \
  --models \
    ${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-14B \
    ${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-32B \
    ${MODEL_DIR}/gpt-oss-20b \
    ${MODEL_DIR}/Llama-3.1-8B-Instruct \
    ${MODEL_DIR}/Qwen2.5-7B-Instruct \
    ${MODEL_DIR}/Qwen3-Coder-30B-A3B-Instruct \
    ${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-1.5B \
    ${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-7B \
    ${MODEL_DIR}/Qwen2.5-32B-Instruct \
    ${MODEL_DIR}/Qwen2.5-Coder-32B-Instruct \
    ${MODEL_DIR}/QwQ-32B \
  --algorithms ae huffman lzw rle \
  --local-files-only \
  --markdown-out paper/rebuttal/full_prompt_tokenizer_report.md \
  --json-out paper/rebuttal/full_prompt_tokenizer_report.json

echo ""
echo "=== Running bitstring & float tokenization analysis ==="
echo ""

python paper/rebuttal/analyze_bitstring_float_tokenization.py \
  --models \
    ${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-14B \
    ${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-32B \
    ${MODEL_DIR}/gpt-oss-20b \
    ${MODEL_DIR}/Llama-3.1-8B-Instruct \
    ${MODEL_DIR}/Qwen2.5-7B-Instruct \
    ${MODEL_DIR}/Qwen3-Coder-30B-A3B-Instruct \
    ${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-1.5B \
    ${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-7B \
    ${MODEL_DIR}/Qwen2.5-32B-Instruct \
    ${MODEL_DIR}/Qwen2.5-Coder-32B-Instruct \
    ${MODEL_DIR}/QwQ-32B \
  --local-files-only \
  --markdown-out paper/rebuttal/bitstring_float_report.md \
  --json-out paper/rebuttal/bitstring_float_report.json
