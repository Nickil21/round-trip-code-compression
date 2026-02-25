# Round-Trip Code Compression

Code repository for a research pipeline on **invertible code reasoning** in LLMs: given either raw inputs or compressed outputs, models are prompted to execute (or invert) classical compression algorithms and recover the missing side of the mapping.

## Abstract

We study whether large language models can perform bidirectional reasoning over lossless compression programs instead of pattern-matching outputs. The benchmark contains four algorithm families (`lzw`, `ae`, `rle`, `huffman`) and four task settings (forward prediction, inverse prediction, each with/without inversion hints).  
The pipeline supports:
- synthetic data generation,
- prompt construction with optional algorithm-name blinding,
- execution-trace extraction,
- natural-language trace translation,
- SFT data construction,
- local (vLLM) and API-based inference,
- exact-match and pass@k evaluation.

## Repository Layout

```
.
├── src/
│   ├── data/
│   │   ├── build_codeio_msg.py
│   │   ├── generate_execution_trace.py
│   │   ├── filter_execution_trace.py
│   │   ├── execution_trace_translation.py
│   │   └── data_construction_sft.py
│   ├── inference/
│   │   ├── batched_api_inference.py
│   │   └── self_reflection.py
│   ├── eval/
│   │   ├── check_io_pred_acc_mp.py
│   │   └── calc_pass_at_k.py
│   ├── ablation/
│   │   ├── build_tokenization_ablation.py
│   │   ├── check_tokenization_ablation.py
│   │   └── compare_ablation_results.py
│   └── core/
├── scripts/
│   ├── generate_data.py
│   ├── zero-shot-inference.sh
│   ├── model-finetune.sh
│   ├── generate-execution-trace.sh
│   ├── self-reflection.sh
│   └── tokenization-ablation.sh
├── paper/
│   ├── figures/
│   └── rebuttal/
├── LLaMA-Factory/   # submodule used for finetuning workflows
├── requirements.txt
└── LICENSE
```

## Task Definition

For each algorithm and sample, we build prompts covering:
- `output_execution_prediction`
- `output_execution_prediction_with_inversion`
- `input_execution_prediction`
- `input_execution_prediction_with_inversion`

Primary metric is exact correctness on parsed outputs; aggregated reporting includes `pass@k`.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional `.env` (used for API-based inference/reflection):

```ini
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
HF_API_KEY=...
```

## Reproducibility Pipeline (Single Algorithm)

Below is an end-to-end local run for `lzw`.

### 1) Generate synthetic benchmark data

```bash
python scripts/generate_data.py \
  --algorithms lzw \
  --source mixed \
  --count 50
```

Output is written under `data/processed/lzw/`.

### 2) Build prompt messages

```bash
python src/data/build_codeio_msg.py \
  --input_file data/processed/lzw/data.jsonl \
  --output_file data/processed/lzw/codeio_1k_msg.jsonl \
  --algorithm lzw \
  --prompt_type zero_shot \
  --blind
```

### 3) Generate raw execution traces

```bash
python src/data/generate_execution_trace.py \
  --data_dir data/processed \
  --algorithm lzw \
  --input_file codeio_1k_msg.jsonl \
  --output_file codeio_1k_msg_executed.jsonl
```

### 4) Filter traces

```bash
python src/data/filter_execution_trace.py \
  --data_dir data/processed/ \
  --algorithm lzw \
  --input_file codeio_1k_msg_executed.jsonl \
  --output_file codeio_1k_msg_executed_filtered.pkl
```

### 5) Translate traces into natural-language reasoning

```bash
python src/data/execution_trace_translation.py \
  --data_dir data/processed/ \
  --algorithm lzw \
  --input_file codeio_1k_msg_executed_filtered.pkl \
  --output_file codeio_1k_msg_executed_filtered_translated.pkl \
  --translator_model Qwen/Qwen3-32B \
  --num_gpus 1
```

### 6) Build SFT data

```bash
python src/data/data_construction_sft.py \
  --data_dir data/processed/ \
  --algorithm lzw \
  --input_file codeio_1k_msg_executed_filtered_translated.pkl \
  --output_file LLaMA-Factory/data/lzw_training_data_sft.jsonl \
  --trained_model Qwen/QwQ-32B
```

## Inference

### Local vLLM

```bash
python src/inference/batched_api_inference.py \
  --model Qwen/Qwen3-32B \
  --input data/processed/lzw/codeio_1k_msg.jsonl \
  --output data/processed/lzw/codeio_1k_gens_model_qwen3_32b_temp_0.2_n5.jsonl \
  --temperature 0.2 \
  --num_completions 5 \
  --tp_size 1 \
  --max_tokens 16384
```

### OpenAI API

```bash
python src/inference/batched_api_inference.py \
  --model gpt-4.1-mini \
  --input data/processed/lzw/codeio_1k_msg.jsonl \
  --output data/processed/lzw/codeio_1k_gens_model_gpt_4_1_mini_temp_0.2_n5.jsonl \
  --temperature 0.2 \
  --num_completions 5 \
  --use_openai \
  --workers 32
```

## Evaluation

### Verify generations

```bash
python src/eval/check_io_pred_acc_mp.py \
  --parsed_file_name data/processed/lzw/data.jsonl \
  --pred_file_name data/processed/lzw/codeio_1k_gens_model_qwen3_32b_temp_0.2_n5.jsonl \
  --res_file_name data/processed/lzw/codeio_1k_gens_model_qwen3_32b_temp_0.2_n5_verified.jsonl \
  --algo lzw
```

### Compute pass@k

```bash
python src/eval/calc_pass_at_k.py \
  --verified_file data/processed/lzw/*_verified.jsonl \
  --k 5
```

## Batch Experiment Scripts

The `scripts/` directory contains SLURM-ready orchestration:
- `zero-shot-inference.sh`: grid over algorithm/model/temperature.
- `model-finetune.sh`: inference with finetuned adapters.
- `generate-execution-trace.sh`: trace generation + translation.
- `self-reflection.sh`: critique/revise loop over verified outputs.
- `tokenization-ablation.sh`: alternative output-format ablations.

## Notes for Artifact Review

- Key randomness is seeded in data generation and major scripts.
- Intermediate artifacts are stored per algorithm under `data/processed/<algo>/`.
- Existing scripts include retry logic for OOM and backend fallbacks in HPC settings.
- `LLaMA-Factory/` is included for finetuning recipes and adapter checkpoints.

## License

MIT. See [LICENSE](LICENSE).
