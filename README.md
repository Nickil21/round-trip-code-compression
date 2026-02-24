# Round-Trip Code Compression

A research framework for training Large Language Models to perform **bidirectional compression algorithm tasks** — encoding inputs to compressed outputs and inverting the process to recover original inputs — using natural language execution traces.

## Overview

This project investigates whether LLMs can learn to invert lossless compression algorithms by reasoning through step-by-step execution traces rather than by memorizing algorithm implementations. The pipeline covers four canonical compression algorithms:

| Algorithm | Output Type | Example Output |
|-----------|-------------|----------------|
| **LZW** (Lempel-Ziv-Welch) | List of integers | `[84, 79, 66, 256, ...]` |
| **Arithmetic Encoding (AE)** | Float | `0.6319081203782178` |
| **RLE** (Run-Length Encoding) | List of `[char, count]` pairs | `[["a", 3], ["b", 2], ["a", 1]]` |
| **Huffman Coding** | List of integers | `[102, 114, 111, ...]` |

The core research question: given a compressed output and (optionally) the reference algorithm code or a natural language trace of its execution, can an LLM reconstruct the original input?

## Pipeline

```
Raw Data
   │
   ▼
Prompt Construction (build_codeio_msg.py)
   │  Zero-shot / few-shot; algorithm name sanitized to prevent memorization
   ▼
Execution Trace Generation (generate_execution_trace.py)
   │  snoop-based tracing → natural language trace translation
   ▼
SFT Training Data Construction (data_construction_sft.py)
   │  Filtered by sequence length (<32K tokens)
   ▼
Model Fine-tuning (LLaMA-Factory / LoRA on QwQ-32B)
   │
   ▼
Batched Inference (batched_api_inference.py)
   │  vLLM (local GPU) or OpenAI API; LoRA adapter loading
   ▼
Evaluation (check_io_pred_acc_mp.py)
   │  Exact match + numerical tolerance; Pass@K
   ▼
Results / Analysis
```

## Repository Structure

```
.
├── src/
│   ├── utils.py                        # File I/O, HF model resolution, JSON extraction
│   ├── helper.py                       # Model size/category metadata (40+ models)
│   ├── codeio_utils.py                 # Frequency dict loading, I/O validation
│   ├── prompts.py                      # Prompt templates (zero-shot, few-shot, trace-augmented)
│   ├── build_codeio_msg.py             # Prompt construction for I/O prediction
│   ├── generate_execution_trace.py     # snoop-based execution tracing (multiprocessing)
│   ├── execution_trace_translation.py  # Raw trace → natural language
│   ├── data_construction_sft.py        # SFT training data builder
│   ├── batched_api_inference.py        # vLLM + OpenAI batched inference
│   ├── self_reflection.py              # Critique → revise reflection loops
│   ├── check_io_pred_acc_mp.py         # Multiprocessing accuracy evaluation
│   ├── calc_pass_at_k.py               # Pass@K metric computation
│   ├── calc_openai_usage.py            # API cost tracking
│   ├── build_tokenization_ablation.py  # Tokenization ablation data builder
│   ├── check_tokenization_ablation.py  # Tokenization ablation evaluation
│   └── compare_ablation_results.py     # Ablation result comparison
├── LLaMA-Factory/                      # Git submodule for supervised fine-tuning
│   ├── bash-scripts/finetune.sh        # LoRA fine-tuning script (QwQ-32B)
│   └── data/                           # SFT training datasets (JSONL/JSON per algorithm)
├── slurm-scripts/                      # HPC cluster job submission scripts
│   ├── model-inference.sh              # Grid search over models × algorithms × temps
│   ├── model-finetune.sh               # Fine-tuning job submission
│   ├── self-reflection.sh              # Reflection loop jobs
│   ├── generate-execution-trace.sh     # Trace generation jobs
│   └── just-inference.sh               # Single-configuration inference
├── flowchart/                          # Paper figures (Mermaid diagrams)
├── requirements.txt
└── LICENSE                             # MIT
```

## Installation

```bash
# 1. Create and activate environment
conda create -p env python=3.11
conda activate env/

# 2. Install core dependencies
pip install -r requirements.txt

# 3. (Optional) Install LLaMA-Factory for fine-tuning
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
cd ..
```

Set up a `.env` file with your API keys:

```ini
HF_API_KEY=<huggingface_token>
OPENAI_API_KEY=<openai_api_key>
OPENROUTER_API_KEY=<openrouter_api_key>
PYTHONPATH=src
```

## Usage

### 1. Build I/O Prediction Prompts

```bash
python src/build_codeio_msg.py \
  --input_file data/processed/lzw/data.jsonl \
  --output_file data/processed/lzw/codeio_1k_msg.jsonl \
  --algorithm lzw \
  --prompt_type zero_shot \
  --blind   # strips algorithm name from prompt
```

### 2. Generate Execution Traces

```bash
python src/generate_execution_trace.py \
  --data_dir data/processed/ \
  --algorithm lzw \
  --input_file codeio_1k_msg.jsonl
```

### 3. Build SFT Training Data

```bash
python src/data_construction_sft.py \
  --input_file data/processed/lzw/codeio_1k_msg_executed_filtered_translated.pkl \
  --output_file LLaMA-Factory/data/lzw_training_data_sft.jsonl \
  --algorithm lzw
```

### 4. Fine-tune a Model

Uses LLaMA-Factory with LoRA on QwQ-32B across 8 GPUs via DeepSpeed ZeRO-3:

```bash
bash LLaMA-Factory/bash-scripts/finetune.sh
```

Key fine-tuning configuration:
- **Base model**: `Qwen/QwQ-32B`
- **Method**: LoRA (`rank=8`, applied to all layers)
- **Batch size**: 1 per GPU × 8 gradient accumulation steps
- **Learning rate**: `1e-4` with cosine scheduler
- **Epochs**: 3, max sequence length: 2048 tokens
- **Precision**: BF16 with Flash Attention (SDPA backend)

### 5. Run Inference

```bash
# Local GPU via vLLM
python src/batched_api_inference.py \
  --model Qwen/QwQ-32B \
  --input data/processed/lzw/codeio_1k_msg.jsonl \
  --output results/lzw_qwq32b.jsonl \
  --lora_path LLaMA-Factory/finetune/lzw/QwQ-32B/lora/sft/checkpoint-21

# OpenAI API
python src/batched_api_inference.py \
  --model gpt-4o \
  --input data/processed/lzw/codeio_1k_msg.jsonl \
  --output results/lzw_gpt4o.jsonl \
  --api openai
```

### 6. Evaluate Results

```bash
python src/check_io_pred_acc_mp.py \
  --input_file results/lzw_qwq32b.jsonl \
  --output_file results/lzw_qwq32b_eval.csv \
  --algorithm lzw

python src/calc_pass_at_k.py \
  --input_file results/lzw_qwq32b_eval.csv \
  --k 1 5 10
```

### 7. Self-Reflection (Optional)

Iterative critique-and-revise loop to improve model outputs:

```bash
python src/self_reflection.py \
  --input_file results/lzw_qwq32b.jsonl \
  --output_file results/lzw_qwq32b_reflected.jsonl \
  --algorithm lzw \
  --rounds 3
```

## Running on a Cluster (SLURM)

The `slurm-scripts/` directory contains ready-to-submit job scripts for HPC environments:

```bash
# Full grid search: 40+ models × 4 algorithms × 2 temperatures
sbatch slurm-scripts/model-inference.sh

# Fine-tuning job
sbatch slurm-scripts/model-finetune.sh

# Reflection loop
sbatch slurm-scripts/self-reflection.sh
```

Scripts handle dynamic tensor parallelism selection based on model size and include OOM fallback mechanisms (progressive `max_length` reduction).

## Supported Models

The framework supports 40+ models across categories including:

- **Reasoning**: `Qwen/QwQ-32B`, `deepseek-ai/DeepSeek-R1-*`
- **Code Generation**: `Qwen/Qwen2.5-Coder-*`, `codellama/*`, `deepseek-ai/deepseek-coder-*`
- **General Instruction**: `Qwen/Qwen2.5-*`, `meta-llama/Llama-3*`, `mistralai/Mistral-*`
- **OpenAI API**: `gpt-4o`, `gpt-4o-mini`, `o1-mini`, `o3-mini`

See `src/helper.py` for the full model registry.

## Evaluation Metrics

- **Exact Match (EM)**: Strict equality after type normalization
- **Numerical Tolerance**: Floating-point comparison with configurable epsilon (used for AE outputs)
- **Pass@K**: Probability of at least one correct answer in K samples
- **Per-algorithm accuracy**: Results broken down by compression algorithm

## License

MIT License. See [LICENSE](LICENSE) for details.
