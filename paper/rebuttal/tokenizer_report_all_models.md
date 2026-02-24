# Tokenizer Comparison Report

## 1. Aggregate Statistics

| Model | Total Tokens | Mean Tokens/Sample | Std | Mean Chars/Token | Token Range |
|---|---:|---:|---:|---:|---|
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | 854 | 42.7 | 28.27 | 1.6546 | [5, 84] |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | 854 | 42.7 | 28.27 | 1.6546 | [5, 84] |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | 578 | 28.9 | 19.07 | 2.338 | [4, 68] |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | 578 | 28.9 | 19.07 | 2.338 | [4, 68] |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | 854 | 42.7 | 28.27 | 1.6546 | [5, 84] |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | 854 | 42.7 | 28.27 | 1.6546 | [5, 84] |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | 854 | 42.7 | 28.27 | 1.6546 | [5, 84] |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | 854 | 42.7 | 28.27 | 1.6546 | [5, 84] |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 854 | 42.7 | 28.27 | 1.6546 | [5, 84] |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 854 | 42.7 | 28.27 | 1.6546 | [5, 84] |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 854 | 42.7 | 28.27 | 1.6546 | [5, 84] |

## 2. Per-Algorithm Mean Tokens

| Algorithm | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B |
|---| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ae | 64.2 (±28.99) | 64.2 (±28.99) | 31.6 (±11.7) | 31.6 (±11.7) | 64.2 (±28.99) | 64.2 (±28.99) | 64.2 (±28.99) | 64.2 (±28.99) | 64.2 (±28.99) | 64.2 (±28.99) | 64.2 (±28.99) |
| huffman | 54.75 (±26.44) | 54.75 (±26.44) | 36.75 (±16.21) | 36.75 (±16.21) | 54.75 (±26.44) | 54.75 (±26.44) | 54.75 (±26.44) | 54.75 (±26.44) | 54.75 (±26.44) | 54.75 (±26.44) | 54.75 (±26.44) |
| lzw | 34.75 (±26.49) | 34.75 (±26.49) | 32.25 (±25.88) | 32.25 (±25.88) | 34.75 (±26.49) | 34.75 (±26.49) | 34.75 (±26.49) | 34.75 (±26.49) | 34.75 (±26.49) | 34.75 (±26.49) | 34.75 (±26.49) |
| rle | 37.33 (±26.58) | 37.33 (±26.58) | 34.67 (±28.88) | 34.67 (±28.88) | 37.33 (±26.58) | 37.33 (±26.58) | 37.33 (±26.58) | 37.33 (±26.58) | 37.33 (±26.58) | 37.33 (±26.58) | 37.33 (±26.58) |
| preset:mixed | 15.75 (±4.92) | 15.75 (±4.92) | 10.0 (±3.74) | 10.0 (±3.74) | 15.75 (±4.92) | 15.75 (±4.92) | 15.75 (±4.92) | 15.75 (±4.92) | 15.75 (±4.92) | 15.75 (±4.92) | 15.75 (±4.92) |

## 3. Per-Sample Token Counts

| Group | Sample | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B |
|---|---| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ae | `ae_encode("AAABBBCCCDDDEEE112233AABBCC")` | 20 | 20 | 16 | 16 | 20 | 20 | 20 | 20 | 20 | 20 | 20 |
| ae | `freq={'1': 2, '2': 2, '3': 2, 'A': 5, 'B': 5, 'C': 5, 'D': 3, 'E': 3}` | 49 | 49 | 49 | 49 | 49 | 49 | 49 | 49 | 49 | 49 | 49 |
| ae | `low=0.27417980805869453612794382132112219696300061774716374678631300741228707534477073` | 84 | 84 | 31 | 31 | 84 | 84 | 84 | 84 | 84 | 84 | 84 |
| ae | `high=0.27417980805869453612794703228880850174137945152555160950960400485130532635816488` | 84 | 84 | 31 | 31 | 84 | 84 | 84 | 84 | 84 | 84 | 84 |
| ae | `code=0.27417980805869453612794382132112219696300061774716374678631300741228707534477073` | 84 | 84 | 31 | 31 | 84 | 84 | 84 | 84 | 84 | 84 | 84 |
| huffman | `huffman_encode("AAABBBCCCDDDEEE112233AABBCC")` | 21 | 21 | 17 | 17 | 21 | 21 | 21 | 21 | 21 | 21 | 21 |
| huffman | `freq={'1': 2, '2': 2, '3': 2, 'A': 5, 'B': 5, 'C': 5, 'D': 3, 'E': 3}` | 49 | 49 | 49 | 49 | 49 | 49 | 49 | 49 | 49 | 49 | 49 |
| huffman | `codes={'C': '00', '3': '010', 'D': '011', 'E': '100', '1': '1010', '2': '1011', 'A': '110', 'B': '111'}` | 66 | 66 | 51 | 51 | 66 | 66 | 66 | 66 | 66 | 66 | 66 |
| huffman | `bitstream=11011011011111111100000001101101110010010010101010101110110100101101101111110000` | 83 | 83 | 30 | 30 | 83 | 83 | 83 | 83 | 83 | 83 | 83 |
| lzw | `lzw_encode("AAABBBCCCDDDEEE112233AABBCC")` | 21 | 21 | 17 | 17 | 21 | 21 | 21 | 21 | 21 | 21 | 21 |
| lzw | `init_dict={'1': 0, '2': 1, '3': 2, 'A': 3, 'B': 4, 'C': 5, 'D': 6, 'E': 7}` | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| lzw | `codes=[3, 8, 4, 10, 5, 12, 6, 14, 7, 16, 0, 0, 1, 1, 2, 2, 9, 4, 12]` | 63 | 63 | 58 | 58 | 63 | 63 | 63 | 63 | 63 | 63 | 63 |
| lzw | `next_code=26` | 5 | 5 | 4 | 4 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| rle | `rle_encode("AAABBBCCCDDDEEE112233AABBCC")` | 21 | 21 | 17 | 17 | 21 | 21 | 21 | 21 | 21 | 21 | 21 |
| rle | `runs=[('A', 3), ('B', 3), ('C', 3), ('D', 3), ('E', 3), ('1', 2), ('2', 2), ('3', 2), ('A', 2), ('B', 2), ('C', 2)]` | 68 | 68 | 68 | 68 | 68 | 68 | 68 | 68 | 68 | 68 | 68 |
| rle | `compact=A3B3C3D3E3122232A2B2C2` | 23 | 23 | 19 | 19 | 23 | 23 | 23 | 23 | 23 | 23 | 23 |
| preset:mixed | `0101010101010101` | 16 | 16 | 6 | 6 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| preset:mixed | `0.12345678901234567890` | 22 | 22 | 9 | 9 | 22 | 22 | 22 | 22 | 22 | 22 | 22 |
| preset:mixed | `def compress(bits): return bits.count('1')` | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| preset:mixed | `while lo < hi: mid = (lo + hi) // 2` | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 |

## 4. Pairwise Comparison

| Model A | Model B | A Wins | B Wins | Ties | Mean Diff (A−B) | More Efficient |
|---|---|---:|---:|---:|---:|---|
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | 0 | 14 | 6 | +13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | 0 | 14 | 6 | +13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | 0 | 14 | 6 | +13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | 0 | 14 | 6 | +13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | 14 | 0 | 6 | -13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | 14 | 0 | 6 | -13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | 14 | 0 | 6 | -13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | 14 | 0 | 6 | -13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 14 | 0 | 6 | -13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 14 | 0 | 6 | -13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 14 | 0 | 6 | -13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | 14 | 0 | 6 | -13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | 14 | 0 | 6 | -13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | 14 | 0 | 6 | -13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | 14 | 0 | 6 | -13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 14 | 0 | 6 | -13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 14 | 0 | 6 | -13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 14 | 0 | 6 | -13.80 | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 0 | 0 | 20 | +0.00 | Tied |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 0 | 0 | 20 | +0.00 | Tied |

## 5. Worst-Case Fragmentation

| Model | Group | Sample | Tokens | Chars/Token |
|---|---|---|---:|---:|
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | ae | `low=0.274179808058694536127943821321122196963000617747163746...` | 84 | 1.0238 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | ae | `low=0.274179808058694536127943821321122196963000617747163746...` | 84 | 1.0238 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | rle | `runs=[('A', 3), ('B', 3), ('C', 3), ('D', 3), ('E', 3), ('1'...` | 68 | 1.6912 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | rle | `runs=[('A', 3), ('B', 3), ('C', 3), ('D', 3), ('E', 3), ('1'...` | 68 | 1.6912 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | ae | `low=0.274179808058694536127943821321122196963000617747163746...` | 84 | 1.0238 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | ae | `low=0.274179808058694536127943821321122196963000617747163746...` | 84 | 1.0238 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | ae | `low=0.274179808058694536127943821321122196963000617747163746...` | 84 | 1.0238 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | ae | `low=0.274179808058694536127943821321122196963000617747163746...` | 84 | 1.0238 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | ae | `low=0.274179808058694536127943821321122196963000617747163746...` | 84 | 1.0238 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | ae | `low=0.274179808058694536127943821321122196963000617747163746...` | 84 | 1.0238 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | ae | `low=0.274179808058694536127943821321122196963000617747163746...` | 84 | 1.0238 |

## 6. Numeric & Binary String Efficiency

| Model | Samples | Mean Tokens | Mean Chars/Token |
|---|---:|---:|---:|
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | 2 | 19.0 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | 2 | 19.0 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | 2 | 7.5 | 2.5556 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | 2 | 7.5 | 2.5556 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | 2 | 19.0 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | 2 | 19.0 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | 2 | 19.0 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | 2 | 19.0 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 2 | 19.0 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 2 | 19.0 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 2 | 19.0 | 1.0 |

## 7. Token Vocabulary Overlap

| Model A | Model B | Shared Tokens | Union | Jaccard |
|---|---|---:|---:|---:|
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | 82 | 155 | 0.529 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | 84 | 154 | 0.5455 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-14B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | 82 | 155 | 0.529 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | 84 | 154 | 0.5455 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-32B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | 152 | 155 | 0.9806 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | 82 | 155 | 0.529 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | 82 | 155 | 0.529 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | 82 | 155 | 0.529 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | 82 | 155 | 0.529 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 82 | 155 | 0.529 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 82 | 155 | 0.529 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/gpt-oss-20b | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 82 | 155 | 0.529 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | 84 | 154 | 0.5455 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | 84 | 154 | 0.5455 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | 84 | 154 | 0.5455 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | 84 | 154 | 0.5455 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 84 | 154 | 0.5455 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 84 | 154 | 0.5455 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Llama-3.1-8B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 84 | 154 | 0.5455 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-7B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen3-Coder-30B-A3B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-1.5B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/DeepSeek-R1-Distill-Qwen-7B | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-32B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 84 | 84 | 1.0 |
| /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/Qwen2.5-Coder-32B-Instruct | /lus/lfs1aip2/projects/u6cg/nmaveli/hf-cache/models/QwQ-32B | 84 | 84 | 1.0 |
