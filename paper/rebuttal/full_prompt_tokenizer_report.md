# Full-Prompt Tokenizer Comparison

Token counts for the **complete prompt** the LLM receives (system message + preamble + IO requirements + reference code + input/output).

## 1. Token Counts by Algorithm and Task

| Algorithm | Task | DeepSeek-R1-Distill-Qwen-14B | DeepSeek-R1-Distill-Qwen-32B | gpt-oss-20b | Llama-3.1-8B-Instruct | Qwen2.5-7B-Instruct | Qwen3-Coder-30B-A3B-Instruct | DeepSeek-R1-Distill-Qwen-1.5B | DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-32B-Instruct | Qwen2.5-Coder-32B-Instruct | QwQ-32B |
|---|---| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ae | encode | 768 | 768 | 762 | 764 | 768 | 768 | 768 | 768 | 768 | 768 | 768 |
| ae | decode | 829 | 829 | 817 | 819 | 829 | 829 | 829 | 829 | 829 | 829 | 829 |
| ae | encode_inv | 860 | 860 | 854 | 856 | 860 | 860 | 860 | 860 | 860 | 860 | 860 |
| ae | decode_inv | 819 | 819 | 807 | 809 | 819 | 819 | 819 | 819 | 819 | 819 | 819 |
| huffman | encode | 866 | 866 | 868 | 862 | 866 | 866 | 866 | 866 | 866 | 866 | 866 |
| huffman | decode | 1024 | 1024 | 980 | 974 | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 |
| huffman | encode_inv | 954 | 954 | 939 | 934 | 954 | 954 | 954 | 954 | 954 | 954 | 954 |
| huffman | decode_inv | 1018 | 1018 | 991 | 984 | 1018 | 1018 | 1018 | 1018 | 1018 | 1018 | 1018 |
| lzw | encode | 690 | 690 | 683 | 684 | 690 | 690 | 690 | 690 | 690 | 690 | 690 |
| lzw | decode | 795 | 795 | 766 | 767 | 795 | 795 | 795 | 795 | 795 | 795 | 795 |
| lzw | encode_inv | 762 | 762 | 755 | 756 | 762 | 762 | 762 | 762 | 762 | 762 | 762 |
| lzw | decode_inv | 805 | 805 | 776 | 777 | 805 | 805 | 805 | 805 | 805 | 805 | 805 |
| rle | encode | 665 | 665 | 663 | 661 | 665 | 665 | 665 | 665 | 665 | 665 | 665 |
| rle | decode | 669 | 669 | 668 | 669 | 669 | 669 | 669 | 669 | 669 | 669 | 669 |
| rle | encode_inv | 647 | 647 | 641 | 643 | 647 | 647 | 647 | 647 | 647 | 647 | 647 |
| rle | decode_inv | 769 | 769 | 772 | 769 | 769 | 769 | 769 | 769 | 769 | 769 | 769 |

## 2. Mean Tokens per Algorithm (averaged across 4 task types)

| Algorithm | DeepSeek-R1-Distill-Qwen-14B | DeepSeek-R1-Distill-Qwen-32B | gpt-oss-20b | Llama-3.1-8B-Instruct | Qwen2.5-7B-Instruct | Qwen3-Coder-30B-A3B-Instruct | DeepSeek-R1-Distill-Qwen-1.5B | DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-32B-Instruct | Qwen2.5-Coder-32B-Instruct | QwQ-32B |
|---| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ae | 819.0 | 819.0 | 810.0 | 812.0 | 819.0 | 819.0 | 819.0 | 819.0 | 819.0 | 819.0 | 819.0 |
| huffman | 965.5 | 965.5 | 944.5 | 938.5 | 965.5 | 965.5 | 965.5 | 965.5 | 965.5 | 965.5 | 965.5 |
| lzw | 763.0 | 763.0 | 745.0 | 746.0 | 763.0 | 763.0 | 763.0 | 763.0 | 763.0 | 763.0 | 763.0 |
| rle | 687.5 | 687.5 | 686.0 | 685.5 | 687.5 | 687.5 | 687.5 | 687.5 | 687.5 | 687.5 | 687.5 |

## 3. Pairwise Token Difference (Model A − Model B, averaged across tasks)

| Algorithm | DeepSeek-R1-Distill-Qwen-14B vs DeepSeek-R1-Distill-Qwen-32B | DeepSeek-R1-Distill-Qwen-14B vs gpt-oss-20b | DeepSeek-R1-Distill-Qwen-14B vs Llama-3.1-8B-Instruct | DeepSeek-R1-Distill-Qwen-14B vs Qwen2.5-7B-Instruct | DeepSeek-R1-Distill-Qwen-14B vs Qwen3-Coder-30B-A3B-Instruct | DeepSeek-R1-Distill-Qwen-14B vs DeepSeek-R1-Distill-Qwen-1.5B | DeepSeek-R1-Distill-Qwen-14B vs DeepSeek-R1-Distill-Qwen-7B | DeepSeek-R1-Distill-Qwen-14B vs Qwen2.5-32B-Instruct | DeepSeek-R1-Distill-Qwen-14B vs Qwen2.5-Coder-32B-Instruct | DeepSeek-R1-Distill-Qwen-14B vs QwQ-32B | DeepSeek-R1-Distill-Qwen-32B vs gpt-oss-20b | DeepSeek-R1-Distill-Qwen-32B vs Llama-3.1-8B-Instruct | DeepSeek-R1-Distill-Qwen-32B vs Qwen2.5-7B-Instruct | DeepSeek-R1-Distill-Qwen-32B vs Qwen3-Coder-30B-A3B-Instruct | DeepSeek-R1-Distill-Qwen-32B vs DeepSeek-R1-Distill-Qwen-1.5B | DeepSeek-R1-Distill-Qwen-32B vs DeepSeek-R1-Distill-Qwen-7B | DeepSeek-R1-Distill-Qwen-32B vs Qwen2.5-32B-Instruct | DeepSeek-R1-Distill-Qwen-32B vs Qwen2.5-Coder-32B-Instruct | DeepSeek-R1-Distill-Qwen-32B vs QwQ-32B | gpt-oss-20b vs Llama-3.1-8B-Instruct | gpt-oss-20b vs Qwen2.5-7B-Instruct | gpt-oss-20b vs Qwen3-Coder-30B-A3B-Instruct | gpt-oss-20b vs DeepSeek-R1-Distill-Qwen-1.5B | gpt-oss-20b vs DeepSeek-R1-Distill-Qwen-7B | gpt-oss-20b vs Qwen2.5-32B-Instruct | gpt-oss-20b vs Qwen2.5-Coder-32B-Instruct | gpt-oss-20b vs QwQ-32B | Llama-3.1-8B-Instruct vs Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct vs Qwen3-Coder-30B-A3B-Instruct | Llama-3.1-8B-Instruct vs DeepSeek-R1-Distill-Qwen-1.5B | Llama-3.1-8B-Instruct vs DeepSeek-R1-Distill-Qwen-7B | Llama-3.1-8B-Instruct vs Qwen2.5-32B-Instruct | Llama-3.1-8B-Instruct vs Qwen2.5-Coder-32B-Instruct | Llama-3.1-8B-Instruct vs QwQ-32B | Qwen2.5-7B-Instruct vs Qwen3-Coder-30B-A3B-Instruct | Qwen2.5-7B-Instruct vs DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5-7B-Instruct vs DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-7B-Instruct vs Qwen2.5-32B-Instruct | Qwen2.5-7B-Instruct vs Qwen2.5-Coder-32B-Instruct | Qwen2.5-7B-Instruct vs QwQ-32B | Qwen3-Coder-30B-A3B-Instruct vs DeepSeek-R1-Distill-Qwen-1.5B | Qwen3-Coder-30B-A3B-Instruct vs DeepSeek-R1-Distill-Qwen-7B | Qwen3-Coder-30B-A3B-Instruct vs Qwen2.5-32B-Instruct | Qwen3-Coder-30B-A3B-Instruct vs Qwen2.5-Coder-32B-Instruct | Qwen3-Coder-30B-A3B-Instruct vs QwQ-32B | DeepSeek-R1-Distill-Qwen-1.5B vs DeepSeek-R1-Distill-Qwen-7B | DeepSeek-R1-Distill-Qwen-1.5B vs Qwen2.5-32B-Instruct | DeepSeek-R1-Distill-Qwen-1.5B vs Qwen2.5-Coder-32B-Instruct | DeepSeek-R1-Distill-Qwen-1.5B vs QwQ-32B | DeepSeek-R1-Distill-Qwen-7B vs Qwen2.5-32B-Instruct | DeepSeek-R1-Distill-Qwen-7B vs Qwen2.5-Coder-32B-Instruct | DeepSeek-R1-Distill-Qwen-7B vs QwQ-32B | Qwen2.5-32B-Instruct vs Qwen2.5-Coder-32B-Instruct | Qwen2.5-32B-Instruct vs QwQ-32B | Qwen2.5-Coder-32B-Instruct vs QwQ-32B |
|---| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ae | +0.0 | +9.0 | +7.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +9.0 | +7.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -2.0 | -9.0 | -9.0 | -9.0 | -9.0 | -9.0 | -9.0 | -9.0 | -7.0 | -7.0 | -7.0 | -7.0 | -7.0 | -7.0 | -7.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |
| huffman | +0.0 | +21.0 | +27.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +21.0 | +27.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +6.0 | -21.0 | -21.0 | -21.0 | -21.0 | -21.0 | -21.0 | -21.0 | -27.0 | -27.0 | -27.0 | -27.0 | -27.0 | -27.0 | -27.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |
| lzw | +0.0 | +18.0 | +17.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +18.0 | +17.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -1.0 | -18.0 | -18.0 | -18.0 | -18.0 | -18.0 | -18.0 | -18.0 | -17.0 | -17.0 | -17.0 | -17.0 | -17.0 | -17.0 | -17.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |
| rle | +0.0 | +1.5 | +2.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +1.5 | +2.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.5 | -1.5 | -1.5 | -1.5 | -1.5 | -1.5 | -1.5 | -1.5 | -2.0 | -2.0 | -2.0 | -2.0 | -2.0 | -2.0 | -2.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |

## 4. Prompt Composition (char lengths)

Shows how much of the prompt is shared preamble vs algorithm-specific content.

| Algorithm | Task | Total Chars | Total Tokens (first model) |
|---|---|---:|---:|
| ae | encode | 3049 | 768 |
| ae | decode | 3345 | 829 |
| ae | encode_inv | 3519 | 860 |
| ae | decode_inv | 3286 | 819 |
| huffman | encode | 3371 | 866 |
| huffman | decode | 3619 | 1024 |
| huffman | encode_inv | 3630 | 954 |
| huffman | decode_inv | 3771 | 1018 |
| lzw | encode | 2937 | 690 |
| lzw | decode | 3133 | 795 |
| lzw | encode_inv | 3242 | 762 |
| lzw | decode_inv | 3239 | 805 |
| rle | encode | 2820 | 665 |
| rle | decode | 2728 | 669 |
| rle | encode_inv | 2788 | 647 |
| rle | decode_inv | 3171 | 769 |
