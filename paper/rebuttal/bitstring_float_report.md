# Bitstring & High-Precision Float Tokenization Analysis

How do tokenizers handle the **outputs** of Huffman (bitstrings) and AE (high-precision floats)?

## 1. Overview: Token Counts

| Input | Type | Value (truncated) | Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct |
|---|---|---| ---: | ---: |
| `AAABBBCCCDDDEEE112233AABBCC` | ae_float | `0.274179808058694536127943821321122196963000617747...` | 82 | 29 |
| `AAABBBCCCDDDEEE112233AABBCC` | huffman_bitstream | `11011011011111111100000001101101110010010010101010...` | 80 | 27 |
| `HELLO` | ae_float | `0.22752` | 7 | 4 |
| `HELLO` | huffman_bitstream | `0100111110` | 10 | 4 |
| `abcdefghijklmnopqrstuvwxyz` | ae_float | `0.001599999999999999999999999999999999830802506931...` | 84 | 30 |
| `abcdefghijklmnopqrstuvwxyz` | huffman_bitstream | `01100011010111001111100001000110010100111010010101...` | 124 | 42 |
| `AAAAAABBBBBBCCCCCCDDDDDDEEEEEE` | ae_float | `0.000016001024065540193230258176` | 32 | 12 |
| `AAAAAABBBBBBCCCCCCDDDDDDEEEEEE` | huffman_bitstream | `11011011011011011011111111111111111100000000000001...` | 72 | 24 |
| `The quick brown fox jumps over...` | ae_float | `0.196118921131261592716943205974741895263354765305...` | 82 | 29 |
| `The quick brown fox jumps over...` | huffman_bitstream | `11111011100101011001100111100010100001001111100000...` | 194 | 65 |

## 2. Chars per Token

| Input | Type | Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct |
|---|---| ---: | ---: |
| `AAABBBCCCDDDEEE112233AABBCC` | ae_float | 1.0 | 2.8276 |
| `AAABBBCCCDDDEEE112233AABBCC` | huffman_bitstream | 1.0 | 2.963 |
| `HELLO` | ae_float | 1.0 | 1.75 |
| `HELLO` | huffman_bitstream | 1.0 | 2.5 |
| `abcdefghijklmnopqrstuvwxyz` | ae_float | 1.0 | 2.8 |
| `abcdefghijklmnopqrstuvwxyz` | huffman_bitstream | 1.0 | 2.9524 |
| `AAAAAABBBBBBCCCCCCDDDDDDEEEEEE` | ae_float | 1.0 | 2.6667 |
| `AAAAAABBBBBBCCCCCCDDDDDDEEEEEE` | huffman_bitstream | 1.0 | 3.0 |
| `The quick brown fox jumps over...` | ae_float | 1.0 | 2.8276 |
| `The quick brown fox jumps over...` | huffman_bitstream | 1.0 | 2.9846 |

## 3. Token Length Distribution

What fraction of tokens are 1-char, 2-char, 3-char, or 4+ chars.

| Model | Input | Type | 1-char% | 2-char% | 3-char% | 4+char% | Mean Len | Max Len |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | `AAABBBCCCDDDEEE11223...` | ae_float | 100.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1 |
| Qwen2.5-7B-Instruct | `AAABBBCCCDDDEEE11223...` | huffman_bitstream | 100.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1 |
| Qwen2.5-7B-Instruct | `HELLO` | ae_float | 100.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1 |
| Qwen2.5-7B-Instruct | `HELLO` | huffman_bitstream | 100.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1 |
| Qwen2.5-7B-Instruct | `abcdefghijklmnopqrst...` | ae_float | 100.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1 |
| Qwen2.5-7B-Instruct | `abcdefghijklmnopqrst...` | huffman_bitstream | 100.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1 |
| Qwen2.5-7B-Instruct | `AAAAAABBBBBBCCCCCCDD...` | ae_float | 100.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1 |
| Qwen2.5-7B-Instruct | `AAAAAABBBBBBCCCCCCDD...` | huffman_bitstream | 100.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1 |
| Qwen2.5-7B-Instruct | `The quick brown fox ...` | ae_float | 100.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1 |
| Qwen2.5-7B-Instruct | `The quick brown fox ...` | huffman_bitstream | 100.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1 |
| Llama-3.1-8B-Instruct | `AAABBBCCCDDDEEE11223...` | ae_float | 6.9 | 3.4 | 89.7 | 0.0 | 2.83 | 3 |
| Llama-3.1-8B-Instruct | `AAABBBCCCDDDEEE11223...` | huffman_bitstream | 0.0 | 3.7 | 96.3 | 0.0 | 2.96 | 3 |
| Llama-3.1-8B-Instruct | `HELLO` | ae_float | 50.0 | 25.0 | 25.0 | 0.0 | 1.75 | 3 |
| Llama-3.1-8B-Instruct | `HELLO` | huffman_bitstream | 25.0 | 0.0 | 75.0 | 0.0 | 2.5 | 3 |
| Llama-3.1-8B-Instruct | `abcdefghijklmnopqrst...` | ae_float | 10.0 | 0.0 | 90.0 | 0.0 | 2.8 | 3 |
| Llama-3.1-8B-Instruct | `abcdefghijklmnopqrst...` | huffman_bitstream | 2.4 | 0.0 | 97.6 | 0.0 | 2.95 | 3 |
| Llama-3.1-8B-Instruct | `AAAAAABBBBBBCCCCCCDD...` | ae_float | 16.7 | 0.0 | 83.3 | 0.0 | 2.67 | 3 |
| Llama-3.1-8B-Instruct | `AAAAAABBBBBBCCCCCCDD...` | huffman_bitstream | 0.0 | 0.0 | 100.0 | 0.0 | 3.0 | 3 |
| Llama-3.1-8B-Instruct | `The quick brown fox ...` | ae_float | 6.9 | 3.4 | 89.7 | 0.0 | 2.83 | 3 |
| Llama-3.1-8B-Instruct | `The quick brown fox ...` | huffman_bitstream | 0.0 | 1.5 | 98.5 | 0.0 | 2.98 | 3 |

## 4. Digit/Binary Purity of Tokens

How many tokens consist purely of digits (0-9) or binary (0/1).

| Model | Input | Type | Tokens | Pure Digit | %Digit | Pure Binary | %Binary |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | `AAABBBCCCDDDEEE11223...` | ae_float | 82 | 81 | 98.8 | 20 | 24.4 |
| Qwen2.5-7B-Instruct | `AAABBBCCCDDDEEE11223...` | huffman_bitstream | 80 | 80 | 100.0 | 80 | 100.0 |
| Qwen2.5-7B-Instruct | `HELLO` | ae_float | 7 | 6 | 85.7 | 1 | 14.3 |
| Qwen2.5-7B-Instruct | `HELLO` | huffman_bitstream | 10 | 10 | 100.0 | 10 | 100.0 |
| Qwen2.5-7B-Instruct | `abcdefghijklmnopqrst...` | ae_float | 84 | 83 | 98.8 | 11 | 13.1 |
| Qwen2.5-7B-Instruct | `abcdefghijklmnopqrst...` | huffman_bitstream | 124 | 124 | 100.0 | 124 | 100.0 |
| Qwen2.5-7B-Instruct | `AAAAAABBBBBBCCCCCCDD...` | ae_float | 32 | 31 | 96.9 | 15 | 46.9 |
| Qwen2.5-7B-Instruct | `AAAAAABBBBBBCCCCCCDD...` | huffman_bitstream | 72 | 72 | 100.0 | 72 | 100.0 |
| Qwen2.5-7B-Instruct | `The quick brown fox ...` | ae_float | 82 | 81 | 98.8 | 19 | 23.2 |
| Qwen2.5-7B-Instruct | `The quick brown fox ...` | huffman_bitstream | 194 | 194 | 100.0 | 194 | 100.0 |
| Llama-3.1-8B-Instruct | `AAABBBCCCDDDEEE11223...` | ae_float | 29 | 28 | 96.6 | 2 | 6.9 |
| Llama-3.1-8B-Instruct | `AAABBBCCCDDDEEE11223...` | huffman_bitstream | 27 | 27 | 100.0 | 27 | 100.0 |
| Llama-3.1-8B-Instruct | `HELLO` | ae_float | 4 | 3 | 75.0 | 1 | 25.0 |
| Llama-3.1-8B-Instruct | `HELLO` | huffman_bitstream | 4 | 4 | 100.0 | 4 | 100.0 |
| Llama-3.1-8B-Instruct | `abcdefghijklmnopqrst...` | ae_float | 30 | 29 | 96.7 | 2 | 6.7 |
| Llama-3.1-8B-Instruct | `abcdefghijklmnopqrst...` | huffman_bitstream | 42 | 42 | 100.0 | 42 | 100.0 |
| Llama-3.1-8B-Instruct | `AAAAAABBBBBBCCCCCCDD...` | ae_float | 12 | 11 | 91.7 | 3 | 25.0 |
| Llama-3.1-8B-Instruct | `AAAAAABBBBBBCCCCCCDD...` | huffman_bitstream | 24 | 24 | 100.0 | 24 | 100.0 |
| Llama-3.1-8B-Instruct | `The quick brown fox ...` | ae_float | 29 | 28 | 96.6 | 1 | 3.4 |
| Llama-3.1-8B-Instruct | `The quick brown fox ...` | huffman_bitstream | 65 | 65 | 100.0 | 65 | 100.0 |

## 5. Cross-Model Token Ratio

Ratio of token counts between model pairs. >1 means first model uses more tokens.

| Input | Type | Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct |
|---|---| ---: |
| `AAABBBCCCDDDEEE112233AABB...` | ae_float | 2.83 |
| `AAABBBCCCDDDEEE112233AABB...` | huffman_bitstream | 2.96 |
| `HELLO` | ae_float | 1.75 |
| `HELLO` | huffman_bitstream | 2.5 |
| `abcdefghijklmnopqrstuvwxy...` | ae_float | 2.8 |
| `abcdefghijklmnopqrstuvwxy...` | huffman_bitstream | 2.95 |
| `AAAAAABBBBBBCCCCCCDDDDDDE...` | ae_float | 2.67 |
| `AAAAAABBBBBBCCCCCCDDDDDDE...` | huffman_bitstream | 3.0 |
| `The quick brown fox jumps...` | ae_float | 2.83 |
| `The quick brown fox jumps...` | huffman_bitstream | 2.98 |

## 6. Example Token Splits (first input)

Showing how each tokenizer splits the AE float and Huffman bitstream for the first input.


### ae_float

**Qwen2.5-7B-Instruct** (82 tokens, 1.0 chars/tok):

```
0 | . | 2 | 7 | 4 | 1 | 7 | 9 | 8 | 0 | 8 | 0 | 5 | 8 | 6 | 9 | 4 | 5 | 3 | 6 | 1 | 2 | 7 | 9 | 4 | 3 | 8 | 2 | 1 | 3 | 2 | 1 | 1 | 2 | 2 | 1 | 9 | 6 | 9 | 6 | 3 | 0 | 0 | 0 | 6 | 1 | 7 | 7 | 4 | 7 | 1 | 6 | 3 | 7 | 4 | 6 | 7 | 8 | 6 | 3 | 1 | 3 | 0 | 0 | 7 | 4 | 1 | 2 | 2 | 8 | 7 | 0 | 7 | 5 | 3 | 4 | 4 | 7 | 7 | 0 | 7 | 3
```

**Llama-3.1-8B-Instruct** (29 tokens, 2.8276 chars/tok):

```
0 | . | 274 | 179 | 808 | 058 | 694 | 536 | 127 | 943 | 821 | 321 | 122 | 196 | 963 | 000 | 617 | 747 | 163 | 746 | 786 | 313 | 007 | 412 | 287 | 075 | 344 | 770 | 73
```


### huffman_bitstream

**Qwen2.5-7B-Instruct** (80 tokens, 1.0 chars/tok):

```
1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0
```

**Llama-3.1-8B-Instruct** (27 tokens, 2.963 chars/tok):

```
110 | 110 | 110 | 111 | 111 | 111 | 000 | 000 | 011 | 011 | 011 | 100 | 100 | 100 | 101 | 010 | 101 | 011 | 101 | 101 | 001 | 011 | 011 | 011 | 111 | 100 | 00
```


## 7. Summary Statistics (averaged across all inputs)


### ae_float

| Model | Mean Tokens | Mean Chars/Token | Mean %1-char | Mean %3+char |
|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 57.4 | 1.0 | 100.0 | 0.0 |
| Llama-3.1-8B-Instruct | 20.8 | 2.5744 | 18.1 | 75.5 |

### huffman_bitstream

| Model | Mean Tokens | Mean Chars/Token | Mean %1-char | Mean %3+char |
|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 96.0 | 1.0 | 100.0 | 0.0 |
| Llama-3.1-8B-Instruct | 32.4 | 2.88 | 5.5 | 93.5 |
