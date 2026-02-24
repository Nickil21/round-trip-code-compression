#!/usr/bin/env python3
"""
Tokenize the *full prompts* the LLM sees during compression tasks,
not just the isolated algorithm outputs.

Usage:
  python paper/rebuttal/tokenize_full_prompts.py \
    --models /path/to/ModelA /path/to/ModelB \
    --input-string "AAABBBCCCDDDEEE112233AABBCC" \
    --local-files-only \
    --markdown-out paper/rebuttal/full_prompt_tokenizer_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable

from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Re-use the prompt templates from the main codebase
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from prompts import (
    OUTPUT_PRED_TEMPLATE,
    INPUT_PRED_TEMPLATE,
    OUTPUT_PRED_TEMPLATE_INV_EXTRA,
    INPUT_PRED_TEMPLATE_INV_EXTRA,
)

TASK_TYPES = [
    ("output_execution_prediction", "encode"),
    ("input_execution_prediction", "decode"),
    ("output_execution_prediction_with_inversion", "encode_inv"),
    ("input_execution_prediction_with_inversion", "decode_inv"),
]

SYSTEM_MSG = (
    "You are a helpful programming assistant designed to execute code. "
    "You must verify your own output via a round\u2010trip check and "
    "self\u2010correct before returning the final JSON."
)


# ---------------------------------------------------------------------------
# Algorithm definitions (encoding/decoding fns, io_req, expected outputs)
# ---------------------------------------------------------------------------

def _build_rle(input_str: str) -> dict:
    # Run RLE
    if not input_str:
        encoded = []
    else:
        runs, prev, cnt = [], input_str[0], 1
        for ch in input_str[1:]:
            if ch == prev:
                cnt += 1
            else:
                runs.append((prev, cnt))
                prev, cnt = ch, 1
        runs.append((prev, cnt))
        encoded = runs

    encoding_fn = (
        "def main_solution(uncompressed):\n"
        "    if not uncompressed:\n"
        "        return []\n"
        "    result = []\n"
        "    prev_char = uncompressed[0]\n"
        "    count = 1\n"
        "    for c in uncompressed[1:]:\n"
        "        if c == prev_char:\n"
        "            count += 1\n"
        "        else:\n"
        "            result.append((prev_char, count))\n"
        "            prev_char = c\n"
        "            count = 1\n"
        "    result.append((prev_char, count))\n"
        "    return result"
    )
    decoding_fn = (
        "def main_solution(compressed):\n"
        "    result = []\n"
        "    for char, count in compressed:\n"
        "        result.append(char * count)\n"
        "    return ''.join(result)"
    )
    io_req_enc = (
        "Input:\n  `uncompressed` (str): The input string to be compressed.\n\n"
        "Output:\n  `return` (list of tuple): A list of (char, count) tuples "
        "representing the RLE compressed string."
    )
    io_req_dec = (
        "Input:\n  `compressed` (list of tuple): A list of (char, count) tuples "
        "from the RLE compression.\n\n"
        "Output:\n  `return` (str): The original uncompressed string."
    )
    return {
        "encoding_fn": encoding_fn,
        "decoding_fn": decoding_fn,
        "io_req_encoding": io_req_enc,
        "io_req_decoding": io_req_dec,
        "input": input_str,
        "output": str(encoded),
    }


def _build_lzw(input_str: str) -> dict:
    # Run LZW
    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}
    w, result = "", []
    for c in input_str:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
    if w:
        result.append(dictionary[w])

    encoding_fn = (
        "def main_solution(uncompressed):\n"
        "    dict_size = 256\n"
        "    dictionary = {chr(i): i for i in range(dict_size)}\n"
        '    w = ""\n'
        "    result = []\n"
        "    for c in uncompressed:\n"
        "        wc = w + c\n"
        "        if wc in dictionary:\n"
        "            w = wc\n"
        "        else:\n"
        "            result.append(dictionary[w])\n"
        "            dictionary[wc] = dict_size\n"
        "            dict_size += 1\n"
        "            w = c\n"
        "    if w:\n"
        "        result.append(dictionary[w])\n"
        "    return result"
    )
    decoding_fn = (
        "def main_solution(compressed):\n"
        "    dict_size = 256\n"
        "    dictionary = {i: chr(i) for i in range(dict_size)}\n"
        "    result = []\n"
        "    w = chr(compressed.pop(0))\n"
        "    result.append(w)\n"
        "    for k in compressed:\n"
        "        if k in dictionary:\n"
        "            entry = dictionary[k]\n"
        "        elif k == dict_size:\n"
        "            entry = w + w[0]\n"
        "        else:\n"
        '            raise ValueError("Bad compressed k: %s" % k)\n'
        "        result.append(entry)\n"
        "        dictionary[dict_size] = w + entry[0]\n"
        "        dict_size += 1\n"
        "        w = entry\n"
        '    return "".join(result)'
    )
    io_req_enc = (
        "Input:\n  `uncompressed` (str): The input string to be compressed. "
        "It should consist of standard ASCII characters.\n\n"
        "Output:\n  `return` (list of int): A list of integers representing "
        "the compressed form of the input string using LZW encoding."
    )
    io_req_dec = (
        "Input:\n  `compressed` (list of int): A list of integers representing "
        "the compressed form of the input string using LZW encoding.\n\n"
        "Output:\n  `return` (str): The original input string before compression."
    )
    return {
        "encoding_fn": encoding_fn,
        "decoding_fn": decoding_fn,
        "io_req_encoding": io_req_enc,
        "io_req_decoding": io_req_dec,
        "input": input_str,
        "output": str(result),
    }


def _build_ae(input_str: str) -> dict:
    freq = dict(sorted(Counter(input_str).items()))
    freq["EOF"] = 1
    total = sum(freq.values())
    symbols = sorted(freq.keys())
    cum_counts = {}
    running = 0
    for sym in symbols:
        cum_counts[sym] = running
        running += freq[sym]
    low, high = 0.0, 1.0
    for c in list(input_str) + ["EOF"]:
        width = high - low
        high = low + width * (cum_counts[c] + freq[c]) / total
        low = low + width * cum_counts[c] / total
    encoded = (low + high) / 2

    encoding_fn = (
        "def main_solution(uncompressed):\n"
        f"    freq = {freq}\n"
        "    total = sum(freq.values())\n"
        "    symbols = sorted(freq.keys())\n"
        "    cum_counts = {}\n"
        "    running = 0\n"
        "    for sym in symbols:\n"
        "        cum_counts[sym] = running\n"
        "        running += freq[sym]\n"
        "    low, high = 0.0, 1.0\n"
        "    for c in list(uncompressed) + ['EOF']:\n"
        "        width = high - low\n"
        "        high = low + width * (cum_counts[c] + freq[c]) / total\n"
        "        low = low + width * cum_counts[c] / total\n"
        "    return (low + high) / 2"
    )
    decoding_fn = (
        "def main_solution(compressed):\n"
        f"    freq = {freq}\n"
        "    total = sum(freq.values())\n"
        "    symbols = sorted(freq.keys())\n"
        "    cum_counts = {}\n"
        "    running = 0\n"
        "    for s in symbols:\n"
        "        cum_counts[s] = running\n"
        "        running += freq[s]\n"
        "    low, high = 0.0, 1.0\n"
        "    result = []\n"
        "    while True:\n"
        "        width = high - low\n"
        "        scaled = (compressed - low) / width * total\n"
        "        for s in symbols:\n"
        "            if cum_counts[s] <= scaled < cum_counts[s] + freq[s]:\n"
        "                symbol = s\n"
        "                break\n"
        "        if symbol == 'EOF':\n"
        "            break\n"
        "        result.append(symbol)\n"
        "        high = low + width * (cum_counts[symbol] + freq[symbol]) / total\n"
        "        low = low + width * cum_counts[symbol] / total\n"
        "    return ''.join(result)"
    )
    io_req_enc = (
        "Input:\n  `uncompressed` (str): The input string to be compressed. "
        "It should consist of standard ASCII characters.\n\n"
        "Output:\n  `return` (float): A probability value representing the "
        "compressed form of the input string using Arithmetic Encoding."
    )
    io_req_dec = (
        "Input:\n  `compressed` (float): A probability value representing the "
        "compressed form of the input string using Arithmetic Encoding.\n\n"
        "Output:\n  `return` (str): The original input string before compression."
    )
    return {
        "encoding_fn": encoding_fn,
        "decoding_fn": decoding_fn,
        "io_req_encoding": io_req_enc,
        "io_req_decoding": io_req_dec,
        "input": input_str,
        "output": str(encoded),
    }


def _build_huffman(input_str: str) -> dict:
    import heapq
    from collections import namedtuple
    from itertools import count as icount

    freq = dict(sorted(Counter(input_str).items()))
    uid = icount()
    heap = [(f, next(uid), sym) for sym, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        f1, _, left = heapq.heappop(heap)
        f2, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, next(uid), (left, right)))
    root = heap[0][2]
    codebook: dict[str, str] = {}

    def walk(node, prefix):
        if isinstance(node, str):
            codebook[node] = prefix or "0"
            return
        walk(node[0], prefix + "0")
        walk(node[1], prefix + "1")

    walk(root, "")
    bitstr = "".join(codebook[c] for c in input_str)
    padding = (-len(bitstr)) % 8
    bitstr += "0" * padding
    encoded_bytes = [int(bitstr[i : i + 8], 2) for i in range(0, len(bitstr), 8)]

    encoding_fn = (
        "def main_solution(uncompressed):\n"
        "    from collections import namedtuple\n"
        "    import heapq\n"
        f"    freq = {freq}\n"
        "    Node = namedtuple('Node', ['freq','symbol','left','right'])\n"
        "    Node.__lt__ = lambda a,b: a.freq < b.freq\n"
        "    heap = [Node(f,sym,None,None) for sym,f in freq.items()]\n"
        "    heapq.heapify(heap)\n"
        "    while len(heap) > 1:\n"
        "        l = heapq.heappop(heap)\n"
        "        r = heapq.heappop(heap)\n"
        "        heapq.heappush(heap, Node(l.freq+r.freq, None, l, r))\n"
        "    codebook = {}\n"
        "    def walk(node, prefix):\n"
        "        if node.symbol is not None:\n"
        "            codebook[node.symbol] = prefix or '0'\n"
        "        else:\n"
        "            walk(node.left, prefix+'0')\n"
        "            walk(node.right, prefix+'1')\n"
        "    walk(heap[0], '')\n"
        "    bitstr = ''.join(codebook[c] for c in uncompressed)\n"
        "    padding = (-len(bitstr)) % 8\n"
        "    bitstr += '0'*padding\n"
        "    return [int(bitstr[i:i+8],2) for i in range(0,len(bitstr),8)], codebook, padding"
    )
    decoding_fn = (
        "def main_solution(compressed):\n"
        "    from collections import namedtuple\n"
        "    import heapq\n"
        f"    codebook = {codebook}\n"
        f"    padding = {padding}\n"
        f"    freq = {freq}\n"
        "    Node = namedtuple('Node', ['freq','symbol','left','right'])\n"
        "    Node.__lt__ = lambda a,b: a.freq < b.freq\n"
        "    heap = [Node(f,sym,None,None) for sym,f in freq.items()]\n"
        "    heapq.heapify(heap)\n"
        "    while len(heap) > 1:\n"
        "        l = heapq.heappop(heap)\n"
        "        r = heapq.heappop(heap)\n"
        "        heapq.heappush(heap, Node(l.freq+r.freq, None, l, r))\n"
        "    root = heap[0]\n"
        "    bitstr = ''.join(map(lambda x: '{:08b}'.format(x), compressed))\n"
        "    bitstr = bitstr[:-padding] if padding else bitstr\n"
        "    result = []\n"
        "    node = root\n"
        "    for bit in bitstr:\n"
        "        node = node.left if bit=='0' else node.right\n"
        "        if node.symbol is not None:\n"
        "            result.append(node.symbol)\n"
        "            node = root\n"
        "    return ''.join(result)"
    )
    io_req_enc = (
        "Input:\n  `uncompressed` (str): The input string to be compressed.\n"
        "Output:\n  `return` (list,int,int): A tuple of (encoded_bytes, codebook, padding)."
    )
    io_req_dec = (
        "Input:\n  `compressed` (tuple): The tuple (encoded_bytes, codebook, padding).\n"
        "Output:\n  `return` (str): The original uncompressed string."
    )
    return {
        "encoding_fn": encoding_fn,
        "decoding_fn": decoding_fn,
        "io_req_encoding": io_req_enc,
        "io_req_decoding": io_req_dec,
        "input": input_str,
        "output": str((encoded_bytes, codebook, padding)),
    }


ALGO_BUILDERS = {"rle": _build_rle, "lzw": _build_lzw, "ae": _build_ae, "huffman": _build_huffman}


# ---------------------------------------------------------------------------
# Prompt assembly (mirrors build_codeio_msg.py logic)
# ---------------------------------------------------------------------------

def assemble_prompt(algo_data: dict, task_type: str) -> str:
    if task_type == "output_execution_prediction":
        template = OUTPUT_PRED_TEMPLATE
        refcode = algo_data["encoding_fn"]
        io_req = algo_data["io_req_encoding"]
    elif task_type == "input_execution_prediction":
        template = INPUT_PRED_TEMPLATE
        refcode = algo_data["decoding_fn"]
        io_req = algo_data["io_req_decoding"]
    elif task_type == "output_execution_prediction_with_inversion":
        template = OUTPUT_PRED_TEMPLATE_INV_EXTRA
        refcode = algo_data["decoding_fn"]
        io_req = algo_data["io_req_encoding"]
    elif task_type == "input_execution_prediction_with_inversion":
        template = INPUT_PRED_TEMPLATE_INV_EXTRA
        refcode = algo_data["encoding_fn"]
        io_req = algo_data["io_req_decoding"]
    else:
        raise ValueError(task_type)

    prompt = template.replace("<<<<io_req>>>>", io_req).replace("<<<<refcode>>>>", refcode)

    if task_type.startswith("output_"):
        prompt = prompt.replace("<<<<input>>>>", algo_data["input"])
    else:
        prompt = prompt.replace("<<<<output>>>>", algo_data["output"])

    return prompt


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

@dataclass
class PromptTokenResult:
    model: str
    algorithm: str
    task_type: str
    task_short: str
    prompt_char_len: int
    prompt_token_len: int
    chars_per_token: float


def model_short_name(path: str) -> str:
    return os.path.basename(path.rstrip("/"))


def tokenize_prompts(
    model_path: str, prompts: list[tuple[str, str, str, str]], args
) -> list[PromptTokenResult]:
    tok = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    name = model_short_name(model_path)
    results = []
    for algo, task_type, task_short, prompt_text in prompts:
        full_text = SYSTEM_MSG + "\n" + prompt_text
        ids = tok.encode(full_text, add_special_tokens=False)
        tlen = len(ids)
        clen = len(full_text)
        results.append(
            PromptTokenResult(
                model=name,
                algorithm=algo,
                task_type=task_type,
                task_short=task_short,
                prompt_char_len=clen,
                prompt_token_len=tlen,
                chars_per_token=round(clen / tlen, 4) if tlen else 0.0,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_markdown(path: str, all_results: list[PromptTokenResult], models: list[str], algos: list[str]):
    lines: list[str] = []
    w = lines.append

    w("# Full-Prompt Tokenizer Comparison\n")
    w("Token counts for the **complete prompt** the LLM receives (system message + "
      "preamble + IO requirements + reference code + input/output).\n")

    # Summary table: algorithm x task x model
    w("## 1. Token Counts by Algorithm and Task\n")
    header = "| Algorithm | Task | " + " | ".join(models) + " |"
    sep = "|---|---| " + " | ".join(["---:"] * len(models)) + " |"
    w(header)
    w(sep)

    by_key = {(r.model, r.algorithm, r.task_short): r for r in all_results}
    task_shorts = list(dict.fromkeys(r.task_short for r in all_results))

    for algo in algos:
        for ts in task_shorts:
            row = f"| {algo} | {ts} |"
            for m in models:
                r = by_key.get((m, algo, ts))
                row += f" {r.prompt_token_len if r else 'N/A'} |"
            w(row)

    # Per-algorithm average
    w("\n## 2. Mean Tokens per Algorithm (averaged across 4 task types)\n")
    header = "| Algorithm | " + " | ".join(models) + " |"
    sep = "|---| " + " | ".join(["---:"] * len(models)) + " |"
    w(header)
    w(sep)
    for algo in algos:
        row = f"| {algo} |"
        for m in models:
            vals = [by_key[(m, algo, ts)].prompt_token_len for ts in task_shorts if (m, algo, ts) in by_key]
            mean = round(sum(vals) / len(vals), 1) if vals else 0
            row += f" {mean} |"
        w(row)

    # Cross-model delta
    if len(models) >= 2:
        w("\n## 3. Pairwise Token Difference (Model A − Model B, averaged across tasks)\n")
        w("| Algorithm | " + " | ".join(f"{models[i]} vs {models[j]}" for i in range(len(models)) for j in range(i+1, len(models))) + " |")
        w("|---| " + " | ".join(["---:"] * (len(models) * (len(models)-1) // 2)) + " |")
        for algo in algos:
            row = f"| {algo} |"
            for i in range(len(models)):
                for j in range(i+1, len(models)):
                    vals_a = [by_key[(models[i], algo, ts)].prompt_token_len for ts in task_shorts if (models[i], algo, ts) in by_key]
                    vals_b = [by_key[(models[j], algo, ts)].prompt_token_len for ts in task_shorts if (models[j], algo, ts) in by_key]
                    diff = round(sum(vals_a) / len(vals_a) - sum(vals_b) / len(vals_b), 1)
                    row += f" {diff:+.1f} |"
            w(row)

    # Breakdown: what fraction of total tokens is the "variable" part
    w("\n## 4. Prompt Composition (char lengths)\n")
    w("Shows how much of the prompt is shared preamble vs algorithm-specific content.\n")
    # Just show char lengths for first model (same across all)
    w("| Algorithm | Task | Total Chars | Total Tokens (first model) |")
    w("|---|---|---:|---:|")
    first_model = models[0]
    for algo in algos:
        for ts in task_shorts:
            r = by_key.get((first_model, algo, ts))
            if r:
                w(f"| {algo} | {ts} | {r.prompt_char_len} | {r.prompt_token_len} |")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote Markdown report to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Tokenize full LLM prompts for compression tasks.")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--algorithms", nargs="+", default=list(ALGO_BUILDERS.keys()),
                    choices=list(ALGO_BUILDERS.keys()))
    p.add_argument("--input-string", default="AAABBBCCCDDDEEE112233AABBCC")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--markdown-out", type=str, default=None)
    p.add_argument("--json-out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Build all prompts
    prompts: list[tuple[str, str, str, str]] = []  # (algo, task_type, task_short, text)
    for algo in args.algorithms:
        algo_data = ALGO_BUILDERS[algo](args.input_string)
        for task_type, task_short in TASK_TYPES:
            text = assemble_prompt(algo_data, task_type)
            prompts.append((algo, task_type, task_short, text))

    # Tokenize with each model
    all_results: list[PromptTokenResult] = []
    failures = []
    for model in args.models:
        try:
            all_results.extend(tokenize_prompts(model, prompts, args))
        except Exception as exc:
            failures.append({"model": model, "error": f"{exc.__class__.__name__}: {exc}"})

    models = list(dict.fromkeys(r.model for r in all_results))

    # Print summary
    print(f"\n{'='*60}")
    print("FULL-PROMPT TOKENIZER COMPARISON")
    print(f"{'='*60}")
    print(f"Input string: {args.input_string}")
    print(f"Models: {models}")
    print(f"Algorithms: {args.algorithms}\n")

    by_key = {(r.model, r.algorithm, r.task_short): r for r in all_results}
    task_shorts = list(dict.fromkeys(r.task_short for r in all_results))
    for algo in args.algorithms:
        print(f"\n--- {algo.upper()} ---")
        for ts in task_shorts:
            tokens = []
            for m in models:
                r = by_key.get((m, algo, ts))
                tokens.append(f"{m}={r.prompt_token_len}" if r else f"{m}=N/A")
            print(f"  {ts:12s}: {', '.join(tokens)}")

    if failures:
        print("\n=== Failures ===")
        for f in failures:
            print(f"  {f['model']}: {f['error']}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"results": [asdict(r) for r in all_results], "failures": failures},
                      f, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON to: {args.json_out}")

    if args.markdown_out and models:
        write_markdown(args.markdown_out, all_results, models, args.algorithms)


if __name__ == "__main__":
    main()
