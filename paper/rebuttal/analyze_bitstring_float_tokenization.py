#!/usr/bin/env python3
"""
Detailed analysis of how tokenizers handle bitstrings (Huffman) and
high-precision floating-point numbers (Arithmetic Encoding).

Produces per-model stats on:
  - Token length distribution (1-char, 2-char, 3-char+ tokens)
  - Digit grouping patterns for numeric substrings
  - Chars-per-token specifically for the numeric/binary portions
  - Whether token boundaries fall on meaningful boundaries
  - Comparison across multiple input strings of varying length

Usage:
  python paper/rebuttal/analyze_bitstring_float_tokenization.py \
    --models /path/to/ModelA /path/to/ModelB \
    --local-files-only \
    --markdown-out paper/rebuttal/bitstring_float_report.md \
    --json-out paper/rebuttal/bitstring_float_report.json
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal, getcontext
from fractions import Fraction
from itertools import count as icount
from typing import Any

from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def ae_encode(text: str) -> str:
    """Return the AE-encoded float as a high-precision decimal string."""
    freq = dict(sorted(Counter(text).items()))
    total = sum(freq.values())
    cumulative: dict[str, tuple[Fraction, Fraction]] = {}
    running = Fraction(0, 1)
    for ch, f in freq.items():
        low = running
        high = running + Fraction(f, total)
        cumulative[ch] = (low, high)
        running = high

    low = Fraction(0, 1)
    high = Fraction(1, 1)
    for ch in text:
        span = high - low
        c_low, c_high = cumulative[ch]
        high = low + span * c_high
        low = low + span * c_low

    getcontext().prec = 80
    return str(Decimal(low.numerator) / Decimal(low.denominator))


def huffman_encode(text: str) -> str:
    """Return the Huffman-encoded bitstream string."""
    freq = dict(sorted(Counter(text).items()))
    if len(freq) == 1:
        return "0" * len(text)

    uid = icount()
    heap = [(f, next(uid), ch) for ch, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        f1, _, left = heapq.heappop(heap)
        f2, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, next(uid), (left, right)))

    root = heap[0][2]
    codes: dict[str, str] = {}

    def walk(node, prefix: str) -> None:
        if isinstance(node, str):
            codes[node] = prefix or "0"
            return
        walk(node[0], prefix + "0")
        walk(node[1], prefix + "1")

    walk(root, "")
    return "".join(codes[ch] for ch in text)


# ---------------------------------------------------------------------------
# Token analysis helpers
# ---------------------------------------------------------------------------

@dataclass
class TokenDetail:
    """Analysis of a single token in the context of a numeric/binary string."""
    token_str: str
    token_id: int
    char_len: int
    is_pure_digit: bool     # all chars are 0-9
    is_pure_binary: bool    # all chars are 0 or 1
    is_pure_numeric: bool   # all chars are 0-9 or '.' or 'e' or '-' or '+'


@dataclass
class SegmentAnalysis:
    """Full analysis for one (model, sample_type, input_string) combination."""
    model: str
    sample_type: str        # "ae_float" or "huffman_bitstream"
    input_string: str
    raw_value: str          # the actual AE float or bitstream
    char_len: int
    token_count: int
    chars_per_token: float

    # Token-length distribution
    n_1char_tokens: int
    n_2char_tokens: int
    n_3char_tokens: int
    n_4plus_char_tokens: int

    # Digit grouping
    pct_1char: float
    pct_2char: float
    pct_3char: float
    pct_4plus: float
    mean_token_char_len: float
    max_token_char_len: int

    # Purity stats (what fraction of tokens are pure digit/binary)
    n_pure_digit_tokens: int
    n_pure_binary_tokens: int
    pct_pure_digit: float
    pct_pure_binary: float

    # Actual token splits (for display)
    tokens: list[str]


def analyze_segment(
    model_name: str,
    tok,
    sample_type: str,
    input_string: str,
    raw_value: str,
) -> SegmentAnalysis:
    """Tokenize raw_value and compute detailed stats."""
    ids = tok.encode(raw_value, add_special_tokens=False)
    token_strs = tok.convert_ids_to_tokens(ids)
    n = len(ids)
    char_len = len(raw_value)

    details = []
    for tid, ts in zip(ids, token_strs):
        # Some tokenizers use special prefix chars (e.g. Ġ for space, ▁ for sentencepiece)
        clean = ts.replace("Ġ", "").replace("▁", "").replace("Ã", "").replace("ĉ", "")
        clen = len(clean) if clean else len(ts)
        details.append(TokenDetail(
            token_str=ts,
            token_id=tid,
            char_len=clen,
            is_pure_digit=bool(re.fullmatch(r"[0-9]+", clean)),
            is_pure_binary=bool(re.fullmatch(r"[01]+", clean)),
            is_pure_numeric=bool(re.fullmatch(r"[0-9.eE+\-]+", clean)),
        ))

    n_1 = sum(1 for d in details if d.char_len == 1)
    n_2 = sum(1 for d in details if d.char_len == 2)
    n_3 = sum(1 for d in details if d.char_len == 3)
    n_4p = sum(1 for d in details if d.char_len >= 4)
    n_pure_digit = sum(1 for d in details if d.is_pure_digit)
    n_pure_binary = sum(1 for d in details if d.is_pure_binary)
    char_lens = [d.char_len for d in details]
    mean_cl = round(sum(char_lens) / len(char_lens), 2) if char_lens else 0.0

    return SegmentAnalysis(
        model=model_name,
        sample_type=sample_type,
        input_string=input_string,
        raw_value=raw_value[:120] + ("..." if len(raw_value) > 120 else ""),
        char_len=char_len,
        token_count=n,
        chars_per_token=round(char_len / n, 4) if n else 0.0,
        n_1char_tokens=n_1,
        n_2char_tokens=n_2,
        n_3char_tokens=n_3,
        n_4plus_char_tokens=n_4p,
        pct_1char=round(100 * n_1 / n, 1) if n else 0.0,
        pct_2char=round(100 * n_2 / n, 1) if n else 0.0,
        pct_3char=round(100 * n_3 / n, 1) if n else 0.0,
        pct_4plus=round(100 * n_4p / n, 1) if n else 0.0,
        mean_token_char_len=mean_cl,
        max_token_char_len=max(char_lens) if char_lens else 0,
        n_pure_digit_tokens=n_pure_digit,
        n_pure_binary_tokens=n_pure_binary,
        pct_pure_digit=round(100 * n_pure_digit / n, 1) if n else 0.0,
        pct_pure_binary=round(100 * n_pure_binary / n, 1) if n else 0.0,
        tokens=[d.token_str for d in details],
    )


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

INPUT_STRINGS = [
    "AAABBBCCCDDDEEE112233AABBCC",
    "HELLO",
    "abcdefghijklmnopqrstuvwxyz",
    "AAAAAABBBBBBCCCCCCDDDDDDEEEEEE",
    "The quick brown fox jumps over the lazy dog",
]


def model_short_name(path: str) -> str:
    return os.path.basename(path.rstrip("/"))


def run_analysis(model_paths: list[str], input_strings: list[str], args) -> list[SegmentAnalysis]:
    all_results: list[SegmentAnalysis] = []

    for model_path in model_paths:
        name = model_short_name(model_path)
        print(f"Loading tokenizer: {name}")
        tok = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            use_fast=True,
        )

        for inp in input_strings:
            # AE float
            ae_val = ae_encode(inp)
            all_results.append(analyze_segment(name, tok, "ae_float", inp, ae_val))

            # Huffman bitstream
            huff_val = huffman_encode(inp)
            all_results.append(analyze_segment(name, tok, "huffman_bitstream", inp, huff_val))

    return all_results


def write_markdown(path: str, results: list[SegmentAnalysis], models: list[str]):
    lines: list[str] = []
    w = lines.append

    w("# Bitstring & High-Precision Float Tokenization Analysis\n")
    w("How do tokenizers handle the **outputs** of Huffman (bitstrings) and AE (high-precision floats)?\n")

    # --- Table 1: Overview ---
    w("## 1. Overview: Token Counts\n")
    w("| Input | Type | Value (truncated) | " + " | ".join(models) + " |")
    w("|---|---|---| " + " | ".join(["---:"] * len(models)) + " |")

    by_key = {(r.model, r.sample_type, r.input_string): r for r in results}
    inputs = list(dict.fromkeys(r.input_string for r in results))
    for inp in inputs:
        for stype in ["ae_float", "huffman_bitstream"]:
            first = by_key.get((models[0], stype, inp))
            if not first:
                continue
            val_trunc = first.raw_value[:50] + ("..." if len(first.raw_value) > 50 else "")
            row = f"| `{inp[:30]}{'...' if len(inp) > 30 else ''}` | {stype} | `{val_trunc}` |"
            for m in models:
                r = by_key.get((m, stype, inp))
                row += f" {r.token_count if r else 'N/A'} |"
            w(row)

    # --- Table 2: Chars per token ---
    w("\n## 2. Chars per Token\n")
    w("| Input | Type | " + " | ".join(models) + " |")
    w("|---|---| " + " | ".join(["---:"] * len(models)) + " |")
    for inp in inputs:
        for stype in ["ae_float", "huffman_bitstream"]:
            row = f"| `{inp[:30]}{'...' if len(inp) > 30 else ''}` | {stype} |"
            for m in models:
                r = by_key.get((m, stype, inp))
                row += f" {r.chars_per_token if r else 'N/A'} |"
            w(row)

    # --- Table 3: Token length distribution ---
    w("\n## 3. Token Length Distribution\n")
    w("What fraction of tokens are 1-char, 2-char, 3-char, or 4+ chars.\n")
    w("| Model | Input | Type | 1-char% | 2-char% | 3-char% | 4+char% | Mean Len | Max Len |")
    w("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for m in models:
        for inp in inputs:
            for stype in ["ae_float", "huffman_bitstream"]:
                r = by_key.get((m, stype, inp))
                if not r:
                    continue
                inp_short = inp[:20] + ("..." if len(inp) > 20 else "")
                w(f"| {m} | `{inp_short}` | {stype} | {r.pct_1char} | {r.pct_2char} "
                  f"| {r.pct_3char} | {r.pct_4plus} | {r.mean_token_char_len} | {r.max_token_char_len} |")

    # --- Table 4: Digit purity ---
    w("\n## 4. Digit/Binary Purity of Tokens\n")
    w("How many tokens consist purely of digits (0-9) or binary (0/1).\n")
    w("| Model | Input | Type | Tokens | Pure Digit | %Digit | Pure Binary | %Binary |")
    w("|---|---|---|---:|---:|---:|---:|---:|")
    for m in models:
        for inp in inputs:
            for stype in ["ae_float", "huffman_bitstream"]:
                r = by_key.get((m, stype, inp))
                if not r:
                    continue
                inp_short = inp[:20] + ("..." if len(inp) > 20 else "")
                w(f"| {m} | `{inp_short}` | {stype} | {r.token_count} | {r.n_pure_digit_tokens} "
                  f"| {r.pct_pure_digit} | {r.n_pure_binary_tokens} | {r.pct_pure_binary} |")

    # --- Table 5: Cross-model comparison ---
    if len(models) >= 2:
        w("\n## 5. Cross-Model Token Ratio\n")
        w("Ratio of token counts between model pairs. >1 means first model uses more tokens.\n")
        pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]
        # Only show unique tokenizer families (skip identical ones)
        w("| Input | Type | " + " | ".join(f"{a} / {b}" for a, b in pairs) + " |")
        w("|---|---| " + " | ".join(["---:"] * len(pairs)) + " |")
        for inp in inputs:
            for stype in ["ae_float", "huffman_bitstream"]:
                row = f"| `{inp[:25]}{'...' if len(inp) > 25 else ''}` | {stype} |"
                for ma, mb in pairs:
                    ra = by_key.get((ma, stype, inp))
                    rb = by_key.get((mb, stype, inp))
                    if ra and rb and rb.token_count > 0:
                        ratio = round(ra.token_count / rb.token_count, 2)
                        row += f" {ratio} |"
                    else:
                        row += " N/A |"
                w(row)

    # --- Section 6: Example token splits ---
    w("\n## 6. Example Token Splits (first input)\n")
    w("Showing how each tokenizer splits the AE float and Huffman bitstream for the first input.\n")
    first_inp = inputs[0]
    for stype in ["ae_float", "huffman_bitstream"]:
        w(f"\n### {stype}\n")
        for m in models:
            r = by_key.get((m, stype, first_inp))
            if not r:
                continue
            w(f"**{m}** ({r.token_count} tokens, {r.chars_per_token} chars/tok):\n")
            # Show tokens with pipe separators
            w(f"```")
            w(f"{' | '.join(r.tokens)}")
            w(f"```\n")

    # --- Summary ---
    w("\n## 7. Summary Statistics (averaged across all inputs)\n")
    for stype in ["ae_float", "huffman_bitstream"]:
        w(f"\n### {stype}\n")
        w("| Model | Mean Tokens | Mean Chars/Token | Mean %1-char | Mean %3+char |")
        w("|---|---:|---:|---:|---:|")
        for m in models:
            subset = [r for r in results if r.model == m and r.sample_type == stype]
            if not subset:
                continue
            mean_tok = round(sum(r.token_count for r in subset) / len(subset), 1)
            mean_cpt = round(sum(r.chars_per_token for r in subset) / len(subset), 4)
            mean_1c = round(sum(r.pct_1char for r in subset) / len(subset), 1)
            mean_3p = round(sum(r.pct_3char + r.pct_4plus for r in subset) / len(subset), 1)
            w(f"| {m} | {mean_tok} | {mean_cpt} | {mean_1c} | {mean_3p} |")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"\nWrote Markdown report to: {path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze tokenizer behavior on bitstrings and high-precision floats."
    )
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--input-strings", nargs="+", default=INPUT_STRINGS,
                    help="Input strings to encode with AE/Huffman.")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--markdown-out", type=str, default=None)
    p.add_argument("--json-out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    results = run_analysis(args.models, args.input_strings, args)

    models = list(dict.fromkeys(r.model for r in results))

    # Print summary to stdout
    print(f"\n{'=' * 70}")
    print("BITSTRING & HIGH-PRECISION FLOAT TOKENIZATION ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Models: {models}")
    print(f"Input strings: {len(args.input_strings)}")

    for stype in ["ae_float", "huffman_bitstream"]:
        print(f"\n--- {stype.upper()} ---")
        for m in models:
            subset = [r for r in results if r.model == m and r.sample_type == stype]
            mean_tok = round(sum(r.token_count for r in subset) / len(subset), 1)
            mean_cpt = round(sum(r.chars_per_token for r in subset) / len(subset), 4)
            mean_1c = round(sum(r.pct_1char for r in subset) / len(subset), 1)
            mean_3p = round(sum(r.pct_3char + r.pct_4plus for r in subset) / len(subset), 1)
            print(f"  {m:40s}: mean_tokens={mean_tok:6.1f}, chars/tok={mean_cpt:.4f}, "
                  f"%1-char={mean_1c:5.1f}%, %3+char={mean_3p:5.1f}%")

    # Show example splits for first input
    first_inp = args.input_strings[0]
    print(f"\n--- EXAMPLE SPLITS (input: '{first_inp}') ---")
    for stype in ["ae_float", "huffman_bitstream"]:
        print(f"\n  {stype}:")
        for m in models:
            subset = [r for r in results if r.model == m and r.sample_type == stype
                      and r.input_string == first_inp]
            if subset:
                r = subset[0]
                print(f"    [{m}] {r.token_count} tokens, {r.chars_per_token} chars/tok")
                print(f"      value: {r.raw_value}")
                print(f"      split: {' | '.join(r.tokens[:30])}{'...' if len(r.tokens) > 30 else ''}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"results": [asdict(r) for r in results]}, f, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON to: {args.json_out}")

    if args.markdown_out and models:
        write_markdown(args.markdown_out, results, models)


if __name__ == "__main__":
    main()
