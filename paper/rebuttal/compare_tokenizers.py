#!/usr/bin/env python3
"""
Compare tokenizer behavior across LLMs on the same text samples.

Examples:
  python paper/rebuttal/compare_tokenizers.py \
    --models Qwen/Qwen2.5-7B-Instruct openai/gpt-oss-20b \
    --algorithms ae huffman lzw rle \
    --preset mixed \
    --local-files-only \
    --json-out paper/rebuttal/tokenizer_report.json

  python paper/rebuttal/compare_tokenizers.py \
    --models /path/to/local/modelA /path/to/local/modelB \
    --samples-file paper/rebuttal/samples.txt
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, getcontext
from fractions import Fraction
from itertools import count
from typing import Iterable

from transformers import AutoTokenizer

ALGO_NAMES = ["ae", "huffman", "lzw", "rle"]

PRESET_SAMPLES = {
    "compression": [
        "0101010101010101",
        "00000000000000000000000000000000",
        "11111111111111111111111111111111",
        "0.12345678901234567890",
        "3.141592653589793238462643383279",
        "1e-20",
        "interval=[0.12345678901234567,0.12345678901234568)",
        "hex=deadbeefcafebabe",
    ],
    "code": [
        "def f(x):\n    return x * x + 1",
        "for i in range(10): print(i, end=' ')",
        "if (a[i] & 1) == 0: even += 1",
        "arr.sort(key=lambda t: (t[1], -t[0]))",
    ],
    "mixed": [
        "0101010101010101",
        "0.12345678901234567890",
        "def compress(bits): return bits.count('1')",
        "while lo < hi: mid = (lo + hi) // 2",
    ],
}


@dataclass
class SampleResult:
    model: str
    sample_group: str
    sample: str
    char_len: int
    token_len: int
    chars_per_token: float
    token_ids: list[int]
    tokens: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tokenizer outputs across models.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model IDs or local model directories.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Add one inline sample. Can be provided multiple times.",
    )
    parser.add_argument(
        "--samples-file",
        type=str,
        default=None,
        help="Path to text file with one sample per line.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_SAMPLES.keys()),
        default=None,
        help="Optional built-in sample set to include.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGO_NAMES,
        default=ALGO_NAMES,
        help="Compression algorithms to generate encoding samples for.",
    )
    parser.add_argument(
        "--input-string",
        type=str,
        default="AAABBBCCCDDDEEE112233AABBCC",
        help="Alphanumeric input string used to generate AE/Huffman/LZW/RLE encoding samples.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load models/tokenizers from local cache/files only.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer.",
    )
    parser.add_argument(
        "--use-slow",
        action="store_true",
        help="Force slow tokenizer implementation (use_fast=False).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write full JSON results.",
    )
    parser.add_argument(
        "--markdown-out",
        type=str,
        default=None,
        help="Optional path to write a Markdown summary table (e.g. paper/rebuttal/tokenizer_report.md).",
    )
    return parser.parse_args()


def read_samples(samples_file: str | None) -> list[str]:
    if not samples_file:
        return []
    out: list[str] = []
    with open(samples_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if s.strip():
                out.append(s)
    return out


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def dedupe_pairs_keep_order(items: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _format_decimal(frac: Fraction, precision: int = 80) -> str:
    getcontext().prec = precision
    return str(Decimal(frac.numerator) / Decimal(frac.denominator))


def _huffman_encode(text: str) -> tuple[dict[str, int], dict[str, str], str]:
    freq = dict(sorted(Counter(text).items()))
    if len(freq) == 1:
        only = next(iter(freq))
        codes = {only: "0"}
        return freq, codes, "0" * len(text)

    uid = count()
    heap = []
    for ch, f in freq.items():
        heapq.heappush(heap, (f, next(uid), ch))

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
        left, right = node
        walk(left, prefix + "0")
        walk(right, prefix + "1")

    walk(root, "")
    bitstream = "".join(codes[ch] for ch in text)
    return freq, codes, bitstream


def _lzw_encode(text: str) -> tuple[dict[str, int], list[int], int]:
    alphabet = sorted(set(text))
    dictionary = {ch: i for i, ch in enumerate(alphabet)}
    next_code = len(dictionary)
    output: list[int] = []
    w = ""

    for c in text:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            output.append(dictionary[w])
            dictionary[wc] = next_code
            next_code += 1
            w = c
    if w:
        output.append(dictionary[w])
    init_dict = {ch: i for i, ch in enumerate(alphabet)}
    return init_dict, output, next_code


def _rle_encode(text: str) -> tuple[list[tuple[str, int]], str]:
    if not text:
        return [], ""
    runs: list[tuple[str, int]] = []
    cur = text[0]
    cnt = 1
    for ch in text[1:]:
        if ch == cur:
            cnt += 1
        else:
            runs.append((cur, cnt))
            cur = ch
            cnt = 1
    runs.append((cur, cnt))
    compact = "".join(f"{ch}{n}" for ch, n in runs)
    return runs, compact


def _ae_encode(text: str) -> tuple[dict[str, int], str, str, str]:
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

    return freq, _format_decimal(low), _format_decimal(high), _format_decimal(low)


def build_algo_samples_from_input(input_str: str, algorithms: list[str]) -> list[tuple[str, str]]:
    if not input_str:
        raise ValueError("--input-string cannot be empty.")
    if not input_str.isalnum():
        raise ValueError("--input-string must be alphanumeric only.")

    grouped: list[tuple[str, str]] = []

    if "ae" in algorithms:
        freq, low, high, code = _ae_encode(input_str)
        grouped.extend(
            [
                ("ae", f'ae_encode("{input_str}")'),
                ("ae", f"freq={freq}"),
                ("ae", f"low={low}"),
                ("ae", f"high={high}"),
                ("ae", f"code={code}"),
            ]
        )
    if "huffman" in algorithms:
        freq, codes, bits = _huffman_encode(input_str)
        grouped.extend(
            [
                ("huffman", f'huffman_encode("{input_str}")'),
                ("huffman", f"freq={freq}"),
                ("huffman", f"codes={codes}"),
                ("huffman", f"bitstream={bits}"),
            ]
        )
    if "lzw" in algorithms:
        init_dict, codes, next_code = _lzw_encode(input_str)
        grouped.extend(
            [
                ("lzw", f'lzw_encode("{input_str}")'),
                ("lzw", f"init_dict={init_dict}"),
                ("lzw", f"codes={codes}"),
                ("lzw", f"next_code={next_code}"),
            ]
        )
    if "rle" in algorithms:
        runs, compact = _rle_encode(input_str)
        grouped.extend(
            [
                ("rle", f'rle_encode("{input_str}")'),
                ("rle", f"runs={runs}"),
                ("rle", f"compact={compact}"),
            ]
        )

    return grouped


def load_tokenizer(model: str, args: argparse.Namespace):
    return AutoTokenizer.from_pretrained(
        model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        use_fast=not args.use_slow,
    )


def evaluate_model(
    model: str, grouped_samples: list[tuple[str, str]], args: argparse.Namespace
) -> list[SampleResult]:
    tok = load_tokenizer(model, args)
    results: list[SampleResult] = []
    for sample_group, sample in grouped_samples:
        token_ids = tok.encode(sample, add_special_tokens=False)
        tokens = tok.convert_ids_to_tokens(token_ids)
        token_len = len(token_ids)
        chars_per_token = (len(sample) / token_len) if token_len else 0.0
        results.append(
            SampleResult(
                model=model,
                sample_group=sample_group,
                sample=sample,
                char_len=len(sample),
                token_len=token_len,
                chars_per_token=round(chars_per_token, 4),
                token_ids=token_ids,
                tokens=tokens,
            )
        )
    return results


def print_report(
    all_results: list[SampleResult], grouped_samples: list[tuple[str, str]], models: list[str]
) -> None:
    by_key = {(r.model, r.sample_group, r.sample): r for r in all_results}
    groups = dedupe_keep_order([group for group, _ in grouped_samples])

    for group in groups:
        current_samples = [sample for g, sample in grouped_samples if g == group]
        print(f"\n=== {group.upper()} Token Length Comparison ===")
        header = ["sample", *models]
        print(" | ".join(header))
        print("-" * (len(" | ".join(header)) + 4))
        for sample in current_samples:
            row = [sample]
            for model in models:
                r = by_key[(model, group, sample)]
                row.append(str(r.token_len))
            print(" | ".join(row))

        print(f"\n=== {group.upper()} Detailed Splits ===")
        for model in models:
            print(f"\n[{model}]")
            for sample in current_samples:
                r = by_key[(model, group, sample)]
                print(f"- sample: {sample}")
                print(
                    f"  chars: {r.char_len}, tokens: {r.token_len}, chars/token: {r.chars_per_token}"
                )
                print(f"  split: {r.tokens}")


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def print_insights(
    all_results: list[SampleResult], grouped_samples: list[tuple[str, str]], models: list[str]
) -> dict:
    """Print actionable aggregate insights and return summary dict for JSON export."""
    by_key = {(r.model, r.sample_group, r.sample): r for r in all_results}
    groups = dedupe_keep_order([group for group, _ in grouped_samples])

    # ── 1. Per-model aggregate statistics ──
    print("\n" + "=" * 60)
    print("AGGREGATE TOKENIZER STATISTICS")
    print("=" * 60)
    model_stats: dict[str, dict] = {}
    for model in models:
        model_results = [r for r in all_results if r.model == model]
        token_lens = [r.token_len for r in model_results]
        cpts = [r.chars_per_token for r in model_results]
        stats = {
            "total_tokens": sum(token_lens),
            "total_chars": sum(r.char_len for r in model_results),
            "mean_token_len": round(sum(token_lens) / len(token_lens), 2),
            "std_token_len": round(_std(token_lens), 2),
            "mean_chars_per_token": round(sum(cpts) / len(cpts), 4),
            "max_token_len": max(token_lens),
            "min_token_len": min(token_lens),
        }
        model_stats[model] = stats
        print(f"\n[{model}]")
        print(f"  Total tokens across all samples : {stats['total_tokens']}")
        print(f"  Total characters                : {stats['total_chars']}")
        print(f"  Mean token count per sample     : {stats['mean_token_len']} (std={stats['std_token_len']})")
        print(f"  Mean chars/token                : {stats['mean_chars_per_token']}")
        print(f"  Token count range               : [{stats['min_token_len']}, {stats['max_token_len']}]")

    # ── 2. Per-algorithm breakdown ──
    print("\n" + "=" * 60)
    print("PER-ALGORITHM BREAKDOWN (mean tokens per sample)")
    print("=" * 60)
    algo_stats: dict[str, dict[str, dict]] = {}
    header = f"{'algorithm':<16}" + "".join(f"{m:<30}" for m in models)
    print(header)
    print("-" * len(header))
    for group in groups:
        current_samples = [sample for g, sample in grouped_samples if g == group]
        algo_stats[group] = {}
        row = f"{group:<16}"
        for model in models:
            lens = [by_key[(model, group, s)].token_len for s in current_samples]
            mean_len = round(sum(lens) / len(lens), 2)
            std_len = round(_std(lens), 2)
            algo_stats[group][model] = {"mean": mean_len, "std": std_len, "samples": len(lens)}
            row += f"{mean_len:>8} (±{std_len:<6})" + " " * (30 - len(f"{mean_len:>8} (±{std_len:<6})"))
        print(row)

    # ── 3. Pairwise model comparison ──
    if len(models) >= 2:
        print("\n" + "=" * 60)
        print("PAIRWISE TOKEN COUNT COMPARISON")
        print("=" * 60)
        pairwise_stats = {}
        for i, m1 in enumerate(models):
            for m2 in models[i + 1 :]:
                wins_m1, wins_m2, ties = 0, 0, 0
                total_diff = 0
                diffs = []
                for group, sample in grouped_samples:
                    r1 = by_key[(m1, group, sample)]
                    r2 = by_key[(m2, group, sample)]
                    diff = r1.token_len - r2.token_len
                    diffs.append(diff)
                    total_diff += diff
                    if r1.token_len < r2.token_len:
                        wins_m1 += 1
                    elif r2.token_len < r1.token_len:
                        wins_m2 += 1
                    else:
                        ties += 1
                mean_diff = round(total_diff / len(diffs), 2)
                pct_fewer_m1 = round(100 * wins_m1 / len(diffs), 1)
                pct_fewer_m2 = round(100 * wins_m2 / len(diffs), 1)
                pair_key = f"{m1} vs {m2}"
                pairwise_stats[pair_key] = {
                    "wins_model_a": wins_m1,
                    "wins_model_b": wins_m2,
                    "ties": ties,
                    "mean_token_diff_a_minus_b": mean_diff,
                }
                print(f"\n  {m1}  vs  {m2}")
                print(f"    {m1} uses fewer tokens: {wins_m1}/{len(diffs)} samples ({pct_fewer_m1}%)")
                print(f"    {m2} uses fewer tokens: {wins_m2}/{len(diffs)} samples ({pct_fewer_m2}%)")
                print(f"    Tied: {ties}/{len(diffs)} samples")
                print(f"    Mean token diff (A - B): {mean_diff:+.2f}")
                if mean_diff < 0:
                    print(f"    --> {m1} is more token-efficient on average")
                elif mean_diff > 0:
                    print(f"    --> {m2} is more token-efficient on average")
                else:
                    print(f"    --> Both tokenizers are equally efficient on average")

    # ── 4. Worst-case fragmentation (highest token count per sample) ──
    print("\n" + "=" * 60)
    print("WORST-CASE FRAGMENTATION (most tokens for a single sample)")
    print("=" * 60)
    worst_cases = {}
    for model in models:
        model_results = [r for r in all_results if r.model == model]
        worst = max(model_results, key=lambda r: r.token_len)
        worst_cases[model] = {
            "sample": worst.sample,
            "group": worst.sample_group,
            "token_len": worst.token_len,
            "chars_per_token": worst.chars_per_token,
        }
        print(f"\n  [{model}]")
        print(f"    Sample : {worst.sample[:80]}{'...' if len(worst.sample) > 80 else ''}")
        print(f"    Group  : {worst.sample_group}")
        print(f"    Tokens : {worst.token_len}  (chars/token={worst.chars_per_token})")

    # ── 5. Digit / binary handling (tokenizer-friendliness for numeric strings) ──
    print("\n" + "=" * 60)
    print("NUMERIC & BINARY STRING EFFICIENCY")
    print("=" * 60)
    numeric_stats = {}
    for model in models:
        model_results = [r for r in all_results if r.model == model]
        numeric = [r for r in model_results if all(c in "0123456789.e-+" for c in r.sample)]
        if not numeric:
            continue
        lens = [r.token_len for r in numeric]
        cpts = [r.chars_per_token for r in numeric]
        numeric_stats[model] = {
            "n_samples": len(numeric),
            "mean_tokens": round(sum(lens) / len(lens), 2),
            "mean_chars_per_token": round(sum(cpts) / len(cpts), 4),
        }
        print(f"  [{model}] {len(numeric)} numeric samples: "
              f"mean tokens={numeric_stats[model]['mean_tokens']}, "
              f"mean chars/token={numeric_stats[model]['mean_chars_per_token']}")

    # ── 6. Vocabulary overlap on these samples ──
    if len(models) >= 2:
        print("\n" + "=" * 60)
        print("TOKEN VOCABULARY OVERLAP (unique tokens used on these samples)")
        print("=" * 60)
        vocab_sets: dict[str, set[str]] = {}
        for model in models:
            model_results = [r for r in all_results if r.model == model]
            vocab_sets[model] = set()
            for r in model_results:
                vocab_sets[model].update(r.tokens)
            print(f"  [{model}] unique tokens used: {len(vocab_sets[model])}")
        overlap_stats = {}
        for i, m1 in enumerate(models):
            for m2 in models[i + 1 :]:
                inter = vocab_sets[m1] & vocab_sets[m2]
                union = vocab_sets[m1] | vocab_sets[m2]
                jaccard = round(len(inter) / len(union), 4) if union else 0.0
                overlap_stats[f"{m1} & {m2}"] = {
                    "intersection": len(inter),
                    "union": len(union),
                    "jaccard": jaccard,
                }
                print(f"  {m1} ∩ {m2}: {len(inter)} shared tokens, "
                      f"Jaccard={jaccard} ({len(union)} union)")

    # ── Build summary dict for JSON ──
    summary = {
        "model_aggregates": model_stats,
        "per_algorithm": algo_stats,
        "worst_case_fragmentation": worst_cases,
        "numeric_string_efficiency": numeric_stats,
    }
    if len(models) >= 2:
        summary["pairwise_comparison"] = pairwise_stats
        summary["vocab_overlap"] = overlap_stats
    return summary


def write_markdown(
    path: str,
    all_results: list[SampleResult],
    grouped_samples: list[tuple[str, str]],
    models: list[str],
    insights: dict,
) -> None:
    by_key = {(r.model, r.sample_group, r.sample): r for r in all_results}
    groups = dedupe_keep_order([group for group, _ in grouped_samples])
    lines: list[str] = []
    w = lines.append

    w("# Tokenizer Comparison Report\n")

    # ── Table 1: Aggregate stats ──
    w("## 1. Aggregate Statistics\n")
    w("| Model | Total Tokens | Mean Tokens/Sample | Std | Mean Chars/Token | Token Range |")
    w("|---|---:|---:|---:|---:|---|")
    for model in models:
        s = insights["model_aggregates"][model]
        w(f"| {model} | {s['total_tokens']} | {s['mean_token_len']} | {s['std_token_len']} "
          f"| {s['mean_chars_per_token']} | [{s['min_token_len']}, {s['max_token_len']}] |")

    # ── Table 2: Per-algorithm ──
    w("\n## 2. Per-Algorithm Mean Tokens\n")
    header = "| Algorithm | " + " | ".join(models) + " |"
    sep = "|---| " + " | ".join(["---:"] * len(models)) + " |"
    w(header)
    w(sep)
    for group in groups:
        row = f"| {group} |"
        for model in models:
            a = insights["per_algorithm"][group][model]
            row += f" {a['mean']} (±{a['std']}) |"
        w(row)

    # ── Table 3: Per-sample token counts ──
    w("\n## 3. Per-Sample Token Counts\n")
    header = "| Group | Sample | " + " | ".join(models) + " |"
    sep = "|---|---| " + " | ".join(["---:"] * len(models)) + " |"
    w(header)
    w(sep)
    for group in groups:
        current_samples = [sample for g, sample in grouped_samples if g == group]
        for sample in current_samples:
            safe = sample.replace("|", "\\|")
            row = f"| {group} | `{safe}` |"
            for model in models:
                r = by_key[(model, group, sample)]
                row += f" {r.token_len} |"
            w(row)

    # ── Table 4: Pairwise comparison ──
    if "pairwise_comparison" in insights:
        w("\n## 4. Pairwise Comparison\n")
        w("| Model A | Model B | A Wins | B Wins | Ties | Mean Diff (A−B) | More Efficient |")
        w("|---|---|---:|---:|---:|---:|---|")
        for pair_key, p in insights["pairwise_comparison"].items():
            m1, m2 = pair_key.split(" vs ")
            diff = p["mean_token_diff_a_minus_b"]
            winner = m1 if diff < 0 else (m2 if diff > 0 else "Tied")
            w(f"| {m1} | {m2} | {p['wins_model_a']} | {p['wins_model_b']} "
              f"| {p['ties']} | {diff:+.2f} | {winner} |")

    # ── Table 5: Worst-case fragmentation ──
    w("\n## 5. Worst-Case Fragmentation\n")
    w("| Model | Group | Sample | Tokens | Chars/Token |")
    w("|---|---|---|---:|---:|")
    for model in models:
        wc = insights["worst_case_fragmentation"][model]
        safe = wc["sample"][:60].replace("|", "\\|")
        if len(wc["sample"]) > 60:
            safe += "..."
        w(f"| {model} | {wc['group']} | `{safe}` | {wc['token_len']} | {wc['chars_per_token']} |")

    # ── Table 6: Numeric efficiency ──
    if insights.get("numeric_string_efficiency"):
        w("\n## 6. Numeric & Binary String Efficiency\n")
        w("| Model | Samples | Mean Tokens | Mean Chars/Token |")
        w("|---|---:|---:|---:|")
        for model in models:
            if model in insights["numeric_string_efficiency"]:
                ns = insights["numeric_string_efficiency"][model]
                w(f"| {model} | {ns['n_samples']} | {ns['mean_tokens']} | {ns['mean_chars_per_token']} |")

    # ── Table 7: Vocab overlap ──
    if "vocab_overlap" in insights:
        w("\n## 7. Token Vocabulary Overlap\n")
        w("| Model A | Model B | Shared Tokens | Union | Jaccard |")
        w("|---|---|---:|---:|---:|")
        for pair_key, v in insights["vocab_overlap"].items():
            m1, m2 = pair_key.split(" & ")
            w(f"| {m1} | {m2} | {v['intersection']} | {v['union']} | {v['jaccard']} |")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote Markdown report to: {path}")


def main() -> None:
    args = parse_args()

    grouped_samples: list[tuple[str, str]] = build_algo_samples_from_input(
        args.input_string, args.algorithms
    )

    if args.preset:
        grouped_samples.extend(("preset:" + args.preset, sample) for sample in PRESET_SAMPLES[args.preset])

    grouped_samples.extend(("custom", sample) for sample in args.sample)
    grouped_samples.extend(("custom", sample) for sample in read_samples(args.samples_file))
    grouped_samples = dedupe_pairs_keep_order(grouped_samples)

    if not grouped_samples:
        raise SystemExit("No samples provided. Use --sample and/or --samples-file.")

    all_results: list[SampleResult] = []
    failures: list[dict[str, str]] = []

    for model in args.models:
        try:
            all_results.extend(evaluate_model(model, grouped_samples, args))
        except Exception as exc:
            failures.append({"model": model, "error": f"{exc.__class__.__name__}: {exc}"})

    succeeded_models = dedupe_keep_order([r.model for r in all_results])
    insights_summary: dict = {}
    if succeeded_models:
        print_report(all_results, grouped_samples, succeeded_models)
        insights_summary = print_insights(all_results, grouped_samples, succeeded_models)
    else:
        print("No models loaded successfully.")

    if failures:
        print("\n=== Load Failures ===")
        for item in failures:
            print(f"- {item['model']}: {item['error']}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        payload = {
            "models_requested": args.models,
            "models_succeeded": succeeded_models,
            "grouped_samples": [{"group": g, "sample": s} for g, s in grouped_samples],
            "results": [asdict(r) for r in all_results],
            "insights": insights_summary,
            "failures": failures,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON report to: {args.json_out}")

    if args.markdown_out and succeeded_models:
        write_markdown(args.markdown_out, all_results, grouped_samples, succeeded_models, insights_summary)


if __name__ == "__main__":
    main()
