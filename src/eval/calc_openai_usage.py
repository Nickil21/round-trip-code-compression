#!/usr/bin/env python3
"""
calc_openai_usage.py — Query the OpenAI Usage API and report token usage + cost estimates.

Usage:
  python src/eval/calc_openai_usage.py                        # current month, table view
  python src/eval/calc_openai_usage.py --start 2026-01-01     # from a date
  python src/eval/calc_openai_usage.py --start 2025-01-01 --end 2025-01-31
  python src/eval/calc_openai_usage.py --bucket 1h            # hourly buckets
  python src/eval/calc_openai_usage.py --json                 # raw JSON dump

  # Local estimation from output JSONL files (no API key scope required):
  python src/eval/calc_openai_usage.py --local
  python src/eval/calc_openai_usage.py --local --data_dir data/processed_test
"""

import os
import sys
import glob
import json
import argparse
import datetime
from collections import defaultdict

import httpx
from dotenv import load_dotenv
from tabulate import tabulate

# ─── Pricing: USD per 1M tokens (as of Feb 2026) ───────────────────────────
# Keys are prefix-matched against the model name returned by the API.
PRICING = {
    "gpt-4.1-mini":  {"input": 0.40,  "cached": 0.10,  "output": 1.60},
    "gpt-4.1-nano":  {"input": 0.10,  "cached": 0.025, "output": 0.40},
    "gpt-4.1":       {"input": 2.00,  "cached": 0.50,  "output": 8.00},
    "gpt-4o-mini":   {"input": 0.15,  "cached": 0.075, "output": 0.60},
    "gpt-4o":        {"input": 2.50,  "cached": 1.25,  "output": 10.00},
}
# More-specific prefixes must come before less-specific ones (already ordered above).


def _price_for(model: str):
    """Return the pricing dict for a model, matched by prefix (longest wins)."""
    for prefix in PRICING:
        if model.startswith(prefix):
            return PRICING[prefix]
    return None


def _cost(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float | None:
    p = _price_for(model)
    if p is None:
        return None
    non_cached = max(0, input_tokens - cached_tokens)
    return (
        non_cached    * p["input"]  +
        cached_tokens * p["cached"] +
        output_tokens * p["output"]
    ) / 1_000_000


_USAGE_URL = "https://api.openai.com/v1/organization/usage/completions"


def _fetch_usage(api_key: str, start_ts: int, end_ts: int, bucket_width: str) -> list[dict]:
    """
    Pull all pages from the completions usage endpoint via direct HTTP (httpx).
    Works with any openai SDK version.
    Requires an org-level/admin API key; project keys may get a 403.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    # group_by must be passed as a repeated param: group_by[]=model
    base_params = [
        ("start_time",    start_ts),
        ("end_time",      end_ts),
        ("bucket_width",  bucket_width),
        ("limit",         168),
        ("group_by[]",    "model"),
    ]

    buckets: list[dict] = []
    next_page = None

    with httpx.Client(timeout=30.0) as http:
        while True:
            params = base_params[:]
            if next_page:
                params.append(("page", next_page))

            resp = http.get(_USAGE_URL, headers=headers, params=params)

            if resp.status_code == 403:
                raise PermissionError(resp.text)
            resp.raise_for_status()

            body = resp.json()
            buckets.extend(body.get("data", []))

            if not body.get("has_more"):
                break
            next_page = body.get("next_page")
            if not next_page:
                break

    return buckets


def _aggregate(buckets: list[dict]) -> dict:
    """Sum usage across all time buckets, keyed by model name."""
    agg = defaultdict(lambda: {"requests": 0, "input": 0, "cached": 0, "output": 0})
    for bucket in buckets:
        for r in bucket.get("results", []):
            m = r.get("model") or "unknown"
            agg[m]["requests"] += r.get("num_model_requests",  0) or 0
            agg[m]["input"]    += r.get("input_tokens",        0) or 0
            agg[m]["cached"]   += r.get("input_cached_tokens", 0) or 0
            agg[m]["output"]   += r.get("output_tokens",       0) or 0
    return agg


def _print_table(agg: dict, start: datetime.date, end: datetime.date) -> None:
    rows = []
    total_cost = 0.0

    for model_name, v in sorted(agg.items()):
        cost = _cost(model_name, v["input"], v["cached"], v["output"])
        if cost is not None:
            total_cost += cost
        rows.append([
            model_name,
            f"{v['requests']:,}",
            f"{v['input']:,}",
            f"{v['cached']:,}",
            f"{v['output']:,}",
            f"${cost:.4f}" if cost is not None else "n/a",
        ])

    # Totals row
    tot_req = sum(v["requests"] for v in agg.values())
    tot_in  = sum(v["input"]    for v in agg.values())
    tot_cac = sum(v["cached"]   for v in agg.values())
    tot_out = sum(v["output"]   for v in agg.values())
    rows.append(["TOTAL", f"{tot_req:,}", f"{tot_in:,}", f"{tot_cac:,}", f"{tot_out:,}", f"${total_cost:.4f}"])

    print(tabulate(
        rows,
        headers=["Model", "Requests", "Input Tokens", "Cached Tokens", "Output Tokens", "Est. Cost (USD)"],
        tablefmt="github",
    ))
    if start.year > 1970:
        print(f"\nPeriod : {start} → {end}")
    print("Prices : see PRICING dict in this script; cached = half input rate.")
    print("Note   : Estimates may differ from the OpenAI dashboard due to rounding.")


def _tiktoken_encode(model: str, text: str) -> int:
    """Count tokens for a string using tiktoken, with a safe fallback."""
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Rough fallback: ~4 chars per token
        return max(1, len(text) // 4)


def _local_aggregate(data_dir: str) -> dict:
    """
    Scan all OpenAI output JSONL files under data_dir, count tokens using tiktoken,
    and aggregate by model.  Only processes rows where 'model' is an OpenAI model
    (no '/' in the name).
    """
    pattern = os.path.join(data_dir, "**", "*.jsonl")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return {}

    agg = defaultdict(lambda: {"requests": 0, "input": 0, "cached": 0, "output": 0})

    for path in sorted(files):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                model_name = row.get("model", "")
                if not model_name or "/" in model_name:
                    continue  # skip HuggingFace models

                # Count input tokens from messages
                messages = row.get("messages", [])
                input_text = " ".join(m.get("content", "") for m in messages)
                input_toks = _tiktoken_encode(model_name, input_text)

                # Count output tokens
                output_text = row.get("output", "") or ""
                output_toks = _tiktoken_encode(model_name, output_text)

                agg[model_name]["requests"] += 1
                agg[model_name]["input"]    += input_toks
                agg[model_name]["output"]   += output_toks

    return agg


def main():
    load_dotenv()

    p = argparse.ArgumentParser(description="Show OpenAI token usage and cost estimates.")
    p.add_argument("--start",    default=None,
                   help="Start date YYYY-MM-DD (default: first day of current month).")
    p.add_argument("--end",      default=None,
                   help="End date YYYY-MM-DD exclusive (default: tomorrow).")
    p.add_argument("--bucket",   default="1d", choices=["1m", "1h", "1d"],
                   help="Aggregation bucket width (default: 1d).")
    p.add_argument("--json",     action="store_true",
                   help="Dump raw API response as JSON instead of a table.")
    p.add_argument("--local",    action="store_true",
                   help="Estimate usage from local output JSONL files (no api.usage.read scope needed).")
    p.add_argument("--data_dir", default="data/processed",
                   help="Root directory to scan for output JSONL files (used with --local).")
    args = p.parse_args()

    # ── Local estimation mode ────────────────────────────────────────────────
    if args.local:
        print(f"Scanning local JSONL files under '{args.data_dir}' …")
        agg = _local_aggregate(args.data_dir)
        if not agg:
            sys.exit(f"No OpenAI model output found under '{args.data_dir}'.")
        today = datetime.date.today()
        print("(Token counts are tiktoken estimates; cached tokens assumed 0.)\n")
        _print_table(agg, start=datetime.date(1970, 1, 1), end=today)
        return

    # ── API mode ─────────────────────────────────────────────────────────────
    today = datetime.date.today()
    start = datetime.date.fromisoformat(args.start) if args.start else today.replace(day=1)
    end   = datetime.date.fromisoformat(args.end)   if args.end   else today + datetime.timedelta(days=1)

    if start >= end:
        sys.exit(f"--start ({start}) must be before --end ({end}).")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set (checked env and .env).")

    start_ts = int(datetime.datetime.combine(start, datetime.time.min,
                                              tzinfo=datetime.timezone.utc).timestamp())
    end_ts   = int(datetime.datetime.combine(end,   datetime.time.min,
                                              tzinfo=datetime.timezone.utc).timestamp())

    print(f"Fetching completions usage: {start} → {end} (bucket={args.bucket}) …")
    try:
        buckets = _fetch_usage(api_key, start_ts, end_ts, args.bucket)
    except PermissionError as e:
        sys.exit(
            f"Permission denied: {e}\n"
            "The usage API requires an org-level or admin API key with 'api.usage.read' scope.\n"
            "Fix: platform.openai.com/api-keys → Create key → Permissions → All (or enable Usage Read).\n"
            "Alternative: run with --local to estimate from output JSONL files instead."
        )

    if args.json:
        print(json.dumps(buckets, indent=2, default=str))
        return

    agg = _aggregate(buckets)
    if not agg:
        print("No usage found for this period.")
        return

    _print_table(agg, start, end)


if __name__ == "__main__":
    main()
