"""
Verify model predictions for the tokenization ablation study.

Mirrors check_io_pred_acc_mp.py but:
  - Only handles output_execution_prediction tasks.
  - Uses item["output_alt"] as the expected value (the pre-computed alternative
    ground truth stored in each sample by build_tokenization_ablation.py).
  - Uses variant-specific validators instead of the original numeric validators.

Usage:
    python src/ablation/check_tokenization_ablation.py \
        --parsed_file_name  data/processed/huffman/data.jsonl \
        --pred_file_name    data/processed/huffman/codeio_alt_base64_gens_...jsonl \
        --res_file_name     data/processed/huffman/codeio_alt_base64_gens_..._verified.jsonl \
        --algo              huffman \
        --variant           base64
"""

import os
import re
import sys
import json
import math
import base64
import pandas as pd
from itertools import islice
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.utils import read_jsonl, load_jsonl_yield, write_jsonl
from eval.check_io_pred_acc_mp import (
    _json_safe,
    extract_last_complete_json,
    _parse_ae_decimal,
    _ae_equal,
)


# ─────────────────────────────────────────────────────────────────────────────
# Variant-specific validators
# ─────────────────────────────────────────────────────────────────────────────

def _validate_huffman_base64(pred, expected):
    """pred: str like 'qg==', expected: list[int] (original byte array)."""
    if not isinstance(pred, str):
        raise TypeError(f"Huffman base64: expected str, got {type(pred).__name__}")
    try:
        decoded = list(base64.b64decode(pred.strip()))
    except Exception as e:
        raise ValueError(f"Huffman base64: could not base64-decode '{pred[:40]}': {e}")
    # expected here is the *alternative* ground truth (also a str, already converted)
    # We just check the type is right; value comparison is done by check_pred_alt.


def _validate_huffman_hex(pred, expected):
    """pred: str like 'aa00', expected: list[int] (original byte array)."""
    if not isinstance(pred, str):
        raise TypeError(f"Huffman hex: expected str, got {type(pred).__name__}")
    try:
        bytes.fromhex(pred.strip().lower())
    except Exception as e:
        raise ValueError(f"Huffman hex: could not hex-decode '{pred[:40]}': {e}")


def _validate_lzw_csv(pred, expected):
    """pred: str like '85,256,257', expected: str (comma-sep ground truth)."""
    if not isinstance(pred, str):
        raise TypeError(f"LZW csv: expected str, got {type(pred).__name__}")
    try:
        [int(x.strip()) for x in pred.strip().split(",")]
    except Exception as e:
        raise ValueError(f"LZW csv: could not parse as comma-sep ints '{pred[:60]}': {e}")


def _validate_rle_compact(pred, expected):
    """pred: str like 'U8A3', expected: str (compact ground truth)."""
    if not isinstance(pred, str):
        raise TypeError(f"RLE compact: expected str, got {type(pred).__name__}")
    parsed = re.findall(r"([A-Za-z .])(\d+)", pred)
    if not parsed and pred.strip():
        raise ValueError(f"RLE compact: could not parse '{pred[:60]}'")


def _validate_huffman_spaced_decimal(pred, expected):
    """pred: str like '170 0 255', expected: str (space-sep decimal ground truth)."""
    if not isinstance(pred, str):
        raise TypeError(f"Huffman spaced_decimal: expected str, got {type(pred).__name__}")
    try:
        vals = [int(x.strip()) for x in pred.strip().split()]
        if any(v < 0 or v > 255 for v in vals):
            raise ValueError("byte value out of range")
    except Exception as e:
        raise ValueError(f"Huffman spaced_decimal: could not parse '{pred[:60]}': {e}")


def _validate_huffman_char_hex(pred, expected):
    """pred: str like 'a a 0 0 f f', expected: str (space-sep hex chars ground truth)."""
    if not isinstance(pred, str):
        raise TypeError(f"Huffman char_hex: expected str, got {type(pred).__name__}")
    chars = pred.strip().split()
    valid = set("0123456789abcdefABCDEF")
    if not all(len(c) == 1 and c in valid for c in chars):
        raise ValueError(f"Huffman char_hex: invalid hex chars in '{pred[:60]}'")


def _validate_huffman_binary(pred, expected):
    """pred: str like '10101010 00000000', expected: str (space-sep 8-bit binary)."""
    if not isinstance(pred, str):
        raise TypeError(f"Huffman binary: expected str, got {type(pred).__name__}")
    try:
        for token in pred.strip().split():
            if len(token) != 8 or not all(c in "01" for c in token):
                raise ValueError(f"not an 8-bit binary string: '{token}'")
    except Exception as e:
        raise ValueError(f"Huffman binary: could not parse '{pred[:60]}': {e}")


def _validate_lzw_spaced(pred, expected):
    """pred: str like '85 256 257', expected: str (space-sep ints ground truth)."""
    if not isinstance(pred, str):
        raise TypeError(f"LZW spaced: expected str, got {type(pred).__name__}")
    try:
        [int(x.strip()) for x in pred.strip().split()]
    except Exception as e:
        raise ValueError(f"LZW spaced: could not parse as space-sep ints '{pred[:60]}': {e}")


def _validate_rle_spaced(pred, expected):
    """pred: str like 'U 8 A 3', expected: str (space-sep char/count ground truth)."""
    if not isinstance(pred, str):
        raise TypeError(f"RLE spaced: expected str, got {type(pred).__name__}")
    tokens = pred.strip().split()
    if len(tokens) % 2 != 0:
        raise ValueError(f"RLE spaced: odd number of tokens in '{pred[:60]}'")
    try:
        for i in range(1, len(tokens), 2):
            int(tokens[i])
    except Exception as e:
        raise ValueError(f"RLE spaced: count token not an int in '{pred[:60]}': {e}")


def _validate_ae_fraction(pred, expected):
    """pred: str like '28/29' or 'Fraction(28, 29)', expected: str (same format)."""
    if not isinstance(pred, str):
        raise TypeError(f"AE fraction: expected str, got {type(pred).__name__}")
    from decimal import InvalidOperation
    try:
        _parse_ae_decimal(pred.strip())
    except InvalidOperation as e:
        raise ValueError(f"AE fraction: could not parse as fraction/decimal '{pred[:60]}': {e}")


VARIANT_VALIDATORS = {
    ("huffman", "base64"):         _validate_huffman_base64,
    ("huffman", "hex"):            _validate_huffman_hex,
    ("huffman", "spaced_decimal"): _validate_huffman_spaced_decimal,
    ("huffman", "char_hex"):       _validate_huffman_char_hex,
    ("huffman", "binary"):         _validate_huffman_binary,
    ("lzw",     "csv"):            _validate_lzw_csv,
    ("lzw",     "spaced"):         _validate_lzw_spaced,
    ("rle",     "compact"):        _validate_rle_compact,
    ("rle",     "spaced"):         _validate_rle_spaced,
    ("ae",      "fraction"):       _validate_ae_fraction,
}


# ─────────────────────────────────────────────────────────────────────────────
# Value comparison helpers
# ─────────────────────────────────────────────────────────────────────────────

def _values_equal_alt(pred: str, expected_alt: str, algorithm: str, variant: str) -> bool:
    """
    Compare predicted string against the alternative ground truth string.

    For huffman variants we decode both sides to byte lists so that equivalent
    representations (e.g. upper/lower hex) are treated as equal.
    For lzw/csv and rle/compact we compare the canonical forms.
    """
    pred = pred.strip()
    expected_alt = expected_alt.strip()

    if algorithm == "huffman" and variant == "base64":
        try:
            return list(base64.b64decode(pred)) == list(base64.b64decode(expected_alt))
        except Exception:
            return False

    if algorithm == "huffman" and variant == "hex":
        try:
            return bytes.fromhex(pred.lower()) == bytes.fromhex(expected_alt.lower())
        except Exception:
            return False

    if algorithm == "huffman" and variant == "spaced_decimal":
        try:
            return [int(x) for x in pred.split()] == [int(x) for x in expected_alt.split()]
        except Exception:
            return False

    if algorithm == "huffman" and variant == "char_hex":
        # Compare byte sequences decoded from space-separated hex chars
        try:
            def _char_hex_to_bytes(s):
                chars = s.strip().split()
                hex_str = "".join(chars)
                return bytes.fromhex(hex_str)
            return _char_hex_to_bytes(pred) == _char_hex_to_bytes(expected_alt)
        except Exception:
            return False

    if algorithm == "huffman" and variant == "binary":
        try:
            def _binary_to_bytes(s):
                return bytes(int(tok, 2) for tok in s.strip().split())
            return _binary_to_bytes(pred) == _binary_to_bytes(expected_alt)
        except Exception:
            return False

    if algorithm == "lzw" and variant == "csv":
        try:
            pred_ints = [int(x.strip()) for x in pred.split(",")]
            exp_ints  = [int(x.strip()) for x in expected_alt.split(",")]
            return pred_ints == exp_ints
        except Exception:
            return False

    if algorithm == "lzw" and variant == "spaced":
        try:
            pred_ints = [int(x.strip()) for x in pred.split()]
            exp_ints  = [int(x.strip()) for x in expected_alt.split()]
            return pred_ints == exp_ints
        except Exception:
            return False

    if algorithm == "rle" and variant == "compact":
        def _parse_compact(s):
            return [(c, int(n)) for c, n in re.findall(r"([A-Za-z .])(\d+)", s)]
        try:
            return _parse_compact(pred) == _parse_compact(expected_alt)
        except Exception:
            return False

    if algorithm == "rle" and variant == "spaced":
        try:
            def _parse_spaced(s):
                tokens = s.strip().split()
                return [(tokens[i], int(tokens[i + 1])) for i in range(0, len(tokens), 2)]
            return _parse_spaced(pred) == _parse_spaced(expected_alt)
        except Exception:
            return False

    if algorithm == "ae" and variant == "fraction":
        # Both sides are fraction/decimal strings; use Decimal-based comparison
        # with the same tolerance as the main evaluator.
        try:
            return _ae_equal(pred, expected_alt, float_tol=1e-3)
        except Exception:
            return False

    # Fallback: string equality
    return pred == expected_alt


# ─────────────────────────────────────────────────────────────────────────────
# Core check
# ─────────────────────────────────────────────────────────────────────────────

def check_pred_alt(item, algo: str, variant: str):
    """
    Check a single prediction item for the ablation study.

    Uses item["output_alt"] as ground truth (pre-converted alternative format).
    """
    def resp(status, message, actual=None, predicted=None):
        a = _json_safe(actual)
        p = _json_safe(predicted)
        return {
            "status":    status,
            "message":   message,
            "actual":    json.dumps(a, ensure_ascii=False) if a is not None else None,
            "predicted": json.dumps(p, ensure_ascii=False) if p is not None else None,
        }

    io_pred = item.get("io_pred", "")
    is_input_task = io_pred.startswith("input_")

    out_text = item.get("output", "")
    last_json = extract_last_complete_json(out_text)
    if last_json is None:
        return resp("no answer", "Failed to extract a complete JSON from model output.",
                    actual=item.get("input") if is_input_task else item.get("output_alt"),
                    predicted="no answer")
    if not isinstance(last_json, dict):
        return resp("no answer", "The last JSON is not an object (dict).",
                    actual=item.get("input") if is_input_task else item.get("output_alt"),
                    predicted="no answer")
    if "output" not in last_json:
        return resp("no answer", "No 'output' field in the last JSON.",
                    actual=item.get("input") if is_input_task else item.get("output_alt"),
                    predicted="no answer")

    pred = last_json["output"]

    # ── Input prediction tasks: expected is the original uncompressed string ──
    if is_input_task:
        expected = item.get("input")
        if expected is None:
            return resp("no answer", "Item has no 'input' field.", actual=None, predicted=pred)
        if not isinstance(pred, str):
            pred = str(pred)
        if pred.strip() == expected:
            return resp("correct", f"Correct {io_pred}!", actual=expected, predicted=pred)
        return resp("wrong", f"[Mismatch] {io_pred} is not correct!",
                    actual=expected, predicted=pred)

    # ── Output prediction tasks: expected is output_alt (the alt-format encoding) ──
    expected_alt = item.get("output_alt")
    if expected_alt is None:
        return resp("no answer", "Item has no 'output_alt' field (was it built with "
                    "build_tokenization_ablation.py?)")

    # Coerce to string if the model wrapped the value in a list/dict
    if not isinstance(pred, str):
        pred = str(pred)

    validator_key = (algo, variant)
    validator = VARIANT_VALIDATORS.get(validator_key)
    if validator is None:
        return resp("no answer", f"No validator for {algo}/{variant}",
                    actual=expected_alt, predicted=pred)

    try:
        validator(pred, expected_alt)
    except (TypeError, ValueError) as e:
        return resp("no answer", str(e), actual=expected_alt, predicted=pred)

    if _values_equal_alt(pred, expected_alt, algo, variant):
        return resp("correct", f"Correct {io_pred}!", actual=expected_alt, predicted=pred)
    return resp("wrong", f"[Mismatch] {io_pred} is not correct!",
                actual=expected_alt, predicted=pred)


# ─────────────────────────────────────────────────────────────────────────────
# Batching / IO
# ─────────────────────────────────────────────────────────────────────────────

def batcher(it, batch_size):
    it = iter(it)
    while True:
        chunk = list(islice(it, batch_size))
        if not chunk:
            break
        yield chunk


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify tokenization-ablation predictions."
    )
    parser.add_argument("--parsed_file_name", required=True,
                        help="Original data.jsonl (used only for item count / metadata).")
    parser.add_argument("--pred_file_name",   required=True,
                        help="Inference output JSONL from batched_api_inference.py.")
    parser.add_argument("--res_file_name",    required=True,
                        help="Output verified JSONL path.")
    parser.add_argument("--algo",    required=True, choices=["huffman", "lzw", "rle", "ae"])
    parser.add_argument("--variant", required=True,
                        choices=["base64", "hex", "csv", "compact",
                                 "spaced_decimal", "char_hex", "binary", "spaced", "fraction"])
    parser.add_argument("--batchsize",  type=int,   default=10000)
    parser.add_argument("--float_tol",  type=float, default=1e-3,
                        help="Unused (kept for CLI compatibility with check_io_pred_acc_mp.py).")
    args = parser.parse_args()

    combo = (args.algo, args.variant)
    if combo not in VARIANT_VALIDATORS:
        parser.error(
            f"Invalid combination: --algo {args.algo} --variant {args.variant}. "
            f"Valid: {sorted(VARIANT_VALIDATORS)}"
        )

    # ── Resume support ───────────────────────────────────────────────────────
    # Seed both done_keys AND csv_rows from the existing verified JSONL so
    # the final CSV is always complete across multiple runs.
    done_keys: set = set()
    csv_rows: list = []
    if os.path.exists(args.res_file_name):
        for line in load_jsonl_yield(args.res_file_name):
            key = (
                line.get("itemid"),
                line.get("ioid"),
                line.get("io_pred"),
                line.get("output_index"),
            )
            done_keys.add(key)
            res = line.get("res", {})
            csv_rows.append({
                "model":       line.get("model"),
                "temperature": line.get("temperature"),
                "io_pred":     line.get("io_pred"),
                "algo":        args.algo,
                "variant":     args.variant,
                "input":       line.get("input"),
                "category":    line.get("category"),
                "status":      res.get("status"),
                "actual":      res.get("actual"),
                "predicted":   res.get("predicted"),
                "message":     res.get("message"),
            })
        print(f"Resuming: {len(done_keys)} results already in {args.res_file_name}")

    dt = load_jsonl_yield(args.pred_file_name)

    batch_idx = 0
    for batch in batcher(dt, args.batchsize):
        batch = [
            item for item in batch
            if (
                item.get("itemid"),
                item.get("ioid"),
                item.get("io_pred"),
                item.get("output_index"),
            ) not in done_keys
        ]
        if not batch:
            batch_idx += 1
            continue

        stats = defaultdict(int)
        for item in batch:
            res = check_pred_alt(item, args.algo, args.variant)
            item["res"] = res
            stats[res["status"]] += 1
            csv_rows.append({
                "model":       item.get("model"),
                "temperature": item.get("temperature"),
                "io_pred":     item.get("io_pred"),
                "algo":        args.algo,
                "variant":     args.variant,
                "input":       item.get("input"),
                "category":    item.get("category"),
                "status":      res["status"],
                "actual":      res["actual"],
                "predicted":   res["predicted"],
                "message":     res["message"],
            })
        write_jsonl(batch, args.res_file_name, mode="a")
        print(f"Batch {batch_idx}: {dict(stats)}")
        batch_idx += 1

    csv_path = args.res_file_name.replace(".jsonl", ".csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Results written to {args.res_file_name} and {csv_path}")


if __name__ == "__main__":
    main()
