import os
import re
import json
import ast
import math
import pandas as pd
from itertools import islice
from collections import defaultdict

from codeio_utils import *
from utils import *

# =====================================================================
# Robust JSON extractors (override any imported ones)
# Strategy: explicitly recover the LAST dict that contains an "output" key.
# Prefers the last [ANSWER] ... [/ANSWER] block; tolerates {{...}} wrappers,
# smart quotes, Python literals (tuples, single quotes, True/False/None),
# and trailing commas.
# =====================================================================

_ANSWER_BLOCK_RE = re.compile(r'\[ANSWER\](.*?)\[/ANSWER\]', re.DOTALL | re.IGNORECASE)
_DICT_SPAN_RE    = re.compile(r'\{[^{}]*\}')      # quick spans; we still validate/parse
_DOUBLE_WRAP_RE  = re.compile(r'^\s*\{\{(.*)\}\}\s*$', re.DOTALL)

def _repair_jsonish(s: str) -> str:
    if not s:
        return s
    # smart quotes → plain
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # Python tokens → JSON
    s = s.replace(": None", ": null").replace(" None", " null")
    s = s.replace(": True", ": true").replace(" True", " true")
    s = s.replace(": False", ": false").replace(" False", " false")
    # trailing commas ,} ,]
    s = re.sub(r',\s*([}\]])', r'\1', s)
    return s

def _parse_jsonish(snippet: str):
    # 1) Try JSON as-is
    try:
        return json.loads(snippet)
    except Exception:
        pass
    # 2) Try minimally repaired JSON
    try:
        return json.loads(_repair_jsonish(snippet))
    except Exception:
        pass
    # 3) Fall back to Python literal, then coerce to JSON types
    try:
        lit = ast.literal_eval(snippet)
        return json.loads(json.dumps(lit))
    except Exception:
        return None

def _collapse_double_braces(snippet: str) -> str:
    m = _DOUBLE_WRAP_RE.match(snippet or "")
    return m.group(1) if m else snippet

def _find_last_output_dict(text: str):
    """
    Return the last parsed dict that has an 'output' key (case-insensitive).
    Scans right-to-left over shallow { ... } spans; each candidate is parsed
    with tolerant parsing and accepted only if it's a dict with output.
    """
    if not text:
        return None
    spans = list(_DICT_SPAN_RE.finditer(text))
    for m in reversed(spans):
        frag = text[m.start():m.end()]
        frag = _collapse_double_braces(frag)
        obj = _parse_jsonish(frag)
        if isinstance(obj, dict):
            # case-insensitive 'output' normalization
            if 'output' in obj:
                return obj
            for k in list(obj.keys()):
                if isinstance(k, str) and k.lower() == 'output':
                    obj['output'] = obj.pop(k)
                    return obj
    return None

def sub_extract_last_complete_json(s: str):
    """
    Extract the *last* dict object that contains an 'output' key.
    Prefers the last [ANSWER] ... [/ANSWER] block; otherwise searches whole text.
    Returns the parsed Python dict, or None.
    """
    if not s:
        return None

    # 1) Prefer the last [ANSWER] ... [/ANSWER] block
    blocks = _ANSWER_BLOCK_RE.findall(s)
    if blocks:
        blk = _collapse_double_braces(blocks[-1].strip())
        obj = _find_last_output_dict(blk)
        if obj is not None:
            return obj

    # 2) Search the whole text
    s2 = _collapse_double_braces(s)
    obj = _find_last_output_dict(s2)
    if obj is not None:
        return obj

    # 3) Nothing found
    return None

def extract_last_complete_json(s: str):
    """Compatibility wrapper; keep the function name your code calls."""
    return sub_extract_last_complete_json(s)

# =====================================================================
# Evaluation helpers & validators
# =====================================================================

# minimal algo tags (kept for labeling & validator selection)
ALGO_NAMES = {"lzw", "huffman", "rle", "ae"}

def normalize(x):
    if isinstance(x, tuple): x = list(x)
    if isinstance(x, list):  return [normalize(v) for v in x]
    if isinstance(x, dict):  return {k: normalize(v) for k, v in x.items()}
    return x

def maybe_literal_eval(s):
    if isinstance(s, str):
        try:
            return ast.literal_eval(s)
        except Exception:
            return s
    return s

def deep_equal(a, b, float_tol=1e-3):
    a = normalize(maybe_literal_eval(a))
    b = normalize(maybe_literal_eval(b))

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=float_tol, abs_tol=float_tol)

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y, float_tol=float_tol) for x, y in zip(a, b))

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(deep_equal(a[k], b[k], float_tol=float_tol) for k in a)

    return a == b

# --- strict but lightweight validators (shape-only) ---
def _ensure_list_of_ints(x, ctx):
    if not isinstance(x, list) or not all(isinstance(t, int) for t in x):
        raise TypeError(f"{ctx}: expected list[int], got {type(x).__name__}{' with non-int elements' if isinstance(x, list) else ''}")

def _ensure_list_of_pairs(x, ctx):
    if not isinstance(x, list):
        raise TypeError(f"{ctx}: expected list[2-tuple/list], got {type(x).__name__}")
    for i, el in enumerate(x):
        if not isinstance(el, (list, tuple)) or len(el) != 2:
            raise TypeError(f"{ctx}: element {i} is not a 2-item list/tuple")

def _validate_lzw(pred, expected):
    _ensure_list_of_ints(pred, "LZW output")

def _validate_rle(pred, expected):
    _ensure_list_of_pairs(pred, "RLE output")

def _validate_ae(pred, expected):
    if not isinstance(pred, (int, float)):
        raise TypeError(f"AE output: expected number, got {type(pred).__name__}")

def _validate_huffman(pred, expected):
    # Adaptive to GT structure
    if isinstance(expected, list) and all(isinstance(t, int) for t in expected):
        _ensure_list_of_ints(pred, "Huffman output (flat)")
        return
    if isinstance(expected, list) and len(expected) >= 1 and isinstance(expected[0], list) and all(isinstance(t, int) for t in expected[0]):
        if not isinstance(pred, list) or len(pred) < 1:
            raise TypeError("Huffman output (tupled): expected list with first element list[int]")
        _ensure_list_of_ints(pred[0], "Huffman output first element")
        return
    if not isinstance(pred, type(expected)):
        raise TypeError(f"Huffman output: expected top-level type {type(expected).__name__}, got {type(pred).__name__}")

VALIDATORS = {
    "lzw":     _validate_lzw,
    "rle":     _validate_rle,
    "ae":      _validate_ae,
    "huffman": _validate_huffman,
}

# =====================================================================
# Core check
# =====================================================================

def check_pred(item, algo, *, float_tol=1e-3):
    def resp(status, message, actual=None, predicted=None):
        return {
            "status": status,
            "message": message,
            "actual": json.dumps(actual) if actual is not None else None,
            "predicted": json.dumps(predicted) if predicted is not None else None,
        }

    io_pred = item.get("io_pred", "")
    is_input_task = io_pred.startswith("input_")

    ori = parsed_ios[item["itemid"]]["ios"][item["ioid"]]
    expected = ori["input"] if is_input_task else ori["output"]

    # ---- extract model JSON and always read top-level "output" field ----
    out_text = item.get("output", "")
    last_json = extract_last_complete_json(out_text)
    if last_json is None:
        return resp("no answer", "Failed to extract a complete JSON from model output!", actual=expected, predicted="no answer")

    if not isinstance(last_json, dict):
        return resp("no answer", "The last JSON is not an object (dict)!", actual=expected, predicted="no answer")

    if "output" not in last_json:
        return resp("no answer", "No 'output' field in the last JSON", actual=expected, predicted="no answer")

    pred = last_json["output"]

    # ---- type/shape enforcement ----
    if is_input_task:
        # input_* tasks: predicted must be a string (since we always fetch from "output")
        if not isinstance(expected, str):
            return resp("no answer", "Ground-truth input is not a string (dataset issue?)", actual=expected, predicted=pred)
        if not isinstance(pred, str):
            return resp("no answer", "For input_* tasks, predicted 'output' must be a string", actual=expected, predicted=pred)
    else:
        # output_* tasks: coerce stringified structures when possible
        if isinstance(pred, str):
            # Try JSON then Python literal
            try:
                pred = json.loads(pred)
            except json.JSONDecodeError:
                try:
                    pred = ast.literal_eval(pred)
                except Exception:
                    pass
        # Validate shape according to algo
        if algo not in VALIDATORS:
            return resp("no answer", f"Unknown algo '{algo}' for validation", actual=expected, predicted=pred)
        try:
            VALIDATORS[algo](pred, expected)
        except TypeError as e:
            return resp("no answer", str(e), actual=expected, predicted=pred)

    # ---- value comparison ----
    correct = deep_equal(pred, expected, float_tol=float_tol)
    if correct:
        return resp("correct", f"Correct {io_pred}!", actual=expected, predicted=pred)
    return resp("wrong", f"[Mismatch] {io_pred} is not correct!", actual=expected, predicted=pred)

# =====================================================================
# batching / io
# =====================================================================

def batcher(it, batch_size):
    it = iter(it)
    while True:
        chunk = list(islice(it, batch_size))
        if not chunk:
            break
        yield chunk

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed_file_name", type=str, required=True)
    parser.add_argument("--pred_file_name",   type=str, required=True)
    parser.add_argument("--res_file_name",    type=str, required=True)
    parser.add_argument("--batchsize",        type=int, default=10000)
    parser.add_argument("--algo",             type=str, required=True, choices=sorted(ALGO_NAMES))
    parser.add_argument("--float_tol",        type=float, default=1e-3)
    args = parser.parse_args()

    global parsed_ios
    parsed_ios = read_jsonl(args.parsed_file_name)

    dt = load_jsonl_yield(args.pred_file_name)

    csv_rows, batch_idx = [], 0
    for batch in batcher(dt, args.batchsize):
        stats = defaultdict(int)
        for item in batch:
            res = check_pred(item, args.algo, float_tol=args.float_tol)
            item["res"] = res
            stats[res["status"]] += 1
            csv_rows.append({
                "model": item.get("model"),
                "temperature": item.get("temperature"),
                "io_pred": item.get("io_pred"),
                "algo": args.algo,
                "input": item.get("input"),
                "category": item.get("category"),
                "actual": res["actual"],
                "predicted": res["predicted"],
                "message": res["message"]  # message column
            })
        write_jsonl(batch, args.res_file_name, mode="a")
        print(f"Batch {batch_idx}: {stats}")
        batch_idx += 1

    pd.DataFrame(csv_rows).to_csv(args.res_file_name.replace(".jsonl", ".csv"), index=False)

if __name__ == "__main__":
    main()
