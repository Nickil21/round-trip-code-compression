import os
import re
import sys
import json
import ast
import math
import pandas as pd
from decimal import Decimal, InvalidOperation
from itertools import islice
from collections import defaultdict

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

from core.utils import read_jsonl, load_jsonl_yield, write_jsonl



# =====================================================================
# JSON safety
# =====================================================================

def _json_safe(obj):
    """Recursively convert non-JSON-serialisable values (Ellipsis, NaN, Inf) to None."""
    if obj is Ellipsis:
        return None
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    return obj

def _sanitize_text_for_csv(x):
    """Convert potentially invalid Unicode (e.g., lone surrogates) into safe escaped text."""
    if not isinstance(x, str):
        return x
    return x.encode("utf-8", errors="backslashreplace").decode("utf-8")


# =====================================================================
# Robust JSON extractors
# =====================================================================

_ANSWER_BLOCK_RE = re.compile(r'\[ANSWER\](.*?)\[/ANSWER\]', re.DOTALL | re.IGNORECASE)
_DOUBLE_WRAP_RE  = re.compile(r'^\s*\{\{(.*)\}\}\s*$', re.DOTALL)
_OUTPUT_KEY_RE   = re.compile(r'["\']output["\']\s*:\s*', re.IGNORECASE)


def _replace_python_tokens_outside_strings(s: str) -> str:
    """
    Replace bare Python literals (None/True/False) only when they appear outside
    quoted string literals.
    """
    if not s:
        return s

    n = len(s)
    i = 0
    out = []
    in_str = False
    quote = ""
    esc = False

    def _is_word_char(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    repl = (("None", "null"), ("True", "true"), ("False", "false"))

    while i < n:
        ch = s[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            i += 1
            continue

        if ch in ('"', "'"):
            in_str = True
            quote = ch
            out.append(ch)
            i += 1
            continue

        replaced = False
        for src, dst in repl:
            if s.startswith(src, i):
                prev = s[i - 1] if i > 0 else ""
                nxt_idx = i + len(src)
                nxt = s[nxt_idx] if nxt_idx < n else ""
                if not _is_word_char(prev) and not _is_word_char(nxt):
                    out.append(dst)
                    i += len(src)
                    replaced = True
                    break
        if replaced:
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _repair_jsonish(s: str) -> str:
    if not s:
        return s
    # smart quotes → plain ASCII
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    # Python tokens → JSON equivalents.
    # Use word-boundary anchoring so we don't corrupt substrings like "NoneType", "Trueish", etc.
    # Match Python bare tokens that appear outside quoted strings (best-effort, not perfect).
    s = _replace_python_tokens_outside_strings(s)
    # trailing commas before } or ]
    s = re.sub(r',\s*([}\]])', r'\1', s)
    return s


def _parse_jsonish(snippet: str):
    try:
        return json.loads(snippet)
    except Exception:
        pass
    try:
        return json.loads(_repair_jsonish(snippet))
    except Exception:
        pass
    try:
        lit = ast.literal_eval(snippet)
        return json.loads(json.dumps(lit))
    except Exception:
        return None


def _collapse_double_braces(snippet: str) -> str:
    m = _DOUBLE_WRAP_RE.match(snippet or "")
    return '{' + m.group(1) + '}' if m else snippet


def _find_balanced_dict_spans(text: str) -> list:
    """
    Return (start, end) for every top-level balanced {...} in text,
    correctly skipping braces inside quoted string literals.
    Replaces the previous shallow r'\\{[^{}]*\\}' regex which could not
    match dicts whose values contain nested braces.
    """
    spans = []
    depth = 0
    start = None
    in_str = False
    str_ch = None
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
                continue
            if ch == '\\':
                esc = True
                continue
            if ch == str_ch:
                in_str = False
            continue
        # Always treat double-quotes as string delimiters. For single-quotes,
        # only treat them as delimiters once we are already inside a {...} span.
        # This avoids apostrophes in surrounding prose (e.g. "don't") from
        # swallowing later JSON, while still supporting Python-style dicts.
        if ch == '"' or (ch == "'" and depth > 0):
            in_str = True
            str_ch = ch
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append((start, i + 1))
                    start = None
    return spans


def _find_last_output_dict(text: str):
    """
    Return the last parsed dict that has an 'output' key (case-insensitive).
    Scans right-to-left over properly nested {…} spans.
    """
    if not text:
        return None
    spans = _find_balanced_dict_spans(text)
    for start, end in reversed(spans):
        frag = _collapse_double_braces(text[start:end])
        obj = _parse_jsonish(frag)
        if isinstance(obj, dict):
            if 'output' in obj:
                return obj
            for k in list(obj.keys()):
                if isinstance(k, str) and k.lower() == 'output':
                    obj['output'] = obj.pop(k)
                    return obj
    return None


def _answer_candidates(text: str) -> list:
    """Return [last ANSWER block (if any), full text] as candidate strings."""
    blocks = _ANSWER_BLOCK_RE.findall(text)
    candidates = []
    if blocks:
        candidates.append(blocks[-1].strip())
    candidates.append(text.strip())
    return candidates


def _extract_fallback_output_obj(text: str):
    """
    Fallback parser:
      - parse last [ANSWER] block as JSON-ish value, or
      - parse whole text as JSON-ish value.
    Returns {"output": value} on success, else None.
    """
    if not text:
        return None

    wrapped_fallback = None
    for cand in _answer_candidates(text):
        cand = _collapse_double_braces(cand)
        obj = _parse_jsonish(cand)
        if obj is None:
            continue
        if isinstance(obj, dict) and "output" in obj:
            return obj
        if wrapped_fallback is None:
            wrapped_fallback = {"output": obj}
    return wrapped_fallback


def _extract_truncated_output_obj(text: str):
    """
    Last-resort salvage for truncated answers that started writing:
      {"output": ...
    but did not close quotes/braces before truncation.
    """
    if not text:
        return None

    for cand in _answer_candidates(text):
        matches = list(_OUTPUT_KEY_RE.finditer(cand))
        if not matches:
            continue
        # Prefer the last "output": occurrence in the candidate.
        start = matches[-1].end()
        tail = cand[start:].lstrip()
        if not tail:
            continue

        # Try to parse as a regular JSON-ish scalar first.
        parsed = _parse_jsonish(tail)
        if parsed is not None:
            return {"output": parsed}

        # If the value starts as a quoted string but is truncated, recover as raw text.
        q = tail[0]
        if q in ('"', "'"):
            body = tail[1:]
            esc = False
            closed_at = None
            for i, ch in enumerate(body):
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == q:
                    closed_at = i
                    break
            if closed_at is not None:
                body = body[:closed_at]
            # Trim common trailing delimiters if closing quote/brace was never produced.
            body = re.split(r'\n\[/ANSWER\]|\n```|,\s*["\']\w+["\']\s*:\s*', body, maxsplit=1)[0]
            body = body.rstrip()
            if body:
                return {"output": body}

    return None


def sub_extract_last_complete_json(s: str):
    if not s:
        return None
    # 1) Prefer the last [ANSWER] … [/ANSWER] block
    blocks = _ANSWER_BLOCK_RE.findall(s)
    if blocks:
        blk = _collapse_double_braces(blocks[-1].strip())
        obj = _find_last_output_dict(blk)
        if obj is not None:
            return obj
    # 2) Search the whole text
    return _find_last_output_dict(_collapse_double_braces(s))


def extract_last_complete_json(s: str):
    obj = sub_extract_last_complete_json(s)
    if obj is not None:
        return obj
    obj = _extract_fallback_output_obj(s)
    if obj is not None:
        return obj
    return _extract_truncated_output_obj(s)


# =====================================================================
# Evaluation helpers & validators
# =====================================================================

ALGO_NAMES = {"lzw", "huffman", "rle", "ae"}

# sys.float_info.dig: number of significant decimal digits Python float reliably carries (15)
_FLOAT_SIG_DIGITS = sys.float_info.dig


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


_FRACTION_RE = re.compile(r'^\s*(-?\d+)\s*/\s*(-?\d+)\s*$')
_FRACTION_REPR_RE = re.compile(r'^\s*Fraction\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*$')


def _parse_ae_decimal(s: str) -> Decimal:
    """
    Parse a string that represents an AE output value.  Accepts:
      - Plain decimal literals      e.g. "0.6319081203782178"
      - Fraction notation           e.g. "28/29"
      - Python Fraction repr        e.g. "Fraction(0, 29)"
    Raises InvalidOperation if none of the formats match.
    """
    s = s.strip()
    # Plain decimal
    try:
        return Decimal(s)
    except InvalidOperation:
        pass
    # "a/b"
    m = _FRACTION_RE.match(s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den == 0:
            raise InvalidOperation("zero denominator")
        return Decimal(num) / Decimal(den)
    # "Fraction(a, b)"
    m = _FRACTION_REPR_RE.match(s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den == 0:
            raise InvalidOperation("zero denominator")
        return Decimal(num) / Decimal(den)
    raise InvalidOperation(f"unrecognised AE value format: {s!r}")


def _ae_equal(a, b, float_tol: float = 0.0) -> bool:
    """
    Compare two AE float values using Decimal arithmetic to avoid float64 precision
    loss.  Accepts str, int, or float on either side; str may be a plain decimal,
    a fraction ("28/29"), or a Python Fraction repr ("Fraction(28, 29)").

    float_tol > 0: use that as the relative (and absolute) tolerance, mirroring
                   the behaviour of deep_equal / math.isclose.
    float_tol == 0 (default): fall back to ~15-significant-digit precision
                   (sys.float_info.dig), i.e. bit-exact Python-float comparison.
    """
    try:
        da = _parse_ae_decimal(str(a))
        db = _parse_ae_decimal(str(b))
    except InvalidOperation:
        return str(a).strip() == str(b).strip()
    if da.is_nan() or db.is_nan():
        return str(a).strip() == str(b).strip()
    if da == db:
        return True
    scale = max(abs(da), abs(db))
    if scale == 0:
        return True
    if float_tol > 0:
        tol = max(Decimal(str(float_tol)) * scale, Decimal(str(float_tol)))
    else:
        tol = scale * Decimal(10) ** (-_FLOAT_SIG_DIGITS)
    return abs(da - db) <= tol


def deep_equal(a, b, float_tol=1e-3):
    a = normalize(maybe_literal_eval(a))
    b = normalize(maybe_literal_eval(b))

    # Keep integer-valued outputs exact; tolerance is only for genuine floats.
    if isinstance(a, int) and isinstance(b, int):
        return a == b
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


# --- shape validators ---

def _ensure_list_of_ints(x, ctx):
    if not isinstance(x, list):
        raise TypeError(f"{ctx}: expected list[int], got {type(x).__name__}")
    # Coerce integer-valued floats (e.g. 256.0 → 256) that arise when a model
    # writes JSON like [256.0, 257.0] instead of [256, 257].
    for i, t in enumerate(x):
        if isinstance(t, float) and t.is_integer():
            x[i] = int(t)
        elif not isinstance(t, int):
            raise TypeError(f"{ctx}: element {i} is not an int (got {type(t).__name__})")

def _ensure_list_of_pairs(x, ctx):
    if not isinstance(x, list):
        raise TypeError(f"{ctx}: expected list[2-tuple/list], got {type(x).__name__}")
    for i, el in enumerate(x):
        if not isinstance(el, (list, tuple)) or len(el) != 2:
            raise TypeError(f"{ctx}: element {i} is not a 2-item list/tuple")
        ch, cnt = el
        if not isinstance(ch, str):
            raise TypeError(f"{ctx}: element {i} first item must be str")
        # Coerce integer-valued floats (e.g. 8.0 → 8) that arise from JSON parsing.
        if isinstance(cnt, float) and cnt.is_integer():
            cnt = int(cnt)
            if isinstance(el, list):
                el[1] = cnt
            else:
                # tuples are immutable; wrap to list so downstream code can iterate
                el = [ch, cnt]
                x[i] = el
        if not isinstance(cnt, int):
            raise TypeError(f"{ctx}: element {i} second item must be int")
        if cnt <= 0:
            raise TypeError(f"{ctx}: element {i} second item must be positive int")

def _validate_lzw(pred, expected):
    _ensure_list_of_ints(pred, "LZW output")

def _validate_rle(pred, expected):
    _ensure_list_of_pairs(pred, "RLE output")

def _validate_ae(pred, expected):
    # Accept number or string.  Strings that parse as plain decimals,
    # fractions ("a/b"), or Fraction reprs ("Fraction(a,b)") will be
    # compared numerically in _ae_equal.  Strings that do NOT parse
    # (e.g. the model returned the input string instead of the float)
    # are allowed through here and will compare as "wrong" rather than
    # the less informative "no answer".
    if not isinstance(pred, (int, float, str)):
        raise TypeError(f"AE output: expected number or string, got {type(pred).__name__}")

def _validate_huffman(pred, expected):
    if isinstance(expected, list) and all(isinstance(t, int) for t in expected):
        _ensure_list_of_ints(pred, "Huffman output (flat)")
        return
    if (isinstance(expected, list) and len(expected) >= 1
            and isinstance(expected[0], list)
            and all(isinstance(t, int) for t in expected[0])):
        if not isinstance(pred, list) or len(pred) < 1:
            raise TypeError("Huffman output (tupled): expected list with first element list[int]")
        _ensure_list_of_ints(pred[0], "Huffman output first element")
        return
    if not isinstance(pred, type(expected)):
        raise TypeError(f"Huffman output: expected top-level type {type(expected).__name__}, "
                        f"got {type(pred).__name__}")

VALIDATORS = {
    "lzw":     _validate_lzw,
    "rle":     _validate_rle,
    "ae":      _validate_ae,
    "huffman": _validate_huffman,
}


# --- ground-truth type validators ---
# These mirror VALIDATORS but check the *expected* value from the dataset,
# not the model prediction.  Called early in check_pred to catch dataset
# parsing issues before they produce confusing comparison failures.

def _validate_expected_lzw(x):
    _ensure_list_of_ints(x, "LZW ground-truth")

def _validate_expected_rle(x):
    _ensure_list_of_pairs(x, "RLE ground-truth")

def _validate_expected_ae(x):
    # Ground-truth AE output is always a plain number, unlike the model prediction
    # which may arrive as a string representation.
    if not isinstance(x, (int, float)):
        raise TypeError(f"AE ground-truth: expected number, got {type(x).__name__}")

def _validate_expected_huffman(x):
    if not isinstance(x, list):
        raise TypeError(f"Huffman ground-truth: expected list, got {type(x).__name__}")

EXPECTED_VALIDATORS = {
    "lzw":     _validate_expected_lzw,
    "rle":     _validate_expected_rle,
    "ae":      _validate_expected_ae,
    "huffman": _validate_expected_huffman,
}


# =====================================================================
# Core check
# =====================================================================

def _unwrap_dict(d):
    """Return the string value from a single-key or well-known-key dict, else None."""
    if not isinstance(d, dict):
        return None
    for key in ("return", "input", "output"):
        if key in d:
            return d[key]
    # Fallback: if the model used the function's own parameter name as the key
    # (e.g. {"compressed": "..."} or {"uncompressed": "..."}), and there is
    # exactly one entry, accept whatever value it holds.
    if len(d) == 1:
        return next(iter(d.values()))
    return None


def check_pred(item, algo, parsed_ios, *, float_tol=1e-3):
    def resp(status, message, actual=None, predicted=None):
        a = _json_safe(actual)
        p = _json_safe(predicted)
        return {
            "status":    status,
            "message":   _sanitize_text_for_csv(message),
            # Use ASCII escapes so lone surrogates from model output do not crash CSV writing.
            "actual":    json.dumps(a, ensure_ascii=True) if a is not None else None,
            "predicted": json.dumps(p, ensure_ascii=True) if p is not None else None,
        }

    io_pred_raw = item.get("io_pred", "")
    io_pred = io_pred_raw if isinstance(io_pred_raw, str) else str(io_pred_raw or "")
    is_input_task = io_pred.startswith("input_")

    itemid = item.get("itemid")
    ioid = item.get("ioid")
    try:
        ori = parsed_ios[itemid]["ios"][ioid]
    except Exception as e:
        return resp("no answer", f"Invalid item reference (itemid/ioid): {e}",
                    actual=None, predicted=item.get("output"))
    expected = ori["input"] if is_input_task else ori["output"]

    # Validate the ground-truth type early so dataset issues produce a clear message
    # rather than a confusing type error deep inside the comparison logic.
    if is_input_task:
        if not isinstance(expected, str):
            return resp("no answer", "Ground-truth input is not a string (dataset issue?)",
                        actual=expected, predicted=None)
    elif algo in EXPECTED_VALIDATORS:
        try:
            EXPECTED_VALIDATORS[algo](expected)
        except TypeError as e:
            return resp("no answer", f"Ground-truth output has wrong type (dataset issue?): {e}",
                        actual=expected, predicted=None)

    out_text = item.get("output", "")
    last_json = extract_last_complete_json(out_text)
    if last_json is None:
        return resp("no answer", "Failed to extract a complete JSON from model output!",
                    actual=expected, predicted="no answer")
    if not isinstance(last_json, dict):
        return resp("no answer", "The last JSON is not an object (dict)!",
                    actual=expected, predicted="no answer")
    if "output" not in last_json:
        return resp("no answer", "No 'output' field in the last JSON",
                    actual=expected, predicted="no answer")

    pred = last_json["output"]

    if is_input_task:
        if isinstance(pred, dict):
            unwrapped = _unwrap_dict(pred)
            if unwrapped is not None:
                pred = unwrapped
        # If the model stringified a dict/list, try one more parse + unwrap.
        if isinstance(pred, str):
            try:
                parsed_pred = json.loads(pred)
            except json.JSONDecodeError:
                try:
                    parsed_pred = ast.literal_eval(pred)
                except Exception:
                    parsed_pred = None
            if isinstance(parsed_pred, dict):
                unwrapped = _unwrap_dict(parsed_pred)
                if unwrapped is not None:
                    pred = unwrapped
        if not isinstance(pred, str):
            # Model returned a number (e.g. the compressed float) instead of the
            # decoded string.  Coerce to a string so the result is "wrong" rather
            # than the less informative "no answer".
            pred = str(pred)
    else:
        # Coerce stringified structures when possible
        if isinstance(pred, str):
            try:
                pred = json.loads(pred)
            except json.JSONDecodeError:
                try:
                    pred = ast.literal_eval(pred)
                except Exception:
                    pass
        if algo not in VALIDATORS:
            return resp("no answer", f"Unknown algo '{algo}' for validation",
                        actual=expected, predicted=pred)
        try:
            VALIDATORS[algo](pred, expected)
        except TypeError as e:
            # The model produced a value but with the wrong type/shape.
            # Only use "no answer" when pred is None (model gave nothing parseable);
            # otherwise use "wrong" so the distinction between "gave up" and
            # "tried but wrong type" is preserved in the results.
            if pred is not None:
                return resp("wrong", str(e), actual=expected, predicted=pred)
            return resp("no answer", str(e), actual=expected, predicted=pred)

    # ---- value comparison ----
    # AE uses Decimal-based comparison to avoid float64 precision loss
    if not is_input_task and algo == "ae":
        correct = _ae_equal(pred, expected, float_tol=float_tol)
    else:
        correct = deep_equal(pred, expected, float_tol=float_tol)

    if correct:
        return resp("correct", f"Correct {io_pred}!", actual=expected, predicted=pred)

    # ---- AE input semantic fallback: strip spurious trailing "EOF" ----
    # The AE decoding algorithm uses "EOF" as a sentinel that is never included in
    # the returned string.  Some models trace the loop correctly but accidentally
    # append the sentinel before returning, producing e.g. "UUUUUUUUEOF" instead
    # of "UUUUUUUU".  Accept such predictions if stripping the suffix is correct.
    if is_input_task and algo == "ae" and isinstance(pred, str) and pred.endswith("EOF"):
        try:
            stripped = pred[:-3]  # remove trailing "EOF"
            if deep_equal(stripped, expected, float_tol=float_tol):
                return resp("correct",
                            f"Correct {io_pred} (semantic: spurious trailing EOF stripped)!",
                            actual=expected, predicted=pred)
        except Exception:
            pass

    # ---- RLE semantic fallback ----
    # The canonical RLE function always merges consecutive identical characters,
    # but a non-canonical prediction (e.g. [["s",1],["s",1]] instead of [["s",2]])
    # is semantically valid if it round-trips to the original uncompressed string.
    if not is_input_task and algo == "rle":
        try:
            original_input = ori["input"]
            decoded = "".join(c * n for c, n in pred)
            if decoded == original_input:
                return resp("correct",
                            f"Correct {io_pred} (semantic: non-canonical RLE decodes correctly)!",
                            actual=expected, predicted=pred)
        except Exception:
            pass

    return resp("wrong", f"[Mismatch] {io_pred} is not correct!", actual=expected, predicted=pred)


# =====================================================================
# Batching / IO
# =====================================================================

def batcher(it, batch_size):
    it = iter(it)
    while True:
        chunk = list(islice(it, batch_size))
        if not chunk:
            break
        yield chunk


def _row_key(line):
    # Always return a 4-tuple so keys from the existing verified JSONL and
    # from the pred file have the same arity and can match in done_keys.
    # A mixed 4-tuple / 5-tuple scheme would mean they could never collide.
    out_idx = line.get("output_index")
    if out_idx is None:
        out_idx = line.get("index")
    if out_idx is None:
        # Content-based fallback for legacy files without output_index.
        out_sig = str(line.get("output", ""))[:80]
        in_sig  = str(line.get("input",  ""))[:80]
        out_idx = f"__sig__{in_sig}__{out_sig}"
    return (line.get("itemid"), line.get("ioid"), line.get("io_pred"), out_idx)


def _build_csv_row(item, res, algo):
    return {
        "model":       item.get("model"),
        "temperature": item.get("temperature"),
        "io_pred":     item.get("io_pred"),
        "algo":        algo,
        "input":       item.get("input"),
        "category":    item.get("category"),
        "status":      res.get("status"),
        "actual":      res.get("actual"),
        "predicted":   res.get("predicted"),
        "message":     res.get("message"),
    }


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

    # Key by line index (== itemid assigned in build_codeio_msg.py); data.jsonl has no itemid field.
    parsed_ios = {idx: item for idx, item in enumerate(read_jsonl(args.parsed_file_name))}

    # Load already-written results so reruns skip completed items.
    # Also seed csv_rows from the existing verified JSONL so the final CSV
    # is always complete (previous runs' rows are not lost on resume).
    done_keys: set = set()
    csv_rows: list = []

    if os.path.exists(args.res_file_name):
        for line in load_jsonl_yield(args.res_file_name):
            done_keys.add(_row_key(line))
            csv_rows.append(_build_csv_row(line, line.get("res", {}), args.algo))
        print(f"Resuming: {len(done_keys)} results already in {args.res_file_name}")

    batch_idx = 0
    for batch in batcher(load_jsonl_yield(args.pred_file_name), args.batchsize):
        batch = [item for item in batch if _row_key(item) not in done_keys]
        if not batch:
            batch_idx += 1
            continue

        stats = defaultdict(int)
        for item in batch:
            res = check_pred(item, args.algo, parsed_ios, float_tol=args.float_tol)
            item["res"] = res
            stats[res["status"]] += 1
            done_keys.add(_row_key(item))
            csv_rows.append(_build_csv_row(item, res, args.algo))
        write_jsonl(batch, args.res_file_name, mode="a")
        print(f"Batch {batch_idx}: {stats}")
        batch_idx += 1

    csv_path = os.path.splitext(args.res_file_name)[0] + ".csv"
    pd.DataFrame(csv_rows).to_csv(
        csv_path,
        index=False,
        encoding="utf-8",
        errors="backslashreplace",
    )


if __name__ == "__main__":
    main()
