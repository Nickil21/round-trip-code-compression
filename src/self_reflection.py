#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reflect_only.py — Reflection-only improver (no regeneration), with per-round predictions.

What it does
- Runs critique→revise loops on existing model drafts (no full regeneration).
- Focuses on the last task block for context.
- Robustly extracts/repairs answers into canonical flat JSON: {"output": <scalar>}
  (Nested schemas are NOT used; any nested found is flattened.)
- Computes EM vs ground-truth (numbers/strings), supports early stop.
- Reports dataset-level stats: first round of resolution (0 = initial, 1..N) and unresolved.

Context safety
- Strict prompt budgeting for critique, revision, and format-repair.
- Tail-clamps long prompts; shrinks generation tokens to fit.
- Conservative chars-per-token budgeting (configurable).
- Aligns --model_ctx to --max_model_len so budgeting cannot overshoot the real limit.

CLI highlights
--reflection_rounds, --critique_style {A|B}
--gt_field, --gt_stop_on em
--on_mismatch {message|annotate|keep|force} (+ --gt_allow_leak)
--model_ctx, --max_model_len, --gen_tokens, --safety_margin, --truncate_hard_chars, --chars_per_token
--stats_path (writes aggregate stats JSON)
"""

import os
import re
import json
import math
import datetime as dt
from argparse import ArgumentParser
from typing import Optional, Tuple
from collections import defaultdict

# Optional deps for vLLM path (safe import)
try:
    import torch  # noqa: F401
    from vllm import LLM, SamplingParams
except Exception:
    torch = None
    LLM = None
    SamplingParams = None

client = None  # OpenAI client (lazy init)

# =========================
# Precompiled regex (speed)
# =========================
RE_ANSWER_TAGS = re.compile(r"\[ANSWER\](.*?)\[/ANSWER\]", flags=re.S | re.I)
RE_CODE_FENCE = re.compile(r"```.*?```", flags=re.S)
RE_VERDICT = re.compile(r"\bVERDICT:\s*(KEEP|REVISE)\b", flags=re.I)

# =========================
# Light token estimator & budget helpers
# =========================
def _est_tokens(text: str, cpt: float) -> int:
    """Conservative token estimate: tokens ≈ ceil(len(chars)/cpt)."""
    if not text:
        return 0
    cpt = max(0.5, float(cpt))  # never less than 0.5 to stay conservative
    from math import ceil
    return int(ceil(len(text) / cpt))

def _clamp_tail(s: str, max_chars: int) -> str:
    if s is None:
        return ""
    if len(s) <= max_chars:
        return s
    head_keep = min(800, max_chars // 10)
    tail_keep = max_chars - head_keep
    return s[:head_keep] + "\n...\n" + s[-tail_keep:]

def _fit_prompt_budget(system_text: str, user_text: str, *,
                       max_ctx_tokens: int, gen_tokens: int, safety_margin: int,
                       hard_char_clamp: int, chars_per_token: float) -> tuple[str, str, int]:
    """Ensure system+user+gen+margin <= max_ctx_tokens (by clamping user + shrinking gen)."""
    if len(user_text or "") > hard_char_clamp:
        user_text = _clamp_tail(user_text, hard_char_clamp)
    sys_toks = _est_tokens(system_text or "", chars_per_token)
    usr_toks = _est_tokens(user_text or "", chars_per_token)
    margin = max(0, int(safety_margin))
    gen_tokens = max(1, int(gen_tokens))
    max_prompt = max_ctx_tokens - gen_tokens - margin
    if max_prompt < 0:
        gen_tokens = max(64, max_ctx_tokens - margin - max(0, sys_toks + usr_toks))
        max_prompt = max_ctx_tokens - gen_tokens - margin
    # If still too large, clamp user payload aggressively
    if sys_toks + usr_toks > max_prompt:
        target_usr = max(0, max_prompt - sys_toks)  # tokens allowed for user
        # convert tokens→chars via conservative chars_per_token
        user_text = _clamp_tail(user_text, max(512, int(target_usr * max(0.5, float(chars_per_token)))))
        usr_toks = _est_tokens(user_text, chars_per_token)
        if sys_toks + usr_toks > max_prompt:
            # final fallback: reduce gen tokens further
            spare = max(0, max_ctx_tokens - margin - (sys_toks + usr_toks))
            gen_tokens = max(32, spare)
    return system_text, user_text, gen_tokens

# =========================
# CLI
# =========================
def get_args():
    p = ArgumentParser()
    # IO
    p.add_argument("--input", required=True, type=str, help="Input predictions JSONL.")
    p.add_argument("--output", required=True, type=str, help="Output JSONL with reflected results.")
    p.add_argument("--feedback_path", default=None, type=str,
                   help="Optional JSONL with {index|id, correct:'y'|'n', feedback, ground_truth?}.")
    p.add_argument("--lessons_db", default="lessons.jsonl", type=str, help="Lessons memory JSONL.")
    p.add_argument("--lessons_topk", default=5, type=int, help="How many lessons to inject.")
    p.add_argument("--lesson_scope_field", default="category", type=str,
                   help="Scope field for lessons (e.g., 'category').")

    # Models
    p.add_argument("--use_openai", action="store_true", help="Use OpenAI chat completions.")
    p.add_argument("--model", type=str, help="Model (HF repo id OR local path, or OpenAI name when --use_openai).")
    p.add_argument("--reflector_model", default=None, type=str, help="Model for the critic (default: --model).")
    p.add_argument("--temperature", type=float, help="Generation temperature for revisions (default 0.2).")
    p.add_argument("--max_tokens", default=1024, type=int, help="Max tokens for revisions (upper bound).")

    # vLLM scaling & memory
    p.add_argument("--tp_size", default=None, type=int, help="[Deprecated] vLLM tensor parallel size.")
    p.add_argument("--num_gpus", default=None, type=int, help="Number of GPUs to use (overrides --tp_size).")
    p.add_argument("--max_model_len", default=4096, type=int, help="True backend context window (tokens).")
    p.add_argument("--gpu_memory_utilization", default=0.90, type=float,
                   help="vLLM GPU memory utilization (0.0–1.0).")
    p.add_argument("--rope_scaling", default=None, type=str,
                   help='JSON for RoPE scaling, e.g. \'{"name":"yarn","factor":16.0}\'.')

    # Reflection loop
    p.add_argument("--reflection_rounds", default=2, type=int,
                   help="How many reflect→revise cycles (default 2).")
    p.add_argument("--critique_style", default="A", choices=["A", "B"],
                   help="Non-leaking critique style (A=concise, B=structured).")

    # Ground-truth usage & scoring
    p.add_argument("--gt_field", default="res.actual", type=str,
                   help="Ground truth field name or dotted path (e.g., 'actual' or 'res.actual').")
    p.add_argument("--gt_stop_on", default="em", choices=["em", "none"],
                   help="Objective for early stop (exact match or none).")
    p.add_argument("--gt_allow_leak", action="store_true",
                   help="Allow reviser to see GT (only used when --on_mismatch=force).")

    # Mismatch policy
    p.add_argument("--on_mismatch", default="message",
                   choices=["message", "annotate", "keep", "force"],
                   help="Action when EM==0. Default 'message' replaces output with a non-leaking message.")
    p.add_argument("--mismatch_message", type=str,
                   default="Mismatch! The predicted answer does not match the actual answer. Please reflect and revise.",
                   help="Text to use when reporting a mismatch without leaking GT.")
    p.add_argument("--force_answer_tags", action="store_true",
                   help="(Force mode only) Always wrap final answers in [ANSWER]...[/ANSWER] tags when leaking GT.")

    # Algo tag
    p.add_argument("--algo", type=str, help="Algorithm/task tag (e.g., ae, rle, lzw, huffman).")

    # Subsetting
    p.add_argument("--subset", default=None, type=str,
                   help="Process only a slice of input rows using 0-based 'START:END' (END exclusive).")
    p.add_argument("--offset", default=0, type=int, help="Skip this many rows before processing (0-based).")
    p.add_argument("--limit", default=None, type=int, help="Process at most this many rows.")

    # Offline / cache options
    p.add_argument("--hf_offline", action="store_true",
                   help="Force HF offline mode (no internet). Requires models pre-cached.")
    p.add_argument("--cache_dir", default=None, type=str,
                   help="HF cache directory (defaults to ~/.cache/huggingface/hub if present).")
    p.add_argument("--model_local_dir", default=None, type=str,
                   help="Explicit local model directory (overrides repo id).")

    # IO perf
    p.add_argument("--flush_every", default=200, type=int,
                   help="Write out results every N items (default 200).")

    # Context management
    p.add_argument("--model_ctx", default=4096, type=int,
                   help="Budgeting window (tokens). Will be clamped to --max_model_len.")
    p.add_argument("--gen_tokens", default=512, type=int,
                   help="Target generation tokens per call (auto-reduced to fit).")
    p.add_argument("--safety_margin", default=128, type=int,
                   help="Extra token margin to avoid boundary errors.")
    p.add_argument("--truncate_hard_chars", default=8000, type=int,
                   help="Absolute char clamp for very long blocks (tail kept).")
    p.add_argument("--chars_per_token", default=2.0, type=float,
                   help="Conservative chars-per-token for budgeting (lower = safer).")

    # Stats output
    p.add_argument("--stats_path", default=None, type=str,
                   help="Optional JSON file path to write aggregate per-round stats.")

    # (optional) per-item CSV with success flags etc. If provided, write rows there.
    p.add_argument("--per_item_stats_csv", default=None, type=str,
                   help="Optional CSV path to write per-item success stats.")

    return p.parse_args()

# =========================
# JSONL helpers
# =========================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                pass

def write_jsonl(objs, path, mode="a"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

# =========================
# Conversation focusing
# =========================
def _latest_task_text(messages):
    text = '\n'.join(m.get("content", "") for m in messages)
    anchor = "The input and output requirements are as follows:"
    k = text.rfind(anchor)
    if k != -1:
        text = text[k:]
    j = text.rfind("[PYTHON]")
    if j != -1:
        text = text[max(0, j - 2000):]
    # Hard clamp the conversation block itself (pre-budget) to avoid giant tails
    clamp_chars = int(os.environ.get("REFLECT_CONV_CLAMP", "10000"))
    return _clamp_tail(text, clamp_chars)

# =========================
# Robust JSON extraction utilities (always canonicalize to flat)
# =========================
def _strip_code_fences(text: str) -> str:
    return RE_CODE_FENCE.sub("", text or "")

def _from_possible_nested(obj):
    """
    Accepts many shapes and extracts a scalar:
      {"output": <scalar>}
      {"output": {"return": <scalar>}}
      {"return": <scalar>}
      {"some_key": <scalar>}  # if single-key
      <scalar>  # already scalar
    """
    if isinstance(obj, dict):
        if "output" in obj:
            v = obj["output"]
            if not isinstance(v, dict):
                return v
            # nested dict: prefer single key or 'return'
            if "return" in v and not isinstance(v["return"], dict):
                return v["return"]
            if len(v) == 1:
                only = next(iter(v.values()))
                if not isinstance(only, dict):
                    return only
        # direct 'return'
        if "return" in obj and not isinstance(obj["return"], dict):
            return obj["return"]
        # single-key dict fallback
        if len(obj) == 1:
            only = next(iter(obj.values()))
            if not isinstance(only, dict):
                return only
        return obj
    return obj

def extract_structured_answer(ans) -> Tuple[Optional[str], Optional[object], Optional[None]]:
    """
    Returns (json_text, scalar_value, None) or (None, None, None).
    Canonicalized as: {"output": <value>}
    """
    def _canon(val):
        return json.dumps({"output": val}, ensure_ascii=False), val, None

    if isinstance(ans, dict):
        val = _from_possible_nested(ans)
        try:
            return _canon(val)
        except Exception:
            return None, None, None

    if not isinstance(ans, str):
        ans = str(ans)

    m = RE_ANSWER_TAGS.search(ans or "")
    search = m.group(1) if m else ans
    search = _strip_code_fences(search)
    s = search.strip()

    # Fast path: try whole blob
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            val = _from_possible_nested(obj)
            if isinstance(val, (str, int, float)) or val is None:
                return _canon(val)
        except Exception:
            pass

    # Bounded JSON scan
    scan_window = search[:20000]
    n, i = len(scan_window), 0
    while i < n:
        if scan_window[i] == '{':
            depth, start = 1, i
            i += 1
            while i < n and depth > 0:
                ch = scan_window[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                i += 1
            if depth == 0:
                try:
                    obj = json.loads(scan_window[start:i])
                except Exception:
                    i += 1
                    continue
                val = _from_possible_nested(obj)
                if isinstance(val, (str, int, float)) or val is None:
                    return _canon(val)
        else:
            i += 1
    return None, None, None

# =========================
# EM scoring (numbers & strings)
# =========================
def _norm_text(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.casefold().strip()

def _maybe_unquote_json_string(s):
    if isinstance(s, str) and len(s) >= 2 and s[0] in "'\"" and s[-1] == s[0]:
        try:
            return json.loads(s)
        except Exception:
            return s.strip(s[0])
    return s

def eval_with_gt(answer, gt):
    try:
        gnum = float(gt)
    except Exception:
        gnum = None
    _, a_val, _ = extract_structured_answer(answer)
    if isinstance(a_val, (int, float)) and gnum is not None:
        return {"em": int(math.isclose(float(a_val), gnum, rel_tol=1e-3, abs_tol=1e-3))}
    gts = _maybe_unquote_json_string(gt)
    if isinstance(a_val, str) and isinstance(gts, str):
        return {"em": int(_norm_text(a_val) == _norm_text(gts))}
    return {"em": int(_norm_text(str(answer)) == _norm_text(str(gt)))}

# =========================
# Helper for GT force mode (always flat)
# =========================
def _format_answer_with_gt(expected_format_source, gt, *, force_tags=False):
    try:
        gt_num = float(gt)
        gt_val = int(gt_num) if abs(gt_num - int(gt_num)) < 1e-15 else gt_num
    except Exception:
        gt_val = _maybe_unquote_json_string(gt)
    body = json.dumps({"output": gt_val}, ensure_ascii=False)
    has_tags = force_tags or (("[ANSWER]" in (expected_format_source or "")) and ("[/ANSWER]" in (expected_format_source or "")))
    return f"[ANSWER]\n{body}\n[/ANSWER]" if has_tags else body

# =========================
# Format-only repair (flat) — with budgeting
# =========================
def format_repair(use_openai, model_name, llm, raw_text, *,
                  max_tokens=256, model_ctx=4096, safety_margin=128, truncate_hard_chars=8000, chars_per_token=2.0):
    """If response lacks required JSON, ask model to reformat ONLY (with context budgeting)."""
    system = ("You must output ONLY:\n"
              "[ANSWER]\n{\"output\": <value>}\n[/ANSWER]\n"
              "No other text, code, or explanations.")
    user = ("Rewrite into the required format. If it already contains the needed value, keep it.\n"
            "------\n" + str(raw_text) + "\n------")
    # Budget the prompt so system+user+gen <= model_ctx
    s_txt, u_txt, final_gen = _fit_prompt_budget(
        system, user,
        max_ctx_tokens=int(model_ctx),
        gen_tokens=int(max_tokens),
        safety_margin=int(safety_margin),
        hard_char_clamp=int(truncate_hard_chars),
        chars_per_token=float(chars_per_token),
    )
    fixed = run_chat(
        use_openai, model_name, llm,
        [{"role": "system", "content": s_txt}, {"role": "user", "content": u_txt}],
        for_reflection=True, temperature=0.0, max_tokens=final_gen
    )
    j, val, _ = extract_structured_answer(fixed)
    return fixed, j, val

# =========================
# Model runners
# =========================
def format_messages(messages):
    return '\n'.join([f"{m['role'].capitalize()}: {m['content']}" for m in messages]) + "\nAssistant:"

def _ensure_openai():
    global client
    if client is not None:
        return client
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set and --use_openai supplied.")
    client = OpenAI(api_key=key)
    return client

def run_openai_chat(model_name, messages, temperature, max_tokens, n=1, stop=None, seed=42):
    _ensure_openai()
    resp = client.chat_completions.create(  # type: ignore[attr-defined]
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        seed=seed,
    )
    return [c.message.content.strip() for c in resp.choices]

def run_vllm_chat(llm, messages, sampling_params):
    prompt = format_messages(messages)
    outputs = llm.generate([prompt], sampling_params)
    return [out.text.strip() for out in outputs[0].outputs]

def run_chat(use_openai, model_name, llm, messages, *, for_reflection, temperature, max_tokens):
    temp = 0.2 if temperature is None else temperature
    if use_openai:
        return run_openai_chat(
            model_name, messages,
            temperature=(0.2 if for_reflection else temp),
            max_tokens=max_tokens, n=1, stop=None, seed=42
        )[0]
    else:
        sp = SamplingParams(
            temperature=(0.2 if for_reflection else temp),
            max_tokens=max_tokens,
            top_p=0.95,
            stop=None,
            include_stop_str_in_output=False,
            n=1,
            seed=42
        )
        return run_vLLM_with_sampling(llm, messages, sp)

def run_vLLM_with_sampling(llm, messages, sp):
    return run_vllm_chat(llm, messages, sp)[0]

# =========================
# Prompts
# =========================
CRITIQUE_A_SYSTEM = (
    "You are a concise reviewer. Do not reveal or estimate any expected value. "
    "Flag issues briefly and decide KEEP/REVISE."
)
CRITIQUE_A_USER = (
    "Conversation:\n{conversation}\n\nDraft answer:\n{draft}\n\n"
    "Provide 2–4 short bullet points about correctness/format without revealing the expected value. "
    "End with VERDICT: KEEP or VERDICT: REVISE."
)

CRITIQUE_B_SYSTEM = (
    "You are a structured reviewer. Provide actionable findings and a fix plan. "
    "Do not reveal or approximate any expected value."
)
CRITIQUE_B_USER = (
    "Conversation:\n{conversation}\n\nDraft answer:\n{draft}\n\n"
    "Write sections 'Findings:' and 'Fix:' in bullet points. "
    "Do not include or infer the expected value. "
    "End with VERDICT: KEEP or VERDICT: REVISE."
)

REVISION_SYSTEM_NOLEAK = (
    "You are a careful editor. Revise the draft strictly according to the feedback. "
    "Do not include analysis. Provide only the improved final answer."
)
REVISION_USER_NOLEAK = (
    "Conversation:\n{conversation}\n\nDraft answer:\n{draft}\n\n"
    "Feedback:\n{feedback}\n\n"
    "Now produce the corrected final answer only."
)

REVISION_SYSTEM_LEAK = (
    "You are a careful editor. You may use the provided ground truth to correct the draft. "
    "Do not include analysis or refer to the ground truth explicitly unless required by the task."
)
REVISION_USER_LEAK = (
    "Conversation:\n{conversation}\n\nDraft answer:\n{draft}\n\n"
    "Ground truth:\n{gt}\n\n"
    "Feedback:\n{feedback}\n\n"
    "Now produce the corrected final answer only."
)

# =========================
# vLLM init
# =========================
def init_vllm(model_name, num_gpus, tp_size, max_model_len, gpu_memory_utilization, download_dir=None, offline=False, rope_scaling=None):
    if LLM is None or SamplingParams is None:
        raise RuntimeError("vLLM not available; install vllm to use non-OpenAI mode.")
    tp = num_gpus or tp_size or 1
    print(f"[vLLM] Loading {model_name} (TP={tp}, max_model_len={max_model_len}, gpu_mem_util={gpu_memory_utilization}) ...")
    if download_dir:
        print(f"[vLLM] Using download/cache dir: {download_dir}")
    if offline:
        print("[vLLM] HF offline mode is ON (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1).")
        
    kwargs = dict(
        model=model_name,
        download_dir=download_dir,
        tensor_parallel_size=tp,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization
        )

    if rope_scaling:
        kwargs["rope_scaling"] = rope_scaling
        print(f"[vLLM] rope_scaling = {rope_scaling}")
    
    llm = LLM(**kwargs)
    return llm

# =========================
# Helpers for HF cache paths
# =========================
def hub_root_from_cache_dir(cache_dir: str) -> str:
    if not cache_dir:
        return os.path.expanduser("~/.cache/huggingface/hub")
    return cache_dir if cache_dir.endswith("/hub") else os.path.join(cache_dir, "hub")

def resolve_local_snapshot_dir(repo_id: str, hub_root: str) -> Optional[str]:
    safe_repo = repo_id.replace("/", "--")
    base = os.path.join(hub_root, f"models--{safe_repo}", "snapshots")
    if not os.path.isdir(base):
        return None
    try:
        snaps = [os.path.join(base, d) for d in os.listdir(base)]
        snaps = [d for d in snaps if os.path.isdir(d)]
        snaps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return snaps[0] if snaps else None
    except Exception:
        return None

# =========================
# Main
# =========================
def main():
    args = get_args()
    t0 = dt.datetime.now()
    print(f"Start: {t0:%Y-%m-%d %H:%M:%S}")

    # Align budgeting window to the true backend limit, always.
    if args.model_ctx > args.max_model_len:
        print(f"[ctx] Adjusting --model_ctx {args.model_ctx} -> --max_model_len {args.max_model_len}")
        args.model_ctx = args.max_model_len

    # Cap max_tokens/gen_tokens to max_model_len (safe for both OpenAI and vLLM)
    if args.max_tokens > args.max_model_len:
        print(f"[note] --max_tokens ({args.max_tokens}) > --max_model_len ({args.max_model_len}); capping.")
        args.max_tokens = args.max_model_len
    if args.gen_tokens and args.gen_tokens > args.max_model_len:
        print(f"[note] --gen_tokens ({args.gen_tokens}) > --max_model_len ({args.max_model_len}); capping.")
        args.gen_tokens = args.max_model_len

    # Offline/cache setup
    default_hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    cache_dir = args.cache_dir or (default_hf_cache if os.path.isdir(default_hf_cache)
                                   else os.environ.get("HF_HOME") or default_hf_cache)
    hub_root = hub_root_from_cache_dir(cache_dir)
    os.environ.setdefault("HF_HOME", cache_dir if not cache_dir.endswith("/hub") else os.path.dirname(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub_root)

    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("[offline] HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1")

    # Resolve model path for vLLM
    model_to_load = args.model
    if args.model_local_dir:
        model_to_load = args.model_local_dir
        print(f"[model] Using explicit local model dir: {model_to_load}")
    elif args.model and os.path.isdir(args.model):
        model_to_load = args.model
        print(f"[model] Using local dir (from --model): {model_to_load}")
    elif args.hf_offline and args.model and "/" in (args.model or ""):
        snap = resolve_local_snapshot_dir(args.model, hub_root)
        if snap and os.path.isdir(snap):
            model_to_load = snap
            print(f"[model] Offline map: '{args.model}' → local snapshot: {snap}")
        else:
            raise SystemExit(f"[offline] Could not find local snapshot for '{args.model}' under '{hub_root}'.")

    # Subsetting
    offset = max(0, int(args.offset or 0))
    limit = (None if args.limit is None else max(0, int(args.limit)))
    if args.subset:
        try:
            start, end = args.subset.split(":")
            offset = int(start) if start else 0
            limit = (int(end) - offset) if end else None
        except Exception as e:
            raise SystemExit(f"Invalid --subset '{args.subset}': {e}")
    print(f"[subset] offset={offset}, limit={'∞' if limit is None else limit}")

    # Prepare models
    llm = None
    rope_scaling_cfg = None
    if args.rope_scaling:
        try:
            rope_scaling_cfg = json.loads(args.rope_scaling)
        except Exception as e:
            raise SystemExit(f"Invalid --rope_scaling JSON: {e}")
    if args.use_openai:
        _ensure_openai()
        print(f"Using OpenAI model: {model_to_load}")
    else:
        llm = init_vllm(
            model_name=model_to_load,
            num_gpus=args.num_gpus,
            tp_size=args.tp_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            download_dir=hub_root,
            offline=bool(args.hf_offline),
            rope_scaling=rope_scaling_cfg
        )
        print(f"Using vLLM model: {model_to_load}")

    reflector_model = args.reflector_model or model_to_load

    # Lessons & optional external feedback
    all_lessons = []
    if os.path.exists(args.lessons_db):
        all_lessons = [js for js in load_jsonl(args.lessons_db) if isinstance(js, dict)]
    feedback_map = {}
    if args.feedback_path and os.path.exists(args.feedback_path):
        for row in load_jsonl(args.feedback_path):
            key = row.get("index", row.get("id"))
            if key is not None:
                feedback_map[key] = row

    # Fresh run truncate
    if os.path.exists(args.output) and offset == 0 and (limit is None):
        open(args.output, "w").close()

    # prepare CSV header if requested
    csv_path = args.per_item_stats_csv
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", encoding="utf-8") as fcsv:
                fcsv.write("id,index,algo,model,temperature,em,resolved_round,success,actual,predicted,io_pred_raw,output_raw\n")

    out_buf = []
    line_no = 0
    processed = 0

    # =========================
    # Aggregated stats (per-round resolution)
    # =========================
    total_items = 0
    with_gt = 0
    without_gt = 0
    resolved_at_round = defaultdict(int)  # round -> count (first time EM==1)
    unresolved = 0

    for js in load_jsonl(args.input):
        line_no += 1
        idx0 = line_no - 1
        if idx0 < offset:
            continue
        if limit is not None and processed >= limit:
            break

        messages = js.get("messages") or []
        draft = js.get("output")
        if not messages or draft is None:
            continue

        total_items += 1

        # Lessons scope + message build
        scope = js.get(args.lesson_scope_field) or js.get("meta", {}).get(args.lesson_scope_field) or args.algo or "global"
        scoped = [L for L in all_lessons if L.get("scope", "global") in (scope, "global")]
        scoped.sort(key=lambda L: L.get("ts", 0), reverse=True)
        picked = scoped[: args.lessons_topk]
        if picked:
            kb_lines = ["KNOWN PITFALLS & FIXES (from prior reviews):"]
            for i, L in enumerate(picked, 1):
                fb = str(L.get("feedback") or "").strip()
                if fb:
                    kb_lines.append(f"{i}. {fb}")
            kb = "\n".join(kb_lines)
            messages_for_review = [{"role": "system", "content": kb}] + messages
        else:
            messages_for_review = messages

        # Focus on the last task only
        conv_text = _latest_task_text(messages_for_review)

        # GT helper
        def get_by_dotted(d, path):
            if d is None or not path:
                return None
            cur = d
            for p in path.split("."):
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return None
            return cur

        gt = get_by_dotted(js, args.gt_field)
        has_gt = gt is not None
        with_gt += int(has_gt)
        without_gt += int(not has_gt)

        print(f"[{idx0}] Processing id={js.get('id', idx0)}, GT={gt}, GT_present={has_gt}, scope='{scope}'")

        # External reviewer feedback (optional)
        feedback = feedback_map.get(js.get("index", js.get("id"))) if feedback_map else None
        ext_feedback = (feedback and feedback.get("feedback"))

        # Reflection loop
        critiques = []
        status_msg = None
        scores = {"em": None}
        current = draft

        # Initial parse & optional score
        init_json, init_val, _ = extract_structured_answer(draft)
        if has_gt:
            try:
                scores = eval_with_gt(init_json or draft, gt)
            except Exception:
                scores = {"em": None}
        if init_json is None:
            _, init_json, init_val = format_repair(
                args.use_openai, model_to_load, llm, draft,
                max_tokens=256, model_ctx=int(args.model_ctx),
                safety_margin=int(args.safety_margin),
                truncate_hard_chars=int(args.truncate_hard_chars),
                chars_per_token=float(args.chars_per_token),
            )
            if init_json:
                current = init_json
                if has_gt:
                    scores = eval_with_gt(current, gt)
            else:
                init_json = draft
                current = init_json
        else:
            current = init_json

        print(f"[{idx0}] Round 0 EM: {scores.get('em')}")

        round_outputs = [{
            "round": 0,
            "output": init_json,
            "raw": draft,
            "scores": scores
        }]

        # Only pass GT to reviser when explicitly requested *and* on_mismatch == 'force'
        pass_gt_to_reviser = (args.on_mismatch == "force" and args.gt_allow_leak and has_gt)

        for r in range(args.reflection_rounds):
            if has_gt and args.gt_stop_on == "em" and scores["em"] == 1:
                print(f"[{idx0}] Early stop at round {r} (EM==1).")
                break

            # critique
            if args.critique_style == "B":
                user_prompt = CRITIQUE_B_USER.format(conversation=conv_text, draft=current)
                sys_prompt = CRITIQUE_B_SYSTEM
            else:
                user_prompt = CRITIQUE_A_USER.format(conversation=conv_text, draft=current)
                sys_prompt = CRITIQUE_A_SYSTEM

            # Tag with round number
            sys_prompt = f"[Round {r+1}] " + sys_prompt
            user_prompt = f"[Round {r+1}] " + user_prompt

            # Context budgeting (critique)
            crit_gen = args.gen_tokens if args.gen_tokens is not None else min(args.max_tokens, 1024)
            s_txt, u_txt, final_gen = _fit_prompt_budget(
                sys_prompt, user_prompt,
                max_ctx_tokens=int(args.model_ctx),
                gen_tokens=int(crit_gen),
                safety_margin=int(args.safety_margin),
                hard_char_clamp=int(args.truncate_hard_chars),
                chars_per_token=float(args.chars_per_token),
            )

            # >>>>>>>>>>>>>>>>>>>>>>>>>> PATCH: Assistant ack before CRITIQUE
            assistant_ack_crit = {
                "role": "assistant",
                "content": f"[Round {r+1}] Acknowledged. I will now provide critique on the previous answer."
            }
            crit_msgs = [
                {"role": "system", "content": s_txt},
                assistant_ack_crit,
                {"role": "user", "content": u_txt},
            ]
            # <<<<<<<<<<<<<<<<<<<<<<<<<< PATCH END

            critique = run_chat(args.use_openai, reflector_model, llm, crit_msgs,
                                for_reflection=True, temperature=0.2, max_tokens=final_gen)
            if ext_feedback:
                critique = critique.strip() + "\n- Additional reviewer note: " + str(ext_feedback).strip()
            mver = RE_VERDICT.findall(critique or "")
            verdict = (mver[-1].upper() if mver else ("REVISE" if "REVISE" in (critique or "").upper() else "KEEP"))
            print(f"[{idx0}] Round {r+1} verdict: {verdict}")
            critiques.append({
                "round": r + 1,
                "assistant_marker_critique": assistant_ack_crit["content"],
                "critique": critique,
                "verdict": verdict
            })

            # Skip revision if critic says KEEP and we have nothing to optimize wrt GT
            if verdict == "KEEP" and (not has_gt or scores.get("em") == 1):
                round_outputs.append({
                    "round": r + 1,
                    "assistant_marker_critique": assistant_ack_crit["content"],
                    "critique": critique,
                    "assistant_marker_revision": None,
                    "revision": None,
                    "output": current,
                    "raw": current,
                    "scores": scores
                })
                continue

            # revise messages
            if pass_gt_to_reviser:
                rev_user = f"Conversation:\n{conv_text}\n\nDraft answer:\n{current}\n\nGround truth:\n{gt}\n\nFeedback:\n{critique}\n\nNow produce the corrected final answer only."
                rev_sys = REVISION_SYSTEM_LEAK
            else:
                rev_user = f"Conversation:\n{conv_text}\n\nDraft answer:\n{current}\n\nFeedback:\n{critique}\n\nNow produce the corrected final answer only."
                rev_sys = REVISION_SYSTEM_NOLEAK

            # Tag revision prompts with round too
            rev_sys = f"[Round {r+1}] " + rev_sys
            rev_user = f"[Round {r+1}] " + rev_user

            # Context budgeting (revision)
            sys_txt, usr_txt, final_rev_gen = _fit_prompt_budget(
                rev_sys, rev_user,
                max_ctx_tokens=int(args.model_ctx),
                gen_tokens=int(args.gen_tokens if args.gen_tokens is not None else args.max_tokens),
                safety_margin=int(args.safety_margin),
                hard_char_clamp=int(args.truncate_hard_chars),
                chars_per_token=float(args.chars_per_token),
            )

            # >>>>>>>>>>>>>>>>>>>>>>>>>> (existing) Assistant ack before REVISION
            assistant_ack_rev = {"role": "assistant",
                                 "content": f"[Round {r+1}] Acknowledged. I will reflect on the feedback and revise the answer now."}
            rev_msgs = [
                {"role": "system", "content": sys_txt},
                {"role": "user", "content": usr_txt},
                assistant_ack_rev
            ]
            # <<<<<<<<<<<<<<<<<<<<<<<<<<

            revised = run_chat(
                args.use_openai, model_to_load, llm, rev_msgs,
                for_reflection=False,
                temperature=(args.temperature if args.temperature is not None else 0.2),
                max_tokens=final_rev_gen
            )

            # Extract & repair if needed (budgeted)
            clean_json, _, _ = extract_structured_answer(revised)
            if clean_json is None:
                _, clean_json, _ = format_repair(
                    args.use_openai, model_to_load, llm, revised,
                    max_tokens=256, model_ctx=int(args.model_ctx),
                    safety_margin=int(args.safety_margin),
                    truncate_hard_chars=int(args.truncate_hard_chars),
                    chars_per_token=float(args.chars_per_token),
                )
                if not clean_json:
                    clean_json = revised
            current = clean_json

            # Score after revision
            if has_gt:
                scores = eval_with_gt(clean_json or current, gt)

            print(f"[{idx0}] Round {r+1} EM: {scores.get('em')}")
            round_outputs.append({
                "round": r + 1,
                "assistant_marker_critique": assistant_ack_crit["content"],   # show critiquer ack
                "critique": critique,                                        # show critiquer content
                "assistant_marker_revision": assistant_ack_rev["content"],   # show reviser ack
                "revision": revised,                                         # show reviser raw output
                "output": clean_json,
                "raw": revised,
                "scores": scores
            })

            # Early action on mismatch if using 'force'
            if scores.get("em") == 0 and args.on_mismatch == "force" and has_gt:
                current = _format_answer_with_gt(current, gt, force_tags=args.force_answer_tags)
                scores = eval_with_gt(current, gt)
                print(f"[{idx0}] Forcing final answer with GT (inside loop; LEAK).")
                break

        # Final action after loop
        if has_gt and scores.get("em") == 0:
            if args.on_mismatch == "force":
                current = _format_answer_with_gt(current, gt, force_tags=args.force_answer_tags)
                scores = eval_with_gt(current, gt)
                print(f"[{idx0}] Forcing final answer with GT (post loop; LEAK).")
            elif args.on_mismatch in ("annotate", "keep"):
                status_msg = args.mismatch_message
            elif args.on_mismatch == "message":
                current = args.mismatch_message
                status_msg = args.mismatch_message

        # Final clean
        final_json, _, _ = extract_structured_answer(current)
        final_out = final_json if (final_json is not None) else current

        # Per-item resolved round (only if GT present)
        resolved_round_idx = None
        if has_gt:
            for ro in round_outputs:
                em = (ro.get("scores") or {}).get("em", None)
                if em == 1:
                    resolved_round_idx = ro.get("round", None)
                    break
            if resolved_round_idx is None:
                unresolved += 1
                print(f"[{idx0}] Unresolved after {args.reflection_rounds} rounds.")
            else:
                resolved_at_round[int(resolved_round_idx)] += 1
                print(f"[{idx0}] Resolved first at round {resolved_round_idx}.")

        entry = js.copy()
        entry.update({
            "algo": args.algo,
            "output": (final_out if (args.on_mismatch != "message" or scores.get("em") == 1)
                       else args.mismatch_message),
            "reasoning": {
                "round_outputs": round_outputs,  # includes both critiquer & reviser info + acks
                "critiques": critiques,          # per-round critic entries (with ack + verdict)
                "scores": scores,
                "reflection_rounds": args.reflection_rounds,
                "gt_stop_on": args.gt_stop_on,
                "gt_allow_leak": (args.on_mismatch == "force" and args.gt_allow_leak),
                "scope": scope,
                "resolved_round": resolved_round_idx
            },
        })
        if status_msg:
            entry["reasoning"]["status"] = status_msg

        out_buf.append(entry)
        processed += 1

        # write per-item CSV row if requested
        if csv_path:
            # figure out fields
            actual = None
            try:
                actual = get_by_dotted(js, args.gt_field)
            except Exception:
                actual = None
            predicted = None
            try:
                _, pv, _ = extract_structured_answer(final_out)
                predicted = pv
            except Exception:
                predicted = final_out
            em_val = scores.get("em")
            success = int(em_val == 1) if em_val is not None else ""
            io_pred_raw = js.get("io_pred", "")
            # output_raw = js.get("output", "")

            with open(csv_path, "a", encoding="utf-8") as fcsv:
                fcsv.write(
                    f"{json.dumps(js.get('id', idx0), ensure_ascii=False)},"
                    f"{idx0},"
                    f"{json.dumps(args.algo, ensure_ascii=False)},"
                    f"{json.dumps(args.model, ensure_ascii=False)},"
                    f"{json.dumps(js.get('temperature'), ensure_ascii=False)},"
                    f"{'' if em_val is None else em_val},"
                    f"{'' if resolved_round_idx is None else resolved_round_idx},"
                    f"{success},"
                    f"{json.dumps(actual, ensure_ascii=False)},"
                    f"{json.dumps(predicted, ensure_ascii=False)},"
                    f"{json.dumps(io_pred_raw, ensure_ascii=False)}\n"
                    # f"{json.dumps(output_raw, ensure_ascii=False)}\n"
                )

        if len(out_buf) >= int(args.flush_every):
            write_jsonl(out_buf, args.output, mode="a")
            print(f"[flush] Wrote {len(out_buf)} items to {args.output}")
            out_buf = []

    if out_buf:
        write_jsonl(out_buf, args.output, mode="a")
        print(f"[flush] Wrote final {len(out_buf)} items to {args.output}")

    # =========================
    # Aggregate stats summary
    # =========================
    print("\n=== Reflection Summary ===")
    print(f"Items processed: {total_items}")
    print(f"With GT: {with_gt} | Without GT: {without_gt}")

    # Build per-round incremental & cumulative
    max_round_observed = max(resolved_at_round.keys()) if resolved_at_round else 0
    max_round_to_show = max(max_round_observed, args.reflection_rounds)

    inc_counts = {r: resolved_at_round.get(r, 0) for r in range(max_round_to_show + 1)}
    cum_counts = {}
    run = 0
    for r in range(max_round_to_show + 1):
        run += inc_counts[r]
        cum_counts[r] = run

    def _pct(n):
        return (100.0 * n / with_gt) if with_gt else 0.0

    print("\nResolved first at round:")
    for r in range(max_round_to_show + 1):
        print(f"  Round {r}: {inc_counts[r]} ({_pct(inc_counts[r]):.2f}%)")
    print(f"  Unresolved after {args.reflection_rounds} rounds: {unresolved} ({_pct(unresolved):.2f}%)")

    print("\nCumulative resolved by round:")
    for r in range(max_round_to_show + 1):
        print(f"  ≤ Round {r}: {cum_counts[r]} ({_pct(cum_counts[r]):.2f}%)")

    # Optional stats JSON
    if args.stats_path:
        stats_obj = {
            "processed": total_items,
            "with_gt": with_gt,
            "without_gt": without_gt,
            "reflection_rounds": args.reflection_rounds,
            "resolved_first_at_round": {str(k): int(v) for k, v in inc_counts.items()},
            "resolved_cumulative_by_round": {str(k): int(v) for k, v in cum_counts.items()},
            "unresolved": int(unresolved),
            "unresolved_pct": _pct(unresolved)
        }
        os.makedirs(os.path.dirname(args.stats_path) or ".", exist_ok=True)
        with open(args.stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_obj, f, ensure_ascii=False, indent=2)
        print(f"\n[stats] Wrote aggregate stats to: {args.stats_path}")

    t1 = dt.datetime.now()
    print(f"\nEnd:   {t1:%Y-%m-%d %H:%M:%S}")
    print(f"Elapsed: {str(t1 - t0).split('.')[0]}")

if __name__ == "__main__":
    main()
