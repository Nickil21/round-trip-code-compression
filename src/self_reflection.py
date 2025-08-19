#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reflect_only.py — Reflection-only improver (no regeneration), up to TWO iterations.

New / Important:
- Supports **dotted path** ground-truth lookup (e.g., --gt_field res.actual).
- Numeric-aware EM check for tasks that wrap answers (e.g., [ANSWER] {"output": 0.123} [/ANSWER]).
- --num_gpus: number of GPUs to use with vLLM (maps to tensor_parallel_size)
- --max_model_len: context length for vLLM
- --gpu_memory_utilization: vLLM memory utilization knob (0–1)
- Caps max_tokens to max_model_len with a friendly note
- Subsetting controls:
    * --subset START:END (0-based, END exclusive), e.g. ':100', '100:200', '100:'
    * --offset N and --limit M (if you prefer explicit knobs)

Example:
python reflect_only.py --input preds.jsonl --output preds_reflected.jsonl \
  --model Qwen/Qwen3-32B --algo ae --critique_style B --gt_stop_on em \
  --gt_field res.actual --num_gpus 2 --max_model_len 8192 --gpu_memory_utilization 0.92 \
  --subset 100:200
"""

import os
import re
import json
import math
import time
import datetime as dt
from argparse import ArgumentParser

# Optional deps for vLLM path (safe import; avoid touching torch.cuda.* here)
try:
    import torch  # noqa: F401  (import is fine; don't call torch.cuda.* here)
    from vllm import LLM, SamplingParams
except Exception:
    torch = None
    LLM = None
    SamplingParams = None

client = None  # OpenAI client (lazy init)

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
    p.add_argument("--model", default="Qwen/Qwen3-32B", type=str, help="Model for critic/reviser.")
    p.add_argument("--reflector_model", default=None, type=str, help="Model for the critic (default: --model).")
    p.add_argument("--temperature", default=0.2, type=float, help="Generation temperature for revisions.")
    p.add_argument("--max_tokens", default=4096, type=int, help="Max tokens for revisions.")
    # vLLM scaling & memory
    p.add_argument("--tp_size", default=None, type=int, help="[Deprecated] vLLM tensor parallel size.")
    p.add_argument("--num_gpus", default=None, type=int, help="Number of GPUs to use (overrides --tp_size).")
    p.add_argument("--max_model_len", default=4096, type=int, help="vLLM max model context length.")
    p.add_argument("--gpu_memory_utilization", default=0.90, type=float,
                   help="vLLM GPU memory utilization (0.0–1.0).")

    # Reflection loop — default TWO iterations
    p.add_argument("--reflection_rounds", default=2, type=int,
                   help="How many reflect→revise cycles (default 2).")
    p.add_argument("--critique_style", default="A", choices=["A", "B"],
                   help="Non-leaking critique style (A=concise, B=structured).")

    # Ground-truth usage (non-leaking) — EM only
    p.add_argument("--gt_field", default="actual", type=str,
                   help="Ground truth field name or dotted path (e.g., 'actual' or 'res.actual').")
    p.add_argument("--gt_stop_on", default="em", choices=["em", "none"],
                   help="Objective for early stop (exact match or none).")
    p.add_argument("--gt_allow_leak", action="store_true",
                   help="Allow reviser to see GT (OFF by default; critiques never leak).")

    # Algo tag (for logging/scoping)
    p.add_argument("--algo", type=str,
                   help="Algorithm/task tag (e.g., ae, rle, lzw, huffman).")

    # Subsetting
    p.add_argument("--subset", default=None, type=str,
                   help="Process only a slice of input rows using 0-based 'START:END' (END exclusive). "
                        "Examples: ':100', '100:200', '100:'. If set, overrides --offset/--limit.")
    p.add_argument("--offset", default=0, type=int,
                   help="Skip this many rows before processing (0-based).")
    p.add_argument("--limit", default=None, type=int,
                   help="Process at most this many rows.")

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

def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# =========================
# EM metric & normalization
# =========================
def _norm(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        try:
            s = json.dumps(s, ensure_ascii=False)
        except Exception:
            s = str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_match(pred, gt):
    return int(_norm(pred) == _norm(gt))

def _extract_output_value(ans):
    """Best-effort extraction of a numeric 'output' from the model answer.

    Supports:
      - Direct float/int
      - Dict with {'output': <num>}
      - JSON object embedded in a string (first object with 'output')
      - Fallback: first numeric literal in the string
    """
    if isinstance(ans, (int, float)):
        return float(ans)

    if isinstance(ans, dict):
        if "output" in ans:
            try:
                return float(ans["output"])
            except Exception:
                return None

    if not isinstance(ans, str):
        ans = str(ans)

    # Try to find JSON objects and parse the first that has "output"
    for m in re.finditer(r"\{.*?\}", ans, flags=re.S):
        try:
            obj = json.loads(m.group(0))
            if "output" in obj:
                return float(obj["output"])
        except Exception:
            continue

    # Fallback: first number in the string
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", ans)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None

def eval_with_gt(answer, gt):
    """Prefer numeric comparison if both parse as numbers; else fall back to string EM."""
    a = _extract_output_value(answer)

    # Parse GT as numeric if possible
    g = None
    if isinstance(gt, (int, float)):
        g = float(gt)
    else:
        try:
            g = float(gt)
        except Exception:
            g = None

    if a is not None and g is not None:
        return {"em": int(math.isclose(a, g, rel_tol=0.0, abs_tol=1e-12))}
    return {"em": exact_match(answer, gt)}

# =========================
# Lessons DB
# =========================
def load_lessons(path):
    if not os.path.exists(path):
        return []
    return [js for js in load_jsonl(path) if isinstance(js, dict)]

def scope_for(js, lesson_scope_field, algo):
    # prefer explicit scope field; fall back to algo; then global
    return js.get(lesson_scope_field) or js.get("meta", {}).get(lesson_scope_field) or algo or "global"

def select_lessons(lessons, scope, topk):
    scoped = [L for L in lessons if L.get("scope", "global") == scope or L.get("scope", "global") == "global"]
    scoped.sort(key=lambda L: L.get("ts", 0), reverse=True)  # recency first
    return scoped[:topk]

def lessons_block(lessons):
    if not lessons:
        return ""
    lines = ["KNOWN PITFALLS & FIXES (from prior reviews):"]
    for i, L in enumerate(lessons, 1):
        fb = (L.get("feedback") or "").strip()
        if not fb:
            continue
        lines.append(f"{i}. {fb}")
    return "\n".join(lines)

def prepend_lessons(messages, block_text):
    if not block_text:
        return messages
    return [{"role": "system", "content": block_text}] + messages

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
    resp = client.chat.completions.create(
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
    if use_openai:
        return run_openai_chat(
            model_name, messages,
            temperature=(0.2 if for_reflection else temperature),
            max_tokens=max_tokens, n=1, stop=(None), seed=42
        )[0]
    else:
        sp = SamplingParams(
            temperature=(0.2 if for_reflection else temperature),
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
# Prompts (non-leaking)
# =========================
CRITIQUE_A_SYSTEM = (
    "You are a concise reviewer. Do not reveal or estimate any expected value. "
    "Flag issues briefly and decide KEEP/REVISE."
)
CRITIQUE_A_USER = (
    "Conversation:\n{conversation}\n\n"
    "Draft answer:\n{draft}\n\n"
    "Provide 2–4 short bullet points about correctness/format without revealing the expected value. "
    "End with VERDICT: KEEP or VERDICT: REVISE."
)

CRITIQUE_B_SYSTEM = (
    "You are a structured reviewer. Provide actionable findings and a fix plan. "
    "Do not reveal or approximate any expected value."
)
CRITIQUE_B_USER = (
    "Conversation:\n{conversation}\n\n"
    "Draft answer:\n{draft}\n\n"
    "Write sections 'Findings:' and 'Fix:' in bullet points. "
    "Do not include or infer the expected value. "
    "End with VERDICT: KEEP or VERDICT: REVISE."
)

REVISION_SYSTEM_NOLEAK = (
    "You are a careful editor. Revise the draft strictly according to the feedback. "
    "Do not include analysis. Provide only the improved final answer."
)
REVISION_USER_NOLEAK = (
    "Conversation:\n{conversation}\n\n"
    "Draft answer:\n{draft}\n\n"
    "Feedback:\n{feedback}\n\n"
    "Now produce the corrected final answer only."
)

REVISION_SYSTEM_LEAK = (
    "You are a careful editor. You may use the provided ground truth to correct the draft. "
    "Do not include analysis or refer to the ground truth explicitly unless required by the task."
)
REVISION_USER_LEAK = (
    "Conversation:\n{conversation}\n\n"
    "Draft answer:\n{draft}\n\n"
    "Ground truth:\n{gt}\n\n"
    "Feedback:\n{feedback}\n\n"
    "Now produce the corrected final answer only."
)

# =========================
# Reflection helpers
# =========================
def conversation_to_text(messages):
    return '\n'.join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])

def build_critique_messages(style, conversation, draft):
    if style == "A":
        return [
            {"role": "system", "content": CRITIQUE_A_SYSTEM},
            {"role": "user", "content": CRITIQUE_A_USER.format(conversation=conversation, draft=draft)},
        ]
    # style B
    return [
        {"role": "system", "content": CRITIQUE_B_SYSTEM},
        {"role": "user", "content": CRITIQUE_B_USER.format(conversation=conversation, draft=draft)},
    ]

def reflect_once_nonleak(use_openai, reflector_model, llm, style, conv_text, draft):
    msgs = build_critique_messages(style, conv_text, draft)
    critique = run_chat(use_openai, reflector_model, llm, msgs,
                        for_reflection=True, temperature=0.2, max_tokens=2048)
    verdict = "REVISE" if "VERDICT: REVISE" in critique.upper() else "KEEP"
    return critique, verdict

def revise_with_feedback(use_openai, model, llm, conv_text, draft, feedback,
                         gt=None, allow_leak=False, temperature=0.2, max_tokens=4096):
    if allow_leak and gt is not None:
        msgs = [
            {"role": "system", "content": REVISION_SYSTEM_LEAK},
            {"role": "user", "content": REVISION_USER_LEAK.format(conversation=conv_text, draft=draft, feedback=feedback, gt=gt)},
        ]
    else:
        msgs = [
            {"role": "system", "content": REVISION_SYSTEM_NOLEAK},
            {"role": "user", "content": REVISION_USER_NOLEAK.format(conversation=conv_text, draft=draft, feedback=feedback)},
        ]
    out = run_chat(use_openai, model, llm, msgs,
                   for_reflection=False, temperature=temperature, max_tokens=max_tokens)
    return out

# =========================
# Utilities
# =========================
def _parse_subset_arg(spec):
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError("format must be START:END with END optional (e.g., ':100', '100:', '100:200').")
    start = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if parts[1] else None
    if start < 0 or (end is not None and end < 0):
        raise ValueError("negative indexes are not supported for streaming input.")
    limit = None if end is None else max(0, end - start)
    return start, limit

def get_by_dotted(d, path):
    """Return value from dict 'd' following a dotted path like 'res.actual'."""
    if d is None or not path:
        return None
    cur = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

# =========================
# vLLM init
# =========================
def init_vllm(model_name, num_gpus, tp_size, max_model_len, gpu_memory_utilization):
    if LLM is None or SamplingParams is None:
        raise RuntimeError("vLLM not available; install vllm to use non-OpenAI mode.")

    tensor_parallel_size = num_gpus or tp_size or 1
    print(f"[vLLM] Loading {model_name} (TP={tensor_parallel_size}, max_model_len={max_model_len}, "
          f"gpu_mem_util={gpu_memory_utilization}) ...")

    # IMPORTANT: do not call torch.cuda.is_available() here to avoid initializing CUDA in parent.
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype="auto",  # let vLLM choose; avoids touching torch.cuda in the parent
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    return llm

# =========================
# Main
# =========================
def main():
    args = get_args()
    t0 = dt.datetime.now()
    print(f"Start: {t0:%Y-%m-%d %H:%M:%S}")

    # Resolve subsetting
    offset = max(0, int(args.offset or 0))
    limit = (None if args.limit is None else max(0, int(args.limit)))
    if args.subset:
        try:
            offset, limit = _parse_subset_arg(args.subset)
        except Exception as e:
            raise SystemExit(f"Invalid --subset '{args.subset}': {e}")
    print(f"[subset] offset={offset}, limit={'∞' if limit is None else limit}")

    # Cap max_tokens to max_model_len (vLLM)
    if not args.use_openai and args.max_tokens > args.max_model_len:
        print(f"[note] --max_tokens ({args.max_tokens}) > --max_model_len ({args.max_model_len}); capping to max_model_len.")
        args.max_tokens = args.max_model_len

    # Prepare models
    llm = None
    if args.use_openai:
        _ensure_openai()
        print(f"Using OpenAI model: {args.model}")
    else:
        llm = init_vllm(
            model_name=args.model,
            num_gpus=args.num_gpus,
            tp_size=args.tp_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
        )
        print(f"Using vLLM model: {args.model}")

    reflector_model = args.reflector_model or args.model

    # Load lessons & optional feedback map
    all_lessons = load_lessons(args.lessons_db)
    feedback_map = {}
    if args.feedback_path and os.path.exists(args.feedback_path):
        for row in load_jsonl(args.feedback_path):
            key = row.get("index", row.get("id"))
            if key is not None:
                feedback_map[key] = row

    out_buf = []
    line_no = 0            # 1-based position in the *file*
    processed = 0          # how many we have actually processed this run

    for js in load_jsonl(args.input):
        line_no += 1
        idx0 = line_no - 1  # 0-based index in the file
        if idx0 < offset:
            continue
        if limit is not None and processed >= limit:
            break

        messages = js.get("messages") or []
        draft = js.get("output")

        if not messages or draft is None:
            # Reflection-only: skip if no context or no draft
            continue

        key = js.get("index", js.get("id", line_no))
        scope = scope_for(js, args.lesson_scope_field, args.algo)

        # Inject recent lessons (non-leaking)
        picked = select_lessons(all_lessons, scope, args.lessons_topk)
        kb = lessons_block(picked)
        messages_for_review = prepend_lessons(messages, kb)

        conv_text = conversation_to_text(messages_for_review)

        # External feedback (binary flag + note + optional GT)
        ext = feedback_map.get(key) if feedback_map else None
        ext_feedback = (ext and ext.get("feedback"))

        # GT retrieval supports dotted paths
        ext_gt = get_by_dotted(ext, args.gt_field) if ext else None
        gt = ext_gt if ext_gt is not None else get_by_dotted(js, args.gt_field)

        # Reflection loop — up to args.reflection_rounds (default 2).
        critiques = []
        scores = {"em": None}
        current = draft

        if gt is not None:
            scores = eval_with_gt(current, gt)

        for r in range(args.reflection_rounds):
            # Early stop ONLY if EM satisfied.
            if gt is not None and args.gt_stop_on == "em" and scores["em"] == 1:
                break

            # Non-leaking critique (never mentions GT values)
            critique, verdict = reflect_once_nonleak(
                args.use_openai, reflector_model, llm, args.critique_style, conv_text, current
            )
            if ext_feedback:
                critique = critique.strip() + "\n- Additional reviewer note: " + ext_feedback.strip()

            critiques.append({
                "round": r + 1,
                "critique": critique,
                "verdict": ("REVISE" if "REVISE" in critique.upper() else "KEEP")
            })

            # Always attempt the next revision (up to the limit);
            # we do not stop on KEEP unless EM objective already satisfied.
            current = revise_with_feedback(
                args.use_openai, args.model, llm, conv_text, current, critique,
                gt=gt, allow_leak=args.gt_allow_leak,
                temperature=args.temperature, max_tokens=args.max_tokens
            )

            # Re-score (EM only)
            if gt is not None:
                scores = eval_with_gt(current, gt)

        # Persist a lesson if external feedback present
        if ext_feedback:
            append_jsonl(args.lessons_db, {
                "scope": scope,
                "feedback": ext_feedback,
                "example_gt": gt,
                "example_pred": current,
                "ts": int(time.time()),
            })

        entry = js.copy()
        entry.update({
            "algo": args.algo,
            "output": current,
            "reasoning": {
                "critiques": critiques,
                "scores": scores,              # {"em": 0/1}
                "reflection_rounds": args.reflection_rounds,
                "gt_stop_on": args.gt_stop_on, # "em" or "none"
                "gt_allow_leak": args.gt_allow_leak,
                "lessons_used": [L.get("feedback") for L in picked if L.get("feedback")],
                "scope": scope
            },
        })
        out_buf.append(entry)
        processed += 1

        if len(out_buf) >= 200:
            write_jsonl(out_buf, args.output, mode="a")
            out_buf = []

    if out_buf:
        write_jsonl(out_buf, args.output, mode="a")

    t1 = dt.datetime.now()
    print(f"End:   {t1:%Y-%m-%d %H:%M:%S}")
    print(f"Elapsed: {str(t1 - t0).split('.')[0]}")


if __name__ == "__main__":
    main()
