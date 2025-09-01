#!/usr/bin/env python3
import os
os.environ.pop("SSL_CERT_FILE", None)

import warnings
from dotenv import load_dotenv
load_dotenv()

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import openai
from openai import OpenAI
import datetime
import json
from argparse import ArgumentParser

import torch
import time
from tqdm import tqdm

# vLLM import kept for compatibility; not used directly below,
# but harmless to keep if other code reads it.
from vllm.sampling_params import RequestOutputKind  # noqa: F401

from helper import model_sizes, model_categories


print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device[0]:", torch.cuda.get_device_name(0))

###############################################
max_try_one_call = 2
llm = None
sampling_params = None
use_openai = False
###############################################

# ---------- Offline HF helpers ----------
def hub_root_from_cache_dir(cache_dir: str) -> str:
    if not cache_dir:
        return os.path.expanduser("~/.cache/huggingface/hub")
    return cache_dir if cache_dir.endswith("/hub") else os.path.join(cache_dir, "hub")

def resolve_local_snapshot_dir(repo_id: str, hub_root: str):
    """
    Map a repo like 'Qwen/Qwen3-32B' to the newest local snapshot directory in the HF cache.
    Returns the snapshot path or None if not found.
    """
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
# ----------------------------------------


def init_llm(model_name, temperature, max_tokens, tp_size, num_completions, *, download_dir=None, lora_adapter=None, lora_scaling=1.0):
    """
    Initialize vLLM engine.
    - Tries bf16 first; on OOM, falls back to fp16.
    - If lora_adapter is provided, attempts to attach the PEFT adapter using
      two common vLLM APIs (support varies by version).
    """
    global llm, sampling_params

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95,
        stop=["[/ANSWER]"],
        include_stop_str_in_output=True,
        n=num_completions,
        seed=42,
    )

    def _try_build_llm(**engine_kwargs):
        """Attempt to construct LLM with LoRA across vLLM API variants."""
        if lora_adapter:
            # Variant A (seen in some vLLM releases)
            try:
                return LLM(
                    lora_modules=[{"path": lora_adapter, "scaling": float(lora_scaling)}],
                    **engine_kwargs,
                )
            except TypeError:
                pass
            # Variant B (older/alternative style)
            try:
                return LLM(
                    enable_lora=True,
                    max_loras=1,
                    lora_modules=[("default", lora_adapter)],
                    **engine_kwargs,
                )
            except TypeError:
                warnings.warn(
                    "LoRA not supported by this vLLM build (both API variants failed). "
                    "Proceeding without LoRA."
                )
        # No LoRA
        return LLM(**engine_kwargs)

    engine_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tp_size,
        dtype=torch.bfloat16,
        max_model_len=max_tokens,
        trust_remote_code=True,
        download_dir=download_dir,
    )

    try:
        print(
            f"Loading {model_name} (bf16), TP={tp_size}"
            + (f", LoRA={os.path.basename(lora_adapter)}" if lora_adapter else "")
        )
        llm = _try_build_llm(**engine_kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠️  OOM during bf16 init. Falling back to fp16 (try reducing --max_tokens / increasing TP).")
            engine_kwargs["dtype"] = torch.float16
            llm = _try_build_llm(**engine_kwargs)
        else:
            raise

    print("✅ LLM ready:", llm)


def format_messages(messages):
    return '\n'.join([f"{m['role'].capitalize()}: {m['content']}" for m in messages]) + "\nAssistant:"

def timer(func):
    def format_time(time_delta):
        hours, remainder = divmod(time_delta.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        print("Start:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        print("End:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
        elapsed_time = end_time - start_time
        print("Elapsed:", format_time(elapsed_time))
        return result
    return wrapper

def load_jsonl_yield(path):
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                pass

def check_exists(line):
    return "output" in line and line["output"] is not None

def process_line(js, output_path):
    messages = js['messages']
    response = None

    for i in range(max_try_one_call):
        try:
            # Call the model
            trial_responses = []
            if use_openai:
                resp = client.chat.completions.create(
                    model=js.get('model', model),
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=num_completions,
                    stop=["[/ANSWER]"],
                )
                for idx, choice in enumerate(resp.choices):
                    trial_responses.append((idx, choice.message.content.strip()))
            else:
                prompt = format_messages(messages)
                outputs = llm.generate([prompt], sampling_params)
                for idx, out in enumerate(outputs[0].outputs):
                    trial_responses.append((idx, out.text.strip()))

            break

        except Exception as e:
            print(f"Error: {e}")
            if i < max_try_one_call - 1:
                time.sleep(5)

    # Write out
    for idx, text in trial_responses:
        entry = js.copy()
        entry.update({
            'output': text,
            'output_index': idx,
            'model': model,
            'model_size': model_size,
            'model_category': model_category,
            'temperature': temperature,
            'reasoning': None,
        })
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    return len(trial_responses) > 0



@timer
def process_file(input_file_path, output_file_path):
    print(f"Using {'OpenAI' if use_openai else 'vLLM'} model: {model}")
    inlines = load_jsonl_yield(input_file_path)

    exist = set()
    if os.path.exists(output_file_path):
        with open(output_file_path) as f:
            for line in f:
                try:
                    line = json.loads(line)
                    if check_exists(line):
                        exist.add(line['index'])
                except:
                    pass

    data = []
    for index, js in enumerate(inlines):
        if index in exist:
            continue
        js['index'] = index
        if js.get('messages'):
            data.append(js)

    print("Total lines to process (excluding existing):", len(data))

    good = 0
    bad = 0
    for js in tqdm(data):
        if process_line(js, output_file_path):
            good += 1
        else:
            bad += 1

    print(f"Finished. ✅ Good: {good}, ❌ Bad: {bad}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", default="xx", type=str)
    parser.add_argument("--output", default="xx", type=str)
    parser.add_argument("--model", default="Qwen/Qwen3-32B", type=str)
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("--max_tokens", default=16384, type=int)
    parser.add_argument("--tp_size", default=1, type=int)
    parser.add_argument("--num_completions", default=1, type=int, help="Number of completions to generate per prompt")
    parser.add_argument("--use_openai", action="store_true", help="If set, use OpenAI API instead of vLLM")
    # Offline/HF cache controls
    parser.add_argument("--hf_offline", action="store_true", help="Force Hugging Face offline mode (use local cache only)")
    parser.add_argument("--cache_dir", default=None, type=str, help="HF cache directory (e.g., ~/.cache/huggingface/hub)")
    # NEW: LoRA adapter path (skip disk export/merge)
    parser.add_argument("--lora_adapter", default=None, type=str, help="Path to a PEFT LoRA adapter (e.g., .../lora/sft)")
    parser.add_argument("--lora_scaling", default=1.0, type=float, help="Scaling factor for LoRA (default 1.0)")

    args = parser.parse_args()
    use_openai = args.use_openai

    # Allow {model} in output path templates
    args.output = args.output.format(args.model)

    # Display name for outputs: append +lora if adapter is provided
    lora_adapter = args.lora_adapter
    lora_scaling = args.lora_scaling
    if lora_adapter:
        model = f"{args.model}+lora"
    else:
        model = args.model

    # Metadata keyed by base model id
    model_size = model_sizes[args.model]
    model_category = model_categories[args.model]
    temperature = args.temperature
    max_tokens = args.max_tokens
    tp_size = args.tp_size
    num_completions = args.num_completions

    # ---------- Offline env wiring ----------
    # Establish a concrete cache root
    default_hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    cache_dir = args.cache_dir or (default_hf_cache if os.path.isdir(default_hf_cache) else default_hf_cache)
    hub_root = hub_root_from_cache_dir(cache_dir)

    # Set env so both transformers & vLLM look locally
    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("[offline] HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1")

    # HF_HOME should be the parent of ".../hub"
    os.environ.setdefault("HF_HOME", cache_dir if not cache_dir.endswith("/hub") else os.path.dirname(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub_root)
    print(f"[hf] Using cache root: {hub_root}")
    # ---------------------------------------

    # Resolve local snapshot path if we appear to be given a repo id like "org/model"
    model_to_load = args.model
    local_snapshot = None
    if "/" in args.model and not os.path.isdir(args.model):
        local_snapshot = resolve_local_snapshot_dir(args.model, hub_root)
        if args.hf_offline:
            if local_snapshot and os.path.isdir(local_snapshot):
                print(f"[hf-offline] Using local snapshot for '{args.model}': {local_snapshot}")
                model_to_load = local_snapshot
            else:
                raise SystemExit(f"[hf-offline] Could not find local snapshot for '{args.model}' under '{hub_root}'. "
                                 "Please 'huggingface-cli download' the model first.")
        else:
            # Not strictly offline: still prefer local snapshot if present
            if local_snapshot and os.path.isdir(local_snapshot):
                print(f"[hf] Found local snapshot for '{args.model}': {local_snapshot}")
                model_to_load = local_snapshot

    # if using OpenAI, configure the key
    if use_openai:
        key = os.getenv("OPENAI_API_KEY")
        print("Using OpenAI API key:", key)
        if not key:
            raise RuntimeError("OpenAI API key required for --use_openai")
        openai.api_key = key
        client = OpenAI(api_key=key)
    else:
        # Tokenizer from base model/snapshot (correct even when using LoRA)
        tok_src = model_to_load if os.path.isdir(model_to_load) else args.model
        tokenizer = AutoTokenizer.from_pretrained(
            tok_src,
            trust_remote_code=True,
            local_files_only=bool(args.hf_offline),
            cache_dir=os.path.dirname(hub_root) if hub_root.endswith("/hub") else hub_root,
        )
        # Initialize vLLM (attach LoRA if provided)
        init_llm(
            model_to_load,
            temperature,
            max_tokens,
            tp_size,
            num_completions,
            download_dir=hub_root,     # vLLM will read models from this cache root
            lora_adapter=lora_adapter,
            lora_scaling=lora_scaling,
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    process_file(args.input, args.output)