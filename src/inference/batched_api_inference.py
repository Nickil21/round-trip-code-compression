#!/usr/bin/env python3
import os
os.environ.pop("SSL_CERT_FILE", None)

import warnings
from dotenv import load_dotenv
load_dotenv()

import asyncio
import openai
from openai import OpenAI, AsyncOpenAI
import datetime
import json
from argparse import ArgumentParser

import time
from tqdm import tqdm

# GPU/vLLM imports are optional — the OpenAI path does not need them.
try:
    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind  # noqa: F401
    from transformers import AutoTokenizer
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Device[0]:", torch.cuda.get_device_name(0))
except ImportError as _gpu_err:
    print(f"[warn] GPU/vLLM imports unavailable ({_gpu_err}); OpenAI-only mode.")
    torch = None
    LLM = None
    SamplingParams = None
    AutoTokenizer = None

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

from core.helper import model_sizes, model_categories
from core.utils import hub_root_from_cache_dir, resolve_local_snapshot_dir, load_jsonl_yield

###############################################
max_try_one_call = 2
llm            = None
sampling_params = None
tokenizer      = None   # Fix: declared at module level; set in __main__ for vLLM path
client         = None   # sync OpenAI client (used for non-async fallback / key check)
async_client   = None   # AsyncOpenAI client (used for production async path)
use_openai     = False
workers        = 16     # max concurrent async OpenAI calls (set from --workers)
api_model      = None   # model name sent to the OpenAI API (strips "openai/" prefix if present)
###############################################


def init_llm(model_name, temperature, max_tokens, tp_size, num_completions,
             *, download_dir=None, lora_adapter=None, lora_scaling=1.0):
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
        # Some container builds of vLLM have a bug where the default
        # config_format="auto" is not handled and raises ValueError.
        # Passing "hf" explicitly avoids that default and works for all
        # HuggingFace-format models (Qwen, Llama, etc.).
        config_format="hf",
        # Reuse KV-cache for shared prompt prefixes (system prompt, task description).
        enable_prefix_caching=True,
        # Interleave prefill and decode phases; improves throughput on mixed-length batches.
        enable_chunked_prefill=True,
    )

    # Flags that may not exist in older vLLM builds — stripped one-by-one on TypeError.
    _OPTIONAL_ENGINE_FLAGS = (
        "enable_prefix_caching", "enable_chunked_prefill", "config_format",
    )

    def _build_with_compat_fallback(**kwargs):
        """
        Try building the LLM.  On TypeError, remove unknown optional flags one at a
        time and retry until either the build succeeds or no optional flags remain.
        """
        for _ in range(len(_OPTIONAL_ENGINE_FLAGS) + 1):
            try:
                return _try_build_llm(**kwargs)
            except TypeError as te:
                msg = str(te)
                removed = False
                for flag in _OPTIONAL_ENGINE_FLAGS:
                    if flag in msg and flag in kwargs:
                        print(f"[compat] vLLM does not support '{flag}'; removing and retrying.")
                        kwargs.pop(flag)
                        removed = True
                        break
                if not removed:
                    raise

    try:
        print(
            f"Loading {model_name} (bf16), TP={tp_size}"
            + (f", LoRA={os.path.basename(lora_adapter)}" if lora_adapter else "")
        )
        llm = _build_with_compat_fallback(**engine_kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠️  OOM during bf16 init. Falling back to fp16 (try reducing --max_tokens / increasing TP).")
            engine_kwargs["dtype"] = torch.float16
            llm = _build_with_compat_fallback(**engine_kwargs)
        else:
            raise

    print("✅ LLM ready:", llm)


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


def check_exists(line):
    return "output" in line and line["output"] is not None


def _build_entries(js, trial_responses):
    """Convert (idx, text) pairs into output dicts ready to write."""
    entries = []
    for idx, text in trial_responses:
        entry = js.copy()
        entry.update({
            'output':         text,
            'output_index':   idx,
            'model':          model,
            'model_size':     model_size,
            'model_category': model_category,
            'temperature':    temperature,
            'reasoning':      None,
        })
        entries.append(entry)
    return entries


async def _run_openai_async(data, output_file_path):
    """Fire all OpenAI calls concurrently (bounded by semaphore), write as they complete."""
    semaphore = asyncio.Semaphore(workers)

    async def call_one(js):
        async with semaphore:
            for i in range(max_try_one_call):
                try:
                    resp = await async_client.chat.completions.create(
                        model=api_model,
                        messages=js['messages'],
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                        n=num_completions,
                        stop=["[/ANSWER]"],
                    )
                    return js, [(idx, c.message.content.strip()) for idx, c in enumerate(resp.choices)]
                except Exception as e:
                    print(f"Error: {e}")
                    if i < max_try_one_call - 1:
                        await asyncio.sleep(5)
        return js, []

    good = bad = 0
    if out_dir := os.path.dirname(output_file_path):
        os.makedirs(out_dir, exist_ok=True)

    tasks = [asyncio.ensure_future(call_one(js)) for js in data]
    pbar = tqdm(total=len(tasks))
    with open(output_file_path, 'a', encoding='utf-8') as f:
        for coro in asyncio.as_completed(tasks):
            js, trial_responses = await coro
            entries = _build_entries(js, trial_responses)
            if entries:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                good += 1
            else:
                bad += 1
            pbar.update(1)
    pbar.close()
    return good, bad


def _process_openai_async(data, output_file_path):
    """Sync entry-point: run the async OpenAI pipeline to completion."""
    return asyncio.run(_run_openai_async(data, output_file_path))


VLLM_CHUNK_SIZE = 50  # flush to disk after every N prompts; allows crash recovery


def _process_vllm_batch(data, output_file_path):
    """Generate in chunks and flush after each chunk so partial results survive crashes."""
    if out_dir := os.path.dirname(output_file_path):
        os.makedirs(out_dir, exist_ok=True)

    good = bad = 0
    pbar = tqdm(total=len(data))

    with open(output_file_path, 'a', encoding='utf-8') as f:
        for chunk_start in range(0, len(data), VLLM_CHUNK_SIZE):
            chunk = data[chunk_start : chunk_start + VLLM_CHUNK_SIZE]
            prompts = [
                tokenizer.apply_chat_template(
                    js['messages'], tokenize=False, add_generation_prompt=True
                )
                for js in chunk
            ]
            all_outputs = llm.generate(prompts, sampling_params)
            for js, output in zip(chunk, all_outputs):
                trial_responses = [(idx, out.text.strip()) for idx, out in enumerate(output.outputs)]
                entries = _build_entries(js, trial_responses)
                if entries:
                    for entry in entries:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    good += 1
                else:
                    bad += 1
            f.flush()
            pbar.update(len(chunk))

    pbar.close()
    return good, bad


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
                except Exception:
                    pass

    data = []
    for index, js in enumerate(inlines):
        if index in exist:
            continue
        js['index'] = index
        if js.get('messages'):
            data.append(js)

    print("Total lines to process (excluding existing):", len(data))

    if use_openai:
        print(f"[openai] Running {len(data)} calls with {workers} concurrent async requests")
        good, bad = _process_openai_async(data, output_file_path)
    else:
        good, bad = _process_vllm_batch(data, output_file_path)

    print(f"Finished. ✅ Good: {good}, ❌ Bad: {bad}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input",           default="xx",             type=str)
    parser.add_argument("--output",          default="xx",             type=str)
    parser.add_argument("--model",           default="Qwen/Qwen3-32B", type=str)
    parser.add_argument("--temperature",     default=0.2,              type=float)
    parser.add_argument("--max_tokens",      default=16384,            type=int)
    parser.add_argument("--tp_size",         default=1,                type=int)
    parser.add_argument("--num_completions", default=1,                type=int,
                        help="Number of completions to generate per prompt")
    parser.add_argument("--use_openai",      action="store_true",
                        help="If set, use OpenAI API instead of vLLM")
    parser.add_argument("--hf_offline",      action="store_true",
                        help="Force Hugging Face offline mode (use local cache only)")
    parser.add_argument("--cache_dir",       default=None,             type=str,
                        help="HF cache directory (e.g., ~/.cache/huggingface/hub)")
    parser.add_argument("--lora_adapter",    default=None,             type=str,
                        help="Path to a PEFT LoRA adapter (e.g., .../lora/sft)")
    parser.add_argument("--lora_scaling",    default=1.0,              type=float,
                        help="Scaling factor for LoRA (default 1.0)")
    parser.add_argument("--workers",         default=16,               type=int,
                        help="Parallel threads for OpenAI API calls (default 16)")

    args = parser.parse_args()
    use_openai = args.use_openai
    workers    = args.workers

    # Allow {model} in output path templates
    args.output = args.output.format(args.model)

    # Display name for outputs: append +lora if adapter is provided
    lora_adapter = args.lora_adapter
    lora_scaling = args.lora_scaling
    if lora_adapter:
        model = f"{args.model}+lora"
    else:
        model = args.model

    # API-facing model name: strip "openai/" namespace prefix if present.
    # e.g. "openai/gpt-oss-20b" → "gpt-oss-20b" for the actual API call,
    # while keeping the full name in `model` for metadata/output fields.
    _base = args.model[len("openai/"):] if args.model.startswith("openai/") else args.model
    api_model = f"{_base}+lora" if lora_adapter else _base

    # Metadata keyed by base model id
    model_size     = model_sizes[args.model]
    model_category = model_categories[args.model]
    temperature    = args.temperature
    max_tokens     = args.max_tokens
    tp_size        = args.tp_size
    num_completions = args.num_completions

    # ---------- Offline env wiring ----------
    default_hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    cache_dir = args.cache_dir or default_hf_cache
    hub_root  = hub_root_from_cache_dir(cache_dir)

    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"]      = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("[offline] HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1")

    os.environ.setdefault(
        "HF_HOME",
        cache_dir if not cache_dir.endswith("/hub") else os.path.dirname(cache_dir),
    )
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub_root)
    print(f"[hf] Using cache root: {hub_root}")
    # ----------------------------------------

    # Resolve local snapshot path if given a repo id like "org/model"
    model_to_load  = args.model
    local_snapshot = None
    if "/" in args.model and not os.path.isdir(args.model):
        local_snapshot = resolve_local_snapshot_dir(args.model, hub_root)
        if args.hf_offline:
            if local_snapshot and os.path.isdir(local_snapshot):
                print(f"[hf-offline] Using local snapshot for '{args.model}': {local_snapshot}")
                model_to_load = local_snapshot
            else:
                raise SystemExit(
                    f"[hf-offline] Could not find local snapshot for '{args.model}' under '{hub_root}'. "
                    "Please 'huggingface-cli download' the model first."
                )
        else:
            if local_snapshot and os.path.isdir(local_snapshot):
                print(f"[hf] Found local snapshot for '{args.model}': {local_snapshot}")
                model_to_load = local_snapshot

    if use_openai:
        key = os.getenv("OPENAI_API_KEY")
        print("Using OpenAI API key:", key[:4] + "..." if key else None)
        if not key:
            raise RuntimeError("OpenAI API key required for --use_openai")
        openai.api_key = key
        client       = OpenAI(api_key=key, max_retries=5)
        async_client = AsyncOpenAI(api_key=key, max_retries=5)
    else:
        # Fix: load tokenizer so process_line can apply the model's chat template
        tok_src   = model_to_load if os.path.isdir(model_to_load) else args.model
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
            download_dir=hub_root,
            lora_adapter=lora_adapter,
            lora_scaling=lora_scaling,
        )

    # Fix: guard against empty dirname when output file is in the current directory
    if out_dir := os.path.dirname(args.output):
        os.makedirs(out_dir, exist_ok=True)
    process_file(args.input, args.output)
