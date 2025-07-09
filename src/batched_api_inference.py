import os
os.environ.pop("SSL_CERT_FILE", None)

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

from helper import model_sizes, model_categories

###############################################
max_try_one_call = 2
llm = None
sampling_params = None
use_openai = False
###############################################



def init_llm(model_name, temperature, max_tokens, tp_size):
    global llm, sampling_params
    llm = LLM(model=model_name, tensor_parallel_size=tp_size, dtype=torch.bfloat16, max_model_len=max_tokens, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens, # zero ⇒ greedy
        top_p=1.0,        # consider all tokens
        top_k=1,           # only the single highest-prob token
        stop=["</s>"],
        # top_p=0.95
    )

def format_messages(messages):
    return '\n'.join([f"{m['role'].capitalize()}: {m['content']}" for m in messages]) + "\nAssistant:"

def timer(func):
    def format_time(time_delta):
        hours, remainder = divmod(time_delta.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        print("Start：", start_time.strftime("%Y-%m-%d %H:%M:%S"))
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        print("End：", end_time.strftime("%Y-%m-%d %H:%M:%S"))
        elapsed_time = end_time - start_time
        print("Elapsed：", format_time(elapsed_time))
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
            if use_openai:
                # OPENAI BRANCH
                resp = client.chat.completions.create(
                    model=js.get('model', model),
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response = resp.choices[0].message.content.strip()

            else:
                # VLLM BRANCH
                prompt = format_messages(messages)
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                outputs = llm.generate([text], sampling_params)
                response = outputs[0].outputs[0].text.strip()

            break

        except Exception as e:
            print(f"Error: {e}")
            if i < max_try_one_call - 1:
                time.sleep(5)

    if response:
        js['output'] = response
        js['model'] = model
        js['model_size'] = model_size
        js['model_category'] = model_category
        js['temperature'] = temperature
        js['reasoning'] = None
        with open(output_path, 'a', encoding='utf-8', errors='ignore') as f:
            f.write(json.dumps(js, ensure_ascii=False) + '\n')
        return True
    return False

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
    parser.add_argument("--model", default="Qwen/Qwen3-32B", type=str)  # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  "Qwen/Qwen3-32B-FP8"
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("--max_tokens", default=16384, type=int)
    parser.add_argument("--tp_size", default=4, type=int)
    parser.add_argument("--use_openai", action="store_true", help="If set, use OpenAI API instead of vLLM")

    args = parser.parse_args()
    use_openai = args.use_openai

    args.output = args.output.format(args.model)

    model = args.model
    model_size = model_sizes[args.model]
    model_category = model_categories[args.model]
    temperature = args.temperature
    max_tokens = args.max_tokens
    tp_size = args.tp_size

    # if using OpenAI, configure the key
    if use_openai:
        key = os.getenv("OPENAI_API_KEY")
        print("Using OpenAI API key:", key)
        if not key:
            raise RuntimeError("OpenAI API key required for --use_openai")
        openai.api_key = key
        client = OpenAI(api_key=key)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        init_llm(model, temperature, max_tokens, tp_size)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    process_file(args.input, args.output)
