from vllm import LLM, SamplingParams
import pickle
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import re
import torch
import os
import random
random.seed(123)

def filter_trace(lines):
    filtered_lines = []
    for line in lines:
        if re.search(r' at 0x[0-9a-fA-F]+', line):
            line = line.replace(re.search(r' at 0x[0-9a-fA-F]+', line).group(), "")
        filtered_lines.append(line) 
    filtered_trace = '\n'.join(filtered_lines)
    return filtered_trace

    return prompt

def get_translation_prompt(question, input, code, trace):
    prompt = f"""Given a question, an input to the question, solution code of the question, and an execution trace of the soluction code, your job is to translate the execution trace into a step-by-step solution to solve the question. Here are some rules for translation:

- Use the exact values from the execution trace during the thought process to ensure the correctness of the thought process.
- Do not write code in your thinking process.
- Pretend you are not given the execution trace and you are solving the question by tracing the code by yourself. So, you should not mention that you are following the execution trace even when you are thinking. 

**Question**

{question}
Given the input, {input}, predict the output of the question by thinking step by step to reach the final output.
Here is a solution code that solves the question:
```
{code}
```

**Execution Trace**

```
{trace}
```
"""
    return prompt

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',        default="processed_datasets/")
    p.add_argument('--algorithm',       required=True)
    p.add_argument('--input_file',      default="codeio_1k_msg_executed_filtered.pkl")
    p.add_argument('--output_file',     default="codeio_1k_msg_executed_filtered_translated.pkl")
    p.add_argument('--translator_model',default='Qwen/Qwen3-32B')
    p.add_argument('--num_gpus',        type=int, default=4)
    args = p.parse_args()

    # load data
    path_in = os.path.join(args.data_dir, args.algorithm, args.input_file)
    with open(path_in, 'rb') as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} examples")

    # build prompts and filter too-long ones early
    tokenizer = AutoTokenizer.from_pretrained(args.translator_model, trust_remote_code=True)
    sampling_params = SamplingParams(max_tokens=8192, temperature=0.6, top_p=0.95, top_k=20, min_p=0)

    pkl_path = os.path.join(args.data_dir, args.algorithm, args.input_file)
    with open(pkl_path, "rb") as f:
        total_data = pickle.load(f)
    print(f"Loaded {len(total_data)} examples from {pkl_path}")

    answer_prefix = "<<< Return value from main_solution: "
    messages_list = []
    for data in tqdm(total_data):
        question = (data['problem_description']+'\n'+data['io_requirements']).replace("\n\n", "\n")
        if isinstance(data['input'], str):
            data['input'] = data['input'].strip()
        trace_lines = data['trace'].strip().split('\n')
        trace = filter_trace(trace_lines)
        answer = trace_lines[-1].split(answer_prefix, 1)[1].strip()
        data['answer'] = answer
        prompt = get_translation_prompt(question=question, input=data['input'], code=data['refcode'], trace=trace)
        messages_list.append([{'role': 'user', 'content': prompt}])
        
    assert len(messages_list) == len(total_data)
    
    inputs = tokenizer.apply_chat_template(
        messages_list,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    print(inputs[0])

    # filter out prompts longer than 40960 tokens which the max length for Qwen3-32B
    org_count = len(inputs)
    total_data = [data for i, data in enumerate(total_data) if len(inputs[i]) <= 32768]
    inputs = [input for input in inputs if len(input) <= 32768]
    print(f"Filtered {org_count - len(inputs)} inputs longer than 32768 tokens")
    print(f"Filtered {org_count - len(total_data)} data accordingly")

    llm = LLM(model=args.translator_model, tensor_parallel_size=args.num_gpus, dtype=torch.bfloat16)
    outputs = llm.generate(inputs, sampling_params)
    
    assert len(outputs) == len(total_data)
    
    for output, data in zip(outputs, total_data):
        data['nl_trace'] = output.outputs[0].text.strip()

    with open(args.data_dir + args.algorithm + os.sep + args.output_file, "wb") as f:
        pickle.dump(total_data, f)
