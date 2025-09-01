import os
import pickle
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

def get_training_prompt(question, code, input):
    
    prompt = f"""You are given a question that requires some input and output variables as follows:

{question}

----

Here is the solution code that solves the question:

```
{code}
```

Given the following input:

{input}

Predict the output of the question by tracing the given solution code step by step to reach the final output.
"""

    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Data Construction for SFT')
    parser.add_argument('--data_dir',      type=str, default="processed_datasets/")
    parser.add_argument('--algorithm',     type=str, required=True)
    parser.add_argument('--input_file',    type=str, default="codeio_1k_msg_executed_filtered_translated.pkl")
    parser.add_argument('--output_file',   type=str, default="training_data_sft.jsonl")
    parser.add_argument('--trained_model', type=str, default='Qwen/Qwen2-7B-Instruct')
    args = parser.parse_args()

    # Load your pickle
    pkl_path = os.path.join(args.data_dir, args.algorithm, args.input_file)
    with open(pkl_path, "rb") as f:
        total_data = pickle.load(f)
    print(f"Loaded {len(total_data)} examples from {pkl_path}")

    print(total_data[0]['nl_trace'])
    # Prepare training data 
    messages_list = []
    for data in total_data:
        question = (data['problem_description']+'\n'+data['io_requirements']).replace("\n\n", "\n")
        prompt = get_training_prompt(question=question, code=data['refcode'], input=data['input'])
        # if '</think>' not in data['nl_trace']: # filter out data where thinking is not finished
        #     continue
        # nl_trace = data['nl_trace'].split('</think>')[1].strip()
        nl_trace = data['nl_trace'].strip()
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": nl_trace},
        ]
        messages_list.append(messages)
        
    print(f"Total messages constructed: {len(messages_list)}")

    tokenizer = AutoTokenizer.from_pretrained(args.trained_model)
    inputs = tokenizer.apply_chat_template(
        messages_list,
        tokenize=False,
        add_generation_prompt=True
    ) 
    org_count = len(inputs)
    messages_list = [data for i, data in enumerate(messages_list) if len(inputs[i]) <= 32768]  # filter out messages longer than 32768 tokens
    print(f"{org_count-len(messages_list)} inputs longer than 32768 tokens")
    print(f"Final Data Size: {len(messages_list)}")
    
    import json
    with open(args.output_file, 'w') as f:
        for message in messages_list:
            f.write(json.dumps({'messages': message}) + '\n')