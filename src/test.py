from codeio_utils import extract_last_complete_json

import pandas as pd

# This will load the entire file into memory
df = pd.read_json("processed_datasets/huffman/demo.jsonl", lines=True)

# Then you can iterate over rows if you like:
for idx, row in df.iterrows():
    item = row.to_dict()
    if item.get("io_pred").startswith("output"):
        output_str = item.get("output", "")
        # print(output_str)
        extracted = extract_last_complete_json(output_str)
        print(extracted)
        print("**" * 20)