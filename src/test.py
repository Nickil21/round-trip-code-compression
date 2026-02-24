# from codeio_utils import extract_last_complete_json

# import pandas as pd

# # This will load the entire file into memory
# df = pd.read_json("data/processed/huffman/demo.jsonl", lines=True)

# # Then you can iterate over rows if you like:
# for idx, row in df.iterrows():
#     item = row.to_dict()
#     if item.get("io_pred").startswith("output"):
#         output_str = item.get("output", "")
#         # print(output_str)
#         extracted = extract_last_complete_json(output_str)
#         print(extracted)
#         print("**" * 20)

import pandas as pd

# read the entire file at once
df = pd.read_csv('data/processed/rle/codeio_1k_gens_model_codegemma_7b_it_temp_0.8_n5_verified.csv')
print(df['input'].value_counts())


"""
Batch 0 status: defaultdict(<class 'int'>, {'correct': 162, 'wrong': 1885, 'no answer': 1253})
[81504:] → rle | qwen2.5_7b_instruct @ T=0.8
Please install shortuuid by running 'pip install shortuuid'
I: 1805 O: 1810
Wrote a batch of 3615 items.
Batch 0 status: defaultdict(<class 'int'>, {'correct': 150, 'wrong': 2202, 'no answer': 1263})
[81504:] → rle | mistral_7b_instruct_v0.3 @ T=0.2
Please install shortuuid by running 'pip install shortuuid'
I: 1395 O: 1395
Wrote a batch of 2790 items.
Batch 0 status: defaultdict(<class 'int'>, {'correct': 44, 'no answer': 1595, 'wrong': 1151})
[81504:] → rle | mistral_7b_instruct_v0.3 @ T=0.8
Please install shortuuid by running 'pip install shortuuid'
I: 1830 O: 1830
Wrote a batch of 3660 items.
Batch 0 status: defaultdict(<class 'int'>, {'correct': 52, 'wrong': 1915, 'no answer': 1693})
Traceback (most recent call last):
  File "/lus/lfs1aip2/home/u6cg/nmaveli.u6cg/projects/round-trip-code-compression/src/check_io_pred_acc_mp.py", line 382, in <module>
    main()
  File "/lus/lfs1aip2/home/u6cg/nmaveli.u6cg/projects/round-trip-code-compression/src/check_io_pred_acc_mp.py", line 378, in main
    df_items.to_csv(f"{res_file_name.replace('.jsonl', '.csv')}", index=False)
  File "/lus/lfs1aip2/home/u6cg/nmaveli.u6cg/miniforge3/envs/round-trip-myenv/lib/python3.10/site-packages/pandas/util/_decorators.py", line 333, in wrapper
    return func(*args, **kwargs)
  File "/lus/lfs1aip2/home/u6cg/nmaveli.u6cg/miniforge3/envs/round-trip-myenv/lib/python3.10/site-packages/pandas/core/generic.py", line 3986, in to_csv
    return DataFrameRenderer(formatter).to_csv(
  File "/lus/lfs1aip2/home/u6cg/nmaveli.u6cg/miniforge3/envs/round-trip-myenv/lib/python3.10/site-packages/pandas/io/formats/format.py", line 1014, in to_csv
    csv_formatter.save()
  File "/lus/lfs1aip2/home/u6cg/nmaveli.u6cg/miniforge3/envs/round-trip-myenv/lib/python3.10/site-packages/pandas/io/formats/csvs.py", line 270, in save
    self._save()
  File "/lus/lfs1aip2/home/u6cg/nmaveli.u6cg/miniforge3/envs/round-trip-myenv/lib/python3.10/site-packages/pandas/io/formats/csvs.py", line 275, in _save
    self._save_body()
  File "/lus/lfs1aip2/home/u6cg/nmaveli.u6cg/miniforge3/envs/round-trip-myenv/lib/python3.10/site-packages/pandas/io/formats/csvs.py", line 313, in _save_body
    self._save_chunk(start_i, end_i)
  File "/lus/lfs1aip2/home/u6cg/nmaveli.u6cg/miniforge3/envs/round-trip-myenv/lib/python3.10/site-packages/pandas/io/formats/csvs.py", line 324, in _save_chunk
    libwriters.write_csv_rows(
  File "pandas/_libs/writers.pyx", line 73, in pandas._libs.writers.write_csv_rows
_csv.Error: need to escape, but no escapechar set
[81504:] → rle | yi_coder_9b_chat @ T=0.2
Please install shortuuid by running 'pip install shortuuid'
I: 1990 O: 1990
Wrote a batch of 3980 items.
Batch 0 status: defaultdict(<class 'int'>, {'correct': 161, 'no answer': 2452, 'wrong': 1367})
[81504:] → rle | yi_coder_9b_chat @ T=0.8
Please install shortuuid by running 'pip install shortuuid'
I: 1990 O: 1990
Wrote a batch of 3980 items.
Batch 0 status: defaultdict(<class 'int'>, {'no answer': 2510, 'correct': 126, 'wrong': 1344})
[81504:] → rle | codegemma_7b_it @ T=0.2
Please install shortuuid by running 'pip install shortuuid'
I: 1990 O: 1990
Wrote a batch of 3980 items.
Batch 0 status: defaultdict(<class 'int'>, {'wrong': 1250, 'correct': 30, 'no answer': 2700})
[81504:] → rle | codegemma_7b_it @ T=0.8
Please install shortuuid by running 'pip install shortuuid'
I: 1990 O: 1990
Wrote a batch of 3980 items.
Batch 0 status: defaultdict(<class 'int'>, {'no answer': 2629, 'wrong': 1326, 'correct': 25})
[81504:] → rle | llama_3.2_1b_instruct @ T=0.2
Please install shortuuid by running 'pip install shortuuid'
I: 1990 O: 1990
Wrote a batch of 3980 items.
Batch 0 status: defaultdict(<class 'int'>, {'no answer': 3891, 'wrong': 89})
[81504:] → rle | llama_3.2_1b_instruct @ T=0.8
Please install shortuuid by running 'pip install shortuuid'
I: 1990 O: 1990
Wrote a batch of 3980 items.
Batch 0 status: defaultdict(<class 'int'>, {'no answer': 3631, 'wrong': 349})
[81504:] → rle | deepseek_r1_distill_qwen_1.5b @ T=0.2
"""