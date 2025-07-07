from os import write
from codeio_utils import *
from utils import *
import copy
import multiprocessing as mp
from tqdm import tqdm
from itertools import islice
import json
from multiprocessing.pool import ThreadPool
from collections import defaultdict

# global vars
python_path = "python"
run_path = "./temp/temp/temp"

import math
import json
import pandas as pd


def check_io_pred_acc(item, algo):
    tol = 0.001  # tolerance for AE

    # unified response helper
    def _make_response(status, message, actual=None, predicted=None):
        return {
            "status":    status,
            "message":   message,
            "actual":    json.dumps(actual)    if actual    is not None else None,
            "predicted": json.dumps(predicted) if predicted is not None else None,
        }

    # validate algorithm
    if algo not in ("lzw", "ae", "rle"):
        return _make_response(
            "no answer",
            f"Unknown algorithm '{algo}'",
        )

    # extract last JSON blob
    output_str = item["output"]
    last_json = extract_last_complete_json(output_str)
    if last_json is None:
        return _make_response(
            "no answer",
            "Fail to extract a complete and valid JSON from the output!",
        )
    if not isinstance(last_json, dict):
        return _make_response(
            "no answer",
            "The last JSON is not a dict!",
        )

    # lookup ground truth
    ori_item = parsed_ios[item["itemid"]]
    ori_io   = ori_item["ios"][item["ioid"]]
    io_pred  = item["io_pred"]

    # synonyms for top‐level and nested keys
    synonyms = {
        "input":  ["input", "uncompressed"],
        "output": ["output", "compressed"],
    }

    # decide primary side and expected type/value
    if io_pred.startswith("output"):
        primary       = "output"
        expected      = ori_io["output"]
        expected_type = list if algo == "lzw" else float
    elif io_pred.startswith("input"):
        primary       = "input"
        # assume only one entry in ori_io["input"]
        expected      = next(iter(ori_io["input"].values()))
        expected_type = str
    else:
        return _error(f"Unknown io_pred '{io_pred}'")

    # find a top‐level key the student used
    found_key = None
    for key in synonyms[primary]:
        if key in last_json:
            found_key = key
            break
    if found_key is None:
        a, b = synonyms[primary]
        return _error(f"No field '{a}' or '{b}' in the last JSON!")

    # extract the actual predicted value, handling nested dicts
    raw_value = last_json[found_key]
    pred      = None
    used_key  = None

    if isinstance(raw_value, dict):
        # they did {"input": {"uncompressed": "..."}}
        for inner_key in synonyms[primary]:
            if inner_key in raw_value and isinstance(raw_value[inner_key], expected_type):
                pred     = raw_value[inner_key]
                used_key = f"{found_key}.{inner_key}"
                break
        if pred is None:
            a, b = synonyms[primary]
            return _error(f"No nested field '{a}' or '{b}' inside '{found_key}'!")
    else:
        # they did {"input": "..."} or {"output": [...]}
        if isinstance(raw_value, expected_type):
            pred     = raw_value
            used_key = found_key
        else:
            return _error(f"Field '{found_key}' is not of type {expected_type.__name__}!")

    # correctness check
    if correct:
        return _make_response(
            "correct",
            f"Correct {io_pred}! (used key '{used_key}')",
            actual=expected,
            predicted=pred,
        )

    # mismatch
    msg = (
        "[Mismatch] Your "
        f"{primary} is not correct!\n"
        f"Given {given_side}:      {json.dumps(ori_io[given_side])}\n"
        f"Predicted {primary}:   {pred!r}  (via '{used_key}')\n"
        f"Actual {primary}:      {expected!r}"
    )
    return _make_response(
        "wrong",
        msg,
        actual=expected,
        predicted=pred,
    )



# def check_io_pred_acc(item, algo):
#     result = {}
#     tol = 0.001  # tolerance for AE algorithm
#     import json
#     # assume is_close is defined elsewhere
#     # from your_module import is_close  

#     # small helpers
#     def _error(msg):
#         return {"status": "no answer", "message": msg}
#     def _wrong(msg):
#         return {"status": "wrong",    "message": msg}
#     def _correct(msg):
#         return {"status": "correct",   "message": msg}

#     # validate algorithm
#     if algo not in ("lzw", "ae"):
#         return _error(f"Unknown algorithm '{algo}'")

#     # 1) extract last JSON
#     last_json = extract_last_complete_json(item["output"])
#     if last_json is None:
#         return _error("Fail to extract a complete and valid JSON from the output!")
#     if not isinstance(last_json, dict):
#         return _error("The last JSON is not an object!")

#     # 2) lookup ground truth
#     ori = parsed_ios[item["itemid"]]["ios"][item["ioid"]]

#     # 3) decide expected value, type, and candidate keys
#     io_pred = item["io_pred"]
#     if algo == "lzw":
#         # LZW: outputs are lists, inputs are strings
#         if io_pred.startswith("output"):
#             expected = ori["output"]
#             expected_type = list
#         elif io_pred.startswith("input"):
#             expected = next(iter(ori["input"].values()))
#             expected_type = str
#         else:
#             return _error(f"Unknown io_pred '{io_pred}'")
#     else:  # algo == "ae"
#         # AE: outputs are floats, inputs are strings
#         if io_pred.startswith("output"):
#             expected = ori["output"]
#             expected_type = float
#         elif io_pred.startswith("input"):
#             expected = next(iter(ori["input"].values()))
#             expected_type = str
#         else:
#             return _error(f"Unknown io_pred '{io_pred}'")

#     # same candidate ordering in both cases
#     candidates = ["output", "input"] if io_pred.startswith("output") else ["input", "output"]

#     # 4) pick the predicted value
#     pred = None
#     used_key = None
#     for key in candidates:
#         if key in last_json and isinstance(last_json[key], expected_type):
#             pred = last_json[key]
#             used_key = key
#             break

#     # 5) error if missing or wrong type
#     if pred is None:
#         if not any(k in last_json for k in candidates):
#             return _error(f"No field '{candidates[0]}' or '{candidates[1]}' in the last JSON!")
#         for k in candidates:
#             if k in last_json and not isinstance(last_json[k], expected_type):
#                 return _error(f"Field '{k}' is not of type {expected_type.__name__}!")
#         return _error("Could not locate a valid I/O field in the last JSON!")

#     # 6) correctness check
#     if io_pred.startswith("input"):
#         # input prediction always exact compare
#         correct = (pred == expected)
#     else:
#         # output prediction: LZW exact, AE via is_close
#         if algo == "lzw":
#             correct = (pred == expected)
#         else:  # ae
#             correct = math.isclose(pred, expected, rel_tol=tol, abs_tol=tol)

#     if correct:
#         return _correct(f"Correct {io_pred}! (used key '{used_key}')")

#     # 7) mismatch message
#     if io_pred.startswith("output"):
#         msg = (
#             "[Mismatch] Your output is not correct!\n"
#             f"Given input:       {json.dumps(ori['input'])}\n"
#             f"Predicted output:  {pred}\n"
#             f"Actual output:     {expected}"
#         )

#     else:
#         msg = (
#             "[Mismatch] Your input is not correct!\n"
#             f"Given output:      {json.dumps(ori['output'])}\n"
#             f"Predicted input:   {pred}\n"
#             f"Actual input:      {expected}"
#         )

#     result["actual"] = expected
#     result["predicted"] = pred

#     return _wrong(msg)


# Function to batch items from an iterator
def batcher(iterable, batch_size):
    """Batch an iterator into lists of length batch_size"""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, batch_size))
        if not chunk:
            break
        yield chunk

def get_total_items_with_wc(filename):
    result = subprocess.run(['wc', '-l', filename], stdout=subprocess.PIPE, text=True)
    total_lines = int(result.stdout.split()[0])  # wc输出的形式是: 行数 文件名, 所以只取第一部分
    return total_lines

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed_file_name", type=str)
    parser.add_argument("--pred_file_name", type=str)
    parser.add_argument("--res_file_name", type=str)
    parser.add_argument("--batchsize", type=int)
    parser.add_argument('--algo', type=str, choices=['lzw', 'ae', 'rle'])
    parser.add_argument('--python_path', type=str, default="python")
    parser.add_argument('--run_path', type=str, default="./temp/temp/temp")
    args = parser.parse_args()

    pred_file_name = args.pred_file_name
    res_file_name = args.res_file_name

    global parsed_ios
    parsed_ios = read_jsonl(args.parsed_file_name)

    global python_path, run_path
    python_path = args.python_path
    run_path = args.run_path

    if not os.path.exists(run_path):
        os.makedirs(run_path, exist_ok=True)

    dt = load_jsonl_yield(pred_file_name)

    if os.path.exists(res_file_name):
        existing = get_total_items_with_wc(res_file_name)
    else:
        existing = 0

    print(f"Existing items: {existing}, skipping them.")
    dt = islice(dt, existing, None)

    total_num_items = get_total_items_with_wc(pred_file_name) - existing
    batchsize = args.batchsize

    pbar = tqdm(total=total_num_items)

    csv_rows = []

    for batch_idx, batch in enumerate(batcher(dt, batchsize)):

        batch_i = [item for item in batch if item['io_pred'].startswith('input')]
        batch_o = [item for item in batch if item['io_pred'].startswith('output')]
        assert len(batch_i) + len(batch_o) == len(batch)

        print("I:", len(batch_i), "O:", len(batch_o))
        batchstat = defaultdict(int)

        for item in batch_i:
            res = check_io_pred_acc(item, algo=args.algo)
            item['res'] = res
            batchstat[res['status']] += 1
            # collect for CSV
            csv_rows.append({
                "model": item.get("model"),
                "temperature": item.get("temperature"),
                "actual": json.loads(res.get("actual", "null")),
                "predicted": json.loads(res.get("predicted", "null"))
            })

        for item in batch_o:
            res = check_io_pred_acc(item, algo=args.algo)
            item['res'] = res
            batchstat[res['status']] += 1
            # collect for CSV
            csv_rows.append({
                "model": item.get("model"),
                "temperature": item.get("temperature"),
                "actual": json.loads(res.get("actual", "null")),
                "predicted": json.loads(res.get("predicted", "null"))
            })


        write_jsonl(batch_i, res_file_name, mode='a')
        write_jsonl(batch_o, res_file_name, mode='a')

        print(f"Wrote a batch of {len(batch)} items.")
        print(f"Batch {batch_idx} status: {batchstat}")
        pbar.update(len(batch))

    pbar.close()

    df_items = pd.DataFrame(csv_rows)
    df_items.to_csv(f"{res_file_name.replace('.jsonl', '.csv')}", index=False)


if __name__ == "__main__":
    main()