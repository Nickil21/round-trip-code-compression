from os import write
from codeio_utils import *
from utils import *
import copy
import multiprocessing as mp
from tqdm import tqdm
from itertools import islice
import json
import math
import pandas as pd
from multiprocessing.pool import ThreadPool
from collections import defaultdict

# global vars
python_path = "python"
run_path = "./temp/temp/temp"

ALGO_CONFIG = {
    "lzw":     dict(expected_type=list,          item_type=int,        float_tol=None),
    "huffman": dict(expected_type=list,          item_type=list,        float_tol=None),
    "rle":     dict(expected_type=list,          item_type=(list, tuple), float_tol=None),
    "ae":      dict(expected_type=(int, float),  item_type=None,       float_tol=1e-3),
}

SYNONYMS = {
    "input":  ("input", "uncompressed"),
    "output": ("output", "compressed"),
}

def check_io_pred_acc(item, algo):
    tol = 1e-3

    def _make_response(status, message, actual=None, predicted=None):
        return {
            "status":    status,
            "message":   message,
            "actual":    json.dumps(actual)    if actual    is not None else None,
            "predicted": json.dumps(predicted) if predicted is not None else None,
        }

    # 1) Validate algorithm
    if algo not in ALGO_CONFIG:
        return _make_response("no answer", f"Unknown algorithm '{algo}'")

    # 2) Determine which side to check
    io_pred = item.get("io_pred", "")
    if io_pred.startswith("output"):
        primary = "output"
    elif io_pred.startswith("input"):
        primary = "input"
    else:
        return _make_response("no answer", f"Unknown io_pred '{io_pred}'")

    # 3) Lookup ground truth
    ori      = parsed_ios[item["itemid"]]["ios"][item["ioid"]]
    expected = ori[primary]

    # 4) Extract the student's last JSON
    output_str = item.get("output", "")
    extracted = extract_last_complete_json(output_str)

    # print(f"Extracted JSON: {extracted}")

    if extracted is None:
        return _make_response(
            "no answer",
            "Fail to extract a complete and valid JSON from the output!",
            actual=expected,
            predicted="no answer"
        )

    # If extract_last_complete_json already returned a Python object, use it directly
    if isinstance(extracted, (dict, list)):
        last_json = extracted
    else:
        # Otherwise we got back a JSON string, so parse it
        try:
            last_json = json.loads(extracted)
        except json.JSONDecodeError:
            return _make_response(
                "no answer",
                "Extracted text was not valid JSON!",
                actual=expected,
                predicted="no answer"
            )

    if not isinstance(last_json, dict):
        return _make_response(
            "no answer",
            "The last JSON is not a dict!",
            actual=expected,
            predicted="no answer"
        )
    # 5) Helper to pick top‐level (and optional nested) key
    def extract_predicted(blob, side):
        for top in SYNONYMS[side]:
            if top in blob:
                val, used = blob[top], top
                if isinstance(val, dict):
                    for inner in SYNONYMS[side]:
                        if inner in val:
                            val, used = val[inner], f"{top}.{inner}"
                            break
                    else:
                        raise KeyError(f"No nested field {SYNONYMS[side]} inside '{top}'")
                return val, used
        raise KeyError(f"No field {SYNONYMS[side]} in the last JSON")

    try:
        raw_value, used_key = extract_predicted(last_json, primary)
    except KeyError as e:
        return _make_response("no answer", str(e), actual=expected, predicted="no answer")

    # 6) Decode & type‐check
    if primary == "input":
        if not isinstance(raw_value, str):
            return _make_response(
                "no answer",
                f"Field '{used_key}' is not of type str!",
                actual=expected,
                predicted="no answer"
            )
        pred = raw_value

    else:  # primary == "output"
        cfg = ALGO_CONFIG[algo]

        if algo == "ae":
            if isinstance(raw_value, (int, float)):
                pred = float(raw_value)
            elif isinstance(raw_value, str):
                try:
                    pred = float(raw_value)
                except ValueError:
                    return _make_response(
                        "no answer",
                        f"Field '{used_key}' string value cannot be parsed as float",
                        actual=expected,
                        predicted="no answer"
                    )
            else:
                return _make_response(
                    "no answer",
                    f"Field '{used_key}' is not a float or float‐string!",
                    actual=expected,
                    predicted="no answer"
                )

        elif algo == "huffman":
            # Expect a 3‑element list: [compressed_list, codebook_dict, meta_int]
            # We expect a 3‑element list, but only care about the first (the compressed ints).
            if not (isinstance(raw_value, list) and len(raw_value) >= 1):
                return _make_response(
                    "no answer",
                    f"Expected a list with at least one element for Huffman output, got {type(raw_value).__name__}",
                    actual=expected,
                    predicted="no answer"
                )
            comp = raw_value[0]

            # Validate it’s a list of ints
            if not (isinstance(comp, list) and all(isinstance(x, int) for x in comp)):
                return _make_response(
                    "no answer",
                    f"Compressed output must be list[int], got {comp!r}",
                    actual=expected,
                    predicted="no answer"
                )

            # Use just that for your comparison
            pred = comp

        else:
            # lzw, rle
            if not isinstance(raw_value, cfg["expected_type"]):
                return _make_response(
                    "no answer",
                    f"Field '{used_key}' is not of type {cfg['expected_type'].__name__}!",
                    actual=expected,
                    predicted="no answer"
                )
            if cfg["item_type"] is not None:
                if not all(isinstance(x, cfg["item_type"]) for x in raw_value):
                    return _make_response(
                        "no answer",
                        f"Elements of '{used_key}' must be {cfg['item_type']}!",
                        actual=expected,
                        predicted="no answer"
                    )

            # Normalize RLE lists → tuples
            if algo == "rle":
                try:
                    pred = [tuple(el) for el in raw_value]
                except Exception:
                    return _make_response(
                        "no answer",
                        f"Field '{used_key}' is not a list of 2‑tuples",
                        actual=expected,
                        predicted="no answer"
                    )
            else:
                pred = raw_value

    # 7) Check correctness
    if primary == "input":
        correct = (pred == expected)
    else:
        if algo == "ae":
            correct = math.isclose(pred, expected, rel_tol=tol, abs_tol=tol)
        else:
            correct = (pred == expected)

    # 8) Return result
    if correct:
        return _make_response(
            "correct",
            f"Correct {io_pred}! (used key '{used_key}')",
            actual=expected,
            predicted=pred
        )

    # Mismatch
    other = "input" if primary == "output" else "output"
    msg = (
        "[Mismatch] Your "
        f"{primary} is not correct!\n"
        f"Given {other}:      {json.dumps(ori[other])}\n"
        f"Predicted {primary}:   {pred!r}  (via '{used_key}')\n"
        f"Actual {primary}:      {expected!r}"
    )
    return _make_response(
        "wrong",
        msg,
        actual=expected,
        predicted=pred
    )


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
    parser.add_argument('--algo', type=str, choices=['lzw', 'ae', 'rle', 'huffman'])
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

    # if os.path.exists(res_file_name):
    #     existing = get_total_items_with_wc(res_file_name)
    # else:
    #     existing = 0

    # print(f"Existing items: {existing}, skipping them.")
    # dt = islice(dt, existing, None)

    # total_num_items = get_total_items_with_wc(pred_file_name) - existing
    batchsize = args.batchsize

    # pbar = tqdm(total=total_num_items)

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
            # print(csv_rows)
            # collect for CSV
            csv_rows.append({
                "model": item.get("model"),
                "temperature": item.get("temperature"),
                "io_pred": item.get("io_pred"),
                "category": item.get("category"),
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
                "io_pred": item.get("io_pred"),
                "category": item.get("category"),
                "actual": json.loads(res.get("actual", "null")),
                "predicted": json.loads(res.get("predicted", "null"))
            })


        write_jsonl(batch_i, res_file_name, mode='a')
        write_jsonl(batch_o, res_file_name, mode='a')

        print(f"Wrote a batch of {len(batch)} items.")
        print(f"Batch {batch_idx} status: {batchstat}")
    #     pbar.update(len(batch))

    # pbar.close()

    df_items = pd.DataFrame(csv_rows)
    df_items.to_csv(f"{res_file_name.replace('.jsonl', '.csv')}", index=False)


if __name__ == "__main__":
    main()