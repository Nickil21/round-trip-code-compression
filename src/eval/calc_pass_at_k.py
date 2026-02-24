"""
Compute pass@k scores from a verified JSONL file.

Usage:
    # Single file:
    python src/eval/calc_pass_at_k.py \
        --verified_file data/processed/huffman/codeio_1k_gens_model_gpt_4.1_mini_temp_0.2_n5_verified.jsonl \
        [--k 5]

    # All verified files across all algos (glob):
    python src/eval/calc_pass_at_k.py \
        --verified_file data/processed/*/codeio_1k_gens_model_gpt_oss_20b_temp_0.2_n5_verified.jsonl

    # Everything verified under data/processed/ (default), pivoted by model:
    python src/eval/calc_pass_at_k.py --pivot
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict


# Short display names for the four task types
_TASK_SHORT = {
    "output_execution_prediction":                "o/p pred",
    "output_execution_prediction_with_inversion": "o/p pred+inv",
    "input_execution_prediction":                 "i/p pred",
    "input_execution_prediction_with_inversion":  "i/p pred+inv",
    "OVERALL":                                    "OVERALL",
}

# Canonical task column order
_TASK_ORDER = [
    "output_execution_prediction",
    "output_execution_prediction_with_inversion",
    "input_execution_prediction",
    "input_execution_prediction_with_inversion",
    "OVERALL",
]


def compute_pass_at_k(verified_file: str, k: int):
    groups = defaultdict(list)
    with open(verified_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = (item.get("itemid"), item.get("ioid"), item.get("io_pred"))
            groups[key].append(item.get("res", {}).get("status", ""))

    # pass@k: correct if at least 1 of min(k, n) completions is correct
    by_task_results = defaultdict(list)
    for (itemid, ioid, io_pred), statuses in groups.items():
        correct = "correct" in statuses[:k]
        by_task_results[io_pred].append(correct)

    algo = os.path.basename(os.path.dirname(verified_file))
    model = os.path.basename(verified_file).replace("_verified.jsonl", "").removeprefix("codeio_1k_gens_model_")
    model = re.sub(r"_temp_[\d.]+_n\d+$", "", model)

    rows = []
    for task in sorted(by_task_results):
        results = by_task_results[task]
        n = len(results)
        score = sum(results) / n * 100
        rows.append((algo, model, task, score, n))

    # overall
    all_results = [r for rs in by_task_results.values() for r in rs]
    overall = sum(all_results) / len(all_results) * 100 if all_results else 0.0
    rows.append((algo, model, "OVERALL", overall, len(all_results)))

    return rows


def print_flat(all_rows, k):
    col = max(len(r[2]) for r in all_rows) + 2
    fmt = f"{{:<12}} {{:<{col}}} {{:>9}} {{:>10}}"
    print(fmt.format("algo", "io_pred", f"pass@{k}(%)", "n_samples"))
    print("-" * (12 + col + 22))
    cur_algo = None
    for algo, model, task, score, n in all_rows:
        if algo != cur_algo:
            if cur_algo is not None:
                print()
            cur_algo = algo
        print(fmt.format(algo if task != "OVERALL" else "", task, f"{score:.1f}", n))


def print_pivot(all_rows, k):
    """Pivot: rows = (model, algo), columns = tasks."""
    # Build lookup: (model, algo, task) -> score
    scores = {}
    n_map = {}
    for algo, model, task, score, n in all_rows:
        scores[(model, algo, task)] = score
        n_map[(model, algo, task)] = n

    # Discover tasks present in data, preserve canonical order
    all_tasks = {k[2] for k in scores}
    tasks_present = [t for t in _TASK_ORDER if t in all_tasks]
    col_headers = [_TASK_SHORT.get(t, t) for t in tasks_present]

    # Collect unique (model, algo) pairs, sorted
    seen = {}
    for algo, model, task, _, _ in all_rows:
        seen[(model, algo)] = True
    pairs = sorted(seen.keys())

    # Column widths
    model_w = max(len(m) for m, _ in pairs) + 2
    algo_w  = max(len(a) for _, a in pairs) + 2
    col_w   = max(max(len(h) for h in col_headers), 7) + 2

    n_w = 10
    hdr_fmt = f"{{:<{model_w}}} {{:<{algo_w}}}" + f" {{:>{col_w}}}" * len(tasks_present) + f" {{:>{n_w}}}"
    row_fmt = f"{{:<{model_w}}} {{:<{algo_w}}}" + f" {{:>{col_w}}}" * len(tasks_present) + f" {{:>{n_w}}}"

    print(hdr_fmt.format("model", "algo", *col_headers, "n_samples"))
    print("-" * (model_w + algo_w + col_w * len(tasks_present) + n_w))

    cur_model = None
    for model, algo in pairs:
        if model != cur_model:
            if cur_model is not None:
                print()
            cur_model = model
        cells = []
        for task in tasks_present:
            s = scores.get((model, algo, task))
            cells.append(f"{s:.1f}" if s is not None else "-")
        # n_samples: use OVERALL count if present, else sum across tasks
        n = n_map.get((model, algo, "OVERALL"))
        if n is None:
            n = sum(n_map.get((model, algo, t), 0) for t in tasks_present if t != "OVERALL")
        print(row_fmt.format(model, algo, *cells, n))

    print(f"\n(values are pass@{k} %)")


def main():
    parser = argparse.ArgumentParser(description="Compute pass@k from verified JSONL files.")
    parser.add_argument("--verified_file", nargs="+", default=["data/processed/*/*_verified.jsonl"],
                        help="One or more verified JSONL file paths or glob patterns.")
    parser.add_argument("--k", type=int, default=5,
                        help="k for pass@k (default: 5).")
    parser.add_argument("--pivot", action="store_true",
                        help="Pivot output: rows=(model,algo), columns=tasks.")
    args = parser.parse_args()

    # Expand any glob patterns (useful when the shell doesn't expand them)
    files = []
    for pattern in args.verified_file:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            files.extend(expanded)
        elif not any(c in pattern for c in ('*', '?', '[')):
            # Only treat as a literal path if it contains no glob metacharacters
            files.append(pattern)

    all_rows = []
    for f in files:
        all_rows.extend(compute_pass_at_k(f, args.k))

    if not all_rows:
        print("No data found.")
        return

    if args.pivot:
        print_pivot(all_rows, args.k)
    else:
        print_flat(all_rows, args.k)


if __name__ == "__main__":
    main()
