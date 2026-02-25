"""
Compare original vs. ablation accuracy from verified CSV files.

Reads two verified CSV files (original format and ablation alternative format),
groups by io_pred, computes accuracy as correct/(correct+wrong+no_answer), and
prints a side-by-side comparison table.

Usage:
    python src/ablation/compare_ablation_results.py \
        --original_csv data/processed/huffman/codeio_1k_gens_..._verified.csv \
        --ablation_csv data/processed/huffman/codeio_alt_base64_gens_..._verified.csv \
        --algo huffman \
        --variant base64 \
        [--output_csv results/huffman_base64_comparison.csv]
"""

import argparse
import sys

import pandas as pd


# io_pred labels to display (shorter)
_LABEL_MAP = {
    "output_execution_prediction":              "output_execution_prediction",
    "output_execution_prediction_with_inversion": "output_execution_prediction_with_inv",
    "input_execution_prediction":               "input_execution_prediction",
    "input_execution_prediction_with_inversion":  "input_execution_prediction_with_inv",
}


def _accuracy(df: pd.DataFrame) -> float:
    """Fraction correct over all rows (correct / total)."""
    if len(df) == 0:
        return float("nan")
    n_correct = (df["status"] == "correct").sum()
    return n_correct / len(df)


def _acc_by_io_pred(df: pd.DataFrame) -> dict:
    """Return {io_pred_label: accuracy_float} for every group in df."""
    result = {}
    for io_pred, grp in df.groupby("io_pred"):
        label = _LABEL_MAP.get(io_pred, io_pred)
        result[label] = _accuracy(grp)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare original vs. ablation accuracy."
    )
    parser.add_argument("--original_csv", required=True,
                        help="Path to the original (numeric format) verified CSV.")
    parser.add_argument("--ablation_csv", required=True,
                        help="Path to the ablation (alternative format) verified CSV.")
    parser.add_argument("--algo",    required=True,
                        help="Algorithm name, e.g. huffman, lzw, rle.")
    parser.add_argument("--variant", required=True,
                        help="Variant name, e.g. base64, hex, csv, compact.")
    parser.add_argument("--output_csv", default=None,
                        help="Optional path to write the comparison table as CSV.")
    parser.add_argument(
        "--task_family",
        default="output",
        choices=["output", "input", "both"],
        help="Which io_pred family to compare: output, input, or both.",
    )
    args = parser.parse_args()

    # ── Load CSVs ────────────────────────────────────────────────────────────
    try:
        orig_df = pd.read_csv(args.original_csv)
    except FileNotFoundError:
        sys.exit(f"[error] Original CSV not found: {args.original_csv}")
    try:
        abla_df = pd.read_csv(args.ablation_csv)
    except FileNotFoundError:
        sys.exit(f"[error] Ablation CSV not found: {args.ablation_csv}")

    # Normalise status column (lower-case, strip whitespace)
    for df in (orig_df, abla_df):
        df["status"] = df["status"].astype(str).str.strip().str.lower()

    # ── Filter tasks by family ────────────────────────────────────────────────
    output_tasks = {
        "output_execution_prediction",
        "output_execution_prediction_with_inversion",
    }
    input_tasks = {
        "input_execution_prediction",
        "input_execution_prediction_with_inversion",
    }
    if args.task_family == "output":
        selected_tasks = output_tasks
    elif args.task_family == "input":
        selected_tasks = input_tasks
    else:
        selected_tasks = output_tasks | input_tasks

    orig_df_out  = orig_df[orig_df["io_pred"].isin(selected_tasks)].copy()
    abla_df_out  = abla_df[abla_df["io_pred"].isin(selected_tasks)].copy()

    # ── Accuracy tables ───────────────────────────────────────────────────────
    orig_acc  = _acc_by_io_pred(orig_df_out)
    abla_acc  = _acc_by_io_pred(abla_df_out)

    # Determine model / temperature from data (best-effort)
    model_name = _first_val(abla_df, "model") or _first_val(orig_df, "model") or "?"
    temp_val   = _first_val(abla_df, "temperature") or _first_val(orig_df, "temperature") or "?"

    # ── Print comparison table ────────────────────────────────────────────────
    header = (
        f"\nComparison: {args.algo} | original vs {args.variant} "
        f"({model_name}, T={temp_val})"
    )
    print(header)
    print()

    col_w = max(len(k) for k in (list(orig_acc) + list(abla_acc) + ["io_pred"])) + 2
    fmt = f"{{:<{col_w}}} | {{:>12}} | {{:>12}} | {{:>8}}"
    print(fmt.format("io_pred", "original acc", f"{args.variant} acc", "delta"))
    print("-" * (col_w + 40))

    all_labels = sorted(set(list(orig_acc) + list(abla_acc)))
    rows = []
    for label in all_labels:
        oa = orig_acc.get(label, float("nan"))
        aa = abla_acc.get(label, float("nan"))
        if not (oa != oa) and not (aa != aa):   # neither is NaN
            delta = aa - oa
            delta_str = f"{delta:+.1f}" if not (delta != delta) else "N/A"
        else:
            delta = float("nan")
            delta_str = "N/A"
        oa_pct = f"{oa*100:.1f}%" if not (oa != oa) else "N/A"
        aa_pct = f"{aa*100:.1f}%" if not (aa != aa) else "N/A"
        print(fmt.format(label, oa_pct, aa_pct, delta_str))
        rows.append({
            "algo":         args.algo,
            "variant":      args.variant,
            "io_pred":      label,
            "original_acc": oa,
            f"{args.variant}_acc": aa,
            "delta":        delta,
        })

    # Overall accuracy (all output tasks combined)
    overall_orig = _accuracy(orig_df_out)
    overall_abla = _accuracy(abla_df_out)
    overall_delta = overall_abla - overall_orig
    print("-" * (col_w + 40))
    print(
        fmt.format(
            "OVERALL",
            f"{overall_orig*100:.1f}%",
            f"{overall_abla*100:.1f}%",
            f"{overall_delta:+.1f}",
        )
    )

    # ── Optionally write CSV ─────────────────────────────────────────────────
    if args.output_csv:
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"\nComparison table saved to {args.output_csv}")


def _first_val(df: pd.DataFrame, col: str):
    if col in df.columns and len(df) > 0:
        v = df[col].dropna()
        return v.iloc[0] if len(v) > 0 else None
    return None


if __name__ == "__main__":
    main()
