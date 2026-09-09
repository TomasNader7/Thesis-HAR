"""
build_master_table.py
---------------------
Run AFTER phase3_domain_adaptation.py.

Usage:
    python build_master_table.py

Output:
    tables/master_table.csv   <- paste directly into Word
"""

import os
import json
import pandas as pd

ROOT = os.path.join("results", "phase3")

ORDER = [
    "Baseline (Naive Transfer)",
    "Z-score Normalization",
    "Meta-Only Adaptation (10%)",
    "Fine-Tuning (5%)",
    "Fine-Tuning (10%)",
    "Fine-Tuning (25%)",
    "Fine-Tuning (50%)",
]


def find_metrics_json(root: str) -> list:
    rows = []
    for dirpath, _, filenames in os.walk(root):
        if "metrics.json" in filenames:
            fp = os.path.join(dirpath, "metrics.json")
            with open(fp, "r") as f:
                data = json.load(f)
            rows.append(data)
            print(f"  Found: {fp}  ->  method='{data['method']}'")
    return rows


def main():
    print(f"Scanning: {ROOT}\n")
    rows = find_metrics_json(ROOT)

    if not rows:
        raise RuntimeError(
            f"No metrics.json files found under '{ROOT}'.\n"
            "Make sure phase3_domain_adaptation.py ran successfully first."
        )

    df = pd.DataFrame(rows)

    baseline_mask = df["method"] == "Baseline (Naive Transfer)"
    if not baseline_mask.any():
        raise RuntimeError(
            "No row with method='Baseline (Naive Transfer)' found.\n"
            "Check that exp_feature_distribution_alignment() ran and saved metrics.json."
        )
    baseline_acc = float(df[baseline_mask].iloc[0]["accuracy"])
    print(f"\nBaseline accuracy: {baseline_acc:.4f}")

    out = pd.DataFrame()
    out["Method"]                 = df["method"]
    out["Accuracy (%)"]           = (df["accuracy"] * 100).round(2)
    out["Delta Accuracy vs Baseline"] = ((df["accuracy"] - baseline_acc) * 100).round(2)
    out["Macro F1"]               = df["macro_f1"].round(3)
    out["WALKING F1"]             = df["walking_f1"].round(3)
    out["SITTING F1"]             = df["sitting_f1"].round(3)
    out["STANDING F1"]            = df["standing_f1"].round(3)
    out["STANDING Recall"]        = df["standing_recall"].round(3)

    out["Delta Accuracy vs Baseline"] = out["Delta Accuracy vs Baseline"].apply(
        lambda v: f"+{v:.2f}" if v >= 0 else f"{v:.2f}"
    )

    out["__order"] = out["Method"].apply(
        lambda m: ORDER.index(m) if m in ORDER else 999
    )
    out = out.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    os.makedirs("tables", exist_ok=True)
    csv_path = os.path.join("tables", "master_table.csv")
    out.to_csv(csv_path, index=False)

    print(f"\nSaved -> {csv_path}\n")
    print(out.to_string(index=False))
    print("\nOpen tables/master_table.csv and paste into Word.")


if __name__ == "__main__":
    main()