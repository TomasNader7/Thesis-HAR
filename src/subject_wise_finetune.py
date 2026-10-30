"""
Subject-wise fine-tuning evaluation (HAPT -> WISDM).

The original fine-tuning curve (phase3_domain_adaptation.exp_fine_tuning_learning_curve)
draws fine-tune windows at random and then tests on ALL of WISDM, so the fine-tune windows
are inside the test set. Here the test set is a fixed group of held-out WISDM subjects:

  - each seed shuffles the WISDM subjects and holds out N_TEST of them as the test set
  - fine-tune windows are drawn only from the remaining (pool) subjects
  - every condition is scored on the same held-out subjects' windows

Conditions: naive (MinMax, HAPT only), z-score per dataset (HAPT only), fine-tune 5/10/25/50%.
Outputs: results/phase3/D_subject_wise/{per_seed.csv, summary.csv}
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score

from phase3_domain_adaptation import (
    DATA_DIR, PHASE3_DIR, FrozenBaseStacking, get_base_learners, get_meta_learner,
    load_hapt_wisdm_3class, normalize_minmax_source_only, normalize_zscore_per_dataset,
)

WISDM_S_PATH = DATA_DIR / "interim" / "Filtered_datasets_and_KS_results" / "3class_wisdm_phase2" / "subject_filtered.txt"
OUT_DIR = os.path.join(PHASE3_DIR, "D_subject_wise")
STANDING = 2  # 0-indexed: WALKING 0, SITTING 1, STANDING 2


def fit_stack(X, y):
    model = FrozenBaseStacking(get_base_learners(), get_meta_learner())
    model.fit_base(X, y)
    model.fit_meta(model.oof_proba_features(X, y), y)
    return model


def score(model, X, y):
    y_pred = model.predict(X)
    return {
        "acc": accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
        "standing_recall": recall_score(y, y_pred, labels=[STANDING], average=None, zero_division=0)[0],
    }


def split_for_seed(seed, subjects, subj, n_test):
    """Same split logic for every condition: test subjects, then a shuffled pool of fine-tune windows."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(subjects)
    test_subj, pool_subj = order[:n_test], order[n_test:]
    pool_idx = np.flatnonzero(np.isin(subj, pool_subj))
    rng.shuffle(pool_idx)  # nested subsets: smaller fractions are prefixes of larger ones
    test_idx = np.flatnonzero(np.isin(subj, test_subj))
    return test_idx, pool_idx


def run_meta_only(n_seeds, n_test, frac):
    """Meta-only adaptation: base learners frozen on HAPT, meta-learner retrained on frac of pool windows."""
    X_hapt, y_hapt, X_wisdm, y_wisdm = load_hapt_wisdm_3class()
    subj = np.loadtxt(WISDM_S_PATH).astype(int)
    assert len(subj) == len(y_wisdm), "subject file not row-aligned with WISDM features"
    subjects = np.unique(subj)
    Xs, Xt = normalize_minmax_source_only(X_hapt, X_wisdm)
    base = FrozenBaseStacking(get_base_learners(), get_meta_learner()).fit_base(Xs, y_hapt)
    rows = []
    for seed in range(n_seeds):
        test_idx, pool_idx = split_for_seed(seed, subjects, subj, n_test)
        k = int(np.floor(frac * len(pool_idx)))
        ft = pool_idx[:k]
        base.fit_meta(base.base_proba_features(Xt[ft]), y_wisdm[ft])
        rows.append({"seed": seed, "n_test_subjects": n_test, "n_test_windows": len(test_idx),
                     "method": f"Meta-only {int(frac * 100)}%", "n_ft_windows": k,
                     **score(base, Xt[test_idx], y_wisdm[test_idx])})
        print(f"seed {seed} meta-only {frac:.2f}: acc {rows[-1]['acc']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "per_seed_meta.csv"), index=False)


def make_outputs():
    """Tables and figures (mean +/- std over seeds) from per_seed.csv and per_seed_meta.csv."""
    frames = [pd.read_csv(os.path.join(OUT_DIR, f)) for f in ("per_seed.csv", "per_seed_meta.csv")
              if os.path.exists(os.path.join(OUT_DIR, f))]
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby("method", sort=False)
    stats = g[["acc", "standing_recall", "macro_f1"]].agg(["mean", "std"])
    n_seeds = int(g["seed"].nunique().min())
    order = [m for m in ["Baseline (naive)", "Z-score", "Meta-only 10%", "Fine-tune 5%", "Fine-tune 10%",
                         "Fine-tune 25%", "Fine-tune 50%"] if m in stats.index]
    table = pd.DataFrame({
        "Method": order,
        "Accuracy (%) mean": [round(stats.loc[m, ("acc", "mean")] * 100, 2) for m in order],
        "Accuracy (%) std": [round(stats.loc[m, ("acc", "std")] * 100, 2) for m in order],
        "STANDING recall mean": [round(stats.loc[m, ("standing_recall", "mean")], 3) for m in order],
        "STANDING recall std": [round(stats.loc[m, ("standing_recall", "std")], 3) for m in order],
        "Macro F1 mean": [round(stats.loc[m, ("macro_f1", "mean")], 4) for m in order],
    })
    table.to_csv(os.path.join(OUT_DIR, "comparison_table.csv"), index=False)
    print(table.to_string(index=False))

    ft = [m for m in order if m.startswith("Fine-tune")]
    pcts = [int(m.split()[1].rstrip("%")) for m in ft]
    for metric, ylabel, fname in [("acc", "Accuracy on held-out WISDM subjects", "accuracy_learning_curve.png"),
                                  ("standing_recall", "STANDING recall on held-out WISDM subjects", "standing_recall_curve.png")]:
        fig, ax = plt.subplots(figsize=(9, 6))
        for m, color, ls in [("Baseline (naive)", "red", "--"), ("Z-score", "orange", "--"), ("Meta-only 10%", "purple", ":")]:
            if m in stats.index:
                mu, sd = stats.loc[m, (metric, "mean")], stats.loc[m, (metric, "std")]
                ax.axhline(mu, color=color, linestyle=ls, linewidth=1.5, label=f"{m} ({mu:.3f})")
                ax.axhspan(mu - sd, mu + sd, color=color, alpha=0.08)
        mu = [stats.loc[m, (metric, "mean")] for m in ft]
        sd = [stats.loc[m, (metric, "std")] for m in ft]
        ax.errorbar(pcts, mu, yerr=sd, marker="o", markersize=8, linewidth=2.2, capsize=5,
                    color="#1f77b4", label="Fine-tuning (mean ± std)")
        ax.set_xlim(0, 55)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Labeled target windows used for fine-tuning (% of non-test WISDM windows)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Subject-wise evaluation (HAPT → WISDM, {n_seeds} seeds)")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=300)
        plt.close()
    print("Saved comparison_table.csv, accuracy_learning_curve.png, standing_recall_curve.png")


def main(n_seeds, n_test, fracs):
    os.makedirs(OUT_DIR, exist_ok=True)
    X_hapt, y_hapt, X_wisdm, y_wisdm = load_hapt_wisdm_3class()
    subj = np.loadtxt(WISDM_S_PATH).astype(int)
    assert len(subj) == len(y_wisdm), "subject file not row-aligned with WISDM features"
    subjects = np.unique(subj)
    print(f"WISDM: {len(y_wisdm)} windows, {len(subjects)} subjects; holding out {n_test} per seed")

    # Naive and z-score models train on HAPT only, so fit once and score on each seed's test subjects.
    Xs_mm, Xt_mm = normalize_minmax_source_only(X_hapt, X_wisdm)
    naive = fit_stack(Xs_mm, y_hapt)
    # Z-score matches the thesis: scaler fit per dataset on all WISDM (transductive, no labels used).
    Xs_z, Xt_z = normalize_zscore_per_dataset(X_hapt, X_wisdm)
    zscore = fit_stack(Xs_z, y_hapt)

    rows = []
    for seed in range(n_seeds):
        test_idx, pool_idx = split_for_seed(seed, subjects, subj, n_test)
        y_test = y_wisdm[test_idx]

        base = {"seed": seed, "n_test_subjects": n_test, "n_test_windows": len(test_idx)}
        rows.append({**base, "method": "Baseline (naive)", **score(naive, Xt_mm[test_idx], y_test)})
        rows.append({**base, "method": "Z-score", **score(zscore, Xt_z[test_idx], y_test)})

        for frac in fracs:
            k = int(np.floor(frac * len(pool_idx)))
            ft = pool_idx[:k]
            X_mix = np.vstack([Xs_mm, Xt_mm[ft]])
            y_mix = np.concatenate([y_hapt, y_wisdm[ft]])
            model = fit_stack(X_mix, y_mix)
            rows.append({**base, "method": f"Fine-tune {int(frac * 100)}%", "n_ft_windows": k,
                         "n_ft_subjects": len(np.unique(subj[ft])),
                         **score(model, Xt_mm[test_idx], y_test)})
            print(f"seed {seed} frac {frac:.2f}: acc {rows[-1]['acc']:.4f} "
                  f"({k} ft windows, {rows[-1]['n_ft_subjects']} subjects)", flush=True)

        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "per_seed.csv"), index=False)

    df = pd.DataFrame(rows)
    summary = df.groupby("method", sort=False)[["acc", "macro_f1", "standing_recall"]].agg(["mean", "std"])
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"))
    print(summary.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.10, 0.25, 0.50])
    ap.add_argument("--meta-only", action="store_true", help="run only the meta-only 10%% condition")
    ap.add_argument("--plots", action="store_true", help="only rebuild tables/figures from saved CSVs")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.plots:
        make_outputs()
    elif a.meta_only:
        run_meta_only(a.seeds, a.n_test, 0.10)
        make_outputs()
    else:
        main(a.seeds, a.n_test, a.fracs)
