"""
ROC curves and feature importance for the thesis (re-runs the Phase 3 ensembles).

Reuses the exact code of phase3_domain_adaptation.py (same data, base learners, meta-learner,
out-of-fold meta training, seeds), then checks that the accuracy it gets equals the number
already in the thesis tables (naive 40.82, z-score 74.65) before drawing anything.
If the check fails, the script stops and writes no figure.

Outputs:
  figures/thesis/fig_roc_naive_vs_zscore.png     one-vs-rest ROC per class, full WISDM set
  figures/thesis/fig_feature_importance.png      top-15 tree importances (RF, XGBoost), source-fit
  results/roc_importance/roc_auc.csv             AUC per class + macro, both conditions
  results/roc_importance/feature_importance.csv  all 80 features, RF and XGBoost

Run from the repo root:  python src/make_roc_importance.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_domain_adaptation as p3  # noqa: E402  (same pipeline code as the tables)

ROOT = p3.REPO_ROOT
FIG = ROOT / "figures" / "thesis"
RES = ROOT / "results" / "roc_importance"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

CLASSES = p3.LABELS_3CLASS
BLUE, ORANGE, GREEN = "#4C72B0", "#DD8452", "#55A868"
COLORS = [BLUE, ORANGE, GREEN]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

SAVED = {
    "naive": ROOT / "results/phase3/A_distribution_alignment/baseline/metrics.json",
    "zscore": ROOT / "results/phase3/A_distribution_alignment/zscore/metrics.json",
}


def run_condition(tag, Xs, Xt, y_src, y_tgt):
    model = p3.FrozenBaseStacking(p3.get_base_learners(), p3.get_meta_learner()).fit_base(Xs, y_src)
    model.fit_meta(model.oof_proba_features(Xs, y_src), y_src)
    acc = (model.predict(Xt) == y_tgt).mean()
    saved = json.loads(SAVED[tag].read_text())["accuracy"]
    print(f"[{tag}] recomputed accuracy {acc:.6f} | saved {saved:.6f}")
    if abs(acc - saved) > 1e-3:
        sys.exit(f"STOP: {tag} accuracy does not reproduce the table value; no figures written.")
    proba = model.fitted_meta_.predict_proba(model.base_proba_features(Xt))
    return model, proba


def main():
    X_h, y_h, X_w, y_w = p3.load_hapt_wisdm_3class()

    Xs_n, Xt_n = p3.normalize_minmax_source_only(X_h, X_w)
    Xs_z, Xt_z = p3.normalize_zscore_per_dataset(X_h, X_w)
    _, proba_n = run_condition("naive", Xs_n, Xt_n, y_h, y_w)
    _, proba_z = run_condition("zscore", Xs_z, Xt_z, y_h, y_w)

    # ---------- ROC (one-vs-rest) ----------
    Y = label_binarize(y_w, classes=[0, 1, 2])
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    for ax, proba, title, tag in [(axes[0], proba_n, "Naive transfer", "naive"),
                                  (axes[1], proba_z, "Per-dataset z-score", "zscore")]:
        for k, name in enumerate(CLASSES):
            fpr, tpr, _ = roc_curve(Y[:, k], proba[:, k])
            auc = roc_auc_score(Y[:, k], proba[:, k])
            ax.plot(fpr, tpr, color=COLORS[k], lw=2, label=f"{name} (AUC {auc:.3f})")
            rows.append({"condition": tag, "class": name, "auc": auc})
        macro = roc_auc_score(Y, proba, average="macro")
        rows.append({"condition": tag, "class": "macro", "auc": macro})
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Chance")
        ax.set_title(f"{title}\nmacro AUC {macro:.3f}", fontsize=10)
        ax.set_xlabel("False positive rate")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    axes[0].set_ylabel("True positive rate")
    fig.tight_layout()
    fig.savefig(FIG / "fig_roc_naive_vs_zscore.png", dpi=300)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(RES / "roc_auc.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # ---------- Feature importance (tree models, fit on HAPT only, z-score scaled) ----------
    names = [l.strip() for l in open(p3.HAPT_X_PATH.parent / "feature_names_hapt.txt", encoding="utf-8") if l.strip()]
    names = [n.split(" ", 1)[1] if n.split(" ", 1)[0].isdigit() and " " in n else n for n in names]  # drop leading index
    if len(names) != X_h.shape[1]:
        sys.exit(f"STOP: {len(names)} feature names vs {X_h.shape[1]} columns.")
    imp = {}
    for key, label in [("random_forest", "Random Forest"), ("xgboost", "XGBoost")]:
        m = dict(p3.get_base_learners())[key]
        m.fit(Xs_z, y_h)
        imp[label] = m.feature_importances_
    df = pd.DataFrame({"feature": names, **imp})
    df.to_csv(RES / "feature_importance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6))
    for ax, label, color in zip(axes, imp, [BLUE, ORANGE]):
        top = df.sort_values(label, ascending=False).head(15).iloc[::-1]
        ax.barh(top["feature"], top[label], color=color)
        ax.set_title(f"{label}: top 15 features", fontsize=10)
        ax.set_xlabel("Importance (share of total)")
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_feature_importance.png", dpi=300)
    plt.close(fig)
    print(df.sort_values("Random Forest", ascending=False).head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
