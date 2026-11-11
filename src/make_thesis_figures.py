"""
Thesis figures drawn only from saved result files (no model is re-run).

Outputs go to figures/thesis/:
  fig_confusion_naive_vs_zscore.png      ensemble confusion matrices, full WISDM set
  fig_confusion_base_learners_naive.png  five base learners, naive transfer
  fig_confusion_base_learners_zscore.png five base learners, z-score
  fig_per_seed_accuracy.png              accuracy of each condition over the 10 subject-wise splits
  fig_pipeline.png                       stacking ensemble and the four conditions (diagram)
  fig_subject_split.png                  subject-wise split design (diagram)

Run from the repo root:  python src/make_thesis_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT / "results" / "phase3"
OUT = ROOT / "figures" / "thesis"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["WALKING", "SITTING", "STANDING"]
BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#7F7F7F"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load_cm(path):
    return np.loadtxt(path, delimiter=",", dtype=int)


def draw_cm(ax, cm, title, cmap="Blues"):
    """Row-normalized colors (share of the true class); cell text shows count and percent."""
    pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            color = "white" if pct[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm[i, j]:,}\n({pct[i, j] * 100:.1f}%)", ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES, fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8, rotation=90, va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"{title}\naccuracy {acc * 100:.2f}%", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_confusion_ensemble():
    naive = load_cm(P3 / "A_distribution_alignment/baseline/minmax_source_only_confusion_matrix.csv")
    z = load_cm(P3 / "A_distribution_alignment/zscore/zscore_per_dataset_confusion_matrix.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    draw_cm(axes[0], naive, "Naive transfer (HAPT-fit MinMax)")
    draw_cm(axes[1], z, "Per-dataset z-score")
    fig.tight_layout()
    fig.savefig(OUT / "fig_confusion_naive_vs_zscore.png", dpi=300)
    plt.close(fig)


def fig_confusion_base_learners():
    names = [("sgd", "SGD"), ("svm", "SVM"), ("random_forest", "Random Forest"),
             ("gradient_boosting", "Gradient Boosting"), ("xgboost", "XGBoost")]
    for tag, folder, suffix, fname in [
        ("naive", "0_individual_base_learners", "", "fig_confusion_base_learners_naive.png"),
        ("z-score", "0b_base_learners_zscore", "_zscore", "fig_confusion_base_learners_zscore.png"),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2))
        for ax, (key, label) in zip(axes.flat, names):
            draw_cm(ax, load_cm(P3 / folder / f"{key}{suffix}_confusion_matrix.csv"), f"{label} ({tag})")
        axes.flat[5].axis("off")
        fig.tight_layout(h_pad=2.5, w_pad=1.5)
        fig.savefig(OUT / fname, dpi=300)
        plt.close(fig)


def fig_per_seed():
    a = pd.read_csv(P3 / "D_subject_wise/per_seed.csv")
    m = pd.read_csv(P3 / "D_subject_wise/per_seed_meta.csv")
    m["method"] = "Meta-only 10%"
    df = pd.concat([a[["seed", "method", "acc"]], m[["seed", "method", "acc"]]])
    order = ["Baseline (naive)", "Z-score", "Meta-only 10%", "Fine-tune 5%", "Fine-tune 10%", "Fine-tune 25%", "Fine-tune 50%"]
    labels = ["Naive", "Z-score", "Meta-only\n(10%)", "Fine-tune\n5%", "Fine-tune\n10%", "Fine-tune\n25%", "Fine-tune\n50%"]
    colors = [RED, ORANGE, PURPLE, BLUE, BLUE, BLUE, BLUE]
    wide = df.pivot(index="seed", columns="method", values="acc")[order] * 100
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for s, row in wide.iterrows():
        ax.plot(range(len(order)), row.values, color=GRAY, alpha=0.25, lw=0.9, zorder=1)
    bp = ax.boxplot([wide[c].values for c in order], positions=range(len(order)), widths=0.5, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", lw=1.4), zorder=2)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for i, c in enumerate(order):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(wide)), wide[c].values, s=16, color=colors[i], edgecolor="black", lw=0.4, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy on held-out WISDM subjects (%)")
    ax.set_ylim(30, 100)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_title("Accuracy over the 10 subject-wise splits (20 held-out subjects each)\n"
                 "dots = splits, grey lines connect the same split, box = quartiles", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_seed_accuracy.png", dpi=300)
    plt.close(fig)
    print(wide.describe().loc[["mean", "std"]].round(2))


def box(ax, x, y, w, h, text, fc, fs=9, ec="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="black"))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.1)
    ax.axis("off")
    # source side
    box(ax, 0.1, 3.9, 1.7, 1.0, "HAPT\n(source)\n1,879 windows", "#DCE6F2")
    box(ax, 2.2, 3.9, 1.7, 1.0, "80 features\nper window", "#DCE6F2")
    box(ax, 4.3, 3.3, 2.1, 2.2, "Five base learners\nSGD, SVM,\nRandom Forest,\nXGBoost,\nGradient Boosting", "#E6F0DC")
    box(ax, 6.9, 3.9, 1.8, 1.0, "Class probabilities\n(out-of-fold,\n3-fold)", "#F5EBD5")
    box(ax, 9.1, 3.9, 1.8, 1.0, "Meta-learner\n(Logistic\nRegression)", "#F2DCDC")
    arrow(ax, 1.8, 4.4, 2.2, 4.4)
    arrow(ax, 3.9, 4.4, 4.3, 4.4)
    arrow(ax, 6.4, 4.4, 6.9, 4.4)
    arrow(ax, 8.7, 4.4, 9.1, 4.4)
    ax.text(5.5, 5.8, "Training on the source dataset (HAPT)", ha="center", fontsize=10, style="italic")
    # target side
    box(ax, 0.1, 1.55, 1.7, 1.0, "WISDM\n(target)\n12,470 windows", "#F7E3D3")
    ax.text(5.5, 2.75, "Four ways of using the trained model on the target dataset (WISDM)", ha="center", fontsize=10, style="italic")
    conds = [("1. Naive transfer\nno adaptation", "#F2DCDC"),
             ("2. Z-score\neach dataset rescaled\n(no labels)", "#FBF0C8"),
             ("3. Meta-only\nretrain meta-learner\n(10% labels)", "#E8DDF1"),
             ("4. Fine-tune\nretrain all learners\n(5, 10, 25, 50% labels)", "#DCE6F2")]
    for k, (t, c) in enumerate(conds):
        box(ax, 2.3 + k * 2.2, 1.45, 2.0, 1.2, t, c, fs=8.5)
        arrow(ax, 1.8, 2.05, 2.3 + k * 2.2, 2.05) if k == 0 else None
    for k in range(3):
        arrow(ax, 4.3 + k * 2.2, 2.05, 4.3 + k * 2.2 + 0.0 + 0.2, 2.05) if False else None
    box(ax, 2.3, 0.15, 8.6, 0.7, "Each condition is scored on the same held-out WISDM subjects (20 of 51 subjects, 10 random splits)", "#EEEEEE", fs=9)
    for k in range(4):
        arrow(ax, 3.3 + k * 2.2, 1.45, 3.3 + k * 2.2, 0.85)
    fig.savefig(OUT / "fig_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_subject_split():
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), gridspec_kw={"height_ratios": [1, 1]})
    ax = axes[0]
    ax.set_xlim(0, 51)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Window-level split (first experiment, superseded): the same subject appears on both sides", fontsize=10, loc="left")
    rng = np.random.default_rng(3)
    for s in range(51):
        ax.add_patch(plt.Rectangle((s, 0.15), 0.9, 0.7, fc=BLUE, ec="white"))
        ax.add_patch(plt.Rectangle((s, 0.15), 0.9, 0.7 * 0.3, fc=ORANGE, ec="white"))
    ax.text(25.5, -0.05, "each column = one subject; whole column = test windows (all of WISDM); orange part = windows also used for adaptation", ha="center", fontsize=8)
    ax = axes[1]
    ax.set_xlim(0, 51)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Subject-wise split (used in this thesis): test subjects never contribute adaptation windows", fontsize=10, loc="left")
    for s in range(51):
        c = BLUE if s < 20 else ORANGE
        ax.add_patch(plt.Rectangle((s, 0.15), 0.9, 0.7, fc=c, ec="white"))
    ax.text(10, 0.92, "20 held-out test subjects", ha="center", fontsize=9)
    ax.text(35.5, 0.92, "31 pool subjects (adaptation windows drawn from here)", ha="center", fontsize=9)
    ax.text(25.5, -0.05, "subjects are re-drawn for each of the 10 random splits", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_subject_split.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_confusion_ensemble()
    fig_confusion_base_learners()
    fig_per_seed()
    fig_pipeline()
    fig_subject_split()
    print("saved to", OUT)
