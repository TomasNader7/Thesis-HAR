# Goal: two dataset figures for the proposal paper (Section 3).
#   1) class distribution of windows, HAPT vs WISDM (WALKING / SITTING / STANDING)
#   2) one raw accelerometer window (6.4 s, 20 Hz) per class per dataset
# Reads the same interim labels and raw files the pipeline uses; writes PNGs to figures/dataset_overview/.
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "figures" / "dataset_overview"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["WALKING", "SITTING", "STANDING"]
HAPT_LABEL = {"WALKING": 1, "SITTING": 2, "STANDING": 3}          # y_hapt.txt
WISDM_LABEL = {"WALKING": 1, "SITTING": 4, "STANDING": 5}         # y_wisdm.txt (6-class map)
WISDM_CODE = {"WALKING": "A", "SITTING": "D", "STANDING": "E"}    # raw activity codes
HAPT_ID = {"WALKING": 1, "SITTING": 4, "STANDING": 5}             # RawData/labels.txt ids
HAPT_HZ, TARGET_HZ, WIN = 50, 20, 128

COLORS = {"HAPT": "#4C72B0", "WISDM": "#DD8452"}
AXIS_COLORS = ["#4C72B0", "#55A868", "#C44E52"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def class_counts():
    y_h = np.loadtxt(DATA / "interim" / "hapt_3class_output_phase2" / "y_hapt.txt", dtype=int)
    y_w = np.loadtxt(DATA / "interim" / "wisdm_phase2" / "y_wisdm.txt", dtype=int)
    h = {c: int((y_h == HAPT_LABEL[c]).sum()) for c in CLASSES}
    w = {c: int((y_w == WISDM_LABEL[c]).sum()) for c in CLASSES}
    return h, w


def fig_class_distribution():
    h, w = class_counts()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, name, d in zip(axes, ["HAPT", "WISDM"], [h, w]):
        vals = [d[c] for c in CLASSES]
        total = sum(vals)
        bars = ax.bar(CLASSES, vals, color=COLORS[name])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({100 * v / total:.0f}%)",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{name} (n = {total:,} windows)")
        ax.set_ylabel("Windows (6.4 s, 50% overlap)")
        ax.set_ylim(0, max(vals) * 1.25)
    fig.suptitle("Class distribution of the three shared activities", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "class_distribution_hapt_vs_wisdm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return h, w


def hapt_window(cls):
    root = DATA / "raw" / "hapt" / "RawData"
    labels = pd.read_csv(root / "labels.txt", sep=r"\s+", header=None,
                         names=["exp", "user", "act", "start", "end"])
    seg_df = labels[labels["act"] == HAPT_ID[cls]].sort_values(["user", "exp", "start"])
    up = TARGET_HZ // math.gcd(TARGET_HZ, HAPT_HZ)
    down = HAPT_HZ // math.gcd(TARGET_HZ, HAPT_HZ)
    for _, r in seg_df.iterrows():
        acc = np.loadtxt(root / f"acc_exp{int(r.exp):02d}_user{int(r.user):02d}.txt")
        seg = resample_poly(acc[int(r.start):int(r.end) + 1], up, down, axis=0, padtype="mean")
        if len(seg) >= WIN:
            mid = (len(seg) - WIN) // 2  # mid-segment, away from transitions
            return seg[mid:mid + WIN], int(r.user)
    raise RuntimeError(f"no HAPT segment >= {WIN} samples for {cls}")


def wisdm_window(cls):
    folder = DATA / "raw" / "wisdm" / "phone" / "accel"
    for f in sorted(folder.glob("*.txt")):
        df = pd.read_csv(f, header=None, names=["user", "act", "ts", "x", "y", "z"])
        df["z"] = df["z"].astype(str).str.replace(";", "").astype(float)
        g = df[df["act"] == WISDM_CODE[cls]].sort_values("ts")
        if len(g) >= WIN:
            start = (len(g) // 3)
            start = min(start, len(g) - WIN)
            return g[["x", "y", "z"]].values[start:start + WIN], int(g["user"].iloc[0])
    raise RuntimeError(f"no WISDM data for {cls}")


def fig_raw_signals():
    t = np.arange(WIN) / TARGET_HZ
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True)
    info = {}
    for col, cls in enumerate(CLASSES):
        for row, (name, fn, unit) in enumerate([("HAPT", hapt_window, "g"),
                                                ("WISDM", wisdm_window, "m/s²")]):
            sig, user = fn(cls)
            info[(name, cls)] = user
            ax = axes[row, col]
            for i, ax_name in enumerate("xyz"):
                ax.plot(t, sig[:, i], color=AXIS_COLORS[i], lw=1.1, label=ax_name)
            ax.set_title(f"{name}: {cls} (subject {user})", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Acceleration ({unit})")
            if row == 1:
                ax.set_xlabel("Time (s)")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, title="Axis")
    fig.suptitle("Raw accelerometer window per class (6.4 s at 20 Hz; HAPT resampled from 50 Hz)", y=1.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "raw_signal_samples_hapt_vs_wisdm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return info


if __name__ == "__main__":
    h, w = fig_class_distribution()
    print("HAPT counts:", h)
    print("WISDM counts:", w)
    print("signal subjects:", fig_raw_signals())
    print("saved to", OUT)
