import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.preprocessing import MinMaxScaler

# Reuse exact feature code (critical for alignment)
from feature_extraction_WISDM import compute_features_for_window, PHASE_TAG


TARGET_ACTIVITIES = {"WALKING", "SITTING", "STANDING"}

# HAPT is recorded at 50Hz, WISDM at 20Hz. Resample HAPT to 20Hz before windowing
# so 128-sample windows cover the same 6.4s in both datasets.
HAPT_NATIVE_HZ = 50
TARGET_HZ = 20

# HAPT (UCI 341) has many "transition" activities (e.g., "SITTING_TO_STANDING")
def is_transition(name: str) -> bool:
    name = name.upper()
    return ("TO" in name) or ("TRANSITION" in name) or ("_" in name and "TO" in name.split("_"))


def load_activity_labels(hapt_root: Path) -> dict[int, str]:
    """
    activity_labels.txt format: "<id> <label>"
    """
    labels_path = hapt_root / "activity_labels.txt"
    mapping = {}
    with open(labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            act_id = int(parts[0])
            act_name = parts[1].upper()
            mapping[act_id] = act_name
    return mapping


def load_labels_table(hapt_root: Path) -> pd.DataFrame:
    """
    RawData/labels.txt columns (per UCI 341 description):
      1 exp_id, 2 user_id, 3 activity_id, 4 start, 5 end
    start/end are sample indices at 50Hz
    """
    labels_path = hapt_root / "RawData" / "labels.txt"
    df = pd.read_csv(labels_path, sep=r"\s+", header=None,
                     names=["exp_id", "user_id", "activity_id", "start", "end"])
    return df


def read_acc_file(hapt_root: Path, exp_id: int, user_id: int) -> np.ndarray:
    """
    RawData/acc_expXX_userYY.txt: each row = one sample, columns = x y z
    """
    fname = f"acc_exp{exp_id:02d}_user{user_id:02d}.txt"
    fpath = hapt_root / "RawData" / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Missing accel file: {fpath}")
    arr = np.loadtxt(fpath)  # shape (N, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Unexpected accel shape in {fpath}: {arr.shape}")
    return arr


def resample_segment(segment_xyz: np.ndarray) -> np.ndarray:
    """
    Resample one labelled segment from HAPT's native 50Hz to WISDM's 20Hz
    (ratio 2/5). resample_poly applies an anti-aliasing FIR filter; padtype="mean"
    avoids edge transients from the ~1g gravity offset.
    """
    up, down = TARGET_HZ // math.gcd(TARGET_HZ, HAPT_NATIVE_HZ), HAPT_NATIVE_HZ // math.gcd(TARGET_HZ, HAPT_NATIVE_HZ)
    return resample_poly(segment_xyz, up, down, axis=0, padtype="mean")


def segment_into_windows(signal_xyz: np.ndarray, window_size=128, overlap=0.5) -> list[np.ndarray]:
    step = int(window_size * (1 - overlap))
    windows = []
    for start in range(0, len(signal_xyz) - window_size + 1, step):
        windows.append(signal_xyz[start:start + window_size])
    return windows


def extract_hapt_3class_features(
    hapt_root: str,
    subset_segments: int | None = None,
    window_size: int = 128,
    overlap: float = 0.5,
) -> pd.DataFrame:
    hapt_root = Path(hapt_root)

    act_map = load_activity_labels(hapt_root)
    labels_df = load_labels_table(hapt_root)

    # Map activity_id -> activity_name
    labels_df["activity_name"] = labels_df["activity_id"].map(act_map).fillna("UNKNOWN").str.upper()

    # Keep only perfect-match, non-transition activities
    labels_df = labels_df[labels_df["activity_name"].isin(TARGET_ACTIVITIES)]
    labels_df = labels_df[~labels_df["activity_name"].apply(is_transition)]

    if subset_segments:
        labels_df = labels_df.head(subset_segments)

    all_rows = []

    # Cache accel files so there is no reload of same exp/user repeatedly
    acc_cache: dict[tuple[int, int], np.ndarray] = {}

    for i, row in labels_df.iterrows():
        exp_id = int(row["exp_id"])
        user_id = int(row["user_id"])
        activity = row["activity_name"]
        start = int(row["start"])
        end = int(row["end"])

        key = (exp_id, user_id)
        if key not in acc_cache:
            acc_cache[key] = read_acc_file(hapt_root, exp_id, user_id)

        acc = acc_cache[key]

        # Defensive bounds
        start = max(0, start)
        end = min(len(acc) - 1, end)

        # Slice at the native rate first so labels stay aligned, then resample
        # the segment so a 128-sample window spans the same 6.4s as WISDM.
        segment = acc[start:end + 1]  # inclusive end
        segment = resample_segment(segment)
        if len(segment) < window_size:
            continue

        windows = segment_into_windows(segment, window_size=window_size, overlap=overlap)

        for w in windows:
            feats = compute_features_for_window(w)  # EXACT SAME FEATURE LOGIC AS WISDM
            feats["activity_uci"] = activity
            feats["user_id"] = user_id
            feats["exp_id"] = exp_id
            all_rows.append(feats)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise ValueError("No windows/features extracted. Check label filtering and paths.")

    # Confirm 61 features (excluding metadata)
    metadata_cols = {"activity_uci", "user_id", "exp_id"}
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    expected = 61 if PHASE_TAG == "phase1" else 80
    print(f"HAPT extracted windows: {len(df):,}")
    print(f"Feature columns: {len(feature_cols)} (expected: {expected})")
    if len(feature_cols) != expected:
        print(f"WARNING: feature count != {expected}. You may have changed feature logic.")

    return df


def normalize_to_minus1_plus1(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    metadata_cols = ["activity_uci", "user_id", "exp_id"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    Xn = scaler.fit_transform(df[feature_cols])

    out = pd.DataFrame(Xn, columns=feature_cols)
    out["activity_uci"] = df["activity_uci"].values
    out["user_id"] = df["user_id"].values
    out["exp_id"] = df["exp_id"].values
    return out, scaler


def export_uci_style(df_norm: pd.DataFrame, output_dir: str):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 3-class mapping (keep it simple and explicit)
    label_map = {"WALKING": 1, "SITTING": 2, "STANDING": 3}

    metadata_cols = ["activity_uci", "user_id", "exp_id"]
    feature_cols = [c for c in df_norm.columns if c not in metadata_cols]

    X = df_norm[feature_cols]
    y = df_norm["activity_uci"].map(label_map)
    subj = df_norm["user_id"]

    # Save
    X.to_csv(outdir / "X_hapt.txt", sep=" ", index=False, header=False)
    y.to_csv(outdir / "y_hapt.txt", index=False, header=False)
    subj.to_csv(outdir / "subject_hapt.txt", index=False, header=False)

    # Feature names (for alignment checks)
    with open(outdir / "feature_names_hapt.txt", "w") as f:
        for i, col in enumerate(feature_cols, 1):
            f.write(f"{i} {col}\n")

    with open(outdir / "activity_labels_hapt.txt", "w") as f:
        for name, lab in sorted(label_map.items(), key=lambda x: x[1]):
            f.write(f"{lab} {name}\n")

    print(f"Saved to: {outdir}")
    print(f"X_hapt: {X.shape}, y_hapt: {y.shape}, subjects: {subj.nunique()}")


if __name__ == "__main__":
    # python src/feature_extraction_HAPT.py (run from repo root, or anywhere --
    # paths below are anchored to the repo root via __file__).
    REPO_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = REPO_ROOT / "data"

    # Place the raw UCI HAPT dataset here (see README "Data" section).
    HAPT_ROOT = str(DATA_DIR / "raw" / "hapt")
    OUTPUT_DIR = str(DATA_DIR / "interim" / f"hapt_3class_output_{PHASE_TAG}")

    df = extract_hapt_3class_features(HAPT_ROOT, subset_segments=None, window_size=128, overlap=0.5)
    df_norm, _ = normalize_to_minus1_plus1(df)
    export_uci_style(df_norm, OUTPUT_DIR)
