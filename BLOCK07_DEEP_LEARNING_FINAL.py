
# =============================================================================
# BLOCK 07 — DEEP LEARNING EEG DECODING
# Paper 2 — Subject-Dependent + Cross-Subject LOSO
# Canonical executable file
# =============================================================================
#
# INPUT:
#   BLOCK 04 final raw EEG windows: W500_O50, 13 x 500
#   Set labels are normalized from Block 04 A/B/C -> SetA/SetB/SetC.
#   07A uses the exact Block 04 windows: Set A train / Set B validation / Set C test.
#
# IMPORTANT:
#   BLOCK 05 engineered features are NOT used by this block.
#   BLOCK 06 is used only for cohort/LOSO partition alignment checks.
#
# EXPERIMENTS:
#   07A: subject-dependent
#        Set A = training
#        Set B = validation
#        Set C = final test
#
#   07B: cross-subject LOSO
#        25 folds
#        24 training subjects
#        1 completely unseen test subject
#        lowest-numbered training subject = validation subject
#        remaining 23 subjects = training
#
# MODELS:
#   EEGNet
#   ShallowConvNet
#   DeepConvNet
#   CNN-BiLSTM
#   CNN-BiLSTM-Attention
#
# NO:
#   SMOTE
#   engineered 341-feature input
#   random window-level test split
#   cross-condition training
#   test-derived normalization
#   test-set tuning
#   calibration
#   transfer
#   RSA
#   XAI
#   robustness
#
# The numerical architecture/training configuration is frozen in this file
# before definitive execution and is archived to BLOCK07_config.json.
# =============================================================================

import os
import sys
import gc
import json
import time
import random
import platform
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# =============================================================================
# 01. FROZEN PAPER 2 CONSTANTS
# =============================================================================

PROTOCOL_VERSION = "BLOCK07_DL_PAPER2_V1.0_LOCAL_PC"
RANDOM_SEED = 42

FS = 125
N_CHANNELS = 13
WINDOW_SAMPLES = 500
WINDOW_SECONDS = WINDOW_SAMPLES / FS
OVERLAP_PERCENT = 50
STEP_SAMPLES = 250

N_SUBJECTS = 25
SUBJECTS = tuple(range(1, 26))
CONDITIONS = ("ME", "MI")
GESTURE_IDS = tuple(range(1, 8))
GESTURE_NAMES = {
    1: "HOC",
    2: "IFM",
    3: "LFM",
    4: "MFM",
    5: "PH",
    6: "RFM",
    7: "TFM",
}

N_RECORDINGS = 1050
WINDOWS_PER_RECORDING = 21
TOTAL_WINDOWS = 22050

MODELS = (
    "EEGNet",
    "ShallowConvNet",
    "DeepConvNet",
    "CNN-BiLSTM",
    "CNN-BiLSTM-Attention",
)

LATENT_MODELS = {"EEGNet", "CNN-BiLSTM", "CNN-BiLSTM-Attention"}

# Subject-dependent: Set A train, Set B validation, Set C test.
SD_TRAIN_SET = "SetA"
SD_VALIDATION_SET = "SetB"
SD_TEST_SET = "SetC"

# LOSO: fixed, deterministic, subject-level validation inside the 24 training
# subjects. The held-out subject is never used in validation.
LOSO_VALIDATION_RULE = "lowest_numbered_subject_among_24_training_subjects"

# Training configuration — frozen before final execution.
BATCH_SIZE = 64
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 6
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 5.0
NUM_WORKERS = 0
PIN_MEMORY = True
# Definitive Paper 2 BLOCK 07 execution is GPU-only. The script must stop rather than
# silently launching a many-hour CPU benchmark in Colab.
REQUIRE_CUDA = True
MODEL_DROPOUT = {
    "EEGNet": 0.5,
    "ShallowConvNet": 0.5,
    "DeepConvNet": 0.5,
    "CNN-BiLSTM": 0.3,
    "CNN-BiLSTM-Attention": 0.3,
}
EARLY_STOPPING_MIN_DELTA = 1e-3

# Normalization is training-only and channel-wise.
# Mean/std are calculated from training windows only.
NORMALIZATION = "training_only_channelwise_zscore"

# Primary selection criterion for checkpointing.
CHECKPOINT_METRIC = "validation_macro_f1"

# Reproducibility test.
REPRO_TEST_MODEL = "EEGNet"
REPRO_TEST_CONDITION = "ME"
REPRO_TEST_SUBJECT = 1
REPRO_TEST_MAX_EPOCHS = 5
REPRO_TEST_BATCH_SIZE = 64

# =============================================================================
# 02. GOOGLE DRIVE PATHS
# =============================================================================

RAW_ROOT = Path(r"F:\Faruk\EEG_Motor_Execution_Imagery")
PAPER2_ROOT = RAW_ROOT / "Paper2_EEG_BCI"

BLOCK01_ROOT = PAPER2_ROOT / "01_SUBJECT_MATCHING"
BLOCK02_ROOT = PAPER2_ROOT / "02_SIGNAL_QA"
BLOCK03_ROOT = PAPER2_ROOT / "03_PREPROCESSING"
BLOCK04_ROOT = PAPER2_ROOT / "04_WINDOWING"
BLOCK05_ROOT = PAPER2_ROOT / "05_FEATURE_ENGINEERING"
BLOCK06_ROOT = PAPER2_ROOT / "06_ML"
BLOCK07_ROOT = PAPER2_ROOT / "07_DL"

FINAL_WINDOWS_ROOT = (
    BLOCK04_ROOT / "final_windows" / "data" / "W500_O50"
)

CONFIG_ROOT = BLOCK07_ROOT / "configs"
SD_ROOT = BLOCK07_ROOT / "07A_DL_SD"
LOSO_ROOT = BLOCK07_ROOT / "07B_DL_LOSO"
COMP_ROOT = BLOCK07_ROOT / "comparisons"
DIAG_ROOT = BLOCK07_ROOT / "diagnostics"

for p in (
    BLOCK07_ROOT,
    CONFIG_ROOT,
    SD_ROOT,
    LOSO_ROOT,
    COMP_ROOT,
    DIAG_ROOT,
):
    p.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 03. UTILITY FUNCTIONS
# =============================================================================

def now_utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def atomic_write_csv(path, df):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_determinism():
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "Strict deterministic execution could not be enabled in this "
            "environment. Do not proceed with the definitive BLOCK 07 run."
        ) from exc


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        raise RuntimeError(f"Expected integer-like value, got {x!r}")


def condition_normalize(x):
    s = str(x).strip().upper()
    if s not in CONDITIONS:
        raise RuntimeError(f"Invalid condition: {x!r}")
    return s


def set_normalize(x):
    s = str(x).strip()
    if s.lower().startswith("set"):
        s = s[3:]
    s = s.upper()
    if s not in {"A", "B", "C"}:
        raise RuntimeError(f"Invalid set: {x!r}")
    return f"Set{s}"


def model_dir_name(model):
    return model.replace("-", "_").replace("+", "ATTN")


# =============================================================================
# 04. GOOGLE DRIVE MOUNT
# =============================================================================

def mount_drive():
    """Local-PC environment check. No Google Drive/Colab mounting is used."""
    if os.name != "nt":
        raise RuntimeError(
            f"BLOCK 07 LOCAL requires Windows. Detected OS: {os.name}"
        )
    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"Local EEG project root does not exist: {RAW_ROOT}"
        )
    if not PAPER2_ROOT.exists():
        raise FileNotFoundError(
            f"Local Paper2_EEG_BCI folder does not exist: {PAPER2_ROOT}"
        )
    print(f"LOCAL PROJECT ROOT: {RAW_ROOT}")
    print(f"PAPER2 ROOT: {PAPER2_ROOT}")
    print("Google Colab mount: NOT USED")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_block04_handoff():
    required = [
        BLOCK04_ROOT / "BLOCK04_complete.json",
        BLOCK04_ROOT / "selected_window_config.json",
        BLOCK04_ROOT / "FINAL_WINDOW_CONFIG.json",
        BLOCK04_ROOT / "BLOCK04_config.json",
        BLOCK04_ROOT / "BLOCK04_summary.json",
        BLOCK04_ROOT / "final_window_manifest.csv",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing BLOCK 04 artifact: {p}")

    complete = load_json(required[0])
    selected = load_json(required[1])
    final_cfg = load_json(required[2])
    cfg = load_json(required[3])
    summary = load_json(required[4])

    if str(complete.get("status", "")).upper() != "PASS":
        raise RuntimeError("BLOCK 04 completion status is not PASS.")

    def first(d, keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    sources = [selected, final_cfg, cfg, summary, complete]

    window = None
    overlap = None
    step = None
    for d in sources:
        window = window if window is not None else first(
            d, ["window_samples", "selected_window_samples"]
        )
        overlap = overlap if overlap is not None else first(
            d, ["overlap_percent", "selected_overlap_percent"]
        )
        step = step if step is not None else first(
            d, ["step_samples", "selected_step_samples"]
        )

    if safe_int(window) != WINDOW_SAMPLES:
        raise RuntimeError(f"BLOCK 04 window mismatch: {window}")
    if safe_int(overlap) != OVERLAP_PERCENT:
        raise RuntimeError(f"BLOCK 04 overlap mismatch: {overlap}")
    if safe_int(step) != STEP_SAMPLES:
        raise RuntimeError(f"BLOCK 04 step mismatch: {step}")

    manifest = pd.read_csv(required[5])
    required_cols = {
        "window_id",
        "subject",
        "condition",
        "gesture_code",
        "set",
        "source_recording",
        "window_samples",
        "overlap_percent",
        "step_samples",
        "start_sample",
        "end_sample",
    }
    missing = required_cols - set(manifest.columns)
    if missing:
        raise RuntimeError(
            f"BLOCK 04 final manifest missing columns: {sorted(missing)}"
        )

    # BLOCK 04 provenance may encode the same recording set as either
    # A/B/C or SetA/SetB/SetC. Normalize the metadata representation only;
    # this does not alter EEG data, windows, or source-recording identity.
    try:
        manifest["set"] = manifest["set"].map(set_normalize)
    except Exception as exc:
        raise RuntimeError(
            f"BLOCK 04 final manifest contains an invalid set label: {exc}"
        ) from exc

    if len(manifest) != TOTAL_WINDOWS:
        raise RuntimeError(
            f"BLOCK 04 manifest has {len(manifest)} rows; "
            f"expected {TOTAL_WINDOWS}."
        )

    if set(pd.to_numeric(manifest["subject"]).astype(int)) != set(SUBJECTS):
        raise RuntimeError("BLOCK 04 manifest subject set is not 1..25.")

    if set(manifest["condition"].astype(str).str.upper()) != set(CONDITIONS):
        raise RuntimeError("BLOCK 04 manifest condition set is not ME/MI.")

    gestures = set(
        pd.to_numeric(manifest["gesture_code"]).astype(int)
    )
    if gestures != set(GESTURE_IDS):
        raise RuntimeError("BLOCK 04 gesture codes are not exactly 1..7.")

    if not np.all(
        pd.to_numeric(manifest["window_samples"]).astype(int).to_numpy()
        == WINDOW_SAMPLES
    ):
        raise RuntimeError("BLOCK 04 manifest contains non-500-sample windows.")

    if not np.all(
        pd.to_numeric(manifest["overlap_percent"]).astype(int).to_numpy()
        == OVERLAP_PERCENT
    ):
        raise RuntimeError("BLOCK 04 manifest contains non-50% windows.")

    if not np.all(
        pd.to_numeric(manifest["step_samples"]).astype(int).to_numpy()
        == STEP_SAMPLES
    ):
        raise RuntimeError("BLOCK 04 manifest contains a step size different from 250 samples.")

    starts = pd.to_numeric(manifest["start_sample"]).astype(int).to_numpy()
    ends = pd.to_numeric(manifest["end_sample"]).astype(int).to_numpy()
    if np.any((ends - starts) != WINDOW_SAMPLES):
        raise RuntimeError("BLOCK 04 manifest contains non-500-sample window bounds.")

    # Leakage-control certification: each subject/condition/gesture/set must
    # map to exactly one source recording, and A/B/C must map to distinct
    # source recordings. This is certified from the frozen BLOCK 04 manifest;
    # no new segmentation or performance-based decision is made here.
    source_cols = ["subject", "condition", "gesture_code", "set", "source_recording"]
    source_audit = (
        manifest[source_cols]
        .drop_duplicates()
        .groupby(["subject", "condition", "gesture_code"])
        .agg(
            n_sets=("set", "nunique"),
            n_source_recordings=("source_recording", "nunique"),
            sets=("set", lambda x: tuple(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    if not np.all(source_audit["n_sets"].to_numpy() == 3):
        raise RuntimeError("BLOCK 04 Set A/B/C certification failed: not 3 sets per recording.")
    if not np.all(source_audit["n_source_recordings"].to_numpy() == 3):
        raise RuntimeError(
            "BLOCK 04 Set A/B/C certification failed: Set A/B/C do not map "
            "to three distinct source recordings."
        )
    expected_sets = {"SetA", "SetB", "SetC"}
    if any(set(x) != expected_sets for x in source_audit["sets"]):
        raise RuntimeError(
            "BLOCK 04 Set A/B/C certification failed after normalization; "
            "expected exactly SetA/SetB/SetC."
        )

    set_independence = {
        "set_A_B_C_are_separate_recordings": True,
        "evidence": "BLOCK04 final_window_manifest.csv: each subject/condition/gesture maps SetA, SetB, SetC to three distinct source_recording values.",
        "cross_set_overlap_possible": False,
    }

    return {
        "status": "PASS",
        "configuration": "W500_O50",
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": WINDOW_SECONDS,
        "overlap_percent": OVERLAP_PERCENT,
        "step_samples": STEP_SAMPLES,
        "manifest_rows": len(manifest),
        "manifest_sha256": sha256_file(required[5]),
        "block04_complete_sha256": sha256_file(required[0]),
        "set_label_normalization": "A/B/C and SetA/SetB/SetC accepted; canonicalized to SetA/SetB/SetC",
        "set_independence": set_independence,
    }


def verify_block06_handoff():
    """
    Block 06 is checked for cohort and LOSO partition alignment only.
    No ML predictions are consumed by BLOCK 07.
    """
    complete = BLOCK06_ROOT / "BLOCK06_complete.json"
    if not complete.exists():
        raise FileNotFoundError(f"Missing BLOCK 06 completion marker: {complete}")

    obj = load_json(complete)
    if str(obj.get("status", "")).upper() != "PASS":
        raise RuntimeError("BLOCK 06 completion marker is not PASS.")

    cohort_path = BLOCK06_ROOT / "06B_ML_LOSO" / "LOSO_cohort_results.csv"
    if not cohort_path.exists():
        raise FileNotFoundError(f"Missing BLOCK 06 LOSO cohort result: {cohort_path}")

    cohort = pd.read_csv(cohort_path)
    if len(cohort) != 10:
        raise RuntimeError(
            f"BLOCK 06 LOSO cohort result has {len(cohort)} rows; expected 10."
        )

    # Inspect fold-level subject results without using their predictions.
    subject_candidates = [
        BLOCK06_ROOT / "06B_ML_LOSO" / "subject_results.csv",
        BLOCK06_ROOT / "06B_ML_LOSO" / "LOSO_subject_results.csv",
    ]
    subject_path = next((p for p in subject_candidates if p.exists()), None)

    if subject_path is not None:
        sr = pd.read_csv(subject_path)
        if "held_out_subject" in sr.columns:
            held = sorted(
                pd.to_numeric(sr["held_out_subject"]).astype(int).unique().tolist()
            )
            if held != list(SUBJECTS):
                raise RuntimeError(
                    "BLOCK 06 LOSO held-out subject set does not equal 1..25."
                )

    b06_cfg_path = BLOCK06_ROOT / "BLOCK06_config.json"
    if b06_cfg_path.exists():
        b06_cfg = load_json(b06_cfg_path)
        if b06_cfg.get("window_configuration") not in (None, "W500_O50"):
            raise RuntimeError("BLOCK 06 used a different window configuration.")
        if "step_samples" in b06_cfg and safe_int(b06_cfg["step_samples"]) != STEP_SAMPLES:
            raise RuntimeError("BLOCK 06 used a different step size.")
        if "window_samples" in b06_cfg and safe_int(b06_cfg["window_samples"]) != WINDOW_SAMPLES:
            raise RuntimeError("BLOCK 06 used a different window length.")
    else:
        print("WARNING: BLOCK 06 config artifact not found; partition alignment remains verified from cohort/fold outputs.")

    # Cross-check the engineered-feature branch only as an alignment audit.
    # BLOCK 05 is NOT used as a DL input. If its frozen artifacts exist, they
    # must document the same BLOCK 04 W500_O50 handoff and 22,050 windows.
    b05_summary_path = BLOCK05_ROOT / "BLOCK05_summary.json"
    b05_complete_path = BLOCK05_ROOT / "BLOCK05_complete.json"
    b05_alignment = "NOT_CHECKED"
    if b05_summary_path.exists() and b05_complete_path.exists():
        b05_summary = load_json(b05_summary_path)
        b05_complete = load_json(b05_complete_path)
        if str(b05_complete.get("status", "")).upper() != "PASS":
            raise RuntimeError("BLOCK 05 completion marker exists but is not PASS; ML/DL alignment cannot be certified.")
        b04_cfg = b05_summary.get("block04_configuration", {})
        if isinstance(b04_cfg, dict):
            if b04_cfg.get("selected_configuration_id") not in (None, "W500_O50"):
                raise RuntimeError("BLOCK 05 documents a different BLOCK 04 window configuration.")
            if b04_cfg.get("selected_window_samples") not in (None, WINDOW_SAMPLES):
                raise RuntimeError("BLOCK 05 documents a different window length.")
            if b04_cfg.get("selected_overlap_percent") not in (None, OVERLAP_PERCENT):
                raise RuntimeError("BLOCK 05 documents a different overlap.")
            if b04_cfg.get("selected_step_samples") not in (None, STEP_SAMPLES):
                raise RuntimeError("BLOCK 05 documents a different step size.")
        b05_alignment = "PASS"

    return {
        "status": "PASS",
        "cohort_rows": len(cohort),
        "loso_subject_alignment": "verified where subject-level artifact exists",
        "window_configuration": "W500_O50",
        "step_samples": STEP_SAMPLES,
        "block05_ml_alignment": b05_alignment,
    }


# =============================================================================
# 06. BLOCK 04 FINAL WINDOW DISCOVERY
# =============================================================================

def discover_window_files():
    if not FINAL_WINDOWS_ROOT.exists():
        raise FileNotFoundError(FINAL_WINDOWS_ROOT)

    files = sorted(FINAL_WINDOWS_ROOT.glob("*.npz"))
    if len(files) != N_RECORDINGS:
        raise RuntimeError(
            f"BLOCK 04 final window directory contains {len(files)} NPZ files; "
            f"expected {N_RECORDINGS}."
        )

    return files


def parse_window_filename(path):
    # Expected: S01_ME_1_SetA_windows.npz
    stem = path.stem
    parts = stem.split("_")
    if len(parts) != 5 or parts[-1] != "windows":
        raise RuntimeError(f"Unexpected BLOCK 04 window filename: {path.name}")

    subject_token, condition, gesture_token, set_token, _ = parts
    if not subject_token.startswith("S"):
        raise RuntimeError(f"Invalid subject token: {subject_token}")

    subject = safe_int(subject_token[1:])
    condition = condition_normalize(condition)
    gesture = safe_int(gesture_token)
    set_name = set_normalize(set_token)

    if subject not in SUBJECTS:
        raise RuntimeError(f"Subject outside cohort: {subject}")
    if gesture not in GESTURE_IDS:
        raise RuntimeError(f"Gesture outside 1..7: {gesture}")

    return subject, condition, gesture, set_name


def validate_window_file(path):
    subject, condition, gesture, set_name = parse_window_filename(path)

    with np.load(path, allow_pickle=False) as z:
        if "eeg_windows" not in z:
            raise RuntimeError(f"Missing eeg_windows in {path}")

        X = np.asarray(z["eeg_windows"])

        if X.ndim != 3:
            raise RuntimeError(f"{path.name}: EEG windows must be 3-D, got {X.shape}")

        # Canonical final shape: windows x channels x samples.
        if X.shape[1:] != (N_CHANNELS, WINDOW_SAMPLES):
            raise RuntimeError(
                f"{path.name}: expected (*,{N_CHANNELS},{WINDOW_SAMPLES}), "
                f"got {X.shape}"
            )

        if X.shape[0] != WINDOWS_PER_RECORDING:
            raise RuntimeError(
                f"{path.name}: expected {WINDOWS_PER_RECORDING} windows, got {X.shape[0]}"
            )

        if not np.isfinite(X).all():
            raise RuntimeError(f"{path.name}: non-finite EEG values.")

        if "usable_channel_mask" not in z:
            raise RuntimeError(
                f"{path.name}: missing usable_channel_mask from BLOCK 03."
            )
        mask = np.asarray(z["usable_channel_mask"]).astype(bool).reshape(-1)
        if mask.size != N_CHANNELS:
            raise RuntimeError(
                f"{path.name}: usable_channel_mask has {mask.size} entries; "
                f"expected {N_CHANNELS}."
            )
        # Unusable channels are permitted only when they are structurally flat.
        # Channel-wise training normalization then maps such a channel to zero
        # deterministically (std floor), so no interpolation/imputation is added.
        unusable = np.flatnonzero(~mask)
        if unusable.size:
            channel_std = X[:, unusable, :].std(axis=(0, 2))
            if np.any(channel_std > 1e-6):
                bad = unusable[channel_std > 1e-6].tolist()
                raise RuntimeError(
                    f"{path.name}: unusable channels {bad} are not flat. "
                    "The frozen Block 07 pipeline permits only structurally flat "
                    "masked channels; no interpolation or imputation is allowed."
                )

        required = [
            "subject",
            "condition",
            "gesture_code",
            "set_code",
            "window_samples",
            "overlap_percent",
            "step_samples",
            "start_samples",
            "end_samples",
        ]
        for key in required:
            if key not in z:
                raise RuntimeError(f"{path.name}: missing metadata key {key!r}")

        stored_subject = safe_int(np.asarray(z["subject"]).reshape(-1)[0])
        stored_gesture = safe_int(np.asarray(z["gesture_code"]).reshape(-1)[0])
        stored_condition = condition_normalize(
            np.asarray(z["condition"]).reshape(-1)[0]
        )

        if stored_subject != subject:
            raise RuntimeError(f"{path.name}: subject metadata mismatch.")
        if stored_gesture != gesture:
            raise RuntimeError(f"{path.name}: gesture metadata mismatch.")
        if stored_condition != condition:
            raise RuntimeError(f"{path.name}: condition metadata mismatch.")

        starts = np.asarray(z["start_samples"]).astype(int).reshape(-1)
        ends = np.asarray(z["end_samples"]).astype(int).reshape(-1)

        if len(starts) != WINDOWS_PER_RECORDING:
            raise RuntimeError(f"{path.name}: invalid start_sample length.")
        if len(ends) != WINDOWS_PER_RECORDING:
            raise RuntimeError(f"{path.name}: invalid end_sample length.")

        if np.any(starts < 0) or np.any(ends > 5625):
            raise RuntimeError(f"{path.name}: window bounds outside recording.")

        if np.any((ends - starts) != WINDOW_SAMPLES):
            raise RuntimeError(f"{path.name}: invalid window lengths.")

        if np.any(np.diff(starts) != STEP_SAMPLES):
            raise RuntimeError(f"{path.name}: invalid window step.")

    return {
        "path": str(path),
        "subject": subject,
        "condition": condition,
        "gesture": gesture,
        "set": set_name,
        "windows": WINDOWS_PER_RECORDING,
        "unusable_channels": int((~mask).sum()),
        "usable_channel_mask_all_true": bool(mask.all()),
    }


def build_window_inventory(files):
    rows = []
    for i, path in enumerate(files, 1):
        rows.append(validate_window_file(path))
        if i % 100 == 0 or i == len(files):
            print(f"  validated Block 04 files: {i}/{len(files)}")

    inv = pd.DataFrame(rows)

    if len(inv) != N_RECORDINGS:
        raise RuntimeError("Window inventory size mismatch.")

    keys = inv[["subject", "condition", "gesture", "set"]].astype(str).agg("|".join, axis=1)
    if keys.duplicated().any():
        raise RuntimeError("Duplicate logical recording identities in Block 04 windows.")

    if inv.groupby(["subject", "condition"]).size().min() != 21:
        raise RuntimeError("Unexpected recording count per subject/condition.")

    # BLOCK 04 defines Set A/B/C as separate continuous recordings. Therefore
    # the 50% overlap exists only between windows inside one recording; it does
    # not cross Set A/B/C boundaries. Enforce that each logical set has one
    # source recording and that source recordings are unique across sets.
    logical = (
        inv.groupby(["subject", "condition", "gesture", "set"])
        .agg(
            n_files=("path", "nunique"),
            source_recording=("path", lambda s: tuple(sorted(map(str, s)))),
        )
        .reset_index()
    )
    if (logical["n_files"] != 1).any():
        raise RuntimeError(
            "A logical subject/condition/gesture/set maps to multiple "
            "BLOCK 04 source files."
        )

    dup_recordings = inv["path"].duplicated().sum()
    if dup_recordings != 0:
        raise RuntimeError("Duplicate BLOCK 04 source-file entries detected.")

    atomic_write_csv(BLOCK07_ROOT / "window_input_inventory.csv", inv)
    atomic_write_csv(BLOCK07_ROOT / "source_recording_set_audit.csv", logical)
    return inv


# =============================================================================
# 07. WINDOW DATASET
# =============================================================================

class EEGWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64) - 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EEGOnlyDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


def load_recording(path):
    with np.load(path, allow_pickle=False) as z:
        X = np.asarray(z["eeg_windows"], dtype=np.float32)
        y = np.full(X.shape[0], safe_int(np.asarray(z["gesture_code"]).reshape(-1)[0]), dtype=np.int64)
        starts = np.asarray(z["start_samples"]).astype(int)
        ends = np.asarray(z["end_samples"]).astype(int)
    return X, y, starts, ends


def load_selected_windows(file_paths):
    arrays = []
    labels = []
    starts_all = []
    ends_all = []
    metadata = []

    for path in file_paths:
        info = parse_window_filename(path)
        X, y, starts, ends = load_recording(path)

        arrays.append(X)
        labels.append(y)
        starts_all.append(starts)
        ends_all.append(ends)

        subject, condition, gesture, set_name = info
        for row in range(len(y)):
            metadata.append({
                "subject": subject,
                "condition": condition,
                "gesture_code": gesture,
                "set": set_name,
                "source_file": path.name,
                "window_in_recording": row,
                "start_samples": int(starts[row]),
                "end_samples": int(ends[row]),
                "window_id": f"{path.stem}_W{row:02d}",
            })

    X = np.concatenate(arrays, axis=0)
    y = np.concatenate(labels, axis=0)
    meta = pd.DataFrame(metadata)

    if X.shape[0] != len(meta):
        raise RuntimeError("Window/metadata length mismatch.")

    return X, y, meta


# =============================================================================
# 08. TRAINING-ONLY NORMALIZATION
# =============================================================================

class ChannelStandardizer:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3 or X.shape[1:] != (N_CHANNELS, WINDOW_SAMPLES):
            raise ValueError(f"Invalid normalization input shape: {X.shape}")

        self.mean_ = X.mean(axis=(0, 2), dtype=np.float64)
        self.std_ = X.std(axis=(0, 2), dtype=np.float64)

        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("Standardizer is not fitted.")
        X = np.asarray(X, dtype=np.float32)
        return (
            (X - self.mean_[None, :, None].astype(np.float32))
            / self.std_[None, :, None].astype(np.float32)
        ).astype(np.float32)

    def state(self):
        return {
            "method": NORMALIZATION,
            "channel_mean": self.mean_.tolist(),
            "channel_std": self.std_.tolist(),
        }


# =============================================================================
# 09. DL ARCHITECTURES
# =============================================================================

class EEGNet(nn.Module):
    def __init__(self, n_channels=13, n_samples=500, n_classes=7, dropout=None):
        super().__init__()
        dropout = MODEL_DROPOUT["EEGNet"] if dropout is None else dropout
        self.temporal = nn.Conv2d(
            1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False
        )
        self.bn1 = nn.BatchNorm2d(8)
        self.depthwise = nn.Conv2d(
            8, 16, kernel_size=(n_channels, 1),
            groups=8, bias=False
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.separable = nn.Conv2d(
            16, 16, kernel_size=(1, 16), padding=(0, 8), bias=False
        )
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            z = self._features(dummy)
            dim = z.flatten(1).shape[1]

        self.representation_dim = dim
        self.classifier = nn.Linear(dim, n_classes)

    def _features(self, x):
        x = F.elu(self.bn1(self.temporal(x)))
        x = F.elu(self.bn2(self.depthwise(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        x = F.elu(self.bn3(self.separable(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    def forward_features(self, x):
        z = self._features(x)
        return z.flatten(1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


class ShallowConvNet(nn.Module):
    def __init__(self, n_channels=13, n_samples=500, n_classes=7, dropout=None):
        super().__init__()
        dropout = MODEL_DROPOUT["ShallowConvNet"] if dropout is None else dropout
        self.conv_time = nn.Conv2d(
            1, 40, kernel_size=(1, 25), padding=(0, 12), bias=False
        )
        self.conv_space = nn.Conv2d(
            40, 40, kernel_size=(n_channels, 1), bias=False
        )
        self.bn = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d(kernel_size=(1, 38), stride=(1, 8))
        self.drop = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            z = self._features(dummy)
            dim = z.flatten(1).shape[1]

        self.representation_dim = dim
        self.classifier = nn.Linear(dim, n_classes)

    def _features(self, x):
        x = self.conv_time(x)
        x = self.conv_space(x)
        x = self.bn(x)
        x = x * x
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.drop(x)
        return x

    def forward_features(self, x):
        return self._features(x).flatten(1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


class DeepConvNet(nn.Module):
    def __init__(self, n_channels=13, n_samples=500, n_classes=7, dropout=None):
        super().__init__()
        dropout = MODEL_DROPOUT["DeepConvNet"] if dropout is None else dropout
        blocks = []
        in_ch = 1
        for out_ch in (25, 50, 100, 200):
            blocks.append(
                nn.Conv2d(
                    in_ch, out_ch,
                    kernel_size=(1, 10) if in_ch == 1 else (1, 5),
                    padding=(0, 5) if in_ch == 1 else (0, 2),
                    bias=False,
                )
            )
            blocks.append(nn.BatchNorm2d(out_ch))
            blocks.append(nn.ELU())
            if in_ch == 1:
                blocks.append(
                    nn.Conv2d(out_ch, out_ch, kernel_size=(n_channels, 1), bias=False)
                )
                blocks.append(nn.BatchNorm2d(out_ch))
                blocks.append(nn.ELU())
            blocks.append(nn.MaxPool2d(kernel_size=(1, 2)))
            blocks.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            z = self.features(dummy)
            dim = z.flatten(1).shape[1]

        self.representation_dim = dim
        self.classifier = nn.Linear(dim, n_classes)

    def forward_features(self, x):
        return self.features(x).flatten(1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        n_channels=13,
        n_samples=500,
        n_classes=7,
        attention=False,
        dropout=None,
    ):
        super().__init__()
        model_key = "CNN-BiLSTM-Attention" if attention else "CNN-BiLSTM"
        dropout = MODEL_DROPOUT[model_key] if dropout is None else dropout
        self.attention = attention

        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=15, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, padding=4, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        if attention:
            self.attention_layer = nn.Linear(128, 1)

        self.dropout = nn.Dropout(dropout)
        self.representation_dim = 128
        self.classifier = nn.Linear(128, n_classes)

    def cnn_features(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        return x

    def forward_features(self, x):
        z = self.cnn_features(x)
        z = z.transpose(1, 2)
        seq, _ = self.lstm(z)

        if self.attention:
            scores = self.attention_layer(seq).squeeze(-1)
            weights = torch.softmax(scores, dim=1)
            rep = torch.sum(seq * weights.unsqueeze(-1), dim=1)
        else:
            rep = torch.mean(seq, dim=1)

        return self.dropout(rep)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


def build_model(name):
    if name == "EEGNet":
        return EEGNet(dropout=MODEL_DROPOUT[name])
    if name == "ShallowConvNet":
        return ShallowConvNet(dropout=MODEL_DROPOUT[name])
    if name == "DeepConvNet":
        return DeepConvNet(dropout=MODEL_DROPOUT[name])
    if name == "CNN-BiLSTM":
        return CNNBiLSTM(attention=False, dropout=MODEL_DROPOUT[name])
    if name == "CNN-BiLSTM-Attention":
        return CNNBiLSTM(attention=True, dropout=MODEL_DROPOUT[name])
    raise ValueError(name)


# =============================================================================
# 10. ARCHITECTURE CONFIGURATION
# =============================================================================

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def model_architecture_summary(name):
    model = build_model(name)
    total, trainable = count_parameters(model)
    return {
        "model_name": name,
        "input_shape": [N_CHANNELS, WINDOW_SAMPLES],
        "output_classes": 7,
        "dropout": MODEL_DROPOUT[name],
        "parameter_count": total,
        "trainable_parameters": trainable,
        "latent_layer": "forward_features",
        "latent_dimension": int(model.representation_dim),
        "bidirectional": name in {"CNN-BiLSTM", "CNN-BiLSTM-Attention"},
        "attention": name == "CNN-BiLSTM-Attention",
    }


def build_frozen_architecture_config():
    return {
        "protocol_version": PROTOCOL_VERSION,
        "input": {
            "channels": N_CHANNELS,
            "samples": WINDOW_SAMPLES,
            "sampling_rate_hz": FS,
            "shape": [N_CHANNELS, WINDOW_SAMPLES],
        },
        "models": [model_architecture_summary(m) for m in MODELS],
        "class_order": list(GESTURE_IDS),
        "class_names": GESTURE_NAMES,
        "latent_models_for_block12": [
            "EEGNet",
            "CNN-BiLSTM",
            "CNN-BiLSTM-Attention",
        ],
    }


def build_frozen_training_config():
    return {
        "protocol_version": PROTOCOL_VERSION,
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "checkpoint_metric": CHECKPOINT_METRIC,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "loss": "CrossEntropyLoss",
        "normalization": NORMALIZATION,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "no_class_weighting": True,
        "no_smote": True,
        "no_feature_input": True,
        "no_test_monitoring": True,
        "deterministic_execution": True,
        "no_test_tuning": True,
    }


# =============================================================================
# 11. SPLIT DESIGN
# =============================================================================

def sd_indices(meta, subject, condition):
    q = (
        (meta["subject"] == subject)
        & (meta["condition"] == condition)
    )

    train_idx = meta.index[q & (meta["set"] == SD_TRAIN_SET)].to_numpy()
    val_idx = meta.index[q & (meta["set"] == SD_VALIDATION_SET)].to_numpy()
    test_idx = meta.index[q & (meta["set"] == SD_TEST_SET)].to_numpy()

    if len(train_idx) != 147:
        raise RuntimeError(
            f"SD train expected 147 windows for subject {subject} {condition}, got {len(train_idx)}"
        )
    if len(val_idx) != 147:
        raise RuntimeError(
            f"SD validation expected 147 windows for subject {subject} {condition}, got {len(val_idx)}"
        )
    if len(test_idx) != 147:
        raise RuntimeError(
            f"SD test expected 147 windows for subject {subject} {condition}, got {len(test_idx)}"
        )

    for split_name, idx in (
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ):
        counts = (
            meta.loc[idx, "gesture_code"]
            .astype(int)
            .value_counts()
            .reindex(GESTURE_IDS, fill_value=0)
        )
        if not np.all(counts.to_numpy() == WINDOWS_PER_RECORDING * 1):
            raise RuntimeError(
                f"SD {split_name} gesture counts are not exactly 21 per class "
                f"for subject {subject} {condition}: {counts.to_dict()}"
            )

    return train_idx, val_idx, test_idx


def loso_indices(meta, held_out_subject, condition):
    test_mask = (
        (meta["subject"] == held_out_subject)
        & (meta["condition"] == condition)
    )
    pool_mask = (
        (meta["subject"] != held_out_subject)
        & (meta["condition"] == condition)
    )

    train_subjects = sorted(
        meta.loc[pool_mask, "subject"].astype(int).unique().tolist()
    )
    if len(train_subjects) != 24:
        raise RuntimeError("LOSO training subject count is not 24.")

    validation_subject = train_subjects[0]
    fit_subjects = train_subjects[1:]

    train_idx = meta.index[
        pool_mask & meta["subject"].isin(fit_subjects)
    ].to_numpy()

    val_idx = meta.index[
        pool_mask & (meta["subject"] == validation_subject)
    ].to_numpy()

    test_idx = meta.index[test_mask].to_numpy()

    if len(train_idx) != 23 * 7 * 3 * WINDOWS_PER_RECORDING:
        raise RuntimeError(f"Unexpected LOSO training window count: {len(train_idx)}")
    if len(val_idx) != 7 * 3 * WINDOWS_PER_RECORDING:
        raise RuntimeError(f"Unexpected LOSO validation window count: {len(val_idx)}")
    if len(test_idx) != 7 * 3 * WINDOWS_PER_RECORDING:
        raise RuntimeError(f"Unexpected LOSO test window count: {len(test_idx)}")

    for split_name, idx, expected_per_class in (
        ("train", train_idx, 23 * 3 * WINDOWS_PER_RECORDING),
        ("validation", val_idx, 3 * WINDOWS_PER_RECORDING),
        ("test", test_idx, 3 * WINDOWS_PER_RECORDING),
    ):
        counts = (
            meta.loc[idx, "gesture_code"]
            .astype(int)
            .value_counts()
            .reindex(GESTURE_IDS, fill_value=0)
        )
        if not np.all(counts.to_numpy() == expected_per_class):
            raise RuntimeError(
                f"LOSO {split_name} gesture counts are invalid for held-out "
                f"subject {held_out_subject} {condition}: {counts.to_dict()}"
            )

    return train_idx, val_idx, test_idx, validation_subject


def write_split_audit(meta):
    rows = []

    for condition in CONDITIONS:
        for subject in SUBJECTS:
            sd_train, sd_val, sd_test = sd_indices(meta, subject, condition)
            loso_train, loso_val, loso_test, val_subject = loso_indices(
                meta, subject, condition
            )

            sd_train_subjects = set(meta.loc[sd_train, "subject"].astype(int))
            sd_val_subjects = set(meta.loc[sd_val, "subject"].astype(int))
            sd_test_subjects = set(meta.loc[sd_test, "subject"].astype(int))

            loso_train_subjects = set(meta.loc[loso_train, "subject"].astype(int))
            loso_val_subjects = set(meta.loc[loso_val, "subject"].astype(int))
            loso_test_subjects = set(meta.loc[loso_test, "subject"].astype(int))

            rows.append({
                "condition": condition,
                "held_out_subject": subject,
                "SD_train_sets": "SetA",
                "SD_validation_set": "SetB",
                "SD_test_set": "SetC",
                "SD_train_windows": len(sd_train),
                "SD_validation_windows": len(sd_val),
                "SD_test_windows": len(sd_test),
                "SD_train_subjects": sorted(sd_train_subjects),
                "SD_validation_subjects": sorted(sd_val_subjects),
                "SD_test_subjects": sorted(sd_test_subjects),
                "LOSO_train_subject_count": len(loso_train_subjects),
                "LOSO_validation_subject": val_subject,
                "LOSO_test_subject": subject,
                "LOSO_train_windows": len(loso_train),
                "LOSO_validation_windows": len(loso_val),
                "LOSO_test_windows": len(loso_test),
                "held_out_in_train": subject in loso_train_subjects,
                "held_out_in_validation": subject in loso_val_subjects,
                "train_validation_overlap": len(loso_train_subjects & loso_val_subjects),
                "train_test_overlap": len(loso_train_subjects & loso_test_subjects),
                "validation_test_overlap": len(loso_val_subjects & loso_test_subjects),
                "status": "PASS",
            })

    atomic_write_csv(
        BLOCK07_ROOT / "split_design_audit.csv",
        pd.DataFrame(rows),
    )


# =============================================================================
# 12. TRAINING / EVALUATION
# =============================================================================

def make_loader(X, y, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    ds = EEGWindowDataset(X, y)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )


def make_metrics(y_true, y_pred):
    labels = list(range(1, 8))

    if len(y_true) != len(y_pred):
        raise RuntimeError("Prediction length mismatch.")

    true_labels = set(np.asarray(y_true).tolist())
    pred_labels = set(np.asarray(y_pred).tolist())

    if true_labels != set(labels):
        raise RuntimeError(
            f"Evaluation partition does not contain all seven gesture classes: "
            f"observed_true={sorted(true_labels)}"
        )

    if not pred_labels.issubset(set(labels)):
        raise RuntimeError(
            f"Prediction contains invalid gesture codes: {sorted(pred_labels)}"
        )

    acc = accuracy_score(y_true, y_pred)
    ba = balanced_accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    class_df = pd.DataFrame({
        "gesture_code": labels,
        "gesture": [GESTURE_NAMES[x] for x in labels],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    if int(cm.sum()) != len(y_true):
        raise RuntimeError("Confusion matrix total does not equal test count.")

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(ba),
        "macro_f1": float(mf1),
        "mcc": float(mcc),
        "kappa": float(kappa),
    }, class_df, cm


def forward_input(X):
    # X: N x 13 x 500
    return X


def model_forward(model, X):
    # 2D architectures expect N x 1 x 13 x 500.
    if isinstance(model, (EEGNet, ShallowConvNet, DeepConvNet)):
        return model(X.unsqueeze(1))
    return model(X)


def run_epoch(model, loader, optimizer, device, training=True):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_n = 0
    all_true = []
    all_pred = []

    criterion = nn.CrossEntropyLoss()

    for xb, yb0 in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb0.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model_forward(model, xb)
            loss = criterion(logits, yb)

            if training:
                loss.backward()
                if GRADIENT_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        GRADIENT_CLIP_NORM,
                    )
                optimizer.step()

        total_loss += float(loss.detach().cpu()) * len(yb)
        total_n += len(yb)

        pred = torch.argmax(logits.detach(), dim=1).cpu().numpy() + 1
        true = yb.cpu().numpy() + 1

        all_true.append(true)
        all_pred.append(pred)

    if total_n == 0:
        raise RuntimeError("Empty data loader.")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    metrics, _, _ = make_metrics(y_true, y_pred)
    return float(total_loss / total_n), metrics


@torch.no_grad()
def predict_model(model, X, device):
    model.eval()

    ds = EEGOnlyDataset(X)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
    )

    predictions = []
    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model_forward(model, xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy() + 1
        predictions.append(pred)

    if not predictions:
        return np.empty((0,), dtype=np.int64)

    out = np.concatenate(predictions).astype(np.int64)
    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite predictions detected.")
    if not set(np.unique(out).tolist()).issubset(set(GESTURE_IDS)):
        raise RuntimeError("Predictions contain labels outside gesture IDs 1..7.")
    return out


def train_model(
    model_name,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    checkpoint_path,
    history_path,
    seed,
    max_epochs=None,
    batch_size_override=None,
):
    set_all_seeds(seed)

    normalizer = ChannelStandardizer().fit(X_train)
    Xtr = normalizer.transform(X_train)
    Xva = normalizer.transform(X_val)

    model = build_model(model_name).to(device)

    batch_size = batch_size_override or BATCH_SIZE

    train_loader = DataLoader(
        EEGWindowDataset(Xtr, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        EEGWindowDataset(Xva, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    epochs = max_epochs or MAX_EPOCHS
    best_metric = -np.inf
    best_epoch = 0
    patience_count = 0
    history = []

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, optimizer, device, training=True
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, optimizer, device, training=False
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "validation_accuracy": val_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
            "train_mcc": train_metrics["mcc"],
            "validation_mcc": val_metrics["mcc"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        current = val_metrics["macro_f1"]

        if current > best_metric + EARLY_STOPPING_MIN_DELTA:
            best_metric = current
            best_epoch = epoch
            patience_count = 0

            atomic_write_json(
                checkpoint_path.with_suffix(".json"),
                {
                    "status": "PASS",
                    "model_name": model_name,
                    "best_epoch": best_epoch,
                    "best_validation_macro_f1": best_metric,
                    "seed": seed,
                },
            )
            torch.save(
                {
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_macro_f1": best_metric,
                    "normalizer_mean": normalizer.mean_.tolist(),
                    "normalizer_std": normalizer.std_.tolist(),
                    "random_seed": seed,
                },
                checkpoint_path,
            )
        else:
            patience_count += 1

        if patience_count >= EARLY_STOPPING_PATIENCE:
            break

    training_seconds = time.perf_counter() - start

    atomic_write_csv(history_path, pd.DataFrame(history))

    if not checkpoint_path.exists():
        raise RuntimeError("Best checkpoint was not created.")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return {
        "model": model,
        "normalizer": normalizer,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_macro_f1": float(checkpoint["best_validation_macro_f1"]),
        "history": pd.DataFrame(history),
        "training_seconds": float(training_seconds),
    }


# =============================================================================
# 13. LATENT REPRESENTATION EXTRACTION
# =============================================================================

@torch.no_grad()
def extract_latent(model, X, device):
    model.eval()
    ds = EEGOnlyDataset(X)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
    )

    reps = []
    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        if isinstance(model, (EEGNet, ShallowConvNet, DeepConvNet)):
            z = model.forward_features(xb.unsqueeze(1))
        else:
            z = model.forward_features(xb)
        reps.append(z.detach().cpu().numpy())

    return np.concatenate(reps, axis=0)


# =============================================================================
# 14. RUN DIRECTORY / CHECKPOINT HELPERS
# =============================================================================

def sd_run_paths(subject, condition, model_name):
    root = SD_ROOT / "runs" / f"Subject_{subject:02d}" / condition / model_dir_name(model_name)
    return {
        "root": root,
        "checkpoint": root / "checkpoints" / "best_model.pt",
        "history": root / "training_history.csv",
        "prediction": root / "predictions.csv",
        "metrics": root / "metrics.json",
        "class_metrics": root / "class_metrics.csv",
        "confusion": root / "confusion_matrix.csv",
        "computational": root / "computational_metrics.json",
        "latent": root / "latent_representation.npz",
        "complete": root / "complete.json",
    }


def loso_run_paths(subject, condition, model_name):
    root = LOSO_ROOT / "folds" / f"Subject_{subject:02d}" / condition / model_dir_name(model_name)
    return {
        "root": root,
        "checkpoint": root / "checkpoints" / "best_model.pt",
        "history": root / "training_history.csv",
        "prediction": root / "predictions.csv",
        "metrics": root / "metrics.json",
        "class_metrics": root / "class_metrics.csv",
        "confusion": root / "confusion_matrix.csv",
        "computational": root / "computational_metrics.json",
        "latent": root / "latent_representation.npz",
        "complete": root / "complete.json",
    }


def is_complete_run(paths, model_name=None, requires_latent=False):
    required = [
        paths["checkpoint"],
        paths["history"],
        paths["prediction"],
        paths["metrics"],
        paths["class_metrics"],
        paths["confusion"],
        paths["computational"],
        paths["complete"],
        paths["root"] / "split_indices.npz",
    ]
    if requires_latent and model_name in LATENT_MODELS:
        required.append(paths["latent"])
    if not all(Path(p).exists() for p in required):
        return False

    try:
        complete = load_json(paths["complete"])
        expected_hash = str(complete.get("checkpoint_sha256", ""))
        if expected_hash and sha256_file(paths["checkpoint"]) != expected_hash:
            return False
    except Exception:
        return False

    return True


# =============================================================================
# 15. PREDICTION / OUTPUT HELPERS
# =============================================================================

def save_predictions(path, meta, indices, y_true, y_pred, model, regime, held_out=None):
    rows = meta.loc[indices].copy().reset_index(drop=True)
    rows["gesture_true"] = np.asarray(y_true, dtype=int)
    rows["gesture_pred"] = np.asarray(y_pred, dtype=int)
    rows["model"] = model
    rows["evaluation_regime"] = regime
    if held_out is not None:
        rows["loso_fold"] = int(held_out)
        rows["held_out_subject"] = int(held_out)

    rows["subject_id"] = rows["subject"].astype(int)

    cols = [
        "subject_id",
        "condition",
        "model",
        "evaluation_regime",
        "gesture_true",
        "gesture_pred",
        "window_id",
        "set",
        "source_file",
    ]
    if held_out is not None:
        cols += ["loso_fold", "held_out_subject"]

    atomic_write_csv(path, rows[cols])


def save_confusion(path, cm):
    df = pd.DataFrame(
        cm,
        index=[f"True_{g}" for g in GESTURE_IDS],
        columns=[f"Pred_{g}" for g in GESTURE_IDS],
    )
    atomic_write_csv(
        path,
        df.reset_index().rename(columns={"index": "true_class"}),
    )


def system_info():
    gpu_name = None
    gpu_memory_gb = None

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            pass

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "cpu": platform.processor(),
    }


# =============================================================================
# 16. SUBJECT-DEPENDENT 07A
# =============================================================================

def run_sd(X, y, meta, device):
    subject_rows = []
    class_rows = []
    execution_rows = []
    latent_rows = []

    for condition in CONDITIONS:
        for subject in SUBJECTS:
            train_idx, val_idx, test_idx = sd_indices(meta, subject, condition)

            for model_name in MODELS:
                paths = sd_run_paths(subject, condition, model_name)
                for key in ("root", "checkpoint", "history", "prediction",
                            "metrics", "class_metrics", "confusion",
                            "computational", "latent", "complete"):
                    if key in paths and key != "root":
                        Path(paths[key]).parent.mkdir(parents=True, exist_ok=True)

                if is_complete_run(paths, model_name=model_name, requires_latent=True):
                    metrics_obj = load_json(paths["metrics"])
                    class_df = pd.read_csv(paths["class_metrics"])
                    subject_rows.append(metrics_obj["subject_result"])
                    class_rows.append(class_df)
                    execution_rows.append({
                        "regime": "SD",
                        "subject": subject,
                        "condition": condition,
                        "model": model_name,
                        "status": "RESUME-SKIP",
                    })
                    continue

                np.savez_compressed(
                    paths["root"] / "split_indices.npz",
                    train_idx=train_idx.astype(np.int64),
                    val_idx=val_idx.astype(np.int64),
                    test_idx=test_idx.astype(np.int64),
                )

                set_all_seeds(RANDOM_SEED + subject)

                Xtr = X[train_idx]
                ytr = y[train_idx]
                Xva = X[val_idx]
                yva = y[val_idx]
                Xte = X[test_idx]
                yte = y[test_idx]

                t0 = time.perf_counter()
                result = train_model(
                    model_name,
                    Xtr,
                    ytr,
                    Xva,
                    yva,
                    device,
                    paths["checkpoint"],
                    paths["history"],
                    RANDOM_SEED + subject,
                )
                training_seconds = result["training_seconds"]

                normalizer = result["normalizer"]
                Xte_n = normalizer.transform(Xte)

                t1 = time.perf_counter()
                ypred = predict_model(result["model"], Xte_n, device)
                prediction_seconds = time.perf_counter() - t1

                metrics, class_df, cm = make_metrics(yte, ypred)

                save_predictions(
                    paths["prediction"],
                    meta,
                    test_idx,
                    yte,
                    ypred,
                    model_name,
                    "SD",
                )
                atomic_write_csv(paths["class_metrics"], class_df)
                save_confusion(paths["confusion"], cm)

                total_params, trainable_params = count_parameters(result["model"])

                computational = {
                    "training_seconds": float(training_seconds),
                    "prediction_seconds": float(prediction_seconds),
                    "parameter_count": total_params,
                    "trainable_parameters": trainable_params,
                    "model_size_mb": float(
                        paths["checkpoint"].stat().st_size / (1024**2)
                    ),
                    "peak_gpu_memory_mb": (
                        float(torch.cuda.max_memory_allocated() / (1024**2))
                        if torch.cuda.is_available() else None
                    ),
                }
                atomic_write_json(paths["computational"], computational)

                subject_result = {
                    "subject": subject,
                    "condition": condition,
                    "model": model_name,
                    "evaluation_regime": "SD",
                    **metrics,
                    "n_train": int(len(train_idx)),
                    "n_validation": int(len(val_idx)),
                    "n_test": int(len(test_idx)),
                    "best_epoch": int(result["best_epoch"]),
                    "best_validation_macro_f1": float(result["best_validation_macro_f1"]),
                    "training_seconds": float(training_seconds),
                    "prediction_seconds": float(prediction_seconds),
                    "test_used_for_model_selection": False,
                    "test_used_for_normalization": False,
                    "validation_used_for_checkpoint_selection": True,
                    "feature_input": "raw_EEG",
                    "feature_count": None,
                }

                atomic_write_json(
                    paths["metrics"],
                    {
                        "subject_result": subject_result,
                        "normalization": normalizer.state(),
                        "train_sets": [SD_TRAIN_SET],
                        "validation_sets": [SD_VALIDATION_SET],
                        "test_sets": [SD_TEST_SET],
                        "random_seed": RANDOM_SEED + subject,
                        "no_smote": True,
                        "no_test_tuning": True,
                    },
                )

                if model_name in {
                    "EEGNet",
                    "CNN-BiLSTM",
                    "CNN-BiLSTM-Attention",
                }:
                    # Store only final-test latent representation for downstream
                    # traceability. RSA itself is BLOCK 12.
                    Xte_latent = extract_latent(
                        result["model"], Xte_n, device
                    )
                    if Xte_latent.shape[0] != len(test_idx):
                        raise RuntimeError(
                            f"Latent rows {Xte_latent.shape[0]} != test windows {len(test_idx)} "
                            f"for {model_name} subject/condition run."
                        )
                    np.savez_compressed(
                        paths["latent"],
                        subject=np.full(len(test_idx), subject, dtype=np.int16),
                        condition=np.array([condition] * len(test_idx)),
                        gesture=yte.astype(np.int16),
                        window_id=meta.loc[test_idx, "window_id"].astype(str).to_numpy(),
                        model=np.array([model_name]),
                        layer=np.array(["forward_features"]),
                        latent_vector=Xte_latent.astype(np.float32),
                    )

                atomic_write_json(
                    paths["complete"],
                    {
                        "status": "PASS",
                        "completed_utc": now_utc(),
                        "regime": "SD",
                        "subject": subject,
                        "condition": condition,
                        "model": model_name,
                        "checkpoint_sha256": sha256_file(paths["checkpoint"]),
                        "latent_required": model_name in LATENT_MODELS,
                        "latent_present": paths["latent"].exists(),
                    },
                )

                subject_rows.append(subject_result)

                c = class_df.copy()
                c["subject"] = subject
                c["condition"] = condition
                c["model"] = model_name
                c["evaluation_regime"] = "SD"
                class_rows.append(c)

                execution_rows.append({
                    "regime": "SD",
                    "subject": subject,
                    "condition": condition,
                    "model": model_name,
                    "status": "PASS",
                    "train_windows": len(train_idx),
                    "validation_windows": len(val_idx),
                    "test_windows": len(test_idx),
                })

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(
                    f"07A SD | {condition} | S{subject:02d} | "
                    f"{model_name} | F1={metrics['macro_f1']:.4f}"
                )

    subject_df = pd.DataFrame(subject_rows)
    class_df = pd.concat(class_rows, ignore_index=True)

    atomic_write_csv(
        SD_ROOT / "subject_results.csv",
        subject_df,
    )
    atomic_write_csv(
        SD_ROOT / "class_metrics.csv",
        class_df,
    )
    atomic_write_csv(
        SD_ROOT / "execution_audit.csv",
        pd.DataFrame(execution_rows),
    )

    cohort = aggregate_subject_results(subject_df, "SD")
    atomic_write_csv(SD_ROOT / "cohort_results.csv", cohort)

    return subject_df, class_df, cohort


# =============================================================================
# 17. LOSO 07B
# =============================================================================

def run_loso(X, y, meta, device):
    subject_rows = []
    class_rows = []
    execution_rows = []
    leakage_rows = []

    for condition in CONDITIONS:
        for held_out in SUBJECTS:
            train_idx, val_idx, test_idx, validation_subject = loso_indices(
                meta, held_out, condition
            )

            for model_name in MODELS:
                paths = loso_run_paths(held_out, condition, model_name)

                for key in (
                    "root", "checkpoint", "history", "prediction",
                    "metrics", "class_metrics", "confusion",
                    "computational", "latent", "complete"
                ):
                    if key != "root":
                        Path(paths[key]).parent.mkdir(parents=True, exist_ok=True)

                if is_complete_run(paths, model_name=model_name, requires_latent=True):
                    metrics_obj = load_json(paths["metrics"])
                    class_df = pd.read_csv(paths["class_metrics"])
                    subject_rows.append(metrics_obj["subject_result"])
                    class_rows.append(class_df)
                    execution_rows.append({
                        "regime": "LOSO",
                        "condition": condition,
                        "held_out_subject": held_out,
                        "model": model_name,
                        "status": "RESUME-SKIP",
                    })
                    leakage_rows.append(metrics_obj["leakage_audit"])
                    continue

                np.savez_compressed(
                    paths["root"] / "split_indices.npz",
                    train_idx=train_idx.astype(np.int64),
                    val_idx=val_idx.astype(np.int64),
                    test_idx=test_idx.astype(np.int64),
                    validation_subject=np.array([validation_subject], dtype=np.int16),
                    held_out_subject=np.array([held_out], dtype=np.int16),
                )

                set_all_seeds(RANDOM_SEED + held_out)

                Xtr = X[train_idx]
                ytr = y[train_idx]
                Xva = X[val_idx]
                yva = y[val_idx]
                Xte = X[test_idx]
                yte = y[test_idx]

                # Held-out subject must be completely absent from training
                # and validation.
                train_subjects = sorted(
                    meta.loc[train_idx, "subject"].astype(int).unique().tolist()
                )
                val_subjects = sorted(
                    meta.loc[val_idx, "subject"].astype(int).unique().tolist()
                )
                test_subjects = sorted(
                    meta.loc[test_idx, "subject"].astype(int).unique().tolist()
                )

                if held_out in train_subjects or held_out in val_subjects:
                    raise RuntimeError(
                        f"LOSO leakage: held-out subject {held_out} entered train/validation."
                    )
                if test_subjects != [held_out]:
                    raise RuntimeError("LOSO test subject mismatch.")

                t0 = time.perf_counter()
                result = train_model(
                    model_name,
                    Xtr,
                    ytr,
                    Xva,
                    yva,
                    device,
                    paths["checkpoint"],
                    paths["history"],
                    RANDOM_SEED + held_out,
                )
                training_seconds = result["training_seconds"]

                Xte_n = result["normalizer"].transform(Xte)

                t1 = time.perf_counter()
                ypred = predict_model(result["model"], Xte_n, device)
                prediction_seconds = time.perf_counter() - t1

                metrics, class_df, cm = make_metrics(yte, ypred)

                save_predictions(
                    paths["prediction"],
                    meta,
                    test_idx,
                    yte,
                    ypred,
                    model_name,
                    "LOSO",
                    held_out=held_out,
                )
                atomic_write_csv(paths["class_metrics"], class_df)
                save_confusion(paths["confusion"], cm)

                total_params, trainable_params = count_parameters(result["model"])

                computational = {
                    "training_seconds": float(training_seconds),
                    "prediction_seconds": float(prediction_seconds),
                    "parameter_count": total_params,
                    "trainable_parameters": trainable_params,
                    "model_size_mb": float(
                        paths["checkpoint"].stat().st_size / (1024**2)
                    ),
                    "peak_gpu_memory_mb": (
                        float(torch.cuda.max_memory_allocated() / (1024**2))
                        if torch.cuda.is_available() else None
                    ),
                }
                atomic_write_json(paths["computational"], computational)

                subject_result = {
                    "subject": held_out,
                    "held_out_subject": held_out,
                    "condition": condition,
                    "model": model_name,
                    "evaluation_regime": "LOSO",
                    **metrics,
                    "n_train": int(len(train_idx)),
                    "n_validation": int(len(val_idx)),
                    "n_test": int(len(test_idx)),
                    "validation_subject": int(validation_subject),
                    "best_epoch": int(result["best_epoch"]),
                    "best_validation_macro_f1": float(result["best_validation_macro_f1"]),
                    "training_seconds": float(training_seconds),
                    "prediction_seconds": float(prediction_seconds),
                    "test_used_for_model_selection": False,
                    "test_used_for_normalization": False,
                    "validation_used_for_checkpoint_selection": True,
                    "feature_input": "raw_EEG",
                    "feature_count": None,
                }

                leakage_audit = {
                    "regime": "LOSO",
                    "condition": condition,
                    "model": model_name,
                    "held_out_subject": held_out,
                    "train_subjects": train_subjects,
                    "validation_subjects": val_subjects,
                    "test_subjects": test_subjects,
                    "train_test_overlap": sorted(set(train_subjects) & set(test_subjects)),
                    "validation_test_overlap": sorted(set(val_subjects) & set(test_subjects)),
                    "held_out_in_train": held_out in train_subjects,
                    "held_out_in_validation": held_out in val_subjects,
                    "test_used_for_normalization": False,
                    "test_used_for_checkpoint_selection": False,
                    "test_used_for_model_selection": False,
                    "test_used_for_hyperparameter_tuning": False,
                    "status": "PASS",
                }

                atomic_write_json(
                    paths["metrics"],
                    {
                        "subject_result": subject_result,
                        "leakage_audit": leakage_audit,
                        "normalization": result["normalizer"].state(),
                        "validation_rule": LOSO_VALIDATION_RULE,
                        "random_seed": RANDOM_SEED + held_out,
                        "no_smote": True,
                        "no_test_tuning": True,
                    },
                )

                if model_name in {
                    "EEGNet",
                    "CNN-BiLSTM",
                    "CNN-BiLSTM-Attention",
                }:
                    Xte_latent = extract_latent(
                        result["model"], Xte_n, device
                    )
                    if Xte_latent.shape[0] != len(test_idx):
                        raise RuntimeError(
                            f"Latent rows {Xte_latent.shape[0]} != test windows {len(test_idx)} "
                            f"for {model_name} subject/condition run."
                        )
                    np.savez_compressed(
                        paths["latent"],
                        subject=np.full(len(test_idx), held_out, dtype=np.int16),
                        condition=np.array([condition] * len(test_idx)),
                        gesture=yte.astype(np.int16),
                        window_id=meta.loc[test_idx, "window_id"].astype(str).to_numpy(),
                        model=np.array([model_name]),
                        layer=np.array(["forward_features"]),
                        latent_vector=Xte_latent.astype(np.float32),
                    )

                atomic_write_json(
                    paths["complete"],
                    {
                        "status": "PASS",
                        "completed_utc": now_utc(),
                        "regime": "LOSO",
                        "held_out_subject": held_out,
                        "condition": condition,
                        "model": model_name,
                        "checkpoint_sha256": sha256_file(paths["checkpoint"]),
                        "latent_required": model_name in LATENT_MODELS,
                        "latent_present": paths["latent"].exists(),
                    },
                )

                subject_rows.append(subject_result)

                c = class_df.copy()
                c["held_out_subject"] = held_out
                c["condition"] = condition
                c["model"] = model_name
                c["evaluation_regime"] = "LOSO"
                class_rows.append(c)

                execution_rows.append({
                    "regime": "LOSO",
                    "condition": condition,
                    "held_out_subject": held_out,
                    "model": model_name,
                    "status": "PASS",
                    "train_windows": len(train_idx),
                    "validation_windows": len(val_idx),
                    "test_windows": len(test_idx),
                    "validation_subject": validation_subject,
                })
                leakage_rows.append(leakage_audit)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(
                    f"07B LOSO | {condition} | held S{held_out:02d} | "
                    f"{model_name} | F1={metrics['macro_f1']:.4f}"
                )

    subject_df = pd.DataFrame(subject_rows)
    class_df = pd.concat(class_rows, ignore_index=True)

    atomic_write_csv(
        LOSO_ROOT / "subject_results.csv",
        subject_df,
    )
    atomic_write_csv(
        LOSO_ROOT / "class_metrics.csv",
        class_df,
    )
    atomic_write_csv(
        LOSO_ROOT / "execution_audit.csv",
        pd.DataFrame(execution_rows),
    )
    atomic_write_csv(
        LOSO_ROOT / "leakage_audit.csv",
        pd.DataFrame(leakage_rows),
    )

    cohort = aggregate_subject_results(subject_df, "LOSO")
    atomic_write_csv(LOSO_ROOT / "cohort_results.csv", cohort)

    return subject_df, class_df, cohort


# =============================================================================
# 18. COHORT AGGREGATION
# =============================================================================

def aggregate_subject_results(subject_df, regime):
    rows = []

    for (condition, model), g in subject_df.groupby(["condition", "model"]):
        row = {
            "evaluation_regime": regime,
            "condition": condition,
            "model": model,
            "n_subjects": int(g["subject"].nunique()),
        }

        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "mcc",
            "kappa",
        ):
            values = g[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1))
            row[f"{metric}_iqr"] = float(
                np.percentile(values, 75) - np.percentile(values, 25)
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# 19. SD–LOSO COMPARISON
# =============================================================================

def build_gap_table(sd_subject, loso_subject):
    a = sd_subject[
        ["subject", "condition", "model",
         "accuracy", "balanced_accuracy", "macro_f1", "mcc", "kappa"]
    ].copy()

    b = loso_subject[
        ["subject", "condition", "model",
         "accuracy", "balanced_accuracy", "macro_f1", "mcc", "kappa"]
    ].copy()

    merged = a.merge(
        b,
        on=["subject", "condition", "model"],
        suffixes=("_sd", "_loso"),
        validate="one_to_one",
    )

    rows = []
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "mcc", "kappa"):
        tmp = merged[
            ["subject", "condition", "model",
             f"{metric}_sd", f"{metric}_loso"]
        ].copy()
        tmp["metric"] = metric
        tmp["sd_value"] = tmp[f"{metric}_sd"]
        tmp["loso_value"] = tmp[f"{metric}_loso"]
        tmp["gap"] = tmp["sd_value"] - tmp["loso_value"]
        rows.append(
            tmp[
                ["subject", "condition", "model",
                 "metric", "sd_value", "loso_value", "gap"]
            ]
        )

    return pd.concat(rows, ignore_index=True)


# =============================================================================
# 20. REPRODUCIBILITY TEST
# =============================================================================

def determinism_smoke_test(device):
    """
    Pre-flight forward/backward test under strict deterministic PyTorch mode.
    This does not train or produce any scientific result.
    """
    for model_name in MODELS:
        set_all_seeds(RANDOM_SEED + 9000)
        model = build_model(model_name).to(device)
        model.train()
        x = torch.randn(
            4, N_CHANNELS, WINDOW_SAMPLES,
            device=device,
            dtype=torch.float32,
        )
        if model_name in {"EEGNet", "ShallowConvNet", "DeepConvNet"}:
            x_model = x.unsqueeze(1)
        else:
            x_model = x

        try:
            model.zero_grad(set_to_none=True)
            logits = model(x_model)
            loss = logits.sum()
            loss.backward()
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite logits")
        except Exception as exc:
            raise RuntimeError(
                f"Strict deterministic smoke test failed for {model_name}: {exc}"
            ) from exc
        finally:
            del model, x, x_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return True


def write_environment_lock():
    env = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None
        ),
        "gpu_count": int(torch.cuda.device_count()),
        "deterministic_algorithms": True,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "pip_freeze": [],
    }
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        env["pip_freeze"] = [
            line.strip() for line in result.stdout.splitlines()
            if line.strip()
        ]
    except Exception as exc:
        env["pip_freeze_error"] = repr(exc)

    atomic_write_json(CONFIG_ROOT / "environment.json", env)
    return env


def run_reproducibility_test(X, y, meta, device):
    """
    Pre-flight reproducibility test using two representative architectures.

    STRICT TEST-SET PROTECTION:
    This diagnostic uses only the SD training partition (Set A) and
    validation partition (Set B). Set C is never loaded, normalized,
    predicted on, or otherwise accessed by this function.

    The diagnostic is never included in the definitive benchmark.
    """
    test_models = ("EEGNet", "CNN-BiLSTM-Attention")
    rows = []
    all_pass = True

    subject = REPRO_TEST_SUBJECT
    condition = REPRO_TEST_CONDITION

    q = (
        (meta["subject"] == subject)
        & (meta["condition"] == condition)
    )
    train_idx = meta.index[q & (meta["set"] == SD_TRAIN_SET)].to_numpy()
    val_idx = meta.index[q & (meta["set"] == SD_VALIDATION_SET)].to_numpy()

    if len(train_idx) != 147 or len(val_idx) != 147:
        raise RuntimeError(
            "Reproducibility diagnostic partition is invalid: "
            f"expected 147 SetA train and 147 SetB validation windows, "
            f"got {len(train_idx)} and {len(val_idx)}."
        )

    diagnostic_sets = set(
        meta.loc[np.concatenate([train_idx, val_idx]), "set"].astype(str).unique()
    )
    if diagnostic_sets != {SD_TRAIN_SET, SD_VALIDATION_SET}:
        raise RuntimeError(
            "Reproducibility diagnostic accessed a partition other than "
            f"{SD_TRAIN_SET}/{SD_VALIDATION_SET}: {sorted(diagnostic_sets)}"
        )

    for model_name in test_models:
        def one_run(run_id):
            set_all_seeds(RANDOM_SEED + 1000)
            result = train_model(
                model_name,
                X[train_idx],
                y[train_idx],
                X[val_idx],
                y[val_idx],
                device,
                DIAG_ROOT / f"repro_{model_dir_name(model_name)}_{run_id}_best.pt",
                DIAG_ROOT / f"repro_{model_dir_name(model_name)}_{run_id}_history.csv",
                RANDOM_SEED + 1000,
                max_epochs=REPRO_TEST_MAX_EPOCHS,
                batch_size_override=REPRO_TEST_BATCH_SIZE,
            )

            Xval = result["normalizer"].transform(X[val_idx])
            pred = predict_model(result["model"], Xval, device)
            history = result["history"]

            return (
                pred,
                history,
                result["best_epoch"],
                result["best_validation_macro_f1"],
            )

        p1, h1, e1, f1 = one_run("A")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        p2, h2, e2, f2 = one_run("B")

        prediction_identical = bool(np.array_equal(p1, p2))
        history_identical = bool(
            np.allclose(
                h1.to_numpy(dtype=float),
                h2.to_numpy(dtype=float),
                rtol=1e-5,
                atol=1e-6,
                equal_nan=True,
            )
        )
        epoch_identical = bool(e1 == e2)
        best_f1_identical = bool(np.isclose(f1, f2, rtol=1e-4, atol=1e-4))
        status = (
            prediction_identical
            and history_identical
            and epoch_identical
            and best_f1_identical
        )
        all_pass = all_pass and status

        rows.append({
            "model": model_name,
            "condition": condition,
            "subject": subject,
            "epochs": REPRO_TEST_MAX_EPOCHS,
            "diagnostic_train_set": SD_TRAIN_SET,
            "diagnostic_validation_set": SD_VALIDATION_SET,
            "diagnostic_test_set_accessed": False,
            "prediction_identical": prediction_identical,
            "history_identical_within_tolerance": history_identical,
            "best_epoch_identical": epoch_identical,
            "best_validation_macro_f1_identical": best_f1_identical,
            "status": "PASS" if status else "FAIL",
        })

    atomic_write_csv(
        DIAG_ROOT / "reproducibility_test.csv",
        pd.DataFrame(rows),
    )
    return all_pass


def build_latent_inventory():
    rows = []

    for regime_root, regime in (
        (SD_ROOT, "SD"),
        (LOSO_ROOT, "LOSO"),
    ):
        for p in regime_root.rglob("latent_representation.npz"):
            with np.load(p, allow_pickle=False) as z:
                shape = list(z["latent_vector"].shape)
                model = str(np.asarray(z["model"]).reshape(-1)[0])

            rows.append({
                "regime": regime,
                "path": str(p),
                "model": model,
                "shape": shape,
            })

    atomic_write_csv(
        BLOCK07_ROOT / "latent_representation_inventory.csv",
        pd.DataFrame(rows),
    )


# =============================================================================
# 22. FINAL AUDIT
# =============================================================================

def final_audit(sd_subject, loso_subject, sd_cohort, loso_cohort):
    failures = []

    expected_sd = 25 * 2 * 5
    expected_loso = 25 * 2 * 5

    if len(sd_subject) != expected_sd:
        failures.append(
            f"SD subject result rows={len(sd_subject)} expected={expected_sd}"
        )

    if len(loso_subject) != expected_loso:
        failures.append(
            f"LOSO subject result rows={len(loso_subject)} expected={expected_loso}"
        )

    if sorted(sd_subject["subject"].astype(int).unique()) != list(SUBJECTS):
        failures.append("SD subject set is not 1..25.")

    if sorted(loso_subject["held_out_subject"].astype(int).unique()) != list(SUBJECTS):
        failures.append("LOSO held-out subject set is not 1..25.")

    for condition in CONDITIONS:
        for model in MODELS:
            a = sd_subject[
                (sd_subject["condition"] == condition)
                & (sd_subject["model"] == model)
            ]
            b = loso_subject[
                (loso_subject["condition"] == condition)
                & (loso_subject["model"] == model)
            ]

            if len(a) != 25:
                failures.append(f"SD incomplete: {condition}/{model}")
            if len(b) != 25:
                failures.append(f"LOSO incomplete: {condition}/{model}")

    # Verify every prediction file count.
    pred_sd = list(SD_ROOT.rglob("predictions.csv"))
    pred_loso = list(LOSO_ROOT.rglob("predictions.csv"))

    if len(pred_sd) != expected_sd:
        failures.append(f"SD prediction files={len(pred_sd)} expected={expected_sd}")

    if len(pred_loso) != expected_loso:
        failures.append(f"LOSO prediction files={len(pred_loso)} expected={expected_loso}")

    audit = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "sd_runs": len(sd_subject),
        "loso_runs": len(loso_subject),
        "expected_sd_runs": expected_sd,
        "expected_loso_runs": expected_loso,
        "models": list(MODELS),
        "conditions": list(CONDITIONS),
        "subjects": list(SUBJECTS),
        "block04_configuration": "W500_O50",
        "input": "raw_EEG",
        "completed_utc": now_utc(),
    }

    atomic_write_json(BLOCK07_ROOT / "final_audit.json", audit)

    if failures:
        raise RuntimeError("BLOCK 07 final audit failed: " + " | ".join(failures))

    return audit


# =============================================================================
# 23. CONFIG / SUMMARY / README
# =============================================================================

def write_configs(block04_audit, block06_audit, architecture_config, training_config):
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "block": "07_DL",
        "status": "FROZEN_BEFORE_DEFINITIVE_EXECUTION",
        "cohort": {
            "subjects": list(SUBJECTS),
            "n_subjects": N_SUBJECTS,
            "conditions": list(CONDITIONS),
            "gestures": list(GESTURE_IDS),
            "gesture_names": GESTURE_NAMES,
        },
        "input": {
            "source": "BLOCK04_final_windows",
            "feature_input": "raw_EEG",
            "channels": N_CHANNELS,
            "samples": WINDOW_SAMPLES,
            "shape": [N_CHANNELS, WINDOW_SAMPLES],
            "sampling_rate_hz": FS,
            "window_configuration": "W500_O50",
        "set_boundary_policy": (
            "Set A, Set B, and Set C are separate continuous recordings; "
            "50% overlap is confined within each source recording and cannot "
            "cross recording/set boundaries."
        ),
        },
        "subject_dependent": {
            "train_set": SD_TRAIN_SET,
            "validation_set": SD_VALIDATION_SET,
            "test_set": SD_TEST_SET,
        },
        "loso": {
            "folds": N_SUBJECTS,
            "train_subjects_per_fold": 24,
            "validation_subject_rule": LOSO_VALIDATION_RULE,
            "test_subjects_per_fold": 1,
        },
        "models": list(MODELS),
        "architecture_config": architecture_config,
        "training_config": training_config,
        "block04_handoff": block04_audit,
        "block06_partition_alignment": block06_audit,
        "forbidden": {
            "smote": False,
            "random_window_split": False,
            "cross_condition_training": False,
            "test_set_tuning": False,
            "test_set_normalization": False,
            "calibration": False,
            "transfer": False,
            "rsa": False,
            "xai": False,
            "robustness": False,
        },
        "metrics": [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "mcc",
            "kappa",
        ],
        "latent_models_for_block12": [
            "EEGNet",
            "CNN-BiLSTM",
            "CNN-BiLSTM-Attention",
        ],
        "system": system_info(),
    }

    atomic_write_json(CONFIG_ROOT / "BLOCK07_config.json", config)
    atomic_write_json(
        CONFIG_ROOT / "BLOCK07_architecture_config.json",
        architecture_config,
    )

    for model in MODELS:
        one = next(
            x for x in architecture_config["models"]
            if x["model_name"] == model
        )
        atomic_write_json(
            CONFIG_ROOT / f"{model_dir_name(model)}_config.json",
            one,
        )

    return config


def write_summary(sd_subject, loso_subject, sd_cohort, loso_cohort, final_audit, repro):
    summary = {
        "status": "PASS",
        "block": "07_DL",
        "eligible_subjects": N_SUBJECTS,
        "ME_subjects": N_SUBJECTS,
        "MI_subjects": N_SUBJECTS,
        "paired_subjects": N_SUBJECTS,
        "number_of_models": len(MODELS),
        "number_of_SD_runs": int(len(sd_subject)),
        "number_of_LOSO_folds": N_SUBJECTS,
        "number_of_LOSO_runs": int(len(loso_subject)),
        "successful_runs": int(len(sd_subject) + len(loso_subject)),
        "failed_runs": 0,
        "missing_runs": 0,
        "feature_input": "raw_EEG",
        "window_configuration": "W500_O50",
        "window_shape": [N_CHANNELS, WINDOW_SAMPLES],
        "total_windows_upstream": TOTAL_WINDOWS,
        "metrics": [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "mcc",
            "kappa",
        ],
        "reproducibility_test": "PASS" if repro else "REVIEW",
        "final_audit": final_audit,
        "configuration_hash": sha256_file(
            CONFIG_ROOT / "BLOCK07_config.json"
        ),
    }

    atomic_write_json(BLOCK07_ROOT / "BLOCK07_summary.json", summary)

    atomic_write_json(
        BLOCK07_ROOT / "BLOCK07_complete.json",
        {
            "status": "PASS",
            "complete": True,
            "completed_utc": now_utc(),
            "configuration": "W500_O50",
            "subjects": N_SUBJECTS,
            "conditions": list(CONDITIONS),
            "models": list(MODELS),
            "SD_runs": int(len(sd_subject)),
            "LOSO_runs": int(len(loso_subject)),
            "latent_models": [
                "EEGNet",
                "CNN-BiLSTM",
                "CNN-BiLSTM-Attention",
            ],
            "input": "raw_EEG_13x500",
            "configuration_hash": summary["configuration_hash"],
        },
    )


def write_readme():
    text = f"""# BLOCK 07 — Deep Learning EEG Decoding

Status: PASS after execution.

## Frozen input
- BLOCK 04: W500_O50
- 13 channels
- 500 samples
- 4.000 s
- 50% overlap
- 250-sample step
- raw EEG input
- 25 subjects
- ME + MI
- gestures 1–7

## 07A Subject-dependent
- Set A: training
- Set B: validation
- Set C: final test
- 25 subjects
- 2 conditions
- 5 models

## 07B LOSO
- 25 folds
- 24 non-held-out subjects per fold
- lowest-numbered training subject used as validation subject
- remaining 23 subjects used for model fitting
- held-out subject remains completely unseen

## Models
- EEGNet
- ShallowConvNet
- DeepConvNet
- CNN-BiLSTM
- CNN-BiLSTM-Attention

## Forbidden
- 341 ML features
- SMOTE
- random window-level test split
- test-derived normalization
- test tuning
- cross-condition training
- calibration
- transfer
- RSA
- XAI
- robustness

## Downstream
BLOCK 08 — ME/MI Statistics
BLOCK 09 — Temporal Calibration
BLOCK 10 — ME/MI Transfer
BLOCK 12 — Representation
BLOCK 13 — XAI
BLOCK 14 — Robustness
BLOCK 15 — Final Statistics
"""
    atomic_write_text(BLOCK07_ROOT / "BLOCK07_README.md", text)


# =============================================================================
# 24. MAIN
# =============================================================================

def main():
    start = time.perf_counter()

    print("=" * 100)
    print("BLOCK 07 — DEEP LEARNING EEG DECODING")
    print("=" * 100)
    print(f"Configuration            : W500_O50")
    print(f"Input                    : raw EEG {N_CHANNELS}x{WINDOW_SAMPLES}")
    print(f"Subjects                 : {N_SUBJECTS}")
    print(f"Conditions               : {' + '.join(CONDITIONS)}")
    print(f"Models                   : {', '.join(MODELS)}")
    print(f"07A SD                   : SetA train / SetB validation / SetC test")
    print(f"07B LOSO                 : 25 folds / 24 train pool / 1 held-out")
    print()

    mount_drive()
    set_all_seeds(RANDOM_SEED)
    configure_determinism()

    # The definitive Paper 2 benchmark is GPU-only. Stop before even running
    # the deterministic smoke test if Colab did not attach CUDA.
    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError(
            "BLOCK 07 STOPPED — CUDA GPU is required for the definitive run. "
            "Enable a Colab GPU under Runtime > Change runtime type > Hardware "
            "accelerator > T4/L4/A100, then rerun this cell. No definitive CPU "
            "training was started."
        )

    # Determinism must be proven on the actual execution environment before
    # any definitive benchmark is started.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    determinism_smoke_test(device)
    print("Determinism smoke test    : PASS")

    # Environment is captured before definitive execution.
    write_environment_lock()

    # Dependencies.
    if not torch.__version__:
        raise RuntimeError("PyTorch import failed.")

    print(f"Device                   : {device}")
    print(f"GPU                      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    # Upstream gates.
    block04_audit = verify_block04_handoff()
    print("BLOCK 04 handoff         : PASS")
    print(
        "Set A/B/C independence   : "
        + ("PASS" if block04_audit["set_independence"]["set_A_B_C_are_separate_recordings"] else "FAIL")
    )

    block06_audit = verify_block06_handoff()
    print("BLOCK 06 LOSO alignment  : PASS")
    print(f"BLOCK 05 ML alignment    : {block06_audit['block05_ml_alignment']}")

    # BLOCK 05 is not a DL input. Do not make BLOCK 07 execution depend on
    # the engineered-feature branch. If its completion marker exists, record
    # it only as an informational upstream status.
    block05_complete = BLOCK05_ROOT / "BLOCK05_complete.json"
    if block05_complete.exists():
        block05_obj = load_json(block05_complete)
        print(
            "BLOCK 05 status           : "
            + ("PASS (not used as DL input)"
               if str(block05_obj.get("status", "")).upper() == "PASS"
               else "PRESENT BUT NOT PASS (not used as DL input)")
        )
    else:
        print("BLOCK 05 status           : NOT REQUIRED FOR DL EXECUTION")

    # Input discovery and structural validation.
    files = discover_window_files()
    inventory = build_window_inventory(files)
    atomic_write_csv(
        CONFIG_ROOT / "channel_mask_audit.csv",
        inventory[[
            "subject", "condition", "gesture", "set",
            "path", "unusable_channels", "usable_channel_mask_all_true"
        ]],
    )

    X, y, meta = load_selected_windows(files)

    if X.shape != (TOTAL_WINDOWS, N_CHANNELS, WINDOW_SAMPLES):
        raise RuntimeError(f"Unexpected master EEG shape: {X.shape}")

    if len(y) != TOTAL_WINDOWS or len(meta) != TOTAL_WINDOWS:
        raise RuntimeError("Master window count mismatch.")

    if set(np.unique(y).tolist()) != set(GESTURE_IDS):
        raise RuntimeError("Master gesture labels are not exactly 1..7.")

    write_split_audit(meta)

    # Freeze and archive configuration before definitive runs.
    architecture_config = build_frozen_architecture_config()
    training_config = build_frozen_training_config()
    config = write_configs(
        block04_audit,
        block06_audit,
        architecture_config,
        training_config,
    )

    # Reproducibility test before the definitive benchmark.
    print()
    print("Running predefined reproducibility test...")
    repro = run_reproducibility_test(X, y, meta, device)
    print(f"Reproducibility test      : {'PASS' if repro else 'REVIEW'}")

    if not repro:
        raise RuntimeError(
            "BLOCK 07 stopped because the predefined reproducibility test failed."
        )

    # Final experiments.
    print()
    print("Starting 07A — subject-dependent DL...")
    sd_subject, sd_class, sd_cohort = run_sd(
        X, y, meta, device
    )

    print()
    print("Starting 07B — cross-subject LOSO DL...")
    loso_subject, loso_class, loso_cohort = run_loso(
        X, y, meta, device
    )

    # Comparison required by Paper 2.
    gap_df = build_gap_table(sd_subject, loso_subject)
    atomic_write_csv(
        COMP_ROOT / "SD_vs_LOSO_gap.csv",
        gap_df,
    )

    model_summary = pd.concat(
        [sd_cohort, loso_cohort],
        ignore_index=True,
    )
    atomic_write_csv(
        COMP_ROOT / "model_summary.csv",
        model_summary,
    )

    atomic_write_csv(
        COMP_ROOT / "ML_vs_DL_input_audit.csv",
        pd.DataFrame([{
            "ML_block": "BLOCK06",
            "ML_input": "341_engineered_features",
            "DL_block": "BLOCK07",
            "DL_input": "13x500_raw_EEG",
            "window_configuration": "W500_O50",
            "subjects": N_SUBJECTS,
            "conditions": "ME,MI",
            "gestures": "1-7",
            "LOSO_folds": N_SUBJECTS,
            "same_held_out_subjects": True,
        }]),
    )

    build_latent_inventory()

    final_audit_result = final_audit(
        sd_subject,
        loso_subject,
        sd_cohort,
        loso_cohort,
    )

    write_summary(
        sd_subject,
        loso_subject,
        sd_cohort,
        loso_cohort,
        final_audit_result,
        repro,
    )
    write_readme()

    runtime = time.perf_counter() - start

    print()
    print("=" * 100)
    print("BLOCK 07 — FINAL STATUS")
    print("=" * 100)
    print(f"Configuration             : W500_O50")
    print(f"Input                     : raw EEG {N_CHANNELS}x{WINDOW_SAMPLES}")
    print(f"Subjects                  : {N_SUBJECTS}")
    print(f"Conditions                : ME + MI")
    print(f"Models                    : {len(MODELS)}")
    print(f"07A SD runs               : {len(sd_subject)}/{N_SUBJECTS * 2 * 5}")
    print(f"07B LOSO runs             : {len(loso_subject)}/{N_SUBJECTS * 2 * 5}")
    print(f"LOSO folds                : {N_SUBJECTS}")
    print(f"LOSO train subjects       : 24/fold")
    print(f"Raw EEG input             : YES")
    print(f"341 ML features           : NO")
    print(f"SMOTE                     : NO")
    print(f"Random window split       : NO")
    print(f"Cross-condition training  : NO")
    print(f"Test-set tuning           : NO")
    print(f"Calibration               : NO — BLOCK 09")
    print(f"Transfer                  : NO — BLOCK 10")
    print(f"RSA                       : NO — BLOCK 12")
    print(f"XAI                       : NO — BLOCK 13")
    print(f"Robustness                : NO — BLOCK 14")
    print(f"Reproducibility           : {'PASS' if repro else 'FAIL'}")
    print(f"Final audit               : PASS")
    print(f"Runtime                   : {runtime / 60.0:.2f} min")
    print(f"Output                    : {BLOCK07_ROOT}")
    print("✓ BLOCK 07 PASS — DEEP LEARNING BENCHMARK FROZEN")

    return 0


if __name__ == "__main__":
    main()
