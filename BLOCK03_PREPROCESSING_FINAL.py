# ============================================================
# PAPER 2 — BLOCK 03
# EEG PREPROCESSING & ARTIFACT-CONTROL PIPELINE
#
# FROZEN FOR EXECUTION
# Depends on : BLOCK 02 — SIGNAL QA
# Feeds       : BLOCK 04 — TEMPORAL WINDOWING
#
# PRIMARY COHORT
#   25 paired subjects
#   ME + MI
#   7 gestures
#   Sets A/B/C
#   1050 logical recordings
#   13 EEG channels
#   Protocol sampling rate = 125 Hz
#   Nominal recording length = 5625 samples / 45 s
#
# FROZEN PRIMARY TRANSFORMATION
#   Raw continuous EEG
#       -> 4th-order Butterworth 1–40 Hz
#       -> zero-phase offline filtering
#       -> continuous preprocessed EEG
#
# IMPORTANT — THIS BLOCK DOES NOT:
#   - resample
#   - epoch
#   - reconstruct trials/repetitions
#   - normalize globally
#   - perform ICA
#   - perform CAR / any new re-reference
#   - interpolate bad channels
#   - automatically reject noisy segments
#   - select features
#   - reduce dimensions
#   - balance classes / SMOTE
#   - fit ML/DL models
#   - tune preprocessing from results
#
# BAD-CHANNEL POLICY
#   Block 02 channel-specific diagnostic flags are carried forward.
#   No channel is physically deleted from the saved 13-channel array.
#   No spatial interpolation is performed because verified electrode
#   geometry/montage is not part of the frozen protocol.
#   A per-recording usable_channel_mask is saved for BLOCK 04/05.
#
# IMPORTANT:
#   BLOCK 04 MUST read usable_channel_mask and exclude channels
#   explicitly flagged by BLOCK 02 from downstream calculations.
#
# RAW DATA:
#   NEVER modified, overwritten, renamed, or deleted.
# ============================================================

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import signal
import matplotlib.pyplot as plt


# ============================================================
# 00. PATHS
# ============================================================

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception:
    # Allows local execution when the Drive is already mounted or
    # when the script is being inspected outside Colab.
    pass

RAW_ROOT = Path("/content/drive/MyDrive/EEG_Motor_Execution_Imagery")
PAPER2_ROOT = RAW_ROOT / "Paper2_EEG_BCI"

BLOCK01_ROOT = PAPER2_ROOT / "01_SUBJECT_MATCHING"
BLOCK02_ROOT = PAPER2_ROOT / "02_SIGNAL_QA"
BLOCK03_ROOT = PAPER2_ROOT / "03_PREPROCESSING"

RAW_ME_ROOT = RAW_ROOT / "Motor_Execution_Raw_Data"
RAW_MI_ROOT = RAW_ROOT / "Motor_Imagery_Raw_Data"

PROCESSED_ROOT = BLOCK03_ROOT / "processed"
PROCESSED_ME_ROOT = PROCESSED_ROOT / "ME"
PROCESSED_MI_ROOT = PROCESSED_ROOT / "MI"

MANIFEST_ROOT = BLOCK03_ROOT / "manifests"
DIAGNOSTICS_ROOT = BLOCK03_ROOT / "diagnostics"
FILTER_VALIDATION_ROOT = DIAGNOSTICS_ROOT / "filter_validation"
RAW_FILTERED_ROOT = DIAGNOSTICS_ROOT / "raw_vs_filtered"
PSD_ROOT = DIAGNOSTICS_ROOT / "PSD"

# Block 03 owns only its own output directory.
# RAW_* and BLOCK01/BLOCK02 are never cleaned.
#
# IMPORTANT RESUME RULE:
# If a BLOCK03 checkpoint already exists, NEVER delete BLOCK03_ROOT.
# This preserves completed outputs so the block can resume.
BLOCK03_CHECKPOINT_PREEXISTS = BLOCK03_ROOT / "BLOCK03_checkpoint.json"

if BLOCK03_ROOT.exists() and not BLOCK03_CHECKPOINT_PREEXISTS.exists():
    shutil.rmtree(BLOCK03_ROOT)

for d in [
    PROCESSED_ME_ROOT,
    PROCESSED_MI_ROOT,
    MANIFEST_ROOT,
    FILTER_VALIDATION_ROOT,
    RAW_FILTERED_ROOT,
    PSD_ROOT,
]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 00A. RESUME / CHECKPOINT LOGIC
# ============================================================
# IMPORTANT:
#   BLOCK 03 can be safely interrupted and resumed.
#
#   Resume behavior:
#     - Existing successfully verified processed .npz files are reused.
#     - Existing manifest rows are reused only when the corresponding
#       output file still exists and passes structural/numerical checks.
#     - Failed/incomplete recordings are retried.
#     - No completed recording is re-filtered unnecessarily.
#     - The raw dataset is NEVER modified.
#
#   The checkpoint is deliberately stored inside BLOCK03_ROOT.
#   If BLOCK03_ROOT is manually deleted, the block starts clean.
# ============================================================

CHECKPOINT_PATH = BLOCK03_ROOT / "BLOCK03_checkpoint.json"
PARTIAL_MANIFEST_PATH = MANIFEST_ROOT / "preprocessing_manifest_partial.csv"
PARTIAL_FAILURE_PATH = MANIFEST_ROOT / "preprocessing_failures_partial.csv"

def verify_existing_processed_output(path: Path):
    """
    Verify that an existing processed artifact is safe to reuse.
    Returns (True, metadata) only when all essential invariants pass.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False, None

    try:
        with np.load(path, allow_pickle=False) as z:
            required = [
                "eeg",
                "usable_channel_mask",
                "channel_names",
                "sampling_rate_hz",
                "original_sample_count",
                "original_duration_s",
                "filter_order",
                "low_cut_hz",
                "high_cut_hz",
                "zero_phase",
                "source_raw_relative_path",
            ]
            if any(k not in z.files for k in required):
                return False, None

            X = z["eeg"]
            mask = z["usable_channel_mask"]

            if X.shape != (EXPECTED_SAMPLES, EXPECTED_CHANNELS):
                return False, None
            if mask.shape != (EXPECTED_CHANNELS,):
                return False, None
            if not np.isfinite(X).all():
                return False, None

            fs_saved = int(np.asarray(z["sampling_rate_hz"]).item())
            n_saved = int(np.asarray(z["original_sample_count"]).item())
            order_saved = int(np.asarray(z["filter_order"]).item())
            low_saved = float(np.asarray(z["low_cut_hz"]).item())
            high_saved = float(np.asarray(z["high_cut_hz"]).item())
            zp_saved = bool(np.asarray(z["zero_phase"]).item())

            if fs_saved != FS:
                return False, None
            if n_saved != EXPECTED_SAMPLES:
                return False, None
            if order_saved != FILTER_ORDER:
                return False, None
            if not np.isclose(low_saved, LOW_CUTOFF_HZ):
                return False, None
            if not np.isclose(high_saved, HIGH_CUTOFF_HZ):
                return False, None
            if not zp_saved:
                return False, None

            metadata = {
                "shape": list(X.shape),
                "sha256": sha256_file(path),
                "sampling_rate_hz": fs_saved,
                "filter_order": order_saved,
                "low_cut_hz": low_saved,
                "high_cut_hz": high_saved,
                "zero_phase": zp_saved,
            }
            return True, metadata
    except Exception:
        return False, None


def load_resume_state():
    """
    Load checkpoint and partial manifests if present.
    Never trusts a checkpoint blindly: every reused output is reverified.
    """
    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        try:
            checkpoint = json.loads(
                CHECKPOINT_PATH.read_text(encoding="utf-8")
            )
        except Exception:
            checkpoint = {}

    previous_manifest = pd.DataFrame()
    if PARTIAL_MANIFEST_PATH.exists():
        try:
            previous_manifest = pd.read_csv(PARTIAL_MANIFEST_PATH)
        except Exception:
            previous_manifest = pd.DataFrame()

    previous_failures = pd.DataFrame()
    if PARTIAL_FAILURE_PATH.exists():
        try:
            previous_failures = pd.read_csv(PARTIAL_FAILURE_PATH)
        except Exception:
            previous_failures = pd.DataFrame()

    return checkpoint, previous_manifest, previous_failures


def save_resume_state(manifest_rows, failure_rows, processed_keys, attempted_count):
    """
    Atomically update partial manifest/checkpoint after progress.
    """
    manifest_tmp = PARTIAL_MANIFEST_PATH.with_suffix(".tmp.csv")
    failure_tmp = PARTIAL_FAILURE_PATH.with_suffix(".tmp.csv")
    checkpoint_tmp = CHECKPOINT_PATH.with_suffix(".tmp.json")

    pd.DataFrame(manifest_rows).to_csv(manifest_tmp, index=False)
    pd.DataFrame(failure_rows).to_csv(failure_tmp, index=False)

    checkpoint = {
        "block": "03_PREPROCESSING",
        "resume_enabled": True,
        "last_checkpoint_utc": utc_now(),
        "attempted_count": int(attempted_count),
        "verified_processed_count": int(len(processed_keys)),
        "processed_keys": sorted(processed_keys),
        "failed_keys": sorted({
            logical_key(
                r["canonical_subject"],
                r["condition"],
                r["gesture_code"],
                r["set"],
            )
            for r in failure_rows
        }),
        "config_signature": {
            "sampling_rate_hz": FS,
            "channels": EXPECTED_CHANNELS,
            "samples": EXPECTED_SAMPLES,
            "filter_type": FILTER_TYPE,
            "filter_order": FILTER_ORDER,
            "low_cut_hz": LOW_CUTOFF_HZ,
            "high_cut_hz": HIGH_CUTOFF_HZ,
            "zero_phase": ZERO_PHASE,
            "car": CAR,
            "ica": ICA,
            "resampling": RESAMPLING,
            "normalization": NORMALIZATION,
            "bad_channel_interpolation": BAD_CHANNEL_INTERPOLATION,
        },
    }

    checkpoint_tmp.write_text(
        json.dumps(checkpoint, indent=2, default=str),
        encoding="utf-8",
    )

    manifest_tmp.replace(PARTIAL_MANIFEST_PATH)
    failure_tmp.replace(PARTIAL_FAILURE_PATH)
    checkpoint_tmp.replace(CHECKPOINT_PATH)


def manifest_row_key(row):
    return logical_key(
        row["canonical_subject"],
        row["condition"],
        row["gesture_code"],
        row["set"],
    )


def build_existing_resume_rows(previous_manifest):
    """
    Return only rows whose output artifacts still exist and pass verification.
    """
    reusable = {}
    if previous_manifest.empty:
        return reusable

    for _, row in previous_manifest.iterrows():
        try:
            key = manifest_row_key(row)
            out_path = BLOCK03_ROOT / str(row["output_path"])
            ok, meta = verify_existing_processed_output(out_path)
            if not ok:
                continue

            reusable[key] = row.to_dict()
        except Exception:
            continue

    return reusable


def build_previous_failures(previous_failures):
    """
    Failed recordings are deliberately NOT treated as completed.
    They are retried on resume.
    """
    rows = []
    if previous_failures.empty:
        return rows

    for _, row in previous_failures.iterrows():
        rows.append(row.to_dict())
    return rows


def build_channel_audit_from_manifest(manifest_rows):
    """
    Rebuild the complete 13-channel audit from the final verified manifest.

    This is intentionally reconstructed from the saved processed artifacts
    rather than from only recordings processed during the current runtime.
    Therefore RESUME runs still produce the complete 1050 × 13 audit.
    """
    rows = []

    for row in manifest_rows:
        out_path = BLOCK03_ROOT / str(row["output_path"])

        with np.load(out_path, allow_pickle=False) as z:
            names = [str(x) for x in z["channel_names"]]
            mask = z["usable_channel_mask"].astype(bool)

        if len(names) != EXPECTED_CHANNELS:
            raise RuntimeError(
                f"Channel audit: expected {EXPECTED_CHANNELS} channel names "
                f"for {row['condition']} {row['canonical_subject']} "
                f"{row['gesture']} Set{row['set']}; got {len(names)}."
            )

        if mask.shape != (EXPECTED_CHANNELS,):
            raise RuntimeError(
                f"Channel audit: invalid usable_channel_mask shape "
                f"for {row['condition']} {row['canonical_subject']} "
                f"{row['gesture']} Set{row['set']}: {mask.shape}"
            )

        for ch_name, usable in zip(names, mask):
            rows.append(
                {
                    "canonical_subject": row["canonical_subject"],
                    "condition": row["condition"],
                    "gesture": row["gesture"],
                    "gesture_code": row["gesture_code"],
                    "set": row["set"],
                    "channel": ch_name,
                    "block02_flagged": not bool(usable),
                    "downstream_usable": bool(usable),
                    "flag_source": (
                        "BLOCK02_flat_channel_qa"
                        if not bool(usable)
                        else ""
                    ),
                }
            )

    return rows


# ============================================================
# 01. FROZEN PROTOCOL
# ============================================================

EXPECTED_SUBJECTS = [f"S{i}" for i in range(1, 26)]
EXPECTED_CONDITIONS = ["ME", "MI"]
EXPECTED_GESTURES = {
    1: "HOC",
    2: "IFM",
    3: "LFM",
    4: "MFM",
    5: "PH",
    6: "RFM",
    7: "TFM",
}
EXPECTED_SETS = ["A", "B", "C"]

EXPECTED_RECORDINGS_PER_CONDITION = 25 * 7 * 3
EXPECTED_TOTAL_RECORDINGS = 25 * 7 * 3 * 2

FS = 125
EXPECTED_CHANNELS = 13
EXPECTED_SAMPLES = 5625
NOMINAL_DURATION_S = 45.0

FILTER_TYPE = "Butterworth"
FILTER_ORDER = 4
LOW_CUTOFF_HZ = 1.0
HIGH_CUTOFF_HZ = 40.0
ZERO_PHASE = True

# Explicitly frozen OFF:
RESAMPLING = False
NORMALIZATION = False
ICA = False
AUTOMATIC_ARTIFACT_REJECTION = False
FEATURE_SELECTION = False
DIMENSIONALITY_REDUCTION = False
CLASS_BALANCING = False
SMOTE = False
CAR = False
BAD_CHANNEL_INTERPOLATION = False
TRIAL_RECONSTRUCTION = False
EPOCHING = False

# The acquisition timestamp/sample-index fields are metadata only.
TIMESTAMP_USED_AS_CLOCK = False
SAMPLE_INDEX_USED_AS_CLOCK = False

GESTURE_DIR_TO_CODE = {
    "Hand_Open_Close": 1,
    "Index_Finger_Movement": 2,
    "Little_Finger_Movement": 3,
    "Middle_Finger_Movement": 4,
    "Pen_Holding": 5,
    "Ring_Finger_Movement": 6,
    "Thumb_Finger_Movement": 7,
}


# ============================================================
# 02. UTILITIES
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize_subject(value):
    s = str(value).strip().upper()
    s = s.replace("SUBJECT_", "S").replace("SUBJECT", "S")
    m = re.search(r"S?(\d+)", s)
    if not m:
        raise ValueError(f"Cannot normalize subject ID: {value!r}")
    return f"S{int(m.group(1))}"


def normalize_condition(value):
    s = str(value).strip().upper()
    if s in {"MOTOR_EXECUTION", "EXECUTION", "ME"}:
        return "ME"
    if s in {"MOTOR_IMAGERY", "IMAGERY", "MI"}:
        return "MI"
    return s


def normalize_set(value):
    s = str(value).strip().upper()
    m = re.search(r"(?:SET[_ -]?)?([ABC])$", s)
    if m:
        return m.group(1)
    m = re.search(r"\bSET[_ -]?([ABC])\b", s)
    if m:
        return m.group(1)
    return s


def first_existing(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not identify {label}; tried {candidates}")


def logical_key(subject, condition, gesture_code, set_code):
    return (
        normalize_subject(subject),
        normalize_condition(condition),
        int(gesture_code),
        normalize_set(set_code),
    )


def logical_key_text(key):
    s, c, g, st = key
    return f"{s}_{c}_{EXPECTED_GESTURES[g]}_Set{st}"


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def split_channel_text(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "[]"}:
        return []
    # Support semicolon, comma, pipe, and bracketed representations.
    s = s.strip("[](){}")
    parts = re.split(r"[;,|]", s)
    return [p.strip().strip("'\"") for p in parts if p.strip()]


# ============================================================
# 03. BLOCK 01 VERIFICATION
# ============================================================

print("\n" + "=" * 96)
print("BLOCK 03 — PREPROCESSING & ARTIFACT CONTROL")
print("=" * 96)

inventory_path = BLOCK01_ROOT / "recording_inventory.csv"
if not inventory_path.exists():
    raise FileNotFoundError(
        f"Required BLOCK 01 inventory not found:\n{inventory_path}"
    )

inventory = pd.read_csv(inventory_path)

if len(inventory) != EXPECTED_TOTAL_RECORDINGS:
    raise RuntimeError(
        f"BLOCK 01 inventory has {len(inventory)} rows; "
        f"expected {EXPECTED_TOTAL_RECORDINGS}."
    )

inv_subject_col = first_existing(
    inventory, ["subject_id", "subject", "canonical_subject"], "subject"
)
inv_condition_col = first_existing(
    inventory, ["condition"], "condition"
)
inv_gesture_col = first_existing(
    inventory, ["gesture_code", "gesture"], "gesture code"
)
inv_set_col = first_existing(
    inventory, ["set_code", "set"], "set"
)
inv_path_col = first_existing(
    inventory,
    ["relative_path", "relative_file_path", "path", "source_path"],
    "relative raw path",
)

inventory["_subject"] = inventory[inv_subject_col].map(normalize_subject)
inventory["_condition"] = inventory[inv_condition_col].map(normalize_condition)
inventory["_gesture"] = pd.to_numeric(
    inventory[inv_gesture_col], errors="coerce"
)
inventory["_set"] = inventory[inv_set_col].map(normalize_set)

inv_keys = [
    logical_key(r["_subject"], r["_condition"], r["_gesture"], r["_set"])
    for _, r in inventory.iterrows()
]

if len(set(inv_keys)) != EXPECTED_TOTAL_RECORDINGS:
    raise RuntimeError("BLOCK 01 contains duplicate logical recording keys.")

if set(inventory["_subject"]) != set(EXPECTED_SUBJECTS):
    raise RuntimeError("BLOCK 01 subject set does not equal S1–S25.")

if set(inventory["_condition"]) != set(EXPECTED_CONDITIONS):
    raise RuntimeError("BLOCK 01 condition set is not exactly ME + MI.")

print(f"BLOCK 01 inventory: PASS ({len(inventory)} logical recordings)")


# ============================================================
# 04. BLOCK 02 VERIFICATION
# ============================================================

b02_complete = BLOCK02_ROOT / "BLOCK02_complete.json"
if not b02_complete.exists():
    raise FileNotFoundError(
        f"Required BLOCK 02 completion marker not found:\n{b02_complete}"
    )

with b02_complete.open("r", encoding="utf-8") as f:
    b02_meta = json.load(f)

b02_status_path = BLOCK02_ROOT / "recording_final_qa_status.csv"
if not b02_status_path.exists():
    raise FileNotFoundError(
        f"Required BLOCK 02 QA table not found:\n{b02_status_path}"
    )

qa = pd.read_csv(b02_status_path)

if len(qa) != EXPECTED_TOTAL_RECORDINGS:
    raise RuntimeError(
        f"BLOCK 02 QA table has {len(qa)} rows; "
        f"expected {EXPECTED_TOTAL_RECORDINGS}."
    )

qa_subject_col = first_existing(
    qa, ["subject_id", "subject", "canonical_subject"], "QA subject"
)
qa_condition_col = first_existing(
    qa, ["condition"], "QA condition"
)
qa_gesture_col = first_existing(
    qa, ["gesture_code", "gesture"], "QA gesture"
)
qa_set_col = first_existing(
    qa, ["set_code", "set"], "QA set"
)
qa_status_col = first_existing(
    qa, ["final_qa_status", "structural_status"], "QA final status"
)

qa["_subject"] = qa[qa_subject_col].map(normalize_subject)
qa["_condition"] = qa[qa_condition_col].map(normalize_condition)
qa["_gesture"] = pd.to_numeric(qa[qa_gesture_col], errors="coerce")
qa["_set"] = qa[qa_set_col].map(normalize_set)
qa["_qa_status"] = qa[qa_status_col].astype(str).str.upper()

qa_keys = [
    logical_key(r["_subject"], r["_condition"], r["_gesture"], r["_set"])
    for _, r in qa.iterrows()
]

if len(set(qa_keys)) != EXPECTED_TOTAL_RECORDINGS:
    raise RuntimeError("BLOCK 02 contains duplicate logical recording keys.")

qa_lookup = {k: row for k, row in zip(qa_keys, qa.to_dict("records"))}

print(f"BLOCK 02 QA table: PASS ({len(qa)} logical recordings)")
print("BLOCK 02 QA final status distribution:")
print(qa["_qa_status"].value_counts(dropna=False).to_string())

# The completed Block 02 in this project reports structural PASS for
# all 1050 recordings and uses REVIEW as a diagnostic decision.
# We do not silently exclude REVIEW recordings here.
structural_failures = qa[
    qa["_qa_status"].isin(["FAIL", "FAILED", "EXCLUDE"])
]
if len(structural_failures):
    raise RuntimeError(
        f"BLOCK 02 contains {len(structural_failures)} structural FAIL/EXCLUDE "
        "recordings. Resolve/document them before BLOCK 03."
    )


# ============================================================
# 05. DISCOVER RAW SCIENTIFIC CSVs
# ============================================================
# BLOCK 01 is the authoritative logical recording inventory.
#
# Therefore BLOCK 03 does NOT assume that every *.csv found under the
# raw folders is a scientific recording. Extra CSVs may be audit,
# comparison, diagnostic, duplicate, or other auxiliary files.
#
# We select the 1050 raw recordings ONLY from the paths recorded in
# BLOCK 01, then separately report any additional CSVs found on disk.
# This prevents auxiliary CSVs from being mistaken for recordings.
# ============================================================

def resolve_inventory_raw_path(raw_value, condition):
    raw_value = str(raw_value).strip()
    candidates = []

    p0 = Path(raw_value)
    if p0.is_absolute():
        candidates.append(p0)

    # Most common: path relative to RAW_ROOT.
    candidates.append(RAW_ROOT / raw_value)

    # Also tolerate a path relative to the condition raw folder.
    raw_condition_root = RAW_ME_ROOT if condition == "ME" else RAW_MI_ROOT
    candidates.append(raw_condition_root / raw_value)

    # If inventory stores only a filename or partial relative path,
    # search inside the appropriate raw condition directory.
    basename = Path(raw_value).name
    if basename:
        candidates.extend(raw_condition_root.rglob(basename))

    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


inventory_raw_paths = []
inventory_missing_paths = []

for _, r in inventory.iterrows():
    condition = r["_condition"]
    resolved = resolve_inventory_raw_path(r[inv_path_col], condition)
    if resolved is None:
        inventory_missing_paths.append(
            {
                "subject": r["_subject"],
                "condition": condition,
                "gesture_code": int(r["_gesture"]),
                "set": r["_set"],
                "inventory_path": str(r[inv_path_col]),
            }
        )
    else:
        inventory_raw_paths.append(
            {
                "condition": condition,
                "path": resolved,
                "inventory_path": str(r[inv_path_col]),
            }
        )

if inventory_missing_paths:
    raise RuntimeError(
        "BLOCK 01 references raw files that cannot be found. "
        f"Missing={len(inventory_missing_paths)}. "
        f"First={inventory_missing_paths[0]}"
    )

if len(inventory_raw_paths) != EXPECTED_TOTAL_RECORDINGS:
    raise RuntimeError(
        "BLOCK 01 raw-path resolution did not produce exactly 1050 files: "
        f"{len(inventory_raw_paths)}"
    )

me_files = sorted(
    [x["path"] for x in inventory_raw_paths if x["condition"] == "ME"],
    key=lambda p: str(p).lower(),
)
mi_files = sorted(
    [x["path"] for x in inventory_raw_paths if x["condition"] == "MI"],
    key=lambda p: str(p).lower(),
)

# All CSVs on disk are inspected only for discovery/audit purposes.
# They are NOT automatically treated as recordings.
all_me_csvs = sorted(RAW_ME_ROOT.rglob("*.csv"))
all_mi_csvs = sorted(RAW_MI_ROOT.rglob("*.csv"))

selected_me_set = {p.resolve() for p in me_files}
selected_mi_set = {p.resolve() for p in mi_files}

extra_me_csvs = [p for p in all_me_csvs if p.resolve() not in selected_me_set]
extra_mi_csvs = [p for p in all_mi_csvs if p.resolve() not in selected_mi_set]

print("\nRAW DATA DISCOVERY")
print(f"ME CSV files selected by BLOCK 01: {len(me_files)}")
print(f"MI CSV files selected by BLOCK 01: {len(mi_files)}")
print(f"ME additional CSVs on disk (not in BLOCK 01): {len(extra_me_csvs)}")
print(f"MI additional CSVs on disk (not in BLOCK 01): {len(extra_mi_csvs)}")

if extra_me_csvs:
    print("ME additional CSV examples:")
    for p in extra_me_csvs[:10]:
        print("  -", p)

if extra_mi_csvs:
    print("MI additional CSV examples:")
    for p in extra_mi_csvs[:10]:
        print("  -", p)

if len(me_files) != EXPECTED_RECORDINGS_PER_CONDITION:
    raise RuntimeError(
        f"BLOCK 01-selected ME recordings = {len(me_files)}, "
        f"expected {EXPECTED_RECORDINGS_PER_CONDITION}."
    )

if len(mi_files) != EXPECTED_RECORDINGS_PER_CONDITION:
    raise RuntimeError(
        f"BLOCK 01-selected MI recordings = {len(mi_files)}, "
        f"expected {EXPECTED_RECORDINGS_PER_CONDITION}."
    )

# Preserve an explicit discovery audit.
pd.DataFrame(
    [
        {
            "condition": "ME",
            "selected_by_block01": True,
            "path": str(p.relative_to(RAW_ROOT)),
        }
        for p in me_files
    ]
    + [
        {
            "condition": "MI",
            "selected_by_block01": True,
            "path": str(p.relative_to(RAW_ROOT)),
        }
        for p in mi_files
    ]
    + [
        {
            "condition": "ME",
            "selected_by_block01": False,
            "path": str(p.relative_to(RAW_ROOT)),
        }
        for p in extra_me_csvs
    ]
    + [
        {
            "condition": "MI",
            "selected_by_block01": False,
            "path": str(p.relative_to(RAW_ROOT)),
        }
        for p in extra_mi_csvs
    ]
).to_csv(
    MANIFEST_ROOT / "raw_csv_discovery_audit.csv",
    index=False,
)


# ============================================================
# 06. RAW METADATA PARSER
# ============================================================

def parse_raw_metadata(path: Path, condition: str):
    # Current structure:
    #   Motor_*_Raw_Data / Subject_XX / Gesture_Name / file.csv
    subject_folder = None
    for parent in path.parents:
        if re.match(r"^Subject[_ -]?\d+$", parent.name, flags=re.I):
            subject_folder = parent
            break

    if subject_folder is None:
        # Fallback: search any path component containing Subject + digits.
        for part in path.parts:
            if re.match(r"^Subject[_ -]?\d+$", part, flags=re.I):
                subject_folder = Path(part)
                break

    if subject_folder is None:
        raise ValueError(f"Cannot parse subject folder from {path}")

    subject = normalize_subject(subject_folder.name)

    gesture_folder = path.parent.name
    if gesture_folder not in GESTURE_DIR_TO_CODE:
        # More tolerant matching.
        normalized = re.sub(r"[^a-z]", "", gesture_folder.lower())
        aliases = {
            "handopenclose": 1,
            "indexfingermovement": 2,
            "littlefingermovement": 3,
            "middlefingermovement": 4,
            "penholding": 5,
            "ringfingermovement": 6,
            "thumbfingermovement": 7,
        }
        if normalized not in aliases:
            raise ValueError(
                f"Unknown gesture folder '{gesture_folder}' in {path}"
            )
        gesture_code = aliases[normalized]
    else:
        gesture_code = GESTURE_DIR_TO_CODE[gesture_folder]

    stem_upper = path.stem.upper()
    set_match = re.search(r"(?:^|[_\-\s])SET[_\-]?([ABC])(?:$|[_\-\s\(\)])", stem_upper)
    if not set_match:
        set_match = re.search(r"SET[_\-]?([ABC])", stem_upper)

    if not set_match:
        raise ValueError(f"Cannot parse Set A/B/C from filename: {path.name}")

    set_code = set_match.group(1)

    return subject, normalize_condition(condition), gesture_code, set_code


# ============================================================
# 07. BUILD RAW KEY LOOKUP AND COMPARE WITH BLOCK 01
# ============================================================

raw_records = []
raw_key_seen = {}

for condition, files in [("ME", me_files), ("MI", mi_files)]:
    for path in files:
        subject, cond, gesture_code, set_code = parse_raw_metadata(
            path, condition
        )
        key = logical_key(subject, cond, gesture_code, set_code)

        if key in raw_key_seen:
            raise RuntimeError(
                f"Duplicate raw logical key:\n{logical_key_text(key)}\n"
                f"{raw_key_seen[key]}\n{path}"
            )

        raw_key_seen[key] = str(path)
        raw_records.append(
            {
                "key": key,
                "subject": subject,
                "condition": cond,
                "gesture_code": gesture_code,
                "gesture": EXPECTED_GESTURES[gesture_code],
                "set": set_code,
                "raw_path": path,
            }
        )

expected_keys = set(inv_keys)
raw_keys = set(raw_key_seen)

missing_raw = sorted(expected_keys - raw_keys)
unexpected_raw = sorted(raw_keys - expected_keys)

if missing_raw or unexpected_raw:
    raise RuntimeError(
        "Raw logical identity does not match BLOCK 01.\n"
        f"Missing: {len(missing_raw)}\n"
        f"Unexpected: {len(unexpected_raw)}"
    )

print(f"Raw logical identity: PASS ({len(raw_records)} recordings)")


# ============================================================
# 08. LOAD BLOCK 02 CHANNEL-SPECIFIC FLAGS
# ============================================================

# We deliberately use only channel identities explicitly provided
# by BLOCK 02. We do NOT infer a channel from a recording-level
# near-flat/review/extreme/discontinuity flag.
#
# No channel interpolation and no permanent deletion.
flat_channel_path = BLOCK02_ROOT / "flat_channel_qa.csv"

flat_lookup = {}

if flat_channel_path.exists():
    flat_df = pd.read_csv(flat_channel_path)

    fsub = first_existing(
        flat_df, ["subject_id", "subject", "canonical_subject"], "flat QA subject"
    )
    fcond = first_existing(
        flat_df, ["condition"], "flat QA condition"
    )
    fgesture = first_existing(
        flat_df, ["gesture_code", "gesture"], "flat QA gesture"
    )
    fset = first_existing(
        flat_df, ["set_code", "set"], "flat QA set"
    )
    fchannels = first_existing(
        flat_df,
        ["flat_channels", "bad_channels", "flagged_channels"],
        "flat channel list",
    )

    for _, r in flat_df.iterrows():
        try:
            key = logical_key(
                r[fsub], r[fcond], r[fgesture], r[fset]
            )
        except Exception:
            continue

        channels = split_channel_text(r[fchannels])
        if channels:
            flat_lookup.setdefault(key, set()).update(channels)

print(
    f"Explicit Block 02 flat-channel lookup: "
    f"{sum(bool(v) for v in flat_lookup.values())} recordings"
)


# ============================================================
# 09. EEG COLUMN DETECTION
# ============================================================

def detect_eeg_columns(df):
    """
    Detect the 13 EEG columns without using Timestamp or Sample Index
    as signal channels.

    Priority:
      1) names containing EEG / EEG_ch
      2) exact EEG_ch-01 ... EEG_ch-13 variants
      3) conservative fallback to numeric columns excluding known
         metadata columns.
    """

    cols = list(df.columns)

    def clean(c):
        return re.sub(r"[^A-Z0-9]+", "", str(c).upper())

    # Explicit EEG names.
    explicit = []
    for c in cols:
        cc = clean(c)
        if cc.startswith("EEGCH") or cc.startswith("EEGCHANNEL"):
            explicit.append(c)

    if len(explicit) == EXPECTED_CHANNELS:
        return explicit

    # Generic EEG substring, excluding metadata-like columns.
    eeg_like = []
    for c in cols:
        cc = clean(c)
        if "EEG" in cc and "TIMESTAMP" not in cc:
            eeg_like.append(c)

    if len(eeg_like) == EXPECTED_CHANNELS:
        return eeg_like

    # Exact known channel numbering variants.
    numbered = []
    for n in range(1, EXPECTED_CHANNELS + 1):
        candidates = {
            f"EEG_ch-{n}",
            f"EEG_ch-{n:02d}",
            f"EEG_ch_{n}",
            f"EEG_ch_{n:02d}",
            f"EEG{n}",
            f"EEG{n:02d}",
        }
        hit = next((c for c in cols if str(c) in candidates), None)
        if hit is not None:
            numbered.append(hit)

    if len(numbered) == EXPECTED_CHANNELS:
        return numbered

    # Conservative numeric fallback.
    metadata_names = {
        "sample index",
        "sample_index",
        "timestamp",
        "timestamp formatted",
        "timestamp_formatted",
        "time",
        "time(s)",
        "time_s",
    }

    numeric_cols = []
    for c in cols:
        if str(c).strip().lower() in metadata_names:
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().sum() >= max(10, int(0.95 * len(df))):
            numeric_cols.append(c)

    if len(numeric_cols) == EXPECTED_CHANNELS:
        return numeric_cols

    raise ValueError(
        f"Could not identify exactly {EXPECTED_CHANNELS} EEG columns. "
        f"Columns={cols}"
    )


# ============================================================
# 10. FILTER IMPLEMENTATION
# ============================================================

def design_filter():
    nyquist = FS / 2.0
    if HIGH_CUTOFF_HZ >= nyquist:
        raise ValueError("High cutoff must be below Nyquist frequency.")

    sos = signal.butter(
        FILTER_ORDER,
        [LOW_CUTOFF_HZ / nyquist, HIGH_CUTOFF_HZ / nyquist],
        btype="bandpass",
        output="sos",
    )
    return sos


SOS = design_filter()


def butter_bandpass_zero_phase(X):
    """
    X: samples × channels

    scipy.signal.sosfiltfilt performs deterministic forward-backward
    zero-phase filtering using the library's documented padding/
    boundary mechanism.
    """
    return signal.sosfiltfilt(
        SOS,
        X,
        axis=0,
        padtype="odd",
    )


# ============================================================
# 11. PROCESSING FUNCTION
# ============================================================

def process_one(record):
    key = record["key"]
    raw_path = record["raw_path"]

    qa_row = qa_lookup.get(key)
    if qa_row is None:
        raise RuntimeError(
            f"No BLOCK 02 QA record for {logical_key_text(key)}"
        )

    qa_status = str(qa_row.get("_qa_status", "UNKNOWN")).upper()

    df = pd.read_csv(raw_path)

    eeg_columns = detect_eeg_columns(df)

    X = df[eeg_columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float64
    )

    input_samples, input_channels = X.shape

    if input_channels != EXPECTED_CHANNELS:
        raise ValueError(
            f"Expected {EXPECTED_CHANNELS} EEG channels; got {input_channels}."
        )

    if input_samples != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} samples; got {input_samples}."
        )

    # Block 02 reported zero nonfinite recordings in the frozen dataset.
    # Therefore no repair is expected. We do NOT interpolate or replace
    # nonfinite values in this block; any occurrence is a processing FAIL.
    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN/Inf; no repair is permitted in BLOCK 03.")

    # Explicitly no CAR and no channel interpolation.
    filtered = butter_bandpass_zero_phase(X)

    if filtered.shape != X.shape:
        raise ValueError(
            f"Filtering changed shape from {X.shape} to {filtered.shape}."
        )

    if not np.isfinite(filtered).all():
        raise ValueError("Filtering introduced NaN/Inf.")

    # Explicitly identified unusable channels from Block 02.
    flagged_channels = sorted(flat_lookup.get(key, set()))

    # Match channel labels exactly where possible.
    flagged_indices = []
    normalized_columns = {
        re.sub(r"[^a-z0-9]", "", str(c).lower()): i
        for i, c in enumerate(eeg_columns)
    }

    unresolved_flags = []
    for ch in flagged_channels:
        norm = re.sub(r"[^a-z0-9]", "", str(ch).lower())
        if norm in normalized_columns:
            flagged_indices.append(normalized_columns[norm])
        else:
            # Also tolerate EEG_ch-11 vs EEG_ch-11.0 style strings.
            candidates = [
                i for i, c in enumerate(eeg_columns)
                if str(c).strip().lower() == str(ch).strip().lower()
            ]
            if candidates:
                flagged_indices.extend(candidates)
            else:
                unresolved_flags.append(ch)

    flagged_indices = sorted(set(flagged_indices))

    usable_mask = np.ones(EXPECTED_CHANNELS, dtype=bool)
    usable_mask[flagged_indices] = False

    if not usable_mask.any():
        raise ValueError(
            "All 13 channels are flagged unusable; no valid channel remains."
        )

    # Save ALL 13 channels. The mask, not deletion, controls downstream use.
    condition_dir = PROCESSED_ME_ROOT if record["condition"] == "ME" else PROCESSED_MI_ROOT
    out_name = (
        f"{record['subject']}_{record['condition']}_"
        f"{record['gesture']}_Set{record['set']}_preprocessed.npz"
    )
    output_path = condition_dir / out_name

    relative_raw = str(raw_path.relative_to(RAW_ROOT))
    relative_output = str(output_path.relative_to(BLOCK03_ROOT))

    np.savez_compressed(
        output_path,
        eeg=filtered.astype(np.float32),
        subject=np.array(record["subject"]),
        condition=np.array(record["condition"]),
        gesture_code=np.array(record["gesture_code"], dtype=np.int16),
        gesture=np.array(record["gesture"]),
        set_code=np.array(record["set"]),
        sampling_rate_hz=np.array(FS, dtype=np.int16),
        original_sample_count=np.array(input_samples, dtype=np.int32),
        original_duration_s=np.array(input_samples / FS, dtype=np.float32),
        channel_names=np.array(eeg_columns),
        usable_channel_mask=usable_mask,
        usable_channels=np.array(
            [c for c, ok in zip(eeg_columns, usable_mask) if ok]
        ),
        excluded_from_downstream_channels=np.array(
            [c for c, ok in zip(eeg_columns, usable_mask) if not ok]
        ),
        filter_type=np.array(FILTER_TYPE),
        filter_order=np.array(FILTER_ORDER, dtype=np.int16),
        low_cut_hz=np.array(LOW_CUTOFF_HZ, dtype=np.float32),
        high_cut_hz=np.array(HIGH_CUTOFF_HZ, dtype=np.float32),
        zero_phase=np.array(ZERO_PHASE),
        reference_method=np.array("AS_ACQUIRED; NO_NEW_REFERENCE"),
        car_applied=np.array(False),
        bad_channel_interpolation=np.array(False),
        normalization_applied=np.array(False),
        resampling_applied=np.array(False),
        ica_applied=np.array(False),
        automatic_artifact_rejection=np.array(False),
        timestamp_used_as_clock=np.array(False),
        sample_index_used_as_clock=np.array(False),
        source_raw_relative_path=np.array(relative_raw),
        qa_final_status=np.array(qa_status),
        unresolved_flagged_channels=np.array(unresolved_flags),
    )

    # Verify the saved artifact immediately.
    with np.load(output_path, allow_pickle=False) as z:
        saved = z["eeg"]
        saved_mask = z["usable_channel_mask"]

        if saved.shape != (EXPECTED_SAMPLES, EXPECTED_CHANNELS):
            raise ValueError(
                f"Saved output shape invalid: {saved.shape}"
            )
        if not np.isfinite(saved).all():
            raise ValueError("Saved output contains NaN/Inf.")
        if saved_mask.shape != (EXPECTED_CHANNELS,):
            raise ValueError("Saved usable_channel_mask has wrong shape.")

    return {
        "canonical_subject": record["subject"],
        "original_subject_id": record["subject"],
        "condition": record["condition"],
        "gesture": record["gesture"],
        "gesture_code": record["gesture_code"],
        "set": record["set"],
        "source_filename": raw_path.name,
        "source_path": relative_raw,
        "output_filename": output_path.name,
        "output_path": relative_output,
        "input_samples": input_samples,
        "output_samples": filtered.shape[0],
        "input_channels": input_channels,
        "output_channels": filtered.shape[1],
        "sampling_rate_hz": FS,
        "filter_type": FILTER_TYPE,
        "filter_order": FILTER_ORDER,
        "low_cut_hz": LOW_CUTOFF_HZ,
        "high_cut_hz": HIGH_CUTOFF_HZ,
        "zero_phase": True,
        "reference_method": "AS_ACQUIRED; NO_NEW_REFERENCE",
        "car_applied": False,
        "artifact_action": "FLAG_ONLY_NO_INTERPOLATION",
        "bad_channels_flagged": ";".join(flagged_channels),
        "bad_channels_excluded_downstream": ";".join(
            [eeg_columns[i] for i in flagged_indices]
        ),
        "usable_channel_count": int(usable_mask.sum()),
        "normalization_applied": False,
        "resampling_applied": False,
        "ica_applied": False,
        "automatic_artifact_rejection": False,
        "feature_selection": False,
        "dimensionality_reduction": False,
        "class_balancing": False,
        "smote": False,
        "timestamp_used_as_clock": False,
        "sample_index_used_as_clock": False,
        "qa_final_status": qa_status,
        "preprocessing_status": "PASS",
        "failure_reason": "",
    }


# ============================================================
# 12. EXECUTE ALL 1050 RECORDINGS — RESUMABLE
# ============================================================

print("\n" + "=" * 96)
print("PROCESSING 1050 CONTINUOUS EEG RECORDINGS — RESUME ENABLED")
print("=" * 96)
print("No epoching | No normalization | No CAR | No ICA | No resampling")
print("1–40 Hz | 4th-order Butterworth | zero-phase | 125 Hz | 13 channels")

checkpoint, previous_manifest, previous_failures = load_resume_state()

# Verify that a checkpoint belongs to THIS frozen configuration.
if checkpoint:
    expected_signature = {
        "sampling_rate_hz": FS,
        "channels": EXPECTED_CHANNELS,
        "samples": EXPECTED_SAMPLES,
        "filter_type": FILTER_TYPE,
        "filter_order": FILTER_ORDER,
        "low_cut_hz": LOW_CUTOFF_HZ,
        "high_cut_hz": HIGH_CUTOFF_HZ,
        "zero_phase": ZERO_PHASE,
        "car": CAR,
        "ica": ICA,
        "resampling": RESAMPLING,
        "normalization": NORMALIZATION,
        "bad_channel_interpolation": BAD_CHANNEL_INTERPOLATION,
    }
    old_signature = checkpoint.get("config_signature", {})
    if old_signature and old_signature != expected_signature:
        raise RuntimeError(
            "Existing BLOCK03 checkpoint was created with a different "
            "frozen configuration. Refusing to mix configurations. "
            "Delete 03_PREPROCESSING only if a deliberate clean rerun "
            "is required."
        )

reusable_rows = build_existing_resume_rows(previous_manifest)
previous_failure_rows = build_previous_failures(previous_failures)

manifest_by_key = dict(reusable_rows)
failure_by_key = {}

# Previous failures are retained only for reporting. They are retried.
for row in previous_failure_rows:
    try:
        failure_by_key[manifest_row_key(row)] = row
    except Exception:
        pass

print(f"Existing checkpoint found : {'YES' if checkpoint else 'NO'}")
print(f"Verified reusable outputs  : {len(reusable_rows)}")
print(f"Previous failures to retry : {len(failure_by_key)}")

processed_keys = set(reusable_rows.keys())
attempted_count = 0

start = time.time()

for i, record in enumerate(raw_records, start=1):
    key = record["key"]

    # --------------------------------------------------------
    # RESUME CHECK
    # --------------------------------------------------------
    if key in manifest_by_key:
        # The artifact was independently verified above.
        print(
            f"[{i:4d}/{len(raw_records)}] "
            f"RESUME-SKIP {logical_key_text(key)}"
        )
        continue

    attempted_count += 1

    # If this recording previously failed, retry it now.
    try:
        row = process_one(record)

        manifest_by_key[key] = row
        processed_keys.add(key)

        # Remove stale previous failure after successful retry.
        failure_by_key.pop(key, None)

        status = "PASS"

    except Exception as exc:
        raw_rel = str(record["raw_path"].relative_to(RAW_ROOT))

        failure_by_key[key] = {
            "canonical_subject": record["subject"],
            "condition": record["condition"],
            "gesture": record["gesture"],
            "gesture_code": record["gesture_code"],
            "set": record["set"],
            "source_filename": record["raw_path"].name,
            "source_path": raw_rel,
            "preprocessing_status": "FAIL",
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }

        status = "FAIL"

    # --------------------------------------------------------
    # CHECKPOINT AFTER EVERY RECORDING
    # --------------------------------------------------------
    # This is intentionally conservative. If Colab disconnects,
    # at most the currently executing recording needs to be retried.
    current_manifest_rows = list(manifest_by_key.values())
    current_failure_rows = list(failure_by_key.values())

    save_resume_state(
        current_manifest_rows,
        current_failure_rows,
        processed_keys,
        attempted_count,
    )

    elapsed = time.time() - start

    print(
        f"[{i:4d}/{len(raw_records)}] "
        f"{status:4s} | "
        f"verified={len(processed_keys):4d} "
        f"failed={len(failure_by_key):3d} "
        f"elapsed={elapsed/60:.1f} min"
    )


# Materialize final in-memory tables from verified/retried state.
manifest_rows = list(manifest_by_key.values())
failure_rows = list(failure_by_key.values())

# Deterministic ordering.
manifest_rows = sorted(
    manifest_rows,
    key=lambda r: (
        str(r["condition"]),
        str(r["canonical_subject"]),
        int(r["gesture_code"]),
        str(r["set"]),
    ),
)
failure_rows = sorted(
    failure_rows,
    key=lambda r: (
        str(r["condition"]),
        str(r["canonical_subject"]),
        int(r["gesture_code"]),
        str(r["set"]),
    ),
)

# ============================================================
# 13. SAVE MANIFESTS
# ============================================================

manifest_df = pd.DataFrame(manifest_rows)
fail_df = pd.DataFrame(failure_rows)

# Rebuild the COMPLETE channel audit from all verified final outputs.
# This is essential for correct RESUME behavior: recordings skipped because
# they were already complete must still appear in the final 1050 × 13 audit.
channel_rows = build_channel_audit_from_manifest(manifest_rows)
channel_df = pd.DataFrame(channel_rows)

# Final verified-recording count used for the complete 13-channel audit check.
processed_count = len(manifest_rows)

if len(channel_df) != processed_count * EXPECTED_CHANNELS:
    raise RuntimeError(
        "Complete channel audit size mismatch: "
        f"expected {processed_count * EXPECTED_CHANNELS}, "
        f"found {len(channel_df)}."
    )

manifest_path = MANIFEST_ROOT / "preprocessing_manifest.csv"
failure_path = MANIFEST_ROOT / "preprocessing_failures.csv"
review_path = MANIFEST_ROOT / "preprocessing_review.csv"
channel_path = MANIFEST_ROOT / "channel_usability_manifest.csv"

if len(manifest_df):
    manifest_df.to_csv(manifest_path, index=False)
else:
    pd.DataFrame().to_csv(manifest_path, index=False)

if len(fail_df):
    fail_df.to_csv(failure_path, index=False)
else:
    pd.DataFrame(
        columns=[
            "canonical_subject", "condition", "gesture",
            "gesture_code", "set", "source_filename",
            "source_path", "preprocessing_status", "failure_reason"
        ]
    ).to_csv(failure_path, index=False)

# REVIEW means Block 02 diagnostic review, not automatic exclusion.
if len(manifest_df):
    review_df = manifest_df[
        manifest_df["qa_final_status"].astype(str).str.upper() == "REVIEW"
    ].copy()
else:
    review_df = pd.DataFrame()

review_df.to_csv(review_path, index=False)
channel_df.to_csv(channel_path, index=False)


# ============================================================
# 14. COVERAGE / SUMMARY TABLES
# ============================================================

if len(manifest_df):
    subject_summary = (
        manifest_df.groupby("canonical_subject")
        .agg(
            recordings=("canonical_subject", "size"),
            processed=("preprocessing_status", lambda s: int((s == "PASS").sum())),
            usable_channel_min=("usable_channel_count", "min"),
            usable_channel_max=("usable_channel_count", "max"),
        )
        .reset_index()
    )

    condition_summary = (
        manifest_df.groupby("condition")
        .agg(
            recordings=("condition", "size"),
            processed=("preprocessing_status", lambda s: int((s == "PASS").sum())),
        )
        .reset_index()
    )

    gesture_summary = (
        manifest_df.groupby(["gesture_code", "gesture"])
        .agg(
            recordings=("gesture_code", "size"),
            processed=("preprocessing_status", lambda s: int((s == "PASS").sum())),
        )
        .reset_index()
    )

    coverage = (
        manifest_df.groupby(
            ["canonical_subject", "condition", "gesture_code", "gesture", "set"]
        )
        .size()
        .reset_index(name="processed_recordings")
    )
else:
    subject_summary = pd.DataFrame()
    condition_summary = pd.DataFrame()
    gesture_summary = pd.DataFrame()
    coverage = pd.DataFrame()

subject_summary.to_csv(
    MANIFEST_ROOT / "subject_summary.csv", index=False
)
condition_summary.to_csv(
    MANIFEST_ROOT / "condition_summary.csv", index=False
)
gesture_summary.to_csv(
    MANIFEST_ROOT / "gesture_summary.csv", index=False
)
coverage.to_csv(
    MANIFEST_ROOT / "subject_gesture_set_coverage.csv", index=False
)


# ============================================================
# 15. FILTER VALIDATION — IMPLEMENTATION DIAGNOSTIC
# ============================================================

# Use the first successfully processed recording as a representative
# validation sample. This is not model selection and cannot alter the
# frozen filter.
representative = None
if len(manifest_df):
    representative = manifest_df.iloc[0].to_dict()

if representative is not None:
    rep_raw = RAW_ROOT / representative["source_path"]
    rep_out = BLOCK03_ROOT / representative["output_path"]

    rep_df = pd.read_csv(rep_raw)
    rep_cols = detect_eeg_columns(rep_df)
    rep_X = rep_df[rep_cols].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float64
    )

    with np.load(rep_out, allow_pickle=False) as z:
        rep_filtered = z["eeg"].astype(np.float64)

    ch_idx = 0
    n_plot = min(1500, EXPECTED_SAMPLES)

    plt.figure(figsize=(12, 4))
    plt.plot(
        np.arange(n_plot) / FS,
        rep_X[:n_plot, ch_idx],
        linewidth=0.8,
        label="Raw",
    )
    plt.plot(
        np.arange(n_plot) / FS,
        rep_filtered[:n_plot, ch_idx],
        linewidth=0.8,
        label="Filtered 1–40 Hz",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(
        f"BLOCK 03 raw vs filtered — "
        f"{representative['canonical_subject']} "
        f"{representative['condition']} "
        f"{representative['gesture']} Set{representative['set']} "
        f"— {rep_cols[ch_idx]}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        RAW_FILTERED_ROOT / "representative_raw_vs_filtered.png",
        dpi=180,
    )
    plt.close()

    # PSD comparison.
    freqs_raw, psd_raw = signal.welch(
        rep_X[:, ch_idx],
        fs=FS,
        nperseg=min(1024, EXPECTED_SAMPLES),
    )
    freqs_filt, psd_filt = signal.welch(
        rep_filtered[:, ch_idx],
        fs=FS,
        nperseg=min(1024, EXPECTED_SAMPLES),
    )

    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs_raw, psd_raw, linewidth=0.9, label="Raw")
    plt.semilogy(freqs_filt, psd_filt, linewidth=0.9, label="Filtered")
    plt.axvline(LOW_CUTOFF_HZ, linestyle="--", linewidth=0.8)
    plt.axvline(HIGH_CUTOFF_HZ, linestyle="--", linewidth=0.8)
    plt.xlim(0, FS / 2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title("BLOCK 03 representative PSD validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PSD_ROOT / "representative_psd_raw_vs_filtered.png",
        dpi=180,
    )
    plt.close()

# Frequency-response diagnostic of the exact SOS filter.
w, h = signal.sosfreqz(SOS, worN=4096, fs=FS)

plt.figure(figsize=(10, 4))
plt.plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-12)), linewidth=1.0)
plt.axvline(LOW_CUTOFF_HZ, linestyle="--", linewidth=0.8)
plt.axvline(HIGH_CUTOFF_HZ, linestyle="--", linewidth=0.8)
plt.xlim(0, FS / 2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("BLOCK 03 exact 4th-order Butterworth 1–40 Hz response")
plt.tight_layout()
plt.savefig(
    FILTER_VALIDATION_ROOT / "butterworth_1_40Hz_response.png",
    dpi=180,
)
plt.close()


# ============================================================
# 16. CONFIGURATION / PROVENANCE
# ============================================================

config = {
    "block": "03_PREPROCESSING",
    "status": "FROZEN_FOR_EXECUTION",
    "created_utc": utc_now(),
    "depends_on": "02_SIGNAL_QA",
    "feeds": "04_WINDOWING",
    "cohort": {
        "subjects": 25,
        "paired_ME_MI": True,
        "conditions": ["ME", "MI"],
        "gestures": EXPECTED_GESTURES,
        "sets": EXPECTED_SETS,
        "expected_recordings_total": EXPECTED_TOTAL_RECORDINGS,
    },
    "signal": {
        "sampling_rate_hz": FS,
        "expected_channels": EXPECTED_CHANNELS,
        "expected_samples": EXPECTED_SAMPLES,
        "nominal_duration_s": NOMINAL_DURATION_S,
        "row_order_is_signal_sequence": True,
        "timestamp_used_as_clock": TIMESTAMP_USED_AS_CLOCK,
        "sample_index_used_as_clock": SAMPLE_INDEX_USED_AS_CLOCK,
    },
    "filter": {
        "type": FILTER_TYPE,
        "order": FILTER_ORDER,
        "low_cut_hz": LOW_CUTOFF_HZ,
        "high_cut_hz": HIGH_CUTOFF_HZ,
        "zero_phase": ZERO_PHASE,
        "implementation": "scipy.signal.sosfiltfilt",
        "sos_design": "scipy.signal.butter(..., output='sos')",
        "padtype": "odd",
    },
    "reference": {
        "method": "AS_ACQUIRED; NO_NEW_REFERENCE",
        "car": False,
    },
    "artifact_control": {
        "bad_channel_interpolation": False,
        "ica": False,
        "automatic_artifact_rejection": False,
        "policy": (
            "Carry forward explicit Block 02 channel flags; retain all "
            "13 channels in output; save usable_channel_mask; BLOCK 04/05 "
            "must exclude explicitly flagged channels."
        ),
        "electrode_geometry_invented": False,
    },
    "downstream_exclusions": {
        "automatic_recording_exclusion": False,
        "automatic_subject_exclusion": False,
        "review_recordings_are_excluded": False,
    },
    "transformations_off": {
        "resampling": False,
        "epoching": False,
        "trial_reconstruction": False,
        "normalization": False,
        "feature_selection": False,
        "dimensionality_reduction": False,
        "class_balancing": False,
        "smote": False,
    },
    "reproducibility": {
        "random_operation": False,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": plt.matplotlib.__version__,
    },
    "raw_data_policy": {
        "raw_modified": False,
        "raw_deleted": False,
        "raw_overwritten": False,
    },
    "resume_logic": {
        "enabled": True,
        "checkpoint_file": "BLOCK03_checkpoint.json",
        "partial_manifest_file": "manifests/preprocessing_manifest_partial.csv",
        "partial_failure_file": "manifests/preprocessing_failures_partial.csv",
        "checkpoint_frequency": "after_every_recording",
        "reuse_only_after_output_verification": True,
        "retry_previous_failures": True,
        "max_work_lost_on_interruption": "at_most_current_recording",
        "configuration_mismatch_protection": True,
    },
}

config_path = BLOCK03_ROOT / "BLOCK03_config.json"
write_json(config_path, config)


# ============================================================
# 17. SUMMARY JSON
# ============================================================

processed_count = len(manifest_df)
failed_count = len(fail_df)
review_count = len(review_df)

if processed_count:
    final_manifest_keys = [
        manifest_row_key(r)
        for r in manifest_rows
    ]
    if len(final_manifest_keys) != len(set(final_manifest_keys)):
        raise RuntimeError(
            "Final preprocessing manifest contains duplicate logical recordings."
        )

summary = {
    "block": "03_PREPROCESSING",
    "status": "PENDING",
    "input_recordings": len(raw_records),
    "block01_selected_me_recordings": len(me_files),
    "block01_selected_mi_recordings": len(mi_files),
    "additional_me_csvs_on_disk_not_in_block01": len(extra_me_csvs),
    "additional_mi_csvs_on_disk_not_in_block01": len(extra_mi_csvs),
    "processed_recordings": processed_count,
    "failed_recordings": failed_count,
    "review_recordings": review_count,
    "sampling_rate_hz": FS,
    "channels": EXPECTED_CHANNELS,
    "expected_samples": EXPECTED_SAMPLES,
    "filter_low_hz": LOW_CUTOFF_HZ,
    "filter_high_hz": HIGH_CUTOFF_HZ,
    "filter_order": FILTER_ORDER,
    "filter_type": FILTER_TYPE,
    "zero_phase": True,
    "resampling": False,
    "normalization": False,
    "ica": False,
    "car": False,
    "bad_channel_interpolation": False,
    "automatic_artifact_rejection": False,
    "feature_selection": False,
    "dimensionality_reduction": False,
    "class_balancing": False,
    "smote": False,
    "timestamp_used_as_clock": False,
    "sample_index_used_as_clock": False,
    "raw_data_modified": False,
    "raw_data_deleted": False,
    "resume_enabled": True,
    "checkpoint_exists": CHECKPOINT_PATH.exists(),
}

summary_path = BLOCK03_ROOT / "BLOCK03_summary.json"


# ============================================================
# 18. HASHES
# ============================================================

# Hash config first.
config_hash = sha256_file(config_path)

# Hash Block 01 input inventory and Block 02 QA input.
inventory_hash = sha256_file(inventory_path)
qa_hash = sha256_file(b02_status_path)

# Hash manifests after writing.
manifest_hash = sha256_file(manifest_path)

hashes = {
    "BLOCK03_config.json_sha256": config_hash,
    "BLOCK01_recording_inventory.csv_sha256": inventory_hash,
    "BLOCK02_recording_final_qa_status.csv_sha256": qa_hash,
    "preprocessing_manifest.csv_sha256": manifest_hash,
}
write_json(BLOCK03_ROOT / "BLOCK03_hashes.json", hashes)


# ============================================================
# 19. FINAL PASS GATE
# ============================================================

# Required final conditions.
conditions = {
    "raw_me_count_525": len(me_files) == 525,
    "raw_mi_count_525": len(mi_files) == 525,
    "raw_total_1050": len(raw_records) == 1050,
    "block01_selected_me_525": len(me_files) == 525,
    "block01_selected_mi_525": len(mi_files) == 525,
    "processed_1050": processed_count == 1050,
    "failures_zero": failed_count == 0,
    "manifest_rows_1050": len(manifest_df) == 1050,
    "all_output_samples_5625": (
        processed_count == 0
        or bool((manifest_df["output_samples"] == EXPECTED_SAMPLES).all())
    ),
    "all_output_channels_13": (
        processed_count == 0
        or bool((manifest_df["output_channels"] == EXPECTED_CHANNELS).all())
    ),
    "all_input_samples_5625": (
        processed_count == 0
        or bool((manifest_df["input_samples"] == EXPECTED_SAMPLES).all())
    ),
    "all_input_channels_13": (
        processed_count == 0
        or bool((manifest_df["input_channels"] == EXPECTED_CHANNELS).all())
    ),
    "all_filter_1_40": (
        processed_count == 0
        or bool(
            (manifest_df["low_cut_hz"] == 1.0).all()
            and (manifest_df["high_cut_hz"] == 40.0).all()
            and (manifest_df["filter_order"] == 4).all()
        )
    ),
    "all_zero_phase": (
        processed_count == 0
        or bool(manifest_df["zero_phase"].astype(bool).all())
    ),
    "no_resampling": (
        processed_count == 0
        or bool((~manifest_df["resampling_applied"].astype(bool)).all())
    ),
    "no_normalization": (
        processed_count == 0
        or bool((~manifest_df["normalization_applied"].astype(bool)).all())
    ),
    "no_ica": (
        processed_count == 0
        or bool((~manifest_df["ica_applied"].astype(bool)).all())
    ),
    "no_car": (
        processed_count == 0
        or bool((~manifest_df["car_applied"].astype(bool)).all())
    ),
    "no_bad_channel_interpolation": (
        processed_count == 0
        or bool(
            (manifest_df["artifact_action"] == "FLAG_ONLY_NO_INTERPOLATION").all()
        )
    ),
    "no_feature_selection": (
        processed_count == 0
        or bool((~manifest_df["feature_selection"].astype(bool)).all())
    ),
    "no_dimensionality_reduction": (
        processed_count == 0
        or bool(
            (~manifest_df["dimensionality_reduction"].astype(bool)).all()
        )
    ),
    "no_class_balancing": (
        processed_count == 0
        or bool((~manifest_df["class_balancing"].astype(bool)).all())
    ),
    "no_smote": (
        processed_count == 0
        or bool((~manifest_df["smote"].astype(bool)).all())
    ),
    "raw_policy_clean": True,
    "channel_audit_rows_13650": len(channel_df) == 1050 * 13,
}

final_pass = all(conditions.values())

summary["status"] = "PASS" if final_pass else "FAIL"
summary["pass_conditions"] = conditions
summary["next_block"] = (
    "04_WINDOWING" if final_pass else "STOP_AND_REVIEW"
)
summary["completion_utc"] = utc_now()

write_json(summary_path, summary)


# Completion marker.
complete = {
    "block": "03_PREPROCESSING",
    "status": "PASS" if final_pass else "FAIL",
    "complete": bool(final_pass),
    "input_recordings": len(raw_records),
    "block01_selected_me_recordings": len(me_files),
    "block01_selected_mi_recordings": len(mi_files),
    "additional_me_csvs_on_disk_not_in_block01": len(extra_me_csvs),
    "additional_mi_csvs_on_disk_not_in_block01": len(extra_mi_csvs),
    "processed_recordings": processed_count,
    "failed_recordings": failed_count,
    "review_recordings": review_count,
    "raw_data_modified": False,
    "raw_data_deleted": False,
    "raw_data_overwritten": False,
    "preprocessing": {
        "filter": "4th-order Butterworth 1–40 Hz zero-phase",
        "sampling_rate_hz": 125,
        "channels": 13,
        "continuous_signal_preserved": True,
        "sample_count_preserved": True,
        "resampling": False,
        "normalization": False,
        "ica": False,
        "car": False,
        "bad_channel_interpolation": False,
        "automatic_artifact_rejection": False,
    },
    "bad_channel_policy": {
        "block02_explicit_flags_carried_forward": True,
        "all_13_channels_retained_in_saved_output": True,
        "usable_channel_mask_saved": True,
        "block04_must_apply_mask": True,
    },
    "resume_logic": {
        "enabled": True,
        "checkpoint_file": "BLOCK03_checkpoint.json",
        "checkpoint_frequency": "after_every_recording",
        "reuses_verified_outputs": True,
        "retries_failed_recordings": True,
    },
    "hashes": hashes,
    "conditions": conditions,
    "next_block": "04_WINDOWING" if final_pass else "STOP_AND_REVIEW",
    "completion_utc": utc_now(),
}

complete_path = BLOCK03_ROOT / "BLOCK03_complete.json"
write_json(complete_path, complete)

# Final checkpoint is retained deliberately for audit/resume provenance.
# A PASS checkpoint is not deleted; it documents the completed state.
save_resume_state(
    manifest_rows,
    failure_rows,
    set(manifest_by_key.keys()),
    attempted_count,
)


# ============================================================
# 20. HUMAN-READABLE SUMMARY
# ============================================================

summary_txt = BLOCK03_ROOT / "BLOCK03_summary.txt"

lines = [
    "BLOCK 03 — EEG PREPROCESSING & ARTIFACT-CONTROL PIPELINE",
    "=" * 72,
    f"Status: {'PASS' if final_pass else 'FAIL'}",
    f"Input recordings: {len(raw_records)}",
    f"Processed recordings: {processed_count}",
    f"Failed recordings: {failed_count}",
    f"Review recordings (diagnostic only): {review_count}",
    "",
    "Frozen preprocessing:",
    "  Sampling rate: 125 Hz (no resampling)",
    "  Channels: 13",
    "  Filter: 4th-order Butterworth 1–40 Hz",
    "  Zero-phase: yes",
    "  CAR: no",
    "  ICA: no",
    "  Bad-channel interpolation: no",
    "  Epoching/windowing: no",
    "  Normalization: no",
    "  Feature selection: no",
    "  Dimensionality reduction: no",
    "  Class balancing: no",
    "  SMOTE: no",
    "  Trial reconstruction: no",
    "",
    "Bad-channel handling:",
    "  Explicit Block 02 channel flags are carried forward.",
    "  All 13 channels remain in each saved array.",
    "  usable_channel_mask is saved for downstream BLOCK 04/05.",
    "  No fabricated electrode geometry is used.",
    "",
    "Raw data policy:",
    "  Modified: NO",
    "  Deleted: NO",
    "  Overwritten: NO",
    "",
    "Resume logic:",
    "  Enabled: YES",
    "  Checkpoint: BLOCK03_checkpoint.json",
    "  Checkpoint frequency: after every recording",
    "  Existing outputs are reused only after verification.",
    "  Previous failures are retried.",
    "",
    f"Next block: {'04_WINDOWING' if final_pass else 'STOP_AND_REVIEW'}",
    "",
    "Hashes:",
    f"  BLOCK03_config.json: {config_hash}",
    f"  BLOCK01 inventory: {inventory_hash}",
    f"  BLOCK02 QA table: {qa_hash}",
    f"  preprocessing manifest: {manifest_hash}",
]

summary_txt.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 21. FINAL CONSOLE OUTPUT
# ============================================================

print("\n" + "=" * 96)
print("BLOCK 03 FINAL RESULT")
print("=" * 96)

print(f"STATUS                  : {'PASS' if final_pass else 'FAIL'}")
print(f"INPUT RECORDINGS        : {len(raw_records)}")
print(f"BLOCK01 SELECTED ME     : {len(me_files)}")
print(f"BLOCK01 SELECTED MI     : {len(mi_files)}")
print(f"EXTRA ME CSVs (IGNORED) : {len(extra_me_csvs)}")
print(f"EXTRA MI CSVs (IGNORED) : {len(extra_mi_csvs)}")
print(f"PROCESSED RECORDINGS    : {processed_count}")
print(f"FAILED RECORDINGS       : {failed_count}")
print(f"REVIEW RECORDINGS       : {review_count}")
print(f"SAMPLING RATE           : {FS} Hz")
print(f"CHANNELS                : {EXPECTED_CHANNELS}")
print(f"SAMPLES / RECORDING    : {EXPECTED_SAMPLES}")
print("FILTER                  : 4th-order Butterworth 1–40 Hz")
print("ZERO-PHASE              : YES")
print("RESAMPLING              : NO")
print("CAR                     : NO")
print("ICA                     : NO")
print("BAD-CHANNEL INTERP.     : NO")
print("NORMALIZATION           : NO")
print("WINDOWING               : NO")
print("FEATURE EXTRACTION      : NO")
print("MODEL FITTING           : NO")
print("RAW DATA MODIFIED       : NO")
print("RAW DATA DELETED        : NO")
print("RESUME LOGIC            : ENABLED")
print("CHECKPOINT FREQUENCY    : AFTER EVERY RECORDING")
print(f"CHANNEL AUDIT ROWS      : {len(channel_df)}")

failed_conditions = [k for k, v in conditions.items() if not v]
if failed_conditions:
    print("\nFAILED GATES:")
    for name in failed_conditions:
        print("  -", name)

print("\nOUTPUT ROOT:")
print(BLOCK03_ROOT)

print(
    "\nNEXT BLOCK:",
    "BLOCK 04 — TEMPORAL WINDOWING & WINDOW-CONFIGURATION SELECTION"
    if final_pass
    else "STOP — REVIEW BLOCK 03 FAILURES",
)

if final_pass:
    print("\n✓ BLOCK 03 PASS — READY FOR BLOCK 04")
else:
    print("\n✗ BLOCK 03 FAIL — DO NOT START BLOCK 04")
