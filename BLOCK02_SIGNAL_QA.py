# ============================================================
# PAPER 2 — BLOCK 02 FINAL CLEAN
# SIGNAL QUALITY ASSURANCE (QA)
# ============================================================
#
# PURPOSE
# -------
# Inspect the CURRENT 1,050 scientific EEG recordings established
# by BLOCK 01.
#
# BLOCK 02 IS READ-ONLY WITH RESPECT TO RAW DATA.
#
# It:
#   ✓ uses BLOCK01 recording_inventory.csv as authoritative identity
#   ✓ reconciles every manifest recording against the raw filesystem
#   ✓ reads the EEG signal values
#   ✓ verifies 13 EEG channels
#   ✓ verifies 5,625 samples per recording
#   ✓ checks NaN / Inf
#   ✓ checks flat / near-flat channels
#   ✓ checks highly repeated values
#   ✓ checks empirical extreme-value repetition
#   ✓ checks signal discontinuities
#   ✓ computes exact EEG signal fingerprints
#   ✓ detects exact duplicate EEG recordings
#   ✓ creates subject / condition / gesture / set QA summaries
#   ✓ creates figures useful for QA inspection
#
# IMPORTANT TIMESTAMP DECISION
# ----------------------------
# Timestamp is METADATA ONLY.
#
# Do NOT:
#   ✗ estimate sampling frequency from timestamps
#   ✗ estimate duration from timestamps
#   ✗ resample using timestamps
#   ✗ reject recordings for timestamp irregularity
#   ✗ use Sample Index as the EEG sample clock
#
# The signal sample sequence is the RECORDED ROW ORDER.
#
# Protocol:
#   Fs = 125 Hz
#   expected samples = 5,625
#   nominal duration = 45 s
#
# No filtering, resampling, interpolation, normalization,
# windowing, feature extraction, modeling, statistics, or
# performance-based subject exclusion is performed.
#
# NO AUTOMATIC SUBJECT/RECORDING EXCLUSION
# -----------------------------------------
# QA findings are reported as PASS / REVIEW / FAIL diagnostics.
#
# FAIL is reserved for structural/readability failures:
#   - unreadable CSV
#   - missing required EEG columns
#   - non-numeric signal data
#   - nonfinite signal values
#   - wrong sample count
#
# REVIEW is used for signal-quality diagnostics such as:
#   - flat channel
#   - near-flat channel
#   - highly repeated values
#   - empirical extreme-value repetition
#   - discontinuities
#
# A REVIEW does NOT automatically remove a recording.
#
# DRIVE OUTPUT POLICY
# -------------------
# Only the official BLOCK 02 output directory is written:
#
#   Paper2_EEG_BCI/02_SIGNAL_QA/
#
# No DIAGNOSTIC subdirectory is created by this script.
# Additional exploratory diagnostics should remain Colab-only
# unless explicitly requested.
#
# ============================================================


# ============================================================
# 1. MOUNT DRIVE
# ============================================================

from google.colab import drive
drive.mount("/content/drive")


# ============================================================
# 2. IMPORTS
# ============================================================

from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import re
import shutil
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 3. EXACT PROJECT PATHS
# ============================================================

BASE_DIR = Path(
    "F:\Faruk\EEG_Motor_Execution_Imagery\Paper2_EEG_BCI"
)

ME_RAW = BASE_DIR / "Motor_Execution_Raw_Data"
MI_RAW = BASE_DIR / "Motor_Imagery_Raw_Data"

PROJECT_ROOT = BASE_DIR / "Paper2_EEG_BCI"

BLOCK00_DIR = PROJECT_ROOT / "00_PROTOCOL"
BLOCK01_DIR = PROJECT_ROOT / "01_SUBJECT_MATCHING"
BLOCK02_DIR = PROJECT_ROOT / "02_SIGNAL_QA"

BLOCK00_COMPLETE = BLOCK00_DIR / "BLOCK00_complete.json"
BLOCK00_CONFIG = BLOCK00_DIR / "configs" / "BLOCK00_config.json"

BLOCK01_COMPLETE = BLOCK01_DIR / "BLOCK01_complete.json"
BLOCK01_CONFIG = BLOCK01_DIR / "BLOCK01_config.json"
BLOCK01_INVENTORY = BLOCK01_DIR / "recording_inventory.csv"


# ============================================================
# 4. FROZEN BLOCK 02 CONSTANTS
# ============================================================

BLOCK_ID = "BLOCK02"
BLOCK_VERSION = "PAPER2-BLOCK02-SIGNAL-QA-FINAL-CLEAN-1.1"

EXPECTED_SUBJECTS = 25
EXPECTED_RECORDINGS = 1050

EXPECTED_SAMPLES = 5625
NOMINAL_FS_HZ = 125.0
NOMINAL_DURATION_SECONDS = 45.0

EXPECTED_CHANNELS = [
    f"EEG_ch-{i:02d}"
    for i in range(1, 14)
]

EXPECTED_CHANNEL_COUNT = 13

GESTURES = {
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
}

CONDITIONS = ("ME", "MI")
SETS = ("A", "B", "C")


# ============================================================
# 5. QA THRESHOLDS — FROZEN DIAGNOSTIC DEFINITIONS
# ============================================================

# Structural failures.
NONFINITE_FAIL_COUNT = 0
SAMPLE_COUNT_FAIL_IF_NOT_EQUAL = EXPECTED_SAMPLES
CHANNEL_COUNT_FAIL_IF_NOT_EQUAL = EXPECTED_CHANNEL_COUNT

# Flatness diagnostics.
FLAT_UNIQUE_VALUE_COUNT = 1
NEAR_FLAT_STD_THRESHOLD = 1e-12

# Repeated-value diagnostic.
HIGH_REPEAT_FRACTION_THRESHOLD = 0.95

# Empirical extreme repetition.
# This is NOT called ADC clipping because the ADC limits are
# not documented in the current protocol.
EXTREME_REPEAT_FRACTION_THRESHOLD = 0.01

# Discontinuity diagnostic.
# Robust z-score of first difference:
#   |diff - median(diff)| / (1.4826*MAD(diff))
DISCONTINUITY_ROBUST_Z_THRESHOLD = 20.0
DISCONTINUITY_REVIEW_COUNT = 3

# Fingerprint rounding:
# NONE. Exact signal fingerprint means exact float-array bytes
# after deterministic dtype conversion.
FINGERPRINT_DTYPE = np.dtype("<f8")


# ============================================================
# 6. HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def safe_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def normalized_path_key(path):
    return str(Path(path).resolve()).replace("\\", "/").lower()


def is_audit_path(path, raw_root):
    try:
        rel = path.relative_to(raw_root)
    except ValueError:
        rel = path

    return any(
        part.upper() == "AUDIT"
        for part in rel.parts[:-1]
    )


def discover_scientific_csvs(raw_root):
    """
    Scientific recording definition frozen in BLOCK 01:
        .csv files outside AUDIT directories.
    """
    paths = []

    for path in sorted(raw_root.rglob("*.csv")):
        if path.is_file() and not is_audit_path(path, raw_root):
            paths.append(path)

    return paths


def canonicalize_column_name(name):
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(name).strip().lower(),
    )


def detect_eeg_columns(columns):
    """
    Accept current real-world naming variants such as:
        EEG_ch-01
        EEG_ch-1
        EEG_ch_01
        EEG_ch1

    Return them ordered as EEG channel 1..13.

    No EMG columns are accepted as EEG.
    """
    mapping = {}

    for col in columns:
        text = str(col).strip()

        m = re.fullmatch(
            r"(?i)EEG[_\-]?ch[_\-]?0*(\d+)",
            text,
        )

        if m:
            number = int(m.group(1))
            if 1 <= number <= 13:
                mapping[number] = text

    if len(mapping) != 13:
        return None

    return [
        mapping[i]
        for i in range(1, 14)
    ]


def normalize_eeg_dataframe(df):
    """
    Return the original dataframe and ordered EEG columns.

    Signal values are not changed.
    Column names are not written back to raw files.
    """
    eeg_columns = detect_eeg_columns(df.columns)

    if eeg_columns is None:
        return None, None

    return df, eeg_columns


def read_signal_file(path):
    """
    Read a raw CSV only for QA.

    No raw file is modified.
    """
    df = pd.read_csv(path)

    normalized_df, eeg_columns = normalize_eeg_dataframe(df)

    if normalized_df is None:
        raise ValueError(
            "Could not identify exactly 13 EEG channels."
        )

    signal_df = normalized_df[eeg_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    X = signal_df.to_numpy(dtype=np.float64)

    return df, X, eeg_columns


def signal_fingerprint(X):
    """
    Exact fingerprint of the 5625 x 13 EEG array.

    The values are converted deterministically to little-endian
    float64. No rounding is applied.
    """
    arr = np.asarray(
        X,
        dtype=FINGERPRINT_DTYPE,
        order="C",
    )

    return hashlib.sha256(
        arr.tobytes(order="C")
    ).hexdigest()


def csv_file_sha256(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def median_absolute_deviation(x):
    x = np.asarray(x, dtype=np.float64)

    if x.size == 0:
        return np.nan

    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def longest_run_fraction(values):
    """
    Longest exact repeated-value run / number of samples.
    """
    values = np.asarray(values)

    if values.size == 0:
        return 0.0, 0

    longest = 1
    current = 1

    for i in range(1, len(values)):
        if values[i] == values[i - 1]:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 1

    return (
        float(longest / len(values)),
        int(longest),
    )


def channel_qa(X, subject, condition, gesture_code, gesture_name,
               set_code, filename, relative_path):

    rows = []
    extreme_rows = []
    discontinuity_rows = []

    for ch_idx in range(X.shape[1]):

        values = X[:, ch_idx]

        finite = values[np.isfinite(values)]

        if finite.size == 0:
            unique_count = 0
            mean = std = median = mad = minimum = maximum = np.nan
            repeat_fraction = 1.0
            longest_fraction = 1.0
            longest_run = len(values)
            extreme_min_fraction = 0.0
            extreme_max_fraction = 0.0
            max_robust_z = np.nan
            large_jump_count = 0
            large_jump_fraction = 0.0

        else:

            unique_count = int(np.unique(finite).size)

            mean = float(np.mean(finite))
            std = float(np.std(finite))
            median = float(np.median(finite))
            mad = median_absolute_deviation(finite)
            minimum = float(np.min(finite))
            maximum = float(np.max(finite))

            _, counts = np.unique(
                finite,
                return_counts=True,
            )

            repeat_fraction = float(
                np.max(counts) / finite.size
            )

            longest_fraction, longest_run = (
                longest_run_fraction(finite)
            )

            min_count = np.sum(
                finite == minimum
            )

            max_count = np.sum(
                finite == maximum
            )

            extreme_min_fraction = float(
                min_count / finite.size
            )

            extreme_max_fraction = float(
                max_count / finite.size
            )

            # ------------------------------------------------
            # Discontinuity
            # ------------------------------------------------

            if finite.size >= 3:

                diff = np.diff(values)

                diff = diff[
                    np.isfinite(diff)
                ]

                if diff.size > 0:

                    diff_median = float(
                        np.median(diff)
                    )

                    diff_mad = median_absolute_deviation(
                        diff
                    )

                    robust_scale = (
                        1.4826 * diff_mad
                    )

                    if (
                        np.isfinite(robust_scale)
                        and robust_scale > 0
                    ):

                        robust_z = np.abs(
                            diff - diff_median
                        ) / robust_scale

                        max_robust_z = float(
                            np.max(robust_z)
                        )

                        large_jump_count = int(
                            np.sum(
                                robust_z
                                > DISCONTINUITY_ROBUST_Z_THRESHOLD
                            )
                        )

                    else:
                        max_robust_z = 0.0
                        large_jump_count = 0

                    large_jump_fraction = float(
                        large_jump_count
                        / max(1, diff.size)
                    )

                else:
                    max_robust_z = np.nan
                    large_jump_count = 0
                    large_jump_fraction = 0.0

            else:
                max_robust_z = np.nan
                large_jump_count = 0
                large_jump_fraction = 0.0

        flat = (
            unique_count <= FLAT_UNIQUE_VALUE_COUNT
        )

        near_flat = (
            np.isfinite(std)
            and std <= NEAR_FLAT_STD_THRESHOLD
        )

        highly_repeated = (
            repeat_fraction
            >= HIGH_REPEAT_FRACTION_THRESHOLD
        )

        empirical_extreme_repeat = (
            max(
                extreme_min_fraction,
                extreme_max_fraction,
            )
            >= EXTREME_REPEAT_FRACTION_THRESHOLD
        )

        discontinuity_review = (
            max_robust_z
            > DISCONTINUITY_ROBUST_Z_THRESHOLD
            and
            large_jump_count
            >= DISCONTINUITY_REVIEW_COUNT
        )

        rows.append({
            "condition": condition,
            "subject_id": subject,
            "gesture_code": gesture_code,
            "gesture_name": gesture_name,
            "set_code": set_code,
            "filename": filename,
            "relative_path": relative_path,
            "channel": f"EEG_ch-{ch_idx + 1:02d}",
            "channel_index": ch_idx + 1,
            "finite_sample_count": int(finite.size),
            "unique_finite_values": unique_count,
            "mean": mean,
            "std": std,
            "median": median,
            "mad": mad,
            "minimum": minimum,
            "maximum": maximum,
            "dominant_value_fraction": repeat_fraction,
            "longest_exact_run": longest_run,
            "longest_exact_run_fraction": longest_fraction,
            "flat_channel": flat,
            "near_flat_channel": near_flat,
            "highly_repeated_values": highly_repeated,
            "empirical_extreme_repeat": empirical_extreme_repeat,
            "max_robust_z": max_robust_z,
            "large_jump_count": large_jump_count,
            "large_jump_fraction": large_jump_fraction,
            "discontinuity_review": discontinuity_review,
        })

        extreme_rows.append({
            "condition": condition,
            "subject_id": subject,
            "gesture_code": gesture_code,
            "gesture_name": gesture_name,
            "set_code": set_code,
            "filename": filename,
            "relative_path": relative_path,
            "channel": f"EEG_ch-{ch_idx + 1:02d}",
            "minimum": minimum,
            "maximum": maximum,
            "minimum_repeat_fraction": extreme_min_fraction,
            "maximum_repeat_fraction": extreme_max_fraction,
            "empirical_extreme_repeat": empirical_extreme_repeat,
        })

    return rows, extreme_rows, discontinuity_rows


# ============================================================
# 7. VERIFY BLOCK 00 AND BLOCK 01
# ============================================================

print("=" * 78)
print("PAPER 2 — BLOCK 02 FINAL CLEAN")
print("SIGNAL QUALITY ASSURANCE")
print("=" * 78)
print()

print("VERIFYING BLOCK 00...")

if not BLOCK00_COMPLETE.exists():
    raise FileNotFoundError(
        f"BLOCK 00 completion marker not found:\n{BLOCK00_COMPLETE}"
    )

if not BLOCK00_CONFIG.exists():
    raise FileNotFoundError(
        f"BLOCK 00 configuration not found:\n{BLOCK00_CONFIG}"
    )

block00_complete = json.loads(
    BLOCK00_COMPLETE.read_text(encoding="utf-8")
)

if block00_complete.get("status") != "COMPLETE_AND_FROZEN":
    raise RuntimeError(
        "BLOCK 00 is not COMPLETE_AND_FROZEN."
    )

print("✓ BLOCK 00 = COMPLETE_AND_FROZEN")

print()
print("VERIFYING BLOCK 01...")

if not BLOCK01_COMPLETE.exists():
    raise FileNotFoundError(
        f"BLOCK 01 completion marker not found:\n{BLOCK01_COMPLETE}"
    )

if not BLOCK01_CONFIG.exists():
    raise FileNotFoundError(
        f"BLOCK 01 configuration not found:\n{BLOCK01_CONFIG}"
    )

if not BLOCK01_INVENTORY.exists():
    raise FileNotFoundError(
        f"BLOCK 01 recording inventory not found:\n{BLOCK01_INVENTORY}"
    )

block01_complete = json.loads(
    BLOCK01_COMPLETE.read_text(encoding="utf-8")
)

if block01_complete.get("status") != "PASS":
    raise RuntimeError(
        "BLOCK 01 is not PASS. BLOCK 02 cannot start."
    )

if block01_complete.get("complete_marker") is not True:
    raise RuntimeError(
        "BLOCK 01 completion marker is not valid."
    )

print("✓ BLOCK 01 = PASS")
print("✓ BLOCK 01 recording inventory found")
print()


# ============================================================
# 8. READ AUTHORITATIVE BLOCK 01 INVENTORY
# ============================================================

print("=" * 78)
print("LOADING AUTHORITATIVE BLOCK 01 RECORDING INVENTORY")
print("=" * 78)

inventory = pd.read_csv(
    BLOCK01_INVENTORY,
    dtype=str,
)

required_inventory_columns = {
    "condition",
    "subject_id",
    "gesture_code",
    "gesture_name",
    "set_code",
    "logical_key",
    "relative_path",
    "filename",
    "absolute_path",
}

missing_inventory_columns = (
    required_inventory_columns
    - set(inventory.columns)
)

if missing_inventory_columns:
    raise RuntimeError(
        "BLOCK 01 inventory is missing required columns: "
        + ", ".join(sorted(missing_inventory_columns))
    )

inventory = inventory.fillna("")

print(
    f"BLOCK 01 inventory rows : {len(inventory):,}"
)

if len(inventory) != EXPECTED_RECORDINGS:
    raise RuntimeError(
        f"Expected {EXPECTED_RECORDINGS} BLOCK 01 inventory rows, "
        f"found {len(inventory)}."
    )

if inventory["logical_key"].duplicated().any():
    raise RuntimeError(
        "BLOCK 01 inventory contains duplicate logical keys."
    )

print("✓ 1,050 unique logical recordings loaded")
print()


# ============================================================
# 9. RECONCILE INVENTORY WITH CURRENT RAW FILESYSTEM
# ============================================================

print("=" * 78)
print("RECONCILING BLOCK 01 INVENTORY WITH CURRENT RAW FILESYSTEM")
print("=" * 78)

raw_paths = (
    discover_scientific_csvs(ME_RAW)
    +
    discover_scientific_csvs(MI_RAW)
)

raw_by_path = {
    normalized_path_key(p): p
    for p in raw_paths
}

manifest_path_keys = set()
manifest_not_on_disk = []
inventory_duplicate_paths = []

for _, row in inventory.iterrows():

    path = Path(row["absolute_path"])

    key = normalized_path_key(path)

    if key in manifest_path_keys:
        inventory_duplicate_paths.append(
            row["absolute_path"]
        )

    manifest_path_keys.add(key)

    if not path.exists():
        manifest_not_on_disk.append(
            row["absolute_path"]
        )

raw_not_in_manifest = sorted(
    set(raw_by_path.keys())
    - manifest_path_keys
)

print(f"Current raw scientific CSVs : {len(raw_paths):,}")
print(f"Manifest rows               : {len(inventory):,}")
print(
    f"Manifest paths missing on disk : "
    f"{len(manifest_not_on_disk):,}"
)
print(
    f"Raw CSVs not in manifest       : "
    f"{len(raw_not_in_manifest):,}"
)
print(
    f"Duplicate manifest paths       : "
    f"{len(inventory_duplicate_paths):,}"
)

if (
    manifest_not_on_disk
    or raw_not_in_manifest
    or inventory_duplicate_paths
):
    raise RuntimeError(
        "BLOCK 01 inventory does not reconcile exactly with "
        "the current raw scientific CSV filesystem."
    )

print("✓ RAW ↔ BLOCK 01 RECONCILIATION PASS")
print()


# ============================================================
# 10. CLEAN OLD BLOCK 02 OUTPUT
# ============================================================
#
# ONLY the official BLOCK 02 directory is removed.
# Raw data and BLOCK 01 are never removed.
#
# ============================================================

print("=" * 78)
print("CLEANING OLD BLOCK 02 OUTPUT")
print("=" * 78)

if BLOCK02_DIR.exists():
    print(f"Removing old BLOCK 02 output:")
    print(BLOCK02_DIR)

    shutil.rmtree(BLOCK02_DIR)

    print("✓ Old BLOCK 02 output removed")
else:
    print("✓ No previous BLOCK 02 output found")

BLOCK02_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("✓ Clean BLOCK 02 directory created")
print()


# ============================================================
# 11. QA PROCESSING
# ============================================================

raw_file_rows = []
channel_rows = []
amplitude_rows = []
duration_rows = []
missing_rows = []
flat_rows = []
clipping_rows = []
discontinuity_rows = []
duplicate_rows = []
errors = []

fingerprint_groups = defaultdict(list)

recording_status = []

# Figures: representative only, so Drive is not flooded.
figure_dir = BLOCK02_DIR / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

representatives = {
    "normal": None,
    "flat": None,
    "extreme_repeat": None,
    "discontinuity": None,
}


# ============================================================
# 12. PROCESS EACH RECORDING
# ============================================================

print("=" * 78)
print("READING AND QA-INSPECTING 1,050 EEG RECORDINGS")
print("=" * 78)

for index, row in inventory.iterrows():

    subject = row["subject_id"]
    condition = row["condition"]
    gesture_code = int(row["gesture_code"])
    gesture_name = row["gesture_name"]
    set_code = row["set_code"]
    logical_key = row["logical_key"]
    relative_path = row["relative_path"]
    filename = row["filename"]
    path = Path(row["absolute_path"])

    base = {
        "condition": condition,
        "subject_id": subject,
        "gesture_code": gesture_code,
        "gesture_name": gesture_name,
        "set_code": set_code,
        "logical_key": logical_key,
        "filename": filename,
        "relative_path": relative_path,
    }

    try:

        file_size = path.stat().st_size
        csv_hash = csv_file_sha256(path)

        # ----------------------------------------------------
        # Read CSV
        # ----------------------------------------------------

        df = pd.read_csv(path)

        row_count = int(len(df))
        column_names = [str(c) for c in df.columns]

        eeg_columns = detect_eeg_columns(
            column_names
        )

        schema_pass = (
            eeg_columns is not None
        )

        numeric_conversion_errors = 0

        if schema_pass:

            signal_df = df[eeg_columns].apply(
                pd.to_numeric,
                errors="coerce",
            )

            X = signal_df.to_numpy(
                dtype=np.float64
            )

            # Any NaN created by non-numeric values or already
            # present in the source is treated as nonfinite.
            nonfinite_mask = ~np.isfinite(X)
            nonfinite_count = int(
                np.sum(nonfinite_mask)
            )

            # Count conversion failures separately.
            numeric_conversion_errors = int(
                signal_df.isna().sum().sum()
                - pd.DataFrame(df[eeg_columns]).isna().sum().sum()
            )

            actual_channel_count = int(
                len(eeg_columns)
            )

            actual_sample_count = int(
                X.shape[0]
            )

        else:

            X = None
            nonfinite_count = -1
            actual_channel_count = (
                len(eeg_columns)
                if eeg_columns is not None
                else 0
            )
            actual_sample_count = row_count

        # ----------------------------------------------------
        # Structural checks
        # ----------------------------------------------------

        sample_count_fail = (
            actual_sample_count
            != EXPECTED_SAMPLES
        )

        channel_count_fail = (
            actual_channel_count
            != EXPECTED_CHANNEL_COUNT
        )

        schema_fail = not schema_pass

        if X is not None:

            finite_all = np.isfinite(X).all()

            if not finite_all:
                structural_status = "FAIL"
            elif sample_count_fail:
                structural_status = "FAIL"
            elif channel_count_fail:
                structural_status = "FAIL"
            else:
                structural_status = "PASS"

        else:
            structural_status = "FAIL"

        # ----------------------------------------------------
        # Initialize per-recording diagnostics
        # ----------------------------------------------------

        flat_channel_count = 0
        near_flat_channel_count = 0
        highly_repeated_channel_count = 0
        empirical_extreme_channel_count = 0
        discontinuity_channel_count = 0

        min_signal = np.nan
        max_signal = np.nan
        mean_signal = np.nan
        median_signal = np.nan
        std_signal = np.nan
        mad_signal = np.nan

        duplicate_fingerprint = ""
        signal_fingerprint_value = ""

        if X is not None:

            finite_values = X[
                np.isfinite(X)
            ]

            if finite_values.size:

                min_signal = float(
                    np.min(finite_values)
                )

                max_signal = float(
                    np.max(finite_values)
                )

                mean_signal = float(
                    np.mean(finite_values)
                )

                median_signal = float(
                    np.median(finite_values)
                )

                std_signal = float(
                    np.std(finite_values)
                )

                mad_signal = median_absolute_deviation(
                    finite_values
                )

            # ------------------------------------------------
            # Exact EEG fingerprint
            # ------------------------------------------------

            if (
                X.shape
                ==
                (
                    EXPECTED_SAMPLES,
                    EXPECTED_CHANNEL_COUNT,
                )
                and np.isfinite(X).all()
            ):

                signal_fingerprint_value = (
                    signal_fingerprint(X)
                )

                fingerprint_groups[
                    signal_fingerprint_value
                ].append({
                    "subject_id": subject,
                    "condition": condition,
                    "gesture_code": gesture_code,
                    "gesture_name": gesture_name,
                    "set_code": set_code,
                    "logical_key": logical_key,
                    "filename": filename,
                    "relative_path": relative_path,
                    "absolute_path": str(path),
                    "csv_sha256": csv_hash,
                })

            # ------------------------------------------------
            # Channel QA
            # ------------------------------------------------

            (
                this_channel_rows,
                this_extreme_rows,
                this_discontinuity_rows,
            ) = channel_qa(
                X,
                subject,
                condition,
                gesture_code,
                gesture_name,
                set_code,
                filename,
                relative_path,
            )

            channel_rows.extend(
                this_channel_rows
            )

            amplitude_rows.extend(
                this_extreme_rows
            )

            discontinuity_rows.extend(
                this_discontinuity_rows
            )

            flat_channel_count = sum(
                r["flat_channel"]
                for r in this_channel_rows
            )

            near_flat_channel_count = sum(
                r["near_flat_channel"]
                for r in this_channel_rows
            )

            highly_repeated_channel_count = sum(
                r["highly_repeated_values"]
                for r in this_channel_rows
            )

            empirical_extreme_channel_count = sum(
                r["empirical_extreme_repeat"]
                for r in this_channel_rows
            )

            discontinuity_channel_count = sum(
                r["discontinuity_review"]
                for r in this_channel_rows
            )

            # ------------------------------------------------
            # Recording-level flat/review rows
            # ------------------------------------------------

            if flat_channel_count > 0:

                flat_rows.append({
                    **base,
                    "flat_channel_count": flat_channel_count,
                    "flat_channels": ";".join(
                        r["channel"]
                        for r in this_channel_rows
                        if r["flat_channel"]
                    ),
                    "note": (
                        "Flat channel diagnostic only. "
                        "No automatic exclusion."
                    ),
                })

            if empirical_extreme_channel_count > 0:

                clipping_rows.append({
                    **base,
                    "empirical_extreme_channel_count":
                        empirical_extreme_channel_count,
                    "channels": ";".join(
                        r["channel"]
                        for r in this_channel_rows
                        if r["empirical_extreme_repeat"]
                    ),
                    "interpretation": (
                        "Empirical repeated minimum/maximum "
                        "value diagnostic. Not labeled ADC clipping."
                    ),
                })

            if discontinuity_channel_count > 0:

                discontinuity_rows.append({
                    **base,
                    "discontinuity_channel_count":
                        discontinuity_channel_count,
                    "channels": ";".join(
                        r["channel"]
                        for r in this_channel_rows
                        if r["discontinuity_review"]
                    ),
                    "max_robust_z": max(
                        (
                            r["max_robust_z"]
                            for r in this_channel_rows
                            if np.isfinite(r["max_robust_z"])
                        ),
                        default=np.nan,
                    ),
                    "total_large_jump_count": int(
                        sum(
                            r["large_jump_count"]
                            for r in this_channel_rows
                        )
                    ),
                    "note": (
                        "Discontinuity diagnostic only. "
                        "No automatic exclusion."
                    ),
                })

            # ------------------------------------------------
            # Missing/nonfinite recording rows
            # ------------------------------------------------

            if nonfinite_count > 0:

                missing_rows.append({
                    **base,
                    "nonfinite_count": nonfinite_count,
                    "status": "FAIL",
                })

            # ------------------------------------------------
            # Representative selection
            # ------------------------------------------------

            if (
                representatives["normal"] is None
                and structural_status == "PASS"
                and flat_channel_count == 0
                and empirical_extreme_channel_count == 0
                and discontinuity_channel_count == 0
            ):
                representatives["normal"] = {
                    **base,
                    "absolute_path": str(path),
                    "X": X.copy(),
                }

            if (
                representatives["flat"] is None
                and flat_channel_count > 0
            ):
                representatives["flat"] = {
                    **base,
                    "absolute_path": str(path),
                    "X": X.copy(),
                    "flat_channels": [
                        r["channel"]
                        for r in this_channel_rows
                        if r["flat_channel"]
                    ],
                }

            if (
                representatives["extreme_repeat"] is None
                and empirical_extreme_channel_count > 0
            ):
                representatives["extreme_repeat"] = {
                    **base,
                    "absolute_path": str(path),
                    "X": X.copy(),
                }

            if (
                representatives["discontinuity"] is None
                and discontinuity_channel_count > 0
            ):
                representatives["discontinuity"] = {
                    **base,
                    "absolute_path": str(path),
                    "X": X.copy(),
                }

        # ----------------------------------------------------
        # Recording-level status
        # ----------------------------------------------------

        review_reasons = []

        if flat_channel_count > 0:
            review_reasons.append("FLAT_CHANNEL")

        if near_flat_channel_count > 0:
            review_reasons.append("NEAR_FLAT_CHANNEL")

        if highly_repeated_channel_count > 0:
            review_reasons.append("HIGHLY_REPEATED_VALUES")

        if empirical_extreme_channel_count > 0:
            review_reasons.append("EMPIRICAL_EXTREME_REPEAT")

        if discontinuity_channel_count > 0:
            review_reasons.append("DISCONTINUITY")

        if structural_status == "FAIL":
            final_qa_status = "FAIL"
        elif review_reasons:
            final_qa_status = "REVIEW"
        else:
            final_qa_status = "PASS"

        recording_status.append({
            **base,
            "structural_status": structural_status,
            "qa_status_before_duplicate_audit": final_qa_status,
            "review_reasons": ";".join(review_reasons),

            # Keep recording-level QA counts in the final status
            # table so downstream summaries do not depend on a
            # separate dataframe merge.
            "flat_channel_count": flat_channel_count,
            "near_flat_channel_count": near_flat_channel_count,
            "highly_repeated_channel_count":
                highly_repeated_channel_count,
            "empirical_extreme_channel_count":
                empirical_extreme_channel_count,
            "discontinuity_channel_count":
                discontinuity_channel_count,
        })

        raw_file_rows.append({
            **base,
            "file_size_bytes": file_size,
            "csv_sha256": csv_hash,
            "row_count": row_count,
            "column_count": len(column_names),
            "eeg_channel_count": actual_channel_count,
            "eeg_columns_detected": (
                ";".join(eeg_columns)
                if eeg_columns
                else ""
            ),
            "expected_sample_count": EXPECTED_SAMPLES,
            "sample_count": actual_sample_count,
            "sample_count_pass": not sample_count_fail,
            "channel_count_pass": not channel_count_fail,
            "nonfinite_count": nonfinite_count,
            "nonfinite_pass": (
                nonfinite_count == 0
                if nonfinite_count >= 0
                else False
            ),
            "schema_pass": schema_pass,
            "numeric_conversion_errors": numeric_conversion_errors,
            "min_signal": min_signal,
            "max_signal": max_signal,
            "mean_signal": mean_signal,
            "median_signal": median_signal,
            "std_signal": std_signal,
            "mad_signal": mad_signal,
            "flat_channel_count": flat_channel_count,
            "near_flat_channel_count": near_flat_channel_count,
            "highly_repeated_channel_count":
                highly_repeated_channel_count,
            "empirical_extreme_channel_count":
                empirical_extreme_channel_count,
            "discontinuity_channel_count":
                discontinuity_channel_count,
            "signal_fingerprint": signal_fingerprint_value,
            "structural_status": structural_status,
            "qa_status_before_duplicate_audit": final_qa_status,
            "review_reasons": ";".join(review_reasons),
        })

        duration_rows.append({
            **base,
            "sample_count": actual_sample_count,
            "protocol_fs_hz": NOMINAL_FS_HZ,
            "nominal_duration_seconds":
                NOMINAL_DURATION_SECONDS,
            "expected_samples": EXPECTED_SAMPLES,
            "sample_count_pass": (
                actual_sample_count == EXPECTED_SAMPLES
            ),
            "timestamp_duration_used": False,
            "sample_index_used_as_clock": False,
        })

    except Exception as exc:

        errors.append({
            **base,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(limit=3),
        })

        raw_file_rows.append({
            **base,
            "file_size_bytes": (
                path.stat().st_size
                if path.exists()
                else ""
            ),
            "csv_sha256": "",
            "row_count": "",
            "column_count": "",
            "eeg_channel_count": "",
            "eeg_columns_detected": "",
            "expected_sample_count": EXPECTED_SAMPLES,
            "sample_count": "",
            "sample_count_pass": False,
            "channel_count_pass": False,
            "nonfinite_count": "",
            "nonfinite_pass": False,
            "schema_pass": False,
            "numeric_conversion_errors": "",
            "min_signal": "",
            "max_signal": "",
            "mean_signal": "",
            "median_signal": "",
            "std_signal": "",
            "mad_signal": "",
            "flat_channel_count": "",
            "near_flat_channel_count": "",
            "highly_repeated_channel_count": "",
            "empirical_extreme_channel_count": "",
            "discontinuity_channel_count": "",
            "signal_fingerprint": "",
            "structural_status": "FAIL",
            "qa_status_before_duplicate_audit": "FAIL",
            "review_reasons": "READ_ERROR",
        })

        recording_status.append({
            **base,
            "structural_status": "FAIL",
            "qa_status_before_duplicate_audit": "FAIL",
            "review_reasons": "READ_ERROR",
            "flat_channel_count": 0,
            "near_flat_channel_count": 0,
            "highly_repeated_channel_count": 0,
            "empirical_extreme_channel_count": 0,
            "discontinuity_channel_count": 0,
        })


# ============================================================
# 13. EXACT DUPLICATE SIGNAL AUDIT
# ============================================================

for fingerprint, rows in fingerprint_groups.items():

    if len(rows) > 1:

        for row in rows:

            duplicate_rows.append({
                "signal_fingerprint": fingerprint,
                "duplicate_group_size": len(rows),
                "subject_id": row["subject_id"],
                "condition": row["condition"],
                "gesture_code": row["gesture_code"],
                "gesture_name": row["gesture_name"],
                "set_code": row["set_code"],
                "logical_key": row["logical_key"],
                "filename": row["filename"],
                "relative_path": row["relative_path"],
                "csv_sha256": row["csv_sha256"],
                "duplicate_definition": (
                    "Exact equality of the complete "
                    "5625×13 EEG array after deterministic "
                    "float64 representation."
                ),
            })


# ============================================================
# 14. FINAL RECORDING STATUS AFTER DUPLICATE AUDIT
# ============================================================

duplicate_logical_keys = {
    row["logical_key"]
    for row in duplicate_rows
}

final_recording_rows = []

for row in recording_status:

    key = row["logical_key"]

    final_status = row[
        "qa_status_before_duplicate_audit"
    ]

    reasons = [
        x for x in row["review_reasons"].split(";")
        if x
    ]

    if key in duplicate_logical_keys:
        if final_status != "FAIL":
            final_status = "REVIEW"
        reasons.append("EXACT_DUPLICATE_EEG")

    final_recording_rows.append({
        **row,
        "final_qa_status": final_status,
        "final_review_reasons": ";".join(
            sorted(set(reasons))
        ),
    })


# ============================================================
# 15. SUBJECT / CONDITION / GESTURE / SET SUMMARIES
# ============================================================

def summarize_group(df, group_columns):

    rows = []

    for keys, group in df.groupby(
        group_columns,
        dropna=False,
    ):

        if not isinstance(keys, tuple):
            keys = (keys,)

        item = dict(
            zip(group_columns, keys)
        )

        item.update({
            "recording_count": len(group),
            "pass_count": int(
                (group["final_qa_status"] == "PASS").sum()
            ),
            "review_count": int(
                (group["final_qa_status"] == "REVIEW").sum()
            ),
            "fail_count": int(
                (group["final_qa_status"] == "FAIL").sum()
            ),
            "flat_recording_count": int(
                (
                    group["flat_channel_count"].fillna(0).astype(float)
                    > 0
                ).sum()
            ),
            "near_flat_recording_count": int(
                (
                    group["near_flat_channel_count"].fillna(0).astype(float)
                    > 0
                ).sum()
            ),
            "extreme_repeat_recording_count": int(
                (
                    group["empirical_extreme_channel_count"].fillna(0).astype(float)
                    > 0
                ).sum()
            ),
            "discontinuity_recording_count": int(
                (
                    group["discontinuity_channel_count"].fillna(0).astype(float)
                    > 0
                ).sum()
            ),
        })

        rows.append(item)

    return rows


raw_df = pd.DataFrame(raw_file_rows)
final_df = pd.DataFrame(final_recording_rows)

subject_summary = summarize_group(
    final_df,
    ["subject_id"],
)

condition_summary = summarize_group(
    final_df,
    ["condition"],
)

gesture_summary = summarize_group(
    final_df,
    ["condition", "gesture_code", "gesture_name"],
)

set_summary = summarize_group(
    final_df,
    ["condition", "set_code"],
)

channel_df = pd.DataFrame(channel_rows)

channel_summary = []

if not channel_df.empty:

    for (condition, channel), group in channel_df.groupby(
        ["condition", "channel"]
    ):

        channel_summary.append({
            "condition": condition,
            "channel": channel,
            "recording_channel_rows": len(group),
            "flat_count": int(
                group["flat_channel"].sum()
            ),
            "near_flat_count": int(
                group["near_flat_channel"].sum()
            ),
            "high_repeat_count": int(
                group["highly_repeated_values"].sum()
            ),
            "empirical_extreme_repeat_count": int(
                group["empirical_extreme_repeat"].sum()
            ),
            "discontinuity_count": int(
                group["discontinuity_review"].sum()
            ),
        })


# ============================================================
# 16. FILE-SIZE QA
# ============================================================

file_size_rows = []

for _, row in raw_df.iterrows():

    file_size_rows.append({
        "condition": row["condition"],
        "subject_id": row["subject_id"],
        "gesture_code": row["gesture_code"],
        "gesture_name": row["gesture_name"],
        "set_code": row["set_code"],
        "filename": row["filename"],
        "relative_path": row["relative_path"],
        "file_size_bytes": row["file_size_bytes"],
    })

file_size_df = pd.DataFrame(file_size_rows)

file_size_outliers = []

if not file_size_df.empty:

    sizes = pd.to_numeric(
        file_size_df["file_size_bytes"],
        errors="coerce",
    )

    q1 = sizes.quantile(0.25)
    q3 = sizes.quantile(0.75)
    iqr = q3 - q1

    if np.isfinite(iqr) and iqr > 0:

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (
            (sizes < lower)
            |
            (sizes > upper)
        )

        for _, row in file_size_df[mask].iterrows():

            file_size_outliers.append({
                **row.to_dict(),
                "lower_iqr_bound": lower,
                "upper_iqr_bound": upper,
            })


# ============================================================
# 17. QA EXCLUSION MANIFEST
# ============================================================
#
# This is intentionally a REVIEW manifest.
# No file is deleted and no subject is automatically excluded.
#
# ============================================================

qa_exclusion_rows = []

for row in final_recording_rows:

    if row["final_qa_status"] != "PASS":

        qa_exclusion_rows.append({
            "condition": row["condition"],
            "subject_id": row["subject_id"],
            "gesture_code": row["gesture_code"],
            "gesture_name": row["gesture_name"],
            "set_code": row["set_code"],
            "logical_key": row["logical_key"],
            "filename": row["filename"],
            "relative_path": row["relative_path"],
            "decision": "REVIEW",
            "reason": row["final_review_reasons"],
            "automatic_exclusion": False,
        })


# ============================================================
# 18. REPRESENTATIVE QA FIGURES
# ============================================================

def save_waveform_figure(item, filename, title):

    if item is None:
        return False

    X = item["X"]

    # Plot one representative EEG channel only.
    channel_index = 0

    y = X[:, channel_index]
    t = np.arange(len(y)) / NOMINAL_FS_HZ

    fig = plt.figure(
        figsize=(12, 4.5)
    )

    plt.plot(
        t,
        y,
        linewidth=0.8,
    )

    plt.xlabel(
        "Sample-sequence time (s; nominal Fs = 125 Hz)"
    )

    plt.ylabel(
        "EEG amplitude"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    fig.savefig(
        figure_dir / filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return True


figure_results = {}

figure_results["normal_recording_waveform.png"] = (
    save_waveform_figure(
        representatives["normal"],
        "normal_recording_waveform.png",
        "Representative normal EEG recording — EEG_ch-01",
    )
)

figure_results["flat_channel_waveform.png"] = (
    save_waveform_figure(
        representatives["flat"],
        "flat_channel_waveform.png",
        "Representative flat-channel QA case — EEG_ch-01",
    )
)

figure_results["extreme_repeat_waveform.png"] = (
    save_waveform_figure(
        representatives["extreme_repeat"],
        "extreme_repeat_waveform.png",
        "Representative empirical extreme-repeat QA case — EEG_ch-01",
    )
)

figure_results["discontinuity_waveform.png"] = (
    save_waveform_figure(
        representatives["discontinuity"],
        "discontinuity_waveform.png",
        "Representative discontinuity QA case — EEG_ch-01",
    )
)


# ============================================================
# 19. QA STATUS COUNTS
# ============================================================

final_status_counts = Counter(
    row["final_qa_status"]
    for row in final_recording_rows
)

structural_status_counts = Counter(
    row["structural_status"]
    for row in final_recording_rows
)


# ============================================================
# 20. WRITE OFFICIAL BLOCK 02 OUTPUTS
# ============================================================

print()
print("=" * 78)
print("WRITING OFFICIAL BLOCK 02 OUTPUTS")
print("=" * 78)

write_csv(
    BLOCK02_DIR / "raw_file_qa.csv",
    raw_file_rows,
    list(raw_file_rows[0].keys())
    if raw_file_rows
    else [],
)

write_csv(
    BLOCK02_DIR / "channel_qa.csv",
    channel_rows,
    list(channel_rows[0].keys())
    if channel_rows
    else [],
)

write_csv(
    BLOCK02_DIR / "amplitude_qa.csv",
    amplitude_rows,
    list(amplitude_rows[0].keys())
    if amplitude_rows
    else [],
)

write_csv(
    BLOCK02_DIR / "duration_qa.csv",
    duration_rows,
    list(duration_rows[0].keys())
    if duration_rows
    else [],
)

write_csv(
    BLOCK02_DIR / "missing_value_qa.csv",
    missing_rows,
    list(missing_rows[0].keys())
    if missing_rows
    else [
        "condition",
        "subject_id",
        "gesture_code",
        "gesture_name",
        "set_code",
        "logical_key",
        "filename",
        "relative_path",
        "nonfinite_count",
        "status",
    ],
)

write_csv(
    BLOCK02_DIR / "flat_channel_qa.csv",
    flat_rows,
    list(flat_rows[0].keys())
    if flat_rows
    else [
        "condition",
        "subject_id",
        "gesture_code",
        "gesture_name",
        "set_code",
        "logical_key",
        "filename",
        "relative_path",
        "flat_channel_count",
        "flat_channels",
        "note",
    ],
)

write_csv(
    BLOCK02_DIR / "clipping_qa.csv",
    clipping_rows,
    list(clipping_rows[0].keys())
    if clipping_rows
    else [
        "condition",
        "subject_id",
        "gesture_code",
        "gesture_name",
        "set_code",
        "logical_key",
        "filename",
        "relative_path",
        "empirical_extreme_channel_count",
        "channels",
        "interpretation",
    ],
)

write_csv(
    BLOCK02_DIR / "discontinuity_qa.csv",
    discontinuity_rows,
    list(discontinuity_rows[0].keys())
    if discontinuity_rows
    else [
        "condition",
        "subject_id",
        "gesture_code",
        "gesture_name",
        "set_code",
        "logical_key",
        "filename",
        "relative_path",
        "discontinuity_channel_count",
        "channels",
        "max_robust_z",
        "total_large_jump_count",
        "note",
    ],
)

write_csv(
    BLOCK02_DIR / "duplicate_signal_qa.csv",
    duplicate_rows,
    list(duplicate_rows[0].keys())
    if duplicate_rows
    else [
        "signal_fingerprint",
        "duplicate_group_size",
        "subject_id",
        "condition",
        "gesture_code",
        "gesture_name",
        "set_code",
        "logical_key",
        "filename",
        "relative_path",
        "csv_sha256",
        "duplicate_definition",
    ],
)

write_csv(
    BLOCK02_DIR / "subject_qa_summary.csv",
    subject_summary,
    list(subject_summary[0].keys())
    if subject_summary
    else [],
)

write_csv(
    BLOCK02_DIR / "condition_qa_summary.csv",
    condition_summary,
    list(condition_summary[0].keys())
    if condition_summary
    else [],
)

write_csv(
    BLOCK02_DIR / "gesture_qa_summary.csv",
    gesture_summary,
    list(gesture_summary[0].keys())
    if gesture_summary
    else [],
)

write_csv(
    BLOCK02_DIR / "set_qa_summary.csv",
    set_summary,
    list(set_summary[0].keys())
    if set_summary
    else [],
)

write_csv(
    BLOCK02_DIR / "channel_qa_summary.csv",
    channel_summary,
    list(channel_summary[0].keys())
    if channel_summary
    else [],
)

write_csv(
    BLOCK02_DIR / "file_size_qa.csv",
    file_size_rows,
    [
        "condition",
        "subject_id",
        "gesture_code",
        "gesture_name",
        "set_code",
        "filename",
        "relative_path",
        "file_size_bytes",
    ],
)

write_csv(
    BLOCK02_DIR / "file_size_outliers.csv",
    file_size_outliers,
    list(file_size_outliers[0].keys())
    if file_size_outliers
    else [
        "condition",
        "subject_id",
        "gesture_code",
        "gesture_name",
        "set_code",
        "filename",
        "relative_path",
        "file_size_bytes",
        "lower_iqr_bound",
        "upper_iqr_bound",
    ],
)

write_csv(
    BLOCK02_DIR / "QA_exclusion_manifest.csv",
    qa_exclusion_rows,
    [
        "condition",
        "subject_id",
        "gesture_code",
        "gesture_name",
        "set_code",
        "logical_key",
        "filename",
        "relative_path",
        "decision",
        "reason",
        "automatic_exclusion",
    ],
)

write_csv(
    BLOCK02_DIR / "csv_read_failures.csv",
    errors,
    list(errors[0].keys())
    if errors
    else [
        "condition",
        "subject_id",
        "gesture_code",
        "gesture_name",
        "set_code",
        "logical_key",
        "filename",
        "relative_path",
        "error_type",
        "error_message",
        "traceback",
    ],
)

write_csv(
    BLOCK02_DIR / "recording_final_qa_status.csv",
    final_recording_rows,
    list(final_recording_rows[0].keys())
    if final_recording_rows
    else [],
)


# ============================================================
# 21. MANIFEST RECONCILIATION OUTPUT
# ============================================================

reconciliation_rows = []

for _, row in inventory.iterrows():

    path = Path(row["absolute_path"])

    reconciliation_rows.append({
        "condition": row["condition"],
        "subject_id": row["subject_id"],
        "gesture_code": row["gesture_code"],
        "gesture_name": row["gesture_name"],
        "set_code": row["set_code"],
        "logical_key": row["logical_key"],
        "relative_path": row["relative_path"],
        "filename": row["filename"],
        "manifest_path_exists": path.exists(),
        "is_scientific_csv": (
            path.suffix.lower() == ".csv"
            and not is_audit_path(
                path,
                ME_RAW if row["condition"] == "ME" else MI_RAW,
            )
        ),
    })

write_csv(
    BLOCK02_DIR / "BLOCK02_manifest_reconciliation.csv",
    reconciliation_rows,
    list(reconciliation_rows[0].keys()),
)


# ============================================================
# 22. CONFIGURATION
# ============================================================

config = {
    "block": BLOCK_ID,
    "block_version": BLOCK_VERSION,

    "protocol_sources": {
        "block00_config": str(BLOCK00_CONFIG),
        "block01_config": str(BLOCK01_CONFIG),
        "block01_recording_inventory": str(BLOCK01_INVENTORY),
    },

    "raw_paths": {
        "base": str(BASE_DIR),
        "me": str(ME_RAW),
        "mi": str(MI_RAW),
    },

    "output_path": str(BLOCK02_DIR),

    "cohort": {
        "subjects": EXPECTED_SUBJECTS,
        "conditions": list(CONDITIONS),
        "gestures": GESTURES,
        "sets": list(SETS),
        "expected_recordings": EXPECTED_RECORDINGS,
    },

    "signal_protocol": {
        "channels": EXPECTED_CHANNELS,
        "channel_count": EXPECTED_CHANNEL_COUNT,
        "nominal_fs_hz": NOMINAL_FS_HZ,
        "expected_samples": EXPECTED_SAMPLES,
        "nominal_duration_seconds": NOMINAL_DURATION_SECONDS,
        "signal_sample_sequence": (
            "Recorded row order."
        ),
    },

    "timestamp_policy": {
        "used_for_sampling_frequency": False,
        "used_for_duration": False,
        "used_for_resampling": False,
        "used_for_rejection": False,
        "timestamp_is_metadata_only": True,
    },

    "sample_index_policy": {
        "used_as_signal_clock": False,
        "used_for_resampling": False,
        "sample_index_is_metadata_only": True,
    },

    "qa_thresholds": {
        "nonfinite_fail_count": NONFINITE_FAIL_COUNT,
        "sample_count_must_equal": EXPECTED_SAMPLES,
        "channel_count_must_equal": EXPECTED_CHANNEL_COUNT,
        "flat_unique_value_count": FLAT_UNIQUE_VALUE_COUNT,
        "near_flat_std_threshold": NEAR_FLAT_STD_THRESHOLD,
        "high_repeat_fraction_threshold":
            HIGH_REPEAT_FRACTION_THRESHOLD,
        "empirical_extreme_repeat_fraction_threshold":
            EXTREME_REPEAT_FRACTION_THRESHOLD,
        "discontinuity_robust_z_threshold":
            DISCONTINUITY_ROBUST_Z_THRESHOLD,
        "discontinuity_review_count":
            DISCONTINUITY_REVIEW_COUNT,
    },

    "duplicate_definition": (
        "Exact equality of complete 5625x13 EEG arrays after "
        "deterministic little-endian float64 representation."
    ),

    "extreme_value_definition": (
        "Repeated empirical minimum/maximum values. "
        "Not labeled ADC clipping because documented ADC limits "
        "are unavailable."
    ),

    "automatic_exclusion": False,

    "raw_data_modified": False,
    "raw_data_deleted": False,

    "processing": {
        "filtering": False,
        "resampling": False,
        "interpolation": False,
        "normalization": False,
        "windowing": False,
        "feature_extraction": False,
        "model_training": False,
        "statistical_testing": False,
    },
}


write_json(
    BLOCK02_DIR / "BLOCK02_config.json",
    config,
)


# ============================================================
# 23. SUMMARY
# ============================================================

summary = {
    "block": BLOCK_ID,
    "block_version": BLOCK_VERSION,
    "created_utc": utc_now(),

    "status": (
        "PASS"
        if (
            len(errors) == 0
            and structural_status_counts.get("FAIL", 0) == 0
        )
        else "REVIEW"
    ),

    "raw_data_modified": False,
    "raw_data_deleted": False,
    "automatic_exclusion": False,

    "recordings": {
        "expected": EXPECTED_RECORDINGS,
        "inspected": len(final_recording_rows),
        "pass": final_status_counts.get("PASS", 0),
        "review": final_status_counts.get("REVIEW", 0),
        "fail": final_status_counts.get("FAIL", 0),
        "read_errors": len(errors),
    },

    "structural_qa": {
        "sample_count_failures": int(
            (
                pd.to_numeric(
                    raw_df["sample_count"],
                    errors="coerce",
                )
                != EXPECTED_SAMPLES
            ).sum()
        ),
        "channel_schema_failures": int(
            (
                raw_df["schema_pass"].astype(str).str.lower()
                != "true"
            ).sum()
        ),
        "nonfinite_recordings": int(
            (
                pd.to_numeric(
                    raw_df["nonfinite_count"],
                    errors="coerce",
                ).fillna(-1)
                > 0
            ).sum()
        ),
    },

    "signal_quality_diagnostics": {
        "flat_channel_recordings": len(flat_rows),
        "near_flat_recordings": int(
            (
                pd.to_numeric(
                    raw_df["near_flat_channel_count"],
                    errors="coerce",
                ).fillna(0)
                > 0
            ).sum()
        ),
        "highly_repeated_value_recordings": int(
            (
                pd.to_numeric(
                    raw_df["highly_repeated_channel_count"],
                    errors="coerce",
                ).fillna(0)
                > 0
            ).sum()
        ),
        "empirical_extreme_repeat_recordings": len(
            clipping_rows
        ),
        "discontinuity_recordings": len(
            discontinuity_rows
        ),
    },

    "duplicate_audit": {
        "unique_signal_fingerprints": len(
            fingerprint_groups
        ),
        "duplicate_fingerprint_groups": sum(
            1
            for rows in fingerprint_groups.values()
            if len(rows) > 1
        ),
        "duplicate_recording_rows": len(
            duplicate_rows
        ),
    },

    "reconciliation": {
        "block01_inventory_rows": len(inventory),
        "current_raw_scientific_csvs": len(raw_paths),
        "manifest_not_on_disk": len(manifest_not_on_disk),
        "raw_not_in_manifest": len(raw_not_in_manifest),
        "duplicate_manifest_paths": len(
            inventory_duplicate_paths
        ),
    },

    "timestamp_policy": {
        "timestamp_qa_used": False,
        "timestamp_used_for_duration": False,
        "timestamp_used_for_sampling_rate": False,
        "timestamp_used_for_resampling": False,
    },

    "sample_index_policy": {
        "sample_index_qa_used": False,
        "sample_index_used_as_clock": False,
    },

    "figures": figure_results,

    "next_block": (
        "BLOCK03_PREPROCESSING"
        if len(errors) == 0
        and structural_status_counts.get("FAIL", 0) == 0
        else "BLOCK02_REVIEW"
    ),

    "important_interpretation": (
        "Signal-quality REVIEW flags are diagnostic and do not "
        "constitute automatic subject or recording exclusion. "
        "Any later exclusion must follow the frozen protocol "
        "and be explicitly documented."
    ),
}


write_json(
    BLOCK02_DIR / "BLOCK02_summary.json",
    summary,
)


# ============================================================
# 24. HUMAN-READABLE SUMMARY
# ============================================================

summary_text = f"""
PAPER 2 — BLOCK 02 FINAL CLEAN
SIGNAL QUALITY ASSURANCE
=========================

STATUS
------
{summary["status"]}


COHORT
------
Expected subjects:
{EXPECTED_SUBJECTS}

Expected recordings:
{EXPECTED_RECORDINGS}

Inspected recordings:
{len(final_recording_rows)}


RECORDING STATUS
----------------
PASS:
{final_status_counts.get("PASS", 0)}

REVIEW:
{final_status_counts.get("REVIEW", 0)}

FAIL:
{final_status_counts.get("FAIL", 0)}

CSV read errors:
{len(errors)}


STRUCTURAL QA
-------------
Sample-count failures:
{summary["structural_qa"]["sample_count_failures"]}

EEG schema failures:
{summary["structural_qa"]["channel_schema_failures"]}

Recordings with nonfinite values:
{summary["structural_qa"]["nonfinite_recordings"]}


SIGNAL-QUALITY DIAGNOSTICS
--------------------------
Flat-channel recordings:
{summary["signal_quality_diagnostics"]["flat_channel_recordings"]}

Near-flat recordings:
{summary["signal_quality_diagnostics"]["near_flat_recordings"]}

Highly repeated-value recordings:
{summary["signal_quality_diagnostics"]["highly_repeated_value_recordings"]}

Empirical extreme-repeat recordings:
{summary["signal_quality_diagnostics"]["empirical_extreme_repeat_recordings"]}

Discontinuity recordings:
{summary["signal_quality_diagnostics"]["discontinuity_recordings"]}


EXACT DUPLICATE EEG AUDIT
-------------------------
Unique EEG fingerprints:
{summary["duplicate_audit"]["unique_signal_fingerprints"]}

Duplicate fingerprint groups:
{summary["duplicate_audit"]["duplicate_fingerprint_groups"]}

Duplicate recording rows:
{summary["duplicate_audit"]["duplicate_recording_rows"]}


RECONCILIATION
--------------
BLOCK 01 inventory rows:
{summary["reconciliation"]["block01_inventory_rows"]}

Current raw scientific CSVs:
{summary["reconciliation"]["current_raw_scientific_csvs"]}

Manifest paths not on disk:
{summary["reconciliation"]["manifest_not_on_disk"]}

Raw CSVs not in BLOCK 01 manifest:
{summary["reconciliation"]["raw_not_in_manifest"]}

Duplicate manifest paths:
{summary["reconciliation"]["duplicate_manifest_paths"]}


TIMESTAMP POLICY
----------------
Timestamp QA:
NOT USED

Timestamp for sampling rate:
NOT USED

Timestamp for duration:
NOT USED

Timestamp for resampling:
NOT USED

Sample Index as signal clock:
NOT USED

Signal sequence:
Recorded row order

Protocol nominal Fs:
{NOMINAL_FS_HZ} Hz

Expected samples:
{EXPECTED_SAMPLES}

Nominal duration:
{NOMINAL_DURATION_SECONDS} s


PROCESSING NOT PERFORMED
------------------------
Filtering:
NO

Resampling:
NO

Interpolation:
NO

Normalization:
NO

Windowing:
NO

Feature extraction:
NO

Model training:
NO

Statistical testing:
NO


EXCLUSION POLICY
----------------
Automatic recording exclusion:
NO

Automatic subject exclusion:
NO

QA REVIEW flags are diagnostic only.


RAW DATA
--------
Modified:
NO

Deleted:
NO


OUTPUT
------
{BLOCK02_DIR}


NEXT
----
{summary["next_block"]}
"""

(BLOCK02_DIR / "BLOCK02_summary.txt").write_text(
    summary_text,
    encoding="utf-8",
)


# ============================================================
# 25. COMPLETION MARKER
# ============================================================

completion = {
    "block": BLOCK_ID,
    "block_version": BLOCK_VERSION,
    "status": summary["status"],

    # A BLOCK 02 completion marker means the QA run itself
    # completed. It does NOT mean all recordings are perfect.
    "complete_marker": True,

    "created_utc": utc_now(),

    "raw_data_scanned": True,
    "raw_data_modified": False,
    "raw_data_deleted": False,

    "automatic_exclusion": False,

    "recordings_expected": EXPECTED_RECORDINGS,
    "recordings_inspected": len(final_recording_rows),

    "pass_recordings": final_status_counts.get("PASS", 0),
    "review_recordings": final_status_counts.get("REVIEW", 0),
    "fail_recordings": final_status_counts.get("FAIL", 0),

    "read_error_count": len(errors),

    "sample_count_failure_count":
        summary["structural_qa"]["sample_count_failures"],

    "schema_failure_count":
        summary["structural_qa"]["channel_schema_failures"],

    "nonfinite_recording_count":
        summary["structural_qa"]["nonfinite_recordings"],

    "flat_channel_recording_count":
        summary["signal_quality_diagnostics"]["flat_channel_recordings"],

    "duplicate_fingerprint_group_count":
        summary["duplicate_audit"]["duplicate_fingerprint_groups"],

    "duplicate_recording_row_count":
        summary["duplicate_audit"]["duplicate_recording_rows"],

    "timestamp_qa_used": False,
    "sample_index_qa_used": False,

    "output_directory": str(BLOCK02_DIR),

    "next_block": summary["next_block"],

    "completion_meaning": (
        "BLOCK 02 completed the predefined signal QA inspection. "
        "PASS means no structural/readability failures were found. "
        "REVIEW flags are diagnostic and do not automatically exclude "
        "recordings or subjects."
    ),
}

write_json(
    BLOCK02_DIR / "BLOCK02_complete.json",
    completion,
)


# ============================================================
# 26. FINAL CONSOLE REPORT
# ============================================================

print()
print("=" * 78)
print("BLOCK 02 — FINAL CLEAN RESULT")
print("=" * 78)

print(
    f"STATUS                         : {summary['status']}"
)

print(
    f"Recordings expected            : {EXPECTED_RECORDINGS:,}"
)

print(
    f"Recordings inspected           : {len(final_recording_rows):,}"
)

print(
    f"PASS recordings                : "
    f"{final_status_counts.get('PASS', 0):,}"
)

print(
    f"REVIEW recordings              : "
    f"{final_status_counts.get('REVIEW', 0):,}"
)

print(
    f"FAIL recordings                : "
    f"{final_status_counts.get('FAIL', 0):,}"
)

print(
    f"CSV read errors                : {len(errors):,}"
)

print(
    f"Sample-count failures          : "
    f"{summary['structural_qa']['sample_count_failures']:,}"
)

print(
    f"EEG schema failures            : "
    f"{summary['structural_qa']['channel_schema_failures']:,}"
)

print(
    f"Nonfinite recordings           : "
    f"{summary['structural_qa']['nonfinite_recordings']:,}"
)

print(
    f"Flat-channel recordings        : "
    f"{summary['signal_quality_diagnostics']['flat_channel_recordings']:,}"
)

print(
    f"Near-flat recordings           : "
    f"{summary['signal_quality_diagnostics']['near_flat_recordings']:,}"
    if "near_flat_recordings" in summary["signal_quality_diagnostics"]
    else
    f"Near-flat recordings           : "
    f"{summary['signal_quality_diagnostics']['near_flat_channel_recordings']:,}"
    if "near_flat_recording_count" in summary["signal_quality_diagnostics"]
    else
    f"Near-flat recordings           : "
    f"{summary['signal_quality_diagnostics']['near_flat_channel_recordings']:,}"
)

print(
    f"Highly repeated recordings    : "
    f"{summary['signal_quality_diagnostics']['highly_repeated_value_recordings']:,}"
)

print(
    f"Empirical extreme-repeat recs  : "
    f"{summary['signal_quality_diagnostics']['empirical_extreme_repeat_recordings']:,}"
)

print(
    f"Discontinuity recordings       : "
    f"{summary['signal_quality_diagnostics']['discontinuity_recordings']:,}"
)

print(
    f"Unique EEG fingerprints        : "
    f"{summary['duplicate_audit']['unique_signal_fingerprints']:,}"
)

print(
    f"Duplicate fingerprint groups   : "
    f"{summary['duplicate_audit']['duplicate_fingerprint_groups']:,}"
)

print(
    f"Duplicate recording rows       : "
    f"{summary['duplicate_audit']['duplicate_recording_rows']:,}"
)

print()
print("TIMESTAMP QA: NOT USED")
print("SAMPLE INDEX QA: NOT USED")
print("RAW DATA MODIFIED: NO")
print("RAW DATA DELETED : NO")
print("AUTOMATIC EXCLUSION: NO")

print()
print("OUTPUT DIRECTORY:")
print(BLOCK02_DIR)

print()

if (
    len(errors) == 0
    and structural_status_counts.get("FAIL", 0) == 0
):

    print("=" * 78)
    print("✓ BLOCK 02 QA RUN COMPLETED")
    print("✓ ALL 1,050 RECORDINGS WERE READ SUCCESSFULLY")
    print("✓ NO STRUCTURAL/READABILITY FAILURES")
    print("✓ SIGNAL-QUALITY REVIEW FLAGS, IF ANY, ARE DIAGNOSTIC ONLY")
    print("✓ NO RAW DATA WERE MODIFIED OR DELETED")
    print("✓ READY FOR MANUAL QA REVIEW BEFORE BLOCK 03")
    print("=" * 78)

else:

    print("=" * 78)
    print("BLOCK 02 = REVIEW")
    print("STRUCTURAL/READABILITY ISSUES WERE DETECTED.")
    print("DO NOT PROCEED TO BLOCK 03 YET.")
    print("Inspect the generated QA tables.")
    print("=" * 78)


# ============================================================
# END OF BLOCK 02
# ============================================================
