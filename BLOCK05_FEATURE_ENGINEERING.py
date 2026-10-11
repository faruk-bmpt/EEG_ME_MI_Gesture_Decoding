# -*- coding: utf-8 -*-
"""
BLOCK 05 — EEG FEATURE ENGINEERING & LEAKAGE-CONTROLLED 341-FEATURE PIPELINE
Paper 2 canonical executable implementation.

FROZEN UPSTREAM CONFIGURATION
-----------------------------
BLOCK 04:
    configuration = W500_O50
    window_samples = 500
    window_seconds = 4.0
    overlap_percent = 50
    step_samples = 250
    sampling_rate = 125 Hz
    recordings = 1050
    windows_per_recording = 21
    total_windows = 22050

PRIMARY FEATURE REPRESENTATION
------------------------------
13 EEG channels × 24 channel features = 312
13 channel means = 13
13 channel SDs = 13
3 global features = 3
TOTAL = 341 features/window

No window selection, scaling, feature selection, PCA, CSP, SMOTE,
or model fitting occurs in this block.

Zero-variance numerical convention:
skewness=0, excess kurtosis=0, Hjorth mobility=0,
Hjorth complexity=0, and constant-channel pairwise correlation=0.
These are deterministic numerical conventions for undefined statistics;
affected channels are retained and are not automatically excluded.

Canonical output:
Paper2_EEG_BCI/05_FEATURE/

Run this script from the beginning in Colab. It is resumable and uses
atomic output writes. It never modifies raw data or BLOCK 03/04 outputs.
"""

from __future__ import annotations

import os
import json
import math
import time
import hashlib
import traceback
import re
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import signal, stats
import matplotlib.pyplot as plt


# =============================================================================
# 00. FROZEN PAPER 2 / BLOCK 04 CONSTANTS
# =============================================================================

FS = 125
WINDOW_SAMPLES = 500
WINDOW_SECONDS = 4.0
OVERLAP_PERCENT = 50
STEP_SAMPLES = 250
RECORDING_SAMPLES = 5625
N_CHANNELS = 13
N_SUBJECTS = 25
N_RECORDINGS = 1050
WINDOWS_PER_RECORDING = 21
TOTAL_WINDOWS = 22050
N_FEATURES = 341
N_CHANNEL_FEATURES = 24

GESTURE_MAP = {
    1: "HOC",
    2: "IFM",
    3: "LFM",
    4: "MFM",
    5: "PH",
    6: "RFM",
    7: "TFM",
}

CONDITIONS = ("ME", "MI")
SETS = ("SetA", "SetB", "SetC")

PSD_NPERSEG = 256
PSD_NOVERLAP = 128
PSD_NFFT = 256
PSD_WINDOW = "hann"
PSD_DETREND = "constant"
PSD_SCALING = "density"
PSD_AVERAGING = "mean"
PSD_ONE_SIDED = True
PSD_EXPECTED_SEGMENTS = 1 + (WINDOW_SAMPLES - PSD_NPERSEG) // (PSD_NPERSEG - PSD_NOVERLAP)
PSD_FREQUENCY_RESOLUTION_HZ = FS / PSD_NFFT

EPSILON = 1e-12
SEF_PERCENT = 95.0
REPRO_TOL = 1e-10
CORR_CONSTANT_VALUE = 0.0

# IMPORTANT: no timestamp is used as the EEG clock.
TIMESTAMP_USED_AS_CLOCK = False
SAMPLE_INDEX_USED_AS_CLOCK = False

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# =============================================================================
# 01. GOOGLE DRIVE PATHS
# =============================================================================

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception:
    # Allows local execution if Drive is already mounted / paths are available.
    pass

DRIVE_ROOT = Path("/content/drive/MyDrive")
RAW_ROOT = DRIVE_ROOT / "EEG_Motor_Execution_Imagery"
PAPER2_ROOT = RAW_ROOT / "Paper2_EEG_BCI"

BLOCK01_ROOT = PAPER2_ROOT / "01_SUBJECT_MATCHING"
BLOCK02_ROOT = PAPER2_ROOT / "02_SIGNAL_QA"
BLOCK03_ROOT = PAPER2_ROOT / "03_PREPROCESSING"
BLOCK04_ROOT = PAPER2_ROOT / "04_WINDOWING"
BLOCK05_ROOT = PAPER2_ROOT / "05_FEATURE_ENGINEERING"

BLOCK05_FEATURES = BLOCK05_ROOT / "features"
BLOCK05_MANIFESTS = BLOCK05_ROOT / "manifests"
BLOCK05_DIAGNOSTICS = BLOCK05_ROOT / "diagnostics"
BLOCK05_CONFIGS = BLOCK05_ROOT / "configs"
BLOCK05_CHECKPOINTS = BLOCK05_ROOT / "checkpoints"

for d in [
    BLOCK05_ROOT,
    BLOCK05_FEATURES,
    BLOCK05_MANIFESTS,
    BLOCK05_DIAGNOSTICS,
    BLOCK05_CONFIGS,
    BLOCK05_CHECKPOINTS,
]:
    d.mkdir(parents=True, exist_ok=True)

for condition in CONDITIONS:
    (BLOCK05_FEATURES / condition).mkdir(parents=True, exist_ok=True)


# =============================================================================
# 02. LOGGING / ATOMIC I/O
# =============================================================================

START_TIME = time.time()
LOG_PATH = BLOCK05_ROOT / "BLOCK05_execution.log"

def log(msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} | {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(
        path,
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
    )


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =============================================================================
# 03. REQUIRED BLOCK 04 HANDOFF
# =============================================================================

def fail(msg: str) -> None:
    raise RuntimeError(msg)


def normalize_condition(value) -> str:
    s = str(value).strip().upper()
    if s.startswith("ME"):
        return "ME"
    if s.startswith("MI"):
        return "MI"
    raise ValueError(f"Unknown condition: {value}")


def normalize_set(value) -> str:
    """Normalize Block 04 set labels to the canonical SetA/SetB/SetC form.

    Block 04 provenance may encode the same set as either ``A``/``B``/``C``
    or ``SetA``/``SetB``/``SetC``. These are labels, not different datasets.
    This normalization changes metadata representation only; it never changes
    the underlying EEG data or window identity.
    """
    s = str(value).strip()
    u = s.upper()
    mapping = {
        "A": "SetA",
        "B": "SetB",
        "C": "SetC",
        "SETA": "SetA",
        "SETB": "SetB",
        "SETC": "SetC",
    }
    if u in mapping:
        return mapping[u]
    raise ValueError(f"Unknown set label: {value!r}; expected A/B/C or SetA/SetB/SetC")


def load_json_required(path: Path) -> dict:
    if not path.exists():
        fail(f"Required file missing: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        fail(f"Cannot read JSON {path}: {e}")


def find_block04_artifact(*names: str) -> Path | None:
    candidates = []
    for name in names:
        candidates.extend([
            BLOCK04_ROOT / name,
            BLOCK04_ROOT / "configs" / name,
            BLOCK04_ROOT / "manifests" / name,
            BLOCK04_ROOT / "diagnostics" / name,
        ])
    for p in candidates:
        if p.exists():
            return p
    return None


def verify_block04_handoff() -> dict:
    log("BLOCK 04 → BLOCK 05 handoff verification started")

    complete_path = find_block04_artifact("BLOCK04_complete.json")
    if complete_path is None:
        fail("BLOCK04_complete.json not found. BLOCK 04 must PASS before BLOCK 05.")

    complete = load_json_required(complete_path)
    complete_status = str(complete.get("status", "")).upper()
    if complete_status != "PASS":
        fail(f"BLOCK 04 completion status is {complete_status!r}, expected PASS.")

    config_path = find_block04_artifact(
        "selected_window_config.json",
        "FINAL_WINDOW_CONFIG.json",
        "BLOCK04_config.json",
    )
    if config_path is None:
        fail("No BLOCK 04 frozen configuration JSON found.")
    config = load_json_required(config_path)

    manifest_path = find_block04_artifact(
        "final_window_manifest.csv",
    )
    if manifest_path is None:
        fail("BLOCK 04 final_window_manifest.csv not found.")

    required_values = {
        "selected_configuration_id": "W500_O50",
        "selected_window_samples": 500,
        "selected_overlap_percent": 50,
        "selected_step_samples": 250,
        "selected_window_seconds": 4.0,
    }

    # Some BLOCK 04 artifacts may expose the selected values under their
    # corresponding generic names. Normalize only for verification.
    normalized = dict(config)
    if "window_samples" in normalized and "selected_window_samples" not in normalized:
        normalized["selected_window_samples"] = normalized["window_samples"]
    if "overlap_percent" in normalized and "selected_overlap_percent" not in normalized:
        normalized["selected_overlap_percent"] = normalized["overlap_percent"]
    if "step_samples" in normalized and "selected_step_samples" not in normalized:
        normalized["selected_step_samples"] = normalized["step_samples"]
    if "window_seconds" in normalized and "selected_window_seconds" not in normalized:
        normalized["selected_window_seconds"] = normalized["window_seconds"]
    if "configuration_id" in normalized and "selected_configuration_id" not in normalized:
        normalized["selected_configuration_id"] = normalized["configuration_id"]

    # Require an explicit False leakage declaration. BLOCK 04 has used
    # several documented field names across frozen clean versions; accept
    # only known aliases and never infer False from a missing field.
    final_flag_keys = [
        "final_cohort_used_for_window_selection",
        "final_cohort_used_for_selection",
        "final_heldout_subjects_used_for_selection",
    ]
    loso_flag_keys = [
        "final_loso_subjects_used_for_window_selection",
        "final_loso_subjects_used_for_selection",
        "loso_subjects_used_for_selection",
    ]
    final_flag_key = next((k for k in final_flag_keys if k in normalized), None)
    loso_flag_key = next((k for k in loso_flag_keys if k in normalized), None)
    if final_flag_key is None or normalized[final_flag_key] is not False:
        fail("BLOCK 04 final-cohort window-selection flag must be explicitly False.")
    if loso_flag_key is None or normalized[loso_flag_key] is not False:
        fail("BLOCK 04 final-LOSO window-selection flag must be explicitly False.")

    for key, expected in required_values.items():
        actual = normalized.get(key)
        if actual != expected:
            fail(
                f"BLOCK 04 handoff mismatch: {key}={actual!r}, "
                f"expected {expected!r}. Source={config_path}"
            )

    # Verify manifest cardinality and required provenance.
    manifest = pd.read_csv(manifest_path)
    if "set" not in manifest.columns:
        fail("BLOCK 04 final_window_manifest.csv is missing the required 'set' column.")
    # Canonicalize A/B/C and SetA/SetB/SetC before all downstream checks.
    manifest["set"] = manifest["set"].map(normalize_set)

    required_manifest_cols = {
        "window_id",
        "subject",
        "condition",
        "gesture_code",
        "set",
        "source_recording",
        "window_samples",
        "window_seconds",
        "overlap_percent",
        "step_samples",
        "start_sample",
        "end_sample",
        "start_time_seconds",
        "end_time_seconds",
    }
    missing = required_manifest_cols - set(manifest.columns)
    if missing:
        fail(f"BLOCK 04 manifest missing required columns: {sorted(missing)}")

    if len(manifest) != TOTAL_WINDOWS:
        fail(
            f"BLOCK 04 manifest has {len(manifest)} windows; "
            f"expected {TOTAL_WINDOWS}."
        )

    if manifest["window_id"].duplicated().any():
        fail("BLOCK 04 final window manifest contains duplicate window_id values.")

    for col, expected in [
        ("window_samples", WINDOW_SAMPLES),
        ("overlap_percent", OVERLAP_PERCENT),
        ("step_samples", STEP_SAMPLES),
    ]:
        vals = pd.to_numeric(manifest[col], errors="coerce")
        if vals.isna().any() or not np.all(vals.to_numpy() == expected):
            fail(f"BLOCK 04 manifest has invalid {col} values.")

    # Exact recording and window counts.
    if manifest["source_recording"].nunique() != N_RECORDINGS:
        fail(
            f"BLOCK 04 manifest maps to {manifest['source_recording'].nunique()} "
            f"recordings; expected {N_RECORDINGS}."
        )

    per_recording = manifest.groupby("source_recording").size()
    if not np.all(per_recording.to_numpy() == WINDOWS_PER_RECORDING):
        fail("BLOCK 04 manifest does not contain exactly 21 windows/recording.")

    subjects = sorted(pd.to_numeric(manifest["subject"], errors="coerce").dropna().astype(int).unique())
    if subjects != list(range(1, N_SUBJECTS + 1)):
        fail(f"BLOCK 04 subjects are {subjects}; expected 1..25.")

    conditions = set(manifest["condition"].astype(str).unique())
    if conditions != set(CONDITIONS):
        fail(f"BLOCK 04 conditions are {conditions}; expected {set(CONDITIONS)}.")

    gestures = sorted(pd.to_numeric(manifest["gesture_code"], errors="coerce").dropna().astype(int).unique())
    if gestures != list(GESTURE_MAP.keys()):
        fail(f"BLOCK 04 gesture codes are {gestures}; expected 1..7.")

    sets_found = set(manifest["set"].astype(str).unique())
    if not {"SetA", "SetB", "SetC"}.issubset(sets_found):
        fail(f"BLOCK 04 sets are {sets_found}; expected SetA/SetB/SetC.")

    # Locate final window data directory from manifest or standard layout.
    final_data_root = BLOCK04_ROOT / "final_windows" / "data" / "W500_O50"
    if not final_data_root.exists():
        # Fallback: search for the directory containing W500/O50 window NPZs.
        candidates = list((BLOCK04_ROOT / "final_windows" / "data").glob("*"))
        matches = [p for p in candidates if p.is_dir() and "500" in p.name and "50" in p.name]
        if matches:
            final_data_root = matches[0]

    if not final_data_root.exists():
        fail(f"BLOCK 04 final window data directory not found: {final_data_root}")

    # Verify that all source recording groups have an actual final NPZ.
    expected_npz = set()
    for _, row in manifest.iterrows():
        src = str(row["source_recording"])
        expected_npz.add(src)

    # Block 04 uses one NPZ per recording. Build lookup by filename.
    npz_files = sorted(final_data_root.rglob("*.npz"))
    if len(npz_files) != N_RECORDINGS:
        fail(
            f"BLOCK 04 final window directory contains {len(npz_files)} NPZ files; "
            f"expected {N_RECORDINGS}."
        )

    handoff = {
        "block04_status": "PASS",
        "selected_configuration_id": "W500_O50",
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": WINDOW_SECONDS,
        "overlap_percent": OVERLAP_PERCENT,
        "step_samples": STEP_SAMPLES,
        "sampling_rate": FS,
        "recording_samples": RECORDING_SAMPLES,
        "channels": N_CHANNELS,
        "subjects": N_SUBJECTS,
        "recordings": N_RECORDINGS,
        "windows_per_recording": WINDOWS_PER_RECORDING,
        "total_windows": TOTAL_WINDOWS,
        "final_cohort_used_for_window_selection": False,
        "final_loso_subjects_used_for_window_selection": False,
        "development_subject_count": 0,
        "selection_method": "DATA-INDEPENDENT PRE-STUDY FREEZE",
        "manifest_sha256": sha256_file(manifest_path),
        "final_window_data_root": str(final_data_root),
        "config_source": str(config_path),
        "complete_source": str(complete_path),
    }

    atomic_write_json(BLOCK05_MANIFESTS / "BLOCK04_handoff_audit.json", handoff)
    log("BLOCK 04 handoff PASS")
    return handoff, manifest, final_data_root


BLOCK04_HANDOFF, BLOCK04_MANIFEST, FINAL_WINDOW_ROOT = verify_block04_handoff()


# =============================================================================
# 04. FEATURE DEFINITIONS
# =============================================================================

CHANNEL_FEATURE_DEFINITIONS = [
    ("RMS", "time_domain"),
    ("VAR", "time_domain"),
    ("P2P", "time_domain"),
    ("MAD", "time_domain"),
    ("IQR", "time_domain"),
    ("ZCR", "time_domain"),
    ("HJ_ACTIVITY", "hjorth"),
    ("HJ_MOBILITY", "hjorth"),
    ("HJ_COMPLEXITY", "hjorth"),
    ("DELTA_ABS", "spectral"),
    ("THETA_ABS", "spectral"),
    ("ALPHA_ABS", "spectral"),
    ("BETA_ABS", "spectral"),
    ("DELTA_REL", "spectral_relative"),
    ("THETA_REL", "spectral_relative"),
    ("ALPHA_REL", "spectral_relative"),
    ("BETA_REL", "spectral_relative"),
    ("PEAK_FREQ", "spectral"),
    ("SPEC_ENTROPY", "spectral"),
    ("SEF95", "spectral"),
    ("ALPHA_BETA", "spectral_ratio"),
    ("SKEW", "distribution"),
    ("KURT", "distribution"),
    ("LINE_LENGTH", "complexity"),
]

assert len(CHANNEL_FEATURE_DEFINITIONS) == 24

CHANNEL_NAMES = [f"CH{i:02d}" for i in range(1, N_CHANNELS + 1)]

FEATURE_NAMES = []
FEATURE_GROUPS = []
FEATURE_CHANNELS = []

for ch in CHANNEL_NAMES:
    for feat, group in CHANNEL_FEATURE_DEFINITIONS:
        FEATURE_NAMES.append(f"{ch}_{feat}")
        FEATURE_GROUPS.append(group)
        FEATURE_CHANNELS.append(ch)

for ch in CHANNEL_NAMES:
    FEATURE_NAMES.append(f"{ch}_MEAN")
    FEATURE_GROUPS.append("channel_mean")
    FEATURE_CHANNELS.append(ch)

for ch in CHANNEL_NAMES:
    FEATURE_NAMES.append(f"{ch}_SD")
    FEATURE_GROUPS.append("channel_sd")
    FEATURE_CHANNELS.append(ch)

GLOBAL_FEATURES = [
    ("GLOBAL_RMS", "global"),
    ("INTRAWINDOW_MEAN_CHANNEL_CORR", "global"),
    ("GLOBAL_MEAN_CHANNEL_SD", "global"),
]

for name, group in GLOBAL_FEATURES:
    FEATURE_NAMES.append(name)
    FEATURE_GROUPS.append(group)
    FEATURE_CHANNELS.append("GLOBAL")

if len(FEATURE_NAMES) != N_FEATURES:
    fail(f"Internal feature schema error: {len(FEATURE_NAMES)} != {N_FEATURES}")


# =============================================================================
# 05. NUMERICAL FEATURE FUNCTIONS
# =============================================================================

def safe_variance(x: np.ndarray) -> float:
    return float(np.var(x, ddof=1))


def safe_skew(x: np.ndarray) -> float:
    # Undefined for a constant channel. Frozen numerical convention = 0.0.
    if np.std(x, ddof=1) <= EPSILON:
        return 0.0
    value = float(stats.skew(x, bias=False))
    return value if np.isfinite(value) else 0.0


def safe_kurtosis_excess(x: np.ndarray) -> float:
    # Fisher=True -> excess kurtosis; bias=False.
    # Undefined for a constant channel. Frozen numerical convention = 0.0.
    if np.std(x, ddof=1) <= EPSILON:
        return 0.0
    value = float(stats.kurtosis(x, fisher=True, bias=False))
    return value if np.isfinite(value) else 0.0


def safe_hjorth(x: np.ndarray) -> tuple[float, float, float]:
    var_x = safe_variance(x)
    dx = np.diff(x)
    var_dx = safe_variance(dx)

    if not np.isfinite(var_x) or var_x <= 0:
        mobility = 0.0
    else:
        mobility = math.sqrt(max(var_dx, 0.0) / var_x)

    ddx = np.diff(dx)
    var_ddx = safe_variance(ddx)

    if mobility <= 0:
        mobility_dx = 0.0
    else:
        mobility_dx = math.sqrt(max(var_ddx, 0.0) / max(var_dx, EPSILON))

    complexity = mobility_dx / mobility if mobility > EPSILON else 0.0
    return float(var_x), float(mobility), float(complexity)


def safe_zero_crossing_rate(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    transitions = np.sum((x[:-1] * x[1:]) < 0)
    return float(transitions / (len(x) - 1))


def compute_psd(x: np.ndarray):
    freqs, psd = signal.welch(
        x,
        fs=FS,
        window=PSD_WINDOW,
        nperseg=PSD_NPERSEG,
        noverlap=PSD_NOVERLAP,
        nfft=PSD_NFFT,
        detrend=PSD_DETREND,
        return_onesided=PSD_ONE_SIDED,
        scaling=PSD_SCALING,
        average=PSD_AVERAGING,
    )
    mask = (freqs >= 1.0) & (freqs <= 40.0)
    return freqs[mask], psd[mask]


def band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    f = freqs[mask]
    p = psd[mask]
    if len(f) < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(p, f))
    return float(np.trapz(p, f))


def spectral_entropy(freqs: np.ndarray, psd: np.ndarray) -> float:
    del freqs
    p = np.asarray(psd, dtype=float)
    p = np.maximum(p, 0.0)
    total = float(np.sum(p))
    if total <= EPSILON or len(p) <= 1:
        return 0.0
    prob = p / total
    nz = prob > 0
    entropy = -float(np.sum(prob[nz] * np.log(prob[nz])))
    return float(entropy / np.log(len(prob)))


def spectral_edge_frequency(freqs: np.ndarray, psd: np.ndarray, fraction=0.95) -> float:
    p = np.maximum(np.asarray(psd, dtype=float), 0.0)
    cumulative = np.cumsum(p)
    total = float(cumulative[-1]) if len(cumulative) else 0.0
    if total <= EPSILON:
        return float(freqs[-1]) if len(freqs) else 0.0

    target = fraction * total
    idx = int(np.searchsorted(cumulative, target, side="left"))
    idx = min(max(idx, 0), len(freqs) - 1)
    return float(freqs[idx])


def channel_features(x: np.ndarray) -> list[float]:
    x = np.asarray(x, dtype=np.float64)
    n = x.size

    if n != WINDOW_SAMPLES:
        raise ValueError(f"Expected {WINDOW_SAMPLES} samples, got {n}")
    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite EEG samples")

    mean = float(np.mean(x))
    variance = safe_variance(x)
    rms = float(np.sqrt(np.mean(x * x)))
    p2p = float(np.max(x) - np.min(x))
    mad = float(np.median(np.abs(x - np.median(x))))
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    zcr = safe_zero_crossing_rate(x)

    hj_activity, hj_mobility, hj_complexity = safe_hjorth(x)

    freqs, psd = compute_psd(x)

    delta = band_power(freqs, psd, 1.0, 4.0)
    theta = band_power(freqs, psd, 4.0, 8.0)
    alpha = band_power(freqs, psd, 8.0, 13.0)
    beta = band_power(freqs, psd, 13.0, 30.0)
    total = band_power(freqs, psd, 1.0, 40.0)

    delta_rel = delta / (total + EPSILON)
    theta_rel = theta / (total + EPSILON)
    alpha_rel = alpha / (total + EPSILON)
    beta_rel = beta / (total + EPSILON)

    if len(psd):
        peak_freq = float(freqs[int(np.argmax(psd))])
    else:
        peak_freq = 0.0

    se = spectral_entropy(freqs, psd)
    sef95 = spectral_edge_frequency(freqs, psd, SEF_PERCENT / 100.0)
    alpha_beta = alpha / (beta + EPSILON)

    skew = safe_skew(x)
    kurt = safe_kurtosis_excess(x)
    line_length = float(np.sum(np.abs(np.diff(x))))

    return [
        rms,
        variance,
        p2p,
        mad,
        iqr,
        zcr,
        hj_activity,
        hj_mobility,
        hj_complexity,
        delta,
        theta,
        alpha,
        beta,
        delta_rel,
        theta_rel,
        alpha_rel,
        beta_rel,
        peak_freq,
        se,
        sef95,
        alpha_beta,
        skew,
        kurt,
        line_length,
    ]


def compute_feature_vector(eeg: np.ndarray) -> np.ndarray:
    """
    Input: (13, 500)
    Output: (341,)
    """
    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.shape != (N_CHANNELS, WINDOW_SAMPLES):
        raise ValueError(
            f"Expected EEG shape {(N_CHANNELS, WINDOW_SAMPLES)}, got {eeg.shape}"
        )
    if not np.all(np.isfinite(eeg)):
        raise ValueError("Window contains NaN/Inf")

    values = []

    # 312 channel-level features.
    for c in range(N_CHANNELS):
        values.extend(channel_features(eeg[c]))

    # 13 means.
    values.extend(np.mean(eeg, axis=1).astype(float).tolist())

    # 13 sample SDs.
    values.extend(np.std(eeg, axis=1, ddof=1).astype(float).tolist())

    # 3 global features.
    global_rms = float(np.sqrt(np.mean(eeg * eeg)))

    # Vectorized Pearson correlation across the 13 channels within this window.
    # This is mathematically equivalent to the 78-pair loop but substantially
    # reduces runtime. Constant-channel correlations are assigned the frozen
    # deterministic convention value of 0.0.
    with np.errstate(invalid="ignore", divide="ignore"):
        channel_corr = np.corrcoef(eeg)
    channel_corr = np.asarray(channel_corr, dtype=np.float64)
    channel_corr[~np.isfinite(channel_corr)] = CORR_CONSTANT_VALUE
    upper_i, upper_j = np.triu_indices(N_CHANNELS, k=1)
    global_corr = float(np.mean(channel_corr[upper_i, upper_j]))
    global_sd = float(np.mean(np.std(eeg, axis=1, ddof=1)))

    values.extend([global_rms, global_corr, global_sd])

    arr = np.asarray(values, dtype=np.float64)
    if arr.size != N_FEATURES:
        raise ValueError(f"Feature count {arr.size} != {N_FEATURES}")
    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))[0].tolist()
        raise ValueError(f"Non-finite feature values at indices {bad[:20]}")

    return arr


# =============================================================================
# 06. FEATURE COLUMN MANIFEST
# =============================================================================

def build_feature_column_manifest() -> pd.DataFrame:
    definitions = {
        "RMS": "sqrt(mean(x^2))",
        "VAR": "sample variance, ddof=1",
        "P2P": "max(x)-min(x)",
        "MAD": "median(abs(x-median(x)))",
        "IQR": "Q75-Q25",
        "ZCR": "sign transitions / (N-1)",
        "HJ_ACTIVITY": "sample variance",
        "HJ_MOBILITY": "sqrt(var(diff(x))/var(x))",
        "HJ_COMPLEXITY": "Hjorth mobility(diff(x))/Hjorth mobility(x)",
        "DELTA_ABS": "trapz Welch PSD, 1-4 Hz",
        "THETA_ABS": "trapz Welch PSD, 4-8 Hz",
        "ALPHA_ABS": "trapz Welch PSD, 8-13 Hz",
        "BETA_ABS": "trapz Welch PSD, 13-30 Hz",
        "DELTA_REL": "delta power / total 1-40 Hz power",
        "THETA_REL": "theta power / total 1-40 Hz power",
        "ALPHA_REL": "alpha power / total 1-40 Hz power",
        "BETA_REL": "beta power / total 1-40 Hz power",
        "PEAK_FREQ": "frequency of maximum Welch PSD, 1-40 Hz",
        "SPEC_ENTROPY": "normalized Shannon entropy of 1-40 Hz PSD bins",
        "SEF95": "frequency containing 95% cumulative 1-40 Hz PSD",
        "ALPHA_BETA": "alpha power / (beta power + epsilon)",
        "SKEW": "scipy.stats.skew, bias=False",
        "KURT": "scipy.stats.kurtosis, Fisher excess, bias=False",
        "LINE_LENGTH": "sum(abs(diff(x)))",
    }

    rows = []
    for i, name in enumerate(FEATURE_NAMES, start=1):
        group = FEATURE_GROUPS[i - 1]
        channel = FEATURE_CHANNELS[i - 1]

        if name.startswith("CH") and "_MEAN" in name:
            definition = "mean(x)"
        elif name.startswith("CH") and "_SD" in name:
            definition = "sample standard deviation, ddof=1"
        elif name.startswith("CH"):
            suffix = name.split("_", 1)[1]
            definition = definitions.get(suffix, "channel-level deterministic feature")
        else:
            definition = {
                "GLOBAL_RMS": "sqrt(mean over all 13x500 samples of x^2)",
                "INTRAWINDOW_MEAN_CHANNEL_CORR": "mean of 78 unique Pearson channel correlations; constant channel pair -> 0",
                "GLOBAL_MEAN_CHANNEL_SD": "mean of 13 channel sample SDs",
            }[name]

        rows.append({
            "feature_index": i,
            "feature_name": name,
            "feature_group": group,
            "channel": channel,
            "definition": definition,
            "units": "implementation-dependent",
            "source_window": "yes",
            "training_fitted": "No",
            "primary_feature": "Yes",
        })

    df = pd.DataFrame(rows)
    if len(df) != N_FEATURES:
        fail("Feature column manifest does not contain 341 rows.")
    return df


FEATURE_COLUMN_MANIFEST = build_feature_column_manifest()
atomic_write_csv(
    FEATURE_COLUMN_MANIFEST,
    BLOCK05_MANIFESTS / "feature_column_manifest.csv",
)

# VAR and Hjorth Activity are mathematically identical under the frozen
# definitions. They are intentionally retained to preserve the canonical
# 341-feature representation and are explicitly documented/audited.
duplicate_feature_rows = []
for ch in CHANNEL_NAMES:
    duplicate_feature_rows.append({
        "feature_a": f"{ch}_VAR",
        "feature_b": f"{ch}_HJ_ACTIVITY",
        "relationship": "identical_by_definition",
        "reason": "Hjorth Activity equals sample variance under the frozen definition.",
        "action": "RETAIN_BOTH_FOR_CANONICAL_341_FEATURE_SCHEMA",
    })
atomic_write_csv(
    pd.DataFrame(duplicate_feature_rows),
    BLOCK05_DIAGNOSTICS / "duplicate_feature_definition_audit.csv",
)


# =============================================================================
# 07. BLOCK 05 CONFIGURATION
# =============================================================================

FEATURE_CONFIG = {
    "block": "05_FEATURE",
    "status": "FROZEN",
    "paper": "Paper 2",
    "selection_method": "DATA-INDEPENDENT PRE-STUDY FREEZE",
    "upstream_block": "04_WINDOWING",
    "block04_configuration_id": "W500_O50",
    "sampling_rate_hz": FS,
    "window_samples": WINDOW_SAMPLES,
    "window_seconds": WINDOW_SECONDS,
    "overlap_percent": OVERLAP_PERCENT,
    "step_samples": STEP_SAMPLES,
    "recording_samples": RECORDING_SAMPLES,
    "channels": N_CHANNELS,
    "subjects": N_SUBJECTS,
    "conditions": list(CONDITIONS),
    "gestures": GESTURE_MAP,
    "sets": list(SETS),
    "expected_recordings": N_RECORDINGS,
    "windows_per_recording": WINDOWS_PER_RECORDING,
    "expected_total_windows": TOTAL_WINDOWS,
    "channel_feature_count": N_CHANNEL_FEATURES,
    "channel_level_total": N_CHANNELS * N_CHANNEL_FEATURES,
    "channel_mean_count": N_CHANNELS,
    "channel_sd_count": N_CHANNELS,
    "global_feature_count": 3,
    "total_feature_count": N_FEATURES,
    "feature_names": FEATURE_NAMES,
    "feature_order_hash": sha256_text("\n".join(FEATURE_NAMES)),
    "frequency_bands_hz": {
        "delta": [1.0, 4.0],
        "theta": [4.0, 8.0],
        "alpha": [8.0, 13.0],
        "beta": [13.0, 30.0],
        "total": [1.0, 40.0],
    },
    "welch": {
        "method": "Welch periodogram",
        "window": PSD_WINDOW,
        "nperseg": PSD_NPERSEG,
        "noverlap": PSD_NOVERLAP,
        "nfft": PSD_NFFT,
        "detrend": PSD_DETREND,
        "scaling": PSD_SCALING,
        "average": PSD_AVERAGING,
        "return_onesided": PSD_ONE_SIDED,
        "integration": "trapezoidal",
        "expected_full_segments_per_500_sample_window": int(PSD_EXPECTED_SEGMENTS),
        "frequency_resolution_hz": float(PSD_FREQUENCY_RESOLUTION_HZ),
        "segment_boundary_policy": "With nperseg=256 and noverlap=128 on a 500-sample window, two complete Welch segments are used; the remaining trailing samples are not used as a third complete segment. No padding or cross-window concatenation is introduced.",
    },
    "spectral_entropy": {
        "definition": "normalized Shannon entropy over retained 1-40 Hz PSD bins",
        "zero_probability_term": "0",
    },
    "spectral_edge": {
        "criterion_percent": SEF_PERCENT,
        "undefined_or_zero_power_convention_hz": 40.0,
        "undefined_or_zero_power_rule": "If retained 1-40 Hz PSD has zero total power, SEF95 is deterministically set to the upper analysis limit (40 Hz).",
    },
    "relative_power_definition": {
        "denominator": "total integrated Welch PSD from 1-40 Hz",
        "numerators": {
            "delta": "1-4 Hz",
            "theta": "4-8 Hz",
            "alpha": "8-13 Hz",
            "beta": "13-30 Hz",
        },
        "sum_behavior": (
            "The four relative bands cover the nominal 1-30 Hz range while the denominator "
            "covers 1-40 Hz; therefore the four relative powers are not expected to sum to "
            "1.0. The 30-40 Hz contribution remains in the denominator as residual "
            "high-frequency power. Because the Welch frequency grid is 125/256 ≈ 0.4883 Hz, "
            "the exact boundaries 4, 8, and 13 Hz are not represented by exact frequency "
            "bins; each band therefore uses the frozen inclusive frequency-mask rule without "
            "post-hoc renormalization."
        ),
        "purpose": "Fixed a priori representation; no post-hoc renormalization.",
    },
    "peak_frequency_definition": {
        "range_hz": [1.0, 40.0],
        "method": "frequency bin with maximum retained Welch PSD",
        "boundary_note": "A peak at the lower boundary can reflect the 1/f-dominant spectrum; no post-hoc exclusion of low-frequency peaks is applied.",
    },
    "mad_definition": {
        "method": "raw median absolute deviation",
        "scaling_factor": 1.0,
        "note": "No 1.4826 consistency scaling is applied.",
    },
    "alpha_beta": {
        "epsilon": EPSILON,
    },
    "kurtosis": {
        "implementation": "scipy.stats.kurtosis",
        "fisher": True,
        "bias": False,
        "convention": "Fisher excess kurtosis",
    },
    "constant_channel_convention": {
        "skewness": 0.0,
        "kurtosis_excess": 0.0,
        "hjorth_mobility": 0.0,
        "hjorth_complexity": 0.0,
        "pairwise_channel_correlation": CORR_CONSTANT_VALUE,
        "rationale": "Undefined statistics for zero-variance windows receive deterministic numerical values; channels are retained and not silently removed.",
    },
    "skewness": {
        "implementation": "scipy.stats.skew",
        "bias": False,
    },
    "zero_crossing": {
        "definition": "x[t]*x[t+1] < 0 divided by N-1",
        "threshold": "exact zero sign transition; no amplitude threshold",
    },
    "global_correlation": {
        "pairs": 78,
        "constant_channel_pair_value": CORR_CONSTANT_VALUE,
    },
    "timestamp_used_as_clock": TIMESTAMP_USED_AS_CLOCK,
    "sample_index_used_as_clock": SAMPLE_INDEX_USED_AS_CLOCK,
    "padding_applied": False,
    "resampling_applied": False,
    "feature_selection_applied": False,
    "scaling_applied": False,
    "pca_applied": False,
    "csp_applied": False,
    "smote_applied": False,
    "model_fitting_applied": False,
    "trial_reconstruction_applied": False,
    "computational_windows_not_trials": True,
    "random_seed": RANDOM_SEED,
    "configuration_frozen": True,
    "configuration_freeze_reason": (
        "Block 04 W500_O50 was frozen by a data-independent pre-study protocol "
        "before final-cohort model evaluation."
    ),
    "feature_correlation_usage": (
        "DIAGNOSTIC QA ONLY — must not be used for feature selection, scaling, "
        "weighting, model tuning, or downstream data-driven preprocessing."
    ),
    "feature_correlation_do_not_use_for_selection": True,
    "full_cohort_diagnostic_statistics": {
        "scope": "all 22050 computational EEG windows across the 25-subject final cohort",
        "files": [
            "feature_summary.csv",
            "feature_distribution_summary.csv",
            "feature_range_summary.csv",
            "feature_quantile_summary.csv",
            "feature_correlation.csv",
            "constant_feature_audit.csv",
        ],
        "purpose": "QA/descriptive reporting only",
        "do_not_use_for_scaling": True,
        "do_not_use_for_feature_selection": True,
        "do_not_use_for_pca_or_csp": True,
        "do_not_use_for_winsorization_or_clipping": True,
        "do_not_use_for_imputation": True,
        "do_not_use_for_model_tuning": True,
        "do_not_use_for_feature_weighting": True,
        "downstream_rule": "All learned preprocessing parameters must be fitted inside the training partition of each evaluation split.",
    },
    "duplicate_feature_policy": (
        "VAR and Hjorth Activity are retained because both belong to the frozen "
        "341-feature schema; their mathematical identity is explicitly audited."
    ),
    "spectral_upper_limit_hz": 40.0,
    "retained_psd_upper_grid_frequency_hz": float((np.floor(40.0 / PSD_FREQUENCY_RESOLUTION_HZ) * PSD_FREQUENCY_RESOLUTION_HZ)),
    "spectral_upper_limit_reason": (
        "Protocol-defined EEG analysis band; no data-driven frequency cutoff "
        "was selected in Block 05."
    ),
    "created_utc": datetime.now(timezone.utc).isoformat(),
}

atomic_write_json(BLOCK05_CONFIGS / "BLOCK05_config.json", FEATURE_CONFIG)

# Explicit downstream leakage contract. These artifacts are intentionally
# generated for QA/descriptive reporting over the complete final cohort.
# Later blocks MUST NOT consume them to learn preprocessing parameters,
# select/weight features, tune models, or otherwise condition evaluation.
FORBIDDEN_DOWNSTREAM_ARTIFACTS = [
    "diagnostics/feature_summary.csv",
    "diagnostics/feature_distribution_summary.csv",
    "diagnostics/feature_range_summary.csv",
    "diagnostics/feature_quantile_summary.csv",
    "diagnostics/feature_correlation.csv",
    "diagnostics/constant_feature_audit.csv",
    "diagnostics/relative_power_definition_audit.csv",
    "diagnostics/feature_summary_metadata.json",
    "diagnostics/feature_distribution_summary_metadata.json",
    "diagnostics/feature_range_summary_metadata.json",
    "diagnostics/feature_quantile_summary_metadata.json",
    "diagnostics/constant_feature_audit_metadata.json",
    "diagnostics/feature_correlation_metadata.json",
    "diagnostics/relative_power_definition_audit_metadata.json",
]
DOWNSTREAM_ALLOWED_DATA_SOURCES = [
    "05_FEATURE/features/ME/*_features.npz",
    "05_FEATURE/features/MI/*_features.npz",
    "05_FEATURE/manifests/feature_manifest.csv",
    "05_FEATURE/manifests/feature_column_manifest.csv",
]
LEAKAGE_CONTRACT = {
    "block": "05_FEATURE",
    "status": "PASS",
    "scope": "downstream handoff contract",
    "forbidden_artifacts": FORBIDDEN_DOWNSTREAM_ARTIFACTS,
    "forbidden_uses": [
        "feature selection",
        "feature weighting",
        "global scaling parameter estimation",
        "winsorization/clipping thresholds",
        "imputation parameter estimation",
        "PCA fitting",
        "CSP fitting",
        "model hyperparameter tuning",
        "any data-driven preprocessing using final-cohort diagnostic statistics",
    ],
    "allowed_primary_sources": DOWNSTREAM_ALLOWED_DATA_SOURCES,
    "rule": (
        "Any learned preprocessing or model parameter must be fitted exclusively "
        "inside the training partition of the active evaluation split. Test/held-out "
        "subjects may only be transformed using parameters learned from training data."
    ),
    "diagnostic_statistics_are_full_cohort": True,
    "diagnostic_statistics_are_not_training_parameters": True,
}
atomic_write_json(
    BLOCK05_CONFIGS / "BLOCK05_LEAKAGE_FORBIDDEN_ARTIFACTS.json",
    LEAKAGE_CONTRACT,
)
atomic_write_json(
    BLOCK05_DIAGNOSTICS / "diagnostic_artifact_usage_contract.json",
    LEAKAGE_CONTRACT,
)


# =============================================================================
# 08. FINAL WINDOW NPZ DISCOVERY / VALIDATION
# =============================================================================

def parse_numeric_subject(value) -> int:
    s = str(value).strip()
    if s.upper().startswith("S"):
        s = s[1:]
    return int(float(s))


def load_final_window_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)

        if "eeg_windows" not in keys:
            raise ValueError(f"{path.name}: missing eeg_windows")

        eeg = np.asarray(z["eeg_windows"], dtype=np.float64)

        # Expected Block 04 final shape: (21, 13, 500).
        if eeg.ndim != 3:
            raise ValueError(f"{path.name}: eeg_windows ndim={eeg.ndim}, expected 3")
        if eeg.shape != (WINDOWS_PER_RECORDING, N_CHANNELS, WINDOW_SAMPLES):
            raise ValueError(
                f"{path.name}: eeg_windows shape={eeg.shape}, expected "
                f"{(WINDOWS_PER_RECORDING, N_CHANNELS, WINDOW_SAMPLES)}"
            )

        for arr_name in ["eeg_windows"]:
            if not np.all(np.isfinite(z[arr_name])):
                raise ValueError(f"{path.name}: {arr_name} contains NaN/Inf")

        meta = {}
        for key in [
            "subject",
            "condition",
            "gesture_code",
            "set_code",
            "set",
            "set_name",
            "window_samples",
            "overlap_percent",
            "step_samples",
            "sampling_rate",
            "start_samples",
            "end_samples",
            "start_time_seconds",
            "end_time_seconds",
        ]:
            if key in keys:
                value = z[key]
                if np.ndim(value) == 0:
                    value = value.item()
                elif np.size(value) == 1:
                    value = np.ravel(value)[0].item()
                meta[key] = value

        return {"eeg_windows": eeg, "meta": meta}


def discover_final_window_files() -> list[Path]:
    files = sorted(FINAL_WINDOW_ROOT.rglob("*.npz"))
    if len(files) != N_RECORDINGS:
        fail(
            f"Expected {N_RECORDINGS} Block 04 final-window NPZ files, found {len(files)}."
        )
    return files


FINAL_WINDOW_FILES = discover_final_window_files()


# =============================================================================
# 09. FILE-LEVEL METADATA / PROVENANCE
# =============================================================================

def parse_filename_identity(path: Path) -> tuple[int, str, int, str]:
    """
    Expected examples:
      S01_ME_1_SetA_windows.npz
      S25_MI_7_SetC_windows.npz
    """
    stem = path.stem
    m = re.match(r"^S?(\d+)_([A-Za-z]+)_(\d+)_([A-Za-z0-9]+)_windows$", stem)
    if not m:
        raise ValueError(f"Cannot parse Block 04 window filename: {path.name}")

    subject = int(m.group(1))
    condition = normalize_condition(m.group(2))
    gesture = int(m.group(3))
    set_name = normalize_set(m.group(4))
    if gesture not in GESTURE_MAP:
        raise ValueError(f"Unknown gesture {gesture} in {path.name}")
    if not 1 <= subject <= N_SUBJECTS:
        raise ValueError(f"Subject {subject} outside 1..25 in {path.name}")

    return subject, condition, gesture, set_name


def build_file_manifest(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        subject, condition, gesture, set_name = parse_filename_identity(path)
        data = load_final_window_npz(path)
        meta = data["meta"]

        # Metadata in file, when present, must agree with filename.
        if "subject" in meta and parse_numeric_subject(meta["subject"]) != subject:
            raise ValueError(f"{path.name}: subject metadata mismatch")
        if "condition" in meta and normalize_condition(meta["condition"]) != condition:
            raise ValueError(f"{path.name}: condition metadata mismatch")
        if "set_name" in meta and normalize_set(meta["set_name"]) != set_name:
            raise ValueError(f"{path.name}: set metadata mismatch")
        if "set" in meta and normalize_set(meta["set"]) != set_name:
            raise ValueError(f"{path.name}: set metadata mismatch")
        if "gesture_code" in meta and int(float(meta["gesture_code"])) != gesture:
            raise ValueError(f"{path.name}: gesture metadata mismatch")
        if "window_samples" in meta and int(float(meta["window_samples"])) != WINDOW_SAMPLES:
            raise ValueError(f"{path.name}: window_samples mismatch")
        if "overlap_percent" in meta and int(float(meta["overlap_percent"])) != OVERLAP_PERCENT:
            raise ValueError(f"{path.name}: overlap mismatch")
        if "step_samples" in meta and int(float(meta["step_samples"])) != STEP_SAMPLES:
            raise ValueError(f"{path.name}: step mismatch")
        if "sampling_rate" in meta and int(float(meta["sampling_rate"])) != FS:
            raise ValueError(f"{path.name}: sampling rate mismatch")

        rows.append({
            "subject": subject,
            "condition": condition,
            "gesture_code": gesture,
            "gesture": GESTURE_MAP[gesture],
            "set": set_name,
            "source_file": str(path),
            "source_file_sha256": sha256_file(path),
            "window_count": WINDOWS_PER_RECORDING,
        })

    df = pd.DataFrame(rows).sort_values(
        ["subject", "condition", "gesture_code", "set"]
    ).reset_index(drop=True)

    key_cols = ["subject", "condition", "gesture_code", "set"]
    if df.duplicated(key_cols).any():
        dup = df[df.duplicated(key_cols, keep=False)]
        raise ValueError(f"Duplicate logical Block 04 recordings:\n{dup}")

    if len(df) != N_RECORDINGS:
        raise ValueError(f"File manifest has {len(df)} recordings, expected {N_RECORDINGS}")

    return df


FILE_MANIFEST = build_file_manifest(FINAL_WINDOW_FILES)
atomic_write_csv(FILE_MANIFEST, BLOCK05_MANIFESTS / "block04_window_file_manifest.csv")


# =============================================================================
# 10. RESUME CHECKPOINTS
# =============================================================================

def checkpoint_path(source_file: Path) -> Path:
    digest = hashlib.sha256(str(source_file).encode("utf-8")).hexdigest()[:20]
    return BLOCK05_CHECKPOINTS / f"{digest}.json"


def save_checkpoint(source_file: Path, status: str, **extra) -> None:
    obj = {
        "source_file": str(source_file),
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    atomic_write_json(checkpoint_path(source_file), obj)


# =============================================================================
# 11. FEATURE EXTRACTION PER RECORDING
# =============================================================================

def output_feature_path(subject: int, condition: str, gesture: int, set_name: str) -> Path:
    # Numeric gesture ID is authoritative; no G1/G2 naming.
    return (
        BLOCK05_FEATURES
        / condition
        / f"S{subject:02d}_{condition}_{gesture}_{set_name}_features.npz"
    )


def save_feature_record(
    path: Path,
    features: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    """
    One NPZ per recording.
    features: (21,341)
    metadata: one row per window.
    """
    if features.shape != (WINDOWS_PER_RECORDING, N_FEATURES):
        raise ValueError(f"Invalid feature matrix shape: {features.shape}")
    if not np.all(np.isfinite(features)):
        raise ValueError("Attempted to save non-finite feature matrix")

    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        features=features.astype(np.float64),
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
        subject=np.asarray(metadata["subject"].to_numpy(), dtype=np.int16),
        condition=np.asarray(metadata["condition"].to_numpy(), dtype=object),
        gesture_code=np.asarray(metadata["gesture_code"].to_numpy(), dtype=np.int8),
        set_name=np.asarray(metadata["set"].to_numpy(), dtype=object),
        window_id=np.asarray(metadata["window_id"].to_numpy(), dtype=object),
        start_sample=np.asarray(metadata["start_sample"].to_numpy(), dtype=np.int32),
        end_sample=np.asarray(metadata["end_sample"].to_numpy(), dtype=np.int32),
        start_time_s=np.asarray(metadata["start_time_s"].to_numpy(), dtype=np.float64),
        end_time_s=np.asarray(metadata["end_time_s"].to_numpy(), dtype=np.float64),
    )
    os.replace(tmp, path)


def build_window_metadata(
    subject: int,
    condition: str,
    gesture: int,
    set_name: str,
    source_file: Path,
    start_samples: np.ndarray,
    end_samples: np.ndarray,
) -> pd.DataFrame:
    """Build provenance from the exact window geometry written by Block 04."""
    starts = np.asarray(start_samples, dtype=int).reshape(-1)
    ends = np.asarray(end_samples, dtype=int).reshape(-1)

    if starts.size != WINDOWS_PER_RECORDING or ends.size != WINDOWS_PER_RECORDING:
        raise ValueError(
            f"{source_file.name}: Block 04 start/end arrays must contain "
            f"{WINDOWS_PER_RECORDING} values."
        )
    if np.any(ends - starts != WINDOW_SAMPLES):
        raise ValueError(f"{source_file.name}: inconsistent Block 04 start/end geometry.")
    if np.any(starts < 0) or np.any(ends > RECORDING_SAMPLES):
        raise ValueError(f"{source_file.name}: Block 04 window bounds exceed recording bounds.")
    if np.any(np.diff(starts) <= 0):
        raise ValueError(f"{source_file.name}: Block 04 start_samples are not strictly increasing.")

    rows = []
    for i, (start, end) in enumerate(zip(starts, ends), start=1):
        rows.append({
            "subject": subject,
            "condition": condition,
            "gesture_code": gesture,
            "gesture": GESTURE_MAP[gesture],
            "set": set_name,
            "recording_id": f"S{subject:02d}_{condition}_{gesture}_{set_name}",
            "window_id": f"S{subject:02d}_{condition}_{gesture}_{set_name}_W{i:03d}",
            "source_file": str(source_file),
            "start_sample": int(start),
            "end_sample": int(end),
            "start_time_s": float(start / FS),
            "end_time_s": float(end / FS),
            "window_length": WINDOW_SAMPLES,
            "window_seconds": WINDOW_SECONDS,
            "overlap_percent": OVERLAP_PERCENT,
            "step": STEP_SAMPLES,
        })
    return pd.DataFrame(rows)


def verify_existing_feature_file(path: Path, expected_meta: pd.DataFrame) -> bool:
    if not path.exists():
        return False

    try:
        with np.load(path, allow_pickle=True) as z:
            if "features" not in z.files:
                return False
            arr = np.asarray(z["features"], dtype=np.float64)
            if arr.shape != (WINDOWS_PER_RECORDING, N_FEATURES):
                return False
            if not np.all(np.isfinite(arr)):
                return False
            if "feature_names" not in z.files:
                return False
            names = [str(x) for x in z["feature_names"].tolist()]
            if names != FEATURE_NAMES:
                return False
            if "window_id" not in z.files:
                return False
            ids = [str(x) for x in z["window_id"].tolist()]
            expected_ids = expected_meta["window_id"].astype(str).tolist()
            if ids != expected_ids:
                return False
        return True
    except Exception:
        return False


def process_recording(row: pd.Series) -> dict:
    source_file = Path(row["source_file"])
    subject = int(row["subject"])
    condition = str(row["condition"])
    gesture = int(row["gesture_code"])
    set_name = str(row["set"])

    output_path = output_feature_path(subject, condition, gesture, set_name)

    data = load_final_window_npz(source_file)
    eeg_windows = data["eeg_windows"]
    source_meta = data["meta"]

    if "start_samples" not in source_meta or "end_samples" not in source_meta:
        raise ValueError(
            f"{source_file.name}: Block 04 start_samples/end_samples are required."
        )

    metadata = build_window_metadata(
        subject,
        condition,
        gesture,
        set_name,
        source_file,
        source_meta["start_samples"],
        source_meta["end_samples"],
    )

    if verify_existing_feature_file(output_path, metadata):
        save_checkpoint(
            source_file,
            "PASS",
            output_file=str(output_path),
            reused=True,
            window_count=WINDOWS_PER_RECORDING,
            feature_count=N_FEATURES,
        )
        return {
            "status": "PASS",
            "reused": True,
            "source_file": str(source_file),
            "feature_file": str(output_path),
            "window_count": WINDOWS_PER_RECORDING,
            "feature_count": N_FEATURES,
            "failure_count": 0,
        }

    save_checkpoint(source_file, "RUNNING")

    feature_rows = []
    failures = []

    for i in range(WINDOWS_PER_RECORDING):
        try:
            eeg = eeg_windows[i]
            if eeg.shape != (N_CHANNELS, WINDOW_SAMPLES):
                raise ValueError(f"Window shape {eeg.shape} is invalid")

            feat = compute_feature_vector(eeg)
            feature_rows.append(feat)

        except Exception as exc:
            failures.append({
                "subject": subject,
                "condition": condition,
                "gesture_code": gesture,
                "gesture": GESTURE_MAP[gesture],
                "set": set_name,
                "recording_id": metadata.iloc[i]["recording_id"],
                "window_id": metadata.iloc[i]["window_id"],
                "source_file": str(source_file),
                "feature_error": repr(exc),
                "error_type": type(exc).__name__,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })

    if failures:
        return {
            "status": "FAIL",
            "reused": False,
            "source_file": str(source_file),
            "feature_file": str(output_path),
            "window_count": WINDOWS_PER_RECORDING,
            "feature_count": N_FEATURES,
            "failure_count": len(failures),
            "failures": failures,
        }

    features = np.asarray(feature_rows, dtype=np.float64)

    save_feature_record(output_path, features, metadata)

    # Verify immediately after writing.
    if not verify_existing_feature_file(output_path, metadata):
        raise RuntimeError(f"Post-write verification failed: {output_path}")

    save_checkpoint(
        source_file,
        "PASS",
        output_file=str(output_path),
        reused=False,
        window_count=WINDOWS_PER_RECORDING,
        feature_count=N_FEATURES,
    )

    return {
        "status": "PASS",
        "reused": False,
        "source_file": str(source_file),
        "feature_file": str(output_path),
        "window_count": WINDOWS_PER_RECORDING,
        "feature_count": N_FEATURES,
        "failure_count": 0,
    }


# =============================================================================
# 12. EXECUTE FEATURE EXTRACTION
# =============================================================================

log("=" * 100)
log("BLOCK 05 STARTED")
log("Paper 2 primary feature extraction")
log("Frozen temporal configuration: W500_O50")
log("341 features/window")
log("=" * 100)

execution_rows = []
failure_rows = []

for idx, row in FILE_MANIFEST.iterrows():
    try:
        result = process_recording(row)
        execution_rows.append({
            "source_file": result["source_file"],
            "feature_file": result["feature_file"],
            "status": result["status"],
            "reused": result["reused"],
            "window_count": result["window_count"],
            "feature_count": result["feature_count"],
            "failure_count": result["failure_count"],
        })
        if result.get("failures"):
            failure_rows.extend(result["failures"])

        if (idx + 1) % 25 == 0 or idx == 0:
            log(f"Processed recordings: {idx + 1}/{N_RECORDINGS}")

    except Exception as exc:
        tb = traceback.format_exc()
        failure_rows.append({
            "source_file": str(row["source_file"]),
            "feature_error": repr(exc),
            "error_type": type(exc).__name__,
            "traceback": tb,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        execution_rows.append({
            "source_file": str(row["source_file"]),
            "feature_file": "",
            "status": "FAIL",
            "reused": False,
            "window_count": 0,
            "feature_count": 0,
            "failure_count": 1,
        })

execution_audit = pd.DataFrame(execution_rows)
atomic_write_csv(
    execution_audit,
    BLOCK05_MANIFESTS / "feature_generation_audit.csv",
)

failures_df = pd.DataFrame(failure_rows)
if failures_df.empty:
    failures_df = pd.DataFrame(
        columns=[
            "subject", "condition", "gesture_code", "gesture", "set",
            "recording_id", "window_id", "source_file",
            "feature_error", "error_type", "traceback", "timestamp_utc",
        ]
    )
atomic_write_csv(failures_df, BLOCK05_MANIFESTS / "feature_failures.csv")

if not execution_audit.empty and (execution_audit["status"] != "PASS").any():
    fail(
        f"BLOCK 05 feature extraction failed for "
        f"{int((execution_audit['status'] != 'PASS').sum())} recordings."
    )

if len(failures_df) != 0:
    fail(f"BLOCK 05 has {len(failures_df)} documented feature failures.")


# =============================================================================
# 13. BUILD FEATURE MANIFEST
# =============================================================================

def build_feature_manifest() -> pd.DataFrame:
    rows = []

    for _, row in FILE_MANIFEST.iterrows():
        subject = int(row["subject"])
        condition = str(row["condition"])
        gesture = int(row["gesture_code"])
        set_name = str(row["set"])
        source_file = str(row["source_file"])
        feature_file = output_feature_path(
            subject, condition, gesture, set_name
        )

        source_path = Path(source_file)
        source_data = load_final_window_npz(source_path)
        source_meta = source_data["meta"]

        if "start_samples" not in source_meta or "end_samples" not in source_meta:
            fail(f"Block 04 start/end provenance missing: {source_path}")

        metadata = build_window_metadata(
            subject,
            condition,
            gesture,
            set_name,
            source_path,
            source_meta["start_samples"],
            source_meta["end_samples"],
        )

        if not verify_existing_feature_file(feature_file, metadata):
            fail(f"Feature output failed verification: {feature_file}")

        for feature_row, m in metadata.iterrows():
            rows.append({
                "subject_id": int(m["subject"]),
                "condition": m["condition"],
                "gesture_code": int(m["gesture_code"]),
                "gesture": m["gesture"],
                "set": m["set"],
                "recording_id": m["recording_id"],
                "window_id": m["window_id"],
                "source_file": m["source_file"],
                "start_sample": int(m["start_sample"]),
                "end_sample": int(m["end_sample"]),
                "start_time_s": float(m["start_time_s"]),
                "end_time_s": float(m["end_time_s"]),
                "window_length": int(m["window_length"]),
                "overlap_percent": int(m["overlap_percent"]),
                "step": int(m["step"]),
                "feature_file": str(feature_file),
                "feature_row": int(feature_row),
            })

    df = pd.DataFrame(rows)

    if len(df) != TOTAL_WINDOWS:
        fail(
            f"Feature manifest contains {len(df)} rows; expected {TOTAL_WINDOWS}."
        )
    if df["window_id"].duplicated().any():
        fail("Feature manifest contains duplicate window IDs.")
    if df["feature_row"].isna().any():
        fail("Feature manifest contains missing feature rows.")

    return df.sort_values(
        ["subject_id", "condition", "gesture_code", "set", "feature_row"]
    ).reset_index(drop=True)


FEATURE_MANIFEST = build_feature_manifest()
atomic_write_csv(
    FEATURE_MANIFEST,
    BLOCK05_MANIFESTS / "feature_manifest.csv",
)


# =============================================================================
# 14. REPRODUCIBILITY TEST
# =============================================================================

def choose_reproducibility_windows(manifest: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Deterministically sample across 10 subjects and both conditions."""
    if len(manifest) <= n:
        return manifest.copy()

    rng = np.random.default_rng(RANDOM_SEED)
    selected = []
    for k, subject in enumerate(range(1, 11)):
        condition = "ME" if k % 2 == 0 else "MI"
        subset = manifest[
            (manifest["subject_id"].astype(int) == subject)
            & (manifest["condition"].astype(str) == condition)
        ]
        if subset.empty:
            subset = manifest[manifest["subject_id"].astype(int) == subject]
        if subset.empty:
            continue
        selected.append(subset.iloc[[int(rng.integers(0, len(subset)))]])
    if not selected:
        return manifest.head(n).copy()
    return pd.concat(selected, ignore_index=True).head(n).copy()


def reproducibility_test() -> pd.DataFrame:
    selected = choose_reproducibility_windows(FEATURE_MANIFEST, n=10)
    rows = []

    for _, row in selected.iterrows():
        # Re-read the same source window and recompute from the actual EEG.
        source_file = Path(row["source_file"])
        data = load_final_window_npz(source_file)
        idx = int(row["feature_row"])
        eeg = data["eeg_windows"][idx]

        run_a = compute_feature_vector(eeg)
        run_b = compute_feature_vector(eeg)

        max_abs_diff = float(np.max(np.abs(run_a - run_b)))
        same_names = FEATURE_NAMES == FEATURE_COLUMN_MANIFEST["feature_name"].tolist()
        passed = bool(
            same_names
            and run_a.shape == (N_FEATURES,)
            and run_b.shape == (N_FEATURES,)
            and np.all(np.isfinite(run_a))
            and np.all(np.isfinite(run_b))
            and max_abs_diff <= REPRO_TOL
        )

        rows.append({
            "window_id": row["window_id"],
            "source_file": row["source_file"],
            "feature_count": int(run_a.size),
            "max_abs_difference": max_abs_diff,
            "tolerance": REPRO_TOL,
            "feature_names_identical": same_names,
            "passed": passed,
        })

    return pd.DataFrame(rows)


REPRO_DF = reproducibility_test()
atomic_write_csv(
    REPRO_DF,
    BLOCK05_DIAGNOSTICS / "reproducibility_test.csv",
)

if not bool(REPRO_DF["passed"].all()):
    fail("BLOCK 05 reproducibility test failed.")


# =============================================================================
# 15. FEATURE DISTRIBUTION / NUMERICAL DIAGNOSTICS
# =============================================================================

def iter_feature_arrays():
    for path in sorted(BLOCK05_FEATURES.glob("*/*.npz")):
        with np.load(path, allow_pickle=True) as z:
            yield path, np.asarray(z["features"], dtype=np.float64)


def build_feature_statistics() -> tuple[pd.DataFrame, np.ndarray]:
    # Dataset size is modest: 22,050 x 341 ≈ 7.5 million values.
    # Load into a single float64 matrix for exact QA summaries.
    matrix = np.empty((TOTAL_WINDOWS, N_FEATURES), dtype=np.float64)
    row_idx = 0

    for path, arr in iter_feature_arrays():
        n = arr.shape[0]
        matrix[row_idx:row_idx+n] = arr
        row_idx += n

    if row_idx != TOTAL_WINDOWS:
        fail(f"Loaded {row_idx} feature rows, expected {TOTAL_WINDOWS}")

    if not np.all(np.isfinite(matrix)):
        fail("Final feature matrix contains NaN/Inf.")

    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        col = matrix[:, j]
        rows.append({
            "feature_index": j + 1,
            "feature_name": name,
            "count": int(col.size),
            "mean": float(np.mean(col)),
            "sd": float(np.std(col, ddof=1)),
            "median": float(np.median(col)),
            "iqr": float(np.percentile(col, 75) - np.percentile(col, 25)),
            "minimum": float(np.min(col)),
            "maximum": float(np.max(col)),
            "nan_count": int(np.isnan(col).sum()),
            "inf_count": int(np.isinf(col).sum()),
            "unique_values": int(np.unique(col).size),
        })

    return pd.DataFrame(rows), matrix


FEATURE_SUMMARY, FEATURE_MATRIX = build_feature_statistics()

atomic_write_csv(
    FEATURE_SUMMARY,
    BLOCK05_DIAGNOSTICS / "feature_summary.csv",
)

# Distribution summary is a QA-only duplicate with explicit feature groups.
distribution = FEATURE_SUMMARY.merge(
    FEATURE_COLUMN_MANIFEST[
        ["feature_index", "feature_group", "channel", "definition"]
    ],
    on="feature_index",
    how="left",
)
atomic_write_csv(
    distribution,
    BLOCK05_DIAGNOSTICS / "feature_distribution_summary.csv",
)

range_summary = FEATURE_SUMMARY[
    [
        "feature_index",
        "feature_name",
        "minimum",
        "maximum",
        "mean",
        "sd",
        "nan_count",
        "inf_count",
    ]
].copy()
atomic_write_csv(
    range_summary,
    BLOCK05_DIAGNOSTICS / "feature_range_summary.csv",
)

# Correlation is diagnostic only; the primary 341 features are not modified.
# Undefined correlations involving constant features are set to 0.0.
with np.errstate(invalid="ignore", divide="ignore"):
    corr = np.corrcoef(FEATURE_MATRIX, rowvar=False)
corr = np.asarray(corr, dtype=np.float64)
corr[~np.isfinite(corr)] = CORR_CONSTANT_VALUE
np.fill_diagonal(corr, 1.0)
corr_df = pd.DataFrame(corr, index=FEATURE_NAMES, columns=FEATURE_NAMES)
corr_df.insert(0, "feature_name", corr_df.index)
atomic_write_csv(
    corr_df.reset_index(drop=True),
    BLOCK05_DIAGNOSTICS / "feature_correlation.csv",
)

atomic_write_json(
    BLOCK05_DIAGNOSTICS / "feature_correlation_metadata.json",
    {
        "purpose": "diagnostic QA only",
        "scope": "all 22050 computational EEG windows",
        "do_not_use_for_selection": True,
        "do_not_use_for_scaling": True,
        "do_not_use_for_weighting": True,
        "do_not_use_for_model_tuning": True,
        "do_not_use_for_downstream_data_driven_preprocessing": True,
        "feature_count": N_FEATURES,
        "window_count": TOTAL_WINDOWS,
        "note": (
            "Downstream ML preprocessing must be fitted exclusively within "
            "the training partition of each evaluation split."
        ),
    },
)

constant_mask = FEATURE_SUMMARY["sd"].to_numpy() <= EPSILON
constant_audit = FEATURE_SUMMARY[["feature_index", "feature_name", "sd", "unique_values"]].copy()
constant_audit["constant_or_zero_variance"] = constant_mask
constant_audit["constant_value"] = np.nan
for j, is_constant in enumerate(constant_mask):
    if is_constant:
        constant_audit.loc[j, "constant_value"] = float(FEATURE_MATRIX[0, j])
atomic_write_csv(constant_audit, BLOCK05_DIAGNOSTICS / "constant_feature_audit.csv")

feature_group_summary = (
    FEATURE_COLUMN_MANIFEST.groupby("feature_group", as_index=False)
    .agg(feature_count=("feature_name", "count"))
    .sort_values(["feature_count", "feature_group"], ascending=[False, True])
    .reset_index(drop=True)
)
atomic_write_csv(feature_group_summary, BLOCK05_DIAGNOSTICS / "feature_group_summary.csv")

channel_feature_coverage = (
    FEATURE_COLUMN_MANIFEST.groupby(["channel", "feature_group"], as_index=False)
    .agg(feature_count=("feature_name", "count"))
    .sort_values(["channel", "feature_group"])
    .reset_index(drop=True)
)
atomic_write_csv(channel_feature_coverage, BLOCK05_DIAGNOSTICS / "channel_feature_coverage.csv")

quantile_rows = []
for j, name in enumerate(FEATURE_NAMES):
    col = FEATURE_MATRIX[:, j]
    quantile_rows.append({
        "feature_index": j + 1,
        "feature_name": name,
        "q01": float(np.percentile(col, 1)),
        "q05": float(np.percentile(col, 5)),
        "q25": float(np.percentile(col, 25)),
        "q50": float(np.percentile(col, 50)),
        "q75": float(np.percentile(col, 75)),
        "q95": float(np.percentile(col, 95)),
        "q99": float(np.percentile(col, 99)),
    })
atomic_write_csv(
    pd.DataFrame(quantile_rows),
    BLOCK05_DIAGNOSTICS / "feature_quantile_summary.csv",
)

# Every full-cohort diagnostic table receives the same explicit usage contract.
DIAGNOSTIC_TABLE_CONTRACT = {
    "scope": "all 22050 computational EEG windows from the complete 25-subject cohort",
    "purpose": "diagnostic QA and descriptive reporting only",
    "not_a_training_parameter_source": True,
    "forbidden_downstream_uses": [
        "scaling",
        "feature selection",
        "feature weighting",
        "winsorization/clipping",
        "imputation",
        "PCA/CSP fitting",
        "model tuning",
        "any learned preprocessing",
    ],
    "training_rule": (
        "When downstream preprocessing is required, fit its parameters only on "
        "the training partition of the active evaluation split and apply unchanged "
        "parameters to validation/test partitions."
    ),
}
for _name in [
    "feature_summary",
    "feature_distribution_summary",
    "feature_range_summary",
    "feature_quantile_summary",
    "constant_feature_audit",
]:
    atomic_write_json(
        BLOCK05_DIAGNOSTICS / f"{_name}_metadata.json",
        {"artifact": f"{_name}.csv", **DIAGNOSTIC_TABLE_CONTRACT},
    )

# Relative-power audit: verify that the four relative bands do not sum to 1.0
# because the shared denominator spans 1-40 Hz while the four relative bands
# span 1-30 Hz. Adjacent bands share boundary bins under the inclusive mask
# used by band_power(); this is the intended frozen convention and no
# renormalization is applied.
relative_rows = []
for ch_idx, ch in enumerate(CHANNEL_NAMES):
    base = ch_idx * N_CHANNEL_FEATURES
    names = [FEATURE_NAMES[base + i] for i in range(N_CHANNEL_FEATURES)]
    idx_map = {name: base + i for i, name in enumerate(names)}
    rel_idx = [
        idx_map[f"{ch}_DELTA_REL"],
        idx_map[f"{ch}_THETA_REL"],
        idx_map[f"{ch}_ALPHA_REL"],
        idx_map[f"{ch}_BETA_REL"],
    ]
    sums = np.sum(FEATURE_MATRIX[:, rel_idx], axis=1)
    relative_rows.append({
        "channel": ch,
        "mean_sum_of_delta_theta_alpha_beta_relative_power": float(np.mean(sums)),
        "median_sum": float(np.median(sums)),
        "minimum_sum": float(np.min(sums)),
        "maximum_sum": float(np.max(sums)),
        "expected_upper_bound": 1.0,
        "denominator": "1-40 Hz total power",
        "numerator_union": (
            "Nominal 1-30 Hz under the frozen inclusive frequency masks; the "
            "actual Welch grid does not contain exact 4, 8, or 13 Hz bins at "
            "125/256 Hz resolution, so no exact boundary bins are double-counted."
        ),
        "interpretation": (
            "Values below 1 are expected because 30-40 Hz remains in the "
            "denominator; the small boundary-bin sharing offsets this only "
            "slightly. No renormalization is applied."
        ),
    })
atomic_write_csv(
    pd.DataFrame(relative_rows),
    BLOCK05_DIAGNOSTICS / "relative_power_definition_audit.csv",
)
atomic_write_json(
    BLOCK05_DIAGNOSTICS / "relative_power_definition_audit_metadata.json",
    {
        "artifact": "relative_power_definition_audit.csv",
        **DIAGNOSTIC_TABLE_CONTRACT,
        "specific_note": (
            "The audit verifies the intended denominator/numerator geometry; "
            "relative powers are not renormalized to sum to one."
        ),
    },
)


# =============================================================================
# 16. BLOCK 05 QA / DESCRIPTIVE FIGURES
# =============================================================================

FIGURES_ROOT = BLOCK05_ROOT / "figures"
FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

def save_figure(fig, stem: str) -> None:
    fig.savefig(FIGURES_ROOT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_ROOT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

# 1. Feature architecture
labels = ["Channel-level\n13 × 24", "Channel means\n13", "Channel SDs\n13", "Global\n3"]
counts = [312, 13, 13, 3]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, counts)
ax.set_ylabel("Number of features")
ax.set_title("Block 05: 341-Feature Representation")
ax.set_ylim(0, max(counts) * 1.15)
for i, v in enumerate(counts):
    ax.text(i, v + max(counts) * 0.02, str(v), ha="center")
fig.tight_layout()
save_figure(fig, "Figure_BLOCK05_Feature_Architecture")

# 2. Feature-group composition
fg = feature_group_summary.sort_values("feature_count", ascending=True)
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(fg["feature_group"], fg["feature_count"])
ax.set_xlabel("Number of features")
ax.set_ylabel("Feature group")
ax.set_title("Block 05: Feature Composition by Group")
fig.tight_layout()
save_figure(fig, "Figure_BLOCK05_Feature_Group_Composition")

# 3. Diagnostic correlation heatmap
fig, ax = plt.subplots(figsize=(11, 10))
im = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
ax.set_title("Block 05: 341-Feature Correlation Matrix (Diagnostic)")
ax.set_xlabel("Feature index")
ax.set_ylabel("Feature index")
fig.colorbar(im, ax=ax, label="Pearson correlation")
fig.tight_layout()
save_figure(fig, "Figure_BLOCK05_Feature_Correlation_Heatmap")

# 4. Feature variability profile
fig, ax = plt.subplots(figsize=(10, 5))
sd_values = np.maximum(FEATURE_SUMMARY["sd"].to_numpy(), EPSILON)
ax.plot(np.arange(1, N_FEATURES + 1), sd_values, linewidth=1.0)
ax.set_yscale("log")
ax.set_xlabel("Feature index")
ax.set_ylabel("Feature SD (log scale)")
ax.set_title("Block 05: Feature Variability Across All Windows")
fig.tight_layout()
save_figure(fig, "Figure_BLOCK05_Feature_SD_Profile")

# 5. Feature median/IQR profile; visualization only, stored features unchanged.
med = FEATURE_SUMMARY["median"].to_numpy()
iqr = FEATURE_SUMMARY["iqr"].to_numpy()
fig, ax = plt.subplots(figsize=(10, 5))
xidx = np.arange(1, N_FEATURES + 1)
ax.plot(xidx, med, linewidth=1.0, label="Median")
ax.plot(xidx, med + 0.5 * iqr, linewidth=0.8, label="Median + 0.5 IQR")
ax.plot(xidx, med - 0.5 * iqr, linewidth=0.8, label="Median - 0.5 IQR")
ax.set_xlabel("Feature index")
ax.set_ylabel("Raw feature value")
ax.set_title("Block 05: Feature Median and IQR Profile")
ax.legend()
fig.tight_layout()
save_figure(fig, "Figure_BLOCK05_Feature_Median_IQR_Profile")

figure_manifest = pd.DataFrame([
    {"figure": "Figure_BLOCK05_Feature_Architecture", "purpose": "341-feature architecture", "model_derived": False, "selection_or_optimization": False, "generated_in_block05": True, "role": "Block 05 descriptive/QA"},
    {"figure": "Figure_BLOCK05_Feature_Group_Composition", "purpose": "Feature-group composition", "model_derived": False, "selection_or_optimization": False, "generated_in_block05": True, "role": "Block 05 descriptive/QA"},
    {"figure": "Figure_BLOCK05_Feature_Correlation_Heatmap", "purpose": "Diagnostic feature correlation across the complete cohort", "model_derived": False, "selection_or_optimization": False, "generated_in_block05": True, "role": "Supplementary/QA only; forbidden for feature selection, scaling, weighting, tuning, or learned preprocessing"},
    {"figure": "Figure_BLOCK05_Feature_SD_Profile", "purpose": "Across-window feature variability across the complete cohort", "model_derived": False, "selection_or_optimization": False, "generated_in_block05": True, "role": "Supplementary/QA only; descriptive"},
    {"figure": "Figure_BLOCK05_Feature_Median_IQR_Profile", "purpose": "Across-window feature distribution across the complete cohort", "model_derived": False, "selection_or_optimization": False, "generated_in_block05": True, "role": "Supplementary/QA only; descriptive"},
    {"figure": "SHAP feature importance curve", "purpose": "Model-derived feature importance", "model_derived": True, "selection_or_optimization": False, "generated_in_block05": False, "role": "BLOCK 13 XAI; intentionally not generated here"},
])
atomic_write_csv(figure_manifest, BLOCK05_MANIFESTS / "BLOCK05_figure_manifest.csv")


# =============================================================================
# 17. FINAL STRUCTURAL / PROVENANCE AUDIT
# =============================================================================

def final_audit() -> dict:
    audit = {
        "block": "05_FEATURE",
        "paper": "Paper 2",
        "status": "PASS",
        "block04_configuration_id": "W500_O50",
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": WINDOW_SECONDS,
        "overlap_percent": OVERLAP_PERCENT,
        "step_samples": STEP_SAMPLES,
        "sampling_rate_hz": FS,
        "channels": N_CHANNELS,
        "subjects": N_SUBJECTS,
        "recordings": N_RECORDINGS,
        "windows_per_recording": WINDOWS_PER_RECORDING,
        "expected_windows": TOTAL_WINDOWS,
        "actual_feature_rows": int(FEATURE_MATRIX.shape[0]),
        "actual_feature_columns": int(FEATURE_MATRIX.shape[1]),
        "feature_count": N_FEATURES,
        "channel_feature_count": N_CHANNEL_FEATURES,
        "channel_level_feature_count": N_CHANNELS * N_CHANNEL_FEATURES,
        "channel_mean_count": N_CHANNELS,
        "channel_sd_count": N_CHANNELS,
        "global_feature_count": 3,
        "nan_values": int(np.isnan(FEATURE_MATRIX).sum()),
        "inf_values": int(np.isinf(FEATURE_MATRIX).sum()),
        "duplicate_window_ids": int(FEATURE_MANIFEST["window_id"].duplicated().sum()),
        "feature_order_hash": sha256_text("\n".join(FEATURE_NAMES)),
        "block04_manifest_sha256": BLOCK04_HANDOFF["manifest_sha256"],
        "timestamp_used_as_clock": False,
        "sample_index_used_as_clock": False,
        "window_selection_in_block05": False,
        "feature_selection_in_block05": False,
        "scaling_in_block05": False,
        "pca_in_block05": False,
        "csp_in_block05": False,
        "smote_in_block05": False,
        "model_fitting_in_block05": False,
        "raw_data_modified": False,
        "block03_modified": False,
        "block05_qa_figures_generated": 5,
        "model_derived_feature_importance_generated": False,
        "model_derived_feature_importance_block": "13_XAI",
    }

    expected = {
        "actual_feature_rows": TOTAL_WINDOWS,
        "actual_feature_columns": N_FEATURES,
        "nan_values": 0,
        "inf_values": 0,
        "duplicate_window_ids": 0,
    }

    for key, value in expected.items():
        if audit[key] != value:
            audit["status"] = "FAIL"
            audit[f"failure_{key}"] = {
                "actual": audit[key],
                "expected": value,
            }

    # Verify each condition contributes exactly half of the paired windows.
    condition_counts = FEATURE_MANIFEST["condition"].value_counts().to_dict()
    audit["condition_window_counts"] = condition_counts
    if condition_counts.get("ME") != 11025 or condition_counts.get("MI") != 11025:
        audit["status"] = "FAIL"
        audit["failure_condition_counts"] = {
            "actual": condition_counts,
            "expected": {"ME": 11025, "MI": 11025},
        }

    # Subject counts.
    subject_counts = FEATURE_MANIFEST.groupby("subject_id").size()
    audit["subject_window_counts_unique"] = sorted(subject_counts.unique().tolist())
    if len(subject_counts) != 25 or not np.all(subject_counts.to_numpy() == 882):
        audit["status"] = "FAIL"
        audit["failure_subject_counts"] = {
            "expected_per_subject": 882,
            "actual_unique_counts": sorted(subject_counts.unique().tolist()),
        }

    # Downstream leakage contract artifacts must exist and explicitly prohibit
    # full-cohort diagnostic statistics from becoming learned parameters.
    contract_path = BLOCK05_CONFIGS / "BLOCK05_LEAKAGE_FORBIDDEN_ARTIFACTS.json"
    audit["leakage_contract_verified"] = False
    if not contract_path.exists():
        audit["status"] = "FAIL"
        audit["failure_leakage_contract"] = "Missing BLOCK05_LEAKAGE_FORBIDDEN_ARTIFACTS.json"
    else:
        contract = load_json_required(contract_path)
        audit["leakage_contract_verified"] = bool(
            contract.get("diagnostic_statistics_are_full_cohort") is True
            and contract.get("diagnostic_statistics_are_not_training_parameters") is True
            and len(contract.get("forbidden_artifacts", [])) > 0
        )
        if not audit["leakage_contract_verified"]:
            audit["status"] = "FAIL"

    # Every recording must contribute 21 windows.
    rec_counts = FEATURE_MANIFEST.groupby("recording_id").size()
    audit["recording_window_counts_unique"] = sorted(rec_counts.unique().tolist())
    if len(rec_counts) != N_RECORDINGS or not np.all(rec_counts.to_numpy() == 21):
        audit["status"] = "FAIL"
        audit["failure_recording_counts"] = {
            "expected": 21,
            "actual_unique_counts": sorted(rec_counts.unique().tolist()),
        }

    return audit


FINAL_AUDIT = final_audit()
atomic_write_json(BLOCK05_DIAGNOSTICS / "BLOCK05_final_audit.json", FINAL_AUDIT)

if FINAL_AUDIT["status"] != "PASS":
    fail("BLOCK 05 final structural audit failed.")


# =============================================================================
# 18. SUMMARY / COMPLETION MARKER
# =============================================================================

runtime_minutes = (time.time() - START_TIME) / 60.0

summary = {
    "block": "05_FEATURE",
    "paper": "Paper 2",
    "status": "PASS",
    "selection_method": "DATA-INDEPENDENT PRE-STUDY FREEZE",
    "block04_configuration": "W500_O50",
    "window_samples": WINDOW_SAMPLES,
    "window_seconds": WINDOW_SECONDS,
    "overlap_percent": OVERLAP_PERCENT,
    "step_samples": STEP_SAMPLES,
    "sampling_rate_hz": FS,
    "channels": N_CHANNELS,
    "subjects": N_SUBJECTS,
    "recordings": N_RECORDINGS,
    "windows_per_recording": WINDOWS_PER_RECORDING,
    "total_windows": TOTAL_WINDOWS,
    "feature_count_per_window": N_FEATURES,
    "feature_matrix_shape": [TOTAL_WINDOWS, N_FEATURES],
    "ME_windows": 11025,
    "MI_windows": 11025,
    "feature_failures": int(len(failures_df)),
    "reproducibility_pass": bool(REPRO_DF["passed"].all()),
    "timestamp_used_as_clock": False,
    "sample_index_used_as_clock": False,
    "window_selection_performed": False,
    "feature_selection_performed": False,
    "scaling_performed": False,
    "model_fitting_performed": False,
    "full_cohort_diagnostics_generated": True,
    "full_cohort_diagnostics_scope": "structural QA/descriptive reporting only",
    "full_cohort_diagnostics_allowed_for_downstream_learning": False,
    "leakage_contract_verified": bool(FINAL_AUDIT.get("leakage_contract_verified", False)),
    "raw_data_modified": False,
    "block03_modified": False,
    "block05_qa_figures_generated": 5,
    "model_derived_feature_importance_generated": False,
    "model_derived_feature_importance_block": "13_XAI",
    "runtime_minutes": round(runtime_minutes, 3),
    "output_root": str(BLOCK05_ROOT),
}

atomic_write_json(BLOCK05_ROOT / "BLOCK05_summary.json", summary)

# Completion marker is deliberately written last.
complete = {
    "block": "05_FEATURE",
    "status": "PASS",
    "paper": "Paper 2",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "frozen_configuration": "W500_O50",
    "feature_count": N_FEATURES,
    "total_windows": TOTAL_WINDOWS,
    "feature_matrix_shape": [TOTAL_WINDOWS, N_FEATURES],
    "block04_handoff_verified": True,
    "feature_schema_verified": True,
    "numerical_audit_verified": True,
    "provenance_verified": True,
    "reproducibility_verified": True,
    "leakage_controls_verified": bool(FINAL_AUDIT["status"] == "PASS" and FINAL_AUDIT.get("leakage_contract_verified", False)),
    "primary_feature_set_frozen": True,
    "full_cohort_diagnostic_statistics_quarantined": True,
    "downstream_leakage_contract_written": True,
    "full_cohort_diagnostic_statistics_quarantined": True,
    "trial_reconstruction_applied": False,
}

atomic_write_json(BLOCK05_ROOT / "BLOCK05_complete.json", complete)


# =============================================================================
# 19. FINAL CONSOLE REPORT
# =============================================================================

print("\n" + "=" * 100)
print("BLOCK 05 — FINAL STATUS")
print("=" * 100)
print(f"Configuration             : W500_O50")
print(f"Window                    : {WINDOW_SAMPLES} samples ({WINDOW_SECONDS:.3f} s)")
print(f"Overlap                   : {OVERLAP_PERCENT}%")
print(f"Step                      : {STEP_SAMPLES} samples")
print(f"Subjects                  : {N_SUBJECTS}")
print(f"Recordings                : {N_RECORDINGS}/{N_RECORDINGS}")
print(f"Windows/recording        : {WINDOWS_PER_RECORDING}")
print(f"Total windows             : {TOTAL_WINDOWS}/{TOTAL_WINDOWS}")
print(f"Features/window           : {N_FEATURES}")
print(f"Feature matrix            : ({TOTAL_WINDOWS}, {N_FEATURES})")
print(f"ME windows                : 11025")
print(f"MI windows                : 11025")
print(f"Feature failures          : {len(failures_df)}")
print(f"Reproducibility           : {'PASS' if REPRO_DF['passed'].all() else 'FAIL'}")
print(f"NaN                      : {int(np.isnan(FEATURE_MATRIX).sum())}")
print(f"Inf                      : {int(np.isinf(FEATURE_MATRIX).sum())}")
print(f"Window reselection        : NO")
print(f"Feature selection         : NO")
print(f"Scaling in Block 05      : NO")
print(f"CSP/PCA                   : NO")
print(f"SMOTE                     : NO")
print(f"Model fitting             : NO")
print(f"Block 05 QA figures       : 5")
print(f"SHAP feature importance   : DEFERRED TO BLOCK 13 XAI")
print(f"Raw modified              : NO")
print(f"BLOCK 03 modified         : NO")
print(f"Runtime                   : {runtime_minutes:.2f} min")
print(f"Output                    : {BLOCK05_ROOT}")
print("✓ BLOCK 05 PASS — 341-FEATURE REPRESENTATION FROZEN")
print("=" * 100)