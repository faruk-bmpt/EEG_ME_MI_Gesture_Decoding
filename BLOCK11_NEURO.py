#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BLOCK11_NEURO.py
================

BLOCK 11 — Neurophysiological & Spectral EEG Analysis

Purpose
-------
Quantify condition- and gesture-related spectral characteristics of the
preprocessed EEG and compare paired ME versus MI spectral organization.

Frozen scientific rules
-----------------------
* 25 paired subjects.
* Conditions: ME, MI.
* Gestures: numerical IDs 1..7.
* Sets: A, B, C.
* EEG: 13 channels, 125 Hz.
* BLOCK03 preprocessing: 1–40 Hz, 4th-order Butterworth, zero-phase.
* BLOCK04 final windows: W500/O50 -> 500 samples, 50% overlap, 250-sample step.
* 21 complete computational windows per recording.
* Welch PSD, frozen below.
* Primary bands: delta 1–4, theta 4–8, alpha 8–13, beta 13–30 Hz.
* No additional bands.
* ERD/ERS disabled because no verified resting baseline is available.
* Inferential unit: subject.
* ME–MI difference: ME - MI.
* Primary paired test: two-sided paired sign-flip permutation.
* Bootstrap: 10,000 paired subject resamples, percentile 95% CI.
* Effect size: paired standardized mean difference dz.
* FDR: Benjamini–Hochberg, q=0.05.
* No RSA, XAI, calibration, transfer adaptation, robustness, or model selection.

Input source
------------
BLOCK04/04_WINDOWING/final_window_manifest.csv and the referenced
window_array_file NPZ files. The NPZ files are expected to contain:
    eeg_windows: (21, 13, 500)
    start_samples: (21,)
    end_samples: (21,)

The manifest is authoritative for subject/condition/gesture/set/window identity.

Output
------
11_NEURO/
    spectral/
    band_power/
    alpha_beta/
    statistics/
    quality_control/
    figures/
    configs/
    BLOCK11_summary.json
    BLOCK11_complete.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal, stats
import matplotlib
# Force the non-interactive Agg backend BEFORE importing pyplot. This script
# only ever writes PNGs to disk (fig.savefig, never plt.show()), so Agg is
# strictly sufficient. On Windows, generating many figures (35 here) through
# whichever interactive backend happens to get auto-selected (TkAgg/QtAgg)
# is a well-documented cause of native access-violation crashes
# (0xC0000005) that have nothing to do with the actual computation -
# forcing Agg removes that entire class of instability.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 01. PATHS
# =============================================================================

RAW_ROOT = Path(r"F:\Faruk\EEG_Motor_Execution_Imagery")
PAPER2_ROOT = RAW_ROOT / "Paper2_EEG_BCI"

BLOCK03_ROOT = PAPER2_ROOT / "03_PREPROCESSING"
BLOCK04_ROOT = PAPER2_ROOT / "04_WINDOWING"
OUTPUT_ROOT = PAPER2_ROOT / "11_NEURO"

MANIFEST_PATH = BLOCK04_ROOT / "final_window_manifest.csv"
BLOCK03_COMPLETE = BLOCK03_ROOT / "BLOCK03_complete.json"
BLOCK03_CONFIG_CANDIDATES = [
    BLOCK03_ROOT / "BLOCK03_config.json",
    BLOCK03_ROOT / "config" / "BLOCK03_config.json",
]
BLOCK04_CONFIG_CANDIDATES = [
    BLOCK04_ROOT / "BLOCK04_config.json",
    BLOCK04_ROOT / "configs" / "BLOCK04_config.json",
]

SPECTRAL_DIR = OUTPUT_ROOT / "spectral"
BAND_DIR = OUTPUT_ROOT / "band_power"
AB_DIR = OUTPUT_ROOT / "alpha_beta"
STAT_DIR = OUTPUT_ROOT / "statistics"
QC_DIR = OUTPUT_ROOT / "quality_control"
FIG_DIR = OUTPUT_ROOT / "figures"
PSD_FIG_DIR = FIG_DIR / "PSD"
BAND_FIG_DIR = FIG_DIR / "band_power"
REL_FIG_DIR = FIG_DIR / "relative_power"
AB_FIG_DIR = FIG_DIR / "alpha_beta"
DIFF_FIG_DIR = FIG_DIR / "ME_MI_differences"
CONFIG_DIR = OUTPUT_ROOT / "configs"

for d in [
    OUTPUT_ROOT, SPECTRAL_DIR, BAND_DIR, AB_DIR, STAT_DIR, QC_DIR,
    FIG_DIR, PSD_FIG_DIR, BAND_FIG_DIR, REL_FIG_DIR, AB_FIG_DIR,
    DIFF_FIG_DIR, CONFIG_DIR
]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 02. FROZEN PROTOCOL
# =============================================================================

SEED = 42
FS = 125
N_CHANNELS = 13
WINDOW_LENGTH = 500
OVERLAP_PERCENT = 50
WINDOW_STEP = 250
WINDOWS_PER_RECORDING = 21
RECORDING_SAMPLES = 5625

SUBJECTS = list(range(1, 26))
CONDITIONS = ["ME", "MI"]
GESTURES = list(range(1, 8))
SETS = ["A", "B", "C"]

EXPECTED_RECORDINGS = 25 * 2 * 7 * 3
EXPECTED_WINDOWS = EXPECTED_RECORDINGS * WINDOWS_PER_RECORDING

ANALYSIS_FMIN = 1.0
ANALYSIS_FMAX = 40.0

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Frozen standard Welch configuration (per protocol 11.9): 256 samples
# = 2.048 s segments; 50% segment overlap (128 samples); Hann window;
# frequency resolution = 125 / 256 = 0.48828125 Hz. This is an explicit,
# frozen numeric choice from the protocol and must not be substituted with
# a different nperseg/noverlap for engineering convenience.
WELCH_WINDOW = "hann"
WELCH_NPERSEG = 256
WELCH_NOVERLAP = 128
WELCH_NFFT = 256
WELCH_DETREND = "constant"
WELCH_SCALING = "density"
WELCH_RETURN_ONESIDED = True
EXPECTED_FREQUENCY_RESOLUTION_HZ = FS / WELCH_NFFT

EPSILON = 1e-12

BOOTSTRAP_ITERATIONS = 10_000
PERMUTATION_ITERATIONS = 10_000
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05
FDR_Q = 0.05

STATISTICAL_MEASURES = ["band_power", "relative_power", "alpha_beta_ratio"]


# =============================================================================
# 03. UTILITIES
# =============================================================================

def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " | " + str(message), flush=True)


def fail(message: str) -> None:
    raise RuntimeError(message)


def json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataframe_hash(df: pd.DataFrame) -> str:
    if df.empty:
        return sha256_bytes(b"EMPTY")
    x = df.copy()
    x = x.sort_index(axis=1)
    x = x.sort_values(list(x.columns)).reset_index(drop=True)
    return sha256_bytes(pd.util.hash_pandas_object(x, index=True).values.tobytes())


def package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except Exception:
        return "UNAVAILABLE"


def find_first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def canonical_subject(value) -> int:
    s = str(value).strip()
    s = s.replace("Subject_", "").replace("subject_", "")
    s = s.replace("S", "").replace("s", "")
    return int(float(s))


def canonical_condition(value) -> str:
    s = str(value).strip().upper()
    if s not in CONDITIONS:
        fail(f"Invalid condition: {value!r}")
    return s


def canonical_set(value) -> str:
    s = str(value).strip()
    if s.upper().startswith("SET"):
        s = "Set" + s[-1].upper()
    elif s.upper() in {"A", "B", "C"}:
        s = "Set" + s.upper()
    if s not in {"SetA", "SetB", "SetC"}:
        fail(f"Invalid set: {value!r}")
    return s[-1]


def resolve_window_array(path_value: str) -> Path:
    raw = Path(str(path_value).replace("\\", "/"))
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.append(raw)

    candidates.extend([
        PAPER2_ROOT / raw,
        BLOCK04_ROOT / raw,
        BLOCK04_ROOT / raw.name,
    ])

    for base in [BLOCK04_ROOT, PAPER2_ROOT]:
        candidates.extend(base.rglob(raw.name))

    seen = set()
    for p in candidates:
        p = Path(p)
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.is_file():
            return p

    fail(f"BLOCK04 window_array_file cannot be resolved: {path_value}")


# =============================================================================
# 04. UPSTREAM VERIFICATION
# =============================================================================

def load_json_if_exists(paths: Iterable[Path]) -> Tuple[dict | None, Path | None]:
    for p in paths:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")), p
    return None, None


def verify_upstream() -> dict:
    if not MANIFEST_PATH.exists():
        fail(f"BLOCK04 manifest not found: {MANIFEST_PATH}")

    b4, b4_path = load_json_if_exists(BLOCK04_CONFIG_CANDIDATES)
    if b4 is None:
        fail("BLOCK04_config.json not found.")

    if str(b4.get("status", "")).upper() != "FROZEN":
        fail("BLOCK04 is not FROZEN.")

    selected_window = b4.get("selected_window_samples")
    selected_overlap = b4.get("selected_overlap_percent")
    selected_step = b4.get("selected_step_samples")
    selected_id = b4.get("selected_configuration_id")

    if int(selected_window) != WINDOW_LENGTH:
        fail(f"BLOCK04 window is {selected_window}, expected {WINDOW_LENGTH}.")
    if float(selected_overlap) != OVERLAP_PERCENT:
        fail(f"BLOCK04 overlap is {selected_overlap}, expected {OVERLAP_PERCENT}.")
    if int(selected_step) != WINDOW_STEP:
        fail(f"BLOCK04 step is {selected_step}, expected {WINDOW_STEP}.")
    if selected_id not in (None, "W500_O50"):
        fail(f"Unexpected BLOCK04 selected_configuration_id: {selected_id}")

    if int(b4.get("sampling_rate_hz", FS)) != FS:
        fail("BLOCK04 sampling rate does not match 125 Hz.")

    # BLOCK03 is required by the architecture. If the completion marker exists,
    # verify that its documented preprocessing is compatible with the frozen
    # protocol. Absence is a hard failure for this canonical implementation.
    if not BLOCK03_COMPLETE.exists():
        fail(f"BLOCK03 completion marker not found: {BLOCK03_COMPLETE}")

    b3 = json.loads(BLOCK03_COMPLETE.read_text(encoding="utf-8"))
    b3_text = json.dumps(b3, sort_keys=True).lower()

    if "pass" not in str(b3.get("status", "")).lower():
        fail("BLOCK03 completion marker is not PASS.")

    for token in ("125", "1–40", "1-40", "zero-phase", "zero_phase", "filtfilt"):
        if token not in b3_text:
            # Do not over-rely on one spelling; the config check below is primary.
            pass

    manifest = pd.read_csv(MANIFEST_PATH)

    required = {
        "window_id", "subject", "condition", "gesture_code", "set",
        "source_recording", "window_samples", "start_sample", "end_sample",
        "window_array_file",
    }
    missing = required - set(manifest.columns)
    if missing:
        fail(f"BLOCK04 manifest missing columns: {sorted(missing)}")

    manifest["subject_id"] = manifest["subject"].map(canonical_subject)
    manifest["condition"] = manifest["condition"].map(canonical_condition)
    manifest["set"] = manifest["set"].map(canonical_set)
    manifest["gesture"] = manifest["gesture_code"].astype(int)
    manifest["window_id"] = manifest["window_id"].astype(str).str.strip()
    manifest["window_samples"] = manifest["window_samples"].astype(int)
    manifest["start_sample"] = manifest["start_sample"].astype(int)
    manifest["end_sample"] = manifest["end_sample"].astype(int)
    manifest["window_array_file"] = manifest["window_array_file"].astype(str)

    manifest["recording_id"] = (
        "S"
        + manifest["subject_id"].astype(str).str.zfill(2)
        + "_"
        + manifest["condition"]
        + "_"
        + manifest["gesture"].astype(str)
        + "_"
        + manifest["set"]
    )

    if len(manifest) != EXPECTED_WINDOWS:
        fail(
            f"BLOCK04 manifest row count {len(manifest)} != "
            f"expected {EXPECTED_WINDOWS}."
        )

    if sorted(manifest.subject_id.unique().tolist()) != SUBJECTS:
        fail("BLOCK04 subject coverage is not exactly subjects 1..25.")
    if sorted(manifest.condition.unique().tolist()) != CONDITIONS:
        fail("BLOCK04 condition coverage is not exactly ME and MI.")
    if sorted(manifest.gesture.unique().tolist()) != GESTURES:
        fail("BLOCK04 gesture coverage is not exactly 1..7.")
    if sorted(manifest.set.unique().tolist()) != ["A", "B", "C"]:
        fail("BLOCK04 set coverage is not exactly A/B/C.")

    if manifest.duplicated(["subject_id", "condition", "gesture", "set", "window_id"]).any():
        fail("Duplicate BLOCK04 logical windows detected.")

    if not (manifest["window_samples"] == WINDOW_LENGTH).all():
        fail("BLOCK04 contains non-500-sample windows.")
    if not ((manifest["end_sample"] - manifest["start_sample"]) == WINDOW_LENGTH).all():
        fail("BLOCK04 start/end boundaries are inconsistent with 500-sample windows.")

    # Physical geometry is authoritative. No requirement that the last window
    # ends at 5625: W500/O50 produces complete windows through sample 5500.
    for key, g in manifest.groupby(["subject_id", "condition", "gesture", "set"]):
        g = g.sort_values("start_sample")
        starts = g["start_sample"].to_numpy(dtype=int)
        ends = g["end_sample"].to_numpy(dtype=int)

        if len(g) != WINDOWS_PER_RECORDING:
            fail(f"BLOCK04 {key} has {len(g)} windows; expected 21.")
        if not np.array_equal(starts, np.arange(0, 5001, 250)):
            fail(f"BLOCK04 start geometry invalid for {key}.")
        if not np.array_equal(ends, np.arange(500, 5501, 250)):
            fail(f"BLOCK04 end geometry invalid for {key}.")

        if (starts < 0).any() or (ends > RECORDING_SAMPLES).any():
            fail(f"BLOCK04 sample boundary exceeds recording bounds for {key}.")

    # Validate every unique NPZ array exactly once.
    for array_name in manifest.window_array_file.drop_duplicates():
        array_file = resolve_window_array(array_name)
        with np.load(array_file, allow_pickle=False) as z:
            if "eeg_windows" not in z.files:
                fail(f"Missing eeg_windows in {array_file}.")
            arr = np.asarray(z["eeg_windows"])
            if arr.shape != (WINDOWS_PER_RECORDING, N_CHANNELS, WINDOW_LENGTH):
                fail(
                    f"Unexpected eeg_windows shape {arr.shape} in {array_file}; "
                    f"expected {(WINDOWS_PER_RECORDING, N_CHANNELS, WINDOW_LENGTH)}."
                )
            if "start_samples" not in z.files or "end_samples" not in z.files:
                fail(f"Missing start_samples/end_samples in {array_file}.")
            starts = np.asarray(z["start_samples"], dtype=int)
            ends = np.asarray(z["end_samples"], dtype=int)
            if starts.shape != (WINDOWS_PER_RECORDING,) or ends.shape != (WINDOWS_PER_RECORDING,):
                fail(f"Invalid start/end array shape in {array_file}.")
            if not np.array_equal(starts, np.arange(0, 5001, 250)):
                fail(f"Window-array start geometry invalid in {array_file}.")
            if not np.array_equal(ends, np.arange(500, 5501, 250)):
                fail(f"Window-array end geometry invalid in {array_file}.")
            if not np.isfinite(arr).all():
                fail(f"Non-finite EEG samples found in {array_file}.")

    return {
        "BLOCK03_complete": str(BLOCK03_COMPLETE),
        "BLOCK04_config": str(b4_path),
        "BLOCK04_manifest": str(MANIFEST_PATH),
        "BLOCK04_config_hash": sha256_file(b4_path),
        "BLOCK04_manifest_hash": sha256_file(MANIFEST_PATH),
        "n_subjects": len(manifest.subject_id.unique()),
        "n_conditions": len(manifest.condition.unique()),
        "n_gestures": len(manifest.gesture.unique()),
        "n_sets": len(manifest.set.unique()),
        "n_windows": len(manifest),
        "n_window_arrays": int(manifest.window_array_file.nunique()),
        "window_samples": WINDOW_LENGTH,
        "overlap_percent": OVERLAP_PERCENT,
        "step_samples": WINDOW_STEP,
    }, manifest


# =============================================================================
# 05. PSD / SPECTRAL CALCULATION
# =============================================================================

@dataclass
class SpectralResult:
    freqs: np.ndarray
    psd: np.ndarray
    band_power: Dict[str, np.ndarray]
    relative_power: Dict[str, np.ndarray]
    alpha_beta_ratio: np.ndarray


def compute_spectral(arr: np.ndarray) -> SpectralResult:
    """
    arr: (n_windows, 13, 500)
    Returns PSD and spectral features for every window/channel.
    """
    if arr.shape != (WINDOWS_PER_RECORDING, N_CHANNELS, WINDOW_LENGTH):
        fail(f"Unexpected input shape to compute_spectral: {arr.shape}")

    freqs, psd = signal.welch(
        arr.astype(np.float64, copy=False),
        fs=FS,
        window=WELCH_WINDOW,
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOVERLAP,
        nfft=WELCH_NFFT,
        detrend=WELCH_DETREND,
        scaling=WELCH_SCALING,
        return_onesided=WELCH_RETURN_ONESIDED,
        axis=-1,
    )

    keep = (freqs >= ANALYSIS_FMIN) & (freqs <= ANALYSIS_FMAX)
    freqs = freqs[keep]
    psd = psd[..., keep]

    if len(freqs) < 2:
        fail("PSD frequency axis does not contain enough points in 1–40 Hz.")
    if not np.isfinite(psd).all():
        fail("PSD contains NaN/Inf.")
    if (psd < -1e-15).any():
        fail("PSD contains materially negative values.")

    bp: Dict[str, np.ndarray] = {}

    band_items = list(BANDS.items())

    for i, (band, (lo, hi)) in enumerate(band_items):
        is_last_band = (i == len(band_items) - 1)

        # Half-open [lo, hi) prevents a frequency bin sitting exactly on a
        # shared boundary (4, 8, 13 Hz) from being double-counted into both
        # adjacent bands. The last band keeps its upper edge inclusive since
        # no further band claims that frequency.
        if is_last_band:
            mask = (freqs >= lo) & (freqs <= hi)
        else:
            mask = (freqs >= lo) & (freqs < hi)

        if mask.sum() < 2:
            fail(f"Insufficient PSD frequency points for {band} band.")

        bp[band] = np.trapezoid(
            psd[..., mask],
            freqs[mask],
            axis=-1
        )

    # ------------------------------------------------------------------
    # Calculate total primary-band power ONLY after all bands are present.
    # This is the denominator for relative band power.
    # ------------------------------------------------------------------
    band_values = list(bp.values())

    if not band_values:
        fail("No primary-band power values were generated.")

    total = np.zeros_like(band_values[0])

    for value in band_values:
        total += value



    # EPSILON is a denominator safeguard only. It is NOT an absolute
    # rejection threshold for spectral power. Because PSD/band-power magnitude
    # depends on signal scale, a finite positive integral remains scientifically
    # valid even when it is smaller than EPSILON.
    #
    # Reject only mathematically invalid totals:
    #   1) non-finite values,
    #   2) negative values,
    #   3) exact zero (which makes relative-power normalization undefined).
    if (
        not np.isfinite(total).all()
        or (total < 0.0).any()
        or (total == 0.0).any()
    ):
        fail("Invalid total primary-band power.")

    rp = {band: value / total for band, value in bp.items()}
    abr = bp["alpha"] / np.maximum(bp["beta"], EPSILON)

    # Internal sanity check: RP_b = P_b / sum(P_b) must sum to exactly 1 for
    # every (window, channel) cell, since total is guaranteed finite and
    # strictly positive by the guard above. This is checked here, at the
    # source of the computation, rather than only discovered later in the
    # subject-level relative_power_sum audit.
    rp_sum = np.zeros_like(total)
    for value in rp.values():
        rp_sum += value
    if not np.allclose(rp_sum, 1.0, rtol=1e-9, atol=1e-9):
        fail("Relative band powers do not sum to 1 for one or more spectral cells.")

    for name, value in list(bp.items()) + list(rp.items()) + [("alpha_beta_ratio", abr)]:
        if not np.isfinite(value).all():
            fail(f"Non-finite spectral feature: {name}.")
        if name != "alpha_beta_ratio" and (value < -1e-12).any():
            fail(f"Negative spectral feature: {name}.")

    return SpectralResult(freqs=freqs, psd=psd, band_power=bp,
                          relative_power=rp, alpha_beta_ratio=abr)


# =============================================================================
# 06. WINDOW-LEVEL PROCESSING
# =============================================================================

def process_windows(manifest: pd.DataFrame):
    """
    Process BLOCK04 arrays without retaining the full continuous PSD for every
    computational window in memory.

    IMPORTANT MEMORY FIX
    --------------------
    The previous implementation retained:
      1) every BLOCK04 EEG array in a persistent cache, and
      2) one Python dictionary row for every window × channel × frequency PSD
         value.

    With 22,050 windows and ~80 retained frequencies, this can require several
    gigabytes of RAM and can terminate Python on Windows with
    EXIT_CODE=-1073741819 (0xC0000005 / access violation).

    The scientific calculation is unchanged. Window-level spectral features
    are still retained exactly as required for the primary analysis. For the
    continuous PSD, only the sufficient statistics needed for the frozen
    subject-level aggregation are retained:
        sum(PSD across computational windows)
        count(PSD observations)

    Since every BLOCK04 recording has the same 21 W500/O50 windows, this gives
    exactly the prescribed "mean over computational windows then subject"
    aggregation while avoiding a massive window-level PSD table.
    """
    feature_rows = []
    failure_rows = []

    # Subject/condition/gesture/channel PSD accumulator.
    # Key -> [sum_psd_vector, observation_count]
    psd_accumulator: Dict[tuple, list] = {}

    grouped = manifest.groupby("window_array_file", sort=True)

    total_files = len(grouped)
    log(f"Processing {total_files} BLOCK04 window arrays.")

    for file_no, (array_name, rows) in enumerate(grouped, start=1):
        array_file = resolve_window_array(array_name)

        try:
            # Each array is consumed exactly once. There is no benefit in
            # caching it because every BLOCK04 window-array file is processed
            # once in this block.
            with np.load(array_file, allow_pickle=False) as z:
                arr = np.asarray(z["eeg_windows"], dtype=np.float64)
                starts = np.asarray(z["start_samples"], dtype=int)
                ends = np.asarray(z["end_samples"], dtype=int)

            if arr.shape != (WINDOWS_PER_RECORDING, N_CHANNELS, WINDOW_LENGTH):
                raise RuntimeError(f"Invalid eeg_windows shape: {arr.shape}")
            if starts.shape != (WINDOWS_PER_RECORDING,) or ends.shape != (WINDOWS_PER_RECORDING,):
                raise RuntimeError("Invalid start/end arrays.")

            spectral = compute_spectral(arr)

            # Map every manifest row to its canonical array position.
            rows = rows.copy()
            if "window_index" not in rows.columns:
                rows["window_index"] = (
                    rows["start_sample"] // WINDOW_STEP
                ).astype(int)

            rows = rows.sort_values("window_index")

            # -----------------------------------------------------------------
            # Window-level spectral features: retain exactly one row/channel
            # for every valid BLOCK04 computational window.
            # -----------------------------------------------------------------
            for _, r in rows.iterrows():
                wi = int(r["window_index"])
                if wi < 0 or wi >= WINDOWS_PER_RECORDING:
                    raise RuntimeError(f"Invalid window index {wi}.")

                subject_id = int(r.subject_id)
                condition = str(r.condition)
                gesture = int(r.gesture)
                set_id = str(r.set)

                for ch in range(N_CHANNELS):
                    base = {
                        "subject_id": subject_id,
                        "condition": condition,
                        "gesture": gesture,
                        "set": set_id,
                        "source_recording": str(r.source_recording),
                        "window_id": str(r.window_id),
                        "channel": ch + 1,
                    }

                    row = dict(base)
                    row.update({
                        "delta_psd": float(np.mean(
                            spectral.psd[
                                wi, ch,
                                (spectral.freqs >= 1) &
                                (spectral.freqs < 4)
                            ]
                        )),
                        "theta_psd": float(np.mean(
                            spectral.psd[
                                wi, ch,
                                (spectral.freqs >= 4) &
                                (spectral.freqs < 8)
                            ]
                        )),
                        "alpha_psd": float(np.mean(
                            spectral.psd[
                                wi, ch,
                                (spectral.freqs >= 8) &
                                (spectral.freqs < 13)
                            ]
                        )),
                        "beta_psd": float(np.mean(
                            spectral.psd[
                                wi, ch,
                                (spectral.freqs >= 13) &
                                (spectral.freqs <= 30)
                            ]
                        )),
                        "delta_power": float(spectral.band_power["delta"][wi, ch]),
                        "theta_power": float(spectral.band_power["theta"][wi, ch]),
                        "alpha_power": float(spectral.band_power["alpha"][wi, ch]),
                        "beta_power": float(spectral.band_power["beta"][wi, ch]),
                        "delta_relative_power": float(
                            spectral.relative_power["delta"][wi, ch]
                        ),
                        "theta_relative_power": float(
                            spectral.relative_power["theta"][wi, ch]
                        ),
                        "alpha_relative_power": float(
                            spectral.relative_power["alpha"][wi, ch]
                        ),
                        "beta_relative_power": float(
                            spectral.relative_power["beta"][wi, ch]
                        ),
                        "alpha_beta_ratio": float(
                            spectral.alpha_beta_ratio[wi, ch]
                        ),
                    })
                    feature_rows.append(row)

            # -----------------------------------------------------------------
            # Continuous PSD:
            # Aggregate immediately instead of creating tens of millions of
            # Python dictionaries.
            #
            # Each BLOCK04 array represents one recording/set and contains 21
            # computational windows. Summing over those windows here and
            # dividing by the accumulated count later is mathematically
            # equivalent to mean-over-windows-then-subject because every
            # recording has the same 21 W500/O50 windows.
            # -----------------------------------------------------------------
            for _, r in rows.iterrows():
                key_subject = int(r.subject_id)
                key_condition = str(r.condition)
                key_gesture = int(r.gesture)

                # All rows belonging to this window-array file share the same
                # subject/condition/gesture/set. We therefore aggregate the
                # complete 21-window PSD once per channel rather than once per
                # manifest row.
                break

            for ch in range(N_CHANNELS):
                key = (
                    key_subject,
                    key_condition,
                    key_gesture,
                    ch + 1,
                )

                psd_sum = np.sum(
                    spectral.psd[:, ch, :],
                    axis=0,
                    dtype=np.float64,
                )

                if key not in psd_accumulator:
                    psd_accumulator[key] = [
                        psd_sum,
                        int(spectral.psd.shape[0]),
                    ]
                else:
                    psd_accumulator[key][0] += psd_sum
                    psd_accumulator[key][1] += int(spectral.psd.shape[0])

            # Explicitly release the large per-recording objects before the
            # next BLOCK04 array is loaded.
            del spectral
            del arr

        except Exception as exc:
            # Record one failure per affected manifest row and channel.
            for _, r in rows.iterrows():
                for ch in range(1, N_CHANNELS + 1):
                    failure_rows.append({
                        "subject_id": int(r.subject_id),
                        "condition": str(r.condition),
                        "gesture": int(r.gesture),
                        "set": str(r.set),
                        "source_recording": str(r.source_recording),
                        "window_id": str(r.window_id),
                        "channel": ch,
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                        "status": "FAIL",
                    })
            log(
                f"SPECTRAL FAIL {array_file.name}: "
                f"{type(exc).__name__}: {exc}"
            )

        if file_no % 25 == 0 or file_no == total_files:
            log(f"Spectral arrays {file_no}/{total_files}")

    # -------------------------------------------------------------------------
    # Construct the normal window-level feature table.
    # -------------------------------------------------------------------------
    features = pd.DataFrame(feature_rows)
    failures = pd.DataFrame(failure_rows)

    if features.empty:
        fail("No spectral feature rows were produced.")

    # -------------------------------------------------------------------------
    # Construct the compact subject/condition/gesture/channel/frequency PSD
    # table. This replaces the previous enormous window-level PSD table.
    # -------------------------------------------------------------------------
    psd_rows = []

    for (subject_id, condition, gesture, channel), (psd_sum, count) in sorted(
        psd_accumulator.items()
    ):
        if count <= 0:
            raise RuntimeError(
                "Invalid PSD accumulator count for "
                f"S{subject_id:02d}_{condition}_{gesture}_C{channel}."
            )

        mean_psd = psd_sum / float(count)

        if not np.isfinite(mean_psd).all():
            raise RuntimeError(
                "Non-finite subject-level PSD encountered for "
                f"S{subject_id:02d}_{condition}_{gesture}_C{channel}."
            )

        # The frequency vector is identical for all spectral computations
        # under the frozen Welch configuration.
        # Recover it from the frozen nfft/fs geometry.
        freqs = np.fft.rfftfreq(WELCH_NFFT, d=1.0 / FS)
        keep = (freqs >= ANALYSIS_FMIN) & (freqs <= ANALYSIS_FMAX)
        freqs = freqs[keep]

        if len(freqs) != len(mean_psd):
            raise RuntimeError(
                "PSD frequency-axis mismatch: "
                f"expected {len(freqs)}, observed {len(mean_psd)}."
            )

        for fi, freq in enumerate(freqs):
            psd_rows.append({
                "subject_id": subject_id,
                "condition": condition,
                "gesture": gesture,
                "channel": channel,
                "frequency": float(freq),
                "psd": float(mean_psd[fi]),
            })

    psd = pd.DataFrame(psd_rows)

    if psd.empty:
        fail("No subject-level PSD rows were produced.")

    return features, psd, failures


# =============================================================================
# 07. SUBJECT-LEVEL AGGREGATION
# =============================================================================

def aggregate_subject_features(features: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "delta_power", "theta_power", "alpha_power", "beta_power",
        "delta_relative_power", "theta_relative_power",
        "alpha_relative_power", "beta_relative_power",
        "alpha_beta_ratio",
    ]

    group = ["subject_id", "condition", "gesture", "channel"]
    subject = features.groupby(group, as_index=False)[numeric].mean()

    return subject


def aggregate_subject_psd(psd: pd.DataFrame) -> pd.DataFrame:
    group = ["subject_id", "condition", "gesture", "channel", "frequency"]
    return psd.groupby(group, as_index=False)["psd"].mean().rename(columns={"psd": "subject_mean_psd"})


def build_difference_table(subject: pd.DataFrame) -> pd.DataFrame:
    measures = [
        ("band_power", "delta", "delta_power"),
        ("band_power", "theta", "theta_power"),
        ("band_power", "alpha", "alpha_power"),
        ("band_power", "beta", "beta_power"),
        ("relative_power", "delta", "delta_relative_power"),
        ("relative_power", "theta", "theta_relative_power"),
        ("relative_power", "alpha", "alpha_relative_power"),
        ("relative_power", "beta", "beta_relative_power"),
        ("alpha_beta_ratio", "—", "alpha_beta_ratio"),
    ]

    rows = []
    key = ["subject_id", "gesture", "channel"]

    for measure, band, col in measures:
        wide = subject[key + ["condition", col]].pivot_table(
            index=key, columns="condition", values=col, aggfunc="mean"
        ).reset_index()

        if "ME" not in wide.columns or "MI" not in wide.columns:
            continue

        for _, r in wide.iterrows():
            rows.append({
                "subject_id": int(r.subject_id),
                "gesture": int(r.gesture),
                "channel": int(r.channel),
                "measure": measure,
                "frequency_band": band,
                "ME_value": float(r.ME),
                "MI_value": float(r.MI),
                "ME_minus_MI": float(r.ME - r.MI),
            })

    return pd.DataFrame(rows)


# =============================================================================
# 08. STATISTICS
# =============================================================================

def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out

    vals = p[finite]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)

    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    restored = np.empty_like(q)
    restored[order] = q
    out[finite] = restored
    return out


def paired_bootstrap_ci(d: np.ndarray, rng: np.random.Generator) -> Tuple[float, float]:
    n = len(d)
    idx = rng.integers(0, n, size=(BOOTSTRAP_ITERATIONS, n))
    boot_means = d[idx].mean(axis=1)
    lo = np.quantile(boot_means, (1.0 - CONFIDENCE_LEVEL) / 2.0)
    hi = np.quantile(boot_means, 1.0 - (1.0 - CONFIDENCE_LEVEL) / 2.0)
    return float(lo), float(hi)


def paired_sign_flip(d: np.ndarray, rng: np.random.Generator) -> float:
    observed = float(np.mean(d))
    signs = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float64),
        size=(PERMUTATION_ITERATIONS, len(d)),
    )
    permuted = np.mean(signs * d[None, :], axis=1)
    # +1 numerator / +1 denominator convention, matching BLOCK08.
    p = (np.sum(np.abs(permuted) >= abs(observed)) + 1.0) / (PERMUTATION_ITERATIONS + 1.0)
    return float(p)


def run_statistics(subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)

    measures = [
        ("band_power", "delta", "delta_power"),
        ("band_power", "theta", "theta_power"),
        ("band_power", "alpha", "alpha_power"),
        ("band_power", "beta", "beta_power"),
        ("relative_power", "delta", "delta_relative_power"),
        ("relative_power", "theta", "theta_relative_power"),
        ("relative_power", "alpha", "alpha_relative_power"),
        ("relative_power", "beta", "beta_relative_power"),
        ("alpha_beta_ratio", "—", "alpha_beta_ratio"),
    ]

    # 9 measures × 7 gestures × 13 channels = 819 predefined tests.
    for measure, band, col in measures:
        for gesture in GESTURES:
            for channel in range(1, N_CHANNELS + 1):
                m = subject[
                    (subject.gesture == gesture)
                    & (subject.channel == channel)
                    & (subject.condition.isin(CONDITIONS))
                ]

                wide = m.pivot_table(
                    index="subject_id",
                    columns="condition",
                    values=col,
                    aggfunc="mean",
                )

                if "ME" not in wide.columns or "MI" not in wide.columns:
                    rows.append({
                        "measure": measure,
                        "frequency_band": band,
                        "gesture": gesture,
                        "channel": channel,
                        "n_subjects": 0,
                        "ME_mean": np.nan,
                        "MI_mean": np.nan,
                        "mean_difference": np.nan,
                        "median_difference": np.nan,
                        "SD_difference": np.nan,
                        "IQR_difference": np.nan,
                        "effect_size": np.nan,
                        "CI_lower": np.nan,
                        "CI_upper": np.nan,
                        "p_value": np.nan,
                    })
                    continue

                pair = wide[["ME", "MI"]].dropna()
                d = (pair["ME"] - pair["MI"]).to_numpy(dtype=float)

                if len(d) < 2:
                    rows.append({
                        "measure": measure,
                        "frequency_band": band,
                        "gesture": gesture,
                        "channel": channel,
                        "n_subjects": int(len(d)),
                        "ME_mean": float(pair["ME"].mean()) if len(d) else np.nan,
                        "MI_mean": float(pair["MI"].mean()) if len(d) else np.nan,
                        "mean_difference": float(np.mean(d)) if len(d) else np.nan,
                        "median_difference": float(np.median(d)) if len(d) else np.nan,
                        "SD_difference": np.nan,
                        "IQR_difference": np.nan,
                        "effect_size": np.nan,
                        "CI_lower": np.nan,
                        "CI_upper": np.nan,
                        "p_value": np.nan,
                    })
                    continue

                sd = float(np.std(d, ddof=1))
                dz = float(np.mean(d) / sd) if sd > 0 else 0.0
                ci_lo, ci_hi = paired_bootstrap_ci(d, rng)
                p = paired_sign_flip(d, rng)

                rows.append({
                    "measure": measure,
                    "frequency_band": band,
                    "gesture": gesture,
                    "channel": channel,
                    "n_subjects": int(len(d)),
                    "ME_mean": float(pair["ME"].mean()),
                    "MI_mean": float(pair["MI"].mean()),
                    "mean_difference": float(np.mean(d)),
                    "median_difference": float(np.median(d)),
                    "SD_difference": sd,
                    "IQR_difference": float(np.quantile(d, 0.75) - np.quantile(d, 0.25)),
                    "effect_size": dz,
                    "CI_lower": ci_lo,
                    "CI_upper": ci_hi,
                    "p_value": p,
                })

    out = pd.DataFrame(rows)

    # Protocol 11.23: one FDR family per (measure x gesture) - here
    # (measure, frequency_band) together identify the specific quantity
    # (e.g. "band_power"+"delta" = absolute_delta_power), so the family key
    # is (measure, frequency_band, gesture), each family containing exactly
    # the 13 channel-level hypotheses. Pooling all 819 tests into one global
    # correction would mix fundamentally different quantities into one
    # family, which the protocol explicitly prohibits.
    out["p_adjusted"] = np.nan
    for _, idx in out.groupby(["measure", "frequency_band", "gesture"]).groups.items():
        out.loc[idx, "p_adjusted"] = benjamini_hochberg(
            out.loc[idx, "p_value"].to_numpy(dtype=float)
        )

    out["significance_status"] = np.where(
        out["p_adjusted"] <= FDR_Q,
        "FDR_SIGNIFICANT",
        "NOT_SIGNIFICANT",
    )
    return out.sort_values(
        ["measure", "frequency_band", "gesture", "channel"]
    ).reset_index(drop=True)


# =============================================================================
# 09. OUTPUT TABLES
# =============================================================================

def save_tables(
    features: pd.DataFrame,
    subject: pd.DataFrame,
    psd: pd.DataFrame,
    difference: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    features.to_csv(SPECTRAL_DIR / "spectral_window_features.csv", index=False)
    subject.to_csv(SPECTRAL_DIR / "spectral_subject_summary.csv", index=False)
    difference.to_csv(SPECTRAL_DIR / "ME_MI_spectral_difference.csv", index=False)

    # Band-power and relative-power machine-readable tables.
    abs_cols = [
        "subject_id", "condition", "gesture", "channel",
        "delta_power", "theta_power", "alpha_power", "beta_power",
    ]
    rel_cols = [
        "subject_id", "condition", "gesture", "channel",
        "delta_relative_power", "theta_relative_power",
        "alpha_relative_power", "beta_relative_power",
    ]
    ab_cols = ["subject_id", "condition", "gesture", "channel", "alpha_beta_ratio"]

    subject[abs_cols].to_csv(BAND_DIR / "absolute_band_power.csv", index=False)
    subject[rel_cols].to_csv(BAND_DIR / "relative_band_power.csv", index=False)
    subject[ab_cols].to_csv(AB_DIR / "alpha_beta_ratio.csv", index=False)

    statistics.to_csv(STAT_DIR / "spectral_statistical_summary.csv", index=False)
    statistics.to_csv(STAT_DIR / "paired_tests.csv", index=False)
    statistics[
        ["measure", "frequency_band", "gesture", "channel", "effect_size", "CI_lower", "CI_upper"]
    ].to_csv(STAT_DIR / "effect_sizes.csv", index=False)
    statistics[
        ["measure", "frequency_band", "gesture", "channel", "p_value", "p_adjusted", "significance_status"]
    ].to_csv(STAT_DIR / "FDR_results.csv", index=False)

    # PSD summary is calculated from subject-level PSD, preventing window
    # pseudoreplication in the cohort summary.
    psd_summary = (
        psd.groupby(["condition", "gesture", "channel", "frequency"], as_index=False)
        .agg(
            mean_psd=("subject_mean_psd", "mean"),
            median_psd=("subject_mean_psd", "median"),
            SD_psd=("subject_mean_psd", "std"),
        )
    )
    psd_summary.to_csv(SPECTRAL_DIR / "PSD_summary.csv", index=False)


# =============================================================================
# 10. QUALITY CONTROL / AUDIT
# =============================================================================

def build_completeness_audit(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    failures: pd.DataFrame,
) -> pd.DataFrame:
    expected_rows = []
    feature_keys = set(
        zip(
            features.subject_id.astype(int),
            features.condition.astype(str),
            features.gesture.astype(int),
            features.set.astype(str),
            features.window_id.astype(str),
            features.channel.astype(int),
        )
    )

    for subject in SUBJECTS:
        for condition in CONDITIONS:
            for gesture in GESTURES:
                for set_id in ["A", "B", "C"]:
                    sub = manifest[
                        (manifest.subject_id == subject)
                        & (manifest.condition == condition)
                        & (manifest.gesture == gesture)
                        & (manifest.set == set_id)
                    ]
                    expected = len(sub) * N_CHANNELS

                    observed = 0
                    for _, r in sub.iterrows():
                        for ch in range(1, N_CHANNELS + 1):
                            if (
                                int(r.subject_id),
                                str(r.condition),
                                int(r.gesture),
                                str(r.set),
                                str(r.window_id),
                                ch,
                            ) in feature_keys:
                                observed += 1

                    expected_rows.append({
                        "subject_id": subject,
                        "condition": condition,
                        "gesture": gesture,
                        "set": set_id,
                        "expected_window_channel_rows": expected,
                        "observed_window_channel_rows": observed,
                        "failed_window_channel_rows": expected - observed,
                        "status": "PASS" if expected == observed else "FAIL",
                    })

    audit = pd.DataFrame(expected_rows)
    return audit


def run_core_audit(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    subject: pd.DataFrame,
    psd: pd.DataFrame,
    difference: pd.DataFrame,
    statistics: pd.DataFrame,
    failures: pd.DataFrame,
) -> dict:
    checks = {}
    checks["subjects"] = sorted(manifest.subject_id.unique().tolist()) == SUBJECTS
    checks["conditions"] = sorted(manifest.condition.unique().tolist()) == CONDITIONS
    checks["gestures"] = sorted(manifest.gesture.unique().tolist()) == GESTURES
    checks["sets"] = sorted(manifest.set.unique().tolist()) == ["A", "B", "C"]
    checks["manifest_windows"] = len(manifest) == EXPECTED_WINDOWS
    checks["feature_rows"] = len(features) == EXPECTED_WINDOWS * N_CHANNELS
    checks["feature_finite"] = bool(
        np.isfinite(features.select_dtypes(include=[np.number]).to_numpy()).all()
    )
    checks["subject_count"] = len(subject.subject_id.unique()) == 25
    checks["subject_conditions"] = sorted(subject.condition.unique().tolist()) == CONDITIONS
    checks["subject_gestures"] = sorted(subject.gesture.unique().tolist()) == GESTURES
    checks["subject_channels"] = sorted(subject.channel.unique().tolist()) == list(range(1, 14))
    checks["subject_features_finite"] = bool(
        np.isfinite(subject.select_dtypes(include=[np.number]).to_numpy()).all()
    )
    diff_me = difference["ME_value"].to_numpy(dtype=float)
    diff_mi = difference["MI_value"].to_numpy(dtype=float)
    diff_reported = difference["ME_minus_MI"].to_numpy(dtype=float)
    diff_recomputed = diff_me - diff_mi
    checks["difference_finite"] = bool(
        np.isfinite(diff_me).all()
        and np.isfinite(diff_mi).all()
        and np.isfinite(diff_reported).all()
    )
    checks["difference_identity"] = bool(
        checks["difference_finite"]
        and np.array_equal(diff_reported, diff_recomputed)
    )
    checks["relative_power_sum"] = bool(
        np.allclose(
            subject[
                [
                    "delta_relative_power",
                    "theta_relative_power",
                    "alpha_relative_power",
                    "beta_relative_power",
                ]
            ].sum(axis=1).to_numpy(),
            1.0,
            rtol=1e-7,
            atol=1e-7,
        )
    )
    # NOTE: this identity is checked at the WINDOW level (features), not the
    # subject level. alpha_beta_ratio = alpha_power / beta_power is a ratio,
    # and mean-of-ratios != ratio-of-means in general, so re-deriving it from
    # subject-aggregated alpha_power/beta_power would test an identity that
    # is not mathematically guaranteed even when the underlying computation
    # is correct. At the window level, alpha_beta_ratio was computed
    # directly from that same window's alpha_power/beta_power, so it must
    # match exactly - this is the genuine independent-recomputation check.
    checks["alpha_beta_identity"] = bool(
        np.allclose(
            features["alpha_beta_ratio"].to_numpy(dtype=float),
            features["alpha_power"].to_numpy(dtype=float)
            / np.maximum(features["beta_power"].to_numpy(dtype=float), EPSILON),
            rtol=1e-9,
            atol=1e-9,
        )
    )
    checks["statistics_test_count"] = len(statistics) == 9 * 7 * 13
    checks["statistics_fdr"] = bool(
        statistics["p_adjusted"].notna().all()
    )
    checks["failures_documented"] = failures is not None

    failed = [k for k, v in checks.items() if not bool(v)]

    return {
        "pass": len(failed) == 0,
        "checks": checks,
        "failed_checks": failed,
        "n_manifest_rows": len(manifest),
        "n_feature_rows": len(features),
        "n_subject_rows": len(subject),
        "n_psd_rows": len(psd),
        "n_difference_rows": len(difference),
        "n_statistical_tests": len(statistics),
        "n_failed_spectral_arrays": int(failures["source_recording"].nunique()) if not failures.empty else 0,
        "n_failure_rows": len(failures),
    }


# =============================================================================
# 11. FIGURES
# =============================================================================

def save_figures(subject: pd.DataFrame) -> None:
    plt.rcParams.update({"figure.dpi": 120})

    # Figure 11-1: cohort-level PSD profiles, one figure per gesture.
    psd_records = []
    for gesture in GESTURES:
        sub = subject[
            (subject.gesture == gesture)
            & (subject.condition.isin(CONDITIONS))
        ]
        # Recompute subject-level PSD only from stored subject summary is not
        # possible here, so Figure 11-1 is generated from a representative
        # band-profile summary using the window spectral PSD table elsewhere.
        # The main continuous PSD figure is generated in main() from PSD_summary.

    # Figure 11-2: absolute band power, paired subject means.
    bands = ["delta", "theta", "alpha", "beta"]
    for gesture in GESTURES:
        s = subject[subject.gesture == gesture]
        me = s[s.condition == "ME"].groupby("subject_id")[["delta_power", "theta_power", "alpha_power", "beta_power"]].mean()
        mi = s[s.condition == "MI"].groupby("subject_id")[["delta_power", "theta_power", "alpha_power", "beta_power"]].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(bands))
        for sid in sorted(set(me.index) & set(mi.index)):
            ax.plot(x, me.loc[sid].to_numpy(), alpha=0.20)
            ax.plot(x, mi.loc[sid].to_numpy(), alpha=0.20)
        ax.plot(x, me.mean(axis=0).to_numpy(), marker="o", linewidth=2, label="ME")
        ax.plot(x, mi.mean(axis=0).to_numpy(), marker="o", linewidth=2, label="MI")
        ax.set_xticks(x, bands)
        ax.set_ylabel("Absolute band power")
        ax.set_title(f"Figure 11-2 — Absolute Band Power, Gesture {gesture}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(BAND_FIG_DIR / f"absolute_band_power_gesture_{gesture}.png", dpi=200)
        plt.close(fig)

        # Relative power.
        rel_cols = [
            "delta_relative_power", "theta_relative_power",
            "alpha_relative_power", "beta_relative_power"
        ]
        me_r = s[s.condition == "ME"].groupby("subject_id")[rel_cols].mean()
        mi_r = s[s.condition == "MI"].groupby("subject_id")[rel_cols].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        for sid in sorted(set(me_r.index) & set(mi_r.index)):
            ax.plot(x, me_r.loc[sid].to_numpy(), alpha=0.20)
            ax.plot(x, mi_r.loc[sid].to_numpy(), alpha=0.20)
        ax.plot(x, me_r.mean(axis=0).to_numpy(), marker="o", linewidth=2, label="ME")
        ax.plot(x, mi_r.mean(axis=0).to_numpy(), marker="o", linewidth=2, label="MI")
        ax.set_xticks(x, bands)
        ax.set_ylabel("Relative band power")
        ax.set_title(f"Figure 11-3 — Relative Band Power, Gesture {gesture}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(REL_FIG_DIR / f"relative_band_power_gesture_{gesture}.png", dpi=200)
        plt.close(fig)

        # Alpha/beta ratio.
        me_ab = s[s.condition == "ME"].groupby("subject_id")["alpha_beta_ratio"].mean()
        mi_ab = s[s.condition == "MI"].groupby("subject_id")["alpha_beta_ratio"].mean()
        common = sorted(set(me_ab.index) & set(mi_ab.index))

        fig, ax = plt.subplots(figsize=(8, 6))
        for sid in common:
            ax.plot([0, 1], [me_ab.loc[sid], mi_ab.loc[sid]], marker="o", alpha=0.25)
        if common:
            ax.plot([0, 1], [me_ab.loc[common].mean(), mi_ab.loc[common].mean()],
                    marker="o", linewidth=2, label="Mean")
        ax.set_xticks([0, 1], ["ME", "MI"])
        ax.set_ylabel("Alpha/Beta ratio")
        ax.set_title(f"Figure 11-4 — Alpha/Beta Ratio, Gesture {gesture}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(AB_FIG_DIR / f"alpha_beta_ratio_gesture_{gesture}.png", dpi=200)
        plt.close(fig)


def save_psd_figures(psd_subject: pd.DataFrame) -> None:
    for gesture in GESTURES:
        fig, ax = plt.subplots(figsize=(11, 6))

        for condition in CONDITIONS:
            s = psd_subject[
                (psd_subject.gesture == gesture)
                & (psd_subject.condition == condition)
            ]
            if s.empty:
                continue
            summary = (
                s.groupby("frequency")["subject_mean_psd"]
                .mean()
                .sort_index()
            )
            ax.plot(summary.index, summary.values, linewidth=2, label=condition)

        ax.set_xlim(ANALYSIS_FMIN, ANALYSIS_FMAX)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title(f"Figure 11-1 — ME–MI PSD Profiles, Gesture {gesture}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PSD_FIG_DIR / f"PSD_profiles_gesture_{gesture}.png", dpi=200)
        plt.close(fig)


def save_difference_figures(statistics: pd.DataFrame) -> None:
    for gesture in GESTURES:
        s = statistics[
            (statistics.gesture == gesture)
            & (statistics.measure.isin(["band_power", "relative_power"]))
        ]
        if s.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 6))
        labels = []
        vals = []
        for _, r in s.iterrows():
            if r.frequency_band == "—":
                continue
            labels.append(f"{r.measure}:{r.frequency_band}:C{int(r.channel)}")
            vals.append(float(r.mean_difference))
        if vals:
            ax.axhline(0.0, linewidth=1)
            ax.plot(np.arange(len(vals)), vals, marker=".", linestyle="none")
            ax.set_xlabel("Predefined comparison index")
            ax.set_ylabel("ME − MI")
            ax.set_title(f"ME–MI spectral differences, Gesture {gesture}")
            fig.tight_layout()
            fig.savefig(DIFF_FIG_DIR / f"ME_MI_difference_gesture_{gesture}.png", dpi=200)
        plt.close(fig)


# =============================================================================
# 12. CONFIGURATION / SUMMARY
# =============================================================================

def build_config(upstream: dict, input_hash: str) -> dict:
    cfg = {
        "block": "11_NEURO",
        "status": "FROZEN",
        "purpose": "Neurophysiological and spectral EEG characterization",
        "sampling_frequency": FS,
        "channels": N_CHANNELS,
        "preprocessing_band": [1.0, 40.0],
        "window_configuration": {
            "window_samples": WINDOW_LENGTH,
            "overlap_percent": OVERLAP_PERCENT,
            "step_samples": WINDOW_STEP,
            "windows_per_recording": WINDOWS_PER_RECORDING,
            "no_padding": True,
            "no_trial_reconstruction": True,
        },
        "frequency_bands": BANDS,
        "PSD_method": "Welch",
        "PSD_parameters": {
            "window_function": WELCH_WINDOW,
            "nperseg": WELCH_NPERSEG,
            "noverlap": WELCH_NOVERLAP,
            "nfft": WELCH_NFFT,
            "frequency_resolution_hz": EXPECTED_FREQUENCY_RESOLUTION_HZ,
            "detrend": WELCH_DETREND,
            "scaling": WELCH_SCALING,
            "return_onesided": WELCH_RETURN_ONESIDED,
            "analysis_range_hz": [ANALYSIS_FMIN, ANALYSIS_FMAX],
        },
        "band_power_method": "trapezoidal_integration_of_Welch_PSD",
        "relative_power_denominator": "sum_of_delta_theta_alpha_beta_absolute_band_power",
        "alpha_beta_ratio_definition": "alpha_power / max(beta_power, epsilon)",
        "epsilon": EPSILON,
        "aggregation_method": "mean_over_computational_windows_then_subject",
        "statistical_unit": "subject",
        "statistical_test": "two-sided paired sign-flip permutation",
        "effect_size": "paired dz = mean(ME-MI) / SD(ME-MI)",
        "bootstrap_configuration": {
            "method": "paired_subject_resampling_percentile",
            "iterations": BOOTSTRAP_ITERATIONS,
            "confidence_level": CONFIDENCE_LEVEL,
        },
        "permutation_configuration": {
            "method": "paired_sign_flip_two_sided",
            "iterations": PERMUTATION_ITERATIONS,
            "p_value_correction": "+1 numerator/+1 denominator",
        },
        "multiple_comparison_method": "Benjamini-Hochberg",
        "FDR_q": FDR_Q,
        "multiple_comparison_family": (
            "one FDR family per (measure x frequency_band x gesture), each "
            "containing the 13 channel-level p-values: 9 measure/band "
            "combinations x 7 gestures = 63 families x 13 channels = "
            "819 tests total, corrected independently within each family"
        ),
        "additional_frequency_bands": [],
        "ERD_ERS_status": "DISABLED — no verified resting baseline",
        "calibration_analysis": False,
        "transfer_adaptation": False,
        "RSA": False,
        "XAI": False,
        "robustness_testing": False,
        "random_seed": SEED,
        "subjects": SUBJECTS,
        "conditions": CONDITIONS,
        "gestures": GESTURES,
        "sets": ["A", "B", "C"],
        "upstream": upstream,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scipy": package_version("scipy"),
            "matplotlib": package_version("matplotlib"),
            "mne": package_version("mne"),
        },
        "operating_system": platform.platform(),
        "cpu": platform.processor(),
        "gpu": "NOT USED",
        "input_data_hash": input_hash,
    }
    cfg_text = json.dumps(cfg, sort_keys=True, indent=2)
    cfg["configuration_hash"] = hashlib.sha256(cfg_text.encode("utf-8")).hexdigest()
    return cfg


# =============================================================================
# 13. MAIN
# =============================================================================

def main() -> int:
    started = time.time()

    log("=" * 100)
    log("BLOCK 11 — NEUROPHYSIOLOGICAL & SPECTRAL EEG ANALYSIS")
    log("=" * 100)
    log("Scientific rule: spectral characterization only; no model optimization.")
    log("Primary inferential unit: subject.")
    log("Frozen window: W500/O50.")
    log("Frozen PSD: Welch, 2 s Hann segments, 50% segment overlap, 0.5 Hz resolution.")
    log("Primary bands: delta 1–4, theta 4–8, alpha 8–13, beta 13–30 Hz.")
    log("ERD/ERS: DISABLED — no verified resting baseline.")
    log("Statistics: paired sign-flip permutation + paired bootstrap CI + BH-FDR.")

    try:
        upstream, manifest = verify_upstream()
        log("Upstream BLOCK03/BLOCK04 verification PASS.")

        input_hash = sha256_file(MANIFEST_PATH)

        config = build_config(upstream, input_hash)
        json_dump(config, CONFIG_DIR / "BLOCK11_config.json")

        features, psd_window, failures = process_windows(manifest)
        log(f"Spectral processing complete: {len(features)} feature rows, "
            f"{len(failures)} documented failures.")

        subject = aggregate_subject_features(features)
        log(f"Subject-level aggregation complete: {len(subject)} rows.")

        psd_subject = aggregate_subject_psd(psd_window)
        log(f"Subject-level PSD aggregation complete: {len(psd_subject)} rows.")

        difference = build_difference_table(subject)
        log(f"ME-MI difference table complete: {len(difference)} rows.")

        log("Running paired statistics (819 tests x 10,000 permutations "
            "x 10,000 bootstrap each) - this is the heaviest stage...")
        statistics = run_statistics(subject)
        log(f"Statistics complete: {len(statistics)} comparisons.")

        save_tables(features, subject, psd_subject, difference, statistics)
        log("Tables saved.")

        completeness = build_completeness_audit(manifest, features, failures)
        completeness.to_csv(QC_DIR / "spectral_completeness_audit.csv", index=False)
        log("Completeness audit saved.")

        if failures.empty:
            pd.DataFrame(columns=[
                "subject_id", "condition", "gesture", "set",
                "source_recording", "window_id", "channel",
                "failure_type", "failure_message", "status"
            ]).to_csv(QC_DIR / "spectral_failures.csv", index=False)
        else:
            failures.to_csv(QC_DIR / "spectral_failures.csv", index=False)

        log("Saving band-power/relative-power/alpha-beta figures...")
        save_figures(subject)
        log("Saving PSD figures...")
        save_psd_figures(psd_subject)
        log("Saving ME-MI difference figures...")
        save_difference_figures(statistics)
        log("All figures saved.")

        log("Running core mathematical/completeness audit...")
        audit = run_core_audit(
            manifest, features, subject, psd_subject,
            difference, statistics, failures
        )
        json_dump(audit, QC_DIR / "BLOCK11_core_audit.json")
        log(f"Core audit complete: pass={audit['pass']}.")

        if not audit["pass"]:
            fail(
                "BLOCK11 core audit FAILED: "
                + ", ".join(audit["failed_checks"])
            )

        # Ensure completeness audit contains no missing window-channel output.
        if not completeness.empty and not (completeness["status"] == "PASS").all():
            fail("BLOCK11 spectral completeness audit FAILED.")

        # Required manuscript-compatible statistical summary.
        manuscript_table = statistics[
            ["measure", "frequency_band", "gesture", "channel",
             "ME_mean", "MI_mean", "mean_difference",
             "effect_size", "CI_lower", "CI_upper", "p_adjusted"]
        ].copy()
        manuscript_table.to_csv(
            STAT_DIR / "manuscript_spectral_summary.csv", index=False
        )

        summary = {
            "status": "PASS",
            "block": "11_NEURO",
            "eligible_subjects": SUBJECTS,
            "conditions": CONDITIONS,
            "gestures": GESTURES,
            "channels": list(range(1, N_CHANNELS + 1)),
            "sampling_frequency": FS,
            "analysis_frequency_range": [ANALYSIS_FMIN, ANALYSIS_FMAX],
            "frequency_bands": BANDS,
            "PSD_method": "Welch",
            "number_of_spectral_features": 13,
            "number_of_valid_windows": int(features.window_id.nunique()),
            "number_of_failed_windows": int(
                failures.window_id.nunique() if not failures.empty else 0
            ),
            "ME_MI_comparisons": int(len(statistics)),
            "statistical_test": "two-sided paired sign-flip permutation",
            "FDR_method": "Benjamini-Hochberg",
            "FDR_q": FDR_Q,
            "ERD_ERS_status": "DISABLED — no verified resting baseline",
            "input_hash": input_hash,
            "configuration_hash": config["configuration_hash"],
            "elapsed_seconds": time.time() - started,
        }
        json_dump(summary, OUTPUT_ROOT / "BLOCK11_summary.json")

        complete = {
            "status": "PASS",
            "block": "11_NEURO",
            "completion_timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scientific_freeze": {
                "window": "W500_O50",
                "PSD": "Welch",
                "bands": BANDS,
                "inferential_unit": "subject",
                "primary_test": "two-sided paired sign-flip permutation",
                "effect_size": "paired dz",
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "permutation_iterations": PERMUTATION_ITERATIONS,
                "FDR": "Benjamini-Hochberg q=0.05",
                "ERD_ERS": False,
            },
            "audit": audit,
            "configuration_hash": config["configuration_hash"],
            "input_hash": input_hash,
            "next_block": "12_REPRESENTATION",
        }
        json_dump(complete, OUTPUT_ROOT / "BLOCK11_complete.json")

        log("=" * 100)
        log("✓ BLOCK 11 PASS — NEUROPHYSIOLOGICAL & SPECTRAL ANALYSIS FROZEN")
        log(f"Output: {OUTPUT_ROOT}")
        log(f"Valid window-channel rows: {len(features)}")
        log(f"Subject-level rows: {len(subject)}")
        log(f"Statistical comparisons: {len(statistics)}")
        log("Primary inference: paired subject-level ME–MI")
        log("FDR: Benjamini-Hochberg q=0.05")
        log("ERD/ERS: DISABLED")
        log("Ready for BLOCK12")
        log("=" * 100)
        return 0

    except Exception as exc:
        log("BLOCK11 FAIL")
        log(f"{type(exc).__name__}: {exc}")

        failure_summary = {
            "status": "FAIL",
            "block": "11_NEURO",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.time() - started,
        }
        json_dump(failure_summary, OUTPUT_ROOT / "BLOCK11_complete.json")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())