# ==================================================================================================
# BLOCK 04 — TEMPORAL WINDOWING & STRICT LEAKAGE-FREE WINDOW FREEZE
# PAPER 2 — ME + MI PAIRED EEG COHORT
#
# PURPOSE
# -------
# 1. Consume ONLY BLOCK 03 preprocessed continuous EEG.
# 2. Preserve the frozen 20-candidate temporal configuration space.
# 3. DO NOT use any of the 25 final-cohort subjects to data-drive window selection.
# 4. Require one numerically pre-frozen global window configuration before final-cohort execution.
# 5. Generate final computational EEG windows using that frozen configuration.
# 6. Generate complete audit tables, manifests, diagnostics, figures, JSON, CSV and README files.
# 7. Resume safely after Colab/session interruption.
#
# STRICT LEAKAGE RULE
# -------------------
# All 25 subjects are members of the final Paper 2 cohort and all 25 become final LOSO
# test subjects. Therefore NONE of the 25 subjects may be used to select the final global
# window configuration.
#
# This implementation intentionally DOES NOT perform data-driven window optimization on
# the 25-subject cohort. The 20 configurations remain the predefined candidate space for
# protocol documentation and theoretical diagnostics only.
#
# BEFORE EXECUTION
# ----------------
# Set:
#     FINAL_WINDOW_SAMPLES
#     FINAL_OVERLAP_PERCENT
#
# to the numerically pre-frozen protocol values.
#
# The script will STOP if either value remains None.
#
# IMPORTANT
# ---------
# - No raw EEG modification.
# - No BLOCK 03 modification.
# - No timestamp/sample-index clock.
# - No padding.
# - No cross-recording / cross-gesture / cross-condition windows.
# - No feature extraction.
# - No final-model training.
# - No subject-specific window selection.
# - No ME/MI-specific window selection.
# - No ML/DL-specific window selection.
# - No LOSO test performance used for window selection.
# - No final-cohort validation score used for window selection.
# - Numerical gesture identifiers are 1–7.
#
# FINAL DATA FLOW
# ---------------
# BLOCK 03 continuous EEG
#        ↓
# pre-frozen global L*, O*
#        ↓
# BLOCK 04 final computational windows
#        ↓
# BLOCK 05 / DL / LOSO / SD / transfer / calibration / representation / XAI / robustness
# ==================================================================================================

import os
import re
import gc
import json
import time
import math
import hashlib
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==================================================================================================
# 0. FROZEN PATHS AND STUDY CONTRACT
# ==================================================================================================

DRIVE_ROOT = Path("/content/drive/MyDrive")
PAPER2_ROOT = DRIVE_ROOT / "EEG_Motor_Execution_Imagery" / "Paper2_EEG_BCI"

BLOCK01_ROOT = PAPER2_ROOT / "01_SUBJECT_MATCHING"
BLOCK02_ROOT = PAPER2_ROOT / "02_SIGNAL_QA"
BLOCK03_ROOT = PAPER2_ROOT / "03_PREPROCESSING"
BLOCK04_ROOT = PAPER2_ROOT / "04_WINDOWING"

PROCESSED_ROOT = BLOCK03_ROOT / "processed"
PROCESSED_ME_ROOT = PROCESSED_ROOT / "ME"
PROCESSED_MI_ROOT = PROCESSED_ROOT / "MI"

# --------------------------------------------------------------------------------------------------
# REQUIRED PRE-FROZEN FINAL WINDOW CONFIGURATION
# --------------------------------------------------------------------------------------------------
#
# DO NOT infer these values from the 25-subject EEG cohort.
# DO NOT select these values from final-cohort performance.
# DO NOT leave them None.
#
# Enter the exact values from the amended frozen protocol before execution.
#
FINAL_WINDOW_SAMPLES = 500
FINAL_OVERLAP_PERCENT = 50
# Example syntax ONLY; do not uncomment unless these are explicitly the frozen protocol values:
# FINAL_WINDOW_SAMPLES = 500
# FINAL_OVERLAP_PERCENT = 50

# --------------------------------------------------------------------------------------------------
# Fixed study constants
# --------------------------------------------------------------------------------------------------

WINDOWS = [200, 256, 300, 400, 500]
OVERLAPS = [0, 25, 50, 75]

FS = 125.0
N_SUBJECTS = 25
N_CHANNELS = 13
EXPECTED_SAMPLES = 5625
EXPECTED_RECORDINGS_PER_CONDITION = 25 * 7 * 3
EXPECTED_TOTAL_RECORDINGS = 25 * 7 * 3 * 2

GESTURE_CODE_TO_NAME = {
    1: "HOC",
    2: "IFM",
    3: "LFM",
    4: "MFM",
    5: "PH",
    6: "RFM",
    7: "TFM",
}
GESTURE_NAME_TO_CODE = {v: k for k, v in GESTURE_CODE_TO_NAME.items()}
GESTURE_ORDER = list(range(1, 8))

RANDOM_SEED = 42

BLOCK04_PROTOCOL_VERSION = "BLOCK04_STRICT_LEAKAGE_FREE_PREFROZEN_V1"
WINDOW_GENERATION_VERSION = "BLOCK04_FINAL_WINDOW_GENERATION_STRICT_LEAKAGE_FREE_V1"

# Resume-safe; never delete output automatically.
RESET_BLOCK04 = False
RESUME = True

# ==================================================================================================
# 1. OUTPUT DIRECTORIES
# ==================================================================================================

DIRS = {
    "root": BLOCK04_ROOT,
    "candidate": BLOCK04_ROOT / "candidate_space",
    "diagnostics": BLOCK04_ROOT / "diagnostics",
    "window_count": BLOCK04_ROOT / "diagnostics" / "window_count",
    "duration": BLOCK04_ROOT / "diagnostics" / "duration",
    "selection": BLOCK04_ROOT / "diagnostics" / "selection",
    "figures": BLOCK04_ROOT / "figures",
    "final_windows": BLOCK04_ROOT / "final_windows",
    "final_data": BLOCK04_ROOT / "final_windows" / "data",
    "manifests": BLOCK04_ROOT / "manifests",
    "checkpoints": BLOCK04_ROOT / "checkpoints",
    "unit_records": BLOCK04_ROOT / "checkpoints" / "unit_records",
    "logs": BLOCK04_ROOT / "logs",
}

if RESET_BLOCK04:
    raise RuntimeError(
        "RESET_BLOCK04=True is blocked. Block 04 is resume-safe and must not "
        "delete a partially completed experiment automatically."
    )

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# ==================================================================================================
# 2. LOGGER
# ==================================================================================================

LOG_FILE = DIRS["logs"] / "BLOCK04_WINDOWING.log"

logger = logging.getLogger("BLOCK04_WINDOWING")
logger.setLevel(logging.DEBUG)
if logger.handlers:
    logger.handlers.clear()

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)

logger.addHandler(fh)
logger.addHandler(ch)

START_TIME = time.time()

# ==================================================================================================
# 3. SAFE SERIALIZATION
# ==================================================================================================

def json_safe(obj):
    if obj is pd.NA or obj is pd.NaT:
        return None

    if isinstance(obj, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None

    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, Path):
        return str(obj)

    return obj


def atomic_json_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def atomic_csv_save(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_text_save(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(str(text), encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz_save(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)

# ==================================================================================================
# 4. PREFROZEN CONFIGURATION GATE
# ==================================================================================================

def validate_prefrozen_configuration():
    if FINAL_WINDOW_SAMPLES is None or FINAL_OVERLAP_PERCENT is None:
        raise RuntimeError(
            "\n"
            "BLOCK 04 STOPPED — FINAL WINDOW CONFIGURATION IS NOT NUMERICALLY FROZEN.\n\n"
            "Set FINAL_WINDOW_SAMPLES and FINAL_OVERLAP_PERCENT at the top of this script\n"
            "to the exact values approved in the pre-study frozen protocol.\n\n"
            "STRICT LEAKAGE RULE:\n"
            "The values MUST NOT be selected from the 25-subject final cohort,\n"
            "LOSO performance, transfer, calibration, representation, XAI, or robustness.\n"
        )

    try:
        w = int(FINAL_WINDOW_SAMPLES)
        o = int(FINAL_OVERLAP_PERCENT)
    except Exception as exc:
        raise RuntimeError(
            "FINAL_WINDOW_SAMPLES and FINAL_OVERLAP_PERCENT must be integers."
        ) from exc

    if w not in WINDOWS:
        raise RuntimeError(
            f"Invalid FINAL_WINDOW_SAMPLES={w}. "
            f"Allowed frozen candidates are {WINDOWS}."
        )

    if o not in OVERLAPS:
        raise RuntimeError(
            f"Invalid FINAL_OVERLAP_PERCENT={o}. "
            f"Allowed frozen candidates are {OVERLAPS}."
        )

    step = step_for(w, o)

    if step <= 0:
        raise RuntimeError(
            f"Invalid final step: window={w}, overlap={o}, step={step}"
        )

    return w, o, step

# step_for is defined below; the configuration gate is called after its definition.

# ==================================================================================================
# 5. BLOCK 03 INPUT VALIDATION
# ==================================================================================================

def parse_int(value):
    try:
        return int(value)
    except Exception:
        return None


def decode_scalar(z, key, default=None):
    if key not in z:
        return default

    v = z[key]

    try:
        if np.asarray(v).ndim == 0:
            return np.asarray(v).item()
    except Exception:
        pass

    return v


def canonical_gesture_code_from_name(name):
    s = str(name).strip().upper()

    if s in GESTURE_NAME_TO_CODE:
        return GESTURE_NAME_TO_CODE[s]

    try:
        n = int(float(s))
        if 1 <= n <= 7:
            return n
    except Exception:
        pass

    raise RuntimeError(f"Unknown gesture label from Block03: {name}")


def parse_processed_identity(path):
    """
    Parse the canonical Block03 filename.

    Block03 uses HOC/IFM/LFM/MFM/PH/RFM/TFM in filenames.
    Numeric gesture_code stored inside the NPZ is authoritative.
    """

    m = re.search(
        r"^S?(\d{1,2})_(ME|MI)_([A-Za-z0-9]+)_Set([ABC])_preprocessed\.npz$",
        Path(path).name,
        re.IGNORECASE,
    )

    if not m:
        return None

    subject = int(m.group(1))
    condition = m.group(2).upper()
    gesture_token = m.group(3).upper()
    set_id = m.group(4).upper()

    try:
        gesture_code = canonical_gesture_code_from_name(gesture_token)
    except Exception:
        return None

    gesture_name = GESTURE_CODE_TO_NAME[gesture_code]

    return subject, condition, gesture_code, gesture_name, set_id


def discover_block03_files():
    rows = []

    for condition, root in [
        ("ME", PROCESSED_ME_ROOT),
        ("MI", PROCESSED_MI_ROOT),
    ]:
        if not root.exists():
            raise RuntimeError(
                f"BLOCK 03 processed directory missing: {root}"
            )

        files = sorted(root.glob("*.npz"))

        for p in files:
            ident = parse_processed_identity(p)

            if ident is None:
                continue

            subject, cond, filename_code, filename_name, set_id = ident

            if cond != condition:
                continue

            with np.load(p, allow_pickle=False) as z:
                if "gesture_code" not in z.files:
                    raise RuntimeError(
                        f"BLOCK 03 file is missing numeric gesture_code: {p.name}"
                    )

                saved_code = int(np.asarray(z["gesture_code"]).item())

            if saved_code != filename_code:
                raise RuntimeError(
                    f"Gesture mismatch in BLOCK03 file {p.name}: "
                    f"filename={filename_code}, metadata={saved_code}"
                )

            rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "gesture_code": saved_code,
                    "gesture": GESTURE_CODE_TO_NAME[saved_code],
                    "set": set_id,
                    "path": str(p),
                    "recording": p.stem.replace("_preprocessed", ""),
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No valid BLOCK03 preprocessed recording files were found."
        )

    expected = {
        (s, c, g, st)
        for s in range(1, 26)
        for c in ("ME", "MI")
        for g in range(1, 8)
        for st in ("A", "B", "C")
    }

    observed = {
        (
            int(r.subject),
            str(r.condition),
            int(r.gesture_code),
            str(r.set),
        )
        for r in df.itertuples()
    }

    duplicates = df.duplicated(
        subset=["subject", "condition", "gesture_code", "set"],
        keep=False,
    )

    if duplicates.any():
        raise RuntimeError(
            "Duplicate BLOCK03 recording identities detected:\n"
            + df.loc[duplicates].to_string(index=False)
        )

    missing = sorted(expected - observed)
    extra = sorted(observed - expected)

    if missing or extra:
        raise RuntimeError(
            f"BLOCK03 recording contract failed. "
            f"Expected={len(expected)}, observed={len(observed)}, "
            f"missing={len(missing)}, extra={len(extra)}. "
            f"First missing={missing[:10]}, first extra={extra[:10]}"
        )

    if len(df) != EXPECTED_TOTAL_RECORDINGS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_RECORDINGS} BLOCK03 recordings; "
            f"found {len(df)}."
        )

    return df.sort_values(
        ["condition", "subject", "gesture_code", "set"]
    ).reset_index(drop=True)

# ==================================================================================================
# 6. WINDOW MATHEMATICS
# ==================================================================================================

def step_for(window, overlap):
    window = int(window)
    overlap = int(overlap)

    step = int(round(window * (1.0 - overlap / 100.0)))

    if step <= 0:
        raise RuntimeError(
            f"Invalid step for window={window}, overlap={overlap}"
        )

    return step


def window_starts(n_samples, window, overlap):
    n_samples = int(n_samples)
    window = int(window)
    overlap = int(overlap)

    step = step_for(window, overlap)

    if n_samples < window:
        return []

    return list(
        range(
            0,
            n_samples - window + 1,
            step,
        )
    )


def expected_window_count(n_samples, window, overlap):
    return len(
        window_starts(
            n_samples,
            window,
            overlap,
        )
    )

# Validate final configuration only AFTER step_for exists.
SELECTED_WINDOW, SELECTED_OVERLAP, SELECTED_STEP = (
    validate_prefrozen_configuration()
)

# ==================================================================================================
# 7. CREATE FROZEN CANDIDATE SPACE TABLES
# ==================================================================================================

candidate_rows = []

for w in WINDOWS:
    for o in OVERLAPS:
        step = step_for(w, o)
        candidate_rows.append(
            {
                "configuration_id": f"W{w}_O{o}",
                "window_samples": int(w),
                "window_seconds": float(w / FS),
                "overlap_percent": int(o),
                "overlap_fraction": float(o / 100.0),
                "step_samples": int(step),
                "candidate_status": (
                    "FINAL_PREFROZEN"
                    if w == SELECTED_WINDOW and o == SELECTED_OVERLAP
                    else "PREDEFINED_CANDIDATE_NOT_SELECTED"
                ),
                "selection_method": "data_independent_pre_study_freeze",
                "final_cohort_used_for_window_selection": False,
            }
        )

candidate_df = pd.DataFrame(candidate_rows)

if len(candidate_df) != 20:
    raise RuntimeError(
        f"Expected exactly 20 candidate configurations; found {len(candidate_df)}"
    )

if int(
    (
        (candidate_df["window_samples"] == SELECTED_WINDOW)
        & (candidate_df["overlap_percent"] == SELECTED_OVERLAP)
    ).sum()
) != 1:
    raise RuntimeError(
        "Exactly one candidate must correspond to the frozen final configuration."
    )

atomic_csv_save(
    candidate_df,
    BLOCK04_ROOT / "candidate_configurations.csv",
)

atomic_csv_save(
    candidate_df,
    DIRS["candidate"] / "candidate_configuration_space.csv",
)

# ==================================================================================================
# 8. DATASET DISCOVERY AND FINGERPRINT
# ==================================================================================================

recordings_df = discover_block03_files()


def load_block03_record(path):
    path = Path(path)

    with np.load(path, allow_pickle=False) as z:

        if "eeg" not in z:
            raise RuntimeError(
                f"Missing 'eeg' array: {path}"
            )

        eeg = np.asarray(
            z["eeg"],
            dtype=np.float32,
        )

        # Block03 canonical storage: samples × channels.
        if eeg.shape == (
            EXPECTED_SAMPLES,
            N_CHANNELS,
        ):
            eeg = eeg.T

        elif eeg.shape == (
            N_CHANNELS,
            EXPECTED_SAMPLES,
        ):
            eeg = eeg.copy()

        else:
            raise RuntimeError(
                f"Invalid EEG shape in {path.name}: {eeg.shape}; "
                f"expected either "
                f"({EXPECTED_SAMPLES}, {N_CHANNELS}) or "
                f"({N_CHANNELS}, {EXPECTED_SAMPLES})"
            )

        if not np.isfinite(eeg).all():
            raise RuntimeError(
                f"Non-finite EEG in Block03 output: {path.name}"
            )

        identity = parse_processed_identity(path)

        if identity is None:
            raise RuntimeError(
                f"Could not parse Block03 identity: {path.name}"
            )

        expected_subject, expected_condition, expected_code, _, expected_set = identity

        saved_code = decode_scalar(
            z,
            "gesture_code",
            None,
        )

        saved_name = decode_scalar(
            z,
            "gesture",
            None,
        )

        saved_condition = decode_scalar(
            z,
            "condition",
            None,
        )

        saved_subject = decode_scalar(
            z,
            "subject",
            None,
        )

        saved_set = decode_scalar(
            z,
            "set_code",
            None,
        )

        if saved_code is not None:
            if int(saved_code) != expected_code:
                raise RuntimeError(
                    f"Gesture code mismatch in {path.name}"
                )

        if saved_name is not None:
            saved_name_code = canonical_gesture_code_from_name(
                saved_name
            )

            if saved_name_code != expected_code:
                raise RuntimeError(
                    f"Gesture name/code mismatch in {path.name}"
                )

        if saved_condition is not None:
            if str(saved_condition).upper() != expected_condition:
                raise RuntimeError(
                    f"Condition mismatch in {path.name}"
                )

        if saved_subject is not None:
            saved_subject_text = str(saved_subject).strip()

            if saved_subject_text.upper().startswith("S"):
                saved_subject_text = saved_subject_text[1:]

            try:
                saved_subject_code = int(
                    float(saved_subject_text)
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid saved subject identifier in {path.name}: "
                    f"{saved_subject!r}"
                ) from exc

            if saved_subject_code != expected_subject:
                raise RuntimeError(
                    f"Subject mismatch in {path.name}"
                )

        if saved_set is not None:
            if str(saved_set).upper() != expected_set:
                raise RuntimeError(
                    f"Set mismatch in {path.name}"
                )

        if "usable_channel_mask" not in z:
            raise RuntimeError(
                f"Missing usable_channel_mask in {path.name}"
            )

        mask = np.asarray(
            z["usable_channel_mask"]
        ).astype(bool)

        if mask.shape != (
            N_CHANNELS,
        ):
            raise RuntimeError(
                f"Invalid usable_channel_mask in {path.name}: "
                f"{mask.shape}"
            )

        if not mask.any():
            raise RuntimeError(
                f"No usable EEG channels remain in {path.name}"
            )

        return eeg, mask


def block03_fingerprint(df):
    """
    Content-based SHA256 fingerprint of selected Block03 NPZ files.
    """

    h = hashlib.sha256()

    for p in df["path"].tolist():

        p = Path(p)

        h.update(
            str(p).encode("utf-8")
        )

        with open(
            p,
            "rb",
        ) as f:

            for chunk in iter(
                lambda: f.read(1024 * 1024),
                b"",
            ):
                h.update(chunk)

    return h.hexdigest()


DATASET_FINGERPRINT = block03_fingerprint(
    recordings_df
)

# ==================================================================================================
# 9. STRICT LEAKAGE AUDIT
# ==================================================================================================

FINAL_COHORT_SUBJECTS = list(
    range(
        1,
        N_SUBJECTS + 1,
    )
)

LEAKAGE_AUDIT = {
    "status": "PASS",
    "final_cohort_subject_count": N_SUBJECTS,
    "final_cohort_subjects": FINAL_COHORT_SUBJECTS,
    "development_subjects_used": [],
    "final_loso_subjects_used_for_window_selection": False,
    "final_cohort_data_used_for_window_selection": False,
    "final_cohort_labels_used_for_window_selection": False,
    "final_cohort_signal_quality_used_for_window_selection": False,
    "final_cohort_window_counts_used_for_selection": False,
    "final_loso_performance_used_for_selection": False,
    "transfer_results_used_for_selection": False,
    "calibration_results_used_for_selection": False,
    "representation_results_used_for_selection": False,
    "xai_results_used_for_selection": False,
    "robustness_results_used_for_selection": False,
    "subject_specific_selection": False,
    "condition_specific_selection": False,
    "ml_dl_specific_selection": False,
    "selection_method": "data_independent_pre_study_freeze",
    "frozen_window_samples": SELECTED_WINDOW,
    "frozen_overlap_percent": SELECTED_OVERLAP,
    "frozen_step_samples": SELECTED_STEP,
}

atomic_json_save(
    LEAKAGE_AUDIT,
    BLOCK04_ROOT / "BLOCK04_leakage_audit.json",
)

atomic_csv_save(
    pd.DataFrame(
        [
            {
                "audit_item": k,
                "value": v,
            }
            for k, v in LEAKAGE_AUDIT.items()
            if k not in {
                "final_cohort_subjects",
                "development_subjects_used",
            }
        ]
    ),
    DIRS["selection"] / "strict_leakage_audit.csv",
)

# ==================================================================================================
# 10. PREFROZEN CONFIGURATION ARTIFACTS
# ==================================================================================================

FINAL_CONFIG = {
    "block": "04_WINDOWING",
    "status": "FROZEN",
    "protocol_version": BLOCK04_PROTOCOL_VERSION,
    "window_generation_version": WINDOW_GENERATION_VERSION,
    "dataset_fingerprint": DATASET_FINGERPRINT,

    "candidate_window_lengths_samples": WINDOWS,
    "candidate_overlaps_percent": OVERLAPS,
    "total_candidates": 20,

    "selected_configuration_id": (
        f"W{SELECTED_WINDOW}_O{SELECTED_OVERLAP}"
    ),
    "selected_window_samples": SELECTED_WINDOW,
    "selected_window_seconds": float(
        SELECTED_WINDOW / FS
    ),
    "selected_overlap_fraction": float(
        SELECTED_OVERLAP / 100.0
    ),
    "selected_overlap_percent": SELECTED_OVERLAP,
    "selected_step_samples": SELECTED_STEP,

    "selection_metric": "data_independent_pre_study_freeze",
    "selection_scope": "pre_final_cohort",
    "selection_method": "data_independent_pre_study_freeze",

    "final_cohort_subject_count": N_SUBJECTS,
    "final_cohort_subjects": FINAL_COHORT_SUBJECTS,
    "final_loso_folds": N_SUBJECTS,

    "final_loso_subjects_used_for_window_selection": False,
    "final_cohort_used_for_window_selection": False,
    "development_subjects": [],
    "development_subject_count": 0,

    "me_mi_common": True,
    "ml_dl_common": True,
    "subject_specific_window_selection": False,
    "condition_specific_window_selection": False,

    "historical_250_50_primary": False,
    "historical_250_50_role": "diagnostic_compatibility_only",

    "tie_break_used": False,
    "tie_break_note": (
        "No final-cohort performance ranking is permitted. "
        "The numeric configuration was pre-frozen independently."
    ),

    "timestamp_used_as_clock": False,
    "sample_index_used_as_clock": False,

    "no_padding": True,
    "no_cross_recording": True,
    "no_cross_gesture": True,
    "no_cross_condition": True,

    "feature_extraction_applied": False,
    "model_fitting_during_window_generation": False,

    "frozen_at": datetime.now().isoformat(),
}

atomic_json_save(
    FINAL_CONFIG,
    BLOCK04_ROOT / "selected_window_config.json",
)

atomic_json_save(
    FINAL_CONFIG,
    BLOCK04_ROOT / "FINAL_WINDOW_CONFIG.json",
)

BLOCK04_CONFIG = {
    "block": "04_WINDOWING",
    "status": "FROZEN",
    "protocol_version": BLOCK04_PROTOCOL_VERSION,

    "candidate_window_lengths_samples": WINDOWS,
    "candidate_overlaps_percent": OVERLAPS,
    "total_candidates": 20,

    "sampling_rate_hz": FS,
    "expected_samples_per_recording": EXPECTED_SAMPLES,
    "expected_recordings_per_condition": EXPECTED_RECORDINGS_PER_CONDITION,
    "expected_total_recordings": EXPECTED_TOTAL_RECORDINGS,

    "selected_configuration_id": (
        f"W{SELECTED_WINDOW}_O{SELECTED_OVERLAP}"
    ),

    "selected_window_samples": SELECTED_WINDOW,
    "selected_window_seconds": float(
        SELECTED_WINDOW / FS
    ),
    "selected_overlap_percent": SELECTED_OVERLAP,
    "selected_overlap_fraction": float(
        SELECTED_OVERLAP / 100.0
    ),
    "selected_step_samples": SELECTED_STEP,

    "selection_method": "data_independent_pre_study_freeze",
    "selection_scope": "pre_final_cohort",

    "final_cohort_subject_count": N_SUBJECTS,
    "final_loso_folds": N_SUBJECTS,

    "final_cohort_used_for_window_selection": False,
    "final_loso_subjects_used_for_window_selection": False,

    "development_subjects_used": False,
    "development_subjects": [],

    "window_generation_rules": {
        "no_padding": True,
        "no_cross_recording": True,
        "no_cross_gesture": True,
        "no_cross_condition": True,
        "timestamp_used_as_clock": False,
        "sample_index_used_as_clock": False,
        "feature_extraction": False,
        "model_training": False,
        "same_configuration_me_mi": True,
        "same_configuration_ml_dl": True,
        "subject_specific_window_selection": False,
        "condition_specific_window_selection": False,
    },

    "prohibited_selection_inputs": [
        "final_cohort_eeg",
        "final_cohort_labels",
        "final_cohort_signal_quality",
        "final_cohort_window_counts",
        "final_loso_test_performance",
        "subject_dependent_final_performance",
        "ME_MI_transfer",
        "temporal_calibration",
        "representation_similarity",
        "XAI",
        "robustness",
        "manuscript_narrative",
    ],

    "dataset_fingerprint": DATASET_FINGERPRINT,

    "frozen_at": datetime.now().isoformat(),
}

atomic_json_save(
    BLOCK04_CONFIG,
    BLOCK04_ROOT / "BLOCK04_config.json",
)

# ==================================================================================================
# 11. PREFROZEN CONFIGURATION VERIFICATION
# ==================================================================================================

def verify_frozen_config():
    selected_id = (
        f"W{SELECTED_WINDOW}_O{SELECTED_OVERLAP}"
    )

    expected = {
        "selected_configuration_id": selected_id,
        "selected_window_samples": SELECTED_WINDOW,
        "selected_overlap_percent": SELECTED_OVERLAP,
        "selected_step_samples": SELECTED_STEP,
        "selection_method": "data_independent_pre_study_freeze",
        "final_cohort_used_for_window_selection": False,
        "final_loso_subjects_used_for_window_selection": False,
    }

    for filename in [
        "selected_window_config.json",
        "FINAL_WINDOW_CONFIG.json",
        "BLOCK04_config.json",
    ]:
        path = BLOCK04_ROOT / filename

        if not path.exists():
            raise RuntimeError(
                f"Missing required frozen configuration artifact: {path}"
            )

        obj = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        for key, value in expected.items():
            if obj.get(key) != value:
                raise RuntimeError(
                    f"Frozen configuration verification failed in "
                    f"{filename}: {key}={obj.get(key)!r}, expected {value!r}"
                )

verify_frozen_config()

# ==================================================================================================
# 12. CANDIDATE-THEORETICAL AUDIT — NO FINAL-COHORT OPTIMIZATION
# ==================================================================================================

candidate_math_rows = []

for w in WINDOWS:
    for o in OVERLAPS:

        step = step_for(
            w,
            o,
        )

        count = expected_window_count(
            EXPECTED_SAMPLES,
            w,
            o,
        )

        coverage_end = (
            (count - 1) * step + w
        )

        unused_tail = (
            EXPECTED_SAMPLES - coverage_end
        )

        candidate_math_rows.append(
            {
                "configuration_id": f"W{w}_O{o}",
                "window_samples": w,
                "window_seconds": w / FS,
                "overlap_percent": o,
                "overlap_fraction": o / 100.0,
                "step_samples": step,
                "recording_samples": EXPECTED_SAMPLES,
                "recording_duration_seconds": EXPECTED_SAMPLES / FS,
                "expected_window_count": count,
                "coverage_end_sample": coverage_end,
                "unused_tail_samples": unused_tail,
                "is_final_prefrozen": (
                    w == SELECTED_WINDOW
                    and o == SELECTED_OVERLAP
                ),
                "selection_used_final_cohort": False,
                "selection_status": (
                    "FROZEN_FINAL"
                    if (
                        w == SELECTED_WINDOW
                        and o == SELECTED_OVERLAP
                    )
                    else "CANDIDATE_ONLY"
                ),
            }
        )

candidate_math_df = pd.DataFrame(
    candidate_math_rows
)

atomic_csv_save(
    candidate_math_df,
    DIRS["candidate"] / "candidate_window_math_audit.csv",
)

atomic_csv_save(
    candidate_math_df,
    DIRS["duration"] / "candidate_duration_window_count_audit.csv",
)

# ==================================================================================================
# 13. FIGURE GENERATION
# ==================================================================================================
#
# Figures are protocol-space diagnostics.
# They do NOT optimize or select the final configuration.
#
# Figure 1: candidate duration comparison
# Figure 2: candidate window count comparison
# Figure 3: window count vs duration
# Figure 4: overlap vs window count for each window length
# Figure 5: final frozen configuration highlighted in candidate space
# Figure 6: final window geometry for one 5625-sample recording
# Figure 7: final window start/end positions
#
# No final-cohort performance appears in these figures.
# ==================================================================================================

FIGURE_DPI = 300


def save_figure_dual(fig, stem):
    png_path = DIRS["figures"] / f"{stem}.png"
    pdf_path = DIRS["figures"] / f"{stem}.pdf"

    fig.savefig(
        png_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    return png_path, pdf_path


# --------------------------------------------------------------------------------------------------
# Figure 1 — candidate durations
# --------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(13, 6.5)
)

labels = candidate_math_df["configuration_id"].tolist()
durations = candidate_math_df["window_seconds"].to_numpy(dtype=float)

ax.bar(
    np.arange(len(labels)),
    durations,
    alpha=0.85,
)

ax.axhline(
    SELECTED_WINDOW / FS,
    linestyle="--",
    linewidth=1.5,
    label=(
        f"Frozen final: W{SELECTED_WINDOW}_O{SELECTED_OVERLAP}"
    ),
)

ax.set_title(
    "BLOCK 04 — Predefined Candidate Window Durations"
)

ax.set_xlabel(
    "Candidate configuration"
)

ax.set_ylabel(
    "Window duration (s)"
)

ax.set_xticks(
    np.arange(len(labels))
)

ax.set_xticklabels(
    labels,
    rotation=60,
    ha="right",
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    loc="best"
)

fig.tight_layout()

save_figure_dual(
    fig,
    "Figure_BLOCK04_Candidate_Window_Durations",
)

# --------------------------------------------------------------------------------------------------
# Figure 2 — candidate window counts
# --------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(13, 6.5)
)

counts = candidate_math_df[
    "expected_window_count"
].to_numpy(dtype=int)

ax.bar(
    np.arange(len(labels)),
    counts,
    alpha=0.85,
)

final_count = expected_window_count(
    EXPECTED_SAMPLES,
    SELECTED_WINDOW,
    SELECTED_OVERLAP,
)

final_position = int(
    candidate_math_df.index[
        candidate_math_df["is_final_prefrozen"]
    ][0]
)

ax.scatter(
    [final_position],
    [final_count],
    marker="*",
    s=220,
    zorder=5,
    label=(
        f"Frozen final: W{SELECTED_WINDOW}_O{SELECTED_OVERLAP} "
        f"({final_count} windows/recording)"
    ),
)

ax.set_title(
    "BLOCK 04 — Theoretical Computational Window Count per Recording"
)

ax.set_xlabel(
    "Candidate configuration"
)

ax.set_ylabel(
    "Windows per 5,625-sample recording"
)

ax.set_xticks(
    np.arange(len(labels))
)

ax.set_xticklabels(
    labels,
    rotation=60,
    ha="right",
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    loc="best"
)

fig.tight_layout()

save_figure_dual(
    fig,
    "Figure_BLOCK04_Candidate_Window_Count",
)

# --------------------------------------------------------------------------------------------------
# Figure 3 — duration vs window count
# --------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9.5, 6.5)
)

ax.scatter(
    candidate_math_df["window_seconds"],
    candidate_math_df["expected_window_count"],
    s=75,
    alpha=0.85,
)

for _, r in candidate_math_df.iterrows():

    ax.annotate(
        r["configuration_id"],
        (
            float(r["window_seconds"]),
            int(r["expected_window_count"]),
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

ax.scatter(
    [SELECTED_WINDOW / FS],
    [final_count],
    marker="*",
    s=250,
    zorder=5,
    label=(
        f"Frozen final: W{SELECTED_WINDOW}_O{SELECTED_OVERLAP}"
    ),
)

ax.set_title(
    "BLOCK 04 — Window Duration vs Computational Window Count"
)

ax.set_xlabel(
    "Window duration (s)"
)

ax.set_ylabel(
    "Windows per recording"
)

ax.grid(
    alpha=0.25
)

ax.legend(
    loc="best"
)

fig.tight_layout()

save_figure_dual(
    fig,
    "Figure_BLOCK04_Duration_vs_WindowCount",
)

# --------------------------------------------------------------------------------------------------
# Figure 4 — overlap vs window count
# --------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9.5, 6.5)
)

for w in WINDOWS:

    subset = candidate_math_df[
        candidate_math_df["window_samples"] == w
    ].sort_values(
        "overlap_percent"
    )

    ax.plot(
        subset["overlap_percent"],
        subset["expected_window_count"],
        marker="o",
        label=f"{w} samples",
    )

ax.scatter(
    [SELECTED_OVERLAP],
    [final_count],
    marker="*",
    s=250,
    zorder=5,
    label=(
        f"Frozen final: {SELECTED_WINDOW} samples / "
        f"{SELECTED_OVERLAP}%"
    ),
)

ax.set_title(
    "BLOCK 04 — Effect of Overlap on Computational Window Count"
)

ax.set_xlabel(
    "Overlap (%)"
)

ax.set_ylabel(
    "Windows per recording"
)

ax.set_xticks(
    OVERLAPS
)

ax.grid(
    alpha=0.25
)

ax.legend(
    loc="best"
)

fig.tight_layout()

save_figure_dual(
    fig,
    "Figure_BLOCK04_Overlap_vs_WindowCount",
)

# --------------------------------------------------------------------------------------------------
# Figure 5 — candidate matrix / final freeze
# --------------------------------------------------------------------------------------------------

matrix = (
    candidate_math_df
    .pivot(
        index="window_samples",
        columns="overlap_percent",
        values="expected_window_count",
    )
    .reindex(
        index=WINDOWS,
        columns=OVERLAPS,
    )
)

fig, ax = plt.subplots(
    figsize=(8.5, 6.5)
)

im = ax.imshow(
    matrix.to_numpy(dtype=float),
    aspect="auto",
)

ax.set_title(
    "BLOCK 04 — Predefined Window Candidate Space\n"
    "Final Configuration Frozen Independently of the 25-Subject Cohort"
)

ax.set_xlabel(
    "Overlap (%)"
)

ax.set_ylabel(
    "Window length (samples)"
)

ax.set_xticks(
    np.arange(len(OVERLAPS))
)

ax.set_xticklabels(
    [str(x) for x in OVERLAPS]
)

ax.set_yticks(
    np.arange(len(WINDOWS))
)

ax.set_yticklabels(
    [str(x) for x in WINDOWS]
)

for i, w in enumerate(WINDOWS):

    for j, o in enumerate(OVERLAPS):

        value = int(
            matrix.loc[w, o]
        )

        is_final = (
            w == SELECTED_WINDOW
            and o == SELECTED_OVERLAP
        )

        text = str(value)

        if is_final:
            text = "★ " + text

        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontweight=(
                "bold"
                if is_final
                else "normal"
            ),
        )

cbar = fig.colorbar(
    im,
    ax=ax,
)

cbar.set_label(
    "Windows per recording"
)

fig.tight_layout()

save_figure_dual(
    fig,
    "Figure_BLOCK04_Candidate_Matrix_Frozen_Configuration",
)

# --------------------------------------------------------------------------------------------------
# Figure 6 — final window geometry
# --------------------------------------------------------------------------------------------------

starts_final = window_starts(
    EXPECTED_SAMPLES,
    SELECTED_WINDOW,
    SELECTED_OVERLAP,
)

ends_final = [
    s + SELECTED_WINDOW
    for s in starts_final
]

fig, ax = plt.subplots(
    figsize=(13, 4.5)
)

for idx, (start, end) in enumerate(
    zip(
        starts_final,
        ends_final,
    )
):

    ax.plot(
        [start, end],
        [idx, idx],
        linewidth=4,
    )

ax.set_title(
    f"BLOCK 04 — Final Frozen Window Geometry "
    f"(W={SELECTED_WINDOW}, O={SELECTED_OVERLAP}%)"
)

ax.set_xlabel(
    "Sample index"
)

ax.set_ylabel(
    "Computational window"
)

ax.set_xlim(
    0,
    EXPECTED_SAMPLES
)

ax.grid(
    axis="x",
    alpha=0.25,
)

fig.tight_layout()

save_figure_dual(
    fig,
    "Figure_BLOCK04_Final_Window_Geometry",
)

# --------------------------------------------------------------------------------------------------
# Figure 7 — final start/end positions
# --------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(13, 5.5)
)

ax.plot(
    np.arange(len(starts_final)),
    starts_final,
    marker="o",
    label="Start sample",
)

ax.plot(
    np.arange(len(ends_final)),
    ends_final,
    marker="o",
    label="End sample",
)

ax.set_title(
    f"BLOCK 04 — Final Window Start/End Positions"
)

ax.set_xlabel(
    "Window number"
)

ax.set_ylabel(
    "Sample index"
)

ax.grid(
    alpha=0.25
)

ax.legend(
    loc="best"
)

fig.tight_layout()

save_figure_dual(
    fig,
    "Figure_BLOCK04_Final_Window_Start_End",
)

# --------------------------------------------------------------------------------------------------
# Figure manifest
# --------------------------------------------------------------------------------------------------

figure_manifest = pd.DataFrame(
    [
        {
            "figure_id": "FIG04_01",
            "file_stem": "Figure_BLOCK04_Candidate_Window_Durations",
            "purpose": "Predefined candidate duration comparison",
            "uses_final_cohort_performance": False,
            "used_for_selection": False,
            "selected_configuration": f"W{SELECTED_WINDOW}_O{SELECTED_OVERLAP}",
        },
        {
            "figure_id": "FIG04_02",
            "file_stem": "Figure_BLOCK04_Candidate_Window_Count",
            "purpose": "Theoretical window count comparison",
            "uses_final_cohort_performance": False,
            "used_for_selection": False,
            "selected_configuration": f"W{SELECTED_WINDOW}_O{SELECTED_OVERLAP}",
        },
        {
            "figure_id": "FIG04_03",
            "file_stem": "Figure_BLOCK04_Duration_vs_WindowCount",
            "purpose": "Window duration versus computational burden",
            "uses_final_cohort_performance": False,
            "used_for_selection": False,
            "selected_configuration": f"W{SELECTED_WINDOW}_O{SELECTED_OVERLAP}",
        },
        {
            "figure_id": "FIG04_04",
            "file_stem": "Figure_BLOCK04_Overlap_vs_WindowCount",
            "purpose": "Overlap versus computational window count",
            "uses_final_cohort_performance": False,
            "used_for_selection": False,
            "selected_configuration": f"W{SELECTED_WINDOW}_O{SELECTED_OVERLAP}",
        },
        {
            "figure_id": "FIG04_05",
            "file_stem": "Figure_BLOCK04_Candidate_Matrix_Frozen_Configuration",
            "purpose": "Complete candidate matrix with frozen configuration",
            "uses_final_cohort_performance": False,
            "used_for_selection": False,
            "selected_configuration": f"W{SELECTED_WINDOW}_O{SELECTED_OVERLAP}",
        },
        {
            "figure_id": "FIG04_06",
            "file_stem": "Figure_BLOCK04_Final_Window_Geometry",
            "purpose": "Final frozen window geometry for one recording",
            "uses_final_cohort_performance": False,
            "used_for_selection": False,
            "selected_configuration": f"W{SELECTED_WINDOW}_O{SELECTED_OVERLAP}",
        },
        {
            "figure_id": "FIG04_07",
            "file_stem": "Figure_BLOCK04_Final_Window_Start_End",
            "purpose": "Final frozen window start/end positions",
            "uses_final_cohort_performance": False,
            "used_for_selection": False,
            "selected_configuration": f"W{SELECTED_WINDOW}_O{SELECTED_OVERLAP}",
        },
    ]
)

for row in figure_manifest.itertuples(index=False):

    png = (
        DIRS["figures"]
        / f"{row.file_stem}.png"
    )

    pdf = (
        DIRS["figures"]
        / f"{row.file_stem}.pdf"
    )

    if not png.exists() or not pdf.exists():
        raise RuntimeError(
            f"Missing figure output: {row.file_stem}"
        )

atomic_csv_save(
    figure_manifest,
    DIRS["figures"] / "BLOCK04_figure_manifest.csv",
)

# ==================================================================================================
# 14. ALL-RECORDING WINDOW COUNT AUDIT
# ==================================================================================================

window_count_rows = []

for row in recordings_df.itertuples(index=False):

    eeg, _mask = load_block03_record(
        row.path
    )

    recording_samples = int(
        eeg.shape[1]
    )

    for w, o in zip(
        [SELECTED_WINDOW],
        [SELECTED_OVERLAP],
    ):

        expected = expected_window_count(
            EXPECTED_SAMPLES,
            w,
            o,
        )

        actual = len(
            window_starts(
                recording_samples,
                w,
                o,
            )
        )

        status = (
            "PASS"
            if expected == actual
            else "FAIL"
        )

        window_count_rows.append(
            {
                "subject": int(row.subject),
                "condition": str(row.condition),
                "gesture": int(row.gesture_code),
                "gesture_code": int(row.gesture_code),
                "gesture_name": str(row.gesture),
                "set": str(row.set),
                "recording": str(row.recording),
                "window_samples": int(w),
                "overlap_percent": int(o),
                "step_samples": int(
                    step_for(w, o)
                ),
                "recording_samples": recording_samples,
                "expected_window_count": int(
                    expected
                ),
                "actual_window_count": int(
                    actual
                ),
                "difference": int(
                    actual - expected
                ),
                "status": status,
                "selection_method": "data_independent_pre_study_freeze",
            }
        )

window_count_audit_df = pd.DataFrame(
    window_count_rows
)

if len(window_count_audit_df) != EXPECTED_TOTAL_RECORDINGS:
    raise RuntimeError(
        "Final window-count audit row count failed."
    )

if (
    window_count_audit_df["status"] != "PASS"
).any():
    raise RuntimeError(
        "Final window-count audit FAILED."
    )

atomic_csv_save(
    window_count_audit_df,
    BLOCK04_ROOT / "window_count_audit.csv",
)

atomic_csv_save(
    window_count_audit_df,
    DIRS["window_count"] / "final_window_count_audit.csv",
)

# ==================================================================================================
# 15. DURATION AUDIT
# ==================================================================================================

duration_rows = []

for w in WINDOWS:

    for o in OVERLAPS:

        count = expected_window_count(
            EXPECTED_SAMPLES,
            w,
            o,
        )

        step = step_for(
            w,
            o,
        )

        coverage_end = (
            (count - 1) * step + w
        )

        duration_rows.append(
            {
                "configuration_id": f"W{w}_O{o}",
                "window_samples": w,
                "window_seconds": w / FS,
                "overlap_percent": o,
                "step_samples": step,
                "recording_samples": EXPECTED_SAMPLES,
                "recording_duration_seconds": EXPECTED_SAMPLES / FS,
                "expected_window_count": count,
                "coverage_end_sample": coverage_end,
                "unused_tail_samples": (
                    EXPECTED_SAMPLES - coverage_end
                ),
                "is_final_prefrozen": (
                    w == SELECTED_WINDOW
                    and o == SELECTED_OVERLAP
                ),
                "used_final_cohort_for_selection": False,
            }
        )

duration_df = pd.DataFrame(
    duration_rows
)

atomic_csv_save(
    duration_df,
    DIRS["duration"] / "window_duration_audit.csv",
)

# ==================================================================================================
# 16. FINAL WINDOW OUTPUT PATHS
# ==================================================================================================

FINAL_CONFIG_ID = (
    f"W{SELECTED_WINDOW}_O{SELECTED_OVERLAP}"
)

FINAL_WINDOW_RECORDS_DIR = (
    DIRS["final_data"]
    / FINAL_CONFIG_ID
)

FINAL_WINDOW_RECORDS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def final_record_output_path(row):
    return (
        FINAL_WINDOW_RECORDS_DIR
        / (
            f"S{int(row.subject):02d}_"
            f"{row.condition}_"
            f"{int(row.gesture_code)}_"
            f"Set{row.set}_windows.npz"
        )
    )


def final_record_meta_path(row):
    return (
        DIRS["unit_records"]
        / (
            f"FINAL_S{int(row.subject):02d}_"
            f"{row.condition}_"
            f"{int(row.gesture_code)}_"
            f"Set{row.set}.json"
        )
    )


def valid_final_record_file(
    path,
    expected_windows,
    expected_window,
):
    try:

        with np.load(
            path,
            allow_pickle=False,
        ) as z:

            required = {
                "eeg_windows",
                "start_samples",
                "end_samples",
                "gesture_code",
                "subject",
                "condition",
                "set_code",
                "window_samples",
                "overlap_percent",
                "step_samples",
            }

            if not required.issubset(
                set(z.files)
            ):
                return False

            x = np.asarray(
                z["eeg_windows"]
            )

            starts = np.asarray(
                z["start_samples"]
            )

            ends = np.asarray(
                z["end_samples"]
            )

            if x.shape != (
                expected_windows,
                N_CHANNELS,
                expected_window,
            ):
                return False

            if starts.shape != (
                expected_windows,
            ):
                return False

            if ends.shape != (
                expected_windows,
            ):
                return False

            if not np.isfinite(x).all():
                return False

            if not np.all(
                ends > starts
            ):
                return False

            if expected_windows > 0:

                if int(starts[0]) != 0:
                    return False

                if int(ends[-1]) > EXPECTED_SAMPLES:
                    return False

            if int(
                np.asarray(
                    z["window_samples"]
                ).item()
            ) != expected_window:
                return False

            if int(
                np.asarray(
                    z["overlap_percent"]
                ).item()
            ) != SELECTED_OVERLAP:
                return False

            if int(
                np.asarray(
                    z["step_samples"]
                ).item()
            ) != SELECTED_STEP:
                return False

            return True

    except Exception:
        return False

# ==================================================================================================
# 17. FINAL WINDOW GENERATION
# ==================================================================================================

def generate_final_record(row):

    eeg, usable_mask = load_block03_record(
        row.path
    )

    starts = window_starts(
        eeg.shape[1],
        SELECTED_WINDOW,
        SELECTED_OVERLAP,
    )

    expected_n = expected_window_count(
        EXPECTED_SAMPLES,
        SELECTED_WINDOW,
        SELECTED_OVERLAP,
    )

    if len(starts) != expected_n:
        raise RuntimeError(
            f"Final window count mismatch for {row.recording}: "
            f"{len(starts)} vs {expected_n}"
        )

    windows = np.stack(
        [
            eeg[
                :,
                s:s + SELECTED_WINDOW,
            ]
            for s in starts
        ],
        axis=0,
    ).astype(
        np.float32
    )

    ends = np.asarray(
        [
            s + SELECTED_WINDOW
            for s in starts
        ],
        dtype=np.int32,
    )

    starts_arr = np.asarray(
        starts,
        dtype=np.int32,
    )

    if windows.shape != (
        expected_n,
        N_CHANNELS,
        SELECTED_WINDOW,
    ):
        raise RuntimeError(
            f"Final window shape mismatch for "
            f"{row.recording}: {windows.shape}"
        )

    if not np.isfinite(
        windows
    ).all():
        raise RuntimeError(
            f"Non-finite final windows for "
            f"{row.recording}"
        )

    out = final_record_output_path(
        row
    )

    arrays = {
        "eeg_windows": windows,

        "subject": np.array(
            int(row.subject),
            dtype=np.int16,
        ),

        "condition": np.array(
            str(row.condition)
        ),

        "gesture_code": np.array(
            int(row.gesture_code),
            dtype=np.int16,
        ),

        "gesture": np.array(
            int(row.gesture_code),
            dtype=np.int16,
        ),

        "set_code": np.array(
            str(row.set)
        ),

        "sampling_rate_hz": np.array(
            FS,
            dtype=np.float32,
        ),

        "window_samples": np.array(
            SELECTED_WINDOW,
            dtype=np.int32,
        ),

        "window_seconds": np.array(
            SELECTED_WINDOW / FS,
            dtype=np.float32,
        ),

        "overlap_percent": np.array(
            SELECTED_OVERLAP,
            dtype=np.int16,
        ),

        "step_samples": np.array(
            SELECTED_STEP,
            dtype=np.int32,
        ),

        "start_samples": starts_arr,
        "end_samples": ends,

        "start_time_seconds": (
            starts_arr.astype(
                np.float64
            ) / FS
        ),

        "end_time_seconds": (
            ends.astype(
                np.float64
            ) / FS
        ),

        "source_recording": np.array(
            str(row.recording)
        ),

        "source_block03_path": np.array(
            str(row.path)
        ),

        "usable_channel_mask": usable_mask,

        "timestamp_used_as_clock": np.array(
            False
        ),

        "sample_index_used_as_clock": np.array(
            False
        ),

        "padding_applied": np.array(
            False
        ),

        "feature_extraction_applied": np.array(
            False
        ),

        "window_selection_used_final_cohort": np.array(
            False
        ),

        "selection_method": np.array(
            "data_independent_pre_study_freeze"
        ),
    }

    atomic_npz_save(
        out,
        arrays,
    )

    if not valid_final_record_file(
        out,
        expected_n,
        SELECTED_WINDOW,
    ):
        raise RuntimeError(
            f"Final window artifact validation failed: {out}"
        )

    return out, expected_n

# ==================================================================================================
# 18. FINAL WINDOW RESUME CHECKPOINT
# ==================================================================================================

final_complete = {}

for row in recordings_df.itertuples(
    index=False
):

    meta_path = final_record_meta_path(
        row
    )

    out_path = final_record_output_path(
        row
    )

    expected_n = expected_window_count(
        EXPECTED_SAMPLES,
        SELECTED_WINDOW,
        SELECTED_OVERLAP,
    )

    reusable = False

    if (
        RESUME
        and meta_path.exists()
        and out_path.exists()
    ):

        try:

            obj = json.loads(
                meta_path.read_text(
                    encoding="utf-8"
                )
            )

            reusable = (
                obj.get("status")
                == "COMPLETE"
                and obj.get(
                    "dataset_fingerprint"
                )
                == DATASET_FINGERPRINT
                and obj.get(
                    "selected_configuration_id"
                )
                == FINAL_CONFIG_ID
                and int(
                    obj.get(
                        "window_samples"
                    )
                )
                == SELECTED_WINDOW
                and int(
                    obj.get(
                        "overlap_percent"
                    )
                )
                == SELECTED_OVERLAP
                and obj.get(
                    "selection_method"
                )
                == "data_independent_pre_study_freeze"
                and obj.get(
                    "window_selection_used_final_cohort"
                )
                is False
                and valid_final_record_file(
                    out_path,
                    expected_n,
                    SELECTED_WINDOW,
                )
            )

        except Exception:
            reusable = False

    if reusable:

        final_complete[
            str(row.recording)
        ] = True

        logger.info(
            "FINAL RESUME-SKIP %s | %d/%d",
            row.recording,
            len(final_complete),
            len(recordings_df),
        )

        continue

    try:

        out_path, n_windows = (
            generate_final_record(
                row
            )
        )

        record = {
            "status": "COMPLETE",
            "recording": str(row.recording),
            "subject": int(row.subject),
            "condition": str(row.condition),
            "gesture": int(row.gesture_code),
            "gesture_code": int(row.gesture_code),
            "set": str(row.set),

            "output_file": str(
                out_path.relative_to(
                    BLOCK04_ROOT
                )
            ),

            "window_count": int(
                n_windows
            ),

            "window_samples": int(
                SELECTED_WINDOW
            ),

            "window_seconds": float(
                SELECTED_WINDOW / FS
            ),

            "overlap_percent": int(
                SELECTED_OVERLAP
            ),

            "step_samples": int(
                SELECTED_STEP
            ),

            "dataset_fingerprint":
                DATASET_FINGERPRINT,

            "selected_configuration_id":
                FINAL_CONFIG_ID,

            "selection_method":
                "data_independent_pre_study_freeze",

            "window_selection_used_final_cohort":
                False,

            "protocol_version":
                BLOCK04_PROTOCOL_VERSION,

            "timestamp_used_as_clock":
                False,

            "sample_index_used_as_clock":
                False,

            "padding_applied":
                False,

            "feature_extraction_applied":
                False,

            "generated_at":
                datetime.now().isoformat(),
        }

        atomic_json_save(
            record,
            meta_path,
        )

        final_complete[
            str(row.recording)
        ] = True

        atomic_json_save(
            {
                "block": "04_WINDOWING",
                "selected_configuration_id":
                    FINAL_CONFIG_ID,
                "completed_recordings":
                    len(final_complete),
                "total_recordings":
                    len(recordings_df),
                "updated":
                    datetime.now().isoformat(),
            },
            DIRS["checkpoints"]
            / "BLOCK04_final_window_checkpoint.json",
        )

        logger.info(
            "FINAL COMPLETE %s | %d/%d | windows=%d",
            row.recording,
            len(final_complete),
            len(recordings_df),
            n_windows,
        )

    except Exception as exc:

        logger.exception(
            "FINAL WINDOW FAILED %s: %s",
            row.recording,
            exc,
        )

    finally:
        gc.collect()

# ==================================================================================================
# 19. FINAL WINDOW MANIFEST
# ==================================================================================================

manifest_path = (
    BLOCK04_ROOT
    / "final_window_manifest.csv"
)

manifest_rows = []

for row in recordings_df.itertuples(
    index=False
):

    out_path = final_record_output_path(
        row
    )

    meta_path = final_record_meta_path(
        row
    )

    if not (
        out_path.exists()
        and meta_path.exists()
    ):
        continue

    try:

        with np.load(
            out_path,
            allow_pickle=False,
        ) as z:

            starts = np.asarray(
                z["start_samples"],
                dtype=np.int32,
            )

            ends = np.asarray(
                z["end_samples"],
                dtype=np.int32,
            )

            n = len(starts)

        for j in range(n):

            window_id = (
                f"S{int(row.subject):02d}_"
                f"{row.condition}_"
                f"{int(row.gesture_code)}_"
                f"Set{row.set}_"
                f"W{SELECTED_WINDOW}_"
                f"O{SELECTED_OVERLAP}_"
                f"Window{j + 1:04d}"
            )

            manifest_rows.append(
                {
                    "window_id":
                        window_id,

                    "subject":
                        int(row.subject),

                    "condition":
                        str(row.condition),

                    "gesture":
                        int(row.gesture_code),

                    "gesture_code":
                        int(row.gesture_code),

                    "gesture_name":
                        str(row.gesture),

                    "set":
                        str(row.set),

                    "source_recording":
                        str(row.recording),

                    "window_samples":
                        int(SELECTED_WINDOW),

                    "window_seconds":
                        float(
                            SELECTED_WINDOW / FS
                        ),

                    "overlap_percent":
                        int(SELECTED_OVERLAP),

                    "step_samples":
                        int(SELECTED_STEP),

                    "start_sample":
                        int(starts[j]),

                    "end_sample":
                        int(ends[j]),

                    "start_time_seconds":
                        float(
                            starts[j] / FS
                        ),

                    "end_time_seconds":
                        float(
                            ends[j] / FS
                        ),

                    "source_block03_file":
                        str(row.path),

                    "window_array_file":
                        str(
                            out_path.relative_to(
                                BLOCK04_ROOT
                            )
                        ),

                    "selection_method":
                        "data_independent_pre_study_freeze",

                    "window_selection_used_final_cohort":
                        False,
                }
            )

    except Exception as exc:

        logger.exception(
            "Could not materialize manifest for %s: %s",
            row.recording,
            exc,
        )

final_manifest_df = pd.DataFrame(
    manifest_rows
)

expected_windows_per_recording = (
    expected_window_count(
        EXPECTED_SAMPLES,
        SELECTED_WINDOW,
        SELECTED_OVERLAP,
    )
)

expected_final_windows = (
    EXPECTED_TOTAL_RECORDINGS
    * expected_windows_per_recording
)

if len(final_manifest_df) != expected_final_windows:

    atomic_json_save(
        {
            "block": "04_WINDOWING",
            "status": "PENDING",
            "reason":
                "Final window generation incomplete.",
            "selected_configuration_id":
                FINAL_CONFIG_ID,
            "selected_window_samples":
                SELECTED_WINDOW,
            "selected_overlap_percent":
                SELECTED_OVERLAP,
            "expected_recordings":
                EXPECTED_TOTAL_RECORDINGS,
            "completed_recordings":
                len(final_complete),
            "expected_windows":
                expected_final_windows,
            "manifest_windows":
                len(final_manifest_df),
            "missing_recordings": [
                str(r.recording)
                for r in recordings_df.itertuples(
                    index=False
                )
                if not final_record_output_path(
                    r
                ).exists()
            ],
            "updated":
                datetime.now().isoformat(),
        },
        BLOCK04_ROOT
        / "BLOCK04_summary.json",
    )

    raise RuntimeError(
        f"BLOCK 04 final generation incomplete: "
        f"{len(final_manifest_df)} windows vs "
        f"{expected_final_windows}. "
        f"Rerun to resume."
    )

required_manifest_columns = [
    "window_id",
    "subject",
    "condition",
    "gesture",
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
]

extras = [
    c
    for c in final_manifest_df.columns
    if c not in required_manifest_columns
]

final_manifest_df = final_manifest_df[
    required_manifest_columns + extras
]

atomic_csv_save(
    final_manifest_df,
    manifest_path,
)

# Also place a copy in manifests for convenient navigation.
atomic_csv_save(
    final_manifest_df,
    DIRS["manifests"] / "final_window_manifest.csv",
)

# ==================================================================================================
# 20. WINDOW GENERATION SUMMARY TABLES
# ==================================================================================================

summary_group = (
    final_manifest_df
    .groupby(
        [
            "condition",
            "gesture",
            "gesture_code",
        ],
        as_index=False,
    )
    .agg(
        recording_count=(
            "source_recording",
            "nunique",
        ),
        window_count=(
            "window_id",
            "count",
        ),
    )
)

atomic_csv_save(
    summary_group,
    BLOCK04_ROOT
    / "window_generation_summary.csv",
)

# Recording-level summary.
recording_summary = (
    final_manifest_df
    .groupby(
        [
            "subject",
            "condition",
            "gesture_code",
            "set",
            "source_recording",
        ],
        as_index=False,
    )
    .agg(
        window_count=(
            "window_id",
            "count",
        ),
        first_start_sample=(
            "start_sample",
            "min",
        ),
        last_end_sample=(
            "end_sample",
            "max",
        ),
    )
)

recording_summary["expected_window_count"] = (
    expected_windows_per_recording
)

recording_summary["count_difference"] = (
    recording_summary["window_count"]
    - recording_summary["expected_window_count"]
)

recording_summary["status"] = np.where(
    recording_summary["count_difference"] == 0,
    "PASS",
    "FAIL",
)

atomic_csv_save(
    recording_summary,
    DIRS["manifests"]
    / "final_recording_window_summary.csv",
)

# Subject-level summary.
subject_summary = (
    final_manifest_df
    .groupby(
        [
            "subject",
            "condition",
        ],
        as_index=False,
    )
    .agg(
        recording_count=(
            "source_recording",
            "nunique",
        ),
        window_count=(
            "window_id",
            "count",
        ),
    )
)

atomic_csv_save(
    subject_summary,
    DIRS["manifests"]
    / "subject_window_summary.csv",
)

# Condition-level summary.
condition_summary = (
    final_manifest_df
    .groupby(
        ["condition"],
        as_index=False,
    )
    .agg(
        recording_count=(
            "source_recording",
            "nunique",
        ),
        subject_count=(
            "subject",
            "nunique",
        ),
        window_count=(
            "window_id",
            "count",
        ),
    )
)

atomic_csv_save(
    condition_summary,
    DIRS["manifests"]
    / "condition_window_summary.csv",
)

# ==================================================================================================
# 21. FINAL INTEGRITY / LEAKAGE AUDIT
# ==================================================================================================

if final_manifest_df[
    "window_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate final window IDs detected."
    )

if final_manifest_df[
    "source_recording"
].isna().any():

    raise RuntimeError(
        "Missing source recording in final manifest."
    )

if (
    (
        final_manifest_df["start_sample"]
        < 0
    ).any()
    or (
        final_manifest_df["end_sample"]
        > EXPECTED_SAMPLES
    ).any()
    or (
        final_manifest_df["end_sample"]
        <= final_manifest_df["start_sample"]
    ).any()
):

    raise RuntimeError(
        "Invalid final window bounds."
    )

if (
    final_manifest_df[
        "window_samples"
    ]
    != SELECTED_WINDOW
).any():

    raise RuntimeError(
        "Manifest contains incorrect window length."
    )

if (
    final_manifest_df[
        "overlap_percent"
    ]
    != SELECTED_OVERLAP
).any():

    raise RuntimeError(
        "Manifest contains incorrect overlap."
    )

if (
    final_manifest_df[
        "step_samples"
    ]
    != SELECTED_STEP
).any():

    raise RuntimeError(
        "Manifest contains incorrect step."
    )

# All windows must use numerical gesture identifiers 1–7.
if not set(
    final_manifest_df[
        "gesture_code"
    ].astype(int).unique()
).issubset(
    set(GESTURE_ORDER)
):

    raise RuntimeError(
        "Invalid numerical gesture identifier found."
    )

# Exactly 25 subjects.
observed_subjects = sorted(
    final_manifest_df[
        "subject"
    ].astype(int).unique().tolist()
)

if observed_subjects != FINAL_COHORT_SUBJECTS:

    raise RuntimeError(
        f"Final subject coverage mismatch: "
        f"{observed_subjects}"
    )

# Exactly 1050 recording groups.
recording_groups = (
    final_manifest_df[
        [
            "subject",
            "condition",
            "gesture_code",
            "set",
            "source_recording",
        ]
    ]
    .drop_duplicates()
)

if len(recording_groups) != EXPECTED_TOTAL_RECORDINGS:

    raise RuntimeError(
        f"Final recording group count mismatch: "
        f"{len(recording_groups)} vs "
        f"{EXPECTED_TOTAL_RECORDINGS}"
    )

# Every recording must have exactly the theoretical number of windows.
counts = (
    final_manifest_df
    .groupby(
        [
            "subject",
            "condition",
            "gesture_code",
            "set",
            "source_recording",
        ],
        as_index=False,
    )
    .size()
    .rename(
        columns={
            "size":
                "actual_window_count"
        }
    )
)

if (
    counts[
        "actual_window_count"
    ]
    != expected_windows_per_recording
).any():

    raise RuntimeError(
        "Final per-recording window-count audit failed."
    )

# Check adjacent overlap behavior.
audit_overlap = []

for key, g in final_manifest_df.groupby(
    [
        "subject",
        "condition",
        "gesture_code",
        "set",
        "source_recording",
    ],
    sort=False,
):

    starts = g[
        "start_sample"
    ].to_numpy(
        dtype=int
    )

    ends = g[
        "end_sample"
    ].to_numpy(
        dtype=int
    )

    order = np.argsort(
        starts
    )

    starts = starts[order]
    ends = ends[order]

    overlap_pairs = 0

    for i in range(
        1,
        len(starts),
    ):

        if starts[i] < ends[i - 1]:
            overlap_pairs += 1

    expected_overlap_pairs = (
        len(starts) - 1
        if SELECTED_OVERLAP > 0
        else 0
    )

    overlap_status = (
        "PASS"
        if overlap_pairs
        == expected_overlap_pairs
        else "FAIL"
    )

    audit_overlap.append(
        {
            "subject": key[0],
            "condition": key[1],
            "gesture_code": key[2],
            "set": key[3],
            "source_recording": key[4],
            "window_count": len(starts),
            "overlapping_adjacent_pairs":
                overlap_pairs,
            "expected_overlapping_adjacent_pairs":
                expected_overlap_pairs,
            "overlap_percent":
                SELECTED_OVERLAP,
            "status":
                overlap_status,
        }
    )

overlap_audit_df = pd.DataFrame(
    audit_overlap
)

if (
    overlap_audit_df["status"]
    != "PASS"
).any():

    raise RuntimeError(
        "Final overlap audit failed."
    )

atomic_csv_save(
    overlap_audit_df,
    DIRS["selection"]
    / "final_overlap_audit.csv",
)

# Strict leakage artifact recheck.
final_leakage_check = {
    "final_cohort_subject_count":
        N_SUBJECTS,

    "final_cohort_used_for_window_selection":
        False,

    "final_loso_subjects_used_for_window_selection":
        False,

    "selection_method":
        "data_independent_pre_study_freeze",

    "selected_window_samples":
        SELECTED_WINDOW,

    "selected_overlap_percent":
        SELECTED_OVERLAP,

    "selected_step_samples":
        SELECTED_STEP,

    "development_subjects_used":
        [],

    "status":
        "PASS",
}

atomic_json_save(
    final_leakage_check,
    DIRS["selection"]
    / "final_strict_leakage_audit.json",
)

# ==================================================================================================
# 22. FINAL SUMMARY
# ==================================================================================================

final_recording_count = (
    final_manifest_df[
        [
            "subject",
            "condition",
            "gesture_code",
            "set",
            "source_recording",
        ]
    ]
    .drop_duplicates()
    .shape[0]
)

if final_recording_count != EXPECTED_TOTAL_RECORDINGS:

    raise RuntimeError(
        f"Final recording-count integrity failure: "
        f"{final_recording_count}/"
        f"{EXPECTED_TOTAL_RECORDINGS}"
    )

if len(final_manifest_df) != expected_final_windows:

    raise RuntimeError(
        f"Final window count failure: "
        f"{len(final_manifest_df)}/"
        f"{expected_final_windows}"
    )

# ==================================================================================================
# 23. FINAL BLOCK04 SUMMARY JSON
# ==================================================================================================

BLOCK04_SUMMARY = {
    "block": "04_WINDOWING",

    "status": "PASS",

    "protocol_version":
        BLOCK04_PROTOCOL_VERSION,

    "window_generation_version":
        WINDOW_GENERATION_VERSION,

    "dataset_fingerprint":
        DATASET_FINGERPRINT,

    "candidate_window_lengths":
        WINDOWS,

    "candidate_overlaps_percent":
        OVERLAPS,

    "total_candidates":
        20,

    "selection_method":
        "data_independent_pre_study_freeze",

    "selection_scope":
        "pre_final_cohort",

    "selected_configuration_id":
        FINAL_CONFIG_ID,

    "selected_window_samples":
        SELECTED_WINDOW,

    "selected_window_seconds":
        float(
            SELECTED_WINDOW / FS
        ),

    "selected_overlap_percent":
        SELECTED_OVERLAP,

    "selected_step_samples":
        SELECTED_STEP,

    "final_cohort_subject_count":
        N_SUBJECTS,

    "final_loso_folds":
        N_SUBJECTS,

    "development_subject_count":
        0,

    "development_subjects":
        [],

    "final_cohort_used_for_window_selection":
        False,

    "final_loso_subjects_used_for_window_selection":
        False,

    "me_common":
        True,

    "mi_common":
        True,

    "ml_dl_common":
        True,

    "subject_specific_window_selection":
        False,

    "condition_specific_window_selection":
        False,

    "expected_recordings":
        EXPECTED_TOTAL_RECORDINGS,

    "completed_final_recordings":
        final_recording_count,

    "expected_windows_per_recording":
        expected_windows_per_recording,

    "expected_final_windows":
        expected_final_windows,

    "actual_final_windows":
        int(
            len(final_manifest_df)
        ),

    "final_configuration_frozen":
        True,

    "padding_applied":
        False,

    "feature_extraction_applied":
        False,

    "model_fitting_final":
        False,

    "raw_data_modified":
        False,

    "block03_modified":
        False,

    "timestamp_used_as_clock":
        False,

    "sample_index_used_as_clock":
        False,

    "optimization_using_final_cohort":
        False,

    "optimization_figures_generated":
        True,

    "optimization_figures_are_selection_inputs":
        False,

    "optimization_figure_count":
        7,

    "figure_manifest":
        "figures/BLOCK04_figure_manifest.csv",

    "created_at":
        datetime.now().isoformat(),
}

atomic_json_save(
    BLOCK04_SUMMARY,
    BLOCK04_ROOT / "BLOCK04_summary.json",
)

# ==================================================================================================
# 24. COMPLETION MARKER — WRITTEN LAST
# ==================================================================================================

BLOCK04_COMPLETE = {
    "block": "04_WINDOWING",

    "status": "PASS",

    "complete": True,

    "selected_configuration_id":
        FINAL_CONFIG_ID,

    "selected_window_samples":
        SELECTED_WINDOW,

    "selected_overlap_percent":
        SELECTED_OVERLAP,

    "selected_step_samples":
        SELECTED_STEP,

    "candidate_count":
        20,

    "completed_candidate_count":
        20,

    "candidate_selection_method":
        "data_independent_pre_study_freeze",

    "final_cohort_used_for_window_selection":
        False,

    "final_loso_subjects_used_for_window_selection":
        False,

    "development_subjects_used":
        False,

    "expected_final_recordings":
        EXPECTED_TOTAL_RECORDINGS,

    "completed_final_recordings":
        final_recording_count,

    "expected_final_windows":
        expected_final_windows,

    "actual_final_windows":
        int(
            len(final_manifest_df)
        ),

    "dataset_fingerprint":
        DATASET_FINGERPRINT,

    "protocol_version":
        BLOCK04_PROTOCOL_VERSION,

    "final_configuration_frozen":
        True,

    "generated_at":
        datetime.now().isoformat(),
}

atomic_json_save(
    BLOCK04_COMPLETE,
    BLOCK04_ROOT / "BLOCK04_complete.json",
)

# ==================================================================================================
# 25. HUMAN-READABLE README
# ==================================================================================================

readme = f"""# BLOCK 04 — TEMPORAL WINDOWING

Status: PASS / FROZEN

## Leakage-control architecture

All 25 subjects belong to the final Paper 2 cohort and all 25 are eventual
LOSO test subjects.

No subject from the final cohort was used to select the final window.

Selection method:
DATA-INDEPENDENT PRE-STUDY FREEZE

No final-cohort validation score was used.
No LOSO score was used.
No subject-dependent performance was used.
No transfer result was used.
No calibration result was used.
No representation result was used.
No XAI result was used.
No robustness result was used.

## Cohort

25 paired subjects
ME + MI
7 gestures
Sets A/B/C
1050 recordings
13 EEG channels
125 Hz
5625 samples per recording

## Predefined candidate space

Window samples:
{WINDOWS}

Overlap:
{OVERLAPS}

Total candidates:
20

## Final frozen configuration

Configuration:
{FINAL_CONFIG_ID}

Window:
{SELECTED_WINDOW} samples

Duration:
{SELECTED_WINDOW / FS:.3f} seconds

Overlap:
{SELECTED_OVERLAP}%

Step:
{SELECTED_STEP} samples

Windows per 5625-sample recording:
{expected_windows_per_recording}

Total final computational EEG windows:
{expected_final_windows}

## Important

The 20 configurations document the predefined candidate space.

They are NOT optimized using the 25-subject final cohort.

The final numerical configuration was pre-frozen independently before final-cohort
analysis and is applied identically to every subject, ME, MI, ML, and DL branch.

## Window rules

No padding.
No cross-recording windows.
No cross-gesture windows.
No cross-condition windows.
No timestamp-based clock.
No sample-index clock.
No feature extraction.
No final-model training.
No subject-specific window selection.
No ME/MI-specific window selection.
No ML/DL-specific window selection.

## Gesture identifiers

Numerical gesture identifiers 1–7 are authoritative.

1 = HOC
2 = IFM
3 = LFM
4 = MFM
5 = PH
6 = RFM
7 = TFM

## Figures

The figures in `figures/` are protocol-space diagnostics.
They do not use final-cohort performance and do not determine the selected window.

## Downstream use

The same frozen configuration is used for:

- BLOCK 05 ML feature engineering;
- DL input;
- subject-dependent evaluation;
- 25-fold LOSO;
- ME→MI transfer;
- MI→ME transfer;
- temporal calibration;
- representation analysis;
- XAI;
- robustness;
- final statistics.

## Completion

BLOCK04_complete.json is written last.
"""

atomic_text_save(
    readme,
    BLOCK04_ROOT / "BLOCK04_README.md",
)

# ==================================================================================================
# 26. EXECUTION REPORT TABLE
# ==================================================================================================

execution_report = pd.DataFrame(
    [
        {
            "item": "Candidate configurations",
            "value": 20,
            "status": "PASS",
        },
        {
            "item": "Final window samples",
            "value": SELECTED_WINDOW,
            "status": "PASS",
        },
        {
            "item": "Final overlap percent",
            "value": SELECTED_OVERLAP,
            "status": "PASS",
        },
        {
            "item": "Final step samples",
            "value": SELECTED_STEP,
            "status": "PASS",
        },
        {
            "item": "Final subjects",
            "value": N_SUBJECTS,
            "status": "PASS",
        },
        {
            "item": "Final recordings",
            "value": final_recording_count,
            "status": "PASS",
        },
        {
            "item": "Expected final windows",
            "value": expected_final_windows,
            "status": "PASS",
        },
        {
            "item": "Actual final windows",
            "value": len(final_manifest_df),
            "status": "PASS",
        },
        {
            "item": "Final cohort used for selection",
            "value": False,
            "status": "PASS",
        },
        {
            "item": "LOSO subjects used for selection",
            "value": False,
            "status": "PASS",
        },
        {
            "item": "Development subjects used",
            "value": 0,
            "status": "PASS",
        },
        {
            "item": "Subject-specific selection",
            "value": False,
            "status": "PASS",
        },
        {
            "item": "ME/MI-specific selection",
            "value": False,
            "status": "PASS",
        },
        {
            "item": "ML/DL-specific selection",
            "value": False,
            "status": "PASS",
        },
        {
            "item": "Padding",
            "value": False,
            "status": "PASS",
        },
        {
            "item": "Feature extraction in Block04",
            "value": False,
            "status": "PASS",
        },
        {
            "item": "Raw data modified",
            "value": False,
            "status": "PASS",
        },
        {
            "item": "Block03 modified",
            "value": False,
            "status": "PASS",
        },
    ]
)

atomic_csv_save(
    execution_report,
    BLOCK04_ROOT / "BLOCK04_execution_report.csv",
)

# ==================================================================================================
# 27. FINAL CONSOLE STATUS
# ==================================================================================================

runtime = time.time() - START_TIME

print(
    "\n"
    + "=" * 110
)

print(
    "BLOCK 04 — FINAL STATUS"
)

print(
    "=" * 110
)

print(
    f"Candidate configurations : 20"
)

print(
    f"Selection method         : DATA-INDEPENDENT PRE-STUDY FREEZE"
)

print(
    f"Final configuration      : {FINAL_CONFIG_ID}"
)

print(
    f"Final window             : "
    f"{SELECTED_WINDOW} samples "
    f"({SELECTED_WINDOW / FS:.3f} s)"
)

print(
    f"Final overlap            : "
    f"{SELECTED_OVERLAP}%"
)

print(
    f"Final step               : "
    f"{SELECTED_STEP} samples"
)

print(
    f"Final subjects           : "
    f"{N_SUBJECTS}"
)

print(
    f"Final recordings         : "
    f"{final_recording_count}/"
    f"{EXPECTED_TOTAL_RECORDINGS}"
)

print(
    f"Windows/recording       : "
    f"{expected_windows_per_recording}"
)

print(
    f"Final windows            : "
    f"{len(final_manifest_df)}/"
    f"{expected_final_windows}"
)

print(
    f"Final cohort used select: NO"
)

print(
    f"LOSO subjects used      : NO"
)

print(
    f"Development subjects    : 0"
)

print(
    f"ME + MI common          : YES"
)

print(
    f"ML + DL common          : YES"
)

print(
    f"Subject-specific window : NO"
)

print(
    f"Padding                 : NO"
)

print(
    f"Feature extraction      : NO"
)

print(
    f"Raw modified            : NO"
)

print(
    f"BLOCK 03 modified       : NO"
)

print(
    f"Runtime                 : "
    f"{runtime / 60:.2f} min"
)

print(
    f"Output                  : "
    f"{BLOCK04_ROOT}"
)

print(
    "✓ BLOCK 04 PASS — STRICT LEAKAGE-FREE FINAL WINDOW CONFIGURATION FROZEN"
)

print(
    "=" * 110
)

logger.info(
    "BLOCK 04 PASS | selected=%s | final_windows=%d | "
    "strict_leakage_free=%s | runtime=%.2f min",
    FINAL_CONFIG_ID,
    len(final_manifest_df),
    True,
    runtime / 60,
)
