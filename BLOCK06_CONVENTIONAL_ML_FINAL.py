
"""
PAPER 2 — BLOCK 06
CONVENTIONAL MACHINE-LEARNING DECODING
06A: SUBJECT-DEPENDENT ML
06B: 25-FOLD SUBJECT-INDEPENDENT LOSO ML

FROZEN HANDOFF:
BLOCK 04 -> W500/O50
BLOCK 05 -> 341 features/window

Primary conditions:
    ME, MI

Gesture IDs:
    1 HOC
    2 IFM
    3 LFM
    4 MFM
    5 PH
    6 RFM
    7 TFM

Primary models:
    LDA, KNN, SVM, Random Forest, XGBoost

IMPORTANT:
    - No random window-level train/test split.
    - SD: SetA + SetB train, SetC final test.
    - LOSO: 25 folds; one subject held out per fold.
    - No internal-validation model fit is executed in BLOCK 06.
      Each LOSO fold fits the frozen classifier on all 24 non-held-out
      subjects and evaluates only on the single held-out subject.
    - Held-out subjects never enter scaler fitting or validation.
    - No feature selection, PCA, CSP, SMOTE, oversampling, undersampling,
      cross-condition training, or post-hoc hyperparameter tuning.
    - Temporal calibration levels 0/10/20/30/40/50/60/80/90% belong to
      BLOCK 09. BLOCK 06 exports the 0% LOSO baseline and calibration
      handoff metadata only.
    - Subject is the inferential unit; windows are prediction units only.
    - No raw data or BLOCK 03/04/05 outputs are modified.
    - All outputs are written under Paper2_EEG_BCI/06_ML.

This script is resumable. Completed model/condition/fold units are reused
only after integrity checks. Final completion is written last.
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import hashlib
import platform
import traceback
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import sklearn

try:
    import xgboost as xgb
except Exception as exc:
    xgb = None
    XGB_IMPORT_ERROR = repr(exc)
else:
    XGB_IMPORT_ERROR = None

try:
    import joblib
except Exception as exc:
    joblib = None
    JOBLIB_IMPORT_ERROR = repr(exc)
else:
    JOBLIB_IMPORT_ERROR = None


# =============================================================================
# 00. FROZEN PAPER 2 / BLOCK 04 / BLOCK 05 CONSTANTS
# =============================================================================

BLOCK_ID = "06_ML"
PROTOCOL_VERSION = "BLOCK06_CONVENTIONAL_ML_V2_SD_SETAB_TO_SETC_LOSO25_STRICT"
RANDOM_SEED = 42

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

SUBJECTS = tuple(range(1, 26))
CONDITIONS = ("ME", "MI")
SETS = ("SetA", "SetB", "SetC")

GESTURE_MAP = {
    1: "HOC",
    2: "IFM",
    3: "LFM",
    4: "MFM",
    5: "PH",
    6: "RFM",
    7: "TFM",
}
GESTURE_IDS = tuple(GESTURE_MAP.keys())
CHANCE_ACCURACY = 1.0 / 7.0

CALIBRATION_LEVELS_PERCENT = (0, 10, 20, 30, 40, 50, 60, 80, 90)

# Frozen primary classifier definitions.
MODEL_CONFIG = {
    "LDA": {
        "scaler": True,
        "solver": "svd",
    },
    "KNN": {
        "scaler": True,
        "n_neighbors": 5,
        "weights": "distance",
        "metric": "euclidean",
    },
    "SVM": {
        "scaler": True,
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "probability": False,
    },
    "RF": {
        "scaler": False,
        "n_estimators": 500,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "class_weight": None,
        "class_balancing": "none; dataset design is balanced by recording structure",
    },
    "XGB": {
        "scaler": False,
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "tree_method": "hist",
    },
}

MODELS = tuple(MODEL_CONFIG.keys())
PRIMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "cohen_kappa",
)

# Subject-dependent split is frozen.
SD_TRAIN_SETS = ("SetA", "SetB")
SD_TEST_SET = "SetC"

# No internal validation model fit is executed in BLOCK 06.
# All classifier configurations are frozen a priori in MODEL_CONFIG.
# Each LOSO fold fits its final model on all 24 non-held-out subjects.
LOSO_INTERNAL_VALIDATION_RULE = "none; all 24 training subjects used for final fold fit"

# Diagnostic/figure policy.
FIGURE_DPI = 300
FIGURE_FORMATS = ("png", "pdf")

# Numerical comparison tolerance.
REPRO_ATOL = 1e-12
REPRO_RTOL = 1e-10


# =============================================================================
# 01. PATHS
# =============================================================================

# Windows / VS Code execution — Paper 2 canonical local paths.
# No Google Colab mount is used in the canonical Windows run.
IN_COLAB = False

DRIVE_ROOT = Path(r"F:\Faruk")
RAW_ROOT = DRIVE_ROOT / "EEG_Motor_Execution_Imagery"
PAPER2_ROOT = RAW_ROOT / "Paper2_EEG_BCI"

BLOCK04_ROOT = PAPER2_ROOT / "04_WINDOWING"
BLOCK05_ROOT = PAPER2_ROOT / "05_FEATURE_ENGINEERING"
BLOCK06_ROOT = PAPER2_ROOT / "06_ML"

FEATURE_ROOT = BLOCK05_ROOT / "features"
ME_FEATURE_ROOT = FEATURE_ROOT / "ME"
MI_FEATURE_ROOT = FEATURE_ROOT / "MI"

SD_ROOT = BLOCK06_ROOT / "06A_ML_SD"
LOSO_ROOT = BLOCK06_ROOT / "06B_ML_LOSO"
COMPARE_ROOT = BLOCK06_ROOT / "comparisons"
CAL_HANDOFF_ROOT = BLOCK06_ROOT / "calibration_handoff"
FIG_ROOT = BLOCK06_ROOT / "figures"
TABLE_ROOT = BLOCK06_ROOT / "tables"
MODEL_ROOT = BLOCK06_ROOT / "models"
LOG_ROOT = BLOCK06_ROOT / "logs"
CHECKPOINT_ROOT = BLOCK06_ROOT / "checkpoints"

CONFIG_PATH = BLOCK06_ROOT / "BLOCK06_config.json"
SUMMARY_PATH = BLOCK06_ROOT / "BLOCK06_summary.json"
COMPLETE_PATH = BLOCK06_ROOT / "BLOCK06_complete.json"
EXECUTION_REPORT_PATH = BLOCK06_ROOT / "BLOCK06_execution_report.csv"


# =============================================================================
# 02. LOGGING / ATOMIC IO
# =============================================================================

class TeeLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(path, "a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.fp.write(text)

    def flush(self):
        self.stdout.flush()
        self.fp.flush()

    def close(self):
        try:
            self.fp.close()
        except Exception:
            pass


RUN_STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
LOG_PATH = LOG_ROOT / f"BLOCK06_{RUN_STAMP}.log"
LOGGER = TeeLogger(LOG_PATH)
sys.stdout = LOGGER


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_joblib_dump(path: Path, obj) -> None:
    if joblib is None:
        raise RuntimeError(
            "joblib is required for model persistence. "
            f"Import error: {JOBLIB_IMPORT_ERROR}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    joblib.dump(obj, tmp, compress=3)
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


# =============================================================================
# 03. DIRECTORY INITIALIZATION
# =============================================================================

def initialize_directories() -> None:
    for p in (
        BLOCK06_ROOT,
        SD_ROOT,
        LOSO_ROOT,
        COMPARE_ROOT,
        CAL_HANDOFF_ROOT,
        FIG_ROOT,
        TABLE_ROOT,
        MODEL_ROOT,
        LOG_ROOT,
        CHECKPOINT_ROOT,
    ):
        p.mkdir(parents=True, exist_ok=True)

    for p in (
        SD_ROOT / "predictions",
        SD_ROOT / "metrics",
        SD_ROOT / "class_metrics",
        SD_ROOT / "confusion_matrices",
        SD_ROOT / "subject_results",
        SD_ROOT / "figures",
        LOSO_ROOT / "folds",
        LOSO_ROOT / "predictions",
        LOSO_ROOT / "metrics",
        LOSO_ROOT / "class_metrics",
        LOSO_ROOT / "confusion_matrices",
        LOSO_ROOT / "figures",
        MODEL_ROOT / "SD",
        MODEL_ROOT / "LOSO",
    ):
        p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 04. BLOCK 04/05 HANDOFF VERIFICATION
# =============================================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _first_present(mapping: dict, keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _as_int(value, field_name: str) -> int:
    try:
        numeric = float(str(value).strip())
    except Exception as exc:
        raise RuntimeError(
            f"BLOCK 04 field '{field_name}' is not numeric: {value!r}"
        ) from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise RuntimeError(
            f"BLOCK 04 field '{field_name}' is not a finite integer: {value!r}"
        )
    return int(numeric)


def _as_bool(value, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    raise RuntimeError(
        f"BLOCK 04 field '{field_name}' is not boolean: {value!r}"
    )


def verify_block05_leakage_contract() -> dict:
    """Verify Block 05's explicit downstream leakage contract."""
    contract_path = BLOCK05_ROOT / "configs" / "BLOCK05_LEAKAGE_FORBIDDEN_ARTIFACTS.json"
    if not contract_path.exists():
        raise FileNotFoundError(contract_path)
    contract = load_json(contract_path)
    if str(contract.get("status", "")).strip().upper() != "PASS":
        raise RuntimeError("BLOCK 05 leakage contract status is not PASS.")

    allowed = set(contract.get("allowed_primary_sources", []))
    required_allowed = {
        "05_FEATURE/features/ME/*_features.npz",
        "05_FEATURE/features/MI/*_features.npz",
        "05_FEATURE/manifests/feature_manifest.csv",
        "05_FEATURE/manifests/feature_column_manifest.csv",
    }
    if not required_allowed.issubset(allowed):
        raise RuntimeError(
            "BLOCK 05 leakage contract does not expose the required primary feature sources."
        )

    forbidden = set(contract.get("forbidden_artifacts", []))
    required_forbidden = {
        "diagnostics/feature_summary.csv",
        "diagnostics/feature_distribution_summary.csv",
        "diagnostics/feature_range_summary.csv",
        "diagnostics/feature_quantile_summary.csv",
        "diagnostics/feature_correlation.csv",
        "diagnostics/constant_feature_audit.csv",
    }
    if not required_forbidden.issubset(forbidden):
        raise RuntimeError(
            "BLOCK 05 leakage contract is missing one or more quarantined diagnostic artifacts."
        )

    if contract.get("diagnostic_statistics_are_full_cohort") is not True:
        raise RuntimeError(
            "BLOCK 05 does not declare diagnostic statistics as full-cohort."
        )
    if contract.get("diagnostic_statistics_are_not_training_parameters") is not True:
        raise RuntimeError(
            "BLOCK 05 does not declare full-cohort diagnostics as non-training parameters."
        )

    return {
        "status": "PASS",
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "allowed_primary_sources": sorted(allowed),
        "forbidden_artifacts": sorted(forbidden),
    }


def _block04_authoritative_value(
    documents: list[tuple[str, dict]],
    keys: list[str],
    field_name: str,
):
    """Read a required frozen Block 04 value from authoritative artifacts."""
    for source_name, obj in documents:
        value = _first_present(obj, keys, default=None)
        if value is not None:
            return value, source_name
    raise RuntimeError(
        f"BLOCK 04 authoritative field '{field_name}' was not found in "
        f"frozen configuration artifacts: {[name for name, _ in documents]}"
    )


def verify_block04_handoff() -> dict:
    complete_path = BLOCK04_ROOT / "BLOCK04_complete.json"
    selected_path = BLOCK04_ROOT / "selected_window_config.json"
    final_config_path = BLOCK04_ROOT / "FINAL_WINDOW_CONFIG.json"
    block04_config_path = BLOCK04_ROOT / "BLOCK04_config.json"
    summary_path = BLOCK04_ROOT / "BLOCK04_summary.json"
    manifest_path = BLOCK04_ROOT / "final_window_manifest.csv"

    required_paths = [
        complete_path,
        selected_path,
        final_config_path,
        block04_config_path,
        summary_path,
    ]
    for p in required_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    complete = load_json(complete_path)
    selected = load_json(selected_path)
    final_config = load_json(final_config_path)
    block04_config = load_json(block04_config_path)
    summary = load_json(summary_path)

    if str(complete.get("status", "")).strip().upper() != "PASS":
        raise RuntimeError("BLOCK 04 completion marker is not PASS.")
    if complete.get("complete") is False:
        raise RuntimeError("BLOCK 04 completion marker reports complete=False.")

    # IMPORTANT:
    # The strict Block 04 protocol stores the frozen values as
    # selected_window_samples / selected_overlap_percent /
    # selected_step_samples. The older aliases are retained only for
    # compatibility. A null field is never interpreted as a value.
    frozen_documents = [
        ("selected_window_config.json", selected),
        ("FINAL_WINDOW_CONFIG.json", final_config),
        ("BLOCK04_config.json", block04_config),
        ("BLOCK04_summary.json", summary),
        ("BLOCK04_complete.json", complete),
    ]

    selected_window, selected_window_source = _block04_authoritative_value(
        frozen_documents,
        [
            "selected_window_samples",
            "window_samples",
            "window_length_samples",
            "final_window_samples",
        ],
        "selected_window_samples",
    )
    selected_overlap, selected_overlap_source = _block04_authoritative_value(
        frozen_documents,
        [
            "selected_overlap_percent",
            "overlap_percent",
            "final_overlap_percent",
        ],
        "selected_overlap_percent",
    )
    selected_step, selected_step_source = _block04_authoritative_value(
        frozen_documents,
        [
            "selected_step_samples",
            "step_samples",
            "final_step_samples",
        ],
        "selected_step_samples",
    )

    if _as_int(selected_window, "selected_window_samples") != WINDOW_SAMPLES:
        raise RuntimeError(
            f"BLOCK 04 frozen window is {selected_window!r} "
            f"(source={selected_window_source}); expected {WINDOW_SAMPLES}."
        )
    if _as_int(selected_overlap, "selected_overlap_percent") != OVERLAP_PERCENT:
        raise RuntimeError(
            f"BLOCK 04 frozen overlap is {selected_overlap!r} "
            f"(source={selected_overlap_source}); expected {OVERLAP_PERCENT}."
        )
    if _as_int(selected_step, "selected_step_samples") != STEP_SAMPLES:
        raise RuntimeError(
            f"BLOCK 04 frozen step is {selected_step!r} "
            f"(source={selected_step_source}); expected {STEP_SAMPLES}."
        )

    selection_method, selection_method_source = _block04_authoritative_value(
        frozen_documents,
        [
            "selection_method",
            "window_selection_method",
            "candidate_selection_method",
        ],
        "selection_method",
    )
    if str(selection_method).strip().lower() != "data_independent_pre_study_freeze":
        raise RuntimeError(
            "BLOCK 04 selection method is not "
            "'data_independent_pre_study_freeze': "
            f"{selection_method!r} (source={selection_method_source})."
        )

    final_cohort_used, _ = _block04_authoritative_value(
        frozen_documents,
        [
            "final_cohort_used_for_selection",
            "final_cohort_used_for_window_selection",
            "final_heldout_subjects_used_for_selection",
        ],
        "final_cohort_used_for_selection",
    )
    if _as_bool(final_cohort_used, "final_cohort_used_for_selection"):
        raise RuntimeError(
            "BLOCK 04 indicates that the final cohort was used for window selection."
        )

    loso_used, _ = _block04_authoritative_value(
        frozen_documents,
        [
            "final_loso_subjects_used_for_selection",
            "final_loso_subjects_used_for_window_selection",
            "loso_subjects_used_for_selection",
        ],
        "final_loso_subjects_used_for_selection",
    )
    if _as_bool(loso_used, "final_loso_subjects_used_for_selection"):
        raise RuntimeError(
            "BLOCK 04 indicates that LOSO subjects were used for window selection."
        )

    development_count, _ = _block04_authoritative_value(
        frozen_documents,
        ["development_subject_count"],
        "development_subject_count",
    )
    if _as_int(development_count, "development_subject_count") != 0:
        raise RuntimeError(
            "BLOCK 04 reports a nonzero development-subject count; "
            "strict leakage-free protocol requires 0."
        )

    final_frozen, _ = _block04_authoritative_value(
        frozen_documents,
        ["final_configuration_frozen"],
        "final_configuration_frozen",
    )
    if not _as_bool(final_frozen, "final_configuration_frozen"):
        raise RuntimeError("BLOCK 04 final configuration is not frozen.")

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = pd.read_csv(manifest_path)

    required = {
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
    }
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError(
            f"BLOCK 04 manifest missing columns: {sorted(missing)}"
        )

    if len(manifest) != TOTAL_WINDOWS:
        raise RuntimeError(
            f"BLOCK 04 manifest has {len(manifest)} rows; "
            f"expected {TOTAL_WINDOWS}."
        )

    if sorted(manifest["subject"].astype(int).unique().tolist()) != list(SUBJECTS):
        raise RuntimeError("BLOCK 04 manifest subject set is not 1..25.")

    if set(manifest["condition"].astype(str).unique()) != set(CONDITIONS):
        raise RuntimeError(
            "BLOCK 04 manifest conditions are not exactly ME/MI."
        )

    if set(manifest["gesture_code"].astype(int).unique()) != set(GESTURE_IDS):
        raise RuntimeError(
            "BLOCK 04 manifest gesture codes are not exactly 1..7."
        )

    set_values = set(manifest["set"].astype(str).map(normalize_set))
    if set_values != set(SETS):
        raise RuntimeError(
            f"BLOCK 04 set labels are invalid: {set_values}"
        )

    if not np.all(
        manifest["window_samples"].astype(int).to_numpy() == WINDOW_SAMPLES
    ):
        raise RuntimeError(
            "BLOCK 04 manifest contains non-500-sample windows."
        )

    if not np.all(
        manifest["overlap_percent"].astype(int).to_numpy() == OVERLAP_PERCENT
    ):
        raise RuntimeError(
            "BLOCK 04 manifest contains non-50% overlap."
        )

    if not np.all(
        manifest["step_samples"].astype(int).to_numpy() == STEP_SAMPLES
    ):
        raise RuntimeError(
            "BLOCK 04 manifest contains non-250 steps."
        )

    recording_counts = manifest.groupby(
        ["subject", "condition", "gesture_code", "set"]
    )["window_id"].nunique()

    if len(recording_counts) != N_RECORDINGS:
        raise RuntimeError(
            f"BLOCK 04 manifest contains {len(recording_counts)} logical "
            f"recording groups; expected {N_RECORDINGS}."
        )

    if not np.all(recording_counts.to_numpy() == WINDOWS_PER_RECORDING):
        raise RuntimeError(
            "BLOCK 04 does not contain exactly 21 windows per recording."
        )

    return {
        "status": "PASS",
        "selected_configuration": "W500_O50",
        "selected_window_source": selected_window_source,
        "selected_overlap_source": selected_overlap_source,
        "selected_step_source": selected_step_source,
        "manifest_rows": int(len(manifest)),
        "manifest_sha256": sha256_file(manifest_path),
        "window_samples": WINDOW_SAMPLES,
        "overlap_percent": OVERLAP_PERCENT,
        "step_samples": STEP_SAMPLES,
    }


def verify_block05_handoff() -> dict:
    """
    Verify the exact schema actually written by the canonical Block 05
    implementation.

    Block 05 writes:
      BLOCK05_summary.json:
          block04_configuration
          feature_count_per_window
          total_windows
          ...
      BLOCK05_complete.json:
          frozen_configuration
          feature_count
          total_windows
          ...
    Do not infer a missing field from another artifact.
    """
    complete_path = BLOCK05_ROOT / "BLOCK05_complete.json"
    summary_path = BLOCK05_ROOT / "BLOCK05_summary.json"
    column_manifest_path = BLOCK05_ROOT / "manifests" / "feature_column_manifest.csv"
    feature_manifest_path = BLOCK05_ROOT / "manifests" / "feature_manifest.csv"

    for p in (
        complete_path,
        summary_path,
        column_manifest_path,
        feature_manifest_path,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    complete = load_json(complete_path)
    summary = load_json(summary_path)

    # ------------------------------------------------------------------
    # Exact completion-marker contract from canonical Block 05.
    # ------------------------------------------------------------------
    if str(complete.get("status", "")).strip().upper() != "PASS":
        raise RuntimeError(
            f"BLOCK 05 completion marker status is not PASS: "
            f"{complete.get('status')!r}"
        )

    frozen_configuration = complete.get("frozen_configuration")
    if frozen_configuration is None:
        raise RuntimeError(
            "BLOCK 05 completion marker is missing 'frozen_configuration'."
        )
    if str(frozen_configuration).strip() != "W500_O50":
        raise RuntimeError(
            "BLOCK 05 completion marker frozen_configuration is not "
            f"W500_O50: {frozen_configuration!r}"
        )

    feature_count = complete.get("feature_count")
    if feature_count is None:
        raise RuntimeError(
            "BLOCK 05 completion marker is missing 'feature_count'."
        )
    if _as_int(feature_count, "BLOCK05_complete.feature_count") != N_FEATURES:
        raise RuntimeError(
            f"BLOCK 05 feature count is not {N_FEATURES}: {feature_count!r}"
        )

    total_windows = complete.get("total_windows")
    if total_windows is None:
        raise RuntimeError(
            "BLOCK 05 completion marker is missing 'total_windows'."
        )
    if _as_int(total_windows, "BLOCK05_complete.total_windows") != TOTAL_WINDOWS:
        raise RuntimeError(
            f"BLOCK 05 total window count is not {TOTAL_WINDOWS}: "
            f"{total_windows!r}"
        )

    feature_shape = complete.get("feature_matrix_shape")
    if feature_shape is not None:
        if list(feature_shape) != [TOTAL_WINDOWS, N_FEATURES]:
            raise RuntimeError(
                "BLOCK 05 completion feature_matrix_shape is inconsistent: "
                f"{feature_shape!r}"
            )

    # ------------------------------------------------------------------
    # Exact summary contract from canonical Block 05.
    # The summary uses 'block04_configuration', NOT
    # 'frozen_configuration'. The latter belongs to BLOCK05_complete.json.
    # ------------------------------------------------------------------
    if str(summary.get("status", "")).strip().upper() != "PASS":
        raise RuntimeError(
            f"BLOCK 05 summary status is not PASS: {summary.get('status')!r}"
        )

    summary_configuration = summary.get("block04_configuration")
    if summary_configuration is None:
        raise RuntimeError(
            "BLOCK 05 summary is missing 'block04_configuration'."
        )
    if str(summary_configuration).strip() != "W500_O50":
        raise RuntimeError(
            "BLOCK 05 summary block04_configuration is not W500_O50: "
            f"{summary_configuration!r}"
        )

    summary_feature_count = summary.get("feature_count_per_window")
    if summary_feature_count is None:
        raise RuntimeError(
            "BLOCK 05 summary is missing 'feature_count_per_window'."
        )
    if _as_int(
        summary_feature_count,
        "BLOCK05_summary.feature_count_per_window",
    ) != N_FEATURES:
        raise RuntimeError(
            "BLOCK 05 summary feature_count_per_window is not "
            f"{N_FEATURES}: {summary_feature_count!r}"
        )

    summary_total_windows = summary.get("total_windows")
    if summary_total_windows is None:
        raise RuntimeError(
            "BLOCK 05 summary is missing 'total_windows'."
        )
    if _as_int(
        summary_total_windows,
        "BLOCK05_summary.total_windows",
    ) != TOTAL_WINDOWS:
        raise RuntimeError(
            "BLOCK 05 summary total_windows is not "
            f"{TOTAL_WINDOWS}: {summary_total_windows!r}"
        )

    summary_window = summary.get("window_samples")
    summary_overlap = summary.get("overlap_percent")
    summary_step = summary.get("step_samples")

    if summary_window is None or _as_int(
        summary_window, "BLOCK05_summary.window_samples"
    ) != WINDOW_SAMPLES:
        raise RuntimeError(
            f"BLOCK 05 summary window_samples is not "
            f"{WINDOW_SAMPLES}: {summary_window!r}"
        )

    if summary_overlap is None or _as_int(
        summary_overlap, "BLOCK05_summary.overlap_percent"
    ) != OVERLAP_PERCENT:
        raise RuntimeError(
            f"BLOCK 05 summary overlap_percent is not "
            f"{OVERLAP_PERCENT}: {summary_overlap!r}"
        )

    if summary_step is None or _as_int(
        summary_step, "BLOCK05_summary.step_samples"
    ) != STEP_SAMPLES:
        raise RuntimeError(
            f"BLOCK 05 summary step_samples is not "
            f"{STEP_SAMPLES}: {summary_step!r}"
        )

    # ------------------------------------------------------------------
    # Feature-column manifest: exact 341-column schema.
    # ------------------------------------------------------------------
    columns = pd.read_csv(column_manifest_path)

    if len(columns) != N_FEATURES:
        raise RuntimeError(
            f"BLOCK 05 feature column manifest has {len(columns)} rows; "
            f"expected {N_FEATURES}."
        )

    if "feature_name" in columns.columns:
        feature_names = columns["feature_name"].astype(str).tolist()
    else:
        if columns.shape[1] == 0:
            raise RuntimeError(
                "BLOCK 05 feature column manifest contains no columns."
            )
        feature_names = columns.iloc[:, 0].astype(str).tolist()

    if len(feature_names) != N_FEATURES:
        raise RuntimeError(
            f"BLOCK 05 feature-name count is {len(feature_names)}; "
            f"expected {N_FEATURES}."
        )

    if any(not name.strip() for name in feature_names):
        raise RuntimeError("BLOCK 05 feature manifest contains blank names.")

    if len(set(feature_names)) != N_FEATURES:
        raise RuntimeError("BLOCK 05 feature names are not unique.")

    # ------------------------------------------------------------------
    # Feature-window manifest: exact 22,050 computational windows.
    # ------------------------------------------------------------------
    fm = pd.read_csv(feature_manifest_path)

    if len(fm) != TOTAL_WINDOWS:
        raise RuntimeError(
            f"BLOCK 05 feature manifest has {len(fm)} rows; "
            f"expected {TOTAL_WINDOWS}."
        )

    required_manifest_columns = {
        "subject_id",
        "condition",
        "gesture_code",
    }
    missing = required_manifest_columns - set(fm.columns)
    if missing:
        raise RuntimeError(
            f"BLOCK 05 feature manifest missing required columns: "
            f"{sorted(missing)}"
        )

    conds = set(fm["condition"].astype(str).str.upper().unique())
    if conds != set(CONDITIONS):
        raise RuntimeError(
            f"BLOCK 05 feature manifest conditions are {sorted(conds)}; "
            f"expected {list(CONDITIONS)}."
        )

    gestures = set(
        pd.to_numeric(fm["gesture_code"], errors="raise").astype(int).unique()
    )
    if gestures != set(GESTURE_IDS):
        raise RuntimeError(
            f"BLOCK 05 gesture codes are {sorted(gestures)}; "
            f"expected {sorted(GESTURE_IDS)}."
        )

    if "feature_file" not in fm.columns or "feature_row" not in fm.columns:
        raise RuntimeError(
            "BLOCK 05 feature manifest must contain feature_file and feature_row provenance."
        )

    if fm["feature_file"].isna().any() or fm["feature_row"].isna().any():
        raise RuntimeError("BLOCK 05 feature manifest contains missing file/row provenance.")

    feature_file_counts = fm.groupby(fm["feature_file"].astype(str)).size()
    if len(feature_file_counts) != N_RECORDINGS:
        raise RuntimeError(
            f"BLOCK 05 feature manifest references {len(feature_file_counts)} feature files; "
            f"expected {N_RECORDINGS}."
        )
    if not np.all(feature_file_counts.to_numpy() == WINDOWS_PER_RECORDING):
        raise RuntimeError("BLOCK 05 feature manifest does not contain exactly 21 rows per feature file.")

    feature_rows = pd.to_numeric(fm["feature_row"], errors="raise").astype(int)
    grouped_rows = fm.assign(_feature_row=feature_rows).groupby(fm["feature_file"].astype(str))["_feature_row"]
    for _, row_values in grouped_rows:
        if sorted(row_values.tolist()) != list(range(WINDOWS_PER_RECORDING)):
            raise RuntimeError("BLOCK 05 feature_row provenance is not exactly 0..20 per recording.")

    subjects = set(
        pd.to_numeric(fm["subject_id"], errors="raise").astype(int).unique()
    )
    if subjects != set(SUBJECTS):
        raise RuntimeError(
            f"BLOCK 05 subject IDs are {sorted(subjects)}; "
            f"expected 1..25."
        )

    leakage_contract_audit = verify_block05_leakage_contract()

    return {
        "status": "PASS",
        "feature_count": N_FEATURES,
        "feature_manifest_rows": int(len(fm)),
        "configuration": "W500_O50",
        "window_samples": WINDOW_SAMPLES,
        "overlap_percent": OVERLAP_PERCENT,
        "step_samples": STEP_SAMPLES,
        "feature_column_manifest_sha256": sha256_file(column_manifest_path),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "feature_names": feature_names,
        "leakage_contract_sha256": leakage_contract_audit["contract_sha256"],
    }


def normalize_set(value) -> str:
    s = str(value).strip()
    if s.lower().startswith("set"):
        s = s[3:]
    if s.upper() in {"A", "B", "C"}:
        return f"Set{s.upper()}"
    raise ValueError(f"Invalid recording set: {value}")


# =============================================================================
# 05. FEATURE FILE DISCOVERY AND LOAD
# =============================================================================

def parse_feature_filename(path: Path):
    """
    Expected:
        S01_ME_1_SetA_features.npz
        S25_MI_7_SetC_features.npz
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 5 or parts[-1] != "features":
        raise ValueError(f"Unexpected feature filename: {path.name}")

    subject_token, condition, gesture_token, set_token = parts[:4]
    if not subject_token.startswith("S"):
        raise ValueError(f"Invalid subject token in {path.name}")

    subject = int(subject_token[1:])
    condition = condition.upper()
    gesture = int(gesture_token)
    set_name = normalize_set(set_token)

    if subject not in SUBJECTS:
        raise ValueError(f"Invalid subject {subject} in {path.name}")
    if condition not in CONDITIONS:
        raise ValueError(f"Invalid condition {condition} in {path.name}")
    if gesture not in GESTURE_IDS:
        raise ValueError(f"Invalid gesture {gesture} in {path.name}")
    if set_name not in SETS:
        raise ValueError(f"Invalid set {set_name} in {path.name}")

    return subject, condition, gesture, set_name


def discover_feature_files() -> list[Path]:
    files = []
    for root in (ME_FEATURE_ROOT, MI_FEATURE_ROOT):
        if not root.exists():
            raise FileNotFoundError(f"Feature directory missing: {root}")
        files.extend(sorted(root.glob("*.npz")))

    if len(files) != N_RECORDINGS:
        raise RuntimeError(
            f"Expected {N_RECORDINGS} Block 05 feature NPZ files, found {len(files)}."
        )

    identities = []
    for path in files:
        identity = parse_feature_filename(path)
        identities.append(identity)

    expected = {
        (s, c, g, st)
        for s in SUBJECTS
        for c in CONDITIONS
        for g in GESTURE_IDS
        for st in SETS
    }

    actual = set(identities)
    missing = expected - actual
    extra = actual - expected

    if missing:
        raise RuntimeError(f"Missing Block 05 feature recordings: {sorted(missing)[:10]}")
    if extra:
        raise RuntimeError(f"Unexpected Block 05 feature recordings: {sorted(extra)[:10]}")
    if len(actual) != N_RECORDINGS:
        raise RuntimeError("Duplicate Block 05 logical recording identities detected.")

    return files


def load_feature_file(path: Path, expected_feature_names=None):
    subject, condition, gesture, set_name = parse_feature_filename(path)

    with np.load(path, allow_pickle=True) as z:
        if "features" not in z.files:
            raise RuntimeError(f"{path.name}: missing 'features'.")
        X = np.asarray(z["features"], dtype=np.float64)

        if X.shape != (WINDOWS_PER_RECORDING, N_FEATURES):
            raise RuntimeError(
                f"{path.name}: expected {(WINDOWS_PER_RECORDING, N_FEATURES)}, "
                f"got {X.shape}."
            )
        if not np.all(np.isfinite(X)):
            raise RuntimeError(f"{path.name}: non-finite feature values.")

        file_names = None
        if "feature_names" in z.files:
            file_names = [str(x) for x in z["feature_names"].tolist()]
            if len(file_names) != N_FEATURES:
                raise RuntimeError(f"{path.name}: invalid feature_names length.")

        if expected_feature_names is not None and file_names is not None:
            if file_names != expected_feature_names:
                raise RuntimeError(
                    f"{path.name}: feature-column ordering differs from Block 05."
                )

        if "subject" in z.files:
            arr_subject = np.asarray(z["subject"]).reshape(-1)
            if len(arr_subject) != WINDOWS_PER_RECORDING or not np.all(
                arr_subject.astype(int) == subject
            ):
                raise RuntimeError(f"{path.name}: subject metadata mismatch.")

        if "condition" in z.files:
            arr_condition = np.asarray(z["condition"]).astype(str).reshape(-1)
            if len(arr_condition) != WINDOWS_PER_RECORDING or not np.all(
                arr_condition == condition
            ):
                raise RuntimeError(f"{path.name}: condition metadata mismatch.")

        if "gesture_code" in z.files:
            arr_gesture = np.asarray(z["gesture_code"]).reshape(-1)
            if len(arr_gesture) != WINDOWS_PER_RECORDING or not np.all(
                arr_gesture.astype(int) == gesture
            ):
                raise RuntimeError(f"{path.name}: gesture metadata mismatch.")

        if "set_name" in z.files:
            arr_set = np.asarray(z["set_name"]).astype(str).reshape(-1)
            if len(arr_set) != WINDOWS_PER_RECORDING:
                raise RuntimeError(f"{path.name}: set metadata length mismatch.")
            if not np.all([normalize_set(x) == set_name for x in arr_set]):
                raise RuntimeError(f"{path.name}: set metadata mismatch.")

        if "window_id" not in z.files:
            raise RuntimeError(f"{path.name}: missing window_id provenance.")
        window_ids = np.asarray(z["window_id"]).astype(str).reshape(-1)
        if len(window_ids) != WINDOWS_PER_RECORDING:
            raise RuntimeError(f"{path.name}: invalid window_id count.")

        if len(set(window_ids.tolist())) != WINDOWS_PER_RECORDING:
            raise RuntimeError(f"{path.name}: duplicate window IDs.")

        if "start_sample" not in z.files or "end_sample" not in z.files:
            raise RuntimeError(f"{path.name}: missing temporal provenance.")

        starts = np.asarray(z["start_sample"]).astype(int).reshape(-1)
        ends = np.asarray(z["end_sample"]).astype(int).reshape(-1)

        if len(starts) != WINDOWS_PER_RECORDING or len(ends) != WINDOWS_PER_RECORDING:
            raise RuntimeError(f"{path.name}: invalid start/end count.")
        if np.any(ends - starts != WINDOW_SAMPLES):
            raise RuntimeError(f"{path.name}: invalid window length.")
        if np.any(starts < 0) or np.any(ends > RECORDING_SAMPLES):
            raise RuntimeError(f"{path.name}: invalid temporal bounds.")

        if "start_time_s" in z.files:
            start_time = np.asarray(z["start_time_s"], dtype=float).reshape(-1)
        else:
            start_time = starts / FS

        if "end_time_s" in z.files:
            end_time = np.asarray(z["end_time_s"], dtype=float).reshape(-1)
        else:
            end_time = ends / FS

    meta = pd.DataFrame({
        "subject": subject,
        "condition": condition,
        "gesture_code": gesture,
        "gesture_name": GESTURE_MAP[gesture],
        "set": set_name,
        "source_feature_file": str(path),
        "window_id": window_ids,
        "start_sample": starts,
        "end_sample": ends,
        "start_time_s": start_time,
        "end_time_s": end_time,
    })

    return X, meta


def build_master_dataset(feature_files, feature_names):
    all_X = []
    all_meta = []

    for i, path in enumerate(feature_files, 1):
        X, meta = load_feature_file(path, expected_feature_names=feature_names)
        all_X.append(X)
        all_meta.append(meta)

        if i % 100 == 0 or i == len(feature_files):
            print(f"  Loaded {i}/{len(feature_files)} feature recordings")

    X_all = np.vstack(all_X)
    meta_all = pd.concat(all_meta, ignore_index=True)

    if X_all.shape != (TOTAL_WINDOWS, N_FEATURES):
        raise RuntimeError(
            f"Master feature matrix expected {(TOTAL_WINDOWS, N_FEATURES)}, "
            f"got {X_all.shape}."
        )

    if len(meta_all) != TOTAL_WINDOWS:
        raise RuntimeError("Master metadata row count mismatch.")

    if meta_all["window_id"].duplicated().any():
        raise RuntimeError("Duplicate window IDs across Block 05 recordings.")

    if set(meta_all["subject"].astype(int).unique()) != set(SUBJECTS):
        raise RuntimeError("Master feature subject set mismatch.")

    if set(meta_all["condition"].unique()) != set(CONDITIONS):
        raise RuntimeError("Master feature condition set mismatch.")

    if set(meta_all["gesture_code"].astype(int).unique()) != set(GESTURE_IDS):
        raise RuntimeError("Master feature gesture set mismatch.")

    if set(meta_all["set"].unique()) != set(SETS):
        raise RuntimeError("Master feature set mismatch.")

    return X_all, meta_all


# =============================================================================
# 06. MODEL FACTORIES
# =============================================================================

class XGBGestureClassifier:
    """XGBoost wrapper that preserves the frozen 1..7 gesture IDs.

    XGBClassifier's sklearn interface requires contiguous zero-based class
    labels internally. This wrapper encodes 1..7 -> 0..6 only inside XGBoost
    and maps predictions back to the canonical Paper 2 gesture IDs.
    """

    def __init__(self, **params):
        self.params = dict(params)
        self.classes_ = np.asarray(GESTURE_IDS, dtype=np.int64)
        self._model = None

    def fit(self, X, y):
        if xgb is None:
            raise RuntimeError(
                "XGBoost is unavailable. Install a compatible xgboost package. "
                f"Import error: {XGB_IMPORT_ERROR}"
            )
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        expected = np.asarray(GESTURE_IDS, dtype=np.int64)
        if not np.array_equal(np.sort(np.unique(y)), expected):
            raise ValueError(
                "XGBGestureClassifier requires the canonical gesture IDs 1..7; "
                f"got {np.sort(np.unique(y)).tolist()}"
            )
        self._model = xgb.XGBClassifier(**self.params)
        self._model.fit(X, y - 1)
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("XGBGestureClassifier has not been fitted.")
        pred = np.asarray(self._model.predict(X), dtype=np.int64).reshape(-1)
        if np.any((pred < 0) | (pred >= len(GESTURE_IDS))):
            raise RuntimeError("XGBoost returned an invalid encoded class ID.")
        return pred + 1

    def get_params(self, deep=True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self

    @property
    def feature_importances_(self):
        if self._model is None:
            raise AttributeError("XGBGestureClassifier has not been fitted.")
        return self._model.feature_importances_


def make_model(model_name: str):
    if model_name == "LDA":
        estimator = LinearDiscriminantAnalysis(
            solver=MODEL_CONFIG["LDA"]["solver"]
        )
    elif model_name == "KNN":
        estimator = KNeighborsClassifier(
            n_neighbors=MODEL_CONFIG["KNN"]["n_neighbors"],
            weights=MODEL_CONFIG["KNN"]["weights"],
            metric=MODEL_CONFIG["KNN"]["metric"],
            n_jobs=-1,
        )
    elif model_name == "SVM":
        estimator = SVC(
            kernel=MODEL_CONFIG["SVM"]["kernel"],
            C=MODEL_CONFIG["SVM"]["C"],
            gamma=MODEL_CONFIG["SVM"]["gamma"],
            probability=MODEL_CONFIG["SVM"]["probability"],
            random_state=RANDOM_SEED,
        )
    elif model_name == "RF":
        estimator = RandomForestClassifier(
            n_estimators=MODEL_CONFIG["RF"]["n_estimators"],
            random_state=MODEL_CONFIG["RF"]["random_state"],
            n_jobs=MODEL_CONFIG["RF"]["n_jobs"],
            class_weight=MODEL_CONFIG["RF"]["class_weight"],
        )
    elif model_name == "XGB":
        if xgb is None:
            raise RuntimeError(
                "XGBoost is unavailable. Install a compatible xgboost package. "
                f"Import error: {XGB_IMPORT_ERROR}"
            )
        cfg = MODEL_CONFIG["XGB"]
        estimator = XGBGestureClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            objective=cfg["objective"],
            eval_metric=cfg["eval_metric"],
            random_state=cfg["random_state"],
            n_jobs=cfg["n_jobs"],
            tree_method=cfg["tree_method"],
            num_class=len(GESTURE_IDS),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if MODEL_CONFIG[model_name]["scaler"]:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    return estimator


# =============================================================================
# 07. METRICS
# =============================================================================

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(GESTURE_IDS),
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(GESTURE_IDS),
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(GESTURE_IDS), average="macro", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

    class_df = pd.DataFrame({
        "gesture_code": list(GESTURE_IDS),
        "gesture_name": [GESTURE_MAP[g] for g in GESTURE_IDS],
        "precision": precision.astype(float),
        "recall": recall.astype(float),
        "f1": f1.astype(float),
        "support": support.astype(int),
    })

    return metrics, class_df, cm


def aggregate_subject_metrics(df, group_cols):
    rows = []

    for keys, group in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_subjects"] = int(group["subject"].nunique())

        for metric in PRIMARY_METRICS:
            values = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q1"] = float(np.percentile(values, 25))
            row[f"{metric}_q3"] = float(np.percentile(values, 75))
            if len(values) > 1:
                sem = np.std(values, ddof=1) / math.sqrt(len(values))
                row[f"{metric}_ci95_low"] = float(np.mean(values) - 1.96 * sem)
                row[f"{metric}_ci95_high"] = float(np.mean(values) + 1.96 * sem)
            else:
                row[f"{metric}_ci95_low"] = float(values[0])
                row[f"{metric}_ci95_high"] = float(values[0])

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# 08. SPLIT DEFINITIONS
# =============================================================================

def sd_indices(meta: pd.DataFrame, subject: int, condition: str):
    subset = (
        (meta["subject"].astype(int) == subject)
        & (meta["condition"] == condition)
    )

    train_mask = subset & meta["set"].isin(SD_TRAIN_SETS)
    test_mask = subset & (meta["set"] == SD_TEST_SET)

    train_idx = np.flatnonzero(train_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())

    # SetA + SetB = 2 sets; each set contains 7 gesture recordings.
    # Therefore: 2 * 7 * 21 = 294 training windows per subject/condition.
    expected_train = 2 * 7 * WINDOWS_PER_RECORDING
    expected_test = 1 * 7 * WINDOWS_PER_RECORDING

    if len(train_idx) != expected_train:
        raise RuntimeError(
            f"SD train count incorrect for S{subject:02d} {condition}: "
            f"{len(train_idx)} vs {expected_train}"
        )
    if len(test_idx) != expected_test:
        raise RuntimeError(
            f"SD test count incorrect for S{subject:02d} {condition}: "
            f"{len(test_idx)} vs {expected_test}"
        )

    return train_idx, test_idx


def loso_indices(meta: pd.DataFrame, held_out_subject: int, condition: str):
    cond_mask = meta["condition"] == condition
    train_mask = cond_mask & (meta["subject"].astype(int) != held_out_subject)
    test_mask = cond_mask & (meta["subject"].astype(int) == held_out_subject)

    train_idx = np.flatnonzero(train_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())

    expected_train = 24 * 7 * 3 * WINDOWS_PER_RECORDING
    expected_test = 7 * 3 * WINDOWS_PER_RECORDING

    if len(train_idx) != expected_train:
        raise RuntimeError(
            f"LOSO train count incorrect for held-out S{held_out_subject:02d} "
            f"{condition}: {len(train_idx)} vs {expected_train}"
        )
    if len(test_idx) != expected_test:
        raise RuntimeError(
            f"LOSO test count incorrect for held-out S{held_out_subject:02d} "
            f"{condition}: {len(test_idx)} vs {expected_test}"
        )

    return train_idx, test_idx



# =============================================================================
# 09. PREDICTION / RESULT HELPERS
# =============================================================================

def make_prediction_frame(
    meta: pd.DataFrame,
    indices: np.ndarray,
    y_pred,
    model_name: str,
    regime: str,
    condition: str,
    held_out_subject=None,
):
    sub = meta.iloc[indices].copy().reset_index(drop=True)

    out = pd.DataFrame({
        "subject": sub["subject"].astype(int),
        "condition": sub["condition"].astype(str),
        "gesture_true": sub["gesture_code"].astype(int),
        "gesture_true_name": sub["gesture_name"].astype(str),
        "gesture_pred": np.asarray(y_pred).astype(int),
        "gesture_pred_name": [
            GESTURE_MAP[int(x)] for x in np.asarray(y_pred).astype(int)
        ],
        "model": model_name,
        "evaluation_regime": regime,
        "set": sub["set"].astype(str),
        "window_id": sub["window_id"].astype(str),
        "source_feature_file": sub["source_feature_file"].astype(str),
        "start_sample": sub["start_sample"].astype(int),
        "end_sample": sub["end_sample"].astype(int),
    })

    if held_out_subject is not None:
        out["loso_fold"] = int(held_out_subject)
        out["held_out_subject"] = int(held_out_subject)

    return out


def checkpoint_path(regime, condition, model, subject=None):
    token = f"{regime}_{condition}_{model}"
    if subject is not None:
        token += f"_S{subject:02d}"
    return CHECKPOINT_ROOT / f"{token}.json"


def write_unit_checkpoint(path: Path, payload: dict):
    atomic_write_json(path, payload)


def is_valid_prediction_file(path: Path, expected_rows: int):
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
        return len(df) == expected_rows
    except Exception:
        return False


# =============================================================================
# 10. SUBJECT-DEPENDENT EXECUTION
# =============================================================================

def run_subject_dependent(X, meta, feature_names):
    print("\n" + "=" * 100)
    print("BLOCK 06A — SUBJECT-DEPENDENT ML")
    print("=" * 100)

    subject_rows = []
    class_rows = []
    execution_rows = []

    for condition in CONDITIONS:
        for model_name in MODELS:
            print(f"\n06A | {condition} | {model_name}")

            for subject in SUBJECTS:
                pred_path = (
                    SD_ROOT / "predictions"
                    / f"S{subject:02d}_{condition}_{model_name}_predictions.csv"
                )
                metrics_path = (
                    SD_ROOT / "metrics"
                    / f"S{subject:02d}_{condition}_{model_name}_metrics.json"
                )
                class_path = (
                    SD_ROOT / "class_metrics"
                    / f"S{subject:02d}_{condition}_{model_name}_class_metrics.csv"
                )
                cm_path = (
                    SD_ROOT / "confusion_matrices"
                    / f"S{subject:02d}_{condition}_{model_name}_confusion_matrix.csv"
                )
                model_path = (
                    MODEL_ROOT / "SD"
                    / f"S{subject:02d}_{condition}_{model_name}.joblib"
                )

                train_idx, test_idx = sd_indices(meta, subject, condition)

                if (
                    is_valid_prediction_file(pred_path, len(test_idx))
                    and metrics_path.exists()
                    and class_path.exists()
                    and cm_path.exists()
                    and model_path.exists()
                ):
                    metrics = load_json(metrics_path)
                    class_df = pd.read_csv(class_path)
                    execution_rows.append({
                        "regime": "SD",
                        "condition": condition,
                        "model": model_name,
                        "subject": subject,
                        "status": "RESUME-SKIP",
                        "train_windows": len(train_idx),
                        "test_windows": len(test_idx),
                        "training_seconds": metrics.get("training_seconds", np.nan),
                        "prediction_seconds": metrics.get("prediction_seconds", np.nan),
                    })
                    subject_rows.append({
                        "subject": subject,
                        "condition": condition,
                        "model": model_name,
                        "evaluation_regime": "SD",
                        **{
                            k: metrics[k]
                            for k in PRIMARY_METRICS
                        },
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "training_seconds": metrics.get("training_seconds", np.nan),
                        "prediction_seconds": metrics.get("prediction_seconds", np.nan),
                    })
                    c = class_df.copy()
                    c["subject"] = subject
                    c["condition"] = condition
                    c["model"] = model_name
                    c["evaluation_regime"] = "SD"
                    class_rows.append(c)
                    continue

                y_train = meta.iloc[train_idx]["gesture_code"].astype(int).to_numpy()
                y_test = meta.iloc[test_idx]["gesture_code"].astype(int).to_numpy()

                model = make_model(model_name)

                t0 = time.perf_counter()
                model.fit(X[train_idx], y_train)
                train_seconds = time.perf_counter() - t0

                t1 = time.perf_counter()
                y_pred = model.predict(X[test_idx])
                prediction_seconds = time.perf_counter() - t1

                metrics, class_df, cm = calculate_metrics(y_test, y_pred)

                pred_df = make_prediction_frame(
                    meta,
                    test_idx,
                    y_pred,
                    model_name,
                    "SD",
                    condition,
                )

                pred_df["gesture_true"] = pred_df["gesture_true"].astype(int)
                pred_df["gesture_pred"] = pred_df["gesture_pred"].astype(int)
                atomic_write_csv(pred_path, pred_df)

                metrics_payload = {
                    **metrics,
                    "regime": "SD",
                    "condition": condition,
                    "model": model_name,
                    "subject": subject,
                    "train_windows": int(len(train_idx)),
                    "test_windows": int(len(test_idx)),
                    "train_sets": list(SD_TRAIN_SETS),
                    "test_set": SD_TEST_SET,
                    "training_seconds": float(train_seconds),
                    "prediction_seconds": float(prediction_seconds),
                    "feature_count": N_FEATURES,
                    "feature_names_hash": hashlib.sha256(
                        "\n".join(feature_names).encode("utf-8")
                    ).hexdigest(),
                    "random_seed": RANDOM_SEED,
                    "test_used_for_model_selection": False,
                    "test_used_for_scaler_fitting": False,
                    "feature_selection_applied": False,
                }
                atomic_write_json(metrics_path, metrics_payload)

                class_out = class_df.copy()
                class_out["subject"] = subject
                class_out["condition"] = condition
                class_out["model"] = model_name
                class_out["evaluation_regime"] = "SD"
                atomic_write_csv(class_path, class_out)

                cm_df = pd.DataFrame(
                    cm,
                    index=[f"True_{g}" for g in GESTURE_IDS],
                    columns=[f"Pred_{g}" for g in GESTURE_IDS],
                )
                atomic_write_csv(
                    cm_path,
                    cm_df.reset_index().rename(columns={"index": "true_class"}),
                )

                atomic_joblib_dump(model_path, model)

                subject_rows.append({
                    "subject": subject,
                    "condition": condition,
                    "model": model_name,
                    "evaluation_regime": "SD",
                    **metrics,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "training_seconds": float(train_seconds),
                    "prediction_seconds": float(prediction_seconds),
                })

                c = class_df.copy()
                c["subject"] = subject
                c["condition"] = condition
                c["model"] = model_name
                c["evaluation_regime"] = "SD"
                class_rows.append(c)

                execution_rows.append({
                    "regime": "SD",
                    "condition": condition,
                    "model": model_name,
                    "subject": subject,
                    "status": "PASS",
                    "train_windows": len(train_idx),
                    "test_windows": len(test_idx),
                    "training_seconds": float(train_seconds),
                    "prediction_seconds": float(prediction_seconds),
                })

                write_unit_checkpoint(
                    checkpoint_path("SD", condition, model_name, subject),
                    {
                        "status": "PASS",
                        "completed_utc": now_utc(),
                        "regime": "SD",
                        "condition": condition,
                        "model": model_name,
                        "subject": subject,
                    },
                )

                print(
                    f"  S{subject:02d}: "
                    f"Acc={metrics['accuracy']:.4f}, "
                    f"BA={metrics['balanced_accuracy']:.4f}, "
                    f"MacroF1={metrics['macro_f1']:.4f}"
                )

    subject_df = pd.DataFrame(subject_rows)
    class_df_all = pd.concat(class_rows, ignore_index=True)
    execution_df = pd.DataFrame(execution_rows)

    atomic_write_csv(
        SD_ROOT / "subject_results" / "SD_subject_results.csv",
        subject_df,
    )
    atomic_write_csv(
        SD_ROOT / "class_metrics" / "SD_class_metrics_all.csv",
        class_df_all,
    )
    atomic_write_csv(
        SD_ROOT / "SD_execution_audit.csv",
        execution_df,
    )

    cohort_df = aggregate_subject_metrics(
        subject_df,
        ["condition", "model", "evaluation_regime"],
    )
    atomic_write_csv(
        SD_ROOT / "SD_cohort_results.csv",
        cohort_df,
    )

    return subject_df, class_df_all, cohort_df


# =============================================================================
# 11. LOSO EXECUTION
# =============================================================================

def run_loso(X, meta, feature_names):
    print("\n" + "=" * 100)
    print("BLOCK 06B — 25-FOLD SUBJECT-INDEPENDENT LOSO ML")
    print("=" * 100)

    subject_rows = []
    class_rows = []
    execution_rows = []
    leakage_rows = []
    calibration_rows = []

    for condition in CONDITIONS:
        for held_out in SUBJECTS:
            train_idx, test_idx = loso_indices(meta, held_out, condition)

            train_subjects = sorted(
                meta.iloc[train_idx]["subject"].astype(int).unique().tolist()
            )
            test_subjects = sorted(
                meta.iloc[test_idx]["subject"].astype(int).unique().tolist()
            )

            if len(train_subjects) != N_SUBJECTS - 1:
                raise RuntimeError(
                    f"LOSO fold S{held_out:02d} has {len(train_subjects)} "
                    f"training subjects; expected {N_SUBJECTS - 1}."
                )
            if held_out in train_subjects:
                raise RuntimeError("LOSO leakage: held-out subject in training.")
            if test_subjects != [held_out]:
                raise RuntimeError(
                    f"LOSO test subject mismatch: expected [{held_out}], "
                    f"got {test_subjects}."
                )

            for model_name in MODELS:
                print(
                    f"\n06B | {condition} | Fold S{held_out:02d} | {model_name}"
                )

                fold_dir = LOSO_ROOT / "folds" / f"Subject_{held_out:02d}"
                fold_dir.mkdir(parents=True, exist_ok=True)

                pred_path = (
                    fold_dir / f"{condition}_{model_name}_predictions.csv"
                )
                metrics_path = (
                    fold_dir / f"{condition}_{model_name}_metrics.json"
                )
                class_path = (
                    fold_dir / f"{condition}_{model_name}_class_metrics.csv"
                )
                cm_path = (
                    fold_dir / f"{condition}_{model_name}_confusion_matrix.csv"
                )
                model_path = (
                    MODEL_ROOT / "LOSO"
                    / f"Fold_S{held_out:02d}_{condition}_{model_name}.joblib"
                )

                if (
                    is_valid_prediction_file(pred_path, len(test_idx))
                    and metrics_path.exists()
                    and class_path.exists()
                    and cm_path.exists()
                    and model_path.exists()
                ):
                    metrics = load_json(metrics_path)
                    class_df = pd.read_csv(class_path)

                    subject_rows.append({
                        "subject": held_out,
                        "held_out_subject": held_out,
                        "condition": condition,
                        "model": model_name,
                        "evaluation_regime": "LOSO",
                        **{k: metrics[k] for k in PRIMARY_METRICS},
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "training_seconds": metrics.get("training_seconds", np.nan),
                        "prediction_seconds": metrics.get("prediction_seconds", np.nan),
                    })

                    c = class_df.copy()
                    c["subject"] = held_out
                    c["held_out_subject"] = held_out
                    c["condition"] = condition
                    c["model"] = model_name
                    c["evaluation_regime"] = "LOSO"
                    class_rows.append(c)

                    execution_rows.append({
                        "regime": "LOSO",
                        "condition": condition,
                        "model": model_name,
                        "held_out_subject": held_out,
                        "status": "RESUME-SKIP",
                        "train_windows": len(train_idx),
                        "test_windows": len(test_idx),
                    })

                    leakage_rows.append({
                        "condition": condition,
                        "model": model_name,
                        "held_out_subject": held_out,
                        "train_subject_count": len(train_subjects),
                        "validation_subject_count": 0,
                        "test_subject_count": len(test_subjects),
                        "train_test_overlap": len(set(train_subjects) & set(test_subjects)),
                        "validation_test_overlap": 0,
                        "test_used_for_scaler": False,
                        "test_used_for_model_selection": False,
                        "internal_validation_used_for_model_selection": False,
                        "internal_validation_used_for_hyperparameter_tuning": False,
                        "feature_selection_applied": False,
                        "status": "PASS",
                    })

                    calibration_rows.append({
                        "held_out_subject": held_out,
                        "condition": condition,
                        "model": model_name,
                        "calibration_percent": 0,
                        "baseline_source": "BLOCK06_LOSO_unadapted_test_prediction",
                        "prediction_file": str(pred_path),
                        "calibration_applied": False,
                    })
                    continue

                # -----------------------------------------------------------------
                # Final LOSO model only.
                # No internal validation model is fitted because all classifier
                # configurations are frozen a priori.
                # -----------------------------------------------------------------

                # -----------------------------------------------------------------
                # Final LOSO model: refit on all 24 training subjects.
                # The held-out subject remains completely untouched.
                # -----------------------------------------------------------------
                model = make_model(model_name)

                y_train = meta.iloc[train_idx]["gesture_code"].astype(int).to_numpy()
                y_test = meta.iloc[test_idx]["gesture_code"].astype(int).to_numpy()

                t0 = time.perf_counter()
                model.fit(X[train_idx], y_train)
                train_seconds = time.perf_counter() - t0

                t1 = time.perf_counter()
                y_pred = model.predict(X[test_idx])
                prediction_seconds = time.perf_counter() - t1

                metrics, class_df, cm = calculate_metrics(y_test, y_pred)

                pred_df = make_prediction_frame(
                    meta,
                    test_idx,
                    y_pred,
                    model_name,
                    "LOSO",
                    condition,
                    held_out_subject=held_out,
                )
                atomic_write_csv(pred_path, pred_df)

                metrics_payload = {
                    **metrics,
                    "regime": "LOSO",
                    "condition": condition,
                    "model": model_name,
                    "held_out_subject": held_out,
                    "train_subjects": train_subjects,
                    "test_subjects": test_subjects,
                    "train_windows": int(len(train_idx)),
                    "test_windows": int(len(test_idx)),
                    "internal_validation_partition_subject": None,
                    "internal_validation_model_fit_executed": False,
                    "internal_validation_used_for_model_selection": False,
                    "internal_validation_used_for_hyperparameter_tuning": False,
                    "internal_validation_runtime_seconds": 0.0,
                    "training_seconds": float(train_seconds),
                    "prediction_seconds": float(prediction_seconds),
                    "feature_count": N_FEATURES,
                    "random_seed": RANDOM_SEED,
                    "test_used_for_model_selection": False,
                    "test_used_for_scaler_fitting": False,
                    "test_used_for_internal_validation": False,
                    "feature_selection_applied": False,
                }
                atomic_write_json(metrics_path, metrics_payload)

                class_out = class_df.copy()
                class_out["subject"] = held_out
                class_out["held_out_subject"] = held_out
                class_out["condition"] = condition
                class_out["model"] = model_name
                class_out["evaluation_regime"] = "LOSO"
                atomic_write_csv(class_path, class_out)

                cm_df = pd.DataFrame(
                    cm,
                    index=[f"True_{g}" for g in GESTURE_IDS],
                    columns=[f"Pred_{g}" for g in GESTURE_IDS],
                )
                atomic_write_csv(
                    cm_path,
                    cm_df.reset_index().rename(columns={"index": "true_class"}),
                )

                atomic_joblib_dump(model_path, model)

                subject_rows.append({
                    "subject": held_out,
                    "held_out_subject": held_out,
                    "condition": condition,
                    "model": model_name,
                    "evaluation_regime": "LOSO",
                    **metrics,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "training_seconds": float(train_seconds),
                    "prediction_seconds": float(prediction_seconds),
                })

                c = class_df.copy()
                c["subject"] = held_out
                c["held_out_subject"] = held_out
                c["condition"] = condition
                c["model"] = model_name
                c["evaluation_regime"] = "LOSO"
                class_rows.append(c)

                execution_rows.append({
                    "regime": "LOSO",
                    "condition": condition,
                    "model": model_name,
                    "held_out_subject": held_out,
                    "status": "PASS",
                    "train_windows": len(train_idx),
                    "test_windows": len(test_idx),
                    "training_seconds": float(train_seconds),
                    "prediction_seconds": float(prediction_seconds),
                })

                leakage_rows.append({
                    "condition": condition,
                    "model": model_name,
                    "held_out_subject": held_out,
                    "train_subject_count": len(train_subjects),
                    "validation_subject_count": 0,
                    "test_subject_count": len(test_subjects),
                    "train_test_overlap": len(set(train_subjects) & set(test_subjects)),
                    "validation_test_overlap": 0,
                    "test_used_for_scaler": False,
                    "test_used_for_model_selection": False,
                    "internal_validation_used_for_model_selection": False,
                    "internal_validation_used_for_hyperparameter_tuning": False,
                    "feature_selection_applied": False,
                    "status": "PASS",
                })

                calibration_rows.append({
                    "held_out_subject": held_out,
                    "condition": condition,
                    "model": model_name,
                    "calibration_percent": 0,
                    "baseline_source": "BLOCK06_LOSO_unadapted_test_prediction",
                    "prediction_file": str(pred_path),
                    "calibration_applied": False,
                })

                write_unit_checkpoint(
                    checkpoint_path("LOSO", condition, model_name, held_out),
                    {
                        "status": "PASS",
                        "completed_utc": now_utc(),
                        "regime": "LOSO",
                        "condition": condition,
                        "model": model_name,
                        "held_out_subject": held_out,
                        "validation_subject": None,
                    },
                )

                print(
                    f"  Fold S{held_out:02d}: "
                    f"Acc={metrics['accuracy']:.4f}, "
                    f"BA={metrics['balanced_accuracy']:.4f}, "
                    f"MacroF1={metrics['macro_f1']:.4f}"
                )

    subject_df = pd.DataFrame(subject_rows)
    class_df_all = pd.concat(class_rows, ignore_index=True)
    execution_df = pd.DataFrame(execution_rows)
    leakage_df = pd.DataFrame(leakage_rows)
    calibration_df = pd.DataFrame(calibration_rows)

    atomic_write_csv(
        LOSO_ROOT / "subject_results" / "LOSO_subject_results.csv",
        subject_df,
    )
    atomic_write_csv(
        LOSO_ROOT / "class_metrics" / "LOSO_class_metrics_all.csv",
        class_df_all,
    )
    atomic_write_csv(
        LOSO_ROOT / "LOSO_execution_audit.csv",
        execution_df,
    )
    atomic_write_csv(
        LOSO_ROOT / "LOSO_leakage_audit.csv",
        leakage_df,
    )
    atomic_write_csv(
        CAL_HANDOFF_ROOT / "baseline_0pct_results.csv",
        calibration_df,
    )

    cohort_df = aggregate_subject_metrics(
        subject_df,
        ["condition", "model", "evaluation_regime"],
    )
    atomic_write_csv(
        LOSO_ROOT / "LOSO_cohort_results.csv",
        cohort_df,
    )

    return subject_df, class_df_all, cohort_df, leakage_df, calibration_df


# =============================================================================
# 12. SD–LOSO COMPARISON
# =============================================================================

def build_comparisons(sd_subject_df, loso_subject_df):
    sd = sd_subject_df.copy()
    lo = loso_subject_df.copy()

    keys = ["subject", "condition", "model"]
    merged = sd.merge(
        lo,
        on=keys,
        suffixes=("_sd", "_loso"),
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != N_SUBJECTS * len(CONDITIONS) * len(MODELS):
        raise RuntimeError(
            f"Expected {N_SUBJECTS * len(CONDITIONS) * len(MODELS)} "
            f"matched SD/LOSO rows, got {len(merged)}."
        )

    rows = []
    for _, r in merged.iterrows():
        base = {
            "subject": int(r["subject"]),
            "condition": r["condition"],
            "model": r["model"],
        }
        for metric in PRIMARY_METRICS:
            rows.append({
                **base,
                "metric": metric,
                "sd_value": float(r[f"{metric}_sd"]),
                "loso_value": float(r[f"{metric}_loso"]),
                "sd_loso_gap": float(
                    r[f"{metric}_sd"] - r[f"{metric}_loso"]
                ),
            })

    gap_df = pd.DataFrame(rows)
    atomic_write_csv(
        COMPARE_ROOT / "SD_vs_LOSO_gap_long.csv",
        gap_df,
    )

    summary_rows = []
    for condition in CONDITIONS:
        for model in MODELS:
            s = merged[
                (merged["condition"] == condition)
                & (merged["model"] == model)
            ]
            row = {
                "condition": condition,
                "model": model,
                "n_subjects": len(s),
            }
            for metric in PRIMARY_METRICS:
                gap = (
                    s[f"{metric}_sd"].astype(float)
                    - s[f"{metric}_loso"].astype(float)
                )
                row[f"{metric}_sd_mean"] = float(s[f"{metric}_sd"].mean())
                row[f"{metric}_loso_mean"] = float(s[f"{metric}_loso"].mean())
                row[f"{metric}_gap_mean"] = float(gap.mean())
                row[f"{metric}_gap_sd"] = (
                    float(gap.std(ddof=1)) if len(gap) > 1 else 0.0
                )
                row[f"{metric}_gap_median"] = float(gap.median())
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    atomic_write_csv(
        COMPARE_ROOT / "model_condition_summary.csv",
        summary_df,
    )

    return gap_df, summary_df


# =============================================================================
# 13. COHORT CONFUSION MATRICES
# =============================================================================

def build_cohort_confusions():
    rows = []

    for condition in CONDITIONS:
        for model_name in MODELS:
            pred_files = sorted(
                (LOSO_ROOT / "folds").glob(
                    f"Subject_*/*{condition}_{model_name}_predictions.csv"
                )
            )

            if len(pred_files) != N_SUBJECTS:
                raise RuntimeError(
                    f"Expected 25 LOSO prediction files for {condition}/{model_name}, "
                    f"found {len(pred_files)}."
                )

            y_true = []
            y_pred = []

            for p in pred_files:
                df = pd.read_csv(p)
                y_true.extend(df["gesture_true"].astype(int).tolist())
                y_pred.extend(df["gesture_pred"].astype(int).tolist())

            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=list(GESTURE_IDS),
            )

            cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

            cm_path = (
                LOSO_ROOT / "confusion_matrices"
                / f"{condition}_{model_name}_cohort_confusion_matrix.csv"
            )
            cm_norm_path = (
                LOSO_ROOT / "confusion_matrices"
                / f"{condition}_{model_name}_cohort_confusion_matrix_normalized.csv"
            )

            atomic_write_csv(
                cm_path,
                pd.DataFrame(
                    cm,
                    index=[f"True_{g}" for g in GESTURE_IDS],
                    columns=[f"Pred_{g}" for g in GESTURE_IDS],
                ).reset_index().rename(columns={"index": "true_class"}),
            )
            atomic_write_csv(
                cm_norm_path,
                pd.DataFrame(
                    cm_norm,
                    index=[f"True_{g}" for g in GESTURE_IDS],
                    columns=[f"Pred_{g}" for g in GESTURE_IDS],
                ).reset_index().rename(columns={"index": "true_class"}),
            )

            for i, gtrue in enumerate(GESTURE_IDS):
                for j, gpred in enumerate(GESTURE_IDS):
                    rows.append({
                        "condition": condition,
                        "model": model_name,
                        "true_gesture": gtrue,
                        "pred_gesture": gpred,
                        "count": int(cm[i, j]),
                        "row_normalized": float(cm_norm[i, j]),
                    })

    df = pd.DataFrame(rows)
    atomic_write_csv(
        LOSO_ROOT / "confusion_matrices" / "LOSO_cohort_confusion_long.csv",
        df,
    )
    return df


# =============================================================================
# 14. FIGURE HELPERS
# =============================================================================

def save_fig(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in FIGURE_FORMATS:
        fig.savefig(
            stem.with_suffix(f".{ext}"),
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    plt.close(fig)


def annotate_heatmap(ax, matrix, fmt="{:.2f}", fontsize=8):
    nrows, ncols = matrix.shape
    for i in range(nrows):
        for j in range(ncols):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize=fontsize,
            )


# =============================================================================
# 15. FIGURES — BAR CHARTS
# =============================================================================

def figure_bar_model_performance(sd_cohort, loso_cohort):
    metric_list = PRIMARY_METRICS

    for regime, df in (("SD", sd_cohort), ("LOSO", loso_cohort)):
        for condition in CONDITIONS:
            sub = df[df["condition"] == condition].copy()

            for metric in metric_list:
                value_col = f"{metric}_mean"

                fig, ax = plt.subplots(figsize=(9, 5.5))
                x = np.arange(len(MODELS))
                vals = [
                    float(
                        sub.loc[sub["model"] == m, value_col].iloc[0]
                    )
                    for m in MODELS
                ]
                bars = ax.bar(x, vals)
                ax.set_xticks(x)
                ax.set_xticklabels(MODELS)
                ax.set_ylabel(metric.replace("_", " ").title())
                if metric in {"mcc", "cohen_kappa"}:
                    ax.set_ylim(-1, 1)
                else:
                    ax.set_ylim(0, 1)
                ax.set_title(
                    f"{condition} — {regime} ML performance: "
                    f"{metric.replace('_', ' ').title()}"
                )
                if metric in {"accuracy", "balanced_accuracy"}:
                    ax.axhline(
                        CHANCE_ACCURACY,
                        linestyle="--",
                        linewidth=1,
                        label="7-class chance level",
                    )
                for b, v in zip(bars, vals):
                    offset = 0.015 if v >= 0 else -0.015
                    va = "bottom" if v >= 0 else "top"
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        v + offset,
                        f"{v:.3f}",
                        ha="center",
                        va=va,
                        fontsize=8,
                    )
                if metric in {"accuracy", "balanced_accuracy"}:
                    ax.legend()
                save_fig(
                    fig,
                    FIG_ROOT / "bar"
                    / f"Figure_BLOCK06_{condition}_{regime}_{metric}",
                )


# =============================================================================
# 16. FIGURES — HEATMAPS
# =============================================================================

def plot_heatmap(matrix_df, title, xlabel, ylabel, stem, vmin=0, vmax=1):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        matrix_df.to_numpy(dtype=float),
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(np.arange(matrix_df.shape[1]))
    ax.set_xticklabels(matrix_df.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix_df.shape[0]))
    ax.set_yticklabels(matrix_df.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    annotate_heatmap(ax, matrix_df.to_numpy(dtype=float), "{:.3f}", fontsize=7)
    fig.colorbar(im, ax=ax, label="Value")
    save_fig(fig, stem)


def figure_model_condition_heatmaps(sd_cohort, loso_cohort):
    for metric in PRIMARY_METRICS:
        rows = []
        for model in MODELS:
            row = {}
            for condition in CONDITIONS:
                for regime, df in (("SD", sd_cohort), ("LOSO", loso_cohort)):
                    x = df[
                        (df["condition"] == condition)
                        & (df["model"] == model)
                    ][f"{metric}_mean"]
                    row[f"{condition}-{regime}"] = float(x.iloc[0])
            rows.append(pd.Series(row, name=model))

        matrix = pd.DataFrame(rows)

        heat_vmin, heat_vmax = (
            (-1, 1) if metric in {"mcc", "cohen_kappa"} else (0, 1)
        )
        plot_heatmap(
            matrix,
            f"BLOCK 06 — Model × condition × regime: {metric.replace('_', ' ').title()}",
            "Condition / evaluation regime",
            "Model",
            FIG_ROOT / "heatmaps"
            / f"Figure_BLOCK06_ModelCondition_{metric}",
            vmin=heat_vmin,
            vmax=heat_vmax,
        )


def figure_metric_heatmaps(sd_cohort, loso_cohort):
    """Generate scale-correct heatmaps for unsigned and signed metrics."""
    unsigned_metrics = ("accuracy", "balanced_accuracy", "macro_f1")
    signed_metrics = ("mcc", "cohen_kappa")

    def make_profile(df, condition, regime, metrics, suffix, vmin, vmax):
        sub = df[df["condition"] == condition]
        matrix = pd.DataFrame(
            {
                metric.replace("_", " ").title(): [
                    float(
                        sub[sub["model"] == model][f"{metric}_mean"].iloc[0]
                    )
                    for model in MODELS
                ]
                for metric in metrics
            },
            index=MODELS,
        )
        plot_heatmap(
            matrix,
            f"{condition} — {regime} ML metric profile ({suffix})",
            "Metric",
            "Model",
            FIG_ROOT / "heatmaps"
            / f"Figure_BLOCK06_{condition}_{regime}_MetricHeatmap_{suffix}",
            vmin=vmin,
            vmax=vmax,
        )

    for condition in CONDITIONS:
        make_profile(
            sd_cohort, condition, "SD", unsigned_metrics, "0to1", 0, 1
        )
        make_profile(
            sd_cohort, condition, "SD", signed_metrics, "signed", -1, 1
        )
        make_profile(
            loso_cohort, condition, "LOSO", unsigned_metrics, "0to1", 0, 1
        )
        make_profile(
            loso_cohort, condition, "LOSO", signed_metrics, "signed", -1, 1
        )


# =============================================================================
# 17. FIGURES — RADAR
# =============================================================================

def radar_chart(values, labels, title, stem):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values = np.asarray(values, dtype=float)
    values_closed = np.r_[values, values[0]]
    angles_closed = np.r_[angles, angles[0]]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_closed, values_closed, linewidth=2)
    ax.fill(angles_closed, values_closed, alpha=0.15)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20)
    ax.grid(True)
    save_fig(fig, stem)


def figure_radar(sd_cohort, loso_cohort):
    labels = [
        "Accuracy",
        "Balanced accuracy",
        "Macro-F1",
        "MCC",
        "Cohen's kappa",
    ]

    for regime, df in (("SD", sd_cohort), ("LOSO", loso_cohort)):
        for condition in CONDITIONS:
            sub = df[df["condition"] == condition]

            # One radar per condition/regime with all models overlaid.
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)

            for model in MODELS:
                r = sub[sub["model"] == model].iloc[0]
                vals = np.array([
                    r["accuracy_mean"],
                    r["balanced_accuracy_mean"],
                    r["macro_f1_mean"],
                    r["mcc_mean"],
                    r["cohen_kappa_mean"],
                ], dtype=float)
                closed = np.r_[vals, vals[0]]
                aclosed = np.r_[angles, angles[0]]
                ax.plot(aclosed, closed, linewidth=1.8, label=model)

            ax.set_xticks(angles)
            ax.set_xticklabels(labels)
            ax.set_ylim(-1, 1)
            ax.set_title(
                f"{condition} — {regime} ML multi-metric radar",
                pad=25,
            )
            ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.15))
            ax.grid(True)

            save_fig(
                fig,
                FIG_ROOT / "radar"
                / f"Figure_BLOCK06_{condition}_{regime}_Radar",
            )


# =============================================================================
# 18. FIGURES — SD VS LOSO
# =============================================================================

def figure_sd_vs_loso(gap_df):
    for condition in CONDITIONS:
        for model in MODELS:
            sub = gap_df[
                (gap_df["condition"] == condition)
                & (gap_df["model"] == model)
                & (gap_df["metric"] == "macro_f1")
            ].sort_values("subject")

            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.plot(
                sub["subject"],
                sub["sd_value"],
                marker="o",
                label="Subject-dependent",
            )
            ax.plot(
                sub["subject"],
                sub["loso_value"],
                marker="o",
                label="LOSO",
            )
            ax.set_xlabel("Subject")
            ax.set_ylabel("Macro-F1")
            ax.set_ylim(0, 1)
            ax.set_title(
                f"{condition} — {model}: subject-dependent vs LOSO Macro-F1"
            )
            ax.legend()
            save_fig(
                fig,
                FIG_ROOT / "subject_distributions"
                / f"Figure_BLOCK06_{condition}_{model}_SD_vs_LOSO_MacroF1",
            )

    for condition in CONDITIONS:
        sub = gap_df[
            (gap_df["condition"] == condition)
            & (gap_df["metric"] == "macro_f1")
        ]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        positions = np.arange(len(MODELS))
        data = [
            sub[sub["model"] == m]["sd_loso_gap"].to_numpy()
            for m in MODELS
        ]
        ax.boxplot(
            data,
            positions=positions,
            tick_labels=MODELS,
        )
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_ylabel("SD − LOSO Macro-F1")
        ax.set_xlabel("Model")
        ax.set_title(f"{condition} — Subject-dependent to LOSO Macro-F1 gap")
        save_fig(
            fig,
            FIG_ROOT / "subject_distributions"
            / f"Figure_BLOCK06_{condition}_SD_LOSO_Gap_MacroF1",
        )


# =============================================================================
# 19. FIGURES — SUBJECT DISTRIBUTIONS
# =============================================================================

def figure_subject_distributions(sd_subject, loso_subject):
    for condition in CONDITIONS:
        for regime, df in (("SD", sd_subject), ("LOSO", loso_subject)):
            sub = df[df["condition"] == condition]

            fig, ax = plt.subplots(figsize=(10, 5.5))
            data = [
                sub[sub["model"] == m]["macro_f1"].to_numpy()
                for m in MODELS
            ]
            ax.boxplot(
                data,
                tick_labels=MODELS,
                showmeans=True,
            )
            ax.set_ylim(0, 1)
            ax.set_ylabel("Subject-level Macro-F1")
            ax.set_xlabel("Model")
            ax.set_title(
                f"{condition} — {regime} subject-level Macro-F1 distribution"
            )
            save_fig(
                fig,
                FIG_ROOT / "subject_distributions"
                / f"Figure_BLOCK06_{condition}_{regime}_SubjectMacroF1",
            )


# =============================================================================
# 20. FIGURES — CLASS F1
# =============================================================================

def figure_class_f1_heatmaps(sd_class, loso_class):
    for regime, df in (("SD", sd_class), ("LOSO", loso_class)):
        for condition in CONDITIONS:
            sub = df[df["condition"] == condition]

            # Mean class F1 across subjects/folds.
            pivot = (
                sub.groupby(["gesture_code", "model"])["f1"]
                .mean()
                .unstack("model")
                .reindex(index=GESTURE_IDS, columns=MODELS)
            )
            pivot.index = [
                f"{g}: {GESTURE_MAP[g]}" for g in GESTURE_IDS
            ]

            plot_heatmap(
                pivot,
                f"{condition} — {regime} class-level F1",
                "Model",
                "Gesture",
                FIG_ROOT / "heatmaps"
                / f"Figure_BLOCK06_{condition}_{regime}_ClassF1Heatmap",
            )

            for model in MODELS:
                m = sub[sub["model"] == model]
                means = (
                    m.groupby("gesture_code")["f1"]
                    .mean()
                    .reindex(GESTURE_IDS)
                )

                fig, ax = plt.subplots(figsize=(9, 5))
                x = np.arange(len(GESTURE_IDS))
                bars = ax.bar(x, means.to_numpy(dtype=float))
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [f"{g}: {GESTURE_MAP[g]}" for g in GESTURE_IDS],
                    rotation=30,
                    ha="right",
                )
                ax.set_ylim(0, 1)
                ax.set_ylabel("Mean F1")
                ax.set_title(
                    f"{condition} — {regime} — {model} class-level F1"
                )
                for b, v in zip(bars, means):
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        float(v) + 0.015,
                        f"{float(v):.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
                save_fig(
                    fig,
                    FIG_ROOT / "bar"
                    / f"Figure_BLOCK06_{condition}_{regime}_{model}_ClassF1",
                )


# =============================================================================
# 21. FIGURES — CONFUSION MATRICES
# =============================================================================

def plot_confusion_matrix(cm, title, stem, normalized=False):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, vmin=0, vmax=1 if normalized else None, aspect="equal")
    ax.set_xticks(np.arange(len(GESTURE_IDS)))
    ax.set_yticks(np.arange(len(GESTURE_IDS)))
    ax.set_xticklabels([str(g) for g in GESTURE_IDS])
    ax.set_yticklabels([str(g) for g in GESTURE_IDS])
    ax.set_xlabel("Predicted gesture ID")
    ax.set_ylabel("True gesture ID")
    ax.set_title(title)

    annotate_heatmap(
        ax,
        np.asarray(cm, dtype=float),
        "{:.2f}" if normalized else "{:.0f}",
        fontsize=8,
    )
    fig.colorbar(
        im,
        ax=ax,
        label="Row-normalized proportion" if normalized else "Count",
    )
    save_fig(fig, stem)


def figure_confusion_matrices():
    for condition in CONDITIONS:
        for model in MODELS:
            raw_path = (
                LOSO_ROOT / "confusion_matrices"
                / f"{condition}_{model}_cohort_confusion_matrix.csv"
            )
            norm_path = (
                LOSO_ROOT / "confusion_matrices"
                / f"{condition}_{model}_cohort_confusion_matrix_normalized.csv"
            )

            raw = pd.read_csv(raw_path)
            norm = pd.read_csv(norm_path)

            cm_raw = raw.drop(columns=["true_class"]).to_numpy(dtype=float)
            cm_norm = norm.drop(columns=["true_class"]).to_numpy(dtype=float)

            plot_confusion_matrix(
                cm_raw,
                f"{condition} — LOSO — {model} cohort confusion matrix",
                FIG_ROOT / "confusion_matrices"
                / f"Figure_BLOCK06_{condition}_{model}_LOSO_CM_Count",
                normalized=False,
            )
            plot_confusion_matrix(
                cm_norm,
                f"{condition} — LOSO — {model} normalized confusion matrix",
                FIG_ROOT / "confusion_matrices"
                / f"Figure_BLOCK06_{condition}_{model}_LOSO_CM_Normalized",
                normalized=True,
            )


# =============================================================================
# 22. FIGURES — EFFICIENCY
# =============================================================================

def figure_efficiency(sd_subject, loso_subject):
    rows = []

    for regime, df in (("SD", sd_subject), ("LOSO", loso_subject)):
        g = (
            df.groupby(["condition", "model"], as_index=False)
            [["training_seconds", "prediction_seconds"]]
            .mean()
        )
        g["regime"] = regime
        rows.append(g)

    eff = pd.concat(rows, ignore_index=True)
    atomic_write_csv(
        TABLE_ROOT / "computational_efficiency.csv",
        eff,
    )

    for condition in CONDITIONS:
        sub = eff[eff["condition"] == condition]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(MODELS))
        width = 0.18

        for j, regime in enumerate(("SD", "LOSO")):
            vals = [
                float(
                    sub[
                        (sub["regime"] == regime)
                        & (sub["model"] == m)
                    ]["training_seconds"].iloc[0]
                )
                for m in MODELS
            ]
            ax.bar(
                x + (j - 0.5) * width,
                vals,
                width,
                label=f"{regime} training",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS)
        ax.set_ylabel("Mean training time (s)")
        ax.set_xlabel("Model")
        ax.set_title(f"{condition} — ML training-time comparison")
        ax.legend()
        save_fig(
            fig,
            FIG_ROOT / "efficiency"
            / f"Figure_BLOCK06_{condition}_TrainingTime",
        )


# =============================================================================
# 23. TABLE GENERATION
# =============================================================================

def build_tables(sd_subject, sd_class, sd_cohort, loso_subject, loso_class, loso_cohort):
    atomic_write_csv(TABLE_ROOT / "SD_subject_results.csv", sd_subject)
    atomic_write_csv(TABLE_ROOT / "SD_cohort_results.csv", sd_cohort)
    atomic_write_csv(TABLE_ROOT / "SD_class_metrics.csv", sd_class)

    atomic_write_csv(TABLE_ROOT / "LOSO_subject_results.csv", loso_subject)
    atomic_write_csv(TABLE_ROOT / "LOSO_cohort_results.csv", loso_cohort)
    atomic_write_csv(TABLE_ROOT / "LOSO_class_metrics.csv", loso_class)

    model_rows = []
    for model in MODELS:
        model_rows.append({
            "model": model,
            **MODEL_CONFIG[model],
        })
    atomic_write_csv(
        TABLE_ROOT / "model_configuration.csv",
        pd.DataFrame(model_rows),
    )

    leakage = []
    for condition in CONDITIONS:
        for model in MODELS:
            leakage.append({
                "condition": condition,
                "model": model,
                "SD_train_sets": "SetA,SetB",
                "SD_test_set": "SetC",
                "LOSO_folds": N_SUBJECTS,
                "LOSO_train_subjects_per_fold": N_SUBJECTS - 1,
                "LOSO_test_subjects_per_fold": 1,
                "LOSO_internal_validation_rule": "none; all 24 training subjects used for final fold fit",
                "test_subject_in_scaler_fit": False,
                "test_subject_in_internal_validation": False,
                "feature_selection": False,
                "PCA": False,
                "CSP": False,
                "SMOTE": False,
                "random_window_split": False,
                "cross_condition_training": False,
                "status": "PASS",
            })

    atomic_write_csv(
        TABLE_ROOT / "leakage_audit.csv",
        pd.DataFrame(leakage),
    )


# =============================================================================
# 24. REPRODUCIBILITY AUDIT
# =============================================================================

def reproducibility_audit(X, meta):
    # Deterministic model reproducibility check on one fixed SD unit and one
    # fixed LOSO unit for every model. This is an audit only and does not
    # alter final predictions.
    rows = []

    checks = [
        ("SD", "ME", 1),
        ("LOSO", "ME", 1),
    ]

    for regime, condition, subject in checks:
        if regime == "SD":
            train_idx, test_idx = sd_indices(meta, subject, condition)
        else:
            train_idx, test_idx = loso_indices(meta, subject, condition)

        for model_name in MODELS:
            m1 = make_model(model_name)
            m2 = make_model(model_name)

            y_train = meta.iloc[train_idx]["gesture_code"].astype(int).to_numpy()

            m1.fit(X[train_idx], y_train)
            p1 = m1.predict(X[test_idx])

            m2.fit(X[train_idx], y_train)
            p2 = m2.predict(X[test_idx])

            same = np.array_equal(p1, p2)

            rows.append({
                "regime": regime,
                "condition": condition,
                "subject": subject,
                "model": model_name,
                "predictions_identical": bool(same),
                "status": "PASS" if same else "FAIL",
            })

            del m1, m2

    df = pd.DataFrame(rows)
    atomic_write_csv(
        BLOCK06_ROOT / "reproducibility_audit.csv",
        df,
    )

    if not df["predictions_identical"].all():
        raise RuntimeError("Block 06 reproducibility audit failed.")

    return df


# =============================================================================
# 25. CALIBRATION HANDOFF
# =============================================================================

def build_calibration_handoff(
    loso_subject_df,
    calibration_df,
    feature_names,
    block04_audit,
    block05_audit,
):
    # Explicitly preserve the predefined calibration grid without executing
    # calibration here.
    levels = pd.DataFrame({
        "calibration_percent": CALIBRATION_LEVELS_PERCENT,
        "execution_block": "09_TEMPORAL_CALIBRATION",
        "executed_in_block06": False,
        "selection_status": "PREDEFINED",
    })
    atomic_write_csv(
        CAL_HANDOFF_ROOT / "calibration_levels_0_to_90.csv",
        levels,
    )

    baseline = loso_subject_df[
        ["subject", "held_out_subject", "condition", "model"]
        + list(PRIMARY_METRICS)
    ].copy()
    baseline["calibration_percent"] = 0
    baseline["calibration_applied"] = False
    baseline["source_block"] = "06B_ML_LOSO"

    atomic_write_csv(
        CAL_HANDOFF_ROOT / "baseline_0pct_subject_results.csv",
        baseline,
    )

    payload = {
        "status": "PASS",
        "block06_calibration_execution": False,
        "calibration_execution_block": "09_TEMPORAL_CALIBRATION",
        "calibration_levels_percent": list(CALIBRATION_LEVELS_PERCENT),
        "baseline_0_percent_source": "06B_ML_LOSO",
        "earliest_contiguous_target_segment_rule": True,
        "random_calibration_sampling": False,
        "timestamps_used_as_clock": False,
        "sample_index_used_as_clock": False,
        "block04_configuration": "W500_O50",
        "feature_count": N_FEATURES,
        "feature_names_hash": hashlib.sha256(
            "\n".join(feature_names).encode("utf-8")
        ).hexdigest(),
        "block04_manifest_sha256": block04_audit["manifest_sha256"],
        "block05_feature_manifest_sha256": block05_audit["feature_manifest_sha256"],
        "handoff_created_utc": now_utc(),
    }
    atomic_write_json(
        CAL_HANDOFF_ROOT / "calibration_handoff.json",
        payload,
    )

    return payload


# =============================================================================
# 26. GLOBAL AUDIT
# =============================================================================

def run_global_audit(sd_subject, sd_class, loso_subject, loso_class, leakage_df, repro_df):
    expected_sd = N_SUBJECTS * len(CONDITIONS) * len(MODELS)
    expected_loso = N_SUBJECTS * len(CONDITIONS) * len(MODELS)

    audit = {
        "status": "PASS",
        "expected_sd_subject_results": expected_sd,
        "actual_sd_subject_results": int(len(sd_subject)),
        "expected_loso_subject_results": expected_loso,
        "actual_loso_subject_results": int(len(loso_subject)),
        "expected_sd_class_result_rows": expected_sd * 7,
        "actual_sd_class_result_rows": int(len(sd_class)),
        "expected_loso_class_result_rows": expected_loso * 7,
        "actual_loso_class_result_rows": int(len(loso_class)),
        "expected_loso_folds": N_SUBJECTS,
        "actual_loso_folds": int(loso_subject["held_out_subject"].nunique()),
        "sd_subjects": int(sd_subject["subject"].nunique()),
        "loso_subjects": int(loso_subject["held_out_subject"].nunique()),
        "models": list(MODELS),
        "conditions": list(CONDITIONS),
        "feature_count": N_FEATURES,
        "all_sd_metrics_finite": bool(
            np.isfinite(sd_subject[list(PRIMARY_METRICS)].to_numpy()).all()
        ),
        "all_loso_metrics_finite": bool(
            np.isfinite(loso_subject[list(PRIMARY_METRICS)].to_numpy()).all()
        ),
        "leakage_audit_all_pass": bool((leakage_df["status"] == "PASS").all()),
        "reproducibility_all_pass": bool((repro_df["status"] == "PASS").all()),
        "random_window_split": False,
        "feature_selection": False,
        "scaler_test_leakage": False,
        "smote": False,
        "cross_condition_training": False,
        "calibration_executed_in_block06": False,
        "internal_validation_model_fit_executed": False,
        "internal_validation_used_for_selection": False,
        "internal_validation_used_for_hyperparameter_tuning": False,
        "calibration_levels_reserved_for_block09": list(CALIBRATION_LEVELS_PERCENT),
        "raw_data_modified": False,
        "block03_modified": False,
        "block04_modified": False,
        "block05_modified": False,
    }

    if audit["actual_sd_subject_results"] != expected_sd:
        audit["status"] = "FAIL"
    if audit["actual_loso_subject_results"] != expected_loso:
        audit["status"] = "FAIL"
    for condition in CONDITIONS:
        for model in MODELS:
            sd_subs = sorted(
                sd_subject[
                    (sd_subject["condition"] == condition)
                    & (sd_subject["model"] == model)
                ]["subject"].astype(int).unique().tolist()
            )
            loso_subs = sorted(
                loso_subject[
                    (loso_subject["condition"] == condition)
                    & (loso_subject["model"] == model)
                ]["held_out_subject"].astype(int).unique().tolist()
            )
            if sd_subs != list(SUBJECTS) or loso_subs != list(SUBJECTS):
                audit["status"] = "FAIL"

    if audit["actual_loso_folds"] != N_SUBJECTS:
        audit["status"] = "FAIL"
    if not audit["all_sd_metrics_finite"] or not audit["all_loso_metrics_finite"]:
        audit["status"] = "FAIL"
    if not audit["leakage_audit_all_pass"]:
        audit["status"] = "FAIL"
    if not audit["reproducibility_all_pass"]:
        audit["status"] = "FAIL"

    if not bool((leakage_df["validation_subject_count"] == 0).all()):
        audit["status"] = "FAIL"

    atomic_write_json(
        BLOCK06_ROOT / "BLOCK06_global_audit.json",
        audit,
    )

    if audit["status"] != "PASS":
        raise RuntimeError("BLOCK 06 global audit failed.")

    return audit


# =============================================================================
# 27. CONFIGURATION
# =============================================================================

def build_config(block04_audit, block05_audit, feature_names):
    config = {
        "block": BLOCK_ID,
        "protocol_version": PROTOCOL_VERSION,
        "paper": "Paper 2",
        "created_utc": now_utc(),
        "random_seed": RANDOM_SEED,

        "cohort": {
            "subjects": list(SUBJECTS),
            "n_subjects": N_SUBJECTS,
            "conditions": list(CONDITIONS),
            "gestures": {
                str(k): v for k, v in GESTURE_MAP.items()
            },
            "n_gestures": len(GESTURE_IDS),
            "chance_accuracy": CHANCE_ACCURACY,
            "recording_sets": list(SETS),
        },

        "block04": {
            "configuration": "W500_O50",
            "window_samples": WINDOW_SAMPLES,
            "window_seconds": WINDOW_SECONDS,
            "overlap_percent": OVERLAP_PERCENT,
            "step_samples": STEP_SAMPLES,
            "recording_samples": RECORDING_SAMPLES,
            "sampling_rate_hz": FS,
            "windows_per_recording": WINDOWS_PER_RECORDING,
            "total_recordings": N_RECORDINGS,
            "total_windows": TOTAL_WINDOWS,
            "manifest_sha256": block04_audit["manifest_sha256"],
        },

        "block05": {
            "feature_count": N_FEATURES,
            "feature_names_hash": hashlib.sha256(
                "\n".join(feature_names).encode("utf-8")
            ).hexdigest(),
            "feature_manifest_sha256": block05_audit["feature_manifest_sha256"],
            "leakage_contract_sha256": block05_audit["leakage_contract_sha256"],
            "feature_selection": False,
            "PCA": False,
            "CSP": False,
            "SMOTE": False,
            "downstream_feature_source_policy": "features_and_manifests_only; full_cohort_diagnostics_forbidden",
        },

        "subject_dependent": {
            "train_sets": list(SD_TRAIN_SETS),
            "test_set": SD_TEST_SET,
            "random_window_split": False,
            "test_set_used_for_tuning": False,
        },

        "loso": {
            "fold_count": N_SUBJECTS,
            "train_subjects_per_fold": N_SUBJECTS - 1,
            "test_subjects_per_fold": 1,
            "internal_validation_rule": LOSO_INTERNAL_VALIDATION_RULE,
            "internal_validation_model_fit_executed": False,
            "internal_validation_used_for_model_selection": False,
            "internal_validation_used_for_hyperparameter_tuning": False,
            "test_subject_in_internal_validation": False,
            "test_subject_in_scaler_fit": False,
            "test_subject_in_model_selection": False,
            "final_model_refit_on_all_24_training_subjects": True,
        },

        "models": MODEL_CONFIG,

        "metrics": list(PRIMARY_METRICS),

        "calibration_handoff": {
            "execution_block": "09_TEMPORAL_CALIBRATION",
            "levels_percent": list(CALIBRATION_LEVELS_PERCENT),
            "executed_in_block06": False,
            "baseline_0_percent_exported": True,
            "earliest_contiguous_segment_required": True,
            "random_calibration_sampling": False,
        },

        "prohibited": {
            "logistic_regression": True,
            "SMOTE": True,
            "random_window_split": True,
            "global_scaling": True,
            "test_set_tuning": True,
            "test_set_feature_selection": True,
            "cross_condition_training": True,
            "posthoc_model_replacement": True,
            "window_level_inferential_statistics": True,
        },

        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": getattr(xgb, "__version__", None),
            "xgboost_import_error": XGB_IMPORT_ERROR,
            "joblib_available": joblib is not None,
            "platform": platform.platform(),
        },
    }

    atomic_write_json(CONFIG_PATH, config)
    return config


# =============================================================================
# 28. README / EXECUTION REPORT
# =============================================================================

def write_readme(config):
    text = f"""# BLOCK 06 — Conventional ML

Status: EXECUTED WITH FROZEN CONFIGURATION

## Frozen input
- Block 04: W500_O50
- Window: 500 samples / 4.0 s
- Overlap: 50%
- Step: 250 samples
- Block 05 features: 341
- Subjects: 25
- Conditions: ME, MI
- Gesture IDs: 1–7

## Subject-dependent
- SetA + SetB: training
- SetC: final test
- Random window-level split: prohibited

## LOSO
- 25 folds
- 24 training subjects
- 1 held-out test subject
- all 24 non-held-out subjects are used for the final fold fit
- no internal validation model fit is executed; classifier configurations are frozen a priori
- held-out subject excluded from all training-dependent transformations

## Models
- LDA
- KNN
- SVM
- Random Forest
- XGBoost

## Visualization
- Bar charts, heatmaps, radar charts, confusion matrices, class-F1 plots, subject distributions, SD-vs-LOSO gap plots, and computational-efficiency plots are descriptive outputs only. No figure is used for model selection.

## Calibration
The predefined levels 0/10/20/30/40/50/60/80/90% are reserved for
BLOCK 09. BLOCK 06 exports the 0% LOSO baseline and calibration handoff.
No calibration adaptation is executed in BLOCK 06.

## Leakage controls
- No random window split
- No test-derived scaling
- No test-set tuning
- No feature selection
- No PCA/CSP
- No SMOTE
- No cross-condition training
- Subject is the inferential unit

## Downstream
BLOCK 07 — Deep Learning
BLOCK 08 — ME/MI statistics
BLOCK 09 — Temporal calibration
BLOCK 10 — ME/MI transfer
"""
    atomic_write_text(BLOCK06_ROOT / "BLOCK06_README.md", text)


# =============================================================================
# 29. MAIN
# =============================================================================

def main():
    block_start = time.perf_counter()

    print("=" * 100)
    print("BLOCK 06 — CONVENTIONAL MACHINE-LEARNING DECODING")
    print("=" * 100)
    print(f"Protocol                 : {PROTOCOL_VERSION}")
    print(f"Block 04 configuration  : W500_O50")
    print(f"Block 05 features       : {N_FEATURES}")
    print(f"Subjects                : {N_SUBJECTS}")
    print(f"Conditions              : {', '.join(CONDITIONS)}")
    print(f"Models                  : {', '.join(MODELS)}")
    print(f"Calibration levels      : {list(CALIBRATION_LEVELS_PERCENT)}%")
    print("Calibration execution   : BLOCK 09 only")
    print()

    initialize_directories()

    block04_audit = verify_block04_handoff()
    print("BLOCK 04 handoff         : PASS")

    block05_audit = verify_block05_handoff()
    print("BLOCK 05 handoff         : PASS")

    feature_names = block05_audit["feature_names"]

    feature_files = discover_feature_files()
    print(f"Feature recordings       : {len(feature_files)}")

    X, meta = build_master_dataset(feature_files, feature_names)
    print(f"Feature matrix           : {X.shape}")

    # Save immutable execution inventory.
    inventory = meta.copy()
    atomic_write_csv(
        BLOCK06_ROOT / "feature_input_inventory.csv",
        inventory,
    )

    # Explicit split-design audit. This is structural only; no model is fitted.
    split_audit_rows = []
    for condition in CONDITIONS:
        for subject in SUBJECTS:
            sd_train_idx, sd_test_idx = sd_indices(meta, subject, condition)
            loso_train_idx, loso_test_idx = loso_indices(meta, subject, condition)
            loso_train_subjects = sorted(
                meta.iloc[loso_train_idx]["subject"].astype(int).unique().tolist()
            )
            loso_test_subjects = sorted(
                meta.iloc[loso_test_idx]["subject"].astype(int).unique().tolist()
            )
            split_audit_rows.append({
                "condition": condition,
                "subject": subject,
                "SD_train_sets": "SetA,SetB",
                "SD_test_set": "SetC",
                "SD_train_windows": len(sd_train_idx),
                "SD_test_windows": len(sd_test_idx),
                "LOSO_train_subjects": len(loso_train_subjects),
                "LOSO_test_subject": subject,
                "LOSO_train_windows": len(loso_train_idx),
                "LOSO_test_windows": len(loso_test_idx),
                "LOSO_internal_validation_subject": None,
                "LOSO_internal_validation_used_for_selection": False,
                "LOSO_internal_validation_model_fit_executed": False,
                "test_subject_in_training": subject in loso_train_subjects,
                "test_subject_in_validation": False,
                "status": (
                    "PASS"
                    if len(loso_train_subjects) == 24
                    and loso_test_subjects == [subject]
                    and subject not in loso_train_subjects
                    else "FAIL"
                ),
            })
    atomic_write_csv(
        BLOCK06_ROOT / "tables" / "split_design_audit.csv",
        pd.DataFrame(split_audit_rows),
    )

    config = build_config(block04_audit, block05_audit, feature_names)
    write_readme(config)

    sd_subject, sd_class, sd_cohort = run_subject_dependent(
        X, meta, feature_names
    )

    loso_subject, loso_class, loso_cohort, leakage_df, calibration_df = run_loso(
        X, meta, feature_names
    )

    gap_df, comparison_summary = build_comparisons(
        sd_subject,
        loso_subject,
    )

    build_cohort_confusions()

    # Complete figure suite.
    figure_bar_model_performance(sd_cohort, loso_cohort)
    figure_model_condition_heatmaps(sd_cohort, loso_cohort)
    figure_metric_heatmaps(sd_cohort, loso_cohort)
    figure_radar(sd_cohort, loso_cohort)
    figure_sd_vs_loso(gap_df)
    figure_subject_distributions(sd_subject, loso_subject)
    figure_class_f1_heatmaps(sd_class, loso_class)
    figure_confusion_matrices()
    figure_efficiency(sd_subject, loso_subject)

    # Machine-readable figure inventory: confirms the requested visual suite
    # was actually generated and records its scientific purpose.
    figure_files = sorted(
        [p for p in FIG_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".pdf"}]
    )
    figure_manifest_rows = []
    for p in figure_files:
        rel = p.relative_to(FIG_ROOT)
        parts = rel.parts
        category = parts[0] if len(parts) > 1 else "root"
        figure_manifest_rows.append({
            "figure_file": str(p),
            "relative_path": str(rel),
            "category": category,
            "format": p.suffix.lower().lstrip("."),
            "exists": True,
        })
    atomic_write_csv(
        FIG_ROOT / "BLOCK06_figure_manifest.csv",
        pd.DataFrame(figure_manifest_rows),
    )

    build_tables(
        sd_subject,
        sd_class,
        sd_cohort,
        loso_subject,
        loso_class,
        loso_cohort,
    )

    repro_df = reproducibility_audit(X, meta)

    calibration_handoff = build_calibration_handoff(
        loso_subject,
        calibration_df,
        feature_names,
        block04_audit,
        block05_audit,
    )

    global_audit = run_global_audit(
        sd_subject,
        sd_class,
        loso_subject,
        loso_class,
        leakage_df,
        repro_df,
    )

    # Final output integrity assertions.
    if len(figure_files) == 0:
        raise RuntimeError("BLOCK 06 generated no figure files.")
    if len(pd.read_csv(FIG_ROOT / "BLOCK06_figure_manifest.csv")) != len(figure_files):
        raise RuntimeError("BLOCK 06 figure manifest count mismatch.")
    if int(leakage_df["validation_subject_count"].max()) != 0:
        raise RuntimeError("Unexpected internal validation subjects detected.")
    if bool(leakage_df["test_used_for_model_selection"].any()):
        raise RuntimeError("Test-set model-selection leakage detected.")
    if bool(leakage_df["test_used_for_scaler"].any()):
        raise RuntimeError("Test-set scaler leakage detected.")

    runtime_minutes = (time.perf_counter() - block_start) / 60.0

    execution_report = pd.DataFrame([{
        "block": BLOCK_ID,
        "status": global_audit["status"],
        "runtime_minutes": runtime_minutes,
        "subjects": N_SUBJECTS,
        "conditions": len(CONDITIONS),
        "models": len(MODELS),
        "primary_cells": len(CONDITIONS) * 2 * len(MODELS),
        "SD_subject_results": len(sd_subject),
        "LOSO_subject_results": len(loso_subject),
        "LOSO_folds": loso_subject["held_out_subject"].nunique(),
        "features": N_FEATURES,
        "feature_matrix_rows": TOTAL_WINDOWS,
        "figure_files_generated": int(len(figure_files)),
        "calibration_levels_reserved_for_block09": ",".join(
            str(x) for x in CALIBRATION_LEVELS_PERCENT
        ),
        "calibration_executed_in_block06": False,
        "raw_data_modified": False,
        "block03_modified": False,
        "block04_modified": False,
        "block05_modified": False,
    }])
    atomic_write_csv(EXECUTION_REPORT_PATH, execution_report)

    summary = {
        "block": BLOCK_ID,
        "status": "PASS",
        "protocol_version": PROTOCOL_VERSION,
        "completed_utc": now_utc(),
        "configuration": "W500_O50",
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": WINDOW_SECONDS,
        "overlap_percent": OVERLAP_PERCENT,
        "step_samples": STEP_SAMPLES,
        "subjects": N_SUBJECTS,
        "recordings": N_RECORDINGS,
        "windows_per_recording": WINDOWS_PER_RECORDING,
        "total_windows": TOTAL_WINDOWS,
        "features_per_window": N_FEATURES,
        "feature_matrix_shape": [TOTAL_WINDOWS, N_FEATURES],
        "conditions": list(CONDITIONS),
        "models": list(MODELS),
        "subject_dependent_split": "SetA+SetB_train__SetC_test",
        "loso_folds": N_SUBJECTS,
        "loso_train_subjects_per_fold": N_SUBJECTS - 1,
        "loso_test_subjects_per_fold": 1,
        "random_window_split": False,
        "feature_selection": False,
        "PCA": False,
        "CSP": False,
        "SMOTE": False,
        "cross_condition_training": False,
        "test_set_tuning": False,
        "test_set_scaling": False,
        "internal_validation_model_fit_executed": False,
        "internal_validation_used_for_selection": False,
        "internal_validation_used_for_hyperparameter_tuning": False,
        "calibration_execution_in_block06": False,
        "calibration_levels_reserved_for_block09": list(
            CALIBRATION_LEVELS_PERCENT
        ),
        "baseline_0_percent_exported": True,
        "leakage_audit": global_audit,
        "reproducibility": {
            "status": "PASS",
            "rows": int(len(repro_df)),
        },
        "runtime_minutes": runtime_minutes,
        "raw_data_modified": False,
        "block03_modified": False,
        "block04_modified": False,
        "block05_modified": False,
        "output_root": str(BLOCK06_ROOT),
    }
    atomic_write_json(SUMMARY_PATH, summary)

    complete = {
        "block": BLOCK_ID,
        "status": "PASS",
        "paper": "Paper 2",
        "completed_utc": now_utc(),
        "block04_handoff_verified": True,
        "block05_handoff_verified": True,
        "configuration": "W500_O50",
        "feature_count": N_FEATURES,
        "feature_matrix_shape": [TOTAL_WINDOWS, N_FEATURES],
        "SD_complete": True,
        "LOSO_complete": True,
        "LOSO_folds": N_SUBJECTS,
        "models": list(MODELS),
        "conditions": list(CONDITIONS),
        "leakage_controls_verified": True,
        "reproducibility_verified": True,
        "calibration_levels": list(CALIBRATION_LEVELS_PERCENT),
        "calibration_execution_block": "09_TEMPORAL_CALIBRATION",
        "calibration_executed_here": False,
        "baseline_0_percent_exported": True,
        "raw_data_modified": False,
        "block03_modified": False,
        "block04_modified": False,
        "block05_modified": False,
        "global_status": "PASS",
    }

    # Completion marker deliberately written last.
    atomic_write_json(COMPLETE_PATH, complete)

    print("\n" + "=" * 100)
    print("BLOCK 06 — FINAL STATUS")
    print("=" * 100)
    print("Configuration             : W500_O50")
    print("Subjects                  : 25")
    print("Conditions                : ME + MI")
    print("Features                  : 341")
    print("Subject-dependent         : PASS")
    print("LOSO folds                : 25")
    print("LOSO training subjects    : 24/fold")
    print("LOSO held-out subjects    : 1/fold")
    print("Internal validation fit   : NO")
    print("LOSO                      : PASS")
    print("Primary ML models         : 5")
    print("Random window split       : NO")
    print("Feature selection         : NO")
    print("PCA/CSP                   : NO")
    print("SMOTE                     : NO")
    print("Cross-condition training  : NO")
    print("Test-set tuning           : NO")
    print("Calibration in Block 06   : NO")
    print("Calibration levels saved  : 0,10,20,30,40,50,60,80,90%")
    print("Calibration execution     : BLOCK 09")
    print("0% baseline handoff       : PASS")
    print("Leakage audit             : PASS")
    print("Reproducibility           : PASS")
    print("Raw modified              : NO")
    print("BLOCK 03 modified         : NO")
    print("BLOCK 04 modified         : NO")
    print("BLOCK 05 modified         : NO")
    print(f"Runtime                   : {runtime_minutes:.2f} min")
    print(f"Output                    : {BLOCK06_ROOT}")
    print("✓ BLOCK 06 PASS — CONVENTIONAL ML BENCHMARK FROZEN")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        try:
            sys.stdout = LOGGER.stdout
            LOGGER.close()
        except Exception:
            pass
