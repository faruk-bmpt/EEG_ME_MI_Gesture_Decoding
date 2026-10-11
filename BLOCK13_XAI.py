# -*- coding: utf-8 -*-
"""
PAPER 2 — BLOCK 13
EXPLAINABLE AI, ATTRIBUTION ANALYSIS & ME–MI ATTRIBUTION SIMILARITY

Canonical executable:
    BLOCK13_XAI.py

Frozen scope:
    BLOCK 05 -> 341-feature ML representation
    BLOCK 06 -> frozen ML models / 25-fold LOSO
    BLOCK 07 -> frozen DL models / 25-fold LOSO
    BLOCK 12 -> frozen representation configuration

Primary XAI:
    ML: SHAP
    DL: Integrated Gradients + Occlusion

Primary questions:
    1. Which EEG channels contribute to model decisions?
    2. Which frequency-related ML features contribute?
    3. Which gesture-specific features contribute?
    4. Which temporal regions contribute to DL decisions?
    5. Are ME and MI attribution profiles similar or condition-specific?

STRICT SCIENTIFIC RULES
-----------------------
- No model is retrained, fine-tuned, selected, or modified.
- No XAI result changes the model, features, preprocessing, or windowing.
- Subject is the inferential unit; windows are explanation units only.
- All seven numerical gesture IDs (1..7) remain primary.
- ML explanations use the frozen BLOCK 05 341-feature schema.
- Cross-subject ML explanations use the corresponding frozen LOSO model
  on its held-out subject.
- DL explanations require a verified BLOCK 07 XAI handoff describing the
  frozen model/input contract. A state_dict without architecture metadata
  is NOT guessed.
- The IG baseline is computational only and is never called a physiological
  resting baseline.
- No trigger timing is inferred from attribution peaks.
- Missing/unsupported model artifacts are recorded as FAIL, never silently
  replaced by fabricated or surrogate explanations.
"""

from __future__ import annotations

import gc
import ast
import hashlib
import importlib
import inspect
import json
import os
import platform
import re
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

# --- crash-safety patch: must run BEFORE numpy/sklearn/xgboost are imported ---
import faulthandler
faulthandler.enable(open("BLOCK13_native_crash.log", "w"), all_threads=True)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # xgboost + sklearn OpenMP clash on Windows
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

# ---------------------------------------------------------------------
# Optional but required for the corresponding analysis branch.
# ---------------------------------------------------------------------
try:
    import joblib
except Exception as exc:
    joblib = None
    JOBLIB_ERROR = repr(exc)
else:
    JOBLIB_ERROR = None

try:
    import shap
except Exception as exc:
    shap = None
    SHAP_ERROR = repr(exc)
else:
    SHAP_ERROR = None

# BLOCK06 XGB LOSO artifacts contain a nested xgboost.sklearn.XGBClassifier.
# xgboost MUST be importable before joblib/pickle deserialization; otherwise
# the frozen XGB artifact cannot be restored and no scientifically valid
# surrogate explanation is permitted.
try:
    import xgboost as xgb
except Exception as exc:
    xgb = None
    XGBOOST_ERROR = repr(exc)
else:
    XGBOOST_ERROR = None

try:
    import torch
except Exception as exc:
    torch = None
    TORCH_ERROR = repr(exc)
else:
    TORCH_ERROR = None

try:
    from captum.attr import IntegratedGradients, Occlusion
except Exception as exc:
    IntegratedGradients = None
    Occlusion = None
    CAPTUM_ERROR = repr(exc)
else:
    CAPTUM_ERROR = None



# =============================================================================
# 00A. SERIALIZATION COMPATIBILITY FOR FROZEN BLOCK06 XGB MODELS
# =============================================================================
#
# BLOCK06 XGB LOSO artifacts were serialized when this wrapper lived in the
# training script's __main__ module.  The pickle therefore stores:
#     __main__.XGBGestureClassifier
# with a nested xgboost.sklearn.XGBClassifier in ``_model`` and a public
# gesture label mapping in ``classes_``.
#
# Defining the class here BEFORE joblib.load() is required for deterministic
# deserialization.  This class does NOT retrain or alter the frozen model.
# It only restores the original wrapper interface and delegates inference to
# the serialized XGBoost estimator.
#
# IMPORTANT:
#   - The serialized object remains the source of truth.
#   - No hyperparameters are changed.
#   - No fitting is performed.
#   - The public classes_ are the canonical gesture IDs 1..7.
#   - The nested XGB estimator uses the corresponding zero-based class index.
#
# This compatibility layer is intentionally local to BLOCK13 so that the
# original BLOCK06 artifacts do not need to be regenerated.
# =============================================================================

class XGBGestureClassifier:
    """Compatibility shell for BLOCK06's serialized __main__.XGBGestureClassifier."""

    def __init__(self, **kwargs):
        # During normal construction this is empty.  joblib/pickle restores
        # the serialized state directly, so __init__ is not used for loading.
        self.params = dict(kwargs)
        self._model = None
        self.classes_ = np.arange(1, N_GESTURES + 1, dtype=int)

    def _require_loaded_model(self):
        if not hasattr(self, "_model") or self._model is None:
            raise RuntimeError(
                "XGBGestureClassifier deserialized without its nested _model."
            )
        return self._model

    def predict_proba(self, X):
        model = self._require_loaded_model()
        return np.asarray(model.predict_proba(np.asarray(X, dtype=np.float64)))

    def predict(self, X):
        p = self.predict_proba(X)
        if p.ndim != 2 or p.shape[1] != len(self.classes_):
            raise RuntimeError(
                "XGBGestureClassifier probability output is incompatible with "
                f"classes_: shape={p.shape}, classes={self.classes_}"
            )
        return np.asarray(self.classes_)[np.argmax(p, axis=1)]

    def get_booster(self):
        return self._require_loaded_model().get_booster()

    def get_xgb_params(self):
        return self._require_loaded_model().get_xgb_params()

    def get_params(self, deep=True):
        return dict(getattr(self, "params", {}))

    def set_params(self, **params):
        # This method is present for sklearn compatibility only.  BLOCK13
        # never calls it, because the frozen model must not be modified.
        self.params = dict(getattr(self, "params", {}))
        self.params.update(params)
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "_model") and self._model is not None


def _is_xgb_gesture_wrapper(model) -> bool:
    return (
        model.__class__.__name__ == "XGBGestureClassifier"
        and hasattr(model, "_model")
    )


def _unwrap_tree_estimator(model):
    """Return the actual tree estimator for SHAP TreeExplainer."""
    if _is_xgb_gesture_wrapper(model):
        nested = model._model
        if nested is None:
            raise RuntimeError("XGBGestureClassifier has no nested XGB model.")
        return nested
    return model


def _model_classes_for_xai(model, n_outputs: int) -> np.ndarray:
    """
    Return the canonical class labels corresponding to SHAP output columns.

    For BLOCK06's XGB wrapper, the nested XGB estimator is zero-based (0..6)
    while the wrapper exposes the canonical gesture IDs (1..7).  Output
    column i therefore corresponds to wrapper.classes_[i].
    """
    public_classes = getattr(model, "classes_", None)
    if public_classes is not None:
        classes = np.asarray(public_classes)
        if classes.ndim != 1 or len(classes) != n_outputs:
            raise RuntimeError(
                f"Model classes_ is incompatible with SHAP outputs: "
                f"classes_shape={classes.shape}, n_outputs={n_outputs}"
            )
        return classes

    raise RuntimeError(
        f"{type(model).__name__} does not expose classes_; "
        "BLOCK13 will not guess the gesture-to-output mapping."
    )

# =============================================================================
# 00. FROZEN CONFIGURATION
# =============================================================================

BLOCK_ID = "13_XAI"
PROTOCOL_VERSION = "BLOCK13_XAI_FROZEN_V4_COMPLETE_AUDIT"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FS = 125
WINDOW_SAMPLES = 500
WINDOW_SECONDS = 4.0
OVERLAP_PERCENT = 50
STEP_SAMPLES = 250

N_CHANNELS = 13
N_FEATURES = 341
N_SUBJECTS = 25
N_GESTURES = 7
N_WINDOWS = 22050

SUBJECTS = tuple(range(1, 26))
CONDITIONS = ("ME", "MI")
SETS = ("SetA", "SetB", "SetC")
GESTURES = tuple(range(1, 8))

ML_MODELS = ("LDA", "KNN", "SVM", "RF", "XGB")
DL_MODELS = (
    "EEGNet",
    "ShallowConvNet",
    "DeepConvNet",
    "CNN-BiLSTM",
    "CNN-BiLSTM-Attention",
)

# Deterministic explanation subset:
# 3 windows per subject/condition/gesture, chosen by fixed window indices
# 0, 10, 20. This is not performance-driven sampling.
EXPLANATION_WINDOW_INDICES = (0, 10, 20)

# IG configuration is frozen here and written to BLOCK13_config.json.
IG_STEPS = 64
IG_INTERNAL_BATCH_SIZE = 16
IG_BASELINE = "zero_input_tensor"
IG_TARGET = "true_gesture_class"

# Occlusion configuration.
# One channel x one temporal segment is the primary perturbation unit.
OCCLUSION_SEGMENT_LENGTH = 50
OCCLUSION_STEP = 50
OCCLUSION_REPLACEMENT = "zero"

# Fixed visualization/summary limits.
TOP_FEATURES = 20
TOP_CHANNELS = 13
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
FDR_Q = 0.05

# =============================================================================
# 01. WINDOWS PATHS
# =============================================================================

WINDOWS_ROOT = Path(
    r"F:\Faruk\EEG_Motor_Execution_Imagery\Paper2_EEG_BCI"
)
BLOCK04_ROOT = WINDOWS_ROOT / "04_WINDOWING"
BLOCK05_ROOT = WINDOWS_ROOT / "05_FEATURE_ENGINEERING"
BLOCK06_ROOT = WINDOWS_ROOT / "06_ML"
BLOCK07_ROOT = WINDOWS_ROOT / "07_DL"
BLOCK12_ROOT = WINDOWS_ROOT / "12_REPRESENTATION_ANALYSIS"

BLOCK13_ROOT = WINDOWS_ROOT / "13_XAI"

# Output architecture.
TABLE_ROOT = BLOCK13_ROOT / "tables"
FIGURE_ROOT = BLOCK13_ROOT / "figures"
RAW_ROOT = BLOCK13_ROOT / "raw_attributions"
DIAG_ROOT = BLOCK13_ROOT / "diagnostics"
CONFIG_ROOT = BLOCK13_ROOT / "config"
LOG_ROOT = BLOCK13_ROOT / "logs"

ML_RAW_ROOT = RAW_ROOT / "ML"
DL_RAW_ROOT = RAW_ROOT / "DL"

for p in (
    BLOCK13_ROOT, TABLE_ROOT, FIGURE_ROOT, RAW_ROOT, DIAG_ROOT,
    CONFIG_ROOT, LOG_ROOT, ML_RAW_ROOT, DL_RAW_ROOT
):
    p.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = CONFIG_ROOT / "BLOCK13_config.json"
PREFLIGHT_PATH = DIAG_ROOT / "BLOCK13_preflight.json"
STATUS_PATH = DIAG_ROOT / "BLOCK13_status.json"
MANIFEST_PATH = DIAG_ROOT / "BLOCK13_artifact_manifest.csv"
LOG_PATH = LOG_ROOT / "BLOCK13_execution.log"


# =============================================================================
# 02. LOGGING / ATOMIC I/O
# =============================================================================

def log(message: str) -> None:
    line = f"{datetime.now().strftime('%H:%M:%S')} | {message}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_process_memory_mb() -> float:
    """Resident memory of this process in MB (psutil, else Win32 API, else -1)."""
    try:
        import psutil  # type: ignore
        return float(psutil.Process(os.getpid()).memory_info().rss) / 1048576.0
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            if psapi.GetProcessMemoryInfo(
                k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb
            ):
                return float(pmc.WorkingSetSize) / 1048576.0
        except Exception:
            pass
    return -1.0


def log_memory(tag: str) -> None:
    """Log process memory after forcing a garbage-collection pass."""
    gc.collect()
    mb = get_process_memory_mb()
    if mb >= 0:
        log(f"MEM | {tag}: process_memory={mb:,.0f} MB")
    else:
        log(f"MEM | {tag}: process_memory=unavailable")


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        df.to_csv(
            tmp,
            index=False,
            chunksize=100_000,
        )
        os.replace(tmp, path)

    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_npz(path: Path, **arrays) -> None:
    """Write a compressed NumPy archive atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def raw_ml_path(subject: int, condition: str, model_name: str) -> Path:
    return ML_RAW_ROOT / f"S{subject:02d}_{condition}_{model_name}.npz"


def raw_dl_path(subject: int, condition: str, model_name: str) -> Path:
    return DL_RAW_ROOT / f"S{subject:02d}_{condition}_{model_dir_name(model_name)}.npz"


def write_raw_ml_archive(
    path: Path,
    signed_shap: np.ndarray,
    meta: pd.DataFrame,
    subject: int,
    condition: str,
    model_name: str,
) -> None:
    """Archive one ML explanation per selected BLOCK05 feature row/window."""
    signed = np.asarray(signed_shap, dtype=np.float64)
    if signed.shape != (len(meta), N_FEATURES):
        raise RuntimeError(
            f"ML raw archive shape {signed.shape}; expected ({len(meta)},{N_FEATURES})."
        )
    assert_finite(f"ML raw SHAP {path.name}", signed)
    atomic_npz(
        path,
        signed_shap=signed,
        abs_shap=np.abs(signed),
        subject_id=np.full(len(meta), subject, dtype=np.int16),
        condition=np.asarray([condition] * len(meta)),
        model=np.asarray([model_name] * len(meta)),
        gesture=np.asarray(meta["gesture_code"].to_numpy(dtype=np.int8)),
        set=np.asarray(meta["set"].astype(str).to_numpy()),
        window_index=np.asarray(meta["window_index"].to_numpy(dtype=np.int16)),
        window_id=np.asarray(meta["window_id"].astype(str).to_numpy()),
        source_recording=np.asarray(meta["source_recording"].astype(str).to_numpy()),
        feature_file=np.asarray(meta["feature_file"].astype(str).to_numpy()),
        feature_row=np.asarray(meta["feature_row"].to_numpy(dtype=np.int16)),
        feature_id=np.asarray(FEATURE_SCHEMA["feature_id"].astype(str).to_numpy()),
        feature_name=np.asarray(FEATURE_SCHEMA["feature_name"].astype(str).to_numpy()),
        feature_family=np.asarray(FEATURE_SCHEMA["feature_family"].astype(str).to_numpy()),
        feature_channel=np.asarray(FEATURE_SCHEMA["channel"].astype(str).to_numpy()),
        feature_frequency_group=np.asarray(
            FEATURE_SCHEMA["frequency_group"].fillna("NONE").astype(str).to_numpy()
        ),
        protocol_version=np.asarray(PROTOCOL_VERSION),
        archive_type=np.asarray("per_window_signed_and_absolute_SHAP"),
    )


def write_raw_dl_archive(
    path: Path,
    res: dict,
    rows: pd.DataFrame,
    subject: int,
    condition: str,
    model_name: str,
) -> None:
    """Archive true per-window DL IG/occlusion results, not gesture means."""
    n = len(rows)
    expected = N_GESTURES * len(SETS) * len(EXPLANATION_WINDOW_INDICES)
    if n != expected:
        raise RuntimeError(
            f"DL raw archive metadata has {n} rows; expected {expected}."
        )

    required = {
        "ig_abs_raw", "ig_signed_raw", "occ_abs_raw", "occ_signed_raw",
        "delta_raw", "prediction_raw", "target_raw",
    }
    missing = sorted(required - set(res))
    if missing:
        raise RuntimeError(f"DL raw archive missing result arrays: {missing}")

    arrays = {
        "ig_abs": np.asarray(res["ig_abs_raw"], dtype=np.float32),
        "ig_signed": np.asarray(res["ig_signed_raw"], dtype=np.float32),
        "occlusion_abs": np.asarray(res["occ_abs_raw"], dtype=np.float32),
        "occlusion_signed": np.asarray(res["occ_signed_raw"], dtype=np.float32),
        "ig_convergence_delta": np.asarray(res["delta_raw"], dtype=np.float32),
        "prediction": np.asarray(res["prediction_raw"], dtype=np.int8),
        "target_gesture": np.asarray(res["target_raw"], dtype=np.int8),
    }
    expected_map = {
        "ig_abs": (n, N_CHANNELS, WINDOW_SAMPLES),
        "ig_signed": (n, N_CHANNELS, WINDOW_SAMPLES),
        "occlusion_abs": (n, N_CHANNELS, WINDOW_SAMPLES),
        "occlusion_signed": (n, N_CHANNELS, WINDOW_SAMPLES),
        "ig_convergence_delta": (n,),
        "prediction": (n,),
        "target_gesture": (n,),
    }
    for key, shape in expected_map.items():
        if arrays[key].shape != shape:
            raise RuntimeError(
                f"DL raw archive {path.name}: {key} shape {arrays[key].shape}; "
                f"expected {shape}."
            )
        if not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"DL raw archive {path.name}: {key} contains NaN/Inf.")

    atomic_npz(
        path,
        **arrays,
        subject_id=np.full(n, subject, dtype=np.int16),
        condition=np.asarray([condition] * n),
        model=np.asarray([model_name] * n),
        gesture=np.asarray(rows["gesture_code"].to_numpy(dtype=np.int8)),
        set=np.asarray(rows["set"].astype(str).to_numpy()),
        window_index=np.asarray(rows["window_index"].to_numpy(dtype=np.int16)),
        window_id=np.asarray(rows["window_id"].astype(str).to_numpy()),
        source_recording=np.asarray(rows["source_recording"].astype(str).to_numpy()),
        window_array_file=np.asarray(rows["window_array_file"].astype(str).to_numpy()),
        window_array_sha256=np.asarray(rows["window_array_sha256"].astype(str).to_numpy())
        if "window_array_sha256" in rows.columns
        else np.asarray(["NOT_PRESENT_IN_BLOCK04_MANIFEST"] * n),
        protocol_version=np.asarray(PROTOCOL_VERSION),
        archive_type=np.asarray("per_window_per_gesture_IG_and_occlusion"),
        explanation_window_indices=np.asarray(EXPLANATION_WINDOW_INDICES, dtype=np.int16),
        ig_steps=np.asarray(IG_STEPS, dtype=np.int16),
        ig_internal_batch_size=np.asarray(IG_INTERNAL_BATCH_SIZE, dtype=np.int16),
        occlusion_segment_length=np.asarray(OCCLUSION_SEGMENT_LENGTH, dtype=np.int16),
        occlusion_step=np.asarray(OCCLUSION_STEP, dtype=np.int16),
    )


def validate_raw_dl_archive(
    path: Path,
    subject: int | None = None,
    condition: str | None = None,
    model_name: str | None = None,
) -> tuple[bool, str]:
    """Strictly validate one DL raw attribution archive and explain any failure.

    This validator is intentionally stronger than the aggregate-cache validator.
    It verifies the complete per-window attribution payload, metadata identity,
    protocol/schema provenance, finite values, and the exact 63-window subset.
    """
    if not path.exists():
        return False, "file does not exist"

    expected_n = N_GESTURES * len(SETS) * len(EXPLANATION_WINDOW_INDICES)
    expected_keys = {
        (int(g), str(st), int(w))
        for g in GESTURES
        for st in SETS
        for w in EXPLANATION_WINDOW_INDICES
    }
    required = {
        "ig_abs", "ig_signed", "occlusion_abs", "occlusion_signed",
        "ig_convergence_delta", "prediction", "target_gesture",
        "subject_id", "condition", "model", "gesture", "set",
        "window_index", "window_id", "source_recording", "window_array_file",
        "window_array_sha256", "protocol_version", "archive_type",
        "explanation_window_indices", "ig_steps", "ig_internal_batch_size",
        "occlusion_segment_length", "occlusion_step",
    }
    expected_shapes = {
        "ig_abs": (expected_n, N_CHANNELS, WINDOW_SAMPLES),
        "ig_signed": (expected_n, N_CHANNELS, WINDOW_SAMPLES),
        "occlusion_abs": (expected_n, N_CHANNELS, WINDOW_SAMPLES),
        "occlusion_signed": (expected_n, N_CHANNELS, WINDOW_SAMPLES),
        "ig_convergence_delta": (expected_n,),
        "prediction": (expected_n,),
        "target_gesture": (expected_n,),
        "subject_id": (expected_n,),
        "condition": (expected_n,),
        "model": (expected_n,),
        "gesture": (expected_n,),
        "set": (expected_n,),
        "window_index": (expected_n,),
        "window_id": (expected_n,),
        "source_recording": (expected_n,),
        "window_array_file": (expected_n,),
        "window_array_sha256": (expected_n,),
    }

    try:
        with np.load(path, allow_pickle=True) as z:
            missing = sorted(required - set(z.files))
            if missing:
                return False, f"missing required key(s): {', '.join(missing)}"

            for key, shape in expected_shapes.items():
                arr = np.asarray(z[key])
                if arr.shape != shape:
                    return False, f"{key} shape {arr.shape}; expected {shape}"

            # All numerical attribution and prediction arrays must be finite.
            for key in (
                "ig_abs", "ig_signed", "occlusion_abs", "occlusion_signed",
                "ig_convergence_delta", "prediction", "target_gesture",
                "subject_id", "gesture", "window_index",
            ):
                arr = np.asarray(z[key])
                if not np.isfinite(arr).all():
                    return False, f"{key} contains NaN/Inf"

            protocol_arr = np.asarray(z["protocol_version"]).reshape(-1)
            if protocol_arr.size != 1:
                return False, f"protocol_version metadata has {protocol_arr.size} values; expected 1"
            protocol = str(protocol_arr[0])
            if protocol != PROTOCOL_VERSION:
                return False, f"protocol_version={protocol!r}; expected {PROTOCOL_VERSION!r}"

            archive_arr = np.asarray(z["archive_type"]).reshape(-1)
            if archive_arr.size != 1:
                return False, f"archive_type metadata has {archive_arr.size} values; expected 1"
            archive_type = str(archive_arr[0])
            expected_archive_type = "per_window_per_gesture_IG_and_occlusion"
            if archive_type != expected_archive_type:
                return False, f"archive_type={archive_type!r}; expected {expected_archive_type!r}"

            # Provenance/configuration metadata must match the frozen runtime.
            ewi = np.asarray(z["explanation_window_indices"], dtype=int).reshape(-1)
            if not np.array_equal(ewi, np.asarray(EXPLANATION_WINDOW_INDICES, dtype=int)):
                return False, f"explanation_window_indices={ewi.tolist()}; expected {list(EXPLANATION_WINDOW_INDICES)}"

            scalar_specs = {
                "ig_steps": int(IG_STEPS),
                "ig_internal_batch_size": int(IG_INTERNAL_BATCH_SIZE),
                "occlusion_segment_length": int(OCCLUSION_SEGMENT_LENGTH),
                "occlusion_step": int(OCCLUSION_STEP),
            }
            for key, expected_value in scalar_specs.items():
                vals = np.asarray(z[key]).reshape(-1)
                if vals.size != 1:
                    return False, f"{key} metadata has {vals.size} values; expected 1"
                if int(vals[0]) != expected_value:
                    return False, f"{key}={int(vals[0])}; expected {expected_value}"

            # Exact subject/condition/model identity checks prevent a correctly
            # shaped archive from being attached to the wrong experiment.
            subjects = np.asarray(z["subject_id"], dtype=int)
            conditions = np.asarray(z["condition"]).astype(str)
            models = np.asarray(z["model"]).astype(str)
            if subject is not None and not np.all(subjects == int(subject)):
                return False, f"subject_id contains values other than {int(subject)}"
            if condition is not None and not np.all(conditions == str(condition)):
                return False, f"condition contains values other than {str(condition)!r}"
            if model_name is not None and not np.all(models == str(model_name)):
                return False, f"model contains values other than {str(model_name)!r}"

            gesture = np.asarray(z["gesture"], dtype=int)
            sets = np.asarray(z["set"]).astype(str)
            wi = np.asarray(z["window_index"], dtype=int)
            observed = {(int(g), str(st), int(w)) for g, st, w in zip(gesture, sets, wi)}
            if observed != expected_keys:
                missing_keys = sorted(expected_keys - observed)
                extra_keys = sorted(observed - expected_keys)
                return False, (
                    f"gesture/set/window subset mismatch; "
                    f"missing={missing_keys[:10]}, extra={extra_keys[:10]}"
                )

            # Each expected key must occur exactly once; set equality alone does
            # not detect duplicate rows replacing another key in malformed data.
            key_counts = {}
            for key in zip(gesture, sets, wi):
                norm = (int(key[0]), str(key[1]), int(key[2]))
                key_counts[norm] = key_counts.get(norm, 0) + 1
            duplicates = sorted(k for k, c in key_counts.items() if c != 1)
            if duplicates:
                return False, f"duplicate/missing explanation keys: {duplicates[:10]}"

            # Metadata arrays must also contain finite/non-empty identifiers.
            for key in ("condition", "model", "set", "window_id", "source_recording", "window_array_file"):
                vals = np.asarray(z[key]).astype(str)
                if any(v.strip() == "" for v in vals):
                    return False, f"{key} contains an empty identifier"

            return True, "OK"
    except Exception as exc:
        return False, f"could not read/validate NPZ: {type(exc).__name__}: {exc}"


def raw_dl_archive_is_valid(
    path: Path,
    subject: int | None = None,
    condition: str | None = None,
    model_name: str | None = None,
) -> bool:
    """Boolean compatibility wrapper around the diagnostic raw-archive validator."""
    ok, _ = validate_raw_dl_archive(path, subject, condition, model_name)
    return ok

def count_raw_dl_archives() -> tuple[int, int]:
    expected = len(DL_MODELS) * len(SUBJECTS) * len(CONDITIONS)
    found = sum(
        1
        for subject in SUBJECTS
        for condition in CONDITIONS
        for model in DL_MODELS
        if raw_dl_archive_is_valid(raw_dl_path(subject, condition, model))
    )
    return found, expected


def write_raw_archive_readme() -> None:
    text = (
        "BLOCK 13 RAW ATTRIBUTION ARCHIVE\n"
        "================================\n\n"
        "ML/: one NPZ per subject/condition/model containing per-window true-class "
        "signed and absolute SHAP values for the deterministic 63-window explanation subset.\n"
        "DL/: one NPZ per subject/condition/model containing per-window signed/absolute "
        "Integrated Gradients and occlusion attributions for the same 63 windows.\n\n"
        "The DL manuscript tables are gesture-level means over the 9 windows belonging "
        "to each gesture. These raw archives preserve the individual explanation units "
        "used to produce those means.\n\n"
        "The IG zero tensor is a computational baseline only. Attribution does not establish "
        "anatomical source or causal neurophysiology.\n"
    )
    p = RAW_ROOT / "README.txt"
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


# =============================================================================
# 03. FROZEN BLOCK 05 FEATURE SCHEMA
# =============================================================================

CHANNEL_FEATURE_DEFINITIONS = [
    ("RMS", "time_domain", None),
    ("VAR", "time_domain", None),
    ("P2P", "time_domain", None),
    ("MAD", "time_domain", None),
    ("IQR", "time_domain", None),
    ("ZCR", "time_domain", None),
    ("HJ_ACTIVITY", "hjorth", None),
    ("HJ_MOBILITY", "hjorth", None),
    ("HJ_COMPLEXITY", "hjorth", None),
    ("DELTA_ABS", "spectral", "delta"),
    ("THETA_ABS", "spectral", "theta"),
    ("ALPHA_ABS", "spectral", "alpha"),
    ("BETA_ABS", "spectral", "beta"),
    ("DELTA_REL", "spectral_relative", "delta"),
    ("THETA_REL", "spectral_relative", "theta"),
    ("ALPHA_REL", "spectral_relative", "alpha"),
    ("BETA_REL", "spectral_relative", "beta"),
    ("PEAK_FREQ", "spectral", "broadband"),
    ("SPEC_ENTROPY", "spectral", "broadband"),
    ("SEF95", "spectral", "broadband"),
    ("ALPHA_BETA", "spectral_ratio", "alpha_beta"),
    ("SKEW", "distribution", None),
    ("KURT", "distribution", None),
    ("LINE_LENGTH", "complexity", None),
]

CHANNEL_NAMES = [f"CH{i:02d}" for i in range(1, 14)]

FEATURE_NAMES = []
FEATURE_FAMILIES = []
FEATURE_CHANNELS = []
FEATURE_FREQUENCIES = []

for ch in CHANNEL_NAMES:
    for feat, family, freq in CHANNEL_FEATURE_DEFINITIONS:
        FEATURE_NAMES.append(f"{ch}_{feat}")
        FEATURE_FAMILIES.append(family)
        FEATURE_CHANNELS.append(ch)
        FEATURE_FREQUENCIES.append(freq)

for ch in CHANNEL_NAMES:
    FEATURE_NAMES.append(f"{ch}_MEAN")
    FEATURE_FAMILIES.append("channel_mean")
    FEATURE_CHANNELS.append(ch)
    FEATURE_FREQUENCIES.append(None)

for ch in CHANNEL_NAMES:
    FEATURE_NAMES.append(f"{ch}_SD")
    FEATURE_FAMILIES.append("channel_sd")
    FEATURE_CHANNELS.append(ch)
    FEATURE_FREQUENCIES.append(None)

for name in (
    "GLOBAL_RMS",
    "INTRAWINDOW_MEAN_CHANNEL_CORR",
    "GLOBAL_MEAN_CHANNEL_SD",
):
    FEATURE_NAMES.append(name)
    FEATURE_FAMILIES.append("global")
    FEATURE_CHANNELS.append("GLOBAL")
    FEATURE_FREQUENCIES.append(None)

if len(FEATURE_NAMES) != N_FEATURES:
    raise RuntimeError(
        f"Internal BLOCK13 feature schema error: {len(FEATURE_NAMES)} != {N_FEATURES}"
    )

FEATURE_SCHEMA = pd.DataFrame({
    "feature_index": np.arange(N_FEATURES),
    "feature_id": [f"F{i+1:03d}" for i in range(N_FEATURES)],
    "feature_name": FEATURE_NAMES,
    "feature_family": FEATURE_FAMILIES,
    "channel": FEATURE_CHANNELS,
    "frequency_group": FEATURE_FREQUENCIES,
})


# =============================================================================
# 04. BASIC VALIDATION HELPERS
# =============================================================================

def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"Required {label} is missing: {path}")
    return path


def load_json(path: Path) -> dict:
    require_file(path, "JSON")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_finite(name: str, arr: np.ndarray) -> None:
    a = np.asarray(arr)
    if not np.isfinite(a).all():
        raise RuntimeError(f"{name} contains NaN/Inf.")


def canonical_condition(v) -> str:
    s = str(v).strip().upper()
    if s not in CONDITIONS:
        raise RuntimeError(f"Invalid condition: {v!r}")
    return s


def canonical_set(v) -> str:
    s = str(v).strip().upper()
    if s in {"A", "SETA"}:
        return "SetA"
    if s in {"B", "SETB"}:
        return "SetB"
    if s in {"C", "SETC"}:
        return "SetC"
    raise RuntimeError(f"Invalid set label: {v!r}")


# =============================================================================
# 05. BLOCK 04 HANDOFF
# =============================================================================

def parse_window_index(window_id: str) -> int:
    """
    Parse the frozen canonical BLOCK04 window identity and return the
    zero-based array index.

    Canonical BLOCK04 IDs are:
        S01_ME_1_SetA_W500_O50_Window0001 ... Window0021

    IMPORTANT:
        window_number is 1..21
        window_index  is 0..20

    The previous implementation returned the terminal window number
    directly, so Window0021 became index 21 and failed validation.
    This function deliberately converts the canonical 1-based number to
    the zero-based array position used by the NPZ window arrays.
    """
    text = str(window_id).strip()

    m = re.fullmatch(
        r"S(?P<subject>\d+)_"
        r"(?P<condition>ME|MI)_"
        r"(?P<gesture>\d+)_"
        r"Set(?P<set>[ABC])_"
        r"W(?P<window_size>\d+)_"
        r"O(?P<overlap>\d+)_"
        r"Window(?P<number>\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if not m:
        raise RuntimeError(
            f"Invalid canonical BLOCK04 window_id: {text!r}"
        )

    number = int(m.group("number"))

    if not 1 <= number <= 21:
        raise RuntimeError(
            f"BLOCK04 window_number must be 1..21; "
            f"got {number} from window_id={text!r}"
        )

    # The NPZ arrays are zero-indexed while the canonical manifest
    # window_number is one-indexed.
    return number - 1


def resolve_window_array(path_value: str) -> Path:
    p = Path(str(path_value))
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        BLOCK04_ROOT / str(path_value),
        BLOCK04_ROOT / "final_windows" / "data" / "W500_O50" / str(path_value),
        BLOCK04_ROOT / "final_windows" / "data" / str(path_value),
    ]
    for c in candidates:
        if c.exists():
            return c
    # Last-resort deterministic recursive basename search.
    matches = sorted(BLOCK04_ROOT.rglob(p.name))
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(
        f"Cannot resolve BLOCK04 window_array_file={path_value!r}; "
        f"matches={len(matches)}"
    )


def verify_block04() -> tuple[pd.DataFrame, dict]:
    manifest = require_file(
        BLOCK04_ROOT / "final_window_manifest.csv",
        "BLOCK04 final window manifest",
    )
    df = pd.read_csv(manifest)

    required = {
        "window_id", "subject", "condition", "gesture_code", "set",
        "source_recording", "window_samples", "overlap_percent",
        "step_samples", "start_sample", "end_sample", "window_array_file",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"BLOCK04 manifest missing columns: {sorted(missing)}")

    if len(df) != N_WINDOWS:
        raise RuntimeError(
            f"BLOCK04 manifest has {len(df)} rows; expected {N_WINDOWS}"
        )

    df["subject"] = pd.to_numeric(df["subject"], errors="raise").astype(int)
    df["condition"] = df["condition"].map(canonical_condition)
    df["gesture_code"] = pd.to_numeric(
        df["gesture_code"], errors="raise"
    ).astype(int)
    df["set"] = df["set"].map(canonical_set)
    df["window_samples"] = pd.to_numeric(
        df["window_samples"], errors="raise"
    ).astype(int)
    df["overlap_percent"] = pd.to_numeric(
        df["overlap_percent"], errors="raise"
    ).astype(int)
    df["step_samples"] = pd.to_numeric(
        df["step_samples"], errors="raise"
    ).astype(int)
    df["start_sample"] = pd.to_numeric(
        df["start_sample"], errors="raise"
    ).astype(int)
    df["end_sample"] = pd.to_numeric(
        df["end_sample"], errors="raise"
    ).astype(int)

    if set(df["subject"]) != set(SUBJECTS):
        raise RuntimeError("BLOCK04 subject IDs are not exactly 1..25.")
    if set(df["condition"]) != set(CONDITIONS):
        raise RuntimeError("BLOCK04 conditions are not exactly ME/MI.")
    if set(df["gesture_code"]) != set(GESTURES):
        raise RuntimeError("BLOCK04 gesture codes are not exactly 1..7.")
    if set(df["set"]) != set(SETS):
        raise RuntimeError("BLOCK04 sets are not exactly SetA/SetB/SetC.")
    if not (df["window_samples"] == WINDOW_SAMPLES).all():
        raise RuntimeError("BLOCK04 contains non-500-sample windows.")
    if not (df["overlap_percent"] == OVERLAP_PERCENT).all():
        raise RuntimeError("BLOCK04 contains non-50% overlap.")
    if not (df["step_samples"] == STEP_SAMPLES).all():
        raise RuntimeError("BLOCK04 contains non-250-sample steps.")

    # Verify actual array contract without loading every window into memory.
    checked = {}
    for value in df["window_array_file"].astype(str).unique():
        p = resolve_window_array(value)
        with np.load(p, allow_pickle=False) as z:
            if "eeg_windows" not in z.files:
                raise RuntimeError(f"{p}: missing eeg_windows")
            arr = np.asarray(z["eeg_windows"])
            if arr.shape != (21, 13, 500):
                raise RuntimeError(
                    f"{p}: eeg_windows shape {arr.shape}; expected (21,13,500)"
                )
            for k in ("start_samples", "end_samples"):
                if k not in z.files:
                    raise RuntimeError(f"{p}: missing {k}")
                if np.asarray(z[k]).shape != (21,):
                    raise RuntimeError(f"{p}: invalid {k} shape")
        checked[value] = str(p)

    df["window_index"] = df["window_id"].map(parse_window_index)

    # Preserve the canonical one-based manifest identity explicitly.
    # This is useful for auditability and prevents future confusion between
    # Window0001..Window0021 and NPZ array positions 0..20.
    df["window_number"] = df["window_index"] + 1

    if not df["window_index"].between(0, 20).all():
        raise RuntimeError("BLOCK04 window indices must be 0..20.")

    if not df["window_number"].between(1, 21).all():
        raise RuntimeError("BLOCK04 window numbers must be 1..21.")

    # Verify that every recording contains the complete canonical sequence.
    for key, group in df.groupby(
        ["subject", "condition", "gesture_code", "set"],
        sort=False,
    ):
        indices = sorted(group["window_index"].astype(int).tolist())
        numbers = sorted(group["window_number"].astype(int).tolist())

        if indices != list(range(21)):
            raise RuntimeError(
                f"BLOCK04 non-contiguous window_index for {key}: {indices}"
            )

        if numbers != list(range(1, 22)):
            raise RuntimeError(
                f"BLOCK04 non-contiguous window_number for {key}: {numbers}"
            )

    for value, pstr in checked.items():
        p = Path(pstr)
        sub = df[df["window_array_file"].astype(str) == value]
        with np.load(p, allow_pickle=False) as z:
            starts = np.asarray(z["start_samples"], dtype=int)
            ends = np.asarray(z["end_samples"], dtype=int)
        for r in sub.itertuples(index=False):
            idx = int(r.window_index)
            if starts[idx] != int(r.start_sample) or ends[idx] != int(r.end_sample):
                raise RuntimeError(
                    f"BLOCK04 manifest/array mismatch: {r.window_id} -> {p.name}"
                )

    return df, {
        "status": "PASS",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "window_arrays": len(checked),
        "shape": [21, 13, 500],
    }


# =============================================================================
# 06. BLOCK 05 HANDOFF
# =============================================================================

def verify_block05() -> dict:
    complete = load_json(BLOCK05_ROOT / "BLOCK05_complete.json")
    summary = load_json(BLOCK05_ROOT / "BLOCK05_summary.json")

    if str(complete.get("status", "")).upper() != "PASS":
        raise RuntimeError("BLOCK05 completion status is not PASS.")
    if str(complete.get("frozen_configuration", "")) != "W500_O50":
        raise RuntimeError("BLOCK05 frozen configuration is not W500_O50.")
    if int(complete.get("feature_count", -1)) != N_FEATURES:
        raise RuntimeError("BLOCK05 feature count is not 341.")
    if int(complete.get("total_windows", -1)) != N_WINDOWS:
        raise RuntimeError("BLOCK05 total windows is not 22050.")
    if str(summary.get("status", "")).upper() != "PASS":
        raise RuntimeError("BLOCK05 summary status is not PASS.")

    manifest = require_file(
        BLOCK05_ROOT / "manifests" / "feature_manifest.csv",
        "BLOCK05 feature manifest",
    )
    columns = require_file(
        BLOCK05_ROOT / "manifests" / "feature_column_manifest.csv",
        "BLOCK05 feature-column manifest",
    )

    cm = pd.read_csv(columns)
    if len(cm) != N_FEATURES:
        raise RuntimeError(f"BLOCK05 column manifest has {len(cm)} rows.")

    # Prefer the authoritative manifest schema; verify its names against
    # the frozen feature definition.
    name_col = "feature_name" if "feature_name" in cm.columns else cm.columns[0]
    actual_names = cm[name_col].astype(str).tolist()
    if actual_names != FEATURE_NAMES:
        raise RuntimeError(
            "BLOCK05 feature-name order differs from the frozen 341-feature schema."
        )

    fm = pd.read_csv(manifest)
    if len(fm) != N_WINDOWS:
        raise RuntimeError(
            f"BLOCK05 feature manifest has {len(fm)} rows; expected {N_WINDOWS}"
        )
    required = {"subject_id", "condition", "gesture_code", "set",
                "feature_file", "feature_row"}
    missing = required - set(fm.columns)
    if missing:
        raise RuntimeError(
            f"BLOCK05 feature manifest missing columns: {sorted(missing)}"
        )

    return {
        "status": "PASS",
        "complete": str(BLOCK05_ROOT / "BLOCK05_complete.json"),
        "feature_manifest": str(manifest),
        "feature_column_manifest": str(columns),
        "feature_manifest_sha256": sha256_file(manifest),
        "feature_column_manifest_sha256": sha256_file(columns),
    }


# =============================================================================
# 07. BLOCK 06 MODEL HANDOFF
# =============================================================================

def verify_runtime_dependencies() -> dict:
    """
    Verify the runtime needed to deserialize and explain the frozen BLOCK06
    artifacts BEFORE any expensive XAI computation starts.

    In particular, Fold_*_XGB.joblib stores an xgboost estimator.  If xgboost
    is absent, joblib.load() cannot reconstruct the frozen object.  BLOCK13
    therefore fails early rather than silently skipping XGB or substituting a
    different model.

    The warning seen in earlier runs also showed that the frozen sklearn
    estimators were serialized under sklearn 1.6.1 while the runtime had 1.5.1.
    This is treated as a compatibility error, not ignored, because BLOCK13 is
    an attribution analysis of frozen models and must not change their runtime
    semantics through an unverified estimator version.
    """
    errors = []

    if joblib is None:
        errors.append(f"joblib unavailable: {JOBLIB_ERROR}")
    if shap is None:
        errors.append(f"SHAP unavailable: {SHAP_ERROR}")
    if xgb is None:
        errors.append(
            "xgboost unavailable: " + str(XGBOOST_ERROR) +
            ". BLOCK06 XGBGestureClassifier contains a nested "
            "xgboost.sklearn.XGBClassifier and cannot be deserialized without "
            "the xgboost package."
        )

    sklearn_version = None
    try:
        import sklearn
        sklearn_version = str(sklearn.__version__)
    except Exception as exc:
        errors.append(f"scikit-learn unavailable: {exc!r}")

    # BLOCK06 emitted an InconsistentVersionWarning showing 1.6.1 as the
    # serialization version.  Require that exact version for this frozen XAI
    # analysis rather than accepting an unverified estimator runtime.
    if sklearn_version is not None and sklearn_version != "1.6.1":
        errors.append(
            f"scikit-learn version mismatch: runtime={sklearn_version}, "
            "required_for_frozen_BLOCK06=1.6.1. "
            "Earlier BLOCK13 runs already reported that the frozen StandardScaler, "
            "LDA and Pipeline were serialized under sklearn 1.6.1."
        )

    # XGBoost is part of the serialized object contract.  Do not enforce an
    # invented package version here because the artifact itself is the source
    # of truth; first make the package importable, then validate the nested model
    # after deserialization.
    if errors:
        raise RuntimeError(
            "BLOCK13 runtime preflight failed before XAI computation:\n"
            + "\n".join(f" - {e}" for e in errors)
        )

    return {
        "status": "PASS",
        "joblib": getattr(joblib, "__version__", "unknown"),
        "shap": getattr(shap, "__version__", "unknown"),
        "xgboost": getattr(xgb, "__version__", "unknown"),
        "scikit_learn": sklearn_version,
    }


def verify_block06() -> dict:
    complete_candidates = [
        BLOCK06_ROOT / "BLOCK06_complete.json",
        BLOCK06_ROOT / "BLOCK06_summary.json",
    ]
    existing = [p for p in complete_candidates if p.exists()]
    if not existing:
        raise RuntimeError("No BLOCK06 completion/summary artifact found.")

    info = {"status": "PASS", "model_files": 0, "missing": []}

    model_root = BLOCK06_ROOT / "models" / "LOSO"
    if not model_root.exists():
        raise RuntimeError(f"BLOCK06 LOSO model directory missing: {model_root}")

    expected = []
    for subject in SUBJECTS:
        for condition in CONDITIONS:
            for model in ML_MODELS:
                expected.append(
                    model_root / f"Fold_S{subject:02d}_{condition}_{model}.joblib"
                )

    for p in expected:
        if not p.exists():
            info["missing"].append(str(p))
        else:
            info["model_files"] += 1

    if info["missing"]:
        raise RuntimeError(
            "BLOCK06 LOSO model handoff incomplete. "
            f"Missing {len(info['missing'])}/250 model files. "
            f"First missing: {info['missing'][:3]}"
        )

    return info


# =============================================================================
# 08. FEATURE MATRIX ACCESS
# =============================================================================

FEATURE_FILE_CACHE: dict[str, np.ndarray] = {}


def resolve_feature_file(value: str) -> Path:
    p = Path(str(value))
    candidates = [
        p,
        BLOCK05_ROOT / str(value),
        BLOCK05_ROOT / "features" / str(value),
    ]
    for c in candidates:
        if c.exists():
            return c

    # Search by basename if manifest stores a relative/legacy path.
    matches = sorted(BLOCK05_ROOT.rglob(p.name))
    if len(matches) == 1:
        return matches[0]

    raise RuntimeError(
        f"Cannot resolve BLOCK05 feature_file={value!r}; matches={len(matches)}"
    )


def load_feature_matrix(path: Path) -> np.ndarray:
    key = str(path)
    if key in FEATURE_FILE_CACHE:
        return FEATURE_FILE_CACHE[key]
    with np.load(path, allow_pickle=False) as z:
        if "features" not in z.files:
            raise RuntimeError(f"{path}: missing features")
        X = np.asarray(z["features"], dtype=np.float64)
    if X.shape != (21, N_FEATURES):
        raise RuntimeError(
            f"{path}: feature shape {X.shape}; expected (21,341)"
        )
    assert_finite(str(path), X)
    FEATURE_FILE_CACHE[key] = X
    return X


def build_feature_lookup(block04_df: pd.DataFrame) -> pd.DataFrame:
    fm = pd.read_csv(
        BLOCK05_ROOT / "manifests" / "feature_manifest.csv"
    ).copy()

    fm["subject_id"] = pd.to_numeric(
        fm["subject_id"], errors="raise"
    ).astype(int)
    fm["condition"] = fm["condition"].map(canonical_condition)
    fm["gesture_code"] = pd.to_numeric(
        fm["gesture_code"], errors="raise"
    ).astype(int)
    fm["set"] = fm["set"].map(canonical_set)
    fm["feature_row"] = pd.to_numeric(
        fm["feature_row"], errors="raise"
    ).astype(int)

    # Keep the exact one-to-one window provenance.
    if len(fm) != N_WINDOWS:
        raise RuntimeError("BLOCK05 feature manifest cardinality mismatch.")

    merged = block04_df[
        [
            "window_id", "subject", "condition", "gesture_code", "set",
            "source_recording", "window_index",
        ]
    ].copy()

    # Match using subject/condition/gesture/set/window index.
    fm["window_index"] = fm["feature_row"]
    merged = merged.merge(
        fm[
            [
                "subject_id", "condition", "gesture_code", "set",
                "feature_file", "feature_row", "window_index",
            ]
        ],
        left_on=[
            "subject", "condition", "gesture_code", "set", "window_index"
        ],
        right_on=[
            "subject_id", "condition", "gesture_code", "set", "window_index"
        ],
        how="left",
        validate="one_to_one",
    )

    if merged["feature_file"].isna().any():
        raise RuntimeError("BLOCK05 feature provenance could not be joined to BLOCK04.")

    return merged


# =============================================================================
# 09. DETERMINISTIC EXPLANATION SAMPLE
# =============================================================================

def build_explanation_rows(
    feature_lookup: pd.DataFrame,
    subject: int,
    condition: str,
) -> pd.DataFrame:
    rows = feature_lookup[
        (feature_lookup["subject"] == subject)
        & (feature_lookup["condition"] == condition)
        & feature_lookup["window_index"].isin(EXPLANATION_WINDOW_INDICES)
    ].copy()

    # Each selected window index exists independently in SetA, SetB,
    # and SetC. Therefore the deterministic explanation subset contains:
    #   7 gestures × 3 sets × 3 selected windows = 63 rows
    # per subject and condition.
    expected = (
        N_GESTURES
        * len(SETS)
        * len(EXPLANATION_WINDOW_INDICES)
    )

    if len(rows) != expected:
        raise RuntimeError(
            f"Explanation subset incomplete for S{subject:02d} {condition}: "
            f"{len(rows)} vs {expected} "
            f"(gestures={N_GESTURES}, sets={len(SETS)}, "
            f"selected_windows={len(EXPLANATION_WINDOW_INDICES)})"
        )

    # Verify that the 63 rows are structurally complete rather than
    # merely matching the expected row count.
    expected_keys = {
        (gesture, set_name, window_index)
        for gesture in GESTURES
        for set_name in SETS
        for window_index in EXPLANATION_WINDOW_INDICES
    }
    observed_keys = {
        (
            int(r.gesture_code),
            str(r.set),
            int(r.window_index),
        )
        for r in rows.itertuples(index=False)
    }

    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise RuntimeError(
            f"Explanation subset key mismatch for S{subject:02d} {condition}: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    if set(rows["gesture_code"]) != set(GESTURES):
        raise RuntimeError("Explanation subset does not contain all seven gestures.")

    return rows.sort_values(
        ["gesture_code", "set", "window_index"]
    ).reset_index(drop=True)


def materialize_feature_rows(rows: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    X_parts = []
    out_rows = []

    for r in rows.itertuples(index=False):
        path = resolve_feature_file(r.feature_file)
        X = load_feature_matrix(path)
        idx = int(r.feature_row)
        if not 0 <= idx < 21:
            raise RuntimeError(f"Invalid feature_row={idx} in {path}")
        x = X[idx]
        assert_finite(f"{path.name}[{idx}]", x)
        X_parts.append(x)
        out_rows.append({
            "window_id": r.window_id,
            "subject": int(r.subject),
            "condition": r.condition,
            "gesture_code": int(r.gesture_code),
            "set": r.set,
            "source_recording": r.source_recording,
            "window_index": int(r.window_index),
            "feature_file": str(path),
            "feature_row": idx,
        })

    return np.vstack(X_parts), pd.DataFrame(out_rows)


# =============================================================================
# 10. SHAP
# =============================================================================

def make_predict_fn(model):
    def predict_fn(x):
        x = np.asarray(x, dtype=np.float64)
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(x)
            return np.asarray(p)
        if hasattr(model, "decision_function"):
            d = np.asarray(model.decision_function(x))
            if d.ndim == 1:
                d = np.column_stack([-d, d])
            # SHAP only needs a stable model output. Use decision scores
            # without inventing probabilities.
            return d
        pred = np.asarray(model.predict(x))
        classes = np.asarray(getattr(model, "classes_", np.unique(pred)))
        out = np.zeros((len(pred), len(classes)), dtype=float)
        for i, y in enumerate(pred):
            j = int(np.where(classes == y)[0][0])
            out[i, j] = 1.0
        return out
    return predict_fn


def normalize_shap_output(values, n_samples: int, n_features: int):
    """
    Return an array shaped:
        samples x features x classes
    without changing signed SHAP values.
    """
    if isinstance(values, list):
        arrs = [np.asarray(v) for v in values]
        if all(a.ndim == 2 for a in arrs):
            return np.stack(arrs, axis=2)
        raise RuntimeError("Unsupported list-style SHAP output.")

    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr[:, :, None]

    if arr.ndim == 3:
        # Common SHAP layouts:
        # samples x features x classes
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            return arr
        # samples x classes x features
        if arr.shape[0] == n_samples and arr.shape[2] == n_features:
            return np.transpose(arr, (0, 2, 1))

    raise RuntimeError(
        f"Unsupported SHAP value shape {arr.shape}; "
        f"expected 2D or 3D compatible with ({n_samples},{n_features})."
    )


def explain_ml_model(model, X: np.ndarray, y_true: np.ndarray):
    """
    Compute signed SHAP values for the true gesture class.

    Leakage rule:
        The explanation samples are held-out-subject windows and the model is
        the corresponding frozen BLOCK06 LOSO model.  BLOCK13 never fits a
        model and never uses the held-out data as a learned background.

    SHAP reference:
        - Tree models use TreeExplainer on the frozen tree estimator.
        - Non-tree models use a deterministic zero feature vector reference.
          This is explicitly computational and is NOT a physiological
          baseline.
    """
    if shap is None:
        raise RuntimeError(
            f"SHAP is unavailable. Install shap in the BLOCK13 environment. "
            f"{SHAP_ERROR}"
        )

    X = np.asarray(X, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)

    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise RuntimeError(
            f"SHAP input shape {X.shape}; expected (n,{N_FEATURES})."
        )
    if len(y_true) != len(X):
        raise RuntimeError(
            f"SHAP target length {len(y_true)} != input rows {len(X)}."
        )
    assert_finite("SHAP input X", X)

    # XGBGestureClassifier is a serialized wrapper.  SHAP must see the
    # underlying xgboost.sklearn.XGBClassifier, not the wrapper itself.
    shap_model = _unwrap_tree_estimator(model)

    tree_like = (
        _is_xgb_gesture_wrapper(model)
        or shap_model.__class__.__name__ in {
            "RandomForestClassifier",
            "ExtraTreesClassifier",
            "RandomForestRegressor",
            "XGBClassifier",
            "XGBModel",
        }
    )

    try:
        if tree_like:
            explainer = shap.TreeExplainer(shap_model)

            # SHAP has changed this API across releases.  Keep the primary
            # call explicit and use a compatibility fallback only when the
            # installed version rejects check_additivity.
            try:
                values = explainer.shap_values(
                    X,
                    check_additivity=False,
                )
            except TypeError:
                values = explainer.shap_values(X)

        else:
            # Zero is a computational feature-space reference.  It is not
            # described as a resting EEG baseline anywhere in BLOCK13.
            background = np.zeros((1, N_FEATURES), dtype=np.float64)

            predict_fn = make_predict_fn(model)
            try:
                explainer = shap.Explainer(
                    predict_fn,
                    background,
                    algorithm="permutation",
                )
            except Exception:
                explainer = shap.Explainer(
                    predict_fn,
                    background,
                )

            explanation = explainer(
                X,
                max_evals=2 * N_FEATURES + 1,
            )
            values = explanation.values

    except Exception as exc:
        raise RuntimeError(
            f"SHAP explanation failed for {type(model).__name__}: {exc}"
        ) from exc

    vals = normalize_shap_output(values, len(X), N_FEATURES)

    if vals.shape[0] != len(X) or vals.shape[1] != N_FEATURES:
        raise RuntimeError(
            f"Normalized SHAP shape {vals.shape} is incompatible with "
            f"X={X.shape}."
        )

    assert_finite("normalized SHAP values", vals)

    # The class order of SHAP output must be explicitly tied to model.classes_.
    # For the XGB wrapper this is 1..7, while the nested XGB output columns are
    # 0..6; the column POSITION remains identical.
    classes = _model_classes_for_xai(model, vals.shape[2])

    true_indices = np.empty(len(y_true), dtype=int)
    for i, y in enumerate(y_true):
        matches = np.flatnonzero(classes == int(y))
        if len(matches) != 1:
            raise RuntimeError(
                f"Cannot map true gesture {int(y)} to SHAP output classes "
                f"{classes.tolist()} for {type(model).__name__}."
            )
        true_indices[i] = int(matches[0])

    signed_true = np.stack(
        [
            vals[i, :, true_indices[i]]
            for i in range(len(X))
        ],
        axis=0,
    )

    assert_finite("signed true-class SHAP", signed_true)

    return signed_true, vals, classes


# =============================================================================
# 11. ML XAI
# =============================================================================

def run_ml_xai(feature_lookup: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_frames = []
    channel_rows_out = []

    model_root = BLOCK06_ROOT / "models" / "LOSO"

    for subject in SUBJECTS:
        for condition in CONDITIONS:
            rows = build_explanation_rows(feature_lookup, subject, condition)
            X, meta = materialize_feature_rows(rows)
            y = meta["gesture_code"].to_numpy(dtype=int)

            for model_name in ML_MODELS:
                model_path = (
                    model_root
                    / f"Fold_S{subject:02d}_{condition}_{model_name}.joblib"
                )
                if not model_path.exists():
                    raise RuntimeError(f"Missing BLOCK06 model: {model_path}")

                log(
                    f"ML SHAP: S{subject:02d} {condition} {model_name} "
                    f"n={len(X)}"
                )

                cache_dir = BLOCK13_ROOT / "cache_ml_shap"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / f"S{subject:02d}_{condition}_{model_name}.npy"
                signed_true = None
                if cache_path.exists():
                    try:
                        tmp = np.load(cache_path)
                        if tmp.shape == (len(X), N_FEATURES):
                            signed_true = tmp
                            log("   cache hit -> skipping SHAP recompute")
                    except Exception:
                        signed_true = None
                if signed_true is None:
                    try:
                        model = joblib.load(model_path)
                    except AttributeError as exc:
                        raise RuntimeError(
                            f"Unable to deserialize BLOCK06 model {model_path.name}. "
                            "This artifact was serialized with a custom "
                            "__main__.XGBGestureClassifier. BLOCK13 includes the "
                            "compatibility class."
                        ) from exc
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to load frozen BLOCK06 model {model_path}: {exc}"
                        ) from exc

                    if model_name == "XGB":
                        if not _is_xgb_gesture_wrapper(model):
                            raise RuntimeError(
                                f"{model_path.name}: expected serialized "
                                "XGBGestureClassifier with nested _model; got "
                                f"{type(model).__name__}."
                            )
                        public_classes = np.asarray(
                            getattr(model, "classes_", []), dtype=int
                        )
                        if public_classes.tolist() != list(GESTURES):
                            raise RuntimeError(
                                f"{model_path.name}: XGBGestureClassifier classes_ "
                                f"{public_classes.tolist()} != {list(GESTURES)}."
                            )
                        nested = model._model
                        if nested is None or not hasattr(nested, "predict_proba"):
                            raise RuntimeError(
                                f"{model_path.name}: nested frozen XGBoost estimator "
                                "is missing predict_proba()."
                            )

                    signed_true, _, _ = explain_ml_model(model, X, y)
                    np.save(cache_path, np.asarray(signed_true, dtype=np.float64))
                    del model
                    gc.collect()

                # The cache itself is already per-window signed true-class SHAP.
                # Export that exact array to the manuscript/audit-facing raw archive
                # without recomputation, including the one-to-one BLOCK04/BLOCK05
                # provenance for every explanation window.
                raw_path = raw_ml_path(subject, condition, model_name)
                write_raw_ml_archive(
                    raw_path, signed_true, meta, subject, condition, model_name
                )

                # signed_true: n x 341
                abs_mean = np.mean(np.abs(signed_true), axis=0)
                signed_mean = np.mean(signed_true, axis=0)

                # Save raw subject/condition/model/gesture/feature values (vectorised).
                n_w = len(X)
                feature_frames.append(pd.DataFrame({
                    "subject_id": subject,
                    "condition": condition,
                    "model": model_name,
                    "gesture": np.repeat(meta["gesture_code"].to_numpy(dtype=int), N_FEATURES),
                    "window_id": np.repeat(meta["window_id"].to_numpy(), N_FEATURES),
                    "feature_id": np.tile(FEATURE_SCHEMA["feature_id"].to_numpy(), n_w),
                    "feature_name": np.tile(FEATURE_SCHEMA["feature_name"].to_numpy(), n_w),
                    "feature_family": np.tile(FEATURE_SCHEMA["feature_family"].to_numpy(), n_w),
                    "channel": np.tile(FEATURE_SCHEMA["channel"].to_numpy(), n_w),
                    "frequency_group": np.tile(FEATURE_SCHEMA["frequency_group"].to_numpy(), n_w),
                    "shap_signed": np.asarray(signed_true, dtype=np.float64).ravel(),
                    "shap_abs": np.abs(np.asarray(signed_true, dtype=np.float64)).ravel(),
                }))

                # Subject-level feature summary, separated by gesture.
                for gesture in GESTURES:
                    sel = meta["gesture_code"].to_numpy() == gesture
                    if not sel.any():
                        continue
                    a = np.mean(np.abs(signed_true[sel]), axis=0)
                    s = np.mean(signed_true[sel], axis=0)
                    order = np.argsort(-a)
                    ranks = np.empty(N_FEATURES, dtype=int)
                    ranks[order] = np.arange(1, N_FEATURES + 1)

                    feature_frames.append(pd.DataFrame({
                        "subject_id": subject,
                        "condition": condition,
                        "model": model_name,
                        "gesture": int(gesture),
                        "window_id": "SUBJECT_GESTURE_SUMMARY",
                        "feature_id": FEATURE_SCHEMA["feature_id"].to_numpy(),
                        "feature_name": FEATURE_SCHEMA["feature_name"].to_numpy(),
                        "feature_family": FEATURE_SCHEMA["feature_family"].to_numpy(),
                        "channel": FEATURE_SCHEMA["channel"].to_numpy(),
                        "frequency_group": FEATURE_SCHEMA["frequency_group"].to_numpy(),
                        "shap_signed": s,
                        "shap_abs": a,
                        "rank": ranks,
                    }))

                    # Channel aggregation required by protocol.
                    for ch in CHANNEL_NAMES:
                        fs = FEATURE_SCHEMA["channel"].to_numpy() == ch
                        raw = float(np.sum(a[fs]))
                        channel_rows_out.append({
                            "subject_id": subject,
                            "condition": condition,
                            "model": model_name,
                            "gesture": int(gesture),
                            "channel": ch,
                            "raw_importance": raw,
                        })

    fdf = pd.concat(feature_frames, ignore_index=True)
    cdf = pd.DataFrame(channel_rows_out)

    if cdf.empty:
        raise RuntimeError("ML channel attribution table is empty.")

    # Normalize channel importance within each subject/model/condition/gesture.
    totals = cdf.groupby(
        ["subject_id", "condition", "model", "gesture"]
    )["raw_importance"].transform("sum")
    cdf["normalized_importance"] = np.where(
        totals > 0,
        cdf["raw_importance"] / totals,
        np.nan,
    )
    cdf["rank"] = (
        cdf.groupby(
            ["subject_id", "condition", "model", "gesture"]
        )["raw_importance"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    assert_finite(
        "ML normalized channel importance",
        cdf["normalized_importance"].dropna().to_numpy(),
    )

    return fdf, cdf


# =============================================================================
# 12. DL MODEL HANDOFF
# =============================================================================

def discover_dl_handoff() -> dict:
    """
    Strict handoff discovery.

    Accepted:
      1. BLOCK07_ROOT/BLOCK07_XAI_HANDOFF.json
      2. BLOCK07_ROOT/**/BLOCK07_XAI_HANDOFF.json
      3. BLOCK07_ROOT/**/xai_handoff.json

    The handoff must specify how each frozen DL checkpoint is reconstructed.
    A state_dict alone is never guessed.
    """
    candidates = [
        BLOCK07_ROOT / "BLOCK07_XAI_HANDOFF.json",
        BLOCK07_ROOT / "xai_handoff.json",
    ]
    candidates += sorted(BLOCK07_ROOT.rglob("BLOCK07_XAI_HANDOFF.json"))
    candidates += sorted(BLOCK07_ROOT.rglob("xai_handoff.json"))

    candidates = list(dict.fromkeys(candidates))
    candidates = [p for p in candidates if p.exists()]

    if not candidates:
        raise RuntimeError(
            "BLOCK07 XAI handoff not found. "
            "Create/retain the frozen BLOCK07_XAI_HANDOFF.json containing "
            "checkpoint path, architecture import path, initialization kwargs, "
            "input transform, and target-output convention. "
            "BLOCK13 will not guess a DL architecture."
        )

    handoff = load_json(candidates[0])
    if str(handoff.get("status", "")).upper() != "PASS":
        raise RuntimeError("BLOCK07 XAI handoff status is not PASS.")

    if handoff.get("window_samples") != WINDOW_SAMPLES:
        raise RuntimeError("BLOCK07 XAI handoff window length is not 500.")
    if handoff.get("channels") != N_CHANNELS:
        raise RuntimeError("BLOCK07 XAI handoff channel count is not 13.")
    if handoff.get("gesture_ids") != list(GESTURES):
        raise RuntimeError("BLOCK07 XAI handoff gesture IDs are not 1..7.")

    return handoff


def import_from_path(import_path: str):
    module_name, sep, attr = str(import_path).rpartition(".")
    if not sep:
        raise RuntimeError(
            f"Invalid import path {import_path!r}; expected module.Class"
        )
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def build_dl_model(spec: dict):
    if torch is None:
        raise RuntimeError(
            f"PyTorch unavailable: {TORCH_ERROR}"
        )

    checkpoint = Path(spec["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = BLOCK07_ROOT / checkpoint
    require_file(checkpoint, f"DL checkpoint for {spec['model']}")

    mode = str(spec.get("checkpoint_type", "state_dict")).lower()

    if mode == "torchscript":
        model = torch.jit.load(str(checkpoint), map_location="cpu")
    elif mode == "state_dict":
        if "model_class" not in spec:
            raise RuntimeError(
                f"{spec['model']}: state_dict checkpoint has no model_class."
            )
        cls = import_from_path(spec["model_class"])
        kwargs = dict(spec.get("init_kwargs", {}))
        model = cls(**kwargs)
        payload = torch.load(str(checkpoint), map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"{checkpoint}: expected state_dict mapping."
            )
        missing, unexpected = model.load_state_dict(payload, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"{spec['model']}: state_dict mismatch; "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
    else:
        raise RuntimeError(
            f"{spec['model']}: unsupported checkpoint_type={mode!r}"
        )

    model.eval()
    return model


# =============================================================================
# 13. DL INPUT / ATTRIBUTION
# =============================================================================

def load_raw_window_from_row(row) -> np.ndarray:
    p = resolve_window_array(str(row.window_array_file))
    idx = int(row.window_index)

    with np.load(p, allow_pickle=False) as z:
        arr = np.asarray(z["eeg_windows"], dtype=np.float32)
    x = arr[idx]

    if x.shape != (N_CHANNELS, WINDOW_SAMPLES):
        raise RuntimeError(
            f"{p}: window shape {x.shape}; expected (13,500)"
        )
    assert_finite(f"raw window {p.name}[{idx}]", x)
    return x


def make_dl_input(x: np.ndarray, spec: dict):
    """
    Apply ONLY the transformation explicitly frozen by BLOCK07 handoff.
    """
    transform = spec.get("input_transform", {"type": "identity"})
    t = str(transform.get("type", "identity")).lower()

    a = np.asarray(x, dtype=np.float32)

    if t == "identity":
        out = a
    elif t == "zscore_per_channel":
        mu = np.mean(a, axis=1, keepdims=True)
        sd = np.std(a, axis=1, keepdims=True)
        out = (a - mu) / np.maximum(sd, 1e-12)
    elif t == "zscore_global":
        mu = float(np.mean(a))
        sd = float(np.std(a))
        out = (a - mu) / max(sd, 1e-12)
    else:
        raise RuntimeError(
            f"Unsupported BLOCK07 input_transform={t!r}; "
            "BLOCK13 will not invent preprocessing."
        )

    assert_finite("DL transformed input", out)
    return torch.from_numpy(out[None, ...])


def model_target_output(model_output, target_class_zero_based: int):
    if model_output.ndim == 1:
        return model_output
    if model_output.ndim != 2:
        raise RuntimeError(
            f"Unexpected DL model output shape {tuple(model_output.shape)}"
        )
    return model_output[:, target_class_zero_based]


def run_integrated_gradients(model, input_tensor, target_zero_based: int):
    baseline = torch.zeros_like(input_tensor)

    ig = IntegratedGradients(model)
    attr, delta = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=target_zero_based,
        n_steps=IG_STEPS,
        internal_batch_size=IG_INTERNAL_BATCH_SIZE,
        return_convergence_delta=True,
    )

    attr = attr.detach().cpu().numpy()[0]
    delta_value = float(delta.detach().cpu().numpy().reshape(-1)[0])

    if attr.shape != (N_CHANNELS, WINDOW_SAMPLES):
        raise RuntimeError(
            f"IG shape {attr.shape}; expected (13,500)"
        )
    assert_finite("IG attribution", attr)
    if not np.isfinite(delta_value):
        raise RuntimeError("IG convergence delta is non-finite.")

    return attr, delta_value


def run_occlusion(model, input_tensor, target_zero_based: int):
    """Deterministic channel-time occlusion via finite target-logit deltas."""
    if input_tensor.ndim != 3 or tuple(input_tensor.shape[1:]) != (N_CHANNELS, WINDOW_SAMPLES):
        raise RuntimeError(f"Occlusion expected N x {N_CHANNELS} x {WINDOW_SAMPLES}, got {tuple(input_tensor.shape)}")
    if not torch.isfinite(input_tensor).all().item():
        raise RuntimeError("Occlusion input contains NaN/Inf before perturbation.")
    target_zero_based = int(target_zero_based)
    if not 0 <= target_zero_based < N_GESTURES:
        raise RuntimeError(f"Occlusion target index out of range: {target_zero_based}")
    model.eval(); x = input_tensor.detach()
    with torch.no_grad(): base_logits = model(x)
    if not torch.isfinite(base_logits).all().item():
        raise RuntimeError("Occlusion original model output contains NaN/Inf.")
    if base_logits.ndim != 2 or tuple(base_logits.shape) != (x.shape[0], N_GESTURES):
        raise RuntimeError(f"Occlusion baseline logits shape {tuple(base_logits.shape)}; expected ({x.shape[0]},{N_GESTURES}).")
    base_target = base_logits[:, target_zero_based].detach()
    n = int(x.shape[0]); attr = torch.zeros_like(x); counts = torch.zeros_like(x)
    step = int(OCCLUSION_STEP); seg = int(OCCLUSION_SEGMENT_LENGTH)
    if step <= 0 or seg <= 0: raise RuntimeError(f"Invalid occlusion parameters: segment={seg}, step={step}")
    channel_starts = list(range(N_CHANNELS)); time_starts = list(range(0, WINDOW_SAMPLES - seg + 1, step))
    if not time_starts: raise RuntimeError(f"Occlusion segment length {seg} exceeds window length {WINDOW_SAMPLES}.")
    # Keep 16 regions per logical batch, but split the real 9 windows into the
    # validated 3-sample forward units (48 model inputs per forward).
    max_regions_per_batch = 16; max_samples_per_forward = 3
    regions = []; perturbed_regions = []
    def flush_region_batch():
        if not perturbed_regions: return
        for sample_start in range(0, n, max_samples_per_forward):
            sample_end = min(sample_start + max_samples_per_forward, n)
            batch = torch.cat([xp[sample_start:sample_end] for xp in perturbed_regions], dim=0)
            if not torch.isfinite(batch).all().item():
                raise RuntimeError(f"Occlusion perturbed input contains NaN/Inf: regions={len(regions)}, samples={sample_start}:{sample_end}.")
            with torch.no_grad(): logits = model(batch)
            if not torch.isfinite(logits).all().item():
                bad = (~torch.isfinite(logits)).nonzero(as_tuple=False); first = bad[0].tolist() if bad.numel() else []
                raise RuntimeError(f"Occlusion perturbed model output contains NaN/Inf: regions={len(regions)}, model_batch={batch.shape[0]}, samples={sample_start}:{sample_end}, first_bad={first}. BLOCK13 will not replace non-finite attribution values.")
            targets = logits[:, target_zero_based].reshape(len(regions), sample_end - sample_start)
            changes = base_target[sample_start:sample_end].unsqueeze(0) - targets
            if not torch.isfinite(changes).all().item():
                raise RuntimeError(f"Occlusion target-logit change contains NaN/Inf: regions={len(regions)}, samples={sample_start}:{sample_end}.")
            for j, (ch, t0, t1) in enumerate(regions):
                value = changes[j].reshape(sample_end - sample_start, 1, 1)
                attr[sample_start:sample_end, ch:ch + 1, t0:t1] += value
                counts[sample_start:sample_end, ch:ch + 1, t0:t1] += 1.0
        perturbed_regions.clear(); regions.clear()
    for ch in channel_starts:
        for t0 in time_starts:
            t1 = min(t0 + seg, WINDOW_SAMPLES); xp = x.clone(); xp[:, ch, t0:t1] = 0.0
            perturbed_regions.append(xp); regions.append((ch, t0, t1))
            if len(perturbed_regions) >= max_regions_per_batch: flush_region_batch()
    flush_region_batch()
    if (counts == 0).any().item():
        raise RuntimeError(f"Occlusion produced {int((counts == 0).sum().item())} uncovered input elements; check segment/stride configuration.")
    if not torch.isfinite(attr).all().item() or not torch.isfinite(counts).all().item():
        raise RuntimeError("Occlusion accumulation contains NaN/Inf before normalization.")
    arr = (attr / counts).detach().cpu().numpy(); assert_finite("occlusion attribution", arr)
    if arr.shape != (n, N_CHANNELS, WINDOW_SAMPLES):
        raise RuntimeError(f"Occlusion attribution shape {arr.shape}; expected ({n},{N_CHANNELS},{WINDOW_SAMPLES})")
    return arr























































# =============================================================================
# 12B. FROZEN BLOCK07 ARCHITECTURES (verbatim copy of BLOCK07 class source)
# =============================================================================
# Copied unchanged from BLOCK07_DEEP_LEARNING_FINAL.py so that the frozen
# BLOCK07 LOSO checkpoints can be rebuilt and loaded with strict=True.
# BLOCK07 is not imported because importing it would execute its own setup.

import torch.nn as nn
import torch.nn.functional as F

MODEL_DROPOUT = {
    "EEGNet": 0.5,
    "ShallowConvNet": 0.5,
    "DeepConvNet": 0.5,
    "CNN-BiLSTM": 0.3,
    "CNN-BiLSTM-Attention": 0.3,
}

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


def model_dir_name(model):
    # identical to BLOCK07 model_dir_name()
    return model.replace("-", "_").replace("+", "ATTN")


class BlockInputAdapter(nn.Module):
    """
    Adapt Captum's canonical XAI tensor N x 13 x 500 to the exact BLOCK07
    forward convention.  The model classes are loaded from the canonical
    BLOCK07 module, so local ``isinstance`` checks are NOT reliable: a class
    with the same name from another Python module is a different type.
    Therefore the adapter is keyed by the canonical BLOCK07 model name.
    """
    TWO_D_MODELS = {"EEGNet", "ShallowConvNet", "DeepConvNet"}

    def __init__(self, net, model_name):
        super().__init__()
        self.net = net
        self.model_name = str(model_name)
        self.is2d = self.model_name in self.TWO_D_MODELS

    def forward(self, x):
        if x.ndim != 3 or x.shape[-2:] != (N_CHANNELS, WINDOW_SAMPLES):
            raise RuntimeError(
                f"{self.model_name}: XAI adapter expected N x {N_CHANNELS} x "
                f"{WINDOW_SAMPLES}, got {tuple(x.shape)}"
            )
        if self.is2d:
            return self.net(x.unsqueeze(1))
        return self.net(x)


def load_block07_loso_model(model_name: str, subject: int, condition: str):
    """
    Load the frozen BLOCK07 cross-subject LOSO checkpoint whose HELD-OUT
    subject is `subject` (same convention as the BLOCK06 ML LOSO models).
    Returns (adapter_model_in_eval_mode, channel_mean, channel_std).
    """
    ckpt_path = (
        BLOCK07_ROOT / "07B_DL_LOSO" / "folds"
        / f"Subject_{subject:02d}" / condition
        / model_dir_name(model_name) / "checkpoints" / "best_model.pt"
    )
    require_file(ckpt_path, f"BLOCK07 LOSO checkpoint {model_name} S{subject:02d} {condition}")

    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if payload.get("model_name") != model_name:
        raise RuntimeError(
            f"{ckpt_path}: checkpoint model_name={payload.get('model_name')!r} != {model_name!r}"
        )

    net = build_model(model_name)
    net.load_state_dict(payload["state_dict"], strict=True)
    net.eval()

    mean = np.asarray(payload["normalizer_mean"], dtype=np.float32)
    std = np.asarray(payload["normalizer_std"], dtype=np.float32)
    if mean.shape != (N_CHANNELS,) or std.shape != (N_CHANNELS,):
        raise RuntimeError(f"{ckpt_path}: normalizer shape {mean.shape}/{std.shape}")
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std)) and np.all(std > 0)):
        raise RuntimeError(f"{ckpt_path}: invalid normalizer values")

    model = BlockInputAdapter(net)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, mean, std



# =============================================================================
# 12C. AUTHORITATIVE BLOCK07 RECONSTRUCTION
# =============================================================================
# Use the same canonical-source/checkpoint logic already used successfully by
# BLOCK09.  BLOCK13 must not depend on an invented XAI handoff JSON and must not
# silently reconstruct a different architecture.

BLOCK07_MAIN_GUARD_REQUIRED = True
_BLOCK07_MODULE_CACHE = None
_BLOCK07_SOURCE_CACHE = None


def _locate_canonical_block07_source() -> Path:
    candidates = [
        WINDOWS_ROOT / "scripts" / "BLOCK07_DEEP_LEARNING_FINAL.py",
        WINDOWS_ROOT / "BLOCK07_DEEP_LEARNING_FINAL.py",
        Path(__file__).resolve().parent / "BLOCK07_DEEP_LEARNING_FINAL.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Canonical BLOCK07_DEEP_LEARNING_FINAL.py was not found. "
        "BLOCK13 will not guess or substitute a DL architecture."
    )


def _canonical_dl_model_name(value: str) -> str:
    s = str(value).strip().lower().replace("_", "-").replace(" ", "")
    mapping = {
        "eegnet": "EEGNet",
        "shallowconvnet": "ShallowConvNet",
        "deepconvnet": "DeepConvNet",
        "cnn-bilstm": "CNN-BiLSTM",
        "cnnbilstm": "CNN-BiLSTM",
        "cnn-bilstm-attention": "CNN-BiLSTM-Attention",
        "cnnbilstm-attention": "CNN-BiLSTM-Attention",
        "cnnbilstmattention": "CNN-BiLSTM-Attention",
    }
    if s not in mapping:
        raise ValueError(f"Unknown BLOCK07 DL model directory/name: {value!r}")
    return mapping[s]


def _load_authoritative_block07_module():
    global _BLOCK07_MODULE_CACHE, _BLOCK07_SOURCE_CACHE
    if _BLOCK07_MODULE_CACHE is not None:
        return _BLOCK07_MODULE_CACHE, _BLOCK07_SOURCE_CACHE

    path = _locate_canonical_block07_source()
    source_text = path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(path))

    has_main_guard = any(
        isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and any(isinstance(op, ast.Eq) for op in n.test.ops)
        and any(
            isinstance(c, ast.Constant) and c.value == "__main__"
            for c in n.test.comparators
        )
        for n in tree.body
    )
    if BLOCK07_MAIN_GUARD_REQUIRED and not has_main_guard:
        raise RuntimeError(
            f"{path} has no safe __main__ guard; BLOCK13 refuses to import it "
            "because importing it could execute BLOCK07 training."
        )

    spec = importlib.util.spec_from_file_location(
        "paper2_block07_canonical_for_block13", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import canonical BLOCK07 source: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_model"):
        raise RuntimeError(
            "Canonical BLOCK07 module has no build_model(); "
            "BLOCK13 will not reconstruct a substitute architecture."
        )

    _BLOCK07_MODULE_CACHE = module
    _BLOCK07_SOURCE_CACHE = path
    log(f"BLOCK07 canonical architecture source: {path}")
    return module, path


def _discover_exact_block07_checkpoint(subject: int, condition: str, model_name: str):
    folds_root = BLOCK07_ROOT / "07B_DL_LOSO" / "folds"
    if not folds_root.exists():
        raise FileNotFoundError(f"BLOCK07 folds directory not found: {folds_root}")

    candidates = []
    for ckpt in folds_root.rglob("best_model.pt"):
        text = "/".join(part.upper() for part in ckpt.parts)
        subject_ok = bool(
            re.search(rf"(?:^|[/_-])S0*{subject}(?:[/_-]|$)", text)
        )
        if not subject_ok:
            subject_ok = (
                f"SUBJECT_{subject:02d}" in text
                or f"SUBJECT-{subject:02d}" in text
            )
        if not subject_ok or condition.upper() not in text:
            continue

        model_ok = False
        for part in ckpt.parts:
            try:
                if _canonical_dl_model_name(part) == model_name:
                    model_ok = True
                    break
            except ValueError:
                pass
        if not model_ok:
            continue

        complete_candidates = list(ckpt.parent.rglob("complete.json"))
        if len(complete_candidates) != 1:
            complete_candidates = list(ckpt.parent.parent.rglob("complete.json"))
        if len(complete_candidates) != 1:
            raise RuntimeError(
                f"Cannot uniquely locate BLOCK07 complete.json for {ckpt}; "
                f"found {len(complete_candidates)}"
            )
        candidates.append((ckpt, complete_candidates[0]))

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one BLOCK07 checkpoint for "
            f"S{subject:02d} {condition} {model_name}; found {len(candidates)}"
        )
    return candidates[0]


def _extract_block07_normalizer(payload: dict, checkpoint: Path):
    mean_keys = (
        "normalizer_mean", "channel_mean", "normalization_mean",
        "train_mean", "x_mean", "scaler_mean", "mean",
    )
    std_keys = (
        "normalizer_std", "channel_std", "normalization_std",
        "train_std", "x_std", "scaler_std", "std",
    )
    mean = next((payload[k] for k in mean_keys if k in payload), None)
    std = next((payload[k] for k in std_keys if k in payload), None)
    if mean is None or std is None:
        raise RuntimeError(
            f"BLOCK07 training normalizer missing from {checkpoint}. "
            f"Available checkpoint keys: {sorted(map(str, payload.keys()))}"
        )

    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean.shape != (N_CHANNELS,) or std.shape != (N_CHANNELS,):
        raise RuntimeError(
            f"Invalid BLOCK07 normalizer shape at {checkpoint}: "
            f"mean={mean.shape}, std={std.shape}; expected {(N_CHANNELS,)}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise RuntimeError(f"Invalid BLOCK07 normalizer values at {checkpoint}")
    return mean, std


def load_block07_loso_model(model_name: str, subject: int, condition: str):
    """Load the exact frozen BLOCK07 LOSO model for the held-out subject."""
    if torch is None:
        raise RuntimeError(f"PyTorch unavailable: {TORCH_ERROR}")

    block07_module, source_path = _load_authoritative_block07_module()
    checkpoint, complete = _discover_exact_block07_checkpoint(
        subject, condition, model_name
    )

    # Verify the BLOCK07 completion marker when it records a checkpoint hash.
    complete_obj = load_json(complete)
    expected_hash = complete_obj.get("checkpoint_sha256")
    if expected_hash and sha256_file(checkpoint) != expected_hash:
        raise RuntimeError(f"BLOCK07 checkpoint hash mismatch: {checkpoint}")

    try:
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(str(checkpoint), map_location="cpu")

    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError(f"Invalid BLOCK07 checkpoint payload: {checkpoint}")

    stored_name = payload.get("model_name")
    if stored_name is not None and str(stored_name) != model_name:
        raise RuntimeError(
            f"BLOCK07 checkpoint model_name={stored_name!r} does not match "
            f"requested {model_name!r}: {checkpoint}"
        )

    model = block07_module.build_model(model_name)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    mean, std = _extract_block07_normalizer(payload, checkpoint)

    adapter = BlockInputAdapter(model, model_name)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)

    log(
        f"DL checkpoint verified: S{subject:02d} {condition} {model_name} "
        f"checkpoint={checkpoint.name} source={source_path.name}"
    )
    return adapter, mean, std


def _assert_dl_runtime_integrity(model, mean, std, raw_by_gesture: dict, model_name: str):
    """
    Fail loudly (never silently) if the torch CPU runtime is not trustworthy:
      1. exactly 1 intra-op thread,
      2. two identical forward passes are bit-identical and finite,
      3. batched output equals row-by-row output (batch invariance).
    """
    if torch.get_num_threads() != 1:
        raise RuntimeError(
            f"DL XAI requires torch.get_num_threads()==1, got {torch.get_num_threads()}."
        )
    g = GESTURES[0]
    xn = (raw_by_gesture[g] - mean[None, :, None]) / std[None, :, None]
    x = torch.from_numpy(xn.astype(np.float32))
    xb = x.repeat(5, 1, 1)                       # 45 rows, like the occlusion batches
    with torch.no_grad():
        a = model(xb)
        b = model(xb)
        rowwise = torch.cat([model(x[i:i + 1]) for i in range(x.shape[0])])
    if not bool(torch.isfinite(a).all()):
        raise RuntimeError(f"{model_name}: runtime integrity check: non-finite logits.")
    if not torch.equal(a, b):
        raise RuntimeError(f"{model_name}: runtime integrity check: identical forward passes differ.")
    if not torch.allclose(a[:x.shape[0]], rowwise, atol=1e-4, rtol=1e-4):
        raise RuntimeError(
            f"{model_name}: runtime integrity check: batched output != row-by-row output "
            f"(max diff {float((a[:x.shape[0]] - rowwise).abs().max()):.3g})."
        )


def _explain_dl_combo(model, mean, std, raw_by_gesture: dict, model_name: str):
    """
    raw_by_gesture[g] = float32 array (9, 13, 500) of RAW BLOCK04 windows.
    Returns gesture-level means plus true per-window attribution arrays.
    The per-window arrays are retained specifically for the raw attribution archive;
    manuscript-facing tables continue to use the gesture-level means.
    """
    out = {k: [] for k in ("ig_abs", "ig_signed", "occ_abs", "occ_signed", "delta")}
    raw_ig_abs = []
    raw_ig_signed = []
    raw_occ_abs = []
    raw_occ_signed = []
    raw_delta = []
    raw_prediction = []
    raw_target = []
    correct = 0
    total = 0
    for g in GESTURES:
        t_g = time.time()
        log(f"DL XAI: {model_name} gesture {g}/{N_GESTURES} IG + occlusion START")
        raw = raw_by_gesture[g]
        xn = (raw - mean[None, :, None]) / std[None, :, None]   # BLOCK07 training-only z-score
        assert_finite("normalized DL input", xn)
        x = torch.from_numpy(xn.astype(np.float32))
        target = g - 1

        with torch.no_grad():
            logits = model(x)
        if tuple(logits.shape) != (x.shape[0], N_GESTURES):
            raise RuntimeError(f"{model_name}: logits shape {tuple(logits.shape)}")
        predictions = logits.argmax(1).detach().cpu().numpy().astype(np.int8) + 1
        correct += int((logits.argmax(1) == target).sum().item())
        total += int(x.shape[0])

        ig = IntegratedGradients(model)
        attr, delta = ig.attribute(
            x,
            baselines=torch.zeros_like(x),
            target=target,
            n_steps=IG_STEPS,
            internal_batch_size=IG_INTERNAL_BATCH_SIZE,
            return_convergence_delta=True,
        )
        attr = attr.detach().cpu().numpy()          # (9,13,500)
        delta = delta.detach().cpu().numpy().reshape(-1)

        occ = run_occlusion(model, x, target)

        for name, arr in (("IG", attr), ("occlusion", occ), ("IG delta", delta)):
            assert_finite(f"{model_name} {name}", arr)
        if attr.shape[1:] != (N_CHANNELS, WINDOW_SAMPLES) or occ.shape[1:] != (N_CHANNELS, WINDOW_SAMPLES):
            raise RuntimeError(f"{model_name}: bad attribution shape {attr.shape} / {occ.shape}")

        abs_ig = np.abs(attr).astype(np.float32)
        signed_ig = np.asarray(attr, dtype=np.float32)
        abs_occ = np.abs(occ).astype(np.float32)
        signed_occ = np.asarray(occ, dtype=np.float32)
        delta_abs = np.abs(delta).astype(np.float32)

        raw_ig_abs.append(abs_ig)
        raw_ig_signed.append(signed_ig)
        raw_occ_abs.append(abs_occ)
        raw_occ_signed.append(signed_occ)
        raw_delta.append(delta_abs)
        raw_prediction.append(predictions)
        raw_target.append(np.full(x.shape[0], g, dtype=np.int8))

        out["ig_abs"].append(abs_ig.mean(axis=0))
        out["ig_signed"].append(signed_ig.mean(axis=0))
        out["occ_abs"].append(abs_occ.mean(axis=0))
        out["occ_signed"].append(signed_occ.mean(axis=0))
        out["delta"].append(float(np.mean(delta_abs)))

        # Explicit per-gesture release of every large temporary (tensors, graph
        # holders, numpy attribution arrays).  The retained copies live in the
        # raw_* lists; everything else is dropped immediately.
        del ig, attr, delta, occ, logits, x, xn, abs_ig, signed_ig, abs_occ, signed_occ, delta_abs, predictions
        gc.collect()
        log(
            f"DL XAI: {model_name} gesture {g}/{N_GESTURES} IG + occlusion DONE "
            f"({time.time() - t_g:.0f}s)"
        )

    res = {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}   # (7,13,500) / (7,)
    # Gesture-major ordering: g1's 9 windows, then g2's 9 windows, ... g7's 9 windows.
    res["ig_abs_raw"] = np.concatenate(raw_ig_abs, axis=0)
    res["ig_signed_raw"] = np.concatenate(raw_ig_signed, axis=0)
    res["occ_abs_raw"] = np.concatenate(raw_occ_abs, axis=0)
    res["occ_signed_raw"] = np.concatenate(raw_occ_signed, axis=0)
    res["delta_raw"] = np.concatenate(raw_delta, axis=0)
    res["prediction_raw"] = np.concatenate(raw_prediction, axis=0)
    res["target_raw"] = np.concatenate(raw_target, axis=0)
    res["accuracy_subset"] = np.float32(correct / max(total, 1))
    del raw_ig_abs, raw_ig_signed, raw_occ_abs, raw_occ_signed, raw_delta, raw_prediction, raw_target, out
    gc.collect()
    return res


def run_dl_xai(feature_lookup: pd.DataFrame, block04_df: pd.DataFrame):
    """
    DL XAI with the frozen BLOCK07 cross-subject LOSO checkpoints
    (held-out subject = explained subject), Captum IG + Occlusion.

    Output tables are aggregated to one row per
    subject/condition/model/gesture/channel/time_index, averaged over that
    gesture's 9 explanation windows (3 sets x window indices 0,10,20).
    (Per-window 13x500 tables would be ~100 million rows.)
    """
    if torch is None:
        raise RuntimeError(f"PyTorch unavailable: {TORCH_ERROR}")
    if IntegratedGradients is None or Occlusion is None:
        raise RuntimeError(f"Captum unavailable: {CAPTUM_ERROR}")

    torch.manual_seed(42)
    # MEASURED on this machine (BLOCK13_NAN_DIAG2): multi-threaded CPU inference
    # (2, 4, 8, 32 threads) returns NaN/Inf or values that differ from the
    # 1-thread result by up to ~1e37, even with mkldnn disabled.  1 thread is
    # exact and repeatable, so the DL XAI phase is forced to 1 thread.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass   # already initialised; intra-op = 1 is what matters

    cache_dir = BLOCK13_ROOT / "cache_dl_xai"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache schema V3 is deliberately tied to the validated single-thread CPU
    # runtime.  Older caches are NOT trusted merely because the filename exists:
    # they may have been produced before the PyTorch convolution/pooling runtime
    # fix and could therefore contain scientifically invalid attributions.
    CACHE_SCHEMA = "DL_XAI_CACHE_V4_COMPLETE_AUDIT"
    CACHE_TORCH_THREADS = 1

    def cache_file(model_name, subject, condition):
        return cache_dir / f"{model_dir_name(model_name)}_S{subject:02d}_{condition}.npz"

    def cache_is_valid(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with np.load(path, allow_pickle=False) as z:
                required = {
                    "cache_schema", "protocol_version", "torch_num_threads",
                    "ig_abs", "ig_signed", "occ_abs", "occ_signed",
                    "delta", "accuracy_subset",
                }
                if not required.issubset(set(z.files)):
                    return False
                schema = str(np.asarray(z["cache_schema"]).reshape(-1)[0])
                protocol = str(np.asarray(z["protocol_version"]).reshape(-1)[0])
                threads = int(np.asarray(z["torch_num_threads"]).reshape(-1)[0])
                if schema != CACHE_SCHEMA or protocol != PROTOCOL_VERSION or threads != CACHE_TORCH_THREADS:
                    return False
                expected_map = {
                    "ig_abs": (N_GESTURES, N_CHANNELS, WINDOW_SAMPLES),
                    "ig_signed": (N_GESTURES, N_CHANNELS, WINDOW_SAMPLES),
                    "occ_abs": (N_GESTURES, N_CHANNELS, WINDOW_SAMPLES),
                    "occ_signed": (N_GESTURES, N_CHANNELS, WINDOW_SAMPLES),
                    "delta": (N_GESTURES,),
                }
                for key, shape in expected_map.items():
                    arr = np.asarray(z[key])
                    if arr.shape != shape or not np.isfinite(arr).all():
                        return False
                acc = float(np.asarray(z["accuracy_subset"]).reshape(-1)[0])
                if not np.isfinite(acc) or not (0.0 <= acc <= 1.0):
                    return False
                return True
        except Exception:
            return False

    n_expected = N_GESTURES * len(SETS) * len(EXPLANATION_WINDOW_INDICES)
    expected_keys = {
        (g, s, w) for g in GESTURES for s in SETS for w in EXPLANATION_WINDOW_INDICES
    }

    for subject in SUBJECTS:
        for condition in CONDITIONS:
            todo = []
            for m in DL_MODELS:
                cp = cache_file(m, subject, condition)
                rp = raw_dl_path(subject, condition, m)
                if not cache_is_valid(cp) or not raw_dl_archive_is_valid(rp, subject, condition, m):
                    # A valid aggregate cache is not sufficient for a complete
                    # BLOCK13 run: the raw per-window archive must also exist.
                    if cp.exists() and cache_is_valid(cp) and not raw_dl_archive_is_valid(rp, subject, condition, m):
                        _, _raw_reason = validate_raw_dl_archive(rp, subject, condition, m)
                        log(f"DL XAI: raw archive invalid -> recomputing {cp.name}: {_raw_reason}")
                    elif cp.exists():
                        log(f"DL XAI: invalid/stale cache -> recomputing {cp.name}")
                    todo.append(m)
            if not todo:
                log(
                    f"DL XAI: S{subject:02d} {condition}: all {len(DL_MODELS)} models "
                    f"cache-hit (aggregate cache + raw archive valid)"
                )
                continue

            log(f"DL XAI: S{subject:02d} {condition}: recomputing models {todo}")
            log_memory(f"DL XAI S{subject:02d} {condition} before raw-window load")
            rows = block04_df[
                (block04_df["subject"] == subject)
                & (block04_df["condition"] == condition)
                & (block04_df["gesture_code"].isin(GESTURES))
                & (block04_df["window_index"].isin(EXPLANATION_WINDOW_INDICES))
            ].copy()
            if len(rows) != n_expected:
                raise RuntimeError(
                    f"DL explanation subset incomplete: S{subject:02d} {condition}: "
                    f"{len(rows)} vs {n_expected}"
                )
            observed = {
                (int(r.gesture_code), str(r.set), int(r.window_index))
                for r in rows.itertuples(index=False)
            }
            if observed != expected_keys:
                raise RuntimeError(
                    f"DL explanation subset key mismatch: S{subject:02d} {condition}"
                )

            rows = rows.sort_values(["gesture_code", "set", "window_index"])
            raw_by_gesture = {}
            for g in GESTURES:
                sub = rows[rows["gesture_code"] == g]
                raw_by_gesture[g] = np.stack(
                    [load_raw_window_from_row(r) for r in sub.itertuples(index=False)],
                    axis=0,
                ).astype(np.float32)
                if raw_by_gesture[g].shape != (9, N_CHANNELS, WINDOW_SAMPLES):
                    raise RuntimeError(f"S{subject:02d} {condition} g{g}: {raw_by_gesture[g].shape}")

            for model_name in todo:
                t0 = time.time()
                log(f"DL XAI: S{subject:02d} {condition} {model_name} loading frozen checkpoint")
                model, mean, std = load_block07_loso_model(model_name, subject, condition)
                log(f"DL XAI: S{subject:02d} {condition} {model_name} runtime-integrity check")
                _assert_dl_runtime_integrity(model, mean, std, raw_by_gesture, model_name)
                log(f"DL XAI: S{subject:02d} {condition} {model_name} explaining (IG + occlusion)")
                res = _explain_dl_combo(model, mean, std, raw_by_gesture, model_name)
                log_memory(f"DL XAI S{subject:02d} {condition} {model_name} after explain")

                # The raw archive must be written before the aggregate cache is
                # promoted.  If this fails, BLOCK13 fails instead of leaving a
                # misleadingly complete aggregate-only result.
                raw_rows = rows.copy()
                raw_path = raw_dl_path(subject, condition, model_name)
                write_raw_dl_archive(
                    raw_path,
                    res,
                    raw_rows,
                    subject,
                    condition,
                    model_name,
                )

                # Validate immediately after promotion. This prevents a late final
                # audit from discovering a malformed/corrupted raw archive hours
                # after the corresponding XAI calculation completed.
                raw_ok, raw_reason = validate_raw_dl_archive(
                    raw_path, subject, condition, model_name
                )
                if not raw_ok:
                    raise RuntimeError(
                        f"DL raw attribution immediate post-write audit failed: {raw_path}. "
                        f"Reason: {raw_reason}"
                    )

                tmp = cache_file(model_name, subject, condition).with_suffix(".tmp.npz")
                # Persist only finite, shape-validated aggregate attribution arrays
                # in the compact cache. Per-window arrays live in raw_attributions/DL.
                for _name in ("ig_abs", "ig_signed", "occ_abs", "occ_signed", "delta"):
                    _arr = np.asarray(res[_name])
                    if not np.isfinite(_arr).all():
                        raise RuntimeError(f"{model_name}: refusing to cache non-finite {_name}")
                np.savez_compressed(
                    tmp,
                    ig_abs=res["ig_abs"],
                    ig_signed=res["ig_signed"],
                    occ_abs=res["occ_abs"],
                    occ_signed=res["occ_signed"],
                    delta=res["delta"],
                    accuracy_subset=res["accuracy_subset"],
                    cache_schema=np.asarray(CACHE_SCHEMA),
                    protocol_version=np.asarray(PROTOCOL_VERSION),
                    torch_num_threads=np.asarray(CACHE_TORCH_THREADS, dtype=np.int8),
                )
                os.replace(tmp, cache_file(model_name, subject, condition))

                log(
                    f"DL XAI PASS: S{subject:02d} {condition} {model_name} "
                    f"subset_acc={float(res['accuracy_subset']):.3f} "
                    f"mean|IG delta|={float(np.mean(res['delta'])):.4f} "
                    f"({time.time() - t0:.0f}s)"
                )
                # Explicit per-model release.
                del model, mean, std, res, raw_rows, _arr
                gc.collect()
                log_memory(f"DL XAI S{subject:02d} {condition} {model_name} after cleanup")

            # Explicit per-subject/condition release.
            del raw_by_gesture, rows, observed
            gc.collect()
            log_memory(f"DL XAI S{subject:02d} {condition} after cleanup")

    # Final cache/raw audit: every subject/condition/model combination must have
    # both a current aggregate cache and a true per-window raw attribution archive.
    raw_dl_expected = len(DL_MODELS) * len(SUBJECTS) * len(CONDITIONS)
    raw_dl_found = 0
    raw_dl_failures = []
    cache_failures = []
    log(f"DL XAI: final cache/raw audit START ({raw_dl_expected} combinations)")
    audit_i = 0
    for model_name in DL_MODELS:
        for subject in SUBJECTS:
            for condition in CONDITIONS:
                audit_i += 1
                if audit_i % 25 == 0 or audit_i == raw_dl_expected:
                    log(f"DL XAI: final audit {audit_i}/{raw_dl_expected}")
                    gc.collect()
                cp = cache_file(model_name, subject, condition)
                rp = raw_dl_path(subject, condition, model_name)
                if not cache_is_valid(cp):
                    cache_failures.append(str(cp))
                raw_ok, raw_reason = validate_raw_dl_archive(
                    rp, subject, condition, model_name
                )
                if not raw_ok:
                    raw_dl_failures.append({
                        "path": str(rp),
                        "subject": subject,
                        "condition": condition,
                        "model": model_name,
                        "reason": raw_reason,
                    })
                else:
                    raw_dl_found += 1

    if cache_failures or raw_dl_failures:
        details = []
        if cache_failures:
            details.append(
                "DL aggregate cache failures:\n  " + "\n  ".join(cache_failures[:20])
            )
        if raw_dl_failures:
            details.append(
                "DL raw archive failures:\n  " + "\n  ".join(
                    f"{x['path']} -> {x['reason']}" for x in raw_dl_failures[:20]
                )
            )
        raise RuntimeError(
            "DL XAI final audit failed. "
            f"Raw archives valid={raw_dl_found}/{raw_dl_expected}.\n"
            + "\n".join(details)
        )

    log(f"DL XAI: final cache/raw audit PASS raw_valid={raw_dl_found}/{raw_dl_expected}")
    write_raw_archive_readme()

    # ---- assemble compact manuscript tables from the caches (STREAMING) ----
    # ROOT CAUSE OF THE 0xC0000005 CRASH (previous implementation):
    #   * it appended ~1.75 MILLION Python dict records (875,000 temporal rows x 2
    #     tables, 9 fields each, ~1-2 GB of small-object heap) while the
    #     5.97M-row ML SHAP DataFrame and the torch/xgboost/shap native runtimes
    #     were still resident, then
    #   * converted those dict lists into DataFrames (pandas object-array
    #     inference + column copies, several additional GB) and
    #   * validated them with .to_numpy(dtype=float64) copies.
    #   Exhausting the process heap inside pandas/numpy native allocation
    #   surfaces on Windows as an access violation instead of MemoryError.
    # NEW DESIGN: no per-row Python objects at all.
    #   * channel summaries (22,750 rows) -> preallocated numpy arrays, DataFrame
    #     built column-wise from index arrays once at the end;
    #   * temporal summaries (875,000 rows per table) -> built and written one
    #     3,500-row chunk (one model/subject/condition cache file) at a time
    #     directly to a temp CSV, then atomically promoted; never held in RAM;
    #   * only one cache NPZ (4 x 7x13x500 float32 = 0.7 MB) is open at a time.
    # The raw 13x500 per-window maps remain untouched in raw_attributions/DL.
    log("DL XAI: assembling compact IG / occlusion summaries from cache")
    log_memory("DL assemble START")

    n_models = len(DL_MODELS)
    n_subjects = len(SUBJECTS)
    n_conditions = len(CONDITIONS)
    n_gestures = len(GESTURES)
    combo_shape = (n_models, n_subjects, n_conditions, n_gestures, N_CHANNELS)
    n_combos = n_models * n_subjects * n_conditions

    ch_ig_abs = np.full(combo_shape, np.nan, dtype=np.float64)
    ch_ig_signed = np.full(combo_shape, np.nan, dtype=np.float64)
    ch_occ_abs = np.full(combo_shape, np.nan, dtype=np.float64)
    ch_occ_signed = np.full(combo_shape, np.nan, dtype=np.float64)
    ch_delta = np.full(combo_shape[:4], np.nan, dtype=np.float64)
    combo_filled = np.zeros((n_models, n_subjects, n_conditions), dtype=bool)

    ig_temporal_path = TABLE_ROOT / "DL_IG_temporal_summary.csv"
    occ_temporal_path = TABLE_ROOT / "DL_occlusion_temporal_summary.csv"
    ig_temporal_tmp = ig_temporal_path.with_suffix(ig_temporal_path.suffix + ".tmp")
    occ_temporal_tmp = occ_temporal_path.with_suffix(occ_temporal_path.suffix + ".tmp")
    for _p in (ig_temporal_tmp, occ_temporal_tmp):
        if _p.exists():
            _p.unlink()

    rows_per_chunk = n_gestures * WINDOW_SAMPLES                       # 3,500
    expected_temporal_rows = n_combos * rows_per_chunk                 # 875,000
    expected_channel_rows = n_combos * n_gestures * N_CHANNELS         # 22,750

    # Constant per-chunk columns (identical row order to the former record loop:
    # gesture-major, time_index-minor).
    chunk_gesture = np.repeat(np.asarray(GESTURES, dtype=np.int64), WINDOW_SAMPLES)
    chunk_time_index = np.tile(np.arange(WINDOW_SAMPLES, dtype=np.int64), n_gestures)
    chunk_time_seconds = chunk_time_index.astype(np.float64) / float(FS)

    temporal_rows_written = 0
    combo_i = 0
    t_assemble = time.time()

    with open(ig_temporal_tmp, "w", encoding="utf-8", newline="") as fh_ig, \
            open(occ_temporal_tmp, "w", encoding="utf-8", newline="") as fh_occ:
        for mi, model_name in enumerate(DL_MODELS):
            for si, subject in enumerate(SUBJECTS):
                for ci_cond, condition in enumerate(CONDITIONS):
                    combo_i += 1
                    cp = cache_file(model_name, subject, condition)
                    if combo_i == 1 or combo_i % 10 == 0 or combo_i == n_combos:
                        log(
                            f"DL assemble: cache {combo_i}/{n_combos} "
                            f"{model_name} S{subject:02d} {condition}"
                        )

                    with np.load(cp, allow_pickle=False) as z:
                        ig_abs = np.asarray(z["ig_abs"], dtype=np.float32)
                        ig_signed = np.asarray(z["ig_signed"], dtype=np.float32)
                        occ_abs = np.asarray(z["occ_abs"], dtype=np.float32)
                        occ_signed = np.asarray(z["occ_signed"], dtype=np.float32)
                        delta = np.asarray(z["delta"], dtype=np.float32)

                    expected_map_shape = (N_GESTURES, N_CHANNELS, WINDOW_SAMPLES)
                    for _nm, _arr in (
                        ("ig_abs", ig_abs), ("ig_signed", ig_signed),
                        ("occ_abs", occ_abs), ("occ_signed", occ_signed),
                    ):
                        if _arr.shape != expected_map_shape:
                            raise RuntimeError(
                                f"Invalid {_nm} cache shape for S{subject:02d} {condition} "
                                f"{model_name}: {_arr.shape}; expected {expected_map_shape}"
                            )
                        if not np.isfinite(_arr).all():
                            raise RuntimeError(
                                f"Non-finite DL attribution cache ({_nm}): "
                                f"S{subject:02d} {condition} {model_name}"
                            )
                    if delta.shape != (N_GESTURES,) or not np.isfinite(delta).all():
                        raise RuntimeError(
                            f"Invalid/non-finite IG delta cache: "
                            f"S{subject:02d} {condition} {model_name}: {delta.shape}"
                        )

                    t_ig_abs = np.empty((n_gestures, WINDOW_SAMPLES), dtype=np.float64)
                    t_ig_signed = np.empty_like(t_ig_abs)
                    t_occ_abs = np.empty_like(t_ig_abs)
                    t_occ_signed = np.empty_like(t_ig_abs)

                    for gi, gesture in enumerate(GESTURES):
                        ig_map = ig_abs[gi]
                        ig_signed_map = ig_signed[gi]
                        occ_map = occ_abs[gi]
                        occ_signed_map = occ_signed[gi]

                        # Channel summaries: 13 values per gesture (mean over time).
                        ch_ig_abs[mi, si, ci_cond, gi, :] = np.mean(ig_map, axis=1)
                        ch_ig_signed[mi, si, ci_cond, gi, :] = np.mean(ig_signed_map, axis=1)
                        ch_occ_abs[mi, si, ci_cond, gi, :] = np.mean(occ_map, axis=1)
                        ch_occ_signed[mi, si, ci_cond, gi, :] = np.mean(occ_signed_map, axis=1)
                        ch_delta[mi, si, ci_cond, gi] = float(delta[gi])

                        # Temporal summaries: 500 values per gesture (mean over channels).
                        t_ig_abs[gi, :] = np.mean(ig_map, axis=0)
                        t_ig_signed[gi, :] = np.mean(ig_signed_map, axis=0)
                        t_occ_abs[gi, :] = np.mean(occ_map, axis=0)
                        t_occ_signed[gi, :] = np.mean(occ_signed_map, axis=0)

                    for _nm, _arr in (
                        ("temporal IG abs", t_ig_abs), ("temporal IG signed", t_ig_signed),
                        ("temporal occ abs", t_occ_abs), ("temporal occ signed", t_occ_signed),
                    ):
                        if not np.isfinite(_arr).all():
                            raise RuntimeError(
                                f"Non-finite {_nm} summary: "
                                f"S{subject:02d} {condition} {model_name}"
                            )

                    delta_col = np.repeat(delta.astype(np.float64), WINDOW_SAMPLES)

                    # One 3,500-row chunk per table; written immediately, then freed.
                    ig_chunk = pd.DataFrame({
                        "subject_id": np.full(rows_per_chunk, subject, dtype=np.int64),
                        "condition": np.full(rows_per_chunk, condition, dtype=object),
                        "model": np.full(rows_per_chunk, model_name, dtype=object),
                        "gesture": chunk_gesture,
                        "time_index": chunk_time_index,
                        "time_seconds": chunk_time_seconds,
                        "mean_abs_IG": t_ig_abs.reshape(-1),
                        "mean_signed_IG": t_ig_signed.reshape(-1),
                        "ig_convergence_delta": delta_col,
                    })
                    ig_chunk.to_csv(fh_ig, index=False, header=(combo_i == 1))
                    occ_chunk = pd.DataFrame({
                        "subject_id": np.full(rows_per_chunk, subject, dtype=np.int64),
                        "condition": np.full(rows_per_chunk, condition, dtype=object),
                        "model": np.full(rows_per_chunk, model_name, dtype=object),
                        "gesture": chunk_gesture,
                        "time_index": chunk_time_index,
                        "time_seconds": chunk_time_seconds,
                        "mean_abs_occlusion": t_occ_abs.reshape(-1),
                        "prediction_change": t_occ_signed.reshape(-1),
                        "absolute_prediction_change": t_occ_abs.reshape(-1),
                    })
                    occ_chunk.to_csv(fh_occ, index=False, header=(combo_i == 1))
                    temporal_rows_written += len(ig_chunk)
                    if len(ig_chunk) != rows_per_chunk or len(occ_chunk) != rows_per_chunk:
                        raise RuntimeError(
                            f"Temporal chunk cardinality mismatch: S{subject:02d} {condition} "
                            f"{model_name}: IG={len(ig_chunk)} OCC={len(occ_chunk)} "
                            f"expected={rows_per_chunk}"
                        )
                    combo_filled[mi, si, ci_cond] = True

                    # Explicit per-cache release.
                    del ig_abs, ig_signed, occ_abs, occ_signed, delta, delta_col
                    del t_ig_abs, t_ig_signed, t_occ_abs, t_occ_signed
                    del ig_chunk, occ_chunk
                    if combo_i % 10 == 0:
                        gc.collect()
                    if combo_i % 50 == 0 or combo_i == n_combos:
                        log_memory(f"DL assemble after cache {combo_i}/{n_combos}")
            fh_ig.flush()
            fh_occ.flush()
            log(f"DL assemble: model {model_name} complete ({combo_i}/{n_combos} caches)")

    if not combo_filled.all():
        raise RuntimeError(
            f"DL assembly incomplete: {int(combo_filled.sum())}/{n_combos} caches processed."
        )
    if temporal_rows_written != expected_temporal_rows:
        raise RuntimeError(
            f"DL temporal-table cardinality mismatch: written={temporal_rows_written}, "
            f"expected={expected_temporal_rows}"
        )
    for _nm, _arr in (
        ("DL IG channel abs", ch_ig_abs), ("DL IG channel signed", ch_ig_signed),
        ("DL occlusion channel abs", ch_occ_abs),
        ("DL occlusion channel signed", ch_occ_signed),
        ("DL IG delta", ch_delta),
    ):
        if not np.isfinite(_arr).all():
            raise RuntimeError(f"{_nm} summary contains NaN/Inf.")

    # Validate the temp CSVs by streaming (never fully loaded), then promote.
    log("DL assemble: validating temporal CSVs by streaming")
    for _label, _tmp, _numeric_cols in (
        ("DL IG temporal", ig_temporal_tmp,
         ["subject_id", "gesture", "time_index", "time_seconds",
          "mean_abs_IG", "mean_signed_IG", "ig_convergence_delta"]),
        ("DL occlusion temporal", occ_temporal_tmp,
         ["subject_id", "gesture", "time_index", "time_seconds",
          "mean_abs_occlusion", "prediction_change", "absolute_prediction_change"]),
    ):
        n_seen = 0
        for _chunk in pd.read_csv(_tmp, chunksize=250_000):
            if not np.isfinite(_chunk[_numeric_cols].to_numpy(dtype=np.float64)).all():
                raise RuntimeError(f"{_label} CSV contains NaN/Inf.")
            n_seen += len(_chunk)
            del _chunk
        gc.collect()
        if n_seen != expected_temporal_rows:
            raise RuntimeError(
                f"{_label} CSV cardinality mismatch: {n_seen} vs {expected_temporal_rows}"
            )
        log(f"DL assemble: {_label} CSV validated rows={n_seen}")

    os.replace(ig_temporal_tmp, ig_temporal_path)
    os.replace(occ_temporal_tmp, occ_temporal_path)
    log(
        f"DL assemble: temporal CSVs written ({time.time() - t_assemble:.0f}s) -> "
        f"{ig_temporal_path.name}, {occ_temporal_path.name}"
    )
    log_memory("DL assemble after temporal CSVs")

    # Channel tables: build column-wise from index arrays (22,750 rows).
    log("DL assemble: building compact channel tables (column-wise)")
    ix_model, ix_subject, ix_cond, ix_gesture, ix_channel = [
        a.reshape(-1) for a in np.indices(combo_shape)
    ]
    model_col = np.asarray(DL_MODELS, dtype=object)[ix_model]
    subject_col = np.asarray(SUBJECTS, dtype=np.int64)[ix_subject]
    condition_col = np.asarray(CONDITIONS, dtype=object)[ix_cond]
    gesture_col = np.asarray(GESTURES, dtype=np.int64)[ix_gesture]
    channel_col = np.asarray(CHANNEL_NAMES, dtype=object)[ix_channel]
    delta_col_all = ch_delta[ix_model, ix_subject, ix_cond, ix_gesture]

    ig_df = pd.DataFrame({
        "subject_id": subject_col,
        "condition": condition_col,
        "model": model_col,
        "gesture": gesture_col,
        "channel": channel_col,
        "mean_abs_IG": ch_ig_abs.reshape(-1),
        "mean_signed_IG": ch_ig_signed.reshape(-1),
        "ig_convergence_delta": delta_col_all,
    })
    occ_df = pd.DataFrame({
        "subject_id": subject_col,
        "condition": condition_col,
        "model": model_col,
        "gesture": gesture_col,
        "channel": channel_col,
        "mean_abs_occlusion": ch_occ_abs.reshape(-1),
        "prediction_change": ch_occ_signed.reshape(-1),
        "absolute_prediction_change": ch_occ_abs.reshape(-1),
    })
    del ix_model, ix_subject, ix_cond, ix_gesture, ix_channel
    del model_col, subject_col, condition_col, gesture_col, channel_col, delta_col_all
    del ch_ig_abs, ch_ig_signed, ch_occ_abs, ch_occ_signed, ch_delta, combo_filled
    gc.collect()

    if len(ig_df) != expected_channel_rows or len(occ_df) != expected_channel_rows:
        raise RuntimeError(
            f"DL channel-table cardinality mismatch: "
            f"IG={len(ig_df)}, OCC={len(occ_df)}, expected={expected_channel_rows}"
        )
    for name, df in (("DL IG channel", ig_df), ("DL occlusion channel", occ_df)):
        numeric = df.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"{name} table contains NaN/Inf.")
    log(
        f"DL XAI: compact summaries complete: channel_rows={len(ig_df)} "
        f"temporal_rows_per_table={temporal_rows_written}"
    )
    log_memory("DL assemble END")

    # Keep the returned DataFrames compact: 22,750 rows each, not 11.375M.
    # The true per-window maps remain in raw_attributions/DL.
    return ig_df, occ_df



# =============================================================================
# 14. ATTRIBUTION SIMILARITY
# =============================================================================

def safe_spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    if np.all(a[mask] == a[mask][0]) or np.all(b[mask] == b[mask][0]):
        return np.nan
    return float(spearmanr(a[mask], b[mask]).statistic)


def bootstrap_ci(values, seed=BOOTSTRAP_SEED):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])

    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for i in range(BOOTSTRAP_RESAMPLES):
        sample = rng.choice(x, size=len(x), replace=True)
        means[i] = np.mean(sample)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_ml_similarity(channel_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model in ML_MODELS:
        for subject in SUBJECTS:
            for gesture in GESTURES:
                me = channel_df[
                    (channel_df.subject_id == subject)
                    & (channel_df.model == model)
                    & (channel_df.gesture == gesture)
                    & (channel_df.condition == "ME")
                ].sort_values("channel")

                mi = channel_df[
                    (channel_df.subject_id == subject)
                    & (channel_df.model == model)
                    & (channel_df.gesture == gesture)
                    & (channel_df.condition == "MI")
                ].sort_values("channel")

                if len(me) != N_CHANNELS or len(mi) != N_CHANNELS:
                    sim = np.nan
                else:
                    sim = safe_spearman(
                        me["normalized_importance"].to_numpy(),
                        mi["normalized_importance"].to_numpy(),
                    )

                rows.append({
                    "subject_id": subject,
                    "model": model,
                    "gesture": gesture,
                    "channel_similarity": sim,
                    "temporal_similarity": np.nan,
                    "full_map_similarity": np.nan,
                    "similarity_method": "Spearman_channel_normalized_SHAP",
                })

    return pd.DataFrame(rows)


def compute_dl_similarity(
    ig_df: pd.DataFrame,
    occ_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute subject×model×gesture ME–MI attribution similarity.

    Uses compact channel summaries plus the canonical temporal summary written
    by run_dl_xai(). Full-map similarity is computed directly from the compact
    cache arrays, never from a giant pandas channel×time table.
    """
    rows = []

    temporal_path = TABLE_ROOT / "DL_IG_temporal_summary.csv"
    if not temporal_path.exists():
        raise RuntimeError(f"Missing temporal IG summary: {temporal_path}")
    log("PHASE 7: loading DL IG temporal summary (needed columns only)")
    temporal = pd.read_csv(
        temporal_path,
        usecols=["subject_id", "condition", "model", "gesture", "time_index", "mean_abs_IG"],
    )
    # One pass groups the 875,000-row table by (subject, model, gesture, condition)
    # so the 875 x 2 lookups below are dictionary hits instead of repeated
    # full-table boolean masks.  Identical selected rows / values.
    temporal_groups = {
        k: g for k, g in temporal.groupby(
            ["subject_id", "model", "gesture", "condition"], sort=False
        )
    }
    del temporal
    gc.collect()
    _empty_t = pd.DataFrame({"time_index": [], "mean_abs_IG": []})

    for model in DL_MODELS:
        for subject in SUBJECTS:
            for gesture in GESTURES:
                me_ig = ig_df[
                    (ig_df["subject_id"] == subject)
                    & (ig_df["model"] == model)
                    & (ig_df["gesture"] == gesture)
                    & (ig_df["condition"] == "ME")
                ]
                mi_ig = ig_df[
                    (ig_df["subject_id"] == subject)
                    & (ig_df["model"] == model)
                    & (ig_df["gesture"] == gesture)
                    & (ig_df["condition"] == "MI")
                ]

                me_t = temporal_groups.get((subject, model, gesture, "ME"), _empty_t)
                mi_t = temporal_groups.get((subject, model, gesture, "MI"), _empty_t)

                if (
                    len(me_ig) != N_CHANNELS
                    or len(mi_ig) != N_CHANNELS
                    or len(me_t) != WINDOW_SAMPLES
                    or len(mi_t) != WINDOW_SAMPLES
                ):
                    rows.append({
                        "subject_id": subject,
                        "model": model,
                        "gesture": gesture,
                        "channel_similarity": np.nan,
                        "temporal_similarity": np.nan,
                        "full_map_similarity": np.nan,
                        "similarity_method": "Spearman_abs_IG",
                    })
                    continue

                me_c = (
                    me_ig.set_index("channel")
                    .reindex(CHANNEL_NAMES)["mean_abs_IG"]
                    .to_numpy(dtype=float)
                )
                mi_c = (
                    mi_ig.set_index("channel")
                    .reindex(CHANNEL_NAMES)["mean_abs_IG"]
                    .to_numpy(dtype=float)
                )
                me_t = (
                    me_t.sort_values("time_index")["mean_abs_IG"]
                    .to_numpy(dtype=float)
                )
                mi_t = (
                    mi_t.sort_values("time_index")["mean_abs_IG"]
                    .to_numpy(dtype=float)
                )

                # Direct cache access for the complete 13×500 map.
                def cache_map(condition):
                    cp = BLOCK13_ROOT / "cache_dl_xai" / (
                        f"{model_dir_name(model)}_S{subject:02d}_{condition}.npz"
                    )
                    with np.load(cp, allow_pickle=False) as z:
                        arr = np.asarray(z["ig_abs"][gesture - 1], dtype=np.float32)
                    if arr.shape != (N_CHANNELS, WINDOW_SAMPLES):
                        raise RuntimeError(
                            f"Invalid full-map cache shape: {cp}: {arr.shape}"
                        )
                    return arr

                me_map = cache_map("ME").ravel().astype(float)
                mi_map = cache_map("MI").ravel().astype(float)

                rows.append({
                    "subject_id": subject,
                    "model": model,
                    "gesture": gesture,
                    "channel_similarity": safe_spearman(me_c, mi_c),
                    "temporal_similarity": safe_spearman(me_t, mi_t),
                    "full_map_similarity": safe_spearman(me_map, mi_map),
                    "similarity_method": "Spearman_abs_IG",
                })

    return pd.DataFrame(rows)



# =============================================================================
# 15. SUMMARIES / FIGURES
# =============================================================================

def build_xai_summary(
    ml_features: pd.DataFrame,
    ml_channels: pd.DataFrame,
    similarity: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model in sorted(
        set(ML_MODELS) | set(DL_MODELS)
    ):
        if model in ML_MODELS:
            sf = ml_features[
                (ml_features["window_id"] == "SUBJECT_GESTURE_SUMMARY")
                & (ml_features["model"] == model)
            ]
            sc = ml_channels[ml_channels["model"] == model]

            if not sf.empty:
                top = (
                    sf.sort_values(
                        ["subject_id", "condition", "gesture", "shap_abs"],
                        ascending=[True, True, True, False],
                    )
                    .groupby(
                        ["subject_id", "condition", "gesture"],
                        as_index=False,
                    )
                    .head(TOP_FEATURES)
                )
                top_features = top["feature_name"].value_counts().head(TOP_FEATURES).index.tolist()
            else:
                top_features = []

            top_channels = (
                sc.groupby("channel")["raw_importance"]
                .mean()
                .sort_values(ascending=False)
                .head(TOP_CHANNELS)
                .index.tolist()
                if not sc.empty else []
            )

            sim = similarity[similarity["model"] == model]["channel_similarity"]
        else:
            top_features = []
            top_channels = []
            sim = similarity[similarity["model"] == model]["channel_similarity"]

        vals = pd.to_numeric(sim, errors="coerce").to_numpy()
        vals = vals[np.isfinite(vals)]
        ci_lo, ci_hi = bootstrap_ci(vals)

        rows.append({
            "model": model,
            "method": "SHAP" if model in ML_MODELS else "IntegratedGradients",
            "condition": "ME_MI",
            "gesture": "1-7",
            "n_subjects": int(len(vals)),
            "top_features": ";".join(map(str, top_features)),
            "top_channels": ";".join(map(str, top_channels)),
            "top_frequency_groups": "",
            "attribution_similarity_mean": float(np.mean(vals)) if len(vals) else np.nan,
            "attribution_similarity_median": float(np.median(vals)) if len(vals) else np.nan,
            "CI_lower": ci_lo,
            "CI_upper": ci_hi,
        })

    return pd.DataFrame(rows)


def save_ml_figure(ml_channels: pd.DataFrame):
    # One fixed visualization scale across ME/MI for each model.
    for model in ML_MODELS:
        d = ml_channels[ml_channels["model"] == model]
        if d.empty:
            continue

        summary = (
            d.groupby(["condition", "channel"])["normalized_importance"]
            .mean()
            .reset_index()
        )
        piv = summary.pivot(
            index="channel", columns="condition", values="normalized_importance"
        ).reindex(CHANNEL_NAMES)

        fig, ax = plt.subplots(figsize=(9, 6))
        piv.plot(kind="bar", ax=ax)
        ax.set_title(f"{model} — mean normalized SHAP channel attribution")
        ax.set_xlabel("EEG channel")
        ax.set_ylabel("Normalized attribution")
        fig.tight_layout()
        fig.savefig(
            FIGURE_ROOT / f"{model}_SHAP_channel_importance.png",
            dpi=300,
        )
        plt.close(fig)


def save_similarity_figure(similarity: pd.DataFrame):
    summary = (
        similarity.groupby("model")["channel_similarity"]
        .mean()
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    summary.plot(kind="bar", ax=ax)
    ax.axhline(0.0, linewidth=1)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("ME–MI attribution similarity")
    ax.set_title("BLOCK 13 — ME–MI channel attribution similarity")
    fig.tight_layout()
    fig.savefig(
        FIGURE_ROOT / "ME_MI_attribution_similarity.png",
        dpi=300,
    )
    plt.close(fig)


# =============================================================================
# 16. MANUSCRIPT TABLES / FIGURES
# =============================================================================

def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def benjamini_hochberg(pvalues, q=0.05):
    """Benjamini–Hochberg FDR correction; NaNs remain NaN."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not mask.any():
        return out
    vals = p[mask]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    tmp = np.empty_like(adj)
    tmp[order] = adj
    out[mask] = tmp
    return out


def paired_signflip_p(values, n_perm=2000, seed=42):
    """Two-sided paired sign-flip permutation p-value against zero."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    observed = abs(float(np.mean(x)))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    chunk = 500
    exceed = 0
    total = 0
    for start_i in range(0, n_perm, chunk):
        n = min(chunk, n_perm - start_i)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, len(x)))
        perm_means = np.abs((signs * x[None, :]).mean(axis=1))
        exceed += int(np.count_nonzero(perm_means >= observed))
        total += n
    return float((exceed + 1) / (total + 1))


def paired_effect_size(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    sd = float(np.std(x, ddof=1))
    return float(np.mean(x) / sd) if sd > 0 else np.nan


def bootstrap_mean_ci(values, n_boot=2000, seed=42):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    # Vectorized, but bounded to one group at a time.
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def save_xai_statistical_tables(
    ml_channels: pd.DataFrame,
    dl_ig: pd.DataFrame,
    similarity: pd.DataFrame,
) -> dict:
    """Create subject-level paired ME–MI XAI statistics without pseudo-replication."""
    written = {}

    rows = []

    # ML: normalized channel attribution, paired within subject.
    for model in ML_MODELS:
        for gesture in GESTURES:
            for channel in CHANNEL_NAMES:
                me = ml_channels[
                    (ml_channels["model"] == model)
                    & (ml_channels["gesture"] == gesture)
                    & (ml_channels["channel"] == channel)
                    & (ml_channels["condition"] == "ME")
                ].set_index("subject_id")["normalized_importance"]
                mi = ml_channels[
                    (ml_channels["model"] == model)
                    & (ml_channels["gesture"] == gesture)
                    & (ml_channels["channel"] == channel)
                    & (ml_channels["condition"] == "MI")
                ].set_index("subject_id")["normalized_importance"]
                common = me.index.intersection(mi.index)
                diff = (me.loc[common] - mi.loc[common]).to_numpy(dtype=float)
                lo, hi = bootstrap_mean_ci(diff, seed=1000 + gesture)
                rows.append({
                    "branch": "ML",
                    "model": model,
                    "gesture": int(gesture),
                    "channel": channel,
                    "n_subjects": int(len(diff)),
                    "mean_ME_minus_MI": float(np.mean(diff)) if len(diff) else np.nan,
                    "median_ME_minus_MI": float(np.median(diff)) if len(diff) else np.nan,
                    "effect_size_dz": paired_effect_size(diff),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "p_value": paired_signflip_p(diff, seed=2000 + gesture),
                })

    # DL: mean absolute IG channel attribution, paired within subject.
    for model in DL_MODELS:
        for gesture in GESTURES:
            for channel in CHANNEL_NAMES:
                me = dl_ig[
                    (dl_ig["model"] == model)
                    & (dl_ig["gesture"] == gesture)
                    & (dl_ig["channel"] == channel)
                    & (dl_ig["condition"] == "ME")
                ].set_index("subject_id")["mean_abs_IG"]
                mi = dl_ig[
                    (dl_ig["model"] == model)
                    & (dl_ig["gesture"] == gesture)
                    & (dl_ig["channel"] == channel)
                    & (dl_ig["condition"] == "MI")
                ].set_index("subject_id")["mean_abs_IG"]
                common = me.index.intersection(mi.index)
                diff = (me.loc[common] - mi.loc[common]).to_numpy(dtype=float)
                lo, hi = bootstrap_mean_ci(diff, seed=3000 + gesture)
                rows.append({
                    "branch": "DL",
                    "model": model,
                    "gesture": int(gesture),
                    "channel": channel,
                    "n_subjects": int(len(diff)),
                    "mean_ME_minus_MI": float(np.mean(diff)) if len(diff) else np.nan,
                    "median_ME_minus_MI": float(np.median(diff)) if len(diff) else np.nan,
                    "effect_size_dz": paired_effect_size(diff),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "p_value": paired_signflip_p(diff, seed=4000 + gesture),
                })

    diff_df = pd.DataFrame(rows)
    diff_df["p_fdr"] = np.nan
    for (branch, model), idx in diff_df.groupby(["branch", "model"]).groups.items():
        diff_df.loc[idx, "p_fdr"] = benjamini_hochberg(
            diff_df.loc[idx, "p_value"].to_numpy(), q=FDR_Q
        )
    p = TABLE_ROOT / "XAI_ME_MI_paired_channel_statistics.csv"
    atomic_csv(p, diff_df)
    written["XAI_ME_MI_paired_channel_statistics"] = str(p)

    # Similarity inference: subject is the unit, separately for each model/metric/gesture.
    sim_rows = []
    for model in sorted(set(ML_MODELS) | set(DL_MODELS)):
        for gesture in GESTURES:
            d = similarity[
                (similarity["model"] == model)
                & (similarity["gesture"] == gesture)
            ]
            for metric in ("channel_similarity", "temporal_similarity", "full_map_similarity"):
                x = d[metric].to_numpy(dtype=float)
                x = x[np.isfinite(x)]
                lo, hi = bootstrap_mean_ci(x, seed=5000 + gesture)
                sim_rows.append({
                    "model": model,
                    "gesture": int(gesture),
                    "metric": metric,
                    "n_subjects": int(len(x)),
                    "mean": float(np.mean(x)) if len(x) else np.nan,
                    "median": float(np.median(x)) if len(x) else np.nan,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "p_value_vs_zero": paired_signflip_p(x, seed=6000 + gesture),
                })
    sim_df = pd.DataFrame(sim_rows)
    sim_df["p_fdr"] = np.nan
    for model, idx in sim_df.groupby("model").groups.items():
        sim_df.loc[idx, "p_fdr"] = benjamini_hochberg(
            sim_df.loc[idx, "p_value_vs_zero"].to_numpy(), q=FDR_Q
        )
    p = TABLE_ROOT / "XAI_attribution_similarity_statistics.csv"
    atomic_csv(p, sim_df)
    written["XAI_attribution_similarity_statistics"] = str(p)

    return written

def save_xai_tables(
    ml_features: pd.DataFrame,
    ml_channels: pd.DataFrame,
    dl_ig: pd.DataFrame,
    dl_occ: pd.DataFrame,
    similarity: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict:
    """Write compact manuscript-facing XAI tables derived only from computed data."""
    written = {}

    # ML: top features across subject/condition/gesture/model.
    ml_summary = ml_features[
        ml_features["window_id"] == "SUBJECT_GESTURE_SUMMARY"
    ].copy()
    top_ml = (
        ml_summary.groupby(
            ["model", "condition", "gesture", "feature_id", "feature_name",
             "feature_family", "channel", "frequency_group"],
            as_index=False,
        )["shap_abs"]
        .mean()
        .rename(columns={"shap_abs": "mean_abs_shap"})
    )
    top_ml["rank"] = top_ml.groupby(
        ["model", "condition", "gesture"]
    )["mean_abs_shap"].rank(method="min", ascending=False).astype(int)
    top_ml = top_ml[top_ml["rank"] <= TOP_FEATURES].sort_values(
        ["model", "condition", "gesture", "rank"]
    )
    p = TABLE_ROOT / "ML_top_features.csv"
    atomic_csv(p, top_ml)
    written["ML_top_features"] = str(p)

    # ML: channel-level condition/gesture summary.
    p = TABLE_ROOT / "ML_channel_importance.csv"
    atomic_csv(p, ml_channels)
    written["ML_channel_importance"] = str(p)

    p = TABLE_ROOT / "ML_gesture_channel_attribution.csv"
    gesture_channel = (
        ml_channels.groupby(
            ["model", "condition", "gesture", "channel"], as_index=False
        )["normalized_importance"].mean()
    )
    atomic_csv(p, gesture_channel)
    written["ML_gesture_channel_attribution"] = str(p)

    # ML: feature-family and frequency-group summaries.
    family = (
        ml_summary.groupby(
            ["model", "condition", "gesture", "feature_family"], as_index=False
        )["shap_abs"].mean()
        .rename(columns={"shap_abs": "mean_abs_shap"})
    )
    p = TABLE_ROOT / "ML_feature_family_importance.csv"
    atomic_csv(p, family)
    written["ML_feature_family_importance"] = str(p)

    freq = ml_summary.copy()
    freq["frequency_group"] = freq["frequency_group"].fillna("NONE")
    freq = (
        freq.groupby(
            ["model", "condition", "gesture", "frequency_group"], as_index=False
        )["shap_abs"].mean()
        .rename(columns={"shap_abs": "mean_abs_shap"})
    )
    p = TABLE_ROOT / "ML_frequency_group_importance.csv"
    atomic_csv(p, freq)
    written["ML_frequency_group_importance"] = str(p)

    # DL: compact channel and temporal summaries from the full attribution tables.
    ig_channel = (
        dl_ig.groupby(["subject_id", "condition", "model", "gesture", "channel"], as_index=False)
        .agg(mean_abs_IG=("mean_abs_IG", "mean"), mean_signed_IG=("mean_signed_IG", "mean"))
    )
    p = TABLE_ROOT / "DL_IG_channel_summary.csv"
    atomic_csv(p, ig_channel)
    written["DL_IG_channel_summary"] = str(p)

    # Temporal summaries are generated by run_dl_xai() directly to avoid
    # reconstructing 11M-row channel×time DataFrames.
    p = TABLE_ROOT / "DL_IG_temporal_summary.csv"
    if not p.exists():
        raise RuntimeError("Missing canonical DL_IG_temporal_summary.csv.")
    written["DL_IG_temporal_summary"] = str(p)

    occ_channel = (
        dl_occ.groupby(["subject_id", "condition", "model", "gesture", "channel"], as_index=False)
        .agg(mean_abs_occlusion=("mean_abs_occlusion", "mean"), mean_signed_occlusion=("prediction_change", "mean"))
    )
    p = TABLE_ROOT / "DL_occlusion_channel_summary.csv"
    atomic_csv(p, occ_channel)
    written["DL_occlusion_channel_summary"] = str(p)

    p = TABLE_ROOT / "DL_occlusion_temporal_summary.csv"
    if not p.exists():
        raise RuntimeError("Missing canonical DL_occlusion_temporal_summary.csv.")
    written["DL_occlusion_temporal_summary"] = str(p)

    # Similarity summary retains channel, temporal and full-map metrics.
    sim_summary = (
        similarity.groupby(["model", "similarity_method"], as_index=False)
        .agg(
            n_subject_gesture=("subject_id", "size"),
            mean_channel_similarity=("channel_similarity", "mean"),
            median_channel_similarity=("channel_similarity", "median"),
            mean_temporal_similarity=("temporal_similarity", "mean"),
            median_temporal_similarity=("temporal_similarity", "median"),
            mean_full_map_similarity=("full_map_similarity", "mean"),
            median_full_map_similarity=("full_map_similarity", "median"),
        )
    )
    p = TABLE_ROOT / "ME_MI_attribution_similarity_summary.csv"
    atomic_csv(p, sim_summary)
    written["ME_MI_attribution_similarity_summary"] = str(p)

    # The 10-row XAI summary is retained as the manuscript index table.
    p = TABLE_ROOT / "XAI_summary.csv"
    atomic_csv(p, summary)
    written["XAI_summary"] = str(p)

    return written


def save_xai_figures(
    ml_channels: pd.DataFrame,
    dl_ig: pd.DataFrame,
    dl_occ: pd.DataFrame,
    similarity: pd.DataFrame,
) -> list[str]:
    """Create the complete BLOCK13 XAI figure set.

    Every primary plot preserves condition and all seven numerical gesture IDs.
    Raw per-window attribution remains in the NPZ archives; these figures are
    manuscript-facing summaries rather than substitutes for the raw evidence.
    """
    paths = []
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    # Canonical temporal summaries are intentionally loaded from disk.
    ig_temporal = pd.read_csv(TABLE_ROOT / "DL_IG_temporal_summary.csv")
    occ_temporal = pd.read_csv(TABLE_ROOT / "DL_occlusion_temporal_summary.csv")

    # 1. ML channel heatmaps by condition.
    for condition in CONDITIONS:
        pivot = (
            ml_channels[ml_channels["condition"] == condition]
            .groupby(["model", "channel"])["normalized_importance"]
            .mean()
            .unstack("channel")
            .reindex(index=ML_MODELS, columns=CHANNEL_NAMES)
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
        ax.set_xticks(range(N_CHANNELS), CHANNEL_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(len(ML_MODELS)), ML_MODELS)
        ax.set_title(f"ML SHAP channel attribution — {condition}")
        ax.set_xlabel("EEG channel")
        ax.set_ylabel("Model")
        fig.colorbar(im, ax=ax, label="Mean normalized |SHAP|")
        path = FIGURE_ROOT / f"ML_SHAP_channel_heatmap_{condition}.png"
        _save_figure(fig, path); paths.append(str(path))

    # 2. Gesture × channel ML heatmaps, one per condition.
    for condition in CONDITIONS:
        pivot = (
            ml_channels[ml_channels["condition"] == condition]
            .groupby(["gesture", "channel"])["normalized_importance"]
            .mean()
            .unstack("channel")
            .reindex(index=GESTURES, columns=CHANNEL_NAMES)
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
        ax.set_xticks(range(N_CHANNELS), CHANNEL_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(N_GESTURES), [str(g) for g in GESTURES])
        ax.set_title(f"ML gesture-specific channel attribution — {condition}")
        ax.set_xlabel("EEG channel")
        ax.set_ylabel("Gesture ID")
        fig.colorbar(im, ax=ax, label="Mean normalized |SHAP|")
        path = FIGURE_ROOT / f"ML_gesture_channel_heatmap_{condition}.png"
        _save_figure(fig, path); paths.append(str(path))

    # 3. Frequency-group attribution: one compact gesture × frequency plot per condition.
    freq_path = TABLE_ROOT / "ML_frequency_group_importance.csv"
    if freq_path.exists():
        freq = pd.read_csv(freq_path)
        for condition in CONDITIONS:
            d = freq[freq["condition"] == condition]
            if not d.empty:
                pivot = (
                    d.groupby(["gesture", "frequency_group"])["mean_abs_shap"]
                    .mean()
                    .unstack("frequency_group")
                    .reindex(index=GESTURES)
                )
                fig, ax = plt.subplots(figsize=(12, 6))
                im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
                ax.set_xticks(range(len(pivot.columns)), list(pivot.columns), rotation=45, ha="right")
                ax.set_yticks(range(N_GESTURES), [str(g) for g in GESTURES])
                ax.set_title(f"ML frequency-group attribution — {condition}")
                ax.set_xlabel("BLOCK05 frequency group")
                ax.set_ylabel("Gesture ID")
                fig.colorbar(im, ax=ax, label="Mean |SHAP|")
                path = FIGURE_ROOT / f"ML_frequency_gesture_heatmap_{condition}.png"
                _save_figure(fig, path); paths.append(str(path))

    # 4. DL IG channel heatmaps by condition.
    ig_ch = (
        dl_ig.groupby(["model", "condition", "channel"])["mean_abs_IG"].mean()
        .reset_index()
    )
    for condition in CONDITIONS:
        pivot = (
            ig_ch[ig_ch["condition"] == condition]
            .pivot(index="model", columns="channel", values="mean_abs_IG")
            .reindex(index=DL_MODELS, columns=CHANNEL_NAMES)
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
        ax.set_xticks(range(N_CHANNELS), CHANNEL_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(len(DL_MODELS)), DL_MODELS)
        ax.set_title(f"DL Integrated Gradients channel attribution — {condition}")
        ax.set_xlabel("EEG channel")
        ax.set_ylabel("Model")
        fig.colorbar(im, ax=ax, label="Mean |IG|")
        path = FIGURE_ROOT / f"DL_IG_channel_heatmap_{condition}.png"
        _save_figure(fig, path); paths.append(str(path))

    # 5. DL gesture × channel IG heatmaps.
    for condition in CONDITIONS:
        pivot = (
            dl_ig[dl_ig["condition"] == condition]
            .groupby(["gesture", "channel"])["mean_abs_IG"]
            .mean()
            .unstack("channel")
            .reindex(index=GESTURES, columns=CHANNEL_NAMES)
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
        ax.set_xticks(range(N_CHANNELS), CHANNEL_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(N_GESTURES), [str(g) for g in GESTURES])
        ax.set_title(f"DL gesture-specific IG channel attribution — {condition}")
        ax.set_xlabel("EEG channel")
        ax.set_ylabel("Gesture ID")
        fig.colorbar(im, ax=ax, label="Mean |IG|")
        path = FIGURE_ROOT / f"DL_IG_gesture_channel_heatmap_{condition}.png"
        _save_figure(fig, path); paths.append(str(path))

    # 6. DL IG temporal profiles by condition, retaining all models.
    for condition in CONDITIONS:
        d = (
            ig_temporal[ig_temporal["condition"] == condition]
            .groupby(["model", "time_index", "time_seconds"])["mean_abs_IG"]
            .mean().reset_index()
        )
        fig, ax = plt.subplots(figsize=(11, 5))
        for model in DL_MODELS:
            x = d[d["model"] == model]
            if not x.empty:
                ax.plot(x["time_seconds"], x["mean_abs_IG"], label=model)
        ax.set_title(f"DL Integrated Gradients temporal attribution — {condition}")
        ax.set_xlabel("Time within computational window (s)")
        ax.set_ylabel("Mean |IG|")
        ax.legend(fontsize=8)
        path = FIGURE_ROOT / f"DL_IG_temporal_profiles_{condition}.png"
        _save_figure(fig, path); paths.append(str(path))

    # 7. Gesture-specific temporal IG plots, one figure per gesture/condition.
    for condition in CONDITIONS:
        for gesture in GESTURES:
            d = ig_temporal[
                (ig_temporal["condition"] == condition)
                & (ig_temporal["gesture"] == gesture)
            ]
            fig, ax = plt.subplots(figsize=(11, 4.5))
            for model in DL_MODELS:
                x = d[d["model"] == model].sort_values("time_index")
                if not x.empty:
                    ax.plot(x["time_seconds"], x["mean_abs_IG"], label=model)
            ax.set_title(
                f"DL IG temporal attribution — {condition}, gesture {gesture}"
            )
            ax.set_xlabel("Time within computational window (s)")
            ax.set_ylabel("Mean |IG|")
            ax.legend(fontsize=8)
            path = FIGURE_ROOT / f"DL_IG_temporal_gesture_{condition}_{gesture}.png"
            _save_figure(fig, path); paths.append(str(path))

    # 8. Occlusion channel heatmaps.
    occ_ch = (
        dl_occ.groupby(["model", "condition", "channel"])["mean_abs_occlusion"].mean()
        .reset_index()
    )
    for condition in CONDITIONS:
        pivot = (
            occ_ch[occ_ch["condition"] == condition]
            .pivot(index="model", columns="channel", values="mean_abs_occlusion")
            .reindex(index=DL_MODELS, columns=CHANNEL_NAMES)
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
        ax.set_xticks(range(N_CHANNELS), CHANNEL_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(len(DL_MODELS)), DL_MODELS)
        ax.set_title(f"DL occlusion channel attribution — {condition}")
        ax.set_xlabel("EEG channel")
        ax.set_ylabel("Model")
        fig.colorbar(im, ax=ax, label="Mean |occlusion|")
        path = FIGURE_ROOT / f"DL_occlusion_channel_heatmap_{condition}.png"
        _save_figure(fig, path); paths.append(str(path))

    # 9. ME–MI attribution similarity: channel, temporal, full map.
    for metric, label, filename in (
        ("channel_similarity", "Channel similarity", "channel"),
        ("temporal_similarity", "Temporal similarity", "temporal"),
        ("full_map_similarity", "Full-map similarity", "full_map"),
    ):
        d = similarity.groupby("model")[metric].mean().reindex(
            list(ML_MODELS) + list(DL_MODELS)
        )
        fig, ax = plt.subplots(figsize=(11, 5))
        d.plot(kind="bar", ax=ax)
        ax.axhline(0.0, linewidth=1)
        ax.set_ylim(-1.0, 1.0)
        ax.set_ylabel(label)
        ax.set_xlabel("Model")
        ax.set_title(f"ME–MI attribution similarity — {label}")
        path = FIGURE_ROOT / f"ME_MI_attribution_similarity_{filename}.png"
        _save_figure(fig, path); paths.append(str(path))

    # 10. Gesture × model attribution-similarity heatmap.
    pivot = (
        similarity.groupby(["model", "gesture"])["channel_similarity"]
        .mean().unstack("gesture")
        .reindex(index=list(ML_MODELS) + list(DL_MODELS), columns=GESTURES)
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest",
                   vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(N_GESTURES), [str(g) for g in GESTURES])
    ax.set_yticks(range(len(pivot.index)), list(pivot.index))
    ax.set_xlabel("Gesture ID")
    ax.set_ylabel("Model")
    ax.set_title("ME–MI channel attribution similarity by gesture")
    fig.colorbar(im, ax=ax, label="Spearman similarity")
    path = FIGURE_ROOT / "ME_MI_attribution_similarity_gesture_heatmap.png"
    _save_figure(fig, path); paths.append(str(path))

    # 11. IG vs occlusion consistency, computed at subject/model/gesture level.
    consistency_rows = []
    for model in DL_MODELS:
        for subject in SUBJECTS:
            for condition in CONDITIONS:
                for gesture in GESTURES:
                    a = dl_ig[
                        (dl_ig["model"] == model)
                        & (dl_ig["subject_id"] == subject)
                        & (dl_ig["condition"] == condition)
                        & (dl_ig["gesture"] == gesture)
                    ].set_index("channel")["mean_abs_IG"].reindex(CHANNEL_NAMES)
                    b = dl_occ[
                        (dl_occ["model"] == model)
                        & (dl_occ["subject_id"] == subject)
                        & (dl_occ["condition"] == condition)
                        & (dl_occ["gesture"] == gesture)
                    ].set_index("channel")["mean_abs_occlusion"].reindex(CHANNEL_NAMES)
                    consistency_rows.append({
                        "subject_id": subject,
                        "condition": condition,
                        "model": model,
                        "gesture": int(gesture),
                        "channel_rank_correlation": safe_spearman(
                            a.to_numpy(dtype=float), b.to_numpy(dtype=float)
                        ),
                    })
    consistency = pd.DataFrame(consistency_rows)
    atomic_csv(TABLE_ROOT / "IG_occlusion_channel_consistency.csv", consistency)

    summary = (
        consistency.groupby(["model", "gesture"])["channel_rank_correlation"]
        .mean().unstack("gesture").reindex(columns=GESTURES)
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(summary.to_numpy(dtype=float), aspect="auto",
                   interpolation="nearest", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(N_GESTURES), [str(g) for g in GESTURES])
    ax.set_yticks(range(len(summary.index)), list(summary.index))
    ax.set_xlabel("Gesture ID")
    ax.set_ylabel("DL model")
    ax.set_title("IG–occlusion channel attribution consistency")
    fig.colorbar(im, ax=ax, label="Spearman rank correlation")
    path = FIGURE_ROOT / "IG_occlusion_channel_consistency_heatmap.png"
    _save_figure(fig, path); paths.append(str(path))

    # Preserve original per-ML-model figures.
    save_ml_figure(ml_channels)
    for model in ML_MODELS:
        paths.append(str(FIGURE_ROOT / f"{model}_SHAP_channel_importance.png"))

    return paths



# =============================================================================
# 17. MAIN
# =============================================================================

def main():
    started = time.time()
    status = {
        "block": BLOCK_ID,
        "protocol_version": PROTOCOL_VERSION,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "phases": {},
    }

    config = {
        "block_id": BLOCK_ID,
        "protocol_version": PROTOCOL_VERSION,
        "random_seed": RANDOM_SEED,
        "sampling_rate_hz": FS,
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": WINDOW_SECONDS,
        "overlap_percent": OVERLAP_PERCENT,
        "step_samples": STEP_SAMPLES,
        "channels": N_CHANNELS,
        "features": N_FEATURES,
        "subjects": list(SUBJECTS),
        "conditions": list(CONDITIONS),
        "gesture_ids": list(GESTURES),
        "ml_models": list(ML_MODELS),
        "dl_models": list(DL_MODELS),
        "explanation_window_indices": list(EXPLANATION_WINDOW_INDICES),
        "ml_shap_background": "zero_feature_vector_for_non_tree_models; TreeExplainer_native_for_tree_models",
        "ml_target": "true_gesture_class",
        "ig_baseline": IG_BASELINE,
        "ig_steps": IG_STEPS,
        "ig_internal_batch_size": IG_INTERNAL_BATCH_SIZE,
        "ig_target": IG_TARGET,
        "occlusion_segment_length": OCCLUSION_SEGMENT_LENGTH,
        "occlusion_step": OCCLUSION_STEP,
        "occlusion_replacement": OCCLUSION_REPLACEMENT,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "fdr_q": FDR_Q,
        "interpretation_warning": (
            "Attribution is model sensitivity/use of recorded input information; "
            "it is not proof of anatomical source or causal neurophysiology."
        ),
        "resting_baseline_available": False,
        "temporal_peak_interpretation": "highly attributed temporal region within computational window",
    }
    atomic_json(CONFIG_PATH, config)

    try:
        log("=" * 90)
        log("BLOCK 13 — XAI START")
        log(f"Root: {BLOCK13_ROOT}")

        # -------------------------------
        # Phase 1: upstream handoffs
        # -------------------------------
        log("PHASE 1/7 BLOCK04 HANDOFF START")
        block04_df, b4 = verify_block04()
        status["phases"]["block04"] = b4
        log(
            f"PHASE 1/7 BLOCK04 HANDOFF PASS: "
            f"windows={len(block04_df)} arrays={b4['window_arrays']}"
        )

        log("PHASE 2/7 BLOCK05 HANDOFF START")
        b5 = verify_block05()
        status["phases"]["block05"] = b5
        log("PHASE 2/7 BLOCK05 HANDOFF PASS: 341 features")

        log("PHASE 3/7 RUNTIME COMPATIBILITY PREFLIGHT START")
        runtime = verify_runtime_dependencies()
        status["phases"]["runtime_preflight"] = runtime
        log(
            "PHASE 3/7 RUNTIME COMPATIBILITY PREFLIGHT PASS: "
            f"sklearn={runtime['scikit_learn']} "
            f"xgboost={runtime['xgboost']} shap={runtime['shap']}"
        )

        log("PHASE 3/7 BLOCK06 HANDOFF START")
        b6 = verify_block06()
        status["phases"]["block06"] = b6
        log("PHASE 3/7 BLOCK06 HANDOFF PASS: 250 LOSO model files")

        # -------------------------------
        # Phase 2: feature provenance
        # -------------------------------
        log("PHASE 4/7 BUILD FEATURE PROVENANCE")
        feature_lookup = build_feature_lookup(block04_df)
        if len(feature_lookup) != N_WINDOWS:
            raise RuntimeError("Feature lookup cardinality mismatch.")
        status["phases"]["feature_provenance"] = {
            "status": "PASS",
            "rows": len(feature_lookup),
        }
        log("PHASE 4/7 FEATURE PROVENANCE PASS")

        # -------------------------------
        # Phase 3: ML XAI
        # -------------------------------
        if joblib is None:
            raise RuntimeError(
                f"joblib unavailable; ML XAI cannot execute: {JOBLIB_ERROR}"
            )

        log("PHASE 5/7 ML SHAP START")
        ml_features, ml_channels = run_ml_xai(feature_lookup)

        ml_raw_expected = len(ML_MODELS) * len(SUBJECTS) * len(CONDITIONS)
        ml_raw_found = sum(1 for subject in SUBJECTS for condition in CONDITIONS for model in ML_MODELS
                           if raw_ml_path(subject, condition, model).exists())
        if ml_raw_found != ml_raw_expected:
            raise RuntimeError(
                f"ML raw attribution archive incomplete: {ml_raw_found}/{ml_raw_expected}"
            )

        ml_feature_path = TABLE_ROOT / "ML_SHAP_feature_importance.csv"
        ml_channel_path = TABLE_ROOT / "ML_channel_importance.csv"
        atomic_csv(ml_feature_path, ml_features)
        atomic_csv(ml_channel_path, ml_channels)

        if ml_features.empty or ml_channels.empty:
            raise RuntimeError("ML XAI produced empty attribution tables.")

        status["phases"]["ml_shap"] = {
            "status": "PASS",
            "feature_rows": len(ml_features),
            "channel_rows": len(ml_channels),
            "raw_archive_files": ml_raw_found,
        }
        log(
            f"PHASE 5/7 ML SHAP PASS: "
            f"feature_rows={len(ml_features)} channel_rows={len(ml_channels)}"
        )
        log("PHASE 5 COMPLETE — transitioning to PHASE 6")

        # -------------------------------
        # Phase 4: DL XAI
        # -------------------------------
        log("PHASE 6/7 DL IG + OCCLUSION START")
        if IntegratedGradients is None or Occlusion is None:
            raise RuntimeError(
                f"Captum unavailable; DL XAI cannot execute: {CAPTUM_ERROR}"
            )
        if torch is None:
            raise RuntimeError(
                f"PyTorch unavailable; DL XAI cannot execute: {TORCH_ERROR}"
            )

        log_memory("PHASE 6 before run_dl_xai")
        log("PHASE 6: CALLING run_dl_xai()")
        dl_ig, dl_occ = run_dl_xai(feature_lookup, block04_df)
        log("PHASE 6: run_dl_xai() RETURNED")
        log_memory("PHASE 6 after run_dl_xai")
        log("PHASE 6: checking raw DL archive count")
        raw_dl_found, raw_dl_expected = count_raw_dl_archives()
        log(
            f"PHASE 6: raw DL archive count = {raw_dl_found}/{raw_dl_expected}"
        )
        if raw_dl_found != raw_dl_expected:
            raise RuntimeError(
                f"DL raw attribution archive incomplete after run: "
                f"{raw_dl_found}/{raw_dl_expected}"
            )
        gc.collect()
        log("PHASE 6: memory cleanup after DL XAI PASS")
        log_memory("PHASE 6 after cleanup")

        ig_path = TABLE_ROOT / "DL_IG_channel_summary.csv"
        occ_path = TABLE_ROOT / "DL_occlusion_channel_summary.csv"
        log("PHASE 6: writing DL channel summary CSVs")
        atomic_csv(ig_path, dl_ig)
        atomic_csv(occ_path, dl_occ)
        log("PHASE 6: DL channel summary CSVs written")

        if dl_ig.empty or dl_occ.empty:
            raise RuntimeError("DL XAI produced empty attribution tables.")

        status["phases"]["dl_xai"] = {
            "status": "PASS",
            "ig_rows": len(dl_ig),
            "occlusion_rows": len(dl_occ),
            "raw_archive_files": raw_dl_found,
        }
        log(
            f"PHASE 6/7 DL IG + OCCLUSION PASS: "
            f"IG_rows={len(dl_ig)} occlusion_rows={len(dl_occ)}"
        )

        # -------------------------------
        # Phase 5: similarity and summaries
        # -------------------------------
        log("PHASE 7/7 ATTRIBUTION SIMILARITY + SUMMARY START")

        ml_similarity = compute_ml_similarity(ml_channels)
        dl_similarity = compute_dl_similarity(dl_ig, dl_occ)
        similarity = pd.concat(
            [ml_similarity, dl_similarity],
            ignore_index=True,
        )

        similarity_path = TABLE_ROOT / "ME_MI_attribution_similarity.csv"
        atomic_csv(similarity_path, similarity)

        # Statistical XAI outputs use the subject as the unit of inference.
        xai_stats_manifest = save_xai_statistical_tables(
            ml_channels, dl_ig, similarity
        )

        summary = build_xai_summary(
            ml_features,
            ml_channels,
            similarity,
        )
        summary_path = TABLE_ROOT / "XAI_summary.csv"
        atomic_csv(summary_path, summary)

        table_manifest = save_xai_tables(
            ml_features, ml_channels, dl_ig, dl_occ, similarity, summary
        )
        table_manifest.update(xai_stats_manifest)
        figure_paths = save_xai_figures(ml_channels, dl_ig, dl_occ, similarity)

        status["phases"]["manuscript_outputs"] = {
            "status": "PASS",
            "table_count": len(table_manifest),
            "figure_count": len(figure_paths),
            "tables": table_manifest,
            "figures": figure_paths,
        }

        status["phases"]["similarity_summary"] = {
            "status": "PASS",
            "similarity_rows": len(similarity),
            "summary_rows": len(summary),
        }
        status["phases"]["raw_attribution_archive"] = {
            "status": "PASS",
            "ml_files": ml_raw_found,
            "ml_expected_files": ml_raw_expected,
            "dl_files": raw_dl_found,
            "dl_expected_files": raw_dl_expected,
            "archive_root": str(RAW_ROOT),
            "ml_archive_type": "per-window true-class signed/absolute SHAP",
            "dl_archive_type": "per-window signed/absolute IG and occlusion",
        }

        # -------------------------------
        # Artifact manifest
        # -------------------------------
        artifacts = []
        for p in sorted(BLOCK13_ROOT.rglob("*")):
            if p.is_file():
                try:
                    artifacts.append({
                        "path": str(p),
                        "size_bytes": p.stat().st_size,
                        "sha256": sha256_file(p),
                    })
                except Exception:
                    pass
        atomic_csv(MANIFEST_PATH, pd.DataFrame(artifacts))

        status["status"] = "PASS"
        status["completed_utc"] = datetime.now(timezone.utc).isoformat()
        status["elapsed_seconds"] = time.time() - started
        atomic_json(STATUS_PATH, status)

        log("=" * 90)
        log("✓ BLOCK 13 PASS — XAI / ATTRIBUTION ANALYSIS FROZEN")
        log(f"ML SHAP rows: {len(ml_features)}")
        log(f"ML channel rows: {len(ml_channels)}")
        log(f"DL IG rows: {len(dl_ig)}")
        log(f"DL occlusion rows: {len(dl_occ)}")
        log(f"Raw ML attribution archives: {ml_raw_found}/{ml_raw_expected}")
        log(f"Raw DL attribution archives: {raw_dl_found}/{raw_dl_expected}")
        log(f"Attribution similarity rows: {len(similarity)}")
        log(f"Output: {BLOCK13_ROOT}")
        log("=" * 90)

    except Exception as exc:
        status["status"] = "FAIL"
        status["completed_utc"] = datetime.now(timezone.utc).isoformat()
        status["elapsed_seconds"] = time.time() - started
        status["error"] = repr(exc)
        status["traceback"] = traceback.format_exc()
        atomic_json(STATUS_PATH, status)

        log("=" * 90)
        log("✗ BLOCK 13 FAIL")
        log(f"{type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        log(f"Diagnostic status: {STATUS_PATH}")
        log("=" * 90)
        raise


if __name__ == "__main__":
    main()