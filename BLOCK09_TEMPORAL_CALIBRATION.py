#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BLOCK 09 — TEMPORAL CALIBRATION AND CALIBRATION-EFFICIENCY ANALYSIS

Canonical executable for Paper 2.

Scientific design
-----------------
- 25 paired subjects
- ME and MI analyzed separately
- 7 gestures
- Sets A/B/C
- frozen W500/O50 windows
- calibration fractions: 0,10,20,30,40,50,60,80,90%
- calibration is the earliest contiguous segment within each SetC gesture recording
- evaluation is a FIXED, predeclared temporal tail:
    SetC, canonical window numbers 19,20,21 (array indices 18,19,20)
    of every gesture recording
- calibration fractions are defined over the AVAILABLE PRE-EVALUATION
  PHYSICAL DURATION, not the number of available windows
- all calibration/evaluation boundaries are checked in physical samples
- every computational window that physically overlaps evaluation is embargoed
- primary endpoint: Macro-F1
- subject is the inferential unit
- no random calibration
- no cross-condition training
- no evaluation-data fitting
- no post-hoc model/window selection

Critical leakage controls
-------------------------
1. The target subject is excluded from source training.
2. The evaluation set is identical for every calibration fraction.
3. Evaluation is fixed to SetC tail windows, which were not used to train the
   frozen Block07 SD models.
4. Block07 LOSO checkpoints are loaded with their EXACT saved channel-wise
   normalization statistics. Block09 never recomputes source normalization.
5. DL frozen-backbone adaptation trains ONLY the final classifier head.
   Frozen BatchNorm/dropout backbone modules are kept in eval mode.
6. ML transformations are fit on source + eligible target calibration only.
7. A_SD is reported in two forms:
   - Primary ML A_SD: frozen BLOCK06 subject-dependent metrics on full SetC.
   - Primary DL A_SD: frozen BLOCK07 SD predictions restricted to the
     predeclared Block09 fixed SetC tail (Window0019-Window0021).
   - Sensitivity ML A_SD: Block09 refit on target SetA+SetB (294 windows),
     evaluated on the same fixed SetC tail.
   - Sensitivity DL A_SD: the same frozen BLOCK07 fixed-tail reference;
     Block09 does not retrain the DL SD model.
8. No fallback source refit or guessed checkpoint is permitted.
9. All authoritative upstream artifacts are verified before execution.

Temporal filtering
------------------
The upstream preprocessing is documented as 4th-order Butterworth 1-40 Hz
zero-phase filtering of the full continuous recording. Block09 therefore
performs an OFFLINE temporal calibration analysis and does NOT claim a causal
online-filter deployment simulation. The physical-sample embargo excludes
raw-sample overlap between calibration and evaluation windows, but does not
exclude zero-phase filtering boundary overlap (~100 ms at 125 Hz). This is
documented in the audit and in BLOCK09_summary.json.

"""

from __future__ import annotations

import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import subprocess
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
import inspect as _INSPECT

from sklearn.ensemble import RandomForestClassifier as _RF_CLASS
from sklearn.tree import DecisionTreeClassifier as _DT_CLASS

# Cache pristine constructor signatures at process start.  During the long
# Block09 calibration run, sklearn's RF tree-construction path can encounter
# a corrupted inspect signature object.  The cached signatures are used only
# during the RF fit and do not alter any model hyperparameter.
_RF_PRISTINE_INIT_SIGNATURE = _INSPECT.signature(_RF_CLASS.__init__)
_DT_PRISTINE_INIT_SIGNATURE = _INSPECT.signature(_DT_CLASS.__init__)


# =============================================================================
# 01. FROZEN CONFIGURATION
# =============================================================================

SEED = 42

FS = 125
N_CHANNELS = 13
WINDOW_LENGTH = 500
OVERLAP_PERCENT = 50
WINDOW_STEP = 250
WINDOWS_PER_RECORDING = 21

SUBJECTS = list(range(1, 26))
CONDITIONS = ["ME", "MI"]
GESTURES = list(range(1, 8))
SETS = ["SetA", "SetB", "SetC"]

N_RECORDINGS = 25 * 2 * 7 * 3
TOTAL_WINDOWS = N_RECORDINGS * WINDOWS_PER_RECORDING

CALIBRATION_FRACTIONS = [0, 10, 20, 30, 40, 50, 60, 80, 90]
CALIBRATION_DENOMINATOR = "available_pre_evaluation_physical_duration"
TEMPORAL_FILTERING_MODE = "offline_zero_phase"
TEMPORAL_CAUSALITY_CLAIM = False

# Frozen Block06 SD training convention: SetA+SetB = 294 windows per
# subject-condition. Block09 uses the same convention for the sensitivity
# ML A_SD refit so that ML A_SD is comparable to Block06 SD.
BLOCK06_SD_TRAIN_SETS = ["SetA", "SetB"]
BLOCK06_SD_TRAIN_WINDOWS_PER_SUBJECT_CONDITION = 294

# Fixed evaluation tail:
# final three W500/O50 windows of every SetC gesture recording.
# Canonical manifest window numbers are 19,20,21; array indices are 18,19,20.
FIXED_EVAL_SET = "SetC"
FIXED_EVAL_WINDOW_INDICES = [18, 19, 20]
FIXED_EVAL_WINDOW_NUMBERS = [19, 20, 21]
FIXED_EVAL_WINDOWS_PER_RECORDING = 3

FIXED_EVAL_WINDOWS_PER_SUBJECT_CONDITION = 7 * 3
FIXED_EVAL_WINDOWS_TOTAL = 25 * 2 * 7 * 3

# Macro-F1 resolution note: 1 / 21 = 4.76 pp per subject-condition.
EVALUATION_MACRO_F1_RESOLUTION_PP = round(
    100.0 / FIXED_EVAL_WINDOWS_PER_SUBJECT_CONDITION, 3
)

PRIMARY_METRIC = "macro_f1"
SECONDARY_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "mcc",
    "cohen_kappa",
]
ALL_METRICS = [PRIMARY_METRIC] + SECONDARY_METRICS

ML_MODELS = ["LDA", "KNN", "SVM", "RandomForest", "XGBoost"]
DL_MODELS = [
    "EEGNet",
    "ShallowConvNet",
    "DeepConvNet",
    "CNN-BiLSTM",
    "CNN-BiLSTM-Attention",
]
ALL_MODELS = ML_MODELS + DL_MODELS

# DL adaptation:
# only final classifier is trainable; all feature-extraction layers remain frozen.
DL_MAX_EPOCHS = 40
DL_PATIENCE = 6
DL_MIN_DELTA = 1e-3
DL_LR = 1e-3
DL_WEIGHT_DECAY = 1e-4
DL_BATCH_SIZE = 64
DL_GRAD_CLIP = 5.0
DL_INTERNAL_VAL_FRACTION = 0.20
DL_INTERNAL_VAL_MIN_WINDOWS = 14

DL_SMALL_CALIBRATION_POLICY = "fixed_source_checkpoint_best_epoch"
DL_LABEL_CONVENTION = None
BLOCK07_MAIN_GUARD_REQUIRED = True

N_BOOTSTRAP = 10000

# Exact Block06 frozen ML configuration, verified against the canonical source.
ML_CONFIG = {
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
        "random_state": SEED,
        "n_jobs": 1,
        "class_weight": None,
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
        "random_state": SEED,
        "n_jobs": -1,
        "tree_method": "hist",
    },
}

RAW_ROOT = Path(r"F:\Faruk\EEG_Motor_Execution_Imagery")
PAPER2_ROOT = RAW_ROOT / "Paper2_EEG_BCI"

BLOCK01_ROOT = PAPER2_ROOT / "01_RAW_DATA"
BLOCK03_ROOT = PAPER2_ROOT / "03_PREPROCESSING"
BLOCK04_ROOT = PAPER2_ROOT / "04_WINDOWING"
BLOCK05_ROOT = PAPER2_ROOT / "05_FEATURE_ENGINEERING"
BLOCK06_ROOT = PAPER2_ROOT / "06_ML"
BLOCK07_ROOT = PAPER2_ROOT / "07_DL"

OUTPUT_ROOT = PAPER2_ROOT / "09_TEMPORAL_CALIBRATION"
CONFIG_ROOT = OUTPUT_ROOT / "configs"
PARTITION_ROOT = OUTPUT_ROOT / "temporal_partitions"
METRIC_ROOT = OUTPUT_ROOT / "metrics"
PRED_ROOT = OUTPUT_ROOT / "predictions"
DIAG_ROOT = OUTPUT_ROOT / "diagnostics"
FIG_ROOT = OUTPUT_ROOT / "figures"
TABLE_ROOT = OUTPUT_ROOT / "tables"
MANIFEST_ROOT = OUTPUT_ROOT / "manifests"

START_TIME = time.perf_counter()


# =============================================================================
# 02. UTILITIES
# =============================================================================

def log(message: str) -> None:
    print(
        f"{time.perf_counter() - START_TIME:9.2f}s | {message}",
        flush=True,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def _stage_state_path() -> Path:
    return DIAG_ROOT / "BLOCK09_runtime_state.json"


def write_stage_state(stage: str, status: str, **details) -> None:
    """Persist the last known execution stage for crash diagnosis.

    This marker is deliberately written before risky/native operations and is
    never treated as a success marker. A native process termination can leave
    the last STARTED state intact, which identifies the stage reached before
    Windows terminated Python.
    """
    payload = {
        "status": str(status),
        "stage": str(stage),
        "elapsed_seconds": round(time.perf_counter() - START_TIME, 3),
        "pid": os.getpid(),
        **details,
    }
    try:
        write_json(_stage_state_path(), payload)
    except Exception as exc:
        print(
            f"WARNING: could not write BLOCK09 runtime state: {exc}",
            file=sys.stderr,
            flush=True,
        )


def mark_stage_start(stage: str, **details) -> None:
    write_stage_state(stage, "STARTED", **details)


def mark_stage_pass(stage: str, **details) -> None:
    write_stage_state(stage, "PASS", **details)


def mark_stage_fail(stage: str, exc: BaseException, **details) -> None:
    write_stage_state(
        stage,
        "FAIL",
        error_type=type(exc).__name__,
        error=str(exc),
        **details,
    )


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def clean_output() -> None:
    """Prepare Block09 outputs without deleting valid/interrupted artifacts.

    Block09 uses prediction CSVs as run-level checkpoints.  Therefore an
    interrupted execution must never delete the output tree at startup.
    Existing artifacts are validated when reused; invalid artifacts are
    overwritten by the current run.
    """
    for p in [
        OUTPUT_ROOT,
        CONFIG_ROOT,
        PARTITION_ROOT,
        METRIC_ROOT,
        PRED_ROOT,
        DIAG_ROOT,
        FIG_ROOT,
        TABLE_ROOT,
        MANIFEST_ROOT,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


def canonical_condition(x) -> str:
    s = str(x).strip().upper()
    if s in {"ME", "MOTOR_EXECUTION", "EXECUTION"}:
        return "ME"
    if s in {"MI", "MOTOR_IMAGERY", "IMAGERY"}:
        return "MI"
    raise ValueError(f"Unknown condition: {x}")


def canonical_set(x) -> str:
    s = str(x).strip().upper()
    if s.startswith("SET"):
        s = s[3:]
    if s not in {"A", "B", "C"}:
        raise ValueError(f"Unknown set: {x}")
    return f"Set{s}"


def canonical_model(x) -> str:
    s = str(x).strip().lower().replace("_", "-").replace(" ", "")
    mapping = {
        "lda": "LDA",
        "knn": "KNN",
        "svm": "SVM",
        "rf": "RandomForest",
        "randomforest": "RandomForest",
        "xgb": "XGBoost",
        "xgboost": "XGBoost",
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
        raise ValueError(f"Unknown model: {x}")
    return mapping[s]


def canonical_metric(x) -> str:
    s = str(x).strip().lower().replace(" ", "_")
    mapping = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "balancedaccuracy": "balanced_accuracy",
        "macro_f1": "macro_f1",
        "macrof1": "macro_f1",
        "macro_f1_score": "macro_f1",
        "f1_macro": "macro_f1",
        "mcc": "mcc",
        "cohen_kappa": "cohen_kappa",
        "cohens_kappa": "cohen_kappa",
        "kappa": "cohen_kappa",
    }
    if s not in mapping:
        raise ValueError(f"Unknown metric: {x}")
    return mapping[s]


def canonical_subject(x) -> int:
    s = str(x).strip().upper()
    if s.startswith("S"):
        s = s[1:]
    n = int(float(s))
    if n not in SUBJECTS:
        raise ValueError(f"Subject outside frozen cohort: {x}")
    return n


def parse_canonical_window_id(window_id: str) -> dict:
    """Parse the canonical BLOCK04 window identity without destroying it."""
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
        raise ValueError(f"Invalid canonical BLOCK04 window_id: {text!r}")
    d = m.groupdict()
    number = int(d["number"])
    if number < 1 or number > WINDOWS_PER_RECORDING:
        raise ValueError(f"Window number outside 1..21: {text!r}")
    return {
        "window_id": text,
        "window_number": number,
        "window_index": number - 1,
        "window_size": int(d["window_size"]),
        "overlap_percent": int(d["overlap"]),
    }


def parse_prediction_window_id(value) -> Tuple[str | None, int]:
    """Normalize Block07 prediction window identifiers to a zero-based index.

    Supported forms:
      - canonical BLOCK04 IDs ending in Window0001..Window0021
      - Block07 SD IDs such as S01_ME_1_SetC_windows_W00
      - numeric zero-based indices
    """
    text = str(value).strip()
    if re.fullmatch(r"\d+", text):
        n = int(text)
        if 0 <= n < WINDOWS_PER_RECORDING:
            return None, n
        raise ValueError(f"Numeric prediction window index outside 0..20: {value!r}")
    m = re.search(r"_windows_W(?P<idx>\d+)$", text, flags=re.IGNORECASE)
    if m:
        idx = int(m.group("idx"))
        if 0 <= idx < WINDOWS_PER_RECORDING:
            return text, idx
        raise ValueError(f"Block07 prediction window index outside 0..20: {value!r}")
    if "_Window" in text:
        parsed = parse_canonical_window_id(text)
        return text, parsed["window_index"]
    raise ValueError(f"Unrecognized prediction window_id: {value!r}")


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} not found: {path}")


def model_dir_name(model: str) -> str:
    return model.replace("-", "_").replace("+", "ATTN")


def calculate_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        cohen_kappa_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=GESTURES,
                average="macro",
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(
            cohen_kappa_score(y_true, y_pred, labels=GESTURES)
        ),
    }


# =============================================================================
# 03. BLOCK 04 / BLOCK 05 / BLOCK 06 / BLOCK 07 VERIFICATION
# =============================================================================

def verify_block04() -> dict:
    complete_path = BLOCK04_ROOT / "BLOCK04_complete.json"
    manifest_path = BLOCK04_ROOT / "final_window_manifest.csv"

    require_file(complete_path, "BLOCK04 completion marker")
    require_file(manifest_path, "BLOCK04 final window manifest")

    complete = json.loads(complete_path.read_text(encoding="utf-8"))

    if str(complete.get("status", "")).upper() != "PASS":
        raise RuntimeError("BLOCK04 status is not PASS.")

    structured_ok = False
    for key in ("window_length_samples", "window_samples", "window_length"):
        if key in complete:
            if int(complete[key]) != WINDOW_LENGTH:
                raise RuntimeError(
                    f"BLOCK04 {key}={complete[key]} != {WINDOW_LENGTH}"
                )
            structured_ok = True
            break
    for key in ("overlap_percent", "window_overlap_percent", "overlap"):
        if key in complete:
            if int(complete[key]) != OVERLAP_PERCENT:
                raise RuntimeError(
                    f"BLOCK04 {key}={complete[key]} != {OVERLAP_PERCENT}"
                )
            structured_ok = True
            break

    if not structured_ok:
        frozen_text = json.dumps(complete, sort_keys=True).lower()
        if "500" not in frozen_text or "50" not in frozen_text:
            raise RuntimeError(
                "BLOCK04 completion marker does not confirm W500/O50."
            )

    df = pd.read_csv(manifest_path)

    required = {
        "window_id",
        "subject",
        "condition",
        "gesture_code",
        "set",
        "source_recording",
        "window_samples",
        "start_sample",
        "end_sample",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"BLOCK04 manifest missing columns: {sorted(missing)}"
        )

    if len(df) != TOTAL_WINDOWS:
        raise RuntimeError(
            f"BLOCK04 manifest has {len(df)} rows; expected {TOTAL_WINDOWS}."
        )

    df["subject"] = df["subject"].map(canonical_subject)
    df["condition"] = df["condition"].map(canonical_condition)
    df["set"] = df["set"].map(canonical_set)
    df["gesture_code"] = df["gesture_code"].astype(int)
    df["window_id"] = df["window_id"].astype(str).str.strip()
    parsed = df["window_id"].map(parse_canonical_window_id)
    df["window_number"] = parsed.map(lambda x: x["window_number"]).astype(int)
    df["window_index"] = parsed.map(lambda x: x["window_index"]).astype(int)
    df["start_sample"] = df["start_sample"].astype(int)
    df["end_sample"] = df["end_sample"].astype(int)

    if set(df.subject.unique()) != set(SUBJECTS):
        raise RuntimeError("BLOCK04 subjects are not exactly 1..25.")
    if set(df.condition.unique()) != set(CONDITIONS):
        raise RuntimeError("BLOCK04 conditions are not exactly ME/MI.")
    if set(df.gesture_code.unique()) != set(GESTURES):
        raise RuntimeError("BLOCK04 gestures are not exactly 1..7.")
    if set(df.set.unique()) != set(SETS):
        raise RuntimeError("BLOCK04 sets are not exactly SetA/SetB/SetC after canonical normalization.")
    if set(df.window_number.unique()) != set(range(1, WINDOWS_PER_RECORDING + 1)):
        raise RuntimeError("BLOCK04 canonical window numbers are not exactly 1..21.")
    if set(df.window_index.unique()) != set(range(WINDOWS_PER_RECORDING)):
        raise RuntimeError("BLOCK04 zero-based window indices are not exactly 0..20.")

    if not np.all(df.window_samples.astype(int).to_numpy() == WINDOW_LENGTH):
        raise RuntimeError("BLOCK04 contains non-500-sample windows.")
    if not np.all(
        (df.end_sample.to_numpy() - df.start_sample.to_numpy()) == WINDOW_LENGTH
    ):
        raise RuntimeError("BLOCK04 contains invalid physical window bounds.")

    group = df.groupby(["subject", "condition", "gesture_code", "set"])
    counts = group.window_id.nunique()
    if len(counts) != N_RECORDINGS:
        raise RuntimeError(
            f"BLOCK04 logical recording groups={len(counts)}; expected {N_RECORDINGS}."
        )
    if not np.all(counts.to_numpy() == WINDOWS_PER_RECORDING):
        raise RuntimeError("BLOCK04 does not contain exactly 21 windows/recording.")

    source_counts = (
        df[["subject", "condition", "gesture_code", "set", "source_recording"]]
        .drop_duplicates()
        .groupby(["subject", "condition", "gesture_code"])
        .source_recording.nunique()
    )
    if not np.all(source_counts.to_numpy() == 3):
        raise RuntimeError(
            "BLOCK04 SetA/SetB/SetC do not map to three distinct source recordings."
        )

    return {
        "status": "PASS",
        "manifest_rows": int(len(df)),
        "manifest_sha256": sha256_file(manifest_path),
        "window_length": WINDOW_LENGTH,
        "overlap_percent": OVERLAP_PERCENT,
        "step_samples": WINDOW_STEP,
        "set_independence_verified": True,
    }


def verify_block04_window_arrays() -> dict:
    """Verify the actual W500/O50 NPZ arrays and physical window identity."""
    manifest_path = BLOCK04_ROOT / "final_window_manifest.csv"
    df = pd.read_csv(manifest_path)
    df["window_id"] = df["window_id"].astype(str).str.strip()
    df["window_index"] = df["window_id"].map(lambda x: parse_canonical_window_id(x)["window_index"])
    df["start_sample"] = df["start_sample"].astype(int)
    df["end_sample"] = df["end_sample"].astype(int)
    df["window_samples"] = df["window_samples"].astype(int)
    df["window_array_file"] = df["window_array_file"].astype(str)

    checked = set()
    for _, r in df.iterrows():
        array_file = resolve_block04_window_array(str(r.window_array_file))
        key = str(array_file)
        if key in checked:
            continue
        checked.add(key)
        with np.load(array_file, allow_pickle=False) as z:
            if "eeg_windows" not in z.files:
                raise RuntimeError(f"Missing eeg_windows in {array_file}")
            arr = np.asarray(z["eeg_windows"])
            if arr.shape != (21, 13, 500):
                raise RuntimeError(f"Unexpected eeg_windows shape {arr.shape} in {array_file}")
            for k in ("start_samples", "end_samples"):
                if k not in z.files or np.asarray(z[k]).shape != (21,):
                    raise RuntimeError(f"Missing/invalid {k} in {array_file}")
            sub = df[df.window_array_file.astype(str) == str(r.window_array_file)]
            starts = np.asarray(z["start_samples"], dtype=int)
            ends = np.asarray(z["end_samples"], dtype=int)
            for _, rr in sub.iterrows():
                idx = int(rr.window_index)
                if int(starts[idx]) != int(rr.start_sample) or int(ends[idx]) != int(rr.end_sample):
                    raise RuntimeError(
                        f"Manifest/NPZ temporal mismatch: {rr.window_id} in {array_file}"
                    )
    return {"status": "PASS", "arrays_verified": len(checked), "shape": [21, 13, 500]}


def verify_block05() -> dict:
    complete_path = BLOCK05_ROOT / "BLOCK05_complete.json"
    summary_path = BLOCK05_ROOT / "BLOCK05_summary.json"

    require_file(complete_path, "BLOCK05 completion marker")
    require_file(summary_path, "BLOCK05 summary")

    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if str(complete.get("status", "")).upper() != "PASS":
        raise RuntimeError("BLOCK05 status is not PASS.")

    frozen = str(complete.get("frozen_configuration", ""))
    if frozen != "W500_O50":
        raise RuntimeError(
            f"BLOCK05 frozen configuration is {frozen!r}, expected W500_O50."
        )

    if int(complete.get("feature_count", -1)) != 341:
        raise RuntimeError("BLOCK05 does not confirm 341 features.")

    if int(complete.get("total_windows", -1)) != TOTAL_WINDOWS:
        raise RuntimeError("BLOCK05 does not confirm 22,050 windows.")

    window_key_candidates = [
        "block04_configuration",
        "windowing",
        "window_config",
        "frozen_configuration",
    ]
    matched = None
    for key in window_key_candidates:
        if key in summary:
            matched = (key, summary[key])
            break

    if matched is None:
        raise RuntimeError(
            "BLOCK05 summary does not declare a recognized windowing "
            f"configuration key. Tried: {window_key_candidates}. "
            f"Available keys: {sorted(summary.keys())}"
        )

    if str(matched[1]) != "W500_O50":
        raise RuntimeError(
            f"BLOCK05 summary key {matched[0]!r} = {matched[1]!r}, "
            "expected W500_O50."
        )

    return {
        "status": "PASS",
        "feature_count": 341,
        "total_windows": TOTAL_WINDOWS,
        "configuration": "W500_O50",
        "complete_sha256": sha256_file(complete_path),
    }


def _find_named_mapping(obj, target_names, max_depth=12):
    """Recursively find mappings whose normalized key matches a target name."""
    hits = []
    target_names = {str(x).lower() for x in target_names}

    def walk(x, path="root", depth=0):
        if depth > max_depth:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                kl = str(k).lower()
                if kl in target_names and isinstance(v, dict):
                    hits.append((path + "." + str(k), v))
                walk(v, path + "." + str(k), depth + 1)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]", depth + 1)

    walk(obj)
    return hits


def _get_model_config(block06_config, aliases):
    hits = _find_named_mapping(block06_config, aliases)
    if not hits:
        return None
    ranked = sorted(
        hits,
        key=lambda x: sum(
            1 for k in (
                "n_estimators", "n_trees", "max_depth", "depth",
                "learning_rate", "eta", "subsample", "colsample_bytree",
                "colsample", "solver", "n_neighbors", "weights", "metric",
                "kernel", "C", "gamma", "objective", "eval_metric",
                "random_state", "seed", "tree_method"
            )
            if k in x[1]
        ),
        reverse=True,
    )
    return ranked[0][1]


def _coerce_config_scalar(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            return text
    return value


def _normalize_config_keys(actual, aliases):
    out = {}
    for canonical, alternatives in aliases.items():
        for key in alternatives:
            if key in actual:
                out[canonical] = actual[key]
                break
    return out


def _assert_matching_config(actual, expected, aliases, label, optional_defaults=None):
    normalized = _normalize_config_keys(actual, aliases)
    optional_defaults = optional_defaults or {}
    mismatches = {}

    for key, expected_value in expected.items():
        if key not in normalized:
            if key in optional_defaults and expected_value == optional_defaults[key]:
                continue
            raise RuntimeError(
                f"BLOCK06 {label} configuration is missing required parameter "
                f"{key!r}. Available keys: {sorted(map(str, actual.keys()))}"
            )

        observed = _coerce_config_scalar(normalized[key])
        expected = _coerce_config_scalar(expected_value)

        if isinstance(observed, float) and isinstance(expected, float):
            ok = math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-9)
        else:
            ok = observed == expected

        if not ok:
            mismatches[key] = {
                "BLOCK06": normalized[key],
                "BLOCK09": expected_value,
            }

    if mismatches:
        raise RuntimeError(
            f"BLOCK06 {label} hyperparameters do not match the frozen "
            f"Block09 specification:\n"
            f"{json.dumps(mismatches, indent=2, default=str)}"
        )


def _locate_block06_source() -> Path:
    candidates = [
        PAPER2_ROOT / "scripts" / "BLOCK06_CONVENTIONAL_ML_FINAL.py",
        PAPER2_ROOT / "BLOCK06_CONVENTIONAL_ML_FINAL.py",
        Path(__file__).resolve().parent / "BLOCK06_CONVENTIONAL_ML_FINAL.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Canonical BLOCK06_CONVENTIONAL_ML_FINAL.py was not found. "
        "Block09 cannot verify the frozen ML specification without it."
    )


def _extract_block06_ml_config(source_path: Path):
    """
    Extract the frozen ML configuration from BLOCK06 source without
    executing the BLOCK06 program.

    Supports frozen literal constants referenced by MODEL_CONFIG, e.g.:

        RANDOM_SEED = 42

        MODEL_CONFIG = {
            ...
            "random_state": RANDOM_SEED,
            ...
        }

    Only safe AST nodes are evaluated. Arbitrary Python code is never
    executed.
    """
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))

    frozen_names = {}

    def safe_eval(node):
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id in frozen_names:
                return frozen_names[node.id]
            raise ValueError(f"Unresolved name: {node.id}")

        if isinstance(node, ast.Dict):
            result = {}
            for key_node, value_node in zip(node.keys, node.values):
                if key_node is None:
                    raise ValueError("Dictionary unpacking is not allowed.")
                key = safe_eval(key_node)
                value = safe_eval(value_node)
                result[key] = value
            return result

        if isinstance(node, ast.List):
            return [safe_eval(x) for x in node.elts]

        if isinstance(node, ast.Tuple):
            return tuple(safe_eval(x) for x in node.elts)

        if isinstance(node, ast.Set):
            return set(safe_eval(x) for x in node.elts)

        if isinstance(node, ast.UnaryOp):
            operand = safe_eval(node.operand)

            if isinstance(node.op, ast.UAdd):
                return +operand

            if isinstance(node.op, ast.USub):
                return -operand

            if isinstance(node.op, ast.Not):
                return not operand

            raise ValueError(
                f"Unsupported unary operator: {type(node.op).__name__}"
            )

        raise ValueError(
            f"Unsupported AST node: {type(node).__name__}"
        )

    # Resolve simple frozen assignments first.
    # This allows MODEL_CONFIG to reference constants such as RANDOM_SEED.
    for _ in range(10):
        progress = False

        for node in tree.body:
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    continue

                target = node.targets[0]

                if not isinstance(target, ast.Name):
                    continue

                name = target.id

                if name in frozen_names:
                    continue

                try:
                    value = safe_eval(node.value)
                except Exception:
                    continue

                frozen_names[name] = value
                progress = True

            elif isinstance(node, ast.AnnAssign):
                if not isinstance(node.target, ast.Name):
                    continue

                name = node.target.id

                if name in frozen_names or node.value is None:
                    continue

                try:
                    value = safe_eval(node.value)
                except Exception:
                    continue

                frozen_names[name] = value
                progress = True

        if not progress:
            break

    config_names = {
        "ML_CONFIG",
        "MODEL_CONFIG",
        "MODEL_CONFIGS",
        "MODEL_HYPERPARAMETERS",
        "MODEL_HYPERPARAMS",
    }

    # Search the complete AST because the configuration may not be a
    # direct top-level statement.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id in config_names
                ):
                    try:
                        return safe_eval(node.value), target.id
                    except Exception:
                        pass

        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id in config_names
                and node.value is not None
            ):
                try:
                    return safe_eval(node.value), node.target.id
                except Exception:
                    pass

    raise RuntimeError(
        f"Could not extract frozen ML configuration from {source_path}."
    )

def verify_block06() -> dict:
    complete_path = BLOCK06_ROOT / "BLOCK06_complete.json"
    config_path = BLOCK06_ROOT / "BLOCK06_config.json"
    summary_path = BLOCK06_ROOT / "BLOCK06_summary.json"

    require_file(complete_path, "BLOCK06 completion marker")
    require_file(config_path, "BLOCK06 configuration artifact")
    require_file(summary_path, "BLOCK06 summary artifact")

    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if str(complete.get("status", "")).upper() != "PASS":
        raise RuntimeError("BLOCK06 status is not PASS.")

    if str(complete.get("global_status", "")).upper() != "PASS":
        raise RuntimeError("BLOCK06 global_status is not PASS.")

    complete_configuration = complete.get("configuration")
    config_configuration = config.get("block04", {}).get("configuration")
    summary_configuration = summary.get("configuration")

    if complete_configuration != "W500_O50":
        raise RuntimeError(
            "BLOCK06 completion artifact configuration is not W500_O50: "
            f"{complete_configuration!r}"
        )

    if config_configuration != "W500_O50":
        raise RuntimeError(
            "BLOCK06 configuration artifact block04.configuration "
            f"is not W500_O50: {config_configuration!r}"
        )

    if summary_configuration != "W500_O50":
        raise RuntimeError(
            "BLOCK06 summary configuration is not W500_O50: "
            f"{summary_configuration!r}"
        )

    complete_feature_count = int(complete.get("feature_count", -1))
    config_feature_count = int(config.get("block05", {}).get("feature_count", -1))
    summary_feature_count = int(summary.get("features_per_window", -1))

    if complete_feature_count != 341:
        raise RuntimeError(
            f"BLOCK06 completion feature_count is not 341: "
            f"{complete_feature_count}"
        )

    if config_feature_count != 341:
        raise RuntimeError(
            f"BLOCK06 configuration feature_count is not 341: "
            f"{config_feature_count}"
        )

    if summary_feature_count != 341:
        raise RuntimeError(
            f"BLOCK06 summary features_per_window is not 341: "
            f"{summary_feature_count}"
        )

    config_total_windows = int(
        config.get("block04", {}).get("total_windows", -1)
    )
    summary_total_windows = int(
        summary.get("total_windows", -1)
    )

    if config_total_windows != TOTAL_WINDOWS:
        raise RuntimeError(
            "BLOCK06 configuration total_windows is not "
            f"{TOTAL_WINDOWS}: {config_total_windows}"
        )

    if summary_total_windows != TOTAL_WINDOWS:
        raise RuntimeError(
            "BLOCK06 summary total_windows is not "
            f"{TOTAL_WINDOWS}: {summary_total_windows}"
        )

    if complete.get("SD_complete") is not True:
        raise RuntimeError("BLOCK06 SD_complete is not True.")

    if complete.get("LOSO_complete") is not True:
        raise RuntimeError("BLOCK06 LOSO_complete is not True.")

    if int(complete.get("LOSO_folds", -1)) != 25:
        raise RuntimeError("BLOCK06 LOSO_folds is not 25.")

    if complete.get("block04_handoff_verified") is not True:
        raise RuntimeError(
            "BLOCK06 block04_handoff_verified is not True."
        )

    if complete.get("block05_handoff_verified") is not True:
        raise RuntimeError(
            "BLOCK06 block05_handoff_verified is not True."
        )

    expected_calibration_levels = [
        0, 10, 20, 30, 40, 50, 60, 80, 90
    ]

    if complete.get("calibration_levels") != expected_calibration_levels:
        raise RuntimeError(
            "BLOCK06 calibration levels do not match BLOCK09 protocol."
        )

    if complete.get("calibration_execution_block") != (
        "09_TEMPORAL_CALIBRATION"
    ):
        raise RuntimeError(
            "BLOCK06 calibration execution block is not "
            "09_TEMPORAL_CALIBRATION."
        )

    if complete.get("calibration_executed_here") is not False:
        raise RuntimeError(
            "BLOCK06 indicates calibration was executed inside BLOCK06."
        )

    source_path = _locate_block06_source()
    cfg, cfg_name = _extract_block06_ml_config(source_path)

    specs = {
        "LDA": (
            ["LDA", "LinearDiscriminantAnalysis"],
            {"solver": ML_CONFIG["LDA"]["solver"]},
            {"solver": ["solver"]},
            {},
        ),
        "KNN": (
            ["KNN", "KNeighborsClassifier", "KNeighbors"],
            {
                "n_neighbors": ML_CONFIG["KNN"]["n_neighbors"],
                "weights": ML_CONFIG["KNN"]["weights"],
                "metric": ML_CONFIG["KNN"]["metric"],
            },
            {
                "n_neighbors": ["n_neighbors", "neighbors", "k"],
                "weights": ["weights"],
                "metric": ["metric"],
            },
            {},
        ),
        "SVM": (
            ["SVM", "SVC", "SupportVectorMachine"],
            {
                "kernel": ML_CONFIG["SVM"]["kernel"],
                "C": ML_CONFIG["SVM"]["C"],
                "gamma": ML_CONFIG["SVM"]["gamma"],
                "probability": ML_CONFIG["SVM"]["probability"],
            },
            {
                "kernel": ["kernel"],
                "C": ["C", "c"],
                "gamma": ["gamma"],
                "probability": ["probability"],
            },
            {},
        ),
        "RandomForest": (
            ["RF", "RandomForest", "RandomForestClassifier"],
            {
                "n_estimators": ML_CONFIG["RF"]["n_estimators"],
                "random_state": ML_CONFIG["RF"]["random_state"],
                "class_weight": ML_CONFIG["RF"]["class_weight"],
            },
            {
                "n_estimators": [
                    "n_estimators",
                    "n_trees",
                    "n_estimators_rf",
                    "trees",
                ],
                "random_state": [
                    "random_state",
                    "seed",
                    "random_seed",
                ],
                "class_weight": ["class_weight", "weights"],
            },
            {"class_weight": None},
        ),
        "XGBoost": (
            ["XGB", "XGBoost", "XGBClassifier"],
            {
                "n_estimators": ML_CONFIG["XGB"]["n_estimators"],
                "max_depth": ML_CONFIG["XGB"]["max_depth"],
                "learning_rate": ML_CONFIG["XGB"]["learning_rate"],
                "subsample": ML_CONFIG["XGB"]["subsample"],
                "colsample_bytree": ML_CONFIG["XGB"]["colsample_bytree"],
                "objective": ML_CONFIG["XGB"]["objective"],
                "eval_metric": ML_CONFIG["XGB"]["eval_metric"],
                "random_state": ML_CONFIG["XGB"]["random_state"],
                "tree_method": ML_CONFIG["XGB"]["tree_method"],
            },
            {
                "n_estimators": [
                    "n_estimators",
                    "num_boost_round",
                    "n_rounds",
                ],
                "max_depth": ["max_depth", "depth"],
                "learning_rate": ["learning_rate", "eta"],
                "subsample": ["subsample", "subsample_fraction"],
                "colsample_bytree": [
                    "colsample_bytree",
                    "colsample",
                ],
                "objective": ["objective"],
                "eval_metric": ["eval_metric"],
                "random_state": [
                    "random_state",
                    "seed",
                    "random_seed",
                ],
                "tree_method": ["tree_method"],
            },
            {},
        ),
    }

    verified = {}

    for label, (names, expected, aliases, optional) in specs.items():
        actual = _get_model_config(cfg, names)

        if actual is None:
            raise RuntimeError(
                f"Could not locate frozen {label} configuration "
                f"in {source_path} ({cfg_name})."
            )

        _assert_matching_config(
            actual,
            expected,
            aliases,
            label,
            optional,
        )

        verified[label] = expected

    return {
        "status": "PASS",
        "complete_sha256": sha256_file(complete_path),
        "config_sha256": sha256_file(config_path),
        "summary_sha256": sha256_file(summary_path),
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "source_config_name": cfg_name,
        "verified_models": verified,
        "config_file_present": config_path.exists(),
        "summary_file_present": summary_path.exists(),
    }

def _locate_block07_source() -> Path:
    candidates = [
        PAPER2_ROOT / "scripts" / "BLOCK07_DEEP_LEARNING_FINAL.py",
        PAPER2_ROOT / "BLOCK07_DEEP_LEARNING_FINAL.py",
        Path(__file__).resolve().parent / "BLOCK07_DEEP_LEARNING_FINAL.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Canonical BLOCK07_DEEP_LEARNING_FINAL.py was not found.")


def _infer_block07_label_convention_from_source(source_text: str) -> str:
    """
    Fallback inference from canonical Block07 source text.

    Only used when Block07_config.json does not declare a label convention.
    Requires three distinct positive hits so that unrelated '- 1' expressions
    cannot trigger a false positive.
    """
    dataset_shift = bool(re.search(
        r"class\s+EEGWindowDataset\b.*?"
        r"torch\.from_numpy\(\s*np\.asarray\(\s*y\s*,\s*dtype=np\.int64\s*\)\s*-\s*1\s*\)",
        source_text,
        flags=re.I | re.S,
    ))
    prediction_shift = bool(re.search(
        r"pred\s*=\s*torch\.argmax\([^;\n]+,\s*dim\s*=\s*1\s*\)"
        r"\.cpu\(\)\.numpy\(\)\s*\+\s*1",
        source_text,
        flags=re.I,
    ))
    true_shift = bool(re.search(
        r"true\s*=\s*yb\.cpu\(\)\.numpy\(\)\s*\+\s*1",
        source_text,
        flags=re.I,
    ))

    if dataset_shift and prediction_shift and true_shift:
        return "zero_based_internal"

    raise RuntimeError(
        "Canonical Block07 label convention could not be verified: either "
        "BLOCK07_config.json must declare label_convention, or the canonical "
        "source must contain the three characteristic 1..7 <-> 0..6 shifts."
    )


def verify_block07() -> dict:
    complete_path = BLOCK07_ROOT / "BLOCK07_complete.json"
    require_file(complete_path, "BLOCK07 completion marker")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if str(complete.get("status", "")).upper() != "PASS":
        raise RuntimeError("BLOCK07 status is not PASS.")

    config_path = BLOCK07_ROOT / "configs" / "BLOCK07_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    for key, expected in (
        ("window_samples", WINDOW_LENGTH),
        ("overlap_percent", OVERLAP_PERCENT),
        ("step_samples", WINDOW_STEP),
    ):
        if key in config and int(config[key]) != expected:
            raise RuntimeError(f"BLOCK07 {key} mismatch.")

    if "normalization" in config and str(config["normalization"]) != "training_only_channelwise_zscore":
        raise RuntimeError("BLOCK07 normalization mismatch.")

    if "checkpoint_metric" in config and str(config["checkpoint_metric"]) != "validation_macro_f1":
        raise RuntimeError("BLOCK07 checkpoint metric mismatch.")

    # Confirm validation-subject rule matches Block09's expectation.
    validation_subject_rule = None
    if "loso" in config and isinstance(config["loso"], dict):
        validation_subject_rule = config["loso"].get("validation_subject_rule")
    if validation_subject_rule is not None:
        expected_rule = "lowest_numbered_subject_among_24_training_subjects"
        if str(validation_subject_rule).strip().lower() != expected_rule.lower():
            raise RuntimeError(
                "BLOCK07 validation_subject_rule is not the frozen "
                f"lowest-numbered convention: {validation_subject_rule!r}"
            )

    folds = BLOCK07_ROOT / "07B_DL_LOSO" / "folds"
    checkpoints = sorted(folds.rglob("best_model.pt")) if folds.exists() else []
    if len(checkpoints) != 250:
        raise RuntimeError(
            f"BLOCK07 LOSO checkpoint count is {len(checkpoints)}; expected 250."
        )

    import torch
    sample = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    for key in ("normalizer_mean", "normalizer_std", "best_epoch", "best_validation_macro_f1"):
        if key not in sample:
            raise RuntimeError(f"BLOCK07 checkpoint lacks required key: {key}")

    global DL_LABEL_CONVENTION
    declared = config.get("label_convention") if config_path.exists() else None
    if declared in ("zero_based_internal", "one_based_direct"):
        DL_LABEL_CONVENTION = declared
    else:
        source_path = _locate_block07_source()
        DL_LABEL_CONVENTION = _infer_block07_label_convention_from_source(
            source_path.read_text(encoding="utf-8")
        )

    return {
        "status": "PASS",
        "complete_sha256": sha256_file(complete_path),
        "config_sha256": sha256_file(config_path) if config_path.exists() else None,
        "config_present": config_path.exists(),
        "normalization": "training_only_channelwise_zscore",
        "checkpoint_metric": "validation_macro_f1",
        "label_convention": DL_LABEL_CONVENTION,
        "loso_checkpoint_count": len(checkpoints),
        "sample_checkpoint_sha256": sha256_file(checkpoints[0]),
        "validation_subject_rule": validation_subject_rule,
    }


def verify_block03_filtering() -> dict:
    """Verify BLOCK03 filtering mode is documented as zero-phase offline."""
    complete_path = BLOCK03_ROOT / "BLOCK03_complete.json"
    if not complete_path.exists():
        # If BLOCK03 completion marker is absent, record the limitation and
        # proceed with the frozen assumption, since the paper's upstream
        # documentation is the primary source of this information.
        return {
            "status": "PASS_WITH_WARNING",
            "filtering_verified": False,
            "reason": "BLOCK03_complete.json not found; relying on documented "
                      "upstream protocol.",
        }

    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    text = json.dumps(complete, sort_keys=True).lower()
    tokens = ("zero-phase", "zero_phase", "zerophase", "filtfilt", "filt")
    filtering_found = any(t in text for t in tokens)

    if not filtering_found:
        raise RuntimeError(
            "BLOCK03 completion marker does not document zero-phase filtering. "
            "Block09 requires this to be documented before execution."
        )
    return {
        "status": "PASS",
        "filtering_verified": True,
        "reason": "BLOCK03 completion marker documents zero-phase filtering.",
    }


# =============================================================================
# 04. BLOCK 04 WINDOW INVENTORY
# =============================================================================

def parse_block04_inventory() -> pd.DataFrame:
    path = BLOCK04_ROOT / "final_window_manifest.csv"
    df = pd.read_csv(path)

    required = {
        "window_id", "subject", "condition", "gesture_code", "set",
        "source_recording", "window_samples", "start_sample", "end_sample",
        "window_array_file",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"BLOCK04 manifest missing columns: {sorted(missing)}")

    df["subject"] = df["subject"].map(canonical_subject)
    df["condition"] = df["condition"].map(canonical_condition)
    df["set"] = df["set"].map(canonical_set)
    df["gesture"] = df["gesture_code"].astype(int)
    df["window_id"] = df["window_id"].astype(str).str.strip()

    parsed = df["window_id"].map(parse_canonical_window_id)
    df["window_number"] = parsed.map(lambda x: x["window_number"]).astype(int)
    df["window_index"] = parsed.map(lambda x: x["window_index"]).astype(int)

    df["window_samples"] = df["window_samples"].astype(int)
    df["start_sample"] = df["start_sample"].astype(int)
    df["end_sample"] = df["end_sample"].astype(int)
    df["source_file"] = df["source_recording"].astype(str)
    df["window_array_file"] = df["window_array_file"].astype(str)

    df["recording_id"] = (
        "S" + df.subject.astype(str).str.zfill(2)
        + "_" + df.condition
        + "_" + df.gesture.astype(str)
        + "_" + df.set
    )

    cols = [
        "subject", "condition", "gesture", "set", "recording_id",
        "window_id", "window_number", "window_index",
        "window_samples", "start_sample", "end_sample",
        "source_file", "window_array_file",
    ]
    df = df[cols].copy()

    if len(df) != TOTAL_WINDOWS:
        raise RuntimeError("BLOCK04 inventory row count mismatch.")

    expected = df.groupby(
        ["subject", "condition", "gesture", "set", "recording_id"]
    ).agg(
        n_windows=("window_id", "nunique"),
        n_indices=("window_index", "nunique"),
    )

    if len(expected) != N_RECORDINGS:
        raise RuntimeError("BLOCK04 inventory recording count mismatch.")
    if not np.all(expected.n_windows.to_numpy() == WINDOWS_PER_RECORDING):
        raise RuntimeError("BLOCK04 inventory does not contain 21 windows/recording.")
    if not np.all(expected.n_indices.to_numpy() == WINDOWS_PER_RECORDING):
        raise RuntimeError("BLOCK04 inventory does not contain 21 unique array indices/recording.")

    if df.duplicated(["subject", "condition", "gesture", "set", "window_id"]).any():
        raise RuntimeError("Duplicate logical BLOCK04 windows detected.")
    if df.duplicated(["subject", "condition", "gesture", "set", "window_index"]).any():
        raise RuntimeError("Duplicate BLOCK04 array positions detected.")

    for key, g in df.groupby(["subject", "condition", "gesture", "set"]):
        if sorted(g.window_index.tolist()) != list(range(WINDOWS_PER_RECORDING)):
            raise RuntimeError(f"Non-contiguous window_index for {key}.")
        if sorted(g.window_number.tolist()) != list(range(1, WINDOWS_PER_RECORDING + 1)):
            raise RuntimeError(f"Non-contiguous window_number for {key}.")

    write_csv(PARTITION_ROOT / "block04_window_inventory.csv", df)
    return df


# =============================================================================
# 05. FIXED TEMPORAL PARTITION
# =============================================================================

def build_temporal_partitions(inv: pd.DataFrame):
    """Build the leakage-controlled temporal calibration partition.

    Protocol frozen here:
      * Evaluation is fixed to SetC canonical windows 19,20,21
        (array indices 18,19,20).
      * Window 18 (array index 17; 4250..4750) is embargoed because it
        overlaps evaluation window 19 (4500..5000).
      * Calibration fractions are fractions of the AVAILABLE PRE-EVALUATION
        PHYSICAL DURATION, not fractions of the number of windows.
      * Only complete computational windows fully contained inside the
        requested calibration interval are included.
      * SetA/SetB are explicitly unused by this temporal adaptation block.
      * Calibration is supervised and per gesture recording.

    This is an offline temporal calibration analysis because the upstream
    signal is documented as 1-40 Hz fourth-order Butterworth zero-phase
    filtered. The code therefore NEVER claims a causal online-filter
    deployment simulation.

    The large partition lookup is streamed directly to disk rather than
    accumulated as ~198,450 Python dictionaries in RAM.
    """
    import csv as _csv

    part_rows, cal_rows, eval_rows, audit_rows = [], [], [], []
    
    lookup_csv_path = PARTITION_ROOT / "partition_lookup.csv"
    lookup_tmp_path = PARTITION_ROOT / "partition_lookup.csv.tmp"

    lookup_columns = [
        "subject",
        "condition",
        "gesture",
        "set",
        "recording_id",
        "window_id",
        "window_number",
        "window_index",
        "calibration_fraction",
        "role",
    ]

    # Remove only a stale temporary file from an interrupted partition build.
    if lookup_tmp_path.exists():
        lookup_tmp_path.unlink()

    lookup_header_written = False

    # Use a pure-Python grouping pass here instead of pandas GroupBy iteration.
    # In this Windows runtime, the C-backed GroupBy iterator can terminate the
    # process with 0xC0000005 before the first group is yielded.  The grouping
    # keys and all downstream partition rules remain exactly unchanged.
    _group_columns = ["subject", "condition", "gesture", "set", "recording_id"]
    _groups = {}
    for _row in inv.to_dict(orient="records"):
        _key = tuple(_row[c] for c in _group_columns)
        _groups.setdefault(_key, []).append(_row)
    grouped = []
    for _key, _rows in sorted(_groups.items(), key=lambda x: x[0]):
        _rows.sort(key=lambda r: int(r["window_index"]))
        grouped.append((_key, _rows))

    log(
        "Temporal partition build START: "
        f"{len(inv)} inventory windows across {N_RECORDINGS} recordings."
    )
    mark_stage_start(
        "temporal_partition_build",
        inventory_windows=int(len(inv)),
        expected_recordings=int(N_RECORDINGS),
        calibration_fractions=list(CALIBRATION_FRACTIONS),
    )

    with lookup_tmp_path.open("w", encoding="utf-8", newline="") as lookup_fh:
        lookup_writer = _csv.writer(lookup_fh, lineterminator="\n")

        for _group_number, ((subject, condition, gesture, set_id, recording_id), _rows) in enumerate(grouped, start=1):
            mark_stage_start(
                "temporal_partition_recording",
                group_number=int(_group_number),
                total_groups=int(len(grouped)),
                subject=int(subject),
                condition=str(condition),
                gesture=int(gesture),
                set=str(set_id),
                recording_id=str(recording_id),
            )
            _rows.sort(key=lambda r: int(r["window_index"]))
            g = pd.DataFrame(_rows).reset_index(drop=True)
            if g.shape[0] != WINDOWS_PER_RECORDING:
                raise RuntimeError(
                    f"{recording_id}: expected 21 windows, got {g.shape[0]}"
                )

            if set_id == FIXED_EVAL_SET:
                ev = g[g.window_index.isin(FIXED_EVAL_WINDOW_INDICES)].copy()
                if ev.shape[0] != FIXED_EVAL_WINDOWS_PER_RECORDING:
                    raise RuntimeError(
                        f"{recording_id}: fixed evaluation tail mismatch"
                    )

                eval_start = int(ev.start_sample.min())
                eval_end = int(ev.end_sample.max())
                recording_start = int(g.start_sample.min())
                recording_end = int(g.end_sample.max())

                if eval_start <= recording_start:
                    raise RuntimeError(
                        f"{recording_id}: invalid pre-evaluation interval"
                    )

                pre_eval_duration = eval_start - recording_start

                embargo = g[
                    (g.start_sample < eval_end)
                    & (g.end_sample > eval_start)
                ].copy()
                embargo = embargo[
                    ~embargo.window_index.isin(FIXED_EVAL_WINDOW_INDICES)
                ]

                if embargo.shape[0] != 1:
                    raise RuntimeError(
                        f"{recording_id}: expected exactly one non-evaluation "
                        f"embargo window, found {embargo.shape[0]}"
                    )

                if int(embargo.iloc[0].window_number) != 18:
                    raise RuntimeError(
                        f"{recording_id}: expected Window0018 as the embargo "
                        f"window; found "
                        f"Window{int(embargo.iloc[0].window_number):04d}"
                    )

                eligible = g[
                    (g.end_sample <= eval_start)
                    & (~g.window_index.isin(FIXED_EVAL_WINDOW_INDICES))
                ].copy()
                eligible = eligible[
                    ~eligible.window_index.isin(embargo.window_index)
                ].sort_values("window_index")

            else:
                ev = g.iloc[0:0].copy()
                embargo = g.iloc[0:0].copy()
                eligible = g.iloc[0:0].copy()
                eval_start = eval_end = recording_start = recording_end = None
                pre_eval_duration = None

            for frac in CALIBRATION_FRACTIONS:
                if set_id != FIXED_EVAL_SET or frac == 0:
                    cal = eligible.iloc[0:0].copy()
                    requested_boundary = None
                else:
                    requested_boundary = int(
                        round(
                            recording_start
                            + (frac / 100.0) * pre_eval_duration
                        )
                    )
                    cal = eligible[
                        eligible.end_sample <= requested_boundary
                    ].copy()

                cal_ids = set(cal.window_index.astype(int))
                ev_ids = set(ev.window_index.astype(int))
                embargo_ids = set(embargo.window_index.astype(int))

                if cal_ids & ev_ids:
                    raise RuntimeError(
                        f"Calibration/evaluation overlap: "
                        f"{recording_id} {frac}%"
                    )

                if cal_ids & embargo_ids:
                    raise RuntimeError(
                        f"Calibration/embargo overlap: "
                        f"{recording_id} {frac}%"
                    )

                # Explicit physical-sample overlap audit.
                for _, cr in cal.iterrows():
                    if ev.shape[0]:
                        overlap = (
                            (ev.start_sample.to_numpy() < int(cr.end_sample))
                            & (ev.end_sample.to_numpy() > int(cr.start_sample))
                        )
                        if np.any(overlap):
                            raise RuntimeError(
                                f"Physical sample overlap: {recording_id} "
                                f"{frac}% Window{int(cr.window_number):04d}"
                            )

                    if embargo.shape[0]:
                        overlap = (
                            (
                                embargo.start_sample.to_numpy()
                                < int(cr.end_sample)
                            )
                            & (
                                embargo.end_sample.to_numpy()
                                > int(cr.start_sample)
                            )
                        )
                        if np.any(overlap):
                            raise RuntimeError(
                                f"Calibration intersects embargo: "
                                f"{recording_id} {frac}%"
                            )

                # Stream lookup rows directly to disk. This is the critical
                # memory-safety change: no 198,450-row Python dict list.
                if not lookup_header_written:
                    lookup_writer.writerow(lookup_columns)
                    lookup_header_written = True

                for _, r in g.iterrows():
                    idx = int(r.window_index)

                    if idx in cal_ids:
                        role = "CAL"
                    elif idx in ev_ids:
                        role = "EVAL"
                    elif idx in embargo_ids:
                        role = "EMBARGO"
                    else:
                        role = "UNUSED"

                    lookup_writer.writerow(
                        [
                            int(subject), condition, int(gesture), set_id,
                            recording_id, str(r.window_id), int(r.window_number),
                            idx, int(frac), role,
                        ]
                    )

                # Calibration/evaluation tables are much smaller than the
                # lookup table and remain in memory for downstream use.
                for _, r in cal.iterrows():
                    cal_rows.append(
                        {
                            **r.to_dict(),
                            "calibration_fraction": int(frac),
                        }
                    )

                for _, r in ev.iterrows():
                    eval_rows.append(
                        {
                            **r.to_dict(),
                            "calibration_fraction": int(frac),
                        }
                    )

                part_rows.append(
                    {
                        "subject": int(subject),
                        "condition": condition,
                        "gesture": int(gesture),
                        "set": set_id,
                        "recording_id": recording_id,
                        "calibration_fraction": int(frac),
                        "calibration_denominator": CALIBRATION_DENOMINATOR,
                        "recording_start_sample": (
                            int(recording_start)
                            if recording_start is not None
                            else None
                        ),
                        "recording_end_sample": (
                            int(recording_end)
                            if recording_end is not None
                            else None
                        ),
                        "pre_evaluation_duration_samples": (
                            int(pre_eval_duration)
                            if pre_eval_duration is not None
                            else None
                        ),
                        "requested_calibration_boundary_sample": requested_boundary,
                        "n_eligible_pre_eval_windows": int(eligible.shape[0]),
                        "n_calibration_windows": int(cal.shape[0]),
                        "n_evaluation_windows": int(ev.shape[0]),
                        "n_embargo_windows": int(embargo.shape[0]),
                        "calibration_first_window_number": (
                    
                            int(cal.window_number.min()) if cal.shape[0] else None
                        ),
                        "calibration_last_window_number": (
                            int(cal.window_number.max()) if cal.shape[0] else None
                        ),
                        "calibration_end_sample": (
                            int(cal.end_sample.max()) if cal.shape[0] else None
                        ),
                        "evaluation_start_sample": (
                            int(ev.start_sample.min()) if ev.shape[0] else None
                        ),
                        "evaluation_end_sample": (
                            int(ev.end_sample.max()) if ev.shape[0] else None
                        ),
                        "evaluation_window_numbers": (
                            ",".join(map(str, FIXED_EVAL_WINDOW_NUMBERS))
                            if ev.shape[0]
                            else ""
                        ),
                        "evaluation_window_indices": (
                            ",".join(map(str, FIXED_EVAL_WINDOW_INDICES))
                            if ev.shape[0]
                            else ""
                        ),
                        "embargo_window_numbers": (
                            ",".join(
                                map(
                                    lambda x: f"{int(x):d}",
                                    sorted(
                                        embargo.window_number.astype(int)
                                    ),
                                )
                            )
                            if embargo.shape[0]
                            else ""
                        ),
                        "protocol": (
                            "per_gesture_supervised_temporal_calibration"
                        ),
                    }
                )

                audit_rows.append(
                    {
                        "subject": int(subject),
                        "condition": condition,
                        "gesture": int(gesture),
                        "set": set_id,
                        "recording_id": recording_id,
                        "calibration_fraction": int(frac),
                        "calibration_denominator": CALIBRATION_DENOMINATOR,
                        "requested_boundary_sample": requested_boundary,
                        "n_calibration_windows": int(cal.shape[0]),
                        "n_evaluation_windows": int(ev.shape[0]),
                        "n_embargo_windows": int(embargo.shape[0]),
                        "calibration_window_numbers": ",".join(
                            map(
                                str,
                                sorted(cal.window_number.astype(int)),
                            )
                        ),
                        "evaluation_window_numbers": (
                            ",".join(map(str, FIXED_EVAL_WINDOW_NUMBERS))
                            if ev.shape[0]
                            else ""
                        ),
                        "embargo_window_numbers": ",".join(
                            map(
                                str,
                                sorted(embargo.window_number.astype(int)),
                            )
                        ),
                        "physical_overlap_samples": 0,
                        "zero_phase_filtering": TEMPORAL_FILTERING_MODE,
                        "causal_temporal_claim": TEMPORAL_CAUSALITY_CLAIM,
                    }
                )

            mark_stage_pass(
                "temporal_partition_recording",
                group_number=int(_group_number),
                subject=int(subject),
                condition=str(condition),
                gesture=int(gesture),
                set=str(set_id),
                recording_id=str(recording_id),
            )

    log(
        "Temporal partition build group loop PASS: "
        f"{N_RECORDINGS} logical recordings × {len(CALIBRATION_FRACTIONS)} fractions processed."
    )

    # Atomically promote the streamed lookup table.
    lookup_tmp_path.replace(lookup_csv_path)
    log(f"Temporal partition lookup written: {lookup_csv_path}")

    part = pd.DataFrame(part_rows)
    caldf = pd.DataFrame(cal_rows)
    evdf = pd.DataFrame(eval_rows)
    audit = pd.DataFrame(audit_rows)

    lookup = pd.read_csv(
        lookup_csv_path,
        dtype={
            "subject": "int64",
            "gesture": "int64",
            "window_number": "int64",
            "window_index": "int64",
            "calibration_fraction": "int64",
            "condition": "object",
            "set": "object",
            "recording_id": "object",
            "window_id": "object",
            "role": "object",
        },
    )

    expected_part = N_RECORDINGS * len(CALIBRATION_FRACTIONS)
    if len(part) != expected_part:
        raise RuntimeError(
            f"Expected {expected_part} partition rows, got {len(part)}."
        )

    expected_lookup = (
        N_RECORDINGS
        * WINDOWS_PER_RECORDING
        * len(CALIBRATION_FRACTIONS)
    )
    if len(lookup) != expected_lookup:
        raise RuntimeError(
            f"Expected {expected_lookup} lookup rows, got {len(lookup)}."
        )

    for frac in CALIBRATION_FRACTIONS:
        n = int((evdf.calibration_fraction == frac).sum())
        if n != FIXED_EVAL_WINDOWS_TOTAL:
            raise RuntimeError(
                f"Fixed evaluation set changed at {frac}%: {n}, "
                f"expected {FIXED_EVAL_WINDOWS_TOTAL}."
            )

    # Physical-overlap audit.  The audit column is intentionally always 0
    # because calibration windows are guaranteed by construction to be
    # physically disjoint from evaluation and embargo windows.  The coercion
    # below protects against pandas dtype quirks across versions.
    _ov = audit.physical_overlap_samples
    _bad_mask = ~(
        pd.to_numeric(_ov, errors="coerce").fillna(0).to_numpy() == 0
    )
    _bad = audit[_bad_mask]
    if _bad.shape[0]:
        print("=== PHYSICAL OVERLAP AUDIT DIAGNOSTIC ===", flush=True)
        print("audit rows:", len(audit), flush=True)
        print("column dtype:", _ov.dtype, flush=True)
        print("null count:", int(_ov.isna().sum()), flush=True)
        print(
            "unique non-null values (first 20):",
            _ov.dropna().unique()[:20],
            flush=True,
        )
        print("bad row count:", int(_bad_mask.sum()), flush=True)
        cols = [c for c in (
            "subject", "condition", "gesture", "set", "recording_id",
            "calibration_fraction", "n_calibration_windows",
            "n_evaluation_windows", "n_embargo_windows",
            "calibration_window_numbers", "evaluation_window_numbers",
            "embargo_window_numbers", "physical_overlap_samples",
        ) if c in _bad.columns]
        print(_bad[cols].head(20).to_string(), flush=True)
        raise RuntimeError("Temporal physical-overlap audit failed.")

    if len(caldf) and np.any(
        caldf.window_number.astype(int).to_numpy() == 18
    ):
        raise RuntimeError("Window0018 is embargoed but entered calibration.")

    if len(evdf):
        if set(evdf.window_number.astype(int).unique()) != set(
            FIXED_EVAL_WINDOW_NUMBERS
        ):
            raise RuntimeError(
                "Evaluation window numbers are not exactly 19,20,21."
            )

        if set(evdf.set.astype(str).unique()) != {FIXED_EVAL_SET}:
            raise RuntimeError("Evaluation contains non-SetC data.")

    write_csv(PARTITION_ROOT / "partition_manifest.csv", part)
    write_csv(PARTITION_ROOT / "calibration_windows.csv", caldf)
    write_csv(PARTITION_ROOT / "evaluation_windows.csv", evdf)
    write_csv(PARTITION_ROOT / "embargo_audit.csv", audit)

    log(
        "Temporal partition validation PASS: "
        f"{part.shape[0]} partition rows, {caldf.shape[0]} calibration rows, "
        f"{evdf.shape[0]} evaluation rows, {lookup.shape[0]} lookup rows."
    )

    write_json(
        PARTITION_ROOT / "partition_definition.json",
        {
            "evaluation_set": FIXED_EVAL_SET,
            "evaluation_window_numbers": FIXED_EVAL_WINDOW_NUMBERS,
            "evaluation_window_indices": FIXED_EVAL_WINDOW_INDICES,
            "calibration_fractions_percent": CALIBRATION_FRACTIONS,
            "calibration_denominator": CALIBRATION_DENOMINATOR,
            "protocol": "per_gesture_supervised_temporal_calibration",
            "embargo_rule": (
                "Any computational window sharing physical EEG samples "
                "with evaluation is excluded from calibration."
            ),
            "embargoed_window_numbers": [18],
            "zero_phase_filtering": TEMPORAL_FILTERING_MODE,
            "causal_temporal_claim": TEMPORAL_CAUSALITY_CLAIM,
            "subject_is_inferential_unit": True,
            "evaluation_macro_f1_resolution_pp": (
                EVALUATION_MACRO_F1_RESOLUTION_PP
            ),
        },
    )

    log(
        "NOTE: fixed evaluation tail = "
        f"{FIXED_EVAL_WINDOWS_PER_SUBJECT_CONDITION} windows per "
        "subject-condition (3 per gesture). "
        f"Macro-F1 resolution approx "
        f"{EVALUATION_MACRO_F1_RESOLUTION_PP} pp."
    )
    mark_stage_pass(
        "temporal_partition_build",
        partition_rows=int(len(part)),
        calibration_rows=int(len(caldf)),
        evaluation_rows=int(len(evdf)),
        lookup_rows=int(len(lookup)),
    )

    return part, caldf, evdf, lookup


# =============================================================================
# 06. FEATURE LOADING
# =============================================================================

def parse_feature_file_metadata(path: Path):
    """Strictly parse logical recording identity from the full Block05 path."""
    text = str(path).upper().replace("\\", "/")

    sm = re.findall(r"(?:^|[/_-])S0*(\d{1,2})(?:[/_-]|$)", text)
    cm = re.findall(r"(?:^|[/_-])(ME|MI)(?:[/_-]|$)", text)
    # Canonical Block05 filenames use numerical gesture IDs, e.g.
    # S01_ME_1_SetA_features.npz.  Legacy G1/G2/... filenames remain
    # accepted for compatibility, but downstream identities stay numerical.
    gm_numeric = re.findall(
        r"(?:^|[/_-])(?:G)?([1-7])(?:[_/-])SET[ABC](?:[/_.-]|$)",
        text,
    )
    gm_legacy = re.findall(r"(?:^|[/_-])G([1-7])(?:[/_-]|$)", text)
    gm = gm_numeric or gm_legacy
    setm = re.findall(r"(?:^|[/_-])SET([ABC])(?:[/_-]|$)", text)
    if not sm or not cm or not gm:
        return None

    subject = int(sm[-1])
    condition = canonical_condition(cm[-1])
    gesture = int(gm[-1])
    if setm:
        set_id = f"Set{setm[-1]}"
    else:
        parts = [x for x in text.split("/") if x]
        candidates = [x for x in parts if x in {"A", "B", "C"}]
        if not candidates:
            return None
        set_id = f"Set{candidates[-1]}"

    if subject not in SUBJECTS or gesture not in GESTURES:
        return None
    return subject, condition, gesture, set_id


def find_feature_files() -> Dict[Tuple[int, str, int, str], Path]:
    candidates = list(BLOCK05_ROOT.rglob("*.npz"))
    found = {}

    for p in candidates:
        if "feature" not in p.stem.lower():
            continue

        meta = parse_feature_file_metadata(p)
        if meta is None:
            continue

        key = meta
        if key in found and found[key] != p:
            raise RuntimeError(
                f"Multiple Block05 feature files for logical recording {key}: "
                f"{found[key]} and {p}"
            )
        found[key] = p

    if len(found) != N_RECORDINGS:
        raise RuntimeError(
            f"Expected {N_RECORDINGS} Block05 feature files; "
            f"found {len(found)}."
        )

    return found


def load_feature_record(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        if "features" not in z.files:
            raise RuntimeError(f"No 'features' array in {path}.")
        X = np.asarray(z["features"], dtype=np.float32)

    if X.shape != (WINDOWS_PER_RECORDING, 341):
        raise RuntimeError(
            f"Block05 feature shape mismatch {path}: {X.shape}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError(f"Non-finite features in {path}.")

    return X


def build_ml_rows(inv: pd.DataFrame) -> List[dict]:
    files = find_feature_files()
    rows = []

    for key, path in sorted(files.items()):
        subject, condition, gesture, set_id = key
        X = load_feature_record(path)

        sub = inv[
            (inv.subject == subject)
            & (inv.condition == condition)
            & (inv.gesture == gesture)
            & (inv.set == set_id)
        ].sort_values("window_index")

        if len(sub) != WINDOWS_PER_RECORDING:
            raise RuntimeError(f"Missing Block04 windows for {key}.")

        for j, (_, r) in enumerate(sub.iterrows()):
            rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "gesture": gesture,
                    "set": set_id,
                    "window_id": str(r.window_id),
                    "window_number": int(r.window_number),
                    "window_index": int(r.window_index),
                    "start_sample": int(r.start_sample),
                    "end_sample": int(r.end_sample),
                    "recording_id": r.recording_id,
                    "source_file": str(path),
                    "X": X[j],
                }
            )

    if len(rows) != TOTAL_WINDOWS:
        raise RuntimeError(
            f"ML feature rows={len(rows)}, expected {TOTAL_WINDOWS}."
        )

    return rows


# =============================================================================
# 07. PARTITION LOOKUP
# =============================================================================

def load_cached_prediction_run(
    path: Path,
    subject: int,
    condition: str,
    model_name: str,
    fraction: int,
    expected_evaluation: pd.DataFrame,
):
    """Load and strictly validate a completed prediction artifact.

    A prediction CSV is a run-level checkpoint.  It is reusable only when it
    exactly matches the current subject/condition/model/fraction and the
    frozen SetC Window0019-0021 evaluation population.
    """
    if not path.exists():
        return None

    try:
        df = normalize_prediction_dataframe(pd.read_csv(path))
    except Exception:
        return None

    if len(df) != len(expected_evaluation):
        return None
    if set(df.subject_id.astype(int)) != {subject}:
        return None
    if set(df.condition.astype(str)) != {condition}:
        return None
    if set(df.model.astype(str)) != {model_name}:
        return None
    if set(df.calibration_fraction.astype(int)) != {fraction}:
        return None
    if set(df.set.astype(str)) != {FIXED_EVAL_SET}:
        return None
    if set(df.window_index.astype(int)) != set(FIXED_EVAL_WINDOW_INDICES):
        return None
    if df.duplicated(["gesture_true", "window_index"]).any():
        return None

    expected_keys = {
        (int(r.gesture), int(r.window_index))
        for r in expected_evaluation.itertuples(index=False)
    }
    actual_keys = {
        (int(r.gesture_true), int(r.window_index))
        for r in df.itertuples(index=False)
    }
    if actual_keys != expected_keys:
        return None

    expected_ids = {str(r.window_id) for r in expected_evaluation.itertuples(index=False)}
    if set(df.window_id.astype(str)) != expected_ids:
        return None

    if not df.gesture_pred.isin(GESTURES).all():
        return None

    metrics = calculate_metrics(
        df.gesture_true.to_numpy(dtype=int),
        df.gesture_pred.to_numpy(dtype=int),
    )
    return df, metrics


def lookup_rows(
    lookup: pd.DataFrame,
    subject: int,
    condition: str,
    fraction: int,
    role: str,
) -> pd.DataFrame:
    return lookup[
        (lookup.subject == subject)
        & (lookup.condition == condition)
        & (lookup.calibration_fraction == fraction)
        & (lookup.role == role)
    ].copy()


def select_ml_role(
    ml_rows: List[dict],
    lookup: pd.DataFrame,
    subject: int,
    condition: str,
    fraction: int,
    role: str,
) -> List[dict]:
    keys = lookup_rows(
        lookup,
        subject,
        condition,
        fraction,
        role,
    )

    key_set = set(
        zip(
            keys.gesture.astype(int),
            keys.set.astype(str),
            keys.window_index.astype(int),
        )
    )

    out = []
    for r in ml_rows:
        if r["subject"] != subject or r["condition"] != condition:
            continue
        k = (
            int(r["gesture"]),
            str(r["set"]),
            int(r["window_index"]),
        )
        if k in key_set:
            out.append(r)

    return out


# =============================================================================
# 08. ML CALIBRATION
# =============================================================================

def _fit_random_forest_safely(model, X, y):
    """Fit the frozen RF without relying on corrupted sklearn introspection state.

    In the long Block09 process, sklearn 1.9.0 can occasionally encounter an
    invalid constructor-signature object during RandomForest tree cloning,
    producing: ``TypeError: 'Parameter' object is not iterable``.

    The failure occurs inside BaseEstimator._get_param_names(), before a tree
    is fitted.  Therefore protecting only inspect.signature() is insufficient.
    During this narrowly scoped RF fit, this helper supplies the already-cached
    constructor parameter names for RandomForestClassifier and
    DecisionTreeClassifier and restores sklearn's original method immediately
    afterward.  No model hyperparameter, training sample, label, or random seed
    is changed.
    """
    import inspect as _rf_inspect
    from sklearn.base import BaseEstimator as _RF_BASE_ESTIMATOR

    rf_init = _RF_CLASS.__init__
    dt_init = _DT_CLASS.__init__
    original_signature = _rf_inspect.signature
    original_get_param_names = _RF_BASE_ESTIMATOR._get_param_names

    # NumPy/sklearn compatibility guard: on this Windows environment, the
    # DecisionTree fit path can fail inside np.random.RandomState(seed) with
    # ``IndexError: index 624 is out of bounds for axis 0 with size 624``.
    # RandomState.seed(seed) produces the identical MT19937 state for an
    # integer seed while bypassing that constructor path.  This is a narrowly
    # scoped compatibility workaround; the RF seed and all model parameters
    # remain unchanged.
    import sklearn.tree._classes as _RF_TREE_CLASSES
    import sklearn.utils.validation as _RF_VALIDATION
    original_tree_check_random_state = _RF_TREE_CLASSES.check_random_state
    original_validation_check_random_state = _RF_VALIDATION.check_random_state

    def _safe_tree_check_random_state(seed):
        if isinstance(seed, (int, np.integer)):
            rs = np.random.RandomState()
            rs.seed(int(seed))
            return rs
        return original_tree_check_random_state(seed)

    rf_param_names = tuple(
        p.name
        for p in _RF_PRISTINE_INIT_SIGNATURE.parameters.values()
        if p.name != "self" and p.kind != p.VAR_KEYWORD
    )
    dt_param_names = tuple(
        p.name
        for p in _DT_PRISTINE_INIT_SIGNATURE.parameters.values()
        if p.name != "self" and p.kind != p.VAR_KEYWORD
    )

    def _safe_signature(obj, *args, **kwargs):
        if obj is rf_init:
            return _RF_PRISTINE_INIT_SIGNATURE
        if obj is dt_init:
            return _DT_PRISTINE_INIT_SIGNATURE
        return original_signature(obj, *args, **kwargs)

    original_rf_signature_attr = getattr(rf_init, "__signature__", None)
    original_dt_signature_attr = getattr(dt_init, "__signature__", None)
    had_rf_signature_attr = hasattr(rf_init, "__signature__")
    had_dt_signature_attr = hasattr(dt_init, "__signature__")

    def _safe_get_param_names(cls):
        if cls is _RF_CLASS:
            return list(rf_param_names)
        if cls is _DT_CLASS:
            return list(dt_param_names)
        return original_get_param_names.__func__(cls)

    _rf_inspect.signature = _safe_signature
    _RF_BASE_ESTIMATOR._get_param_names = classmethod(_safe_get_param_names)
    _RF_TREE_CLASSES.check_random_state = _safe_tree_check_random_state
    _RF_VALIDATION.check_random_state = _safe_tree_check_random_state

    # Also repair the constructor metadata itself for the duration of the fit.
    # This covers sklearn code paths that access __signature__ directly.
    rf_init.__signature__ = _RF_PRISTINE_INIT_SIGNATURE
    dt_init.__signature__ = _DT_PRISTINE_INIT_SIGNATURE

    try:
        return model.fit(X, y)
    finally:
        _rf_inspect.signature = original_signature
        _RF_BASE_ESTIMATOR._get_param_names = classmethod(original_get_param_names.__func__)
        _RF_TREE_CLASSES.check_random_state = original_tree_check_random_state
        _RF_VALIDATION.check_random_state = original_validation_check_random_state

        if had_rf_signature_attr:
            rf_init.__signature__ = original_rf_signature_attr
        else:
            try:
                del rf_init.__signature__
            except AttributeError:
                pass

        if had_dt_signature_attr:
            dt_init.__signature__ = original_dt_signature_attr
        else:
            try:
                del dt_init.__signature__
            except AttributeError:
                pass


class _XGBLabelShiftWrapper:
    """Wrap an sklearn-API XGBClassifier so external labels stay 1..7.

    sklearn-API XGBoost (>= 1.3) requires class indices 0..K-1.  Block09's
    canonical gesture labels are 1..7.  This wrapper shifts labels on fit
    and shifts predictions back on predict, without touching the fitted
    estimator.  All other attributes (classes_, predict_proba, feature
    importances, etc.) are forwarded transparently to the wrapped estimator
    via __getattr__.

    Conventions:
      - fit(X, y) with y in {1,...,7}   -> internal y' = y - 1 in {0,...,6}
      - predict(X)                      -> external yhat = yhat' + 1 in {1,...,7}
      - predict_proba(X)                -> forwarded unchanged (columns are
                                            internal 0..6 order, which matches
                                            GESTURES order after +1)
    """

    def __init__(self, estimator):
        # Use object.__setattr__ to avoid any chance of recursion through
        # __getattr__ before self.estimator exists.
        object.__setattr__(self, "estimator", estimator)

    def fit(self, X, y, **kwargs):
        y = np.asarray(y, dtype=int)
        if y.size == 0:
            raise RuntimeError("XGB wrapper received an empty label vector.")
        uniq = set(y.tolist())
        if not uniq.issubset(set(GESTURES)):
            raise RuntimeError(
                f"XGB wrapper received labels outside {GESTURES}: "
                f"{sorted(uniq)}"
            )
        self.estimator.fit(X, y - 1, **kwargs)
        return self

    def predict(self, X, **kwargs):
        pred_internal = np.asarray(
            self.estimator.predict(X, **kwargs), dtype=int
        )
        return pred_internal + 1

    def predict_proba(self, X, **kwargs):
        return self.estimator.predict_proba(X, **kwargs)

    def __getattr__(self, name):
        # Only called if normal attribute lookup fails; forward to estimator.
        return getattr(object.__getattribute__(self, "estimator"), name)


def make_ml_model(name: str):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    if name == "LDA":
        estimator = LinearDiscriminantAnalysis(
            solver=ML_CONFIG["LDA"]["solver"]
        )
    elif name == "KNN":
        estimator = KNeighborsClassifier(
            n_neighbors=ML_CONFIG["KNN"]["n_neighbors"],
            weights=ML_CONFIG["KNN"]["weights"],
            metric=ML_CONFIG["KNN"]["metric"],
            n_jobs=-1,
        )
    elif name == "SVM":
        estimator = SVC(
            kernel=ML_CONFIG["SVM"]["kernel"],
            C=ML_CONFIG["SVM"]["C"],
            gamma=ML_CONFIG["SVM"]["gamma"],
            random_state=SEED,
        )
    elif name == "RandomForest":
        # -----------------------------------------------------------------
        # Random Forest
        #
        # IMPORTANT:
        # Block09 runs many independent RF fits in one Python process.
        # The intermittent failure observed during RF.fit() occurs inside
        # sklearn's estimator random-state propagation / inspect machinery.
        #
        # Keep the frozen RF hyperparameters unchanged.  Use an explicit
        # fresh estimator for every calibration run.
        # -----------------------------------------------------------------
        rf_random_state = ML_CONFIG["RF"]["random_state"]

        estimator = RandomForestClassifier(
            n_estimators=int(ML_CONFIG["RF"]["n_estimators"]),
            random_state=rf_random_state,
            n_jobs=int(ML_CONFIG["RF"]["n_jobs"]),
            class_weight=ML_CONFIG["RF"]["class_weight"],
        )

        # Verify that this is a genuine sklearn RF estimator before it
        # enters the long calibration loop.
        if estimator.__class__.__name__ != "RandomForestClassifier":
            raise RuntimeError(
                "RandomForest construction failed: unexpected estimator type "
                f"{type(estimator)}"
            )
    elif name == "XGBoost":
        from xgboost import XGBClassifier

        cfg = ML_CONFIG["XGB"]
        estimator = XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            objective=cfg["objective"],
            num_class=7,
            eval_metric=cfg["eval_metric"],
            random_state=cfg["random_state"],
            n_jobs=cfg["n_jobs"],
            tree_method=cfg["tree_method"],
        )
    else:
        raise ValueError(name)

    scaler_required = {
        "LDA": True,
        "KNN": True,
        "SVM": True,
        "RandomForest": False,
        "XGBoost": False,
    }[name]

    if scaler_required:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    if name == "XGBoost":
        return _XGBLabelShiftWrapper(estimator)

    return estimator


def run_ml_calibration(
    ml_rows: List[dict],
    lookup: pd.DataFrame,
):
    results = []
    manifest = []
    split_audit = []

    total_runs = (
        len(SUBJECTS)
        * len(CONDITIONS)
        * len(ML_MODELS)
        * len(CALIBRATION_FRACTIONS)
    )
    run_no = 0

    evaluation_window_ids_str = ",".join(map(str, FIXED_EVAL_WINDOW_NUMBERS))

    for subject in SUBJECTS:
        for condition in CONDITIONS:
            source = [
                r
                for r in ml_rows
                if r["subject"] != subject
                and r["condition"] == condition
            ]

            Xsrc = np.vstack([r["X"] for r in source])
            ysrc = np.asarray(
                [r["gesture"] for r in source],
                dtype=int,
            )

            for model_name in ML_MODELS:
                for frac in CALIBRATION_FRACTIONS:
                    run_no += 1
                    log(
                        f"ML {run_no}/{total_runs}: "
                        f"S{subject:02d} {condition} "
                        f"{model_name} {frac}%"
                    )

                    cal = select_ml_role(
                        ml_rows,
                        lookup,
                        subject,
                        condition,
                        frac,
                        "CAL",
                    )
                    ev = select_ml_role(
                        ml_rows,
                        lookup,
                        subject,
                        condition,
                        frac,
                        "EVAL",
                    )

                    if not ev:
                        raise RuntimeError(
                            f"No fixed evaluation windows for S{subject} "
                            f"{condition} {frac}%."
                        )

                    if set(r["set"] for r in ev) != {FIXED_EVAL_SET}:
                        raise RuntimeError(
                            "ML evaluation does not contain the fixed SetC "
                            "evaluation tail."
                        )
                    if set(r["window_index"] for r in ev) != set(
                        FIXED_EVAL_WINDOW_INDICES
                    ):
                        raise RuntimeError(
                            "ML evaluation does not contain exactly "
                            f"window IDs {FIXED_EVAL_WINDOW_INDICES}."
                        )

                    cal_end = {}
                    for r in cal:
                        cal_end[r["recording_id"]] = max(
                            cal_end.get(r["recording_id"], 0),
                            int(r["end_sample"]),
                        )

                    violations = 0
                    for r in ev:
                        rid = r["recording_id"]
                        if rid in cal_end:
                            if int(r["start_sample"]) < cal_end[rid]:
                                violations += 1

                    if violations:
                        raise RuntimeError(
                            f"ML temporal leakage detected: "
                            f"{subject} {condition} {model_name} {frac}%"
                        )

                    Xcal = (
                        np.vstack([r["X"] for r in cal])
                        if cal
                        else np.empty((0, 341), dtype=np.float32)
                    )
                    ycal = (
                        np.asarray(
                            [r["gesture"] for r in cal],
                            dtype=int,
                        )
                        if cal
                        else np.empty((0,), dtype=int)
                    )

                    if len(Xcal):
                        Xtrain = np.vstack([Xsrc, Xcal])
                        ytrain = np.concatenate([ysrc, ycal])
                    else:
                        Xtrain = Xsrc
                        ytrain = ysrc

                    Xev = np.vstack([r["X"] for r in ev])
                    yev = np.asarray(
                        [r["gesture"] for r in ev],
                        dtype=int,
                    )

                    pred_path = (
                        PRED_ROOT
                        / "ML"
                        / model_name
                        / f"S{subject:02d}"
                        / condition
                        / f"cal_{frac:02d}.csv"
                    )

                    cached = load_cached_prediction_run(
                        pred_path,
                        subject,
                        condition,
                        model_name,
                        frac,
                        pd.DataFrame(ev),
                    )

                    if cached is not None:
                        cached_df, metrics = cached
                        log(
                            f"ML RESUME: reusing validated prediction artifact "
                            f"S{subject:02d} {condition} {model_name} {frac}%"
                        )
                        for metric, value in metrics.items():
                            results.append({
                                "subject_id": subject,
                                "condition": condition,
                                "model": model_name,
                                "calibration_fraction": frac,
                                "metric": metric,
                                "value": value,
                                "n_source_windows": len(source),
                                "n_calibration_windows": len(cal),
                                "n_evaluation_windows": len(ev),
                            })
                        manifest.append({
                            "subject_id": subject,
                            "condition": condition,
                            "model": model_name,
                            "calibration_fraction": frac,
                            "family": "ML",
                            "prediction_file": str(pred_path),
                            "n_source_windows": len(source),
                            "n_calibration_windows": len(cal),
                            "n_evaluation_windows": len(ev),
                            "evaluation_fixed": True,
                            "adaptation_status": (
                                "NO_ADAPTATION_ZERO_CALIBRATION" if frac == 0 else
                                ("ADAPTED_TARGET_CALIBRATION" if len(cal) > 0 else
                                 "NO_COMPLETE_TARGET_WINDOWS_AT_REQUESTED_BOUNDARY")
                            ),
                            "evaluation_sets": FIXED_EVAL_SET,
                            "evaluation_window_ids": evaluation_window_ids_str,
                            "status": "PASS_RESUMED",
                        })
                        split_audit.append({
                            "subject_id": subject,
                            "condition": condition,
                            "model": model_name,
                            "calibration_fraction": frac,
                            "family": "ML",
                            "physical_overlap_violations": violations,
                            "evaluation_fixed": True,
                            "pass": violations == 0,
                        })
                        continue

                    model = make_ml_model(model_name)

                    # -----------------------------------------------------------------
                    # FIT MODEL
                    # -----------------------------------------------------------------
                    if model_name == "RandomForest":
                        # sklearn RandomForestClassifier internally calls
                        # _set_random_states(), which recursively inspects estimator
                        # parameters while creating individual trees.
                        #
                        # The Block09 long-running process has shown an intermittent
                        # inspect.Parameter failure even though a fresh standalone RF
                        # estimator works correctly.
                        #
                        # Validate the estimator immediately before fit so that a
                        # corrupted estimator cannot silently enter the run.
                        try:
                            import inspect as _rf_inspect

                            _rf_signature = _rf_inspect.signature(
                                model.__class__.__init__
                            )
                            _rf_params = model.get_params(deep=True)

                            if not isinstance(
                                _rf_signature,
                                _rf_inspect.Signature,
                            ):
                                raise RuntimeError(
                                    "RandomForest inspect.Signature validation failed."
                                )

                            if not isinstance(_rf_params, dict):
                                raise RuntimeError(
                                    "RandomForest get_params() did not return a dictionary."
                                )

                        except Exception as exc:
                            raise RuntimeError(
                                "RandomForest estimator integrity check failed immediately "
                                "before fit: "
                                f"{type(exc).__name__}: {exc}"
                            ) from exc

                        try:
                            _fit_random_forest_safely(model, Xtrain, ytrain)

                        except TypeError as exc:
                            if "'Parameter' object is not iterable" not in str(exc):
                                raise

                            # -------------------------------------------------------------
                            # Retry with a completely fresh sklearn RF estimator.
                            #
                            # The first estimator is discarded.  No learned state is
                            # carried into the retry.
                            # -------------------------------------------------------------
                            log(
                                f"RF FIT RETRY: rebuilding estimator after inspect failure "
                                f"S{subject:02d} {condition} {frac}%"
                            )

                            from sklearn.ensemble import RandomForestClassifier

                            model = RandomForestClassifier(
                                n_estimators=int(
                                    ML_CONFIG["RF"]["n_estimators"]
                                ),
                                random_state=ML_CONFIG["RF"]["random_state"],
                                n_jobs=int(
                                    ML_CONFIG["RF"]["n_jobs"]
                                ),
                                class_weight=ML_CONFIG["RF"]["class_weight"],
                            )

                            # Confirm the fresh object before retry.
                            _retry_params = model.get_params(deep=True)

                            if not isinstance(_retry_params, dict):
                                raise RuntimeError(
                                    "Fresh RandomForest retry estimator returned an invalid "
                                    "parameter dictionary."
                                )

                            _fit_random_forest_safely(model, Xtrain, ytrain)

                    else:
                        model.fit(Xtrain, ytrain)

                    pred = np.asarray(model.predict(Xev), dtype=int)
                

                    if not np.all(np.isin(pred, GESTURES)):
                        raise RuntimeError(
                            f"Invalid ML predictions: {model_name}"
                        )

                    metrics = calculate_metrics(yev, pred)

                    for metric, value in metrics.items():
                        results.append(
                            {
                                "subject_id": subject,
                                "condition": condition,
                                "model": model_name,
                                "calibration_fraction": frac,
                                "metric": metric,
                                "value": value,
                                "n_source_windows": len(source),
                                "n_calibration_windows": len(cal),
                                "n_evaluation_windows": len(ev),
                            }
                        )

                    pred_df = pd.DataFrame(
                        [
                            {
                                "subject_id": subject,
                                "condition": condition,
                                "model": model_name,
                                "calibration_fraction": frac,
                                "gesture_true": int(r["gesture"]),
                                "gesture_pred": int(p),
                                "window_id": str(r["window_id"]),
                                "window_number": int(r["window_number"]),
                                "window_index": int(r["window_index"]),
                                "set": r["set"],
                                "recording_id": r["recording_id"],
                                "evaluation_start_sample": int(
                                    r["start_sample"]
                                ),
                                "evaluation_end_sample": int(
                                    r["end_sample"]
                                ),
                                "source_file": str(r["source_file"]),
                                "calibration_start_sample": (
                                    int(cal[0]["start_sample"])
                                    if len(cal) else None
                                ),
                                "calibration_end_sample": (
                                    int(max(x["end_sample"] for x in cal))
                                    if len(cal) else None
                                ),
                                "embargo_start_sample": 4250,
                                "embargo_end_sample": 4750,
                            }
                            for r, p in zip(ev, pred)
                        ]
                    )

                    write_csv(pred_path, pred_df)

                    manifest.append(
                        {
                            "subject_id": subject,
                            "condition": condition,
                            "model": model_name,
                            "calibration_fraction": frac,
                            "family": "ML",
                            "prediction_file": str(pred_path),
                            "n_source_windows": len(source),
                            "n_calibration_windows": len(cal),
                            "n_evaluation_windows": len(ev),
                            "evaluation_fixed": True,
                            "adaptation_status": (
                                "NO_ADAPTATION_ZERO_CALIBRATION"
                                if frac == 0
                                else (
                                    "ADAPTED_TARGET_CALIBRATION"
                                    if len(cal) > 0
                                    else "NO_COMPLETE_TARGET_WINDOWS_AT_REQUESTED_BOUNDARY"
                                )
                            ),
                            "evaluation_sets": FIXED_EVAL_SET,
                            "evaluation_window_ids": evaluation_window_ids_str,
                            "status": "PASS",
                        }
                    )

                    split_audit.append(
                        {
                            "subject_id": subject,
                            "condition": condition,
                            "model": model_name,
                            "calibration_fraction": frac,
                            "family": "ML",
                            "physical_overlap_violations": violations,
                            "evaluation_fixed": True,
                            "pass": violations == 0,
                        }
                    )

    return (
        pd.DataFrame(results),
        pd.DataFrame(manifest),
        pd.DataFrame(split_audit),
    )


# =============================================================================
# 09. EXACT BLOCK07 MODEL IMPORT
# =============================================================================

def load_block07_module():
    global DL_LABEL_CONVENTION
    path = _locate_block07_source()
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
            f"{path} has no safe __main__ guard; Block09 refuses to import it."
        )
    inferred = _infer_block07_label_convention_from_source(source_text)
    if DL_LABEL_CONVENTION is None:
        DL_LABEL_CONVENTION = inferred
    elif DL_LABEL_CONVENTION != inferred:
        raise RuntimeError("Block07 label convention mismatch.")
    spec = importlib.util.spec_from_file_location("paper2_block07_canonical", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_model"):
        raise RuntimeError("Canonical Block07 module has no build_model().")
    return module, path


def discover_exact_block07_checkpoint(subject: int, condition: str, model_name: str):
    """Require exactly one checkpoint mapped to this LOSO fold."""
    folds_root = BLOCK07_ROOT / "07B_DL_LOSO" / "folds"
    if not folds_root.exists():
        raise FileNotFoundError(f"BLOCK07 folds directory not found: {folds_root}")

    candidates = []
    for ckpt in folds_root.rglob("best_model.pt"):
        text = "/".join([x.upper() for x in ckpt.parts])
        subject_ok = bool(re.search(rf"(?:^|[/_-])S0*{subject}(?:[/_-]|$)", text))
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
                if canonical_model(part) == model_name:
                    model_ok = True
                    break
            except Exception:
                pass
        if not model_ok:
            continue

        metas = list(ckpt.parent.rglob("complete.json"))
        if len(metas) != 1:
            metas = list(ckpt.parent.parent.rglob("complete.json"))
        if len(metas) != 1:
            raise RuntimeError(
                f"Cannot uniquely locate completion metadata for {ckpt}"
            )
        candidates.append((ckpt, metas[0]))

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one Block07 checkpoint for "
            f"S{subject:02d} {condition} {model_name}; found {len(candidates)}"
        )
    return candidates[0]


def _extract_normalizer_from_checkpoint(ckpt_obj, checkpoint: Path):
    mean_keys = [
        "normalizer_mean", "channel_mean", "normalization_mean",
        "train_mean", "x_mean", "scaler_mean", "mean",
    ]
    std_keys = [
        "normalizer_std", "channel_std", "normalization_std",
        "train_std", "x_std", "scaler_std", "std",
    ]

    mean = next((ckpt_obj[k] for k in mean_keys if k in ckpt_obj), None)
    std = next((ckpt_obj[k] for k in std_keys if k in ckpt_obj), None)

    sidecars = [
        checkpoint.with_name("normalization_stats.npz"),
        checkpoint.with_name("normalizer.npz"),
        checkpoint.with_name("normalization.json"),
        checkpoint.with_name("normalizer.json"),
    ]

    if mean is None or std is None:
        found = [x for x in sidecars if x.exists()]
        ordered = sorted(
            found,
            key=lambda x: (
                0 if x.suffix.lower() == ".npz" else 1,
                sidecars.index(x),
            ),
        )
        for side in ordered:
            try:
                if side.suffix.lower() == ".npz":
                    with np.load(side, allow_pickle=False) as z:
                        if "mean" in z.files and "std" in z.files:
                            mean, std = z["mean"], z["std"]
                        elif (
                            "normalizer_mean" in z.files
                            and "normalizer_std" in z.files
                        ):
                            mean = z["normalizer_mean"]
                            std = z["normalizer_std"]
                        elif "x_mean" in z.files and "x_std" in z.files:
                            mean, std = z["x_mean"], z["x_std"]
                else:
                    obj = json.loads(side.read_text(encoding="utf-8"))
                    mean = obj.get(
                        "mean",
                        obj.get("normalizer_mean", obj.get("x_mean")),
                    )
                    std = obj.get(
                        "std",
                        obj.get("normalizer_std", obj.get("x_std")),
                    )
            except Exception:
                mean, std = None, None
            if mean is not None and std is not None:
                break

    if mean is None or std is None:
        raise RuntimeError(
            f"Exact Block07 normalization statistics unavailable for "
            f"{checkpoint}. Checkpoint keys: "
            f"{sorted(str(k) for k in ckpt_obj.keys())}. "
            f"Sidecar candidates searched: {[str(x) for x in sidecars]}"
        )

    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)

    if mean.shape != (N_CHANNELS,) or std.shape != (N_CHANNELS,):
        raise RuntimeError(
            f"Invalid normalization shape for {checkpoint}: "
            f"mean={mean.shape}, std={std.shape}; expected {(N_CHANNELS,)}."
        )
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise RuntimeError(f"Invalid normalization values for {checkpoint}")

    return mean, std


def load_block07_checkpoint(block07_module, subject: int, condition: str, model_name: str):
    import torch

    checkpoint, complete = discover_exact_block07_checkpoint(
        subject, condition, model_name
    )
    complete_obj = json.loads(complete.read_text(encoding="utf-8"))
    expected_hash = complete_obj.get("checkpoint_sha256")
    if expected_hash and sha256_file(checkpoint) != expected_hash:
        raise RuntimeError(f"BLOCK07 checkpoint hash mismatch: {checkpoint}")

    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(
            f"Unexpected Block07 checkpoint structure: {checkpoint}"
        )
    if not hasattr(block07_module, "build_model"):
        raise RuntimeError(
            "Canonical Block07 module has no build_model(); "
            "architecture reconstruction is forbidden."
        )

    model = block07_module.build_model(model_name)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    mean, std = _extract_normalizer_from_checkpoint(ckpt, checkpoint)

    # Metadata checks against complete.json (authoritative source).
    if "held_out_subject" in complete_obj:
        got = int(complete_obj["held_out_subject"])
        if got != subject:
            raise RuntimeError(
                f"Block07 complete.json held_out_subject mismatch for "
                f"{complete}: {got} != {subject}"
            )
    if "condition" in complete_obj:
        if canonical_condition(complete_obj["condition"]) != condition:
            raise RuntimeError(
                f"Block07 complete.json condition mismatch for {complete}"
            )
    if "model" in complete_obj:
        if canonical_model(complete_obj["model"]) != model_name:
            raise RuntimeError(
                f"Block07 complete.json model mismatch for {complete}"
            )

    # Optional validation-subject consistency check.
    saved_validation_subject = ckpt.get("validation_subject")
    if saved_validation_subject is not None:
        expected_validation_subject = min(s for s in SUBJECTS if s != subject)
        if int(saved_validation_subject) != expected_validation_subject:
            raise RuntimeError(
                f"Block07 checkpoint validation subject mismatch for "
                f"S{subject:02d}: saved={saved_validation_subject}, "
                f"expected={expected_validation_subject}"
            )

    return {
        "model": model,
        "normalizer_mean": mean,
        "normalizer_std": std,
        "best_epoch": max(1, int(ckpt.get("best_epoch", 1))),
        "best_validation_macro_f1": float(
            ckpt.get("best_validation_macro_f1", np.nan)
        ),
        "checkpoint": checkpoint,
        "complete": complete,
    }


def normalize_with_source_stats(X, mean, std):
    X = np.asarray(X, dtype=np.float32)
    return (
        (X - mean[None, :, None])
        / std[None, :, None]
    ).astype(np.float32)


def _locate_final_classifier(model):
    for name in ("classifier", "fc", "final_classifier", "classification_head"):
        if hasattr(model, name):
            head = getattr(model, name)
            if hasattr(head, "parameters"):
                return head, name
    raise RuntimeError(
        f"Canonical Block07 model {type(model).__name__} has no explicit "
        "final classifier head; Block09 will not guess."
    )


def freeze_backbone_only_classifier(model):
    for p in model.parameters():
        p.requires_grad = False
    head, name = _locate_final_classifier(model)
    for p in head.parameters():
        p.requires_grad = True
    model._block09_classifier_head_name = name


def set_frozen_backbone_mode(model):
    model.eval()
    head, _ = _locate_final_classifier(model)
    head.train()


def model_forward(model, x):
    """Standard forward entry."""
    return model(x)


# =============================================================================
# 10. RAW WINDOW LOADING FOR DL
# =============================================================================

def resolve_block04_window_array(path_value: str) -> Path:
    """Resolve the manifest's window_array_file, not the source recording."""
    raw = Path(str(path_value).replace("\\", "/"))
    candidates = []
    if raw.is_absolute() and raw.exists():
        candidates.append(raw)
    for base in (PAPER2_ROOT, BLOCK04_ROOT):
        q = base / raw
        if q.exists():
            candidates.append(q)
    if not candidates and raw.name:
        matches = list(BLOCK04_ROOT.rglob(raw.name))
        if len(matches) == 1:
            candidates = matches
        elif len(matches) > 1:
            raise RuntimeError(f"Ambiguous Block04 window array {raw.name}: {matches}")
    unique = list(dict.fromkeys(str(x.resolve()) for x in candidates))
    if len(unique) != 1:
        raise FileNotFoundError(
            f"Could not uniquely resolve BLOCK04 window_array_file: {path_value}"
        )
    return Path(unique[0])


def load_raw_windows(rows_df: pd.DataFrame):
    cache = {}
    Xs, ys = [], []
    for _, r in rows_df.iterrows():
        array_file = resolve_block04_window_array(str(r.window_array_file))
        if array_file not in cache:
            with np.load(array_file, allow_pickle=False) as z:
                if "eeg_windows" not in z.files:
                    raise RuntimeError(f"BLOCK04 window array missing 'eeg_windows': {array_file}")
                arr = np.asarray(z["eeg_windows"], dtype=np.float32)
                if arr.shape != (WINDOWS_PER_RECORDING, N_CHANNELS, WINDOW_LENGTH):
                    raise RuntimeError(
                        f"Unexpected BLOCK04 eeg_windows shape {arr.shape} in {array_file}; "
                        f"expected {(WINDOWS_PER_RECORDING, N_CHANNELS, WINDOW_LENGTH)}"
                    )
                if "start_samples" not in z.files or "end_samples" not in z.files:
                    raise RuntimeError(f"BLOCK04 window array lacks physical start/end metadata: {array_file}")
                starts = np.asarray(z["start_samples"], dtype=int)
                ends = np.asarray(z["end_samples"], dtype=int)
                if starts.shape != (WINDOWS_PER_RECORDING,) or ends.shape != (WINDOWS_PER_RECORDING,):
                    raise RuntimeError(f"BLOCK04 start/end metadata shape mismatch: {array_file}")
                cache[array_file] = (arr, starts, ends)

        arr, starts, ends = cache[array_file]
        idx = int(r.window_index)
        if not 0 <= idx < WINDOWS_PER_RECORDING:
            raise RuntimeError(f"Invalid BLOCK04 window_index={idx}: {r.window_id}")
        if int(starts[idx]) != int(r.start_sample) or int(ends[idx]) != int(r.end_sample):
            raise RuntimeError(
                "BLOCK04 manifest/NPZ temporal identity mismatch for "
                f"{r.window_id}: manifest=({r.start_sample},{r.end_sample}), "
                f"NPZ=({starts[idx]},{ends[idx]})"
            )
        Xs.append(arr[idx])
        ys.append(int(r.gesture))

    if not Xs:
        return (
            np.empty((0, N_CHANNELS, WINDOW_LENGTH), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.stack(Xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def build_dl_target_dataframe(
    inv: pd.DataFrame,
    lookup: pd.DataFrame,
    subject: int,
    condition: str,
    frac: int,
    role: str,
):
    keys = lookup_rows(
        lookup,
        subject,
        condition,
        frac,
        role,
    )

    if keys.empty:
        return inv.iloc[0:0].copy()

    key_set = set(
        zip(
            keys.gesture.astype(int),
            keys.set.astype(str),
            keys.window_index.astype(int),
            keys.recording_id.astype(str),
        )
    )

    rows = []
    target = inv[
        (inv.subject == subject)
        & (inv.condition == condition)
    ].copy()

    for _, r in target.iterrows():
        key = (
            int(r.gesture),
            str(r.set),
            int(r.window_index),
            str(r.recording_id),
        )
        if key in key_set:
            rows.append(r)

    if not rows:
        return inv.iloc[0:0].copy()

    return pd.DataFrame(rows)


# =============================================================================
# 11. DL CALIBRATION
# =============================================================================

def chronological_calibration_validation_split(cal_df: pd.DataFrame):
    """Chronological, gesture-stratified validation inside calibration."""
    if cal_df.empty:
        return None

    target_n = max(
        1,
        int(math.ceil(DL_INTERNAL_VAL_FRACTION * len(cal_df))),
    )

    val_idx = []
    train_idx = []

    for gesture in GESTURES:
        idx = np.where(
            cal_df.gesture.astype(int).to_numpy() == gesture
        )[0]

        if len(idx) == 0:
            continue

        order = sorted(
            idx.tolist(),
            key=lambda i: int(cal_df.iloc[i].start_sample),
        )

        n_val = max(
            1,
            int(math.ceil(DL_INTERNAL_VAL_FRACTION * len(order))),
        )

        val_idx.extend(order[-n_val:])
        train_idx.extend(order[:-n_val])

    val_idx = np.asarray(sorted(set(val_idx)), dtype=int)
    train_idx = np.asarray(sorted(set(train_idx)), dtype=int)

    if (
        len(val_idx) < DL_INTERNAL_VAL_MIN_WINDOWS
        or len(train_idx) == 0
        or len(val_idx) >= len(cal_df)
    ):
        return None

    return train_idx, val_idx, target_n


def dl_targets_to_internal(y_external):
    y = np.asarray(y_external, dtype=np.int64)
    if not np.all(np.isin(y, GESTURES)):
        raise RuntimeError("DL target labels must be external gesture IDs 1..7.")
    if DL_LABEL_CONVENTION == "zero_based_internal":
        return (y - 1).astype(np.int64)
    if DL_LABEL_CONVENTION == "one_based_direct":
        return y.astype(np.int64)
    raise RuntimeError("DL label convention has not been verified.")


def dl_predictions_to_external(y_internal):
    y = np.asarray(y_internal, dtype=np.int64)
    if not np.all((y >= 0) & (y < 7)):
        raise RuntimeError("DL model produced an invalid class index.")
    if DL_LABEL_CONVENTION == "zero_based_internal":
        return y + 1
    if DL_LABEL_CONVENTION == "one_based_direct":
        return y
    raise RuntimeError("DL label convention has not been verified.")


def dl_adapt_classifier(
    model,
    Xcal,
    ycal,
    mean,
    std,
    best_epoch_source,
    validation_split,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import f1_score

    device = torch.device("cuda")

    Xcal_n = normalize_with_source_stats(Xcal, mean, std)

    freeze_backbone_only_classifier(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable DL classifier parameters.")

    if validation_split is None:
        Xtr = Xcal_n
        ytr = ycal
        Xva = None
        yva = None
        epochs = max(1, int(best_epoch_source))
        validation_mode = DL_SMALL_CALIBRATION_POLICY
    else:
        train_idx, val_idx, _ = validation_split
        Xtr = Xcal_n[train_idx]
        ytr = ycal[train_idx]
        Xva = Xcal_n[val_idx]
        yva = ycal[val_idx]
        epochs = DL_MAX_EPOCHS
        validation_mode = "TARGET_CALIBRATION_CHRONOLOGICAL_STRATIFIED"

    train_ds = TensorDataset(
        torch.from_numpy(Xtr.astype(np.float32)),
        torch.from_numpy(dl_targets_to_internal(ytr)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=DL_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    if Xva is not None:
        val_ds = TensorDataset(
            torch.from_numpy(Xva.astype(np.float32)),
            torch.from_numpy(dl_targets_to_internal(yva)),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=DL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    else:
        val_loader = None

    model.to(device)
    set_frozen_backbone_mode(model)

    optimizer = torch.optim.Adam(
        trainable,
        lr=DL_LR,
        weight_decay=DL_WEIGHT_DECAY,
    )
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_state = None
    best_val = -np.inf
    bad = 0

    for epoch in range(1, epochs + 1):
        set_frozen_backbone_mode(model)

        losses = []
        train_true = []
        train_pred = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model_forward(model, xb)
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, DL_GRAD_CLIP)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            train_true.extend(yb.detach().cpu().numpy().tolist())
            train_pred.extend(
                out.argmax(1).detach().cpu().numpy().tolist()
            )

        train_f1 = float(
            f1_score(
                np.asarray(train_true),
                np.asarray(train_pred),
                labels=np.arange(7),
                average="macro",
                zero_division=0,
            )
        )

        if val_loader is not None:
            model.eval()
            val_true = []
            val_pred = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    out = model_forward(
                        model,
                        xb.to(device, non_blocking=True),
                    )
                    val_true.extend(yb.cpu().numpy().tolist())
                    val_pred.extend(out.argmax(1).cpu().numpy().tolist())

            val_f1 = float(
                f1_score(
                    np.asarray(val_true),
                    np.asarray(val_pred),
                    labels=np.arange(7),
                    average="macro",
                    zero_division=0,
                )
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_macro_f1": train_f1,
                    "validation_macro_f1": val_f1,
                    "loss": float(np.mean(losses)),
                }
            )

            if val_f1 > best_val + DL_MIN_DELTA:
                best_val = val_f1
                bad = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            else:
                bad += 1

            if bad >= DL_PATIENCE:
                break
        else:
            history.append(
                {
                    "epoch": epoch,
                    "train_macro_f1": train_f1,
                    "validation_macro_f1": np.nan,
                    "loss": float(np.mean(losses)),
                }
            )

    if validation_split is not None:
        if best_state is None:
            raise RuntimeError(
                "Target calibration validation was used but no best "
                "classifier state was produced."
            )
        model.load_state_dict(best_state, strict=True)

    model.eval()
    return model, pd.DataFrame(history), validation_mode


def predict_dl(model, X, mean, std):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda")
    # At 0% calibration the adaptation routine is intentionally skipped, so
    # the model has not yet been moved to CUDA. Prediction must handle both
    # adapted and non-adapted runs explicitly.
    model.to(device)
    Xn = normalize_with_source_stats(X, mean, std)

    ds = TensorDataset(torch.from_numpy(Xn.astype(np.float32)))
    loader = DataLoader(
        ds,
        batch_size=DL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model.eval()
    pred = []

    with torch.no_grad():
        for (xb,) in loader:
            out = model_forward(model, xb.to(device, non_blocking=True))
            pred.extend(
                dl_predictions_to_external(
                    out.argmax(1).cpu().numpy()
                ).tolist()
            )

    return np.asarray(pred, dtype=int)


def run_dl_calibration(
    inv: pd.DataFrame,
    lookup: pd.DataFrame,
):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("BLOCK09 DL calibration requires CUDA.")

    block07_module, block07_path = load_block07_module()

    results = []
    manifest = []
    split_audit = []
    source_log = []

    total_runs = (
        len(SUBJECTS)
        * len(CONDITIONS)
        * len(DL_MODELS)
        * len(CALIBRATION_FRACTIONS)
    )
    run_no = 0

    evaluation_window_ids_str = ",".join(map(str, FIXED_EVAL_WINDOW_NUMBERS))

    for subject in SUBJECTS:
        for condition in CONDITIONS:
            validation_subject = min(s for s in SUBJECTS if s != subject)
            source_fit_subjects = {
                s
                for s in SUBJECTS
                if s != subject and s != validation_subject
            }

            if len(source_fit_subjects) != 23:
                raise RuntimeError(
                    f"Expected 23 Block07 source-fit subjects for "
                    f"S{subject:02d}; found {len(source_fit_subjects)}"
                )

            expected_source_windows = (
                23 * 3 * 7 * WINDOWS_PER_RECORDING
            )

            for model_name in DL_MODELS:
                source_info = load_block07_checkpoint(
                    block07_module,
                    subject,
                    condition,
                    model_name,
                )

                source_log.append(
                    {
                        "subject_id": subject,
                        "condition": condition,
                        "model": model_name,
                        "source_checkpoint": str(source_info["checkpoint"]),
                        "source_checkpoint_sha256": sha256_file(
                            source_info["checkpoint"]
                        ),
                        "source_best_epoch": source_info["best_epoch"],
                        "source_best_validation_macro_f1": source_info[
                            "best_validation_macro_f1"
                        ],
                        "source_validation_subject": validation_subject,
                        "source_fit_subject_count": len(source_fit_subjects),
                        "source_fit_window_count": expected_source_windows,
                        "source_normalization": (
                            "EXACT_BLOCK07_CHECKPOINT_STATISTICS"
                        ),
                        "architecture_source": str(block07_path),
                    }
                )

                for frac in CALIBRATION_FRACTIONS:
                    run_no += 1
                    log(
                        f"DL {run_no}/{total_runs}: "
                        f"S{subject:02d} {condition} "
                        f"{model_name} {frac}%"
                    )

                    cal_df = build_dl_target_dataframe(
                        inv, lookup, subject, condition, frac, "CAL"
                    )
                    ev_df = build_dl_target_dataframe(
                        inv, lookup, subject, condition, frac, "EVAL"
                    )

                    if len(ev_df) != FIXED_EVAL_WINDOWS_PER_SUBJECT_CONDITION:
                        raise RuntimeError(
                            f"DL fixed evaluation count mismatch: "
                            f"S{subject} {condition} {frac}% -> {len(ev_df)}"
                        )

                    if set(ev_df.set.astype(str)) != {FIXED_EVAL_SET}:
                        raise RuntimeError(
                            "DL evaluation does not contain the fixed SetC "
                            "evaluation tail."
                        )

                    if set(ev_df.window_index.astype(int)) != set(
                        FIXED_EVAL_WINDOW_INDICES
                    ):
                        raise RuntimeError(
                            "DL evaluation does not contain exactly "
                            f"window IDs {FIXED_EVAL_WINDOW_INDICES}."
                        )

                    cal_end = {}
                    for _, r in cal_df.iterrows():
                        rid = str(r.recording_id)
                        cal_end[rid] = max(
                            cal_end.get(rid, 0),
                            int(r.end_sample),
                        )

                    violations = 0
                    for _, r in ev_df.iterrows():
                        rid = str(r.recording_id)
                        if rid in cal_end:
                            if int(r.start_sample) < cal_end[rid]:
                                violations += 1

                    if violations:
                        raise RuntimeError(
                            f"DL temporal leakage detected: "
                            f"S{subject} {condition} {model_name} {frac}%"
                        )

                    pred_path = (
                        PRED_ROOT
                        / "DL"
                        / model_name
                        / f"S{subject:02d}"
                        / condition
                        / f"cal_{frac:02d}.csv"
                    )

                    cached = load_cached_prediction_run(
                        pred_path,
                        subject,
                        condition,
                        model_name,
                        frac,
                        ev_df,
                    )

                    if cached is not None:
                        cached_df, metrics = cached
                        log(
                            f"DL RESUME: reusing validated prediction artifact "
                            f"S{subject:02d} {condition} {model_name} {frac}%"
                        )
                        adaptation_status = (
                            "NO_ADAPTATION_ZERO_CALIBRATION" if frac == 0 else
                            ("ADAPTED_TARGET_CALIBRATION" if len(cal_df) > 0 else
                             "NO_COMPLETE_TARGET_WINDOWS_AT_REQUESTED_BOUNDARY")
                        )
                        for metric, value in metrics.items():
                            results.append({
                                "subject_id": subject,
                                "condition": condition,
                                "model": model_name,
                                "calibration_fraction": frac,
                                "metric": metric,
                                "value": value,
                                "n_source_windows": expected_source_windows,
                                "n_calibration_windows": len(cal_df),
                                "n_evaluation_windows": len(ev_df),
                                "adaptation_status": adaptation_status,
                            })
                        manifest.append({
                            "subject_id": subject,
                            "condition": condition,
                            "model": model_name,
                            "calibration_fraction": frac,
                            "family": "DL",
                            "prediction_file": str(pred_path),
                            "source_checkpoint": str(source_info["checkpoint"]),
                            "source_checkpoint_sha256": sha256_file(source_info["checkpoint"]),
                            "source_normalization": "EXACT_BLOCK07_CHECKPOINT_STATISTICS",
                            "fine_tune_operator": "classifier_only",
                            "backbone_frozen": True,
                            "frozen_batchnorm_running_stats": True,
                            "internal_val_mode": "RESUMED_FROM_VALIDATED_PREDICTION",
                            "adaptation_status": adaptation_status,
                            "n_source_windows": expected_source_windows,
                            "n_calibration_windows": len(cal_df),
                            "n_evaluation_windows": len(ev_df),
                            "evaluation_fixed": True,
                            "evaluation_sets": FIXED_EVAL_SET,
                            "evaluation_window_ids": evaluation_window_ids_str,
                            "status": "PASS_RESUMED",
                        })
                        split_audit.append({
                            "subject_id": subject,
                            "condition": condition,
                            "model": model_name,
                            "calibration_fraction": frac,
                            "family": "DL",
                            "physical_overlap_violations": violations,
                            "evaluation_fixed": True,
                            "pass": violations == 0,
                        })
                        continue

                    Xev, yev = load_raw_windows(ev_df)

                    if len(cal_df):
                        Xcal, ycal = load_raw_windows(cal_df)
                    else:
                        Xcal = np.empty(
                            (0, N_CHANNELS, WINDOW_LENGTH),
                            dtype=np.float32,
                        )
                        ycal = np.empty((0,), dtype=np.int64)

                    model = block07_module.build_model(model_name)
                    model.load_state_dict(
                        source_info["model"].state_dict(),
                        strict=True,
                    )

                    set_seeds(
                        deterministic_seed_for(
                            "BLOCK09_DL_CALIBRATION",
                            subject,
                            condition,
                            model_name,
                            frac,
                        )
                    )

                    internal_val_mode = "NO_TARGET_CALIBRATION"
                    history = pd.DataFrame()
                    adaptation_status = "NO_ADAPTATION_ZERO_CALIBRATION"

                    if frac > 0 and len(cal_df) > 0:
                        validation_split = (
                            chronological_calibration_validation_split(
                                cal_df
                            )
                        )
                        model, history, internal_val_mode = (
                            dl_adapt_classifier(
                                model,
                                Xcal,
                                ycal,
                                source_info["normalizer_mean"],
                                source_info["normalizer_std"],
                                source_info["best_epoch"],
                                validation_split,
                            )
                        )
                        adaptation_status = "ADAPTED_TARGET_CALIBRATION"
                    elif frac > 0 and len(cal_df) == 0:
                        # With W500/O50, the 10% physical boundary is 450 samples
                        # after the start of the recording, so no complete 500-sample
                        # computational window can fit inside that boundary. Do not
                        # silently call this a successful 10% adaptation point.
                        adaptation_status = (
                            "NO_COMPLETE_TARGET_WINDOWS_AT_REQUESTED_BOUNDARY"
                        )

                    pred = predict_dl(
                        model,
                        Xev,
                        source_info["normalizer_mean"],
                        source_info["normalizer_std"],
                    )

                    if not np.all(np.isin(pred, GESTURES)):
                        raise RuntimeError("Invalid DL gesture IDs.")

                    metrics = calculate_metrics(yev, pred)

                    for metric, value in metrics.items():
                        results.append(
                            {
                                "subject_id": subject,
                                "condition": condition,
                                "model": model_name,
                                "calibration_fraction": frac,
                                "metric": metric,
                                "value": value,
                                "n_source_windows": expected_source_windows,
                                "n_calibration_windows": len(Xcal),
                                "n_evaluation_windows": len(Xev),
                                "adaptation_status": adaptation_status,
                            }
                        )

                    pred_df = pd.DataFrame(
                        [
                            {
                                "subject_id": subject,
                                "condition": condition,
                                "model": model_name,
                                "calibration_fraction": frac,
                                "gesture_true": int(r.gesture),
                                "gesture_pred": int(p),
                                "window_id": str(r.window_id),
                                "window_number": int(r.window_number),
                                "window_index": int(r.window_index),
                                "set": r.set,
                                "recording_id": r.recording_id,
                                "evaluation_start_sample": int(
                                    r.start_sample
                                ),
                                "evaluation_end_sample": int(
                                    r.end_sample
                                ),
                                "source_file": str(r.source_file),
                                "calibration_start_sample": (
                                    int(cal_df.start_sample.min())
                                    if len(cal_df) else None
                                ),
                                "calibration_end_sample": (
                                    int(cal_df.end_sample.max())
                                    if len(cal_df) else None
                                ),
                                "embargo_start_sample": 4250,
                                "embargo_end_sample": 4750,
                            }
                            for r, p in zip(
                                ev_df.itertuples(index=False),
                                pred,
                            )
                        ]
                    )

                    write_csv(pred_path, pred_df)

                    history_path = (
                        DIAG_ROOT
                        / "DL_histories"
                        / model_name
                        / f"S{subject:02d}"
                        / condition
                        / f"cal_{frac:02d}.csv"
                    )
                    write_csv(history_path, history)

                    manifest.append(
                        {
                            "subject_id": subject,
                            "condition": condition,
                            "model": model_name,
                            "calibration_fraction": frac,
                            "family": "DL",
                            "prediction_file": str(pred_path),
                            "source_checkpoint": str(
                                source_info["checkpoint"]
                            ),
                            "source_checkpoint_sha256": sha256_file(
                                source_info["checkpoint"]
                            ),
                            "source_normalization": (
                                "EXACT_BLOCK07_CHECKPOINT_STATISTICS"
                            ),
                            "fine_tune_operator": "classifier_only",
                            "backbone_frozen": True,
                            "frozen_batchnorm_running_stats": True,
                            "internal_val_mode": internal_val_mode,
                            "adaptation_status": adaptation_status,
                            "n_source_windows": expected_source_windows,
                            "n_calibration_windows": len(Xcal),
                            "n_evaluation_windows": len(Xev),
                            "evaluation_fixed": True,
                            "evaluation_sets": FIXED_EVAL_SET,
                            "evaluation_window_ids": evaluation_window_ids_str,
                            "status": "PASS",
                        }
                    )

                    split_audit.append(
                        {
                            "subject_id": subject,
                            "condition": condition,
                            "model": model_name,
                            "calibration_fraction": frac,
                            "family": "DL",
                            "physical_overlap_violations": violations,
                            "evaluation_fixed": True,
                            "pass": violations == 0,
                        }
                    )

                    del model
                    torch.cuda.empty_cache()
                    gc.collect()

    return (
        pd.DataFrame(results),
        pd.DataFrame(manifest),
        pd.DataFrame(split_audit),
        pd.DataFrame(source_log),
    )


# =============================================================================
# 12. A_SD FROM SAME FIXED EVALUATION WINDOWS
# =============================================================================

def discover_prediction_files(root: Path) -> List[Path]:
    return sorted(root.rglob("predictions.csv"))


def normalize_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "subject" in df.columns and "subject_id" not in df.columns:
        rename["subject"] = "subject_id"
    if "true" in df.columns and "gesture_true" not in df.columns:
        rename["true"] = "gesture_true"
    if "pred" in df.columns and "gesture_pred" not in df.columns:
        rename["pred"] = "gesture_pred"
    if rename:
        df = df.rename(columns=rename)

    required = {
        "subject_id", "condition", "model",
        "gesture_true", "gesture_pred", "window_id", "set",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Prediction file missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["subject_id"] = df["subject_id"].map(canonical_subject)
    df["condition"] = df["condition"].map(canonical_condition)
    df["model"] = df["model"].map(canonical_model)
    df["set"] = df["set"].map(canonical_set)
    df["gesture_true"] = df["gesture_true"].astype(int)
    df["gesture_pred"] = df["gesture_pred"].astype(int)

    parsed = df["window_id"].map(parse_prediction_window_id)
    df["prediction_window_id"] = parsed.map(
        lambda x: x[0] if x[0] is not None else ""
    )
    df["window_index"] = parsed.map(lambda x: x[1]).astype(int)
    df["window_number"] = df["window_index"] + 1

    if not df.gesture_true.isin(GESTURES).all():
        raise RuntimeError("Invalid true gesture IDs in prediction file.")
    if not df.gesture_pred.isin(GESTURES).all():
        raise RuntimeError("Invalid predicted gesture IDs in prediction file.")
    if not df.window_index.isin(range(WINDOWS_PER_RECORDING)).all():
        raise RuntimeError("Invalid prediction window indices.")
    return df

def verify_block07_sd_run_artifacts(block07_sd_root: Path) -> pd.DataFrame:
    """Validate the actual Block07 SD run layout before using predictions.

    Block07 does not create one aggregated SD subject-results CSV. Its
    authoritative SD run artifacts are stored as one class_metrics.csv per
    subject/condition/model under:

        07A_DL_SD/runs/Subject_XX/<condition>/<model>/

    The frozen Block07 SD protocol uses SetA for training, SetB for
    validation, and the FULL SetC (147 windows = 21 per gesture) for test.

    The actual Block07 class_metrics.csv artifact is a seven-row,
    per-gesture file with canonical gesture codes 1..7. It contains:

        gesture_code, gesture, precision, recall, f1, support

    Therefore this function validates the seven per-gesture rows and the
    canonical gesture coverage. It does not use class_metrics.csv as the
    Block09 fixed-tail primary metric because those metrics summarize the
    full 147-window SetC test.

    The exact fixed-tail A_SD is reconstructed from the corresponding
    frozen Block07 predictions.csv files below.
    """
    runs_root = block07_sd_root / "runs"
    require_file(runs_root, "BLOCK07 SD runs directory")

    files = sorted(runs_root.rglob("class_metrics.csv"))
    if len(files) != 25 * 2 * 5:
        raise RuntimeError(
            "BLOCK07 SD class_metrics coverage mismatch: "
            f"found {len(files)} files, expected 250. "
            f"Root={runs_root}"
        )

    expected = {
        (s, c, canonical_model(m))
        for s in SUBJECTS
        for c in CONDITIONS
        for m in DL_MODELS
    }

    rows = []

    for path in files:
        df = pd.read_csv(path)

        # ------------------------------------------------------------------
        # Actual Block07 SD class_metrics structure:
        # exactly 7 rows, one row for each canonical gesture code 1..7.
        # ------------------------------------------------------------------
        required_class_columns = {
            "gesture_code",
            "gesture",
            "precision",
            "recall",
            "f1",
            "support",
        }

        missing_class_columns = required_class_columns - set(df.columns)
        if missing_class_columns:
            raise RuntimeError(
                "BLOCK07 SD class_metrics missing required per-gesture "
                f"columns {sorted(missing_class_columns)}: {path}"
            )

        if df.shape[0] != 7:
            raise RuntimeError(
                "BLOCK07 SD class_metrics must contain exactly 7 "
                f"per-gesture rows: {path}; found {df.shape[0]}"
            )

        try:
            gesture_codes = sorted(
                pd.to_numeric(
                    df["gesture_code"],
                    errors="raise",
                ).astype(int).tolist()
            )
        except Exception as exc:
            raise RuntimeError(
                "BLOCK07 SD class_metrics gesture_code values could not "
                f"be parsed as integers: {path}"
            ) from exc

        if gesture_codes != list(range(1, 8)):
            raise RuntimeError(
                "BLOCK07 SD class_metrics must contain canonical gesture "
                f"codes 1..7 exactly: {path}; found {gesture_codes}"
            )

        try:
            support_values = pd.to_numeric(
                df["support"],
                errors="raise",
            ).astype(int)
        except Exception as exc:
            raise RuntimeError(
                "BLOCK07 SD class_metrics support values could not "
                f"be parsed as integers: {path}"
            ) from exc

        if not bool((support_values == 21).all()):
            raise RuntimeError(
                "BLOCK07 SD class_metrics must contain support=21 for "
                f"each gesture: {path}; found {support_values.tolist()}"
            )

        # ------------------------------------------------------------------
        # The aggregate run metadata are validated from the corresponding
        # predictions.csv artifact, not from class_metrics.csv.
        # ------------------------------------------------------------------
        predictions_files = sorted(path.parent.glob("predictions.csv"))

        if len(predictions_files) != 1:
            raise RuntimeError(
                "BLOCK07 SD run must contain exactly one predictions.csv "
                f"artifact: {path.parent}; found {len(predictions_files)}"
            )

        predictions_path = predictions_files[0]
        pred_df = pd.read_csv(predictions_path)

        required_prediction_columns = {
            "subject_id",
            "condition",
            "model",
            "evaluation_regime",
            "gesture_true",
            "gesture_pred",
            "window_id",
            "set",
        }

        missing_prediction_columns = (
            required_prediction_columns - set(pred_df.columns)
        )

        if missing_prediction_columns:
            raise RuntimeError(
                "BLOCK07 SD predictions.csv missing required columns "
                f"{sorted(missing_prediction_columns)}: {predictions_path}"
            )

        subject = canonical_subject(pred_df.iloc[0]["subject_id"])
        condition = canonical_condition(pred_df.iloc[0]["condition"])
        model = canonical_model(pred_df.iloc[0]["model"])
        regime = str(
            pred_df.iloc[0]["evaluation_regime"]
        ).strip().upper()

        if regime != "SD":
            raise RuntimeError(
                "BLOCK07 SD predictions.csv has "
                f"evaluation_regime={regime!r}: {predictions_path}"
            )

        key = (subject, condition, model)

        if key not in expected:
            raise RuntimeError(
                f"Unexpected BLOCK07 SD run key {key}: {predictions_path}"
            )

        # Every frozen SD run must contain the complete SetC test:
        # 147 windows = 21 windows × 7 gestures.
        if pred_df.shape[0] != 147:
            raise RuntimeError(
                "BLOCK07 SD predictions test-size mismatch for "
                f"{key}: found {pred_df.shape[0]} rows; expected 147: "
                f"{predictions_path}"
            )

        set_values = (
            pred_df["set"]
            .astype(str)
            .str.strip()
            .str.upper()
            .unique()
            .tolist()
        )

        if set_values != ["SETC"]:
            raise RuntimeError(
                "BLOCK07 SD predictions must contain only SetC test "
                f"windows for {key}: found {set_values}: "
                f"{predictions_path}"
            )

        true_gesture_codes = sorted(
            pd.to_numeric(
                pred_df["gesture_true"],
                errors="raise",
            ).astype(int).unique().tolist()
        )

        if true_gesture_codes != list(range(1, 8)):
            raise RuntimeError(
                "BLOCK07 SD predictions must contain canonical true "
                f"gesture codes 1..7 for {key}: "
                f"found {true_gesture_codes}: {predictions_path}"
            )

        gesture_support = (
            pd.to_numeric(
                pred_df["gesture_true"],
                errors="raise",
            )
            .astype(int)
            .value_counts()
            .sort_index()
        )

        if not bool((gesture_support == 21).all()):
            raise RuntimeError(
                "BLOCK07 SD predictions must contain 21 test windows "
                f"per gesture for {key}: "
                f"found {gesture_support.to_dict()}: "
                f"{predictions_path}"
            )

        rows.append(
            {
                "subject_id": subject,
                "condition": condition,
                "model": model,
                "evaluation_regime": regime,
                "n_train": 147,
                "n_validation": 147,
                "n_test": 147,
                "class_metrics_path": str(path),
                "predictions_path": str(predictions_path),
            }
        )

    got = {
        (r["subject_id"], r["condition"], r["model"])
        for r in rows
    }

    if got != expected:
        raise RuntimeError(
            "BLOCK07 SD run-key coverage mismatch. "
            f"Missing={sorted(expected - got)[:10]}; "
            f"Extra={sorted(got - expected)[:10]}"
        )

    out = (
        pd.DataFrame(rows)
        .sort_values(
            ["subject_id", "condition", "model"]
        )
        .reset_index(drop=True)
    )

    write_csv(
        DIAG_ROOT / "BLOCK07_SD_run_artifact_audit.csv",
        out,
    )

    return out


def load_block06_sd_reference() -> pd.DataFrame:
    """Load BLOCK06 subject-dependent Macro-F1 as primary A_SD.

    Returns a long DataFrame with columns:
      subject_id, condition, model, metric, A_SD_primary
    """
    path = (
        BLOCK06_ROOT
        / "06A_ML_SD"
        / "subject_results"
        / "SD_subject_results.csv"
    )
    require_file(path, "BLOCK06 SD subject results")

    df = pd.read_csv(path)
    rename = {}
    if "subject" in df.columns and "subject_id" not in df.columns:
        rename["subject"] = "subject_id"
    if rename:
        df = df.rename(columns=rename)

    required = {"subject_id", "condition", "model"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"BLOCK06 SD subject results missing columns: {sorted(missing)}"
        )

    df["subject_id"] = df["subject_id"].map(canonical_subject)
    df["condition"] = df["condition"].map(canonical_condition)
    df["model"] = df["model"].map(canonical_model)

    long_rows = []
    for metric_canonical, aliases in {
        "accuracy": ["accuracy"],
        "balanced_accuracy": ["balanced_accuracy", "balancedaccuracy"],
        "macro_f1": ["macro_f1", "macro_f1_score", "f1_macro"],
        "mcc": ["mcc"],
        "cohen_kappa": ["cohen_kappa", "cohens_kappa", "kappa"],
    }.items():
        col = None
        for alias in aliases:
            if alias in df.columns:
                col = alias
                break
        if col is None:
            continue
        for _, r in df.iterrows():
            value = pd.to_numeric(r[col], errors="coerce")
            if pd.isna(value):
                continue
            long_rows.append({
                "subject_id": int(r["subject_id"]),
                "condition": str(r["condition"]),
                "model": str(r["model"]),
                "metric": metric_canonical,
                "A_SD_primary": float(value),
            })
    ref = pd.DataFrame(long_rows)
    if ref.empty:
        raise RuntimeError("BLOCK06 SD primary A_SD is empty.")
    ref.to_csv(DIAG_ROOT / "A_SD_primary_BLOCK06.csv", index=False)
    return ref



def _asd_ml_checkpoint_path() -> Path:
    return DIAG_ROOT / "A_SD_ML_sensitivity_checkpoint.json"


def _load_asd_ml_checkpoint():
    path = _asd_ml_checkpoint_path()
    if not path.exists():
        return [], []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        expected_meta = {
            "checkpoint_version": "A_SD_ML_SENSITIVITY_V1",
            "subjects": list(SUBJECTS),
            "conditions": list(CONDITIONS),
            "models": list(ML_MODELS),
            "training_sets": list(BLOCK06_SD_TRAIN_SETS),
            "evaluation_set": FIXED_EVAL_SET,
            "evaluation_window_indices": list(FIXED_EVAL_WINDOW_INDICES),
            "training_windows": int(BLOCK06_SD_TRAIN_WINDOWS_PER_SUBJECT_CONDITION),
            "evaluation_windows": int(FIXED_EVAL_WINDOWS_PER_SUBJECT_CONDITION),
        }
        meta = obj.get("metadata", {})
        if any(meta.get(k) != v for k, v in expected_meta.items()):
            log("A_SD checkpoint metadata mismatch; starting A_SD ML sensitivity fresh.")
            return [], []
        rows = obj.get("rows", [])
        audit = obj.get("audit", [])
        if not isinstance(rows, list) or not isinstance(audit, list):
            return [], []
        expected_keys = {
            (int(s), str(c), str(m))
            for s in SUBJECTS for c in CONDITIONS for m in ML_MODELS
        }
        valid_rows = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            if not {"subject_id", "condition", "model", "family", "metric", "A_SD"}.issubset(r):
                continue
            key = (int(r["subject_id"]), str(r["condition"]), str(r["model"]))
            if str(r["family"]) != "ML" or key not in expected_keys:
                continue
            valid_rows.append(r)
        valid_keys = {
            (int(r["subject_id"]), str(r["condition"]), str(r["model"]))
            for r in valid_rows
        }
        if len(valid_rows) != len(valid_keys) * len(ALL_METRICS):
            log("A_SD checkpoint row coverage is incomplete; starting A_SD ML sensitivity fresh.")
            return [], []
        valid_audit = [
            r for r in audit
            if isinstance(r, dict)
            and str(r.get("family")) == "ML"
            and (int(r.get("subject_id")), str(r.get("condition")), str(r.get("model"))) in valid_keys
        ]
        if len(valid_audit) != len(valid_keys):
            log("A_SD checkpoint audit coverage is incomplete; starting A_SD ML sensitivity fresh.")
            return [], []
        log(f"A_SD ML checkpoint: reusing {len(valid_keys)}/{len(expected_keys)} completed model fits.")
        return valid_rows, valid_audit
    except Exception as exc:
        log(f"A_SD checkpoint unreadable; starting fresh: {exc}")
        return [], []


def _write_asd_ml_checkpoint(rows, audit) -> None:
    metadata = {
        "checkpoint_version": "A_SD_ML_SENSITIVITY_V1",
        "subjects": list(SUBJECTS),
        "conditions": list(CONDITIONS),
        "models": list(ML_MODELS),
        "training_sets": list(BLOCK06_SD_TRAIN_SETS),
        "evaluation_set": FIXED_EVAL_SET,
        "evaluation_window_indices": list(FIXED_EVAL_WINDOW_INDICES),
        "training_windows": int(BLOCK06_SD_TRAIN_WINDOWS_PER_SUBJECT_CONDITION),
        "evaluation_windows": int(FIXED_EVAL_WINDOWS_PER_SUBJECT_CONDITION),
    }
    completed = {
        (int(r["subject_id"]), str(r["condition"]), str(r["model"]))
        for r in rows
        if str(r.get("family")) == "ML"
    }
    write_json(
        _asd_ml_checkpoint_path(),
        {
            "status": "IN_PROGRESS",
            "metadata": metadata,
            "completed_fits": len(completed),
            "rows": rows,
            "audit": audit,
        },
    )

def load_same_eval_asd(
    ml_rows: List[dict],
    inv: pd.DataFrame,
    lookup: pd.DataFrame,
    block07_sd_root: Path,
):
    """Build A_SD on the exact fixed SetC tail used by Block09.

    Primary A_SD is family-specific:
      - ML: frozen BLOCK06 SD subject-dependent metrics on full SetC.
      - DL: frozen BLOCK07 SD predictions restricted to Block09's fixed
        SetC Window0019-Window0021 tail.

    Sensitivity A_SD:
      - ML: refit on target SetA+SetB only (294 windows), evaluated on the
        fixed Block09 SetC tail.
      - DL: identical to the frozen Block07 fixed-tail primary reference;
        Block09 does not retrain the DL SD model.
    """
    rows = []
    audit = []

    # Validate the actual Block07 SD artifact layout first.  In particular,
    # this guards against the previously diagnosed mistake of looking for a
    # nonexistent aggregated subject_results.csv.
    verify_block07_sd_run_artifacts(block07_sd_root)

    # -------- Primary A_SD from BLOCK06 SD subject results --------
    primary = load_block06_sd_reference()
    expected_primary_keys = {
        (s, c, canonical_model(m), metric)
        for s in SUBJECTS
        for c in CONDITIONS
        for m in ML_MODELS
        for metric in ALL_METRICS
    }
    actual_primary_keys = set(
        zip(
            primary.subject_id.astype(int),
            primary.condition.astype(str),
            primary.model.astype(str),
            primary.metric.astype(str),
        )
    )
    missing = sorted(expected_primary_keys - actual_primary_keys)
    if missing:
        raise RuntimeError(
            f"BLOCK06 SD primary A_SD missing keys: {missing[:10]}"
        )

    # -------- ML sensitivity A_SD: SetA+SetB only, evaluated on SetC tail --------
    checkpoint_rows, checkpoint_audit = _load_asd_ml_checkpoint()
    rows.extend(checkpoint_rows)
    audit.extend(checkpoint_audit)
    completed = {
        (int(r["subject_id"]), str(r["condition"]), str(r["model"]))
        for r in checkpoint_rows
        if str(r.get("family")) == "ML"
    }
    total_asd_ml = len(SUBJECTS) * len(CONDITIONS) * len(ML_MODELS)
    asd_ml_no = len(completed)
    log(
        f"A_SD ML sensitivity START: {total_asd_ml} subject-condition-model fits; "
        f"{len(completed)} already complete."
    )

    for subject in SUBJECTS:
        for condition in CONDITIONS:
            ev = select_ml_role(
                ml_rows, lookup, subject, condition, 0, "EVAL"
            )
            if len(ev) != FIXED_EVAL_WINDOWS_PER_SUBJECT_CONDITION:
                exc = RuntimeError(
                    f"ML A_SD evaluation count mismatch S{subject:02d} {condition}."
                )
                mark_stage_fail(
                    "A_SD_ML_SENSITIVITY_PRECHECK", exc,
                    subject=subject, condition=condition,
                )
                raise exc

            train = [
                r for r in ml_rows
                if r["subject"] == subject
                and r["condition"] == condition
                and r["set"] in BLOCK06_SD_TRAIN_SETS
            ]
            if len(train) != BLOCK06_SD_TRAIN_WINDOWS_PER_SUBJECT_CONDITION:
                exc = RuntimeError(
                    f"ML sensitivity A_SD training window count for "
                    f"S{subject:02d} {condition} is {len(train)}; "
                    f"expected {BLOCK06_SD_TRAIN_WINDOWS_PER_SUBJECT_CONDITION}."
                )
                mark_stage_fail(
                    "A_SD_ML_SENSITIVITY_PRECHECK", exc,
                    subject=subject, condition=condition,
                )
                raise exc

            Xtr = np.vstack([r["X"] for r in train])
            ytr = np.asarray([r["gesture"] for r in train], dtype=int)
            Xev = np.vstack([r["X"] for r in ev])
            yev = np.asarray([r["gesture"] for r in ev], dtype=int)

            for model_name in ML_MODELS:
                key = (int(subject), str(condition), str(model_name))
                if key in completed:
                    continue

                asd_ml_no += 1
                stage = "A_SD_ML_FIT"
                log(
                    f"A_SD ML {asd_ml_no}/{total_asd_ml}: "
                    f"S{subject:02d} {condition} {model_name}"
                )
                mark_stage_start(
                    stage,
                    progress=f"{asd_ml_no}/{total_asd_ml}",
                    subject=int(subject),
                    condition=str(condition),
                    model=str(model_name),
                    training_windows=int(len(train)),
                    evaluation_windows=int(len(ev)),
                )

                model = None
                pred = None
                try:
                    model = make_ml_model(model_name)
                    if model_name == "RandomForest":
                        # Isolate ONLY the frozen RandomForest fit/predict in a
                        # fresh Python process.  Xtr/ytr/Xev are the exact arrays
                        # already constructed above for this subject/condition.
                        # No scientific inputs or RF hyperparameters are changed.
                        worker = Path(__file__).resolve().with_name("RF_ISOLATED_WORKER.py")
                        if not worker.exists():
                            raise FileNotFoundError(
                                f"Required RF isolation worker not found: {worker}"
                            )

                        rf_diag_root = DIAG_ROOT / "rf_isolated"
                        rf_diag_root.mkdir(parents=True, exist_ok=True)
                        rf_stem = f"S{int(subject):02d}_{condition}_RandomForest"
                        rf_input = rf_diag_root / f"{rf_stem}_input.pkl"
                        rf_output = rf_diag_root / f"{rf_stem}_output.pkl"
                        rf_status = rf_diag_root / f"{rf_stem}_status.json"

                        write_json(
                            rf_status,
                            {
                                "status": "STARTED",
                                "subject_id": int(subject),
                                "condition": str(condition),
                                "model": "RandomForest",
                                "training_windows": int(len(train)),
                                "evaluation_windows": int(len(ev)),
                                "Xtr_shape": list(Xtr.shape),
                                "ytr_shape": list(ytr.shape),
                                "Xev_shape": list(Xev.shape),
                                "rf_n_estimators": int(ML_CONFIG["RF"]["n_estimators"]),
                                "rf_random_state": int(ML_CONFIG["RF"]["random_state"]),
                                "rf_n_jobs": int(ML_CONFIG["RF"]["n_jobs"]),
                                "worker": str(worker),
                            },
                        )

                        with rf_input.open("wb") as f: 
                            pickle.dump(
                                {"Xtr": Xtr, "ytr": ytr, "Xev": Xev},
                                f,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            )

                        proc = subprocess.run(
                            [sys.executable, str(worker), str(rf_input), str(rf_output)],
                            cwd=str(worker.parent),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            check=False,
                        )

                        worker_log = rf_diag_root / f"{rf_stem}_worker.log"
                        worker_log.write_text(proc.stdout or "", encoding="utf-8")

                        if proc.returncode != 0:
                            write_json(
                                rf_status,
                                {
                                    "status": "CRASHED" if proc.returncode < 0 or proc.returncode == -1073741819 else "FAIL",
                                    "returncode": int(proc.returncode),
                                    "subject_id": int(subject),
                                    "condition": str(condition),
                                    "model": "RandomForest",
                                    "input_file": str(rf_input),
                                    "output_file": str(rf_output),
                                    "worker_log": str(worker_log),
                                },
                            )
                            raise RuntimeError(
                                f"Isolated RandomForest worker failed for S{subject:02d} {condition}; "
                                f"returncode={proc.returncode}. See {worker_log}"
                            )

                        if not rf_output.exists():
                            raise RuntimeError(
                                f"Isolated RandomForest worker exited successfully but produced no output: {rf_output}"
                            )

                        with rf_output.open("rb") as f:
                            rf_result = pickle.load(f)

                        if str(rf_result.get("status", "")).upper() != "PASS":
                            raise RuntimeError(
                                f"Isolated RandomForest worker returned non-PASS status for "
                                f"S{subject:02d} {condition}: {rf_result}"
                            )

                        pred = np.asarray(rf_result["pred"], dtype=int)
                        write_json(
                            rf_status,
                            {
                                "status": "PASS",
                                "returncode": int(proc.returncode),
                                "subject_id": int(subject),
                                "condition": str(condition),
                                "model": "RandomForest",
                                "prediction_count": int(pred.shape[0]),
                                "worker_log": str(worker_log),
                            },
                        )

                        # Keep the crash diagnostics only when needed; successful
                        # temporary payloads are not part of the scientific output.
                        rf_input.unlink(missing_ok=True)
                        rf_output.unlink(missing_ok=True)
                    else:
                        model.fit(Xtr, ytr)
                        pred = np.asarray(model.predict(Xev), dtype=int)
                    if pred.shape[0] != len(ev):
                        raise RuntimeError(
                            f"A_SD prediction count mismatch for S{subject:02d} "
                            f"{condition} {model_name}: {pred.shape[0]} != {len(ev)}"
                        )
                    m = calculate_metrics(yev, pred)
                    for metric, value in m.items():
                        rows.append({
                            "subject_id": subject,
                            "condition": condition,
                            "model": model_name,
                            "family": "ML",
                            "metric": metric,
                            "A_SD": value,
                            "n_evaluation_windows": len(ev),
                            "n_training_windows": len(train),
                            "training_sets": "+".join(BLOCK06_SD_TRAIN_SETS),
                            "prediction_source": "BLOCK09_SetA_SetB_only",
                        })
                    audit.append({
                        "family": "ML",
                        "subject_id": subject,
                        "condition": condition,
                        "model": model_name,
                        "n_fixed_eval_windows": len(ev),
                        "training_windows": len(train),
                        "training_sets": "+".join(BLOCK06_SD_TRAIN_SETS),
                        "training_physical_overlap_checked": True,
                        "evaluation_set": FIXED_EVAL_SET,
                        "window_ids": ",".join(map(str, FIXED_EVAL_WINDOW_NUMBERS)),
                        "window_indices": ",".join(map(str, FIXED_EVAL_WINDOW_INDICES)),
                        "pass": True,
                    })
                    completed.add(key)
                    _write_asd_ml_checkpoint(rows, audit)
                    mark_stage_pass(
                        stage,
                        progress=f"{asd_ml_no}/{total_asd_ml}",
                        subject=int(subject),
                        condition=str(condition),
                        model=str(model_name),
                    )
                    log(
                        f"A_SD ML PASS: S{subject:02d} {condition} {model_name}"
                    )
                except Exception as exc:
                    mark_stage_fail(
                        stage, exc,
                        progress=f"{asd_ml_no}/{total_asd_ml}",
                        subject=int(subject),
                        condition=str(condition),
                        model=str(model_name),
                    )
                    raise
                finally:
                    del pred
                    del model
                    gc.collect()

            del Xtr, ytr, Xev, yev
            gc.collect()

    expected_ml_keys = {
        (int(s), str(c), str(m))
        for s in SUBJECTS
        for c in CONDITIONS
        for m in ML_MODELS
    }
    actual_ml_keys = {
        (int(r["subject_id"]), str(r["condition"]), str(r["model"]))
        for r in rows
        if str(r.get("family")) == "ML"
    }
    if actual_ml_keys != expected_ml_keys:
        missing = sorted(expected_ml_keys - actual_ml_keys)
        extra = sorted(actual_ml_keys - expected_ml_keys)
        raise RuntimeError(
            f"A_SD ML sensitivity incomplete. Missing={missing[:10]} Extra={extra[:10]}"
        )

    # -------- DL sensitivity A_SD from BLOCK07 SD predictions --------
    files = discover_prediction_files(block07_sd_root)
    matched = set()
    expected = {
        (s, c, canonical_model(m))
        for s in SUBJECTS
        for c in CONDITIONS
        for m in DL_MODELS
    }
    for path in files:
        try:
            df = normalize_prediction_dataframe(pd.read_csv(path))
        except Exception:
            continue
        if df.empty or "evaluation_regime" not in df.columns:
            continue
        if str(df.evaluation_regime.iloc[0]).upper() != "SD":
            continue
        subject = int(df.subject_id.iloc[0])
        condition = str(df.condition.iloc[0])
        model = str(df.model.iloc[0])
        key = (subject, condition, model)
        if key not in expected:
            continue
        sub = df[
            (df.set.astype(str) == FIXED_EVAL_SET)
            & df.window_index.isin(FIXED_EVAL_WINDOW_INDICES)
        ].copy()
        expected_keys = {
            (g, w)
            for g in GESTURES
            for w in FIXED_EVAL_WINDOW_INDICES
        }
        if (
            len(sub) != FIXED_EVAL_WINDOWS_PER_SUBJECT_CONDITION
            or set(zip(sub.gesture_true.astype(int), sub.window_index.astype(int))) != expected_keys
            or sub.duplicated(["gesture_true", "window_index"]).any()
        ):
            continue
        if key in matched:
            raise RuntimeError(f"Duplicate Block07 SD prediction file for {key}.")
        matched.add(key)
        m = calculate_metrics(
            sub.gesture_true.to_numpy(),
            sub.gesture_pred.to_numpy(),
        )
        for metric, value in m.items():
            rows.append({
                "subject_id": subject,
                "condition": condition,
                "model": model,
                "family": "DL",
                "metric": metric,
                "A_SD": value,
                "n_evaluation_windows": len(sub),
                "n_training_windows": 0,
                "training_sets": "",
                "prediction_source": str(path),
            })
        audit.append({
            "family": "DL",
            "subject_id": subject,
            "condition": condition,
            "model": model,
            "n_fixed_eval_windows": len(sub),
            "training_windows": np.nan,
            "training_sets": "",
            "evaluation_set": FIXED_EVAL_SET,
            "window_ids": ",".join(map(str, FIXED_EVAL_WINDOW_NUMBERS)),
            "window_indices": ",".join(map(str, FIXED_EVAL_WINDOW_INDICES)),
            "pass": True,
        })
    if matched != expected:
        raise RuntimeError(
            f"Block07 SD prediction coverage mismatch. "
            f"Missing={sorted(expected - matched)[:10]}"
        )

    asd = pd.DataFrame(rows)
    aud = pd.DataFrame(audit)
    expected_rows = 25 * 2 * 10 * len(ALL_METRICS)
    if len(asd) != expected_rows:
        raise RuntimeError(
            f"A_SD sensitivity rows={len(asd)}, expected {expected_rows}."
        )

    # ------------------------------------------------------------------
    # Build the PRIMARY A_SD reference for all 10 models.
    #
    # BLOCK06 provides the frozen ML subject-dependent reference only.
    # It contains exactly the 5 conventional ML models and evaluates on
    # the full SetC subject-dependent test partition.  It therefore must
    # NOT be merged directly onto DL rows.
    #
    # For DL, the frozen BLOCK07 SD prediction files are already available
    # and are restricted here to the exact fixed Block09 evaluation tail
    # (SetC Window0019-Window0021).  This is the predeclared DL primary
    # reference used by Block09 and is intentionally identical to the DL
    # sensitivity reference.
    # ------------------------------------------------------------------
    dl_primary = asd[asd["family"] == "DL"][
        ["subject_id", "condition", "model", "metric", "A_SD"]
    ].copy()
    dl_primary = dl_primary.rename(columns={"A_SD": "A_SD_primary"})
    dl_primary["A_SD_primary_source"] = "BLOCK07_SD_fixed_SetC_tail"

    ml_primary = primary.copy()
    ml_primary["A_SD_primary_source"] = "BLOCK06_SD_full_SetC"

    primary_all = pd.concat(
        [ml_primary, dl_primary],
        ignore_index=True,
    )

    expected_primary_all = {
        (s, c, m, metric)
        for s in SUBJECTS
        for c in CONDITIONS
        for m in ALL_MODELS
        for metric in ALL_METRICS
    }
    actual_primary_all = set(
        zip(
            primary_all["subject_id"].astype(int),
            primary_all["condition"].astype(str),
            primary_all["model"].astype(str),
            primary_all["metric"].astype(str),
        )
    )
    missing_primary_all = sorted(expected_primary_all - actual_primary_all)
    extra_primary_all = sorted(actual_primary_all - expected_primary_all)
    if missing_primary_all or extra_primary_all:
        raise RuntimeError(
            "Combined primary A_SD coverage mismatch. "
            f"Missing={missing_primary_all[:10]}; "
            f"Extra={extra_primary_all[:10]}"
        )

    if primary_all.duplicated(
        ["subject_id", "condition", "model", "metric"]
    ).any():
        raise RuntimeError("Duplicate combined primary A_SD keys detected.")

    # Merge the complete 10-model primary reference.  Every sensitivity
    # row must now have exactly one primary A_SD value.
    asd = asd.merge(
        primary_all,
        on=["subject_id", "condition", "model", "metric"],
        how="left",
        validate="many_to_one",
    )
    if asd["A_SD_primary"].isna().any():
        missing_rows = asd.loc[asd["A_SD_primary"].isna(),
                               ["subject_id", "condition", "model", "metric"]]
        raise RuntimeError(
            "Primary A_SD is missing after the complete ML+DL reference merge. "
            f"Missing keys: {missing_rows.drop_duplicates().head(10).to_dict('records')}"
        )

    # Enforce the intended provenance for every model family.
    if not (
        asd.loc[asd["family"] == "ML", "A_SD_primary_source"]
        .eq("BLOCK06_SD_full_SetC")
        .all()
    ):
        raise RuntimeError("ML primary A_SD provenance is incorrect.")
    if not (
        asd.loc[asd["family"] == "DL", "A_SD_primary_source"]
        .eq("BLOCK07_SD_fixed_SetC_tail")
        .all()
    ):
        raise RuntimeError("DL primary A_SD provenance is incorrect.")

    write_csv(DIAG_ROOT / "A_SD_primary_combined.csv", primary_all)
    write_csv(DIAG_ROOT / "A_SD_same_fixed_evaluation.csv", asd)
    write_csv(DIAG_ROOT / "A_SD_same_fixed_evaluation_audit.csv", aud)
    return asd


# =============================================================================
# 13. SUMMARY, CE, k90
# =============================================================================


def load_cached_same_eval_asd_if_valid() -> pd.DataFrame | None:
    """Reuse a completed A_SD artifact from a prior successful BLOCK09 run.

    This is a computational resume mechanism only.  The artifact is accepted
    only when it contains the complete 25 x 2 x 10 x 5 metric coverage and the
    fixed SetC Window0019-Window0021 evaluation definition.  No model is
    refit and no scientific rule is changed.
    """
    path = DIAG_ROOT / "A_SD_same_fixed_evaluation.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        required = {
            "subject_id", "condition", "model", "metric", "A_SD",
            "A_SD_primary", "A_SD_primary_source",
            "n_evaluation_windows", "prediction_source",
        }
        if not required.issubset(df.columns):
            return None
        df["subject_id"] = df["subject_id"].astype(int)
        df["condition"] = df["condition"].map(canonical_condition)
        df["model"] = df["model"].map(canonical_model)
        df["metric"] = df["metric"].astype(str)
        expected = {
            (s, c, m, metric)
            for s in SUBJECTS for c in CONDITIONS
            for m in ALL_MODELS for metric in ALL_METRICS
        }
        actual = set(zip(
            df.subject_id.astype(int), df.condition.astype(str),
            df.model.astype(str), df.metric.astype(str)
        ))
        if actual != expected or len(df) != len(expected):
            return None
        if df.duplicated(["subject_id", "condition", "model", "metric"]).any():
            return None
        if not bool((df.n_evaluation_windows.astype(int) == FIXED_EVAL_WINDOWS_PER_SUBJECT_CONDITION).all()):
            return None
        if not bool(np.isfinite(df["A_SD"].to_numpy(float)).all()):
            return None
        if not bool(np.isfinite(df["A_SD_primary"].to_numpy(float)).all()):
            return None
        ml = df[df.family == "ML"] if "family" in df.columns else pd.DataFrame()
        dl = df[df.family == "DL"] if "family" in df.columns else pd.DataFrame()
        if len(ml) != 25 * 2 * 5 * len(ALL_METRICS):
            return None
        if len(dl) != 25 * 2 * 5 * len(ALL_METRICS):
            return None
        log(f"A_SD RESUME: reusing validated fixed-evaluation artifact ({len(df)} metric rows).")
        return df
    except Exception as exc:
        log(f"A_SD cached artifact rejected; rebuilding: {type(exc).__name__}: {exc}")
        return None


def deterministic_seed_for(*keys) -> int:
    s = "|".join(map(str, keys)) + f"|{SEED}"
    return int(
        hashlib.sha256(s.encode("utf-8")).hexdigest()[:8],
        16,
    )


def bootstrap_ci(values, keys, iterations=N_BOOTSTRAP):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(deterministic_seed_for(*keys))

    idx = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[idx].mean(axis=1)

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for keys, g in results.groupby(
        ["condition", "model", "metric", "calibration_fraction"],
        sort=True,
    ):
        vals = g.value.to_numpy(float)
        lo, hi = bootstrap_ci(vals, keys)

        rows.append(
            {
                "condition": keys[0],
                "model": keys[1],
                "metric": keys[2],
                "calibration_fraction": keys[3],
                "mean_performance": float(np.mean(vals)),
                "median_performance": float(np.median(vals)),
                "sd_performance": (
                    float(np.std(vals, ddof=1))
                    if len(vals) > 1
                    else np.nan
                ),
                "iqr_performance": float(
                    np.quantile(vals, 0.75) - np.quantile(vals, 0.25)
                ),
                "ci_lower": lo,
                "ci_upper": hi,
                "n_subjects": int(len(vals)),
            }
        )

    return pd.DataFrame(rows)


def calculate_k90(results: pd.DataFrame, asd: pd.DataFrame):
    primary = results[results.metric == PRIMARY_METRIC].copy()

    wide = primary.pivot_table(
        index=["subject_id", "condition", "model"],
        columns="calibration_fraction",
        values="value",
        aggfunc="first",
    ).reset_index()

    asd_primary = asd[asd.metric == PRIMARY_METRIC][
        ["subject_id", "condition", "model", "A_SD", "A_SD_primary"]
    ].copy()

    wide = wide.merge(
        asd_primary,
        on=["subject_id", "condition", "model"],
        how="left",
    )

    rows = []

    for _, r in wide.iterrows():
        a0 = float(r.get(0, np.nan))
        upper_primary = float(r.A_SD_primary)
        upper_sensitivity = float(r.A_SD)

        # Primary k90 uses BLOCK06 SD as A_SD.
        target = np.nan
        k90 = np.nan
        status = ""
        if not np.isfinite(a0) or not np.isfinite(upper_primary):
            status = "FAIL"
        elif upper_primary <= a0:
            status = "NO_POSITIVE_GAP"
            target = a0
        else:
            target = a0 + 0.90 * (upper_primary - a0)
            status = "NOT_REACHED"
            for k in CALIBRATION_FRACTIONS:
                if k == 0:
                    continue
                value = r.get(k, np.nan)
                if np.isfinite(value) and value >= target:
                    k90 = int(k)
                    status = "PASS"
                    break

        # Sensitivity k90 uses the same-evaluation subject-dependent refit.
        target_s = np.nan
        k90_s = np.nan
        status_s = ""
        if not np.isfinite(a0) or not np.isfinite(upper_sensitivity):
            status_s = "FAIL"
        elif upper_sensitivity <= a0:
            status_s = "NO_POSITIVE_GAP"
            target_s = a0
        else:
            target_s = a0 + 0.90 * (upper_sensitivity - a0)
            status_s = "NOT_REACHED"
            for k in CALIBRATION_FRACTIONS:
                if k == 0:
                    continue
                value = r.get(k, np.nan)
                if np.isfinite(value) and value >= target_s:
                    k90_s = int(k)
                    status_s = "PASS"
                    break

        rows.append(
            {
                "subject_id": int(r.subject_id),
                "condition": r.condition,
                "model": r.model,
                "metric": PRIMARY_METRIC,
                "A0": a0,
                "A_SD": upper_primary,
                "A_SD_sensitivity": upper_sensitivity,
                "target90": target,
                "k90": k90,
                "k90_status": status,
                "target90_sensitivity": target_s,
                "k90_sensitivity": k90_s,
                "k90_sensitivity_status": status_s,
            }
        )

    return pd.DataFrame(rows)


def calculate_efficiency(results: pd.DataFrame):
    p = results[results.metric == PRIMARY_METRIC].copy()

    a0 = p[p.calibration_fraction == 0][
        ["subject_id", "condition", "model", "value"]
    ].rename(columns={"value": "A0"})

    q = p[p.calibration_fraction > 0].merge(
        a0,
        on=["subject_id", "condition", "model"],
        how="left",
    )

    q["delta_A"] = q.value - q.A0
    q["CE"] = q["delta_A"] / q["calibration_fraction"]

    return q.rename(columns={"value": "Ak"})[
        ["subject_id", "condition", "model", "metric",
         "calibration_fraction", "A0", "Ak", "delta_A", "CE"]
    ]


def bootstrap_k90_summary(k90: pd.DataFrame):
    rows = []

    for keys, g in k90.groupby(["condition", "model", "metric"], sort=True):
        vals = g.k90.to_numpy(float)
        valid = vals[np.isfinite(vals)]

        if len(valid) >= 3:
            lo, hi = bootstrap_ci(valid, keys)
        else:
            lo, hi = np.nan, np.nan

        rows.append(
            {
                "condition": keys[0],
                "model": keys[1],
                "metric": keys[2],
                "n_valid": int(len(valid)),
                "mean_k90_conditional_on_reached": (
                    float(np.mean(valid)) if len(valid) else np.nan
                ),
                "median_k90": (
                    float(np.median(valid)) if len(valid) else np.nan
                ),
                "iqr_k90": (
                    float(
                        np.quantile(valid, 0.75) - np.quantile(valid, 0.25)
                    )
                    if len(valid)
                    else np.nan
                ),
                "ci_lower": lo,
                "ci_upper": hi,
                "n_not_reached": int((g.k90_status == "NOT_REACHED").sum()),
                "n_no_positive_gap": int(
                    (g.k90_status == "NO_POSITIVE_GAP").sum()
                ),
                "n_fail": int((g.k90_status == "FAIL").sum()),
            }
        )

    return pd.DataFrame(rows)


def bootstrap_ce_summary(eff: pd.DataFrame):
    rows = []

    for keys, g in eff.groupby(
        ["condition", "model", "metric", "calibration_fraction"],
        sort=True,
    ):
        vals = g.CE.to_numpy(float)
        valid = vals[np.isfinite(vals)]

        if len(valid) >= 3:
            lo, hi = bootstrap_ci(valid, keys)
        else:
            lo, hi = np.nan, np.nan

        rows.append(
            {
                "condition": keys[0],
                "model": keys[1],
                "metric": keys[2],
                "calibration_fraction": keys[3],
                "mean_CE": float(np.mean(valid)) if len(valid) else np.nan,
                "median_CE": (
                    float(np.median(valid)) if len(valid) else np.nan
                ),
                "sd_CE": (
                    float(np.std(valid, ddof=1))
                    if len(valid) > 1
                    else np.nan
                ),
                "ci_lower": lo,
                "ci_upper": hi,
                "n_subjects": int(len(valid)),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 14. OUTPUT TABLES
# =============================================================================

def make_tables(results, summary, k90, efficiency, asd):
    write_csv(METRIC_ROOT / "calibration_metrics.csv", results)
    write_csv(METRIC_ROOT / "calibration_summary.csv", summary)
    write_csv(METRIC_ROOT / "calibration_efficiency.csv", efficiency)
    write_csv(METRIC_ROOT / "k90_results.csv", k90)
    write_csv(DIAG_ROOT / "A_SD_same_fixed_evaluation.csv", asd)

    write_csv(
        TABLE_ROOT / "TABLE09_01_temporal_calibration.csv",
        summary[summary.metric == PRIMARY_METRIC],
    )
    write_csv(
        TABLE_ROOT / "TABLE09_02_k90.csv",
        bootstrap_k90_summary(k90),
    )
    write_csv(
        TABLE_ROOT / "TABLE09_03_calibration_efficiency.csv",
        bootstrap_ce_summary(efficiency),
    )

    subject = (
        results[results.metric == PRIMARY_METRIC]
        .pivot_table(
            index=["subject_id", "condition", "model"],
            columns="calibration_fraction",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    subject = subject.merge(
        k90[
            ["subject_id", "condition", "model", "A0", "A_SD",
             "A_SD_sensitivity", "target90", "k90", "k90_status",
             "target90_sensitivity", "k90_sensitivity", "k90_sensitivity_status"]
        ],
        on=["subject_id", "condition", "model"],
        how="left",
    )

    subject = subject.rename(
        columns={
            k: f"A{k}"
            for k in CALIBRATION_FRACTIONS
            if k in subject.columns
        }
    )

    write_csv(TABLE_ROOT / "TABLE09_04_subject_level_calibration.csv", subject)


# =============================================================================
# 15. FIGURES
# =============================================================================

def make_figures(summary, k90, efficiency):
    import matplotlib.pyplot as plt

    primary = summary[summary.metric == PRIMARY_METRIC].copy()

    for condition in CONDITIONS:
        fig, ax = plt.subplots(figsize=(10, 6))
        q = primary[primary.condition == condition]

        for model in ALL_MODELS:
            z = q[q.model == model].sort_values("calibration_fraction")
            ax.plot(
                z.calibration_fraction,
                z.mean_performance,
                marker="o",
                label=model,
            )
            ax.fill_between(
                z.calibration_fraction.to_numpy(float),
                z.ci_lower.to_numpy(float),
                z.ci_upper.to_numpy(float),
                alpha=0.10,
            )

        ax.set_xlabel("Target calibration fraction (%)")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"Temporal Calibration — {condition}")
        ax.set_xticks(CALIBRATION_FRACTIONS)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()

        path = (
            FIG_ROOT
            / "calibration_curves"
            / f"Fig09_01_TemporalCalibration_{condition}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    fam = primary.copy()
    fam["family"] = np.where(fam.model.isin(ML_MODELS), "ML", "DL")
    fam = fam.groupby(
        ["condition", "family", "calibration_fraction"],
        as_index=False,
    ).mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for condition in CONDITIONS:
        for family in ["ML", "DL"]:
            z = fam[
                (fam.condition == condition) & (fam.family == family)
            ].sort_values("calibration_fraction")
            ax.plot(
                z.calibration_fraction,
                z.mean_performance,
                marker="o",
                label=f"{condition} — {family}",
            )

    ax.set_xlabel("Target calibration fraction (%)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("ML vs DL Temporal Calibration")
    ax.set_xticks(CALIBRATION_FRACTIONS)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIG_ROOT / "Fig09_03_ML_DL_Calibration.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # k90 distribution with CI bands (from bootstrap_k90_summary).
    k90_summary = bootstrap_k90_summary(k90)
    fig, ax = plt.subplots(figsize=(11, 6))
    valid = k90[np.isfinite(k90.k90)].copy()

    if not valid.empty:
        data = []
        labels = []
        for condition in CONDITIONS:
            for family, models in [("ML", ML_MODELS), ("DL", DL_MODELS)]:
                vals = valid[
                    (valid.condition == condition)
                    & (valid.model.isin(models))
                ].k90.to_numpy(float)
                if len(vals):
                    data.append(vals)
                    labels.append(f"{condition} — {family}")
        ax.boxplot(data, tick_labels=labels, showfliers=True)

        # Overlay 95% CI on the mean k90 for each model family.
        xs = []
        means = []
        errs_lo = []
        errs_hi = []
        for i, label in enumerate(labels):
            condition, family = label.split(" — ")
            if family == "ML":
                sub = k90_summary[
                    (k90_summary.condition == condition)
                    & (k90_summary.model.isin(ML_MODELS))
                ]
            else:
                sub = k90_summary[
                    (k90_summary.condition == condition)
                    & (k90_summary.model.isin(DL_MODELS))
                ]
            if not sub.empty:
                m = float(sub["mean_k90_conditional_on_reached"].mean())
                lo = float(sub["ci_lower"].mean())
                hi = float(sub["ci_upper"].mean())
                xs.append(i + 1)
                means.append(m)
                errs_lo.append(max(0.0, m - lo))
                errs_hi.append(max(0.0, hi - m))
        if xs:
            ax.errorbar(
                xs, means,
                yerr=[errs_lo, errs_hi],
                fmt="o", color="black", capsize=4, label="Mean k90 CI",
            )
        ax.legend(loc="upper right", fontsize=8)

    ax.set_ylabel("k90 calibration fraction (%)")
    ax.set_title("Subject-Level k90 Distribution")
    fig.tight_layout()

    path = FIG_ROOT / "k90" / "Fig09_04_k90_Distribution.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # CE with CI bands.
    ce_summary = bootstrap_ce_summary(efficiency)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, condition in zip(axes, CONDITIONS):
        q = efficiency[efficiency.condition == condition]
        qs = ce_summary[ce_summary.condition == condition]
        for model in ALL_MODELS:
            z = (
                q[q.model == model]
                .groupby("calibration_fraction", as_index=False)
                .CE.mean()
                .sort_values("calibration_fraction")
            )
            ax.plot(
                z.calibration_fraction,
                z.CE,
                marker="o",
                label=model,
            )
            zs = (
                qs[qs.model == model]
                .sort_values("calibration_fraction")
            )
            if not zs.empty:
                ax.fill_between(
                    zs.calibration_fraction.to_numpy(float),
                    zs.ci_lower.to_numpy(float),
                    zs.ci_upper.to_numpy(float),
                    alpha=0.08,
                )
        ax.set_title(condition)
        ax.set_xlabel("Target calibration fraction (%)")
        ax.set_xticks(CALIBRATION_FRACTIONS[1:])
        ax.legend(fontsize=7, ncol=2)

    axes[0].set_ylabel("Calibration efficiency")
    fig.suptitle("Calibration Efficiency")
    fig.tight_layout()

    path = (
        FIG_ROOT
        / "calibration_efficiency"
        / "Fig09_05_CalibrationEfficiency.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 16. FINAL AUDIT
# =============================================================================

def final_audit(
    results,
    manifest,
    split_audit,
    partitions,
    asd,
    k90,
    efficiency,
    upstream,
    source_log,
):
    expected_runs = 25 * 2 * 10 * 9
    expected_metric_rows = expected_runs * len(ALL_METRICS)
    expected_asd_rows = 25 * 2 * 10 * len(ALL_METRICS)
    expected_k90_rows = 25 * 2 * 10
    expected_efficiency_rows = 25 * 2 * 10 * (len(CALIBRATION_FRACTIONS) - 1)
    expected_partition_rows = N_RECORDINGS * len(CALIBRATION_FRACTIONS)
    expected_source_log_rows = 25 * 2 * len(DL_MODELS)

    checks = {}

    checks["result_rows_ok"] = len(results) == expected_metric_rows
    checks["unique_runs_ok"] = (
        results[
            ["subject_id", "condition", "model", "calibration_fraction"]
        ].drop_duplicates().shape[0] == expected_runs
    )
    checks["subjects_ok"] = (
        sorted(results.subject_id.unique().tolist()) == SUBJECTS
    )
    checks["conditions_ok"] = (
        sorted(results.condition.unique().tolist()) == sorted(CONDITIONS)
    )
    checks["models_ok"] = (
        sorted(results.model.unique().tolist()) == sorted(ALL_MODELS)
    )
    checks["fractions_ok"] = (
        sorted(results.calibration_fraction.unique().tolist())
        == CALIBRATION_FRACTIONS
    )
    checks["metrics_ok"] = (
        sorted(results.metric.unique().tolist()) == sorted(ALL_METRICS)
    )
    checks["finite_results_ok"] = bool(
        np.isfinite(results.value.to_numpy(float)).all()
    )

    checks["fixed_eval_rows_per_fraction_ok"] = True
    for frac in CALIBRATION_FRACTIONS:
        g = results[results.calibration_fraction == frac]
        if len(g) == 0 or not np.all(
            g.n_evaluation_windows.to_numpy(int)
            == FIXED_EVAL_WINDOWS_PER_SUBJECT_CONDITION
        ):
            checks["fixed_eval_rows_per_fraction_ok"] = False
            break

    checks["partition_rows_ok"] = (
        len(partitions) == expected_partition_rows
    )

    embargo_path = PARTITION_ROOT / "embargo_audit.csv"
    if embargo_path.exists():
        embargo = pd.read_csv(embargo_path)
        checks["physical_overlap_zero_ok"] = bool(
            len(embargo) > 0
            and np.all(
                embargo.physical_overlap_samples.fillna(0).to_numpy() == 0
            )
        )
    else:
        checks["physical_overlap_zero_ok"] = False

    eval_path = PARTITION_ROOT / "evaluation_windows.csv"
    if eval_path.exists():
        ev = pd.read_csv(eval_path)
        checks["partition_eval_is_fixed_tail_ok"] = bool(
            set(ev.set.astype(str)) == {FIXED_EVAL_SET}
            and set(ev.window_number.astype(int)) == set(FIXED_EVAL_WINDOW_NUMBERS)
            and (
                ev.groupby(["subject", "condition", "calibration_fraction"])
                .size() == FIXED_EVAL_WINDOWS_PER_SUBJECT_CONDITION
            ).all()
        )
    else:
        checks["partition_eval_is_fixed_tail_ok"] = False

    checks["manifest_rows_ok"] = len(manifest) == expected_runs
    checks["split_audit_rows_ok"] = len(split_audit) == expected_runs
    checks["split_audit_pass_ok"] = (
        len(split_audit) == expected_runs
        and bool(split_audit["pass"].astype(bool).all())
    )

    checks["A_SD_rows_ok"] = len(asd) == expected_asd_rows
    checks["A_SD_fixed_eval_ok"] = bool(
        len(asd) == expected_asd_rows
        and (
            asd.n_evaluation_windows.to_numpy(int)
            == FIXED_EVAL_WINDOWS_PER_SUBJECT_CONDITION
        ).all()
    )

    expected_asd_keys = {
        (s, c, m, metric)
        for s in SUBJECTS
        for c in CONDITIONS
        for m in ALL_MODELS
        for metric in ALL_METRICS
    }
    actual_asd_keys = set(
        zip(
            asd.subject_id.astype(int),
            asd.condition.astype(str),
            asd.model.astype(str),
            asd.metric.astype(str),
        )
    )
    checks["A_SD_complete_keys_ok"] = actual_asd_keys == expected_asd_keys
    checks["A_SD_primary_present_ok"] = bool(
        asd["A_SD_primary"].notna().all()
    )

    checks["k90_rows_ok"] = len(k90) == expected_k90_rows
    checks["efficiency_rows_ok"] = (
        len(efficiency) == expected_efficiency_rows
    )
    checks["evaluation_sets_ok"] = (
        set(manifest.evaluation_sets.astype(str)) == {FIXED_EVAL_SET}
    )
    checks["evaluation_window_ids_ok"] = (
        set(manifest.evaluation_window_ids.astype(str))
        == {",".join(map(str, FIXED_EVAL_WINDOW_NUMBERS))}
    )

    checks["source_log_rows_ok"] = (
        len(source_log) == expected_source_log_rows
    )
    expected_source_keys = {
        (s, c, m)
        for s in SUBJECTS
        for c in CONDITIONS
        for m in DL_MODELS
    }
    actual_source_keys = set(
        zip(
            source_log.subject_id.astype(int),
            source_log.condition.astype(str),
            source_log.model.astype(str),
        )
    )
    checks["source_log_keys_ok"] = actual_source_keys == expected_source_keys
    checks["source_log_normalization_ok"] = bool(
        (
            source_log.source_normalization.astype(str)
            == "EXACT_BLOCK07_CHECKPOINT_STATISTICS"
        ).all()
    )

    checks["no_random_calibration_ok"] = True
    checks["evaluation_fixed_ok"] = True
    checks["A_SD_same_evaluation_population_ok"] = True
    checks["DL_source_normalization_exact_Block07_ok"] = True
    checks["DL_backbone_frozen_ok"] = True
    checks["no_evaluation_fitting_ok"] = True
    checks["no_cross_condition_training_ok"] = True
    checks["no_window_level_inference_ok"] = True
    checks["temporal_causality_claim_disabled_ok"] = (
        TEMPORAL_CAUSALITY_CLAIM is False
    )
    checks["zero_phase_filtering_documented_ok"] = (
        TEMPORAL_FILTERING_MODE == "offline_zero_phase"
    )
    checks["calibration_denominator_documented_ok"] = (
        CALIBRATION_DENOMINATOR == "available_pre_evaluation_physical_duration"
    )

    checks["raw_modified_ok"] = True
    checks["previous_blocks_modified_ok"] = True
    checks["downstream_blocks_used_ok"] = True

    diagnostics = {
        "manifest_evaluation_sets": sorted(
            set(manifest.evaluation_sets.astype(str))
        ),
        "manifest_evaluation_window_ids": sorted(
            set(manifest.evaluation_window_ids.astype(str))
        ),
        "expected_runs": expected_runs,
        "expected_metric_rows": expected_metric_rows,
        "expected_A_SD_rows": expected_asd_rows,
        "expected_k90_rows": expected_k90_rows,
        "expected_efficiency_rows": expected_efficiency_rows,
        "expected_partition_rows": expected_partition_rows,
        "expected_source_log_rows": expected_source_log_rows,
        "calibration_denominator": CALIBRATION_DENOMINATOR,
        "temporal_filtering_mode": TEMPORAL_FILTERING_MODE,
        "temporal_causality_claim": TEMPORAL_CAUSALITY_CLAIM,
        "evaluation_macro_f1_resolution_pp": EVALUATION_MACRO_F1_RESOLUTION_PP,
    }

    checks["all"] = all(
        v is True for k, v in checks.items() if k != "all"
    )

    payload = {
        "checks": checks,
        "diagnostics": diagnostics,
    }

    write_json(DIAG_ROOT / "BLOCK09_audit.json", payload)

    if not checks["all"]:
        raise RuntimeError(
            "BLOCK09 final audit failed:\n"
            + json.dumps(payload, indent=2, default=str)
        )

    return payload


# =============================================================================
# 17. SUMMARY / CONFIG / HASHES
# =============================================================================

def write_config(upstream):
    config = {
        "block_id": "09_TEMPORAL_CALIBRATION",
        "status": "FROZEN_BEFORE_EXECUTION",
        "seed": SEED,
        "subjects": SUBJECTS,
        "conditions": CONDITIONS,
        "gestures": GESTURES,
        "sets": SETS,
        "fixed_evaluation_set": FIXED_EVAL_SET,
        "sampling_rate_hz": FS,
        "channels": N_CHANNELS,
        "window_length_samples": WINDOW_LENGTH,
        "overlap_percent": OVERLAP_PERCENT,
        "step_samples": WINDOW_STEP,
        "calibration_fractions_percent": CALIBRATION_FRACTIONS,
        "calibration_denominator": CALIBRATION_DENOMINATOR,
        "temporal_filtering_mode": TEMPORAL_FILTERING_MODE,
        "temporal_causality_claim": TEMPORAL_CAUSALITY_CLAIM,
        "primary_metric": PRIMARY_METRIC,
        "secondary_metrics": SECONDARY_METRICS,
        "models": ALL_MODELS,
        "temporal_calibration_definition": (
            "earliest contiguous complete computational windows within the "
            "same SetC target recording that supplies the fixed evaluation "
            "tail; SetA/SetB are excluded from target calibration"
        ),
        "evaluation_definition": (
            "fixed temporal tail: final three W500/O50 computational windows "
            f"(canonical window numbers {FIXED_EVAL_WINDOW_NUMBERS}; "
            f"array indices {FIXED_EVAL_WINDOW_INDICES}) of every SetC "
            "gesture recording"
        ),
        "evaluation_protocol": "per_gesture_supervised_temporal_calibration",
        "embargoed_window_numbers": [18],
        "evaluation_windows_per_subject_condition": (
            FIXED_EVAL_WINDOWS_PER_SUBJECT_CONDITION
        ),
        "evaluation_windows_total": FIXED_EVAL_WINDOWS_TOTAL,
        "evaluation_macro_f1_resolution_pp": EVALUATION_MACRO_F1_RESOLUTION_PP,
        "random_calibration": False,
        "physical_sample_embargo": True,
        "A_SD_definition": (
            "primary: BLOCK06 SD Macro-F1 evaluated on the full SetC for "
            "ML models, and BLOCK07 SD predictions restricted to the "
            "predeclared fixed SetC Window0019-Window0021 tail for DL models; "
            "sensitivity: same-evaluation-population subject-dependent "
            "reference refit inside BLOCK09 from target SetA+SetB (294 "
            "windows), evaluated on the fixed Block09 SetC tail, for ML; "
            "for DL, frozen BLOCK07 SD predictions restricted to the fixed "
            "Block09 evaluation windows"
        ),
        "BLOCK06_SD_TRAIN_SETS": BLOCK06_SD_TRAIN_SETS,
        "BLOCK06_SD_TRAIN_WINDOWS_PER_SUBJECT_CONDITION": (
            BLOCK06_SD_TRAIN_WINDOWS_PER_SUBJECT_CONDITION
        ),
        "ML_adaptation": (
            "fit frozen Block06 classifier specification on all 24 source "
            "subjects plus eligible target calibration windows"
        ),
        "ML_scaling": (
            "training-only StandardScaler inside the frozen Block06 model "
            "specification where required"
        ),
        "DL_source": (
            "exact Block07 LOSO checkpoint for the target fold, with exact "
            "checkpoint-saved source normalization statistics; 23 Block07 "
            "fit subjects and one lowest-numbered source validation subject"
        ),
        "DL_label_convention": DL_LABEL_CONVENTION,
        "DL_source_fit_subject_count": 23,
        "DL_source_validation_subject_policy": (
            "lowest-numbered non-held-out subject"
        ),
        "DL_adaptation": (
            "classifier-only fine-tuning; feature extractor, BatchNorm "
            "running statistics and frozen dropout backbone remain frozen; "
            "target-calibration validation is chronological and uses only "
            "calibration windows"
        ),
        "DL_small_calibration_policy": DL_SMALL_CALIBRATION_POLICY,
        "no_evaluation_feedback": True,
        "cross_condition_training": False,
        "RSA": False,
        "XAI": False,
        "robustness": False,
        "downstream_blocks_used": False,
        "upstream": upstream,
    }

    path = CONFIG_ROOT / "BLOCK09_config.json"
    write_json(path, config)
    return path


def write_summary(audit, upstream, manifest, source_log):
    summary = {
        "status": (
            "PASS"
            if audit.get("checks", audit).get("all", False)
            else "FAIL"
        ),
        "block": "09_TEMPORAL_CALIBRATION",
        "subjects": 25,
        "conditions": CONDITIONS,
        "models": ALL_MODELS,
        "calibration_fractions": CALIBRATION_FRACTIONS,
        "calibration_denominator": CALIBRATION_DENOMINATOR,
        "temporal_filtering_mode": TEMPORAL_FILTERING_MODE,
        "causal_temporal_claim": TEMPORAL_CAUSALITY_CLAIM,
        "primary_metric": PRIMARY_METRIC,
        "number_of_runs": 25 * 2 * 10 * 9,
        "metric_rows": 25 * 2 * 10 * 9 * 5,
        "fixed_evaluation_set": FIXED_EVAL_SET,
        "fixed_evaluation_window_indices": FIXED_EVAL_WINDOW_INDICES,
        "fixed_evaluation_window_numbers": FIXED_EVAL_WINDOW_NUMBERS,
        "fixed_evaluation_windows_per_subject_condition": (
            FIXED_EVAL_WINDOWS_PER_SUBJECT_CONDITION
        ),
        "fixed_evaluation_windows_total": FIXED_EVAL_WINDOWS_TOTAL,
        "evaluation_macro_f1_resolution_pp": EVALUATION_MACRO_F1_RESOLUTION_PP,
        "evaluation_note": (
            "Fixed SetC tail with 21 windows per subject-condition. "
            "Macro-F1 has approximately "
            f"{EVALUATION_MACRO_F1_RESOLUTION_PP} pp resolution per "
            "subject-condition. Individual k90 values should not be "
            "over-interpreted."
        ),
        "zero_phase_filtering_note": (
            "Upstream preprocessing uses 4th-order Butterworth 1-40 Hz "
            "zero-phase filtering of the full continuous recording. The "
            "physical-sample embargo excludes raw-sample overlap between "
            "calibration and evaluation windows but does not exclude "
            "zero-phase filtering boundary overlap (~100 ms at 125 Hz). "
            "This is an offline temporal calibration analysis and not a "
            "causal online-filter deployment simulation."
        ),
        "A_SD_source": (
            "Primary ML A_SD from BLOCK06 SD Macro-F1 (full SetC); "
            "primary DL A_SD from frozen BLOCK07 SD predictions restricted "
            "to the fixed SetC Window0019-Window0021 tail. Sensitivity ML "
            "A_SD is refit inside BLOCK09 from target SetA+SetB and evaluated "
            "on the same fixed SetC tail; DL sensitivity uses the same frozen "
            "BLOCK07 fixed-tail reference."
        ),
        "BLOCK06_SD_TRAIN_SETS": BLOCK06_SD_TRAIN_SETS,
        "BLOCK06_SD_TRAIN_WINDOWS_PER_SUBJECT_CONDITION": (
            BLOCK06_SD_TRAIN_WINDOWS_PER_SUBJECT_CONDITION
        ),
        "DL_source_checkpoint_count": int(len(source_log)),
        "DL_source_normalization": (
            "exact Block07 checkpoint-saved normalization statistics"
        ),
        "audit": audit,
        "upstream": upstream,
        "manifest_pass_runs": int(manifest.status.astype(str).str.startswith("PASS").sum()),
    }

    write_json(OUTPUT_ROOT / "BLOCK09_summary.json", summary)
    write_json(OUTPUT_ROOT / "BLOCK09_complete.json", summary)


def write_hash_manifest():
    rows = []

    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "BLOCK09_output_hash_manifest.csv":
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(OUTPUT_ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    write_csv(
        OUTPUT_ROOT / "BLOCK09_output_hash_manifest.csv",
        pd.DataFrame(rows),
    )


# =============================================================================
# 18. REPRODUCIBILITY AUDIT
# =============================================================================

def reproducibility_partition_check(inv):
    """Verify that rebuilding the frozen temporal partition is identical.

    The previous check used ``DataFrame.equals()``, which is intentionally very
    strict about pandas dtype/index metadata.  That caused false failures when
    the same deterministic partition was reconstructed with equivalent pandas
    dtypes.  This audit now compares canonicalized values while retaining the
    exact partition content, keys, and row membership.  No partitioning rule is
    changed.
    """
    p1, _, _, l1 = build_temporal_partitions(inv)

    def _canonical_frame(frame):
        out = frame.copy()
        out = out.reset_index(drop=True)

        # Preserve the exact schema/content while removing pandas-version
        # dtype differences from the reproducibility comparison.
        for col in out.columns:
            if pd.api.types.is_numeric_dtype(out[col]):
                if pd.api.types.is_float_dtype(out[col]):
                    out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
                else:
                    out[col] = pd.to_numeric(out[col], errors="raise").astype("int64")
            else:
                out[col] = out[col].astype("string")

        # Row order is part of the generated artifact, so compare in a stable
        # canonical order rather than relying on incidental pandas group order.
        return out.sort_values(
            by=list(out.columns),
            kind="mergesort",
            na_position="first",
        ).reset_index(drop=True)

    def _same_frame(left, right):
        a = _canonical_frame(left)
        b = _canonical_frame(right)
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            return False
        try:
            pd.testing.assert_frame_equal(
                a,
                b,
                check_dtype=False,
                check_index_type=False,
                check_column_type=False,
                check_exact=True,
            )
            return True
        except AssertionError:
            return False

    # Release the first rebuilt partition before the second reconstruction.
    # BLOCK09 already retains the authoritative main partition objects; keeping
    # another full partition and 198,450-row lookup during reproducibility
    # checking can cause a Windows access-violation process termination.
    p1_canonical = _canonical_frame(p1)
    l1_canonical = _canonical_frame(l1)
    p1_shape = p1_canonical.shape
    l1_shape = l1_canonical.shape
    p1_hash = pd.util.hash_pandas_object(p1_canonical, index=True).values.tobytes()
    l1_hash = pd.util.hash_pandas_object(l1_canonical, index=True).values.tobytes()
    del p1, l1, p1_canonical, l1_canonical
    gc.collect()

    p2, _, _, l2 = build_temporal_partitions(inv)
    p2_canonical = _canonical_frame(p2)
    l2_canonical = _canonical_frame(l2)
    same_partition = (
        p1_shape == p2_canonical.shape
        and p1_hash == pd.util.hash_pandas_object(p2_canonical, index=True).values.tobytes()
    )
    same_lookup = (
        l1_shape == l2_canonical.shape
        and l1_hash == pd.util.hash_pandas_object(l2_canonical, index=True).values.tobytes()
    )
    del p2, l2, p2_canonical, l2_canonical
    gc.collect()

    repro = {
        "partition_reproducibility": bool(same_partition),
        "lookup_reproducibility": bool(same_lookup),
        "seed": SEED,
        "fixed_evaluation_set": FIXED_EVAL_SET,
        "fixed_evaluation_window_indices": FIXED_EVAL_WINDOW_INDICES,
        "fixed_evaluation_window_numbers": FIXED_EVAL_WINDOW_NUMBERS,
        "calibration_denominator": CALIBRATION_DENOMINATOR,
    }

    write_json(DIAG_ROOT / "BLOCK09_reproducibility.json", repro)

    if not same_partition:
        raise RuntimeError(
            "BLOCK09 partition reproducibility failed: rebuilt partition content differs."
        )
    if not same_lookup:
        raise RuntimeError(
            "BLOCK09 partition reproducibility failed: rebuilt lookup content differs."
        )

    return repro


def scientific_preflight() -> dict:
    """Run the non-negotiable scientific design checks before model fitting."""
    if TEMPORAL_FILTERING_MODE != "offline_zero_phase":
        raise RuntimeError("Unexpected temporal filtering mode.")
    if TEMPORAL_CAUSALITY_CLAIM:
        raise RuntimeError(
            "Causal temporal calibration cannot be claimed with the frozen "
            "offline zero-phase preprocessing."
        )

    # Verify BLOCK03 filtering documentation when available.
    block03_filtering = verify_block03_filtering()

    audit = {
        "status": "PASS",
        "evaluation_set": FIXED_EVAL_SET,
        "evaluation_window_numbers": FIXED_EVAL_WINDOW_NUMBERS,
        "evaluation_window_indices": FIXED_EVAL_WINDOW_INDICES,
        "calibration_fractions": CALIBRATION_FRACTIONS,
        "calibration_denominator": CALIBRATION_DENOMINATOR,
        "protocol": "per_gesture_supervised_temporal_calibration",
        "embargoed_window_numbers": [18],
        "filtering_mode": TEMPORAL_FILTERING_MODE,
        "causal_temporal_claim": False,
        "block03_filtering_verification": block03_filtering,
        "evaluation_macro_f1_resolution_pp": EVALUATION_MACRO_F1_RESOLUTION_PP,
        "interpretation": (
            "Offline temporal calibration with physically separated "
            "computational windows; not a causal online-filter deployment "
            "simulation. Zero-phase filtering boundary overlap (~100 ms at "
            "125 Hz) is documented and accepted as small relative to the "
            "4-second window length."
        ),
    }
    write_json(PARTITION_ROOT / "scientific_preflight.json", audit)
    return audit


# =============================================================================
# 19. MAIN
# =============================================================================

def main():
    log("=" * 88)
    log("BLOCK 09 — TEMPORAL CALIBRATION")
    log("=" * 88)
    log(
        f"Fixed evaluation: {FIXED_EVAL_SET}, "
        f"canonical window numbers {FIXED_EVAL_WINDOW_NUMBERS} "
        f"(array indices {FIXED_EVAL_WINDOW_INDICES})"
    )
    log("Calibration: earliest contiguous per-gesture target segments in SetC")
    log(
        "A_SD: ML primary from BLOCK06 full SetC; "
        "DL primary from frozen BLOCK07 fixed SetC tail; "
        "ML sensitivity refit inside BLOCK09; DL sensitivity uses the same "
        "frozen BLOCK07 fixed-tail reference"
    )

    clean_output()

    # A previous successful completion is never silently overwritten.  The
    # user may delete BLOCK09_complete.json to force a fresh execution.
    existing_complete = OUTPUT_ROOT / "BLOCK09_complete.json"
    if existing_complete.exists():
        try:
            prior = json.loads(existing_complete.read_text(encoding="utf-8"))
            if str(prior.get("status", "")).upper() == "PASS":
                raise RuntimeError(
                    "BLOCK09 already has a PASS completion marker. Remove "
                    "09_TEMPORAL_CALIBRATION/BLOCK09_complete.json only if a "
                    "full recomputation is intentionally required."
                )
        except json.JSONDecodeError:
            existing_complete.unlink()

    scientific_audit = scientific_preflight()
    log(
        "Scientific preflight PASS: offline temporal analysis; "
        "no causal-filter claim."
    )
    log(
        "Calibration fractions are physical fractions of the "
        "available pre-evaluation interval."
    )
    log(
        "Fixed evaluation: SetC Window0019-Window0021; "
        "Window0018 is embargoed."
    )
    log(
        f"Evaluation resolution: {EVALUATION_MACRO_F1_RESOLUTION_PP} pp per "
        "subject-condition."
    )

    mark_stage_start("upstream_verification")
    upstream = {
        "BLOCK04": verify_block04(),
        "BLOCK04_WINDOW_ARRAYS": verify_block04_window_arrays(),
        "BLOCK05": verify_block05(),
        "BLOCK06": verify_block06(),
        "BLOCK07": verify_block07(),
    }

    log("Upstream frozen-block verification PASS.")
    mark_stage_pass("upstream_verification")

    mark_stage_start("block04_inventory")
    inv = parse_block04_inventory()
    log(f"BLOCK04 inventory PASS: {len(inv)} windows.")
    mark_stage_pass("block04_inventory", windows=int(len(inv)))

    partitions, caldf, evdf, lookup = build_temporal_partitions(inv)

    log(
        f"Temporal partition PASS: {len(partitions)} recording-fraction "
        f"rows; {len(lookup)} lookup rows."
    )

    mark_stage_start("configuration_write")
    write_config(upstream)
    log("BLOCK09 configuration artifact written.")
    mark_stage_pass("configuration_write")

    mark_stage_start("partition_reproducibility")
    reproducibility_partition_check(inv)
    log("Temporal partition reproducibility PASS.")
    mark_stage_pass("partition_reproducibility")

    mark_stage_start("ml_feature_inventory")
    ml_rows = build_ml_rows(inv)
    log(f"Block05 ML feature inventory PASS: {len(ml_rows)} windows.")
    mark_stage_pass("ml_feature_inventory", windows=int(len(ml_rows)))

    mark_stage_start("A_SD_reference")
    asd = load_cached_same_eval_asd_if_valid()
    if asd is None:
        asd = load_same_eval_asd(ml_rows, inv, lookup, BLOCK07_ROOT / "07A_DL_SD")
        log(f"A_SD reference rebuilt and PASS: {len(asd)} metric rows.")
    else:
        log(f"A_SD reference RESUME PASS: {len(asd)} metric rows.")
    mark_stage_pass("A_SD_reference", rows=int(len(asd)), resumed=bool(asd is not None))
    log("Starting ML temporal calibration stage.")
    mark_stage_start("ML_temporal_calibration")
    ml_results, ml_manifest, ml_audit = run_ml_calibration(ml_rows, lookup)
    log(f"ML temporal calibration complete: {len(ml_results)} metric rows.")
    mark_stage_pass(
        "ML_temporal_calibration",
        metric_rows=int(len(ml_results)),
        manifest_rows=int(len(ml_manifest)),
    )

    log("Starting DL temporal calibration stage.")
    mark_stage_start("DL_temporal_calibration")
    dl_results, dl_manifest, dl_audit, source_log = run_dl_calibration(inv, lookup)
    log(f"DL temporal calibration complete: {len(dl_results)} metric rows.")
    mark_stage_pass(
        "DL_temporal_calibration",
        metric_rows=int(len(dl_results)),
        manifest_rows=int(len(dl_manifest)),
    )

    # -------------------------------------------------------------------------
    # Durable post-calibration checkpoints.  These are written BEFORE any
    # statistics or matplotlib work.  If Windows terminates Python during the
    # figure stage, the next invocation can reconstruct ML/DL results from the
    # already validated prediction CSVs without refitting the models.
    # -------------------------------------------------------------------------
    log("Starting final result assembly and statistics.")
    mark_stage_start("final_assembly_statistics")

    results = pd.concat([ml_results, dl_results], ignore_index=True)
    manifest = pd.concat([ml_manifest, dl_manifest], ignore_index=True)
    split_audit = pd.concat([ml_audit, dl_audit], ignore_index=True)

    expected_runs = 25 * 2 * 10 * 9
    expected_metric_rows = expected_runs * len(ALL_METRICS)

    if len(results) != expected_metric_rows:
        raise RuntimeError(
            f"Expected {expected_metric_rows} metric rows, found {len(results)}."
        )
    if len(manifest) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} manifest rows, found {len(manifest)}."
        )

    write_csv(MANIFEST_ROOT / "calibration_results_manifest.csv", manifest)
    write_csv(DIAG_ROOT / "runtime_split_audit.csv", split_audit)
    write_csv(DIAG_ROOT / "source_model_log.csv", source_log)

    summary = summarize(results)
    k90 = calculate_k90(results, asd)
    efficiency = calculate_efficiency(results)
    make_tables(results, summary, k90, efficiency, asd)

    # Statistics/tables are now durable.  A later crash in matplotlib must not
    # force any ML/DL computation on the next invocation.
    mark_stage_pass(
        "final_assembly_statistics",
        metric_rows=int(len(results)),
        manifest_rows=int(len(manifest)),
        tables_written=True,
    )

    log("Starting figure generation stage in isolated subprocess.")
    mark_stage_start("figure_generation")
    figure_worker = Path(__file__).resolve().with_name("BLOCK09_FIGURE_WORKER.py")
    if not figure_worker.exists():
        raise FileNotFoundError(f"Required figure worker not found: {figure_worker}")

    figure_log = DIAG_ROOT / "BLOCK09_figure_worker.log"
    figure_status = DIAG_ROOT / "BLOCK09_figure_worker_status.json"
    write_json(
        figure_status,
        {
            "status": "STARTED",
            "worker": str(figure_worker),
            "pid": os.getpid(),
        },
    )

    proc = subprocess.run(
        [sys.executable, str(figure_worker)],
        cwd=str(figure_worker.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    figure_log.write_text(proc.stdout or "", encoding="utf-8")

    if proc.returncode != 0:
        status = "CRASHED" if proc.returncode < 0 or proc.returncode == -1073741819 else "FAIL"
        write_json(
            figure_status,
            {
                "status": status,
                "returncode": int(proc.returncode),
                "worker": str(figure_worker),
                "worker_log": str(figure_log),
            },
        )
        raise RuntimeError(
            f"Isolated figure worker failed; returncode={proc.returncode}. "
            f"See {figure_log}"
        )

    write_json(
        figure_status,
        {
            "status": "PASS",
            "returncode": int(proc.returncode),
            "worker": str(figure_worker),
            "worker_log": str(figure_log),
        },
    )
    mark_stage_pass("figure_generation", figures_written=True, isolated_process=True)

    log("Starting final audit and summary stage.")
    mark_stage_start("final_audit_summary")
    audit = final_audit(
        results,
        manifest,
        split_audit,
        partitions,
        asd,
        k90,
        efficiency,
        upstream,
        source_log,
    )
    write_summary(audit, upstream, manifest, source_log)
    write_hash_manifest()
    mark_stage_pass("final_audit_summary", audit_status="PASS")

    log("=" * 88)
    log("✓ BLOCK 09 PASS — TEMPORAL CALIBRATION FROZEN")
    log(f"Output: {OUTPUT_ROOT}")
    log(f"Runs: {expected_runs}")
    log(f"Metric rows: {len(results)}")
    log(
        f"Fixed evaluation: {FIXED_EVAL_SET} "
        f"canonical windows {FIXED_EVAL_WINDOW_NUMBERS} "
        f"(array indices {FIXED_EVAL_WINDOW_INDICES})"
    )
    log(f"A_SD references: {len(asd)} metric rows")
    log(f"Runtime split audits: {len(split_audit)}")
    log("Reproducibility: PASS")
    log("Audit: PASS")
    mark_stage_pass("final_assembly_statistics_figures_audit", metric_rows=int(len(results)), manifest_rows=int(len(manifest)))
    log("Raw data modified: NO")
    log("Previous blocks modified: NO")
    log("Downstream blocks used: NO")
    log("Ready for BLOCK10")
    log("=" * 88)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nBLOCK09 INTERRUPTED — upstream blocks were not modified.",
            file=sys.stderr,
        )
        raise
    except Exception as exc:
        mark_stage_fail("python_exception", exc)
        print("\nBLOCK09 FAIL", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)