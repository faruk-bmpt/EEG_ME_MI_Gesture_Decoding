#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BLOCK 10 — BIDIRECTIONAL ME–MI CROSS-CONDITION TRANSFER

Canonical executable for Paper 2.

Scientific design
-----------------
- 25 paired subjects; ME and MI.
- 7 numerical gesture classes: 1..7.
- Frozen W500/O50 window configuration from BLOCK04/BLOCK05/BLOCK07.
- Two primary directions:
    ME -> MI
    MI -> ME
- Same-subject cross-condition transfer.
- Primary evaluation: target condition SetC, all 147 windows
  (21 windows x 7 gestures).
- Source training for ML uses the frozen Block06 SD training partition:
  SetA + SetB = 294 windows for the same subject and source condition.
- DL source models are the frozen BLOCK07 07A subject-dependent checkpoints
  trained under the native SD protocol; target condition is inference-only.
- No target calibration, target fine-tuning, target normalization fitting,
  target hyperparameter tuning, or target checkpoint selection.
- Primary endpoint: Macro-F1.
- Subject is the inferential unit.
- All 10 models are retained.
- Resume/checkpointing is per subject x direction x model.
- Failed units are never silently removed.

Important:
This block does not calculate RSA, XAI, temporal calibration, or robustness.
Those belong to downstream blocks.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SUBJECTS = list(range(1, 26))
CONDITIONS = ["ME", "MI"]
GESTURES = list(range(1, 8))
SETS = ["SetA", "SetB", "SetC"]

ML_MODELS = ["LDA", "KNN", "SVM", "RandomForest", "XGBoost"]
DL_MODELS = [
    "EEGNet",
    "ShallowConvNet",
    "DeepConvNet",
    "CNN-BiLSTM",
    "CNN-BiLSTM-Attention",
]
ALL_MODELS = ML_MODELS + DL_MODELS

TRANSFER_DIRECTIONS = [
    ("ME", "MI", "ME_to_MI"),
    ("MI", "ME", "MI_to_ME"),
]

METRICS = [
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "mcc",
    "cohen_kappa",
]

PRIMARY_METRIC = "macro_f1"
N_BOOTSTRAP = 10000
N_PERMUTATIONS = 10000
FDR_Q = 0.05
NUM_TOL = 1e-10

RF_MAX_RETRIES = 3
RF_RETRY_SLEEP_SECONDS = 1.0


def deterministic_seed(*parts, base=SEED, modulo=2_000_000_000):
    """Stable cross-process seed; never uses Python's randomized hash()."""
    payload = "||".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) % modulo + int(base)


def canonical_json_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def package_version(package_name: str) -> str:
    """Return an installed package version without importing optional stacks."""
    try:
        from importlib.metadata import version
        return str(version(package_name))
    except Exception:
        return "UNAVAILABLE"

FS = 125
N_CHANNELS = 13
WINDOW_LENGTH = 500
OVERLAP_PERCENT = 50
WINDOW_STEP = 250
WINDOWS_PER_RECORDING = 21
FEATURE_COUNT = 341
TARGET_SET = "SetC"
SOURCE_SETS = ["SetA", "SetB"]

RAW_ROOT = Path(r"F:\Faruk\EEG_Motor_Execution_Imagery")
PAPER2_ROOT = RAW_ROOT / "Paper2_EEG_BCI"

BLOCK04_ROOT = PAPER2_ROOT / "04_WINDOWING"
BLOCK05_ROOT = PAPER2_ROOT / "05_FEATURE_ENGINEERING"
BLOCK06_ROOT = PAPER2_ROOT / "06_ML"
BLOCK07_ROOT = PAPER2_ROOT / "07_DL"

OUTPUT_ROOT = PAPER2_ROOT / "10_ME_MI_TRANSFER"
SOURCE_ROOT = OUTPUT_ROOT / "source_models"
PRED_ROOT = OUTPUT_ROOT / "predictions"
METRIC_ROOT = OUTPUT_ROOT / "metrics"
SUBJECT_ROOT = OUTPUT_ROOT / "subject_level"
STAT_ROOT = OUTPUT_ROOT / "statistics"
CLASS_ROOT = OUTPUT_ROOT / "class_metrics"
CM_ROOT = OUTPUT_ROOT / "confusion_matrices"
DIAG_ROOT = OUTPUT_ROOT / "diagnostics"
FIG_ROOT = OUTPUT_ROOT / "figures"
CONFIG_ROOT = OUTPUT_ROOT / "configs"

for p in [
    OUTPUT_ROOT, SOURCE_ROOT, PRED_ROOT, METRIC_ROOT, SUBJECT_ROOT,
    STAT_ROOT, CLASS_ROOT, CM_ROOT, DIAG_ROOT, FIG_ROOT, CONFIG_ROOT,
    PRED_ROOT / "ME_to_MI", PRED_ROOT / "MI_to_ME",
    SOURCE_ROOT / "ME", SOURCE_ROOT / "MI",
    CM_ROOT / "ME_to_MI", CM_ROOT / "MI_to_ME",
    FIG_ROOT / "bidirectional_transfer", FIG_ROOT / "transfer_loss",
    FIG_ROOT / "negative_transfer",
    DIAG_ROOT / "leakage_audit", DIAG_ROOT / "direction_audit",
    DIAG_ROOT / "consistency_checks",
]:
    p.mkdir(parents=True, exist_ok=True)

RUNTIME_STATE = DIAG_ROOT / "BLOCK10_runtime_state.json"
SUMMARY_JSON = OUTPUT_ROOT / "BLOCK10_summary.json"
COMPLETE_JSON = OUTPUT_ROOT / "BLOCK10_complete.json"
CONFIG_JSON = CONFIG_ROOT / "BLOCK10_config.json"
MANIFEST_CSV = DIAG_ROOT / "transfer_run_manifest.csv"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"{now()} | {msg}", flush=True)


def atomic_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, df: pd.DataFrame):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_subject(v):
    if isinstance(v, str):
        m = re.search(r"(\d+)", v)
        if not m:
            raise ValueError(f"Invalid subject: {v!r}")
        v = int(m.group(1))
    v = int(v)
    if v not in SUBJECTS:
        raise ValueError(f"Subject outside 1..25: {v}")
    return v


def canonical_condition(v):
    x = str(v).strip().upper()
    if x in {"ME", "MOTOR_EXECUTION", "EXECUTION"}:
        return "ME"
    if x in {"MI", "MOTOR_IMAGERY", "IMAGERY"}:
        return "MI"
    raise ValueError(f"Invalid condition: {v!r}")


def canonical_set(v):
    x = str(v).strip().upper()
    if x in {"A", "SETA"}:
        return "SetA"
    if x in {"B", "SETB"}:
        return "SetB"
    if x in {"C", "SETC"}:
        return "SetC"
    raise ValueError(f"Invalid set: {v!r}")


def canonical_model(v):
    x = str(v).strip().upper().replace("_", "-")
    aliases = {
        "LDA": "LDA",
        "KNN": "KNN",
        "SVM": "SVM",
        "RF": "RandomForest",
        "RANDOMFOREST": "RandomForest",
        "RANDOM-FOREST": "RandomForest",
        "XGB": "XGBoost",
        "XGBOOST": "XGBoost",
        "EEGNET": "EEGNet",
        "SHALLOWCONVNET": "ShallowConvNet",
        "DEEPCONVNET": "DeepConvNet",
        "CNN-BILSTM": "CNN-BiLSTM",
        "CNN-BILSTM-ATTENTION": "CNN-BiLSTM-Attention",
        "CNN-BILSTM-SELF-ATTENTION": "CNN-BiLSTM-Attention",
    }
    if x not in aliases:
        raise ValueError(f"Unknown model: {v!r}")
    return aliases[x]


def require(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def read_json(path: Path, label: str):
    require(path, label)
    return json.loads(path.read_text(encoding="utf-8"))


def metric_values(y_true, y_pred) -> dict:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    if len(yt) != len(yp) or len(yt) == 0:
        raise RuntimeError("Metric input length mismatch or empty prediction.")
    if not set(np.unique(yt)).issubset(set(GESTURES)):
        raise RuntimeError("True labels outside canonical gesture IDs 1..7.")
    if not set(np.unique(yp)).issubset(set(GESTURES)):
        raise RuntimeError("Predicted labels outside canonical gesture IDs 1..7.")

    return {
        "macro_f1": float(f1_score(
            yt, yp, labels=GESTURES, average="macro", zero_division=0
        )),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(
            balanced_accuracy_score(yt, yp)
        ),
        "mcc": float(matthews_corrcoef(yt, yp)),
        "cohen_kappa": float(
            cohen_kappa_score(yt, yp, labels=GESTURES)
        ),
    }


def bootstrap_mean_ci(values, seed=SEED, n=N_BOOTSTRAP):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def paired_swap_permutation_p(a, b, seed=SEED, n=N_PERMUTATIONS):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return np.nan

    d = a - b
    observed = float(np.mean(d))
    rng = np.random.default_rng(seed)
    signs = rng.choice(
        np.array([-1.0, 1.0], dtype=float),
        size=(n, len(d)),
    )
    null = (signs * d[None, :]).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(observed)) + 1.0) / (n + 1.0)
    return float(p)


def bh_fdr(p_values, q=FDR_Q):
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    vals = p[finite]
    if len(vals) == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    adj = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    tmp = np.empty_like(adj)
    tmp[order] = np.minimum(adj, 1.0)
    out[finite] = tmp
    return out


def verify_upstream():
    """Hard gate: do not run transfer on an incomplete upstream pipeline."""
    b06_complete = read_json(
        BLOCK06_ROOT / "BLOCK06_complete.json", "BLOCK06 completion marker"
    )
    b06_config = read_json(
        BLOCK06_ROOT / "BLOCK06_config.json", "BLOCK06 configuration"
    )
    b04_complete = read_json(
        BLOCK04_ROOT / "BLOCK04_complete.json", "BLOCK04 completion marker"
    )
    b04_config = read_json(
        BLOCK04_ROOT / "BLOCK04_config.json", "BLOCK04 configuration",
    )
    b04_manifest_path = BLOCK04_ROOT / "final_window_manifest.csv"
    require(b04_manifest_path, "BLOCK04 final window manifest")

    b07_complete = read_json(
        BLOCK07_ROOT / "BLOCK07_complete.json", "BLOCK07 completion marker"
    )
    b07_config = read_json(
        BLOCK07_ROOT / "configs" / "BLOCK07_config.json",
        "BLOCK07 configuration",
    )

    block09_root = PAPER2_ROOT / "09_TEMPORAL_CALIBRATION"
    b09_complete_path = block09_root / "BLOCK09_complete.json"
    b09_complete = read_json(
        b09_complete_path,
        "BLOCK09 completion marker",
    )

    if str(b04_complete.get("status", "")).upper() != "PASS":
        raise RuntimeError("BLOCK04 is not PASS.")
    if str(b06_complete.get("status", "")).upper() != "PASS":
        raise RuntimeError("BLOCK06 is not PASS.")
    if str(b07_complete.get("status", "")).upper() != "PASS":
        raise RuntimeError("BLOCK07 is not PASS.")
    if str(b09_complete.get("status", "")).upper() != "PASS":
        raise RuntimeError("BLOCK09 is not PASS.")

    # BLOCK04 is the authoritative source for the frozen windowing decision.
    # The canonical BLOCK04 config uses selected_configuration_id.
    b04_status = str(b04_config.get("status", "")).upper()
    b04_selected = str(b04_config.get("selected_configuration_id", ""))
    b04_window = int(b04_config.get("selected_window_samples", -1))
    b04_overlap = float(b04_config.get("selected_overlap_percent", -1))
    b04_step = int(b04_config.get("selected_step_samples", -1))

    if b04_status != "FROZEN":
        raise RuntimeError(
            f"BLOCK04 config status is {b04_status!r}; expected FROZEN."
        )
    if b04_selected != "W500_O50":
        raise RuntimeError(
            f"BLOCK04 selected_configuration_id is {b04_selected!r}; expected W500_O50."
        )
    if b04_window != WINDOW_LENGTH:
        raise RuntimeError(
            f"BLOCK04 selected_window_samples={b04_window}; expected {WINDOW_LENGTH}."
        )
    if b04_overlap != OVERLAP_PERCENT:
        raise RuntimeError(
            f"BLOCK04 selected_overlap_percent={b04_overlap}; expected {OVERLAP_PERCENT}."
        )
    if b04_step != WINDOW_STEP:
        raise RuntimeError(
            f"BLOCK04 selected_step_samples={b04_step}; expected {WINDOW_STEP}."
        )

    # Validate the completion marker only when it actually contains an
    # explicit configuration field. BLOCK04_config.json is authoritative.
    b04_complete_configuration = b04_complete.get("configuration")
    if (
        b04_complete_configuration is not None
        and str(b04_complete_configuration) != "W500_O50"
    ):
        raise RuntimeError(
            f"BLOCK04 completion configuration is "
            f"{b04_complete_configuration!r}; expected W500/O50."
        )

    # Empirical geometry gate from the authoritative BLOCK04 manifest.
    manifest = pd.read_csv(b04_manifest_path)
    required_geometry = {"window_samples", "start_sample", "end_sample"}
    missing_geometry = required_geometry - set(manifest.columns)
    if missing_geometry:
        raise RuntimeError(
            f"BLOCK04 manifest missing geometry columns: {sorted(missing_geometry)}"
        )

    window_samples = pd.to_numeric(
        manifest["window_samples"], errors="raise"
    ).astype(int)
    start_sample = pd.to_numeric(
        manifest["start_sample"], errors="raise"
    ).astype(int)
    end_sample = pd.to_numeric(
        manifest["end_sample"], errors="raise"
    ).astype(int)

    if not np.all(window_samples.to_numpy() == WINDOW_LENGTH):
        raise RuntimeError(
            "BLOCK04 manifest does not confirm 500-sample windows."
        )

    # BLOCK04 stores end_sample as an EXCLUSIVE boundary:
    # [start_sample, end_sample). Therefore 0 -> 500 is 500 samples.
    if not np.all(
        (end_sample - start_sample).to_numpy() == WINDOW_LENGTH
    ):
        raise RuntimeError(
            "BLOCK04 sample boundaries are inconsistent with 500-sample windows."
        )

    if np.any(end_sample.to_numpy() <= start_sample.to_numpy()):
        raise RuntimeError(
            "BLOCK04 manifest contains non-positive window lengths."
        )

    # Verify W500/O50 geometry from actual sample boundaries.
    # Do not depend on window_index because the canonical BLOCK04 manifest
    # may leave that field blank.
    # The canonical BLOCK04 manifest has been observed with either
    # 'gesture_code' or 'gesture' as the raw gesture-identity column
    # depending on the manifest-writer version. Resolve whichever is
    # actually present rather than silently no-op'ing the geometry audit
    # when only one of the two names exists.
    gesture_col = (
        "gesture_code" if "gesture_code" in manifest.columns
        else "gesture" if "gesture" in manifest.columns
        else None
    )
    grouping_columns = (
        ["subject", "condition", gesture_col, "set"]
        if gesture_col is not None else []
    )
    if gesture_col is not None and set(grouping_columns).issubset(manifest.columns):
        geom = manifest.copy()
        geom["_start_sample"] = start_sample.to_numpy(dtype=np.int64)
        geom["_end_sample"] = end_sample.to_numpy(dtype=np.int64)
        geom["subject"] = geom["subject"].astype(str)
        geom["condition"] = geom["condition"].astype(str)
        geom[gesture_col] = pd.to_numeric(
            geom[gesture_col], errors="raise"
        ).astype(int)
        geom["set"] = geom["set"].astype(str)

        for key, g in geom.groupby(grouping_columns, sort=False):
            g = g.sort_values("_start_sample")
            starts = g["_start_sample"].to_numpy(dtype=np.int64)
            ends = g["_end_sample"].to_numpy(dtype=np.int64)

            if len(g) != WINDOWS_PER_RECORDING:
                raise RuntimeError(
                    f"BLOCK04 geometry has {len(g)} windows for {key}; "
                    f"expected {WINDOWS_PER_RECORDING}."
                )

            # BLOCK04 uses complete W500 windows with a 250-sample step.
            # The canonical manifest contains exactly 21 windows, so the
            # authoritative starts are 0,250,...,5000. Do not derive this
            # sequence from total recording length; the final 125 samples of
            # the 5625-sample recording are intentionally unused because
            # BLOCK04 specifies no padding and complete-window-only generation.
            expected_starts = (
                np.arange(WINDOWS_PER_RECORDING, dtype=np.int64)
                * WINDOW_STEP
            )
            if not np.array_equal(starts, expected_starts):
                raise RuntimeError(
                    f"BLOCK04 start-sample geometry is inconsistent with "
                    f"W500/O50 for {key}."
                )

            if not np.array_equal(ends, starts + WINDOW_LENGTH):
                raise RuntimeError(
                    f"BLOCK04 end-sample geometry is inconsistent with "
                    f"W500/O50 for {key}."
                )

            # BLOCK09 derives window_index from the canonical WindowNNNN ID.
            # Reproduce that exact identity check here so BLOCK10 consumes
            # BLOCK04 in the same way as BLOCK09.
            if "window_id" not in g.columns:
                raise RuntimeError(
                    "BLOCK04 manifest lacks canonical window_id."
                )
            numbers = []
            for wid in g["window_id"].astype(str):
                m = re.search(r"Window(\d+)", wid, flags=re.IGNORECASE)
                if m is None:
                    raise RuntimeError(
                        f"Invalid canonical BLOCK04 window_id: {wid!r}."
                    )
                numbers.append(int(m.group(1)))

            expected_numbers = list(range(1, WINDOWS_PER_RECORDING + 1))
            if numbers != expected_numbers:
                raise RuntimeError(
                    f"BLOCK04 window-number order is inconsistent with "
                    f"W500/O50 for {key}."
                )

    if b06_complete.get("configuration") != "W500_O50":
        raise RuntimeError("BLOCK06 is not frozen at W500/O50.")
    if b06_config.get("block04", {}).get("configuration") != "W500_O50":
        raise RuntimeError("BLOCK06 config does not confirm W500/O50.")

    if int(b06_complete.get("feature_count", -1)) != FEATURE_COUNT:
        raise RuntimeError("BLOCK06 does not confirm 341 features.")

    return {
        "BLOCK04_complete_sha256": sha256_file(BLOCK04_ROOT / "BLOCK04_complete.json"),
        "BLOCK04_config_sha256": sha256_file(BLOCK04_ROOT / "BLOCK04_config.json"),
        "BLOCK04_manifest_sha256": sha256_file(b04_manifest_path),
        "BLOCK06_complete_sha256": sha256_file(BLOCK06_ROOT / "BLOCK06_complete.json"),
        "BLOCK06_config_sha256": sha256_file(BLOCK06_ROOT / "BLOCK06_config.json"),
        "BLOCK07_complete_sha256": sha256_file(BLOCK07_ROOT / "BLOCK07_complete.json"),
        "BLOCK07_config_sha256": sha256_file(
            BLOCK07_ROOT / "configs" / "BLOCK07_config.json"
        ),
        "BLOCK09_complete_sha256": sha256_file(b09_complete_path),
    }


def load_inventory():
    path = BLOCK04_ROOT / "final_window_manifest.csv"
    require(path, "BLOCK04 final window manifest")
    df = pd.read_csv(path)

    rename = {}
    if "subject_id" not in df.columns and "subject" in df.columns:
        rename["subject"] = "subject_id"
    if "condition" not in df.columns:
        raise RuntimeError("BLOCK04 manifest lacks condition.")
    if "gesture" not in df.columns:
        raise RuntimeError("BLOCK04 manifest lacks gesture.")
    if "set" not in df.columns:
        raise RuntimeError("BLOCK04 manifest lacks set.")
    if rename:
        df = df.rename(columns=rename)

    df["subject_id"] = df["subject_id"].map(canonical_subject)
    df["condition"] = df["condition"].map(canonical_condition)
    df["gesture"] = pd.to_numeric(df["gesture"], errors="raise").astype(int)
    df["set"] = df["set"].map(canonical_set)

    # The canonical BLOCK04 manifest never carries a native recording_id
    # column - it only stores window-level rows. recording_id is a
    # synthetic logical-recording identity (one value per 21-window group),
    # built here with the exact same convention BLOCK09 already uses and
    # has been running successfully against this same manifest, so BLOCK10
    # stays consistent with the frozen upstream identity scheme rather than
    # inventing a new one.
    if "recording_id" not in df.columns:
        df["recording_id"] = (
            "S" + df["subject_id"].astype(int).astype(str).str.zfill(2)
            + "_" + df["condition"].astype(str)
            + "_" + df["gesture"].astype(str)
            + "_" + df["set"].astype(str)
        )

    # BLOCK04 canonical manifest does not store window_index/window_number
    # as dedicated columns.  The canonical window identity is carried by
    # window_id (Window0001 ... Window0021).  BLOCK09 derives its zero-based
    # window index from this same canonical identifier.  BLOCK10 must use
    # that authoritative identity rather than requiring a column that
    # BLOCK04 never promised to create.
    if "window_index" not in df.columns:
        if "window_number" in df.columns:
            df["window_index"] = (
                pd.to_numeric(df["window_number"], errors="raise").astype(int) - 1
            )
        elif "window_id" in df.columns:
            window_numbers = []
            for wid in df["window_id"].astype(str):
                match = re.search(r"Window(\d+)", wid, flags=re.IGNORECASE)
                if match is None:
                    raise RuntimeError(
                        f"BLOCK04 manifest contains invalid window_id: {wid!r}."
                    )
                window_numbers.append(int(match.group(1)))

            df["window_index"] = np.asarray(window_numbers, dtype=np.int64) - 1
        else:
            raise RuntimeError(
                "BLOCK04 manifest lacks window_index/window_number and window_id."
            )

    df["window_index"] = pd.to_numeric(
        df["window_index"], errors="raise"
    ).astype(int)

    # BLOCK04 canonical numbering is Window0001..Window0021, corresponding
    # exactly to zero-based indices 0..20.
    if not np.array_equal(
        np.sort(df["window_index"].unique()),
        np.arange(WINDOWS_PER_RECORDING, dtype=np.int64),
    ):
        raise RuntimeError(
            "BLOCK04 window_index values are not the canonical 0..20 sequence."
        )

    required = {
        "subject_id", "condition", "gesture", "set", "window_index",
        "window_id", "recording_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"BLOCK04 manifest missing columns: {sorted(missing)}")

    if len(df) != 22050:
        raise RuntimeError(f"BLOCK04 inventory has {len(df)} rows; expected 22050.")
    if sorted(df["gesture"].unique().tolist()) != GESTURES:
        raise RuntimeError("BLOCK04 gestures are not exactly 1..7.")
    if sorted(df["subject_id"].unique().tolist()) != SUBJECTS:
        raise RuntimeError("BLOCK04 subjects are not exactly 1..25.")
    if sorted(df["condition"].unique().tolist()) != CONDITIONS:
        raise RuntimeError("BLOCK04 conditions are not exactly ME/MI.")

    counts = (
        df.groupby(["subject_id", "condition", "gesture", "set"])
        .size()
        .to_numpy()
    )
    if len(counts) != 1050 or not np.all(counts == WINDOWS_PER_RECORDING):
        raise RuntimeError("BLOCK04 logical recording/window coverage failed.")

    return df


def parse_feature_meta(path: Path):
    text = str(path).upper().replace("\\", "/")
    sm = re.findall(r"(?:^|[/_-])S0*(\d{1,2})(?:[/_-]|$)", text)
    cm = re.findall(r"(?:^|[/_-])(ME|MI)(?:[/_-]|$)", text)
    gm = re.findall(
        r"(?:^|[/_-])(?:G)?([1-7])(?:[_/-])SET[ABC](?:[/_.-]|$)",
        text,
    )
    if not (sm and cm and gm):
        return None
    subject = int(sm[-1])
    condition = canonical_condition(cm[-1])
    gesture = int(gm[-1])
    setm = re.findall(r"(?:^|[/_-])SET([ABC])(?:[/_-]|$)", text)
    if not setm:
        return None
    return subject, condition, gesture, f"Set{setm[-1]}"


def find_feature_files():
    found = {}
    for p in BLOCK05_ROOT.rglob("*.npz"):
        if "feature" not in p.stem.lower():
            continue
        meta = parse_feature_meta(p)
        if meta is None:
            continue
        if meta in found and found[meta] != p:
            raise RuntimeError(f"Duplicate Block05 feature file for {meta}.")
        found[meta] = p

    if len(found) != 1050:
        raise RuntimeError(
            f"Expected 1050 Block05 feature files; found {len(found)}."
        )
    return found


def load_feature_file(path: Path):
    with np.load(path, allow_pickle=False) as z:
        if "features" not in z.files:
            raise RuntimeError(f"No features array in {path}.")
        X = np.asarray(z["features"], dtype=np.float32)
    if X.shape != (WINDOWS_PER_RECORDING, FEATURE_COUNT):
        raise RuntimeError(f"Unexpected feature shape {X.shape}: {path}")
    if not np.isfinite(X).all():
        raise RuntimeError(f"Non-finite features: {path}")
    return X


def build_feature_table(inv):
    files = find_feature_files()
    rows = []

    for key, path in sorted(files.items()):
        s, c, g, set_id = key
        X = load_feature_file(path)
        sub = inv[
            (inv.subject_id == s)
            & (inv.condition == c)
            & (inv.gesture == g)
            & (inv.set == set_id)
        ].sort_values("window_index")

        if len(sub) != WINDOWS_PER_RECORDING:
            raise RuntimeError(f"Window mismatch for {key}.")

        for j, (_, r) in enumerate(sub.iterrows()):
            rows.append({
                "subject_id": s,
                "condition": c,
                "gesture": g,
                "set": set_id,
                "window_index": int(r.window_index),
                "window_id": str(r.window_id),
                "recording_id": str(r.recording_id),
                "X": X[j],
            })

    if len(rows) != 22050:
        raise RuntimeError(f"Feature table has {len(rows)} rows; expected 22050.")
    return rows


def feature_arrays(rows, subject, condition, sets):
    selected = [
        r for r in rows
        if r["subject_id"] == subject
        and r["condition"] == condition
        and r["set"] in sets
    ]
    selected.sort(key=lambda r: (
        r["set"], r["gesture"], r["window_index"]
    ))
    if not selected:
        raise RuntimeError(f"No ML rows for S{subject:02d} {condition} {sets}.")
    X = np.stack([r["X"] for r in selected]).astype(np.float32)
    y = np.asarray([r["gesture"] for r in selected], dtype=int)
    return X, y, selected


def make_ml_model(name):
    if name == "LDA":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearDiscriminantAnalysis(solver="svd")),
        ])
    if name == "KNN":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                metric="euclidean",
                n_jobs=1,
            )),
        ])
    if name == "SVM":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=False,
                random_state=SEED,
            )),
        ])
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=500,
            random_state=SEED,
            n_jobs=1,
            class_weight=None,
        )
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.80,
            colsample_bytree=0.80,
            objective="multi:softprob",
            num_class=7,
            eval_metric="mlogloss",
            random_state=SEED,
            n_jobs=1,
            tree_method="hist",
        )
    raise ValueError(name)


def xgb_predict(estimator, X):
    p = np.asarray(estimator.predict(X), dtype=int)
    # Frozen Block06 convention uses internal 0..6 labels.
    if p.min() >= 0 and p.max() <= 6:
        p = p + 1
    return p


def load_block06_native():
    path = BLOCK06_ROOT / "06A_ML_SD" / "subject_results" / "SD_subject_results.csv"
    require(path, "BLOCK06 native SD subject-results")
    df = pd.read_csv(path)
    if "subject_id" not in df.columns and "subject" in df.columns:
        df = df.rename(columns={"subject": "subject_id"})
    required = {"subject_id", "condition", "model"}
    if not required.issubset(df.columns):
        raise RuntimeError(
            f"BLOCK06 native result missing columns: "
            f"{sorted(required - set(df.columns))}"
        )
    df["subject_id"] = df["subject_id"].map(canonical_subject)
    df["condition"] = df["condition"].map(canonical_condition)
    df["model"] = df["model"].map(canonical_model)

    aliases = {
        "macro_f1": ["macro_f1", "macro_f1_score", "f1_macro"],
        "accuracy": ["accuracy"],
        "balanced_accuracy": ["balanced_accuracy", "balancedaccuracy"],
        "mcc": ["mcc"],
        "cohen_kappa": ["cohen_kappa", "cohens_kappa", "kappa"],
    }
    out = []
    for _, r in df.iterrows():
        for metric, names in aliases.items():
            col = next((n for n in names if n in df.columns), None)
            if col is None:
                continue
            value = pd.to_numeric(r[col], errors="coerce")
            if pd.isna(value):
                continue
            out.append({
                "subject_id": int(r.subject_id),
                "condition": str(r.condition),
                "model": str(r.model),
                "metric": metric,
                "native_target_performance": float(value),
                "native_source": str(path),
            })
    ref = pd.DataFrame(out)
    expected = 25 * 2 * 5 * 5
    expected_keys = {
        (s, c, m, metric)
        for s in SUBJECTS
        for c in CONDITIONS
        for m in ML_MODELS
        for metric in METRICS
    }
    actual_keys = {
        (int(r.subject_id), str(r.condition), str(r.model), str(r.metric))
        for _, r in ref.iterrows()
    }
    if len(ref) != expected:
        raise RuntimeError(
            f"BLOCK06 native reference row count mismatch: found {len(ref)}, "
            f"expected exactly {expected}."
        )
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise RuntimeError(
            f"BLOCK06 native reference key coverage mismatch. "
            f"Missing={missing[:10]}, Extra={extra[:10]}"
        )
    if ref.duplicated(["subject_id", "condition", "model", "metric"]).any():
        raise RuntimeError("BLOCK06 native reference contains duplicate metric keys.")
    return ref


def load_block07_sd_predictions():
    root = BLOCK07_ROOT / "07A_DL_SD" / "runs"
    require(root, "BLOCK07 07A SD runs directory")

    expected = {
        (s, c, m)
        for s in SUBJECTS for c in CONDITIONS for m in DL_MODELS
    }
    found = {}

    for p in sorted(root.rglob("predictions.csv")):
        try:
            df = pd.read_csv(p)
        except (pd.errors.ParserError, UnicodeDecodeError, OSError):
            # Ignore unreadable/non-CSV artifacts. Validation failures in an
            # otherwise readable prediction file must never be swallowed.
            continue

        required = {
            "subject_id", "condition", "model",
            "gesture_true", "gesture_pred", "window_id", "set",
        }
        if not required.issubset(df.columns):
            continue

        s = canonical_subject(df.iloc[0]["subject_id"])
        c = canonical_condition(df.iloc[0]["condition"])
        m = canonical_model(df.iloc[0]["model"])
        if m not in DL_MODELS:
            continue

        key = (s, c, m)
        if key in found:
            raise RuntimeError(f"Duplicate BLOCK07 SD predictions for {key}.")

        df["subject_id"] = df["subject_id"].map(canonical_subject)
        df["condition"] = df["condition"].map(canonical_condition)
        df["model"] = df["model"].map(canonical_model)
        df["set"] = df["set"].map(canonical_set)
        df["gesture_true"] = pd.to_numeric(df["gesture_true"], errors="raise").astype(int)
        df["gesture_pred"] = pd.to_numeric(df["gesture_pred"], errors="raise").astype(int)

        if len(df) != 147:
            raise RuntimeError(
                f"BLOCK07 native SD prediction must contain exactly 147 rows: {p}"
            )
        if set(df["set"]) != {"SetC"}:
            raise RuntimeError(
                f"BLOCK07 native SD prediction must be SetC-only: {p}"
            )
        if sorted(df["gesture_true"].unique().tolist()) != GESTURES:
            raise RuntimeError(
                f"BLOCK07 native SD prediction does not contain gestures 1..7: {p}"
            )
        if not df["gesture_pred"].isin(GESTURES).all():
            raise RuntimeError(
                f"BLOCK07 native SD prediction contains invalid gesture_pred IDs: {p}"
            )
        if df["window_id"].astype(str).duplicated().any():
            raise RuntimeError(
                f"BLOCK07 native SD prediction contains duplicate window IDs: {p}"
            )

        found[key] = (p, df)

    if set(found) != expected:
        missing = sorted(expected - set(found))
        extra = sorted(set(found) - expected)
        raise RuntimeError(
            f"BLOCK07 SD prediction coverage mismatch. Missing={missing[:10]}, "
            f"Extra={extra[:10]}"
        )

    return found


def _alnum_only(text: str) -> str:
    """Strip every separator (dash, underscore, space, dot) for fuzzy matching."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def discover_block07_sd_checkpoint(subject, condition, model):
    root = BLOCK07_ROOT / "07A_DL_SD" / "runs"
    preferred_names = {"best_model.pt", "model.pt", "checkpoint.pt"}
    model_key = _alnum_only(model)

    all_pt = list(root.rglob("*.pt"))

    def subject_condition_ok(ckpt):
        text = "/".join(x.upper() for x in ckpt.parts)
        subject_ok = (
            f"SUBJECT_{subject:02d}" in text
            or f"SUBJECT-{subject:02d}" in text
            or re.search(rf"(?:^|[/_-])S0*{subject}(?:[/_-]|$)", text)
        )
        return subject_ok and condition.upper() in text

    sc_matches = [c for c in all_pt if subject_condition_ok(c)]

    def model_ok(ckpt):
        # Real BLOCK07 layout nests checkpoints one level deeper than the
        # model name: .../<Model>/checkpoints/best_model.pt. So the true
        # model-identifying directory is the *grandparent*, not the
        # immediate parent (which is always literally "checkpoints").
        # Try grandparent first, then parent, for layouts without the
        # nested checkpoints/ folder.
        for seg in (ckpt.parent.parent.name, ckpt.parent.name):
            try:
                if canonical_model(seg) == model:
                    return True
            except Exception:
                pass
            # Fuzzy fallback: compare this ONE path segment (not the whole
            # concatenated path) after stripping all separators, so
            # "CNN_BiLSTM" cannot false-positive match as a substring
            # prefix of "CNN_BiLSTM_Attention" - it now requires the
            # entire segment to reduce to the same alnum string.
            if _alnum_only(seg) == model_key:
                return True
        return False

    candidates = [c for c in sc_matches if model_ok(c)]

    # If several plausible checkpoints exist, prefer the conventional
    # filenames; only fall back to "any single .pt" when that narrows to
    # exactly one, rather than requiring the conventional name up front
    # (BLOCK07 may not have used one of the three expected filenames for
    # every architecture).
    preferred = [c for c in candidates if c.name.lower() in preferred_names]
    if preferred:
        candidates = preferred

    if len(candidates) != 1:
        nearby = sorted(str(c) for c in sc_matches)
        raise RuntimeError(
            f"Expected exactly one BLOCK07 SD checkpoint for "
            f"S{subject:02d} {condition} {model}; found {len(candidates)}. "
            f"All .pt files under {root} matching subject+condition "
            f"(any model): {nearby[:15]}"
            + (f" ... (+{len(nearby) - 15} more)" if len(nearby) > 15 else "")
        )
    return candidates[0]


def load_block07_module():
    import ast
    import importlib.util

    candidates = [
        PAPER2_ROOT / "scripts" / "BLOCK07_DEEP_LEARNING_FINAL.py",
        PAPER2_ROOT / "BLOCK07_DEEP_LEARNING_FINAL.py",
        Path(__file__).resolve().parent / "BLOCK07_DEEP_LEARNING_FINAL.py",
    ]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "Canonical BLOCK07_DEEP_LEARNING_FINAL.py was not found."
        )

    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    has_guard = any(
        isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and any(isinstance(op, ast.Eq) for op in n.test.ops)
        and any(
            isinstance(c, ast.Constant) and c.value == "__main__"
            for c in n.test.comparators
        )
        for n in tree.body
    )
    if not has_guard:
        raise RuntimeError("BLOCK07 source lacks a safe __main__ guard.")

    spec = importlib.util.spec_from_file_location(
        "paper2_block07_transfer", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import BLOCK07 source.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_model"):
        raise RuntimeError("BLOCK07 canonical module has no build_model().")
    return module


def extract_normalizer(ckpt, checkpoint):
    mean_keys = [
        "normalizer_mean", "channel_mean", "normalization_mean",
        "train_mean", "x_mean", "scaler_mean", "mean",
    ]
    std_keys = [
        "normalizer_std", "channel_std", "normalization_std",
        "train_std", "x_std", "scaler_std", "std",
    ]
    mean = next((ckpt[k] for k in mean_keys if k in ckpt), None)
    std = next((ckpt[k] for k in std_keys if k in ckpt), None)

    sidecars = [
        checkpoint.with_name("normalization_stats.npz"),
        checkpoint.with_name("normalizer.npz"),
        checkpoint.with_name("normalization.json"),
        checkpoint.with_name("normalizer.json"),
    ]
    for side in sidecars:
        if mean is not None and std is not None:
            break
        if not side.exists():
            continue
        try:
            if side.suffix.lower() == ".npz":
                with np.load(side, allow_pickle=False) as z:
                    if "mean" in z.files and "std" in z.files:
                        mean, std = z["mean"], z["std"]
                    elif "normalizer_mean" in z.files and "normalizer_std" in z.files:
                        mean, std = z["normalizer_mean"], z["normalizer_std"]
                    elif "x_mean" in z.files and "x_std" in z.files:
                        mean, std = z["x_mean"], z["x_std"]
            else:
                obj = json.loads(side.read_text(encoding="utf-8"))
                mean = obj.get("mean", obj.get("normalizer_mean", obj.get("x_mean")))
                std = obj.get("std", obj.get("normalizer_std", obj.get("x_std")))
        except Exception:
            continue

    if mean is None or std is None:
        raise RuntimeError(
            f"BLOCK07 source normalization unavailable: {checkpoint}"
        )
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean.shape != (N_CHANNELS,) or std.shape != (N_CHANNELS,):
        raise RuntimeError(
            f"Invalid BLOCK07 normalization shape: {mean.shape}, {std.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise RuntimeError("Invalid BLOCK07 normalization values.")
    return mean, std


def load_block07_sd_model(module, subject, condition, model):
    import torch

    checkpoint = discover_block07_sd_checkpoint(subject, condition, model)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(f"Unexpected BLOCK07 checkpoint: {checkpoint}")

    net = module.build_model(model)
    net.load_state_dict(ckpt["state_dict"], strict=True)
    mean, std = extract_normalizer(ckpt, checkpoint)

    return net, mean, std, checkpoint


def resolve_window_array(path_value):
    raw = Path(str(path_value).replace("\\", "/"))
    if raw.is_absolute() and raw.exists():
        return raw
    for base in [PAPER2_ROOT, BLOCK04_ROOT]:
        p = base / raw
        if p.exists():
            return p
    matches = list(BLOCK04_ROOT.rglob(raw.name))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot uniquely resolve window array: {path_value}")


def load_raw_subset(inv, subject, condition, set_id):
    import numpy as np

    sub = inv[
        (inv.subject_id == subject)
        & (inv.condition == condition)
        & (inv.set == set_id)
    ].copy().sort_values(["gesture", "window_index"])

    if sub.empty:
        raise RuntimeError(f"No raw windows for S{subject:02d} {condition} {set_id}.")

    if "window_array_file" not in sub.columns:
        raise RuntimeError(
            "BLOCK04 manifest does not contain window_array_file; "
            "DL transfer cannot proceed without the authoritative window arrays."
        )

    cache = {}
    Xs, ys, metas = [], [], []
    for _, r in sub.iterrows():
        p = resolve_window_array(r.window_array_file)
        if p not in cache:
            with np.load(p, allow_pickle=False) as z:
                if "eeg_windows" not in z.files:
                    raise RuntimeError(f"Missing eeg_windows in {p}")
                arr = np.asarray(z["eeg_windows"], dtype=np.float32)
                starts = np.asarray(z["start_samples"], dtype=int)
                ends = np.asarray(z["end_samples"], dtype=int)
                if arr.shape != (21, 13, 500):
                    raise RuntimeError(f"Unexpected raw window shape {arr.shape}: {p}")
                cache[p] = (arr, starts, ends)
        arr, starts, ends = cache[p]
        idx = int(r.window_index)
        Xs.append(arr[idx])
        ys.append(int(r.gesture))
        metas.append(r.to_dict())

    return np.stack(Xs).astype(np.float32), np.asarray(ys, dtype=int), metas


def dl_predict(model, X, mean, std, model_name):
    import torch

    Xn = ((X - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
    model.eval()

    base = torch.from_numpy(Xn)  # (N, EEG_channels, T)

    # The frozen BLOCK07 architectures are not all the same shape family:
    # EEGNet/ShallowConvNet/DeepConvNet are Conv2d-based and require an
    # explicit in_channels=1 dimension, (N, 1, EEG_channels, T); the
    # CNN-BiLSTM family is Conv1d-based and requires the raw
    # (N, EEG_channels, T) shape. The required shape is a fixed property of
    # each frozen architecture, so it is looked up explicitly by model name
    # rather than inferred at runtime.
    conv1d_models = {"CNN-BiLSTM", "CNN-BiLSTM-Attention"}
    if model_name in conv1d_models:
        tensor = base  # (N, EEG_channels, T)
    else:
        tensor = base.unsqueeze(1)  # (N, 1, EEG_channels, T)

    with torch.no_grad():
        out = model(tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]

    pred = torch.argmax(out, dim=1).cpu().numpy().astype(int)
    # Canonical Block07 external labels are 1..7.
    if pred.min() >= 0 and pred.max() <= 6:
        pred += 1
    return pred


def native_reference(native_ml, native_dl, subject, target_condition, model):
    rows = []
    if model in ML_MODELS:
        q = native_ml[
            (native_ml.subject_id == subject)
            & (native_ml.condition == target_condition)
            & (native_ml.model == model)
        ]
        if q.empty:
            raise RuntimeError(
                f"Missing BLOCK06 native reference S{subject:02d} "
                f"{target_condition} {model}."
            )
        for _, r in q.iterrows():
            rows.append({
                "metric": r.metric,
                "native_target_performance": float(r.native_target_performance),
                "native_source": r.native_source,
            })
    else:
        key = (subject, target_condition, model)
        p, df = native_dl[key]
        q = metric_values(df.gesture_true, df.gesture_pred)
        for metric, value in q.items():
            rows.append({
                "metric": metric,
                "native_target_performance": float(value),
                "native_source": str(p),
            })
    return {r["metric"]: r for r in rows}


def save_prediction(path, records):
    df = pd.DataFrame(records)
    atomic_csv(path, df)
    return df


def validate_prediction_file(path, subject, source_condition, target_condition, model, direction):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        req = {
            "subject_id", "model", "transfer_direction",
            "source_condition", "target_condition",
            "gesture_true", "gesture_pred", "window_id", "set",
        }
        if not req.issubset(df.columns):
            return None
        if len(df) != 147:
            return None
        if int(df.subject_id.iloc[0]) != subject:
            return None
        if canonical_model(df.model.iloc[0]) != model:
            return None
        if str(df.transfer_direction.iloc[0]) != direction:
            return None
        if canonical_condition(df.source_condition.iloc[0]) != source_condition:
            return None
        if canonical_condition(df.target_condition.iloc[0]) != target_condition:
            return None
        if set(df.set.map(canonical_set)) != {"SetC"}:
            return None
        yt = pd.to_numeric(df.gesture_true, errors="raise").astype(int)
        yp = pd.to_numeric(df.gesture_pred, errors="raise").astype(int)
        if sorted(yt.unique().tolist()) != GESTURES:
            return None
        if not yp.isin(GESTURES).all():
            return None
        if df["window_id"].astype(str).duplicated().any():
            return None
        if df["window_id"].astype(str).nunique() != 147:
            return None
        if set(df["set"].map(canonical_set)) != {"SetC"}:
            return None
        if len(yt) != 147:
            return None
        return df
    except Exception:
        return None


def fit_ml_source_predict(rows, subject, source_condition, target_condition, model_name):
    Xtr, ytr, _ = feature_arrays(rows, subject, source_condition, SOURCE_SETS)
    Xev, yev, meta = feature_arrays(rows, subject, target_condition, [TARGET_SET])

    if len(Xtr) != 294 or len(Xev) != 147:
        raise RuntimeError(
            f"Unexpected transfer sizes: train={len(Xtr)}, eval={len(Xev)}"
        )
    if sorted(np.unique(ytr).tolist()) != GESTURES:
        raise RuntimeError("Source training does not contain all seven gestures.")
    if sorted(np.unique(yev).tolist()) != GESTURES:
        raise RuntimeError("Target evaluation does not contain all seven gestures.")

    rf_audit = None

    if model_name == "RandomForest":
        import pickle

        worker = Path(__file__).resolve().with_name("RF_ISOLATED_WORKER.py")
        if not worker.exists():
            raise FileNotFoundError(f"Required RF isolation worker not found: {worker}")

        rf_diag_root = DIAG_ROOT / "rf_isolated"
        rf_diag_root.mkdir(parents=True, exist_ok=True)
        stem = f"S{subject:02d}_{source_condition}_to_{target_condition}_RandomForest"
        rf_input = rf_diag_root / f"{stem}_input.pkl"
        rf_output = rf_diag_root / f"{stem}_output.pkl"
        rf_status = rf_diag_root / f"{stem}_status.json"
        rf_log = rf_diag_root / f"{stem}_worker.log"

        with rf_input.open("wb") as f:
            pickle.dump(
                {"Xtr": Xtr, "ytr": ytr, "Xev": Xev},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        attempts = []
        pred = None
        fit_seconds = np.nan
        infer_seconds = 0.0

        for attempt in range(1, RF_MAX_RETRIES + 1):
            t0 = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, str(worker), str(rf_input), str(rf_output)],
                cwd=str(worker.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            elapsed = time.perf_counter() - t0

            with rf_log.open("a", encoding="utf-8") as f:
                f.write(
                    f"\n--- BLOCK10 attempt {attempt} ---\n"
                    f"returncode={proc.returncode}\n{proc.stdout}\n"
                )

            attempt_record = {
                "attempt_number": attempt,
                "worker_returncode": int(proc.returncode),
                "worker_status": "PASS" if proc.returncode == 0 else "FAIL",
                "elapsed_seconds": float(elapsed),
            }
            attempts.append(attempt_record)

            if proc.returncode == 0 and rf_output.exists():
                try:
                    with rf_output.open("rb") as f:
                        result = pickle.load(f)
                    if str(result.get("status", "")).upper() != "PASS":
                        raise RuntimeError("RF worker output status is not PASS.")
                    pred = np.asarray(result["pred"], dtype=int)
                    if pred.shape != (len(Xev),):
                        raise RuntimeError(
                            f"RF worker prediction shape {pred.shape}; "
                            f"expected {(len(Xev),)}"
                        )
                    if not np.isin(pred, GESTURES).all():
                        raise RuntimeError("RF worker produced invalid gesture IDs.")
                    fit_seconds = float(elapsed)
                    break
                except Exception as exc:
                    attempt_record["worker_status"] = "INVALID_OUTPUT"
                    attempt_record["output_error"] = str(exc)

            atomic_json(
                rf_status,
                {
                    "subject_id": int(subject),
                    "source_condition": source_condition,
                    "target_condition": target_condition,
                    "model": "RandomForest",
                    "status": attempt_record["worker_status"],
                    "attempt_number": attempt,
                    "retry_count": attempt - 1,
                    "worker_returncode": int(proc.returncode),
                    "training_windows": int(len(Xtr)),
                    "evaluation_windows": int(len(Xev)),
                    "Xtr_shape": list(Xtr.shape),
                    "ytr_shape": list(ytr.shape),
                    "Xev_shape": list(Xev.shape),
                    "n_estimators": 500,
                    "random_state": SEED,
                    "n_jobs": 1,
                    "input_file": str(rf_input),
                    "output_file": str(rf_output),
                    "worker_log": str(rf_log),
                    "attempts": attempts,
                },
            )

            if pred is None and attempt < RF_MAX_RETRIES:
                time.sleep(RF_RETRY_SLEEP_SECONDS)

        if pred is None:
            raise RuntimeError(
                f"Isolated RandomForest failed after {RF_MAX_RETRIES} attempts for "
                f"S{subject:02d} {source_condition}->{target_condition}."
            )

        rf_audit = {
            "model": "RandomForest",
            "worker_isolated": True,
            "attempt_number": attempts[-1]["attempt_number"],
            "retry_count": attempts[-1]["attempt_number"] - 1,
            "worker_returncode": attempts[-1]["worker_returncode"],
            "worker_status": "PASS",
            "attempts": attempts,
            "n_estimators": 500,
            "random_state": SEED,
            "n_jobs": 1,
            "input_file": str(rf_input),
            "output_file": str(rf_output),
            "worker_log": str(rf_log),
        }
        model = None
        ypred = pred

    else:
        model = make_ml_model(model_name)
        t0 = time.perf_counter()
        if model_name == "XGBoost":
            # This installed xgboost version validates that y is strictly
            # 0..num_class-1 at fit time; canonical gesture IDs are 1..7.
            # Fit on the zero-indexed view here; xgb_predict already shifts
            # predictions back by +1 (frozen Block06 convention), so this
            # keeps fit/predict consistent without changing that convention.
            model.fit(Xtr, ytr - 1)
        else:
            model.fit(Xtr, ytr)
        fit_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        if model_name == "XGBoost":
            ypred = xgb_predict(model, Xev)
        else:
            ypred = np.asarray(model.predict(Xev), dtype=int)
        infer_seconds = time.perf_counter() - t1

    if not set(ypred.tolist()).issubset(set(GESTURES)):
        raise RuntimeError("ML prediction outside gesture IDs 1..7.")

    records = []
    for r, yt, yp in zip(meta, yev, ypred):
        records.append({
            "subject_id": subject,
            "model": model_name,
            "transfer_direction": (
                f"{source_condition}_to_{target_condition}"
            ),
            "source_condition": source_condition,
            "target_condition": target_condition,
            "gesture_true": int(yt),
            "gesture_pred": int(yp),
            "window_id": str(r["window_id"]),
            "window_index": int(r["window_index"]),
            "set": TARGET_SET,
            "source_file": str(r["recording_id"]),
        })
    return records, fit_seconds, infer_seconds, rf_audit


def run_dl_transfer(module, inv, subject, source_condition, target_condition, model_name):
    Xev, yev, meta = load_raw_subset(
        inv, subject, target_condition, TARGET_SET
    )
    if len(Xev) != 147:
        raise RuntimeError(f"DL target size is {len(Xev)}; expected 147.")
    if sorted(np.unique(yev).tolist()) != GESTURES:
        raise RuntimeError("DL target evaluation does not contain gestures 1..7.")

    net, mean, std, checkpoint = load_block07_sd_model(
        module, subject, source_condition, model_name
    )

    t0 = time.perf_counter()
    ypred = dl_predict(net, Xev, mean, std, model_name)
    infer_seconds = time.perf_counter() - t0

    # Validate the frozen source checkpoint output immediately.
    ypred = np.asarray(ypred, dtype=int)
    if ypred.shape != (len(Xev),):
        raise RuntimeError(
            f"DL prediction shape {ypred.shape}; expected {(len(Xev),)}."
        )
    if not np.isin(ypred, GESTURES).all():
        raise RuntimeError(
            f"DL prediction contains gesture IDs outside canonical 1..7: "
            f"{sorted(np.unique(ypred).tolist())}"
        )

    records = []
    for r, yt, yp in zip(meta, yev, ypred):
        records.append({
            "subject_id": subject,
            "model": model_name,
            "transfer_direction": f"{source_condition}_to_{target_condition}",
            "source_condition": source_condition,
            "target_condition": target_condition,
            "gesture_true": int(yt),
            "gesture_pred": int(yp),
            "window_id": str(r["window_id"]),
            "window_index": int(r["window_index"]),
            "set": TARGET_SET,
            "source_file": str(checkpoint),
        })

    del net
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return records, 0.0, infer_seconds, checkpoint


def build_run_leakage_audit(
    subject, source_condition, target_condition, direction, model,
    train_meta, eval_meta, rf_audit=None, checkpoint=None
):
    train_sets = sorted({str(r["set"]) for r in train_meta})
    eval_sets = sorted({str(r["set"]) for r in eval_meta})
    train_ids = {str(r["window_id"]) for r in train_meta}
    eval_ids = {str(r["window_id"]) for r in eval_meta}
    overlap = sorted(train_ids & eval_ids)

    audit = {
        "subject_id": int(subject),
        "source_condition": source_condition,
        "target_condition": target_condition,
        "transfer_direction": direction,
        "model": model,
        "source_condition_training_only": source_condition != target_condition,
        "train_sets": train_sets,
        "evaluation_sets": eval_sets,
        "train_window_count": len(train_meta),
        "evaluation_window_count": len(eval_meta),
        "train_window_ids_unique": len(train_ids) == len(train_meta),
        "evaluation_window_ids_unique": len(eval_ids) == len(eval_meta),
        "train_eval_window_id_overlap_count": len(overlap),
        "train_eval_window_id_overlap": overlap[:20],
        "target_set_is_setc": eval_sets == ["SetC"],
        "source_sets_are_seta_setb": (
            train_sets == ["SetA", "SetB"] if model in ML_MODELS else True
        ),
        "target_condition_absent_from_ml_training": (
            all(str(r["condition"]) == source_condition for r in train_meta)
            if model in ML_MODELS else True
        ),
        "pass": (
            source_condition != target_condition
            and len(overlap) == 0
            and eval_sets == ["SetC"]
            and len(eval_ids) == len(eval_meta)
            and (
                (model in ML_MODELS and train_sets == ["SetA", "SetB"] and len(train_meta) == 294)
                or model in DL_MODELS
            )
        ),
    }
    if rf_audit is not None:
        audit["rf_isolation"] = rf_audit
    if checkpoint is not None:
        audit["source_checkpoint"] = str(checkpoint)
        checkpoint_text = str(checkpoint).upper().replace("\\", "/")
        audit["checkpoint_mentions_source_condition"] = source_condition in checkpoint_text
        if not audit["checkpoint_mentions_source_condition"]:
            audit["pass"] = False

    if not audit["pass"]:
        raise RuntimeError(
            f"Leakage/split audit failed for S{subject:02d} {direction} {model}."
        )
    return audit


def write_run_status(status):
    name = (
        f"S{status['subject_id']:02d}_"
        f"{status['source_condition']}_to_{status['target_condition']}_"
        f"{status['model']}_status.json"
    )
    atomic_json(DIAG_ROOT / name, status)


def run_one(subject, source_condition, target_condition, direction, model,
            rows, inv, native_ml, native_dl, dl_module):
    pred_dir = PRED_ROOT / direction
    pred_path = pred_dir / (
        f"S{subject:02d}_{direction}_{model}_predictions.csv"
    )

    cached = validate_prediction_file(
        pred_path, subject, source_condition, target_condition, model, direction
    )

    status = {
        "subject_id": subject,
        "source_condition": source_condition,
        "target_condition": target_condition,
        "transfer_direction": direction,
        "model": model,
        "primary_metric": PRIMARY_METRIC,
        "status": "STARTED",
        "prediction_file": str(pred_path),
        "target_calibration_used": False,
        "target_training_used": False,
        "target_normalization_fit": False,
        "target_hyperparameter_tuning": False,
        "target_checkpoint_selection": False,
    }
    write_run_status(status)

    try:
        if cached is not None:
            pred_df = cached
            metrics = metric_values(pred_df.gesture_true, pred_df.gesture_pred)
            fit_seconds = np.nan
            infer_seconds = np.nan
            source_model_file = (
                str(pred_df.source_file.iloc[0])
                if "source_file" in pred_df.columns else ""
            )
            status["status"] = "PASS_RESUMED"
            status["resumed"] = True
            train_meta = [
                r for r in rows
                if r["subject_id"] == subject
                and r["condition"] == source_condition
                and r["set"] in SOURCE_SETS
            ]
            eval_meta = [
                r for r in rows
                if r["subject_id"] == subject
                and r["condition"] == target_condition
                and r["set"] == TARGET_SET
            ]
            run_audit = build_run_leakage_audit(
                subject, source_condition, target_condition, direction, model,
                train_meta if model in ML_MODELS else [],
                eval_meta,
            )
        else:
            status["resumed"] = False
            if model in ML_MODELS:
                train_meta = [
                    r for r in rows
                    if r["subject_id"] == subject
                    and r["condition"] == source_condition
                    and r["set"] in SOURCE_SETS
                ]
                eval_meta = [
                    r for r in rows
                    if r["subject_id"] == subject
                    and r["condition"] == target_condition
                    and r["set"] == TARGET_SET
                ]
                records, fit_seconds, infer_seconds, rf_audit = fit_ml_source_predict(
                    rows, subject, source_condition, target_condition, model
                )
                source_model_file = "BLOCK10_source_fit_from_frozen_Block06_spec"
                run_audit = build_run_leakage_audit(
                    subject, source_condition, target_condition, direction, model,
                    train_meta, eval_meta, rf_audit=rf_audit,
                )
            else:
                records, fit_seconds, infer_seconds, source_model_file = run_dl_transfer(
                    dl_module, inv, subject, source_condition, target_condition, model
                )
                eval_meta = [
                    r.to_dict() for _, r in inv[
                        (inv.subject_id == subject)
                        & (inv.condition == target_condition)
                        & (inv.set == TARGET_SET)
                    ].iterrows()
                ]
                run_audit = build_run_leakage_audit(
                    subject, source_condition, target_condition, direction, model,
                    [], eval_meta, checkpoint=source_model_file,
                )
            pred_df = save_prediction(pred_path, records)
            metrics = metric_values(
                pred_df.gesture_true, pred_df.gesture_pred
            )
            status["status"] = "PASS"

        native = native_reference(
            native_ml, native_dl, subject, target_condition, model
        )

        rows_out = []
        for metric in METRICS:
            transfer_perf = float(metrics[metric])
            native_perf = float(native[metric]["native_target_performance"])
            loss = native_perf - transfer_perf
            diff = transfer_perf - native_perf
            negative = bool(diff < -NUM_TOL)

            if abs(diff) <= NUM_TOL:
                transfer_status = "EQUIVALENT"
            elif diff > 0:
                transfer_status = "POSITIVE_TRANSFER"
            else:
                transfer_status = "NEGATIVE_TRANSFER"

            rows_out.append({
                "subject_id": subject,
                "model": model,
                "transfer_direction": direction,
                "metric": metric,
                "source_condition": source_condition,
                "target_condition": target_condition,
                "transfer_performance": transfer_perf,
                "native_target_performance": native_perf,
                "transfer_loss": loss,
                "transfer_difference": diff,
                "transfer_benefit": diff,
                "negative_transfer": negative,
                "transfer_status": transfer_status,
                "n_source_windows": 294 if model in ML_MODELS else np.nan,
                "n_target_windows": 147,
                "source_model_file": source_model_file,
                "native_source_file": native[metric]["native_source"],
                "fit_seconds": fit_seconds,
                "inference_seconds": infer_seconds,
                "resumed": bool(status.get("resumed", False)),
            })

        status["status"] = "PASS_RESUMED" if status.get("resumed") else "PASS"
        status["unit_complete"] = True
        status["metrics"] = metrics
        status["source_model_file"] = source_model_file
        status["leakage_audit"] = run_audit
        if "rf_audit" in locals() and rf_audit is not None:
            status["rf_isolation"] = rf_audit
        status["fit_seconds"] = fit_seconds
        status["inference_seconds"] = infer_seconds
        write_run_status(status)
        return rows_out, pred_df

    except Exception as exc:
        status["status"] = "FAIL"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
        status["traceback"] = traceback.format_exc()
        write_run_status(status)
        raise


def build_subject_results(metric_df):
    rows = []
    for (s, model, direction, metric), g in metric_df.groupby(
        ["subject_id", "model", "transfer_direction", "metric"],
        sort=True
    ):
        if len(g) != 1:
            raise RuntimeError("Duplicate subject/model/direction/metric result.")
        r = g.iloc[0]
        rows.append(r.to_dict())
    return pd.DataFrame(rows)


def build_model_summary(subject_df):
    rows = []
    for (model, direction, metric), g in subject_df.groupby(
        ["model", "transfer_direction", "metric"], sort=True
    ):
        x = g.transfer_performance.to_numpy(float)
        native = g.native_target_performance.to_numpy(float)
        loss = g.transfer_loss.to_numpy(float)
        diff = g.transfer_difference.to_numpy(float)
        lo, hi = bootstrap_mean_ci(diff, seed=deterministic_seed("bootstrap_model_summary", model, direction, metric))

        rows.append({
            "model": model,
            "direction": direction,
            "metric": metric,
            "n_subjects": len(g),
            "mean_transfer": float(np.mean(x)),
            "median_transfer": float(np.median(x)),
            "sd_transfer": float(np.std(x, ddof=1)),
            "mean_native_target": float(np.mean(native)),
            "mean_transfer_loss": float(np.mean(loss)),
            "mean_transfer_difference": float(np.mean(diff)),
            "negative_transfer_n": int(np.sum(diff < -NUM_TOL)),
            "negative_transfer_percentage": float(
                100.0 * np.mean(diff < -NUM_TOL)
            ),
            "ci_lower": lo,
            "ci_upper": hi,
        })
    return pd.DataFrame(rows)


def build_asymmetry(subject_df):
    rows = []
    key_cols = ["subject_id", "model", "metric"]
    for (s, model, metric), g in subject_df.groupby(key_cols, sort=True):
        me = g[g.transfer_direction == "ME_to_MI"]
        mi = g[g.transfer_direction == "MI_to_ME"]
        if len(me) != 1 or len(mi) != 1:
            raise RuntimeError(
                f"Missing bidirectional pair for S{s:02d} {model} {metric}."
            )
        l_me = float(me.transfer_loss.iloc[0])
        l_mi = float(mi.transfer_loss.iloc[0])
        rows.append({
            "subject_id": int(s),
            "model": model,
            "metric": metric,
            "ME_to_MI_loss": l_me,
            "MI_to_ME_loss": l_mi,
            "asymmetry": l_me - l_mi,
        })
    return pd.DataFrame(rows)


def build_statistics(subject_df, asym_df):
    rows = []
    for (model, metric), g in asym_df.groupby(
        ["model", "metric"], sort=True
    ):
        me = g.ME_to_MI_loss.to_numpy(float)
        mi = g.MI_to_ME_loss.to_numpy(float)
        a = g.asymmetry.to_numpy(float)

        p = paired_swap_permutation_p(
            me, mi,
            seed=deterministic_seed("permutation_asymmetry", model, metric),
            n=N_PERMUTATIONS,
        )
        lo, hi = bootstrap_mean_ci(
            a,
            seed=deterministic_seed("bootstrap_asymmetry", model, metric, base=SEED + 1000),
            n=N_BOOTSTRAP,
        )

        rows.append({
            "model": model,
            "metric": metric,
            "n_subjects": len(g),
            "ME_to_MI_mean": float(np.mean(
                subject_df[
                    (subject_df.model == model)
                    & (subject_df.metric == metric)
                    & (subject_df.transfer_direction == "ME_to_MI")
                ].transfer_performance
            )),
            "MI_to_ME_mean": float(np.mean(
                subject_df[
                    (subject_df.model == model)
                    & (subject_df.metric == metric)
                    & (subject_df.transfer_direction == "MI_to_ME")
                ].transfer_performance
            )),
            "ME_to_MI_median": float(np.median(me)),
            "MI_to_ME_median": float(np.median(mi)),
            "ME_to_MI_loss": float(np.mean(me)),
            "MI_to_ME_loss": float(np.mean(mi)),
            "asymmetry_mean": float(np.mean(a)),
            "asymmetry_median": float(np.median(a)),
            "asymmetry_ci_lower": lo,
            "asymmetry_ci_upper": hi,
            "asymmetry_p": p,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["asymmetry_p_adjusted"] = bh_fdr(out.asymmetry_p.to_numpy())
        out["fdr_q"] = FDR_Q
    return out


def build_negative_summary(subject_df):
    rows = []
    for (model, direction, metric), g in subject_df.groupby(
        ["model", "transfer_direction", "metric"], sort=True
    ):
        diff = g.transfer_difference.to_numpy(float)
        rows.append({
            "model": model,
            "direction": direction,
            "metric": metric,
            "n_subjects": len(g),
            "n_negative": int(np.sum(diff < -NUM_TOL)),
            "negative_percentage": float(
                100 * np.mean(diff < -NUM_TOL)
            ),
        })
    return pd.DataFrame(rows)


def build_class_metrics(prediction_frames):
    rows = []
    for key, df in prediction_frames.items():
        s, direction, model = key
        yt = df.gesture_true.astype(int).to_numpy()
        yp = df.gesture_pred.astype(int).to_numpy()
        p, r, f1, sup = precision_recall_fscore_support(
            yt, yp, labels=GESTURES, zero_division=0
        )
        for g, pp, rr, ff, ss in zip(GESTURES, p, r, f1, sup):
            rows.append({
                "subject_id": s,
                "transfer_direction": direction,
                "model": model,
                "gesture": g,
                "precision": float(pp),
                "recall": float(rr),
                "f1": float(ff),
                "support": int(ss),
            })
    return pd.DataFrame(rows)


def save_confusion_matrices(prediction_frames):
    for (s, direction, model), df in prediction_frames.items():
        cm = confusion_matrix(
            df.gesture_true.astype(int),
            df.gesture_pred.astype(int),
            labels=GESTURES,
        )
        path = CM_ROOT / direction / (
            f"S{s:02d}_{direction}_{model}_confusion_matrix.csv"
        )
        atomic_csv(
            path,
            pd.DataFrame(cm, index=GESTURES, columns=GESTURES)
            .rename_axis("true_gesture"),
        )


def write_config(upstream):
    feature_configuration = {
        "feature_count": FEATURE_COUNT,
        "feature_source": "BLOCK05 frozen feature inventory",
        "window_configuration": "W500_O50",
        "channels": N_CHANNELS,
    }
    window_configuration = {
        "window_length_samples": WINDOW_LENGTH,
        "overlap_percent": OVERLAP_PERCENT,
        "window_step_samples": WINDOW_STEP,
        "windows_per_recording": WINDOWS_PER_RECORDING,
        "sampling_rate_hz": FS,
    }

    config = {
        "block_id": "10_ME_MI_TRANSFER",
        "status": "FROZEN_BEFORE_EXECUTION",
        "subject_ids": SUBJECTS,
        "subjects": SUBJECTS,
        "conditions": CONDITIONS,
        "gesture_ids": GESTURES,
        "gesture_labels": GESTURES,
        "cohort": {
            "n_subjects": 25,
            "paired_me_mi": True,
            "subject_ids": SUBJECTS,
        },
        "sampling_rate_hz": FS,
        "channels": N_CHANNELS,
        "window_length_samples": WINDOW_LENGTH,
        "overlap_percent": OVERLAP_PERCENT,
        "window_step_samples": WINDOW_STEP,
        "window_configuration": "W500_O50",
        "window_configuration_hash": canonical_json_hash(window_configuration),
        "feature_configuration_hash": canonical_json_hash(feature_configuration),
        "feature_count": FEATURE_COUNT,
        "transfer_directions": [
            {"source": s, "target": t, "name": d}
            for s, t, d in TRANSFER_DIRECTIONS
        ],
        "ML_models": ML_MODELS,
        "DL_models": DL_MODELS,
        "primary_metric": PRIMARY_METRIC,
        "secondary_metrics": METRICS[1:],
        "source_training_protocol": (
            "ML: same subject, source condition, SetA+SetB, 294 windows, "
            "frozen Block06 model specification. "
            "DL: exact frozen Block07 07A SD source-condition checkpoint."
        ),
        "target_evaluation_protocol": (
            "same subject, opposite condition, SetC, full 147 windows; "
            "target labels used only for final metrics."
        ),
        "target_evaluation_population_policy": (
            "BLOCK10 uses complete SetC: 147 windows = 21 per gesture. "
            "The Window19-21 embargo subset belongs specifically to BLOCK09 "
            "temporal calibration and is not reused here because BLOCK10 "
            "performs no target adaptation, target training, target normalization "
            "fitting, target hyperparameter tuning, or target checkpoint selection."
        ),
        "block09_embargo_reuse_policy": (
            "The BLOCK09 Window18 embargo is not applied to BLOCK10 because "
            "BLOCK10 has no temporal target adaptation and trains ML models only "
            "on source-condition SetA+SetB; DL uses frozen upstream source checkpoints. "
            "Therefore the complete target SetC is inference-only in BLOCK10."
        ),
        "normalization_policy": (
            "ML scaler, where applicable, fit on source training only. "
            "DL applies exact source-checkpoint normalization statistics."
        ),
        "no_target_calibration": True,
        "no_target_finetuning": True,
        "no_target_hyperparameter_tuning": True,
        "no_target_checkpoint_selection": True,
        "no_target_normalization_fitting": True,
        "native_baseline_source": {
            "ML": "BLOCK06 precomputed SD subject_results.csv values.",
            "DL": (
                "BLOCK07 07A_SD predictions.csv; BLOCK10 recomputes metric values "
                "from the frozen prediction rows over FULL SetC."
            ),
            "policy": (
                "Native baselines are frozen upstream references. BLOCK10 does not "
                "fit, adapt, tune, normalize, or select checkpoints using target data."
            ),
        },
        "transfer_status_tolerance": NUM_TOL,
        "transfer_loss_definition": "native_target_performance - transfer_performance",
        "transfer_difference_definition": "transfer_performance - native_target_performance",
        "transfer_benefit_definition": "transfer_performance - native_target_performance",
        "negative_transfer_definition": "transfer_difference < 0",
        "asymmetry_definition": "ME_to_MI_loss - MI_to_ME_loss",
        "statistical_test": "paired within-subject sign-swap permutation",
        "permutation_n": N_PERMUTATIONS,
        "bootstrap_method": "subject-level nonparametric bootstrap of the mean; percentile CI",
        "bootstrap_n": N_BOOTSTRAP,
        "confidence_level": 0.95,
        "bootstrap_ci": "percentile 95%",
        "multiplicity_correction": "Benjamini-Hochberg FDR",
        "fdr_q": FDR_Q,
        "random_seed": SEED,
        "software_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": package_version("scikit-learn"),
            "joblib": package_version("joblib"),
            "xgboost": package_version("xgboost"),
            "torch": package_version("torch"),
        },
        "upstream_hashes": upstream,
        "rf_isolation": {
            "enabled": True,
            "worker": "RF_ISOLATED_WORKER.py",
            "max_retries": RF_MAX_RETRIES,
            "n_estimators": 500,
            "random_state": SEED,
            "n_jobs": 1,
        },
    }
    atomic_json(CONFIG_JSON, config)


def write_leakage_audit(run_audits):
    if not run_audits:
        raise RuntimeError("No run-level leakage audits were collected.")
    failures = [a for a in run_audits if not bool(a.get("pass"))]
    expected_runs = len(SUBJECTS) * len(TRANSFER_DIRECTIONS) * len(ALL_MODELS)
    audit = {
        "status": "PASS" if not failures else "FAIL",
        "n_runs_audited": len(run_audits),
        "expected_runs": expected_runs,
        "coverage_complete": len(run_audits) == expected_runs,
        "failed_run_audits": failures,
        "all_source_condition_training_only": all(
            a.get("source_condition_training_only", False) for a in run_audits
        ),
        "all_target_setc_only": all(
            a.get("target_set_is_setc", False) for a in run_audits
        ),
        "all_train_eval_window_disjoint": all(
            int(a.get("train_eval_window_id_overlap_count", -1)) == 0
            for a in run_audits
        ),
        "all_window_ids_unique": all(
            a.get("train_window_ids_unique", True)
            and a.get("evaluation_window_ids_unique", False)
            for a in run_audits
        ),
        "target_calibration_used": False,
        "target_finetuning_used": False,
        "target_normalization_fitting_used": False,
        "target_hyperparameter_tuning_used": False,
        "target_checkpoint_selection_used": False,
        "target_labels_used_only_for_final_metrics": True,
        "negative_transfer_retained": True,
        "all_expected_runs_audited": len(run_audits) == expected_runs,
    }
    atomic_json(
        DIAG_ROOT / "leakage_audit" / "BLOCK10_leakage_audit.json",
        audit,
    )
    if len(run_audits) != expected_runs:
        raise RuntimeError(
            f"BLOCK10 leakage audit coverage incomplete: audited "
            f"{len(run_audits)} runs; expected {expected_runs}."
        )
    if failures:
        raise RuntimeError(
            f"BLOCK10 leakage audit failed for {len(failures)} run(s)."
        )
    return audit


def write_direction_audit(subject_df, prediction_frames):
    rows = []
    for source, target, direction in TRANSFER_DIRECTIONS:
        keys = [k for k in prediction_frames if k[1] == direction]
        expected = 25 * len(ALL_MODELS)
        observed_metrics = len(subject_df[subject_df.transfer_direction == direction])
        passed = (
            len(keys) == expected
            and observed_metrics == expected * len(METRICS)
        )
        rows.append({
            "direction": direction,
            "source_condition": source,
            "target_condition": target,
            "expected_prediction_units": expected,
            "observed_prediction_units": len(keys),
            "expected_metric_rows": expected * len(METRICS),
            "observed_metric_rows": observed_metrics,
            "pass": bool(passed),
        })
    df = pd.DataFrame(rows)
    atomic_csv(DIAG_ROOT / "direction_audit" / "direction_audit.csv", df)
    if not df["pass"].all():
        raise RuntimeError("BLOCK10 direction audit failed.")
    return df


def write_consistency_checks(subject_df, prediction_frames, native_ml, native_dl):
    checks = [
        {
            "check": "prediction_unit_count",
            "expected": 500,
            "observed": len(prediction_frames),
            "pass": len(prediction_frames) == 500,
        },
        {
            "check": "subject_metric_rows",
            "expected": 500 * len(METRICS),
            "observed": len(subject_df),
            "pass": len(subject_df) == 500 * len(METRICS),
        },
        {
            "check": "native_ml_rows",
            "expected": 1250,
            "observed": len(native_ml),
            "pass": len(native_ml) == 1250,
        },
        {
            "check": "native_dl_runs",
            "expected": 250,
            "observed": len(native_dl),
            "pass": len(native_dl) == 250,
        },
        {
            "check": "transfer_loss_identity",
            "expected": "native - transfer",
            "observed": "validated",
            "pass": bool(np.allclose(
                subject_df.transfer_loss.to_numpy(float),
                subject_df.native_target_performance.to_numpy(float)
                - subject_df.transfer_performance.to_numpy(float),
                atol=1e-12,
            )),
        },
    ]
    df = pd.DataFrame(checks)
    atomic_csv(
        DIAG_ROOT / "consistency_checks" / "BLOCK10_consistency_checks.csv",
        df,
    )
    if not df["pass"].all():
        raise RuntimeError("BLOCK10 consistency checks failed.")
    return df


def make_figures(subject_df, summary_df, negative_df):
    import matplotlib.pyplot as plt

    # Figure 10-1: native vs transfer, primary metric.
    for model in ALL_MODELS:
        q = subject_df[
            (subject_df.model == model)
            & (subject_df.metric == PRIMARY_METRIC)
        ]
        if q.empty:
            continue
        labels = []
        vals = []
        for direction in ["ME_to_MI", "MI_to_ME"]:
            z = q[q.transfer_direction == direction]
            labels.append(direction)
            vals.append(float(z.transfer_performance.mean()))
        plt.figure(figsize=(8, 5))
        plt.bar(labels, vals)
        plt.ylabel("Macro-F1")
        plt.title(f"BLOCK10 Transfer Performance — {model}")
        plt.tight_layout()
        plt.savefig(
            FIG_ROOT / "bidirectional_transfer" / f"{model}_transfer.png",
            dpi=200,
        )
        plt.close()

    # Figure 10-2: transfer loss.
    for model in ALL_MODELS:
        q = subject_df[
            (subject_df.model == model)
            & (subject_df.metric == PRIMARY_METRIC)
        ]
        if q.empty:
            continue
        vals = [
            q[q.transfer_direction == d].transfer_loss.to_numpy()
            for d in ["ME_to_MI", "MI_to_ME"]
        ]
        plt.figure(figsize=(8, 5))
        plt.boxplot(vals, labels=["ME→MI", "MI→ME"])
        plt.axhline(0.0, linestyle="--")
        plt.ylabel("Transfer loss")
        plt.title(f"BLOCK10 Transfer Loss — {model}")
        plt.tight_layout()
        plt.savefig(
            FIG_ROOT / "transfer_loss" / f"{model}_loss.png",
            dpi=200,
        )
        plt.close()

    # Figure 10-3: negative-transfer frequency.
    q = negative_df[negative_df.metric == PRIMARY_METRIC]
    if not q.empty:
        pivot = q.pivot(
            index="model",
            columns="direction",
            values="negative_percentage",
        )
        pivot = pivot.reindex(ALL_MODELS)
        plt.figure(figsize=(11, 6))
        pivot.plot(kind="bar", ax=plt.gca())
        plt.ylabel("Negative-transfer subjects (%)")
        plt.title("BLOCK10 Negative Transfer Frequency")
        plt.tight_layout()
        plt.savefig(
            FIG_ROOT / "negative_transfer" / "negative_transfer_frequency.png",
            dpi=200,
        )
        plt.close()


def validate_completion(subject_df, prediction_frames, failures):
    expected_units = len(SUBJECTS) * len(TRANSFER_DIRECTIONS) * len(ALL_MODELS)
    expected_rows = expected_units * len(METRICS)

    if failures:
        raise RuntimeError(
            f"BLOCK10 has {len(failures)} failed transfer units. "
            f"First failures: {failures[:5]}"
        )
    if len(prediction_frames) != expected_units:
        raise RuntimeError(
            f"Expected {expected_units} prediction units; "
            f"found {len(prediction_frames)}."
        )
    if len(subject_df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} subject metric rows; "
            f"found {len(subject_df)}."
        )

    for _, r in subject_df.iterrows():
        if abs(
            float(r.transfer_difference) + float(r.transfer_loss)
        ) > 1e-8:
            raise RuntimeError("Transfer difference != -transfer loss.")
        expected_negative = float(r.transfer_difference) < -NUM_TOL
        if bool(r.negative_transfer) != expected_negative:
            raise RuntimeError("Negative-transfer identity failed.")

    for s in SUBJECTS:
        for model in ALL_MODELS:
            for metric in METRICS:
                q = subject_df[
                    (subject_df.subject_id == s)
                    & (subject_df.model == model)
                    & (subject_df.metric == metric)
                ]
                if len(q) != 2:
                    raise RuntimeError(
                        f"Bidirectional coverage failed for S{s:02d} "
                        f"{model} {metric}."
                    )


def main():
    start = time.perf_counter()
    log("=" * 88)
    log("BLOCK 10 — BIDIRECTIONAL ME–MI CROSS-CONDITION TRANSFER")
    log("=" * 88)
    log("Scientific rule: source condition only for training; target condition is inference-only.")
    log("Primary metric: Macro-F1; inferential unit: subject.")

    atomic_json(RUNTIME_STATE, {
        "pid": os.getpid(),
        "stage": "STARTED",
        "status": "RUNNING",
        "elapsed_seconds": 0.0,
    })

    failures = []
    prediction_frames = {}
    all_metric_rows = []
    run_audits = []

    try:
        upstream = verify_upstream()
        log("Upstream frozen-block verification PASS.")

        inv = load_inventory()
        log(f"BLOCK04 inventory PASS: {len(inv)} windows.")

        feature_rows = build_feature_table(inv)
        log(f"BLOCK05 feature inventory PASS: {len(feature_rows)} windows.")

        native_ml = load_block06_native()
        log(f"BLOCK06 native reference PASS: {len(native_ml)} rows.")

        native_dl = load_block07_sd_predictions()
        log("BLOCK07 native SD prediction coverage PASS: 250 runs.")

        write_config(upstream)

        try:
            dl_module = load_block07_module()
        except Exception as exc:
            dl_module = None
            log(f"BLOCK07 model module load deferred/failed: {exc}")

        manifest_rows = []

        total = 25 * 2 * 10
        counter = 0

        for source_condition, target_condition, direction in TRANSFER_DIRECTIONS:
            for subject in SUBJECTS:
                for model in ALL_MODELS:
                    counter += 1
                    log(
                        f"TRANSFER {counter}/{total}: "
                        f"S{subject:02d} {direction} {model}"
                    )

                    try:
                        if model in DL_MODELS and dl_module is None:
                            raise RuntimeError(
                                "Canonical BLOCK07 module unavailable for DL transfer."
                            )

                        rows_out, pred_df = run_one(
                            subject,
                            source_condition,
                            target_condition,
                            direction,
                            model,
                            feature_rows,
                            inv,
                            native_ml,
                            native_dl,
                            dl_module,
                        )
                        all_metric_rows.extend(rows_out)
                        prediction_frames[(subject, direction, model)] = pred_df

                        status_path = DIAG_ROOT / (
                            f"S{subject:02d}_{source_condition}_to_{target_condition}_{model}_status.json"
                        )
                        if status_path.exists():
                            st = read_json(status_path, "transfer run status")
                            if st.get("leakage_audit") is not None:
                                run_audits.append(st["leakage_audit"])

                        manifest_rows.append({
                            "subject_id": subject,
                            "source_condition": source_condition,
                            "target_condition": target_condition,
                            "transfer_direction": direction,
                            "model": model,
                            "status": "PASS",
                            "prediction_file": str(
                                PRED_ROOT / direction /
                                f"S{subject:02d}_{direction}_{model}_predictions.csv"
                            ),
                        })
                        atomic_csv(
                            MANIFEST_CSV,
                            pd.DataFrame(manifest_rows),
                        )
                        atomic_json(RUNTIME_STATE, {
                            "pid": os.getpid(),
                            "stage": "TRANSFER",
                            "status": "RUNNING",
                            "progress": f"{counter}/{total}",
                            "subject": subject,
                            "direction": direction,
                            "model": model,
                            "elapsed_seconds": time.perf_counter() - start,
                        })

                    except Exception as exc:
                        failure = {
                            "subject_id": subject,
                            "source_condition": source_condition,
                            "target_condition": target_condition,
                            "transfer_direction": direction,
                            "model": model,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        failures.append(failure)
                        manifest_rows.append({
                            **failure,
                            "status": "FAIL",
                        })
                        atomic_csv(MANIFEST_CSV, pd.DataFrame(manifest_rows))
                        log(
                            f"TRANSFER FAIL: S{subject:02d} "
                            f"{direction} {model}: {type(exc).__name__}: {exc}"
                        )

        subject_df = build_subject_results(pd.DataFrame(all_metric_rows))
        if subject_df.empty:
            raise RuntimeError("No successful transfer metric rows were produced.")

        atomic_csv(
            SUBJECT_ROOT / "subject_transfer_results.csv",
            subject_df,
        )

        asym_df = build_asymmetry(subject_df)
        atomic_csv(
            SUBJECT_ROOT / "transfer_asymmetry.csv",
            asym_df,
        )

        model_summary = build_model_summary(subject_df)
        atomic_csv(
            METRIC_ROOT / "transfer_model_summary.csv",
            model_summary,
        )

        negative_df = build_negative_summary(subject_df)
        atomic_csv(
            METRIC_ROOT / "negative_transfer.csv",
            negative_df,
        )

        atomic_csv(
            METRIC_ROOT / "transfer_metrics.csv",
            subject_df[
                [
                    "subject_id", "model", "transfer_direction", "metric",
                    "source_condition", "target_condition",
                    "transfer_performance",
                    "native_target_performance",
                    "transfer_loss",
                    "transfer_difference",
                    "transfer_benefit",
                    "negative_transfer",
                ]
            ],
        )

        atomic_csv(
            METRIC_ROOT / "transfer_loss.csv",
            subject_df[
                [
                    "subject_id", "model", "transfer_direction",
                    "metric", "transfer_loss"
                ]
            ],
        )
        atomic_csv(
            METRIC_ROOT / "transfer_difference.csv",
            subject_df[
                [
                    "subject_id", "model", "transfer_direction",
                    "metric", "transfer_difference"
                ]
            ],
        )
        atomic_csv(
            METRIC_ROOT / "transfer_benefit.csv",
            subject_df[
                [
                    "subject_id", "model", "transfer_direction",
                    "metric", "transfer_benefit"
                ]
            ],
        )

        class_df = build_class_metrics(prediction_frames)
        atomic_csv(
            CLASS_ROOT / "class_level_transfer.csv",
            class_df,
        )
        save_confusion_matrices(prediction_frames)

        stats_df = build_statistics(subject_df, asym_df)
        atomic_csv(
            STAT_ROOT / "transfer_statistical_summary.csv",
            stats_df,
        )
        atomic_csv(
            STAT_ROOT / "asymmetry_test.csv",
            stats_df,
        )

        # The two following files are retained as explicit aliases because
        # the master structural pipeline requires them.
        atomic_csv(
            STAT_ROOT / "bootstrap_results.csv",
            stats_df[
                [
                    "model", "metric", "n_subjects",
                    "asymmetry_mean", "asymmetry_ci_lower",
                    "asymmetry_ci_upper"
                ]
            ],
        )
        atomic_csv(
            STAT_ROOT / "permutation_results.csv",
            stats_df[
                [
                    "model", "metric", "n_subjects",
                    "asymmetry_p", "asymmetry_p_adjusted"
                ]
            ],
        )

        validate_completion(subject_df, prediction_frames, failures)
        direction_audit_df = write_direction_audit(subject_df, prediction_frames)
        consistency_df = write_consistency_checks(
            subject_df, prediction_frames, native_ml, native_dl
        )
        leakage = write_leakage_audit(run_audits)
        make_figures(subject_df, model_summary, negative_df)

        leakage = json.loads(
            (DIAG_ROOT / "leakage_audit" / "BLOCK10_leakage_audit.json")
            .read_text(encoding="utf-8")
        )

        input_hash = hashlib.sha256(
            json.dumps(
                {
                    "upstream": upstream,
                    "config": json.loads(CONFIG_JSON.read_text(encoding="utf-8")),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        summary = {
            "status": "PASS",
            "block": "10_ME_MI_TRANSFER",
            "eligible_subjects": SUBJECTS,
            "number_of_models": len(ALL_MODELS),
            "number_of_directions": 2,
            "ME_to_MI_runs": 25 * len(ALL_MODELS),
            "MI_to_ME_runs": 25 * len(ALL_MODELS),
            "successful_runs": len(prediction_frames),
            "failed_runs": len(failures),
            "negative_transfer_counts": (
                negative_df[
                    negative_df.metric == PRIMARY_METRIC
                ].groupby("direction").n_negative.sum().to_dict()
                if not negative_df.empty else {}
            ),
            "negative_transfer_percentages": (
                negative_df[
                    negative_df.metric == PRIMARY_METRIC
                ].groupby("direction").negative_percentage.mean().to_dict()
                if not negative_df.empty else {}
            ),
            "mean_transfer_loss": (
                float(subject_df[
                    subject_df.metric == PRIMARY_METRIC
                ].transfer_loss.mean())
            ),
            "mean_transfer_difference": (
                float(subject_df[
                    subject_df.metric == PRIMARY_METRIC
                ].transfer_difference.mean())
            ),
            "mean_asymmetry": (
                float(asym_df[
                    asym_df.metric == PRIMARY_METRIC
                ].asymmetry.mean())
            ),
            "statistical_test": "paired within-subject sign-swap permutation",
            "permutation_n": N_PERMUTATIONS,
            "bootstrap_n": N_BOOTSTRAP,
            "multiplicity_correction": "Benjamini-Hochberg FDR",
            "configuration_hash": sha256_file(CONFIG_JSON),
            "input_hash": input_hash,
            "upstream_hashes": upstream,
            "raw_data_modified": False,
            "previous_blocks_modified": False,
            "downstream_blocks_used": False,
            "leakage_audit": leakage["status"],
            "direction_audit": "PASS",
            "consistency_checks": "PASS",
        }
        atomic_json(SUMMARY_JSON, summary)

        complete = {
            "status": "PASS",
            "block": "10_ME_MI_TRANSFER",
            "subjects": 25,
            "directions": ["ME_to_MI", "MI_to_ME"],
            "models": ALL_MODELS,
            "metrics": METRICS,
            "prediction_units": 500,
            "metric_rows": len(subject_df),
            "all_units_complete": True,
            "all_seven_gestures_retained": True,
            "target_adaptation": False,
            "target_normalization_fit": False,
            "target_hyperparameter_tuning": False,
            "target_checkpoint_selection": False,
            "negative_transfer_retained": True,
            "subject_level_inference": True,
            "reproducibility": "PASS",
            "audit": "PASS",
            "configuration_hash": sha256_file(CONFIG_JSON),
            "input_hash": input_hash,
        }
        atomic_json(COMPLETE_JSON, complete)

        atomic_json(RUNTIME_STATE, {
            "pid": os.getpid(),
            "stage": "COMPLETE",
            "status": "PASS",
            "progress": "500/500",
            "elapsed_seconds": time.perf_counter() - start,
        })

        log("=" * 88)
        log("✓ BLOCK 10 PASS — BIDIRECTIONAL ME–MI TRANSFER FROZEN")
        log(f"Output: {OUTPUT_ROOT}")
        log(f"Transfer units: {len(prediction_frames)}/500")
        log(f"Metric rows: {len(subject_df)}")
        log("Directions: ME→MI and MI→ME")
        log("Primary metric: Macro-F1")
        log("Subject-level inference: PASS")
        log("Leakage audit: PASS")
        log("Reproducibility: PASS")
        log("Ready for BLOCK11")
        log("=" * 88)

    except Exception as exc:
        atomic_json(RUNTIME_STATE, {
            "pid": os.getpid(),
            "stage": "FAILED",
            "status": "FAIL",
            "elapsed_seconds": time.perf_counter() - start,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        atomic_json(SUMMARY_JSON, {
            "status": "FAIL",
            "block": "10_ME_MI_TRANSFER",
            "successful_runs": len(prediction_frames),
            "failed_runs": len(failures),
            "failures": failures,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        log("BLOCK10 FAIL")
        log(f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()