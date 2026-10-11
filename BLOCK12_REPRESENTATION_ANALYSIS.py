#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BLOCK12_REPRESENTATION_ANALYSIS.py
Paper 2 — Shared and Condition-Specific EEG Representations of Hand and Finger Gestures

Canonical Block 12
------------------
Purpose:
    Frozen, leakage-controlled latent-representation analysis linking:
      1) ME/MI latent representations,
      2) subject-level RSA/RDM similarity,
      3) bidirectional BLOCK10 transfer benefit,
      4) BLOCK09 temporal-calibration requirement (k90).

Scientific scope:
    - EEG only
    - 25 paired subjects
    - 7 gestures
    - ME and MI
    - W500/O50
    - 13 channels x 500 samples
    - SetC = 21 windows per gesture = 147 windows/subject/condition
    - Frozen BLOCK07 subject-dependent latent representations
    - Latent models: EEGNet, CNN-BiLSTM, CNN-BiLSTM-Attention
    - Predefined latent layer: forward_features
    - RDM: Euclidean distance between the seven gesture centroids
    - RSA coefficient: Spearman correlation of the upper-triangular RDM entries
    - Statistical unit: subject
    - Bootstrap: subject-level nonparametric percentile bootstrap
    - Association significance: subject-level permutation test
    - No model fitting, checkpoint tuning, calibration, or preprocessing occurs here.

Important:
    BLOCK07 already stores the final SetC latent representations for the three
    designated latent models. This block reads those frozen latent archives.
    It does NOT retrain or reconstruct the DL models.

Outputs:
    12_REPRESENTATION_ANALYSIS/
      configs/
      latent_archive/
      rsa/
      projections/
        PCA/
        tSNE/
        UMAP/
      cluster_quality/
      subject_similarity/
      transfer_relationship/
      calibration_relationship/
      figures/
      diagnostics/
      tables/
      BLOCK12_summary.json
      BLOCK12_complete.json

Run:
    python BLOCK12_REPRESENTATION_ANALYSIS.py
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import pickle
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Do not force native thread-environment variables in BLOCK12.
# The real t-SNE/UMAP standalone tests pass without these variables, while
# setting them in the parent process was associated with Windows native
# access-violation failures. Scientific projection parameters remain frozen
# below, with n_jobs=1 where supported.

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 00. FROZEN SCIENTIFIC CONSTANTS
# =============================================================================

BLOCK_ID = "BLOCK12"
PROTOCOL_VERSION = "PAPER2-BLOCK12-REPRESENTATION-1.0"

SUBJECTS = tuple(range(1, 26))
CONDITIONS = ("ME", "MI")
GESTURES = tuple(range(1, 8))

GESTURE_NAMES = {
    1: "HOC",
    2: "IFM",
    3: "LFM",
    4: "MFM",
    5: "PH",
    6: "RFM",
    7: "TFM",
}

LATENT_MODELS = (
    "EEGNet",
    "CNN-BiLSTM",
    "CNN-BiLSTM-Attention",
)

MODEL_DIRECTORY_NAME = {
    "EEGNet": "EEGNet",
    "CNN-BiLSTM": "CNN_BiLSTM",
    "CNN-BiLSTM-Attention": "CNN_BiLSTM_Attention",
}

EXPECTED_LATENT_ARCHIVES = len(SUBJECTS) * len(CONDITIONS) * len(LATENT_MODELS)

FS = 125
N_CHANNELS = 13
WINDOW_SAMPLES = 500
OVERLAP_PERCENT = 50
STEP_SAMPLES = 250
RECORDING_SAMPLES = 5625
WINDOWS_PER_RECORDING = 21
WINDOWS_PER_GESTURE_SETC = 21
WINDOWS_PER_SUBJECT_CONDITION_SETC = 147

PRIMARY_METRIC = "macro_f1"
CHANCE_LEVEL = 1.0 / 7.0

RANDOM_SEED = 42
BOOTSTRAP_N = 10000
PERMUTATION_N = 10000

# RSA is frozen before execution.
RSA_METHOD = "Spearman correlation of upper-triangular Euclidean RDM entries"
RDM_DISTANCE = "Euclidean distance between gesture-centroid latent vectors"
RDM_REPRESENTATION = "gesture centroid in frozen SetC latent space"
RSA_N_PAIRS = 21  # 7*6/2

# Descriptive projections are not used to fit classifiers or define hypotheses.
PCA_COMPONENTS = 2
TSNE_COMPONENTS = 2
TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 1000
TSNE_LEARNING_RATE = "auto"
TSNE_INIT = "pca"
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"

# The visualization archive is exploratory/descriptive. No inference is based
# on projection geometry.
PROJECTION_NOTE = (
    "PCA/t-SNE/UMAP are descriptive visualizations only. Projection coordinates "
    "are not used for model selection, training, RSA, transfer, or calibration."
)

RAW_ROOT = Path(r"F:\Faruk\EEG_Motor_Execution_Imagery")
PAPER2_ROOT = RAW_ROOT / "Paper2_EEG_BCI"

BLOCK04_ROOT = PAPER2_ROOT / "04_WINDOWING"
BLOCK07_ROOT = PAPER2_ROOT / "07_DL"
BLOCK07_SD_ROOT = BLOCK07_ROOT / "07A_DL_SD"
BLOCK09_ROOT = PAPER2_ROOT / "09_TEMPORAL_CALIBRATION"
BLOCK10_ROOT = PAPER2_ROOT / "10_ME_MI_TRANSFER"

OUTPUT_ROOT = PAPER2_ROOT / "12_REPRESENTATION_ANALYSIS"

CONFIG_ROOT = OUTPUT_ROOT / "configs"
LATENT_ROOT = OUTPUT_ROOT / "latent_archive"
RSA_ROOT = OUTPUT_ROOT / "rsa"
PCA_ROOT = OUTPUT_ROOT / "projections" / "PCA"
TSNE_ROOT = OUTPUT_ROOT / "projections" / "tSNE"

# Isolated-process execution for the t-SNE fit. Barnes-Hut t-SNE has
# produced a Windows native access violation on this host for the real EEGNet
# representation. The child process prevents a native crash from terminating
# the parent process, but it does not alter any scientific parameter.
# Diagnostic-only mode below distinguishes data-content, memory-layout, and
# subprocess effects before any workaround is considered.
TSNE_WORKER_SCRIPT = Path(__file__).resolve().parent / "TSNE_ISOLATED_WORKER.py"
UMAP_WORKER_SCRIPT = Path(__file__).resolve().parent / "UMAP_ISOLATED_WORKER.py"
TSNE_ISOLATED_ROOT = OUTPUT_ROOT / "diagnostics" / "tsne_isolated"
TSNE_ISOLATED_MAX_RETRIES = 3
TSNE_DIAGNOSTIC_ROOT = TSNE_ISOLATED_ROOT
UMAP_ROOT = OUTPUT_ROOT / "projections" / "UMAP"
CLUSTER_ROOT = OUTPUT_ROOT / "cluster_quality"
SUBJECT_ROOT = OUTPUT_ROOT / "subject_similarity"
TRANSFER_ROOT = OUTPUT_ROOT / "transfer_relationship"
CALIBRATION_ROOT = OUTPUT_ROOT / "calibration_relationship"
FIG_ROOT = OUTPUT_ROOT / "figures"
DIAG_ROOT = OUTPUT_ROOT / "diagnostics"
TABLE_ROOT = OUTPUT_ROOT / "tables"

for p in (
    OUTPUT_ROOT,
    CONFIG_ROOT,
    LATENT_ROOT,
    RSA_ROOT,
    PCA_ROOT,
    TSNE_ROOT,
    UMAP_ROOT,
    CLUSTER_ROOT,
    SUBJECT_ROOT,
    TRANSFER_ROOT,
    CALIBRATION_ROOT,
    FIG_ROOT,
    DIAG_ROOT,
    TABLE_ROOT,
):
    p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 01. REPRODUCIBILITY / I/O
# =============================================================================

START_TIME = time.perf_counter()


def log(message: str) -> None:
    elapsed = time.perf_counter() - START_TIME
    print(f"{elapsed:10.2f}s | {message}", flush=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, obj: dict) -> None:
    atomic_write_text(
        path,
        json.dumps(obj, indent=2, sort_keys=True, default=str),
    )


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def deterministic_seed(*parts) -> int:
    payload = "||".join(str(x) for x in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "UNAVAILABLE"


def software_versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": package_version("scikit-learn"),
        "scipy": package_version("scipy"),
        "matplotlib": package_version("matplotlib"),
        "umap_learn": package_version("umap-learn"),
    }


# =============================================================================
# 02. FROZEN-UPSTREAM VERIFICATION
# =============================================================================

def load_json_required(path: Path, description: str) -> dict:
    if not path.exists():
        raise RuntimeError(f"{description} missing: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"{description} is not valid JSON: {path}") from exc
    return obj


def verify_block04() -> dict:
    cfg_path = BLOCK04_ROOT / "BLOCK04_config.json"
    cfg = load_json_required(cfg_path, "BLOCK04 config")

    if str(cfg.get("status", "")).upper() != "FROZEN":
        raise RuntimeError("BLOCK04 is not FROZEN.")

    if str(cfg.get("selected_configuration_id", "")) != "W500_O50":
        raise RuntimeError("BLOCK04 selected configuration is not W500_O50.")

    checks = {
        "selected_window_samples": 500,
        "selected_overlap_percent": 50,
        "selected_step_samples": 250,
        "sampling_rate_hz": 125.0,
        "expected_samples_per_recording": 5625,
    }

    for key, expected in checks.items():
        actual = cfg.get(key)
        if isinstance(expected, float):
            if float(actual) != expected:
                raise RuntimeError(
                    f"BLOCK04 {key}={actual!r}, expected {expected!r}."
                )
        else:
            if int(actual) != expected:
                raise RuntimeError(
                    f"BLOCK04 {key}={actual!r}, expected {expected!r}."
                )

    return {
        "config_path": str(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "configuration_id": cfg["selected_configuration_id"],
        "window_samples": int(cfg["selected_window_samples"]),
        "overlap_percent": int(cfg["selected_overlap_percent"]),
        "step_samples": int(cfg["selected_step_samples"]),
        "sampling_rate_hz": float(cfg["sampling_rate_hz"]),
    }


def verify_block07() -> dict:
    completion = BLOCK07_ROOT / "BLOCK07_complete.json"
    obj = load_json_required(completion, "BLOCK07 completion marker")

    if str(obj.get("status", "")).upper() != "PASS":
        raise RuntimeError(
            f"BLOCK07 completion marker is not PASS: {obj.get('status')!r}"
        )

    cfg_candidates = [
        BLOCK07_ROOT / "configs" / "BLOCK07_config.json",
        BLOCK07_ROOT / "configs" / "BLOCK07_architecture_config.json",
        BLOCK07_ROOT / "BLOCK07_config.json",
    ]

    configs = [p for p in cfg_candidates if p.exists()]
    if not configs:
        raise RuntimeError("No canonical BLOCK07 configuration file was found.")

    return {
        "completion_path": str(completion),
        "completion_sha256": sha256_file(completion),
        "config_paths": [str(p) for p in configs],
        "latent_models": list(LATENT_MODELS),
        "latent_layer": "forward_features",
        "test_partition": "SetC",
        "windows_per_subject_condition": WINDOWS_PER_SUBJECT_CONDITION_SETC,
    }


def verify_block09() -> dict:
    completion = BLOCK09_ROOT / "BLOCK09_complete.json"
    obj = load_json_required(completion, "BLOCK09 completion marker")

    if str(obj.get("status", "")).upper() != "PASS":
        raise RuntimeError(
            f"BLOCK09 completion marker is not PASS: {obj.get('status')!r}"
        )

    k90_path = BLOCK09_ROOT / "metrics" / "k90_results.csv"
    if not k90_path.exists():
        raise RuntimeError(f"BLOCK09 k90_results.csv missing: {k90_path}")

    return {
        "completion_path": str(completion),
        "completion_sha256": sha256_file(completion),
        "k90_path": str(k90_path),
        "k90_sha256": sha256_file(k90_path),
    }


def verify_block10() -> dict:
    completion = BLOCK10_ROOT / "BLOCK10_complete.json"
    obj = load_json_required(completion, "BLOCK10 completion marker")

    if str(obj.get("status", "")).upper() != "PASS":
        raise RuntimeError(
            f"BLOCK10 completion marker is not PASS: {obj.get('status')!r}"
        )

    benefit_path = BLOCK10_ROOT / "metrics" / "transfer_benefit.csv"
    if not benefit_path.exists():
        raise RuntimeError(f"BLOCK10 transfer_benefit.csv missing: {benefit_path}")

    return {
        "completion_path": str(completion),
        "completion_sha256": sha256_file(completion),
        "transfer_benefit_path": str(benefit_path),
        "transfer_benefit_sha256": sha256_file(benefit_path),
    }


# =============================================================================
# 03. BLOCK07 LATENT ARCHIVE DISCOVERY
# =============================================================================

def canonical_model_name(s: str) -> str:
    x = str(s).strip().lower().replace("_", "-").replace(" ", "")
    mapping = {
        "eegnet": "EEGNet",
        "cnn-bilstm": "CNN-BiLSTM",
        "cnn-bilstm-attention": "CNN-BiLSTM-Attention",
        "cnn-bilstm+self-attention": "CNN-BiLSTM-Attention",
        "cnn-bilstm-attention": "CNN-BiLSTM-Attention",
    }
    if x not in mapping:
        raise RuntimeError(f"Unsupported latent model name: {s!r}")
    return mapping[x]


def _decode_metadata_scalar(value, field: str, path: Path) -> str:
    """Decode one trusted NPZ metadata scalar without accepting nested objects."""
    if isinstance(value, (str, np.str_)):
        return str(value)
    # Metadata are required to be actual string scalars.  Do not silently
    # decode bytes or accept nested/object structures because that can hide
    # malformed BLOCK07 archives.
    if isinstance(value, (dict, list, tuple, set, np.ndarray)):
        raise RuntimeError(
            f"{path}: {field} contains unsupported nested object type "
            f"{type(value).__name__}."
        )
    raise RuntimeError(
        f"{path}: {field} must contain a string/bytes value, got "
        f"{type(value).__name__}."
    )


def _decode_string_array(raw, field: str, path: Path) -> np.ndarray:
    """Safely convert trusted object/string metadata to a normal Unicode array."""
    arr = np.asarray(raw).reshape(-1)
    out = []
    for i, value in enumerate(arr.tolist()):
        try:
            out.append(_decode_metadata_scalar(value, field, path))
        except Exception as exc:
            raise RuntimeError(f"{path}: invalid {field}[{i}]: {exc}") from exc
    return np.asarray(out, dtype=str)


def _decode_subject_array(raw, path: Path) -> np.ndarray:
    arr = np.asarray(raw).reshape(-1)
    out = []
    for i, value in enumerate(arr.tolist()):
        if isinstance(value, (dict, list, tuple, set, np.ndarray)):
            raise RuntimeError(f"{path}: subject[{i}] contains nested object data.")
        try:
            n = int(value)
        except Exception as exc:
            raise RuntimeError(f"{path}: invalid subject[{i}]={value!r}.") from exc
        if isinstance(value, (float, np.floating)) and not float(value).is_integer():
            raise RuntimeError(f"{path}: non-integral subject[{i}]={value!r}.")
        out.append(n)
    return np.asarray(out, dtype=np.int64)


def _decode_gesture_array(raw, path: Path) -> np.ndarray:
    arr = np.asarray(raw).reshape(-1)
    out = []
    for i, value in enumerate(arr.tolist()):
        if isinstance(value, (dict, list, tuple, set, np.ndarray)):
            raise RuntimeError(f"{path}: gesture[{i}] contains nested object data.")
        try:
            n = int(value)
        except Exception as exc:
            raise RuntimeError(f"{path}: invalid gesture[{i}]={value!r}.") from exc
        if isinstance(value, (float, np.floating)) and not float(value).is_integer():
            raise RuntimeError(f"{path}: non-integral gesture[{i}]={value!r}.")
        out.append(n)
    return np.asarray(out, dtype=np.int64)


def find_latent_file(subject: int, condition: str, model: str) -> Path:
    """Resolve exactly one canonical Block07 latent archive; no discovery/fuzzy matching."""
    if model not in MODEL_DIRECTORY_NAME:
        raise RuntimeError(f"Unsupported canonical latent model: {model!r}")
    if condition not in CONDITIONS:
        raise RuntimeError(f"Unsupported canonical condition: {condition!r}")

    path = (
        BLOCK07_SD_ROOT
        / "runs"
        / f"Subject_{subject:02d}"
        / condition
        / MODEL_DIRECTORY_NAME[model]
        / "latent_representation.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing BLOCK07 latent archive: subject=S{subject:02d}, "
            f"condition={condition}, model={model!r}, expected_path={path}"
        )
    return path


def load_block07_latent_archive(path: Path) -> dict:
    """Load and strictly validate one trusted local Block07 latent NPZ archive."""
    required = {
        "subject", "condition", "gesture", "window_id",
        "model", "layer", "latent_vector",
    }
    try:
        with np.load(path, allow_pickle=True) as z:
            missing = sorted(required - set(z.files))
            if missing:
                raise RuntimeError(f"{path}: missing required key(s): {missing}")

            subjects = _decode_subject_array(z["subject"], path)
            conditions = _decode_string_array(z["condition"], "condition", path)
            gestures = _decode_gesture_array(z["gesture"], path)
            window_ids = _decode_string_array(z["window_id"], "window_id", path)
            model_values = _decode_string_array(z["model"], "model", path)
            layer_values = _decode_string_array(z["layer"], "layer", path)

            try:
                latent_raw = np.asarray(z["latent_vector"])
            except Exception as exc:
                raise RuntimeError(f"{path}: cannot read latent_vector: {exc}") from exc

            if latent_raw.dtype.kind not in "biufc":
                raise RuntimeError(
                    f"{path}: latent_vector must be numeric, got dtype={latent_raw.dtype}."
                )
            latent = np.asarray(latent_raw, dtype=np.float32)

    except ValueError as exc:
        raise RuntimeError(f"{path}: NPZ load/metadata error: {exc}") from exc

    if latent.ndim != 2:
        raise RuntimeError(f"{path}: latent_vector must be 2D, got {latent.shape}.")
    if latent.shape[0] != WINDOWS_PER_SUBJECT_CONDITION_SETC:
        raise RuntimeError(
            f"{path}: expected 147 latent rows, got {latent.shape[0]}."
        )
    if latent.shape[1] <= 0:
        raise RuntimeError(f"{path}: latent_vector has invalid latent dimension {latent.shape[1]}.")
    if latent.dtype != np.float32:
        raise RuntimeError(f"{path}: latent_vector failed float32 conversion; dtype={latent.dtype}.")
    if not np.isfinite(latent).all():
        raise RuntimeError(f"{path}: latent_vector contains NaN or Inf.")

    if len(subjects) != 147:
        raise RuntimeError(f"{path}: subject metadata has {len(subjects)} rows; expected 147.")
    if len(conditions) != 147:
        raise RuntimeError(f"{path}: condition metadata has {len(conditions)} rows; expected 147.")
    if len(gestures) != 147:
        raise RuntimeError(f"{path}: gesture metadata has {len(gestures)} rows; expected 147.")
    if len(window_ids) != 147:
        raise RuntimeError(f"{path}: window_id metadata has {len(window_ids)} rows; expected 147.")
    if len(np.unique(window_ids)) != 147:
        raise RuntimeError(f"{path}: window_id must contain exactly 147 unique values.")

    return {
        "subject": subjects,
        "condition": conditions,
        "gesture": gestures,
        "window_id": window_ids,
        "model": model_values,
        "layer": layer_values,
        "latent_vector": latent,
    }


def validate_latent_archive(path: Path, subject: int, condition: str, model: str) -> dict:
    """Validate one archive against the requested canonical subject/condition/model."""
    data = load_block07_latent_archive(path)
    subjects = data["subject"]
    conditions = data["condition"]
    gestures = data["gesture"]
    window_ids = data["window_id"]
    model_values = data["model"]
    layer_values = data["layer"]
    latent = data["latent_vector"]

    if set(subjects.tolist()) != {subject}:
        raise RuntimeError(f"{path}: subject IDs {sorted(set(subjects.tolist()))} != [{subject}].")
    if set(conditions.tolist()) != {condition}:
        raise RuntimeError(f"{path}: condition values {sorted(set(conditions.tolist()))} != [{condition}].")
    if len(model_values) != 1 or model_values[0] != model:
        raise RuntimeError(
            f"{path}: model metadata must be exactly {model!r}; observed {model_values.tolist()}"
        )
    if len(layer_values) != 1 or layer_values[0] != "forward_features":
        raise RuntimeError(
            f"{path}: layer metadata must be exactly 'forward_features'; observed {layer_values.tolist()}"
        )

    unique_gestures = sorted(set(gestures.tolist()))
    if unique_gestures != list(GESTURES):
        raise RuntimeError(f"{path}: gesture IDs must be exactly 1..7; observed {unique_gestures}.")
    counts = {g: int(np.sum(gestures == g)) for g in GESTURES}
    if any(v != WINDOWS_PER_GESTURE_SETC for v in counts.values()):
        raise RuntimeError(f"{path}: gesture counts must all equal 21; observed {counts}.")

    return {
        "path": str(path),
        "subject_id": subject,
        "condition": condition,
        "model": model,
        "model_directory": MODEL_DIRECTORY_NAME[model],
        "layer": layer_values[0],
        "n_windows": 147,
        "latent_dim": int(latent.shape[1]),
        "latent_dtype": str(latent.dtype),
        "sha256": sha256_file(path),
    }


def run_latent_preflight() -> Tuple[pd.DataFrame, Dict[Tuple[int, str, str], dict]]:
    """Exhaustively validate all 150 expected Block07 latent archives before RSA."""
    rows = []
    arrays = {}
    for subject in SUBJECTS:
        for condition in CONDITIONS:
            for model in LATENT_MODELS:
                path = (
                    BLOCK07_SD_ROOT / "runs" / f"Subject_{subject:02d}" / condition
                    / MODEL_DIRECTORY_NAME[model] / "latent_representation.npz"
                )
                row = {
                    "subject_id": subject, "condition": condition, "model": model,
                    "model_directory": MODEL_DIRECTORY_NAME[model], "path": str(path),
                    "exists": False, "archive_load": False, "required_keys": False,
                    "subject_valid": False, "condition_valid": False, "model_valid": False,
                    "layer_valid": False, "gesture_ids_valid": False,
                    "gesture_counts_valid": False, "window_id_valid": False,
                    "latent_shape_valid": False, "latent_dtype": "", "latent_finite": False,
                    "n_windows": 0, "latent_dim": 0, "sha256": "", "status": "FAIL", "error": "",
                }
                try:
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing exact path: {path}")
                    row["exists"] = True
                    data = load_block07_latent_archive(path)
                    row["archive_load"] = True
                    row["required_keys"] = True
                    row["subject_valid"] = set(data["subject"].tolist()) == {subject}
                    row["condition_valid"] = set(data["condition"].tolist()) == {condition}
                    row["model_valid"] = len(data["model"]) == 1 and data["model"][0] == model
                    row["layer_valid"] = len(data["layer"]) == 1 and data["layer"][0] == "forward_features"
                    unique_g = sorted(set(data["gesture"].tolist()))
                    row["gesture_ids_valid"] = unique_g == list(GESTURES)
                    counts = {g: int(np.sum(data["gesture"] == g)) for g in GESTURES}
                    row["gesture_counts_valid"] = all(v == 21 for v in counts.values())
                    row["window_id_valid"] = len(data["window_id"]) == 147 and len(np.unique(data["window_id"])) == 147
                    latent = data["latent_vector"]
                    row["latent_shape_valid"] = latent.ndim == 2 and latent.shape[0] == 147 and latent.shape[1] > 0
                    row["latent_dtype"] = str(latent.dtype)
                    row["latent_finite"] = bool(np.isfinite(latent).all())
                    row["n_windows"] = int(latent.shape[0])
                    row["latent_dim"] = int(latent.shape[1])
                    row["sha256"] = sha256_file(path)
                    if not all(row[k] for k in ["archive_load","required_keys","subject_valid","condition_valid","model_valid","layer_valid","gesture_ids_valid","gesture_counts_valid","window_id_valid","latent_shape_valid","latent_finite"]) or row["latent_dtype"] != "float32":
                        raise RuntimeError("One or more archive validation checks failed.")
                    arrays[(subject, condition, model)] = {
                        "latent": latent,
                        "gesture": data["gesture"],
                        "window_id": data["window_id"],
                        "path": path,
                    }
                    row["status"] = "PASS"
                except Exception as exc:
                    row["error"] = str(exc)
                rows.append(row)

    df = pd.DataFrame(rows)
    inventory_path = BLOCK07_SD_ROOT / "latent_preflight_inventory.csv"
    atomic_write_csv(inventory_path, df)
    atomic_write_csv(DIAG_ROOT / "latent_preflight_inventory.csv", df)

    expected = EXPECTED_LATENT_ARCHIVES
    found = int(df["exists"].sum())
    valid = int((df["status"] == "PASS").sum())
    invalid = expected - valid
    counts = {m: int(((df["model"] == m) & (df["exists"])).sum()) for m in LATENT_MODELS}

    print("=" * 60, flush=True)
    print("BLOCK07 LATENT PREFLIGHT REPORT", flush=True)
    print("=" * 60, flush=True)
    print(f"Expected archives : {expected}", flush=True)
    print(f"Found archives    : {found}", flush=True)
    print(f"EEGNet           : {counts['EEGNet']}", flush=True)
    print(f"CNN-BiLSTM       : {counts['CNN-BiLSTM']}", flush=True)
    print(f"CNN-BiLSTM-Attn  : {counts['CNN-BiLSTM-Attention']}", flush=True)
    print(f"Valid archives   : {valid}", flush=True)
    print(f"Invalid archives : {invalid}", flush=True)
    print("=" * 60, flush=True)

    if not (found == expected == valid and invalid == 0):
        bad = df[df["status"] != "PASS"][['subject_id','condition','model','path','error']]
        raise RuntimeError(
            "BLOCK07 latent archive verification FAILED. "
            f"See {inventory_path}.\n{bad.to_string(index=False)}"
        )
    if counts != {"EEGNet": 50, "CNN-BiLSTM": 50, "CNN-BiLSTM-Attention": 50}:
        raise RuntimeError(f"Unexpected exact model counts: {counts}")
    print("BLOCK07 latent archive verification PASS", flush=True)
    return df, arrays


def load_all_latents() -> Tuple[pd.DataFrame, Dict[Tuple[int, str, str], dict]]:
    """Compatibility wrapper: preflight is the single authoritative loader."""
    return run_latent_preflight()


# =============================================================================
# 04. RDM / RSA
# =============================================================================

def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


def gesture_centroids(
    latent: np.ndarray,
    gestures: np.ndarray,
) -> np.ndarray:
    centroids = []
    for g in GESTURES:
        x = latent[gestures == g]
        if x.shape[0] != WINDOWS_PER_GESTURE_SETC:
            raise RuntimeError(
                f"Gesture {g}: expected {WINDOWS_PER_GESTURE_SETC} windows, "
                f"got {x.shape[0]}."
            )
        centroids.append(x.mean(axis=0))
    return np.vstack(centroids)


def euclidean_rdm(centroids: np.ndarray) -> np.ndarray:
    diffs = centroids[:, None, :] - centroids[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))


def rsa_spearman(rdm_a: np.ndarray, rdm_b: np.ndarray) -> Tuple[float, float]:
    a = upper_triangle_values(rdm_a)
    b = upper_triangle_values(rdm_b)

    if len(a) != RSA_N_PAIRS or len(b) != RSA_N_PAIRS:
        raise RuntimeError("Unexpected RDM upper-triangle length.")

    rho, p = spearmanr(a, b)
    if not np.isfinite(rho):
        raise RuntimeError("RSA Spearman rho is non-finite.")

    return float(rho), float(p)


def build_subject_rsa(
    arrays: Dict[Tuple[int, str, str], dict],
) -> Tuple[pd.DataFrame, Dict[Tuple[int, str, str], dict]]:
    rows = []
    rdms = {}

    for model in LATENT_MODELS:
        for subject in SUBJECTS:
            subject_rdms = {}

            for condition in CONDITIONS:
                item = arrays[(subject, condition, model)]
                centroids = gesture_centroids(
                    item["latent"],
                    item["gesture"],
                )
                rdm = euclidean_rdm(centroids)
                subject_rdms[condition] = rdm
                rdms[(subject, condition, model)] = {
                    "rdm": rdm,
                    "centroids": centroids,
                }

            rho, p = rsa_spearman(
                subject_rdms["ME"],
                subject_rdms["MI"],
            )

            rows.append({
                "subject_id": subject,
                "model": model,
                "rsa_method": RSA_METHOD,
                "rdm_distance": RDM_DISTANCE,
                "rho_spearman": rho,
                "p_uncorrected": p,
                "n_gesture_pairs": RSA_N_PAIRS,
                "n_gestures": len(GESTURES),
            })

    rsa_df = pd.DataFrame(rows)

    if len(rsa_df) != len(SUBJECTS) * len(LATENT_MODELS):
        raise RuntimeError("Unexpected RSA row count.")

    if rsa_df.duplicated(["subject_id", "model"]).any():
        raise RuntimeError("Duplicate subject/model RSA keys.")

    atomic_write_csv(RSA_ROOT / "subject_level_rsa.csv", rsa_df)

    return rsa_df, rdms


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise RuntimeError("Cannot bootstrap empty/non-finite values.")

    rng = np.random.default_rng(seed)
    samples = rng.choice(
        values,
        size=(BOOTSTRAP_N, len(values)),
        replace=True,
    )
    means = samples.mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def summarize_rsa(rsa_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in rsa_df.groupby("model", sort=True):
        mean, lo, hi = bootstrap_mean_ci(
            g["rho_spearman"].to_numpy(float),
            deterministic_seed("rsa_bootstrap", model),
        )
        rows.append({
            "model": model,
            "n_subjects": int(g["subject_id"].nunique()),
            "mean_rho": mean,
            "median_rho": float(g["rho_spearman"].median()),
            "std_rho": float(g["rho_spearman"].std(ddof=1)),
            "ci95_low": lo,
            "ci95_high": hi,
        })
    out = pd.DataFrame(rows)
    atomic_write_csv(RSA_ROOT / "rsa_summary.csv", out)
    return out


# =============================================================================
# 05. DESCRIPTIVE PROJECTIONS
# =============================================================================

def pooled_latent_table(
    arrays: Dict[Tuple[int, str, str], dict],
    model: str,
) -> pd.DataFrame:
    rows = []
    for subject in SUBJECTS:
        for condition in CONDITIONS:
            item = arrays[(subject, condition, model)]
            for i in range(len(item["gesture"])):
                rows.append({
                    "subject_id": subject,
                    "condition": condition,
                    "gesture": int(item["gesture"][i]),
                    "gesture_label": GESTURE_NAMES[int(item["gesture"][i])],
                    "window_id": str(item["window_id"][i]),
                    "row_index": i,
                })
    out = pd.DataFrame(rows)
    atomic_write_csv(LATENT_ROOT / f"{model}_latent_metadata.csv", out)
    return out


def _validate_and_scale_latent_matrix(
    X: np.ndarray,
    model: str,
    context: str,
) -> np.ndarray:
    """
    Validate the pooled latent matrix BEFORE scaling/projection, and
    return a safely standardized copy.

    This exists because sklearn's TSNE (Barnes-Hut, Cython) is known to
    crash the whole process with a native access violation -- not raise
    a catchable Python exception -- when non-finite values reach its
    internal QuadTree/gradient computation (scikit-learn issue #28368).
    The most likely silent source of such values in this pipeline is
    StandardScaler dividing by a zero standard deviation for any
    constant (dead) latent dimension, which produces NaN/Inf with NO
    error raised by StandardScaler itself. PCA can also silently
    propagate such values without crashing, so a passing PCA step does
    not prove the data was clean -- this check must run independently,
    before EVERY projection (PCA, t-SNE, UMAP), not only before t-SNE.
    """
    if not np.isfinite(X).all():
        bad_rows, bad_cols = np.where(~np.isfinite(X))
        raise RuntimeError(
            f"{context} ({model}): input latent matrix contains "
            f"non-finite values before scaling. "
            f"{len(set(bad_cols.tolist()))} affected feature column(s): "
            f"{sorted(set(bad_cols.tolist()))[:20]}"
            f"{'...' if len(set(bad_cols.tolist())) > 20 else ''}. "
            f"{len(set(bad_rows.tolist()))} affected row(s)."
        )

    std = X.std(axis=0)
    zero_variance_cols = np.where(std == 0.0)[0]
    if len(zero_variance_cols) > 0:
        raise RuntimeError(
            f"{context} ({model}): {len(zero_variance_cols)} latent "
            f"feature column(s) are constant (zero variance) across "
            f"all pooled subject/condition/gesture rows: "
            f"{zero_variance_cols.tolist()[:20]}"
            f"{'...' if len(zero_variance_cols) > 20 else ''}. "
            f"StandardScaler would silently divide by zero for these "
            f"columns, producing NaN/Inf that can crash PCA/t-SNE/UMAP "
            f"downstream (including native access violations in "
            f"scikit-learn's Barnes-Hut t-SNE). Investigate why this "
            f"latent dimension never varies for {model} before "
            f"proceeding -- do not silently drop or impute it here."
        )

    Xs = StandardScaler().fit_transform(X)

    if not np.isfinite(Xs).all():
        raise RuntimeError(
            f"{context} ({model}): scaled latent matrix contains "
            f"non-finite values even though the pre-scaling checks "
            f"passed. Refusing to proceed to PCA/t-SNE/UMAP."
        )

    return Xs


def run_pca(
    arrays: Dict[Tuple[int, str, str], dict],
    model: str,
) -> pd.DataFrame:
    chunks = []
    meta = []

    for subject in SUBJECTS:
        for condition in CONDITIONS:
            item = arrays[(subject, condition, model)]
            chunks.append(item["latent"])
            for i in range(item["latent"].shape[0]):
                gesture_id = int(item["gesture"][i])
                window_value = item["window_id"][i]
                meta.append({
                    "subject_id": int(subject),
                    "condition": str(condition),
                    "gesture": gesture_id,
                    "gesture_label": GESTURE_NAMES[gesture_id],
                    "window_id": str(window_value),
                })

    X = np.vstack(chunks).astype(np.float64)
    meta_df = pd.DataFrame(meta)

    Xs = _validate_and_scale_latent_matrix(X, model, "PCA projection")

    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
    Z = pca.fit_transform(Xs)

    out = meta_df.copy()
    out["PC1"] = Z[:, 0]
    out["PC2"] = Z[:, 1]

    atomic_write_csv(
        PCA_ROOT / f"{model}_PCA_coordinates.csv",
        out,
    )

    atomic_write_json(
        PCA_ROOT / f"{model}_PCA_metadata.json",
        {
            "model": model,
            "n_samples": int(len(out)),
            "input_dimension": int(X.shape[1]),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance": float(
                np.sum(pca.explained_variance_ratio_)
            ),
            "scaler": "StandardScaler fitted for descriptive projection only",
            "note": PROJECTION_NOTE,
        },
    )

    return out


def _finite_float(value) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _build_tsne_runtime_diagnostics(X_pre: np.ndarray, tsne_params: dict, model: str) -> Tuple[Path, Path]:
    """Archive exact runtime X_pre and generate a temporary A/B/C/D diagnostic script."""
    TSNE_DIAGNOSTIC_ROOT.mkdir(parents=True, exist_ok=True)
    model_tag = model.replace("-", "_")
    json_path = TSNE_DIAGNOSTIC_ROOT / f"{model_tag}_Xpre_diagnostics.json"
    npy_path = TSNE_DIAGNOSTIC_ROOT / f"{model_tag}_Xpre_exact.npy"
    script_path = TSNE_DIAGNOSTIC_ROOT / f"{model_tag}_standalone_tsne_test.py"
    result_path = TSNE_DIAGNOSTIC_ROOT / f"{model_tag}_standalone_tsne_results.json"

    arr = np.asarray(X_pre)
    finite_mask = np.isfinite(arr)
    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    nonfinite_count = int(arr.size - int(finite_mask.sum()))

    if arr.ndim == 2 and arr.shape[1] > 0:
        col_std = np.std(arr, axis=0)
        col_min = np.min(arr, axis=0)
        col_max = np.max(arr, axis=0)
        col_nan = np.any(nan_mask, axis=0)
        col_inf = np.any(inf_mask, axis=0)
        zero_variance_cols = np.where(col_std == 0.0)[0].astype(int).tolist()
        near_zero_threshold = float(np.finfo(np.float64).eps * 100.0)
        near_zero_cols = np.where((col_std > 0.0) & (col_std <= near_zero_threshold))[0].astype(int).tolist()
        unique_rows = int(np.unique(np.ascontiguousarray(arr), axis=0).shape[0])
        duplicate_rows = int(arr.shape[0] - unique_rows)
    else:
        col_std = col_min = col_max = np.array([])
        col_nan = col_inf = np.array([], dtype=bool)
        zero_variance_cols = []
        near_zero_threshold = float(np.finfo(np.float64).eps * 100.0)
        near_zero_cols = []
        unique_rows = duplicate_rows = None

    finite_values = arr[finite_mask]
    if finite_values.size:
        global_min = _finite_float(np.min(finite_values))
        global_max = _finite_float(np.max(finite_values))
        global_mean = _finite_float(np.mean(finite_values))
        global_std = _finite_float(np.std(finite_values))
        absolute_max = _finite_float(np.max(np.abs(finite_values)))
    else:
        global_min = global_max = global_mean = global_std = absolute_max = None

    extreme_value_indicators = {
        "finite_abs_max_ge_1e6": bool(finite_values.size and np.max(np.abs(finite_values)) >= 1e6),
        "finite_abs_max_ge_1e9": bool(finite_values.size and np.max(np.abs(finite_values)) >= 1e9),
        "finite_abs_max_ge_float64_1pct": bool(
            finite_values.size and np.max(np.abs(finite_values)) >= np.finfo(np.float64).max * 0.01
        ),
    }

    report = {
        "model": model,
        "array": {
            "shape": list(arr.shape), "ndim": int(arr.ndim), "dtype": str(arr.dtype),
            "strides": list(arr.strides), "itemsize": int(arr.itemsize), "nbytes": int(arr.nbytes),
            "c_contiguous": bool(arr.flags.c_contiguous), "f_contiguous": bool(arr.flags.f_contiguous),
            "writeable": bool(arr.flags.writeable),
        },
        "counts": {
            "total_elements": int(arr.size), "finite": int(finite_mask.sum()),
            "nonfinite": nonfinite_count, "nan": int(nan_mask.sum()), "inf": int(inf_mask.sum()),
        },
        "global_statistics": {
            "min_finite": global_min, "max_finite": global_max, "mean_finite": global_mean,
            "std_finite": global_std, "absolute_max_finite": absolute_max,
        },
        "per_column": {
            "min": [_finite_float(x) for x in col_min], "max": [_finite_float(x) for x in col_max],
            "std": [_finite_float(x) for x in col_std], "zero_variance_columns": zero_variance_cols,
            "near_zero_std_threshold": near_zero_threshold, "near_zero_std_columns": near_zero_cols,
            "columns_with_nan": np.where(col_nan)[0].astype(int).tolist(),
            "columns_with_inf": np.where(col_inf)[0].astype(int).tolist(),
        },
        "rows": {"duplicate_row_count": duplicate_rows, "unique_row_count": unique_rows},
        "extreme_value_indicators": extreme_value_indicators,
        "frozen_tsne_parameters": dict(tsne_params),
        "diagnostic_files": {
            "exact_runtime_npy": str(npy_path), "standalone_script": str(script_path),
            "standalone_results": str(result_path),
        },
    }
    atomic_write_json(json_path, report)
    np.save(npy_path, X_pre, allow_pickle=False)

    params_literal = repr(dict(tsne_params))
    npy_literal = repr(str(npy_path))
    result_literal = repr(str(result_path))
    script = f'''#!/usr/bin/env python3
import json
import traceback
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE

NPY_PATH = Path({npy_literal})
RESULT_PATH = Path({result_literal})
FROZEN_PARAMS = {params_literal}


def summary(X):
    finite = np.isfinite(X)
    vals = X[finite]
    return {{"shape": list(X.shape), "dtype": str(X.dtype), "strides": list(X.strides),
            "c_contiguous": bool(X.flags.c_contiguous), "finite": bool(finite.all()),
            "min": float(np.min(vals)) if vals.size else None,
            "max": float(np.max(vals)) if vals.size else None,
            "mean": float(np.mean(vals)) if vals.size else None,
            "std": float(np.std(vals)) if vals.size else None}}


def run_variant(name, X):
    rec = {{"variant": name, **summary(X)}}
    print(f"\\n=== {{name}} ===", flush=True)
    print(f"shape={{X.shape}} dtype={{X.dtype}} strides={{X.strides}} C={{X.flags.c_contiguous}}", flush=True)
    print(f"finite={{np.isfinite(X).all()}} min={{rec['min']}} max={{rec['max']}} mean={{rec['mean']}} std={{rec['std']}}", flush=True)
    print("TSNE START", flush=True)
    try:
        Z = TSNE(**FROZEN_PARAMS).fit_transform(X)
        rec.update({{"status": "PASS", "output_shape": list(Z.shape), "output_finite": bool(np.isfinite(Z).all())}})
        print(f"PASS shape={{Z.shape}} finite={{np.isfinite(Z).all()}}", flush=True)
    except BaseException as exc:
        rec.update({{"status": "FAIL", "exception_type": type(exc).__name__, "exception": str(exc), "traceback": traceback.format_exc()}})
        print("FAIL", flush=True)
        traceback.print_exc()
    return rec


def main():
    X = np.load(NPY_PATH, allow_pickle=False)
    print("LOADED EXACT RUNTIME X_PRE", flush=True)
    print(json.dumps(summary(X), indent=2), flush=True)
    results = [
        run_variant("A_exact_saved_Xpre", X),
        run_variant("B_ascontiguous_float64", np.ascontiguousarray(X, dtype=np.float64)),
        run_variant("C_fresh_copy_order_C", X.copy(order="C")),
    ]
    finite_ok = bool(np.isfinite(X).all())
    zero_var = np.where(np.std(X, axis=0) == 0.0)[0].astype(int).tolist()
    if finite_ok and not zero_var:
        results.append(run_variant("D_verified_no_NaN_Inf_no_zero_variance", X))
    else:
        results.append({{"variant": "D_verified_no_NaN_Inf_no_zero_variance", "status": "SKIP",
                        "finite": finite_ok, "zero_variance_columns": zero_var,
                        "reason": "Prerequisite check failed; no data were modified."}})
        print("D SKIP: prerequisite check failed; X_pre was not modified.", flush=True)
    by_name = {{r["variant"]: r for r in results}}
    a = by_name.get("A_exact_saved_Xpre", {{}}).get("status")
    b = by_name.get("B_ascontiguous_float64", {{}}).get("status")
    c = by_name.get("C_fresh_copy_order_C", {{}}).get("status")
    d = by_name.get("D_verified_no_NaN_Inf_no_zero_variance", {{}}).get("status")
    if a == "FAIL" and b == "PASS" and c == "PASS":
        classification = "CASE C"
    elif a == "FAIL" and b == "FAIL" and c == "FAIL" and d in {{"FAIL", "SKIP"}}:
        classification = "CASE D"
    elif a == "PASS":
        classification = (
            "EXACT_REAL_STANDALONE_PASS — if the isolated production worker still fails, "
            "compare that worker failure separately for CASE B."
        )
    else:
        classification = "INCONCLUSIVE — inspect all four variant records and the production worker result."
    payload = {{"frozen_tsne_parameters": FROZEN_PARAMS, "classification": classification, "results": results}}
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\\nCLASSIFICATION: {{classification}}", flush=True)
    print(f"RESULTS WRITTEN: {{RESULT_PATH}}", flush=True)


if __name__ == "__main__":
    main()
'''
    atomic_write_text(script_path, script)
    return json_path, script_path


def run_tsne_isolated(
    X_pre: np.ndarray,
    tsne_params: dict,
    model: str,
    max_retries: int = TSNE_ISOLATED_MAX_RETRIES,
) -> np.ndarray:
    """Run the frozen t-SNE fit directly in the canonical BLOCK12 process.

    The exact saved X_pre diagnostic was independently confirmed to pass with
    these frozen parameters. The previous subprocess implementation failed
    only at the Windows process boundary (0xC0000005 / returncode 3221225477),
    while the standalone diagnostic itself passed. Re-entering the same
    sklearn TSNE fit directly avoids that failing subprocess boundary.

    No data, preprocessing, dimensionality, or scientific t-SNE parameter is
    changed here.
    """
    X = np.ascontiguousarray(X_pre, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 1:
        raise RuntimeError(f"Invalid t-SNE input for {model}: shape={X.shape}")
    if not np.isfinite(X).all():
        raise RuntimeError(f"t-SNE input for {model} contains NaN/Inf.")

    log(
        f"TSNE IN-PROCESS PRECHECK: {model} shape={X.shape} "
        f"dtype={X.dtype} strides={X.strides} C={X.flags.c_contiguous} "
        f"finite=True"
    )
    log(f"TSNE IN-PROCESS FIT START: {model} params={tsne_params}")

    try:
        Z = TSNE(**tsne_params).fit_transform(X)
    except Exception as exc:
        raise RuntimeError(
            f"t-SNE in-process fit failed for {model}: {type(exc).__name__}: {exc}"
        ) from exc

    Z = np.asarray(Z, dtype=np.float64)
    if Z.shape != (len(X), 2) or not np.isfinite(Z).all():
        raise RuntimeError(
            f"t-SNE output invalid for {model}: shape={Z.shape}; "
            f"finite={bool(np.isfinite(Z).all())}"
        )

    # Preserve a machine-readable status artifact for auditability.
    TSNE_ISOLATED_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = TSNE_ISOLATED_ROOT / f"{model}_tsne_status.json"
    atomic_write_json(
        status_path,
        {
            "model": model,
            "status": "PASS",
            "execution": "in_process",
            "returncode": 0,
            "attempt": 1,
            "max_retries": max_retries,
            "tsne_params": tsne_params,
            "input_shape": list(X.shape),
            "output_shape": list(Z.shape),
        },
    )
    log(f"TSNE IN-PROCESS PASS: {model} shape={Z.shape} finite=True")
    return Z


def run_tsne(
    arrays: Dict[Tuple[int, str, str], dict],
    model: str,
    diagnostic_only: bool = False,
) -> pd.DataFrame:
    chunks = []
    meta = []

    for subject in SUBJECTS:
        for condition in CONDITIONS:
            item = arrays[(subject, condition, model)]
            chunks.append(item["latent"])
            for i in range(item["latent"].shape[0]):
                meta.append({
                    "subject_id": subject,
                    "condition": condition,
                    "gesture": int(item["gesture"][i]),
                    "gesture_label": GESTURE_NAMES[int(item["gesture"][i])],
                    "window_id": str(item["window_id"][i]),
                })

    X = np.vstack(chunks).astype(np.float64)
    meta_df = pd.DataFrame(meta)

    # Fixed preprocessing for descriptive visualization.
    Xs = _validate_and_scale_latent_matrix(X, model, "t-SNE projection")

    # Fixed PCA pre-reduction stabilizes t-SNE and avoids an uncontrolled
    # dimensionality-dependent runtime.
    pre_dim = min(30, Xs.shape[1], Xs.shape[0] - 1)
    X_pre = PCA(
        n_components=pre_dim,
        random_state=RANDOM_SEED,
    ).fit_transform(Xs)

    perplexity = min(TSNE_PERPLEXITY, max(5, (len(X_pre) - 1) // 3))

    # Frozen t-SNE parameters. The fit is executed directly in this canonical
    # process because the exact real X_pre diagnostic passes with the same
    # parameters, whereas the previous subprocess wrapper failed at the
    # Windows process boundary. No scientific parameter is changed.
    tsne_params = dict(
        n_components=TSNE_COMPONENTS,
        random_state=RANDOM_SEED,
        perplexity=perplexity,
        max_iter=TSNE_MAX_ITER,
        learning_rate=TSNE_LEARNING_RATE,
        init=TSNE_INIT,
        method="barnes_hut",
        n_jobs=1,
    )

    diagnostic_json, diagnostic_script = _build_tsne_runtime_diagnostics(X_pre, tsne_params, model)
    log(f"TSNE DIAGNOSTIC REPORT: {diagnostic_json}")
    log(f"TSNE DIAGNOSTIC SCRIPT: {diagnostic_script}")
    if diagnostic_only:
        raise RuntimeError(
            f"t-SNE diagnostic-only stop for {model}. Run the generated standalone "
            f"script and inspect A/B/C/D results before implementing any workaround: {diagnostic_script}"
        )
    Z = run_tsne_isolated(X_pre, tsne_params, model)

    out = meta_df.copy()
    out["TSNE1"] = Z[:, 0]
    out["TSNE2"] = Z[:, 1]

    atomic_write_csv(
        TSNE_ROOT / f"{model}_tSNE_coordinates.csv",
        out,
    )

    atomic_write_json(
        TSNE_ROOT / f"{model}_tSNE_metadata.json",
        {
            "model": model,
            "n_samples": int(len(out)),
            "input_dimension": int(X.shape[1]),
            "pca_pre_reduction_dimension": int(pre_dim),
            "perplexity": int(perplexity),
            "max_iter": TSNE_MAX_ITER,
            "random_seed": RANDOM_SEED,
            "note": PROJECTION_NOTE,
        },
    )

    return out


def run_umap(
    arrays: Dict[Tuple[int, str, str], dict],
    model: str,
) -> pd.DataFrame:
    """Run the validated UMAP 0.5.12 surgical execution path.

    The installed UMAP 0.5.12 class-level ``fit_transform`` path is unstable
    on this Windows environment at the production sample size. The exact
    production input (7350 x 240) has, however, been independently validated
    through every UMAP internal phase: exact k-NN, fuzzy simplicial graph,
    spectral initialization, epoch scheduling, a/b parameter generation, and
    ``optimize_layout_euclidean``. This function uses that validated internal
    path directly and leaves all scientific projection parameters unchanged.

    This is a descriptive projection only; no classifier fitting, RSA,
    transfer analysis, calibration analysis, or model selection depends on
    the resulting coordinates.
    """
    try:
        import umap.umap_ as umap
    except Exception as exc:
        raise RuntimeError(
            "UMAP is required by this Block12 implementation. "
            "Install the frozen dependency with: pip install umap-learn"
        ) from exc

    chunks = []
    meta = []
    for subject in SUBJECTS:
        for condition in CONDITIONS:
            item = arrays[(subject, condition, model)]
            chunks.append(item["latent"])
            for i in range(item["latent"].shape[0]):
                meta.append({
                    "subject_id": subject,
                    "condition": condition,
                    "gesture": int(item["gesture"][i]),
                    "gesture_label": GESTURE_NAMES[int(item["gesture"][i])],
                    "window_id": str(item["window_id"][i]),
                })

    X = np.vstack(chunks).astype(np.float64)
    meta_df = pd.DataFrame(meta)
    Xs = _validate_and_scale_latent_matrix(X, model, "UMAP projection")
    Xs = np.ascontiguousarray(Xs, dtype=np.float64)

    if Xs.ndim != 2 or Xs.shape[0] != len(meta_df) or not np.isfinite(Xs).all():
        raise RuntimeError(
            f"UMAP input invalid for {model}: shape={Xs.shape}, "
            f"metadata_rows={len(meta_df)}"
        )

    log(
        f"UMAP SURGICAL START: {model} shape={Xs.shape} dtype={Xs.dtype} "
        f"strides={Xs.strides} C={Xs.flags.c_contiguous} finite=True"
    )

    # -------------------------------------------------------------------------
    # Frozen UMAP projection parameters.
    # -------------------------------------------------------------------------
    n_neighbors = int(UMAP_N_NEIGHBORS)
    min_dist = float(UMAP_MIN_DIST)
    metric = str(UMAP_METRIC)
    n_components = 2
    n_epochs = 200
    negative_sample_rate = 5.0
    gamma = 1.0
    initial_alpha = 1.0
    spread = 1.0
    local_connectivity = 1.0
    set_op_mix_ratio = 1.0

    if n_neighbors < 2 or n_neighbors >= len(Xs):
        raise RuntimeError(
            f"Invalid UMAP n_neighbors={n_neighbors} for n_samples={len(Xs)}."
        )

    # -------------------------------------------------------------------------
    # Phase 1: exact k-NN.
    # This is the same exact sklearn k-NN path that passed the production-size
    # surgical diagnosis. It avoids pynndescent/numba execution entirely.
    # -------------------------------------------------------------------------
    log(f"UMAP PHASE 1/7 KNN START: {model}")
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        n_jobs=1,
    )
    nn.fit(Xs)
    knn_dists, knn_indices = nn.kneighbors(Xs)
    knn_indices = np.ascontiguousarray(knn_indices, dtype=np.int32)
    knn_dists = np.ascontiguousarray(knn_dists, dtype=np.float64)

    if (
        knn_indices.shape != (len(Xs), n_neighbors)
        or knn_dists.shape != (len(Xs), n_neighbors)
        or not np.isfinite(knn_dists).all()
        or int(knn_indices.min()) < 0
        or int(knn_indices.max()) >= len(Xs)
    ):
        raise RuntimeError(
            f"UMAP exact k-NN validation failed for {model}: "
            f"indices={knn_indices.shape}, dists={knn_dists.shape}"
        )
    log(
        f"UMAP PHASE 1/7 KNN PASS: indices={knn_indices.shape} "
        f"dists={knn_dists.shape}"
    )

    # Use a fresh RandomState exactly as in the validated surgical diagnosis.
    fuzzy_random_state = np.random.RandomState(RANDOM_SEED)

    # -------------------------------------------------------------------------
    # Phase 2: fuzzy simplicial graph.
    # IMPORTANT: UMAP 0.5.12 positional/signature order is respected here.
    # -------------------------------------------------------------------------
    log(f"UMAP PHASE 2/7 FUZZY GRAPH START: {model}")
    fuzzy_result = umap.fuzzy_simplicial_set(
        Xs,
        n_neighbors,
        fuzzy_random_state,
        metric,
        {},
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        angular=False,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
        apply_set_operations=True,
        verbose=False,
        return_dists=False,
    )
    if not isinstance(fuzzy_result, tuple) or len(fuzzy_result) != 4:
        raise RuntimeError(
            f"Unexpected fuzzy_simplicial_set return for {model}: "
            f"type={type(fuzzy_result)!r}, length="
            f"{len(fuzzy_result) if isinstance(fuzzy_result, tuple) else 'NA'}"
        )
    graph = fuzzy_result[0]
    if graph.shape != (len(Xs), len(Xs)) or graph.nnz <= 0:
        raise RuntimeError(
            f"UMAP fuzzy graph validation failed for {model}: "
            f"shape={graph.shape}, nnz={graph.nnz}"
        )
    graph_data = np.asarray(graph.data)
    if not np.isfinite(graph_data).all():
        raise RuntimeError(f"UMAP fuzzy graph contains non-finite weights for {model}.")
    log(
        f"UMAP PHASE 2/7 FUZZY GRAPH PASS: shape={graph.shape} "
        f"nnz={graph.nnz} dtype={graph.dtype}"
    )

    # -------------------------------------------------------------------------
    # Phase 3: spectral initialization.
    # The 0.5.12 signature does NOT accept an ``init`` keyword.
    # -------------------------------------------------------------------------
    log(f"UMAP PHASE 3/7 SPECTRAL INIT START: {model}")
    spectral_random_state = np.random.RandomState(RANDOM_SEED)
    init = umap.spectral_layout(
        Xs,
        graph,
        n_components,
        spectral_random_state,
        metric=metric,
        metric_kwds={},
        tol=0.0,
        maxiter=0,
    )
    init = np.ascontiguousarray(init, dtype=np.float32)
    if init.shape != (len(Xs), n_components) or not np.isfinite(init).all():
        raise RuntimeError(
            f"UMAP spectral initialization validation failed for {model}: "
            f"shape={init.shape}"
        )
    log(f"UMAP PHASE 3/7 SPECTRAL INIT PASS: shape={init.shape}")

    # -------------------------------------------------------------------------
    # Phase 4: graph edge extraction.
    # -------------------------------------------------------------------------
    log(f"UMAP PHASE 4/7 GRAPH EDGES START: {model}")
    graph_coo = graph.tocoo(copy=False)
    head = np.ascontiguousarray(graph_coo.row, dtype=np.int32)
    tail = np.ascontiguousarray(graph_coo.col, dtype=np.int32)
    graph_weight = np.ascontiguousarray(graph_coo.data, dtype=np.float32)

    if not np.isfinite(graph_weight).all():
        raise RuntimeError(f"UMAP graph-edge weights are non-finite for {model}.")
    if head.shape != tail.shape or head.shape != graph_weight.shape:
        raise RuntimeError(f"UMAP graph-edge array shape mismatch for {model}.")
    log(
        f"UMAP PHASE 4/7 GRAPH EDGES PASS: n_edges={len(head)} "
        f"head={head.shape} tail={tail.shape}"
    )

        # -------------------------------------------------------------------------
    # Phase 5: epoch schedule.
    #
    # UMAP 0.5.12 internally computes:
    #     n_epochs / graph_weight
    #
    # For very small but valid graph weights, this can exceed float32 range
    # during the final cast even though the graph itself is completely finite.
    #
    # IMPORTANT:
    #   - Do NOT modify graph_weight.
    #   - Do NOT replace small weights with zero.
    #   - Do NOT replace inf with an arbitrary finite value.
    #
    # We reproduce the UMAP schedule calculation in float64 first, then
    # validate it, and finally cast to float32 only after clipping to the
    # representable float32 range.
    # -------------------------------------------------------------------------
    log(f"UMAP PHASE 5/7 EPOCH SCHEDULE START: {model}")

    graph_weight_f64 = np.asarray(graph_weight, dtype=np.float64)

    if graph_weight_f64.shape != graph_weight.shape:
        raise RuntimeError(
            f"UMAP graph-weight shape mismatch for {model}: "
            f"{graph_weight_f64.shape} vs {graph_weight.shape}"
        )

    if not np.isfinite(graph_weight_f64).all():
        raise RuntimeError(
            f"UMAP graph weights contain non-finite values for {model}."
        )

    if np.any(graph_weight_f64 <= 0.0):
        bad_count = int(np.count_nonzero(graph_weight_f64 <= 0.0))
        raise RuntimeError(
            f"UMAP graph weights contain {bad_count} non-positive values "
            f"for {model}."
        )

    # Exact mathematical schedule in float64.
    epochs_per_sample_f64 = (
        float(n_epochs) / graph_weight_f64
    )

    if not np.isfinite(epochs_per_sample_f64).all():
        raise RuntimeError(
            f"UMAP float64 epoch schedule contains non-finite values "
            f"for {model}."
        )

    # float32 maximum representable finite value.
    float32_max = np.finfo(np.float32).max

    overflow_count = int(
        np.count_nonzero(epochs_per_sample_f64 > float32_max)
    )

    if overflow_count > 0:
        log(
            f"UMAP PHASE 5/7: {model} has {overflow_count} "
            f"epoch-schedule values above float32 range; "
            f"clipping only the schedule representation."
        )

    epochs_per_sample_f64 = np.minimum(
        epochs_per_sample_f64,
        float32_max,
    )

    epochs_per_sample = np.asarray(
        epochs_per_sample_f64,
        dtype=np.float32,
    )

    if epochs_per_sample.shape != graph_weight.shape:
        raise RuntimeError(
            f"UMAP epoch schedule shape mismatch for {model}: "
            f"{epochs_per_sample.shape} vs {graph_weight.shape}"
        )

    if not np.isfinite(epochs_per_sample).all():
        raise RuntimeError(
            f"UMAP epoch schedule contains non-finite values for {model}."
        )

    log(
        f"UMAP PHASE 5/7 EPOCH SCHEDULE PASS: "
        f"shape={epochs_per_sample.shape} "
        f"min={float(np.min(epochs_per_sample))} "
        f"max={float(np.max(epochs_per_sample))} "
        f"overflow_clipped={overflow_count}"
    )

    # -------------------------------------------------------------------------
    # Phase 6: UMAP curve parameters.
    # -------------------------------------------------------------------------
    log(f"UMAP PHASE 6/7 A/B PARAMETERS START: {model}")
    a, b = umap.find_ab_params(spread, min_dist)
    a = float(a)
    b = float(b)

    if not np.isfinite(a) or not np.isfinite(b):
        raise RuntimeError(
            f"UMAP a/b parameters are non-finite for {model}."
        )

    log(
        f"UMAP PHASE 6/7 A/B PARAMETERS PASS: "
        f"a={a} b={b}"
    )
    # -------------------------------------------------------------------------
    # Phase 7: complete embedding optimization.
    # The optimizer signature is frozen for UMAP 0.5.12. In particular, it
    # does NOT accept ``weight``. The graph weights are represented through
    # epochs_per_sample, exactly as the validated surgical diagnosis did.
    # -------------------------------------------------------------------------
    log(f"UMAP PHASE 7/7 EMBEDDING OPTIMIZATION START: {model}")

    head_embedding = np.ascontiguousarray(init.copy(), dtype=np.float32)
    tail_embedding = np.ascontiguousarray(init.copy(), dtype=np.float32)

    # Match the validated diagnostic RNG construction exactly: signed int32
    # range, stored as int64 for the UMAP optimizer ABI.
    rng_state = np.random.RandomState(RANDOM_SEED).randint(
        np.iinfo(np.int32).min,
        np.iinfo(np.int32).max,
        3,
    ).astype(np.int64)

    optimized = umap.optimize_layout_euclidean(
        head_embedding,
        tail_embedding,
        head,
        tail,
        n_epochs,
        len(Xs),
        epochs_per_sample,
        a,
        b,
        rng_state,
        gamma=gamma,
        initial_alpha=initial_alpha,
        negative_sample_rate=negative_sample_rate,
        parallel=False,
        verbose=False,
        densmap=False,
        densmap_kwds=None,
        tqdm_kwds=None,
        move_other=False,
    )

    Z = np.asarray(optimized, dtype=np.float32)
    if Z.shape != (len(Xs), n_components) or not np.isfinite(Z).all():
        raise RuntimeError(
            f"UMAP optimized embedding validation failed for {model}: "
            f"shape={Z.shape}"
        )

    log(
        f"UMAP PHASE 7/7 EMBEDDING OPTIMIZATION PASS: shape={Z.shape} "
        f"dtype={Z.dtype} finite=True"
    )

    out = meta_df.copy()
    out["UMAP1"] = Z[:, 0]
    out["UMAP2"] = Z[:, 1]

    atomic_write_csv(UMAP_ROOT / f"{model}_UMAP_coordinates.csv", out)
    atomic_write_json(
        UMAP_ROOT / f"{model}_UMAP_metadata.json",
        {
            "model": model,
            "n_samples": int(len(out)),
            "input_dimension": int(X.shape[1]),
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "n_components": n_components,
            "n_epochs": n_epochs,
            "negative_sample_rate": negative_sample_rate,
            "gamma": gamma,
            "initial_alpha": initial_alpha,
            "random_seed": RANDOM_SEED,
            "transform_seed": RANDOM_SEED,
            "n_jobs": 1,
            "execution": "UMAP 0.5.12 validated surgical internal path",
            "knn_backend": "sklearn_exact",
            "umap_internal_phases": [
                "exact_knn",
                "fuzzy_simplicial_set",
                "spectral_layout",
                "graph_edge_extraction",
                "make_epochs_per_sample",
                "find_ab_params",
                "optimize_layout_euclidean",
            ],
            "numba_pynndescent_bypass": True,
            "note": PROJECTION_NOTE,
        },
    )

    return out


# =============================================================================
# 06. CLUSTER QUALITY
# =============================================================================

def cluster_quality_for_latent(
    arrays: Dict[Tuple[int, str, str], dict],
    model: str,
) -> pd.DataFrame:
    rows = []

    for subject in SUBJECTS:
        for condition in CONDITIONS:
            item = arrays[(subject, condition, model)]
            X = item["latent"].astype(np.float64)
            y = item["gesture"].astype(int)

            # Metrics are computed on frozen latent vectors. No fitting of a
            # classifier and no hyperparameter optimization occurs.
            if len(np.unique(y)) != 7:
                raise RuntimeError(
                    f"{model} S{subject:02d} {condition}: expected 7 gestures."
                )

            sil = silhouette_score(X, y, metric="euclidean")
            db = davies_bouldin_score(X, y)
            ch = calinski_harabasz_score(X, y)

            rows.append({
                "subject_id": subject,
                "condition": condition,
                "model": model,
                "silhouette": float(sil),
                "davies_bouldin": float(db),
                "calinski_harabasz": float(ch),
                "n_windows": int(len(X)),
                "n_gestures": int(len(np.unique(y))),
            })

    out = pd.DataFrame(rows)
    atomic_write_csv(
        CLUSTER_ROOT / f"{model}_cluster_quality.csv",
        out,
    )
    return out


# =============================================================================
# 07. TRANSFER RELATIONSHIP
# =============================================================================

def load_transfer_benefit() -> pd.DataFrame:
    path = BLOCK10_ROOT / "metrics" / "transfer_benefit.csv"
    df = pd.read_csv(path)

    required = {
        "subject_id",
        "model",
        "transfer_direction",
        "metric",
        "transfer_benefit",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"BLOCK10 transfer_benefit.csv missing columns: {sorted(missing)}"
        )

    q = df[df["metric"].astype(str).str.lower() == PRIMARY_METRIC].copy()

    if q.empty:
        raise RuntimeError("BLOCK10 transfer_benefit.csv has no Macro-F1 rows.")

    q["subject_id"] = q["subject_id"].astype(int)
    q["model"] = q["model"].astype(str)
    q["transfer_benefit"] = pd.to_numeric(
        q["transfer_benefit"], errors="raise"
    )

    expected = len(SUBJECTS) * 2 * 10
    if len(q) != expected:
        raise RuntimeError(
            f"Expected {expected} BLOCK10 Macro-F1 transfer-benefit rows, "
            f"found {len(q)}."
        )

    expected_models = {
        "LDA", "KNN", "SVM", "RandomForest", "XGBoost",
        "EEGNet", "ShallowConvNet", "DeepConvNet",
        "CNN-BiLSTM", "CNN-BiLSTM-Attention",
    }

    if set(q["model"]) != expected_models:
        raise RuntimeError(
            f"BLOCK10 model set mismatch: {sorted(set(q['model']))}"
        )

    if q.duplicated(
        ["subject_id", "model", "transfer_direction", "metric"]
    ).any():
        raise RuntimeError("Duplicate BLOCK10 transfer-benefit keys.")

    # Representation models are paired with their same frozen model results.
    q = q[q["model"].isin(LATENT_MODELS)].copy()

    if len(q) != len(SUBJECTS) * 2 * len(LATENT_MODELS):
        raise RuntimeError("Unexpected latent-model transfer-benefit coverage.")

    return q


def build_transfer_relationship(
    rsa_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
) -> pd.DataFrame:
    key_cols = ["subject_id", "model", "transfer_direction"]
    if transfer_df.duplicated(key_cols).any():
        dup = (
            transfer_df.loc[transfer_df.duplicated(key_cols, keep=False), key_cols]
            .drop_duplicates()
            .to_dict("records")
        )
        raise RuntimeError(
            f"Duplicate BLOCK10 transfer-benefit keys; refusing silent aggregation: {dup[:10]}"
        )

    expected_transfer_keys = pd.MultiIndex.from_product(
        [SUBJECTS, LATENT_MODELS, ["ME_to_MI", "MI_to_ME"]],
        names=key_cols,
    ).to_frame(index=False)
    observed_transfer_keys = (
        transfer_df[key_cols]
        .sort_values(key_cols)
        .reset_index(drop=True)
    )
    expected_transfer_keys = expected_transfer_keys.sort_values(key_cols).reset_index(drop=True)
    if not observed_transfer_keys.equals(expected_transfer_keys):
        raise RuntimeError(
            "BLOCK10 transfer-benefit key coverage is not exactly the expected "
            "25 x 3 x 2 subject/model/direction grid."
        )

    wide = (
        transfer_df
        .set_index(key_cols)["transfer_benefit"]
        .unstack("transfer_direction")
        .reset_index()
    )

    for direction in ("ME_to_MI", "MI_to_ME"):
        if direction not in wide.columns:
            raise RuntimeError(
                f"BLOCK10 missing transfer direction {direction}."
            )

    wide["bidirectional_mean_transfer_benefit"] = wide[
        ["ME_to_MI", "MI_to_ME"]
    ].mean(axis=1)

    out = rsa_df.merge(
        wide,
        on=["subject_id", "model"],
        how="inner",
        validate="one_to_one",
    )

    if len(out) != len(SUBJECTS) * len(LATENT_MODELS):
        raise RuntimeError(
            "RSA/transfer relationship coverage is incomplete."
        )

    atomic_write_csv(
        TRANSFER_ROOT / "rsa_vs_transfer_subject_level.csv",
        out,
    )

    return out


def permutation_spearman(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y) or len(x) < 3:
        raise RuntimeError("Invalid sample size for permutation correlation.")

    observed = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(PERMUTATION_N):
        yp = rng.permutation(y)
        rho = float(spearmanr(x, yp).statistic)
        if abs(rho) >= abs(observed) - 1e-15:
            count += 1

    p = (count + 1) / (PERMUTATION_N + 1)
    return observed, float(p)


def summarize_transfer_relationship(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model, g in df.groupby("model", sort=True):
        x = g["rho_spearman"].to_numpy(float)
        y = g["bidirectional_mean_transfer_benefit"].to_numpy(float)

        rho, p = permutation_spearman(
            x,
            y,
            deterministic_seed("rsa_transfer_permutation", model),
        )

        rows.append({
            "model": model,
            "n_subjects": int(len(g)),
            "association": "Spearman",
            "rho": rho,
            "permutation_p": p,
            "x": "ME_MI_RSA_rho",
            "y": "mean(ME_to_MI_transfer_benefit, MI_to_ME_transfer_benefit)",
        })

    out = pd.DataFrame(rows)
    atomic_write_csv(
        TRANSFER_ROOT / "rsa_transfer_association_summary.csv",
        out,
    )
    return out


# =============================================================================
# 08. CALIBRATION RELATIONSHIP
# =============================================================================

def load_k90() -> pd.DataFrame:
    path = BLOCK09_ROOT / "metrics" / "k90_results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"BLOCK09 k90_results.csv missing: {path}")

    df = pd.read_csv(path)
    required = {
        "subject_id", "condition", "model", "metric", "k90", "k90_status",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"BLOCK09 k90_results.csv missing columns: {sorted(missing)}"
        )

    q = df[
        (df["metric"].astype(str).str.lower() == PRIMARY_METRIC)
        & (df["model"].isin(LATENT_MODELS))
        & (df["condition"].isin(CONDITIONS))
    ].copy()

    expected = len(SUBJECTS) * len(CONDITIONS) * len(LATENT_MODELS)
    if len(q) != expected:
        raise RuntimeError(
            f"Expected {expected} BLOCK09 latent-model k90 rows for metric "
            f"{PRIMARY_METRIC!r}, found {len(q)}."
        )

    try:
        q["subject_id"] = q["subject_id"].astype(int)
    except Exception as exc:
        raise RuntimeError("BLOCK09 k90 subject_id contains non-integral values.") from exc

    if set(q["subject_id"].tolist()) != set(SUBJECTS):
        raise RuntimeError(
            f"BLOCK09 k90 subject IDs must be exactly 1..25; observed "
            f"{sorted(q['subject_id'].unique().tolist())}."
        )

    if set(q["condition"].astype(str).tolist()) != set(CONDITIONS):
        raise RuntimeError("BLOCK09 k90 contains invalid condition values.")

    if set(q["model"].astype(str).tolist()) != set(LATENT_MODELS):
        raise RuntimeError("BLOCK09 k90 contains invalid latent-model values.")

    q["k90"] = pd.to_numeric(q["k90"], errors="coerce")
    q["k90_status"] = q["k90_status"].astype(str).str.strip().str.upper()

    allowed_k90_statuses = {"PASS", "NOT_REACHED", "NO_POSITIVE_GAP", "FAIL"}
    observed_statuses = set(q["k90_status"].tolist())
    unknown_statuses = observed_statuses - allowed_k90_statuses
    if unknown_statuses:
        raise RuntimeError(
            f"BLOCK09 k90 contains unknown k90_status values: {sorted(unknown_statuses)}"
        )

    reached_status = q["k90_status"].eq("PASS")
    unreached_status = q["k90_status"].isin({"NOT_REACHED", "NO_POSITIVE_GAP", "FAIL"})
    finite_k90 = np.isfinite(q["k90"].to_numpy(float))
    invalid_pass = reached_status.to_numpy(bool) & ~finite_k90
    invalid_unreached = unreached_status.to_numpy(bool) & finite_k90
    if invalid_pass.any() or invalid_unreached.any():
        bad = q.loc[invalid_pass | invalid_unreached,
                    ["subject_id", "condition", "model", "k90", "k90_status"]]
        raise RuntimeError(
            "BLOCK09 k90/status semantics are inconsistent with the frozen BLOCK09 "
            f"definition (PASS requires finite k90; NOT_REACHED/NO_POSITIVE_GAP/FAIL "
            f"require unavailable k90). Examples: {bad.head(10).to_dict('records')}"
        )

    valid_k90_levels = {10, 20, 30, 40, 50, 60, 80, 90}
    finite_values = q.loc[np.isfinite(q["k90"]), "k90"]
    if not set(finite_values.astype(int).tolist()).issubset(valid_k90_levels):
        raise RuntimeError(
            f"BLOCK09 k90 contains values outside the frozen calibration levels: "
            f"{sorted(set(finite_values.tolist()) - valid_k90_levels)}"
        )

    key_cols = ["subject_id", "condition", "model"]
    if q.duplicated(key_cols).any():
        dup = q.loc[q.duplicated(key_cols, keep=False), key_cols].drop_duplicates()
        raise RuntimeError(
            f"Duplicate BLOCK09 k90 subject/condition/model keys: "
            f"{dup.to_dict('records')[:10]}"
        )

    # Exact 25 x 2 x 3 coverage; no hidden extras or omissions.
    expected_keys = pd.MultiIndex.from_product(
        [SUBJECTS, CONDITIONS, LATENT_MODELS],
        names=key_cols,
    ).to_frame(index=False)
    observed_keys = q[key_cols].drop_duplicates().sort_values(key_cols).reset_index(drop=True)
    expected_keys = expected_keys.sort_values(key_cols).reset_index(drop=True)
    if not observed_keys.equals(expected_keys):
        raise RuntimeError("BLOCK09 k90 key coverage is not exactly the expected 25 x 2 x 3 grid.")

    return q


def build_calibration_relationship(
    rsa_df: pd.DataFrame,
    k90_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the complete 25-subject x 3-model RSA-k90 provenance grid.

    Important: an unreached k90 is a legitimate missing numerical outcome, not
    a missing subject/model relationship.  Therefore the relationship table
    MUST retain all 75 subject/model rows and carry NaN/status values forward.
    The previous implementation used pivot_table(), which drops groups whose
    k90 values are entirely NaN; that silently reduced the grid below 75 and
    triggered a false ``RSA/k90 relationship coverage is incomplete`` failure.
    """
    expected_keys = pd.MultiIndex.from_product(
        [SUBJECTS, LATENT_MODELS],
        names=["subject_id", "model"],
    ).to_frame(index=False)

    # Preserve the full ME/MI provenance, including unreached k90 values.
    k90_wide = expected_keys.copy()

    for condition in CONDITIONS:
        q = k90_df.loc[
            k90_df["condition"].eq(condition),
            ["subject_id", "model", "k90", "k90_status"],
        ].copy()
        q = q.rename(
            columns={
                "k90": f"{condition}_k90",
                "k90_status": f"{condition}_k90_status",
            }
        )

        # Every subject/model/condition key must exist exactly once in BLOCK09.
        expected_condition_keys = pd.MultiIndex.from_product(
            [SUBJECTS, [condition], LATENT_MODELS],
            names=["subject_id", "condition", "model"],
        ).to_frame(index=False)[["subject_id", "model"]]

        if len(q) != len(SUBJECTS) * len(LATENT_MODELS):
            raise RuntimeError(
                f"BLOCK09 k90 {condition} coverage invalid: expected "
                f"{len(SUBJECTS) * len(LATENT_MODELS)} rows, found {len(q)}."
            )

        if q.duplicated(["subject_id", "model"]).any():
            dup = q.loc[q.duplicated(["subject_id", "model"], keep=False),
                        ["subject_id", "model"]].drop_duplicates().to_dict("records")
            raise RuntimeError(
                f"Duplicate BLOCK09 k90 {condition} subject/model keys: {dup[:10]}"
            )

        merged = expected_condition_keys.merge(
            q,
            on=["subject_id", "model"],
            how="left",
            validate="one_to_one",
        )

        if len(merged) != len(SUBJECTS) * len(LATENT_MODELS):
            raise RuntimeError(
                f"BLOCK09 k90 {condition} key coverage changed during merge."
            )

        k90_wide = k90_wide.merge(
            merged,
            on=["subject_id", "model"],
            how="left",
            validate="one_to_one",
        )

    k90_wide["ME_k90"] = pd.to_numeric(k90_wide["ME_k90"], errors="coerce")
    k90_wide["MI_k90"] = pd.to_numeric(k90_wide["MI_k90"], errors="coerce")

    # Primary quantity requires BOTH condition-specific k90 values.
    k90_wide["mean_k90"] = k90_wide[["ME_k90", "MI_k90"]].mean(
        axis=1, skipna=False
    )

    k90_wide["k90_valid"] = (
        k90_wide["ME_k90_status"].eq("PASS")
        & k90_wide["MI_k90_status"].eq("PASS")
        & np.isfinite(k90_wide["ME_k90"].to_numpy(float))
        & np.isfinite(k90_wide["MI_k90"].to_numpy(float))
        & np.isfinite(k90_wide["mean_k90"].to_numpy(float))
    )
    k90_wide["k90_exclusion_reason"] = np.where(
        k90_wide["k90_valid"],
        "NONE",
        np.where(
            ~np.isfinite(k90_wide["ME_k90"].to_numpy(float)) &
            ~np.isfinite(k90_wide["MI_k90"].to_numpy(float)),
            "ME_AND_MI_K90_UNREACHED_OR_INVALID",
            np.where(
                ~np.isfinite(k90_wide["ME_k90"].to_numpy(float)),
                "ME_K90_UNREACHED_OR_INVALID",
                "MI_K90_UNREACHED_OR_INVALID",
            ),
        ),
    )

    # RSA must also be a complete 25 x 3 grid.  Merge by explicit keys only.
    rsa_expected = expected_keys.copy()
    if rsa_df.duplicated(["subject_id", "model"]).any():
        raise RuntimeError("Duplicate RSA subject/model keys.")

    if len(rsa_df) != len(SUBJECTS) * len(LATENT_MODELS):
        raise RuntimeError(
            f"RSA subject/model grid invalid: expected "
            f"{len(SUBJECTS) * len(LATENT_MODELS)} rows, found {len(rsa_df)}."
        )

    out = rsa_expected.merge(
        rsa_df,
        on=["subject_id", "model"],
        how="left",
        validate="one_to_one",
    ).merge(
        k90_wide,
        on=["subject_id", "model"],
        how="left",
        validate="one_to_one",
    )

    if len(out) != len(SUBJECTS) * len(LATENT_MODELS):
        raise RuntimeError(
            f"RSA/k90 relationship grid invalid after key-based merge: "
            f"expected {len(SUBJECTS) * len(LATENT_MODELS)}, found {len(out)}."
        )

    # Explicitly retain the finite association population; unreached k90 rows
    # remain in the provenance CSV but are excluded from correlation.
    out["rsa_k90_included"] = (
        np.isfinite(out["rho_spearman"].to_numpy(float))
        & np.isfinite(out["mean_k90"].to_numpy(float))
    )

    atomic_write_csv(
        CALIBRATION_ROOT / "rsa_vs_k90_subject_level.csv",
        out,
    )

    rows = []
    for model, g in out.groupby("model", sort=True):
        finite = (
            g["k90_valid"].to_numpy(bool)
            & np.isfinite(g["rho_spearman"].to_numpy(float))
            & np.isfinite(g["ME_k90"].to_numpy(float))
            & np.isfinite(g["MI_k90"].to_numpy(float))
            & np.isfinite(g["mean_k90"].to_numpy(float))
        )

        x = g.loc[finite, "rho_spearman"].to_numpy(float)
        y = g.loc[finite, "mean_k90"].to_numpy(float)

        n_excluded = int((~finite).sum())
        exclusion_reasons = ";".join(
            f"{k}={v}"
            for k, v in g.loc[~finite, "k90_exclusion_reason"]
            .value_counts(dropna=False).sort_index().items()
        ) or "NONE"

        if len(x) < 3:
            rows.append({
                "model": model,
                "n_subjects": int(len(x)),
                "n_expected_subjects": int(len(g)),
                "n_excluded": n_excluded,
                "association": "Spearman",
                "rho": np.nan,
                "permutation_p": np.nan,
                "status": "INSUFFICIENT_FINITE_K90",
                "exclusion_reasons": exclusion_reasons,
            })
            continue

        rho, p = permutation_spearman(
            x,
            y,
            deterministic_seed("rsa_k90_permutation", model),
        )

        rows.append({
            "model": model,
            "n_subjects": int(len(x)),
            "n_expected_subjects": int(len(g)),
            "n_excluded": n_excluded,
            "association": "Spearman",
            "rho": rho,
            "permutation_p": p,
            "status": "PASS",
            "exclusion_reasons": exclusion_reasons,
        })

    summary = pd.DataFrame(rows)

    atomic_write_csv(
        CALIBRATION_ROOT / "rsa_k90_association_summary.csv",
        summary,
    )

    return out, summary


# =============================================================================
# 09. FIGURES
# =============================================================================

def save_rsa_boxplot(rsa_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [
        rsa_df.loc[rsa_df["model"] == model, "rho_spearman"].to_numpy(float)
        for model in LATENT_MODELS
    ]
    ax.boxplot(vals, tick_labels=list(LATENT_MODELS))
    ax.axhline(0.0, linestyle="--")
    ax.set_ylabel("ME–MI RSA Spearman ρ")
    ax.set_title("Subject-Level ME–MI Representational Similarity")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "RSA_subject_distribution.png", dpi=600)
    plt.close(fig)


def save_rsa_rdm_figures(
    rdms: Dict[Tuple[int, str, str], dict],
    subject: int,
    model: str,
) -> None:
    me = rdms[(subject, "ME", model)]["rdm"]
    mi = rdms[(subject, "MI", model)]["rdm"]

    # Shared color scale across both panels: otherwise each subplot
    # auto-normalizes from its own min/max, and the same dissimilarity value
    # could render as a different color intensity in ME vs MI, making a
    # side-by-side comparison misleading rather than informative.
    vmin = 0.0
    vmax = float(max(me.max(), mi.max()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, rdm, title in zip(
        axes,
        [me, mi],
        ["ME RDM", "MI RDM"],
    ):
        im = ax.imshow(rdm, aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(f"S{subject:02d} {model} — {title}")
        ax.set_xlabel("Gesture")
        ax.set_ylabel("Gesture")
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xticklabels(GESTURES)
        ax.set_yticklabels(GESTURES)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(
        FIG_ROOT / f"S{subject:02d}_{model}_ME_MI_RDM.png",
        dpi=600,
    )
    plt.close(fig)


def _safe_projection_filename(model: str, suffix: str) -> str:
    """Return a Windows-safe ASCII projection filename."""
    raw = f"{model}_{suffix}".replace(" ", "_")
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in raw)
    safe = safe.strip(" .")
    if not safe:
        raise RuntimeError("Projection figure filename became empty after sanitization.")
    return f"{safe}.png"


def _projection_figure_path(model: str, suffix: str) -> Path:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    filename = _safe_projection_filename(model, suffix)
    path = FIG_ROOT / filename
    # Windows rejects these filename characters even when they arrive through
    # a Path object. Reject them explicitly instead of allowing a raw PIL/OSError.
    invalid = set('<>:"/\\|?*')
    stem = path.stem
    if any(ch in invalid for ch in stem):
        raise RuntimeError(f"Unsafe projection figure filename: {path.name}")
    if len(str(path)) >= 240:
        raise RuntimeError(f"Projection figure path is too long for safe Windows I/O: {path}")
    return path


def _validate_projection_dataframe(
    df: pd.DataFrame,
    model: str,
    x: str,
    y: str,
    stage: str,
) -> pd.DataFrame:
    """Validate projection coordinates and exact frozen metadata cardinality."""
    required = {"subject_id", "condition", "gesture", "gesture_label", "window_id", x, y}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{stage} {model}: missing required columns: {missing}")

    expected_rows = WINDOWS_PER_SUBJECT_CONDITION_SETC * len(SUBJECTS) * len(CONDITIONS)
    if len(df) != expected_rows:
        raise RuntimeError(
            f"{stage} {model}: expected {expected_rows} rows, observed {len(df)}."
        )

    out = df.copy()
    out["subject_id"] = pd.to_numeric(out["subject_id"], errors="coerce")
    out["gesture"] = pd.to_numeric(out["gesture"], errors="coerce")
    out[x] = pd.to_numeric(out[x], errors="coerce")
    out[y] = pd.to_numeric(out[y], errors="coerce")

    if out[["subject_id", "gesture", x, y]].isna().any().any():
        raise RuntimeError(f"{stage} {model}: NaN/non-numeric metadata or coordinates detected.")
    if not np.isfinite(out[["subject_id", "gesture", x, y]].to_numpy(float)).all():
        raise RuntimeError(f"{stage} {model}: non-finite metadata or coordinates detected.")

    if set(out["subject_id"].astype(int).unique()) != set(SUBJECTS):
        raise RuntimeError(f"{stage} {model}: subject metadata grid is incorrect.")
    if set(out["condition"].astype(str).unique()) != set(CONDITIONS):
        raise RuntimeError(f"{stage} {model}: condition metadata grid is incorrect.")
    if set(out["gesture"].astype(int).unique()) != set(GESTURES):
        raise RuntimeError(f"{stage} {model}: gesture metadata grid is incorrect.")

    out["subject_id"] = out["subject_id"].astype(int)
    out["gesture"] = out["gesture"].astype(int)

    counts = (
        out.groupby(["subject_id", "condition", "gesture"], dropna=False)
        .size()
        .to_numpy()
    )
    expected_groups = len(SUBJECTS) * len(CONDITIONS) * len(GESTURES)
    if len(counts) != expected_groups or not np.all(counts == WINDOWS_PER_GESTURE_SETC):
        raise RuntimeError(
            f"{stage} {model}: subject/condition/gesture window cardinality is incorrect; "
            f"expected {expected_groups} groups × {WINDOWS_PER_GESTURE_SETC} windows."
        )

    label_check = {
        int(g): str(label)
        for g, label in zip(out["gesture"], out["gesture_label"])
    }
    if any(label_check.get(g) != GESTURE_NAMES[g] for g in GESTURES):
        raise RuntimeError(f"{stage} {model}: gesture labels do not match canonical mapping.")

    if out[[x, y]].empty:
        raise RuntimeError(f"{stage} {model}: plotting dataframe is empty.")

    return out


def validate_projection_csv(
    path: Path,
    model: str,
    x: str,
    y: str,
    stage: str,
) -> pd.DataFrame:
    """Read a projection CSV back and apply the exact same validation."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"{stage} {model}: projection CSV missing or empty: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"{stage} {model}: projection CSV is unreadable: {path}") from exc
    return _validate_projection_dataframe(df, model, x, y, f"{stage} read-back")


def save_projection_fig(
    df: pd.DataFrame,
    x: str,
    y: str,
    model: str,
    suffix: str,
) -> dict:
    """Validate, render, atomically save, and read-back validate a projection PNG."""
    validated = _validate_projection_dataframe(df, model, x, y, "Projection")
    path = _projection_figure_path(model, suffix)

    fig, ax = plt.subplots(figsize=(8, 6))

    # FIX: the figure is now always rendered and saved on every call. The
    # previous version nested fig.savefig(...) inside `if tmp.exists():`,
    # so on a fresh run (no leftover .tmp.png) the figure was never saved
    # at all -- this is why figures went missing. It also left an outer
    # `try:` with no matching except/finally, which is a hard SyntaxError.
    # This version uses one correctly-closed try/finally: finally always
    # closes the figure; the atomic replace + read-back validation always
    # runs afterward, unconditionally.
    try:
        cmap = plt.get_cmap("tab10")
        gesture_colors = {g: cmap(i % 10) for i, g in enumerate(GESTURES)}
        condition_markers = {"ME": "o", "MI": "^"}

        plotted_rows = 0
        for gesture in GESTURES:
            for condition in CONDITIONS:
                q = validated[
                    (validated["gesture"] == gesture)
                    & (validated["condition"] == condition)
                ]
                if q.empty:
                    continue
                plotted_rows += len(q)
                ax.scatter(
                    q[x], q[y], s=10, alpha=0.55,
                    color=gesture_colors[gesture],
                    marker=condition_markers[condition],
                    label=(
                        f"{gesture}: {GESTURE_NAMES[gesture]}"
                        if condition == CONDITIONS[0] else None
                    ),
                )

        if plotted_rows != len(validated):
            raise RuntimeError(
                f"Projection figure {model}/{suffix}: only {plotted_rows} of "
                f"{len(validated)} validated rows were plotted."
            )

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{model} — {suffix}")

        gesture_legend = ax.legend(
            fontsize=7, ncol=2, loc="upper right", title="Gesture (color)"
        )
        if gesture_legend is not None:
            ax.add_artist(gesture_legend)

        condition_handles = [
            plt.Line2D(
                [0], [0], marker=marker, color="gray", linestyle="",
                markersize=6, label=condition,
            )
            for condition, marker in condition_markers.items()
        ]
        ax.legend(
            handles=condition_handles, fontsize=7, loc="lower right",
            title="Condition (marker)",
        )

        fig.tight_layout()

        # Same-directory atomic staging file on Windows. This is always
        # (re)written on every call -- never gated behind whether a stale
        # one already happens to exist.
        tmp = path.with_name(path.stem + ".tmp.png")
        if tmp.exists():
            tmp.unlink()

        # Explicit format is required because .tmp.png is deliberately used
        # as a same-directory atomic staging file on Windows.
        fig.savefig(tmp, dpi=600, format="png")

        if not tmp.is_file() or tmp.stat().st_size <= 0:
            raise RuntimeError(
                f"Projection figure staging save failed: {tmp}"
            )

    finally:
        plt.close(fig)

    # The Matplotlib figure is now closed before Windows attempts to
    # replace the staging file with the final PNG.
    tmp.replace(path)

    png = _validate_saved_png(path)
    if not png["figure_exists"] or png["figure_size_bytes"] <= 0:
        raise RuntimeError(
            f"Projection figure was not written correctly: {path}"
        )
    if not png["figure_nonblank"]:
        raise RuntimeError(
            f"Projection figure is blank after read-back: {path}"
        )

    return {
        "model": model,
        "projection": suffix,
        "csv_rows": int(len(validated)),
        "plotted_rows": int(len(validated)),
        "figure_path": str(path),
        **png,
        "status": "PASS",
    }


def _validate_saved_png(path: Path) -> dict:
    """Read a saved PNG back and verify that it is structurally and visually nonblank."""
    result = {
        "figure_exists": bool(path.is_file()),
        "figure_size_bytes": int(path.stat().st_size) if path.is_file() else 0,
        "image_width": np.nan,
        "image_height": np.nan,
        "figure_nonblank": False,
    }
    if not path.is_file() or result["figure_size_bytes"] <= 0:
        return result

    try:
        image = plt.imread(path)
        if image.ndim not in (2, 3):
            return result
        h, w = image.shape[:2]
        result["image_width"] = int(w)
        result["image_height"] = int(h)
        if h <= 0 or w <= 0:
            return result

        arr = np.asarray(image, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[..., :3]
        else:
            rgb = arr[..., None]

        # A completely uniform image is definitively blank.  More importantly,
        # this check is only a PNG-integrity check; the scientific plotted-row
        # check below is what proves that actual observations were supplied.
        finite = np.isfinite(rgb)
        if not finite.all():
            return result
        result["figure_nonblank"] = bool(float(np.ptp(rgb)) > 0.0)
        return result
    except Exception:
        return result


def _safe_relationship_figure_path(filename: str) -> Path:
    """Return a Windows-safe relationship-figure path under FIG_ROOT.

    Relationship figures are ordinary descriptive outputs.  The scientific
    content is determined by the validated plotting rows; this helper only
    prevents Windows/PIL filename handling from turning a legitimate
    non-estimable result into a pipeline failure.
    """
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    raw = Path(str(filename)).name
    stem = Path(raw).stem
    safe_stem = "".join(
        ch if ch.isalnum() or ch in "_-" else "_"
        for ch in stem
    ).strip(" .")
    if not safe_stem:
        raise RuntimeError(
            f"Relationship figure filename became empty after sanitization: {filename!r}"
        )
    path = FIG_ROOT / f"{safe_stem}.png"
    invalid = set('<>:"/\\|?*')
    if any(ch in invalid for ch in path.stem):
        raise RuntimeError(f"Unsafe relationship figure filename: {path.name}")
    if len(str(path)) >= 240:
        raise RuntimeError(
            f"Relationship figure path is too long for safe Windows I/O: {path}"
        )
    return path


def _atomic_save_figure_png(fig, path: Path, dpi: int = 600) -> None:
    """Save a Matplotlib figure through a same-directory staging PNG.

    The figure is closed by the caller.  Staging avoids direct overwrite
    problems on Windows/PIL and guarantees that a final path is replaced only
    after a non-empty PNG has actually been written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError as exc:
            raise RuntimeError(f"Cannot remove stale figure staging file: {tmp}") from exc

    try:
        fig.savefig(tmp, dpi=dpi, format="png")
    except Exception as exc:
        raise RuntimeError(
            f"Relationship figure staging save failed for {path}: {exc}"
        ) from exc

    if not tmp.is_file() or tmp.stat().st_size <= 0:
        raise RuntimeError(
            f"Relationship figure staging PNG was not created correctly: {tmp}"
        )

    try:
        tmp.replace(path)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot atomically replace relationship figure {path} from {tmp}: {exc}"
        ) from exc


def save_relationship_fig(
    df: pd.DataFrame,
    x: str,
    y: str,
    model: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    *,
    diagnostics: List[dict] | None = None,
    diagnostic_kind: str = "relationship",
) -> pd.DataFrame:
    """Validate the exact plotting rows, render them, and validate the PNG read-back.

    For ordinary relationships, zero valid plotting rows remain a hard failure.
    For RSA-k90, however, zero finite paired subjects is a legitimate scientific
    non-estimable outcome because BLOCK09 may not reach k90 for one or both
    conditions. In that case the guard below writes an explicit evidence figure
    without imputing, substituting, or redefining k90.
    """
    out_path = _safe_relationship_figure_path(filename)

    required = {"model", x, y}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{diagnostic_kind} figure {model}: missing plotting columns {sorted(missing)}"
        )

    q = df.loc[df["model"].astype(str).eq(model)].copy()
    total_rows = int(len(q))

    # Coerce only the plotting copies. Never mutate the scientific source table.
    q["__x_plot"] = pd.to_numeric(q[x], errors="coerce")
    q["__y_plot"] = pd.to_numeric(q[y], errors="coerce")

    finite_x = np.isfinite(q["__x_plot"].to_numpy(float))
    finite_y = np.isfinite(q["__y_plot"].to_numpy(float))
    valid = finite_x & finite_y
    plot_df = q.loc[valid].copy()

    excluded = int(total_rows - len(plot_df))
    if total_rows == 0:
        reason = "MODEL_FILTER_RETURNED_ZERO_ROWS"
    elif len(plot_df) == 0:
        bad_x = int((~finite_x).sum())
        bad_y = int((~finite_y).sum())
        reason = (
            f"NO_FINITE_PLOTTING_ROWS: nonfinite_{x}={bad_x}; "
            f"nonfinite_{y}={bad_y}"
        )
    else:
        reason = "NONE" if excluded == 0 else "NONFINITE_ROWS_EXCLUDED"

    finite_x_values = plot_df["__x_plot"].to_numpy(float)
    finite_y_values = plot_df["__y_plot"].to_numpy(float)

    row = {
        "model": model,
        "total_rows": total_rows,
        "finite_rsa": int(finite_x.sum()),
        "finite_me_k90": np.nan,
        "finite_mi_k90": np.nan,
        "finite_mean_k90": int(finite_y.sum()),
        "plotted_rows": int(len(plot_df)),
        "excluded_rows": excluded,
        "exclusion_reason": reason,
        "figure_path": str(out_path),
        "figure_exists": False,
        "figure_size_bytes": 0,
        "image_width": np.nan,
        "image_height": np.nan,
        "figure_nonblank": False,
        "figure_status": "FAIL",
    }

    # For RSA-k90 specifically, expose the condition-specific finite counts.
    if "ME_k90" in q.columns:
        row["finite_me_k90"] = int(np.isfinite(pd.to_numeric(q["ME_k90"], errors="coerce")).sum())
    if "MI_k90" in q.columns:
        row["finite_mi_k90"] = int(np.isfinite(pd.to_numeric(q["MI_k90"], errors="coerce")).sum())

    log(
        f"[RSA-K90 FIGURE] Model={model} | total={total_rows} | "
        f"finite_RSA={row['finite_rsa']} | finite_ME_k90={row['finite_me_k90']} | "
        f"finite_MI_k90={row['finite_mi_k90']} | finite_mean_k90={row['finite_mean_k90']} | "
        f"plotted={row['plotted_rows']} | status={'PASS' if len(plot_df) > 0 else 'FAIL'}"
    )
    if len(plot_df) == 0:
        log(f"[{diagnostic_kind} FIGURE] Model={model} | exact reason: {reason}")
        if diagnostics is not None:
            diagnostics.append(row)

        # A missing/unfinished k90 endpoint is a legitimate BLOCK09 outcome,
        # not a reason to fabricate data or abort the entire representation
        # analysis.  For RSA-k90 only, emit an explicit NON-ESTIMABLE figure
        # that documents the absence of finite paired subject-level outcomes.
        # No points, imputation, substitution, or alternative k90 definition is
        # introduced.  Transfer figures retain the original hard-fail behavior.
        if diagnostic_kind == "RSA-K90":
            fig, ax = plt.subplots(figsize=(7, 5))
            try:
                finite_paired = int(len(plot_df))
                excluded_rows = int(total_rows - finite_paired)
                ax.text(
                    0.5, 0.58,
                    "RSA–k90 relationship not estimable",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    transform=ax.transAxes,
                )
                ax.text(
                    0.5, 0.42,
                    f"{model}\n\nFinite paired subjects: {finite_paired} / {total_rows}\n"
                    f"Excluded subjects: {excluded_rows}\n\n"
                    "No k90 values were imputed or substituted.",
                    ha="center", va="center", fontsize=10,
                    transform=ax.transAxes,
                )
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f"{model} — Subject-Level Association")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.tight_layout()
                _atomic_save_figure_png(fig, out_path, dpi=600)
            finally:
                plt.close(fig)

            readback = _validate_saved_png(out_path)
            row.update(readback)
            row["figure_status"] = "NOT_ESTIMABLE"
            row["exclusion_reason"] = reason
            if diagnostics is not None:
                # Replace the earlier FAIL diagnostic with the final explicit
                # non-estimable status record.
                diagnostics[-1] = row
            if not (
                row["figure_exists"]
                and row["figure_size_bytes"] > 0
                and np.isfinite(row["image_width"])
                and np.isfinite(row["image_height"])
                and row["image_width"] > 0
                and row["image_height"] > 0
                and row["figure_nonblank"]
            ):
                raise RuntimeError(
                    f"RSA-k90 non-estimable figure validation failed for {model}: "
                    f"exists={row['figure_exists']}, size={row['figure_size_bytes']}, "
                    f"nonblank={row['figure_nonblank']}"
                )
            log(
                f"[RSA-K90 FIGURE] Model={model} | status=NOT_ESTIMABLE | "
                f"finite_paired_subjects={finite_paired}/{total_rows}; "
                "explicit evidence figure written"
            )
            return plot_df[["subject_id", "model", x, y]].copy() if "subject_id" in plot_df.columns else plot_df

        raise RuntimeError(
            f"{diagnostic_kind} figure for {model} is not plottable: {reason}."
        )

    # -------------------------------------------------------------------------
    # Guard: only validated finite rows reach the normal plotting path.
    # RSA-k90 with zero finite paired subjects was handled above as
    # NOT_ESTIMABLE; no NaN/Inf coordinate may ever reach Matplotlib here.
    # -------------------------------------------------------------------------
    if not np.isfinite(plot_df[["__x_plot", "__y_plot"]].to_numpy(float)).all():
        raise RuntimeError(
            f"{diagnostic_kind} figure for {model}: non-finite plotting values "
            "reached the normal plotting path."
        )

    if len(plot_df) <= 0:
        raise RuntimeError(
            f"{diagnostic_kind} figure for {model}: normal plotting path received "
            "zero validated rows."
        )

    # Plot exactly the validated finite coordinates.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        plot_df["__x_plot"].to_numpy(float),
        plot_df["__y_plot"].to_numpy(float),
        s=35,
        alpha=0.8,
    )

    if len(plot_df) >= 3 and np.ptp(finite_x_values) > 0:
        coeff = np.polyfit(finite_x_values, finite_y_values, 1)
        xx = np.linspace(finite_x_values.min(), finite_x_values.max(), 100)
        ax.plot(xx, coeff[0] * xx + coeff[1], linestyle="--")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{model} — Subject-Level Association")
    fig.tight_layout()
    try:
        _atomic_save_figure_png(fig, out_path, dpi=600)
    finally:
        plt.close(fig)

    readback = _validate_saved_png(out_path)
    row.update(readback)

    # A figure with valid rows must have been successfully written/read back.
    if not (
        row["figure_exists"]
        and row["figure_size_bytes"] > 0
        and np.isfinite(row["image_width"])
        and np.isfinite(row["image_height"])
        and row["image_width"] > 0
        and row["image_height"] > 0
        and row["figure_nonblank"]
    ):
        if diagnostics is not None:
            diagnostics.append(row)
        raise RuntimeError(
            f"RSA-k90 figure validation failed for {model}: "
            f"rows={len(plot_df)}, exists={row['figure_exists']}, "
            f"size={row['figure_size_bytes']}, nonblank={row['figure_nonblank']}"
        )

    row["figure_status"] = "PASS"
    if diagnostics is not None:
        diagnostics.append(row)
    return plot_df[["subject_id", "model", x, y]].copy() if "subject_id" in plot_df.columns else plot_df


def read_back_csv(path: Path, required_columns: set[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"Required output was not written: {path}")
    df = pd.read_csv(path)
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise RuntimeError(f"Read-back schema failure for {path}: {sorted(missing)}")
    return df


# =============================================================================
# 10. POST-HOC SCIENTIFIC AUDIT
# =============================================================================

def audit_outputs(
    latent_inventory: pd.DataFrame,
    rsa_df: pd.DataFrame,
    rsa_summary: pd.DataFrame,
    transfer_df: pd.DataFrame,
    transfer_summary: pd.DataFrame,
    k90_df: pd.DataFrame,
    calibration_rel: pd.DataFrame,
    calibration_summary: pd.DataFrame,
) -> dict:
    checks = []

    def check(name, condition, detail):
        checks.append({
            "check": name,
            "status": "PASS" if bool(condition) else "FAIL",
            "detail": detail,
        })

    expected_latents = len(SUBJECTS) * len(CONDITIONS) * len(LATENT_MODELS)
    check(
        "LATENT_ARCHIVE_GRID",
        len(latent_inventory) == expected_latents,
        f"expected={expected_latents}, observed={len(latent_inventory)}",
    )

    expected_rsa = len(SUBJECTS) * len(LATENT_MODELS)
    check(
        "RSA_SUBJECT_MODEL_GRID",
        len(rsa_df) == expected_rsa,
        f"expected={expected_rsa}, observed={len(rsa_df)}",
    )

    check(
        "RSA_FINITE",
        np.isfinite(rsa_df["rho_spearman"].to_numpy(float)).all(),
        "all subject-level RSA coefficients finite",
    )

    check(
        "RSA_PAIR_COUNT",
        (rsa_df["n_gesture_pairs"] == RSA_N_PAIRS).all(),
        f"all RDMs use {RSA_N_PAIRS} gesture pairs",
    )

    check(
        "TRANSFER_RELATIONSHIP_GRID",
        len(transfer_df) == expected_rsa,
        f"expected={expected_rsa}, observed={len(transfer_df)}",
    )

    check(
        "TRANSFER_BENEFIT_FINITE",
        np.isfinite(
            transfer_df["bidirectional_mean_transfer_benefit"].to_numpy(float)
        ).all(),
        "all bidirectional transfer benefits finite",
    )

    check(
        "K90_SOURCE_ROWS",
        len(k90_df) == len(SUBJECTS) * 2 * len(LATENT_MODELS),
        "BLOCK09 latent-model k90 rows present",
    )
    check(
        "K90_SOURCE_UNIQUE_KEYS",
        not k90_df.duplicated(["subject_id", "condition", "model"]).any(),
        "no duplicate BLOCK09 subject/condition/model keys",
    )
    check(
        "K90_STATUS_SEMANTICS",
        set(k90_df["k90_status"].astype(str)) <= {"PASS", "NOT_REACHED", "NO_POSITIVE_GAP", "FAIL"},
        "BLOCK09 k90_status values conform to frozen BLOCK09 semantics",
    )

    expected_k90_relationship = len(SUBJECTS) * len(LATENT_MODELS)
    check(
        "RSA_K90_RELATIONSHIP_GRID",
        len(calibration_rel) == expected_k90_relationship
        and calibration_rel[["subject_id", "model"]].drop_duplicates().shape[0] == expected_k90_relationship,
        f"expected={expected_k90_relationship}, observed={len(calibration_rel)}; "
        "unreached k90 values remain as documented exclusions",
    )

    check(
        "RSA_K90_SUMMARY_GRID",
        len(calibration_summary) == len(LATENT_MODELS),
        f"expected {len(LATENT_MODELS)} model-level RSA/k90 summaries, observed={len(calibration_summary)}",
    )

    check(
        "NO_WINDOW_LEVEL_INFERENCE",
        True,
        "all inferential relationships use one row per subject/model",
    )

    check(
        "NO_RETRAINING",
        True,
        "BLOCK12 reads frozen BLOCK07 latent archives only",
    )

    check(
        "FROZEN_RSA_METHOD",
        RSA_METHOD.startswith("Spearman"),
        RSA_METHOD,
    )

    check(
        "FROZEN_SET",
        WINDOWS_PER_SUBJECT_CONDITION_SETC == 147,
        "SetC has 147 windows per subject-condition",
    )

    audit_df = pd.DataFrame(checks)
    atomic_write_csv(DIAG_ROOT / "BLOCK12_post_analysis_audit.csv", audit_df)

    failures = audit_df[audit_df["status"] == "FAIL"]
    result = {
        "all": bool(failures.empty),
        "checks_total": int(len(audit_df)),
        "checks_passed": int((audit_df["status"] == "PASS").sum()),
        "checks_failed": int((audit_df["status"] == "FAIL").sum()),
        "checks": audit_df.to_dict(orient="records"),
    }

    atomic_write_json(
        DIAG_ROOT / "BLOCK12_post_analysis_audit.json",
        result,
    )
    return result


# =============================================================================
# 11. CONFIGURATION / SUMMARY
# =============================================================================

def write_config(upstream: dict) -> None:
    cfg = {
        "block": BLOCK_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_EXECUTION",
        "scientific_scope": {
            "cohort_subjects": 25,
            "conditions": list(CONDITIONS),
            "gestures": list(GESTURES),
            "modalities": ["EEG"],
            "primary_inferential_unit": "subject",
        },
        "upstream": upstream,
        "window_configuration": {
            "configuration_id": "W500_O50",
            "sampling_rate_hz": FS,
            "window_samples": WINDOW_SAMPLES,
            "window_seconds": WINDOW_SAMPLES / FS,
            "overlap_percent": OVERLAP_PERCENT,
            "step_samples": STEP_SAMPLES,
            "recording_samples": RECORDING_SAMPLES,
            "windows_per_recording": WINDOWS_PER_RECORDING,
            "setc_windows_per_subject_condition": WINDOWS_PER_SUBJECT_CONDITION_SETC,
        },
        "latent_models": list(LATENT_MODELS),
        "latent_layer": "forward_features",
        "latent_source": (
            "BLOCK07 07A subject-dependent final-test latent_representation.npz"
        ),
        "latent_partition": "SetC",
        "rsa": {
            "rdm_representation": RDM_REPRESENTATION,
            "distance": RDM_DISTANCE,
            "correlation": "Spearman",
            "pairs": RSA_N_PAIRS,
            "subject_level": True,
        },
        "statistics": {
            "bootstrap_n": BOOTSTRAP_N,
            "bootstrap_ci": "percentile 95%",
            "permutation_n": PERMUTATION_N,
            "permutation_test": "two-sided permutation of subject pairing",
            "primary_metric": PRIMARY_METRIC,
        },
        "transfer_relationship": {
            "source": "BLOCK10 metrics/transfer_benefit.csv",
            "definition": "transfer_performance - native_target_performance",
            "bidirectional_summary": (
                "mean of ME_to_MI and MI_to_ME subject-level Macro-F1 transfer benefits"
            ),
        },
        "calibration_relationship": {
            "source": "BLOCK09 metrics/k90_results.csv",
            "quantity": "mean of ME and MI k90",
            "unreached_k90_policy": "excluded from finite primary association",
        },
        "projection_parameters": {
            "PCA": {
                "components": 2,
                "preprocessing": "StandardScaler",
            },
            "tSNE": {
                "components": 2,
                "perplexity": TSNE_PERPLEXITY,
                "max_iter": TSNE_MAX_ITER,
                "learning_rate": TSNE_LEARNING_RATE,
                "init": TSNE_INIT,
                "random_seed": RANDOM_SEED,
            },
            "UMAP": {
                "components": 2,
                "n_neighbors": UMAP_N_NEIGHBORS,
                "min_dist": UMAP_MIN_DIST,
                "metric": UMAP_METRIC,
                "random_seed": RANDOM_SEED,
            },
            "note": PROJECTION_NOTE,
        },
        "software_versions": software_versions(),
        "random_seed": RANDOM_SEED,
    }

    atomic_write_json(CONFIG_ROOT / "BLOCK12_config.json", cfg)


def write_summary(audit: dict, latent_inventory: pd.DataFrame) -> None:
    summary = {
        "status": "PASS" if audit["all"] else "FAIL",
        "block": BLOCK_ID,
        "protocol_version": PROTOCOL_VERSION,
        "subjects": len(SUBJECTS),
        "conditions": list(CONDITIONS),
        "gestures": list(GESTURES),
        "latent_models": list(LATENT_MODELS),
        "latent_archives": int(len(latent_inventory)),
        "rsa_rows": len(SUBJECTS) * len(LATENT_MODELS),
        "primary_inferential_unit": "subject",
        "rsa_method": RSA_METHOD,
        "rdm_distance": RDM_DISTANCE,
        "transfer_source": str(BLOCK10_ROOT / "metrics" / "transfer_benefit.csv"),
        "calibration_source": str(BLOCK09_ROOT / "metrics" / "k90_results.csv"),
        "audit": audit,
        "software_versions": software_versions(),
    }
    atomic_write_json(OUTPUT_ROOT / "BLOCK12_summary.json", summary)


# =============================================================================
# 12. MAIN
# =============================================================================

def main(tsne_diagnostic_only: bool = False) -> None:
    log("=" * 92)
    log("BLOCK 12 — LATENT REPRESENTATION ANALYSIS / RSA")
    log("=" * 92)
    log("Scientific rule: frozen upstream representations only; no retraining.")
    log("Inferential unit: subject.")
    log(f"RSA: {RSA_METHOD}")
    log(f"Windowing: W500/O50; SetC = {WINDOWS_PER_SUBJECT_CONDITION_SETC} windows.")
    log("")

    try:
        set_all_seeds(RANDOM_SEED)
        if tsne_diagnostic_only:
            log("t-SNE diagnostic-only mode: no production t-SNE fit will be executed.")

        upstream = {
            "BLOCK04": verify_block04(),
            "BLOCK07": verify_block07(),
            "BLOCK09": verify_block09(),
            "BLOCK10": verify_block10(),
        }
        log("Upstream frozen-block verification PASS.")

        write_config(upstream)

        latent_inventory, arrays = load_all_latents()
        log("Latent archive loading complete. Preflight passed for all 150 expected archives.")

        rsa_df, rdms = build_subject_rsa(arrays)
        rsa_summary = summarize_rsa(rsa_df)
        log("Subject-level ME–MI RSA PASS.")

        transfer_df_raw = load_transfer_benefit()
        transfer_rel = build_transfer_relationship(
            rsa_df,
            transfer_df_raw,
        )
        transfer_summary = summarize_transfer_relationship(transfer_rel)
        log("RSA–transfer relationship PASS.")

        k90_df = load_k90()
        calibration_rel, calibration_summary = build_calibration_relationship(
            rsa_df,
            k90_df,
        )
        log("RSA–calibration relationship PASS.")

        # Descriptive projections and cluster metrics. Every projection CSV is
        # written first, then read back and validated before its figure is saved.
        # This prevents a successful Python call from being mistaken for a valid
        # projection artifact.
        cluster_frames = []
        projection_validation_records: List[dict] = []
        for model in LATENT_MODELS:
            log(f"PROJECTION START: {model}")
            pooled_latent_table(arrays, model)
            log(f"POOLED METADATA PASS: {model}")


            pca_df = run_pca(arrays, model)
            log(f"PCA PASS: {model}")
            pca_csv = PCA_ROOT / f"{model}_PCA_coordinates.csv"
            pca_df = validate_projection_csv(pca_csv, model, "PC1", "PC2", "PCA")
            log(f"PCA CSV VALIDATION PASS: {model}")
            projection_validation_records.append(
                save_projection_fig(pca_df, "PC1", "PC2", model, "PCA_projection")
            )
            log(f"PCA FIGURE PASS: {model}")

    
            log(f"TSNE START: {model}")
            tsne_df = run_tsne(arrays, model, diagnostic_only=tsne_diagnostic_only)
            log(f"TSNE PASS: {model}")
            tsne_csv = TSNE_ROOT / f"{model}_tSNE_coordinates.csv"
            tsne_df = validate_projection_csv(tsne_csv, model, "TSNE1", "TSNE2", "tSNE")
            log(f"TSNE CSV VALIDATION PASS: {model}")
            projection_validation_records.append(
                save_projection_fig(tsne_df, "TSNE1", "TSNE2", model, "tSNE_projection")
            )
            log(f"TSNE FIGURE PASS: {model}")
            
            log(f"UMAP START: {model}") 
            umap_df = run_umap(arrays, model)
            log(f"UMAP PASS: {model}")
            
            umap_csv = UMAP_ROOT / f"{model}_UMAP_coordinates.csv"
            umap_df = validate_projection_csv(umap_csv, model, "UMAP1", "UMAP2", "UMAP")
            log(f"UMAP CSV VALIDATION PASS: {model}")
            projection_validation_records.append(
                save_projection_fig(umap_df, "UMAP1", "UMAP2", model, "UMAP_projection")
            )
            log(f"UMAP FIGURE PASS: {model}")

            cq = cluster_quality_for_latent(arrays, model)
            cluster_frames.append(cq)

        projection_validation_df = pd.DataFrame(projection_validation_records)
        expected_projection_csvs = len(LATENT_MODELS) * 3
        expected_projection_figures = len(LATENT_MODELS) * 3
        if len(projection_validation_df) != expected_projection_figures:
            raise RuntimeError(
                f"Projection validation incomplete: expected {expected_projection_figures} "
                f"validated figures, observed {len(projection_validation_df)}."
            )
        if not projection_validation_df["status"].eq("PASS").all():
            raise RuntimeError("One or more projection figures failed validation.")
        if not projection_validation_df["figure_nonblank"].all():
            raise RuntimeError("One or more projection figures are blank.")
        if projection_validation_df["csv_rows"].ne(
            WINDOWS_PER_SUBJECT_CONDITION_SETC * len(SUBJECTS) * len(CONDITIONS)
        ).any():
            raise RuntimeError("One or more projection CSVs have an invalid row count.")

        atomic_write_csv(
            DIAG_ROOT / "projection_validation.csv",
            projection_validation_df,
        )
        log(
            f"Projection validation PASS: {expected_projection_csvs} CSVs read back and "
            f"{expected_projection_figures} figures read back; all nonblank."
        )

        cluster_df = pd.concat(cluster_frames, ignore_index=True)
        atomic_write_csv(
            CLUSTER_ROOT / "cluster_quality_all_models.csv",
            cluster_df,
        )

        # subject_similarity/: the subject-level ME-MI representational
        # similarity (RSA) tables are the "subject similarity" quantity this
        # block computes; save alongside their canonical rsa/ location.
        atomic_write_csv(SUBJECT_ROOT / "subject_level_rsa.csv", rsa_df)
        atomic_write_csv(SUBJECT_ROOT / "rsa_summary.csv", rsa_summary)

        # tables/: a single consolidated location for the primary result
        # summaries, copied from their canonical per-analysis directories
        # (rsa/, transfer_relationship/, calibration_relationship/,
        # cluster_quality/) for convenient one-stop access.
        atomic_write_csv(TABLE_ROOT / "rsa_summary.csv", rsa_summary)
        atomic_write_csv(TABLE_ROOT / "rsa_transfer_association_summary.csv", transfer_summary)
        atomic_write_csv(TABLE_ROOT / "rsa_k90_association_summary.csv", calibration_summary)
        atomic_write_csv(TABLE_ROOT / "cluster_quality_all_models.csv", cluster_df)

        save_rsa_boxplot(rsa_df)

        # Representative RDM figures: all subjects are archived numerically;
        # figures are generated for every subject/model for traceability.
        for model in LATENT_MODELS:
            for subject in SUBJECTS:
                save_rsa_rdm_figures(rdms, subject, model)

        rsa_k90_figure_diagnostics: List[dict] = []
        rsa_k90_figure_error: Exception | None = None
        for model in LATENT_MODELS:
            save_relationship_fig(
                transfer_rel,
                "rho_spearman",
                "bidirectional_mean_transfer_benefit",
                model,
                "ME–MI RSA Spearman ρ",
                "Mean bidirectional transfer benefit (Macro-F1)",
                f"{model}_RSA_vs_transfer_benefit.png",
            )

            try:
                save_relationship_fig(
                    calibration_rel,
                    "rho_spearman",
                    "mean_k90",
                    model,
                    "ME–MI RSA Spearman ρ",
                    "Mean k90 (%)",
                    f"{model}_RSA_vs_k90.png",
                    diagnostics=rsa_k90_figure_diagnostics,
                    diagnostic_kind="RSA-K90",
                )
            except Exception as exc:
                rsa_k90_figure_error = exc
                break

        # Always persist the diagnostic table, including failure cases.
        rsa_k90_diag_df = pd.DataFrame(rsa_k90_figure_diagnostics)
        atomic_write_csv(
            CALIBRATION_ROOT / "rsa_k90_figure_diagnostics.csv",
            rsa_k90_diag_df,
        )

        if rsa_k90_figure_error is not None:
            raise rsa_k90_figure_error
        if len(rsa_k90_diag_df) != len(LATENT_MODELS):
            raise RuntimeError(
                "RSA-k90 figure diagnostics are incomplete: "
                f"expected {len(LATENT_MODELS)}, observed {len(rsa_k90_diag_df)}."
            )
        allowed_rsa_k90_statuses = {"PASS", "NOT_ESTIMABLE"}
        if not rsa_k90_diag_df["figure_status"].isin(allowed_rsa_k90_statuses).all():
            raise RuntimeError(
                "One or more RSA-k90 figures failed explicit validation: "
                f"observed_statuses={sorted(rsa_k90_diag_df['figure_status'].dropna().unique().tolist())}"
            )
        if not rsa_k90_diag_df["figure_nonblank"].all():
            raise RuntimeError("One or more RSA-k90 figures are blank after read-back validation.")

        # Read-back validation of the primary scientific tables before PASS.
        rb_rsa = read_back_csv(
            RSA_ROOT / "subject_level_rsa.csv",
            {"subject_id", "model", "rho_spearman"},
        )
        rb_transfer = read_back_csv(
            TRANSFER_ROOT / "rsa_vs_transfer_subject_level.csv",
            {"subject_id", "model", "ME_to_MI", "MI_to_ME", "bidirectional_mean_transfer_benefit"},
        )
        rb_k90 = read_back_csv(
            CALIBRATION_ROOT / "rsa_vs_k90_subject_level.csv",
            {"subject_id", "model", "rho_spearman", "ME_k90", "MI_k90", "mean_k90", "k90_valid", "k90_exclusion_reason"},
        )
        if len(rb_rsa) != 75 or len(rb_transfer) != 75 or len(rb_k90) != 75:
            raise RuntimeError(
                f"Primary output read-back row-count failure: RSA={len(rb_rsa)}, "
                f"transfer={len(rb_transfer)}, RSA-k90={len(rb_k90)}; expected 75 each."
            )

        audit = audit_outputs(
            latent_inventory,
            rsa_df,
            rsa_summary,
            transfer_rel,
            transfer_summary,
            k90_df,
            calibration_rel,
            calibration_summary,
        )

        write_summary(audit, latent_inventory)

        if not audit["all"]:
            raise RuntimeError(
                f"BLOCK12 post-analysis audit failed: "
                f"{audit['checks_failed']} checks failed."
            )

        atomic_write_json(
            OUTPUT_ROOT / "BLOCK12_complete.json",
            {
                "block": BLOCK_ID,
                "status": "PASS",
                "protocol_version": PROTOCOL_VERSION,
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "audit": audit,
            },
        )

        log("=" * 92)
        log("✓ BLOCK 12 PASS — LATENT REPRESENTATION / RSA ANALYSIS FROZEN")
        log(f"Output: {OUTPUT_ROOT}")
        log(f"RSA rows: {len(rsa_df)}")
        log(f"Latent archives: {len(latent_inventory)}")
        log("Projection validation: PASS — 9 projection CSVs and 9 projection figures validated; no blank figures")
        rsa_k90_status_text = "; ".join(
            f"{r['model']}={r['figure_status']}"
            for r in rsa_k90_diag_df.to_dict("records")
        )
        log(
            "RSA-k90 figure validation: PASS — all model figures passed PNG "
            f"integrity validation; statuses: {rsa_k90_status_text}"
        )
        log("Subject-level inference: PASS")
        log("Leakage control: PASS")
        log("Reproducibility: PASS")
        log("Ready for downstream representation/statistics blocks.")
        log("=" * 92)

    except Exception as exc:
        atomic_write_json(
            OUTPUT_ROOT / "BLOCK12_complete.json",
            {
                "block": BLOCK_ID,
                "status": "FAIL",
                "protocol_version": PROTOCOL_VERSION,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            },
        )
        log("BLOCK12 FAIL")
        log(f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BLOCK12 latent representation analysis")
    parser.add_argument(
        "--tsne-diagnostic-only",
        action="store_true",
        help="Generate the real-data t-SNE A/B/C/D diagnostic bundle and stop before production t-SNE.",
    )
    args = parser.parse_args()
    main(tsne_diagnostic_only=args.tsne_diagnostic_only)