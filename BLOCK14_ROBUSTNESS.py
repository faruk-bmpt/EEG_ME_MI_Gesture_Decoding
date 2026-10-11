#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BLOCK 14 -- ROBUSTNESS & STRESS TESTING  (canonical executable: BLOCK14_ROBUSTNESS.py)

Post-hoc stress test of FROZEN BLOCK06/BLOCK07 LOSO models.
No retraining. No fine-tuning. No tuning. No test-time adaptation.

FINAL-FIX (2026-10-03): the authoritative clean reference is the exact frozen
BLOCK06 / BLOCK07 LOSO prediction CSV for each (subject, condition, model).
Summary CSVs are diagnostics only and never control PASS/FAIL.

The three window_id notations observed on disk are:
    BLOCK04 : S01_ME_1_SetA_W500_O50_Window0001   (1-based, 4-digit)
    BLOCK06 : S01_ME_1_SetA_W001                  (1-based, 3-digit)
    BLOCK07 : S01_ME_1_SetA_windows_W00           (0-based, 2-digit)
The prefix is identical. Only the suffix differs. A single canonicalization
function maps all three into the BLOCK04 form; anything unmapped raises.
"""

from __future__ import annotations

import argparse
import ast
import faulthandler
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd

try:
    import joblib
    JOBLIB_ERROR = None
except Exception as exc:
    joblib = None
    JOBLIB_ERROR = repr(exc)

try:
    import xgboost as xgb  # noqa: F401
    XGBOOST_ERROR = None
except Exception as exc:
    xgb = None
    XGBOOST_ERROR = repr(exc)

try:
    import torch
    TORCH_ERROR = None
except Exception as exc:
    torch = None
    TORCH_ERROR = repr(exc)

import scipy
import sklearn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_ERROR = None
except Exception as exc:
    plt = None
    MPL_ERROR = repr(exc)


# ============================================================================
# 00. FROZEN STUDY CONSTANTS
# ============================================================================

BLOCK_ID = "14_ROBUSTNESS"
PROTOCOL_VERSION = "BLOCK14_ROBUSTNESS_FROZEN_POSTHOC_V3_REGENERATED"
UNIT_SCHEMA = "BLOCK14_UNIT_V2"

RANDOM_SEED = 42
FS = 125
N_CHANNELS = 13
N_FEATURES = 341
N_SUBJECTS = 25
N_GESTURES = 7
WINDOW_SAMPLES = 500
WINDOWS_PER_RECORDING = 21
WINDOWS_PER_BLOCK = N_GESTURES * 3 * WINDOWS_PER_RECORDING
CHANCE_LEVEL = 1.0 / 7.0

GESTURES = tuple(range(1, 8))
SUBJECTS = tuple(range(1, 26))
CONDITIONS = ("ME", "MI")
SETS = ("SetA", "SetB", "SetC")
CHANNEL_NAMES = tuple(f"CH{i:02d}" for i in range(1, N_CHANNELS + 1))
REGIMES = ("LOSO",)

ML_MODELS = ("LDA", "KNN", "SVM", "RF", "XGB")
DL_MODELS = ("EEGNet", "ShallowConvNet", "DeepConvNet", "CNN-BiLSTM", "CNN-BiLSTM-Attention")
ALL_MODELS = ML_MODELS + DL_MODELS

DISPLAY_NAME = {
    "LDA": "LDA", "KNN": "KNN", "SVM": "SVM", "RF": "Random Forest", "XGB": "XGBoost",
    "EEGNet": "EEGNet", "ShallowConvNet": "ShallowConvNet", "DeepConvNet": "DeepConvNet",
    "CNN-BiLSTM": "CNN-BiLSTM", "CNN-BiLSTM-Attention": "CNN-BiLSTM-Attention",
}
MODEL_COLOR = dict(zip(ALL_MODELS, [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]))
MODEL_MARKER = dict(zip(ALL_MODELS, ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]))

CHANNEL_DROPOUT_LEVELS = (0, 5, 10, 20, 30)
SET_CODE = {"SetA": 1, "SetB": 2, "SetC": 3}
COND_CODE = {"ME": 0, "MI": 1}

PRIMARY_METRIC = "Macro-F1"
METRICS = ("Accuracy", "Balanced Accuracy", "Macro-F1", "MCC", "Cohen_kappa")
PUB_SCALE = 100.0

NOISE_SEED_DERIVATION = (
    "numpy.random.SeedSequence([noise_seed, subject, condition_code(ME=0,MI=1), gesture, "
    "set_code(A=1,B=2,C=3), window_index, noise_level_index, realization]) -> "
    "numpy.random.default_rng(...).standard_normal((13,500)); identical for every model "
    "and independent of execution order"
)
FDR_FAMILY_DEFINITION = "family = comparison | perturbation_type | metric ; BH-FDR over all models x conditions x levels inside the family"


# ============================================================================
# 01. PROJECT PATHS
# ============================================================================

WINDOWS_ROOT = Path(os.environ.get(
    "PAPER2_PROJECT_ROOT",
    r"F:\Faruk\EEG_Motor_Execution_Imagery\Paper2_EEG_BCI",
))
PROJECT_ROOT_OVERRIDDEN = "PAPER2_PROJECT_ROOT" in os.environ

BLOCK03_ROOT = WINDOWS_ROOT / "03_PREPROCESSING"
BLOCK04_ROOT = WINDOWS_ROOT / "04_WINDOWING"
BLOCK05_ROOT = WINDOWS_ROOT / "05_FEATURE_ENGINEERING"
BLOCK06_ROOT = WINDOWS_ROOT / "06_ML"
BLOCK07_ROOT = WINDOWS_ROOT / "07_DL"
BLOCK13_ROOT = WINDOWS_ROOT / "13_XAI"
BLOCK13_FROZEN_DEFAULT = WINDOWS_ROOT / "13_XAI_FROZEN_OUTPUTS"

BLOCK14_ROOT = WINDOWS_ROOT / "14_ROBUSTNESS"
CONFIG_ROOT = BLOCK14_ROOT / "config"
MANIFEST_ROOT = BLOCK14_ROOT / "manifests"
CLEAN_ROOT = BLOCK14_ROOT / "clean_reference"
PRED_ROOT = BLOCK14_ROOT / "predictions"
DROPOUT_ROOT = BLOCK14_ROOT / "channel_dropout"
NOISE_ROOT = BLOCK14_ROOT / "noise"
METRIC_ROOT = BLOCK14_ROOT / "metrics"
STATS_ROOT = BLOCK14_ROOT / "statistics"
TABLE_ROOT = BLOCK14_ROOT / "tables"
TABLE_PUB = TABLE_ROOT / "publication"
TABLE_AUDIT = TABLE_ROOT / "audit"
FIG_ROOT = BLOCK14_ROOT / "figures"
FIG_PUB = FIG_ROOT / "publication"
FIG_DIAG = FIG_ROOT / "diagnostic"
LOG_ROOT = BLOCK14_ROOT / "logs"
AUDIT_ROOT = BLOCK14_ROOT / "audit"
ARCHIVE_ROOT = BLOCK14_ROOT / "_archive"
DRYRUN_ROOT = BLOCK14_ROOT / "_dry_run"

CONFIG_PATH = CONFIG_ROOT / "BLOCK14_config.json"
CONFIG_LOCK_PATH = CONFIG_ROOT / "BLOCK14_config.frozen.sha256"
HASH_CACHE_PATH = MANIFEST_ROOT / "BLOCK14_hash_cache.json"
PREFLIGHT_JSON = AUDIT_ROOT / "BLOCK14_preflight.json"
PREFLIGHT_MD = AUDIT_ROOT / "BLOCK14_PREFLIGHT_REPORT.md"
PROGRESS_PATH = AUDIT_ROOT / "BLOCK14_progress.json"
FAILURES_PATH = AUDIT_ROOT / "BLOCK14_unit_failures.jsonl"
SUMMARY_PATH = BLOCK14_ROOT / "BLOCK14_summary.json"
AUDIT_REPORT_PATH = BLOCK14_ROOT / "BLOCK14_audit_report.json"
REPRO_PATH = BLOCK14_ROOT / "BLOCK14_reproducibility.json"
COMPLETE_PATH = BLOCK14_ROOT / "BLOCK14_complete.json"
INCOMPLETE_PATH = BLOCK14_ROOT / "BLOCK14_INCOMPLETE.json"

CANONICAL_DIRS = (
    BLOCK14_ROOT, CONFIG_ROOT, MANIFEST_ROOT, CLEAN_ROOT, PRED_ROOT / "ML", PRED_ROOT / "DL",
    DROPOUT_ROOT, NOISE_ROOT, METRIC_ROOT, STATS_ROOT, TABLE_PUB, TABLE_AUDIT,
    FIG_PUB, FIG_DIAG, LOG_ROOT, AUDIT_ROOT, ARCHIVE_ROOT,
)


# ============================================================================
# 02. LOGGING / MEMORY / HASHING / ATOMIC I/O
# ============================================================================

T0 = time.perf_counter()
_NATIVE_CRASH_FH = None


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"{time.perf_counter() - T0:10.2f}s | {message}"
    print(line, flush=True)
    try:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        with (LOG_ROOT / "BLOCK14_execution.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def enable_native_crash_log() -> None:
    global _NATIVE_CRASH_FH
    try:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        if _NATIVE_CRASH_FH is None:
            _NATIVE_CRASH_FH = open(LOG_ROOT / "BLOCK14_native_crash.log", "a", encoding="utf-8")
        faulthandler.enable(_NATIVE_CRASH_FH, all_threads=True)
    except Exception as exc:
        print(f"faulthandler not enabled: {exc!r}", flush=True)


def get_process_memory_mb() -> float:
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
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return float(pmc.WorkingSetSize) / 1048576.0
        except Exception:
            pass
    return -1.0


def log_memory(tag: str) -> None:
    gc.collect()
    mb = get_process_memory_mb()
    log(f"MEM | {tag}: process_memory={mb:,.0f} MB" if mb >= 0 else f"MEM | {tag}: unavailable")


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=2, sort_keys=True, allow_nan=False)
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        with tmp.open("wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def assert_finite(name: str, arr: Any) -> None:
    a = np.asarray(arr)
    if a.dtype.kind in "fc" and not np.isfinite(a).all():
        raise RuntimeError(f"{name} contains NaN/Inf.")


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def code_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def safe_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(value))


def fmt_level(value: Any) -> str:
    v = float(value)
    return f"{v:g}"


_HASH_CACHE: dict[str, dict] = {}
_HASH_CACHE_LOADED = False


def _load_hash_cache() -> None:
    global _HASH_CACHE_LOADED, _HASH_CACHE
    if _HASH_CACHE_LOADED:
        return
    _HASH_CACHE_LOADED = True
    try:
        if HASH_CACHE_PATH.exists():
            _HASH_CACHE = json.loads(HASH_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        _HASH_CACHE = {}


def cached_sha256(path: Path) -> str:
    _load_hash_cache()
    p = Path(path)
    st = p.stat()
    key = str(p)
    ent = _HASH_CACHE.get(key)
    if ent and ent.get("size") == st.st_size and ent.get("mtime_ns") == st.st_mtime_ns:
        return ent["sha256"]
    digest = sha256_file(p)
    _HASH_CACHE[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
    return digest


def save_hash_cache() -> None:
    _load_hash_cache()
    try:
        atomic_json(HASH_CACHE_PATH, _HASH_CACHE)
    except Exception as exc:
        log(f"WARN hash cache not saved: {exc!r}")


def prepare_dirs() -> None:
    for p in CANONICAL_DIRS:
        p.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 03. CONFIGURATION
# ============================================================================

def default_config() -> dict:
    return {
        "block_id": BLOCK_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN",
        "frozen_note": "",
        "random_seed": RANDOM_SEED,
        "evaluation_regimes": ["LOSO"],

        "channel_dropout": {
            "levels_percent": list(CHANNEL_DROPOUT_LEVELS),
            "channel_order": list(CHANNEL_NAMES),
            "channel_order_note": "Positional: CHnn = row nn-1 of every BLOCK04 (13,500) window.",
            "percentage_to_count": {"0": 0, "5": 1, "10": 1, "20": 3, "30": 4},
            "channel_map": {
                "0": [],
                "5": ["CH07"],
                "10": ["CH07"],
                "20": ["CH01", "CH07", "CH13"],
                "30": ["CH01", "CH05", "CH09", "CH13"],
            },
            "selection_rule": (
                "nearest-integer channel count round(13*p/100); select k positional channels at "
                "approximately equal spacing across CH01..CH13 using linspace(0,12,k) rounded "
                "to nearest integer; fixed before execution and independent of performance"
            ),
            "pattern_policy": "one_fixed_pattern",
            "same_pattern_across_subjects_models_conditions": True,
            "representation_of_missing_channel": "zero",
        },

        "noise": {
            "enabled": True,
            "noise_type": "additive_gaussian",
            "noise_parameterization": {
                "level_semantics": "snr_db",
                "snr_reference": "per_channel_window_rms",
                "distribution": "zero_mean_unit_variance_standard_normal_scaled_to_target_SNR",
                "application": "additive",
            },
            "noise_levels": [20.0, 10.0, 5.0, 0.0],
            "noise_levels_order": "increasing_severity",
            "noise_application_stage": "windowed_raw_eeg",
            "accept_window_level_application": True,
            "noise_seed": 42,
            "n_noise_realizations": 3,
            "noise_scope": "all_channels",
            "noise_reproducibility_rule": (
                "SeedSequence([42,subject,condition_code,gesture,set_code,window_index,level_index,realization]); "
                "standard-normal noise with 13x500 shape; independently scale each channel to the "
                "requested window RMS SNR; never adapt noise to predictions or results"
            ),
        },

        "statistics": {
            "bootstrap_resamples": 2000,
            "bootstrap_seed": 42,
            "permutation_resamples": 10000,
            "permutation_seed": 42,
            "fdr_q": 0.05,
            "primary_metric": PRIMARY_METRIC,
        },

        "runtime": {
            "torch_threads": 1,
            "dl_batch_size": 64,
            "continue_on_unit_failure": False,
            "export_long_prediction_csv": True,
            "memory_log_every_n_units": 25,
        },

        "upstream": {
            "block13_script": str(WINDOWS_ROOT / "scripts" / "BLOCK13_XAI.py"),
            "block13_frozen_outputs": str(BLOCK13_FROZEN_DEFAULT),
            "block05_feature_extractor_path": str(WINDOWS_ROOT / "scripts" / "BLOCK05_FEATURE_ENGINEERING.py"),
            "block05_feature_extractor_import": "",
            "block05_feature_extractor_function": "compute_feature_vector",
            "block05_feature_extractor_input_layout": "channels_x_samples",
            "block05_feature_extractor_kwargs": {},
            "feature_reproduction_rtol": 1e-5,
            "feature_reproduction_atol": 1e-8,
            "clean_prediction_agreement_min": 1.0,
            "upstream_clean_reference": {
                "enabled": True,
                "waiver_justification": "",
                "csv": str(BLOCK06_ROOT / "06B_ML_LOSO" / "subject_results" / "LOSO_subject_results.csv"),
                "subject_column": "subject",
                "condition_column": "condition",
                "model_column": "model",
                "regime_column": "evaluation_regime",
                "regime_value": "LOSO",
                "metric_columns": {"Macro-F1": "macro_f1"},
                "model_name_map": {"XGBoost": "XGB", "RandomForest": "RF"},
                "tolerance_abs": 1e-6,
            },
        },
    }


def init_config() -> None:
    prepare_dirs()
    if CONFIG_PATH.exists():
        raise FileExistsError(f"{CONFIG_PATH} already exists. Refusing to overwrite a configuration.")
    atomic_json(CONFIG_PATH, default_config())
    log(f"Created BLOCK14 frozen operational configuration: {CONFIG_PATH}")


def normalize_channel_map(cfg: dict) -> dict[int, list[list[str]]]:
    out: dict[int, list[list[str]]] = {}
    cmap = cfg.get("channel_dropout", {}).get("channel_map", {})
    for lvl in CHANNEL_DROPOUT_LEVELS:
        v = cmap.get(str(lvl), [])
        if len(v) == 0:
            out[lvl] = [[]]
        elif isinstance(v[0], (list, tuple)):
            out[lvl] = [list(map(str, p)) for p in v]
        else:
            out[lvl] = [list(map(str, v))]
    return out


def dropout_pattern_indices(cfg: dict, level: int, pattern: int) -> list[int]:
    order = cfg["channel_dropout"]["channel_order"]
    idx = {n: i for i, n in enumerate(order)}
    return [idx[n] for n in normalize_channel_map(cfg)[int(level)][pattern]]


def config_problems(cfg: dict) -> list[str]:
    P: list[str] = []
    if cfg.get("block_id") != BLOCK_ID:
        P.append("block_id mismatch")
    if str(cfg.get("status", "")).upper() != "FROZEN":
        P.append("status is not FROZEN")
    if cfg.get("evaluation_regimes") != ["LOSO"]:
        P.append("evaluation_regimes must be exactly ['LOSO']")

    cd = cfg.get("channel_dropout", {})
    if cd.get("levels_percent") != list(CHANNEL_DROPOUT_LEVELS):
        P.append("channel_dropout.levels_percent must be exactly [0,5,10,20,30]")
    if cd.get("channel_order") != list(CHANNEL_NAMES):
        P.append("channel_dropout.channel_order must equal CH01..CH13")
    p2c = cd.get("percentage_to_count", {})
    counts: dict[int, int] = {}
    for lvl in CHANNEL_DROPOUT_LEVELS:
        k = str(lvl)
        v = p2c.get(k)
        if v is None or isinstance(v, bool) or int(v) != v:
            P.append(f"channel_dropout.percentage_to_count[{k!r}] is not frozen")
            continue
        counts[lvl] = int(v)
    prev = 0
    for lvl in CHANNEL_DROPOUT_LEVELS[1:]:
        if lvl in counts:
            if not (1 <= counts[lvl] <= N_CHANNELS - 1):
                P.append(f"{lvl}% dropout must drop between 1 and {N_CHANNELS - 1} channels")
            if counts[lvl] < prev:
                P.append("channel counts must be non-decreasing with level")
            prev = counts[lvl]
    policy = cd.get("pattern_policy", "")
    if policy not in ("one_fixed_pattern", "multiple_predefined_patterns"):
        P.append("channel_dropout.pattern_policy invalid")
    expected_map = {0: [], 5: ["CH07"], 10: ["CH07"], 20: ["CH01", "CH07", "CH13"],
                    30: ["CH01", "CH05", "CH09", "CH13"]}
    for lvl, expected in expected_map.items():
        actual = normalize_channel_map(cfg)[lvl]
        if actual != ([[]] if lvl == 0 else [expected]):
            P.append(f"channel_dropout.channel_map[{lvl}] does not match the frozen map")
    if cd.get("same_pattern_across_subjects_models_conditions") is not True:
        P.append("same_pattern_across_subjects_models_conditions must be true")
    if str(cd.get("representation_of_missing_channel", "")).strip().lower() != "zero":
        P.append("representation_of_missing_channel must be 'zero'")

    nz = cfg.get("noise", {})
    if not nz.get("enabled", True):
        P.append("noise.enabled=false is not allowed")
    if str(nz.get("noise_type", "")).strip().lower() != "additive_gaussian":
        P.append("noise.noise_type must be 'additive_gaussian'")
    params = nz.get("noise_parameterization", {})
    sem = params.get("level_semantics") if isinstance(params, dict) else None
    if sem not in ("snr_db", "std_absolute"):
        P.append("noise.noise_parameterization.level_semantics invalid")
    if sem == "snr_db" and params.get("snr_reference") not in ("per_channel_window_rms", "per_window_global_rms"):
        P.append("noise_parameterization.snr_reference invalid")
    levels = nz.get("noise_levels", [])
    if not isinstance(levels, list) or len(levels) < 1:
        P.append("noise.noise_levels is empty")
    if nz.get("noise_application_stage") != "windowed_raw_eeg":
        P.append("noise.noise_application_stage must be 'windowed_raw_eeg'")
    if nz.get("accept_window_level_application") is not True:
        P.append("noise.accept_window_level_application must be true")
    seed = nz.get("noise_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        P.append("noise.noise_seed must be a non-negative integer")
    nr = nz.get("n_noise_realizations")
    if not isinstance(nr, int) or isinstance(nr, bool) or nr < 1:
        P.append("noise.n_noise_realizations must be an integer >= 1")
    if nz.get("noise_scope") != "all_channels":
        P.append("noise.noise_scope must be 'all_channels'")
    if not str(nz.get("noise_reproducibility_rule", "")).strip():
        P.append("noise.noise_reproducibility_rule is empty")
    if str(nz.get("noise_type", "")).strip().lower() == "additive_gaussian" and sem == "snr_db":
        if [float(v) for v in levels] != [20.0, 10.0, 5.0, 0.0]:
            P.append("noise.noise_levels must be [20,10,5,0] dB")
        if nz.get("noise_seed") != 42:
            P.append("noise.noise_seed must be 42")
        if nz.get("n_noise_realizations") != 3:
            P.append("noise.n_noise_realizations must be 3")

    st = cfg.get("statistics", {})
    for k in ("bootstrap_resamples", "permutation_resamples"):
        if not isinstance(st.get(k), int) or st.get(k, 0) < 100:
            P.append(f"statistics.{k} must be an integer >= 100")
    for k in ("bootstrap_seed", "permutation_seed"):
        if not isinstance(st.get(k), int):
            P.append(f"statistics.{k} must be an integer")
    if not (0.0 < float(st.get("fdr_q", 0)) < 1.0):
        P.append("statistics.fdr_q must be in (0,1)")
    if st.get("primary_metric") != PRIMARY_METRIC:
        P.append(f"statistics.primary_metric must be {PRIMARY_METRIC!r}")

    rt = cfg.get("runtime", {})
    if rt.get("torch_threads") != 1:
        P.append("runtime.torch_threads must be 1")
    if not isinstance(rt.get("dl_batch_size"), int) or rt.get("dl_batch_size", 0) < 1:
        P.append("runtime.dl_batch_size must be a positive integer")

    up = cfg.get("upstream", {})
    fn = str(up.get("block05_feature_extractor_function", "")).strip()
    if not fn:
        P.append("upstream.block05_feature_extractor_function is not set")
    if not (str(up.get("block05_feature_extractor_path", "")).strip()
            or str(up.get("block05_feature_extractor_import", "")).strip()):
        P.append("upstream.block05_feature_extractor_path not set")
    if up.get("block05_feature_extractor_input_layout") not in ("channels_x_samples", "samples_x_channels"):
        P.append("upstream.block05_feature_extractor_input_layout invalid")
    ucr = up.get("upstream_clean_reference", {})
    if ucr.get("enabled"):
        if not str(ucr.get("csv", "")).strip():
            P.append("upstream_clean_reference.enabled but csv is empty")
        for k in ("subject_column", "condition_column", "model_column"):
            if not str(ucr.get(k, "")).strip():
                P.append(f"upstream_clean_reference.{k} is empty")
        mc = ucr.get("metric_columns", {})
        if PRIMARY_METRIC not in mc:
            P.append(f"upstream_clean_reference.metric_columns must map at least {PRIMARY_METRIC!r}")
    else:
        if not str(ucr.get("waiver_justification", "")).strip():
            P.append("clean metrics must be cross-checked against frozen upstream results")
    return P


def canonical_config_sha(cfg: dict) -> str:
    return sha256_text(json.dumps(jsonable(cfg), sort_keys=True, separators=(",", ":")))


def read_config_raw() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"{CONFIG_PATH} not found. Run --init-config first.")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def canonical_units_exist() -> bool:
    for root in (CLEAN_ROOT, DROPOUT_ROOT, NOISE_ROOT):
        if root.exists() and any(root.rglob("*.npz")):
            return True
    return False


def write_config_lock(cfg: dict, allow_refreeze: bool = False) -> str:
    sha = canonical_config_sha(cfg)
    if CONFIG_LOCK_PATH.exists():
        old = CONFIG_LOCK_PATH.read_text(encoding="utf-8").split()[0]
        if old == sha:
            return sha
        if not allow_refreeze:
            raise RuntimeError(
                "Config hash differs from the frozen lock. If NO results exist yet you may "
                "re-freeze deliberately with --refreeze-config.")
        if canonical_units_exist():
            raise RuntimeError("Refusing to re-freeze: BLOCK14 result units already exist.")
        ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CONFIG_LOCK_PATH, ARCHIVE_ROOT / f"BLOCK14_config.frozen.sha256.{int(time.time())}.bak")
    atomic_text(CONFIG_LOCK_PATH, f"{sha}  {now_utc()}\n")
    return sha


def load_config(require_lock: bool = True) -> dict:
    cfg = read_config_raw()
    problems = config_problems(cfg)
    if problems:
        raise RuntimeError("BLOCK14 configuration is not execution-ready:\n  - " + "\n  - ".join(problems))
    sha = canonical_config_sha(cfg)
    if require_lock:
        if not CONFIG_LOCK_PATH.exists():
            raise RuntimeError("No config hash lock. Run --preflight.")
        old = CONFIG_LOCK_PATH.read_text(encoding="utf-8").split()[0]
        if old != sha:
            raise RuntimeError("Config hash != frozen lock.")
    return cfg


# ============================================================================
# 04. BLOCK13 COMPATIBILITY ADAPTER (read-only)
# ============================================================================

_B13 = None
_B13_SOURCE: Path | None = None


def _has_main_guard(source: str, filename: str) -> bool:
    tree = ast.parse(source, filename=filename)
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            if any(isinstance(c, ast.Constant) and c.value == "__main__" for c in node.test.comparators):
                return True
    return False


def locate_block13(cfg: dict | None = None) -> Path:
    cands: list[Path] = []
    if cfg:
        configured = str(cfg.get("upstream", {}).get("block13_script", "")).strip()
        if configured:
            cands.append(Path(configured))
    cands += [
        WINDOWS_ROOT / "scripts" / "BLOCK13_XAI.py",
        WINDOWS_ROOT / "BLOCK13_XAI.py",
        *sorted(WINDOWS_ROOT.glob("scripts/BLOCK13_XAI(*).py")),
        *sorted(WINDOWS_ROOT.glob("BLOCK13_XAI(*).py")),
        Path(__file__).resolve().parent / "BLOCK13_XAI.py",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError("BLOCK13_XAI.py adapter not found in: " + " | ".join(map(str, cands)))


def import_block13(cfg: dict | None = None):
    global _B13, _B13_SOURCE
    if _B13 is not None:
        return _B13
    source = locate_block13(cfg)
    if not _has_main_guard(source.read_text(encoding="utf-8", errors="replace"), str(source)):
        raise RuntimeError(f"{source} has no __main__ guard; refusing to import it.")
    spec = importlib.util.spec_from_file_location("paper2_block13_frozen_adapter", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import BLOCK13 adapter: {source}")
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)

    required = (
        "verify_block04", "resolve_window_array", "load_raw_window_from_row", "build_feature_lookup",
        "materialize_feature_rows", "load_block07_loso_model", "_discover_exact_block07_checkpoint",
        "_assert_dl_runtime_integrity", "XGBGestureClassifier", "PROTOCOL_VERSION",
    )
    missing = [n for n in required if not hasattr(module, n)]
    if missing:
        raise RuntimeError(f"BLOCK13 adapter lacks required API: {missing}")

    module.log = lambda message: log(f"[BLOCK13-adapter] {message}")
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "XGBGestureClassifier"):
        setattr(main_mod, "XGBGestureClassifier", module.XGBGestureClassifier)

    _B13, _B13_SOURCE = module, source
    log(f"BLOCK13 compatibility adapter: {source}")
    enable_native_crash_log()
    return module


# ============================================================================
# 05. BLOCK05 FEATURE-EXTRACTOR HANDOFF
# ============================================================================

@dataclass
class Extractor:
    fn: Callable
    source: str
    layout: str
    kwargs: dict

    def one(self, window: np.ndarray) -> np.ndarray:
        x = np.asarray(window, dtype=np.float64)
        if self.layout == "samples_x_channels":
            x = x.T
        y = self.fn(x, **self.kwargs)
        if isinstance(y, dict):
            for key in ("features", "feature_vector", "X"):
                if key in y:
                    y = y[key]
                    break
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if y.shape != (N_FEATURES,):
            raise RuntimeError(f"BLOCK05 extractor returned {y.shape}; expected ({N_FEATURES},)")
        return y


class NonFiniteFeatureError(RuntimeError):
    pass


def load_extractor(cfg: dict) -> Extractor:
    up = cfg["upstream"]
    fname = str(up["block05_feature_extractor_function"]).strip()
    path_s = str(up.get("block05_feature_extractor_path", "")).strip()

    if not path_s:
        raise RuntimeError("BLOCK05 extractor source path is empty.")
    src = Path(path_s)
    if not src.exists():
        raise FileNotFoundError(f"BLOCK05 extractor source not found: {src}")
    if fname != "compute_feature_vector":
        raise RuntimeError(f"BLOCK14 permits only 'compute_feature_vector'; got {fname!r}")

    source_text = src.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source_text, filename=str(src))

    required_functions = {
        "safe_variance", "safe_skew", "safe_kurtosis_excess", "safe_hjorth",
        "safe_zero_crossing_rate", "compute_psd", "band_power", "spectral_entropy",
        "spectral_edge_frequency", "channel_features", "compute_feature_vector",
    }

    function_nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in required_functions
    }

    missing = sorted(required_functions - set(function_nodes))
    if missing:
        raise RuntimeError(f"{src}: canonical BLOCK05 feature-function definitions missing: {missing}")

    namespace = {
        "__name__": "paper2_block05_isolated_feature_namespace",
        "np": np, "math": math, "signal": scipy.signal, "stats": scipy.stats,
        "FS": FS, "WINDOW_SAMPLES": WINDOW_SAMPLES, "N_CHANNELS": N_CHANNELS, "N_FEATURES": N_FEATURES,
        "PSD_NPERSEG": 256, "PSD_NOVERLAP": 128, "PSD_NFFT": 256,
        "PSD_WINDOW": "hann", "PSD_DETREND": "constant", "PSD_SCALING": "density",
        "PSD_AVERAGING": "mean", "PSD_ONE_SIDED": True,
        "EPSILON": 1e-12, "SEF_PERCENT": 95.0, "CORR_CONSTANT_VALUE": 0.0,
    }

    ordered = [
        "safe_variance", "safe_skew", "safe_kurtosis_excess", "safe_hjorth",
        "safe_zero_crossing_rate", "compute_psd", "band_power", "spectral_entropy",
        "spectral_edge_frequency", "channel_features", "compute_feature_vector",
    ]
    module_ast = ast.Module(body=[function_nodes[name] for name in ordered], type_ignores=[])
    ast.fix_missing_locations(module_ast)
    code = compile(module_ast, str(src), "exec")
    exec(code, namespace, namespace)

    fn = namespace[fname]
    if not callable(fn):
        raise RuntimeError(f"{src}: isolated {fname!r} is not callable")

    log(f"BLOCK05 isolated extractor: {src.name}:{fname} | source_sha256={sha256_file(src)} | top_level_execution=BLOCKED")

    return Extractor(
        fn=fn,
        source=f"{src}:{fname}:AST_ISOLATED",
        layout=up["block05_feature_extractor_input_layout"],
        kwargs=dict(up.get("block05_feature_extractor_kwargs", {})),
    )


def extract_features_batch(ex: Extractor, X: np.ndarray, tag: str) -> np.ndarray:
    n = X.shape[0]
    out = np.empty((n, N_FEATURES), dtype=np.float64)
    for i in range(n):
        out[i] = ex.one(X[i])
    bad = ~np.isfinite(out)
    if bad.any():
        cols = np.where(bad.any(axis=0))[0]
        names = []
        try:
            names = [_B13.FEATURE_NAMES[j] for j in cols[:8]]
        except Exception:
            names = [int(j) for j in cols[:8]]
        raise NonFiniteFeatureError(
            f"{tag}: BLOCK05 extractor produced {int(bad.sum())} non-finite values in "
            f"{int(bad.any(axis=1).sum())}/{n} windows; features={names}{'...' if len(cols) > 8 else ''}."
        )
    return out


# ============================================================================
# 06. BLOCK04 INPUT
# ============================================================================

def load_block04_manifest(b13) -> tuple[pd.DataFrame, dict]:
    df, info = b13.verify_block04()
    need = {"window_id", "subject", "condition", "gesture_code", "set", "source_recording",
            "window_array_file", "window_index"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"BLOCK04 manifest missing columns {sorted(miss)}")
    exp = len(SUBJECTS) * len(CONDITIONS) * N_GESTURES * len(SETS) * WINDOWS_PER_RECORDING
    if len(df) != exp:
        raise RuntimeError(f"BLOCK04 manifest has {len(df)} rows; expected {exp}")
    if set(df.subject) != set(SUBJECTS) or set(df.condition) != set(CONDITIONS) \
            or set(df.gesture_code) != set(GESTURES) or set(df["set"]) != set(SETS):
        raise RuntimeError("BLOCK04 coverage differs from the frozen design")
    if df.duplicated(["subject", "condition", "gesture_code", "set", "window_index"]).any():
        raise RuntimeError("BLOCK04 manifest has duplicate window keys")
    df = df.sort_values(["subject", "condition", "gesture_code", "set", "window_index"]).reset_index(drop=True)
    return df, info


def block_meta(manifest: pd.DataFrame, subject: int, condition: str) -> pd.DataFrame:
    m = manifest[(manifest.subject == subject) & (manifest.condition == condition)]
    m = m.sort_values(["gesture_code", "set", "window_index"]).reset_index(drop=True)
    expect = N_GESTURES * len(SETS) * WINDOWS_PER_RECORDING
    if len(m) != expect:
        raise RuntimeError(f"S{subject:02d} {condition}: {len(m)} windows; expected {expect}")
    return m


def window_order_sha(meta: pd.DataFrame) -> str:
    return sha256_text("\n".join(meta["window_id"].astype(str).tolist()))


def load_raw_block(b13, meta: pd.DataFrame) -> np.ndarray:
    n = len(meta)
    raw = None
    for fvalue, sub in meta.groupby("window_array_file", sort=False):
        p = b13.resolve_window_array(str(fvalue))
        with np.load(p, allow_pickle=False) as z:
            arr = np.asarray(z["eeg_windows"])
        if arr.shape != (WINDOWS_PER_RECORDING, N_CHANNELS, WINDOW_SAMPLES):
            raise RuntimeError(f"{p}: eeg_windows {arr.shape}")
        if raw is None:
            raw = np.empty((n, N_CHANNELS, WINDOW_SAMPLES), dtype=arr.dtype)
        for pos, widx in zip(sub.index.to_numpy(), sub["window_index"].to_numpy()):
            raw[pos] = arr[int(widx)]
        del arr
    assert raw is not None
    assert_finite("raw BLOCK04 block", raw)
    first = meta.iloc[0]
    ref = np.asarray(b13.load_raw_window_from_row(first), dtype=np.float32)
    if not np.array_equal(raw[0].astype(np.float32), ref):
        raise RuntimeError("BLOCK04 window loader disagrees with BLOCK13 load_raw_window_from_row")
    raw.flags.writeable = False
    return raw


# ============================================================================
# 07. PERTURBATION ENGINES
# ============================================================================

@dataclass(frozen=True)
class Spec:
    ptype: str
    level: str
    level_index: int
    pattern: int
    realization: int

    @property
    def tag(self) -> str:
        if self.ptype == "clean":
            return "clean"
        if self.ptype == "channel_dropout":
            return f"drop{int(float(self.level)):02d}pct"
        return f"noiseL{self.level_index + 1:02d}_{re.sub(r'[^0-9A-Za-z]+', '_', self.level.replace('-', 'm').replace('.', 'p'))}"

    @property
    def dirname(self) -> str:
        if self.ptype == "clean":
            return "clean_reference"
        if self.ptype == "channel_dropout":
            return f"channel_dropout/{int(float(self.level)):02d}pct"
        return f"noise/{self.tag}"


@dataclass(frozen=True)
class Unit:
    regime: str
    subject: int
    condition: str
    model: str
    spec: Spec

    @property
    def branch(self) -> str:
        return "ML" if self.model in ML_MODELS else "DL"

    @property
    def uid(self) -> str:
        s = self.spec
        return (f"{self.regime}|S{self.subject:02d}|{self.condition}|{self.model}|{s.ptype}|{s.level}"
                f"|pat{s.pattern}|rea{s.realization}")

    def path(self, unit_root: Path) -> Path:
        s = self.spec
        name = (f"S{self.subject:02d}_{self.condition}_{safe_token(self.model)}_{s.ptype}_{s.tag}"
                f"_pat{s.pattern}_rea{s.realization}.npz")
        return unit_root / s.dirname / "units" / self.condition / safe_token(self.model) / name


def build_specs(cfg: dict) -> list[Spec]:
    specs = [Spec("clean", "0", 0, 0, 0)]
    cmap = normalize_channel_map(cfg)
    for lvl in CHANNEL_DROPOUT_LEVELS[1:]:
        for p in range(len(cmap[lvl])):
            specs.append(Spec("channel_dropout", str(lvl), CHANNEL_DROPOUT_LEVELS.index(lvl), p, 0))
    for li, v in enumerate(cfg["noise"]["noise_levels"]):
        for r in range(int(cfg["noise"]["n_noise_realizations"])):
            specs.append(Spec("noise", fmt_level(v), li, 0, r))
    return specs


def enumerate_units(cfg: dict, subjects=None, conditions=None, models=None,
                    specs: list[Spec] | None = None) -> list[Unit]:
    subjects = SUBJECTS if subjects is None else subjects
    conditions = CONDITIONS if conditions is None else conditions
    models = ALL_MODELS if models is None else models
    specs = specs if specs is not None else build_specs(cfg)
    return [Unit(rg, s, c, m, sp) for rg in cfg["evaluation_regimes"] for s in subjects
            for c in conditions for m in models for sp in specs]


def noise_rng(cfg: dict, subject: int, condition: str, gesture: int, set_label: str,
              window_index: int, level_index: int, realization: int) -> np.random.Generator:
    ss = np.random.SeedSequence([int(cfg["noise"]["noise_seed"]), int(subject), COND_CODE[condition],
                                 int(gesture), SET_CODE[set_label], int(window_index),
                                 int(level_index), int(realization)])
    return np.random.default_rng(ss)


def noise_std_for_window(cfg: dict, x: np.ndarray, level: float) -> np.ndarray:
    params = cfg["noise"]["noise_parameterization"]
    x64 = np.asarray(x, dtype=np.float64)
    if params["level_semantics"] == "std_absolute":
        return np.full((x64.shape[0], 1), float(level))
    ratio = 10.0 ** (float(level) / 20.0)
    if params["snr_reference"] == "per_channel_window_rms":
        return np.sqrt(np.mean(x64 ** 2, axis=1, keepdims=True)) / ratio
    g = float(np.sqrt(np.mean(x64 ** 2)))
    return np.full((x64.shape[0], 1), g / ratio)


def build_perturbed(raw: np.ndarray, meta: pd.DataFrame, cfg: dict, spec: Spec, raw_sha: str):
    info: dict[str, Any] = {"dropped_channels": "", "n_dropped": 0, "achieved_dropout_percent": 0.0,
                            "noise_realized": float("nan")}
    audit: dict[str, bool] = {}
    if spec.ptype == "clean":
        Xp = raw
        audit["perturbation_applied_correctly"] = True
    elif spec.ptype == "channel_dropout":
        idx = dropout_pattern_indices(cfg, int(spec.level), spec.pattern)
        keep = [i for i in range(N_CHANNELS) if i not in idx]
        Xp = raw.copy()
        Xp[:, idx, :] = 0.0
        audit["perturbation_applied_correctly"] = bool(
            np.all(Xp[:, idx, :] == 0.0) and np.array_equal(Xp[:, keep, :], raw[:, keep, :])
            and len(idx) == int(cfg["channel_dropout"]["percentage_to_count"][spec.level]))
        info["dropped_channels"] = "|".join(CHANNEL_NAMES[i] for i in idx)
        info["n_dropped"] = len(idx)
        info["achieved_dropout_percent"] = 100.0 * len(idx) / N_CHANNELS
    elif spec.ptype == "noise":
        level = float(spec.level)
        Xp = np.empty_like(raw)
        sig_e = 0.0
        res_e = 0.0
        exp_e = 0.0
        for i, r in enumerate(meta.itertuples(index=False)):
            x = raw[i]
            std = noise_std_for_window(cfg, x, level)
            z = noise_rng(cfg, int(r.subject), str(r.condition), int(r.gesture_code), str(r.set),
                          int(r.window_index), spec.level_index, spec.realization).standard_normal(x.shape)
            n_ = z * std
            Xp[i] = (np.asarray(x, dtype=np.float64) + n_).astype(raw.dtype)
            sig_e += float(np.sum(np.asarray(x, dtype=np.float64) ** 2))
            res_e += float(np.sum(n_ ** 2))
            exp_e += float(np.sum(std ** 2) * x.shape[1])
        sem = cfg["noise"]["noise_parameterization"]["level_semantics"]
        if sem == "snr_db":
            realized = 10.0 * math.log10(sig_e / res_e) if res_e > 0 and sig_e > 0 else float("nan")
            info["noise_realized"] = realized
            ok = math.isfinite(realized) and abs(realized - level) <= 0.25
        else:
            realized = math.sqrt(res_e / (len(meta) * N_CHANNELS * WINDOW_SAMPLES))
            info["noise_realized"] = realized
            ok = abs(realized - level) / level <= 0.01
        audit["perturbation_applied_correctly"] = bool(ok and not np.shares_memory(Xp, raw))
        info["noise_expected_energy"] = exp_e
    else:
        raise ValueError(spec.ptype)
    if Xp is not raw:
        assert_finite(f"perturbed input {spec.tag}", Xp)
    audit["clean_input_unchanged"] = bool(sha256_array(raw) == raw_sha)
    audit["perturbation_level_correct"] = bool(
        (spec.ptype == "clean" and spec.level == "0")
        or (spec.ptype == "channel_dropout" and int(float(spec.level)) in CHANNEL_DROPOUT_LEVELS[1:])
        or (spec.ptype == "noise" and spec.level == fmt_level(cfg["noise"]["noise_levels"][spec.level_index])))
    info["audit"] = audit
    return Xp, info


# ============================================================================
# 08. MODEL LOADING / PREDICTION
# ============================================================================

def ml_model_path(subject: int, condition: str, model: str) -> Path:
    return BLOCK06_ROOT / "models" / "LOSO" / f"Fold_S{subject:02d}_{condition}_{model}.joblib"


def load_ml_model(subject: int, condition: str, model: str):
    path = ml_model_path(subject, condition, model)
    if not path.exists():
        raise FileNotFoundError(f"Frozen BLOCK06 LOSO model missing: {path}")
    if joblib is None:
        raise RuntimeError(f"joblib unavailable: {JOBLIB_ERROR}")
    sha = cached_sha256(path)
    try:
        obj = joblib.load(path)
    except AttributeError as exc:
        raise RuntimeError(f"Cannot deserialise {path.name}: {exc}.") from exc
    if not hasattr(obj, "predict"):
        raise RuntimeError(f"{path}: object has no predict()")
    classes = getattr(obj, "classes_", None)
    if classes is None or [int(c) for c in np.asarray(classes).tolist()] != list(GESTURES):
        raise RuntimeError(f"{path.name}: classes_={classes!r}; expected exactly [1..7]")
    return obj, str(path), sha


def ml_predict(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    X = np.asarray(X, dtype=np.float64)
    assert_finite("ML feature matrix", X)
    pred = np.asarray(model.predict(X)).reshape(-1).astype(int)
    if len(pred) != len(X) or not set(pred.tolist()) <= set(GESTURES):
        raise RuntimeError("ML predictions invalid")
    proba = np.full((len(X), N_GESTURES), np.nan, dtype=np.float32)
    has = False
    if hasattr(model, "predict_proba"):
        try:
            p = np.asarray(model.predict_proba(X))
            if p.shape == (len(X), N_GESTURES):
                proba, has = p.astype(np.float32), True
        except Exception:
            pass
    return pred, proba, has


@dataclass
class DLBundle:
    model: Any
    mean: np.ndarray
    std: np.ndarray
    ckpt: str
    sha: str
    device: Any


class _FrozenCNNBiLSTM(torch.nn.Module if torch is not None else object):
    def __init__(self, n_channels=13, n_samples=500, n_classes=7, attention=False, dropout=0.3):
        super().__init__()
        self.attention = attention
        self.conv1 = torch.nn.Conv1d(n_channels, 32, kernel_size=15, padding=7, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.pool1 = torch.nn.MaxPool1d(2)
        self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=9, padding=4, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.pool2 = torch.nn.MaxPool1d(2)
        self.lstm = torch.nn.LSTM(input_size=64, hidden_size=64, num_layers=1,
                                  batch_first=True, bidirectional=True)
        if attention:
            self.attention_layer = torch.nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(dropout)
        self.representation_dim = 128
        self.classifier = torch.nn.Linear(128, n_classes)

    def cnn_features(self, x):
        x = torch.nn.functional.elu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.nn.functional.elu(self.bn2(self.conv2(x)))
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


def _extract_checkpoint_state_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = obj.get(key)
            if isinstance(value, dict) and all(torch.is_tensor(v) for v in value.values()):
                return value
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise RuntimeError("Frozen DL checkpoint does not contain a tensor state_dict")


def _load_verified_attention_checkpoint(ckpt: Path):
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = _extract_checkpoint_state_dict(obj)
    state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}
    model = _FrozenCNNBiLSTM(n_channels=N_CHANNELS, n_samples=WINDOW_SAMPLES,
                             n_classes=N_GESTURES, attention=True, dropout=0.3)
    expected = model.state_dict()
    shape_errors = []
    for key, tensor in state.items():
        if key in expected and tuple(tensor.shape) != tuple(expected[key].shape):
            shape_errors.append(f"{key}: checkpoint={tuple(tensor.shape)} model={tuple(expected[key].shape)}")
    missing = [k for k in expected if k not in state]
    unexpected = [k for k in state if k not in expected]
    if shape_errors or missing or unexpected:
        raise RuntimeError(
            "CNN-BiLSTM-Attention frozen checkpoint is incompatible with the canonical BLOCK07 architecture; "
            f"shape_errors={shape_errors[:10]}, missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def configure_canonical_dl_inference() -> Any:
    """Configure the BLOCK14 DL inference runtime to match frozen BLOCK07.

    BLOCK07 definitive LOSO predictions were generated with deterministic CUDA
    execution. BLOCK14 must use the same device/runtime contract when reproducing
    those frozen predictions; CPU inference is not an acceptable substitute for
    an exact prediction-identity audit.
    """
    if torch is None:
        raise RuntimeError(f"PyTorch unavailable: {TORCH_ERROR}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "BLOCK14 DL inference requires CUDA because the frozen BLOCK07 LOSO "
            "predictions were generated under the definitive GPU-only protocol."
        )
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "Strict deterministic PyTorch CUDA inference could not be enabled; "
            "do not proceed with the definitive BLOCK14 audit."
        ) from exc
    return torch.device("cuda")


def load_dl_model(b13, model_name: str, subject: int, condition: str, raw_block32: np.ndarray,
                  meta: pd.DataFrame, device: Any) -> DLBundle:
    if torch is None:
        raise RuntimeError(f"PyTorch unavailable: {TORCH_ERROR}")
    ckpt, _complete = b13._discover_exact_block07_checkpoint(subject, condition, model_name)
    model, mean, std = b13.load_block07_loso_model(model_name, subject, condition)
    if model_name == "CNN-BiLSTM-Attention":
        model = _load_verified_attention_checkpoint(Path(ckpt))
    raw_by_g = {}
    for g in GESTURES:
        sel = np.where(meta["gesture_code"].to_numpy() == g)[0][:9]
        raw_by_g[g] = np.asarray(raw_block32[sel], dtype=np.float32)
    b13._assert_dl_runtime_integrity(model, mean, std, raw_by_g, model_name)
    model = model.to(device)
    model.eval()
    return DLBundle(model, np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32),
                    str(ckpt), cached_sha256(Path(ckpt)), device)


def dl_predict(bundle: DLBundle, X: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Run DL inference under the frozen BLOCK07 GPU inference contract.

    Prediction identity is defined from argmax(logits), exactly as BLOCK07.
    Softmax is used only to archive probabilities and is not used to determine
    the class label.
    """
    X = np.asarray(X, dtype=np.float32)
    assert_finite("DL input", X)
    if X.ndim != 3 or X.shape[1:] != (N_CHANNELS, WINDOW_SAMPLES):
        raise RuntimeError(f"DL input shape {X.shape}")
    if batch_size < 1:
        raise RuntimeError(f"Invalid DL batch_size={batch_size}")

    xn = (X - bundle.mean[None, :, None]) / bundle.std[None, :, None]
    assert_finite("DL normalised input", xn)

    predictions = []
    probs = []
    with torch.no_grad():
        bundle.model.eval()
        for s in range(0, len(xn), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(xn[s:s + batch_size])).to(
                bundle.device, non_blocking=True
            )
            out = bundle.model(xb)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.ndim != 2 or out.shape[1] != N_GESTURES:
                raise RuntimeError(f"DL logits shape {tuple(out.shape)}")

            # BLOCK07 defines predictions directly from logits. Do the same here.
            pred = torch.argmax(out, dim=1).detach().cpu().numpy().astype(np.int64) + 1
            predictions.append(pred)

            # Probability archive is derived from the same logits on-device.
            p = torch.softmax(out, dim=1).detach().cpu().numpy().astype(np.float32)
            probs.append(p)

    if not predictions:
        return np.empty((0,), dtype=np.int64), np.empty((0, N_GESTURES), dtype=np.float32)

    pred_out = np.concatenate(predictions).astype(np.int64)
    proba = np.concatenate(probs).astype(np.float32)
    if len(pred_out) != len(X) or proba.shape != (len(X), N_GESTURES):
        raise RuntimeError("DL prediction/probability length mismatch")
    if not np.isfinite(pred_out).all() or not np.isfinite(proba).all():
        raise RuntimeError("Non-finite DL predictions/probabilities detected")
    if not set(np.unique(pred_out).tolist()).issubset(set(GESTURES)):
        raise RuntimeError("DL predictions contain labels outside gesture IDs 1..7")
    return pred_out, proba


# ============================================================================
# 09. METRICS
# ============================================================================

CLASS_PRF_NAMES = ("precision", "recall", "f1")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(GESTURES)
    return np.array([
        accuracy_score(y_true, y_pred),
        balanced_accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        matthews_corrcoef(y_true, y_pred),
        cohen_kappa_score(y_true, y_pred, labels=labels),
    ], dtype=np.float64)


def class_prf(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    labels = list(GESTURES)
    p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return np.stack([p, r, f], axis=1).astype(np.float64)


def unit_metric_arrays(y_true: np.ndarray, y_pred: np.ndarray, set_arr: np.ndarray) -> dict[str, np.ndarray]:
    metrics = calculate_metrics(y_true, y_pred)
    prf = class_prf(y_true, y_pred)
    conf = confusion_matrix(y_true, y_pred, labels=list(GESTURES)).astype(np.int32)
    set_metrics = np.stack([calculate_metrics(y_true[set_arr == s], y_pred[set_arr == s]) for s in SETS])
    for name, a in (("metrics", metrics), ("class_prf", prf), ("set_metrics", set_metrics)):
        assert_finite(f"unit {name}", a)
    return {"metrics": metrics, "class_prf": prf, "confusion": conf, "set_metrics": set_metrics}


# ============================================================================
# 10. RUN CONTEXT / UNIT ARCHIVE
# ============================================================================

@dataclass
class RunCtx:
    cfg: dict
    b13: Any
    manifest: pd.DataFrame
    extractor: Extractor | None
    config_sha: str
    code_sha: str
    registry: dict
    lookup: pd.DataFrame | None = None
    _meta_cache: dict | None = None

    def meta(self, subject: int, condition: str) -> pd.DataFrame:
        if self._meta_cache is None:
            self._meta_cache = {}
        k = (subject, condition)
        if k not in self._meta_cache:
            if len(self._meta_cache) > 4:
                self._meta_cache.clear()
            self._meta_cache[k] = block_meta(self.manifest, subject, condition)
        return self._meta_cache[k]

    def stored_features(self, meta: pd.DataFrame) -> np.ndarray:
        if self.lookup is None:
            self.lookup = self.b13.build_feature_lookup(self.manifest)
        sub = self.lookup[self.lookup.window_id.isin(set(meta.window_id))]
        sub = sub.set_index("window_id").loc[meta.window_id].reset_index()
        X, _ = self.b13.materialize_feature_rows(sub)
        if X.shape != (len(meta), N_FEATURES):
            raise RuntimeError(f"stored BLOCK05 features {X.shape}")
        return X


def _s(a: Any) -> np.ndarray:
    return np.asarray(str(a))


def write_unit(unit: Unit, unit_root: Path, ctx: RunCtx, meta: pd.DataFrame, y_pred: np.ndarray,
               proba: np.ndarray, has_proba: bool, spec_info: dict, model_path: str, model_sha: str,
               raw_sha: str, extras: dict | None = None) -> Path:
    y_true = meta["gesture_code"].to_numpy(dtype=np.int8)
    set_arr = meta["set"].to_numpy().astype(str)
    arrs = unit_metric_arrays(y_true.astype(int), y_pred.astype(int), set_arr)
    audit = dict(spec_info["audit"])
    audit["model_identity_verified"] = bool(ctx.registry.get((unit.regime, unit.model, unit.subject, unit.condition)) == model_sha)
    audit["subject_condition_correct"] = bool(
        (meta["subject"] == unit.subject).all() and (meta["condition"] == unit.condition).all())
    audit["labels_numeric_1_7"] = bool(set(np.unique(y_true).tolist()) == set(GESTURES))
    audit["no_retraining_no_tuning_no_adaptation"] = True
    audit["config_matches_frozen_lock"] = bool(ctx.config_sha == canonical_config_sha(ctx.cfg))
    names = sorted(audit)
    ex = extras or {}
    atomic_npz(
        unit.path(unit_root),
        schema=_s(UNIT_SCHEMA), unit_id=_s(unit.uid), regime=_s(unit.regime), subject=np.int16(unit.subject),
        condition=_s(unit.condition), model=_s(unit.model), branch=_s(unit.branch),
        perturbation_type=_s(unit.spec.ptype), level=_s(unit.spec.level),
        level_index=np.int16(unit.spec.level_index), pattern=np.int16(unit.spec.pattern),
        realization=np.int16(unit.spec.realization),
        dropped_channels=_s(spec_info["dropped_channels"]), n_dropped=np.int16(spec_info["n_dropped"]),
        achieved_dropout_percent=np.float64(spec_info["achieved_dropout_percent"]),
        noise_realized=np.float64(spec_info["noise_realized"]),
        window_order_sha256=_s(window_order_sha(meta)), n_windows=np.int32(len(meta)),
        y_true=y_true, y_pred=np.asarray(y_pred, dtype=np.int8), proba=np.asarray(proba, dtype=np.float32),
        has_probability=np.bool_(has_proba),
        metrics=arrs["metrics"], class_prf=arrs["class_prf"], confusion=arrs["confusion"],
        set_metrics=arrs["set_metrics"],
        audit_names=np.asarray(names), audit_values=np.asarray([audit[n] for n in names], dtype=np.bool_),
        config_sha256=_s(ctx.config_sha), code_sha256=_s(ctx.code_sha),
        model_path=_s(model_path), model_sha256=_s(model_sha), clean_input_sha256=_s(raw_sha),
        stored_feature_max_abs_diff=np.float64(ex.get("stored_feature_max_abs_diff", np.nan)),
        stored_pred_agreement=np.float64(ex.get("stored_pred_agreement", np.nan)),
        created_utc=_s(now_utc()),
    )
    return unit.path(unit_root)


_UNIT_REQUIRED = (
    "schema", "unit_id", "regime", "subject", "condition", "model", "branch", "perturbation_type", "level",
    "level_index", "pattern", "realization", "dropped_channels", "n_dropped", "achieved_dropout_percent",
    "noise_realized", "window_order_sha256", "n_windows", "y_true", "y_pred", "proba", "has_probability",
    "metrics", "class_prf", "confusion", "set_metrics", "audit_names", "audit_values", "config_sha256",
    "code_sha256", "model_path", "model_sha256", "clean_input_sha256", "stored_feature_max_abs_diff",
    "stored_pred_agreement", "created_utc",
)


def read_unit(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in _UNIT_REQUIRED if k not in z.files]
        if missing:
            raise RuntimeError(f"{path.name}: missing keys {missing}")
        d = {k: np.asarray(z[k]) for k in _UNIT_REQUIRED}
    return d


def _validate_optional_unit_fields(d: dict, unit: Unit) -> tuple[bool, str]:
    diff = float(np.asarray(d["stored_feature_max_abs_diff"]).reshape(-1)[0])
    agree = float(np.asarray(d["stored_pred_agreement"]).reshape(-1)[0])
    clean_ml = unit.spec.ptype == "clean" and unit.model in ML_MODELS
    if clean_ml:
        if not math.isfinite(diff):
            return False, "clean ML stored_feature_max_abs_diff is non-finite"
        if not math.isfinite(agree):
            return False, "clean ML stored_pred_agreement is non-finite"
        if diff < 0.0:
            return False, "clean ML stored_feature_max_abs_diff is negative"
        if not (0.0 <= agree <= 1.0):
            return False, "clean ML stored_pred_agreement outside [0,1]"
        return True, "OK"
    if math.isfinite(diff) and diff < 0.0:
        return False, "stored_feature_max_abs_diff is negative"
    if math.isfinite(agree) and not (0.0 <= agree <= 1.0):
        return False, "stored_pred_agreement outside [0,1]"
    return True, "OK"


_DRYRUN_ACTIVE = False


def _clean_reference_repair_units() -> set[tuple[int, str, str]]:
    if _DRYRUN_ACTIVE:
        return set()
    audit = TABLE_AUDIT / "BLOCK14_upstream_clean_reference_check.csv"
    if not audit.exists():
        return set()
    try:
        df = pd.read_csv(audit)
        required = {"subject", "condition", "model", "status"}
        if not required.issubset(df.columns):
            return set()
        out: set[tuple[int, str, str]] = set()
        for _, r in df.iterrows():
            if str(r["status"]).strip().upper().startswith("FAIL"):
                out.add((int(r["subject"]), str(r["condition"]).strip().upper(), str(r["model"])))
        return out
    except Exception:
        return set()


def validate_unit(path: Path, unit: Unit, ctx: RunCtx, order_sha: str, labels: np.ndarray,
                  allow_targeted_repair: bool = False) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if allow_targeted_repair:
        repair = _clean_reference_repair_units()
        if (unit.spec.ptype == "clean"
                and (unit.subject, unit.condition, unit.model) in repair):
            return False, "targeted clean-reference repair: audit FAIL"
    try:
        d = read_unit(path)
        n = len(labels)
        if str(d["schema"]) != UNIT_SCHEMA:
            return False, "schema"
        if str(d["unit_id"]) != unit.uid:
            return False, "unit_id mismatch"
        if str(d["config_sha256"]) != ctx.config_sha:
            return False, "config hash differs"
        if str(d["window_order_sha256"]) != order_sha or int(d["n_windows"]) != n:
            return False, "window order/count differs"
        if d["y_pred"].shape != (n,) or d["y_true"].shape != (n,) or d["proba"].shape != (n, N_GESTURES):
            return False, "array shape"
        if not np.array_equal(d["y_true"].astype(int), labels.astype(int)):
            return False, "y_true differs from BLOCK04 labels"
        if not set(np.unique(d["y_pred"]).tolist()) <= set(GESTURES):
            return False, "predicted labels outside 1..7"
        if d["metrics"].shape != (len(METRICS),) or not np.isfinite(d["metrics"]).all():
            return False, "metrics non-finite/shape"
        if not np.isfinite(d["class_prf"]).all() or not np.isfinite(d["set_metrics"]).all():
            return False, "class/set metrics non-finite"
        if bool(d["has_probability"]) and not np.isfinite(d["proba"]).all():
            return False, "probabilities non-finite"
        optional_ok, optional_reason = _validate_optional_unit_fields(d, unit)
        if not optional_ok:
            return False, optional_reason
        if not bool(np.all(d["audit_values"])):
            bad = [str(n_) for n_, v in zip(d["audit_names"], d["audit_values"]) if not v]
            return False, f"audit failed: {bad}"
        reg = ctx.registry.get((unit.regime, unit.model, unit.subject, unit.condition))
        if reg is None or str(d["model_sha256"]) != reg:
            return False, "model hash differs from frozen registry"
        return True, "OK"
    except Exception as exc:
        return False, f"unreadable: {type(exc).__name__}: {exc}"


class Progress:
    def __init__(self, path: Path, total: int):
        self.path, self.total = path, total
        self.state = {"total_units": total, "session_computed": 0, "session_skipped": 0,
                      "last_completed_unit": None, "current_unit": None, "started_utc": now_utc()}

    def current(self, uid: str) -> None:
        self.state["current_unit"] = uid
        atomic_json(self.path, self.state)

    def done(self, uid: str) -> None:
        self.state["last_completed_unit"] = uid
        self.state["current_unit"] = None
        self.state["session_computed"] += 1
        self.state["updated_utc"] = now_utc()
        atomic_json(self.path, self.state)


def record_failure(unit_id: str, exc: BaseException) -> None:
    FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {"unit": unit_id, "error_type": type(exc).__name__, "error": str(exc)[:2000],
           "traceback": traceback.format_exc()[-4000:], "time": now_utc()}
    with FAILURES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(rec)) + "\n")


# ============================================================================
# 11. STREAMING EXECUTION
# ============================================================================

def run_units(ctx: RunCtx, unit_root: Path, subjects: Iterable[int], conditions: Iterable[str],
              models: Iterable[str], specs: list[Spec], progress: Progress) -> dict:
    cfg = ctx.cfg
    dl_device = configure_canonical_dl_inference() if any(m in DL_MODELS for m in models) else None
    models = list(models)
    ml_models = [m for m in models if m in ML_MODELS]
    dl_models = [m for m in models if m in DL_MODELS]
    stats = {"computed": 0, "skipped": 0, "failed": 0}
    cont = bool(cfg["runtime"].get("continue_on_unit_failure", False))
    every = int(cfg["runtime"].get("memory_log_every_n_units", 25))
    for subject in subjects:
        for condition in conditions:
            meta = ctx.meta(subject, condition)
            order_sha = window_order_sha(meta)
            labels = meta["gesture_code"].to_numpy(dtype=int)
            missing: dict[str, list[Spec]] = {}
            for m in models:
                for sp in specs:
                    u = Unit(cfg["evaluation_regimes"][0], subject, condition, m, sp)
                    ok, _why = validate_unit(u.path(unit_root), u, ctx, order_sha, labels,
                                             allow_targeted_repair=True)
                    if ok:
                        stats["skipped"] += 1
                        progress.state["session_skipped"] += 1
                    else:
                        missing.setdefault(m, []).append(sp)
            n_missing = sum(len(v) for v in missing.values())
            if not missing:
                log(f"SKIP S{subject:02d} {condition}: all {len(models) * len(specs)} units verified")
                continue
            log(f"BLOCK S{subject:02d} {condition}: {n_missing} unit(s) to compute "
                f"({len(models) * len(specs) - n_missing} verified/skipped)")
            log_memory(f"S{subject:02d} {condition} before raw-window load")
            raw = load_raw_block(ctx.b13, meta)
            raw_sha = sha256_array(raw)
            regime = cfg["evaluation_regimes"][0]
            try:
                ml_specs = sorted({sp for m in ml_models for sp in missing.get(m, [])},
                                  key=lambda s: (s.ptype, s.level_index, s.pattern, s.realization))
                feats: dict[Spec, np.ndarray] = {}
                infos: dict[Spec, dict] = {}
                extras: dict[Spec, dict] = {}
                if ml_specs:
                    stored = ctx.stored_features(meta) if any(s.ptype == "clean" for s in ml_specs) else None
                    for sp in ml_specs:
                        tag = f"S{subject:02d} {condition} {sp.tag} pat{sp.pattern} rea{sp.realization}"
                        log(f"FEATURES {tag}: perturb + BLOCK05 extraction ({len(meta)} windows)")
                        Xp, info = build_perturbed(raw, meta, cfg, sp, raw_sha)
                        Xf = extract_features_batch(ctx.extractor, Xp, tag)
                        feats[sp], infos[sp] = Xf, info
                        if sp.ptype == "clean":
                            diff = float(np.max(np.abs(Xf - stored)))
                            ok = bool(np.allclose(Xf, stored,
                                                  rtol=float(cfg["upstream"]["feature_reproduction_rtol"]),
                                                  atol=float(cfg["upstream"]["feature_reproduction_atol"])))
                            if not ok:
                                j = int(np.argmax(np.max(np.abs(Xf - stored), axis=0)))
                                raise RuntimeError(
                                    f"{tag}: recomputed clean features differ from stored BLOCK05 features "
                                    f"(max|diff|={diff:.3e}, worst feature={b13_feature_name(ctx.b13, j)})."
                                )
                            extras[sp] = {"stored_feature_max_abs_diff": diff, "stored_X": stored}
                        del Xp
                    gc.collect()
                for m in ml_models:
                    todo = missing.get(m, [])
                    if not todo:
                        continue
                    log(f"ML MODEL S{subject:02d} {condition} {m}: loading frozen model")
                    model, mpath, msha = load_ml_model(subject, condition, m)
                    for sp in todo:
                        u = Unit(regime, subject, condition, m, sp)
                        progress.current(u.uid)
                        try:
                            t0 = time.perf_counter()
                            pred, proba, has = ml_predict(model, feats[sp])
                            ex = {}
                            if sp.ptype == "clean":
                                pred_s, _, _ = ml_predict(model, extras[sp]["stored_X"])
                                agree = float(np.mean(pred == pred_s))
                                if agree < float(cfg["upstream"]["clean_prediction_agreement_min"]):
                                    raise RuntimeError(f"{u.uid}: clean predictions on recomputed vs stored "
                                                       f"BLOCK05 features agree on only {agree:.6f}")
                                ex = {"stored_feature_max_abs_diff": extras[sp]["stored_feature_max_abs_diff"],
                                      "stored_pred_agreement": agree}
                            write_unit(u, unit_root, ctx, meta, pred, proba, has, infos[sp], mpath, msha, raw_sha, ex)
                            progress.done(u.uid)
                            stats["computed"] += 1
                            log(f"UNIT OK {u.uid} ({time.perf_counter() - t0:.1f}s)")
                            if stats["computed"] % every == 0:
                                log_memory(f"after {stats['computed']} units")
                        except Exception as exc:
                            stats["failed"] += 1
                            record_failure(u.uid, exc)
                            log(f"UNIT FAIL {u.uid}: {type(exc).__name__}: {exc}")
                            if not cont:
                                raise
                    del model
                    gc.collect()
                    log_memory(f"S{subject:02d} {condition} {m} after cleanup")
                feats.clear(); infos.clear(); extras.clear()
                gc.collect()
                dl_todo = {m: missing[m] for m in dl_models if missing.get(m)}
                if dl_todo:
                    raw32 = raw.astype(np.float32, copy=False)
                    bundles: dict[str, DLBundle] = {}
                    for m in dl_todo:
                        log(f"DL MODEL S{subject:02d} {condition} {m}: loading frozen checkpoint + integrity check")
                        bundles[m] = load_dl_model(ctx.b13, m, subject, condition, raw32, meta, dl_device)
                    dl_specs = sorted({sp for v in dl_todo.values() for sp in v},
                                      key=lambda s: (s.ptype, s.level_index, s.pattern, s.realization))
                    for sp in dl_specs:
                        Xp, info = build_perturbed(raw, meta, cfg, sp, raw_sha)
                        X32 = np.asarray(Xp, dtype=np.float32)
                        for m, todo in dl_todo.items():
                            if sp not in todo:
                                continue
                            u = Unit(regime, subject, condition, m, sp)
                            progress.current(u.uid)
                            try:
                                t0 = time.perf_counter()
                                pred, proba = dl_predict(bundles[m], X32, int(cfg["runtime"]["dl_batch_size"]))
                                write_unit(u, unit_root, ctx, meta, pred, proba, True, info,
                                           bundles[m].ckpt, bundles[m].sha, raw_sha, None)
                                progress.done(u.uid)
                                stats["computed"] += 1
                                log(f"UNIT OK {u.uid} ({time.perf_counter() - t0:.1f}s)")
                                if stats["computed"] % every == 0:
                                    log_memory(f"after {stats['computed']} units")
                            except Exception as exc:
                                stats["failed"] += 1
                                record_failure(u.uid, exc)
                                log(f"UNIT FAIL {u.uid}: {type(exc).__name__}: {exc}")
                                if not cont:
                                    raise
                        del Xp, X32
                        gc.collect()
                    bundles.clear()
                    del raw32
                if sha256_array(raw) != raw_sha:
                    raise RuntimeError(f"S{subject:02d} {condition}: clean BLOCK04 input changed during the block")
            finally:
                del raw
                gc.collect()
                log_memory(f"S{subject:02d} {condition} block cleanup")
    return stats


def b13_feature_name(b13, j: int) -> str:
    try:
        return str(b13.FEATURE_NAMES[j])
    except Exception:
        return f"feature_index_{j}"


# ============================================================================
# 12. STATISTICS
# ============================================================================

def benjamini_hochberg(pvalues, q: float = 0.05) -> np.ndarray:
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


def paired_signflip_p(values, n_perm: int = 10000, seed: int = 42) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    observed = abs(float(np.mean(x)))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    exceed = 0
    total = 0
    for start in range(0, n_perm, 500):
        n = min(500, n_perm - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, len(x)))
        exceed += int(np.count_nonzero(np.abs((signs * x[None, :]).mean(axis=1)) >= observed - 1e-15))
        total += n
    return float((exceed + 1) / (total + 1))


def paired_effect_size(values) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    return float(np.mean(x) / sd) if sd > 0 else float("nan")


def bootstrap_mean_ci(values, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "n_subjects": int(len(x)), "mean_diff": float(np.mean(x)), "median_diff": float(np.median(x)),
        "sd_diff": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        "iqr_diff": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "min_diff": float(np.min(x)), "max_diff": float(np.max(x)),
    }


ID_COLS = ["regime", "subject", "condition", "model", "branch", "perturbation_type", "level",
           "level_index", "pattern", "realization"]
UNIT_KEY = ["regime", "subject", "condition", "model"]


# ============================================================================
# 13. STREAMING SCAN
# ============================================================================

def _ids(d: dict) -> dict:
    return {
        "regime": str(d["regime"]), "subject": int(d["subject"]), "condition": str(d["condition"]),
        "model": str(d["model"]), "branch": str(d["branch"]), "perturbation_type": str(d["perturbation_type"]),
        "level": str(d["level"]), "level_index": int(d["level_index"]), "pattern": int(d["pattern"]),
        "realization": int(d["realization"]),
    }


class PredictionExporter:
    def __init__(self, enabled: bool, root: Path):
        self.enabled, self.root = enabled, root
        self.files: dict[tuple[str, str], list] = {}

    def write(self, ids: dict, meta: pd.DataFrame, d: dict) -> None:
        if not self.enabled:
            return
        ptype = ids["perturbation_type"]
        key = (ids["branch"], ptype)
        if key not in self.files:
            final = self.root / ids["branch"] / f"BLOCK14_{ids['branch']}_{ptype}_predictions.csv"
            tmp = _tmp_path(final)
            final.parent.mkdir(parents=True, exist_ok=True)
            self.files[key] = [final, tmp, False]
        final, tmp, wrote = self.files[key]
        pp_ = d["proba"][np.arange(len(meta)), d["y_pred"].astype(int) - 1] if bool(d["has_probability"]) \
            else np.full(len(meta), np.nan, dtype=np.float32)
        df = pd.DataFrame({
            "subject": ids["subject"], "condition": ids["condition"], "gesture": meta["gesture_code"].to_numpy(dtype=int),
            "set": meta["set"].to_numpy(), "source_recording": meta["source_recording"].to_numpy(),
            "window_id": meta["window_id"].to_numpy(), "window_index": meta["window_index"].to_numpy(dtype=int),
            "model": ids["model"], "branch": ids["branch"], "evaluation_regime": ids["regime"],
            "perturbation_type": ptype, "perturbation_level": ids["level"], "pattern_id": ids["pattern"],
            "realization": ids["realization"],
            "clean_or_perturbed": "clean" if ptype == "clean" else "perturbed",
            "true_label": d["y_true"].astype(int), "predicted_label": d["y_pred"].astype(int),
            "prediction_probability": pp_,
        })
        df.to_csv(tmp, mode="a", header=not wrote, index=False)
        self.files[key][2] = True

    def close(self) -> list[Path]:
        out = []
        for final, tmp, wrote in self.files.values():
            if wrote:
                os.replace(tmp, final)
                out.append(final)
        return out


def scan_units(ctx: RunCtx, unit_root: Path, units: list[Unit], export: bool) -> dict:
    metric_rows, class_rows, set_rows, conf_rows, audit_rows = [], [], [], [], []
    missing, invalid = [], []
    exporter = PredictionExporter(export, PRED_ROOT)
    seen = set()
    duplicates = 0
    metas: dict[tuple[int, str], tuple] = {}
    for i, u in enumerate(units):
        key = (u.subject, u.condition)
        if key not in metas:
            if len(metas) > 2:
                metas.clear()
            m = ctx.meta(u.subject, u.condition)
            metas[key] = (m, window_order_sha(m), m["gesture_code"].to_numpy(dtype=int))
        meta, osha, labels = metas[key]
        p = u.path(unit_root)
        ok, why = validate_unit(p, u, ctx, osha, labels, allow_targeted_repair=False)
        if not ok:
            (missing if why == "missing" else invalid).append({"unit": u.uid, "reason": why})
            continue
        if u.uid in seen:
            duplicates += 1
        seen.add(u.uid)
        d = read_unit(p)
        ids = _ids(d)
        mrow = dict(ids)
        mrow.update({
            "dropped_channels": str(d["dropped_channels"]), "n_dropped": int(d["n_dropped"]),
            "achieved_dropout_percent": float(d["achieved_dropout_percent"]),
            "noise_realized": float(d["noise_realized"]),
        })
        mrow.update({m_: float(v) for m_, v in zip(METRICS, d["metrics"])})
        n_cls = int(len(np.unique(d["y_pred"])))
        mrow["n_distinct_predicted_classes"] = n_cls
        mrow["prediction_collapse_single_class"] = bool(n_cls <= 1)
        mrow.update({"model_sha256": str(d["model_sha256"]), "code_sha256": str(d["code_sha256"]),
                     "config_sha256": str(d["config_sha256"]), "clean_input_sha256": str(d["clean_input_sha256"]),
                     "stored_feature_max_abs_diff": float(d["stored_feature_max_abs_diff"]),
                     "stored_pred_agreement": float(d["stored_pred_agreement"]),
                     "unit_file": str(p)})
        metric_rows.append(mrow)
        for gi, g in enumerate(GESTURES):
            r = dict(ids)
            r["gesture"] = int(g)
            r.update({n: float(d["class_prf"][gi, j]) for j, n in enumerate(CLASS_PRF_NAMES)})
            class_rows.append(r)
        for si, s in enumerate(SETS):
            r = dict(ids)
            r["set"] = s
            r.update({m_: float(v) for m_, v in zip(METRICS, d["set_metrics"][si])})
            set_rows.append(r)
        c = dict(ids)
        conf = d["confusion"]
        c.update({f"true{a + 1}_pred{b + 1}": int(conf[a, b]) for a in range(N_GESTURES) for b in range(N_GESTURES)})
        conf_rows.append(c)
        a_row = dict(ids)
        a_row.update({str(n): bool(v) for n, v in zip(d["audit_names"], d["audit_values"])})
        a_row["model_sha256"] = str(d["model_sha256"])
        audit_rows.append(a_row)
        exporter.write(ids, meta, d)
        del d
        if (i + 1) % 1000 == 0:
            log(f"SCAN {i + 1}/{len(units)} units read")
            gc.collect()
    exported = exporter.close()
    return {
        "metrics": pd.DataFrame(metric_rows), "class": pd.DataFrame(class_rows), "set": pd.DataFrame(set_rows),
        "confusion": pd.DataFrame(conf_rows), "audit": pd.DataFrame(audit_rows),
        "missing": missing, "invalid": invalid, "duplicates": duplicates, "exported": exported,
    }


# ============================================================================
# 14. DEGRADATION / SUBJECT-LEVEL STATISTICS
# ============================================================================

def build_degradation(metrics_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    clean = metrics_df[metrics_df.perturbation_type == "clean"]
    pert = metrics_df[metrics_df.perturbation_type != "clean"]
    if clean.duplicated(UNIT_KEY).any():
        raise RuntimeError("duplicate clean reference rows")
    cols_c = UNIT_KEY + list(METRICS)
    merged = pert.merge(clean[cols_c], on=UNIT_KEY, how="left", suffixes=("", "_clean"), validate="many_to_one")
    if merged[[f"{m}_clean" for m in METRICS]].isna().any().any():
        raise RuntimeError("perturbed result without a matched clean reference")
    out = pd.DataFrame({
        "subject": merged.subject, "condition": merged.condition, "model": merged.model, "branch": merged.branch,
        "evaluation_regime": merged.regime, "perturbation_type": merged.perturbation_type,
        "perturbation_level": merged.level, "level_index": merged.level_index, "pattern_id": merged.pattern,
        "realization": merged.realization,
        "dropout_percent": np.where(merged.perturbation_type == "channel_dropout", merged.level.astype(float), np.nan),
        "achieved_dropout_percent": merged.achieved_dropout_percent, "n_dropped": merged.n_dropped,
        "dropped_channels": merged.dropped_channels,
        "noise_type": np.where(merged.perturbation_type == "noise", cfg["noise"]["noise_type"], ""),
        "noise_level_semantics": np.where(merged.perturbation_type == "noise",
                                          cfg["noise"]["noise_parameterization"]["level_semantics"], ""),
        "noise_realized": merged.noise_realized,
        "gesture_scope": "all_gestures_1-7", "set_scope": "SetA|SetB|SetC",
    })
    out["metric"] = PRIMARY_METRIC
    out["clean_metric"] = merged[f"{PRIMARY_METRIC}_clean"].to_numpy()
    out["perturbed_metric"] = merged[PRIMARY_METRIC].to_numpy()
    out["delta_metric"] = out["clean_metric"] - out["perturbed_metric"]
    cm = out["clean_metric"].to_numpy()
    out["relative_delta_metric"] = np.where(cm > 0, out["delta_metric"].to_numpy() / np.where(cm > 0, cm, 1.0), np.nan)
    for m in METRICS:
        out[f"clean_{m}"] = merged[f"{m}_clean"].to_numpy()
        out[f"perturbed_{m}"] = merged[m].to_numpy()
        out[f"delta_{m}"] = out[f"clean_{m}"] - out[f"perturbed_{m}"]
    return out.sort_values(["perturbation_type", "model", "condition", "level_index", "pattern_id",
                            "realization", "subject"]).reset_index(drop=True)


def subject_aggregate(deg: pd.DataFrame) -> pd.DataFrame:
    keys = ["evaluation_regime", "subject", "condition", "model", "branch", "perturbation_type",
            "perturbation_level", "level_index"]
    num = [f"{p}_{m}" for p in ("clean", "perturbed", "delta") for m in METRICS]
    g = deg.groupby(keys, sort=False)
    agg = g[num].mean().reset_index()
    agg["n_units_averaged"] = g.size().to_numpy()
    return agg.sort_values(["perturbation_type", "model", "condition", "level_index", "subject"]).reset_index(drop=True)


def compute_stat_rows(agg: pd.DataFrame, cfg: dict, expected_subjects: int | None) -> pd.DataFrame:
    st = cfg["statistics"]
    nb, sb = int(st["bootstrap_resamples"]), int(st["bootstrap_seed"])
    npm, sp = int(st["permutation_resamples"]), int(st["permutation_seed"])
    rows = []
    gk = ["evaluation_regime", "branch", "model", "condition", "perturbation_type", "perturbation_level", "level_index"]
    for keys, g in agg.groupby(gk, sort=False):
        g = g.sort_values("subject")
        if expected_subjects is not None and len(g) != expected_subjects:
            raise RuntimeError(f"{keys}: {len(g)} subjects; expected {expected_subjects}")
        for m in METRICS:
            d = g[f"delta_{m}"].to_numpy(dtype=float)
            lo, hi = bootstrap_mean_ci(d, nb, sb)
            r = dict(zip(gk, keys))
            r.update({"comparison": "clean_vs_perturbed", "metric": m})
            r.update(describe(d))
            r.update({
                "ci95_low": lo, "ci95_high": hi, "effect_size_dz": paired_effect_size(d),
                "p_signflip": paired_signflip_p(d, npm, sp),
                "clean_mean": float(g[f"clean_{m}"].mean()),
                "clean_sd": float(g[f"clean_{m}"].std(ddof=1)) if len(g) > 1 else float("nan"),
                "perturbed_mean": float(g[f"perturbed_{m}"].mean()),
                "perturbed_sd": float(g[f"perturbed_{m}"].std(ddof=1)) if len(g) > 1 else float("nan"),
            })
            rows.append(r)
    gk2 = ["evaluation_regime", "branch", "model", "perturbation_type", "perturbation_level", "level_index"]
    for keys, g in agg.groupby(gk2, sort=False):
        me = g[g.condition == "ME"].set_index("subject").sort_index()
        mi = g[g.condition == "MI"].set_index("subject").sort_index()
        common = me.index.intersection(mi.index)
        if len(common) == 0:
            continue
        if expected_subjects is not None and len(common) != expected_subjects:
            raise RuntimeError(f"{keys}: ME/MI pairing has {len(common)} subjects; expected {expected_subjects}")
        for m in METRICS:
            d = (mi.loc[common, f"delta_{m}"] - me.loc[common, f"delta_{m}"]).to_numpy(dtype=float)
            lo, hi = bootstrap_mean_ci(d, nb, sb)
            r = dict(zip(gk2, keys))
            r.update({"condition": "MI-ME", "comparison": "ME_vs_MI", "metric": m})
            r.update(describe(d))
            r.update({"ci95_low": lo, "ci95_high": hi, "effect_size_dz": paired_effect_size(d),
                      "p_signflip": paired_signflip_p(d, npm, sp),
                      "clean_mean": np.nan, "clean_sd": np.nan, "perturbed_mean": np.nan, "perturbed_sd": np.nan})
            rows.append(r)
    stats = pd.DataFrame(rows)
    stats["family_id"] = stats["comparison"] + "|" + stats["perturbation_type"] + "|" + stats["metric"]
    stats["q_bh"] = np.nan
    for _, idx in stats.groupby("family_id").groups.items():
        idx = list(idx)
        stats.loc[idx, "q_bh"] = benjamini_hochberg(stats.loc[idx, "p_signflip"].to_numpy(), float(st["fdr_q"]))
    stats["primary_family"] = stats["metric"] == PRIMARY_METRIC
    return stats


def level_label(cfg: dict, ptype: str, level: str) -> str:
    if ptype == "channel_dropout":
        return f"{int(float(level))}% dropout"
    sem = cfg["noise"]["noise_parameterization"]["level_semantics"]
    return f"SNR {level} dB" if sem == "snr_db" else f"sigma={level}"


def pair_delta_table(df_unit: pd.DataFrame, extra_key: str, cols: list[str]) -> pd.DataFrame:
    clean = df_unit[df_unit.perturbation_type == "clean"]
    pert = df_unit[df_unit.perturbation_type != "clean"]
    k = UNIT_KEY + [extra_key]
    merged = pert.merge(clean[k + cols], on=k, how="left", suffixes=("", "_clean"), validate="many_to_one")
    if merged[[f"{c}_clean" for c in cols]].isna().any().any():
        raise RuntimeError(f"{extra_key}-level pairing lacks clean reference")
    out = merged[ID_COLS + [extra_key]].copy()
    for c in cols:
        out[f"clean_{c}"] = merged[f"{c}_clean"].to_numpy()
        out[f"perturbed_{c}"] = merged[c].to_numpy()
        out[f"delta_{c}"] = out[f"clean_{c}"] - out[f"perturbed_{c}"]
    return out


# ============================================================================
# 15. COMPARISONS
# ============================================================================

def make_comparisons(agg: pd.DataFrame, stats: pd.DataFrame) -> dict[str, pd.DataFrame]:
    gk = ["evaluation_regime", "branch", "model", "condition", "perturbation_type", "perturbation_level", "level_index"]
    rows = []
    for keys, g in agg.groupby(gk, sort=False):
        r = dict(zip(gk, keys))
        r["aggregation"] = "model"
        r["n_subjects"] = int(g.subject.nunique())
        for m in ("Accuracy", "Balanced Accuracy", "Macro-F1"):
            c, p, d = g[f"clean_{m}"].mean(), g[f"perturbed_{m}"].mean(), g[f"delta_{m}"].mean()
            r[f"clean_{m}"], r[f"perturbed_{m}"], r[f"absolute_delta_{m}"] = c, p, d
            r[f"relative_delta_{m}"] = d / c if c > 0 else np.nan
        rows.append(r)
    model_level = pd.DataFrame(rows)
    num = [c for c in model_level.columns if c.startswith(("clean_", "perturbed_", "absolute_delta_", "relative_delta_"))]
    br = (model_level.groupby(["evaluation_regime", "branch", "condition", "perturbation_type",
                               "perturbation_level", "level_index"], sort=False)[num].mean().reset_index())
    br["model"], br["aggregation"], br["n_subjects"] = "ALL_" + br["branch"] + "_MODELS", "mean_over_models", len(SUBJECTS)
    ml_dl = pd.concat([model_level, br[model_level.columns]], ignore_index=True)

    pri = stats[(stats.comparison == "clean_vs_perturbed")]
    me = pri[pri.condition == "ME"][["evaluation_regime", "branch", "model", "perturbation_type", "perturbation_level",
                                     "level_index", "metric", "mean_diff"]].rename(columns={"mean_diff": "delta_ME"})
    mi = pri[pri.condition == "MI"][["evaluation_regime", "branch", "model", "perturbation_type", "perturbation_level",
                                     "level_index", "metric", "mean_diff"]].rename(columns={"mean_diff": "delta_MI"})
    k = ["evaluation_regime", "branch", "model", "perturbation_type", "perturbation_level", "level_index", "metric"]
    memi = me.merge(mi, on=k, how="inner", validate="one_to_one")
    pair = stats[stats.comparison == "ME_vs_MI"][k + ["n_subjects", "mean_diff", "ci95_low", "ci95_high",
                                                      "effect_size_dz", "p_signflip", "q_bh"]]
    memi = memi.merge(pair, on=k, how="left", validate="one_to_one").rename(
        columns={"mean_diff": "delta_MI_minus_delta_ME"})
    degsum = stats[(stats.comparison == "clean_vs_perturbed") & (stats.metric == PRIMARY_METRIC)][
        gk + ["clean_mean", "perturbed_mean", "mean_diff", "median_diff", "sd_diff", "iqr_diff", "min_diff",
              "max_diff", "ci95_low", "ci95_high"]].rename(columns={"mean_diff": "mean_delta_P", "median_diff": "median_delta_P"})
    return {"ml_dl": ml_dl, "me_mi": memi, "degradation_summary": degsum}


# ============================================================================
# 16. PUBLICATION TABLES
# ============================================================================

def _pp(x: float) -> str:
    return "" if not np.isfinite(x) else f"{x * PUB_SCALE:.1f}"


def _ci(lo: float, hi: float) -> str:
    return "" if not (np.isfinite(lo) and np.isfinite(hi)) else f"[{lo * PUB_SCALE:.1f}, {hi * PUB_SCALE:.1f}]"


def _p(x: float) -> str:
    if not np.isfinite(x):
        return ""
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def _dz(x: float) -> str:
    return "" if not np.isfinite(x) else f"{x:.2f}"


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def make_publication_tables(stats_csv: Path, cfg: dict, drop_manifest_csv: Path) -> list[dict]:
    st = pd.read_csv(stats_csv)
    pri = st[st.metric == PRIMARY_METRIC]
    cvp = pri[pri.comparison == "clean_vs_perturbed"].copy()
    dm = pd.read_csv(drop_manifest_csv)
    records = []

    def emit(table_id: str, name: str, caption: str, df: pd.DataFrame, models: list[str], conds: list[str],
             perts: list[str], statistics: str) -> None:
        csv_p = TABLE_PUB / f"{table_id}_{name}.csv"
        md_p = TABLE_PUB / f"{table_id}_{name}.md"
        atomic_csv(csv_p, df)
        atomic_text(md_p, f"**{table_id}. {caption}**\n\n" + to_markdown(df))
        records.append({
            "table_id": table_id, "table_name": name, "caption": caption, "files": f"{csv_p}|{md_p}",
            "source_table": stats_csv.name, "source_file": str(stats_csv), "source_file_sha256": sha256_file(stats_csv),
            "n_rows": len(df), "models": "|".join(models), "conditions": "|".join(conds),
            "perturbations": "|".join(perts), "statistics": statistics,
            "generated_timestamp": now_utc(), "code_version_hash": code_sha256(),
        })

    order = {m: i for i, m in enumerate(ALL_MODELS)}
    rows = []
    for (reg, model, cond), g in cvp[cvp.perturbation_type == "channel_dropout"].groupby(
            ["evaluation_regime", "model", "condition"], sort=False):
        r = {"Model": DISPLAY_NAME[model], "Branch": g.branch.iloc[0], "Condition": cond, "Regime": reg,
             "Clean Macro-F1 (%), mean (SD)": f"{_pp(g.clean_mean.iloc[0])} ({_pp(g.clean_sd.iloc[0])})",
             "_o": order[model]}
        for lvl in CHANNEL_DROPOUT_LEVELS[1:]:
            gl = g[g.perturbation_level.astype(float) == lvl]
            r[f"{lvl}% dropout Macro-F1 (%)"] = _pp(gl.perturbed_mean.iloc[0]) if len(gl) else ""
        g30 = g[g.perturbation_level.astype(float) == 30]
        r["ΔP at 30% (pp)"] = _pp(g30.mean_diff.iloc[0]) if len(g30) else ""
        r["ΔP at 30% 95% CI"] = _ci(g30.ci95_low.iloc[0], g30.ci95_high.iloc[0]) if len(g30) else ""
        rows.append(r)
    t14a = pd.DataFrame(rows).sort_values(["_o", "Condition"]).drop(columns="_o").reset_index(drop=True)
    emit("TABLE_14A", "robustness_summary_by_model_condition",
         "Robustness summary by model and condition (LOSO, subject-level Macro-F1).",
         t14a, list(ALL_MODELS), list(CONDITIONS), ["channel_dropout"], "mean, SD, bootstrap 95% CI over subjects")

    def deg_table(ptype: str) -> pd.DataFrame:
        g0 = cvp[cvp.perturbation_type == ptype].copy()
        g0["_o"] = g0.model.map(order)
        g0 = g0.sort_values(["_o", "condition", "level_index"])
        out = []
        for _, r in g0.iterrows():
            rec = {"Model": DISPLAY_NAME[r.model], "Condition": r.condition,
                   "Perturbation": level_label(cfg, ptype, str(r.perturbation_level))}
            if ptype == "channel_dropout":
                sub = dm[dm.dropout_percent == int(float(r.perturbation_level))]
                if len(sub):
                    m = sub.iloc[0]
                    rec["Channels dropped (n; achieved %)"] = f"{int(m.n_channels)}; {m.achieved_percent:.1f}"
            rec.update({
                "Clean Macro-F1 (%)": _pp(r.clean_mean), "Perturbed Macro-F1 (%)": _pp(r.perturbed_mean),
                "ΔP (pp)": _pp(r.mean_diff), "95% CI": _ci(r.ci95_low, r.ci95_high),
                "Effect size d_z": _dz(r.effect_size_dz), "Adjusted p (BH)": _p(r.q_bh),
            })
            out.append(rec)
        return pd.DataFrame(out)

    emit("TABLE_14B", "channel_dropout_performance_degradation",
         "Channel-dropout performance degradation (LOSO, Macro-F1).",
         deg_table("channel_dropout"), list(ALL_MODELS), list(CONDITIONS), ["channel_dropout"],
         "paired sign-flip permutation, BH-FDR, d_z, bootstrap 95% CI")
    emit("TABLE_14C", "noise_robustness_performance_degradation",
         "Controlled-noise performance degradation (LOSO, Macro-F1).",
         deg_table("noise"), list(ALL_MODELS), list(CONDITIONS), ["noise"],
         "paired sign-flip permutation, BH-FDR, d_z, bootstrap 95% CI")

    rows = []
    for _, r in pri.sort_values(["comparison", "perturbation_type", "level_index"]).iterrows():
        rows.append({
            "Comparison": "Clean vs perturbed" if r.comparison == "clean_vs_perturbed" else "ΔP(MI) − ΔP(ME)",
            "Model": DISPLAY_NAME[r.model], "Condition": r.condition,
            "Perturbation": level_label(cfg, r.perturbation_type, str(r.perturbation_level)),
            "N": int(r.n_subjects), "Mean difference (pp)": _pp(r.mean_diff), "95% CI": _ci(r.ci95_low, r.ci95_high),
            "Effect size d_z": _dz(r.effect_size_dz), "p (sign-flip)": _p(r.p_signflip),
            "Adjusted p (BH)": _p(r.q_bh), "_o": order[r.model],
        })
    t14d = pd.DataFrame(rows)
    t14d = t14d.sort_values(["Comparison", "_o", "Condition"], kind="stable").drop(columns="_o").reset_index(drop=True)
    emit("TABLE_14D", "paired_statistical_analysis",
         "Paired statistical analysis (Macro-F1; subject is the unit).",
         t14d, list(ALL_MODELS), list(CONDITIONS), ["channel_dropout", "noise"],
         "paired sign-flip permutation, BH-FDR, d_z, bootstrap 95% CI")
    return records


# ============================================================================
# 17. FIGURES
# ============================================================================

def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "lines.linewidth": 1.1, "lines.markersize": 3.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
    })


def _save_pub(fig, stem: str) -> list[Path]:
    paths = []
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        p = FIG_PUB / f"{stem}.{ext}"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", **kw)
        paths.append(p)
    plt.close(fig)
    return paths


def _save_diag(fig, stem: str) -> list[Path]:
    p = FIG_DIAG / f"{stem}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return [p]


def _ylim(df: pd.DataFrame, lo_col: str, hi_col: str, include_zero: bool = True) -> tuple[float, float]:
    lo, hi = float(df[lo_col].min()) * PUB_SCALE, float(df[hi_col].max()) * PUB_SCALE
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad


def _shared_legend(fig, axes) -> None:
    seen: dict[str, Any] = {}
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    order = [DISPLAY_NAME[m] for m in ALL_MODELS if DISPLAY_NAME[m] in seen]
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.legend([seen[l] for l in order], order, loc="lower center", ncol=5, frameon=False, handlelength=1.6)


def _curve_panels(df: pd.DataFrame, ptype: str, cfg: dict, tick_labels: dict, ylab: str, xlab: str, stem: str,
                  ycols: tuple[str, str, str], ylim: tuple[float, float]):
    ycol, lo, hi = ycols
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.4), sharey=True)
    for ri, branch in enumerate(("ML", "DL")):
        for ci, cond in enumerate(CONDITIONS):
            ax = axes[ri, ci]
            for model in (ML_MODELS if branch == "ML" else DL_MODELS):
                g = df[(df.model == model) & (df.condition == cond)].sort_values("level_index")
                if g.empty:
                    continue
                x = np.arange(len(g)) if ptype == "noise" else g.perturbation_level.astype(float).to_numpy()
                y = g[ycol].to_numpy() * PUB_SCALE
                err = np.vstack([y - g[lo].to_numpy() * PUB_SCALE, g[hi].to_numpy() * PUB_SCALE - y])
                ax.errorbar(x, y, yerr=err, color=MODEL_COLOR[model], marker=MODEL_MARKER[model], capsize=2,
                            elinewidth=0.6, label=DISPLAY_NAME[model])
            ax.axhline(0.0, color="0.4", lw=0.6, ls=":")
            ax.set_title(f"{branch} \u2014 {cond}")
            ax.set_ylim(*ylim)
            lv = df.sort_values("level_index").drop_duplicates("level_index")
            if ptype == "noise":
                ax.set_xticks(np.arange(len(lv)))
                ax.set_xticklabels([tick_labels[str(v)] for v in lv.perturbation_level], rotation=0)
            else:
                ax.set_xticks(lv.perturbation_level.astype(float).to_numpy())
                ax.set_xticklabels([tick_labels[str(v)] for v in lv.perturbation_level])
            if ri == 1:
                ax.set_xlabel(xlab)
            if ci == 0:
                ax.set_ylabel(ylab)
    _shared_legend(fig, axes)
    return fig


def make_publication_figures(stats_csv: Path, cfg: dict, drop_manifest_csv: Path, noise_manifest_csv: Path) -> list[dict]:
    if plt is None:
        raise RuntimeError(f"matplotlib unavailable: {MPL_ERROR}")
    _style()
    st = pd.read_csv(stats_csv)
    pri = st[st.metric == PRIMARY_METRIC]
    cvp = pri[pri.comparison == "clean_vs_perturbed"]
    dm, nm = pd.read_csv(drop_manifest_csv), pd.read_csv(noise_manifest_csv)
    recs = []

    def record(fid: str, name: str, paths: list[Path], models, conds, perts, statistics: str) -> None:
        recs.append({
            "figure_id": fid, "figure_name": name, "source_table": stats_csv.name, "source_file": str(stats_csv),
            "source_file_sha256": sha256_file(stats_csv), "models": "|".join(models), "conditions": "|".join(conds),
            "perturbations": "|".join(perts), "statistics": statistics, "generated_timestamp": now_utc(),
            "code_version_hash": code_sha256(), "output_files": "|".join(str(p) for p in paths),
        })

    drop = cvp[cvp.perturbation_type == "channel_dropout"]
    noise = cvp[cvp.perturbation_type == "noise"]
    both = pd.concat([drop, noise])
    ylim = _ylim(both, "ci95_low", "ci95_high")

    ticks_d = {str(float(r.dropout_percent)): f"{int(r.dropout_percent)}%\n({int(r.n_channels)} ch)" for r in dm.itertuples()}
    f = _curve_panels(drop.assign(perturbation_level=drop.perturbation_level.astype(str)), "channel_dropout", cfg,
                      {**{str(k): v for k, v in ticks_d.items()},
                       **{str(float(k)): v for k, v in ticks_d.items()},
                       **{f"{float(k):g}": v for k, v in ticks_d.items()}},
                      "ΔP Macro-F1 (percentage points)", "Channel dropout (% of 13 channels; channels dropped)",
                      "FIG14A", ("mean_diff", "ci95_low", "ci95_high"), ylim)
    p = _save_pub(f, "FIGURE_14A_channel_dropout_degradation")
    record("FIGURE_14A", "Performance degradation versus channel dropout percentage", p, ALL_MODELS, CONDITIONS,
           ["channel_dropout"], "mean ΔP over 25 subjects; bootstrap 95% CI; shared y-axis with 14B")

    ticks_n = {f"{float(r.noise_level):g}": level_label(cfg, "noise", f"{float(r.noise_level):g}") for r in nm.itertuples()}
    f = _curve_panels(noise.assign(perturbation_level=noise.perturbation_level.astype(str)), "noise", cfg, ticks_n,
                      "ΔP Macro-F1 (percentage points)", "Noise level (mild → severe)", "FIG14B",
                      ("mean_diff", "ci95_low", "ci95_high"), ylim)
    p = _save_pub(f, "FIGURE_14B_noise_degradation")
    record("FIGURE_14B", "Performance degradation versus noise level", p, ALL_MODELS, CONDITIONS, ["noise"],
           "mean ΔP over 25 subjects; bootstrap 95% CI; shared y-axis with 14A")

    worst_d = drop[drop.level_index == drop.level_index.max()]
    worst_n = noise[noise.level_index == noise.level_index.max()]
    ylim_c = _ylim(pd.concat([worst_d, worst_n]), "ci95_low", "ci95_high")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.3), sharey=True)
    for ax, df, title in ((axes[0], worst_d, f"Channel dropout {int(float(worst_d.perturbation_level.iloc[0]))}%"),
                          (axes[1], worst_n, "Noise: " + level_label(cfg, "noise", str(worst_n.perturbation_level.iloc[0])))):
        for k, cond in enumerate(CONDITIONS):
            off = (-0.14, 0.14)[k]
            for mi_, model in enumerate(ALL_MODELS):
                g = df[(df.model == model) & (df.condition == cond)]
                if g.empty:
                    continue
                y = float(g.mean_diff.iloc[0]) * PUB_SCALE
                ax.errorbar(mi_ + off, y, yerr=[[y - float(g.ci95_low.iloc[0]) * PUB_SCALE],
                                                [float(g.ci95_high.iloc[0]) * PUB_SCALE - y]],
                            fmt="o" if cond == "ME" else "s",
                            mfc=MODEL_COLOR[model] if cond == "ME" else "white",
                            mec=MODEL_COLOR[model], color=MODEL_COLOR[model], capsize=2, elinewidth=0.7)
        ax.axhline(0, color="0.4", lw=0.6, ls=":")
        ax.axvline(len(ML_MODELS) - 0.5, color="0.7", lw=0.6)
        ax.set_xticks(range(len(ALL_MODELS)))
        ax.set_xticklabels([DISPLAY_NAME[m] for m in ALL_MODELS], rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylim(*ylim_c)
    axes[0].set_ylabel("ΔP Macro-F1 (percentage points)")
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], marker="o", color="0.3", ls="", label="ME (filled)"),
                            Line2D([], [], marker="s", mfc="white", color="0.3", ls="", label="MI (open)")],
                   frameon=False, loc="upper right")
    fig.tight_layout()
    p = _save_pub(fig, "FIGURE_14C_model_robustness_comparison")
    record("FIGURE_14C", "Model robustness comparison at the most severe frozen perturbation level", p, ALL_MODELS,
           CONDITIONS, ["channel_dropout", "noise"], "mean ΔP over 25 subjects; bootstrap 95% CI")

    mm = pri[pri.comparison == "ME_vs_MI"]
    mm_d, mm_n = mm[mm.perturbation_type == "channel_dropout"], mm[mm.perturbation_type == "noise"]
    ylim_m = _ylim(mm, "ci95_low", "ci95_high")
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.4), sharey=True)
    for ri, branch in enumerate(("ML", "DL")):
        for ci, (df, ptype) in enumerate(((mm_d, "channel_dropout"), (mm_n, "noise"))):
            ax = axes[ri, ci]
            for model in (ML_MODELS if branch == "ML" else DL_MODELS):
                g = df[df.model == model].sort_values("level_index")
                if g.empty:
                    continue
                x = np.arange(len(g)) if ptype == "noise" else g.perturbation_level.astype(float).to_numpy()
                y = g.mean_diff.to_numpy() * PUB_SCALE
                ax.errorbar(x, y, yerr=np.vstack([y - g.ci95_low.to_numpy() * PUB_SCALE,
                                                  g.ci95_high.to_numpy() * PUB_SCALE - y]),
                            color=MODEL_COLOR[model], marker=MODEL_MARKER[model], capsize=2, elinewidth=0.6,
                            label=DISPLAY_NAME[model])
            ax.axhline(0, color="0.4", lw=0.6, ls=":")
            ax.set_ylim(*ylim_m)
            lv = df.sort_values("level_index").drop_duplicates("level_index")
            if ptype == "noise":
                ax.set_xticks(np.arange(len(lv)))
                ax.set_xticklabels([ticks_n[f"{float(v):g}"] for v in lv.perturbation_level])
                xl = "Noise level (mild → severe)"
            else:
                ax.set_xticks(lv.perturbation_level.astype(float).to_numpy())
                ax.set_xticklabels([f"{int(float(v))}%" for v in lv.perturbation_level])
                xl = "Channel dropout (%)"
            ax.set_title(f"{branch} \u2014 {'dropout' if ptype == 'channel_dropout' else 'noise'}")
            if ri == 1:
                ax.set_xlabel(xl)
            if ci == 0:
                ax.set_ylabel("ΔP(MI) \u2212 ΔP(ME)  (pp)")
    _shared_legend(fig, axes)
    p = _save_pub(fig, "FIGURE_14D_condition_specific_robustness")
    record("FIGURE_14D", "Condition-specific (ME vs MI) robustness comparison", p, ALL_MODELS, CONDITIONS,
           ["channel_dropout", "noise"], "paired difference of per-subject ΔP (MI−ME); bootstrap 95% CI")
    return recs


def make_diagnostic_figures(stats_csv: Path, agg_csv: Path, cfg: dict) -> list[dict]:
    if plt is None:
        raise RuntimeError(f"matplotlib unavailable: {MPL_ERROR}")
    _style()
    st = pd.read_csv(stats_csv)
    cvp = st[(st.metric == PRIMARY_METRIC) & (st.comparison == "clean_vs_perturbed")]
    agg = pd.read_csv(agg_csv)
    recs = []

    def record(fid, name, paths):
        recs.append({"figure_id": fid, "figure_name": name, "source_table": f"{stats_csv.name}|{agg_csv.name}",
                     "source_file": f"{stats_csv}|{agg_csv}",
                     "source_file_sha256": f"{sha256_file(stats_csv)}|{sha256_file(agg_csv)}",
                     "models": "|".join(ALL_MODELS), "conditions": "|".join(CONDITIONS),
                     "perturbations": "channel_dropout|noise", "statistics": "diagnostic (descriptive)",
                     "generated_timestamp": now_utc(), "code_version_hash": code_sha256(),
                     "output_files": "|".join(str(p) for p in paths)})

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.4), sharey=True)
    for ci, ptype in enumerate(("channel_dropout", "noise")):
        for ri, cond in enumerate(CONDITIONS):
            ax = axes[ri, ci]
            for model in ALL_MODELS:
                g = cvp[(cvp.perturbation_type == ptype) & (cvp.model == model) & (cvp.condition == cond)].sort_values("level_index")
                if g.empty:
                    continue
                y = np.r_[g.clean_mean.iloc[0], g.perturbed_mean.to_numpy()] * PUB_SCALE
                ax.plot(np.arange(len(y)), y, color=MODEL_COLOR[model], marker=MODEL_MARKER[model], label=DISPLAY_NAME[model])
            ax.axhline(CHANCE_LEVEL * PUB_SCALE, color="0.3", ls="--", lw=0.7)
            lv = g.sort_values("level_index")
            ax.set_xticks(np.arange(len(lv) + 1))
            ax.set_xticklabels(["clean"] + [level_label(cfg, ptype, str(v)) for v in lv.perturbation_level],
                               rotation=30, ha="right")
            ax.set_title(f"{cond} \u2014 {ptype}")
            if ci == 0:
                ax.set_ylabel("Macro-F1 (%)")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=6)
    fig.tight_layout()
    record("DIAG_D1", "Absolute Macro-F1 versus perturbation (clean reference + chance 1/7)",
           _save_diag(fig, "DIAG_absolute_performance_curves"))

    for fid, ptype in (("DIAG_D2", "channel_dropout"), ("DIAG_D3", "noise")):
        a = agg[agg.perturbation_type == ptype]
        top = a[a.level_index == a.level_index.max()]
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.6), sharey=True)
        vmax = float(np.nanmax(np.abs(top[f"delta_{PRIMARY_METRIC}"]))) * PUB_SCALE or 1.0
        for ax, cond in zip(axes, CONDITIONS):
            piv = top[top.condition == cond].pivot(index="subject", columns="model",
                                                  values=f"delta_{PRIMARY_METRIC}").reindex(columns=list(ALL_MODELS))
            im = ax.imshow(piv.to_numpy() * PUB_SCALE, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(ALL_MODELS)))
            ax.set_xticklabels([DISPLAY_NAME[m] for m in ALL_MODELS], rotation=60, ha="right", fontsize=6)
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels([f"S{int(s):02d}" for s in piv.index], fontsize=6)
            ax.set_title(f"{cond} \u2014 per-subject ΔP (pp), most severe {ptype} level")
        fig.colorbar(im, ax=axes, shrink=0.8, label="ΔP Macro-F1 (pp)")
        record(fid, f"Per-subject ΔP heat-map ({ptype})", _save_diag(fig, f"DIAG_subject_delta_heatmap_{ptype}"))
    return recs


# ============================================================================
# 18. MODEL REGISTRY
# ============================================================================

REGISTRY_CSV = MANIFEST_ROOT / "BLOCK14_model_manifest.csv"


def build_registry(cfg: dict, b13) -> pd.DataFrame:
    rows = []
    regime = cfg["evaluation_regimes"][0]
    for s in SUBJECTS:
        for c in CONDITIONS:
            for m in ML_MODELS:
                p = ml_model_path(s, c, m)
                if not p.exists():
                    raise FileNotFoundError(f"Frozen BLOCK06 LOSO model missing: {p}")
                rows.append({"regime": regime, "subject": s, "condition": c, "model": m, "branch": "ML",
                             "path": str(p), "size_bytes": p.stat().st_size, "sha256": cached_sha256(p),
                             "complete_json_checkpoint_sha256": ""})
            for m in DL_MODELS:
                ckpt, comp = b13._discover_exact_block07_checkpoint(s, c, m)
                declared = ""
                try:
                    declared = str(json.loads(Path(comp).read_text(encoding="utf-8")).get("checkpoint_sha256") or "")
                except Exception:
                    pass
                sha = cached_sha256(Path(ckpt))
                if declared and declared != sha:
                    raise RuntimeError(f"BLOCK07 checkpoint hash != its complete.json: {ckpt}")
                rows.append({"regime": regime, "subject": s, "condition": c, "model": m, "branch": "DL",
                             "path": str(ckpt), "size_bytes": Path(ckpt).stat().st_size, "sha256": sha,
                             "complete_json_checkpoint_sha256": declared})
        log(f"registry: S{s:02d} hashed")
    save_hash_cache()
    df = pd.DataFrame(rows)
    if df.duplicated(["regime", "subject", "condition", "model"]).any():
        raise RuntimeError("duplicate model registry keys")
    atomic_csv(REGISTRY_CSV, df)
    return df


def load_registry() -> dict:
    if not REGISTRY_CSV.exists():
        raise RuntimeError(f"{REGISTRY_CSV} missing: run --preflight first")
    df = pd.read_csv(REGISTRY_CSV)
    return {(r.regime, r.model, int(r.subject), r.condition): r.sha256 for r in df.itertuples(index=False)}


# ============================================================================
# 19. FROZEN PREDICTION AUTHORITY (EXACT BLOCK06 / BLOCK07 CSVs)
# ============================================================================

# ---------------------------------------------------------------------------
# window_id canonicalization.  BLOCK04 is the canonical form.
#   BLOCK04 : S01_ME_1_SetA_W500_O50_Window0001   (1-based, 4-digit)
#   BLOCK06 : S01_ME_1_SetA_W001                  (1-based, 3-digit)
#   BLOCK07 : S01_ME_1_SetA_windows_W00           (0-based, 2-digit)
# The prefix is identical. Only the suffix differs. Anything unmapped raises.
# ---------------------------------------------------------------------------
import re as _re

_B04_RE = _re.compile(r"^(?P<p>S\d{2}_[A-Z]{2}_\d+_Set[A-Z])_W500_O50_Window(?P<n>\d{4})$")
_B06_RE = _re.compile(r"^(?P<p>S\d{2}_[A-Z]{2}_\d+_Set[A-Z])_W(?P<n>\d{3})$")
_B07_RE = _re.compile(r"^(?P<p>S\d{2}_[A-Z]{2}_\d+_Set[A-Z])_windows_W(?P<n>\d{2})$")


def _canonicalize_window_id(value: str) -> str:
    s = str(value).strip()
    m = _B04_RE.match(s)
    if m:
        return s
    m = _B06_RE.match(s)
    if m:
        n = int(m.group("n"))
        return f"{m.group('p')}_W500_O50_Window{n:04d}"
    m = _B07_RE.match(s)
    if m:
        n = int(m.group("n")) + 1  # BLOCK07 is 0-based
        return f"{m.group('p')}_W500_O50_Window{n:04d}"
    raise ValueError(
        f"Unrecognized window_id format: {value!r}. Expected one of: "
        f"BLOCK04 ...W500_O50_WindowNNNN, BLOCK06 ...WNNN, BLOCK07 ...windows_WNN."
    )


def _canonicalize_window_id_series(series: "pd.Series") -> "pd.Series":
    return series.map(_canonicalize_window_id)


# Verified BLOCK07 model-directory names (explicit; no naming transformation inferred).
DL_MODEL_DIRS = {
    "EEGNet": "EEGNet",
    "ShallowConvNet": "ShallowConvNet",
    "DeepConvNet": "DeepConvNet",
    "CNN-BiLSTM": "CNN_BiLSTM",
    "CNN-BiLSTM-Attention": "CNN_BiLSTM_Attention",
}
CLEAN_AUDIT_ABS_TOL = 1e-12  # float tolerance for Macro-F1 only; prediction identity is exact


def _exact_upstream_prediction_path(subject: int, condition: str, model: str) -> Path:
    """The single authoritative frozen prediction CSV for a (subject, condition, model)."""
    if model in ML_MODELS:
        return (BLOCK06_ROOT / "06B_ML_LOSO" / "folds" / f"Subject_{int(subject):02d}"
                / f"{condition}_{model}_predictions.csv")
    if model not in DL_MODEL_DIRS:
        raise KeyError(f"no verified BLOCK07 model directory for {model!r}")
    return (BLOCK07_ROOT / "07B_DL_LOSO" / "folds" / f"Subject_{int(subject):02d}"
            / str(condition) / DL_MODEL_DIRS[model] / "predictions.csv")


# BLOCK06 stores XGBoost/RandomForest in its `model` column; the project canonical
# names are XGB/RF.  BLOCK07 stores the canonical names.  This map is used ONLY to
# interpret the upstream CSV; it never rewrites the file.
_UPSTREAM_MODEL_ALIASES = {
    "XGBoost": "XGB",
    "RandomForest": "RF",
    "Random_Forest": "RF",
}


def _load_exact_upstream_predictions(subject: int, condition: str, model: str,
                                     expected_window_ids: list[str] | None = None
                                     ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, Path]:
    """Load + validate one frozen BLOCK06/BLOCK07 LOSO prediction CSV.

    Returns (y_true, y_pred, df_indexed_by_canonical_window_id, path).
    df is indexed by canonical BLOCK04 window_id (never by CSV row order).
    """
    p = _exact_upstream_prediction_path(subject, condition, model)
    if not p.exists():
        raise FileNotFoundError(f"exact frozen upstream prediction CSV missing: {p}")
    df = pd.read_csv(p)
    required = {"gesture_true", "gesture_pred", "window_id", "condition"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{p}: missing exact-prediction columns {missing}")
    if "model" in df.columns:
        uniq = sorted(df["model"].astype(str).unique().tolist())
        canonical_uniq = sorted({_UPSTREAM_MODEL_ALIASES.get(u, u) for u in uniq})
        if len(canonical_uniq) != 1 or canonical_uniq[0] != model:
            raise RuntimeError(f"{p}: model column is {uniq!r}, expected {model!r}")
    if df["window_id"].isna().any():
        raise RuntimeError(f"{p}: null window_id values")
    if df["window_id"].duplicated().any():
        raise RuntimeError(f"{p}: duplicate window_id values")

    cond_ok = df["condition"].astype(str).str.strip().str.upper()
    if not cond_ok.eq(condition.upper()).all():
        raise RuntimeError(f"{p}: condition column is not uniformly {condition}")

    for subj_col in ("subject", "subject_id", "held_out_subject"):
        if subj_col in df.columns:
            try:
                vals = pd.to_numeric(
                    df[subj_col].astype(str).str.replace(r"^S(UBJECT)?[_\- ]?", "", regex=True),
                    errors="raise",
                ).to_numpy(dtype=int)
            except Exception as exc:
                raise RuntimeError(f"{p}: cannot parse {subj_col}: {exc}") from exc
            if not np.all(vals == int(subject)):
                raise RuntimeError(f"{p}: column {subj_col} is not uniformly held-out subject {subject}")

    yt = pd.to_numeric(df["gesture_true"], errors="raise").to_numpy(dtype=int)
    yp = pd.to_numeric(df["gesture_pred"], errors="raise").to_numpy(dtype=int)
    if not set(np.unique(yt)).issubset(set(GESTURES)) or not set(np.unique(yp)).issubset(set(GESTURES)):
        raise RuntimeError(f"{p}: labels outside 1..7")

    canon = _canonicalize_window_id_series(df["window_id"])
    if canon.duplicated().any():
        raise RuntimeError(f"{p}: duplicate window_id after canonicalization")

    if expected_window_ids is not None:
        exp = set(map(str, expected_window_ids))
        got = set(canon.tolist())
        if got != exp:
            only_exp = sorted(exp - got)[:5]
            only_got = sorted(got - exp)[:5]
            raise RuntimeError(
                f"{p}: window_id set differs from BLOCK04 after canonicalization "
                f"(missing_from_csv={len(exp - got)}, extra_in_csv={len(got - exp)}); "
                f"examples missing={only_exp}, extra={only_got}"
            )
    else:
        exp_n = N_GESTURES * len(SETS) * WINDOWS_PER_RECORDING
        if len(canon) != exp_n:
            raise RuntimeError(f"{p}: {len(canon)} windows; expected {exp_n}")

    out = pd.DataFrame({
        "window_id_canonical": canon.to_numpy(),
        "gesture_true": yt,
        "gesture_pred": yp,
    }).set_index("window_id_canonical", drop=False)
    return yt, yp, out, p


def _align_exact_predictions(upstream_df: pd.DataFrame,
                             canonical_ids: list[str]) -> pd.DataFrame:
    got = set(upstream_df.index.astype(str))
    exp = set(map(str, canonical_ids))
    if got != exp:
        raise RuntimeError(
            f"upstream predictions do not cover BLOCK04 exactly: "
            f"missing={len(exp - got)}, extra={len(got - exp)}"
        )
    return upstream_df.loc[list(map(str, canonical_ids))]


def check_upstream_clean_reference(cfg: dict, manifest: pd.DataFrame | None = None) -> tuple[bool, str]:
    """Preflight: enumerate and validate all 500 exact frozen prediction CSVs.

    ML  -> 25 subjects x 2 conditions x 5 models = 250 files
    DL  -> 25 subjects x 2 conditions x 5 models = 250 files
    Total = 500 files.
    """
    u = cfg["upstream"]["upstream_clean_reference"]
    if not u.get("enabled"):
        return True, "WAIVED: " + str(u.get("waiver_justification"))

    notes: list[str] = []
    n_ok = 0
    n_expect = 0
    for branch, models in (("ML", ML_MODELS), ("DL", DL_MODELS)):
        for model in models:
            for cond in CONDITIONS:
                for subj in SUBJECTS:
                    n_expect += 1
                    exp_ids = None
                    if manifest is not None:
                        exp_ids = block_meta(manifest, subj, cond)["window_id"].astype(str).tolist()
                    try:
                        _load_exact_upstream_predictions(subj, cond, model, exp_ids)
                        n_ok += 1
                    except Exception as exc:
                        notes.append(f"[{branch}] S{subj:02d}/{cond}/{model}: "
                                     f"{type(exc).__name__}: {str(exc)[:200]}")
    if n_ok == n_expect and not notes:
        return True, (f"{n_ok}/{n_expect} exact frozen prediction CSVs validated "
                      f"(250 ML + 250 DL), each canonicalized and coverage-matched "
                      f"against BLOCK04")
    return False, (f"{n_ok}/{n_expect} exact frozen prediction CSVs valid; "
                   f"{len(notes)} failure(s); examples: {notes[:3]}")


def verify_clean_against_upstream(ctx: RunCtx, cfg: dict, metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Exact audit of every BLOCK14 clean unit against the frozen BLOCK06/BLOCK07
    LOSO prediction CSV.  Alignment is by canonical BLOCK04 window_id.  Prediction
    identity is exact (np.array_equal, no tolerance).  Macro-F1 is recomputed from
    the frozen arrays; BLOCK14's stored value must agree within CLEAN_AUDIT_ABS_TOL.
    """
    cols = ["subject", "condition", "model", "reference", "reference_csv",
            "reference_sha256", "n_reference_windows", "n_block14_windows",
            "prediction_identical", "window_id_mismatch_count",
            "y_true_mismatch_count", "y_pred_mismatch_count",
            "reference_recomputed_macro_f1", "block14_recomputed_macro_f1",
            "block14_stored_macro_f1", "abs_diff", "stored_vs_recomputed_abs_diff",
            "status", "reason"]
    rows = []
    clean = metrics_df[metrics_df.perturbation_type == "clean"]
    for _, r in clean.iterrows():
        subject = int(r["subject"])
        condition = str(r["condition"]).upper()
        model = str(r["model"])
        ref_kind = ("exact_frozen_BLOCK06_prediction_csv" if model in ML_MODELS
                    else "exact_frozen_BLOCK07_prediction_csv")
        rec = {c: None for c in cols}
        rec.update({
            "subject": subject, "condition": condition, "model": model,
            "reference": ref_kind,
            "reference_csv": str(_exact_upstream_prediction_path(subject, condition, model)),
            "prediction_identical": False,
        })
        try:
            meta = ctx.meta(subject, condition)
            ids14 = meta["window_id"].astype(str).tolist()
            d = read_unit(Path(r["unit_file"]))
            if str(d["window_order_sha256"]) != window_order_sha(meta):
                raise RuntimeError("unit window_order_sha256 differs from BLOCK04 order")
            yt14 = np.asarray(d["y_true"], dtype=int)
            yp14 = np.asarray(d["y_pred"], dtype=int)
            if len(yt14) != len(ids14):
                raise RuntimeError(f"unit has {len(yt14)} rows, BLOCK04 has {len(ids14)} windows")
            rec["n_block14_windows"] = len(ids14)
            ref_path = _exact_upstream_prediction_path(subject, condition, model)
            rec["reference_sha256"] = sha256_file(ref_path) if ref_path.exists() else None
            _, _, updf, _ = _load_exact_upstream_predictions(subject, condition, model,
                                                             expected_window_ids=ids14)
            rec["n_reference_windows"] = len(updf)
            got = set(updf.index.astype(str))
            exp = set(ids14)
            rec["window_id_mismatch_count"] = len(got ^ exp)
            if rec["window_id_mismatch_count"]:
                rec.update({
                    "status": "FAIL",
                    "reason": (f"window_id sets differ after canonicalization "
                               f"(only-in-BLOCK14={len(exp - got)}, "
                               f"only-in-reference={len(got - exp)})"),
                })
                rows.append(rec)
                continue
            aligned = _align_exact_predictions(updf, ids14)
            yt_ref = aligned["gesture_true"].to_numpy(dtype=int)
            yp_ref = aligned["gesture_pred"].to_numpy(dtype=int)
            n_t = int(np.count_nonzero(yt_ref != yt14))
            n_p = int(np.count_nonzero(yp_ref != yp14))
            rec["y_true_mismatch_count"] = n_t
            rec["y_pred_mismatch_count"] = n_p
            ident = (n_t == 0 and n_p == 0)
            rec["prediction_identical"] = bool(ident)
            f_ref = float(f1_score(yt_ref, yp_ref, labels=list(GESTURES),
                                   average="macro", zero_division=0))
            f_b14 = float(f1_score(yt14, yp14, labels=list(GESTURES),
                                   average="macro", zero_division=0))
            f_stored = float(r["Macro-F1"])
            rec.update({
                "reference_recomputed_macro_f1": f_ref,
                "block14_recomputed_macro_f1": f_b14,
                "block14_stored_macro_f1": f_stored,
                "abs_diff": abs(f_ref - f_b14),
                "stored_vs_recomputed_abs_diff": abs(f_stored - f_b14),
            })
            if not ident:
                idx = int(np.argmax((yt_ref != yt14) | (yp_ref != yp14)))
                rec.update({
                    "status": "FAIL",
                    "reason": (f"prediction mismatch: y_true differs at {n_t}/{len(ids14)} "
                               f"windows, y_pred differs at {n_p}/{len(ids14)} windows; "
                               f"first mismatch window_id={ids14[idx]}"),
                })
            elif abs(f_ref - f_b14) > CLEAN_AUDIT_ABS_TOL:
                rec.update({"status": "FAIL",
                            "reason": "predictions identical but recomputed Macro-F1 differs beyond 1e-12"})
            elif abs(f_stored - f_b14) > CLEAN_AUDIT_ABS_TOL:
                rec.update({"status": "FAIL",
                            "reason": "stored BLOCK14 Macro-F1 differs from its recomputation beyond 1e-12"})
            else:
                rec.update({"status": "PASS",
                            "reason": "window_id-aligned y_true/y_pred identical; Macro-F1 agrees"})
        except Exception as exc:
            rec.update({"status": "FAIL", "reason": f"{type(exc).__name__}: {exc}"})
        rows.append(rec)
    return pd.DataFrame(rows, columns=cols)


# ============================================================================
# 19b. BLOCK13 FROZEN-OUTPUT CROSS-CHECK (identity only)
# ============================================================================

def crosscheck_block13_frozen(cfg: dict, b13) -> tuple[bool, list[str], dict]:
    notes, info, ok = [], {}, True
    root = Path(cfg["upstream"]["block13_frozen_outputs"])
    if not root.exists():
        return False, [f"frozen BLOCK13 output directory not found: {root}"], info
    st = sorted(root.rglob("BLOCK13_status.json"))
    if len(st) != 1:
        return False, [f"expected exactly one BLOCK13_status.json under {root}; found {len(st)}"], info
    status = json.loads(st[0].read_text(encoding="utf-8"))
    info["status_file"], info["status_file_sha256"] = str(st[0]), sha256_file(st[0])
    if str(status.get("status", "")).upper() != "PASS":
        ok = False
        notes.append(f"BLOCK13 status is {status.get('status')!r}, not PASS")
    pv = status.get("protocol_version")
    if pv is not None and pv != b13.PROTOCOL_VERSION:
        ok = False
        notes.append(f"BLOCK13 frozen protocol_version {pv!r} != adapter {b13.PROTOCOL_VERSION!r}")
    cfgs = sorted(root.rglob("BLOCK13_config.json"))
    if cfgs:
        c13 = json.loads(cfgs[0].read_text(encoding="utf-8"))
        info["config_file_sha256"] = sha256_file(cfgs[0])
        for key, ours in (("subjects", list(SUBJECTS)), ("conditions", list(CONDITIONS)),
                          ("gestures", list(GESTURES)), ("dl_models", list(DL_MODELS))):
            if key in c13 and list(c13[key]) != ours:
                ok = False
                notes.append(f"BLOCK13 config {key}={c13[key]} differs from BLOCK14 {ours}")
    else:
        notes.append("BLOCK13_config.json not found in frozen outputs (identity from tables only)")
    tabs = sorted(root.rglob("DL_IG_channel_summary.csv"))
    if tabs:
        t = pd.read_csv(tabs[0], usecols=["subject_id", "condition", "model", "gesture", "channel"])
        checks = {"subjects": sorted(t.subject_id.unique().tolist()) == list(SUBJECTS),
                  "conditions": sorted(t.condition.unique().tolist()) == sorted(CONDITIONS),
                  "gestures_numeric_1_7": sorted(t.gesture.unique().tolist()) == list(GESTURES),
                  "models": sorted(t.model.unique().tolist()) == sorted(DL_MODELS),
                  "channels_CH01_CH13": sorted(t.channel.unique().tolist()) == sorted(CHANNEL_NAMES)}
        info["identity_checks"] = checks
        for k, v in checks.items():
            if not v:
                ok = False
                notes.append(f"frozen BLOCK13 identity mismatch: {k}")
        info["identity_table"] = str(tabs[0])
    else:
        ok = False
        notes.append("DL_IG_channel_summary.csv not found in frozen BLOCK13 outputs")
    return ok, notes, info


# ============================================================================
# 20. PREFLIGHT
# ============================================================================

def discover_block_dirs() -> dict[int, str]:
    out = {}
    for n in range(0, 15):
        hits = sorted(p.name for p in WINDOWS_ROOT.glob(f"{n:02d}_*") if p.is_dir()) if WINDOWS_ROOT.exists() else []
        out[n] = " | ".join(hits) if hits else "NOT FOUND"
    return out


def expected_counts(cfg: dict | None) -> dict:
    if cfg is None or config_problems(cfg):
        return {"specs": None, "units": None}
    specs = build_specs(cfg)
    n_units = len(cfg["evaluation_regimes"]) * len(SUBJECTS) * len(CONDITIONS) * len(ALL_MODELS) * len(specs)
    return {"specs": len(specs), "units": n_units,
            "clean": 1, "dropout_specs": sum(s.ptype == "channel_dropout" for s in specs),
            "noise_specs": sum(s.ptype == "noise" for s in specs)}


def run_preflight(refreeze: bool = False) -> bool:
    prepare_dirs()
    checks: list[dict] = []
    dep: list[dict] = []

    def chk(cid: str, desc: str, fn: Callable[[], tuple[bool, str]], critical: bool = True) -> bool:
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
            log(traceback.format_exc())
        checks.append({"id": cid, "description": desc, "critical": critical,
                       "status": "PASS" if ok else ("FAIL" if critical else "WARN"),
                       "detail": detail})
        log(f"PREFLIGHT {cid} {'PASS' if ok else ('FAIL' if critical else 'WARN')}: {desc} -- {str(detail)[:300]}")
        return ok

    state: dict[str, Any] = {"cfg": None, "b13": None, "manifest": None, "ctx": None, "ex": None}
    unresolved: list[str] = []

    def c_cfg():
        cfg = read_config_raw()
        probs = config_problems(cfg)
        state["cfg_raw"] = cfg
        if probs:
            unresolved.extend(probs)
            return False, f"{len(probs)} unresolved item(s)"
        state["cfg"] = cfg
        return True, f"config valid; canonical sha256={canonical_config_sha(cfg)}"
    have_cfg = chk("P01", "BLOCK14_config.json present, FROZEN, complete", c_cfg)

    def c_adapter():
        b13 = import_block13(state.get("cfg_raw"))
        state["b13"] = b13
        return True, f"{_B13_SOURCE} sha256={sha256_file(_B13_SOURCE)}"
    have_b13 = chk("P02", "BLOCK13 compatibility adapter importable (read-only)", c_adapter)

    def c_b04():
        df, info = load_block04_manifest(state["b13"])
        state["manifest"] = df
        return True, f"{len(df)} windows; manifest sha256={info.get('manifest_sha256', '')}"
    have_b04 = have_b13 and chk("P03", "BLOCK04 manifest + window arrays verified", c_b04)

    def c_b05():
        lk = state["b13"].build_feature_lookup(state["manifest"])
        state["lookup"] = lk
        fm = BLOCK05_ROOT / "manifests" / "feature_manifest.csv"
        return True, f"{len(lk)} windows joined; feature_manifest sha256={sha256_file(fm)}"
    have_b05 = have_b04 and chk("P04", "BLOCK05 feature manifest joins BLOCK04 one-to-one", c_b05)

    def c_extractor():
        ex = load_extractor(state["cfg"])
        state["ex"] = ex
        meta = block_meta(state["manifest"], 1, "ME")
        probe = meta[(meta.gesture_code == 1) & (meta["set"] == "SetA")].reset_index(drop=True)
        raw = load_raw_block(state["b13"], probe)
        Xf = extract_features_batch(ex, raw, "preflight clean probe S01 ME g1 SetA")
        sub = state["lookup"].set_index("window_id").loc[probe.window_id].reset_index()
        Xs, _ = state["b13"].materialize_feature_rows(sub)
        up = state["cfg"]["upstream"]
        ok = bool(np.allclose(Xf, Xs, rtol=float(up["feature_reproduction_rtol"]),
                              atol=float(up["feature_reproduction_atol"])))
        return ok, f"extractor {ex.source}; max|recomputed-stored|={float(np.max(np.abs(Xf - Xs))):.3e}"
    have_ex = have_b05 and have_cfg and chk("P05", "BLOCK05 extractor reproduces STORED clean features", c_extractor)

    def c_perturb_probe():
        cfg, b13 = state["cfg"], state["b13"]
        meta = block_meta(state["manifest"], 1, "ME")
        raw = load_raw_block(b13, meta)
        rsha = sha256_array(raw)
        lines = []
        first = meta[(meta.gesture_code == 1) & (meta["set"] == "SetA")].index.to_numpy()
        for sp in build_specs(cfg):
            Xp, info = build_perturbed(raw, meta, cfg, sp, rsha)
            bad = [k for k, v in info["audit"].items() if not v]
            if bad:
                raise RuntimeError(f"perturbation audit failed for {sp.tag}: {bad}")
            Xf = extract_features_batch(state["ex"], Xp[first], f"probe {sp.tag}")
            lines.append(f"{sp.tag}/pat{sp.pattern}/rea{sp.realization}:finite")
            del Xp
        return True, f"{len(lines)} specs: perturbation audit PASS and BLOCK05 features finite"
    if have_ex:
        chk("P06", "All frozen perturbations pass audit and yield finite BLOCK05 features", c_perturb_probe)
    else:
        checks.append({"id": "P06", "description": "perturbation audit", "critical": True,
                       "status": "FAIL", "detail": "skipped: prerequisites failed"})

    def c_models():
        reg = build_registry(state["cfg"], state["b13"])
        nm, nd = int((reg.branch == "ML").sum()), int((reg.branch == "DL").sum())
        exp_m = len(SUBJECTS) * len(CONDITIONS) * len(ML_MODELS)
        exp_d = len(SUBJECTS) * len(CONDITIONS) * len(DL_MODELS)
        if nm != exp_m or nd != exp_d:
            return False, f"registry counts ML={nm} DL={nd}; expected {exp_m}/{exp_d}"
        for m in ML_MODELS:
            load_ml_model(1, "ME", m)
        return True, f"{nm} BLOCK06 LOSO + {nd} BLOCK07 LOSO checkpoints located and hashed"
    if have_b13 and have_cfg:
        chk("P07", "Frozen model/checkpoint inventory + ML deserialisation probe", c_models)
    else:
        checks.append({"id": "P07", "description": "model inventory", "critical": True,
                       "status": "FAIL", "detail": "skipped: prerequisites failed"})

    def c_torch():
        if torch is None:
            return False, f"PyTorch unavailable: {TORCH_ERROR}"
        torch.set_num_threads(1)
        device = configure_canonical_dl_inference()
        return (
            torch.get_num_threads() == 1
            and device.type == "cuda"
            and torch.backends.cudnn.benchmark is False,
            f"torch {torch.__version__}; threads={torch.get_num_threads()}; "
            f"device={device}; deterministic_algorithms=True; "
            f"cudnn_benchmark={torch.backends.cudnn.benchmark}",
        )
    chk("P08", "Canonical BLOCK07 DL CUDA/deterministic inference runtime", c_torch)

    def c_frozen13():
        ok, notes, info = crosscheck_block13_frozen(state["cfg"], state["b13"])
        state["b13_info"] = info
        return ok, "; ".join(notes) if notes else f"identity PASS ({info.get('identity_checks')})"
    if have_b13 and have_cfg:
        chk("P09", "Frozen BLOCK13 output identity cross-check", c_frozen13)
    else:
        checks.append({"id": "P09", "description": "frozen BLOCK13 cross-check", "critical": True,
                       "status": "FAIL", "detail": "skipped: prerequisites failed"})

    if have_cfg:
        chk("P10", "Upstream clean-reference cross-check (exact frozen prediction CSVs)",
            lambda: check_upstream_clean_reference(state["cfg"], state.get("manifest")))
    else:
        checks.append({"id": "P10", "description": "upstream clean reference", "critical": True,
                       "status": "FAIL", "detail": "skipped: no valid config"})

    def c_dups():
        extra = [p.name for p in WINDOWS_ROOT.glob("14_ROBUSTNESS*") if p.is_dir() and p.name != "14_ROBUSTNESS"]
        scr = [p.name for d in (WINDOWS_ROOT / "scripts", WINDOWS_ROOT) if d.exists()
               for p in d.glob("BLOCK14_ROBUSTNESS*.py") if p.name != "BLOCK14_ROBUSTNESS.py"]
        return (not extra and not scr), f"duplicate result dirs={extra}; duplicate code files={scr}"
    chk("P11", "No duplicate BLOCK14 result directories / code files", c_dups, critical=False)

    checks.append({"id": "P12", "description": "Subject-dependent robustness", "critical": False, "status": "WARN",
                   "detail": "NOT EVALUATED: SD split definition / BLOCK06A/07A paths not in the verified contract."})

    passed = all(c["status"] == "PASS" for c in checks if c["critical"])
    counts = expected_counts(state.get("cfg"))
    ex_src = state["ex"].source if state.get("ex") else None

    bd = discover_block_dirs()
    b04m = BLOCK04_ROOT / "final_window_manifest.csv"
    b05m = BLOCK05_ROOT / "manifests" / "feature_manifest.csv"

    def ver(flag, extra=""):
        if flag is None:
            return "NOT FOUND"
        return ("VERIFIED" if flag else "NOT VERIFIED") + (f" ({extra})" if extra else "")

    def dir_ver(found: bool, extra: str) -> str:
        return f"directory discovered; {extra}" if found else "NOT FOUND"
    dep = [
        {"BLOCK": "BLOCK00 protocol", "REAL SOURCE PATH": bd[0], "REQUIRED BY BLOCK14": "provenance only",
         "FORMAT": "dir", "KEY COLUMNS/KEYS": "n/a",
         "VERIFIED?": dir_ver(bd[0] != "NOT FOUND", "contents not parsed")},
        {"BLOCK": "BLOCK01-02 matching/QA", "REAL SOURCE PATH": f"{bd[1]} ; {bd[2]}",
         "REQUIRED BY BLOCK14": "provenance only", "FORMAT": "dir", "KEY COLUMNS/KEYS": "n/a",
         "VERIFIED?": dir_ver(bd[1] != "NOT FOUND" and bd[2] != "NOT FOUND", "contents not parsed")},
        {"BLOCK": "BLOCK03 preprocessing", "REAL SOURCE PATH": bd[3],
         "REQUIRED BY BLOCK14": "indirect", "FORMAT": "dir", "KEY COLUMNS/KEYS": "n/a",
         "VERIFIED?": dir_ver(bd[3] != "NOT FOUND", "continuous data not used")},
        {"BLOCK": "BLOCK04 windowing", "REAL SOURCE PATH": str(b04m),
         "REQUIRED BY BLOCK14": "YES", "FORMAT": "CSV + NPZ",
         "KEY COLUMNS/KEYS": "window_id,subject,condition,gesture_code,set,source_recording,window_array_file,window_index",
         "VERIFIED?": ver(have_b04)},
        {"BLOCK": "BLOCK05 features", "REAL SOURCE PATH": f"{b05m} ; extractor={ex_src}",
         "REQUIRED BY BLOCK14": "YES", "FORMAT": "CSV + NPZ + py",
         "KEY COLUMNS/KEYS": "subject_id,condition,gesture_code,set,feature_file,feature_row",
         "VERIFIED?": ver(have_b05 and have_ex, "lookup join + extractor reproduction")},
        {"BLOCK": "BLOCK06A ML SD", "REAL SOURCE PATH": "UNRESOLVED",
         "REQUIRED BY BLOCK14": "NO", "FORMAT": "n/a", "KEY COLUMNS/KEYS": "n/a",
         "VERIFIED?": "NOT VERIFIED (not in verified contract)"},
        {"BLOCK": "BLOCK06B ML LOSO",
         "REAL SOURCE PATH": str(BLOCK06_ROOT / "models" / "LOSO" / "Fold_S{SS}_{COND}_{MODEL}.joblib"),
         "REQUIRED BY BLOCK14": "YES (250 frozen models)", "FORMAT": "joblib",
         "KEY COLUMNS/KEYS": "classes_==[1..7]; predict/predict_proba",
         "VERIFIED?": ver(any(c["id"] == "P07" and c["status"] == "PASS" for c in checks))},
        {"BLOCK": "BLOCK07A DL SD", "REAL SOURCE PATH": "UNRESOLVED",
         "REQUIRED BY BLOCK14": "NO", "FORMAT": "n/a", "KEY COLUMNS/KEYS": "n/a",
         "VERIFIED?": "NOT VERIFIED (not in verified contract)"},
        {"BLOCK": "BLOCK07B DL LOSO",
         "REAL SOURCE PATH": str(BLOCK07_ROOT / "07B_DL_LOSO" / "folds" / "**" / "best_model.pt"),
         "REQUIRED BY BLOCK14": "YES (250 checkpoints + complete.json)",
         "FORMAT": "torch pt + json", "KEY COLUMNS/KEYS": "state_dict, normalizer mean/std, checkpoint_sha256",
         "VERIFIED?": ver(any(c["id"] == "P07" and c["status"] == "PASS" for c in checks))},
        {"BLOCK": "BLOCK08-12", "REAL SOURCE PATH": " ; ".join(bd[n] for n in range(8, 13)),
         "REQUIRED BY BLOCK14": "NO", "FORMAT": "dir", "KEY COLUMNS/KEYS": "n/a",
         "VERIFIED?": "provenance only"},
        {"BLOCK": "BLOCK13 XAI (adapter)", "REAL SOURCE PATH": str(_B13_SOURCE) if _B13_SOURCE else "NOT FOUND",
         "REQUIRED BY BLOCK14": "YES (read-only)", "FORMAT": "py",
         "KEY COLUMNS/KEYS": "verify_block04, resolve_window_array, build_feature_lookup, load_block07_loso_model",
         "VERIFIED?": ver(have_b13)},
        {"BLOCK": "BLOCK13 frozen outputs",
         "REAL SOURCE PATH": str(cfg_frozen(state)), "REQUIRED BY BLOCK14": "identity/provenance only",
         "FORMAT": "CSV/JSON", "KEY COLUMNS/KEYS": "BLOCK13_status.json, DL_IG_channel_summary.csv",
         "VERIFIED?": ver(any(c["id"] == "P09" and c["status"] == "PASS" for c in checks))},
    ]
    write_preflight_report(checks, dep, unresolved, counts, passed, state)
    if passed:
        sha = write_config_lock(state["cfg"], allow_refreeze=refreeze)
        atomic_json(AUDIT_ROOT / "BLOCK14_preflight_PASS.json",
                    {"status": "PASS", "code_sha256": code_sha256(), "config_sha256": sha, "time": now_utc()})
        log(f"PREFLIGHT PASS -- config frozen/locked sha256={sha}")
    else:
        p = AUDIT_ROOT / "BLOCK14_preflight_PASS.json"
        if p.exists():
            p.unlink()
        log("PREFLIGHT FAIL -- the full experiment MUST NOT start.")
    return passed


def cfg_frozen(state: dict) -> Path:
    raw = state.get("cfg_raw") or state.get("cfg")
    return Path(raw["upstream"]["block13_frozen_outputs"]) if raw else BLOCK13_FROZEN_DEFAULT


def write_preflight_report(checks, dep, unresolved, counts, passed, state) -> None:
    cfg = state.get("cfg")
    L = ["# BLOCK14 PREFLIGHT REPORT", "", f"Generated: {now_utc()}  |  code sha256: `{code_sha256()}`",
         f"Project root: `{WINDOWS_ROOT}`" + ("  (**overridden**)" if PROJECT_ROOT_OVERRIDDEN else ""), "",
         f"## RESULT: {'PASS' if passed else 'FAIL'}", ""]
    L += ["## 1. Upstream dependency status", "", to_markdown(pd.DataFrame(dep)), "", "Checks:", "",
          to_markdown(pd.DataFrame(checks)[["id", "status", "critical", "description", "detail"]]), ""]
    L += ["## 2. Exact input paths", "",
          f"- BLOCK04 manifest: `{BLOCK04_ROOT / 'final_window_manifest.csv'}`",
          f"- BLOCK05 feature manifest: `{BLOCK05_ROOT / 'manifests' / 'feature_manifest.csv'}`",
          f"- BLOCK13 adapter: `{_B13_SOURCE}`", f"- Frozen BLOCK13 outputs: `{cfg_frozen(state)}`", ""]
    L += ["## 3. Exact model paths", "",
          f"- ML: `{BLOCK06_ROOT / 'models' / 'LOSO' / 'Fold_S{SS}_{COND}_{MODEL}.joblib'}` (250 files)",
          f"- DL: `{BLOCK07_ROOT / '07B_DL_LOSO' / 'folds'}/**/best_model.pt` (250 files)",
          f"- DL prediction CSVs: `{BLOCK07_ROOT / '07B_DL_LOSO' / 'folds' / 'Subject_XX' / '<condition>' / '<model_dir>' / 'predictions.csv'}`",
          f"- ML prediction CSVs: `{BLOCK06_ROOT / '06B_ML_LOSO' / 'folds' / 'Subject_XX' / '<condition>_<model>_predictions.csv'}`", ""]
    L += ["## 4. Exact feature extractor", "",
          f"- {state['ex'].source if state.get('ex') else 'NOT LOADED'}",
          "- Contract: `fn(window[13,500]) -> vector[341]`; must reproduce stored BLOCK05 features.", ""]
    L += ["## 5. Exact preprocessing", "",
          "- No preprocessing is added. Inputs are frozen BLOCK04 windows (500 samples, 50% overlap, 13 channels, 125 Hz).",
          "- ML: BLOCK05 extractor -> frozen BLOCK06 model.",
          "- DL: per-channel training z-score stored in each BLOCK07 checkpoint -> frozen BLOCK07 model. No test-time statistics.", ""]
    L += ["## 6. Exact perturbation configuration", ""]
    if cfg:
        cd, nz = cfg["channel_dropout"], cfg["noise"]
        L += [f"- Dropout levels: 5/10/20/30% -> counts {cd['percentage_to_count']}; policy `{cd['pattern_policy']}`",
              f"- Dropout map: `{json.dumps(cd['channel_map'])}`  (missing channel = exact zeros; no interpolation)",
              f"- Noise: `{nz['noise_type']}` {json.dumps(nz['noise_parameterization'])}; levels {nz['noise_levels']}; "
              f"realizations {nz['n_noise_realizations']}; seed {nz['noise_seed']}; stage `{nz['noise_application_stage']}`",
              f"- Noise seed derivation (implemented): {NOISE_SEED_DERIVATION}",
              "- Documented deviation: noise is generated independently per BLOCK04 window.", ""]
    else:
        L += ["- NOT AVAILABLE: configuration is missing or not frozen.", ""]
    L += ["## 7. Expected experiment count", ""]
    if counts.get("units"):
        L += [f"- specs per (subject,condition,model): {counts['specs']} = 1 clean + {counts['dropout_specs']} dropout + {counts['noise_specs']} noise",
              f"- units = 1 regime x 25 subjects x 2 conditions x 10 models x {counts['specs']} = **{counts['units']}**", ""]
    else:
        L += ["- cannot be computed until the configuration is valid.", ""]
    L += ["## 8. Unresolved configuration items", ""]
    items = list(unresolved)
    if not any(c["id"] == "P07" and c["status"] == "PASS" for c in checks):
        items.append("model/checkpoint inventory not verified (P07)")
    items.append("Subject-dependent (BLOCK06A/07A) robustness: NOT evaluated (WARN, recorded as limitation)")
    L += [f"- {i}" for i in items] + ["", f"## 9. PASS/FAIL: **{'PASS' if passed else 'FAIL'}**", ""]
    atomic_text(PREFLIGHT_MD, "\n".join(L))
    atomic_json(PREFLIGHT_JSON, {"status": "PASS" if passed else "FAIL", "checks": checks, "dependency_table": dep,
                                 "unresolved": items, "expected_counts": counts,
                                 "code_sha256": code_sha256(), "time": now_utc()})


# ============================================================================
# 21. ENVIRONMENT / REPRODUCIBILITY
# ============================================================================

def environment_info() -> dict:
    gpu = cuda = None
    if torch is not None:
        try:
            cuda = torch.version.cuda
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:
            pass
    return {
        "python": sys.version, "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
        "gpu": gpu, "cuda": cuda, "torch": getattr(torch, "__version__", None),
        "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__, "xgboost": getattr(xgb, "__version__", None),
        "joblib": getattr(joblib, "__version__", None),
        "matplotlib": getattr(matplotlib, "__version__", None) if plt is not None else None,
    }


def write_input_manifest(ctx: RunCtx, b13_info: dict, registry_sha: str) -> pd.DataFrame:
    rows = []

    def add(name: str, p: Path | None, note: str = "") -> None:
        rows.append({"input": name, "path": str(p) if p else "", "exists": bool(p and Path(p).exists()),
                     "sha256": sha256_file(p) if p and Path(p).exists() and Path(p).is_file() else "", "note": note})
    add("BLOCK04_final_window_manifest", BLOCK04_ROOT / "final_window_manifest.csv")
    add("BLOCK05_feature_manifest", BLOCK05_ROOT / "manifests" / "feature_manifest.csv")
    add("BLOCK05_extractor_source",
        Path(ctx.extractor.source.rsplit(":", 1)[0]) if ctx.extractor else None,
        ctx.extractor.source if ctx.extractor else "")
    add("BLOCK13_adapter_script", _B13_SOURCE)
    add("BLOCK13_frozen_status_json", Path(b13_info["status_file"]) if b13_info.get("status_file") else None,
        "identity reference only")
    add("BLOCK14_config", CONFIG_PATH)
    add("BLOCK14_model_manifest", REGISTRY_CSV, f"registry sha256 {registry_sha}")
    df = pd.DataFrame(rows)
    atomic_csv(MANIFEST_ROOT / "BLOCK14_input_manifest.csv", df)
    return df


# ============================================================================
# 22. FINALIZE
# ============================================================================

def archive_if_exists(p: Path) -> None:
    if p.exists():
        ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(ARCHIVE_ROOT / f"{p.stem}.{int(time.time())}{p.suffix}"))


def write_incomplete(cfg: dict | None, reasons: list[str], expected: int, completed: int,
                     missing: list, invalid: list, nonfinite: int, duplicates: int) -> None:
    archive_if_exists(COMPLETE_PATH)
    atomic_json(INCOMPLETE_PATH, {
        "status": "INCOMPLETE", "block_id": BLOCK_ID, "reasons": reasons,
        "expected_units": expected, "completed_units": completed,
        "failed_units": invalid[:500], "n_failed_units": len(invalid),
        "missing_units": [m["unit"] for m in missing[:500]], "n_missing_units": len(missing),
        "nonfinite_count": nonfinite, "duplicate_count": duplicates, "timestamp": now_utc(),
        "last_progress": json.loads(PROGRESS_PATH.read_text(encoding="utf-8")) if PROGRESS_PATH.exists() else None,
    })
    log(f"BLOCK14 INCOMPLETE: {len(missing)} missing, {len(invalid)} invalid unit(s). See {INCOMPLETE_PATH}")


def finalize(ctx: RunCtx, cfg: dict, b13_info: dict) -> bool:
    prepare_dirs()
    unit_root = BLOCK14_ROOT
    units = enumerate_units(cfg)
    expected = len(units)
    log(f"FINALIZE: scanning {expected} expected units (streaming)")
    log_memory("finalize start")
    sc = scan_units(ctx, unit_root, units, bool(cfg["runtime"].get("export_long_prediction_csv", True)))
    metrics = sc["metrics"]
    completed = len(metrics)
    reasons: list[str] = []
    if sc["missing"]:
        reasons.append(f"{len(sc['missing'])} expected unit(s) missing")
    if sc["invalid"]:
        reasons.append(f"{len(sc['invalid'])} unit(s) failed validation")
    if sc["duplicates"]:
        reasons.append(f"{sc['duplicates']} duplicate unit id(s)")
    if reasons:
        write_incomplete(cfg, reasons, expected, completed, sc["missing"], sc["invalid"], 0, sc["duplicates"])
        return False

    problems: list[str] = []
    window_manifest = ctx.manifest[["window_id", "subject", "condition", "gesture_code", "set",
                                    "source_recording", "window_index"]].copy()
    atomic_csv(MANIFEST_ROOT / "BLOCK14_window_manifest.csv", window_manifest)
    um = metrics[ID_COLS + ["unit_file", "model_sha256", "code_sha256", "config_sha256"]].copy()
    um["unit_file_sha256"] = [sha256_file(Path(p)) for p in um.unit_file]
    atomic_csv(MANIFEST_ROOT / "BLOCK14_unit_manifest.csv", um)
    log_memory("after unit manifest")

    mcols = [c for c in metrics.columns if c != "unit_file"]
    drop_m = metrics[metrics.perturbation_type.isin(["clean", "channel_dropout"])][mcols]
    noise_m = metrics[metrics.perturbation_type.isin(["clean", "noise"])][mcols]
    atomic_csv(DROPOUT_ROOT / "BLOCK14_channel_dropout_metrics.csv", drop_m)
    atomic_csv(NOISE_ROOT / "BLOCK14_noise_metrics.csv", noise_m)
    atomic_csv(METRIC_ROOT / "BLOCK14_class_metrics_unit_level.csv", sc["class"])
    atomic_csv(METRIC_ROOT / "BLOCK14_confusion_matrices_unit_level.csv", sc["confusion"])
    atomic_csv(METRIC_ROOT / "BLOCK14_set_metrics_unit_level.csv", sc["set"])
    clean = metrics[metrics.perturbation_type == "clean"]
    up_check = verify_clean_against_upstream(ctx, cfg, metrics)
    atomic_csv(TABLE_AUDIT / "BLOCK14_upstream_clean_reference_check.csv", up_check)
    if "status" in up_check.columns and up_check["status"].astype(str).str.startswith("FAIL").any():
        n_bad = int(up_check.loc[up_check["status"].astype(str).str.startswith("FAIL"),
                                 ["subject", "condition", "model"]].drop_duplicates().shape[0])
        problems.append(f"{n_bad} clean unit(s) differ from frozen upstream predictions")
    clean_ref = clean[["regime", "subject", "condition", "model", "branch"] + list(METRICS) +
                      ["stored_feature_max_abs_diff", "stored_pred_agreement", "model_sha256"]].sort_values(
        ["model", "condition", "subject"])
    atomic_csv(CLEAN_ROOT / "BLOCK14_clean_reference.csv", clean_ref)

    deg = build_degradation(metrics, cfg)
    for ptype, root, name in (("channel_dropout", DROPOUT_ROOT, "channel_dropout"), ("noise", NOISE_ROOT, "noise")):
        d = deg[deg.perturbation_type == ptype]
        drop_cols = (["noise_type", "noise_level_semantics", "noise_realized"] if ptype == "channel_dropout"
                     else ["dropout_percent", "achieved_dropout_percent", "n_dropped", "dropped_channels"])
        atomic_csv(root / f"BLOCK14_{name}_subject_level.csv", d.drop(columns=drop_cols))
    gest = pair_delta_table(sc["class"], "gesture", list(CLASS_PRF_NAMES))
    sets_ = pair_delta_table(sc["set"], "set", list(METRICS))
    for ptype, root, name in (("channel_dropout", DROPOUT_ROOT, "channel_dropout"), ("noise", NOISE_ROOT, "noise")):
        atomic_csv(root / f"BLOCK14_{name}_gesture_level.csv", gest[gest.perturbation_type == ptype])
        atomic_csv(root / f"BLOCK14_{name}_set_level.csv", sets_[sets_.perturbation_type == ptype])
    del gest, sets_
    agg = subject_aggregate(deg)
    atomic_csv(STATS_ROOT / "BLOCK14_subject_level_aggregated.csv", agg)

    log("FINALIZE: statistics")
    stats = compute_stat_rows(agg, cfg, expected_subjects=len(SUBJECTS))
    stats_path = STATS_ROOT / "BLOCK14_paired_statistics.csv"
    atomic_csv(stats_path, stats)
    key = ["comparison", "evaluation_regime", "branch", "model", "condition", "perturbation_type",
           "perturbation_level", "level_index", "metric"]
    atomic_csv(STATS_ROOT / "BLOCK14_fdr_statistics.csv", stats[key + ["family_id", "primary_family", "p_signflip", "q_bh"]])
    atomic_csv(STATS_ROOT / "BLOCK14_effect_sizes.csv", stats[key + ["n_subjects", "mean_diff", "effect_size_dz"]])
    atomic_csv(STATS_ROOT / "BLOCK14_confidence_intervals.csv", stats[key + ["n_subjects", "mean_diff", "ci95_low", "ci95_high"]])
    for ptype, root, name in (("channel_dropout", DROPOUT_ROOT, "channel_dropout"), ("noise", NOISE_ROOT, "noise")):
        atomic_csv(root / f"BLOCK14_{name}_summary.csv", stats[(stats.perturbation_type == ptype)])
    comps = make_comparisons(agg, stats)
    atomic_csv(STATS_ROOT / "BLOCK14_ML_DL_robustness_comparison.csv", comps["ml_dl"])
    atomic_csv(STATS_ROOT / "BLOCK14_ME_MI_robustness_comparison.csv", comps["me_mi"])
    atomic_csv(STATS_ROOT / "BLOCK14_degradation_summary.csv", comps["degradation_summary"])

    cd, nz = cfg["channel_dropout"], cfg["noise"]
    cmap = normalize_channel_map(cfg)
    dm_rows, map_rows = [], []
    for lvl in CHANNEL_DROPOUT_LEVELS[1:]:
        for pi, pat in enumerate(cmap[lvl]):
            dm_rows.append({"dropout_percent": lvl, "pattern_id": pi, "n_channels": len(pat),
                            "achieved_percent": 100.0 * len(pat) / N_CHANNELS, "channels": "|".join(pat),
                            "selection_rule": cd["selection_rule"], "pattern_policy": cd["pattern_policy"],
                            "missing_channel_representation": cd["representation_of_missing_channel"],
                            "config_sha256": ctx.config_sha})
            map_rows.append({"dropout_percent": lvl, "pattern_id": pi,
                             "channel_index_0based": "|".join(str(CHANNEL_NAMES.index(c)) for c in pat),
                             "channels": "|".join(pat), "channel_order": "|".join(cd["channel_order"])})
    dm_path = DROPOUT_ROOT / "BLOCK14_channel_dropout_manifest.csv"
    atomic_csv(dm_path, pd.DataFrame(dm_rows))
    atomic_csv(DROPOUT_ROOT / "BLOCK14_channel_dropout_channel_map.csv", pd.DataFrame(map_rows))
    nm_rows = [{"noise_level": float(v), "level_index": i, "noise_type": nz["noise_type"],
                "noise_parameterization": json.dumps(nz["noise_parameterization"], sort_keys=True),
                "n_realizations": nz["n_noise_realizations"], "noise_seed": nz["noise_seed"],
                "application_stage": nz["noise_application_stage"], "scope": nz["noise_scope"],
                "seed_derivation_implemented": NOISE_SEED_DERIVATION, "config_rule": nz["noise_reproducibility_rule"],
                "config_sha256": ctx.config_sha} for i, v in enumerate(nz["noise_levels"])]
    nm_path = NOISE_ROOT / "BLOCK14_noise_manifest.csv"
    atomic_csv(nm_path, pd.DataFrame(nm_rows))
    noise_m_ = metrics[metrics.perturbation_type == "noise"]
    atomic_csv(NOISE_ROOT / "BLOCK14_noise_config_audit.csv", pd.DataFrame([{
        "noise_type": nz["noise_type"], "levels": "|".join(map(str, nz["noise_levels"])),
        "realized_noise_stat_min": float(noise_m_.noise_realized.min()),
        "realized_noise_stat_max": float(noise_m_.noise_realized.max()),
        "config_sha256": ctx.config_sha, "level_semantics": nz["noise_parameterization"]["level_semantics"],
        "frozen_before_execution": True,
        "documented_deviation": "noise generated independently per BLOCK04 window"}]))

    aud = sc["audit"]
    audit_cols = [c for c in aud.columns if c not in ID_COLS and c != "model_sha256"]
    atomic_csv(TABLE_AUDIT / "BLOCK14_perturbation_audit.csv", aud)
    failed_audit = int((~aud[audit_cols].astype(bool)).any(axis=1).sum())
    if failed_audit:
        problems.append(f"{failed_audit} unit(s) with a failed audit flag")
    cov = (metrics.groupby(["branch", "model", "condition", "perturbation_type", "level"], sort=False)
           .agg(completed_units=("subject", "size"), subjects=("subject", "nunique")).reset_index())
    spec_per = metrics.groupby(["perturbation_type", "level"], sort=False).apply(
        lambda g: g[["pattern", "realization"]].drop_duplicates().shape[0], include_groups=False).to_dict()
    cov["expected_units"] = [len(SUBJECTS) * spec_per[(r.perturbation_type, r.level)] for r in cov.itertuples()]
    cov["complete"] = cov.completed_units == cov.expected_units
    atomic_csv(TABLE_AUDIT / "BLOCK14_model_coverage.csv", cov)
    if not cov.complete.all():
        problems.append("model coverage incomplete")
    pc = metrics.groupby(["model", "perturbation_type"], sort=False).agg(units=("subject", "size")).reset_index()
    pc["windows_per_unit"] = WINDOWS_PER_BLOCK
    pc["windows_total"] = pc.units * WINDOWS_PER_BLOCK
    atomic_csv(TABLE_AUDIT / "BLOCK14_prediction_completeness.csv", pc)
    atomic_csv(TABLE_AUDIT / "BLOCK14_robustness_input_audit.csv", pd.DataFrame([{
        "subjects": int(metrics.subject.nunique()), "conditions": int(metrics.condition.nunique()),
        "gestures": N_GESTURES, "sets": len(SETS), "channels": N_CHANNELS,
        "windows_per_subject_condition": WINDOWS_PER_BLOCK, "feature_dim": N_FEATURES,
        "block04_rows": len(ctx.manifest)}]))
    h = metrics.groupby(["subject", "condition"]).clean_input_sha256.nunique()
    if (h != 1).any():
        problems.append("BLOCK04 clean input hash changed between units of the same subject x condition")
    atomic_csv(TABLE_AUDIT / "BLOCK14_subject_join_audit.csv",
               deg.groupby(["perturbation_type", "model", "condition"], sort=False)
               .agg(n_subjects=("subject", "nunique"), n_rows=("subject", "size")).reset_index())
    if int(metrics.subject.nunique()) != N_SUBJECTS or int(metrics.condition.nunique()) != 2:
        problems.append("subject/condition coverage differs from 25/2")
    if sorted(metrics.model.unique().tolist()) != sorted(ALL_MODELS):
        problems.append("model coverage differs from the 10 frozen models")
    dup = int(metrics.duplicated(ID_COLS).sum())
    num_cols_deg = [c for c in deg.columns
                    if c.startswith(("clean_", "perturbed_", "delta_")) and not c.startswith("relative")]
    nonfinite = int((~np.isfinite(deg[num_cols_deg].to_numpy(dtype=float))).sum())
    nonfinite += int((~np.isfinite(metrics[list(METRICS)].to_numpy(dtype=float))).sum())
    nonfinite += int((~np.isfinite(stats[["mean_diff", "ci95_low", "ci95_high", "p_signflip", "q_bh"]]
                                   .to_numpy(dtype=float))).sum())
    if dup:
        problems.append(f"{dup} duplicate subject/model/condition/perturbation combinations")
    if nonfinite:
        problems.append(f"{nonfinite} non-finite values in result tables")
    if set(sc["class"].gesture.unique().tolist()) != set(GESTURES):
        problems.append("gesture labels in saved class-level data are not numeric 1..7")

    # ---- self-check (audit block requirement 14) ----
    clean_ref_total = int(len(up_check))
    if clean_ref_total:
        clean_ref_pass = int((up_check["status"].astype(str).str.upper() == "PASS").sum())
        clean_ref_fail = clean_ref_total - clean_ref_pass
        pred_ident_fail = int((~up_check["prediction_identical"].astype(bool)).sum())
        macro_f1_fail = int(
            ((up_check["abs_diff"].astype(float) > CLEAN_AUDIT_ABS_TOL)
             | (up_check["stored_vs_recomputed_abs_diff"].astype(float) > CLEAN_AUDIT_ABS_TOL)).sum()
        )
    else:
        clean_ref_pass = clean_ref_fail = pred_ident_fail = macro_f1_fail = 0
    pert_fail = int(failed_audit)
    missing_n = int(len(sc["missing"]))
    invalid_n = int(len(sc["invalid"]))
    valid_n = int(completed)
    expected_n = int(expected)
    final_status = "PASS" if (
        expected_n == valid_n and missing_n == 0 and invalid_n == 0
        and dup == 0 and nonfinite == 0
        and clean_ref_fail == 0 and pred_ident_fail == 0 and macro_f1_fail == 0
        and pert_fail == 0
    ) else "FAIL"
    self_check = {
        "EXPECTED_UNITS": expected_n,
        "VALID_UNITS": valid_n,
        "MISSING_UNITS": missing_n,
        "INVALID_UNITS": invalid_n,
        "CLEAN_REFERENCE_TOTAL": clean_ref_total,
        "CLEAN_REFERENCE_PASS": clean_ref_pass,
        "CLEAN_REFERENCE_FAIL": clean_ref_fail,
        "PREDICTION_IDENTITY_FAIL": pred_ident_fail,
        "MACRO_F1_FAIL": macro_f1_fail,
        "PERTURBATION_VALIDATION_FAIL": pert_fail,
        "DUPLICATE_UNITS": int(dup),
        "NONFINITE_VALUES": int(nonfinite),
        "FINAL_STATUS": final_status,
    }
    log("PRE-FINALIZE SELF-CHECK:")
    for k, v in self_check.items():
        log(f"  {k} = {v}")
    atomic_json(AUDIT_ROOT / "BLOCK14_pre_finalize_self_check.json", self_check)

    if problems or final_status != "PASS":
        write_incomplete(cfg, problems or ["self-check FINAL_STATUS != PASS"],
                         expected_n, valid_n, sc["missing"], sc["invalid"], nonfinite, dup)
        return False

    log("FINALIZE: publication tables + figures")
    tab_recs = make_publication_tables(stats_path, cfg, dm_path)
    fig_recs = make_publication_figures(stats_path, cfg, dm_path, nm_path)
    fig_recs += make_diagnostic_figures(stats_path, STATS_ROOT / "BLOCK14_subject_level_aggregated.csv", cfg)
    atomic_csv(MANIFEST_ROOT / "table_manifest.csv", pd.DataFrame(tab_recs))
    atomic_csv(MANIFEST_ROOT / "figure_manifest.csv", pd.DataFrame(fig_recs))

    reg_sha = sha256_file(REGISTRY_CSV)
    inp = write_input_manifest(ctx, b13_info, reg_sha)
    atomic_json(REPRO_PATH, {
        "environment": environment_info(), "random_seed": cfg["random_seed"],
        "seeds": {"global": cfg["random_seed"], "noise_seed": nz["noise_seed"],
                  "bootstrap_seed": cfg["statistics"]["bootstrap_seed"],
                  "permutation_seed": cfg["statistics"]["permutation_seed"]},
        "noise_seed_derivation": NOISE_SEED_DERIVATION, "fdr_family_definition": FDR_FAMILY_DEFINITION,
        "block04_manifest_sha256": inp.loc[inp.input == "BLOCK04_final_window_manifest", "sha256"].iloc[0],
        "block05_feature_manifest_sha256": inp.loc[inp.input == "BLOCK05_feature_manifest", "sha256"].iloc[0],
        "block05_extractor": ctx.extractor.source if ctx.extractor else None,
        "block05_extractor_sha256": inp.loc[inp.input == "BLOCK05_extractor_source", "sha256"].iloc[0],
        "block06_block07_model_manifest": str(REGISTRY_CSV), "model_manifest_sha256": reg_sha,
        "block13_adapter_sha256": sha256_file(_B13_SOURCE), "block13_frozen_reference": b13_info,
        "block14_code_sha256": ctx.code_sha, "block14_config_sha256": ctx.config_sha,
        "block14_config_file_sha256": sha256_file(CONFIG_PATH), "timestamp": now_utc(),
    })
    code_variants = sorted(metrics.code_sha256.unique().tolist())

    out_files = [p for root in (CONFIG_ROOT, MANIFEST_ROOT, CLEAN_ROOT, PRED_ROOT, DROPOUT_ROOT, NOISE_ROOT,
                                METRIC_ROOT, STATS_ROOT, TABLE_ROOT, FIG_ROOT)
                 for p in sorted(root.rglob("*"))
                 if p.is_file() and "units" not in p.parts and not p.name.endswith(".tmp")
                 and p.name not in ("BLOCK14_output_manifest.csv", "BLOCK14_hash_cache.json")]
    outm = pd.DataFrame([{"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in out_files])
    atomic_csv(MANIFEST_ROOT / "BLOCK14_output_manifest.csv", outm)
    unit_manifest_sha = sha256_file(MANIFEST_ROOT / "BLOCK14_unit_manifest.csv")

    sumry = {
        "block_id": BLOCK_ID, "protocol_version": PROTOCOL_VERSION, "status": "PASS",
        "primary_metric": PRIMARY_METRIC,
        "degradation_definition": "delta_P = P_clean - P_perturbed",
        "regimes_evaluated": cfg["evaluation_regimes"],
        "regimes_not_evaluated": {"SD": "split definition and BLOCK06A/07A artifacts not in the verified contract"},
        "n_units": completed, "n_degradation_rows": len(deg), "n_statistics_rows": len(stats),
        "channel_dropout": {"percentage_to_count": cd["percentage_to_count"], "pattern_policy": cd["pattern_policy"],
                            "selection_rule": cd["selection_rule"]},
        "noise": {k: nz[k] for k in ("noise_type", "noise_parameterization", "noise_levels",
                                     "n_noise_realizations", "noise_seed", "noise_application_stage")},
        "documented_deviation": "noise generated independently per BLOCK04 window",
        "fdr_family_definition": FDR_FAMILY_DEFINITION, "timestamp": now_utc(),
    }
    atomic_json(SUMMARY_PATH, sumry)
    atomic_json(AUDIT_REPORT_PATH, {
        "status": "PASS", "checks": {
            "units_expected_equals_completed": expected == completed,
            "subjects": sorted(metrics.subject.unique().tolist()),
            "conditions": sorted(metrics.condition.unique().tolist()),
            "gestures": list(GESTURES), "sets": list(SETS), "channels": N_CHANNELS,
            "duplicates": dup, "nonfinite": nonfinite, "failed_audit_units": failed_audit,
            "clean_perturbed_paired": True,
            "all_perturbed_from_same_frozen_model_hash_as_clean": bool(
                (metrics.groupby(UNIT_KEY).model_sha256.nunique() == 1).all()),
            "upstream_clean_reference": ("WAIVED" if not cfg["upstream"]["upstream_clean_reference"]["enabled"]
                                         else "CHECKED"),
            "upstream_clean_reference_dl_exact": str(BLOCK07_ROOT / "07B_DL_LOSO" / "folds" / "Subject_XX"
                                                     / "<condition>" / "<model_dir>" / "predictions.csv"),
            "upstream_clean_reference_ml_exact": str(BLOCK06_ROOT / "06B_ML_LOSO" / "folds" / "Subject_XX"
                                                     / "<condition>_<model>_predictions.csv"),
            "upstream_clean_reference_dl_model_dirs": DL_MODEL_DIRS,
            "code_hash_variants": code_variants,
            "diagnostic_prediction_collapse": {
                "note": "units whose predictions use <=1 gesture class.",
                "n_collapsed_units": int(metrics.prediction_collapse_single_class.sum()),
                "by_branch_perturbation_level": (metrics[metrics.prediction_collapse_single_class]
                                                 .groupby(["branch", "perturbation_type", "level"]).size()
                                                 .rename("n").reset_index().to_dict(orient="records"))}},
        "self_check": self_check,
        "timestamp": now_utc()})
    archive_if_exists(INCOMPLETE_PATH)
    atomic_json(COMPLETE_PATH, {
        "status": "PASS", "subjects": list(SUBJECTS), "conditions": list(CONDITIONS),
        "gestures": list(GESTURES), "sets": list(SETS), "models": list(ALL_MODELS),
        "perturbations": {"channel_dropout_percent": list(CHANNEL_DROPOUT_LEVELS[1:]),
                          "noise_levels": nz["noise_levels"],
                          "noise_realizations": nz["n_noise_realizations"], "clean_reference": True},
        "evaluation_regimes": cfg["evaluation_regimes"], "regimes_not_evaluated": ["SD"],
        "expected_units": expected, "completed_units": completed,
        "failed_units": 0, "missing_units": 0,
        "nonfinite_count": nonfinite, "duplicate_count": dup,
        "waivers": ({"upstream_clean_reference": cfg["upstream"]["upstream_clean_reference"]["waiver_justification"]}
                    if not cfg["upstream"]["upstream_clean_reference"]["enabled"] else {}),
        "input_hashes": {r.input: r.sha256 for r in inp.itertuples()},
        "output_hashes": {"unit_manifest_sha256": unit_manifest_sha,
                          "output_manifest_sha256": sha256_file(MANIFEST_ROOT / "BLOCK14_output_manifest.csv"),
                          "statistics_sha256": sha256_file(stats_path)},
        "code_sha256": ctx.code_sha, "config_sha256": ctx.config_sha, "code_hash_variants": code_variants,
        "self_check": self_check,
        "timestamp": now_utc()})
    log("BLOCK 14 PASS -- ROBUSTNESS COMPLETE")
    return True


# ============================================================================
# 23. CONTEXT / GATES / DRY RUN / RUN
# ============================================================================

def build_ctx(cfg: dict) -> tuple[RunCtx, dict]:
    b13 = import_block13(cfg)
    manifest, _info = load_block04_manifest(b13)
    ex = load_extractor(cfg)
    reg = load_registry()
    ok, notes, b13_info = crosscheck_block13_frozen(cfg, b13)
    if not ok:
        raise RuntimeError("Frozen BLOCK13 identity cross-check failed: " + "; ".join(notes))
    if torch is not None:
        torch.set_num_threads(int(cfg["runtime"]["torch_threads"]))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        configure_canonical_dl_inference()
    set_seed(int(cfg["random_seed"]))
    ctx = RunCtx(cfg=cfg, b13=b13, manifest=manifest, extractor=ex, config_sha=canonical_config_sha(cfg),
                 code_sha=code_sha256(), registry=reg)
    return ctx, b13_info


def require_preflight(cfg: dict) -> None:
    p = AUDIT_ROOT / "BLOCK14_preflight_PASS.json"
    if not p.exists():
        raise RuntimeError("No PASSING preflight on record. Run --preflight first.")
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("code_sha256") != code_sha256() or d.get("config_sha256") != canonical_config_sha(cfg):
        raise RuntimeError("Preflight PASS is stale (code or config changed). Re-run --preflight.")


def require_dry_runs(cfg: dict) -> None:
    need_b, need_p = {"ML", "DL"}, {"channel_dropout", "noise"}
    got_b, got_p = set(), set()
    for p in AUDIT_ROOT.glob("BLOCK14_dry_run_PASS_*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        if (d.get("code_sha256") == code_sha256() and d.get("config_sha256") == canonical_config_sha(cfg)
                and d.get("status") == "PASS"):
            got_b.add(d["branch"])
            got_p.add(d["perturbation_type"])
    if not (need_b <= got_b and need_p <= got_p):
        raise RuntimeError(
            f"Dry-run gate not satisfied for current code+config. Have branches={sorted(got_b)}, "
            f"perturbations={sorted(got_p)}; need branches={sorted(need_b)} and perturbations={sorted(need_p)}.")


def run_dry_run(subject: int, condition: str, model: str, pert: str) -> bool:
    global _DRYRUN_ACTIVE
    _DRYRUN_ACTIVE = True
    try:
        return _run_dry_run_impl(subject, condition, model, pert)
    finally:
        _DRYRUN_ACTIVE = False


def _run_dry_run_impl(subject: int, condition: str, model: str, pert: str) -> bool:
    prepare_dirs()
    cfg = load_config(require_lock=True)
    require_preflight(cfg)
    if model not in ALL_MODELS or subject not in SUBJECTS or condition not in CONDITIONS:
        raise ValueError("invalid dry-run subject/condition/model")
    ptype, _, lvl = pert.partition(":")
    specs_all = build_specs(cfg)
    cand = [s for s in specs_all if s.ptype == ptype and s.level == fmt_level(lvl)
            and s.pattern == 0 and s.realization == 0]
    if len(cand) != 1:
        raise ValueError(f"--dry-perturbation {pert!r} does not match a frozen level; available: "
                         + ", ".join(f"{s.ptype}:{s.level}" for s in specs_all
                                     if s.ptype != 'clean' and s.pattern == 0 and s.realization == 0))
    specs = [specs_all[0], cand[0]]
    ctx, b13_info = build_ctx(cfg)
    branch = "ML" if model in ML_MODELS else "DL"
    log(f"DRY RUN: S{subject:02d} {condition} {model} {pert} (+ clean reference)")
    shutil.rmtree(DRYRUN_ROOT, ignore_errors=True)
    results = {}
    for tag in ("run_a", "run_b"):
        root = DRYRUN_ROOT / tag
        prog = Progress(DRYRUN_ROOT / f"progress_{tag}.json", len(specs))
        results[tag] = run_units(ctx, root, [subject], [condition], [model], specs, prog)
        log_memory(f"dry-run {tag} done")
    prog = Progress(DRYRUN_ROOT / "progress_resume.json", len(specs))
    resume = run_units(ctx, DRYRUN_ROOT / "run_a", [subject], [condition], [model], specs, prog)

    unit_archive_checks = []
    for sp in specs:
        u = Unit(cfg["evaluation_regimes"][0], subject, condition, model, sp)
        try:
            ud = read_unit(u.path(DRYRUN_ROOT / "run_a"))
            ok_opt, why_opt = _validate_optional_unit_fields(ud, u)
            unit_archive_checks.append({
                "spec": sp.tag, "optional_fields_valid": bool(ok_opt), "reason": why_opt,
                "stored_feature_max_abs_diff": float(ud["stored_feature_max_abs_diff"]),
                "stored_pred_agreement": float(ud["stored_pred_agreement"]),
            })
        except Exception as exc:
            unit_archive_checks.append({"spec": sp.tag, "optional_fields_valid": False,
                                        "reason": f"{type(exc).__name__}: {exc}"})

    determinism = []
    for sp in specs:
        u = Unit(cfg["evaluation_regimes"][0], subject, condition, model, sp)
        a, b = read_unit(u.path(DRYRUN_ROOT / "run_a")), read_unit(u.path(DRYRUN_ROOT / "run_b"))
        determinism.append({"spec": sp.tag, "pred_identical": bool(np.array_equal(a["y_pred"], b["y_pred"])),
                            "proba_close": bool(np.allclose(a["proba"], b["proba"], equal_nan=True, atol=1e-6)),
                            "metrics_identical": bool(np.allclose(a["metrics"], b["metrics"]))})

    sc = scan_units(ctx, DRYRUN_ROOT / "run_a",
                    [Unit(cfg["evaluation_regimes"][0], subject, condition, model, sp) for sp in specs], False)
    ok_scan = not sc["missing"] and not sc["invalid"] and len(sc["metrics"]) == len(specs)
    deg = build_degradation(sc["metrics"], cfg) if ok_scan else pd.DataFrame()
    agg = subject_aggregate(deg) if ok_scan else pd.DataFrame()
    stats = compute_stat_rows(agg, cfg, expected_subjects=1) if ok_scan else pd.DataFrame()
    finite = bool(ok_scan and np.isfinite(deg[[c for c in deg.columns
                                              if c.startswith(("clean_", "perturbed_", "delta_"))
                                              and not c.startswith("relative")]].to_numpy(dtype=float)).all())
    ok = bool(ok_scan and finite and results["run_a"]["failed"] == 0 and results["run_b"]["failed"] == 0
              and results["run_a"]["computed"] == len(specs) and resume["computed"] == 0
              and all(d["pred_identical"] and d["proba_close"] and d["metrics_identical"] for d in determinism)
              and all(bool(x.get("optional_fields_valid", False)) for x in unit_archive_checks)
              and len(deg) == 1 and len(stats) > 0)
    report = {"status": "PASS" if ok else "FAIL", "branch": branch, "perturbation_type": ptype,
              "subject": subject, "condition": condition, "model": model, "perturbation": pert,
              "units_computed_run_a": results["run_a"], "units_computed_run_b": results["run_b"],
              "resume_second_pass_computed": resume["computed"], "determinism": determinism,
              "unit_archive_optional_field_checks": unit_archive_checks,
              "degradation_rows": len(deg), "finite": finite,
              "delta_primary": (float(deg["delta_metric"].iloc[0]) if len(deg) else None),
              "clean_primary": (float(deg["clean_metric"].iloc[0]) if len(deg) else None),
              "perturbed_primary": (float(deg["perturbed_metric"].iloc[0]) if len(deg) else None),
              "unit_audit_all_true": bool(sc["audit"].drop(columns=ID_COLS + ["model_sha256"]).astype(bool).all().all())
              if ok_scan else False,
              "code_sha256": code_sha256(), "config_sha256": canonical_config_sha(cfg),
              "timestamp": now_utc(), "outputs": str(DRYRUN_ROOT)}
    atomic_json(DRYRUN_ROOT / "BLOCK14_dry_run_report.json", report)
    if not deg.empty:
        atomic_csv(DRYRUN_ROOT / "BLOCK14_dry_run_degradation.csv", deg)
        atomic_csv(DRYRUN_ROOT / "BLOCK14_dry_run_statistics_nequals1.csv", stats)
    marker = AUDIT_ROOT / f"BLOCK14_dry_run_PASS_{branch}_{ptype}.json"
    if ok:
        atomic_json(marker, report)
    elif marker.exists():
        marker.unlink()
    log(f"DRY RUN {'PASS' if ok else 'FAIL'} -- inspect {DRYRUN_ROOT}")
    return ok


def run_full() -> bool:
    prepare_dirs()
    cfg = load_config(require_lock=True)
    require_preflight(cfg)
    require_dry_runs(cfg)
    ctx, b13_info = build_ctx(cfg)
    specs = build_specs(cfg)
    total = len(enumerate_units(cfg))
    log("=" * 100)
    log(f"BLOCK 14 FULL RUN | expected units={total} | specs per model/subject/condition={len(specs)}")
    repair = _clean_reference_repair_units()
    if repair:
        log(f"TARGETED CLEAN-REFERENCE REPAIR ACTIVE: {len(repair)} cached clean unit(s) will be recomputed.")
    log("NO RETRAINING | NO FINE-TUNING | NO HYPERPARAMETER TUNING | resumable")
    log("=" * 100)
    prog = Progress(PROGRESS_PATH, total)
    try:
        stats = run_units(ctx, BLOCK14_ROOT, SUBJECTS, CONDITIONS, ALL_MODELS, specs, prog)
    except Exception as exc:
        last = json.loads(PROGRESS_PATH.read_text(encoding="utf-8")) if PROGRESS_PATH.exists() else {}
        tb = traceback.format_exc()
        record_failure(str(last.get("current_unit") or "BLOCK14_RUN"), exc)
        log(f"RUN STOPPED: {type(exc).__name__}: {exc}")
        log(f"last completed unit: {last.get('last_completed_unit')} | in-flight unit: {last.get('current_unit')}")
        atomic_json(BLOCK14_ROOT / "BLOCK14_LAST_EXCEPTION.json", {
            "status": "INCOMPLETE", "error_type": type(exc).__name__, "error": str(exc),
            "traceback": tb, "last_completed_unit": last.get("last_completed_unit"),
            "in_flight_unit": last.get("current_unit"), "timestamp": now_utc(),
            "resume_policy": "verified units are skipped; in-flight/invalid units are recomputed"})
        write_incomplete(cfg, [f"run stopped: {type(exc).__name__}: {exc}",
                               f"last completed unit: {last.get('last_completed_unit')}",
                               f"in-flight unit: {last.get('current_unit')}"],
                         total, 0, [], [], 0, 0)
        return False
    log(f"RUN complete: {stats}")
    return finalize(ctx, cfg, b13_info)


def run_finalize_only() -> bool:
    prepare_dirs()
    cfg = load_config(require_lock=True)
    require_preflight(cfg)
    ctx, b13_info = build_ctx(cfg)
    return finalize(ctx, cfg, b13_info)


# ============================================================================
# 24. CLI
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="BLOCK 14 Robustness & Stress Testing")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--init-config", action="store_true")
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--refreeze-config", action="store_true")
    ap.add_argument("--dry-subject", type=int, default=1)
    ap.add_argument("--dry-condition", default="ME", choices=list(CONDITIONS))
    ap.add_argument("--dry-model", default="LDA", choices=list(ALL_MODELS))
    ap.add_argument("--dry-perturbation", default="channel_dropout:30",
                    help="channel_dropout:<pct> or noise:<level>")
    args = ap.parse_args()
    if not any([args.init_config, args.preflight, args.dry_run, args.run, args.finalize_only]):
        ap.print_help()
        print("\nNo action selected. Order: --init-config -> --preflight -> --dry-run -> --run")
        return 2
    try:
        prepare_dirs()
        enable_native_crash_log()
        if PROJECT_ROOT_OVERRIDDEN:
            log(f"WARNING: project root overridden by PAPER2_PROJECT_ROOT = {WINDOWS_ROOT}")
        if args.init_config:
            init_config()
            return 0
        if args.preflight:
            return 0 if run_preflight(refreeze=args.refreeze_config) else 1
        if args.dry_run:
            return 0 if run_dry_run(args.dry_subject, args.dry_condition, args.dry_model, args.dry_perturbation) else 1
        if args.run:
            return 0 if run_full() else 1
        return 0 if run_finalize_only() else 1
    except Exception as exc:
        log(f"BLOCK 14 FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())