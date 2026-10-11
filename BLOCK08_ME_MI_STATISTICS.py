#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BLOCK 08 — ME–MI STATISTICAL ANALYSIS

Purpose
-------
Statistically compare Motor Execution (ME) and Motor Imagery (MI) decoding
performance at the SUBJECT level using the frozen outputs of BLOCK 06 and
BLOCK 07.

Frozen design
-------------
25 paired subjects
2 conditions: ME, MI
2 evaluation regimes: SD, LOSO
10 models: 5 ML + 5 DL
5 metrics: Accuracy, Balanced Accuracy, Macro-F1, MCC, Cohen's kappa
100 predefined comparisons

Statistics
----------
D_i = ME_i - MI_i
paired dz
10,000 paired-subject bootstrap resamples, percentile 95% CI
10,000 two-sided paired sign-flip permutations
Benjamini-Hochberg FDR, q=0.05

Important
---------
- Subject is the inferential unit.
- Windows are never treated as independent subjects.
- ME and MI are paired explicitly by subject_id.
- No model is retrained.
- No prediction is modified.
- No subject is removed because of performance.
- No statistical test is selected after seeing results.
- No downstream Block 09–14 information is used.
- Only BLOCK 08 output is created/cleaned.
- The script discovers the actual frozen BLOCK 06/07 result files rather
  than assuming one exact filename for every upstream artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# FROZEN CONFIGURATION
# ============================================================================

BLOCK_NAME = "BLOCK08_ME_MI_STATISTICS"
STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"

N_SUBJECTS = 25
SUBJECTS = [f"S{i:02d}" for i in range(1, N_SUBJECTS + 1)]

CONDITIONS = ["ME", "MI"]
REGIMES = ["SD", "LOSO"]

ML_MODELS = [
    "LDA",
    "KNN",
    "SVM",
    "RandomForest",
    "XGBoost",
]

DL_MODELS = [
    "EEGNet",
    "ShallowConvNet",
    "DeepConvNet",
    "CNN-BiLSTM",
    "CNN-BiLSTM-Attention",
]

MODELS = ML_MODELS + DL_MODELS

METRICS = [
    "Accuracy",
    "Balanced Accuracy",
    "Macro-F1",
    "MCC",
    "Cohen's kappa",
]

PRIMARY_METRICS = [
    "Accuracy",
    "Balanced Accuracy",
    "Macro-F1",
]

SECONDARY_METRICS = [
    "MCC",
    "Cohen's kappa",
]

BOOTSTRAP_ITERATIONS = 10_000
PERMUTATION_ITERATIONS = 10_000
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05
FDR_Q = 0.05
SEED = 42

# Prediction-level consistency audit tolerance. The authoritative subject-level
# metrics are read from BLOCK06/BLOCK07; prediction files are independently
# rescored only as a consistency check. Small CSV/float serialization differences
# are tolerated, while materially different metric calculations are rejected.
PREDICTION_METRIC_ATOL = 1e-10
PREDICTION_METRIC_RTOL = 1e-8

EXPECTED_TESTS = len(REGIMES) * len(MODELS) * len(METRICS)

GESTURES = list(range(1, 8))
CHANCE_ACCURACY = 1.0 / 7.0

PERFORMANCE_ROUND = 3
EFFECT_ROUND = 3
CI_ROUND = 3

START_TIME = time.time()
UTC_START = datetime.now(timezone.utc).isoformat()


# ============================================================================
# PATH RESOLUTION
# ============================================================================

def resolve_project_root() -> Path:
    candidates = [
        Path(r"F:\Faruk\EEG_Motor_Execution_Imagery\Paper2_EEG_BCI"),
        Path(r"F:\Faruk\OFS_Paper_Work\Data_Set_Paper_Work\EEG_EMG_BIOZ_DATASET\Paper2_EEG_BCI"),
        Path.cwd() / "Paper2_EEG_BCI",
        Path.cwd(),
    ]

    for p in candidates:
        if (p / "06_ML").exists() and (p / "07_DL").exists():
            return p

    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "06_ML").exists() and (p / "07_DL").exists():
            return p

    raise FileNotFoundError(
        "Could not locate Paper2_EEG_BCI containing both 06_ML and 07_DL."
    )


PROJECT_ROOT = resolve_project_root()
BLOCK06_ROOT = PROJECT_ROOT / "06_ML"
BLOCK07_ROOT = PROJECT_ROOT / "07_DL"
BLOCK08_ROOT = PROJECT_ROOT / "08_ME_MI_STATISTICS"

SUBJECT_DIR = BLOCK08_ROOT / "subject_level"
STAT_DIR = BLOCK08_ROOT / "statistics"
FDR_DIR = BLOCK08_ROOT / "multiple_comparisons"
FIG_DIR = BLOCK08_ROOT / "figures"
SUPP_DIR = BLOCK08_ROOT / "supplementary"
TABLE_DIR = BLOCK08_ROOT / "tables"
CONFIG_DIR = BLOCK08_ROOT / "configs"
AUDIT_DIR = BLOCK08_ROOT / "audit"


# ============================================================================
# LOGGING
# ============================================================================

def log(msg: str) -> None:
    elapsed = time.time() - START_TIME
    print(f"{elapsed:9.2f}s | {msg}", flush=True)


def fail(message: str) -> None:
    raise RuntimeError(message)


# ============================================================================
# FILE UTILITIES
# ============================================================================

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def csv_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def normalize_name(x: Any) -> str:
    s = str(x).strip().lower()
    s = s.replace("κ", "kappa")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def normalize_model_name(x: Any) -> str:
    s = str(x).strip().lower()
    s = s.replace("_", "").replace("-", "").replace(" ", "")
    s = s.replace("+", "")
    aliases = {
        "lda": "LDA",
        "knn": "KNN",
        "svm": "SVM",
        "rf": "RandomForest",
        "randomforest": "RandomForest",
        "randomforestclassifier": "RandomForest",
        "xgb": "XGBoost",
        "xgboost": "XGBoost",
        "eegnet": "EEGNet",
        "shallowconvnet": "ShallowConvNet",
        "deepconvnet": "DeepConvNet",
        "cnnbilstm": "CNN-BiLSTM",
        "cnnbilstmattention": "CNN-BiLSTM-Attention",
        "cnnbilstmselfattention": "CNN-BiLSTM-Attention",
    }
    return aliases.get(s, str(x).strip())


def normalize_condition(x: Any) -> str | None:
    s = str(x).strip().upper()
    if s in {"ME", "MOTOR_EXECUTION", "MOTOREXECUTION", "EXECUTION"}:
        return "ME"
    if s in {"MI", "MOTOR_IMAGERY", "MOTORIMAGERY", "IMAGERY"}:
        return "MI"
    return None


def normalize_regime(x: Any) -> str | None:
    s = str(x).strip().upper()
    if s in {"SD", "SUBJECT_DEPENDENT", "SUBJECTDEPENDENT", "SUBJECT-DEPENDENT"}:
        return "SD"
    if s in {"LOSO", "LEAVE_ONE_SUBJECT_OUT", "LEAVE-ONE-SUBJECT-OUT"}:
        return "LOSO"
    return None


def normalize_subject(x: Any) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    m = re.search(r"(?<!\d)(?:S)?(\d{1,2})(?!\d)", s)
    if not m:
        return None
    n = int(m.group(1))
    if 1 <= n <= 25:
        return f"S{n:02d}"
    return None


def find_column(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    normalized = {normalize_name(c): c for c in df.columns}
    for alias in aliases:
        key = normalize_name(alias)
        if key in normalized:
            return normalized[key]
    return None


def safe_float(x: Any) -> float:
    if pd.isna(x):
        return float("nan")
    return float(x)


# ============================================================================
# OUTPUT MANAGEMENT
# ============================================================================

def prepare_output() -> None:
    if BLOCK08_ROOT.exists():
        for child in BLOCK08_ROOT.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    for d in [
        SUBJECT_DIR,
        STAT_DIR,
        FDR_DIR,
        FIG_DIR,
        SUPP_DIR,
        TABLE_DIR,
        CONFIG_DIR,
        AUDIT_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# UPSTREAM CONFIGURATION AUDIT
# ============================================================================

def _collect_numeric_values_by_key(
    obj: Any,
    target_keys: set[str],
    out: list[tuple[str, int]],
    prefix: str = "",
) -> None:
    """Recursively collect integer values for normalized JSON keys."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_norm = normalize_name(key)
            current = f"{prefix}.{key}" if prefix else str(key)
            if key_norm in target_keys:
                if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                    out.append((current, int(value)))
                elif isinstance(value, float) and float(value).is_integer():
                    out.append((current, int(value)))
            _collect_numeric_values_by_key(value, target_keys, out, current)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            _collect_numeric_values_by_key(value, target_keys, out, f"{prefix}[{i}]")


def _unique_positive_candidates(
    cfg: dict[str, Any],
    keys: Iterable[str],
) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    _collect_numeric_values_by_key(
        cfg,
        {normalize_name(k) for k in keys},
        found,
    )
    return sorted({(path, value) for path, value in found if value > 0})


def _get_first_present(
    cfg: dict[str, Any],
    keys: Iterable[str],
) -> tuple[str, Any] | None:
    """
    Return (matched_key, value) for the first key present at any nesting level.
    Used by the protocol audit to accept a small set of canonical spellings
    while still *requiring* at least one of them to exist.
    """
    target = {normalize_name(k) for k in keys}

    def _walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if normalize_name(key) in target:
                    return value
                found = _walk(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _walk(item)
                if found is not None:
                    return found
        return None

    value = _walk(cfg)
    if value is None:
        return None
    return (next(iter(keys)), value)


def verify_upstream_protocol_configuration() -> dict[str, Any]:
    """
    Cross-check the frozen upstream configurations for the cohort, conditions,
    gesture space, and W500/O50 handoff. This prevents BLOCK08 from silently
    comparing metrics produced under incompatible task definitions.

    The audit REQUIRES each category to be discoverable in at least one
    recognized spelling. If a category cannot be found, the audit fails
    rather than silently passing.
    """
    paths = {
        "BLOCK06": BLOCK06_ROOT / "BLOCK06_config.json",
        "BLOCK07": BLOCK07_ROOT / "configs" / "BLOCK07_config.json",
    }

    result: dict[str, Any] = {}

    subject_keys = [
        "subjects",
        "subject_ids",
        "subject_id_list",
        "cohort_subjects",
    ]
    condition_keys = [
        "conditions",
        "condition_list",
        "motor_conditions",
        "tasks",
    ]
    n_subject_keys = [
        "n_subjects",
        "cohort_size",
        "number_of_subjects",
        "num_subjects",
    ]
    gesture_keys = [
        "gestures",
        "gesture_ids",
        "gesture_classes",
        "gesture_set",
        "classes",
    ]

    for label, path in paths.items():
        if not path.exists():
            fail(
                f"Missing {label} configuration required for BLOCK08 protocol audit: {path}"
            )

        cfg = read_json_if_exists(path)
        if cfg is None:
            fail(f"Could not parse {label} configuration: {path}")

        # --- Cohort size ---------------------------------------------------
        n_subject_hit = _get_first_present(cfg, n_subject_keys)
        if n_subject_hit is None:
            fail(
                f"{label} config does not declare a recognized cohort-size key. "
                f"Tried: {n_subject_keys}"
            )
        n_subjects = n_subject_hit[1]
        try:
            n_subjects_int = int(n_subjects)
        except (TypeError, ValueError):
            fail(
                f"{label} config cohort size is not an integer: {n_subjects!r}"
            )
        if n_subjects_int != N_SUBJECTS:
            fail(
                f"{label} config cohort size {n_subjects_int} != {N_SUBJECTS}"
            )

        # --- Subjects ------------------------------------------------------
        subject_hit = _get_first_present(cfg, subject_keys)
        if subject_hit is None:
            fail(
                f"{label} config does not declare a recognized subject-list key. "
                f"Tried: {subject_keys}"
            )
        subject_values = subject_hit[1]
        if not isinstance(subject_values, (list, tuple)):
            fail(
                f"{label} config subject list is not a list/tuple: "
                f"{type(subject_values).__name__}"
            )
        normalized_subjects = sorted(
            normalize_subject(x) for x in subject_values
        )
        if None in normalized_subjects:
            fail(
                f"{label} config subject list contains unrecognized entries."
            )
        if normalized_subjects != SUBJECTS:
            fail(
                f"{label} config subject cohort does not match S01–S25."
            )

        # --- Conditions ----------------------------------------------------
        condition_hit = _get_first_present(cfg, condition_keys)
        if condition_hit is None:
            fail(
                f"{label} config does not declare a recognized condition-list key. "
                f"Tried: {condition_keys}"
            )
        condition_values = condition_hit[1]
        if not isinstance(condition_values, (list, tuple)):
            fail(
                f"{label} config condition list is not a list/tuple: "
                f"{type(condition_values).__name__}"
            )
        normalized_conditions = sorted(
            normalize_condition(x) for x in condition_values
        )
        if None in normalized_conditions:
            fail(
                f"{label} config condition list contains unrecognized entries."
            )
        if normalized_conditions != sorted(CONDITIONS):
            fail(
                f"{label} config conditions do not match ME/MI."
            )

        # --- Gesture space -------------------------------------------------
        gesture_hit = _get_first_present(cfg, gesture_keys)
        if gesture_hit is None:
            fail(
                f"{label} config does not declare a recognized gesture-space key. "
                f"Tried: {gesture_keys}"
            )
        gestures_value = gesture_hit[1]

        if isinstance(gestures_value, dict):
            gesture_ids = sorted(
                int(k) for k in gestures_value.keys() if str(k).isdigit()
            )
        elif isinstance(gestures_value, (list, tuple)):
            try:
                gesture_ids = sorted(int(x) for x in gestures_value)
            except (TypeError, ValueError):
                fail(
                    f"{label} config gesture list contains non-integer entries."
                )
        else:
            fail(
                f"{label} config gesture space is neither list nor dict: "
                f"{type(gestures_value).__name__}"
            )

        if gesture_ids != GESTURES:
            fail(
                f"{label} config gesture space {gesture_ids} "
                f"does not match frozen 1..7."
            )

        result[label] = {
            "path": str(path),
            "n_subjects": n_subjects_int,
            "subjects_verified": True,
            "conditions_verified": True,
            "gesture_space_verified": True,
            "gesture_ids": gesture_ids,
        }

    return result


# ============================================================================
# UPSTREAM COMPLETION CHECKS
# ============================================================================

def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def verify_upstream_completion() -> dict[str, Any]:
    checks: dict[str, Any] = {}

    b06_complete = BLOCK06_ROOT / "BLOCK06_complete.json"
    b07_complete = BLOCK07_ROOT / "BLOCK07_complete.json"

    checks["BLOCK06_complete_exists"] = b06_complete.exists()
    checks["BLOCK07_complete_exists"] = b07_complete.exists()

    if not checks["BLOCK06_complete_exists"]:
        fail(f"Missing required upstream marker: {b06_complete}")

    if not checks["BLOCK07_complete_exists"]:
        fail(f"Missing required upstream marker: {b07_complete}")

    b06 = read_json_if_exists(b06_complete) or {}
    b07 = read_json_if_exists(b07_complete) or {}

    checks["BLOCK06_status"] = b06.get("status")
    checks["BLOCK07_status"] = b07.get("status")

    if b06.get("status") != "PASS":
        fail(f"BLOCK06 is not PASS: {b06.get('status')!r}")

    if b07.get("status") != "PASS":
        fail(f"BLOCK07 is not PASS: {b07.get('status')!r}")

    return checks


# ============================================================================
# CSV READING
# ============================================================================

def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Could not read CSV: {path}\n{e}") from e


# ============================================================================
# BLOCK 06 AUTHORITATIVE SUBJECT-LEVEL RESULTS
# ============================================================================

def discover_block06_subject_files() -> list[tuple[Path, str]]:
    candidates = [
        (BLOCK06_ROOT / "06A_ML_SD" / "subject_results" / "SD_subject_results.csv", "SD"),
        (BLOCK06_ROOT / "06B_ML_LOSO" / "subject_results" / "LOSO_subject_results.csv", "LOSO"),
    ]

    missing = [str(p) for p, _ in candidates if not p.exists()]
    if missing:
        fail(
            "Missing authoritative BLOCK06 subject-level result files:\n"
            + "\n".join(missing)
        )

    return candidates


def canonicalize_subject_result_file(
    df: pd.DataFrame,
    regime: str,
    source: Path,
    source_type: str,
) -> pd.DataFrame:
    subject_col = find_column(
        df,
        [
            "subject_id",
            "subject",
            "held_out_subject",
            "heldout_subject",
            "test_subject",
            "participant",
            "participant_id",
        ],
    )
    condition_col = find_column(
        df,
        ["condition", "cond", "motor_condition", "task", "state"],
    )
    model_col = find_column(
        df,
        ["model", "model_name", "classifier", "architecture"],
    )
    regime_col = find_column(
        df,
        ["evaluation_regime", "regime", "evaluation", "protocol"],
    )

    if subject_col is None or condition_col is None or model_col is None:
        return pd.DataFrame()

    out = pd.DataFrame({
        "subject_id": df[subject_col].map(normalize_subject),
        "condition": df[condition_col].map(normalize_condition),
        "model": df[model_col].map(normalize_model_name),
        "regime": regime,
        "source_file": str(source),
        "source_type": source_type,
    })

    if regime_col is not None:
        declared_regime = df[regime_col].map(normalize_regime)
        bad = declared_regime.notna() & (declared_regime != regime)
        if bad.any():
            fail(
                f"Declared evaluation_regime conflicts with path regime in {source}."
            )

    metric_aliases = {
        "Accuracy": ["accuracy", "acc"],
        "Balanced Accuracy": ["balanced_accuracy", "balanced_acc", "bal_accuracy", "bal_acc"],
        "Macro-F1": ["macro_f1", "macro_f1_score", "f1_macro"],
        "MCC": ["mcc", "matthews_corrcoef", "matthews_correlation"],
        "Cohen's kappa": [
            "cohen_kappa",
            "cohens_kappa",
            "cohen's kappa",
            "kappa",
            "cohen_k",
        ],
    }

    for metric, aliases in metric_aliases.items():
        col = find_column(df, aliases)
        if col is None:
            fail(
                f"Missing required metric '{metric}' in authoritative "
                f"BLOCK06/BLOCK07 subject-level file: {source}"
            )
        out[metric] = pd.to_numeric(df[col], errors="coerce")

    for alias in ["n_test", "test_windows", "n_validation", "n_train"]:
        col = find_column(df, [alias])
        if col is not None:
            out[alias] = pd.to_numeric(df[col], errors="coerce")

    return out


def validate_authoritative_subject_results(
    df: pd.DataFrame,
    expected_models: list[str] | tuple[str, ...],
    regime: str,
    source_label: str,
) -> pd.DataFrame:
    if df.empty:
        fail(f"{source_label}: empty subject-level result table.")

    required = [
        "subject_id",
        "condition",
        "model",
        "regime",
        *METRICS,
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        fail(
            f"{source_label}: missing canonical columns: {missing_cols}"
        )

    invalid_subject = df["subject_id"].isna()
    invalid_condition = df["condition"].isna()
    invalid_model = ~df["model"].isin(expected_models)
    invalid_regime = df["regime"] != regime

    invalid = (
        invalid_subject
        | invalid_condition
        | invalid_model
        | invalid_regime
    )

    if invalid.any():
        bad = df[invalid].copy()
        csv_write(
            bad,
            AUDIT_DIR / f"{source_label}_invalid_subject_results.csv",
        )
        fail(
            f"{source_label}: invalid subject/condition/model/regime rows: "
            f"{len(bad)}"
        )

    key = ["subject_id", "condition", "model", "regime"]
    dup = df.duplicated(key, keep=False)
    if dup.any():
        bad = df[dup].sort_values(key)
        csv_write(
            bad,
            AUDIT_DIR / f"{source_label}_duplicate_subject_results.csv",
        )
        fail(
            f"{source_label}: duplicate logical subject-level results: "
            f"{len(bad)}"
        )

    validate_metric_ranges(df, source_label)

    expected = pd.MultiIndex.from_product(
        [
            SUBJECTS,
            CONDITIONS,
            list(expected_models),
        ],
        names=["subject_id", "condition", "model"],
    ).to_frame(index=False)

    expected["regime"] = regime

    present_keys = df[key].drop_duplicates()
    missing = expected.merge(
        present_keys,
        on=key,
        how="left",
        indicator=True,
    )
    missing = missing[missing["_merge"] != "both"].drop(columns=["_merge"])

    if not missing.empty:
        csv_write(
            missing,
            AUDIT_DIR / f"{source_label}_missing_subject_results.csv",
        )
        fail(
            f"{source_label}: missing logical subject-level results: "
            f"{len(missing)}"
        )

    unexpected = df.merge(
        expected,
        on=key,
        how="left",
        indicator=True,
    )
    unexpected = unexpected[unexpected["_merge"] != "both"].drop(columns=["_merge"])
    if not unexpected.empty:
        csv_write(
            unexpected,
            AUDIT_DIR / f"{source_label}_unexpected_subject_results.csv",
        )
        fail(
            f"{source_label}: unexpected subject-level results: "
            f"{len(unexpected)}"
        )

    expected_rows = N_SUBJECTS * len(CONDITIONS) * len(expected_models)
    if len(df) != expected_rows:
        fail(
            f"{source_label}: row count {len(df)} != expected {expected_rows}"
        )

    return df.sort_values(key).reset_index(drop=True)


def load_block06() -> pd.DataFrame:
    frames = []

    for path, regime in discover_block06_subject_files():
        df = read_csv(path)
        parsed = canonicalize_subject_result_file(
            df,
            regime,
            path,
            "BLOCK06_SUBJECT_RESULTS",
        )
        parsed = validate_authoritative_subject_results(
            parsed,
            ML_MODELS,
            regime,
            f"BLOCK06_{regime}",
        )
        frames.append(parsed)
        log(
            f"BLOCK06 authoritative subject results accepted: "
            f"{path} | {len(parsed)} rows"
        )

    out = pd.concat(frames, ignore_index=True)

    if len(out) != N_SUBJECTS * len(CONDITIONS) * len(ML_MODELS) * len(REGIMES):
        fail("BLOCK06 combined subject-level row count is incorrect.")

    return out


# ============================================================================
# BLOCK 07 AUTHORITATIVE SUBJECT-LEVEL RESULTS
# ============================================================================

def discover_block07_subject_files() -> list[tuple[Path, str]]:
    candidates = [
        (BLOCK07_ROOT / "07A_DL_SD" / "subject_results.csv", "SD"),
        (BLOCK07_ROOT / "07B_DL_LOSO" / "subject_results.csv", "LOSO"),
    ]

    missing = [str(p) for p, _ in candidates if not p.exists()]
    if missing:
        fail(
            "Missing authoritative BLOCK07 subject-level result files:\n"
            + "\n".join(missing)
        )

    return candidates


def load_block07() -> pd.DataFrame:
    frames = []

    for path, regime in discover_block07_subject_files():
        df = read_csv(path)

        parsed = canonicalize_subject_result_file(
            df,
            regime,
            path,
            "BLOCK07_SUBJECT_RESULTS",
        )

        parsed = validate_authoritative_subject_results(
            parsed,
            DL_MODELS,
            regime,
            f"BLOCK07_{regime}",
        )

        frames.append(parsed)

        log(
            f"BLOCK07 authoritative subject results accepted: "
            f"{path} | {len(parsed)} rows"
        )

    out = pd.concat(frames, ignore_index=True)

    expected_rows = (
        N_SUBJECTS
        * len(CONDITIONS)
        * len(DL_MODELS)
        * len(REGIMES)
    )

    if len(out) != expected_rows:
        fail(
            f"BLOCK07 combined subject-level row count {len(out)} "
            f"!= expected {expected_rows}"
        )

    return out


# ============================================================================
# PREDICTION-LEVEL CONSISTENCY AUDIT
# ============================================================================

def infer_model_from_path(path: Path) -> str:
    """Infer exactly one frozen DL model from a prediction-file path.

    The most specific model token is checked first so that
    CNN-BiLSTM-Attention is never mistaken for CNN-BiLSTM.
    """
    text = str(path).replace("\\", "/").lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)

    ordered = [
        ("cnnbilstmattention", "CNN-BiLSTM-Attention"),
        ("cnnbilstmselfattention", "CNN-BiLSTM-Attention"),
        ("cnnbilstm", "CNN-BiLSTM"),
        ("eegnet", "EEGNet"),
        ("shallowconvnet", "ShallowConvNet"),
        ("deepconvnet", "DeepConvNet"),
    ]

    for token, model in ordered:
        if token in compact:
            return model

    raise ValueError(
        f"Could not infer a frozen DL model from prediction path: {path}."
    )


def infer_model_from_path_or_csv(
    path: Path,
    df: pd.DataFrame,
) -> pd.Series:
    """
    Determine the model label for each row.

    Priority:
      1. Explicit model column in the CSV, if it maps to a known canonical
         model name for every row.
      2. Path-based inference, used only when the CSV column is absent, or
         when the CSV contains values that do not map to known model names.

    Path-based inference is deferred so a valid CSV model column never
    triggers a spurious path-inference failure.
    """
    model_col = find_column(
        df,
        ["model", "model_name", "classifier", "architecture"],
    )

    if model_col is None:
        inferred = infer_model_from_path(path)
        return pd.Series(inferred, index=df.index, dtype="object")

    mapped = df[model_col].map(normalize_model_name)

    known_models = set(DL_MODELS + ML_MODELS)
    missing = mapped.isna() | ~mapped.isin(known_models)

    if missing.any():
        inferred = infer_model_from_path(path)
        mapped = mapped.astype("object")
        mapped.loc[missing] = inferred

    return mapped


def prediction_metrics_for_subject(
    df: pd.DataFrame,
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        cohen_kappa_score,
    )

    y_true = pd.to_numeric(
        df["true_label"],
        errors="coerce",
    ).to_numpy()

    y_pred = pd.to_numeric(
        df["pred_label"],
        errors="coerce",
    ).to_numpy()

    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Non-numeric prediction labels.")

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    if set(y_true.tolist()) != set(GESTURES):
        raise ValueError(
            "Prediction partition does not contain all seven true gesture classes: "
            f"{sorted(set(y_true.tolist()))}"
        )

    if not set(y_pred.tolist()).issubset(set(GESTURES)):
        raise ValueError(
            "Prediction contains labels outside 1..7: "
            f"{sorted(set(y_pred.tolist()) - set(GESTURES))}"
        )

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Macro-F1": float(
            f1_score(
                y_true,
                y_pred,
                labels=GESTURES,
                average="macro",
                zero_division=0,
            )
        ),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Cohen's kappa": float(
            cohen_kappa_score(y_true, y_pred, labels=GESTURES)
        ),
    }


def load_prediction_file_for_audit(
    path: Path,
    regime: str,
) -> pd.DataFrame:
    df = read_csv(path)

    subject_col = find_column(
        df,
        [
            "subject_id",
            "subject",
            "held_out_subject",
            "heldout_subject",
            "test_subject",
            "participant_id",
            "participant",
        ],
    )
    condition_col = find_column(
        df,
        ["condition", "cond", "motor_condition", "task", "state"],
    )
    true_col = find_column(
        df,
        [
            "gesture_true",
            "true_label",
            "true",
            "y_true",
            "actual",
            "target",
            "label",
            "y",
        ],
    )
    pred_col = find_column(
        df,
        [
            "gesture_pred",
            "predicted_label",
            "prediction",
            "pred",
            "y_pred",
            "pred_label",
            "output",
        ],
    )

    if not all(
        c is not None
        for c in [subject_col, condition_col, true_col, pred_col]
    ):
        return pd.DataFrame()

    model_values = infer_model_from_path_or_csv(path, df)

    out = pd.DataFrame({
        "subject_id": df[subject_col].map(normalize_subject),
        "condition": df[condition_col].map(normalize_condition),
        "true_label": pd.to_numeric(df[true_col], errors="coerce"),
        "pred_label": pd.to_numeric(df[pred_col], errors="coerce"),
        "regime": regime,
        "model": model_values,
        "source_file": str(path),
    })

    for alias in ["window_id", "source_file", "source_feature_file", "set"]:
        col = find_column(df, [alias])
        if col is not None and alias not in out.columns:
            out[alias] = df[col].astype(str)

    return out


def discover_all_prediction_files() -> list[tuple[Path, str, str]]:
    """
    Discover only BLOCK07 DL prediction files for the consistency audit.

    BLOCK06 authoritative ML subject-level results are sufficient for ML
    inference and BLOCK06 is not required to emit per-window prediction CSVs.
    """
    found: list[tuple[Path, str, str]] = []

    roots = [
        (BLOCK07_ROOT / "07A_DL_SD", "SD", "BLOCK07"),
        (BLOCK07_ROOT / "07B_DL_LOSO", "LOSO", "BLOCK07"),
    ]

    for root, regime, source in roots:
        if not root.exists():
            fail(f"Missing BLOCK07 prediction root: {root}")

        for path in sorted(root.rglob("*predictions.csv")):
            if path.is_file():
                found.append((path, regime, source))

    return found


def derive_expected_prediction_rows() -> dict[str, int]:
    """
    Derive SD/LOSO prediction row counts from frozen upstream configs.

    No literal 147/441 row counts are used as source of truth. The counts are
    derived from the frozen window count, gesture count, and set count declared
    by the upstream protocol/configuration. The resulting values are asserted
    against the frozen study geometry (7 gestures, 3 sets, 21 windows/recording).
    """
    config_paths = [
        BLOCK07_ROOT / "configs" / "BLOCK07_config.json",
        BLOCK06_ROOT / "BLOCK06_config.json",
    ]

    configs: list[tuple[str, dict[str, Any]]] = []
    for path in config_paths:
        cfg = read_json_if_exists(path)
        if cfg is not None:
            configs.append((str(path), cfg))

    if not configs:
        fail(
            "No readable upstream configuration is available to derive "
            "prediction row counts."
        )

    window_candidates: list[tuple[str, int]] = []
    for path, cfg in configs:
        window_candidates.extend(
            (f"{path}:{where}", value)
            for where, value in _unique_positive_candidates(
                cfg,
                [
                    "windows_per_recording",
                    "windows_per_recording_per_set",
                    "n_windows_per_recording",
                    "window_count_per_recording",
                    "windows_per_file",
                    "windows_per_source_recording",
                ],
            )
        )

    window_values = sorted({value for _, value in window_candidates})
    if window_values != [21]:
        fail(
            "Could not unambiguously derive the frozen windows-per-recording "
            f"value from upstream configs. Candidates={window_candidates}"
        )
    windows_per_recording = window_values[0]

    gesture_candidates: list[tuple[str, int]] = []
    for path, cfg in configs:
        gesture_candidates.extend(
            (f"{path}:{where}", value)
            for where, value in _unique_positive_candidates(
                cfg,
                ["n_gestures", "number_of_gestures", "gesture_count"],
            )
        )

    gesture_values = sorted({value for _, value in gesture_candidates})
    if gesture_values and gesture_values != [len(GESTURES)]:
        fail(
            "Upstream config gesture count conflicts with frozen 7-class design: "
            f"{gesture_candidates}"
        )
    n_gestures = len(GESTURES)

    set_candidates: list[tuple[str, int]] = []
    for path, cfg in configs:
        set_candidates.extend(
            (f"{path}:{where}", value)
            for where, value in _unique_positive_candidates(
                cfg,
                ["n_sets", "number_of_sets", "set_count", "sets_per_recording"],
            )
        )

    set_values = sorted({value for _, value in set_candidates})
    if set_values and set_values != [3]:
        fail(
            "Upstream config set count conflicts with frozen A/B/C design: "
            f"{set_candidates}"
        )
    n_sets = 3

    expected = {
        "SD": windows_per_recording * n_gestures,
        "LOSO": windows_per_recording * n_gestures * n_sets,
    }

    if expected != {"SD": 147, "LOSO": 441}:
        fail(
            "Derived prediction row counts do not match the frozen study geometry: "
            f"{expected}"
        )

    return expected


def audit_predictions_against_authoritative_results(
    authoritative: pd.DataFrame,
) -> dict[str, Any]:
    files = discover_all_prediction_files()

    expected_file_count = (
        len(REGIMES)
        * len(CONDITIONS)
        * len(DL_MODELS)
        * N_SUBJECTS
    )

    expected_prediction_rows = derive_expected_prediction_rows()

    if len(files) != expected_file_count:
        fail(
            f"Prediction file count {len(files)} != expected "
            f"{expected_file_count}."
        )

    logical_rows = []
    mismatch_rows = []
    invalid_rows = []
    duplicate_window_rows = []

    for path, regime, source in files:
        parsed = load_prediction_file_for_audit(path, regime)

        if parsed.empty:
            invalid_rows.append({
                "file": str(path),
                "reason": "prediction_schema_not_matched",
            })
            continue

        if parsed["subject_id"].isna().any():
            invalid_rows.append({
                "file": str(path),
                "reason": "invalid_subject_id",
            })
            continue

        if parsed["condition"].isna().any():
            invalid_rows.append({
                "file": str(path),
                "reason": "invalid_condition",
            })
            continue

        if parsed["model"].isna().any():
            invalid_rows.append({
                "file": str(path),
                "reason": "model_missing",
            })
            continue

        if not parsed["model"].isin(DL_MODELS).all():
            invalid_rows.append({
                "file": str(path),
                "reason": "unexpected_model",
            })
            continue

        subjects = sorted(parsed["subject_id"].unique())
        conditions = sorted(parsed["condition"].unique())
        models = sorted(parsed["model"].unique())

        if len(subjects) != 1 or len(conditions) != 1 or len(models) != 1:
            invalid_rows.append({
                "file": str(path),
                "reason": "multiple_logical_groups_in_one_prediction_file",
            })
            continue

        subject = subjects[0]
        condition = conditions[0]
        model = models[0]

        expected_n = expected_prediction_rows[regime]

        if len(parsed) != expected_n:
            invalid_rows.append({
                "file": str(path),
                "reason": "unexpected_prediction_row_count",
                "observed_rows": len(parsed),
                "expected_rows": expected_n,
            })
            continue

        try:
            derived = prediction_metrics_for_subject(parsed)
        except Exception as e:
            invalid_rows.append({
                "file": str(path),
                "reason": str(e),
            })
            continue

        if "window_id" in parsed.columns:
            dup = parsed.duplicated(["window_id"], keep=False)
            if dup.any():
                duplicate_window_rows.append(
                    parsed.loc[dup].assign(audit_file=str(path))
                )

        logical_key = {
            "subject_id": subject,
            "condition": condition,
            "model": model,
            "regime": regime,
        }

        logical_rows.append({
            **logical_key,
            "source": source,
            "source_file": str(path),
            "prediction_rows": len(parsed),
            **derived,
        })

        auth = authoritative[
            (authoritative["subject_id"] == subject)
            & (authoritative["condition"] == condition)
            & (authoritative["model"] == model)
            & (authoritative["regime"] == regime)
        ]

        if len(auth) != 1:
            invalid_rows.append({
                "file": str(path),
                "reason": "authoritative_result_not_unique",
                **logical_key,
            })
            continue

        auth_row = auth.iloc[0]

        for metric in METRICS:
            diff = float(derived[metric] - float(auth_row[metric]))
            if not np.isclose(
                derived[metric],
                float(auth_row[metric]),
                atol=PREDICTION_METRIC_ATOL,
                rtol=PREDICTION_METRIC_RTOL,
            ):
                mismatch_rows.append({
                    **logical_key,
                    "source_file": str(path),
                    "metric": metric,
                    "prediction_derived": derived[metric],
                    "authoritative_subject_result": float(auth_row[metric]),
                    "absolute_difference": abs(diff),
                })

    logical_df = pd.DataFrame(logical_rows)
    invalid_df = pd.DataFrame(invalid_rows)
    mismatch_df = pd.DataFrame(mismatch_rows)

    if duplicate_window_rows:
        dup_windows = pd.concat(duplicate_window_rows, ignore_index=True)
    else:
        dup_windows = pd.DataFrame()

    csv_write(logical_df, AUDIT_DIR / "prediction_file_inventory.csv")
    csv_write(invalid_df, AUDIT_DIR / "prediction_file_invalid_rows.csv")
    csv_write(mismatch_df, AUDIT_DIR / "prediction_vs_subject_result_mismatches.csv")
    csv_write(dup_windows, AUDIT_DIR / "prediction_duplicate_window_rows.csv")

    key_cols = ["subject_id", "condition", "model", "regime"]

    if not invalid_df.empty:
        fail(
            f"Prediction-level audit found {len(invalid_df)} invalid files/conditions."
        )

    if not mismatch_df.empty:
        fail(
            "Prediction-derived metrics do not exactly match authoritative "
            f"subject results: {len(mismatch_df)} mismatches."
        )

    if not dup_windows.empty:
        fail(
            f"Prediction files contain duplicated window IDs: {len(dup_windows)} rows."
        )

    if len(logical_df) != expected_file_count:
        fail(
            f"Prediction logical inventory {len(logical_df)} != "
            f"expected {expected_file_count}."
        )

    if logical_df.duplicated(key_cols).any():
        fail("Duplicate logical prediction runs detected.")

    expected_keys = pd.MultiIndex.from_product(
        [SUBJECTS, CONDITIONS, DL_MODELS, REGIMES],
        names=key_cols,
    )

    observed_keys = pd.MultiIndex.from_frame(logical_df[key_cols])

    missing = expected_keys.difference(observed_keys)
    unexpected = observed_keys.difference(expected_keys)

    if len(missing):
        fail(f"Missing prediction logical keys: {len(missing)}")
    if len(unexpected):
        fail(f"Unexpected prediction logical keys: {len(unexpected)}")

    return {
        "prediction_files_found": len(files),
        "prediction_files_expected": expected_file_count,
        "invalid_files": len(invalid_df),
        "metric_mismatches": len(mismatch_df),
        "duplicate_window_rows": len(dup_windows),
        "logical_prediction_runs": len(logical_df),
        "prediction_rows_per_run": expected_prediction_rows,
        "metric_tolerance": {
            "absolute": PREDICTION_METRIC_ATOL,
            "relative": PREDICTION_METRIC_RTOL,
        },
        "pass": True,
    }


# ============================================================================
# METRIC VALIDATION
# ============================================================================

def validate_metric_ranges(df: pd.DataFrame, label: str) -> None:
    bounds = {
        "Accuracy": (0.0, 1.0),
        "Balanced Accuracy": (0.0, 1.0),
        "Macro-F1": (0.0, 1.0),
        "MCC": (-1.0, 1.0),
        "Cohen's kappa": (-1.0, 1.0),
    }

    bad_rows = []

    for metric, (lo, hi) in bounds.items():
        if metric not in df.columns:
            continue

        vals = pd.to_numeric(df[metric], errors="coerce")
        invalid = vals.isna() | ~np.isfinite(vals) | (vals < lo) | (vals > hi)

        if invalid.any():
            temp = df.loc[invalid].copy()
            temp["invalid_metric"] = metric
            temp["lower_bound"] = lo
            temp["upper_bound"] = hi
            bad_rows.append(temp)

    if bad_rows:
        bad = pd.concat(bad_rows, ignore_index=True)
        csv_write(
            bad,
            AUDIT_DIR / f"{label}_invalid_metric_values.csv",
        )
        fail(
            f"{label} contains invalid metric values: {len(bad)} rows."
        )


# ============================================================================
# MERGING BLOCK 06 + BLOCK 07
# ============================================================================

def prepare_all_subject_level_results(
    block06: pd.DataFrame,
    block07: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "subject_id",
        "condition",
        "model",
        "regime",
        "source_file",
        "source_type",
    ]

    b06 = block06[cols + METRICS].copy()
    b07 = block07[cols + METRICS].copy()

    b06 = b06[b06["model"].isin(ML_MODELS)].copy()
    b07 = b07[b07["model"].isin(DL_MODELS)].copy()

    all_results = pd.concat([b06, b07], ignore_index=True)

    expected_rows = N_SUBJECTS * len(CONDITIONS) * len(MODELS) * len(REGIMES)

    logical_key = ["subject_id", "condition", "model", "regime"]

    duplicate = all_results.duplicated(logical_key, keep=False)

    if duplicate.any():
        dup = all_results[duplicate].sort_values(logical_key)
        csv_write(dup, AUDIT_DIR / "subject_level_duplicate_keys.csv")
        fail(
            "Duplicate subject-level logical results exist across BLOCK06/BLOCK07."
        )

    expected = pd.MultiIndex.from_product(
        [SUBJECTS, CONDITIONS, MODELS, REGIMES],
        names=logical_key,
    ).to_frame(index=False)

    merged = expected.merge(
        all_results,
        on=logical_key,
        how="left",
        indicator=True,
    )

    missing = merged[merged["_merge"] != "both"].copy()

    if not missing.empty:
        csv_write(missing, AUDIT_DIR / "subject_level_missing_keys.csv")
        fail(
            f"Missing subject-level results: {len(missing)} "
            f"of expected {expected_rows}."
        )

    merged.drop(columns=["_merge"], inplace=True)

    validate_metric_ranges(merged, "SUBJECT_LEVEL")

    if len(merged) != expected_rows:
        fail(
            f"Subject-level row count mismatch: {len(merged)} != {expected_rows}"
        )

    csv_write(
        merged.sort_values(logical_key),
        SUBJECT_DIR / "subject_level_all.csv",
    )
    csv_write(
        merged[merged["regime"] == "SD"].sort_values(logical_key),
        SUBJECT_DIR / "SD_subject_level.csv",
    )
    csv_write(
        merged[merged["regime"] == "LOSO"].sort_values(logical_key),
        SUBJECT_DIR / "LOSO_subject_level.csv",
    )

    return merged.sort_values(logical_key).reset_index(drop=True)


# ============================================================================
# PAIRED ANALYSIS
# ============================================================================

def paired_data_for(
    subject_level: pd.DataFrame,
    regime: str,
    model: str,
    metric: str,
) -> pd.DataFrame:
    sub = subject_level[
        (subject_level["regime"] == regime)
        & (subject_level["model"] == model)
    ][["subject_id", "condition", metric]].copy()

    pivot = sub.pivot(
        index="subject_id",
        columns="condition",
        values=metric,
    ).reset_index()

    if "ME" not in pivot.columns or "MI" not in pivot.columns:
        fail(f"Missing ME/MI pair for {regime}/{model}/{metric}")

    pivot = pivot[["subject_id", "ME", "MI"]].copy()
    pivot.rename(
        columns={"ME": "ME_value", "MI": "MI_value"},
        inplace=True,
    )

    pivot = pivot.sort_values("subject_id").reset_index(drop=True)

    if pivot["subject_id"].duplicated().any():
        fail(f"Duplicate subject pairing in {regime}/{model}/{metric}")

    if pivot["ME_value"].isna().any() or pivot["MI_value"].isna().any():
        missing = pivot[
            pivot["ME_value"].isna() | pivot["MI_value"].isna()
        ]
        csv_write(missing, AUDIT_DIR / "missing_ME_MI_pairs.csv")
        fail(f"Missing ME/MI paired value in {regime}/{model}/{metric}")

    pivot["difference"] = pivot["ME_value"] - pivot["MI_value"]

    if not np.allclose(
        pivot["difference"].to_numpy(),
        pivot["ME_value"].to_numpy() - pivot["MI_value"].to_numpy(),
        atol=1e-12,
        rtol=0.0,
    ):
        fail(f"D_i verification failed for {regime}/{model}/{metric}")

    if len(pivot) > N_SUBJECTS:
        fail(f"n_paired > 25 for {regime}/{model}/{metric}")

    return pivot


def dz_effect_size(d: np.ndarray) -> float:
    if len(d) < 2:
        return float("nan")

    sd = float(np.std(d, ddof=1))

    if sd == 0.0:
        if float(np.mean(d)) == 0.0:
            return float("nan")
        return math.copysign(float("inf"), float(np.mean(d)))

    return float(np.mean(d) / sd)


def paired_bootstrap(
    differences: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """
    Paired-subject bootstrap.

    Returns:
        bootstrap mean-difference distribution (length = iterations)
        bootstrap dz distribution (length = iterations; may contain non-finite
            entries at indices where the resample SD is exactly zero; these are
            retained in the archived array and excluded only from the CI)
        mean-difference CI lower/upper
        dz CI lower/upper
    """
    n = len(differences)

    indices = rng.integers(0, n, size=(iterations, n))

    sampled = differences[indices]
    boot_mean = sampled.mean(axis=1)
    boot_sd = sampled.std(axis=1, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        boot_dz = boot_mean / boot_sd

    finite_dz = boot_dz[np.isfinite(boot_dz)]

    alpha = 1.0 - CONFIDENCE_LEVEL

    mean_lower = float(np.quantile(boot_mean, alpha / 2.0))
    mean_upper = float(np.quantile(boot_mean, 1.0 - alpha / 2.0))

    if len(finite_dz) == 0:
        dz_lower = float("nan")
        dz_upper = float("nan")
    else:
        dz_lower = float(np.quantile(finite_dz, alpha / 2.0))
        dz_upper = float(np.quantile(finite_dz, 1.0 - alpha / 2.0))

    return (
        boot_mean,
        boot_dz,
        mean_lower,
        mean_upper,
        dz_lower,
        dz_upper,
    )


def paired_sign_flip_test(
    differences: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[np.ndarray, float, float]:
    observed = float(np.mean(differences))
    n = len(differences)

    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(iterations, n),
    )

    permuted = (signs * differences[None, :]).mean(axis=1)

    extreme = int(np.count_nonzero(np.abs(permuted) >= abs(observed)))
    p = (extreme + 1.0) / (iterations + 1.0)

    return permuted, observed, float(p)


def analyze_one(
    paired: pd.DataFrame,
    regime: str,
    model: str,
    metric: str,
    rng_boot: np.random.Generator,
    rng_perm: np.random.Generator,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    d = paired["difference"].to_numpy(dtype=float)

    if not np.all(np.isfinite(d)):
        fail(f"Nonfinite paired differences: {regime}/{model}/{metric}")

    me = paired["ME_value"].to_numpy(dtype=float)
    mi = paired["MI_value"].to_numpy(dtype=float)

    (
        boot_mean,
        boot_dz,
        ci_lo,
        ci_hi,
        dz_ci_lo,
        dz_ci_hi,
    ) = paired_bootstrap(d, rng_boot, BOOTSTRAP_ITERATIONS)

    perm, observed, p_raw = paired_sign_flip_test(
        d, rng_perm, PERMUTATION_ITERATIONS
    )

    sd = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
    q25, q75 = np.quantile(d, [0.25, 0.75])

    dz = dz_effect_size(d)

    row = {
        "evaluation_regime": regime,
        "model": model,
        "metric": metric,
        "n_paired": int(len(d)),
        "mean_ME": float(np.mean(me)),
        "mean_MI": float(np.mean(mi)),
        "mean_difference": float(np.mean(d)),
        "median_difference": float(np.median(d)),
        "sd_difference": sd,
        "iqr_difference": float(q75 - q25),
        "effect_size_dz": dz,
        "effect_size_dz_ci_lower": dz_ci_lo,
        "effect_size_dz_ci_upper": dz_ci_hi,
        "bootstrap_ci_lower": ci_lo,
        "bootstrap_ci_upper": ci_hi,
        "observed_statistic": observed,
        "p_raw": p_raw,
        "n_ME_greater": int(np.sum(d > 0)),
        "n_MI_greater": int(np.sum(d < 0)),
        "n_equal": int(np.sum(d == 0)),
        "ME_greater_percent": float(np.mean(d > 0) * 100.0),
        "MI_greater_percent": float(np.mean(d < 0) * 100.0),
        "equal_percent": float(np.mean(d == 0) * 100.0),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "permutation_iterations": PERMUTATION_ITERATIONS,
        "bootstrap_method": "paired_subject_resampling_percentile",
        "permutation_method": "paired_sign_flip_two_sided",
    }

    return row, boot_mean, boot_dz, perm


def run_paired_statistics(
    subject_level: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    summary_rows = []
    bootstrap_mean_rows = []
    bootstrap_dz_rows = []
    permutation_rows = []
    subject_rows = []

    comparison_id = 0

    for regime in REGIMES:
        for model in MODELS:
            for metric in METRICS:
                comparison_id += 1

                paired = paired_data_for(subject_level, regime, model, metric)

                subject_tmp = paired.copy()
                subject_tmp["evaluation_regime"] = regime
                subject_tmp["model"] = model
                subject_tmp["metric"] = metric

                subject_rows.append(
                    subject_tmp[
                        [
                            "subject_id",
                            "evaluation_regime",
                            "model",
                            "metric",
                            "ME_value",
                            "MI_value",
                            "difference",
                        ]
                    ]
                )

                rng_boot = np.random.default_rng(SEED + comparison_id * 100_003)
                rng_perm = np.random.default_rng(SEED + comparison_id * 200_003)

                row, boot_mean, boot_dz, perm = analyze_one(
                    paired, regime, model, metric, rng_boot, rng_perm
                )

                row["comparison_id"] = comparison_id
                summary_rows.append(row)

                bootstrap_mean_rows.append(boot_mean)
                bootstrap_dz_rows.append(boot_dz)
                permutation_rows.append(perm)

    summary = pd.DataFrame(summary_rows)
    subject_df = pd.concat(subject_rows, ignore_index=True)
    bootstrap_mean_array = np.asarray(bootstrap_mean_rows, dtype=float)
    bootstrap_dz_array = np.asarray(bootstrap_dz_rows, dtype=float)
    permutation_array = np.asarray(permutation_rows, dtype=float)

    return (
        summary,
        subject_df,
        bootstrap_mean_array,
        bootstrap_dz_array,
        permutation_array,
    )


# ============================================================================
# FDR
# ============================================================================

def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)

    if np.any(~np.isfinite(p)):
        fail("Nonfinite raw p-value encountered before FDR correction.")

    n = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]

    adjusted = np.empty(n, dtype=float)

    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * n / rank
        running = min(running, value)
        adjusted[i] = running

    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def apply_fdr(summary: pd.DataFrame) -> pd.DataFrame:
    if len(summary) != EXPECTED_TESTS:
        fail(
            f"Expected {EXPECTED_TESTS} statistical tests, "
            f"found {len(summary)}."
        )

    summary = summary.copy()
    summary["p_adjusted"] = benjamini_hochberg(
        summary["p_raw"].to_numpy(dtype=float)
    )
    summary["FDR_significant"] = summary["p_adjusted"] <= FDR_Q

    return summary


# ============================================================================
# TABLES
# ============================================================================

def save_tables(
    paired_subject: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    csv_write(
        paired_subject.sort_values(
            ["evaluation_regime", "model", "metric", "subject_id"]
        ),
        SUBJECT_DIR / "subject_level_ME_MI_difference.csv",
    )

    csv_write(
        paired_subject[
            paired_subject["evaluation_regime"] == "SD"
        ].sort_values(["model", "metric", "subject_id"]),
        SUBJECT_DIR / "SD_ME_MI_difference.csv",
    )

    csv_write(
        paired_subject[
            paired_subject["evaluation_regime"] == "LOSO"
        ].sort_values(["model", "metric", "subject_id"]),
        SUBJECT_DIR / "LOSO_ME_MI_difference.csv",
    )

    summary_sorted = summary.sort_values(
        ["evaluation_regime", "model", "metric"]
    ).reset_index(drop=True)

    csv_write(summary_sorted, STAT_DIR / "ME_MI_statistical_summary.csv")

    csv_write(
        summary_sorted[
            [
                "comparison_id",
                "evaluation_regime",
                "model",
                "metric",
                "n_paired",
                "observed_statistic",
                "p_raw",
            ]
        ],
        STAT_DIR / "paired_test_results.csv",
    )

    csv_write(
        summary_sorted[
            [
                "comparison_id",
                "evaluation_regime",
                "model",
                "metric",
                "mean_difference",
                "median_difference",
                "sd_difference",
                "effect_size_dz",
            ]
        ],
        STAT_DIR / "effect_size_results.csv",
    )

    csv_write(
        summary_sorted[
            [
                "comparison_id",
                "evaluation_regime",
                "model",
                "metric",
                "bootstrap_iterations",
                "bootstrap_method",
                "bootstrap_ci_lower",
                "bootstrap_ci_upper",
                "effect_size_dz",
                "effect_size_dz_ci_lower",
                "effect_size_dz_ci_upper",
            ]
        ],
        STAT_DIR / "bootstrap_results.csv",
    )

    csv_write(
        summary_sorted[
            [
                "comparison_id",
                "evaluation_regime",
                "model",
                "metric",
                "permutation_iterations",
                "permutation_method",
                "observed_statistic",
                "p_raw",
            ]
        ],
        STAT_DIR / "permutation_results.csv",
    )

    csv_write(
        summary_sorted[
            [
                "comparison_id",
                "evaluation_regime",
                "model",
                "metric",
                "p_raw",
                "p_adjusted",
                "FDR_significant",
            ]
        ],
        FDR_DIR / "FDR_adjusted_results.csv",
    )

    csv_write(summary_sorted, TABLE_DIR / "TABLES08_01_complete_100_tests.csv")

    # Main manuscript table: compact, publication-facing Macro-F1 summary.
    # Keep the complete 100-test table separately for transparency.
    primary = summary_sorted[summary_sorted["metric"] == "Macro-F1"].copy()
    primary = primary[[
        "evaluation_regime",
        "model",
        "mean_ME",
        "mean_MI",
        "mean_difference",
        "median_difference",
        "effect_size_dz",
        "effect_size_dz_ci_lower",
        "effect_size_dz_ci_upper",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "p_raw",
        "p_adjusted",
        "FDR_significant",
        "n_paired",
    ]].copy()
    primary = primary.rename(columns={
        "mean_difference": "ME_minus_MI",
        "effect_size_dz": "dz",
        "effect_size_dz_ci_lower": "dz_ci_lower",
        "effect_size_dz_ci_upper": "dz_ci_upper",
        "bootstrap_ci_lower": "difference_ci_lower",
        "bootstrap_ci_upper": "difference_ci_upper",
        "p_raw": "p",
        "p_adjusted": "p_FDR",
    })
    csv_write(primary, TABLE_DIR / "TABLE08_01_primary_MacroF1.csv")

    csv_write(
        paired_subject.sort_values(
            ["evaluation_regime", "model", "metric", "subject_id"]
        ),
        TABLE_DIR / "TABLES08_02_subject_level_results.csv",
    )


def save_resampling_archives(
    bootstrap_mean: np.ndarray,
    bootstrap_dz: np.ndarray,
    permutation: np.ndarray,
) -> None:
    np.savez_compressed(
        STAT_DIR / "bootstrap_distributions.npz",
        bootstrap_mean_difference=bootstrap_mean,
        bootstrap_dz=bootstrap_dz,
        comparison_id=np.arange(1, EXPECTED_TESTS + 1, dtype=np.int32),
    )

    np.savez_compressed(
        STAT_DIR / "permutation_distributions.npz",
        permutation_mean_difference=permutation,
        comparison_id=np.arange(1, EXPECTED_TESTS + 1, dtype=np.int32),
    )


# ============================================================================
# FIGURES
# ============================================================================

def set_publication_style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_paired_macro_f1(paired_subject: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)

    panel_specs = [
        ("SD", "ML", ML_MODELS, axes[0, 0]),
        ("SD", "DL", DL_MODELS, axes[0, 1]),
        ("LOSO", "ML", ML_MODELS, axes[1, 0]),
        ("LOSO", "DL", DL_MODELS, axes[1, 1]),
    ]

    for regime, family, models, ax in panel_specs:
        sub = paired_subject[
            (paired_subject["evaluation_regime"] == regime)
            & (paired_subject["metric"] == "Macro-F1")
            & (paired_subject["model"].isin(models))
        ]

        positions = np.arange(len(models))

        for j, model in enumerate(models):
            p = sub[sub["model"] == model]

            for _, row in p.iterrows():
                ax.plot(
                    [j - 0.16, j + 0.16],
                    [row["ME_value"], row["MI_value"]],
                    linewidth=0.6,
                    alpha=0.28,
                )

            me = p["ME_value"].to_numpy()
            mi = p["MI_value"].to_numpy()

            ax.scatter(
                np.full(len(me), j - 0.16),
                me,
                s=12,
                label="ME" if j == 0 else None,
            )
            ax.scatter(
                np.full(len(mi), j + 0.16),
                mi,
                s=12,
                label="MI" if j == 0 else None,
            )
            ax.plot(
                [j - 0.16, j + 0.16],
                [np.mean(me), np.mean(mi)],
                linewidth=2.2,
                marker="o",
                markersize=5,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(models, rotation=35, ha="right")
        ax.set_title(f"{regime} — {family}")
        ax.set_ylabel("Macro-F1")
        ax.grid(axis="y", alpha=0.18)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    fig.suptitle("Paired Subject-Level ME–MI Macro-F1 Performance", y=1.01)
    save_fig(fig, FIG_DIR / "FIG08_01_paired_ME_MI_performance.png")


def plot_difference_distribution(paired_subject: pd.DataFrame) -> None:
    primary = paired_subject[paired_subject["metric"] == "Macro-F1"].copy()

    combos = [
        ("SD", "ML", ML_MODELS),
        ("SD", "DL", DL_MODELS),
        ("LOSO", "ML", ML_MODELS),
        ("LOSO", "DL", DL_MODELS),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)

    for ax, (regime, family, models) in zip(axes.flat, combos):
        sub = primary[
            (primary["evaluation_regime"] == regime)
            & (primary["model"].isin(models))
        ]

        positions = np.arange(len(models))

        for j, model in enumerate(models):
            d = sub[sub["model"] == model]["difference"].to_numpy()

            x_jitter = np.linspace(-0.10, 0.10, len(d))
            ax.scatter(np.full(len(d), j) + x_jitter, d, s=14, alpha=0.45)

            median = np.median(d)
            q25, q75 = np.quantile(d, [0.25, 0.75])

            ax.plot([j - 0.18, j + 0.18], [median, median], linewidth=2.5)
            ax.plot([j, j], [q25, q75], linewidth=4, alpha=0.65)

        ax.axhline(0.0, linewidth=1.1, linestyle="--")
        ax.set_xticks(positions)
        ax.set_xticklabels(models, rotation=35, ha="right")
        ax.set_title(f"{regime} — {family}")
        ax.set_ylabel("ME − MI Macro-F1")
        ax.grid(axis="y", alpha=0.18)

    fig.suptitle("Subject-Level ME–MI Macro-F1 Differences", y=1.01)
    save_fig(fig, FIG_DIR / "FIG08_02_subject_difference_distribution.png")


def plot_effect_size_forest(summary: pd.DataFrame) -> None:
    sub = summary[summary["metric"] == "Macro-F1"].copy()

    sub["label"] = (
        sub["evaluation_regime"].astype(str)
        + " | "
        + sub["model"].astype(str)
    )

    sub = sub.sort_values(
        ["evaluation_regime", "effect_size_dz"],
        ascending=[True, True],
    ).reset_index(drop=True)

    y = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(10, 10))

    x = sub["effect_size_dz"].to_numpy(dtype=float)
    lo = sub["effect_size_dz_ci_lower"].to_numpy(dtype=float)
    hi = sub["effect_size_dz_ci_upper"].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)

    if valid.any():
        ax.errorbar(
            x[valid],
            y[valid],
            xerr=np.vstack([
                x[valid] - lo[valid],
                hi[valid] - x[valid],
            ]),
            fmt="o",
            markersize=5,
            capsize=3,
            linewidth=1.0,
        )

    invalid = ~valid
    if invalid.any():
        ax.scatter(x[invalid], y[invalid], marker="x", s=35)

    ax.axvline(0.0, linewidth=1.1, linestyle="--")

    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"])
    ax.set_xlabel("Paired effect size (d_z), bootstrap 95% CI")
    ax.set_title("ME–MI Paired Effect Sizes — Macro-F1")
    ax.grid(axis="x", alpha=0.18)

    save_fig(fig, FIG_DIR / "FIG08_03_effect_size_forest.png")


def plot_difference_heatmap(summary: pd.DataFrame) -> None:
    sub = summary[summary["metric"] == "Macro-F1"].copy()

    matrix = sub.pivot(
        index="model",
        columns="evaluation_regime",
        values="mean_difference",
    ).reindex(MODELS)[REGIMES]

    fig, ax = plt.subplots(figsize=(7.5, 8.5))

    vmax = float(np.nanmax(np.abs(matrix.to_numpy())))
    vmax = max(vmax, 1e-9)

    im = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xticks(range(len(REGIMES)))
    ax.set_xticklabels(REGIMES)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(
                j, i, f"{value:.3f}",
                ha="center", va="center", fontsize=8,
            )

    ax.set_xlabel("Evaluation regime")
    ax.set_ylabel("Model")
    ax.set_title("ME–MI Macro-F1 Difference (ME − MI)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean paired difference")

    save_fig(fig, FIG_DIR / "FIG08_04_difference_heatmap.png")


def plot_complete_model_paired(paired_subject: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    axes = axes.ravel()

    for ax, metric in zip(axes[:len(METRICS)], METRICS):
        sub = paired_subject[paired_subject["metric"] == metric]

        combos = [("SD", m) for m in MODELS] + [("LOSO", m) for m in MODELS]

        labels = []

        for j, (regime, model) in enumerate(combos):
            p = sub[
                (sub["evaluation_regime"] == regime)
                & (sub["model"] == model)
            ].sort_values("subject_id")

            labels.append(f"{regime}\n{model}")

            me = p["ME_value"].to_numpy(dtype=float)
            mi = p["MI_value"].to_numpy(dtype=float)

            for k in range(len(me)):
                ax.plot(
                    [j - 0.14, j + 0.14],
                    [me[k], mi[k]],
                    linewidth=0.45,
                    alpha=0.18,
                )

            ax.scatter(
                np.full(len(me), j - 0.14), me, s=8, alpha=0.35,
                label="ME" if j == 0 else None,
            )
            ax.scatter(
                np.full(len(mi), j + 0.14), mi, s=8, alpha=0.35,
                label="MI" if j == 0 else None,
            )
            ax.plot(
                [j - 0.14, j + 0.14],
                [np.mean(me), np.mean(mi)],
                linewidth=2.0, marker="o", markersize=4,
            )

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=75, ha="right")
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.18)

    axes[0].legend(frameon=False)
    if len(axes) > len(METRICS):
        axes[len(METRICS)].axis("off")

    fig.suptitle("Complete Paired ME–MI Model and Metric Comparison", y=0.995)
    save_fig(fig, SUPP_DIR / "FIGS08_01_complete_model_paired.png")


def plot_complete_metric_comparison(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    axes = axes.ravel()

    for ax, metric in zip(axes, METRICS):
        sub = summary[summary["metric"] == metric]

        matrix = sub.pivot(
            index="model",
            columns="evaluation_regime",
            values="mean_difference",
        ).reindex(MODELS)[REGIMES]

        vmax = float(np.nanmax(np.abs(matrix.to_numpy(dtype=float))))
        vmax = max(vmax, 1e-9)

        im = ax.imshow(
            matrix.to_numpy(),
            aspect="auto",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )

        ax.set_xticks(range(2))
        ax.set_xticklabels(REGIMES)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(MODELS)
        ax.set_title(metric)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j, i, f"{matrix.iloc[i, j]:.3f}",
                    ha="center", va="center", fontsize=7,
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean ME − MI")

    axes[-1].axis("off")

    fig.suptitle("Complete ME–MI Difference Across All Metrics", y=0.995)
    save_fig(fig, SUPP_DIR / "FIGS08_02_complete_metric_comparison.png")


def plot_subject_difference_heatmap(paired_subject: pd.DataFrame) -> None:
    sub = paired_subject[paired_subject["metric"] == "Macro-F1"].copy()

    sub["comparison"] = sub["evaluation_regime"] + " | " + sub["model"]

    matrix = sub.pivot(
        index="subject_id",
        columns="comparison",
        values="difference",
    ).reindex(index=SUBJECTS)

    ordered_columns = [
        f"{regime} | {model}"
        for regime in REGIMES
        for model in MODELS
    ]

    matrix = matrix.reindex(columns=ordered_columns)

    fig, ax = plt.subplots(figsize=(18, 8))

    vmax = float(np.nanmax(np.abs(matrix.to_numpy())))
    vmax = max(vmax, 1e-9)

    im = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_yticks(range(len(SUBJECTS)))
    ax.set_yticklabels(SUBJECTS)

    ax.set_xticks(range(len(ordered_columns)))
    ax.set_xticklabels(ordered_columns, rotation=80, ha="right", fontsize=7)

    ax.set_xlabel("Evaluation regime | model")
    ax.set_ylabel("Subject")
    ax.set_title("Subject-Level ME–MI Macro-F1 Differences")

    fig.colorbar(im, ax=ax, label="ME − MI")

    save_fig(fig, SUPP_DIR / "FIGS08_03_subject_difference_heatmap.png")


def plot_difference_distributions(paired_subject: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes = axes.ravel()

    for ax, metric in zip(axes, METRICS):
        sub = paired_subject[paired_subject["metric"] == metric]

        combos = [
            ("SD", "ML", ML_MODELS),
            ("SD", "DL", DL_MODELS),
            ("LOSO", "ML", ML_MODELS),
            ("LOSO", "DL", DL_MODELS),
        ]

        for j, (regime, family, models) in enumerate(combos):
            d_all = []
            for model in models:
                d = sub[
                    (sub["evaluation_regime"] == regime)
                    & (sub["model"] == model)
                ]["difference"].to_numpy()
                if len(d):
                    d_all.append(d)

            if not d_all:
                continue

            values = np.concatenate(d_all)

            ax.hist(
                values,
                bins=25,
                alpha=0.28,
                label=f"{regime}-{family}",
            )

        ax.axvline(0.0, linewidth=1.1, linestyle="--")
        ax.set_title(metric)
        ax.set_xlabel("ME − MI")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.18)

    axes[0].legend(frameon=False, fontsize=8)
    axes[-1].axis("off")

    fig.suptitle("Distribution of Subject-Level ME–MI Differences", y=0.995)
    save_fig(fig, SUPP_DIR / "FIGS08_04_difference_distributions.png")


def generate_figures(
    paired_subject: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    set_publication_style()

    plot_paired_macro_f1(paired_subject)
    plot_difference_distribution(paired_subject)
    plot_effect_size_forest(summary)
    plot_difference_heatmap(summary)

    plot_complete_model_paired(paired_subject)
    plot_complete_metric_comparison(summary)
    plot_subject_difference_heatmap(paired_subject)
    plot_difference_distributions(paired_subject)


# ============================================================================
# AUDIT
# ============================================================================

def build_audit(
    subject_level: pd.DataFrame,
    paired_subject: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    checks["subject_count"] = (
        sorted(subject_level["subject_id"].unique().tolist()) == SUBJECTS
    )
    checks["condition_set"] = (
        sorted(subject_level["condition"].unique().tolist()) == sorted(CONDITIONS)
    )
    checks["regime_set"] = (
        sorted(subject_level["regime"].unique().tolist()) == sorted(REGIMES)
    )
    checks["model_set"] = (
        sorted(subject_level["model"].unique().tolist()) == sorted(MODELS)
    )

    checks["metric_count"] = len(METRICS)
    checks["expected_tests"] = EXPECTED_TESTS
    checks["actual_tests"] = len(summary)

    checks["all_n_paired_le_25"] = bool((summary["n_paired"] <= N_SUBJECTS).all())
    checks["all_n_paired_equal_25"] = bool((summary["n_paired"] == N_SUBJECTS).all())

    checks["difference_formula_verified"] = bool(
        np.allclose(
            paired_subject["difference"].to_numpy(dtype=float),
            (
                paired_subject["ME_value"].to_numpy(dtype=float)
                - paired_subject["MI_value"].to_numpy(dtype=float)
            ),
            atol=1e-12,
            rtol=0.0,
        )
    )

    checks["no_duplicate_subject_level"] = not subject_level.duplicated(
        ["subject_id", "condition", "model", "regime"]
    ).any()

    checks["no_duplicate_paired_rows"] = not paired_subject.duplicated(
        ["subject_id", "evaluation_regime", "model", "metric"]
    ).any()

    checks["all_metric_values_finite"] = bool(
        np.isfinite(subject_level[METRICS].to_numpy(dtype=float)).all()
    )

    checks["all_summary_p_values_finite"] = bool(
        np.isfinite(summary["p_raw"].to_numpy(dtype=float)).all()
    )
    checks["all_adjusted_p_values_finite"] = bool(
        np.isfinite(summary["p_adjusted"].to_numpy(dtype=float)).all()
    )

    checks["fdr_significance_rule_consistent"] = bool(
        (
            summary["FDR_significant"].to_numpy(dtype=bool)
            == (summary["p_adjusted"].to_numpy(dtype=float) <= FDR_Q)
        ).all()
    )

    checks["fdr_applied_to_all_tests"] = (
        summary["p_adjusted"].notna().all()
        and len(summary) == EXPECTED_TESTS
    )

    checks["subject_level_rows"] = len(subject_level)
    checks["subject_level_expected_rows"] = (
        N_SUBJECTS * len(CONDITIONS) * len(MODELS) * len(REGIMES)
    )
    checks["paired_subject_rows"] = len(paired_subject)
    checks["paired_subject_expected_rows"] = EXPECTED_TESTS * N_SUBJECTS

    primary = summary[summary["metric"] == "Macro-F1"]
    checks["primary_macro_f1_rows"] = len(primary)
    checks["primary_macro_f1_expected_rows"] = len(REGIMES) * len(MODELS)

    required_summary_columns = [
        "evaluation_regime",
        "model",
        "metric",
        "n_paired",
        "mean_ME",
        "mean_MI",
        "mean_difference",
        "median_difference",
        "sd_difference",
        "iqr_difference",
        "effect_size_dz",
        "effect_size_dz_ci_lower",
        "effect_size_dz_ci_upper",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "p_raw",
        "p_adjusted",
        "FDR_significant",
    ]

    checks["summary_required_columns"] = all(
        c in summary.columns for c in required_summary_columns
    )

    expected_test_keys = pd.MultiIndex.from_product(
        [REGIMES, MODELS, METRICS],
        names=["evaluation_regime", "model", "metric"],
    )
    observed_test_keys = pd.MultiIndex.from_frame(
        summary[["evaluation_regime", "model", "metric"]]
    )

    checks["all_expected_test_keys_present"] = (
        len(observed_test_keys) == EXPECTED_TESTS
        and len(set(observed_test_keys)) == EXPECTED_TESTS
        and len(expected_test_keys.difference(observed_test_keys)) == 0
    )

    required_figures = [
        FIG_DIR / "FIG08_01_paired_ME_MI_performance.png",
        FIG_DIR / "FIG08_02_subject_difference_distribution.png",
        FIG_DIR / "FIG08_03_effect_size_forest.png",
        FIG_DIR / "FIG08_04_difference_heatmap.png",
        SUPP_DIR / "FIGS08_01_complete_model_paired.png",
        SUPP_DIR / "FIGS08_02_complete_metric_comparison.png",
        SUPP_DIR / "FIGS08_03_subject_difference_heatmap.png",
        SUPP_DIR / "FIGS08_04_difference_distributions.png",
    ]

    figure_status = {}
    for path in required_figures:
        ok = path.exists() and path.stat().st_size > 0
        figure_status[str(path.relative_to(BLOCK08_ROOT))] = ok

    checks["figure_status"] = figure_status
    checks["all_figures_exist"] = all(figure_status.values())

    required_tables = [
        SUBJECT_DIR / "subject_level_ME_MI_difference.csv",
        SUBJECT_DIR / "SD_ME_MI_difference.csv",
        SUBJECT_DIR / "LOSO_ME_MI_difference.csv",
        STAT_DIR / "ME_MI_statistical_summary.csv",
        STAT_DIR / "paired_test_results.csv",
        STAT_DIR / "effect_size_results.csv",
        STAT_DIR / "bootstrap_results.csv",
        STAT_DIR / "permutation_results.csv",
        FDR_DIR / "FDR_adjusted_results.csv",
        FDR_DIR / "FDR_family_definition.json",
        TABLE_DIR / "TABLE08_01_primary_MacroF1.csv",
        TABLE_DIR / "TABLES08_01_complete_100_tests.csv",
        TABLE_DIR / "TABLES08_02_subject_level_results.csv",
    ]

    table_status = {}
    for path in required_tables:
        ok = path.exists() and path.stat().st_size > 0
        table_status[str(path.relative_to(BLOCK08_ROOT))] = ok

    checks["table_status"] = table_status
    checks["all_tables_exist"] = all(table_status.values())

    required_archives = [
        STAT_DIR / "bootstrap_distributions.npz",
        STAT_DIR / "permutation_distributions.npz",
        CONFIG_DIR / "BLOCK08_config.json",
    ]

    archive_status = {}
    for path in required_archives:
        ok = path.exists() and path.stat().st_size > 0
        archive_status[str(path.relative_to(BLOCK08_ROOT))] = ok

    checks["archive_status"] = archive_status
    checks["all_archives_exist"] = all(archive_status.values())

    # Verify that the compact resampling archives contain exactly the frozen
    # 100 comparisons × 10,000 resamples, rather than merely existing.
    archive_shape_ok = True
    archive_iteration_ok = True
    try:
        boot = np.load(STAT_DIR / "bootstrap_distributions.npz")
        perm = np.load(STAT_DIR / "permutation_distributions.npz")
        boot_mean = boot["bootstrap_mean_difference"]
        boot_dz = boot["bootstrap_dz"]
        perm_dist = perm["permutation_mean_difference"]
        boot_ids = boot["comparison_id"]
        perm_ids = perm["comparison_id"]
        archive_shape_ok = (
            boot_mean.shape == (EXPECTED_TESTS, BOOTSTRAP_ITERATIONS)
            and boot_dz.shape == (EXPECTED_TESTS, BOOTSTRAP_ITERATIONS)
            and perm_dist.shape == (EXPECTED_TESTS, PERMUTATION_ITERATIONS)
            and boot_ids.tolist() == list(range(1, EXPECTED_TESTS + 1))
            and perm_ids.tolist() == list(range(1, EXPECTED_TESTS + 1))
        )
        archive_iteration_ok = True
    except Exception as exc:
        archive_shape_ok = False
        archive_iteration_ok = False
        checks["archive_shape_error"] = repr(exc)

    checks["resampling_archive_shape_verified"] = bool(archive_shape_ok)
    checks["resampling_archive_iteration_verified"] = bool(archive_iteration_ok)

    primary_table = TABLE_DIR / "TABLE08_01_primary_MacroF1.csv"
    primary_table_ok = False
    if primary_table.exists() and primary_table.stat().st_size > 0:
        try:
            pt = pd.read_csv(primary_table)
            required_primary_cols = {
                "evaluation_regime", "model", "mean_ME", "mean_MI",
                "ME_minus_MI", "median_difference", "dz",
                "dz_ci_lower", "dz_ci_upper",
                "difference_ci_lower", "difference_ci_upper",
                "p", "p_FDR", "FDR_significant", "n_paired",
            }
            primary_table_ok = (
                len(pt) == len(REGIMES) * len(MODELS)
                and required_primary_cols.issubset(pt.columns)
                and pt["n_paired"].eq(N_SUBJECTS).all()
            )
        except Exception as exc:
            checks["primary_table_error"] = repr(exc)

    checks["primary_table_valid"] = bool(primary_table_ok)

    checks["pass"] = all([
        checks["subject_count"],
        checks["condition_set"],
        checks["regime_set"],
        checks["model_set"],
        checks["actual_tests"] == EXPECTED_TESTS,
        checks["subject_level_rows"] == checks["subject_level_expected_rows"],
        checks["paired_subject_rows"] == checks["paired_subject_expected_rows"],
        checks["primary_macro_f1_rows"] == checks["primary_macro_f1_expected_rows"],
        checks["summary_required_columns"],
        checks["all_expected_test_keys_present"],
        checks["all_n_paired_le_25"],
        checks["all_n_paired_equal_25"],
        checks["difference_formula_verified"],
        checks["no_duplicate_subject_level"],
        checks["no_duplicate_paired_rows"],
        checks["all_metric_values_finite"],
        checks["all_summary_p_values_finite"],
        checks["all_adjusted_p_values_finite"],
        checks["fdr_significance_rule_consistent"],
        checks["fdr_applied_to_all_tests"],
        checks["all_figures_exist"],
        checks["all_tables_exist"],
        checks["all_archives_exist"],
        checks["resampling_archive_shape_verified"],
        checks["resampling_archive_iteration_verified"],
        checks["primary_table_valid"],
    ])

    return checks


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def dataframe_hash(df: pd.DataFrame) -> str:
    canonical = df.copy()
    canonical = canonical.sort_index(axis=1)

    for col in canonical.columns:
        if canonical[col].dtype == object:
            canonical[col] = canonical[col].map(
                lambda x: "" if pd.isna(x) else str(x)
            )

    payload = pd.util.hash_pandas_object(
        canonical, index=True,
    ).to_numpy(dtype=np.uint64).tobytes()

    return hashlib.sha256(payload).hexdigest()


def reproducibility_check(
    subject_level: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    """
    Recompute all 100 summary statistics using the exact frozen seed stream.
    The second run does not write files and is compared against the first run.
    """
    rerun_rows = []
    comparison_id = 0

    for regime in REGIMES:
        for model in MODELS:
            for metric in METRICS:
                comparison_id += 1

                paired = paired_data_for(subject_level, regime, model, metric)

                rng_boot = np.random.default_rng(SEED + comparison_id * 100_003)
                rng_perm = np.random.default_rng(SEED + comparison_id * 200_003)

                row, _, _, _ = analyze_one(
                    paired, regime, model, metric, rng_boot, rng_perm
                )

                row["comparison_id"] = comparison_id
                rerun_rows.append(row)

    rerun = pd.DataFrame(rerun_rows).sort_values("comparison_id").reset_index(drop=True)
    first = summary.sort_values("comparison_id").reset_index(drop=True)

    compare_cols = [
        "comparison_id",
        "n_paired",
        "mean_ME",
        "mean_MI",
        "mean_difference",
        "median_difference",
        "sd_difference",
        "iqr_difference",
        "effect_size_dz",
        "effect_size_dz_ci_lower",
        "effect_size_dz_ci_upper",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "observed_statistic",
        "p_raw",
    ]

    a = first[compare_cols].copy()
    b = rerun[compare_cols].copy()

    numeric_cols = [
        c for c in compare_cols
        if c != "comparison_id" and c != "n_paired"
    ]

    same = (
        a["comparison_id"].equals(b["comparison_id"])
        and a["n_paired"].equals(b["n_paired"])
        and np.allclose(
            a[numeric_cols].to_numpy(dtype=float),
            b[numeric_cols].to_numpy(dtype=float),
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
    )

    result = {
        "reproducibility_pass": bool(same),
        "seed": SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "permutation_iterations": PERMUTATION_ITERATIONS,
        "first_run_hash": dataframe_hash(a),
        "repeat_run_hash": dataframe_hash(b),
    }

    json_dump(result, AUDIT_DIR / "BLOCK08_reproducibility.json")

    return result


# ============================================================================
# CONFIGURATION / SUMMARY
# ============================================================================

def build_config(
    upstream_checks: dict[str, Any],
    upstream_protocol_checks: dict[str, Any],
    prediction_rows_per_run: dict[str, int],
) -> dict[str, Any]:
    return {
        "block": BLOCK_NAME,
        "status": "FROZEN",
        "created_utc": UTC_START,
        "cohort": {
            "subjects": N_SUBJECTS,
            "subject_ids": SUBJECTS,
            "conditions": CONDITIONS,
        },
        "evaluation_regimes": REGIMES,
        "models": {
            "ML": ML_MODELS,
            "DL": DL_MODELS,
            "all": MODELS,
        },
        "metrics": {
            "all": METRICS,
            "primary": PRIMARY_METRICS,
            "secondary": SECONDARY_METRICS,
        },
        "statistical_unit": "subject",
        "pairing_key": "subject_id",
        "paired_difference": "ME - MI",
        "effect_size": "paired dz = mean(D) / SD(D)",
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "method": "paired subject resampling, percentile CI",
            "confidence_level": CONFIDENCE_LEVEL,
        },
        "permutation": {
            "iterations": PERMUTATION_ITERATIONS,
            "method": "paired sign flip",
            "alternative": "two-sided",
            "p_value_correction": "+1 numerator/+1 denominator",
        },
        "multiple_comparisons": {
            "number_of_predefined_tests": EXPECTED_TESTS,
            "method": "Benjamini-Hochberg FDR",
            "q": FDR_Q,
            "primary_family": "all 100 predefined comparisons",
        },
        "random_seed": SEED,
        "no_retraining": True,
        "no_test_tuning": True,
        "no_subject_exclusion": True,
        "no_window_level_inference": True,
        "no_downstream_information_used": True,
        "upstream_completion_checks": upstream_checks,
        "upstream_protocol_configuration_checks": upstream_protocol_checks,
        "authoritative_metric_sources": {
            "ML": "BLOCK06 subject_results.csv",
            "DL": "BLOCK07 subject_results.csv",
        },
        "prediction_consistency_audit": {
            "source": "BLOCK07_DL_prediction_files_only",
            "all_frozen_BLOCK07_prediction_files_checked": True,
            "prediction_metrics_recomputed_only_for_audit": True,
            "required_prediction_rows_per_run": prediction_rows_per_run,
            "metric_tolerance": {
                "absolute": PREDICTION_METRIC_ATOL,
                "relative": PREDICTION_METRIC_RTOL,
            },
            "bootstrap_dz_note": (
                "bootstrap_dz may contain non-finite entries at resample "
                "indices where SD(D*) == 0; these are retained in the "
                "archived npz for reproducibility and excluded only from CI "
                "computation."
            ),
        },
        "downstream_blocks_not_used": [
            "09_TEMPORAL_CALIBRATION",
            "10_ME_MI_TRANSFER",
            "11_NEURO",
            "12_REPRESENTATION",
            "13_XAI",
            "14_ROBUSTNESS",
            "15_FINAL_STATISTICS",
        ],
    }


def build_summary(
    audit: dict[str, Any],
    prediction_audit: dict[str, Any],
    reproducibility: dict[str, Any],
    summary: pd.DataFrame,
    config_hash: str,
    input_hash: str,
) -> dict[str, Any]:
    return {
        "status": STATUS_PASS if (
            audit["pass"]
            and prediction_audit["pass"]
            and reproducibility["reproducibility_pass"]
        ) else STATUS_FAIL,
        "block": BLOCK_NAME,
        "created_utc": UTC_START,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "number_of_subjects": N_SUBJECTS,
        "number_of_paired_subjects": int(summary["n_paired"].min()),
        "number_of_models": len(MODELS),
        "number_of_metrics": len(METRICS),
        "number_of_evaluation_regimes": len(REGIMES),
        "number_of_primary_tests": EXPECTED_TESTS,
        "primary_test": "two-sided paired sign-flip permutation",
        "effect_size": "paired dz",
        "bootstrap_method": "paired subject resampling percentile",
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "permutation_method": "paired sign flip",
        "permutation_iterations": PERMUTATION_ITERATIONS,
        "FDR_method": "Benjamini-Hochberg",
        "FDR_threshold": FDR_Q,
        "significant_tests_raw": int((summary["p_raw"] <= ALPHA).sum()),
        "significant_tests_adjusted": int((summary["p_adjusted"] <= FDR_Q).sum()),
        "configuration_hash": config_hash,
        "input_hash": input_hash,
        "audit_pass": audit["pass"],
        "prediction_audit_pass": prediction_audit["pass"],
        "prediction_files_checked": prediction_audit["prediction_files_found"],
        "prediction_files_expected": prediction_audit["prediction_files_expected"],
        "prediction_metric_mismatches": prediction_audit["metric_mismatches"],
        "prediction_rows_per_run": prediction_audit.get("prediction_rows_per_run"),
        "reproducibility_pass": reproducibility["reproducibility_pass"],
    }


# ============================================================================
# OUTPUT HASH MANIFEST
# ============================================================================

def build_output_hash_manifest() -> pd.DataFrame:
    rows = []

    for p in sorted(BLOCK08_ROOT.rglob("*")):
        if p.is_file():
            try:
                rows.append({
                    "relative_path": str(p.relative_to(BLOCK08_ROOT)),
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                })
            except Exception:
                pass

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df[
            df["relative_path"] != "audit/BLOCK08_output_hash_manifest.csv"
        ]

    csv_write(df, AUDIT_DIR / "BLOCK08_output_hash_manifest.csv")
    return df


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    try:
        print("=" * 88)
        print("BLOCK 08 — ME–MI STATISTICAL ANALYSIS")
        print("=" * 88)
        log(f"Project root: {PROJECT_ROOT}")
        log(f"Output root:  {BLOCK08_ROOT}")
        log(f"Expected comparisons: {EXPECTED_TESTS}")
        log(f"Seed: {SEED}")
        log(f"Bootstrap iterations: {BOOTSTRAP_ITERATIONS}")
        log(f"Permutation iterations: {PERMUTATION_ITERATIONS}")

        prepare_output()

        # ------------------------------------------------------------------
        # 1. Verify upstream completion and protocol configuration
        # ------------------------------------------------------------------
        upstream_checks = verify_upstream_completion()
        json_dump(
            upstream_checks,
            AUDIT_DIR / "upstream_completion_checks.json",
        )

        upstream_protocol_checks = verify_upstream_protocol_configuration()
        json_dump(
            upstream_protocol_checks,
            AUDIT_DIR / "upstream_protocol_configuration_checks.json",
        )

        # ------------------------------------------------------------------
        # 2. Load frozen BLOCK06 authoritative subject-level ML results
        # ------------------------------------------------------------------
        log("Loading BLOCK06 authoritative subject-level ML results...")
        block06 = load_block06()

        log(f"BLOCK06 subject-level rows: {len(block06)}")

        # ------------------------------------------------------------------
        # 3. Load frozen BLOCK07 authoritative subject-level DL results
        # ------------------------------------------------------------------
        log("Loading BLOCK07 authoritative subject-level DL results...")
        block07 = load_block07()

        log(f"BLOCK07 subject-level rows: {len(block07)}")

        # ------------------------------------------------------------------
        # 4. Combine all frozen subject-level results
        # ------------------------------------------------------------------
        log("Validating complete subject-level design...")
        subject_level = prepare_all_subject_level_results(block06, block07)

        # ------------------------------------------------------------------
        # 4b. Independent prediction-level consistency audit.
        #
        # Authoritative subject-level results are used for inference.
        # Frozen BLOCK07 prediction files are rescored only to verify that
        # they reproduce those authoritative metrics exactly.
        # ------------------------------------------------------------------
        log("Auditing all frozen BLOCK07 prediction files against authoritative subject results...")
        prediction_audit = audit_predictions_against_authoritative_results(subject_level)
        json_dump(
            prediction_audit,
            AUDIT_DIR / "prediction_level_audit.json",
        )

        if not prediction_audit["pass"]:
            fail("Prediction-level consistency audit failed.")

        input_hash = dataframe_hash(
            subject_level[
                [
                    "subject_id",
                    "condition",
                    "model",
                    "regime",
                    *METRICS,
                ]
            ]
        )

        log(
            "Subject-level design PASS: "
            f"{len(subject_level)} rows "
            "(25 subjects × 2 conditions × 10 models × 2 regimes)"
        )

        # ------------------------------------------------------------------
        # 5. Configuration (single source of truth for prediction row counts)
        # ------------------------------------------------------------------
        prediction_rows_per_run = prediction_audit["prediction_rows_per_run"]

        config = build_config(
            upstream_checks,
            upstream_protocol_checks,
            prediction_rows_per_run,
        )
        config_text = json.dumps(config, sort_keys=True, indent=2)
        config_hash = sha256_text(config_text)

        config["configuration_hash"] = config_hash

        json_dump(config, CONFIG_DIR / "BLOCK08_config.json")

        # ------------------------------------------------------------------
        # 6. Paired statistics
        # ------------------------------------------------------------------
        log("Running 100 predefined paired statistical comparisons...")

        (
            summary,
            paired_subject,
            bootstrap_mean,
            bootstrap_dz,
            permutation,
        ) = run_paired_statistics(subject_level)

        # ------------------------------------------------------------------
        # 7. FDR
        # ------------------------------------------------------------------
        log("Applying Benjamini-Hochberg FDR to all 100 tests...")
        summary = apply_fdr(summary)

        fdr_family = {
            "method": "Benjamini-Hochberg FDR",
            "q": FDR_Q,
            "primary_family": "all 100 predefined ME-MI paired comparisons",
            "regimes": REGIMES,
            "models": MODELS,
            "metrics": METRICS,
            "number_of_tests": EXPECTED_TESTS,
            "hypothesis": "ME - MI = 0",
            "adjusted_significance_rule": "p_adjusted <= 0.05",
        }
        json_dump(fdr_family, FDR_DIR / "FDR_family_definition.json")

        # ------------------------------------------------------------------
        # 8. Save tables and complete distributions
        # ------------------------------------------------------------------
        log("Writing statistical tables...")
        save_tables(paired_subject, summary)

        save_resampling_archives(bootstrap_mean, bootstrap_dz, permutation)

        # ------------------------------------------------------------------
        # 9. Figures
        # ------------------------------------------------------------------
        log("Generating publication and supplementary figures...")
        generate_figures(paired_subject, summary)

        # ------------------------------------------------------------------
        # 10. Audit
        # ------------------------------------------------------------------
        log("Running final statistical audit...")
        audit = build_audit(subject_level, paired_subject, summary)

        json_dump(audit, AUDIT_DIR / "BLOCK08_audit.json")

        if not audit["pass"]:
            fail(
                "BLOCK08 statistical audit FAILED. "
                "See audit/BLOCK08_audit.json."
            )

        # ------------------------------------------------------------------
        # 11. Reproducibility
        # ------------------------------------------------------------------
        log("Running deterministic reproducibility check...")
        reproducibility = reproducibility_check(subject_level, summary)

        if not reproducibility["reproducibility_pass"]:
            fail("BLOCK08 reproducibility check FAILED.")

        # ------------------------------------------------------------------
        # 12. Final summary
        # ------------------------------------------------------------------
        final_summary = build_summary(
            audit,
            prediction_audit,
            reproducibility,
            summary,
            config_hash,
            input_hash,
        )

        json_dump(final_summary, BLOCK08_ROOT / "BLOCK08_summary.json")

        # ------------------------------------------------------------------
        # 13. Completion marker
        # ------------------------------------------------------------------
        complete = {
            "status": STATUS_PASS,
            "block": BLOCK_NAME,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": str(PROJECT_ROOT),
            "output_root": str(BLOCK08_ROOT),
            "cohort_size": N_SUBJECTS,
            "conditions": CONDITIONS,
            "evaluation_regimes": REGIMES,
            "models": MODELS,
            "metrics": METRICS,
            "expected_statistical_tests": EXPECTED_TESTS,
            "actual_statistical_tests": len(summary),
            "paired_subject_inference": True,
            "window_level_pseudoreplication": False,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "permutation_iterations": PERMUTATION_ITERATIONS,
            "FDR_method": "Benjamini-Hochberg",
            "FDR_q": FDR_Q,
            "primary_test": "two-sided paired sign-flip permutation",
            "effect_size": "paired dz",
            "reproducibility_pass": reproducibility["reproducibility_pass"],
            "audit_pass": audit["pass"],
            "prediction_audit_pass": prediction_audit["pass"],
            "prediction_files_checked": prediction_audit["prediction_files_found"],
            "prediction_files_expected": prediction_audit["prediction_files_expected"],
            "prediction_metric_mismatches": prediction_audit["metric_mismatches"],
            "prediction_rows_per_run": prediction_audit.get("prediction_rows_per_run"),
            "input_hash": input_hash,
            "configuration_hash": config_hash,
            "raw_data_modified": False,
            "previous_blocks_modified": False,
            "downstream_blocks_used": False,
            "next_block": "BLOCK09",
        }

        json_dump(complete, BLOCK08_ROOT / "BLOCK08_complete.json")

        # ------------------------------------------------------------------
        # 14. Final human-readable summary
        # ------------------------------------------------------------------
        runtime_minutes = (time.time() - START_TIME) / 60.0

        with (BLOCK08_ROOT / "BLOCK08_summary.txt").open("w", encoding="utf-8") as f:
            f.write("BLOCK 08 — ME–MI STATISTICAL ANALYSIS\n")
            f.write("=" * 60 + "\n")
            f.write("STATUS: PASS\n")
            f.write(f"Runtime minutes: {runtime_minutes:.2f}\n")
            f.write(f"Subjects: {N_SUBJECTS}\n")
            f.write("Conditions: ME, MI\n")
            f.write("Regimes: SD, LOSO\n")
            f.write(f"Models: {len(MODELS)}\n")
            f.write(f"Metrics: {len(METRICS)}\n")
            f.write(f"Predefined tests: {EXPECTED_TESTS}\n")
            f.write(f"Paired subjects per test: {summary['n_paired'].min()}\n")
            f.write(f"Bootstrap iterations: {BOOTSTRAP_ITERATIONS}\n")
            f.write(f"Permutation iterations: {PERMUTATION_ITERATIONS}\n")
            f.write("FDR: Benjamini-Hochberg, q=0.05\n")
            f.write(f"Raw p <= 0.05: {(summary['p_raw'] <= ALPHA).sum()}\n")
            f.write(f"FDR q <= 0.05: {(summary['p_adjusted'] <= FDR_Q).sum()}\n")
            f.write(f"Reproducibility: {reproducibility['reproducibility_pass']}\n")
            f.write(f"Audit: {audit['pass']}\n")
            f.write("Prediction-level audit: PASS\n")
            f.write("Raw data modified: NO\n")
            f.write("Previous blocks modified: NO\n")
            f.write("Downstream blocks used: NO\n")
            f.write("Next block: BLOCK09\n")

        # ------------------------------------------------------------------
        # 15. Final output hashes
        # ------------------------------------------------------------------
        build_output_hash_manifest()

        print()
        print("=" * 88)
        print("✓ BLOCK 08 PASS — ME–MI STATISTICAL ANALYSIS FROZEN")
        print("=" * 88)
        print(f"Output: {BLOCK08_ROOT}")
        print(f"Comparisons: {EXPECTED_TESTS}")
        print(f"Paired subjects/test: {summary['n_paired'].min()}")
        print(f"Bootstrap: {BOOTSTRAP_ITERATIONS}")
        print(f"Permutation: {PERMUTATION_ITERATIONS}")
        print(f"FDR-significant tests: {(summary['p_adjusted'] <= FDR_Q).sum()}")
        print("Reproducibility: PASS")
        print("Audit: PASS")
        print("Raw data modified: NO")
        print("Previous blocks modified: NO")
        print("Downstream blocks used: NO")
        print("Ready for BLOCK09.")
        print("=" * 88)

        return 0

    except Exception as e:
        runtime_minutes = (time.time() - START_TIME) / 60.0

        try:
            BLOCK08_ROOT.mkdir(parents=True, exist_ok=True)
            failure = {
                "status": STATUS_FAIL,
                "block": BLOCK_NAME,
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_minutes": runtime_minutes,
                "error": repr(e),
                "project_root": str(PROJECT_ROOT),
                "raw_data_modified": False,
                "previous_blocks_modified": False,
            }
            json_dump(failure, BLOCK08_ROOT / "BLOCK08_failure.json")
        except Exception:
            pass

        print()
        print("=" * 88)
        print("✗ BLOCK 08 FAIL")
        print("=" * 88)
        print(str(e))
        print("=" * 88)

        return 1


if __name__ == "__main__":
    raise SystemExit(main())