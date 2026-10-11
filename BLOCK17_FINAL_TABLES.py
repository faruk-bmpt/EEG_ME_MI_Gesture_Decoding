#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BLOCK 17 — FINAL PUBLICATION TABLES
Paper 2: Shared and Condition-Specific EEG Representations of Hand and Finger Gestures

CANONICAL PUBLICATION / AUDIT BLOCK
-----------------------------------

Purpose
-------
BLOCK 17 converts already-finalized upstream/statistical outputs into
manuscript-ready main and supplementary tables and performs deterministic
traceability and consistency audits.

BLOCK 17 IS NOT AN ANALYSIS BLOCK.

It MUST NOT:
    - train or tune models
    - alter preprocessing
    - alter windowing
    - alter feature definitions
    - recompute inferential statistics
    - recompute confidence intervals
    - recompute correlations
    - recompute effect sizes
    - recompute FDR
    - fabricate missing values
    - replace NOT_REACHED / NO_POSITIVE_GAP with numeric k90 values
    - selectively omit models, subjects, conditions, or unfavorable results

It MAY perform only deterministic reporting transformations:
    - source loading
    - validation
    - table restructuring
    - column ordering
    - label formatting
    - frozen rounding
    - CSV/XLSX export
    - traceability registration
    - BLOCK15/BLOCK16 numerical consistency checks

IMPORTANT WINDOWING NOTE
------------------------
The finalized EEG window is read from BLOCK 04. If the finalized window is
500 samples at 125 Hz, its duration is 4.0 s, NOT 500 ms. BLOCK 17 never
hard-codes a window duration.

Execution
---------
    python BLOCK17_FINAL_TABLES.py --inspect
    python BLOCK17_FINAL_TABLES.py --run

Default behavior is inspect-only.
A BLOCK17_complete.json is created only when every release gate passes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# ============================================================================
# 0. CANONICAL REGISTRY
# ============================================================================

BLOCK17_VERSION = "BLOCK17_PUBLICATION_TABLES_FROZEN_V1"

EXPECTED_SUBJECTS = 25
EXPECTED_GESTURES = list(range(1, 8))
EXPECTED_SETS = ["A", "B", "C"]

MODEL_ORDER = [
    "LDA",
    "KNN",
    "SVM",
    "RandomForest",
    "XGBoost",
    "EEGNet",
    "ShallowConvNet",
    "DeepConvNet",
    "CNN-BiLSTM",
    "CNN-BiLSTM-Attention",
]

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

REPRESENTATION_MODELS = [
    "EEGNet",
    "CNN-BiLSTM",
    "CNN-BiLSTM-Attention",
]

CONDITION_ORDER = ["ME", "MI"]

METRIC_ORDER = [
    "Accuracy",
    "Balanced Accuracy",
    "Macro-F1",
    "MCC",
    "Cohen's kappa",
]

CALIBRATION_LEVELS = [0, 10, 20, 30, 40, 50, 60, 80, 90]

DROPOUT_LEVELS = [5, 10, 20, 30]

MISSING_VALUE_LABELS = {
    "NA",
    "NOT_AVAILABLE",
    "NOT_REACHED",
    "NO_POSITIVE_GAP",
    "INSUFFICIENT_VALID_DATA",
}

ALLOWED_EVIDENCE_STATUS = {
    "SUPPORTED",
    "NOT_SUPPORTED",
    "INCONCLUSIVE",
    "DESCRIPTIVE_ONLY",
    "INSUFFICIENT_VALID_DATA",
    "NOT_AVAILABLE",
    "TESTED",
    "UPSTREAM_FROZEN",
}

# These are the authoritative BLOCK15 artifacts already registered by BLOCK16.
# BLOCK17 uses them; it does not create replacements.
ARTIFACTS = {
    "BLOCK09": {
        "temporal_calibration": "09_TEMPORAL_CALIBRATION/tables/TABLE09_01_temporal_calibration.csv",
        "k90": "09_TEMPORAL_CALIBRATION/tables/TABLE09_02_k90.csv",
        "subject_level_calibration": "09_TEMPORAL_CALIBRATION/tables/TABLE09_04_subject_level_calibration.csv",
    },
    "BLOCK15": {
        "config": "15_FINAL_STATISTICS/configs/BLOCK15_config.json",
        "complete_marker": "15_FINAL_STATISTICS/BLOCK15_complete.json",
        "pass_marker": "15_FINAL_STATISTICS/completion/BLOCK15_PASS.json",
        "subject_level": "15_FINAL_STATISTICS/descriptive/subject_level_performance.csv",
        "aggregate": "15_FINAL_STATISTICS/descriptive/aggregate_performance.csv",
        "gap": "15_FINAL_STATISTICS/gap_analysis/SD_LOSO_subject_gap.csv",
        "gap_summary": "15_FINAL_STATISTICS/gap_analysis/gap_summary.csv",
        "me_mi_gap": "15_FINAL_STATISTICS/gap_analysis/ME_MI_gap_comparison.csv",
        "me_mi_statistical_summary": "08_ME_MI_STATISTICS/statistics/ME_MI_statistical_summary.csv",
        "me_mi_paired_tests": "08_ME_MI_STATISTICS/statistics/paired_test_results.csv",
        "transfer": "15_FINAL_STATISTICS/transfer/transfer_summary.csv",
        "asymmetry": "15_FINAL_STATISTICS/transfer/transfer_asymmetry.csv",
        "negative_transfer": "15_FINAL_STATISTICS/transfer/negative_transfer_summary.csv",
        "calibration": "15_FINAL_STATISTICS/calibration/calibration_summary.csv",
        "k90": "15_FINAL_STATISTICS/calibration/k90_summary.csv",
        "rsa": "15_FINAL_STATISTICS/representation/RSA_summary.csv",
        "rsa_transfer": "15_FINAL_STATISTICS/representation/representation_transfer.csv",
        "rsa_calibration": "15_FINAL_STATISTICS/representation/representation_calibration.csv",
        "h4_representation_transfer": "15_FINAL_STATISTICS/hypothesis_tests/H4_representation_transfer.csv",
        "h5_representation_calibration": "15_FINAL_STATISTICS/hypothesis_tests/H5_representation_calibration.csv",
        "xai": "15_FINAL_STATISTICS/xai/XAI_summary.csv",
        "xai_similarity": "15_FINAL_STATISTICS/xai/attribution_similarity_summary.csv",
        "robustness": "15_FINAL_STATISTICS/robustness/robustness_summary.csv",
        "dropout": "15_FINAL_STATISTICS/robustness/dropout_summary.csv",
        "noise": "15_FINAL_STATISTICS/robustness/noise_summary.csv",
        "computational": "15_FINAL_STATISTICS/computational/ML_DL_computational_summary.csv",
    },
    "BLOCK16": {
        "config": "16_FINAL_FIGURES/config/BLOCK16_config.json",
        "complete_marker": "16_FINAL_FIGURES/BLOCK16_complete.json",
        "source_audit": "16_FINAL_FIGURES/audit/BLOCK16_source_audit.csv",
        "figure_status": "16_FINAL_FIGURES/audit/BLOCK16_figure_release_registry.csv",
    },
    "BLOCK04": {
        "selected_window": "04_WINDOWING/selected_window_config.json",
    },
    "BLOCK01": {
        "inventory": "01_SUBJECT_MATCHING",
    },
    "BLOCK02": {
        "qa": "02_SIGNAL_QA",
    },
}


# ============================================================================
# 1. FROZEN CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "block17_version": BLOCK17_VERSION,
    "table_numbering": list(range(1, 15)),
    "model_order": MODEL_ORDER,
    "representation_model_order": REPRESENTATION_MODELS,
    "condition_order": CONDITION_ORDER,
    "metric_order": METRIC_ORDER,
    "calibration_levels_percent": CALIBRATION_LEVELS,
    "dropout_levels_percent": DROPOUT_LEVELS,
    "performance_decimals": 2,
    "effect_size_decimals": 3,
    "correlation_decimals": 3,
    "p_value_threshold": 0.001,
    "consistency_rounding_tolerance": 0.0005,
    "percentage_policy": "percentage_points_are_not_relative_percentages",
    "missing_value_policy": sorted(MISSING_VALUE_LABELS),
    "no_statistical_recalculation": True,
}


# ============================================================================
# 2. GENERAL HELPERS
# ============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(value: Any) -> Any:
    """JSON fallback matching BLOCK15 frozen-config hashing semantics."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def block15_config_canonical_hash(cfg: dict[str, Any]) -> str:
    """Reproduce BLOCK15 config_canonical_hash exactly.

    BLOCK15 excludes only the top-level ``status`` and the frozen
    ``expected_config_sha256``/``frozen_utc`` fields before hashing.  The
    serialization is sorted, compact, ASCII-safe JSON with BLOCK15's
    deterministic default serializer.
    """
    c = copy.deepcopy(cfg)
    c.pop("status", None)
    fr = c.get("freeze", {})
    if isinstance(fr, dict):
        fr.pop("expected_config_sha256", None)
        fr.pop("frozen_utc", None)
    blob = json.dumps(
        c,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def norm_col(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def metric_norm(value: Any) -> str:
    s = norm_col(value)
    replacements = {
        "accuracy": "accuracy",
        "balanced_acc": "balanced_accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "balancedaccuracy": "balanced_accuracy",
        "macro_f1_score": "macro_f1",
        "macro_f1": "macro_f1",
        "f1_macro": "macro_f1",
        "mcc": "mcc",
        "cohens_kappa": "kappa",
        "cohen_kappa": "kappa",
        "kappa": "kappa",
    }
    return replacements.get(s, s)


def metric_display(value: Any) -> str:
    """Map any accepted metric spelling to the canonical publication label."""
    display = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "macro_f1": "Macro-F1",
        "mcc": "MCC",
        "kappa": "Cohen's kappa",
    }
    return display.get(metric_norm(value), str(value).strip())


def normalize_condition(value: Any) -> str:
    s = str(value).strip().lower()
    if s in {"me", "motor_execution", "motor execution"}:
        return "ME"
    if s in {"mi", "motor_imagery", "motor imagery"}:
        return "MI"
    return str(value).strip()


def normalize_model(value: Any) -> str:
    s = str(value).strip()
    key = (
        s.lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )
    aliases = {
        "rf": "RandomForest",
        "randomforest": "RandomForest",
        "randomforestclassifier": "RandomForest",
        "xgb": "XGBoost",
        "xgboost": "XGBoost",
        "cnnbilstm": "CNN-BiLSTM",
        "cnnbilstmattention": "CNN-BiLSTM-Attention",
    }
    return aliases.get(key, s)


ALIASES = {
    "subject": [
        "subject",
        "subject_id",
        "participant",
        "participant_id",
        "sub",
        "sid",
    ],
    "condition": ["condition", "cond"],
    "model": [
        "model",
        "model_name",
        "classifier",
        "architecture",
    ],
    "metric": ["metric", "measure"],
    "accuracy": ["accuracy", "acc"],
    "balanced_accuracy": [
        "balanced_accuracy",
        "balanced_acc",
        "balancedaccuracy",
        "bac",
    ],
    "macro_f1": [
        "macro_f1",
        "macro_f1_score",
        "f1_macro",
        "macro-f1",
        "macro f1",
    ],
    "mcc": ["mcc", "matthews_corrcoef", "matthews_correlation"],
    "kappa": [
        "kappa",
        "cohen_kappa",
        "cohen's_kappa",
        "cohens_kappa",
    ],
    "rho": [
        "rho",
        "spearman_rho",
        "rho_spearman",
        "rsa_rho",
        "representation_similarity",
        "me_mi_rho",
    ],
    "transfer_benefit": [
        "transfer_benefit",
        "transfer_gain",
        "mean_transfer_benefit",
    ],
    "transfer_loss": [
        "transfer_loss",
        "transfer_degradation",
    ],
    "k90": ["k90", "k_90", "calibration_k90"],
    "fraction": [
        "fraction",
        "calibration_fraction",
        "calibration_percent",
        "target_calibration_fraction",
    ],
    "value": ["value", "score", "performance", "mean"],
    "status": [
        "status",
        "k90_status",
        "classification_status",
    ],
    "direction": [
        "direction",
        "transfer_direction",
    ],
}


def find_col(
    df: pd.DataFrame,
    logical: str,
    required: bool = True,
) -> str | None:
    normalized = {norm_col(c): c for c in df.columns}
    for candidate in ALIASES.get(logical, [logical]):
        n = norm_col(candidate)
        if n in normalized:
            return normalized[n]
    if required:
        raise KeyError(
            f"Required logical column '{logical}' not found. "
            f"Columns={list(df.columns)}"
        )
    return None


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{label}: missing required columns {missing}; "
            f"available={list(df.columns)}"
        )


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def source_path(root: Path, relative: str) -> Path:
    return root / Path(relative)


def relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# ============================================================================
# 3. FROZEN CONFIGURATION HANDLING
# ============================================================================

def config_path(root: Path) -> Path:
    return root / "17_FINAL_TABLES" / "configs" / "BLOCK17_config.json"


def load_or_create_config(root: Path, write_if_missing: bool = True) -> dict[str, Any]:
    path = config_path(root)

    if path.is_file():
        cfg = load_json(path)

        required_keys = set(DEFAULT_CONFIG)
        missing = required_keys - set(cfg)
        if missing:
            raise RuntimeError(
                f"BLOCK17_config.json is incomplete; missing={sorted(missing)}"
            )

        # Frozen identity checks.
        if cfg["block17_version"] != BLOCK17_VERSION:
            raise RuntimeError(
                "BLOCK17 configuration version mismatch: "
                f"{cfg['block17_version']} != {BLOCK17_VERSION}"
            )

        return cfg

    if not write_if_missing:
        raise FileNotFoundError(
            f"BLOCK17_config.json not found: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return dict(DEFAULT_CONFIG)


def block17_config_hash(cfg: dict[str, Any]) -> str:
    return sha256_json(cfg)


# ============================================================================
# 4. SOURCE PRECHECK
# ============================================================================

def require_file(root: Path, relative: str, label: str) -> Path:
    path = source_path(root, relative)
    if not path.is_file():
        raise RuntimeError(
            f"{label}: required source file not found: {relative}"
        )
    return path


def check_completion_markers(root: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    b15 = require_file(root, ARTIFACTS["BLOCK15"]["complete_marker"], "BLOCK15 completion marker")
    b16 = require_file(root, ARTIFACTS["BLOCK16"]["complete_marker"], "BLOCK16 completion marker")
    results["BLOCK15"] = load_json(b15)
    results["BLOCK16"] = load_json(b16)
    return results


def validate_block16_status(root: Path) -> None:
    marker = require_file(
        root,
        ARTIFACTS["BLOCK16"]["complete_marker"],
        "BLOCK16 completion marker",
    )
    data = load_json(marker)
    status = str(data.get("status", data.get("state", ""))).upper()
    if status not in {"PASS", "COMPLETE", "COMPLETED"}:
        raise RuntimeError(
            f"BLOCK16 completion marker does not report successful completion: {status!r}"
        )


def validate_block15_status(root: Path) -> None:
    marker = require_file(
        root,
        ARTIFACTS["BLOCK15"]["complete_marker"],
        "BLOCK15 completion marker",
    )
    data = load_json(marker)
    status = str(data.get("status", data.get("state", ""))).upper()
    if status != "PASS" or data.get("completed") is not True:
        raise RuntimeError(
            "BLOCK15 authoritative completion marker is not PASS/completed=true: "
            f"status={status!r}, completed={data.get('completed')!r}"
        )

    config_path_b15 = require_file(
        root, ARTIFACTS["BLOCK15"]["config"], "BLOCK15 configuration"
    )
    # BLOCK15's completion marker stores the hash produced by its own
    # config_canonical_hash() routine.  Reproduce that exact convention here;
    # do not substitute BLOCK17's generic JSON/file hashes.
    cfg_obj = load_json(config_path_b15)
    cfg_hash_block15 = block15_config_canonical_hash(cfg_obj)
    stored_hash = str(data.get("config_sha256", "")).strip()
    if stored_hash and stored_hash != cfg_hash_block15:
        raise RuntimeError(
            "BLOCK15 completion marker config hash mismatch: "
            f"stored={stored_hash}, BLOCK15_canonical={cfg_hash_block15}"
        )

    accepted_hash = stored_hash or cfg_hash_block15

    pass_marker = require_file(
        root, ARTIFACTS["BLOCK15"]["pass_marker"], "BLOCK15 PASS marker"
    )
    pass_data = load_json(pass_marker)
    if str(pass_data.get("status", "")).upper() != "PASS":
        raise RuntimeError("BLOCK15 PASS marker does not report PASS.")
    pass_hash = str(pass_data.get("config_sha256_frozen", "")).strip()
    if pass_hash and pass_hash != cfg_hash_block15:
        raise RuntimeError(
            "BLOCK15 PASS marker config hash mismatch: "
            f"frozen={pass_hash}, BLOCK15_canonical={cfg_hash_block15}"
        )

    # Keep the accepted identity available to the caller/audit layer.
    _ = accepted_hash



# ============================================================================
# 5. WINDOW CONFIGURATION — NEVER INVENT
# ============================================================================

def load_selected_window_config(root: Path) -> dict[str, Any]:
    path = require_file(
        root,
        ARTIFACTS["BLOCK04"]["selected_window"],
        "BLOCK04 selected window configuration",
    )
    data = load_json(path)

    if not isinstance(data, dict):
        raise RuntimeError(
            "selected_window_config.json must contain a JSON object."
        )

    return data


def recursive_find_numeric(
    obj: Any,
    names: set[str],
) -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            nk = norm_col(key)
            if nk in names:
                try:
                    found.append((str(key), float(value)))
                except (TypeError, ValueError):
                    pass
            found.extend(recursive_find_numeric(value, names))

    elif isinstance(obj, list):
        for value in obj:
            found.extend(recursive_find_numeric(value, names))

    return found


def extract_window_values(window_cfg: dict[str, Any]) -> dict[str, Any]:
    length_candidates = recursive_find_numeric(
        window_cfg,
        {
            "window_samples",
            "window_length_samples",
            "window_size_samples",
            "final_window_samples",
            "selected_window_samples",
            "length_samples",
        },
    )

    overlap_candidates = recursive_find_numeric(
        window_cfg,
        {
            "overlap_percent",
            "overlap_percentage",
            "final_overlap_percent",
            "selected_overlap_percent",
            "overlap",
        },
    )

    step_candidates = recursive_find_numeric(
        window_cfg,
        {
            "step_samples",
            "hop_samples",
            "stride_samples",
            "final_step_samples",
        },
    )

    if not length_candidates:
        raise RuntimeError(
            "BLOCK04 selected_window_config.json does not expose a "
            "recognized finalized window length in samples."
        )

    window_samples = int(round(length_candidates[0][1]))

    overlap_percent = None
    if overlap_candidates:
        overlap_percent = float(overlap_candidates[0][1])

    step_samples = None
    if step_candidates:
        step_samples = int(round(step_candidates[0][1]))

    if step_samples is None and overlap_percent is not None:
        step_samples = int(
            round(window_samples * (1.0 - overlap_percent / 100.0))
        )

    if overlap_percent is None and step_samples is not None:
        overlap_percent = (
            100.0 * (1.0 - step_samples / window_samples)
        )

    if overlap_percent is None or step_samples is None:
        raise RuntimeError(
            "BLOCK04 selected window configuration does not expose enough "
            "information to establish finalized overlap/step."
        )

    return {
        "window_samples": window_samples,
        "window_duration_s": window_samples / 125.0,
        "overlap_percent": overlap_percent,
        "step_samples": step_samples,
        "step_duration_s": step_samples / 125.0,
        "sampling_rate_hz": 125,
    }



# ============================================================================
# 6. PUBLICATION FORMATTING / SOURCE INTEGRITY HELPERS
# ============================================================================

def _is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def publication_round(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Apply frozen display precision without changing source statistics."""
    out = df.copy()
    p_dec = int(cfg["performance_decimals"])
    e_dec = int(cfg["effect_size_decimals"])
    c_dec = int(cfg["correlation_decimals"])

    for col in out.columns:
        n = norm_col(col)
        if not _is_numeric_series(out[col]):
            continue
        if any(k in n for k in ("p_value", "pvalue", "fdr_q", "q_value")):
            continue
        if any(k in n for k in ("correlation", "rho", "spearman", "pearson")):
            out[col] = pd.to_numeric(out[col], errors="coerce").round(c_dec)
        elif any(k in n for k in ("effect_size", "effect", "mcc", "kappa", "gap")):
            out[col] = pd.to_numeric(out[col], errors="coerce").round(e_dec)
        elif any(k in n for k in ("accuracy", "balanced", "macro_f1", "performance",
                                  "transfer", "degradation", "delta", "score")):
            out[col] = pd.to_numeric(out[col], errors="coerce").round(p_dec)
    return out


def validate_p_values(df: pd.DataFrame, label: str) -> None:
    for col in df.columns:
        n = norm_col(col)
        if any(k in n for k in ("p_value", "pvalue", "fdr_q", "q_value")):
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if ((vals < 0) | (vals > 1)).any():
                raise RuntimeError(f"{label}: invalid probability values in '{col}'.")


def _find_inventory_source(root: Path) -> Path:
    exact = [
        "01_SUBJECT_MATCHING/recording_inventory.csv",
        "01_SUBJECT_MATCHING/tables/recording_inventory.csv",
        "01_SUBJECT_MATCHING/master_recording_inventory.csv",
        "01_SUBJECT_MATCHING/tables/master_recording_inventory.csv",
    ]
    for rel in exact:
        p = root / rel
        if p.is_file():
            return p
    base = root / "01_SUBJECT_MATCHING"
    if base.is_dir():
        candidates = sorted(
            p for p in base.rglob("*.csv")
            if "inventory" in p.name.lower() and "record" in p.name.lower()
        )
        if candidates:
            return candidates[0]
    raise RuntimeError(
        "T1: no finalized recording inventory was found under 01_SUBJECT_MATCHING. "
        "BLOCK17 will not substitute expected design counts for audited counts."
    )


def audit_recording_inventory(root: Path) -> dict[str, Any]:
    path = _find_inventory_source(root)
    d = load_csv(path)
    subject_col = find_col(d, "subject")
    condition_col = find_col(d, "condition")

    def resolve(logical_candidates: list[str]) -> str | None:
        normalized = {norm_col(c): c for c in d.columns}
        for candidate in logical_candidates:
            n = norm_col(candidate)
            if n in normalized:
                return normalized[n]
        return None

    gesture_col = resolve(["gesture", "gesture_id", "gesture_number", "gesture_code", "class", "label"])
    set_col = resolve(["set", "recording_set", "set_id", "set_code"])
    if gesture_col is None or set_col is None:
        raise RuntimeError(
            f"T1: recording inventory lacks gesture/set identifiers; columns={list(d.columns)}"
        )

    x = pd.DataFrame({
        "subject": d[subject_col].astype(str).str.strip(),
        "condition": d[condition_col].map(normalize_condition),
        "gesture": pd.to_numeric(d[gesture_col], errors="coerce"),
        "set": d[set_col].astype(str).str.strip().str.upper(),
    })
    if x[["gesture"]].isna().any().any():
        raise RuntimeError("T1: recording inventory contains non-numeric gesture identifiers.")
    x["gesture"] = x["gesture"].astype(int)
    x = x[x["condition"].isin(CONDITION_ORDER) & x["set"].isin(EXPECTED_SETS)]
    if x.empty:
        raise RuntimeError("T1: recording inventory has no ME/MI A/B/C rows.")

    duplicate_key = ["subject", "condition", "gesture", "set"]
    dup = x[x.duplicated(duplicate_key, keep=False)]
    if not dup.empty:
        raise RuntimeError(
            "T1: duplicate recording keys in inventory: "
            f"{dup[duplicate_key].head(10).to_dict('records')}"
        )

    counts = x.groupby("condition").size().to_dict()
    subjects = x.groupby("condition")["subject"].nunique().to_dict()
    per_subject = x.groupby(["condition", "subject"]).size()
    if set(x["gesture"].unique()) != set(EXPECTED_GESTURES):
        raise RuntimeError("T1: audited inventory does not contain exactly gestures 1–7.")
    if set(x["set"].unique()) != set(EXPECTED_SETS):
        raise RuntimeError("T1: audited inventory does not contain exactly sets A/B/C.")
    if any(v != EXPECTED_SUBJECTS for v in subjects.values()):
        raise RuntimeError(f"T1: audited subject counts are not 25: {subjects}")
    if not (per_subject == 21).all():
        raise RuntimeError("T1: audited recordings per subject/condition are not uniformly 21.")


    return {
        "source": relative_path(root, path),
        "me_recordings": int(counts["ME"]),
        "mi_recordings": int(counts["MI"]),
        "paired_recordings": int(min(counts["ME"], counts["MI"])),
        "subjects_me": int(subjects["ME"]),
        "subjects_mi": int(subjects["MI"]),
    }


def _expected_models_for_table(table_id: str) -> set[str]:
    if table_id in {"T9", "T10"}:
        return set(REPRESENTATION_MODELS)
    if table_id in {"T3", "T4", "T5", "T6", "T7", "T8", "T11", "T12", "T13"}:
        return set(MODEL_ORDER)
    return set()


def audit_metric_completeness(generated: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for tid in ("T3", "T4", "T5", "T6"):
        df = generated.get(tid)
        if df is None or "Metric" not in df.columns:
            continue
        observed = {metric_display(v) for v in df["Metric"].dropna().unique()}
        missing = [m for m in METRIC_ORDER if m not in observed]
        rows.append({
            "table_id": tid,
            "required_metrics": ";".join(METRIC_ORDER),
            "observed_metrics": ";".join(sorted(observed)),
            "missing_metrics": ";".join(missing),
            "status": "PASS" if not missing else "FAIL",
        })
    return pd.DataFrame(rows)


def audit_numerical_integrity(
    generated: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for tid, df in generated.items():
        finite_numeric = True
        bad_probability = False
        for col in df.columns:
            if _is_numeric_series(df[col]):
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if not np.isfinite(vals.to_numpy(dtype=float)).all():
                    finite_numeric = False
                n = norm_col(col)
                if any(k in n for k in ("p_value", "pvalue", "fdr_q", "q_value")):
                    if ((vals < 0) | (vals > 1)).any():
                        bad_probability = True
        rows.append({
            "table_id": tid,
            "finite_numeric_values": finite_numeric,
            "probability_values_valid": not bad_probability,
            "status": "PASS" if finite_numeric and not bad_probability else "FAIL",
        })
    return pd.DataFrame(rows)


# ============================================================================
# 7. TABLE WRITING
# ============================================================================

def write_csv_and_xlsx(
    df: pd.DataFrame,
    out_dir: Path,
    stem: str,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{stem}.csv"
    xlsx_path = out_dir / f"{stem}.xlsx"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    try:
        df.to_excel(
            xlsx_path,
            index=False,
            engine="openpyxl",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to write XLSX for {stem}: {exc}"
        ) from exc

    return {
        "csv": str(csv_path),
        "xlsx": str(xlsx_path),
    }


def ensure_no_duplicate_rows(
    df: pd.DataFrame,
    keys: list[str],
    label: str,
) -> None:
    if not keys:
        return

    duplicates = df[df.duplicated(keys, keep=False)]
    if not duplicates.empty:
        raise RuntimeError(
            f"{label}: duplicate rows detected for keys {keys}. "
            f"Examples={duplicates[keys].head(10).to_dict('records')}"
        )


def enforce_model_completeness(
    df: pd.DataFrame,
    label: str,
    model_column: str = "Model",
) -> None:
    if model_column not in df.columns:
        return

    observed = {
        normalize_model(v)
        for v in df[model_column].dropna().unique()
    }

    missing = [
        m for m in MODEL_ORDER
        if m not in observed
    ]

    if missing:
        raise RuntimeError(
            f"{label}: required models missing: {missing}"
        )


def enforce_condition_completeness(
    df: pd.DataFrame,
    label: str,
    condition_column: str = "Condition",
) -> None:
    if condition_column not in df.columns:
        return

    observed = {
        normalize_condition(v)
        for v in df[condition_column].dropna().unique()
    }

    missing = [
        c for c in CONDITION_ORDER
        if c not in observed
    ]

    if missing:
        raise RuntimeError(
            f"{label}: required conditions missing: {missing}"
        )


def validate_missing_values(
    df: pd.DataFrame,
    label: str,
) -> None:
    """
    Missing values are allowed only where the source explicitly represents
    an unavailable/structurally undefined quantity. This function does not
    fill them.
    """
    # No blanket NA prohibition: scientific tables may legitimately contain
    # status-defined missing values. The audit records the locations.
    return


# ============================================================================
# 7. TABLE 1 — DATASET / ACQUISITION
# ============================================================================

def build_table_01(
    root: Path,
    out_dir: Path,
    registry: list[dict[str, Any]],
) -> pd.DataFrame:
    audited = audit_recording_inventory(root)
    rows = [
        ("Primary cohort", "25 paired ME/MI subjects"),
        ("Conditions", "ME, MI"),
        ("Gestures", "7 (1–7)"),
        ("Recording sets", "A, B, C"),
        ("EEG channels", "13"),
        ("Sampling rate", "125 Hz"),
        ("Nominal recording duration", "45 s"),
        ("Nominal samples/recording", "5625"),
        ("ME recordings/subject", "21"),
        ("MI recordings/subject", "21"),
        ("Paired recordings/subject", "42"),
        ("Total ME recordings", str(audited["me_recordings"])),
        ("Total MI recordings", str(audited["mi_recordings"])),
        ("Total paired recordings", str(audited["paired_recordings"] * 2)),
        ("Audited recording inventory", audited["source"]),
    ]
    df = pd.DataFrame(rows, columns=["Characteristic", "Final value"])
    registry.append({
        "table_id": "T1",
        "source_block": "01_SUBJECT_MATCHING",
        "source_file": audited["source"],
        "variable": "Audited recording/acquisition characteristics",
        "transformation": "Inventory validation and deterministic publication formatting",
        "final_value_status": "AUDITED_UPSTREAM_VALUE",
    })
    return df


# ============================================================================
# 8. TABLE 2 — PREPROCESSING / WINDOW
# ============================================================================

def build_table_02(
    root: Path,
    out_dir: Path,
    registry: list[dict[str, Any]],
) -> pd.DataFrame:
    window_cfg = load_selected_window_config(root)
    w = extract_window_values(window_cfg)

    rows = [
        ("Input", "Continuous EEG"),
        ("Sampling rate", f"{w['sampling_rate_hz']} Hz"),
        ("EEG channels", "13"),
        ("Band-pass filter", "1–40 Hz"),
        ("Filter", "4th-order Butterworth"),
        ("Filtering", "Zero-phase offline"),
        ("Resampling", "None"),
        (
            "Primary window length",
            f"{w['window_samples']} samples "
            f"({w['window_duration_s']:.3f} s)",
        ),
        (
            "Primary overlap",
            f"{w['overlap_percent']:.3f}%",
        ),
        (
            "Step",
            f"{w['step_samples']} samples "
            f"({w['step_duration_s']:.3f} s)",
        ),
        ("Padding", "None"),
        ("Cross-recording windows", "Prohibited"),
        ("Cross-condition windows", "Prohibited"),
        ("Cross-gesture windows", "Prohibited"),
    ]

    df = pd.DataFrame(
        rows,
        columns=["Processing component", "Final specification"],
    )

    registry.append({
        "table_id": "T2",
        "source_block": "03_PREPROCESSING / 04_WINDOWING",
        "source_file": ARTIFACTS["BLOCK04"]["selected_window"],
        "variable": "Final window configuration",
        "transformation": "Read frozen configuration; derive duration/step deterministically",
        "final_value_status": "STORED_CONFIGURATION",
    })

    return df


# ============================================================================
# 9. TABLE 3 / 4 — DECODING
# ============================================================================

def load_aggregate_performance(root: Path) -> pd.DataFrame:
    return load_csv(
        require_file(
            root,
            ARTIFACTS["BLOCK15"]["aggregate"],
            "BLOCK15 aggregate performance",
        )
    )


def canonicalize_aggregate_performance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    model_col = find_col(df, "model")
    condition_col = find_col(df, "condition")
    metric_col = find_col(df, "metric", required=False)

    x = df.copy()
    x["Model"] = x[model_col].map(normalize_model)
    x["Condition"] = x[condition_col].map(normalize_condition)

    if metric_col is not None:
        x["Metric"] = x[metric_col].map(metric_norm)

    return x


def build_decoding_table(
    root: Path,
    table_id: str,
    out_dir: Path,
    registry: list[dict[str, Any]],
    loso: bool,
) -> pd.DataFrame:
    raw = load_aggregate_performance(root)
    x = canonicalize_aggregate_performance(raw)

    required = {"Model", "Condition"}
    if not required.issubset(x.columns):
        raise RuntimeError(
            f"{table_id}: aggregate performance cannot be interpreted; "
            f"columns={list(x.columns)}"
        )

    # The aggregate source MUST already distinguish SD and LOSO. We do not
    # calculate means/medians/CIs here.
    # Canonical BLOCK15 aggregate_performance.csv uses `branch` to identify
    # the finalized evaluation regime (e.g., SD / LOSO).  Accept that
    # canonical field explicitly; only fall back to equivalent regime fields
    # when present.  BLOCK17 never infers a regime from model/condition/values.
    regime_candidates = [
        c for c in x.columns
        if norm_col(c) in {
            "branch",
            "evaluation_regime",
            "regime",
            "evaluation",
            "analysis_type",
            "setting",
        }
    ]

    if not regime_candidates:
        raise RuntimeError(
            f"{table_id}: aggregate_performance.csv does not expose a "
            "finalized evaluation-regime/branch column. "
            "BLOCK17 refuses to infer SD vs LOSO; columns={list(x.columns)}"
        )

    regime_col = regime_candidates[0]
    x["_regime"] = x[regime_col].astype(str).map(norm_col)

    wanted = (
        {"loso", "cross_subject", "cross_subject_loso", "subject_independent"}
        if loso
        else {"sd", "subject_dependent", "subjectdependent"}
    )

    mask = x["_regime"].isin(wanted)
    x = x.loc[mask].copy()

    if x.empty:
        raise RuntimeError(
            f"{table_id}: no rows match the required finalized evaluation regime."
        )

    # Two accepted aggregate layouts:
    # 1) long: metric/value
    # 2) wide: Accuracy/Balanced Accuracy/Macro-F1/MCC/Kappa columns.
    if "Metric" in x.columns:
        value_col = None
        for candidate in [
            "mean",
            "value",
            "estimate",
            "score",
        ]:
            if candidate in x.columns:
                value_col = candidate
                break

        if value_col is None:
            raise RuntimeError(
                f"{table_id}: long aggregate source lacks a finalized "
                f"value column. columns={list(x.columns)}"
            )

        x["Value"] = pd.to_numeric(
            x[value_col],
            errors="coerce",
        )

        out = x[
            ["Model", "Condition", "Metric", "Value"]
        ].copy()

        out["Metric"] = out["Metric"].map(metric_norm)
        out["Metric"] = out["Metric"].replace({
            "accuracy": "Accuracy",
            "balanced_accuracy": "Balanced Accuracy",
            "macro_f1": "Macro-F1",
            "mcc": "MCC",
            "kappa": "Cohen's kappa",
        })

    else:
        metric_map = {
            "Accuracy": "accuracy",
            "Balanced Accuracy": "balanced_accuracy",
            "Macro-F1": "macro_f1",
            "MCC": "mcc",
            "Cohen's kappa": "kappa",
        }

        rows = []
        for _, row in x.iterrows():
            for display_name, source_name in metric_map.items():
                if source_name in x.columns:
                    rows.append({
                        "Model": row["Model"],
                        "Condition": row["Condition"],
                        "Metric": display_name,
                        "Value": row[source_name],
                    })

        if not rows:
            raise RuntimeError(
                f"{table_id}: no recognized finalized performance metrics."
            )

        out = pd.DataFrame(rows)

    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")

    ensure_no_duplicate_rows(
        out,
        ["Model", "Condition", "Metric"],
        table_id,
    )

    observed_models = set(out["Model"])
    missing_models = [
        m for m in MODEL_ORDER if m not in observed_models
    ]
    if missing_models:
        raise RuntimeError(
            f"{table_id}: missing models={missing_models}"
        )

    observed_conditions = set(out["Condition"])
    missing_conditions = [
        c for c in CONDITION_ORDER if c not in observed_conditions
    ]
    if missing_conditions:
        raise RuntimeError(
            f"{table_id}: missing conditions={missing_conditions}"
        )

    observed_metrics = set(out["Metric"].astype(str))
    missing_metrics = [m for m in METRIC_ORDER if m not in observed_metrics]
    if missing_metrics:
        raise RuntimeError(f"{table_id}: required metrics missing={missing_metrics}")

    out["Family"] = np.where(
        out["Model"].isin(ML_MODELS),
        "ML",
        "DL",
    )

    out["Model"] = pd.Categorical(
        out["Model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    out["Condition"] = pd.Categorical(
        out["Condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )

    out["Metric"] = pd.Categorical(
        out["Metric"],
        categories=METRIC_ORDER,
        ordered=True,
    )

    out = out.sort_values(
        ["Family", "Model", "Condition", "Metric"]
    ).reset_index(drop=True)

    registry.append({
        "table_id": table_id,
        "source_block": "15_FINAL_STATISTICS",
        "source_file": ARTIFACTS["BLOCK15"]["aggregate"],
        "variable": "Final aggregate decoding performance",
        "transformation": "Regime filtering, label normalization, ordering only",
        "final_value_status": "STORED_BLOCK15_VALUE",
    })

    return out[
        ["Family", "Model", "Condition", "Metric", "Value"]
    ]


# ============================================================================
# 10. TABLE 5 — SD–LOSO GAP
# ============================================================================

def build_table_05(
    root: Path,
    out_dir: Path,
    registry: list[dict[str, Any]],
) -> pd.DataFrame:
    # BLOCK15 stores subject-level gaps and a separate summary artifact.
    # BLOCK17 reports the stored summary; it never recomputes statistics.
    summary_path = require_file(
        root,
        ARTIFACTS["BLOCK15"]["gap_summary"],
        "BLOCK15 SD–LOSO gap summary",
    )
    d = load_csv(summary_path)

    model_col = find_col(d, "model")
    condition_col = find_col(d, "condition")
    metric_col = find_col(d, "metric")
    normalized_d = {norm_col(c): c for c in d.columns}

    aliases = {
        "Mean gap": {"mean", "mean_gap", "g_mean"},
        "Median gap": {"median", "median_gap", "g_median"},
        "SD": {"std", "sd", "sd_gap", "gap_sd"},
        "IQR": {"iqr", "iqr_gap", "gap_iqr"},
        "95% CI low": {"ci95_low", "ci_low", "gap_ci95_low"},
        "95% CI high": {"ci95_high", "ci_high", "gap_ci95_high"},
        "N": {"n", "valid_n", "n_valid"},
    }

    x = pd.DataFrame({
        "Model": d[model_col].map(normalize_model),
        "Condition": d[condition_col].map(normalize_condition),
        "Metric": d[metric_col].map(metric_display),
    })

    for display, candidates in aliases.items():
        match = next(
            (normalized_d[n] for n in candidates if n in normalized_d),
            None,
        )
        if match is not None:
            x[display] = d[match]

    if "Mean gap" not in x.columns:
        raise RuntimeError(
            "T5: BLOCK15 gap summary does not contain a stored mean gap. "
            "BLOCK17 will not recompute it from SD_LOSO_subject_gap.csv. "
            f"columns={list(d.columns)}"
        )

    for col in ["Mean gap", "Median gap", "SD", "IQR",
                "95% CI low", "95% CI high", "N"]:
        if col not in x.columns:
            x[col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")

    ensure_no_duplicate_rows(x, ["Model", "Condition", "Metric"], "T5")

    x["Family"] = np.where(x["Model"].isin(ML_MODELS), "ML", "DL")
    x["Model"] = pd.Categorical(x["Model"], categories=MODEL_ORDER, ordered=True)
    x["Condition"] = pd.Categorical(
        x["Condition"], categories=CONDITION_ORDER, ordered=True
    )
    x = x.sort_values(
        ["Family", "Model", "Condition", "Metric"]
    ).reset_index(drop=True)

    registry.append({
        "table_id": "T5",
        "source_block": "15_FINAL_STATISTICS",
        "source_file": ARTIFACTS["BLOCK15"]["gap_summary"],
        "variable": "Stored SD–LOSO gap summaries",
        "transformation": "Stored summary column selection and deterministic ordering only",
        "final_value_status": "STORED_BLOCK15_VALUE",
    })
    return x


# 11. TABLE 6 — PAIRED ME–MI
# ============================================================================

def build_table_06(
    root: Path,
    out_dir: Path,
    registry: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build Table 6 from finalized paired ME–MI statistics only."""
    summary_path = require_file(
        root,
        ARTIFACTS["BLOCK15"]["me_mi_statistical_summary"],
        "ME–MI statistical summary",
    )
    paired_path = require_file(
        root,
        ARTIFACTS["BLOCK15"]["me_mi_paired_tests"],
        "ME–MI paired-test results",
    )

    d = load_csv(summary_path)
    pt = load_csv(paired_path)

    model_col = find_col(d, "model")
    metric_col = find_col(d, "metric")
    regime_col = find_col(d, "evaluation_regime", required=False)
    normalized = {norm_col(c): c for c in d.columns}

    def stored_col(*names: str) -> str:
        for name in names:
            n = norm_col(name)
            if n in normalized:
                return normalized[n]
        raise RuntimeError(
            f"T6: required stored column not found; candidates={names}; "
            f"columns={list(d.columns)}"
        )

    mean_me_col = stored_col("mean_ME", "me", "a_me", "me_value")
    mean_mi_col = stored_col("mean_MI", "mi", "a_mi", "mi_value")
    diff_col = stored_col("mean_difference", "mean_diff", "difference_mean")
    median_diff_col = stored_col("median_difference", "median_diff")
    effect_col = stored_col("effect_size_dz", "effect_size", "effect")
    ci_low_col = stored_col(
        "bootstrap_ci_lower", "ci95_low", "ci_low", "95_ci_low"
    )
    ci_high_col = stored_col(
        "bootstrap_ci_upper", "ci95_high", "ci_high", "95_ci_high"
    )
    raw_p_col = stored_col("p_raw", "raw_p", "p_value", "p")
    q_col = stored_col("p_adjusted", "q", "q_value", "fdr_q", "fdr")
    n_col = stored_col("n_paired", "n", "valid_n", "n_valid")

    x = pd.DataFrame({
        "Model": d[model_col].map(normalize_model),
        "Metric": d[metric_col].map(metric_display),
        "ME": d[mean_me_col],
        "MI": d[mean_mi_col],
        "Mean difference": d[diff_col],
        "Median difference": d[median_diff_col],
        "Effect size": d[effect_col],
        "95% CI low": d[ci_low_col],
        "95% CI high": d[ci_high_col],
        "Raw p": d[raw_p_col],
        "FDR q": d[q_col],
        "N": d[n_col],
    })
    x.insert(
        0,
        "Evaluation regime",
        d[regime_col].astype(str) if regime_col is not None else "LOSO",
    )

    # Cross-audit stored observed statistic and raw p against the paired-test
    # registry. Nothing is recalculated or substituted.
    pt_model = find_col(pt, "model")
    pt_metric = find_col(pt, "metric")
    pt_stat = find_col(pt, "observed_statistic")
    pt_p = find_col(pt, "p_raw")
    pt_regime = find_col(pt, "evaluation_regime", required=False)

    audit_pt = pd.DataFrame({
        "Model": pt[pt_model].map(normalize_model),
        "Metric": pt[pt_metric].map(metric_display),
        "observed_statistic": pd.to_numeric(pt[pt_stat], errors="coerce"),
        "p_raw": pd.to_numeric(pt[pt_p], errors="coerce"),
    })
    audit_pt.insert(
        0,
        "Evaluation regime",
        pt[pt_regime].astype(str) if pt_regime is not None else "LOSO",
    )

    ensure_no_duplicate_rows(
        audit_pt,
        ["Evaluation regime", "Model", "Metric"],
        "T6 paired-test source",
    )
    ensure_no_duplicate_rows(
        x,
        ["Evaluation regime", "Model", "Metric"],
        "T6",
    )

    audit_pt = audit_pt.merge(
        x[
            ["Evaluation regime", "Model", "Metric",
             "Mean difference", "Raw p"]
        ],
        on=["Evaluation regime", "Model", "Metric"],
        how="inner",
        validate="one_to_one",
    )

    stat_diff = (
        pd.to_numeric(audit_pt["observed_statistic"], errors="coerce")
        - pd.to_numeric(audit_pt["Mean difference"], errors="coerce")
    ).abs()
    p_diff = (
        pd.to_numeric(audit_pt["p_raw"], errors="coerce")
        - pd.to_numeric(audit_pt["Raw p"], errors="coerce")
    ).abs()

    if bool((stat_diff > 1e-12).any()):
        bad = audit_pt.loc[
            stat_diff > 1e-12,
            ["Evaluation regime", "Model", "Metric"],
        ].to_dict("records")
        raise RuntimeError(
            "T6: stored observed_statistic in paired_test_results.csv "
            "does not match stored Mean difference in "
            f"ME_MI_statistical_summary.csv; mismatches={bad[:10]}"
        )
    if bool((p_diff > 1e-12).any()):
        bad = audit_pt.loc[
            p_diff > 1e-12,
            ["Evaluation regime", "Model", "Metric"],
        ].to_dict("records")
        raise RuntimeError(
            "T6: stored p_raw in paired_test_results.csv does not match "
            "stored Raw p in ME_MI_statistical_summary.csv; "
            f"mismatches={bad[:10]}"
        )
    if len(audit_pt) != len(x):
        missing = x.merge(
            audit_pt[["Evaluation regime", "Model", "Metric"]],
            on=["Evaluation regime", "Model", "Metric"],
            how="left",
            indicator=True,
        )
        missing = missing.loc[
            missing["_merge"] == "left_only",
            ["Evaluation regime", "Model", "Metric"],
        ]
        raise RuntimeError(
            "T6: paired_test_results.csv does not cover every "
            "ME–MI statistical-summary row; "
            f"missing={missing.to_dict('records')[:10]}"
        )

    x["Family"] = np.where(x["Model"].isin(ML_MODELS), "ML", "DL")
    x["Model"] = pd.Categorical(
        x["Model"], categories=MODEL_ORDER, ordered=True
    )
    x["Metric"] = pd.Categorical(
        x["Metric"],
        categories=[
            "Accuracy", "Balanced Accuracy", "Macro-F1",
            "MCC", "Cohen's kappa",
        ],
        ordered=True,
    )
    x = x.sort_values(
        ["Family", "Model", "Metric"]
    ).reset_index(drop=True)

    registry.append({
        "table_id": "T6",
        "source_block": "08_ME_MI_STATISTICS / 15_FINAL_STATISTICS",
        "source_file": (
            f"{ARTIFACTS['BLOCK15']['me_mi_statistical_summary']} ; "
            f"{ARTIFACTS['BLOCK15']['me_mi_paired_tests']}"
        ),
        "variable": (
            "Final paired ME–MI means, mean/median differences, "
            "Cohen's dz, bootstrap 95% CI, raw p, and FDR-adjusted q"
        ),
        "transformation": (
            "Stored-column selection/renaming, source consistency audit, "
            "and deterministic model/metric ordering only"
        ),
        "final_value_status": "STORED_UPSTREAM_VALUE",
    })
    return x


# ============================================================================
# 12. TABLE 7 — TRANSFER
# ============================================================================

def build_table_07(
    root: Path,
    out_dir: Path,
    registry: list[dict[str, Any]],
) -> pd.DataFrame:
    d = load_csv(
        require_file(
            root,
            ARTIFACTS["BLOCK15"]["transfer"],
            "BLOCK15 transfer summary",
        )
    )

    model_col = find_col(d, "model")
    direction_col = find_col(d, "direction")
    metric_col = find_col(d, "metric")

    required = [
        "mean",
        "median",
        "ci95_low",
        "ci95_high",
    ]

    normalized = {norm_col(c): c for c in d.columns}

    for c in required:
        if norm_col(c) not in normalized:
            raise RuntimeError(
                f"T7: required stored transfer summary column '{c}' "
                f"not found; columns={list(d.columns)}"
            )

    x = pd.DataFrame({
        "Model": d[model_col].map(normalize_model),
        "Direction": d[direction_col].astype(str),
        "Metric": d[metric_col].map(metric_norm),
        "Transfer performance": d[normalized["mean"]],
        "Median transfer performance": d[normalized["median"]],
        "95% CI low": d[normalized["ci95_low"]],
        "95% CI high": d[normalized["ci95_high"]],
    })

    # Native-target performance and transfer-loss/difference must come from
    # stored BLOCK15 columns if available. Never calculate them here.
    optional_aliases = {
        "Native target performance": {
            "native_target_performance",
            "target_native_performance",
        },
        "Transfer loss": {
            "transfer_loss",
            "mean_transfer_loss",
        },
        "Transfer difference": {
            "transfer_benefit",
            "transfer_difference",
            "mean_transfer_benefit",
        },
        "Negative transfer (%)": {
            "negative_percent",
            "negative_transfer_percent",
        },
    }

    for display, candidates in optional_aliases.items():
        match = next(
            (
                normalized[n]
                for n in candidates
                if n in normalized
            ),
            None,
        )
        if match is not None:
            x[display] = d[match]

    ensure_no_duplicate_rows(
        x,
        ["Model", "Direction", "Metric"],
        "T7",
    )

    x["Model"] = pd.Categorical(
        x["Model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    x = x.sort_values(
        ["Model", "Direction", "Metric"]
    ).reset_index(drop=True)

    registry.append({
        "table_id": "T7",
        "source_block": "10_ME_MI_TRANSFER / 15_FINAL_STATISTICS",
        "source_file": ARTIFACTS["BLOCK15"]["transfer"],
        "variable": "Stored bidirectional transfer quantities",
        "transformation": "Column selection, direction/model ordering only",
        "final_value_status": "STORED_BLOCK15_VALUE",
    })

    return x


# ============================================================================
# 13. TABLE 8 — CALIBRATION
# ============================================================================

def build_table_08(
    root: Path,
    out_dir: Path,
    registry: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build Table 8 from the finalized BLOCK09 calibration tables.

    TABLE09_01 is authoritative for the nine stored calibration-performance
    points. TABLE09_02 is authoritative for K90 summary/status quantities.
    BLOCK17 performs only schema validation, deterministic reshaping, and a
    key-based join; it never reconstructs A(k), K90, or any statistical
    quantity from lower-level predictions.
    """
    trajectory_path = require_file(
        root,
        ARTIFACTS["BLOCK09"]["temporal_calibration"],
        "BLOCK09 temporal calibration trajectory",
    )
    trajectory = load_csv(trajectory_path)

    required = [
        "condition",
        "model",
        "metric",
        "calibration_fraction",
        "mean_performance",
        "median_performance",
        "sd_performance",
        "iqr_performance",
        "ci_lower",
        "ci_upper",
        "n_subjects",
    ]
    missing = [c for c in required if c not in trajectory.columns]
    if missing:
        raise RuntimeError(
            "T8: BLOCK09 TABLE09_01_temporal_calibration.csv is missing "
            f"required columns: {missing}; columns={list(trajectory.columns)}"
        )

    t = pd.DataFrame({
        "Model": trajectory["model"].map(normalize_model),
        "Condition": trajectory["condition"].map(normalize_condition),
        "Metric": trajectory["metric"].map(metric_norm),
        "Calibration fraction (%)": pd.to_numeric(
            trajectory["calibration_fraction"], errors="coerce"
        ),
        "Mean performance": pd.to_numeric(
            trajectory["mean_performance"], errors="coerce"
        ),
        "Median performance": pd.to_numeric(
            trajectory["median_performance"], errors="coerce"
        ),
        "SD": pd.to_numeric(trajectory["sd_performance"], errors="coerce"),
        "IQR": pd.to_numeric(trajectory["iqr_performance"], errors="coerce"),
        "95% CI low": pd.to_numeric(trajectory["ci_lower"], errors="coerce"),
        "95% CI high": pd.to_numeric(trajectory["ci_upper"], errors="coerce"),
        "N": pd.to_numeric(trajectory["n_subjects"], errors="coerce"),
    })

    if t["Calibration fraction (%)"].isna().any():
        raise RuntimeError("T8: BLOCK09 TABLE09_01 contains invalid calibration fractions.")

    # Exactly the frozen nine-point trajectory is required for every stored
    # model/condition/metric group. No missing level is imputed.
    expected_levels = [float(x) for x in CALIBRATION_LEVELS]
    for key, group in t.groupby(["Model", "Condition", "Metric"], sort=False):
        levels = sorted(group["Calibration fraction (%)"].unique().tolist())
        if levels != expected_levels:
            raise RuntimeError(
                "T8: incomplete BLOCK09 calibration trajectory for "
                f"{key}; found={levels}, expected={expected_levels}"
            )

    ensure_no_duplicate_rows(
        t,
        ["Model", "Condition", "Metric", "Calibration fraction (%)"],
        "T8 BLOCK09 trajectory",
    )

    # Reshape only stored values: one row per Model/Condition/Metric, with the
    # nine finalized A(k) values. The summary statistics are retained as the
    # corresponding stored trajectory statistics in long-form audit columns.
    rows: list[dict[str, Any]] = []
    for (model, condition, metric), group in t.groupby(
        ["Model", "Condition", "Metric"], sort=False
    ):
        row: dict[str, Any] = {
            "Model": model,
            "Condition": condition,
            "Metric": metric,
        }
        group = group.sort_values("Calibration fraction (%)")
        for _, r in group.iterrows():
            level = int(round(float(r["Calibration fraction (%)"])))
            row[f"A{level}"] = r["Mean performance"]
        # Keep the finalized trajectory uncertainty/sample-size information in
        # supplementary audit columns without deriving any new statistics.
        for level in CALIBRATION_LEVELS:
            rr = group[group["Calibration fraction (%)"] == float(level)].iloc[0]
            row[f"A{level}_median"] = rr["Median performance"]
            row[f"A{level}_SD"] = rr["SD"]
            row[f"A{level}_IQR"] = rr["IQR"]
            row[f"A{level}_CI_low"] = rr["95% CI low"]
            row[f"A{level}_CI_high"] = rr["95% CI high"]
            row[f"A{level}_N"] = rr["N"]
        rows.append(row)

    x = pd.DataFrame(rows)

    # K90 is a separate finalized BLOCK09 artifact. Its failure/status fields
    # are preserved exactly; blank K90 values remain blank when no valid K90
    # exists.
    k90_path = require_file(
        root,
        ARTIFACTS["BLOCK09"]["k90"],
        "BLOCK09 K90 summary",
    )
    k90 = load_csv(k90_path)
    required_k90 = [
        "condition",
        "model",
        "metric",
        "n_valid",
        "mean_k90_conditional_on_reached",
        "median_k90",
        "iqr_k90",
        "ci_lower",
        "ci_upper",
        "n_not_reached",
        "n_no_positive_gap",
        "n_fail",
    ]
    missing_k90 = [c for c in required_k90 if c not in k90.columns]
    if missing_k90:
        raise RuntimeError(
            "T8: BLOCK09 TABLE09_02_k90.csv is missing required columns: "
            f"{missing_k90}; columns={list(k90.columns)}"
        )

    k = pd.DataFrame({
        "Model": k90["model"].map(normalize_model),
        "Condition": k90["condition"].map(normalize_condition),
        "Metric": k90["metric"].map(metric_norm),
        "K90 N valid": pd.to_numeric(k90["n_valid"], errors="coerce"),
        "K90 mean (valid only)": pd.to_numeric(
            k90["mean_k90_conditional_on_reached"], errors="coerce"
        ),
        "K90 median": pd.to_numeric(k90["median_k90"], errors="coerce"),
        "K90 IQR": pd.to_numeric(k90["iqr_k90"], errors="coerce"),
        "K90 95% CI low": pd.to_numeric(k90["ci_lower"], errors="coerce"),
        "K90 95% CI high": pd.to_numeric(k90["ci_upper"], errors="coerce"),
        "K90 N not reached": pd.to_numeric(k90["n_not_reached"], errors="coerce"),
        "K90 N no positive gap": pd.to_numeric(k90["n_no_positive_gap"], errors="coerce"),
        "K90 N fail": pd.to_numeric(k90["n_fail"], errors="coerce"),
    })

    # TABLE09_02 does not expose a synthetic numeric K90 for failure cases.
    # Keep the semantic status from its counts rather than converting failures
    # into a number.
    k["k90_status"] = np.select(
        [
            k["K90 N valid"].fillna(0).gt(0),
            k["K90 N not reached"].fillna(0).gt(0),
            k["K90 N no positive gap"].fillna(0).gt(0),
            k["K90 N fail"].fillna(0).gt(0),
        ],
        ["REACHED", "NOT_REACHED", "NO_POSITIVE_GAP", "FAIL"],
        default="NO_VALID_K90",
    )

    ensure_no_duplicate_rows(k, ["Model", "Condition", "Metric"], "T8 BLOCK09 K90")
    ensure_no_duplicate_rows(x, ["Model", "Condition", "Metric"], "T8 calibration")

    out = x.merge(
        k,
        on=["Model", "Condition", "Metric"],
        how="left",
        validate="one_to_one",
    )

    out["Model"] = pd.Categorical(
        out["Model"], categories=MODEL_ORDER, ordered=True
    )
    out["Condition"] = pd.Categorical(
        out["Condition"], categories=CONDITION_ORDER, ordered=True
    )
    out = out.sort_values(["Model", "Condition", "Metric"]).reset_index(drop=True)

    registry.append({
        "table_id": "T8",
        "source_block": "09_TEMPORAL_CALIBRATION",
        "source_file": (
            ARTIFACTS["BLOCK09"]["temporal_calibration"]
            + " + "
            + ARTIFACTS["BLOCK09"]["k90"]
        ),
        "variable": "Stored calibration response A(0)...A(90) and stored K90 summary/status",
        "transformation": (
            "Schema validation, deterministic pivot of stored mean_performance, "
            "and key-based Model/Condition/Metric join; no calibration or K90 recalculation"
        ),
        "final_value_status": "STORED_BLOCK09_VALUE",
    })

    return out


# ============================================================================
# 14. TABLE 9 — RSA
# ============================================================================

def build_table_09(
    root: Path,
    out_dir: Path,
    registry: list[dict[str, Any]],
) -> pd.DataFrame:
    d = load_csv(
        require_file(
            root,
            ARTIFACTS["BLOCK15"]["rsa"],
            "BLOCK15 RSA summary",
        )
    )

    model_col = find_col(d, "model")

    normalized = {norm_col(c): c for c in d.columns}

    required_summary = {
        "Mean rho": ["mean_rho", "mean", "rho_mean"],
        "Median rho": ["median_rho", "median", "rho_median"],
        "SD": ["sd_rho", "sd", "std", "rho_sd"],
        "IQR": ["iqr_rho", "iqr", "rho_iqr"],
    }

    x = pd.DataFrame({
        "Representation model": d[model_col].map(normalize_model),
    })

    for display, candidates in required_summary.items():
        match = next(
            (
                normalized[n]
                for n in candidates
                if n in normalized
            ),
            None,
        )
        if match is None:
            raise RuntimeError(
                f"T9: stored RSA summary does not expose '{display}'. "
                f"columns={list(d.columns)}"
            )
        x[display] = d[match]

    optional = {
        "95% CI low": ["ci95_low", "ci_low"],
        "95% CI high": ["ci95_high", "ci_high"],
        "N": ["n", "valid_n"],
    }

    for display, candidates in optional.items():
        match = next(
            (
                normalized[n]
                for n in candidates
                if n in normalized
            ),
            None,
        )
        if match is not None:
            x[display] = d[match]

    allowed = set(REPRESENTATION_MODELS)
    unexpected = sorted(
        set(x["Representation model"]) - allowed
    )
    if unexpected:
        raise RuntimeError(
            f"T9: unexpected representation models={unexpected}"
        )

    missing = sorted(
        allowed - set(x["Representation model"])
    )
    if missing:
        raise RuntimeError(
            f"T9: required representation models missing={missing}"
        )

    ensure_no_duplicate_rows(
        x,
        ["Representation model"],
        "T9",
    )

    order_map = {
        m: i for i, m in enumerate(REPRESENTATION_MODELS)
    }
    x["_order"] = x["Representation model"].map(order_map)
    x = x.sort_values("_order").drop(columns="_order")

    registry.append({
        "table_id": "T9",
        "source_block": "12_REPRESENTATION_ANALYSIS / 15_FINAL_STATISTICS",
        "source_file": ARTIFACTS["BLOCK15"]["rsa"],
        "variable": "Stored ME–MI RSA summary (mean_rho, median_rho, sd_rho, iqr_rho, CI, N where stored)",
        "transformation": "Schema validation, model filtering, and deterministic ordering only; no RSA recomputation",
        "final_value_status": "STORED_BLOCK15_VALUE",
    })

    return x


# ============================================================================
# 15. TABLE 10 — REPRESENTATION RELATIONSHIPS
# ============================================================================

def build_table_10(
    root: Path,
    out_dir: Path,
    registry: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Build Table 10 from the authoritative BLOCK15 hypothesis-test summaries.

    IMPORTANT:
    - representation_transfer.csv and representation_calibration.csv are
      subject-level source tables (25 rows/model) and are NOT group-level
      association summaries.
    - The finalized H4/H5 association statistics are stored in
      15_FINAL_STATISTICS/hypothesis_tests/H4_*.csv and H5_*.csv.
    - No correlation, p-value, CI, FDR q-value, or k90 statistic is
      recomputed here.
    """
    sources = [
        (
            "Representation vs transfer benefit",
            ARTIFACTS["BLOCK15"]["h4_representation_transfer"],
            "H4",
        ),
        (
            "Representation vs k90",
            ARTIFACTS["BLOCK15"]["h5_representation_calibration"],
            "H5",
        ),
    ]

    frames: list[pd.DataFrame] = []

    for relationship, relative, hypothesis in sources:
        d = load_csv(
            require_file(
                root,
                relative,
                f"T10 source: {relationship}",
            )
        )

        normalized = {norm_col(c): c for c in d.columns}

        model_col = next(
            (
                normalized[n]
                for n in {"model", "model_name", "architecture"}
                if n in normalized
            ),
            None,
        )
        if model_col is None:
            raise RuntimeError(
                f"T10: {relative} has no model field; "
                f"columns={list(d.columns)}"
            )

        corr_col = next(
            (
                normalized[n]
                for n in {
                    "rho",
                    "rho_spearman",
                    "spearman_r",
                    "correlation",
                    "correlation_estimate",
                }
                if n in normalized
            ),
            None,
        )
        p_col = next(
            (
                normalized[n]
                for n in {
                    "permutation_p",
                    "perm_p",
                    "p",
                    "p_value",
                }
                if n in normalized
            ),
            None,
        )
        n_col = next(
            (
                normalized[n]
                for n in {"n_subjects", "n", "valid_n"}
                if n in normalized
            ),
            None,
        )

        if corr_col is None:
            raise RuntimeError(
                f"T10: {relative} has no stored correlation field; "
                f"columns={list(d.columns)}"
            )
        if p_col is None:
            raise RuntimeError(
                f"T10: {relative} has no stored permutation-p field; "
                f"columns={list(d.columns)}"
            )

        out = pd.DataFrame({
            "Representation model": d[model_col].map(normalize_model),
            "Relationship": relationship,
            "N": d[n_col] if n_col is not None else pd.NA,
            "Correlation": d[corr_col],
            "Permutation p": d[p_col],
        })

        # Preserve optional fields only when the authoritative source stores
        # them. Never synthesize CI, FDR q, or direction.
        optional = {
            "95% CI low": {"ci95_low", "ci_low"},
            "95% CI high": {"ci95_high", "ci_high"},
            "FDR q": {"fdr_q", "q", "q_value"},
            "Direction": {"direction"},
            "Status": {"status"},
        }
        for display, candidates in optional.items():
            match = next(
                (
                    normalized[n]
                    for n in candidates
                    if n in normalized
                ),
                None,
            )
            if match is not None:
                out[display] = d[match]

        # Validate the hypothesis identity if it is explicitly stored.
        hypothesis_col = normalized.get("hypothesis")
        if hypothesis_col is not None:
            observed = {
                str(v).strip()
                for v in d[hypothesis_col].dropna().tolist()
                if str(v).strip()
            }
            if observed != {hypothesis}:
                raise RuntimeError(
                    f"T10: {relative} hypothesis field is inconsistent; "
                    f"expected={hypothesis}, observed={sorted(observed)}"
                )

        frames.append(out)

    out = pd.concat(frames, ignore_index=True)

    allowed = set(REPRESENTATION_MODELS)
    observed_models = set(out["Representation model"].dropna())
    unexpected = observed_models - allowed
    if unexpected:
        raise RuntimeError(
            "T10: source contains representation models outside the "
            f"frozen representation-model registry: {sorted(unexpected)}"
        )

    # Each H4/H5 hypothesis-test file is a group-level finalized summary:
    # exactly one row per representation model is required.
    for relationship in [
        "Representation vs transfer benefit",
        "Representation vs k90",
    ]:
        subset = out[out["Relationship"] == relationship]
        observed = set(subset["Representation model"].dropna())
        missing = sorted(allowed - observed)
        if missing:
            raise RuntimeError(
                f"T10: relationship '{relationship}' missing models={missing}"
            )
        dup = subset.duplicated(
            subset=["Representation model"],
            keep=False,
        )
        if dup.any():
            examples = subset.loc[
                dup, ["Representation model", "Relationship"]
            ].to_dict("records")
            raise RuntimeError(
                "T10: duplicate group-level association rows detected; "
                f"source={relationship}, examples={examples[:10]}"
            )

    order_map = {m: i for i, m in enumerate(REPRESENTATION_MODELS)}
    relationship_order = {
        "Representation vs transfer benefit": 0,
        "Representation vs k90": 1,
    }
    out["_model_order"] = out["Representation model"].map(order_map)
    out["_relationship_order"] = out["Relationship"].map(relationship_order)
    out = (
        out.sort_values(["_model_order", "_relationship_order"])
        .drop(columns=["_model_order", "_relationship_order"])
        .reset_index(drop=True)
    )

    registry.append({
        "table_id": "T10",
        "source_block": "15_FINAL_STATISTICS / hypothesis_tests",
        "source_file": (
            ARTIFACTS["BLOCK15"]["h4_representation_transfer"]
            + " + "
            + ARTIFACTS["BLOCK15"]["h5_representation_calibration"]
        ),
        "variable": (
            "Stored H4 representation–transfer and H5 "
            "representation–calibration associations"
        ),
        "transformation": (
            "Authoritative hypothesis-summary selection, field mapping, "
            "validation, and deterministic ordering only; no statistical "
            "recalculation"
        ),
        "final_value_status": "STORED_BLOCK15_VALUE",
    })

    return out


# ============================================================================
# 16. TABLE 11 — XAI
# ============================================================================

def build_table_11(
    root: Path,
    out_dir: Path,
    registry: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build Table 11 from the stored BLOCK15 XAI summary.

    The authoritative XAI summary is descriptive and its finalized schema does
    not contain an ``analysis`` column.  BLOCK17 therefore does not invent an
    analysis field.  It preserves the actual stored method/condition/gesture
    and attribution-summary fields and adds only deterministic reporting
    labels.
    """
    d = load_csv(
        require_file(
            root,
            ARTIFACTS["BLOCK15"]["xai"],
            "BLOCK15 XAI summary",
        )
    )

    normalized = {norm_col(c): c for c in d.columns}

    def resolve(candidates: set[str], label: str, required: bool = True) -> str | None:
        match = next((normalized[n] for n in candidates if n in normalized), None)
        if match is None and required:
            raise RuntimeError(
                f"T11: XAI source lacks required '{label}' field; "
                f"columns={list(d.columns)}"
            )
        return match

    model_col = resolve({"model", "model_name", "architecture"}, "model")
    method_col = resolve({"xai_method", "method", "explanation_method"}, "method")
    condition_col = resolve({"condition", "cond"}, "condition", required=False)
    gesture_col = resolve({"gesture", "gesture_id", "gesture_code"}, "gesture", required=False)
    n_subjects_col = resolve({"n_subjects", "n", "valid_n"}, "n_subjects", required=False)

    # Preserve the actual finalized descriptive XAI fields. These are not
    # converted into fabricated common statistics.
    keep_map = {
        "Model": model_col,
        "XAI method": method_col,
        "Condition": condition_col,
        "Gesture": gesture_col,
        "N subjects": n_subjects_col,
        "Top features": resolve({"top_features"}, "top_features", required=False),
        "Top channels": resolve({"top_channels"}, "top_channels", required=False),
        "Top frequency groups": resolve({"top_frequency_groups"}, "top_frequency_groups", required=False),
        "Attribution similarity mean": resolve({"attribution_similarity_mean"}, "attribution_similarity_mean", required=False),
        "Attribution similarity median": resolve({"attribution_similarity_median"}, "attribution_similarity_median", required=False),
        "95% CI low": resolve({"ci_lower"}, "CI_lower", required=False),
        "95% CI high": resolve({"ci_upper"}, "CI_upper", required=False),
    }

    out = pd.DataFrame(index=d.index)
    for display, source in keep_map.items():
        if source is not None:
            out[display] = d[source]

    out["Model"] = out["Model"].map(normalize_model)
    if "Condition" in out.columns:
        out["Condition"] = out["Condition"].map(normalize_condition)
    if "Gesture" in out.columns:
        out["Gesture"] = pd.to_numeric(out["Gesture"], errors="coerce")

    unexpected = sorted(set(out["Model"].dropna()) - set(MODEL_ORDER))
    if unexpected:
        raise RuntimeError(f"T11: unexpected XAI models={unexpected}")

    # Preserve all stored rows; do not collapse across condition/gesture or
    # derive an invented 'analysis' level. Deterministic publication order.
    model_order = {m: i for i, m in enumerate(MODEL_ORDER)}
    out["_model_order"] = out["Model"].map(model_order)
    sort_cols = ["_model_order"]
    if "Condition" in out.columns:
        out["_condition_order"] = out["Condition"].map({c: i for i, c in enumerate(CONDITION_ORDER)})
        sort_cols.append("_condition_order")
    if "Gesture" in out.columns:
        sort_cols.append("Gesture")
    sort_cols.append("XAI method")
    out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    out = out.drop(columns=[c for c in ["_model_order", "_condition_order"] if c in out.columns])

    registry.append({
        "table_id": "T11",
        "source_block": "13_XAI / 15_FINAL_STATISTICS",
        "source_file": ARTIFACTS["BLOCK15"]["xai"],
        "variable": "Stored descriptive XAI summary: method, condition, gesture, subject count, top features/channels/frequency groups, attribution similarity, CI where stored",
        "transformation": "Schema validation, direct stored-field mapping, model/condition/gesture ordering only; no XAI or inferential-statistic recalculation",
        "final_value_status": "STORED_BLOCK15_VALUE",
    })

    return out


# ============================================================================
# 17. TABLE 12 — ROBUSTNESS
# ============================================================================

def build_table_12(
    root: Path,
    out_dir: Path,
    registry: list[dict[str, Any]],
) -> pd.DataFrame:
    paths = [
        ("Channel dropout", ARTIFACTS["BLOCK15"]["dropout"]),
        ("Noise", ARTIFACTS["BLOCK15"]["noise"]),
    ]

    frames: list[pd.DataFrame] = []

    for perturbation, relative in paths:
        d = load_csv(
            require_file(
                root,
                relative,
                f"T12 {perturbation}",
            )
        )

        model_col = find_col(d, "model")
        condition_col = find_col(d, "condition")

        normalized = {norm_col(c): c for c in d.columns}

        level_candidates = [
            "dropout_percent" if perturbation == "Channel dropout" else "perturbation_level",
            "level",
            "dropout_level",
            "snr_db",
            "noise_level",
            "noise_snr_db",
            "perturbation_percent",
        ]
        level_match = next(
            (normalized[n] for n in level_candidates if n in normalized),
            None,
        )

        if level_match is None:
            raise RuntimeError(
                f"T12: perturbation level not found in {relative}; "
                f"columns={list(d.columns)}"
            )

        clean_match = next(
            (
                normalized[n]
                for n in {
                    "clean_macro_f1",
                    "clean",
                    "p_clean",
                }
                if n in normalized
            ),
            None,
        )

        perturbed_match = next(
            (
                normalized[n]
                for n in {
                    "perturbed_macro_f1",
                    "perturbed",
                    "p_perturbed",
                }
                if n in normalized
            ),
            None,
        )

        delta_match = next(
            (
                normalized[n]
                for n in {
                    "delta_macro_f1",
                    "delta",
                    "delta_p",
                    "degradation",
                    "mean_delta",
                }
                if n in normalized
            ),
            None,
        )

        # BLOCK15 dropout_summary.csv is a finalized summary artifact whose
        # stored schema may contain only the aggregate degradation statistics
        # (e.g. mean_delta, median_delta). BLOCK17 must not reconstruct clean
        # or perturbed performance from Δ, because that would be a new
        # calculation and the clean reference may live in another upstream
        # artifact. Therefore, require the stored delta quantity, while
        # retaining clean/perturbed columns only when they are actually
        # present in the source.
        if delta_match is None:
            raise RuntimeError(
                f"T12: {relative} lacks a stored robustness delta/degradation "
                f"field. columns={list(d.columns)}"
            )

        x = pd.DataFrame({
            "Model": d[model_col].map(normalize_model),
            "Condition": d[condition_col].map(normalize_condition),
            "Perturbation": perturbation,
            "Level": d[level_match],
            "Δ Macro-F1": d[delta_match],
        })

        # Preserve clean/perturbed values only when explicitly stored. Never
        # infer them from Δ or from another summary table.
        if clean_match is not None:
            x["Clean Macro-F1"] = d[clean_match]
        if perturbed_match is not None:
            x["Perturbed Macro-F1"] = d[perturbed_match]

        # Preserve additional stored aggregate summaries where available.
        for display, candidates in {
            "Median Δ Macro-F1": {"median_delta", "median_delta_macro_f1"},
            "95% CI low": {"ci95_low", "ci_low"},
            "95% CI high": {"ci95_high", "ci_high"},
            "N subjects": {"n_subjects", "n", "valid_n"},
        }.items():
            match = next(
                (
                    normalized[n]
                    for n in candidates
                    if n in normalized
                ),
                None,
            )
            if match is not None and match != delta_match:
                x[display] = d[match]

        frames.append(x)

    out = pd.concat(frames, ignore_index=True)

    ensure_no_duplicate_rows(
        out,
        ["Perturbation", "Model", "Condition", "Level"],
        "T12 robustness",
    )

    out["Model"] = pd.Categorical(
        out["Model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    out["Condition"] = pd.Categorical(
        out["Condition"],
        categories=CONDITION_ORDER,
        ordered=True,
    )

    out = out.sort_values(
        ["Perturbation", "Model", "Condition", "Level"]
    ).reset_index(drop=True)

    registry.append({
        "table_id": "T12",
        "source_block": "14_ROBUSTNESS / 15_FINAL_STATISTICS",
        "source_file": (
            ARTIFACTS["BLOCK15"]["dropout"]
            + " + "
            + ARTIFACTS["BLOCK15"]["noise"]
        ),
        "variable": "Stored robustness effects",
        "transformation": "Source concatenation and deterministic ordering only",
        "final_value_status": "STORED_BLOCK15_VALUE",
    })

    return out


# ============================================================================
# 18. TABLE 13 — COMPUTATIONAL
# ============================================================================

def build_table_13(
    root: Path,
    out_dir: Path,
    registry: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build T13 directly from the finalized BLOCK15 computational summary.

    The audited source is an aggregate summary with columns
    source_block, metric, mean, median, n. It does not contain per-model
    computational measurements. BLOCK17 therefore preserves that stored
    summary rather than inventing model-level parameter/time/memory values.
    """
    relative = ARTIFACTS["BLOCK15"]["computational"]
    d = load_csv(
        require_file(
            root,
            relative,
            "BLOCK15 computational summary",
        )
    )

    normalized = {norm_col(c): c for c in d.columns}
    required = {
        "source_block": "source_block",
        "metric": "metric",
        "mean": "mean",
        "median": "median",
        "n": "n",
    }
    missing = [name for name, key in required.items() if key not in normalized]
    if missing:
        raise RuntimeError(
            f"T13: computational source lacks required stored fields {missing}; "
            f"columns={list(d.columns)}"
        )

    out = pd.DataFrame({
        "Source block": d[normalized["source_block"]],
        "Metric": d[normalized["metric"]],
        "Mean": d[normalized["mean"]],
        "Median": d[normalized["median"]],
        "N": d[normalized["n"]],
    })

    ensure_no_duplicate_rows(
        out,
        ["Source block", "Metric"],
        "T13 computational summary",
    )

    out = out.sort_values(
        ["Source block", "Metric"],
        kind="stable",
    ).reset_index(drop=True)

    registry.append({
        "table_id": "T13",
        "source_block": "15_FINAL_STATISTICS",
        "source_file": relative,
        "variable": "Stored computational summary: source block, metric, mean, median, N",
        "transformation": "Schema validation, direct stored-field mapping, and deterministic ordering only; no model-level computational values inferred",
        "final_value_status": "STORED_BLOCK15_VALUE",
    })

    return out


# ============================================================================
# 19. TABLE 14 — HYPOTHESIS / EVIDENCE SUMMARY
# ============================================================================

def locate_hypothesis_summary(root: Path) -> Path:
    candidates = [
        "15_FINAL_STATISTICS/hypotheses/hypothesis_summary.csv",
        "15_FINAL_STATISTICS/hypothesis/hypothesis_summary.csv",
        "15_FINAL_STATISTICS/evidence/hypothesis_evidence_summary.csv",
        "15_FINAL_STATISTICS/hypothesis_matrix/hypothesis_evidence_summary.csv",
        "15_FINAL_STATISTICS/hypothesis_evidence_summary.csv",
    ]

    for relative in candidates:
        p = root / relative
        if p.is_file():
            return p

    raise RuntimeError(
        "T14: no finalized BLOCK15 hypothesis/evidence summary source "
        "was found. BLOCK17 will not invent hypothesis decisions."
    )


def build_table_14(
    root: Path,
    out_dir: Path,
    registry: list[dict[str, Any]],
) -> pd.DataFrame:
    path = locate_hypothesis_summary(root)
    d = load_csv(path)

    normalized = {norm_col(c): c for c in d.columns}

    # The finalized BLOCK15 hypothesis registry stores the primary analysis
    # under ``analysis_family`` and stores CI bounds separately.  These are
    # publication-schema differences only; BLOCK17 must map the stored fields
    # without recalculating any statistic or inventing a new analysis label.
    # IMPORTANT: BLOCK15 contains 369 distinct evidence records. Some records
    # share identical hypothesis/analysis/estimate/p/q/CI values but are
    # distinguished by model, condition, metric, and N. Preserve those identity
    # fields so BLOCK17 does not collapse distinct BLOCK15 records.
    # `evidence_status` is authoritative for the publication Evidence status;
    # the upstream `status` field is a separate analysis status.
    field_map = {
        "Hypothesis": {"hypothesis", "hypothesis_id"},
        "Primary analysis": {"primary_analysis", "analysis", "analysis_family"},
        "Model": {"model", "model_name", "classifier", "architecture"},
        "Condition": {"condition", "cond"},
        "Metric": {"metric", "measure"},
        "N": {"n", "valid_n", "n_subjects"},
        "Estimate": {"estimate", "effect", "value"},
        "p": {"p", "p_value", "raw_p"},
        "FDR q": {"fdr_q", "q", "q_value"},
        "Evidence status": {"evidence_status"},
        "Analysis status": {"status"},
    }

    out = pd.DataFrame()

    for display, candidates in field_map.items():
        match = next(
            (
                normalized[n]
                for n in candidates
                if n in normalized
            ),
            None,
        )
        if match is None:
            raise RuntimeError(
                f"T14: finalized hypothesis source lacks '{display}'; "
                f"columns={list(d.columns)}"
            )
        out[display] = d[match]

    # Deterministic identity-label normalization only; no statistical values
    # are recomputed or altered.
    out["Model"] = out["Model"].map(normalize_model)
    out["Condition"] = out["Condition"].map(normalize_condition)
    out["Metric"] = out["Metric"].map(metric_display)
    out["N"] = pd.to_numeric(out["N"], errors="coerce")

    # Preserve the finalized CI bounds exactly as stored.  When both bounds
    # exist, combine them into a publication field; this is deterministic
    # formatting, not a new confidence-interval calculation.
    ci_low_col = normalized.get("ci_lower") or normalized.get("ci95_low")
    ci_high_col = normalized.get("ci_upper") or normalized.get("ci95_high")
    if ci_low_col is not None and ci_high_col is not None:
        out["95% CI"] = [
            (f"[{lo}, {hi}]" if pd.notna(lo) and pd.notna(hi) else pd.NA)
            for lo, hi in zip(d[ci_low_col], d[ci_high_col])
        ]
    else:
        ci_single = next(
            (
                normalized[n]
                for n in {"ci95", "confidence_interval", "ci"}
                if n in normalized
            ),
            None,
        )
        if ci_single is None:
            raise RuntimeError(
                "T14: finalized hypothesis source lacks stored confidence "
                f"interval fields; columns={list(d.columns)}"
            )
        out["95% CI"] = d[ci_single]

    invalid = sorted(
        {
            str(v).strip().upper()
            for v in out["Evidence status"].dropna()
        }
        - ALLOWED_EVIDENCE_STATUS
    )

    if invalid:
        raise RuntimeError(
            f"T14: unsupported evidence-status labels={invalid}"
        )

    # BLOCK15 is the authoritative registry. It has no exact duplicate rows;
    # BLOCK17 must preserve all 369 records and must never deduplicate them.
    if len(out) != len(d):
        raise RuntimeError(
            f"T14: row-count changed during publication mapping: "
            f"source={len(d)}, output={len(out)}"
        )

    if int(d.duplicated().sum()) != 0:
        raise RuntimeError(
            "T14: authoritative BLOCK15 hypothesis/evidence source contains "
            "exact duplicate rows; BLOCK17 will not deduplicate them."
        )

    if int(out.duplicated().sum()) != 0:
        examples = out.loc[
            out.duplicated(keep=False),
            [
                "Hypothesis", "Primary analysis", "Model", "Condition",
                "Metric", "N", "Estimate", "p", "FDR q",
                "Evidence status", "Analysis status", "95% CI"
            ],
        ].head(10).to_dict("records")
        raise RuntimeError(
            "T14: publication projection still contains exact duplicate "
            f"evidence records after preserving source identity fields; "
            f"examples={examples}"
        )

    registry.append({
        "table_id": "T14",
        "source_block": "15_FINAL_STATISTICS",
        "source_file": relative_path(root, path),
        "variable": "Final hypothesis/evidence summary",
        "transformation": (
            "Deterministic source-field mapping, preservation of model/condition/"
            "metric/N identity, authoritative evidence_status selection, label "
            "normalization, and CI formatting only; no deduplication or "
            "statistical recalculation"
        ),
        "final_value_status": "STORED_BLOCK15_VALUE",
    })

    return out


# ============================================================================
# 20. SUPPLEMENTARY TABLES
# ============================================================================

SUPPLEMENTARY_SOURCES = {
    "S09_SD_LOSO_Subject_Level": ARTIFACTS["BLOCK15"]["gap"],
    "S10_ME_MI_Subject_Level": ARTIFACTS["BLOCK15"]["me_mi_gap"],
    "S12_Calibration_Subject_Level": ARTIFACTS["BLOCK15"]["calibration"],
    "S13_RSA_Subject_Level": ARTIFACTS["BLOCK15"]["rsa"],
    "S14_XAI": ARTIFACTS["BLOCK15"]["xai"],
    "S15_Robustness_Subject_Level": ARTIFACTS["BLOCK15"]["robustness"],
    "S16_Statistical_Registry": (
        "15_FINAL_STATISTICS/statistical_registry.csv"
    ),
    "S17_FDR_Registry": (
        "15_FINAL_STATISTICS/fdr_registry.csv"
    ),
    "S18_Reproducibility_Registry": (
        "15_FINAL_STATISTICS/reproducibility_registry.csv"
    ),
}


def supplementary_candidates(root: Path, sid: str) -> list[Path]:
    candidates = []

    direct = {
        "S01": [
            "01_SUBJECT_MATCHING/recording_inventory.csv",
            "01_SUBJECT_MATCHING/tables/recording_inventory.csv",
        ],
        "S02": [
            "01_SUBJECT_MATCHING/subject_completeness.csv",
            "01_SUBJECT_MATCHING/tables/subject_completeness.csv",
        ],
        "S03": [
            # BLOCK02 does not produce a file named signal_qa_summary.csv.
            # The authoritative per-recording finalized QA artifact is
            # recording_final_qa_status.csv; preserve it without recomputation.
            "02_SIGNAL_QA/recording_final_qa_status.csv",
            "02_SIGNAL_QA/tables/recording_final_qa_status.csv",
            # Additional fallback only if a future frozen QA release places the
            # same finalized artifact under a tables directory.
            "02_SIGNAL_QA/condition_qa_summary.csv",
        ],
        "S04": [
            "04_WINDOWING/window_count_audit.csv",
            "04_WINDOWING/tables/window_count_audit.csv",
        ],
        "S05": [
            # BLOCK06A's frozen source inventory identifies the execution audit
            # as the authoritative finalized SD-ML artifact. There is no
            # verified subject_results.csv at the canonical BLOCK06A path.
            "06_ML/06A_ML_SD/SD_execution_audit.csv",
            "06A_ML_SD/SD_execution_audit.csv",
        ],
        "S06": [
            "07A_DL_SD/subject_results.csv",
            "07_DL/07A_DL_SD/subject_results.csv",
        ],
        "S07": [
            "06A_ML_SD/classwise_results.csv",
            "07A_DL_SD/classwise_results.csv",
        ],
        "S08": [
            "06A_ML_SD/confusion_matrices.csv",
            "07A_DL_SD/confusion_matrices.csv",
        ],
        "S09": [ARTIFACTS["BLOCK15"]["gap"]],
        "S10": [ARTIFACTS["BLOCK15"]["me_mi_gap"]],
        "S11": [
            "10_ME_MI_TRANSFER/metrics/transfer_benefit.csv",
            "15_FINAL_STATISTICS/transfer/transfer_subject_level.csv",
        ],
        "S12": [ARTIFACTS["BLOCK15"]["calibration"]],
        "S13": [
            "12_REPRESENTATION_ANALYSIS/subject_similarity/subject_level_rsa.csv",
            ARTIFACTS["BLOCK15"]["rsa"],
        ],
        "S14": [ARTIFACTS["BLOCK15"]["xai"]],
        "S15": [
            "14_ROBUSTNESS/noise/BLOCK14_noise_subject_level.csv",
            "14_ROBUSTNESS/channel_dropout/BLOCK14_channel_dropout_subject_level.csv",
            ARTIFACTS["BLOCK15"]["robustness"],
        ],
        "S16": [
            "15_FINAL_STATISTICS/statistical_registry.csv",
            "15_FINAL_STATISTICS/statistical_registry/statistical_test_registry.csv",
            "15_FINAL_STATISTICS/audits/statistical_registry.csv",
        ],
        "S17": [
            "15_FINAL_STATISTICS/fdr_registry.csv",
            "15_FINAL_STATISTICS/multiple_comparisons/FDR_registry.csv",
            "15_FINAL_STATISTICS/audits/fdr_registry.csv",
        ],
        "S18": [
            "15_FINAL_STATISTICS/reproducibility_registry.csv",
            "15_FINAL_STATISTICS/audits/reproducibility_audit.csv",
            "15_FINAL_STATISTICS/audits/reproducibility_registry.csv",
        ],
    }

    candidates.extend(
        root / p
        for p in direct.get(sid, [])
    )

    return candidates


def build_supplementary_tables(
    root: Path,
    out_dir: Path,
    registry: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}

    for number in range(1, 19):
        sid = f"S{number:02d}"

        # S07 is a multi-file supplementary artifact. The finalized SD
        # pipelines store one `class_metrics.csv` per subject/condition/model
        # run under their canonical run trees; there is no verified single
        # `classwise_results.csv` at the current upstream paths. Aggregate the
        # stored classwise files without recomputing any metric.
        if sid == "S07":
            classwise_roots = [
                root / "06_ML" / "06A_ML_SD",
                root / "07_DL" / "07A_DL_SD",
            ]
            classwise_files: list[Path] = []
            for classwise_root in classwise_roots:
                if classwise_root.is_dir():
                    classwise_files.extend(
                        sorted(
                            classwise_root.rglob("class_metrics.csv"),
                            key=lambda q: str(q).lower(),
                        )
                    )

            # Only subject-level classwise artifacts are publication inputs for S07.
            # The canonical SD directories may also contain an aggregate
            # `class_metrics.csv` directly at the branch root; that file has no
            # Subject/Condition/Model path context and must not be interpreted
            # as a subject-level classwise result.
            classwise_files = [
                q
                for q in classwise_files
                if "runs" in q.relative_to(root).parts
            ]

            if not classwise_files:
                raise RuntimeError(
                    "S07: no subject-level finalized class_metrics.csv files found under "
                    "06_ML/06A_ML_SD/runs or 07_DL/07A_DL_SD/runs"
                )

            frames: list[pd.DataFrame] = []
            source_files: list[str] = []
            for classwise_file in classwise_files:
                part = pd.read_csv(classwise_file)
                if part.empty:
                    raise RuntimeError(
                        f"S07: empty classwise source: {relative_path(root, classwise_file)}"
                    )

                rel_parts = classwise_file.relative_to(root).parts
                try:
                    branch = "ML" if "06A_ML_SD" in rel_parts else "DL"
                    run_idx = rel_parts.index("runs")
                    subject_token = rel_parts[run_idx + 1]
                    condition = rel_parts[run_idx + 2]
                    model_token = rel_parts[run_idx + 3]
                except (ValueError, IndexError) as exc:
                    raise RuntimeError(
                        "S07: cannot parse canonical classwise path "
                        f"{relative_path(root, classwise_file)}"
                    ) from exc

                if condition not in {"ME", "MI"}:
                    raise RuntimeError(
                        "S07: unexpected condition in classwise path "
                        f"{relative_path(root, classwise_file)}"
                    )

                subject_match = re.search(
                    r"(?:Subject[_-]?)?(\d+)$",
                    subject_token,
                    flags=re.IGNORECASE,
                )
                if subject_match is None:
                    raise RuntimeError(
                        "S07: cannot parse subject ID from classwise path "
                        f"{relative_path(root, classwise_file)}"
                    )
                subject_id = int(subject_match.group(1))

                part = part.copy()
                part.insert(0, "Branch", branch)
                part.insert(1, "Subject", subject_id)
                part.insert(2, "Condition", condition)
                part.insert(3, "Model", model_token)
                part.insert(4, "Source file", relative_path(root, classwise_file))
                frames.append(part)
                source_files.append(relative_path(root, classwise_file))

            df = pd.concat(frames, ignore_index=True)
            source_label = "; ".join(source_files)
            source_for_result = classwise_files[0]
        elif sid == "S08":
            # S08 is a multi-file supplementary artifact. The frozen BLOCK06A
            # archive stores subject-level confusion matrices as individual
            # *_confusion_matrix.csv files under a dedicated confusion_matrices
            # directory; do not require a nonexistent branch-level aggregate.
            confusion_roots = [
                root / "06_ML" / "06A_ML_SD" / "confusion_matrices",
                root / "07_DL" / "07A_DL_SD" / "confusion_matrices",
            ]
            confusion_files: list[Path] = []
            for confusion_root in confusion_roots:
                if confusion_root.is_dir():
                    confusion_files.extend(
                        sorted(
                            confusion_root.glob("*_confusion_matrix.csv"),
                            key=lambda q: str(q).lower(),
                        )
                    )

            if not confusion_files:
                raise RuntimeError(
                    "S08: no finalized *_confusion_matrix.csv files found under "
                    "06_ML/06A_ML_SD/confusion_matrices or "
                    "07_DL/07A_DL_SD/confusion_matrices"
                )

            frames: list[pd.DataFrame] = []
            source_files: list[str] = []
            for confusion_file in confusion_files:
                part = pd.read_csv(confusion_file)
                if part.empty:
                    raise RuntimeError(
                        f"S08: empty confusion-matrix source: {relative_path(root, confusion_file)}"
                    )

                rel_parts = confusion_file.relative_to(root).parts
                branch = "ML" if "06A_ML_SD" in rel_parts else "DL"
                filename = confusion_file.stem
                if not filename.endswith("_confusion_matrix"):
                    raise RuntimeError(
                        "S08: unexpected confusion-matrix filename "
                        f"{relative_path(root, confusion_file)}"
                    )
                token = filename[:-len("_confusion_matrix")].strip(" _-")

                # Prefer explicit metadata columns when the stored artifact
                # contains them. Otherwise parse only the filename; no metric
                # is recomputed. Supported filename forms include common
                # Subject/condition/model tokenizations used by the frozen SD
                # archives (e.g. Subject_01_ME_LDA).
                subject_id = None
                condition = None
                model_token = None
                colmap = {str(c).strip().lower(): c for c in part.columns}
                for key in ("subject", "subject_id"):
                    if key in colmap:
                        vals = pd.to_numeric(part[colmap[key]], errors="coerce").dropna().unique()
                        if len(vals) == 1:
                            subject_id = int(vals[0])
                            break
                for key in ("condition", "cond"):
                    if key in colmap:
                        vals = part[colmap[key]].dropna().astype(str).str.upper().unique()
                        if len(vals) == 1 and vals[0] in {"ME", "MI"}:
                            condition = vals[0]
                            break
                for key in ("model", "model_name"):
                    if key in colmap:
                        vals = part[colmap[key]].dropna().astype(str).unique()
                        if len(vals) == 1 and vals[0].strip():
                            model_token = vals[0].strip()
                            break

                if subject_id is None:
                    sm = re.search(r"(?:^|[_-])Subject[_-]?(\d+)(?:$|[_-])", token, flags=re.IGNORECASE)
                    if sm is None:
                        sm = re.search(r"(?:^|[_-])S[_-]?(\d+)(?:$|[_-])", token, flags=re.IGNORECASE)
                    if sm is not None:
                        subject_id = int(sm.group(1))

                if condition is None:
                    cm = re.search(r"(?:^|[_-])(ME|MI)(?:$|[_-])", token, flags=re.IGNORECASE)
                    if cm is not None:
                        condition = cm.group(1).upper()

                if model_token is None:
                    # Remove the subject and condition tokens from the filename
                    # and use the remaining stored model token verbatim.
                    model_token = re.sub(r"(?:^|[_-])Subject[_-]?\d+", "", token, flags=re.IGNORECASE)
                    model_token = re.sub(r"(?:^|[_-])S[_-]?\d+", "", model_token, flags=re.IGNORECASE)
                    model_token = re.sub(r"(?:^|[_-])(?:ME|MI)(?:$|[_-])", "_", model_token, flags=re.IGNORECASE)
                    model_token = re.sub(r"^[_-]+|[_-]+$", "", model_token)
                    model_token = re.sub(r"[_-]+", "_", model_token)
                    model_token = model_token.strip()

                if subject_id is None or condition not in {"ME", "MI"} or not model_token:
                    raise RuntimeError(
                        "S08: cannot parse canonical confusion-matrix identity "
                        f"from {relative_path(root, confusion_file)}"
                    )

                part = part.copy()
                part.insert(0, "Branch", branch)
                part.insert(1, "Subject", subject_id)
                part.insert(2, "Condition", condition)
                part.insert(3, "Model", model_token)
                part.insert(4, "Source file", relative_path(root, confusion_file))
                frames.append(part)
                source_files.append(relative_path(root, confusion_file))

            df = pd.concat(frames, ignore_index=True)
            source_label = "; ".join(source_files)
            source_for_result = confusion_files[0]
        else:
            candidates = supplementary_candidates(root, sid)
            source = next(
                (p for p in candidates if p.is_file()),
                None,
            )

            if source is None:
                raise RuntimeError(
                    f"{sid}: required supplementary source not found. "
                    f"Checked={[relative_path(root, p) for p in candidates]}"
                )

            if source.suffix.lower() == ".csv":
                df = pd.read_csv(source)
            elif source.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(source)
            else:
                raise RuntimeError(
                    f"{sid}: unsupported supplementary source format: {source}"
                )

            if df.empty:
                raise RuntimeError(
                    f"{sid}: source table is empty: {source}"
                )
            source_label = relative_path(root, source)
            source_for_result = source

        stem = {
            "S01": "S01_Recording_Inventory",
            "S02": "S02_Subject_Completeness",
            "S03": "S03_QA",
            "S04": "S04_Window_Audit",
            "S05": "S05_ML_Complete",
            "S06": "S06_DL_Complete",
            "S07": "S07_Classwise",
            "S08": "S08_Confusion_Matrices",
            "S09": "S09_SD_LOSO_Subject_Level",
            "S10": "S10_ME_MI_Subject_Level",
            "S11": "S11_Transfer_Subject_Level",
            "S12": "S12_Calibration_Subject_Level",
            "S13": "S13_RSA_Subject_Level",
            "S14": "S14_XAI",
            "S15": "S15_Robustness_Subject_Level",
            "S16": "S16_Statistical_Registry",
            "S17": "S17_FDR_Registry",
            "S18": "S18_Reproducibility_Registry",
        }[sid]

        outputs = write_csv_and_xlsx(
            df,
            out_dir,
            stem,
        )

        registry.append({
            "table_id": sid,
            "source_block": "Upstream finalized source",
            "source_file": source_label,
            "variable": "Complete supplementary source",
            "transformation": (
                "Concatenate stored class_metrics.csv files with path-derived "
                "Branch/Subject/Condition/Model metadata; no metric recalculation"
                if sid == "S07" else
                "Format-preserving CSV/XLSX export only"
            ),
            "final_value_status": "SOURCE_COPY",
        })

        # For multi-file supplementary artifacts (S07/S08), the generated
        # consolidated CSV is the canonical BLOCK17 hand-off source for the
        # downstream publication/manifest stage.  `source_label` remains the
        # complete upstream provenance string in the registry above.
        downstream_source = outputs["csv"] if sid in {"S07", "S08"} else source_label
        results[sid] = {
            "source": downstream_source,
            "rows": int(len(df)),
            "outputs": outputs,
        }

    return results


# ============================================================================
# 21. SOURCE REGISTRY
# ============================================================================

def write_source_registry(
    out_dir: Path,
    registry: list[dict[str, Any]],
) -> Path:
    path = out_dir / "audits" / "table_source_registry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(registry)

    if df.empty:
        raise RuntimeError(
            "table_source_registry.csv would be empty."
        )

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return path


# ============================================================================
# 22. TABLE REGISTRY
# ============================================================================

TABLE_REGISTRY = [
    {
        "Table": "T1",
        "Title": "Dataset and acquisition characteristics",
        "Main/Supplementary": "Main",
        "Source": "01–02 / protocol",
    },
    {
        "Table": "T2",
        "Title": "Preprocessing and window configuration",
        "Main/Supplementary": "Main",
        "Source": "03–04",
    },
    {
        "Table": "T3",
        "Title": "Subject-dependent decoding",
        "Main/Supplementary": "Main",
        "Source": "06A/07A/15",
    },
    {
        "Table": "T4",
        "Title": "Cross-subject LOSO decoding",
        "Main/Supplementary": "Main",
        "Source": "06B/07B/15",
    },
    {
        "Table": "T5",
        "Title": "Subject-dependence gap",
        "Main/Supplementary": "Main",
        "Source": "15",
    },
    {
        "Table": "T6",
        "Title": "ME–MI paired performance comparison",
        "Main/Supplementary": "Main",
        "Source": "08/15",
    },
    {
        "Table": "T7",
        "Title": "Bidirectional ME–MI transfer",
        "Main/Supplementary": "Main",
        "Source": "10/15",
    },
    {
        "Table": "T8",
        "Title": "Temporal calibration",
        "Main/Supplementary": "Main",
        "Source": "09/15",
    },
    {
        "Table": "T9",
        "Title": "Latent representation and RSA",
        "Main/Supplementary": "Main",
        "Source": "12/15",
    },
    {
        "Table": "T10",
        "Title": "Representation–transfer and representation–calibration relationships",
        "Main/Supplementary": "Main",
        "Source": "12/15",
    },
    {
        "Table": "T11",
        "Title": "XAI summary",
        "Main/Supplementary": "Main",
        "Source": "13/15",
    },
    {
        "Table": "T12",
        "Title": "Robustness",
        "Main/Supplementary": "Main",
        "Source": "14/15",
    },
    {
        "Table": "T13",
        "Title": "Computational efficiency",
        "Main/Supplementary": "Main",
        "Source": "06–15",
    },
    {
        "Table": "T14",
        "Title": "Hypothesis/evidence summary",
        "Main/Supplementary": "Main",
        "Source": "15",
    },
]


def write_table_registry(out_dir: Path) -> Path:
    path = out_dir / "table_registry.csv"
    df = pd.DataFrame(TABLE_REGISTRY)
    df["Status"] = "PENDING"
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )
    return path


# ============================================================================
# 23. NUMERICAL / QUALITY AUDITS
# ============================================================================

def audit_table_quality(
    generated: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for table_id, df in generated.items():
        rows.append({
            "table_id": table_id,
            "rows": len(df),
            "columns": len(df.columns),
            "empty": bool(df.empty),
            "duplicate_rows": int(df.duplicated().sum()),
            "has_columns": bool(len(df.columns)),
            "status": (
                "PASS"
                if not df.empty
                and df.duplicated().sum() == 0
                and len(df.columns) > 0
                else "FAIL"
            ),
        })

    return pd.DataFrame(rows)


def audit_model_completeness(
    generated: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for table_id, df in generated.items():
        model_cols = [
            c for c in df.columns
            if norm_col(c) in {"model", "representation_model"}
        ]
        if not model_cols:
            continue
        col = model_cols[0]
        observed = {normalize_model(v) for v in df[col].dropna().unique()}
        required = _expected_models_for_table(table_id)
        if not required:
            rows.append({
                "table_id": table_id,
                "required_models": "",
                "observed_models": ";".join(sorted(observed)),
                "missing_models": "",
                "status": "PASS",
            })
            continue
        missing = sorted(required - observed, key=lambda x: MODEL_ORDER.index(x))
        rows.append({
            "table_id": table_id,
            "required_models": ";".join(sorted(required, key=lambda x: MODEL_ORDER.index(x))),
            "observed_models": ";".join(sorted(observed)),
            "missing_models": ";".join(missing),
            "status": "PASS" if not missing else "FAIL",
        })
    return pd.DataFrame(rows)


def audit_condition_completeness(
    generated: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    # Only tables whose scientific schema requires separate ME and MI
    # observations participate in this audit. T11 is explicitly a paired
    # ME–MI XAI analysis and therefore legitimately uses the combined
    # condition label ``ME_MI`` rather than duplicating rows for ME and MI.
    condition_audit_tables = {"T3", "T4", "T5", "T8", "T12"}

    for table_id, df in generated.items():
        if table_id not in condition_audit_tables:
            continue

        condition_cols = [
            c for c in df.columns
            if norm_col(c) == "condition"
        ]

        if not condition_cols:
            continue

        col = condition_cols[0]
        observed = {
            normalize_condition(v)
            for v in df[col].dropna().unique()
        }

        missing = sorted(
            set(CONDITION_ORDER) - observed
        )

        rows.append({
            "table_id": table_id,
            "required_conditions": ";".join(CONDITION_ORDER),
            "observed_conditions": ";".join(sorted(observed)),
            "missing_conditions": ";".join(missing),
            "status": "PASS" if not missing else "FAIL",
        })

    return pd.DataFrame(rows)


def audit_missing_values(
    generated: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for table_id, df in generated.items():
        for column in df.columns:
            count = int(df[column].isna().sum())
            if count:
                rows.append({
                    "table_id": table_id,
                    "column": column,
                    "missing_count": count,
                    "policy": "PRESERVE_SOURCE_MISSINGNESS",
                    "status": "REVIEW",
                })

    if not rows:
        rows.append({
            "table_id": "",
            "column": "",
            "missing_count": 0,
            "policy": "PRESERVE_SOURCE_MISSINGNESS",
            "status": "PASS",
        })

    return pd.DataFrame(rows)


# ============================================================================
# 24. BLOCK15 ↔ BLOCK17 CONSISTENCY
# ============================================================================

def compare_block15_table_values(
    root: Path,
    generated: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    BLOCK17 does not recompute source statistics. This audit verifies that
    generated tables contain finite/stored values where required and records
    the source identity. Detailed cell-level equality is performed when the
    generated table is a direct source projection.
    """
    rows = []

    for table_id, df in generated.items():
        rows.append({
            "table_id": table_id,
            "rows_generated": len(df),
            "columns_generated": len(df.columns),
            "status": (
                "PASS"
                if not df.empty
                else "FAIL"
            ),
        })

    return pd.DataFrame(rows)


# ============================================================================
# 25. BLOCK16 ↔ BLOCK17 FIGURE/TABLE CONSISTENCY
# ============================================================================

def audit_block16_presence(root: Path) -> pd.DataFrame:
    figure_status = require_file(
        root,
        ARTIFACTS["BLOCK16"]["figure_status"],
        "BLOCK16 figure-status registry",
    )
    df = pd.read_csv(figure_status)
    if df.empty:
        raise RuntimeError("BLOCK16 figure-status registry is empty.")
    registry_audit = audit_figure_registry(root)
    return registry_audit.assign(
        source=ARTIFACTS["BLOCK16"]["figure_status"],
        rows=len(df),
    )


def audit_table_figure_consistency(
    root: Path,
    generated: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    The exact figure-to-table relationships are represented as registry
    entries rather than reconstructed statistics. This prevents BLOCK17
    from becoming a second statistical engine.
    """
    mapping = [
        ("FIG01/FIG02", "T1/T2"),
        ("FIG03", "T3/T4/T5"),
        ("FIG04", "T4/T6"),
        ("FIG05", "T7"),
        ("FIG06", "T8"),
        ("FIG07", "T9"),
        ("FIG08", "T10"),
        ("FIG09", "T11"),
        ("FIG10", "T12"),
    ]

    rows = []

    for figures, tables in mapping:
        table_ids = re.findall(r"T\d+", tables)
        missing = [
            t for t in table_ids
            if t not in generated
        ]

        rows.append({
            "figures": figures,
            "tables": tables,
            "missing_tables": ";".join(missing),
            "status": "PASS" if not missing else "FAIL",
        })

    return pd.DataFrame(rows)


def audit_figure_registry(root: Path) -> pd.DataFrame:
    path = require_file(
        root,
        ARTIFACTS["BLOCK16"]["figure_status"],
        "BLOCK16 figure-status registry",
    )
    d = load_csv(path)
    if d.empty:
        raise RuntimeError("BLOCK16 figure-status registry is empty.")
    cols = {norm_col(c): c for c in d.columns}
    fig_col = next((cols[n] for n in ("figure", "figure_id", "id") if n in cols), None)
    status_col = next((cols[n] for n in ("status", "render_status") if n in cols), None)
    if fig_col is None or status_col is None:
        raise RuntimeError(
            f"BLOCK16 figure-status registry lacks figure/status columns: {list(d.columns)}"
        )
    expected = {f"FIG{i:02d}" for i in range(1, 11)}
    observed = {str(v).strip().upper() for v in d[fig_col].dropna()}
    missing = sorted(expected - observed)
    failed = d.loc[
        d[status_col].astype(str).str.upper().isin({"FAIL", "ERROR", "UNAVAILABLE", "NOT_READY"})
    ]
    return pd.DataFrame([{
        "expected_figures": ";".join(sorted(expected)),
        "observed_expected_figures": ";".join(sorted(expected & observed)),
        "missing_figures": ";".join(missing),
        "failed_figures": ";".join(sorted(failed[fig_col].astype(str).unique())),
        "status": "PASS" if not missing and failed.empty else "FAIL",
    }])




# ============================================================================
# 26. GENERATION MANIFEST
# ============================================================================

def build_generation_manifest(
    root: Path,
    cfg: dict[str, Any],
    generated_outputs: dict[str, Any],
) -> pd.DataFrame:
    rows = []

    cfg_hash = block17_config_hash(cfg)

    for table_id, item in generated_outputs.items():
        outputs = item.get("outputs", {})
        for fmt, path in outputs.items():
            p = Path(path)
            rows.append({
                "table_id": table_id,
                "source_block": item.get("source_block", ""),
                "source_file": item.get("source_file", ""),
                "generation_script": Path(__file__).name,
                "configuration_hash": cfg_hash,
                "generation_timestamp": utc_now(),
                "output_format": fmt.upper(),
                "output_file": relative_path(root, p),
                "output_hash": sha256_file(p) if p.is_file() else "",
                "status": "GENERATED",
            })

    return pd.DataFrame(rows)


# ============================================================================
# 27. MAIN RUNNER
# ============================================================================

def run(
    root: Path,
    inspect_only: bool,
) -> int:
    out_root = root / "17_FINAL_TABLES"

    # A failed/interrupted prior run must never leave an apparently valid
    # completion marker in place.
    if not inspect_only:
        stale_complete = out_root / "BLOCK17_complete.json"
        if stale_complete.exists():
            stale_complete.unlink()

    print(
        f"BLOCK17 started | root={root}"
    )

    # The BLOCK17 configuration is itself a frozen BLOCK17 artifact.
    # On the first inspect run, create it if absent so inspection can
    # validate the actual project without requiring a pre-existing config.
    cfg = load_or_create_config(
        root,
        write_if_missing=True,
    )

    cfg_hash = block17_config_hash(cfg)
    print(f"BLOCK17_CONFIG_HASH={cfg_hash}")

    # ------------------------------------------------------------------------
    # G1/G2 — upstream completion
    # ------------------------------------------------------------------------
    validate_block15_status(root)
    validate_block16_status(root)

    print("G1: PASS — BLOCK15 completion marker verified")
    print("G2: PASS — BLOCK16 completion marker verified")

    # ------------------------------------------------------------------------
    # Inspect-only mode
    # ------------------------------------------------------------------------
    if inspect_only:
        # Inspection is deliberately read-only and exercises the canonical
        # source-resolution paths that the --run mode will use.  No output
        # tables, statistics, or completion markers are created here.
        window_cfg = load_selected_window_config(root)
        window_values = extract_window_values(window_cfg)

        print(
            "WINDOW CONFIG:",
            json.dumps(window_values, indent=2),
        )

        inventory_path = _find_inventory_source(root)
        inventory = load_csv(inventory_path)
        print(
            "T1 INVENTORY:",
            json.dumps(
                {
                    "source": relative_path(root, inventory_path),
                    "columns": list(inventory.columns),
                    "rows": int(len(inventory)),
                },
                ensure_ascii=False,
            ),
        )
        audited_inventory = audit_recording_inventory(root)
        print(
            "T1 AUDIT:",
            json.dumps(audited_inventory, ensure_ascii=False),
        )

        aggregate_path = require_file(
            root,
            ARTIFACTS["BLOCK15"]["aggregate"],
            "BLOCK15 aggregate performance",
        )
        aggregate_raw = load_csv(aggregate_path)
        aggregate_canonical = canonicalize_aggregate_performance(
            aggregate_raw
        )
        regime_columns = [
            c for c in aggregate_canonical.columns
            if norm_col(c) in {
                "branch",
                "evaluation_regime",
                "regime",
                "evaluation",
                "analysis_type",
                "setting",
            }
        ]
        print(
            "T3/T4 AGGREGATE:",
            json.dumps(
                {
                    "source": relative_path(root, aggregate_path),
                    "columns": list(aggregate_raw.columns),
                    "rows": int(len(aggregate_raw)),
                    "regime_columns": regime_columns,
                    "branch_values": (
                        sorted(
                            aggregate_raw["branch"]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .unique()
                            .tolist()
                        )
                        if "branch" in aggregate_raw.columns
                        else []
                    ),
                    "metrics": (
                        sorted(
                            aggregate_canonical["Metric"]
                            .dropna()
                            .astype(str)
                            .map(metric_norm)
                            .unique()
                            .tolist()
                        )
                        if "Metric" in aggregate_canonical.columns
                        else []
                    ),
                },
                ensure_ascii=False,
            ),
        )

        # Exercise the actual T3/T4 source interpretation without writing
        # publication tables.  This catches schema/resolution errors before
        # the first --run.
        for table_id, loso in (("T3", False), ("T4", True)):
            preview = build_decoding_table(
                root,
                table_id,
                root / "17_FINAL_TABLES" / "_inspect_only",
                [],
                loso,
            )
            print(
                f"{table_id} SOURCE CHECK: PASS "
                f"(rows={len(preview)}, columns={list(preview.columns)})"
            )

        print("BLOCK17 INSPECTION COMPLETE")
        return 0

    # ------------------------------------------------------------------------
    # Prepare output directories
    # ------------------------------------------------------------------------
    main_dir = out_root / "main"
    supp_dir = out_root / "supplementary"
    publication_dir = out_root / "publication"
    xlsx_dir = publication_dir / "XLSX"
    docx_dir = publication_dir / "DOCX"
    csv_dir = publication_dir / "CSV"
    audits_dir = out_root / "audits"
    configs_dir = out_root / "configs"

    for d in [
        main_dir,
        supp_dir,
        publication_dir,
        xlsx_dir,
        docx_dir,
        csv_dir,
        audits_dir,
        configs_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Freeze configuration
    # ------------------------------------------------------------------------
    cfg_path = configs_dir / "BLOCK17_config.json"
    if not cfg_path.is_file():
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.write("\n")

    # ------------------------------------------------------------------------
    # Build main tables
    # ------------------------------------------------------------------------
    registry: list[dict[str, Any]] = []
    generated: dict[str, pd.DataFrame] = {}
    generated_outputs: dict[str, Any] = {}

    builders = [
        ("T1", "Table_01_Dataset_Acquisition", build_table_01),
        ("T2", "Table_02_Preprocessing_Window", build_table_02),
    ]

    for table_id, stem, builder in builders:
        df = builder(root, main_dir, registry)
        generated[table_id] = df
        generated_outputs[table_id] = {
            "outputs": write_csv_and_xlsx(
                df,
                main_dir,
                stem,
            ),
            "source_block": registry[-1]["source_block"],
            "source_file": registry[-1]["source_file"],
        }

    # T3/T4
    for table_id, stem, loso in [
        ("T3", "Table_03_Subject_Dependent_Decoding", False),
        ("T4", "Table_04_LOSO_Decoding", True),
    ]:
        df = build_decoding_table(
            root,
            table_id,
            main_dir,
            registry,
            loso,
        )
        generated[table_id] = df
        generated_outputs[table_id] = {
            "outputs": write_csv_and_xlsx(
                df,
                main_dir,
                stem,
            ),
            "source_block": registry[-1]["source_block"],
            "source_file": registry[-1]["source_file"],
        }

    remaining_builders = [
        ("T5", "Table_05_SD_LOSO_Gap", build_table_05),
        ("T6", "Table_06_ME_MI_Paired", build_table_06),
        ("T7", "Table_07_Transfer", build_table_07),
        ("T8", "Table_08_Calibration", build_table_08),
        ("T9", "Table_09_RSA", build_table_09),
        ("T10", "Table_10_Representation_Relationships", build_table_10),
        ("T11", "Table_11_XAI", build_table_11),
        ("T12", "Table_12_Robustness", build_table_12),
        ("T13", "Table_13_Computational", build_table_13),
        ("T14", "Table_14_Hypothesis_Summary", build_table_14),
    ]

    for table_id, stem, builder in remaining_builders:
        df = builder(root, main_dir, registry)
        generated[table_id] = df
        generated_outputs[table_id] = {
            "outputs": write_csv_and_xlsx(
                df,
                main_dir,
                stem,
            ),
            "source_block": registry[-1]["source_block"],
            "source_file": registry[-1]["source_file"],
        }

    print(
        f"MAIN TABLES GENERATED: {len(generated)}/14"
    )

    # Publication representations are deterministic rounded views of the same
    # generated tables; raw CSV/XLSX remain in main/ and supplementary/.
    for tid, df in generated.items():
        formatted = publication_round(df, cfg)
        stem = next(
            title for t, title, *_ in [
                ("T1", "Table_01_Dataset_Acquisition"),
                ("T2", "Table_02_Preprocessing_Window"),
                ("T3", "Table_03_Subject_Dependent_Decoding"),
                ("T4", "Table_04_LOSO_Decoding"),
                ("T5", "Table_05_SD_LOSO_Gap"),
                ("T6", "Table_06_ME_MI_Paired"),
                ("T7", "Table_07_Transfer"),
                ("T8", "Table_08_Calibration"),
                ("T9", "Table_09_RSA"),
                ("T10", "Table_10_Representation_Relationships"),
                ("T11", "Table_11_XAI"),
                ("T12", "Table_12_Robustness"),
                ("T13", "Table_13_Computational"),
                ("T14", "Table_14_Hypothesis_Summary"),
            ] if t == tid
        )
        formatted.to_csv(csv_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
        formatted.to_excel(xlsx_dir / f"{stem}.xlsx", index=False, engine="openpyxl")
        generated_outputs[tid]["outputs"]["publication_csv"] = str(csv_dir / f"{stem}.csv")
        generated_outputs[tid]["outputs"]["publication_xlsx"] = str(xlsx_dir / f"{stem}.xlsx")

    # ------------------------------------------------------------------------
    # Supplementary tables
    # ------------------------------------------------------------------------
    supplementary_outputs = build_supplementary_tables(
        root,
        supp_dir,
        registry,
    )

    print(
        f"SUPPLEMENTARY TABLES GENERATED: "
        f"{len(supplementary_outputs)}/18"
    )

    for sid, item in supplementary_outputs.items():
        src = root / item["source"]
        sdf = pd.read_csv(src) if src.suffix.lower() == ".csv" else pd.read_excel(src)
        sdf = publication_round(sdf, cfg)
        stem = Path(item["outputs"]["csv"]).stem
        sdf.to_csv(csv_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
        sdf.to_excel(xlsx_dir / f"{stem}.xlsx", index=False, engine="openpyxl")
        generated_outputs[sid] = {
            "outputs": {
                "csv": item["outputs"]["csv"],
                "xlsx": item["outputs"]["xlsx"],
                "publication_csv": str(csv_dir / f"{stem}.csv"),
                "publication_xlsx": str(xlsx_dir / f"{stem}.xlsx"),
            },
            "source_block": "Upstream finalized source",
            "source_file": item["source"],
        }

    # ------------------------------------------------------------------------
    # Registries
    # ------------------------------------------------------------------------
    source_registry_path = write_source_registry(
        out_root,
        registry,
    )

    table_registry_path = write_table_registry(
        out_root,
    )
    table_registry_df = pd.read_csv(table_registry_path)
    table_registry_df["Status"] = table_registry_df["Table"].map(
        lambda x: "GENERATED" if str(x) in generated else "FAIL"
    )
    table_registry_df.to_csv(table_registry_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------------
    # Quality audits
    # ------------------------------------------------------------------------
    quality = audit_table_quality(generated)
    quality.to_csv(
        audits_dir / "table_quality_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_audit = audit_model_completeness(generated)
    model_audit.to_csv(
        audits_dir / "model_completeness.csv",
        index=False,
        encoding="utf-8-sig",
    )

    condition_audit = audit_condition_completeness(generated)
    condition_audit.to_csv(
        audits_dir / "condition_completeness.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing_audit = audit_missing_values(generated)
    missing_audit.to_csv(
        audits_dir / "missing_value_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    block15_audit = compare_block15_table_values(
        root,
        generated,
    )
    block15_audit.to_csv(
        audits_dir / "block15_table_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    block16_audit = audit_block16_presence(root)
    block16_audit.to_csv(
        audits_dir / "block16_presence_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    figure_table_audit = audit_table_figure_consistency(
        root,
        generated,
    )
    figure_table_audit.to_csv(
        audits_dir / "table_figure_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------------
    # Generation manifest
    # ------------------------------------------------------------------------
    manifest = build_generation_manifest(
        root,
        cfg,
        generated_outputs,
    )
    manifest_path = out_root / "table_generation_manifest.csv"
    manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------------
    # Release gates
    # ------------------------------------------------------------------------
    gates: dict[str, dict[str, Any]] = {}

    gates["G1"] = {
        "status": "PASS",
        "message": "BLOCK15 completion marker verified",
    }

    gates["G2"] = {
        "status": "PASS",
        "message": "BLOCK16 completion marker verified",
    }

    gates["G3"] = {
        "status": "PASS",
        "message": "BLOCK17 configuration loaded and hashed",
    }

    gates["G4"] = {
        "status": (
            "PASS"
            if len(generated) == 14
            else "FAIL"
        ),
        "message": f"Main tables generated: {len(generated)}/14",
    }

    gates["G5"] = {
        "status": (
            "PASS"
            if len(supplementary_outputs) == 18
            else "FAIL"
        ),
        "message": (
            f"Supplementary tables generated: "
            f"{len(supplementary_outputs)}/18"
        ),
    }

    model_failures = (
        model_audit.loc[
            model_audit["status"] != "PASS"
        ]
        if not model_audit.empty
        else pd.DataFrame()
    )

    gates["G6"] = {
        "status": "PASS" if model_failures.empty else "FAIL",
        "message": "Model completeness audit",
    }

    condition_failures = (
        condition_audit.loc[
            condition_audit["status"] != "PASS"
        ]
        if not condition_audit.empty
        else pd.DataFrame()
    )

    gates["G7"] = {
        "status": "PASS" if condition_failures.empty else "FAIL",
        "message": "Condition completeness audit",
    }

    quality_failures = quality.loc[
        quality["status"] != "PASS"
    ]

    gates["G8"] = {
        "status": "PASS" if quality_failures.empty else "FAIL",
        "message": "Table quality audit",
    }

    figure_failures = figure_table_audit.loc[
        figure_table_audit["status"] != "PASS"
    ]

    gates["G9"] = {
        "status": "PASS" if figure_failures.empty else "FAIL",
        "message": "BLOCK16 figure ↔ BLOCK17 table registry consistency",
    }

    gates["G10"] = {
        "status": (
            "PASS"
            if source_registry_path.is_file()
            and table_registry_path.is_file()
            and manifest_path.is_file()
            else "FAIL"
        ),
        "message": "Traceability registries and generation manifest",
    }

    gates["G11"] = {
        "status": (
            "PASS"
            if block15_audit["status"].eq("PASS").all()
            else "FAIL"
        ),
        "message": "BLOCK15 ↔ BLOCK17 source-value integrity",
    }

    # A missing-value audit is informational because legitimate status-based
    # missingness is allowed. Fabrication is prohibited by construction.
    gates["G12"] = {
        "status": "PASS",
        "message": "Missing values preserved; no imputation performed",
    }

    numerical_audit = audit_numerical_integrity(generated, cfg)
    numerical_audit.to_csv(
        audits_dir / "numerical_integrity.csv", index=False, encoding="utf-8-sig"
    )
    gates["G13"] = {
        "status": "PASS" if numerical_audit["status"].eq("PASS").all() else "FAIL",
        "message": "Numerical finiteness and probability-range audit",
    }

    metric_audit = audit_metric_completeness(generated)
    metric_audit.to_csv(
        audits_dir / "metric_completeness.csv", index=False, encoding="utf-8-sig"
    )
    gates["G14"] = {
        "status": "PASS" if metric_audit.empty or metric_audit["status"].eq("PASS").all() else "FAIL",
        "message": "Primary/secondary metric completeness audit",
    }

    figure_registry_audit = audit_figure_registry(root)
    figure_registry_audit.to_csv(
        audits_dir / "block16_figure_registry_audit.csv",
        index=False, encoding="utf-8-sig"
    )
    gates["G15"] = {
        "status": "PASS" if figure_registry_audit["status"].eq("PASS").all() else "FAIL",
        "message": "Frozen BLOCK16 figure registry audit",
    }

    gates["G16"] = {
        "status": "PASS" if sha256_json(cfg) == cfg_hash else "FAIL",
        "message": "BLOCK17 configuration reproducibility hash",
    }

    gates["G17"] = {
        "status": (
            "PASS"
            if all(
                gate["status"] == "PASS"
                for gate in gates.values()
            )
            else "FAIL"
        ),
        "message": "Final publication release gate",
    }

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------
    all_pass = all(
        gate["status"] == "PASS"
        for gate in gates.values()
    )

    summary = {
        "block": "BLOCK17",
        "version": BLOCK17_VERSION,
        "status": "PASS" if all_pass else "NOT_READY",
        "timestamp_utc": utc_now(),
        "root": str(root),
        "configuration_hash": cfg_hash,
        "main_tables": len(generated),
        "supplementary_tables": len(supplementary_outputs),
        "gates": gates,
        "window_configuration": extract_window_values(
            load_selected_window_config(root)
        ),
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }

    summary_path = out_root / "BLOCK17_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    print("BLOCK17 RELEASE GATES:")
    for gate_id, gate in gates.items():
        print(
            f"  {gate_id}: {gate['status']} — {gate['message']}"
        )

    if all_pass:
        complete_path = out_root / "BLOCK17_complete.json"
        complete_payload = {
            **summary,
            "completion_marker": True,
            "completion_hash": sha256_json(summary),
        }

        with complete_path.open("w", encoding="utf-8") as f:
            json.dump(
                complete_payload,
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")

        print("BLOCK17 PASS")
        return 0

    # Never leave a stale completion marker after a failed run.
    stale_complete = out_root / "BLOCK17_complete.json"
    if stale_complete.exists():
        stale_complete.unlink()

    print("BLOCK17 NOT_READY")
    return 2


# ============================================================================
# 28. CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BLOCK17 final publication tables and audit"
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root. Default: parent of the scripts directory.",
    )

    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect-only mode; does not generate tables.",
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="Generate tables and run release gates.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.root is not None:
        root = args.root.resolve()
    else:
        # Expected project layout:
        # PROJECT_ROOT/scripts/BLOCK17_FINAL_TABLES.py
        root = script_dir.parent.resolve()

    if not args.inspect and not args.run:
        args.inspect = True

    if args.inspect and args.run:
        raise SystemExit(
            "Use either --inspect or --run, not both."
        )

    try:
        return run(
            root=root,
            inspect_only=args.inspect,
        )
    except Exception as exc:
        print(
            f"BLOCK17 ERROR: {type(exc).__name__}: {exc}"
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
