#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLOCK15_FINAL_STATISTICS.py  (canonical, single file)
=====================================================
Paper 2 -- Final Statistical Integration, Hypothesis Testing & Evidence Synthesis.

Design contract (spec 15.1-15.41)
---------------------------------
* The SUBJECT is the only inferential unit; window-level tables are rejected.
* Frozen upstream inference (BLOCK08 ME-MI, BLOCK10 asymmetry, BLOCK12 H4/H5) is IMPORTED, never re-run,
  and upstream-adjusted q-values are PRESERVED (no double FDR).
* New BLOCK15 families (H1 SD-LOSO gap, gap difference D_G) use the frozen config: two-sided Wilcoxon,
  subject-bootstrap CI, explicit BH-FDR families.
* Missing subjects are reported (ids + reason) and never imputed; incomplete cohorts are INSUFFICIENT_DATA.
* Config freeze is ENFORCED by hash; PASS is written only when every completion-gate check succeeds.
  Otherwise the status is NOT_READY or FAIL with explicit reasons.
* Nothing here retrains, re-predicts, or edits any upstream artifact.
* BLOCK12 authoritative tree is 12_REPRESENTATION_ANALYSIS; legacy 12_REPRESENTATION is not consumed.

Commands
--------
  --inspect        walk the real project tree and write a dependency/schema report (read-only)
  --init-config    write a DRAFT config (refuses to overwrite a FROZEN config)
  --freeze-config  validate the draft, compute the config hash, set status=FROZEN
  --validate-only  verify config freeze + upstream gates, no statistics
  (default)        definitive run
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import scipy
    from scipy import stats
except Exception as exc:  # pragma: no cover
    raise RuntimeError("BLOCK15 requires scipy.") from exc


# =============================================================================
# 0. IMMUTABLE STUDY CONSTANTS
# =============================================================================

BLOCK_ID = "BLOCK15"
BLOCK_NAME = "15_FINAL_STATISTICS"
CONFIG_VERSION = "2.2.0"

N_EXPECTED_SUBJECTS = 25
SUBJECTS = tuple(range(1, N_EXPECTED_SUBJECTS + 1))
CONDITIONS = ("ME", "MI")
GESTURES = tuple(range(1, 8))

PRIMARY_METRIC = "macro_f1"
SECONDARY_METRICS = ("accuracy", "balanced_accuracy", "mcc", "kappa")
ALL_METRICS = (PRIMARY_METRIC,) + SECONDARY_METRICS

MODELS_ML = ("LDA", "KNN", "SVM", "RandomForest", "XGBoost")
MODELS_DL = ("EEGNet", "ShallowConvNet", "DeepConvNet", "CNN-BiLSTM", "CNN-BiLSTM-Attention")
MODELS_LATENT = ("EEGNet", "CNN-BiLSTM", "CNN-BiLSTM-Attention")
ALL_MODELS = MODELS_ML + MODELS_DL

DROPOUT_LEVELS = (5, 10, 20, 30)
CHANCE_LEVEL = 1.0 / 7.0
FDR_Q = 0.05
DEFAULT_BOOTSTRAP_ITERATIONS = 10000
DEFAULT_PERMUTATION_ITERATIONS = 10000
DEFAULT_SEED = 20260915

# Controlled status vocabulary (single source of truth for registry / matrix / summary / audit / completion).
STATUS_VOCAB = (
    "SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE", "DESCRIPTIVE_ONLY", "NOT_TESTED",
    "INSUFFICIENT_DATA", "NOT_AVAILABLE", "NOT_REACHED", "NO_POSITIVE_GAP", "UPSTREAM_FROZEN", "TESTED",
)
INFERENTIAL_STATUSES = ("TESTED", "UPSTREAM_FROZEN")
NON_INFERENTIAL_STATUSES = ("DESCRIPTIVE_ONLY", "NOT_TESTED", "INSUFFICIENT_DATA", "NOT_AVAILABLE")

# Authoritative model-alias map (ONE mapping; every analysis goes through canonical_model()).
MODEL_ALIASES = {
    "lda": "LDA", "knn": "KNN", "svm": "SVM",
    "rf": "RandomForest", "random forest": "RandomForest", "randomforest": "RandomForest",
    "random-forest": "RandomForest", "random_forest": "RandomForest",
    "xgb": "XGBoost", "xgboost": "XGBoost",
    "eegnet": "EEGNet", "shallowconvnet": "ShallowConvNet", "deepconvnet": "DeepConvNet",
    "cnn-bilstm": "CNN-BiLSTM", "cnn_bilstm": "CNN-BiLSTM", "cnnbilstm": "CNN-BiLSTM",
    "cnn-bilstm-attention": "CNN-BiLSTM-Attention", "cnn_bilstm_attention": "CNN-BiLSTM-Attention",
    "cnnbilstmattention": "CNN-BiLSTM-Attention", "cnn-bilstm + self-attention": "CNN-BiLSTM-Attention",
}

CONDITION_NORMALIZATION = {
    "me": "ME", "motor_execution": "ME", "motor execution": "ME",
    "mi": "MI", "motor_imagery": "MI", "motor imagery": "MI",
}

# Aliases are for schema recognition only. NOTE: the ambiguous bare "f1" is intentionally NOT a macro_f1 alias.
COLUMN_ALIASES = {
    "subject": ("subject", "subject_id", "participant", "participant_id", "sub", "sid"),
    "condition": ("condition", "cond"),
    "model": ("model", "model_name", "classifier", "architecture"),
    "metric": ("metric", "measure"),
    "accuracy": ("accuracy", "acc"),
    "balanced_accuracy": ("balanced_accuracy", "balanced_acc", "balancedaccuracy", "bac"),
    "macro_f1": ("macro_f1", "macro_f1_score", "f1_macro", "macro-f1", "macro f1"),
    "mcc": ("mcc", "matthews_corrcoef", "matthews_correlation"),
    "kappa": ("kappa", "cohen_kappa", "cohen's_kappa", "cohens_kappa"),
    "rho": ("rho", "spearman_rho", "rho_spearman", "rsa_rho", "representation_similarity", "me_mi_rho"),
    "transfer_benefit": ("transfer_benefit", "transfer_gain"),
    "transfer_loss": ("transfer_loss", "transfer_degradation"),
    "k90": ("k90", "k_90", "calibration_k90"),
    "fraction": ("fraction", "calibration_fraction", "calibration_percent"),
    "value": ("value", "score", "performance"),
}

# Upstream layout: CANDIDATES ONLY. `--inspect` verifies them against the real tree; nothing is assumed silently.
UPSTREAM_LAYOUT = {
    "06A_ML_SD": ("06_ML/06A_ML_SD", None),
    "06B_ML_LOSO": ("06_ML/06B_ML_LOSO", None),
    "07A_DL_SD": ("07_DL/07A_DL_SD", None),
    "07B_DL_LOSO": ("07_DL/07B_DL_LOSO", None),
    "08_ME_MI_STATISTICS": ("08_ME_MI_STATISTICS", "08_ME_MI_STATISTICS/BLOCK08_complete.json"),
    "09_TEMPORAL_CALIBRATION": ("09_TEMPORAL_CALIBRATION", "09_TEMPORAL_CALIBRATION/BLOCK09_complete.json"),
    "10_ME_MI_TRANSFER": ("10_ME_MI_TRANSFER", "10_ME_MI_TRANSFER/BLOCK10_complete.json"),
    "11_NEURO": ("11_NEURO", "11_NEURO/BLOCK11_complete.json"),
    "12_REPRESENTATION": ("12_REPRESENTATION_ANALYSIS", "12_REPRESENTATION_ANALYSIS/BLOCK12_complete.json"),
    "13_XAI": ("13_XAI_FROZEN_OUTPUTS", None),
    "14_ROBUSTNESS": ("14_ROBUSTNESS", "14_ROBUSTNESS/BLOCK14_complete.json"),
}
REQUIRED_UPSTREAM_BLOCKS = tuple(UPSTREAM_LAYOUT)

# Artifacts named in the project brief (exact paths relative to the Paper2 root). None = must be set after --inspect.
KNOWN_ARTIFACTS = {
    "06A_ML_SD": {
        "sd_execution_audit": "06_ML/06A_ML_SD/SD_execution_audit.csv",
        "class_metrics_dir": "06_ML/06A_ML_SD/class_metrics",
        "confusion_matrices_dir": "06_ML/06A_ML_SD/confusion_matrices",
    },
    "06B_ML_LOSO": {
        "subject_metrics": "06_ML/06B_ML_LOSO/subject_results/LOSO_subject_results.csv",
    },
    "07A_DL_SD": {
        "subject_metrics": "07_DL/07A_DL_SD/subject_results.csv",
    },
    "07B_DL_LOSO": {
        "subject_metrics": "07_DL/07B_DL_LOSO/subject_results.csv",
    },
    "08_ME_MI_STATISTICS": {
        "paired_statistics": "08_ME_MI_STATISTICS/statistics/ME_MI_statistical_summary.csv",
    },
    "09_TEMPORAL_CALIBRATION": {
        "k90_results": "09_TEMPORAL_CALIBRATION/metrics/k90_results.csv",
        "calibration_grid_source": "09_TEMPORAL_CALIBRATION/diagnostics/A_SD_primary_combined.csv",
    },
    "10_ME_MI_TRANSFER": {
        "transfer_benefit": "10_ME_MI_TRANSFER/metrics/transfer_benefit.csv",
        "asymmetry_test": "10_ME_MI_TRANSFER/statistics/asymmetry_test.csv",
    },
    "11_NEURO": {
        "spectral_summary": "11_NEURO/statistics/spectral_statistical_summary.csv",
    },
    "12_REPRESENTATION": {
        "subject_rsa": "12_REPRESENTATION_ANALYSIS/subject_similarity/subject_level_rsa.csv",
        "rsa_transfer_subject": "12_REPRESENTATION_ANALYSIS/transfer_relationship/rsa_vs_transfer_subject_level.csv",
        "rsa_transfer_summary": "12_REPRESENTATION_ANALYSIS/transfer_relationship/rsa_transfer_association_summary.csv",
        "rsa_k90_subject": "12_REPRESENTATION_ANALYSIS/calibration_relationship/rsa_vs_k90_subject_level.csv",
        "rsa_k90_summary": "12_REPRESENTATION_ANALYSIS/calibration_relationship/rsa_k90_association_summary.csv",
    },
    "13_XAI": {
        "xai_summary": "13_XAI_FROZEN_OUTPUTS/tables/XAI_summary.csv",
        "attribution_similarity_summary": "13_XAI_FROZEN_OUTPUTS/tables/ME_MI_attribution_similarity_summary.csv",
        "paired_channel_statistics": "13_XAI_FROZEN_OUTPUTS/tables/XAI_ME_MI_paired_channel_statistics.csv",
        "attribution_similarity_statistics": "13_XAI_FROZEN_OUTPUTS/tables/XAI_attribution_similarity_statistics.csv",
    },
    "14_ROBUSTNESS": {
        "robustness_subject_level": "14_ROBUSTNESS/noise/BLOCK14_noise_subject_level.csv",
        "dropout_subject_level": "14_ROBUSTNESS/channel_dropout/BLOCK14_channel_dropout_subject_level.csv",
        "clean_reference_identity_audit": "14_ROBUSTNESS/tables/audit/BLOCK14_upstream_clean_reference_check.csv",
    },
}

# Spec 15.31 output tree (relative to 15_FINAL_STATISTICS/).
OUTPUT_DIRS = ("descriptive", "hypothesis_tests", "gap_analysis", "transfer", "calibration", "representation",
               "xai", "neuro", "robustness", "computational", "multiple_comparisons", "hypothesis_matrix",
               "statistical_registry", "audits", "configs", "tables", "figures", "completion")
REQUIRED_OUTPUTS = (
    "descriptive/subject_level_performance.csv", "descriptive/aggregate_performance.csv",
    "descriptive/confidence_intervals.csv", "descriptive/distribution_summary.csv",
    "hypothesis_tests/H1_subject_dependency.csv", "hypothesis_tests/H2_ME_MI_similarity.csv",
    "hypothesis_tests/H3_transfer.csv", "hypothesis_tests/H4_representation_transfer.csv",
    "hypothesis_tests/H5_representation_calibration.csv", "hypothesis_tests/H6_ML_DL_comparison.csv",
    "gap_analysis/SD_LOSO_subject_gap.csv", "gap_analysis/ME_MI_gap_comparison.csv", "gap_analysis/gap_summary.csv",
    "transfer/transfer_summary.csv", "transfer/transfer_asymmetry.csv", "transfer/negative_transfer_summary.csv",
    "calibration/calibration_summary.csv", "calibration/k90_summary.csv", "calibration/calibration_efficiency_summary.csv",
    "representation/RSA_summary.csv", "representation/representation_transfer.csv", "representation/representation_calibration.csv",
    "xai/XAI_summary.csv", "xai/attribution_similarity_summary.csv",
    "neuro/spectral_summary.csv", "neuro/ME_MI_spectral_comparison.csv",
    "robustness/robustness_summary.csv", "robustness/dropout_summary.csv", "robustness/noise_summary.csv",
    "computational/ML_DL_computational_summary.csv",
    "multiple_comparisons/FDR_registry.csv", "multiple_comparisons/adjusted_results.csv",
    "hypothesis_matrix/hypothesis_evidence_summary.csv", "statistical_registry/statistical_test_registry.csv",
    "audits/upstream_completion_audit.csv", "audits/subject_integrity_audit.csv",
    "audits/statistical_input_audit.csv", "audits/reproducibility_audit.csv",
    "audits/statistical_registry_audit.csv", "configs/BLOCK15_config.json",
)

REGISTRY_FIELDS = (
    "test_id", "hypothesis", "family", "model", "condition", "metric", "unit", "n", "test", "effect_size",
    "estimate", "ci_lower", "ci_upper", "ci_status", "raw_p", "q", "direction", "status", "evidence_status",
    "source_block", "fdr_family", "fdr_action", "fdr_family_n", "upstream_q", "missing_subjects", "notes",
)
MATRIX_FIELDS = (
    "hypothesis", "hypothesis_description", "analysis_family", "model", "condition", "metric", "N", "estimate",
    "effect_size", "CI_lower", "CI_upper", "raw_p", "q", "direction", "expected_direction", "status",
    "evidence_status", "source_block", "interpretation",
)
HYPOTHESIS_TEXT = {
    "H1": "Subject-dependent decoding exceeds LOSO decoding (SD > LOSO)",
    "H1_GAP": "The SD-LOSO gap differs between ME and MI (D_G = G_ME - G_MI)",
    "H2": "ME-MI representational similarity (subject-level rho) is measurable with retained subject variability",
    "H3": "Cross-condition transfer is bidirectional and may be asymmetric",
    "H4": "Greater ME-MI representation similarity is associated with greater transfer benefit",
    "H5": "Greater ME-MI representation similarity is associated with lower k90 (calibration requirement)",
    "H6": "ML/DL comparative evidence (descriptive; no composite score, no winner)",
}
INTERPRETATION = {
    "H1": "Positive gap = higher SD than LOSO performance. Direction is evaluated separately from significance.",
    "H1_GAP": "Difference in the SD-LOSO gap between conditions; not a mechanism claim.",
    "H2": "Representational similarity only; not identical neural activity or a shared mechanism.",
    "H3": "Asymmetry of transfer loss (ME->MI minus MI->ME); frozen BLOCK10 inference imported, not re-tested.",
    "H4": "Association only; no causal inference.",
    "H5": "Association only; no causal inference. NOT_REACHED / NO_POSITIVE_GAP never converted to numeric k90.",
    "H6": "Descriptive ML/DL characterisation; no composite score.",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_project_root() -> Path:
    """
    Resolve Paper2 root without assuming a particular Windows drive.

    Priority:
      1. PAPER2_ROOT environment variable
      2. --root argument (applied in main)
      3. parent hierarchy containing the expected block directories
      4. current working directory
    """
    env = os.environ.get("PAPER2_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    candidates = [Path.cwd().resolve()]
    here = Path(__file__).resolve().parent
    candidates.extend([here, *here.parents])

    for p in candidates:
        names = {x.name for x in p.iterdir()} if p.is_dir() else set()
        if "15_FINAL_STATISTICS" in names or "14_ROBUSTNESS" in names:
            return p

    return Path.cwd().resolve()


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, level: str, message: str) -> None:
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {level:<7} | {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def info(self, msg: str) -> None:
        self._write("INFO", msg)

    def warning(self, msg: str) -> None:
        self._write("WARN", msg)

    def error(self, msg: str) -> None:
        self._write("ERROR", msg)


def atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON: {path}") from exc


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            df = pd.read_json(path, lines=True)
        else:
            obj = load_json(path)
            if isinstance(obj, list):
                df = pd.DataFrame(obj)
            elif isinstance(obj, dict):
                # Accept a standard record-list container only.
                records = None
                for key in ("records", "rows", "data", "results", "table"):
                    if isinstance(obj.get(key), list):
                        records = obj[key]
                        break
                if records is None:
                    raise RuntimeError(
                        f"JSON does not contain a tabular record list: {path}"
                    )
                df = pd.DataFrame(records)
            else:
                raise RuntimeError(f"Unsupported JSON tabular structure: {path}")
    else:
        raise RuntimeError(f"Unsupported BLOCK15 table artifact: {path}")

    if df.empty:
        raise RuntimeError(f"Configured artifact is empty: {path}")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalized_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def find_column(df: pd.DataFrame, logical: str, required: bool = False) -> Optional[str]:
    aliases = {normalized_col(x) for x in COLUMN_ALIASES.get(logical, ())}
    mapping = {normalized_col(c): c for c in df.columns}

    for alias in aliases:
        if alias in mapping:
            return mapping[alias]

    if required:
        raise RuntimeError(
            f"Required logical column {logical!r} not found. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def canonical_subject(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("Missing subject ID")
    text = str(value).strip()
    m = re.search(r"(\d+)", text)
    if not m:
        raise ValueError(f"Cannot parse subject ID: {value!r}")
    subject = int(m.group(1))
    if subject not in SUBJECTS:
        raise ValueError(f"Subject outside canonical cohort 1..25: {subject}")
    return subject


def canonical_condition(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("Missing condition")
    key = str(value).strip().lower().replace("-", "_")
    key = re.sub(r"\s+", " ", key)
    if key in CONDITION_NORMALIZATION:
        return CONDITION_NORMALIZATION[key]
    raise ValueError(f"Unknown condition: {value!r}")


def coerce_numeric(series: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().any():
        bad = series[out.isna()].head(10).tolist()
        raise RuntimeError(f"Non-numeric values in {name}: {bad}")
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise RuntimeError(f"NaN/Inf values in {name}")
    return out.astype(float)


def standardize_subject_table(
    df: pd.DataFrame,
    *,
    require_model: bool = False,
    require_condition: bool = False,
    require_metric: bool = False,
) -> pd.DataFrame:
    s_col = find_column(df, "subject", required=True)
    c_col = find_column(df, "condition", required=require_condition)
    m_col = find_column(df, "model", required=require_model)
    metric_col = find_column(df, "metric", required=require_metric)

    out = df.copy()
    out["subject"] = out[s_col].map(canonical_subject)

    if c_col:
        out["condition"] = out[c_col].map(canonical_condition)
    if m_col:
        out["model"] = out[m_col].map(canonical_model)
    if metric_col:
        out["metric"] = out[metric_col].astype(str).str.strip()

    for logical in (
        "accuracy", "balanced_accuracy", "macro_f1", "mcc", "kappa",
        "rho", "transfer_benefit", "transfer_loss", "k90", "fraction", "value"
    ):
        col = find_column(out, logical, required=False)
        if col and logical not in out.columns:
            out[logical] = coerce_numeric(out[col], logical)

    return out


def reject_window_level_data(df: pd.DataFrame, context: str) -> None:
    """
    BLOCK15 must not accept a table whose apparent inferential rows are
    window-level. This is deliberately conservative.
    """
    names = {normalized_col(c) for c in df.columns}
    suspicious = {
        "window_id", "window_index", "window_number",
        "start_sample", "end_sample", "window_array_file"
    }
    if names.intersection(suspicious) and "subject" in names:
        # A table can contain window-level fields for provenance, but it cannot
        # be used directly as a subject-level inferential table. Callers must
        # aggregate explicitly from already frozen upstream subject metrics.
        raise RuntimeError(
            f"{context}: apparent window-level table supplied to subject-level "
            "inference. BLOCK15 refuses implicit window aggregation."
        )


def validate_unique_subject_observations(
    df: pd.DataFrame,
    keys: list[str],
    context: str,
) -> None:
    if df.duplicated(keys).any():
        dup = df.loc[df.duplicated(keys, keep=False), keys]
        raise RuntimeError(
            f"{context}: duplicate subject-level observations for keys={keys}:\n"
            f"{dup.head(20).to_string(index=False)}"
        )


def percentile_ci(
    values: np.ndarray,
    *,
    seed: int,
    iterations: int,
    statistic: str = "mean",
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (math.nan, math.nan)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(iterations, len(x)))
    samples = x[idx]

    if statistic == "mean":
        boot = samples.mean(axis=1)
    elif statistic == "median":
        boot = np.median(samples, axis=1)
    else:
        raise ValueError(statistic)

    return (
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def summarize_vector(
    values: Iterable[float],
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "n": 0, "mean": math.nan, "median": math.nan, "sd": math.nan,
            "iqr": math.nan, "min": math.nan, "max": math.nan,
            "ci95_low": math.nan, "ci95_high": math.nan,
        }

    ci_lo, ci_hi = percentile_ci(
        x, seed=seed, iterations=iterations, statistic="mean"
    )
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else math.nan,
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
    }


def rank_biserial_from_wilcoxon(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if len(d) == 0:
        return math.nan

    ranks = stats.rankdata(np.abs(d), method="average")
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    return float((w_plus - w_minus) / (w_plus + w_minus))


def paired_wilcoxon(
    a: Iterable[float],
    b: Iterable[float],
    *,
    min_n: int,
) -> dict[str, Any]:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    d = x - y

    if len(d) < min_n:
        return {
            "status": "NOT_TESTABLE",
            "n": int(len(d)),
            "estimate": math.nan,
            "median_difference": math.nan,
            "effect_size": math.nan,
            "p_raw": math.nan,
            "reason": f"N<{min_n}",
        }

    try:
        result = stats.wilcoxon(
            d,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        p = float(result.pvalue)
    except ValueError:
        p = math.nan

    return {
        "status": "TESTED" if np.isfinite(p) else "NOT_TESTABLE",
        "n": int(len(d)),
        "estimate": float(np.mean(d)),
        "median_difference": float(np.median(d)),
        "effect_size": rank_biserial_from_wilcoxon(d),
        "p_raw": p,
        "reason": "",
    }


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)

    finite = np.isfinite(p)
    if not finite.any():
        return q

    vals = p[finite]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)

    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[finite] = restored
    return q


def prepare_metric_long(df: pd.DataFrame, context: str) -> pd.DataFrame:
    reject_window_level_data(df, context)

    out = standardize_subject_table(
        df,
        require_model=True,
        require_condition=True,
    )

    metric_columns = {}
    for metric in (PRIMARY_METRIC, "accuracy", "balanced_accuracy", "mcc", "kappa"):
        col = find_column(out, metric, required=False)
        if col:
            metric_columns[metric] = col
            if metric not in out.columns:
                out[metric] = coerce_numeric(out[col], metric)

    if not metric_columns:
        raise RuntimeError(
            f"{context}: no supported subject-level metric columns found."
        )

    # Preserve only rows that contain at least one metric.
    keep = ["subject", "condition", "model"] + list(metric_columns.keys())
    out = out[keep].copy()

    validate_unique_subject_observations(
        out, ["subject", "condition", "model"], context
    )
    return out


def extract_rsa(df: pd.DataFrame, context: str) -> pd.DataFrame:
    reject_window_level_data(df, context)

    out = standardize_subject_table(
        df,
        require_model=True,
        require_condition=False,
    )
    rho_col = find_column(out, "rho", required=False)

    if rho_col is None and "rho" not in out.columns:
        raise RuntimeError(f"{context}: RSA rho column not found.")

    if "rho" not in out.columns:
        out["rho"] = coerce_numeric(out[rho_col], "rho")

    # RSA should be one subject/model observation unless BLOCK12 explicitly
    # supplies a different frozen aggregation unit.
    validate_unique_subject_observations(out, ["subject", "model"], context)
    return out[["subject", "model", "rho"]].copy()


# =============================================================================
# 1. MODEL NORMALISATION (single authority)
# =============================================================================

def canonical_model(value: Any) -> str:
    """Map any upstream alias to the canonical Paper 2 model name; unknown names FAIL CLOSED."""
    if pd.isna(value):
        raise ValueError("Missing model")
    raw = str(value).strip()
    key = re.sub(r"\s+", " ", raw.lower().replace("\u2013", "-").replace("\u2014", "-"))
    for k in (key, key.replace(" ", ""), key.replace("_", "-"), key.replace(" ", "-")):
        if k in MODEL_ALIASES:
            return MODEL_ALIASES[k]
    for cand in ALL_MODELS:
        if key == cand.lower():
            return cand
    raise ValueError(f"Unknown model alias {raw!r}; add it to MODEL_ALIASES only after verifying it upstream")


def normalize_model_column(df: pd.DataFrame, col: str = "model") -> pd.DataFrame:
    out = df.copy()
    out[col] = out[col].map(canonical_model)
    return out


# =============================================================================
# 2. CONFIGURATION + FREEZE GATE
# =============================================================================

def default_config() -> dict[str, Any]:
    upstream = {}
    for block, (bdir, marker) in UPSTREAM_LAYOUT.items():
        evidence_map = {
            "06A_ML_SD": "06_ML/06A_ML_SD/SD_execution_audit.csv",
            "06B_ML_LOSO": "06_ML/06B_ML_LOSO/subject_results/LOSO_subject_results.csv",
            "07A_DL_SD": "07_DL/07A_DL_SD/subject_results.csv",
            "07B_DL_LOSO": "07_DL/07B_DL_LOSO/subject_results.csv",
            "13_XAI": "13_XAI_FROZEN_OUTPUTS/tables/XAI_summary.csv",
        }
        upstream[block] = {
            "block_dir": bdir,
            "completion_marker": marker,
            "completion_evidence": evidence_map.get(block),
            "layout_verified": False,
            "artifacts": copy.deepcopy(KNOWN_ARTIFACTS[block]),
        }
    return {
        "block_id": BLOCK_ID,
        "config_version": CONFIG_VERSION,
        "status": "DRAFT",
        "freeze": {
            "primary_metric": PRIMARY_METRIC,
            "secondary_metrics": list(SECONDARY_METRICS),
            "inferential_unit": "subject",
            "expected_subjects": N_EXPECTED_SUBJECTS,
            "subjects": list(SUBJECTS),
            "conditions": list(CONDITIONS),
            "gestures": list(GESTURES),
            "paired_matching_keys": ["subject", "condition", "model"],
            "seed": DEFAULT_SEED,
            "bootstrap_iterations": DEFAULT_BOOTSTRAP_ITERATIONS,
            "permutation_iterations": DEFAULT_PERMUTATION_ITERATIONS,
            "ci_method": "subject_level_percentile_bootstrap_of_mean",
            "new_paired_test": "wilcoxon_signed_rank_two_sided",
            "new_association_test": "spearman",
            "fdr_method": "benjamini_hochberg",
            "fdr_q": FDR_Q,
            "h2_inferential_test": None,
            "calibration_fractions": [0, 10, 20, 30, 40, 50, 60, 80, 90],
            "calibration_fractions_note": "Frozen Paper 2 calibration grid: 0,10,20,30,40,50,60,80,90%.",
            "expected_config_sha256": None,
            "frozen_utc": None,
        },
        "model_mapping": {"aliases": MODEL_ALIASES, "ml": list(MODELS_ML), "dl": list(MODELS_DL),
                          "latent": list(MODELS_LATENT)},
        "inclusion": {"models": list(ALL_MODELS), "conditions": list(CONDITIONS), "metrics": list(ALL_METRICS)},
        "missing_data": {"impute_missing": False, "h1_require_complete_cohort": True,
                         "expected_n": N_EXPECTED_SUBJECTS, "min_valid_subjects": N_EXPECTED_SUBJECTS},
        "fdr": {
            "alpha": FDR_Q,
            "families": {
                "H1_PRIMARY": {"hypothesis": "H1", "metrics": [PRIMARY_METRIC]},
                "H1_SECONDARY": {"hypothesis": "H1", "metrics": list(SECONDARY_METRICS)},
                "H1_GAP_PRIMARY": {"hypothesis": "H1_GAP", "metrics": [PRIMARY_METRIC]},
                "H1_GAP_SECONDARY": {"hypothesis": "H1_GAP", "metrics": list(SECONDARY_METRICS)},
                "H4_RSA_TRANSFER": {"hypothesis": "H4", "metrics": []},
                "H5_RSA_CALIBRATION": {"hypothesis": "H5", "metrics": []},
            },
            "rule": ("A registry row carrying a frozen upstream q (BLOCK10, BLOCK12 if present) keeps it. Only rows "
                     "with a BLOCK15-generated or upstream-raw p and no upstream q are BH-corrected, once, inside "
                     "their configured family. BLOCK08 inference is imported without correction."),
        },
        "hypotheses": {
            "H1": {"status": "confirmatory", "expected_sign": +1, "primary_metric": PRIMARY_METRIC},
            "H1_GAP": {"status": "supporting_exploratory", "expected_sign": 0},
            "H2": {"status": "confirmatory_descriptive", "expected_sign": None},
            "H3": {"status": "confirmatory_imported", "expected_sign": 0},
            "H4": {"status": "confirmatory_imported", "expected_sign": +1},
            "H5": {"status": "confirmatory_imported", "expected_sign": -1},
            "H6": {"status": "descriptive", "expected_sign": None},
        },
        "upstream": upstream,
        "notes": {"artifact_paths_must_be_exact": True, "do_not_overwrite_upstream": True,
                  "windows_are_not_inferential_units": True,
                  "post_hoc_analyses_must_be_labelled_exploratory": True},
    }


def config_canonical_hash(cfg: dict[str, Any]) -> str:
    """Deterministic hash of the frozen content (excludes status + the hash/timestamp fields themselves)."""
    c = copy.deepcopy(cfg)
    c.pop("status", None)
    fr = c.get("freeze", {})
    fr.pop("expected_config_sha256", None)
    fr.pop("frozen_utc", None)
    blob = json.dumps(c, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def config_path_for(root: Path) -> Path:
    return root / BLOCK_NAME / "configs" / "BLOCK15_config.json"


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"BLOCK15_config.json not found: {path}. Run --init-config, then --inspect.")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("block_id") != BLOCK_ID:
        raise RuntimeError(f"Config block_id mismatch: expected {BLOCK_ID}.")
    if cfg.get("config_version") != CONFIG_VERSION:
        raise RuntimeError(
            f"STALE BLOCK15 CONFIG: found version {cfg.get('config_version')!r}, "
            f"expected {CONFIG_VERSION}. Run --init-config to reconstruct the corrected configuration."
        )
    return cfg


def validate_config_structure(cfg: dict[str, Any]) -> list[str]:
    """Return a list of problems (empty = structurally valid). Scientific invariants are non-negotiable."""
    p: list[str] = []
    fr = cfg.get("freeze", {})
    if fr.get("inferential_unit") != "subject":
        p.append("inferential_unit must be 'subject'")
    if fr.get("primary_metric") != PRIMARY_METRIC:
        p.append("primary_metric must be macro_f1")
    if fr.get("fdr_method") != "benjamini_hochberg" or not math.isclose(float(fr.get("fdr_q", -1)), FDR_Q):
        p.append("FDR must be Benjamini-Hochberg q=0.05")
    if int(fr.get("expected_subjects", -1)) != N_EXPECTED_SUBJECTS:
        p.append("expected_subjects must be 25")
    if cfg.get("missing_data", {}).get("impute_missing", True):
        p.append("imputation is forbidden")
    if fr.get("new_paired_test") != "wilcoxon_signed_rank_two_sided":
        p.append("new_paired_test must be wilcoxon_signed_rank_two_sided (spec 15.7A default)")
    fams = cfg.get("fdr", {}).get("families", {})
    if not fams:
        p.append("fdr.families must be explicit")
    secondary = set(SECONDARY_METRICS)
    for name, spec in fams.items():
        mets = set(spec.get("metrics", []))
        if PRIMARY_METRIC in mets and mets & secondary:
            p.append(f"FDR family {name} mixes primary and secondary metrics")
    return p


def freeze_problems(cfg: dict[str, Any]) -> list[str]:
    """Problems that prevent a definitive run: structure, unset artifacts, missing hash, hash drift."""
    p = validate_config_structure(cfg)
    fr = cfg["freeze"]
    if cfg.get("status") != "FROZEN":
        p.append("config status is not FROZEN (run --freeze-config after --inspect)")
    exp = fr.get("expected_config_sha256")
    if not exp:
        p.append("no expected_config_sha256 recorded")
    elif config_canonical_hash(cfg) != exp:
        p.append(f"CONFIG DRIFT: current hash {config_canonical_hash(cfg)[:16]}... != frozen {str(exp)[:16]}...")
    if not fr.get("calibration_fractions"):
        p.append("calibration_fractions not frozen from BLOCK09")
    for block, spec in cfg.get("upstream", {}).items():
        for key, path in spec.get("artifacts", {}).items():
            if path in (None, ""):
                p.append(f"{block}.{key}: artifact path not set")
    return p


# =============================================================================
# 3. UPSTREAM GATES
# =============================================================================

def _abs(root: Path, raw: str) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else root / p


def verify_completion_marker(path: Path, block: str) -> tuple[bool, str, dict[str, Any]]:
    if not path or str(path) in {"", ".", "None"} or not path.is_file():
        return False, f"completion marker missing: {path}", {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"completion marker unreadable: {exc}", {}
    status = str(obj.get("status", "")).upper()
    completed = obj.get("completed", None)
    ok = status in {"PASS", "COMPLETED", "COMPLETE", "SUCCESS"} or completed is True
    if status in {"FAIL", "FAILED", "ERROR", "INCOMPLETE", "PARTIAL", "NOT_READY"} or not ok:
        return False, f"marker is not a definitive PASS (status={status!r}, completed={completed!r})", obj
    if block == "14_ROBUSTNESS":
        for k in ("failed_units", "missing_units", "duplicate_count", "nonfinite_count"):
            if int(obj.get(k, -1)) != 0:
                return False, f"BLOCK14 marker reports {k}={obj.get(k)!r}; expected 0", obj
        if int(obj.get("expected_units", -1)) != int(obj.get("completed_units", -2)) or int(obj.get("expected_units", 0)) <= 0:
            return False, (f"BLOCK14 expected_units={obj.get('expected_units')} != completed_units={obj.get('completed_units')}"), obj
        self_check = obj.get("self_check")
        if isinstance(self_check, dict):
            for key in ("prediction_identity_failures", "identity_failures", "clean_reference_prediction_mismatches"):
                if key in self_check and int(self_check[key]) != 0:
                    return False, f"BLOCK14 self_check reports {key}={self_check[key]!r}; expected 0", obj
    return True, "PASS", obj


def verify_completion_evidence(path: Path, block: str) -> tuple[bool, str, dict[str, Any]]:
    """Validate real-tree evidence where no single top-level completion marker exists."""
    if not path or not path.is_file() or path.stat().st_size == 0:
        return False, f"completion evidence missing/empty: {path}", {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return False, f"completion evidence unreadable: {exc}", {}
    if df.empty:
        return False, f"completion evidence empty: {path}", {}

    expected_rows = {"06A_ML_SD": 250, "06B_ML_LOSO": 250, "07A_DL_SD": 250, "07B_DL_LOSO": 250}
    if block in expected_rows:
        if len(df) != expected_rows[block]:
            return False, f"{block} evidence rows={len(df)}; expected {expected_rows[block]}", {"rows": len(df)}
        if "subject" in df.columns and df["subject"].nunique() != 25:
            return False, f"{block} evidence has {df['subject'].nunique()} subjects; expected 25", {}
        if "condition" in df.columns and set(df["condition"].astype(str).str.upper()) != {"ME", "MI"}:
            return False, f"{block} evidence does not contain exactly ME and MI", {}
        if "model" in df.columns:
            models = {canonical_model(x) for x in df["model"].dropna().unique()}
            expected = set(MODELS_ML if block.startswith("06") else MODELS_DL)
            if models != expected:
                return False, f"{block} model coverage {sorted(models)} != {sorted(expected)}", {}
        if block == "06A_ML_SD" and "status" in df.columns:
            status = df["status"].astype(str).str.strip().str.upper()
            explicit_failure = status.isin({
                "FAIL", "FAILED", "ERROR", "INCOMPLETE", "NOT_READY", "MISSING",
                "ABORTED", "CANCELLED", "CANCELED", "CRASHED", "EXCEPTION"
            })
            if explicit_failure.any():
                return False, f"06A execution audit contains {int(explicit_failure.sum())} explicit failure rows", {}
            # an empty CSV cell is read as NaN, which astype(str) never turns into "": test missingness explicitly
            blank_status = df["status"].isna() | status.eq("")
            if blank_status.any():
                return False, f"06A execution audit contains {int(blank_status.sum())} blank status values", {}
        return True, "PASS", {"rows": len(df), "columns": list(df.columns)}

    if block == "13_XAI":
        root = path
        while root.name != "Paper2_EEG_BCI" and root.parent != root:
            root = root.parent
        required = [
            "13_XAI_FROZEN_OUTPUTS/tables/XAI_summary.csv",
            "13_XAI_FROZEN_OUTPUTS/tables/ME_MI_attribution_similarity_summary.csv",
            "13_XAI_FROZEN_OUTPUTS/tables/XAI_ME_MI_paired_channel_statistics.csv",
            "13_XAI_FROZEN_OUTPUTS/tables/XAI_attribution_similarity_statistics.csv",
        ]
        missing = [rel for rel in required if not (root / rel).is_file() or (root / rel).stat().st_size == 0]
        if missing:
            return False, f"BLOCK13 frozen XAI artifacts missing/empty: {missing}", {"rows": len(df)}
        return True, "PASS", {"rows": len(df), "required_artifacts": required}

    return True, "PASS", {"rows": len(df), "columns": list(df.columns)}

def verify_upstream(cfg: dict[str, Any], root: Path, log: "Logger") -> tuple[dict[str, Any], list[str], pd.DataFrame]:
    evidence: dict[str, Any] = {}
    problems: list[str] = []
    rows = []
    for block in REQUIRED_UPSTREAM_BLOCKS:
        spec = cfg["upstream"][block]
        marker_raw = spec.get("completion_marker")
        evidence_raw = spec.get("completion_evidence")
        if marker_raw:
            marker = _abs(root, marker_raw)
            ok, why, obj = verify_completion_marker(marker, block)
            gate_path = marker
        elif evidence_raw:
            marker = _abs(root, evidence_raw)
            ok, why, obj = verify_completion_evidence(marker, block)
            gate_path = marker
        else:
            ok, why, obj = False, f"no completion marker/evidence configured for {block}", {}
            gate_path = root / "None"
        if not ok:
            problems.append(f"{block}: {why}")
        evidence[block] = {"completion_gate": str(gate_path), "gate_ok": ok, "gate_reason": why,
                           "gate_sha256": sha256_file(gate_path) if gate_path.is_file() else None, "artifacts": {}}
        rows.append({"block": block, "item": "completion_gate", "path": str(gate_path), "exists": gate_path.is_file(),
                     "sha256": evidence[block]["gate_sha256"], "status": "PASS" if ok else "FAIL", "detail": why})
        for key, raw in spec.get("artifacts", {}).items():
            if raw in (None, ""):
                problems.append(f"{block}.{key}: artifact path not configured")
                rows.append({"block": block, "item": key, "path": None, "exists": False, "sha256": None,
                             "status":"FAIL", "detail":"not configured"})
                continue
            pth = _abs(root, raw)
            if pth.is_dir():
                try:
                    exists = any(pth.iterdir())
                except OSError:
                    exists = False
                sha = None
            else:
                exists = pth.is_file() and pth.stat().st_size > 0
                sha = sha256_file(pth) if exists else None
            if not exists:
                problems.append(f"{block}.{key}: artifact missing or empty: {pth}")
            evidence[block]["artifacts"][key] = {"path": str(pth), "sha256": sha}
            rows.append({"block": block, "item": key, "path": str(pth), "exists": exists, "sha256": sha,
                         "status":"PASS" if exists else "FAIL", "detail":""})
        log.info(f"{block}: completion={'OK' if ok else 'NOT OK'}; artifacts={len(spec.get('artifacts', {}))}")
    return evidence, problems, pd.DataFrame(rows)

def validate_runtime_artifacts(cfg: dict[str, Any], root: Path, log: "Logger") -> tuple[bool, list[str], pd.DataFrame]:
    """Deep, read-only preflight of the frozen inputs before config freeze/definitive run.

    This validates the exact real-tree schemas that BLOCK15 consumes. It never writes
    upstream data and never performs inferential testing.
    """
    problems: list[str] = []
    rows: list[dict[str, Any]] = []

    def check_file(label: str, rel: str) -> Path | None:
        path = _abs(root, rel)
        ok = path.is_file() and path.stat().st_size > 0
        rows.append({"item": label, "path": str(path), "exists": ok, "status": "PASS" if ok else "FAIL"})
        if not ok:
            problems.append(f"{label}: missing or empty: {path}")
            return None
        return path

    # Exact BLOCK06/07 subject-level inputs.
    exact_csv = {
        "06A_ML_SD": "06_ML/06A_ML_SD/SD_execution_audit.csv",
        "06B_ML_LOSO": "06_ML/06B_ML_LOSO/subject_results/LOSO_subject_results.csv",
        "07A_DL_SD": "07_DL/07A_DL_SD/subject_results.csv",
        "07B_DL_LOSO": "07_DL/07B_DL_LOSO/subject_results.csv",
    }
    for block, rel in exact_csv.items():
        path = check_file(block, rel)
        if path is None:
            continue
        try:
            df = read_table(path)
            if block == "06A_ML_SD":
                required = {"subject", "condition", "model", "status"}
                if not required.issubset(df.columns):
                    raise RuntimeError(f"missing columns {sorted(required-set(df.columns))}")
                if len(df) != 250 or df["subject"].nunique() != 25:
                    raise RuntimeError(f"rows={len(df)}, subjects={df['subject'].nunique()}; expected 250/25")
                if set(df["condition"].astype(str).str.upper()) != {"ME", "MI"}:
                    raise RuntimeError("condition coverage is not exactly ME/MI")
                if {canonical_model(x) for x in df["model"].dropna().unique()} != set(MODELS_ML):
                    raise RuntimeError("model coverage is not exactly the five ML models")
                status = df["status"].astype(str).str.strip().str.upper()
                explicit_failure = status.isin({
                    "FAIL", "FAILED", "ERROR", "INCOMPLETE", "NOT_READY", "MISSING",
                    "ABORTED", "CANCELLED", "CANCELED", "CRASHED", "EXCEPTION"
                })
                if explicit_failure.any():
                    raise RuntimeError(f"{int(explicit_failure.sum())} explicit failure execution rows")
                blank_status = df["status"].isna() | status.eq("")
                if blank_status.any():
                    raise RuntimeError(f"06A execution audit contains {int(blank_status.sum())} blank status values")
            else:
                df = standardize_subject_table(df, require_model=True, require_condition=True)
                if len(df) != 250 or df["subject"].nunique() != 25:
                    raise RuntimeError(f"rows={len(df)}, subjects={df['subject'].nunique()}; expected 250/25")
                if set(df["condition"].astype(str)) != {"ME", "MI"}:
                    raise RuntimeError("condition coverage is not exactly ME/MI")
                expected = set(MODELS_ML if block == "06B_ML_LOSO" else MODELS_DL)
                if {canonical_model(x) for x in df["model"].dropna().unique()} != expected:
                    raise RuntimeError(f"model coverage does not match expected set: {sorted(expected)}")
                validate_unique_subject_observations(df, ["subject", "condition", "model"], block)
        except Exception as exc:
            problems.append(f"{block}: schema validation failed: {exc}")
            rows.append({"item": f"{block}_schema", "path": str(path), "exists": True, "status": "FAIL", "detail": str(exc)})

    # 06A has 250 frozen per-subject class-metric files and 250 matching confusion matrices.
    class_dir = root / "06_ML" / "06A_ML_SD" / "class_metrics"
    conf_dir = root / "06_ML" / "06A_ML_SD" / "confusion_matrices"
    for label, d in (("06A_class_metrics", class_dir), ("06A_confusion_matrices", conf_dir)):
        ok = d.is_dir()
        rows.append({"item": label, "path": str(d), "exists": ok, "status": "PASS" if ok else "FAIL"})
        if not ok:
            problems.append(f"{label}: directory missing: {d}")
    if class_dir.is_dir() and conf_dir.is_dir():
        cps = sorted(class_dir.glob("*_class_metrics.csv"))
        cfs = sorted(conf_dir.glob("*_confusion_matrix.csv"))
        if len(cps) != 250:
            problems.append(f"06A_class_metrics: files={len(cps)}; expected 250")
        if len(cfs) != 250:
            problems.append(f"06A_confusion_matrices: files={len(cfs)}; expected 250")
        conf_names = {x.name for x in cfs}
        for cp in cps:
            expected_name = cp.name.replace("_class_metrics.csv", "_confusion_matrix.csv")
            if expected_name not in conf_names:
                problems.append(f"06A: missing matching confusion matrix for {cp.name}")
                continue
            try:
                cm = read_table(cp)
                required = {"subject", "condition", "model", "f1", "recall", "support"}
                if len(cm) != 7 or not required.issubset(cm.columns):
                    raise RuntimeError(f"{cp.name}: class metrics must be 7 rows with {sorted(required)}")
                conf = read_table(conf_dir / expected_name)
                pred_cols = [c for c in conf.columns if normalized_col(c).startswith("pred_")]
                if len(conf) != 7 or len(pred_cols) != 7:
                    raise RuntimeError(f"{expected_name}: confusion matrix must be 7x7")
            except Exception as exc:
                problems.append(f"06A archive schema: {exc}")

    # Required frozen markers and exact high-value statistical artifacts.
    marker_blocks = ("08_ME_MI_STATISTICS", "09_TEMPORAL_CALIBRATION", "10_ME_MI_TRANSFER", "11_NEURO", "12_REPRESENTATION", "14_ROBUSTNESS")
    for block in marker_blocks:
        raw = cfg["upstream"][block].get("completion_marker")
        path = _abs(root, raw) if raw else root / "None"
        ok, why, _ = verify_completion_marker(path, "14_ROBUSTNESS" if block == "14_ROBUSTNESS" else block)
        rows.append({"item": f"{block}_marker", "path": str(path), "exists": path.is_file(), "status": "PASS" if ok else "FAIL", "detail": why})
        if not ok:
            problems.append(f"{block}: {why}")

    # Every configured artifact must exist before the config can be frozen.
    # A configured directory is valid when it contains at least one entry.
    for block, spec in cfg.get("upstream", {}).items():
        for key, raw in spec.get("artifacts", {}).items():
            if raw in (None, ""):
                problems.append(f"{block}.{key}: artifact path not configured")
                rows.append({"item": f"{block}.{key}", "path": None, "exists": False, "status": "FAIL"})
                continue
            path = _abs(root, raw)
            if path.is_dir():
                try:
                    exists = any(path.iterdir())
                except OSError:
                    exists = False
            else:
                exists = path.is_file() and path.stat().st_size > 0
            rows.append({"item": f"{block}.{key}", "path": str(path), "exists": exists, "status": "PASS" if exists else "FAIL"})
            if not exists:
                problems.append(f"{block}.{key}: configured artifact missing or empty: {path}")

    # BLOCK14 authoritative clean-reference identity audit: exactly 500 PASS rows.
    b14 = root / "14_ROBUSTNESS" / "tables" / "audit" / "BLOCK14_upstream_clean_reference_check.csv"
    if not b14.is_file():
        problems.append(f"BLOCK14 clean-reference audit missing: {b14}")
    else:
        try:
            q = pd.read_csv(b14)
            bad = q[q["status"].astype(str).str.upper() != "PASS"] if "status" in q.columns else q
            if len(q) != 500 or len(bad) != 0:
                raise RuntimeError(f"rows={len(q)}, non-PASS={len(bad)}; expected 500/0")
            rows.append({"item": "BLOCK14_clean_reference_identity", "path": str(b14), "exists": True, "status": "PASS", "rows": len(q)})
        except Exception as exc:
            problems.append(f"BLOCK14 clean-reference audit failed: {exc}")

    return len(problems) == 0, problems, pd.DataFrame(rows)


def load_art(cfg: dict[str, Any], root: Path, block: str, key: str) -> tuple[pd.DataFrame, Path]:
    raw = cfg["upstream"][block]["artifacts"].get(key)
    if raw in (None, ""):
        raise RuntimeError(f"{block}.{key}: artifact path not configured")
    p = _abs(root, raw)
    return read_table(p), p


# =============================================================================
# 4. STATISTICS
# =============================================================================

def spearman_with_permutation(x: Iterable[float], y: Iterable[float], *, seed: int, iterations: int, min_n: int,
                              alternative: str = "two-sided", ci_iterations: int = 0) -> dict[str, Any]:
    """Subject-level Spearman + permutation. Constant / too-small input is NOT tested (never p=1/(B+1))."""
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    base = {"n": int(len(a)), "rho": math.nan, "p_raw": math.nan, "p_asymptotic": math.nan,
            "ci_lower": math.nan, "ci_upper": math.nan}
    if len(a) < max(min_n, 3):
        return {**base, "status": "INSUFFICIENT_DATA", "reason": f"N<{max(min_n, 3)}"}
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return {**base, "status": "NOT_TESTED", "reason": "INSUFFICIENT_VARIATION (constant variable)"}
    ra, rb = stats.rankdata(a), stats.rankdata(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    p_asym = float(stats.spearmanr(a, b).pvalue)
    rng = np.random.default_rng(seed)
    perm_idx = np.argsort(rng.random((iterations, len(rb))), axis=1)
    rbp = rb[perm_idx]
    ra_c = (ra - ra.mean()) / ra.std()
    rbp_c = (rbp - rbp.mean(axis=1, keepdims=True)) / rbp.std(axis=1, keepdims=True)
    null = (rbp_c * ra_c[None, :]).mean(axis=1)
    if alternative == "greater":
        hits = int(np.sum(null >= rho - 1e-12))
    elif alternative == "less":
        hits = int(np.sum(null <= rho + 1e-12))
    else:
        hits = int(np.sum(np.abs(null) >= abs(rho) - 1e-12))
    out = {**base, "rho": rho, "p_asymptotic": p_asym, "p_raw": float((hits + 1) / (iterations + 1)),
           "status": "TESTED", "reason": ""}
    if ci_iterations:
        idx = rng.integers(0, len(a), size=(ci_iterations, len(a)))
        vals = []
        for r in idx:
            if np.unique(a[r]).size > 1 and np.unique(b[r]).size > 1:
                vals.append(np.corrcoef(stats.rankdata(a[r]), stats.rankdata(b[r]))[0, 1])
        if len(vals) > 0.9 * ci_iterations:
            out["ci_lower"], out["ci_upper"] = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    return out


# =============================================================================
# 5. REGISTRY HELPERS
# =============================================================================

def make_reg_row(**kw: Any) -> dict[str, Any]:
    row = {f: (math.nan if f in ("n", "effect_size", "estimate", "ci_lower", "ci_upper", "raw_p", "q",
                                 "upstream_q", "fdr_family_n") else "") for f in REGISTRY_FIELDS}
    row["unit"] = "subject"
    row["ci_status"] = "AVAILABLE"
    row.update(kw)
    return row


def test_status(raw: str) -> str:
    return {"TESTED": "TESTED", "INSUFFICIENT_DATA": "INSUFFICIENT_DATA"}.get(raw, "NOT_TESTED")


def find_first_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    norm = {normalized_col(c): c for c in df.columns}
    for c in candidates:
        if normalized_col(c) in norm:
            return norm[normalized_col(c)]
    return None


def assign_fdr_families(registry: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = registry.copy()
    fams = cfg["fdr"]["families"]
    for i, r in out.iterrows():
        if str(r["fdr_family"]):
            continue
        hits = [n for n, s in fams.items()
                if s["hypothesis"] == r["hypothesis"] and (not s.get("metrics") or r["metric"] in s["metrics"])]
        if len(hits) > 1:
            raise RuntimeError(f"Registry row {r['test_id']} matches several FDR families: {hits}")
        out.at[i, "fdr_family"] = hits[0] if hits else ""
    return out


def apply_fdr_to_registry(registry: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Preserve upstream q; BH-correct (once) only BLOCK15-eligible rows inside explicit configured families."""
    out = assign_fdr_families(registry, cfg)
    for c in ("upstream_q", "q", "fdr_family_n", "raw_p"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["fdr_action"] = "NOT_APPLICABLE"
    up = out["upstream_q"].notna()
    out.loc[up, "q"] = out.loc[up, "upstream_q"]
    out.loc[up, "fdr_action"] = "PRESERVED_UPSTREAM"
    eligible = (~up) & out["raw_p"].notna() & (out["fdr_family"].astype(str) != "")
    for fam, idx in out[eligible].groupby("fdr_family").groups.items():
        idx = list(idx)
        out.loc[idx, "q"] = bh_fdr(out.loc[idx, "raw_p"].to_numpy(dtype=float))
        out.loc[idx, "fdr_family_n"] = len(idx)
        out.loc[idx, "fdr_action"] = "BLOCK15_BH"
    unassigned = (~up) & out["raw_p"].notna() & (out["fdr_family"].astype(str) == "")
    out.loc[unassigned, "fdr_action"] = "UNASSIGNED_FAMILY"
    return out


def evidence_status(row: pd.Series, cfg: dict[str, Any]) -> str:
    st = str(row["status"])
    if st in NON_INFERENTIAL_STATUSES:
        return st
    if st not in INFERENTIAL_STATUSES:
        return "NOT_TESTED"
    q = pd.to_numeric(row["q"], errors="coerce")
    est = pd.to_numeric(row["estimate"], errors="coerce")
    if not np.isfinite(q) or not np.isfinite(est):
        return "NOT_AVAILABLE"
    if q >= cfg["fdr"]["alpha"]:
        return "INCONCLUSIVE"
    sign = cfg["hypotheses"].get(row["hypothesis"], {}).get("expected_sign")
    if sign in (None, 0):
        return "SUPPORTED"
    return "SUPPORTED" if est * sign > 0 else "NOT_SUPPORTED"   # significant but wrong direction is NOT support


def observed_direction(est: float) -> str:
    if not np.isfinite(est):
        return "NA"
    return "positive" if est > 0 else "negative" if est < 0 else "null"


# =============================================================================
# 6. H1 (SD vs LOSO) + GAP ANALYSIS
# =============================================================================

def build_h1_and_gap(sd: pd.DataFrame, loso: pd.DataFrame, cfg: dict[str, Any]
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns h1_table, h1_registry, subject_gap, gap_comparison(D_G), gap_summary, gap_registry."""
    fr = cfg["freeze"]
    seed, boot = int(fr["seed"]), int(fr["bootstrap_iterations"])
    exp_n = int(cfg["missing_data"]["expected_n"])
    need_complete = bool(cfg["missing_data"]["h1_require_complete_cohort"])
    min_n = int(cfg["missing_data"]["min_valid_subjects"])
    merged = sd.merge(loso, on=["subject", "condition", "model"], suffixes=("_sd", "_loso"), how="outer",
                      validate="one_to_one", indicator=True)
    h1_rows, reg, gap_rows = [], [], []
    for model in cfg["inclusion"]["models"]:
        for cond in cfg["inclusion"]["conditions"]:
            sub = merged[(merged["model"] == model) & (merged["condition"] == cond)]
            for metric in cfg["inclusion"]["metrics"]:
                a_c, b_c = f"{metric}_sd", f"{metric}_loso"
                if a_c not in sub.columns or b_c not in sub.columns:
                    continue
                q = sub[["subject", a_c, b_c]].dropna(subset=[a_c, b_c])
                present = sorted(int(s) for s in q["subject"].unique())
                missing = sorted(set(SUBJECTS) - set(present))
                gap = (q[a_c] - q[b_c]).to_numpy(dtype=float)
                for s, g in zip(q["subject"].tolist(), gap.tolist()):
                    gap_rows.append({"subject": int(s), "condition": cond, "model": model, "metric": metric,
                                     "A_SD": float(q.loc[q["subject"] == s, a_c].iloc[0]),
                                     "A_LOSO": float(q.loc[q["subject"] == s, b_c].iloc[0]), "gap": g})
                desc = summarize_vector(gap, seed=seed, iterations=boot)
                if missing and need_complete:
                    status, test = "INSUFFICIENT_DATA", {"effect_size": math.nan, "p_raw": math.nan}
                    reason = f"missing subjects {missing} (observed N={len(present)} of {exp_n}); no imputation"
                else:
                    t = paired_wilcoxon(q[a_c].to_numpy(), q[b_c].to_numpy(), min_n=min_n)
                    status, test, reason = test_status(t["status"]), t, t.get("reason", "")
                est = desc["mean"]
                h1_rows.append({"hypothesis": "H1", "model": model, "condition": cond, "metric": metric,
                                "n": desc["n"], "missing_subjects": ";".join(map(str, missing)),
                                "mean_gap": est, "median_gap": desc["median"], "sd_gap": desc["sd"],
                                "iqr_gap": desc["iqr"], "ci95_low": desc["ci95_low"], "ci95_high": desc["ci95_high"],
                                "prop_positive_gap": float(np.mean(gap > 0)) if len(gap) else math.nan,
                                "effect_size_rank_biserial": test["effect_size"], "p_raw": test["p_raw"],
                                "expected_direction": "SD > LOSO", "observed_direction": ("SD > LOSO" if est > 0 else "SD < LOSO" if est < 0 else "SD = LOSO") if np.isfinite(est) else "NA",
                                "status": status, "reason": reason})
                reg.append(make_reg_row(
                    test_id=f"H1|{model}|{cond}|{metric}", hypothesis="H1", model=model, condition=cond,
                    metric=metric, n=desc["n"], test="Wilcoxon signed-rank (two-sided), subject-level",
                    effect_size=test["effect_size"], estimate=est, ci_lower=desc["ci95_low"],
                    ci_upper=desc["ci95_high"], raw_p=test["p_raw"], direction=observed_direction(est),
                    status=status, source_block="06A+06B+07A+07B", missing_subjects=";".join(map(str, missing)),
                    notes=reason))
    subject_gap = pd.DataFrame(gap_rows)
    # ---- D_G = G_ME - G_MI (subject level) ----
    comp_rows, greg = [], []
    if not subject_gap.empty:
        for model in cfg["inclusion"]["models"]:
            for metric in cfg["inclusion"]["metrics"]:
                g = subject_gap[(subject_gap["model"] == model) & (subject_gap["metric"] == metric)]
                pv = g.pivot_table(index="subject", columns="condition", values="gap", aggfunc="first")
                if not {"ME", "MI"}.issubset(pv.columns):
                    continue
                pv = pv.dropna(subset=["ME", "MI"])
                missing = sorted(set(SUBJECTS) - set(int(s) for s in pv.index))
                d = (pv["ME"] - pv["MI"]).to_numpy(dtype=float)
                desc = summarize_vector(d, seed=seed, iterations=boot)
                if missing and need_complete:
                    status, t, reason = "INSUFFICIENT_DATA", {"effect_size": math.nan, "p_raw": math.nan}, \
                        f"missing subjects {missing}; no imputation"
                else:
                    t = paired_wilcoxon(pv["ME"].to_numpy(), pv["MI"].to_numpy(), min_n=min_n)
                    status, reason = test_status(t["status"]), t.get("reason", "")
                comp_rows.append({"hypothesis": "H1_GAP", "model": model, "metric": metric, "n": desc["n"],
                                  "missing_subjects": ";".join(map(str, missing)), "mean_DG": desc["mean"],
                                  "median_DG": desc["median"], "sd_DG": desc["sd"], "iqr_DG": desc["iqr"],
                                  "ci95_low": desc["ci95_low"], "ci95_high": desc["ci95_high"],
                                  "effect_size_rank_biserial": t["effect_size"], "p_raw": t["p_raw"],
                                  "observed_direction": observed_direction(desc["mean"]), "status": status})
                greg.append(make_reg_row(
                    test_id=f"H1_GAP|{model}|ME-MI|{metric}", hypothesis="H1_GAP", model=model,
                    condition="ME_minus_MI", metric=metric, n=desc["n"],
                    test="Wilcoxon signed-rank (two-sided) on G_ME vs G_MI", effect_size=t["effect_size"],
                    estimate=desc["mean"], ci_lower=desc["ci95_low"], ci_upper=desc["ci95_high"], raw_p=t["p_raw"],
                    direction=observed_direction(desc["mean"]), status=status, source_block="06A+06B+07A+07B",
                    missing_subjects=";".join(map(str, missing)), notes=reason))
    gap_cmp = pd.DataFrame(comp_rows)
    gs = []
    if not subject_gap.empty:
        for (model, cond, metric), g in subject_gap.groupby(["model", "condition", "metric"]):
            d = summarize_vector(g["gap"], seed=seed, iterations=boot)
            gs.append({"model": model, "condition": cond, "metric": metric, **d,
                       "prop_positive_gap": float((g["gap"] > 0).mean())})
    return (pd.DataFrame(h1_rows), pd.DataFrame(reg), subject_gap, gap_cmp, pd.DataFrame(gs), pd.DataFrame(greg))


# =============================================================================
# 7. H2 / H3 / H4 / H5 / k90 (frozen-upstream imports)
# =============================================================================

def build_h2(rsa: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fr = cfg["freeze"]
    seed, boot = int(fr["seed"]), int(fr["bootstrap_iterations"])
    frozen_test = fr.get("h2_inferential_test")
    rows, reg, dist = [], [], []
    for model in MODELS_LATENT:
        q = rsa[rsa["model"] == model].dropna(subset=["rho"])
        present = sorted(int(s) for s in q["subject"].unique())
        missing = sorted(set(SUBJECTS) - set(present))
        d = summarize_vector(q["rho"].to_numpy(dtype=float), seed=seed, iterations=boot)
        for s, r in zip(q["subject"], q["rho"]):
            dist.append({"model": model, "subject": int(s), "rho": float(r)})
        status = "DESCRIPTIVE_ONLY" if frozen_test is None else "NOT_AVAILABLE"
        rows.append({"hypothesis": "H2", "model": model, "n": d["n"], "missing_subjects": ";".join(map(str, missing)),
                     "missing_reason": ("no valid rho in BLOCK12 subject table" if missing else ""),
                     "mean_rho": d["mean"], "median_rho": d["median"], "sd_rho": d["sd"], "iqr_rho": d["iqr"],
                     "ci95_low": d["ci95_low"], "ci95_high": d["ci95_high"], "status": status,
                     "note": "descriptive: BLOCK12 froze no group-level null test"})
        reg.append(make_reg_row(
            test_id=f"H2|{model}|RSA", hypothesis="H2", family="H2_RSA", model=model, condition="ME_vs_MI",
            metric="rho", n=d["n"], test="descriptive (no frozen group-level null)", estimate=d["mean"],
            ci_lower=d["ci95_low"], ci_upper=d["ci95_high"], direction=observed_direction(d["mean"]),
            status=status, source_block="12_REPRESENTATION_ANALYSIS", missing_subjects=";".join(map(str, missing)),
            notes="CI = BLOCK15 subject-bootstrap of mean rho (descriptive)"))
    return pd.DataFrame(rows), pd.DataFrame(reg), pd.DataFrame(dist)


def build_h3_from_frozen_block10(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Import BLOCK10 asymmetry inference (frozen paired sign-swap permutation, bootstrap CI, BH-FDR). No re-test."""
    need = {"model", "metric", "n_subjects", "asymmetry_mean", "asymmetry_median", "asymmetry_ci_lower",
            "asymmetry_ci_upper", "asymmetry_p", "asymmetry_p_adjusted"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"BLOCK10 asymmetry_test.csv missing columns {sorted(miss)}; found {list(df.columns)}")
    q = normalize_model_column(df)
    q = q[q["metric"].astype(str).map(normalized_col) == PRIMARY_METRIC].copy()
    if q["model"].duplicated().any():
        raise RuntimeError("BLOCK10 asymmetry: duplicate model rows for Macro-F1")
    if set(q["model"]) != set(ALL_MODELS):
        raise RuntimeError(f"BLOCK10 asymmetry model coverage {sorted(set(q['model']))} != frozen 10-model set")
    table = q.copy()
    table["hypothesis"], table["source"], table["status"] = "H3", "10_ME_MI_TRANSFER", "UPSTREAM_FROZEN"
    reg = pd.DataFrame([make_reg_row(
        test_id=f"H3|{r.model}|ME-to-MI_minus_MI-to-ME|{PRIMARY_METRIC}", hypothesis="H3",
        family="H3_TRANSFER_ASYMMETRY", model=r.model, condition="ME_to_MI_minus_MI_to_ME", metric=PRIMARY_METRIC,
        n=int(r.n_subjects), test="BLOCK10 frozen paired sign-swap permutation (10,000), bootstrap CI",
        estimate=float(r.asymmetry_mean), ci_lower=float(r.asymmetry_ci_lower), ci_upper=float(r.asymmetry_ci_upper),
        raw_p=float(r.asymmetry_p), upstream_q=float(r.asymmetry_p_adjusted),
        direction=observed_direction(float(r.asymmetry_mean)), status="UPSTREAM_FROZEN",
        source_block="10_ME_MI_TRANSFER") for r in q.itertuples(index=False)])
    return table, reg


Q_COLS = ("q", "q_value", "fdr_q", "p_fdr", "adjusted_p", "permutation_q", "fdr_adjusted_p", "permutation_p_adjusted")
CI_LO_COLS = ("ci_lower", "ci95_low", "rho_ci_lower", "ci_low", "spearman_ci_lower")
CI_HI_COLS = ("ci_upper", "ci95_high", "rho_ci_upper", "ci_high", "spearman_ci_upper")


def import_block12_association(df: pd.DataFrame, hyp: str, label: str, family: str
                               ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Import frozen BLOCK12 H4/H5 results; preserve p/q/CI exactly; document anything BLOCK12 does not provide."""
    need = {"model", "n_subjects", "rho", "permutation_p"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"BLOCK12 {hyp} summary missing columns {sorted(miss)}; found {list(df.columns)}")
    q = normalize_model_column(df)
    q = q[q["model"].isin(MODELS_LATENT)].copy()
    if len(q) != len(MODELS_LATENT) or q["model"].duplicated().any():
        raise RuntimeError(f"BLOCK12 {hyp}: expected exactly one row per latent model, found {len(q)}")
    qc, lo, hi = find_first_col(q, Q_COLS), find_first_col(q, CI_LO_COLS), find_first_col(q, CI_HI_COLS)
    limits: list[str] = []
    if qc is None:
        limits.append(f"{hyp}: BLOCK12 summary has no q-value column; BLOCK15 applies ONE BH within {family}")
    if lo is None or hi is None:
        limits.append(f"{hyp}: BLOCK12 summary provides no CI; none fabricated (inspect for a frozen bootstrap distribution)")
    table = q.copy()
    table["hypothesis"], table["target"], table["source"] = hyp, label, "12_REPRESENTATION"
    table["status"] = "UPSTREAM_FROZEN"
    rows = []
    for r in q.to_dict("records"):
        has_ci = lo is not None and hi is not None and np.isfinite(pd.to_numeric(r[lo], errors="coerce")) \
            and np.isfinite(pd.to_numeric(r[hi], errors="coerce"))
        rho = pd.to_numeric(r["rho"], errors="coerce")
        p = pd.to_numeric(r["permutation_p"], errors="coerce")
        stt = "UPSTREAM_FROZEN" if np.isfinite(p) and np.isfinite(rho) else "NOT_AVAILABLE"
        rows.append(make_reg_row(
            test_id=f"{hyp}|{r['model']}|{label}", hypothesis=hyp, family=family, model=r["model"],
            condition="ME_vs_MI", metric=f"rho_vs_{label}", n=int(r["n_subjects"]),
            test="BLOCK12 frozen Spearman + subject-level permutation", effect_size=float(rho) if np.isfinite(rho) else math.nan,
            estimate=float(rho) if np.isfinite(rho) else math.nan,
            ci_lower=float(r[lo]) if has_ci else math.nan, ci_upper=float(r[hi]) if has_ci else math.nan,
            ci_status="AVAILABLE" if has_ci else "NOT_AVAILABLE", raw_p=float(p) if np.isfinite(p) else math.nan,
            upstream_q=float(r[qc]) if qc is not None and np.isfinite(pd.to_numeric(r[qc], errors="coerce")) else math.nan,
            direction=observed_direction(float(rho)) if np.isfinite(rho) else "NA", status=stt,
            source_block="12_REPRESENTATION_ANALYSIS", notes="" if has_ci else "CI not provided by BLOCK12"))
    return table, pd.DataFrame(rows), limits


def extract_k90_exact(df: pd.DataFrame, context: str) -> pd.DataFrame:
    need = {"subject_id", "condition", "model", "metric", "k90", "k90_status"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"{context}: BLOCK09 k90 schema missing {sorted(miss)}; found {list(df.columns)}")
    q = normalize_model_column(df)
    q["subject_id"] = q["subject_id"].map(canonical_subject)
    q["condition"] = q["condition"].map(canonical_condition)
    q = q[q["metric"].astype(str).map(normalized_col) == PRIMARY_METRIC].copy()
    q["k90"] = pd.to_numeric(q["k90"], errors="coerce")
    if q.duplicated(["subject_id", "condition", "model"]).any():
        raise RuntimeError(f"{context}: duplicate subject/condition/model k90 keys")
    return q


def summarize_k90(k: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    seed, boot = int(cfg["freeze"]["seed"]), int(cfg["freeze"]["bootstrap_iterations"])
    for (model, cond), g in k.groupby(["model", "condition"]):
        st = g["k90_status"].astype(str)
        valid = g[g["k90"].notna() & ~st.str.upper().isin(["NOT_REACHED", "NO_POSITIVE_GAP"])]
        d = summarize_vector(valid["k90"].to_numpy(dtype=float), seed=seed, iterations=boot)
        rows.append({"model": model, "condition": cond, "n_rows": len(g),
                     "n_subjects": int(g["subject_id"].nunique()),
                     "missing_subjects": ";".join(str(s) for s in sorted(set(SUBJECTS) - set(g["subject_id"]))),
                     "n_valid_k90": len(valid), "n_NOT_REACHED": int((st.str.upper() == "NOT_REACHED").sum()),
                     "n_NO_POSITIVE_GAP": int((st.str.upper() == "NO_POSITIVE_GAP").sum()),
                     "status_counts": json.dumps(st.value_counts().to_dict()),
                     "k90_mean_valid_only": d["mean"], "k90_median_valid_only": d["median"],
                     "k90_ci95_low": d["ci95_low"], "k90_ci95_high": d["ci95_high"],
                     "note": "k90 statistics use valid values only; failures are counted, never imputed"})
    return pd.DataFrame(rows)



def build_transfer_integration(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    need={"subject_id","model","transfer_direction","metric","transfer_benefit"}
    miss=need-set(df.columns)
    if miss: raise RuntimeError(f"BLOCK10 transfer_benefit missing columns {sorted(miss)}")
    q=df.copy(); q["subject"]=q["subject_id"].map(canonical_subject); q["model"]=q["model"].map(canonical_model)
    q["metric_norm"]=q["metric"].astype(str).map(normalized_col); q=q[q["metric_norm"]==PRIMARY_METRIC].copy()
    if len(q) != N_EXPECTED_SUBJECTS*len(ALL_MODELS)*2: raise RuntimeError(f"BLOCK10 transfer_benefit Macro-F1 rows={len(q)}; expected 500")
    if q.duplicated(["subject","model","transfer_direction"]).any(): raise RuntimeError("BLOCK10 transfer_benefit duplicate subject/model/direction")
    q["transfer_benefit"]=coerce_numeric(q["transfer_benefit"],"transfer_benefit")
    summary=[]; neg=[]
    for (m,d),g in q.groupby(["model","transfer_direction"]):
        v=g["transfer_benefit"].to_numpy(float); ds=summarize_vector(v,seed=int(cfg["freeze"]["seed"]),iterations=int(cfg["freeze"]["bootstrap_iterations"]))
        summary.append({"model":m,"transfer_direction":d,"metric":PRIMARY_METRIC,**ds,"negative_n":int((v<0).sum()),"negative_percent":float((v<0).mean()*100)})
    for m,g in q.groupby("model"):
        neg.append({"model":m,"metric":PRIMARY_METRIC,"n_subjects":int(g["subject"].nunique()),"negative_any_direction_n":int(g.groupby("subject")["transfer_benefit"].apply(lambda x:(x<0).any()).sum()),"negative_any_direction_percent":float(g.groupby("subject")["transfer_benefit"].apply(lambda x:(x<0).any()).mean()*100)})
    return q, pd.DataFrame(summary), pd.DataFrame(neg)


def build_calibration_integration(k90: pd.DataFrame, a_sd: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Integrate the frozen BLOCK09 k90 and A_SD artifacts without inventing data.

    The inferential unit remains the subject.  ME--MI paired calibration summaries
    are produced only when both conditions are actually present for a model.
    Missing conditions are reported as unavailable rather than imputed.
    """
    k = extract_k90_exact(k90, "BLOCK09")

    a = a_sd.copy()
    required_a = {"subject_id", "condition", "model", "metric", "A_SD_primary"}
    missing_a = required_a - set(a.columns)
    if missing_a:
        raise RuntimeError(
            "BLOCK09 A_SD primary artifact missing required columns: "
            f"{sorted(missing_a)}; found {list(a.columns)}"
        )

    a["subject_id"] = a["subject_id"].map(canonical_subject)
    a["model"] = a["model"].map(canonical_model)
    a["condition"] = a["condition"].map(canonical_condition)
    a["metric_norm"] = a["metric"].astype(str).map(normalized_col)
    a = a[a["metric_norm"] == PRIMARY_METRIC].copy()
    a["A_SD_primary"] = coerce_numeric(a["A_SD_primary"], "A_SD_primary")

    if a.duplicated(["subject_id", "condition", "model"]).any():
        dup = a[a.duplicated(["subject_id", "condition", "model"], keep=False)]
        raise RuntimeError(
            "BLOCK09 A_SD primary artifact contains duplicate "
            "subject/condition/model observations; "
            f"rows={len(dup)}"
        )

    merged = k.merge(
        a[["subject_id", "condition", "model", "A_SD_primary"]],
        on=["subject_id", "condition", "model"],
        how="left",
        validate="one_to_one",
    )

    rows = []
    for (m, c), g in merged.groupby(["model", "condition"], dropna=False):
        status = g["k90_status"].astype(str).str.strip().str.upper()
        valid = g[
            g["k90"].notna()
            & ~status.isin(["NOT_REACHED", "NO_POSITIVE_GAP"])
        ]
        rows.append({
            "model": m,
            "condition": c,
            "metric": PRIMARY_METRIC,
            "n_subjects": int(g["subject_id"].nunique()),
            "n_valid_k90": int(len(valid)),
            "mean_k90": float(valid["k90"].mean()) if len(valid) else np.nan,
            "median_k90": float(valid["k90"].median()) if len(valid) else np.nan,
            "mean_A_SD_primary": float(g["A_SD_primary"].mean()) if g["A_SD_primary"].notna().any() else np.nan,
            "status": "K90_IMPORTED",
        })

    eff = []
    for m, g in merged.groupby("model", dropna=False):
        p = g.pivot_table(
            index="subject_id",
            columns="condition",
            values="k90",
            aggfunc="first",
        )

        available = {str(c).strip().upper() for c in p.columns}
        has_me = "ME" in available
        has_mi = "MI" in available

        # Re-map columns explicitly so the following calculation is deterministic.
        rename_cols = {}
        for c in p.columns:
            c_norm = str(c).strip().upper()
            if c_norm in {"ME", "MI"}:
                rename_cols[c] = c_norm
        p = p.rename(columns=rename_cols)

        if has_me and has_mi:
            paired = p[["ME", "MI"]].dropna(subset=["ME", "MI"], how="any")
            n_paired = int(len(paired))
            mean_difference = (
                float((paired["ME"] - paired["MI"]).mean())
                if n_paired else np.nan
            )
            paired_status = "PAIRED_ME_MI" if n_paired else "INSUFFICIENT_DATA"
        else:
            n_paired = 0
            mean_difference = np.nan
            if not has_me and not has_mi:
                paired_status = "ME_MI_NOT_AVAILABLE"
            elif not has_me:
                paired_status = "ME_NOT_AVAILABLE"
            else:
                paired_status = "MI_NOT_AVAILABLE"

        eff.append({
            "model": m,
            "n_subjects": int(p.index.nunique()),
            "n_paired_subjects": n_paired,
            "mean_ME_minus_MI_k90": mean_difference,
            "calibration_grid": json.dumps(cfg["freeze"]["calibration_fractions"]),
            "status": paired_status,
        })

    return pd.DataFrame(rows), pd.DataFrame(eff)


def build_robustness_integration(dropout: pd.DataFrame, noise: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    def prep(d):
        q=d.copy(); q["subject"]=q["subject"].map(canonical_subject); q["model"]=q["model"].map(canonical_model)
        q=q[q["metric"].astype(str).map(normalized_col)==PRIMARY_METRIC].copy(); q["delta_metric"]=coerce_numeric(q["delta_metric"],"delta_metric"); return q
    d=prep(dropout); n=prep(noise)
    ds=d.groupby(["model","condition","dropout_percent"],as_index=False).agg(n_subjects=("subject","nunique"),mean_delta=("delta_metric","mean"),median_delta=("delta_metric","median"))
    ns=n.groupby(["model","condition","perturbation_type","perturbation_level"],as_index=False).agg(n_subjects=("subject","nunique"),mean_delta=("delta_metric","mean"),median_delta=("delta_metric","median"))
    summary=pd.concat([ds.assign(perturbation="channel_dropout",level=ds["dropout_percent"]),ns.assign(perturbation="noise",level=ns["perturbation_level"])] ,ignore_index=True,sort=False)
    return summary,ds,ns

def profile_table(df: pd.DataFrame, block: str, key: str, path: Path) -> dict[str, Any]:
    sc, mc, cc = find_column(df, "subject"), find_column(df, "model"), find_column(df, "condition")
    subj = sorted(set(df[sc].map(canonical_subject))) if sc else []
    return {"block": block, "artifact": key, "path": str(path), "rows": len(df), "columns": ";".join(map(str, df.columns)),
            "n_subjects": len(subj), "missing_subjects": ";".join(str(s) for s in sorted(set(SUBJECTS) - set(subj))) if sc else "NA",
            "n_models": int(df[mc].nunique()) if mc else 0, "n_conditions": int(df[cc].nunique()) if cc else 0,
            "duplicate_rows": int(df.duplicated().sum()), "null_cells": int(df.isna().sum().sum())}


# =============================================================================
# 8. DESCRIPTIVE / H6
# =============================================================================

def build_descriptive(sd: pd.DataFrame, loso: pd.DataFrame, cfg: dict[str, Any]):
    seed, boot = int(cfg["freeze"]["seed"]), int(cfg["freeze"]["bootstrap_iterations"])
    long = pd.concat([sd.assign(branch="SD"), loso.assign(branch="LOSO")], ignore_index=True)
    id_cols = ["subject", "condition", "model", "branch"]
    metrics = [m for m in ALL_METRICS if m in long.columns]
    subj_level = long.melt(id_vars=id_cols, value_vars=metrics, var_name="metric", value_name="value").dropna(subset=["value"])
    agg, h6, reg = [], [], []
    for (branch, model, cond, metric), g in subj_level.groupby(["branch", "model", "condition", "metric"]):
        d = summarize_vector(g["value"].to_numpy(dtype=float), seed=seed, iterations=boot)
        q1, q3 = np.percentile(g["value"], [25, 75])
        row = {"branch": branch, "model": model, "condition": cond, "metric": metric, "family": "ML" if model in MODELS_ML else "DL",
               "chance_level": CHANCE_LEVEL if metric in ("accuracy", PRIMARY_METRIC) else math.nan, **d, "q1": q1, "q3": q3}
        agg.append(row)
        if metric in ALL_METRICS:
            h6.append({"hypothesis": "H6", **row,
                       "interpretation": "Descriptive ML/DL characterisation; no composite score, no winner."})
            reg.append(make_reg_row(
                test_id=f"H6|{branch}|{model}|{cond}|{metric}", hypothesis="H6", family="H6_DESCRIPTIVE", model=model,
                condition=cond, metric=metric, n=d["n"], test="descriptive", estimate=d["mean"],
                ci_lower=d["ci95_low"], ci_upper=d["ci95_high"], direction="NA", status="DESCRIPTIVE_ONLY",
                source_block="06+07", notes=branch))
    agg_df = pd.DataFrame(agg)
    return subj_level, agg_df, pd.DataFrame(h6), pd.DataFrame(reg)


# =============================================================================
# 9. MATRIX / REGISTRY AUDIT / ENVIRONMENT
# =============================================================================

def build_hypothesis_matrix(reg: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for r in reg.to_dict("records"):
        h = r["hypothesis"]
        sign = cfg["hypotheses"].get(h, {}).get("expected_sign")
        exp = {1: "positive", -1: "negative", 0: "either (non-directional)"}.get(sign, "none")
        rows.append({"hypothesis": h, "hypothesis_description": HYPOTHESIS_TEXT.get(h, ""),
                     "analysis_family": r["family"] or r["fdr_family"], "model": r["model"], "condition": r["condition"],
                     "metric": r["metric"], "N": r["n"], "estimate": r["estimate"], "effect_size": r["effect_size"],
                     "CI_lower": r["ci_lower"], "CI_upper": r["ci_upper"], "raw_p": r["raw_p"], "q": r["q"],
                     "direction": r["direction"], "expected_direction": exp, "status": r["status"],
                     "evidence_status": r["evidence_status"], "source_block": r["source_block"],
                     "interpretation": INTERPRETATION.get(h, "")})
    m = pd.DataFrame(rows, columns=list(MATRIX_FIELDS))
    roll = []
    for h in HYPOTHESIS_TEXT:
        g = m[m["hypothesis"] == h]
        counts = g["evidence_status"].value_counts().to_dict() if len(g) else {}
        roll.append({"hypothesis": h, "hypothesis_description": HYPOTHESIS_TEXT[h], "n_tests": len(g),
                     **{f"n_{s}": int(counts.get(s, 0)) for s in STATUS_VOCAB},
                     "hypothesis_level_status": "NOT_TESTED" if len(g) == 0 else (
                         "DESCRIPTIVE_ONLY" if set(g["evidence_status"]) <= {"DESCRIPTIVE_ONLY"} else "SEE_COUNTS"),
                     "note": "No composite verdict: counts of per-test evidence statuses are reported."})
    return m, pd.DataFrame(roll)


def audit_registry(reg: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for r in reg.to_dict("records"):
        miss = []
        for f in ("test_id", "hypothesis", "model", "metric", "unit", "test", "status", "source_block"):
            if r[f] in ("", None) or (isinstance(r[f], float) and not np.isfinite(r[f])):
                miss.append(f)
        st = r["status"]
        if st not in STATUS_VOCAB:
            miss.append("status_not_in_vocabulary")
        if r["evidence_status"] not in STATUS_VOCAB:
            miss.append("evidence_status_not_in_vocabulary")
        if st in INFERENTIAL_STATUSES + ("DESCRIPTIVE_ONLY",):
            for f in ("n", "estimate"):
                if not np.isfinite(pd.to_numeric(r[f], errors="coerce")):
                    miss.append(f)
            if r["ci_status"] != "NOT_AVAILABLE" and not (np.isfinite(pd.to_numeric(r["ci_lower"], errors="coerce"))
                                                          and np.isfinite(pd.to_numeric(r["ci_upper"], errors="coerce"))):
                miss.append("ci")
        if st in INFERENTIAL_STATUSES:
            if not np.isfinite(pd.to_numeric(r["raw_p"], errors="coerce")):
                miss.append("raw_p")
            if not np.isfinite(pd.to_numeric(r["q"], errors="coerce")):
                miss.append("q")
            if r["fdr_action"] == "UNASSIGNED_FAMILY":
                miss.append("fdr_family_unassigned")
        rows.append({"test_id": r["test_id"], "hypothesis": r["hypothesis"], "status": st,
                     "ci_status": r["ci_status"], "missing_or_invalid_fields": ";".join(miss), "ok": not miss})
    return pd.DataFrame(rows)


def _pkg_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def environment_snapshot() -> dict[str, Any]:
    import importlib.metadata  # noqa: F401  (ensures submodule is loaded)
    git = None
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=str(Path(__file__).resolve().parent)).stdout.strip() or None
    except Exception:
        pass
    return {"timestamp_utc": utc_now(), "python_version": sys.version, "numpy_version": np.__version__,
            "pandas_version": pd.__version__, "scipy_version": scipy.__version__,
            "sklearn_version": _pkg_version("scikit-learn"), "statsmodels_version": _pkg_version("statsmodels"),
            "torch_version": _pkg_version("torch"), "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "git_commit": git, "script_sha256": sha256_file(Path(__file__).resolve())}


# =============================================================================
# 10. INSPECTION (read-only dependency / schema report)
# =============================================================================

_SKIP_DIRS = {"folds", "units", "checkpoints", "predictions", "__pycache__", "archive", "raw_windows", "windows"}


def inspect_project(root: Path) -> dict[str, Any]:
    rep: dict[str, Any] = {"root": str(root), "timestamp_utc": utc_now(), "top_level": sorted(p.name for p in root.iterdir())
                           if root.is_dir() else [], "blocks": {}}
    for block, (bdir, marker) in UPSTREAM_LAYOUT.items():
        b = _abs(root, bdir)
        info: dict[str, Any] = {"candidate_dir": str(b), "dir_exists": b.is_dir(), "candidate_marker": str(_abs(root, marker)),
                                "marker_exists": _abs(root, marker).is_file(), "markers_found": [], "tables": []}
        if b.is_dir():
            n = 0
            for dp, dn, fn in os.walk(b):
                dn[:] = [d for d in dn if d.lower() not in _SKIP_DIRS]
                depth = len(Path(dp).relative_to(b).parts)
                if depth > 4:
                    dn[:] = []
                for f in sorted(fn):
                    p = Path(dp) / f
                    if "complete" in f.lower() and f.lower().endswith(".json"):
                        try:
                            o = json.loads(p.read_text(encoding="utf-8"))
                            info["markers_found"].append({"path": str(p), "keys": list(o)[:40], "status": o.get("status"),
                                                          "completed": o.get("completed")})
                        except Exception as exc:
                            info["markers_found"].append({"path": str(p), "error": str(exc)})
                    if f.lower().endswith(".csv") and n < 300:
                        n += 1
                        t: dict[str, Any] = {"path": str(p.relative_to(root)), "bytes": p.stat().st_size}
                        try:
                            head = pd.read_csv(p, nrows=3)
                            t["columns"] = list(head.columns)
                            if p.stat().st_size < 25_000_000:
                                d = pd.read_csv(p)
                                t["rows"] = len(d)
                                for logical in ("subject", "condition", "model", "metric"):
                                    c = find_column(d, logical)
                                    if c:
                                        u = d[c].dropna().unique()
                                        t[f"unique_{logical}"] = len(u)
                                        if len(u) <= 30:
                                            t[f"values_{logical}"] = [str(x) for x in sorted(u, key=str)]
                                for c in d.columns:
                                    if "k90_status" in normalized_col(c) or normalized_col(c) in ("fraction", "calibration_fraction", "k"):
                                        u = d[c].dropna().unique()
                                        if len(u) <= 30:
                                            t[f"values_{c}"] = [str(x) for x in sorted(u, key=str)]
                        except Exception as exc:
                            t["error"] = f"{type(exc).__name__}: {exc}"
                        info["tables"].append(t)
        rep["blocks"][block] = info
    return rep


def write_inspection(root: Path) -> Path:
    rep = inspect_project(root)
    d = root / BLOCK_NAME / "audits"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_json(rep, d / "BLOCK15_inspection.json")
    md = ["# BLOCK15 dependency inspection", f"root: `{root}`", f"top level: {rep['top_level']}", ""]
    for b, i in rep["blocks"].items():
        md += [f"## {b}", f"- dir exists: {i['dir_exists']} (`{i['candidate_dir']}`)",
               f"- candidate marker exists: {i['marker_exists']}",
               f"- markers found: {[m.get('path') for m in i['markers_found']]}"]
        for t in i["tables"]:
            md.append(f"  - `{t['path']}` rows={t.get('rows','?')} cols={t.get('columns')}")
        md.append("")
    (d / "BLOCK15_inspection.md").write_text("\n".join(md), encoding="utf-8")
    return d / "BLOCK15_inspection.json"


# =============================================================================
# 11. COMPLETION GATE + MAIN
# =============================================================================

def completion_gate(out: Path, cfg: dict[str, Any], *, upstream_problems: list[str], analysis_errors: list[str],
                    registry: pd.DataFrame, reg_audit: pd.DataFrame, unimplemented: list[str]) -> list[str]:
    reasons: list[str] = []
    reasons += [f"upstream: {p}" for p in upstream_problems]
    reasons += [f"analysis error: {e}" for e in analysis_errors]
    reasons += [f"required analysis not produced: {u}" for u in unimplemented]
    for rel in REQUIRED_OUTPUTS:
        p = out / rel
        if not p.is_file() or p.stat().st_size == 0:
            reasons.append(f"required output missing/empty: {rel}")
    if registry.empty:
        reasons.append("statistical registry is empty")
    else:
        bad = reg_audit[~reg_audit["ok"]]
        if len(bad):
            reasons.append(f"{len(bad)} registry record(s) incomplete, e.g. {bad.head(3)[['test_id','missing_or_invalid_fields']].values.tolist()}")
        if registry.duplicated(["test_id"]).any():
            reasons.append("duplicate test_id in statistical registry")
        req_h = {"H1", "H2", "H3", "H4", "H5", "H6"}
        if not req_h.issubset(set(registry["hypothesis"])):
            reasons.append(f"hypotheses without registered tests: {sorted(req_h - set(registry['hypothesis']))}")
        insuff = registry[registry["status"] == "INSUFFICIENT_DATA"]
        if len(insuff):
            reasons.append(f"{len(insuff)} registered test(s) INSUFFICIENT_DATA (incomplete 25-subject cohort)")
    return reasons


def run_block(root: Path, *, validate_only: bool = False, force: bool = False) -> int:
    out = root / BLOCK_NAME
    for d in OUTPUT_DIRS:
        (out / d).mkdir(parents=True, exist_ok=True)
    log = Logger(out / "BLOCK15_run.log")
    start = time.perf_counter()
    log.info("=" * 80)
    log.info("BLOCK15 -- FINAL STATISTICAL INTEGRATION")
    log.info(f"root={root}")
    cfg_path = config_path_for(root)
    cfg = load_config(cfg_path)

    def finish(status: str, reasons: list[str], extra: Optional[dict] = None) -> int:
        payload = {"block": BLOCK_ID, "status": status, "reasons": reasons, "timestamp_utc": utc_now(),
                   "config_sha256_current": config_canonical_hash(cfg),
                   "config_sha256_frozen": cfg["freeze"].get("expected_config_sha256"), **(extra or {})}
        atomic_write_json(payload, out / "completion" / f"BLOCK15_{status}.json")
        atomic_write_json(payload, out / "BLOCK15_summary.json")
        log.info(f"BLOCK15 {status}: {len(reasons)} reason(s)")
        for r in reasons[:25]:
            log.info(f"  - {r}")
        return 0 if status == "PASS" else (2 if status == "NOT_READY" else 1)

    # --- freeze gate: FAIL CLOSED on drift ---
    fp = freeze_problems(cfg)
    drift = [p for p in fp if p.startswith("CONFIG DRIFT")]
    if drift:
        return finish("FAIL", drift)
    if fp:
        return finish("NOT_READY", [f"freeze gate: {p}" for p in fp])

    evidence, up_problems, up_audit = verify_upstream(cfg, root, log)
    atomic_write_csv(up_audit, out / "audits" / "upstream_completion_audit.csv")
    if up_problems:
        return finish("NOT_READY", [f"upstream: {p}" for p in up_problems])
    if validate_only:
        return finish("READY", [], {"validation_only": True, "statistics_executed": False})
    if (out / "BLOCK15_complete.json").exists() and not force:
        log.info("BLOCK15_complete.json exists; use --force for a reproducible rerun with the SAME frozen config")
        return 0

    errors: list[str] = []
    limits: list[str] = []
    tabs: dict[str, pd.DataFrame] = {}
    regs: list[pd.DataFrame] = []
    prof: list[dict[str, Any]] = []

    def step(name: str, fn):
        try:
            fn()
            log.info(f"step OK: {name}")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            log.error(f"step FAIL: {name}: {exc}")

    sd_all: dict[str, pd.DataFrame] = {}

    def s_h1():
        parts = {}

        # 06A ML SD has no frozen aggregate subject table. Derive subject-level
        # metrics only from its frozen per-subject class metrics and confusion
        # matrices; no window-level aggregation and no BLOCK08 substitution.
        sd_ml_dir = root / "06_ML" / "06A_ML_SD"
        class_dir = sd_ml_dir / "class_metrics"
        conf_dir = sd_ml_dir / "confusion_matrices"
        if not class_dir.is_dir() or not conf_dir.is_dir():
            raise RuntimeError(f"06A frozen metric archives missing: {class_dir} / {conf_dir}")

        rows = []
        for cp in sorted(class_dir.glob("*_class_metrics.csv")):
            cm = read_table(cp)
            required = {"subject", "condition", "model", "f1", "recall", "support"}
            if not required.issubset(cm.columns):
                continue
            cm["subject"] = cm["subject"].map(canonical_subject)
            cm["condition"] = cm["condition"].map(canonical_condition)
            cm["model"] = cm["model"].map(canonical_model)
            cm["f1"] = coerce_numeric(cm["f1"], f"06A {cp.name} f1")
            cm["recall"] = coerce_numeric(cm["recall"], f"06A {cp.name} recall")
            cm["support"] = coerce_numeric(cm["support"], f"06A {cp.name} support")
            if len(cm) != 7 or cm["subject"].nunique() != 1 or cm["condition"].nunique() != 1 or cm["model"].nunique() != 1:
                raise RuntimeError(f"06A class-metrics schema/row-count invalid: {cp}")

            subject = int(cm["subject"].iloc[0])
            cond = cm["condition"].iloc[0]
            model = cm["model"].iloc[0]
            support_total = float(cm["support"].sum())
            if support_total <= 0:
                raise RuntimeError(f"06A {cp}: non-positive total support")

            balanced_accuracy = float(cm["recall"].mean())
            macro_f1 = float(cm["f1"].mean())
            accuracy = float((cm["recall"] * cm["support"]).sum() / support_total)

            conf_path = conf_dir / cp.name.replace("_class_metrics.csv", "_confusion_matrix.csv")
            if not conf_path.is_file():
                raise RuntimeError(f"06A confusion matrix missing for {cp.name}: {conf_path}")
            conf = read_table(conf_path)
            pred_cols = [c for c in conf.columns if normalized_col(c).startswith("pred_")]
            if len(conf) != 7 or len(pred_cols) != 7:
                raise RuntimeError(f"06A confusion matrix schema invalid: {conf_path}")
            matrix = conf[pred_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(matrix).all() or np.any(matrix < 0):
                raise RuntimeError(f"06A confusion matrix contains invalid values: {conf_path}")

            tp = float(np.trace(matrix))
            total = float(matrix.sum())
            row_sum = matrix.sum(axis=1)
            col_sum = matrix.sum(axis=0)
            denom = math.sqrt(max(0.0, (total**2 - float(np.dot(col_sum, col_sum))) *
                                  (total**2 - float(np.dot(row_sum, row_sum)))))
            mcc = float((tp * total - float(np.dot(row_sum, col_sum))) / denom) if denom > 0 else 0.0
            po = float(tp / total) if total > 0 else math.nan
            pe = float(np.dot(row_sum, col_sum) / (total**2)) if total > 0 else math.nan
            kappa = float((po - pe) / (1.0 - pe)) if np.isfinite(pe) and abs(1.0 - pe) > 1e-15 else 0.0

            rows.append({
                "subject": subject, "condition": cond, "model": model,
                "accuracy": accuracy, "balanced_accuracy": balanced_accuracy,
                "macro_f1": macro_f1, "mcc": mcc, "kappa": kappa,
                "evaluation_regime": "SD",
            })

        sd_ml = pd.DataFrame(rows)
        if len(sd_ml) != 25 * 2 * 5:
            raise RuntimeError(f"06A_ML_SD derived subject metrics rows={len(sd_ml)}; expected 250")
        validate_unique_subject_observations(sd_ml, ["subject", "condition", "model"], "06A_ML_SD")
        parts["06A_ML_SD"] = prepare_metric_long(sd_ml, "06A_ML_SD")
        prof.append(profile_table(sd_ml, "06A_ML_SD", "derived_frozen_class_and_confusion_metrics", class_dir))

        for blk in ("06B_ML_LOSO", "07A_DL_SD", "07B_DL_LOSO"):
            df, p = load_art(cfg, root, blk, "subject_metrics")
            parts[blk] = prepare_metric_long(df, blk)
            prof.append(profile_table(df, blk, "subject_metrics", p))
        sd = pd.concat([parts["06A_ML_SD"], parts["07A_DL_SD"]], ignore_index=True)
        lo = pd.concat([parts["06B_ML_LOSO"], parts["07B_DL_LOSO"]], ignore_index=True)
        for nm, d in (("SD", sd), ("LOSO", lo)):
            if d.duplicated(["subject", "condition", "model"]).any():
                raise RuntimeError(f"{nm}: duplicate subject/condition/model observations after ML+DL concat")
            unexpected = sorted(set(d["model"]) - set(ALL_MODELS))
            if unexpected:
                raise RuntimeError(f"{nm}: unexpected models {unexpected}")
        sd_all["sd"], sd_all["lo"] = sd, lo
        h1, r1, sg, gc, gsum, gr = build_h1_and_gap(sd, lo, cfg)
        tabs.update({"h1": h1, "subject_gap": sg, "gap_cmp": gc, "gap_summary": gsum})
        regs.extend([r1, gr])
        subj, agg, h6, r6 = build_descriptive(sd, lo, cfg)
        tabs.update({"subj_level": subj, "agg": agg, "h6": h6})
        regs.append(r6)

    def s_h2():
        df, p = load_art(cfg, root, "12_REPRESENTATION", "subject_rsa")
        prof.append(profile_table(df, "12_REPRESENTATION", "subject_rsa", p))
        rsa = extract_rsa(normalize_model_column(df) if find_column(df, "model") else df, "12_REPRESENTATION")
        rsa = normalize_model_column(rsa)
        t, r, dist = build_h2(rsa, cfg)
        tabs.update({"h2": t, "rsa_dist": dist})
        regs.append(r)

    def s_h3():
        df, p = load_art(cfg, root, "10_ME_MI_TRANSFER", "asymmetry_test")
        prof.append(profile_table(df, "10_ME_MI_TRANSFER", "asymmetry_test", p))
        t, r = build_h3_from_frozen_block10(df)
        tabs["h3"] = t
        regs.append(r)

    def s_h45():
        for key, hyp, label, fam, tk in (("rsa_transfer_summary", "H4", "transfer_benefit", "H4_RSA_TRANSFER", "h4"),
                                          ("rsa_k90_summary", "H5", "k90", "H5_RSA_CALIBRATION", "h5")):
            df, p = load_art(cfg, root, "12_REPRESENTATION", key)
            prof.append(profile_table(df, "12_REPRESENTATION", key, p))
            t, r, lim = import_block12_association(df, hyp, label, fam)
            tabs[tk] = t
            regs.append(r)
            limits.extend(lim)
        for key in ("rsa_transfer_subject", "rsa_k90_subject"):
            df, p = load_art(cfg, root, "12_REPRESENTATION", key)
            prof.append(profile_table(df, "12_REPRESENTATION", key, p))
            tabs[key] = df

    def s_k90():
        df, p = load_art(cfg, root, "09_TEMPORAL_CALIBRATION", "k90_results")
        prof.append(profile_table(df, "09_TEMPORAL_CALIBRATION", "k90_results", p))
        tabs["k90"] = summarize_k90(extract_k90_exact(df, "09_TEMPORAL_CALIBRATION"), cfg)

    for nm, fn in (("H1/gap/descriptive/H6", s_h1), ("H2", s_h2), ("H3", s_h3), ("H4/H5", s_h45), ("k90", s_k90)):
        step(nm, fn)

    unimplemented = []

    # BLOCK10 transfer benefit: frozen upstream descriptive/subject-level data, no re-test.
    def s_transfer():
        df,p=load_art(cfg,root,"10_ME_MI_TRANSFER","transfer_benefit"); prof.append(profile_table(df,"10_ME_MI_TRANSFER","transfer_benefit",p))
        q,ts,neg=build_transfer_integration(df,cfg); tabs["transfer_subject"]=q; tabs["transfer_summary"]=ts; tabs["negative_transfer"]=neg
    step("BLOCK10 transfer integration",s_transfer)

    # BLOCK09: exact frozen k90 + exact primary A_SD source; no invented intermediate curves.
    def s_calibration():
        k,p=load_art(cfg,root,"09_TEMPORAL_CALIBRATION","k90_results")
        # Use the exact inspected frozen A_SD diagnostic artifact.
        ap=root/"09_TEMPORAL_CALIBRATION/diagnostics/A_SD_primary_combined.csv"
        if not ap.is_file(): raise RuntimeError(f"BLOCK09 A_SD primary artifact missing: {ap}")
        at=read_table(ap)
        prof.append(profile_table(k,"09_TEMPORAL_CALIBRATION","k90_results",p))
        prof.append(profile_table(at,"09_TEMPORAL_CALIBRATION","A_SD_primary_combined",ap))
        cs,ce=build_calibration_integration(k,at,cfg); tabs["calibration_summary"]=cs; tabs["calibration_efficiency"]=ce
        tabs["k90"]=summarize_k90(extract_k90_exact(k,"09_TEMPORAL_CALIBRATION"),cfg)
    step("BLOCK09 calibration integration",s_calibration)

    def s_robustness():
        d,p=load_art(cfg,root,"14_ROBUSTNESS","dropout_subject_level"); n,pn=load_art(cfg,root,"14_ROBUSTNESS","robustness_subject_level")
        prof.append(profile_table(d,"14_ROBUSTNESS","dropout_subject_level",p)); prof.append(profile_table(n,"14_ROBUSTNESS","robustness_subject_level",pn))
        rs,ds,ns=build_robustness_integration(d,n); tabs["robustness_summary"]=rs; tabs["dropout_summary"]=ds; tabs["noise_summary"]=ns
    step("BLOCK14 robustness integration",s_robustness)

    def s_xai():
        df,p=load_art(cfg,root,"13_XAI","xai_summary"); prof.append(profile_table(df,"13_XAI","xai_summary",p)); tabs["xai"]=df
        sim=root/"13_XAI_FROZEN_OUTPUTS/tables/ME_MI_attribution_similarity_summary.csv"; tabs["xai_similarity"]=read_table(sim)
    step("BLOCK13 XAI integration",s_xai)

    def s_neuro():
        df,p=load_art(cfg,root,"11_NEURO","spectral_summary"); prof.append(profile_table(df,"11_NEURO","spectral_summary",p)); tabs["neuro"]=df
        diff=root/"11_NEURO/statistics/ME_MI_spectral_difference.csv"
        if diff.is_file(): tabs["neuro_diff"]=read_table(diff)
        else: tabs["neuro_diff"]=pd.DataFrame([{"status":"NOT_AVAILABLE","reason":"ME_MI_spectral_difference.csv not present in audited path"}])
    step("BLOCK11 neuro integration",s_neuro)

    def s_block08():
        df,p=load_art(cfg,root,"08_ME_MI_STATISTICS","paired_statistics"); prof.append(profile_table(df,"08_ME_MI_STATISTICS","paired_statistics",p)); tabs["block08"]=df
    step("BLOCK08 frozen inference import",s_block08)

    registry = pd.concat([r for r in regs if len(r)], ignore_index=True) if regs else pd.DataFrame(columns=list(REGISTRY_FIELDS))
    if len(registry):
        registry = apply_fdr_to_registry(registry, cfg)
        registry["evidence_status"] = registry.apply(lambda r: evidence_status(r, cfg), axis=1)
        registry = registry[list(REGISTRY_FIELDS)]
    reg_audit = audit_registry(registry, cfg) if len(registry) else pd.DataFrame(columns=["test_id", "ok"])
    matrix, rollup = build_hypothesis_matrix(registry, cfg) if len(registry) else (pd.DataFrame(), pd.DataFrame())

    def w(rel: str, df: Optional[pd.DataFrame]) -> None:
        if df is not None and len(df):
            atomic_write_csv(df, out / rel)

    w("descriptive/subject_level_performance.csv", tabs.get("subj_level"))
    agg = tabs.get("agg")
    w("descriptive/aggregate_performance.csv", agg)
    if agg is not None and len(agg):
        w("descriptive/confidence_intervals.csv", agg[["branch", "model", "condition", "metric", "n", "mean", "ci95_low", "ci95_high"]])
        w("descriptive/distribution_summary.csv", agg[["branch", "model", "condition", "metric", "n", "min", "q1", "median", "q3", "max", "iqr", "sd"]])
    for rel, k in (("hypothesis_tests/H1_subject_dependency.csv", "h1"), ("hypothesis_tests/H2_ME_MI_similarity.csv", "h2"),
                   ("hypothesis_tests/H3_transfer.csv", "h3"), ("hypothesis_tests/H4_representation_transfer.csv", "h4"),
                   ("hypothesis_tests/H5_representation_calibration.csv", "h5"), ("hypothesis_tests/H6_ML_DL_comparison.csv", "h6"),
                   ("gap_analysis/SD_LOSO_subject_gap.csv", "subject_gap"), ("gap_analysis/ME_MI_gap_comparison.csv", "gap_cmp"),
                   ("gap_analysis/gap_summary.csv", "gap_summary"), ("calibration/k90_summary.csv", "k90"),
                   ("representation/RSA_summary.csv", "h2"), ("representation/representation_transfer.csv", "rsa_transfer_subject"),
                   ("representation/representation_calibration.csv", "rsa_k90_subject"),
                   ("transfer/transfer_asymmetry.csv", "h3")):
        w(rel, tabs.get(k))
    w("transfer/transfer_summary.csv", tabs.get("transfer_summary"))
    w("transfer/transfer_asymmetry.csv", tabs.get("h3"))
    w("transfer/negative_transfer_summary.csv", tabs.get("negative_transfer"))
    w("calibration/calibration_summary.csv", tabs.get("calibration_summary"))
    w("calibration/k90_summary.csv", tabs.get("k90"))
    w("calibration/calibration_efficiency_summary.csv", tabs.get("calibration_efficiency"))
    w("representation/RSA_summary.csv", tabs.get("h2"))
    w("representation/representation_transfer.csv", tabs.get("rsa_transfer_subject"))
    w("representation/representation_calibration.csv", tabs.get("rsa_k90_subject"))
    w("xai/XAI_summary.csv", tabs.get("xai"))
    w("xai/attribution_similarity_summary.csv", tabs.get("xai_similarity"))
    w("neuro/spectral_summary.csv", tabs.get("neuro"))
    w("neuro/ME_MI_spectral_comparison.csv", tabs.get("neuro_diff"))
    w("robustness/robustness_summary.csv", tabs.get("robustness_summary"))
    w("robustness/dropout_summary.csv", tabs.get("dropout_summary"))
    w("robustness/noise_summary.csv", tabs.get("noise_summary"))

    comp = []
    for blk, d in (("06B_ML_LOSO", tabs.get("transfer_subject")),
                   ("07A_DL_SD", sd_all.get("sd")),
                   ("07B_DL_LOSO", sd_all.get("lo"))):
        if d is not None and len(d):
            for c in ("training_seconds", "prediction_seconds"):
                if c in d.columns:
                    vals = pd.to_numeric(d[c], errors="coerce")
                    comp.append({"source_block": blk, "metric": c, "mean": vals.mean(),
                                 "median": vals.median(), "n": int(vals.notna().sum())})
    w("computational/ML_DL_computational_summary.csv",
      pd.DataFrame(comp) if comp else
      pd.DataFrame([{"source_block":"NOT_AVAILABLE","metric":"runtime","mean":np.nan,"median":np.nan,"n":0}]))

    w("multiple_comparisons/FDR_registry.csv", registry[["test_id", "hypothesis", "fdr_family", "fdr_action", "fdr_family_n", "raw_p", "upstream_q", "q"]] if len(registry) else None)
    w("multiple_comparisons/adjusted_results.csv", registry)
    w("statistical_registry/statistical_test_registry.csv", registry)
    w("hypothesis_matrix/hypothesis_evidence_summary.csv", matrix)
    w("hypothesis_matrix/hypothesis_rollup.csv", rollup)
    w("audits/statistical_registry_audit.csv", reg_audit)
    w("audits/statistical_input_audit.csv", pd.DataFrame(prof))
    subj_rows = []
    for k, d in (("sd", sd_all.get("sd")), ("loso", sd_all.get("lo"))):
        if d is not None:
            for (m, c), g in d.groupby(["model", "condition"]):
                subj_rows.append({"table": k, "model": m, "condition": c, "n_subjects": int(g["subject"].nunique()),
                                  "missing_subjects": ";".join(str(s) for s in sorted(set(SUBJECTS) - set(g["subject"]))),
                                  "duplicate_subject_rows": int(g.duplicated("subject").sum())})
    w("audits/subject_integrity_audit.csv", pd.DataFrame(subj_rows))
    env = environment_snapshot()
    atomic_write_json({"environment": env, "config_sha256": config_canonical_hash(cfg), "seed": cfg["freeze"]["seed"],
                       "upstream": evidence}, out / "audits" / "reproducibility_audit.json")
    w("audits/reproducibility_audit.csv", pd.DataFrame([{"item": k, "value": str(v)} for k, v in env.items()]
                                                       + [{"item": "config_sha256", "value": config_canonical_hash(cfg)}]))
    atomic_write_json(cfg, out / "configs" / "BLOCK15_config.json") if cfg_path.resolve() != (out / "configs" / "BLOCK15_config.json").resolve() else None

    reasons = completion_gate(out, cfg, upstream_problems=up_problems, analysis_errors=errors, registry=registry,
                              reg_audit=reg_audit, unimplemented=unimplemented)
    extra = {"limitations": limits, "n_registry_rows": int(len(registry)),
             "evidence_status_counts": registry["evidence_status"].value_counts().to_dict() if len(registry) else {},
             "runtime_seconds": time.perf_counter() - start,
             "interpretation_limits": ["Performance differences do not prove mechanism.", "Transfer/RSA relations are associative.",
                                       "RSA similarity != identical neural activity.", "Attribution != anatomical localisation.",
                                       "No verified rest baseline: no baseline-referenced ERD/ERS.", "Windows are not independent subjects."]}
    if reasons:
        return finish("NOT_READY", reasons, extra)
    atomic_write_json({"block": BLOCK_ID, "status": "PASS", "completed": True, "timestamp_utc": utc_now(),
                       "config_sha256": config_canonical_hash(cfg), "limitations": limits}, out / "BLOCK15_complete.json")
    return finish("PASS", [], extra)


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper 2 BLOCK15 final statistical integration.")
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--inspect", action="store_true", help="read-only dependency/schema report")
    ap.add_argument("--init-config", action="store_true", help="write a DRAFT config (never overwrites FROZEN)")
    ap.add_argument("--freeze-config", action="store_true", help="validate + hash + freeze the config")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--preflight", action="store_true", help="deep read-only validation of frozen upstream schemas")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).expanduser().resolve() if a.root else discover_project_root()
    out = root / BLOCK_NAME
    (out / "configs").mkdir(parents=True, exist_ok=True)
    cp = config_path_for(root)
    if a.inspect:
        print(f"inspection written: {write_inspection(root)}")
        return 0
    if a.init_config:
        if cp.exists():
            try:
                existing = json.loads(cp.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"REFUSED: existing BLOCK15_config.json is unreadable: {exc}")
                return 1

            existing_version = str(existing.get("config_version", ""))
            existing_status = str(existing.get("status", "")).upper()

            # Older BLOCK15 schemas are obsolete after the corrected protocol
            # integration. Reconstruct the current schema instead of using
            # stale upstream paths.
            if existing_version != CONFIG_VERSION:
                replacement = default_config()
                atomic_write_json(replacement, cp)
                print(
                    f"REPLACED STALE CONFIG: {existing_version or 'unknown'} -> "
                    f"{CONFIG_VERSION}; new config status=DRAFT"
                )
                print(f"DRAFT config written: {cp}")
                return 0

            # Same-version frozen configurations remain immutable.
            if existing_status == "FROZEN":
                print(
                    f"REFUSED: config {CONFIG_VERSION} is already FROZEN; "
                    "a same-version change requires a new exploratory analysis."
                )
                return 1

        atomic_write_json(default_config(), cp)
        print(f"DRAFT config written: {cp}")
        return 0
    if a.preflight:
        try:
            cfg = load_config(cp)
            probs = validate_config_structure(cfg)
            if probs:
                print("PREFLIGHT: CONFIG FAIL")
                for x in probs: print("  -", x)
                return 1
            ok, problems, audit = validate_runtime_artifacts(cfg, root, Logger(out / "logs" / "BLOCK15_preflight.log"))
            audit_path = out / "audits" / "BLOCK15_preflight_audit.csv"
            atomic_write_csv(audit, audit_path)
            if not ok:
                print("PREFLIGHT: FAIL")
                for x in problems: print("  -", x)
                print(f"audit: {audit_path}")
                return 1
            print("PREFLIGHT: PASS")
            print(f"audit: {audit_path}")
            return 0
        except Exception as exc:
            print(f"PREFLIGHT: FAIL: {type(exc).__name__}: {exc}")
            return 1
    if a.freeze_config:
        cfg = load_config(cp)
        if cfg.get("status") == "FROZEN":
            print("already FROZEN"); return 1
        probs = [p for p in freeze_problems({**cfg, "status": "FROZEN", "freeze": {**cfg["freeze"], "expected_config_sha256": "x"}})
                 if not p.startswith("CONFIG DRIFT")]
        if probs:
            print("CANNOT FREEZE:"); [print("  -", p) for p in probs]; return 1
        ok, live_problems, live_audit = validate_runtime_artifacts(cfg, root, Logger(out / "logs" / "BLOCK15_preflight.log"))
        atomic_write_csv(live_audit, out / "audits" / "BLOCK15_preflight_audit.csv")
        if not ok:
            print("CANNOT FREEZE: LIVE PREFLIGHT FAILED")
            [print("  -", p) for p in live_problems]
            return 1
        cfg["freeze"]["frozen_utc"] = utc_now()
        cfg["freeze"]["expected_config_sha256"] = config_canonical_hash(cfg)
        cfg["status"] = "FROZEN"
        atomic_write_json(cfg, cp)
        print(f"FROZEN. sha256={cfg['freeze']['expected_config_sha256']}")
        return 0
    try:
        return run_block(root, validate_only=a.validate_only, force=a.force)
    except Exception as exc:
        Logger(out / "BLOCK15_run.log").error(f"BLOCK15 FAIL: {type(exc).__name__}: {exc}")
        atomic_write_json({"block": BLOCK_ID, "status": "FAIL", "completed": False, "error": f"{type(exc).__name__}: {exc}",
                           "timestamp_utc": utc_now()}, out / "completion" / "BLOCK15_FAIL.json")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())