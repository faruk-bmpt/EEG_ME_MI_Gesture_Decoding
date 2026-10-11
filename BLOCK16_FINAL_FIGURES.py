#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

BLOCK16_FINAL_FIGURES.py

Paper 2 — Final Figure Generation

Purpose

-------

Implement the frozen BLOCK16 figure specification without modifying BLOCK15 or

creating new inferential statistics.

Scientific boundary

-------------------

* BLOCK15/BLOCK14 stored inferential quantities are authoritative.

* BLOCK16 may compute deterministic descriptive plotting quantities only.

* No new p-values, q-values, CIs, effect sizes, correlations, FDR, hypothesis

  tests, model selection, subject selection, or censoring transformations.

* Windows are never treated as inferential subjects.

* A panel whose frozen source cannot be verified is marked UNAVAILABLE and is

  not replaced by a BLOCK16-derived inferential quantity.

* BLOCK14-dependent figures remain blocked unless the strict clean-reference

  audit and completion marker pass.

Usage

-----

    python BLOCK16_FINAL_FIGURES.py --inspect

    python BLOCK16_FINAL_FIGURES.py --run

    python BLOCK16_FINAL_FIGURES.py --run --paper2-root "F:\\...\\Paper2_EEG_BCI"

The Paper 2 root can also be supplied with PAPER2_ROOT.

Canonical filename: BLOCK16_FINAL_FIGURES.py

"""

from __future__ import annotations

import argparse
import io
import ast

import hashlib

import json

import math

import os

import re

import sys

import traceback

from datetime import datetime, timezone

from pathlib import Path

from typing import Any, Iterable

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------

# Frozen implementation configuration

# ---------------------------------------------------------------------------

CONFIG = {

    "block": "BLOCK16",

    "config_version": "1.1.0",

    "specification": "BLOCK16_FINAL_FIGURE_STRUCTURE_FROZEN.md",

    "primary_subjects": 25,

    "conditions": ["ME", "MI"],

    "gestures": list(range(1, 8)),

    "chance_level": 1.0 / 7.0,

    "metrics_primary": ["Macro-F1", "Accuracy", "Balanced Accuracy", "MCC", "Cohen's kappa"],

    "metric_plot": "Macro-F1",

    "models_ml": ["LDA", "KNN", "SVM", "RandomForest", "XGBoost"],

    "models_dl": ["EEGNet", "ShallowConvNet", "DeepConvNet", "CNN-BiLSTM", "CNN-BiLSTM-Attention"],

    "models_representation": ["EEGNet", "CNN-BiLSTM", "CNN-BiLSTM-Attention"],

    "calibration_grid": [0, 10, 20, 30, 40, 50, 60, 80, 90],

    "dropout_nominal": [5, 10, 20, 30],

    "output_formats": ["pdf", "svg", "png"],

    "png_dpi": 600,

    "deterministic_tolerance": 1e-10,

    "main_figures": [f"FIG{i:02d}" for i in range(1, 11)],

    "supplementary_figures": [f"FIG-S{i:02d}" for i in range(1, 15)],

}

FIGURE_STATUS = {

    "FIG01": "SUPPORTING",

    "FIG02": "SUPPORTING",

    "FIG03": "CONFIRMATORY — H1",

    "FIG04": "SUPPORTING",

    "FIG05": "CONFIRMATORY — H3",

    "FIG06": "SUPPORTING / CALIBRATION EVIDENCE",

    "FIG07": "CONFIRMATORY — H2",

    "FIG08": "CONFIRMATORY — H4/H5",

    "FIG09": "EXPLORATORY/DESCRIPTIVE",

    "FIG10": "SUPPORTING / ROBUSTNESS",

}

FIGURE_TITLES = {

    "FIG01": "Experimental workflow and analysis architecture",

    "FIG02": "Paired cohort and analyzed EEG span",

    "FIG03": "Subject-dependent versus LOSO decoding",

    "FIG04": "Motor execution versus motor imagery decoding",

    "FIG05": "Bidirectional ME–MI transfer",

    "FIG06": "Temporal calibration and k90",

    "FIG07": "Latent representation geometry and RSA",

    "FIG08": "Representation similarity versus transfer/calibration",

    "FIG09": "Explainable AI summaries",

    "FIG10": "Robustness under channel dropout and synthetic noise",

}

# Exact frozen artifact candidates from the inspected BLOCK15 code.

ARTIFACTS = {

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

    "12_REPRESENTATION_ANALYSIS": {

        "subject_rsa": "12_REPRESENTATION_ANALYSIS/subject_similarity/subject_level_rsa.csv",

        "rsa_transfer_subject": "12_REPRESENTATION_ANALYSIS/transfer_relationship/rsa_vs_transfer_subject_level.csv",

        "rsa_transfer_summary": "12_REPRESENTATION_ANALYSIS/transfer_relationship/rsa_transfer_association_summary.csv",

        "rsa_k90_subject": "12_REPRESENTATION_ANALYSIS/calibration_relationship/rsa_vs_k90_subject_level.csv",

        "rsa_k90_summary": "12_REPRESENTATION_ANALYSIS/calibration_relationship/rsa_k90_association_summary.csv",

    },

    "13_XAI": {

        "ml_features": "13_XAI_FROZEN_OUTPUTS/tables/ML_top_features.csv",

        "xai_summary": "13_XAI_FROZEN_OUTPUTS/tables/XAI_summary.csv",

        "attribution_similarity_summary": "13_XAI_FROZEN_OUTPUTS/tables/ME_MI_attribution_similarity_summary.csv",

        "paired_channel_statistics": "13_XAI_FROZEN_OUTPUTS/tables/XAI_ME_MI_paired_channel_statistics.csv",

        "attribution_similarity_statistics": "13_XAI_FROZEN_OUTPUTS/tables/XAI_attribution_similarity_statistics.csv",

    },

    "14_ROBUSTNESS": {

        "robustness_subject_level": "14_ROBUSTNESS/noise/BLOCK14_noise_subject_level.csv",

        "dropout_subject_level": "14_ROBUSTNESS/channel_dropout/BLOCK14_channel_dropout_subject_level.csv",

        "clean_reference_identity_audit": "14_ROBUSTNESS/tables/audit/BLOCK14_upstream_clean_reference_check.csv",

        "complete_marker": "14_ROBUSTNESS/BLOCK14_complete.json",

        "config": "14_ROBUSTNESS/BLOCK14_config.json",

    },

    "BLOCK15": {

        "config": "15_FINAL_STATISTICS/configs/BLOCK15_config.json",

        "complete_marker": "15_FINAL_STATISTICS/completion/BLOCK15_complete.json",

        "subject_level": "15_FINAL_STATISTICS/descriptive/subject_level_performance.csv",

        "aggregate": "15_FINAL_STATISTICS/descriptive/aggregate_performance.csv",

        "gap": "15_FINAL_STATISTICS/gap_analysis/SD_LOSO_subject_gap.csv",

        "me_mi_gap": "15_FINAL_STATISTICS/gap_analysis/ME_MI_gap_comparison.csv",

        "transfer": "15_FINAL_STATISTICS/transfer/transfer_summary.csv",

        "asymmetry": "15_FINAL_STATISTICS/transfer/transfer_asymmetry.csv",

        "negative_transfer": "15_FINAL_STATISTICS/transfer/negative_transfer_summary.csv",

        "calibration": "15_FINAL_STATISTICS/calibration/calibration_summary.csv",

        "k90": "15_FINAL_STATISTICS/calibration/k90_summary.csv",

        "rsa": "15_FINAL_STATISTICS/representation/RSA_summary.csv",

        "rsa_transfer": "15_FINAL_STATISTICS/representation/representation_transfer.csv",

        "rsa_calibration": "15_FINAL_STATISTICS/representation/representation_calibration.csv",

        "xai": "15_FINAL_STATISTICS/xai/XAI_summary.csv",

        "xai_similarity": "15_FINAL_STATISTICS/xai/attribution_similarity_summary.csv",

        "robustness": "15_FINAL_STATISTICS/robustness/robustness_summary.csv",

        "dropout": "15_FINAL_STATISTICS/robustness/dropout_summary.csv",

        "noise": "15_FINAL_STATISTICS/robustness/noise_summary.csv",

        "computational": "15_FINAL_STATISTICS/computational/ML_DL_computational_summary.csv",

    },

}

# Candidate BLOCK03/BLOCK04 files are deliberately candidates, not assumptions.

CONFIG_CANDIDATES = {

    "BLOCK03": [

        "03_PREPROCESSING/BLOCK03_config.json",

        "03_PREPROCESSING/config/BLOCK03_config.json",

        "03_PREPROCESSING/BLOCK03_PREPROCESSING_config.json",

    ],

    "BLOCK04": [

        "04_WINDOWING/BLOCK04_config.json",

        "04_WINDOWING/config/BLOCK04_config.json",

        "04_WINDOWING/BLOCK04_WINDOWING_config.json",

    ],

    "PROTOCOL": [

        "00_PROTOCOL/protocol.json",

        "00_PROTOCOL/BLOCK00_config.json",

        "00_PROTOCOL/Paper2_protocol.json",

    ],

}

# Aliases used only to recognize schemas; they never alter source values.

ALIASES = {

    "subject": ["subject", "subject_id", "participant", "participant_id", "sub", "sid"],

    "condition": ["condition", "cond"],

    "model": ["model", "model_name", "classifier", "architecture"],

    "metric": ["metric", "measure"],

    "accuracy": ["accuracy", "acc"],

    "balanced_accuracy": ["balanced_accuracy", "balanced_acc", "balancedaccuracy", "bac"],

    "macro_f1": ["macro_f1", "macro_f1_score", "f1_macro", "macro-f1", "macro f1"],

    "mcc": ["mcc", "matthews_corrcoef", "matthews_correlation"],

    "kappa": ["kappa", "cohen_kappa", "cohen's_kappa", "cohens_kappa"],

    "rho": ["rho", "spearman_rho", "rho_spearman", "rsa_rho", "representation_similarity", "me_mi_rho"],

    "transfer_benefit": ["transfer_benefit", "transfer_gain"],

    "transfer_loss": ["transfer_loss", "transfer_degradation"],

    "k90": ["k90", "k_90", "calibration_k90"],

    "fraction": ["fraction", "calibration_fraction", "calibration_percent"],

    "value": ["value", "score", "performance"],

    "status": ["status", "k90_status", "classification_status"],

    "direction": ["direction", "transfer_direction"],

}

# Exact frozen BLOCK13 gesture order; do not reorder or extend.
GESTURES = list(range(1, 8))

# Exact frozen BLOCK13 feature-family order; do not reorder or extend.
FEATURE_FAMILY_ORDER = ["spectral", "spectral_relative", "spectral_ratio"]

# Exact frozen BLOCK13 model order; do not reorder or extend.
MODEL_ORDER = [
    "LDA",
    "KNN",
    "SVM",
    "RF",
    "XGB",
    "EEGNet",
    "ShallowConvNet",
    "DeepConvNet",
    "CNN-BiLSTM",
    "CNN-BiLSTM-Attention",
]

COLOR_ORDER = {

    "ME": "tab:blue",

    "MI": "tab:orange",

}

# One fixed color-blind-safe palette; no result-dependent color selection.

MODEL_COLORS = {

    "LDA": "#0072B2", "KNN": "#E69F00", "SVM": "#009E73",

    "RandomForest": "#CC79A7", "XGBoost": "#56B4E9",

    "EEGNet": "#0072B2", "ShallowConvNet": "#E69F00",

    "DeepConvNet": "#009E73", "CNN-BiLSTM": "#CC79A7",

    "CNN-BiLSTM-Attention": "#D55E00",

}

MODEL_TICK_LABELS = {
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "ShallowConvNet": "Shallow\nConvNet",
    "DeepConvNet": "Deep\nConvNet",
    "CNN-BiLSTM": "CNN–BiLSTM",
    "CNN-BiLSTM-Attention": "CNN–BiLSTM–\nAttention",
}

def model_tick_label(model_name: str) -> str:
    return MODEL_TICK_LABELS.get(model_name, model_name)

plt.rcParams.update({

    "font.family": "DejaVu Sans",

    "font.size": 9,

    "axes.titlesize": 10,

    "axes.labelsize": 9,

    "legend.fontsize": 8,

    "figure.titlesize": 11,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

})

# ---------------------------------------------------------------------------

# Generic helpers

# ---------------------------------------------------------------------------

def utc_now() -> str:

    return datetime.now(timezone.utc).isoformat()

def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(lambda: f.read(1024 * 1024), b""):

            h.update(chunk)

    return h.hexdigest()

def canonical_json(obj: Any) -> str:

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def config_hash() -> str:

    return hashlib.sha256(canonical_json(CONFIG).encode("utf-8")).hexdigest()

def norm_col(x: Any) -> str:

    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")

def metric_norm(v: Any) -> str:

    return norm_col(v).replace("macro_f1_score", "macro_f1")

def feature_family_label(v: Any) -> str:

    labels = {
        "spectral": "Spectral",
        "spectral_relative": "Relative spectral",
        "spectral_ratio": "Spectral ratio",
    }
    key = norm_col(v)
    return labels.get(key, str(v))

def find_col(df: pd.DataFrame, logical: str, required: bool = True) -> str | None:

    normalized = {norm_col(c): c for c in df.columns}

    for candidate in ALIASES.get(logical, [logical]):

        n = norm_col(candidate)

        if n in normalized:

            return normalized[n]

    if required:

        raise KeyError(f"Required logical column '{logical}' not found. Columns={list(df.columns)}")

    return None

def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:

    return pd.to_numeric(df[col], errors="coerce")

def normalize_condition(v: Any) -> str:

    s = str(v).strip().lower()

    if s in {"me", "motor_execution", "motor execution"}:

        return "ME"

    if s in {"mi", "motor_imagery", "motor imagery"}:

        return "MI"

    return str(v)

def normalize_model(v: Any) -> str:

    s = str(v).strip()

    key = s.lower().replace("_", "").replace("-", "").replace(" ", "")

    aliases = {

        "rf": "RandomForest", "randomforest": "RandomForest",

        "xgb": "XGBoost", "xgboost": "XGBoost",

        "cnnbilstm": "CNN-BiLSTM",

        "cnnbilstmattention": "CNN-BiLSTM-Attention",

    }

    return aliases.get(key, s)

def load_csv(path: Path) -> pd.DataFrame:

    return pd.read_csv(path)

def source_path(root: Path, rel: str) -> Path:

    return root / Path(rel)

def locate_first(root: Path, candidates: Iterable[str]) -> Path | None:

    for rel in candidates:

        p = root / rel

        if p.is_file():

            return p

    return None

def locate_block04_final_source(root: Path) -> Path | None:

    """Locate a readable BLOCK04 final source containing both frozen W/O constants."""

    candidates = [
        root / "scripts" / "BLOCK04_WINDOWING_FINAL_SCIENTIFIC.py",
        root / "04_WINDOWING" / "BLOCK04_WINDOWING_FINAL_SCIENTIFIC.py",
    ]

    for base in [root / "scripts", root / "04_WINDOWING"]:
        if base.is_dir():
            candidates.extend(sorted(
                p for p in base.glob("BLOCK04_WINDOWING_FINAL_SCIENTIFIC*.py")
                if p.is_file()
            ))

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if (
            re.search(r"FINAL_WINDOW_SAMPLES\s*=\s*\d+", txt)
            and re.search(r"FINAL_OVERLAP_PERCENT\s*=\s*\d+", txt)
        ):
            return p
    return None

def write_json(path: Path, obj: Any) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def save_figure(fig: plt.Figure, out_dir: Path, fig_id: str) -> dict:

    out_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    pdf = out_dir / f"{fig_id}.pdf"

    svg = out_dir / f"{fig_id}.svg"

    png = out_dir / f"{fig_id}.png"

    fig.savefig(pdf, bbox_inches="tight")

    fig.savefig(svg, bbox_inches="tight")

    try:
        fig.savefig(png, dpi=CONFIG["png_dpi"], bbox_inches="tight")
    except OSError as exc:
        if getattr(exc, "errno", None) != 22:
            raise
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=CONFIG["png_dpi"], bbox_inches="tight")
        png.write_bytes(buffer.getvalue())

    files["pdf"] = str(pdf)

    files["svg"] = str(svg)

    files["png"] = str(png)

    plt.close(fig)

    return files

def caption_status(fig_id: str) -> str:

    return FIGURE_STATUS[fig_id]

def style_axes(ax):

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.grid(axis="y", alpha=0.15, linewidth=0.6)

def title_footer(fig, fig_id: str, extra: str = ""):

    fig.suptitle(FIGURE_TITLES[fig_id], y=1.02)

    footer = f"Status: {FIGURE_STATUS[fig_id]}"

    if extra:
        footer += f"\n{extra}"

    fig.text(0.02, 0.01, footer, fontsize=8, va="bottom")

def panel_label(ax, label: str):

    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontweight="bold", va="bottom")

def subject_column(df: pd.DataFrame) -> str:

    return find_col(df, "subject")

def prepare_metric_long(df: pd.DataFrame, metric: str = "Macro-F1") -> pd.DataFrame:

    """Recognize either long or wide frozen subject-level tables."""

    d = df.copy()

    if find_col(d, "metric", required=False):

        mc = find_col(d, "metric", required=False)

        vc = find_col(d, "value", required=False)

        if vc:

            out = d.rename(columns={find_col(d, "subject"): "subject",

                                    find_col(d, "condition"): "condition",

                                    find_col(d, "model"): "model",

                                    vc: "value"})

            out["condition"] = out["condition"].map(normalize_condition)

            out["model"] = out["model"].map(normalize_model)

            out["metric"] = out[mc].astype(str)

            return out

    # Wide form: locate common macro-F1 columns.

    sc = find_col(d, "subject")

    cc = find_col(d, "condition", required=False)

    mc = find_col(d, "model", required=False)

    candidates = [c for c in d.columns if "macro" in norm_col(c) and "f1" in norm_col(c)]

    if candidates:

        rows = []

        for c in candidates:

            m = d.copy()

            m["subject"] = m[sc]

            if cc:

                m["condition"] = m[cc].map(normalize_condition)

            else:

                # Try column suffix.

                text = norm_col(c)

                m["condition"] = "ME" if "me" in text and "mi" not in text else "MI" if "mi" in text else ""

            if mc:

                m["model"] = m[mc].map(normalize_model)

            else:

                m["model"] = ""

            m["value"] = pd.to_numeric(m[c], errors="coerce")

            m["metric"] = metric

            rows.append(m[["subject", "condition", "model", "value", "metric"]])

        return pd.concat(rows, ignore_index=True)

    raise KeyError(f"Cannot recognize subject-level metric table columns: {list(d.columns)}")

def read_subject_performance(root: Path) -> pd.DataFrame:

    p = source_path(root, ARTIFACTS["BLOCK15"]["subject_level"])

    if not p.is_file():

        raise FileNotFoundError(p)

    d = load_csv(p)

    return prepare_metric_long(d)

def verify_numeric_finite(df: pd.DataFrame, cols: list[str]) -> bool:

    for c in cols:

        if c not in df.columns:

            return False

        x = pd.to_numeric(df[c], errors="coerce")

        if not np.isfinite(x.dropna()).all():

            return False

    return True

# ---------------------------------------------------------------------------

# Release audit

# ---------------------------------------------------------------------------

def audit_artifact(root: Path, rel: str, required_columns: list[str] | None = None,

                   expected_n: int | None = None, key_columns: list[str] | None = None,

                   numeric_columns: list[str] | None = None,

                   stored_quantity: str | None = None) -> dict:

    p = source_path(root, rel)

    result = {

        "source_file": rel,

        "source_sha256": "",

        "required_columns": "|".join(required_columns or []),

        "columns_present": False,

        "expected_N": expected_n,

        "observed_N": None,

        "finite_check": False,

        "missing_key_count": None,

        "duplicate_key_count": None,

        "stored_quantity": stored_quantity or "",

        "status": "UNAVAILABLE",

        "reason": "",

    }

    if not p.is_file():

        result["reason"] = "source file does not exist"

        return result

    result["source_sha256"] = sha256_file(p)

    try:

        if p.suffix.lower() == ".csv":

            d = load_csv(p)

        elif p.suffix.lower() == ".json":

            obj = json.loads(p.read_text(encoding="utf-8"))

            if isinstance(obj, list):

                d = pd.DataFrame(obj)

            elif isinstance(obj, dict):

                # Marker/config artifacts are audited structurally below.

                d = pd.DataFrame([obj])

            else:

                d = pd.DataFrame()

        else:

            result["reason"] = "unsupported audit source type"

            return result

        result["observed_N"] = len(d)

        required_columns = required_columns or []

        missing = [c for c in required_columns if c not in d.columns]

        result["columns_present"] = len(missing) == 0

        if key_columns:

            key_missing = [c for c in key_columns if c not in d.columns]

            if key_missing:

                result["missing_key_count"] = -1

                result["reason"] = f"missing key columns: {key_missing}"

                return result

            result["missing_key_count"] = int(d[key_columns].isna().any(axis=1).sum())

            result["duplicate_key_count"] = int(d.duplicated(key_columns).sum())

        else:

            result["missing_key_count"] = 0

            result["duplicate_key_count"] = 0

        if numeric_columns:

            result["finite_check"] = verify_numeric_finite(d, numeric_columns)

        else:

            result["finite_check"] = True

        if expected_n is not None and len(d) != expected_n:

            result["reason"] = f"observed N={len(d)} != expected N={expected_n}"

            return result

        if missing:

            result["reason"] = f"missing required columns: {missing}"

            return result

        if result["missing_key_count"] not in (0, None):

            result["reason"] = "missing required key values"

            return result

        if result["duplicate_key_count"] not in (0, None):

            result["reason"] = "duplicate inferential keys"

            return result

        if not result["finite_check"]:

            result["reason"] = "required numeric values are not finite"

            return result

        result["status"] = "AVAILABLE"

        result["reason"] = "artifact schema/size/key/finite audit passed"

        return result

    except Exception as exc:

        result["reason"] = f"read/audit error: {exc}"

        return result

def _locate_block14_config(root: Path, marker_sha: str | None) -> Path | None:
    """Locate the actual BLOCK14 config; never synthesize it from a hash."""
    exact = root / ARTIFACTS["14_ROBUSTNESS"]["config"]
    candidates = []
    if exact.is_file():
        candidates.append(exact)

    # Deterministic recovery: search the project tree for the canonical basename,
    # including archive/backup locations. Prefer a hash match to the completion marker.
    for p in sorted(root.rglob("BLOCK14_config.json")):
        if p.is_file() and p not in candidates:
            candidates.append(p)

    if marker_sha:
        for p in candidates:
            try:
                if sha256_file(p) == marker_sha:
                    return p
            except OSError:
                pass
    return candidates[0] if candidates else None

def audit_block14(root: Path) -> dict:
    marker_rel = ARTIFACTS["14_ROBUSTNESS"]["complete_marker"]
    audit_rel = ARTIFACTS["14_ROBUSTNESS"]["clean_reference_identity_audit"]

    out = {"marker": {}, "config": {}, "clean_reference": {}, "strict_pass": False}

    marker = source_path(root, marker_rel)
    if marker.is_file():
        try:
            obj = json.loads(marker.read_text(encoding="utf-8"))
            out["marker"] = {
                "exists": True,
                "sha256": sha256_file(marker),
                "status": obj.get("status"),
                "expected_units": obj.get("expected_units"),
                "completed_units": obj.get("completed_units"),
                "failed_units": obj.get("failed_units"),
                "missing_units": obj.get("missing_units"),
                "config_sha256": obj.get("config_sha256"),
                "input_config_sha256": (obj.get("input_hashes") or {}).get("BLOCK14_config"),
                "clean_reference_total": obj.get("CLEAN_REFERENCE_TOTAL"),
                "clean_reference_pass": obj.get("CLEAN_REFERENCE_PASS"),
                "clean_reference_fail": obj.get("CLEAN_REFERENCE_FAIL"),
                "prediction_identity_fail": obj.get("PREDICTION_IDENTITY_FAIL"),
            }
        except Exception as exc:
            out["marker"] = {"exists": True, "error": str(exc)}
    else:
        out["marker"] = {"exists": False}

    marker_sha = out.get("marker", {}).get("config_sha256")
    config = _locate_block14_config(root, marker_sha)
    if config is not None:
        try:
            obj = json.loads(config.read_text(encoding="utf-8"))
            actual_sha = sha256_file(config)
            # BLOCK14 records two different hashes:
            #   config_sha256 = canonical JSON-content hash used by execution units
            #   input_hashes.BLOCK14_config = raw-file SHA256 recorded in the input manifest
            # They are intentionally different hash domains and must not be compared to each other.
            input_marker_sha = out.get("marker", {}).get("input_config_sha256")
            canonical_sha = hashlib.sha256(
                json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            ).hexdigest()
            out["config"] = {
                "exists": True,
                "path": str(config.relative_to(root)),
                "sha256": actual_sha,
                "marker_config_sha256": marker_sha,
                "marker_input_config_sha256": input_marker_sha,
                "canonical_sha256": canonical_sha,
                "raw_hash_matches_input_manifest": bool(input_marker_sha and actual_sha == input_marker_sha),
                "canonical_hash_matches_marker": bool(marker_sha and canonical_sha == marker_sha),
                # Backward-compatible aggregate verification flag.
                "hash_matches_marker": bool(
                    (input_marker_sha and actual_sha == input_marker_sha)
                    and (marker_sha and canonical_sha == marker_sha)
                ),
                "data": obj,
            }
        except Exception as exc:
            out["config"] = {"exists": True, "path": str(config), "error": str(exc)}
    else:
        out["config"] = {"exists": False}

    audit = source_path(root, audit_rel)
    if audit.is_file():
        try:
            d = pd.read_csv(audit)
            status_col = find_col(d, "status", required=False)
            pred_col = next((c for c in d.columns if norm_col(c) in {
                "prediction_identical", "predictions_identical", "identity_match"
            }), None)
            invalid = 0
            if status_col:
                invalid += int((d[status_col].astype(str).str.upper() != "PASS").sum())
            if pred_col:
                invalid += int((d[pred_col].astype(str).str.upper().isin({"FALSE", "0", "NO"})).sum())
            out["clean_reference"] = {
                "exists": True,
                "sha256": sha256_file(audit),
                "rows": len(d),
                "status_column": status_col,
                "prediction_column": pred_col,
                "invalid_count": invalid,
            }
        except Exception as exc:
            out["clean_reference"] = {"exists": True, "error": str(exc)}
    else:
        out["clean_reference"] = {"exists": False}

    m = out["marker"]
    c = out["clean_reference"]
    cfg = out["config"]

    expected_clean = m.get("clean_reference_total")
    if not isinstance(expected_clean, int):
        expected_clean = 500

    clean_ok = (
        c.get("exists") is True
        and c.get("rows") == expected_clean
        and c.get("invalid_count", 1) == 0
    )

    # IMPORTANT: BLOCK14 expected_units=8500 is the total perturbation-unit count,
    # not the number of clean-reference audit rows. These are distinct quantities.
    out["strict_pass"] = bool(
        m.get("status") == "PASS"
        and isinstance(m.get("expected_units"), int)
        and m.get("completed_units") == m.get("expected_units")
        and m.get("failed_units") in (0, None)
        and m.get("missing_units") in (0, None)
        and cfg.get("exists") is True
        and cfg.get("raw_hash_matches_input_manifest") is True
        and cfg.get("canonical_hash_matches_marker") is True
        and clean_ok
        and (m.get("clean_reference_fail") in (0, None))
        and (m.get("prediction_identity_fail") in (0, None))
    )
    return out

def audit_sources(root: Path) -> pd.DataFrame:

    rows = []

    # G5: primary subject-level performance.
    # FIG03: 25 subjects × 2 conditions × 10 models × 5 metrics = 2,500 rows.
    rows.append(("FIG03", "A-C", ARTIFACTS["BLOCK15"]["gap"],
                 ["subject", "condition", "model", "metric", "A_SD", "A_LOSO", "gap"],
                 25 * 2 * 10 * 5,
                 ["subject", "condition", "model", "metric"],
                 ["A_SD", "A_LOSO", "gap"],
                 "stored G_i = A_SD − A_LOSO with SD/LOSO subject-level values"))

    # FIG04: 25 subjects × 2 conditions × 10 models × 2 branches × 5 metrics = 5,000 rows.
    rows.append(("FIG04", "A-D", ARTIFACTS["BLOCK15"]["subject_level"],
                 ["subject", "condition", "model", "branch", "metric", "value"],
                 25 * 2 * 10 * 2 * 5,
                 ["subject", "condition", "model", "branch", "metric"],
                 ["value"],
                 "stored subject-level performance; branch retained in inferential key"))

    # FIG05 frozen BLOCK15 transfer_summary currently contains one metric per
    # model/direction combination: 10 models × 2 directions = 20 physical rows.
    # The source audit must validate the artifact that actually exists; it must
    # not assume an additional metric that is not present in the frozen file.
    rows.append(("FIG05", "A-E", ARTIFACTS["BLOCK15"]["transfer"],
                 ["model", "transfer_direction", "metric", "mean"],
                 20,
                 ["model", "transfer_direction", "metric"],
                 ["mean", "median", "ci95_low", "ci95_high", "negative_n", "negative_percent"],
                 "stored transfer-benefit summary quantities"))

    # FIG06 frozen BLOCK15 calibration_summary currently contains one metric
    # per model/condition combination: 10 models × 2 conditions = 20 physical
    # rows.  k90 is audited separately from the subject-level BLOCK09 source.
    rows.append(("FIG06", "A-B", ARTIFACTS["BLOCK15"]["calibration"],
                 ["model", "condition", "metric", "n_subjects", "n_valid_k90",
                  "mean_k90", "median_k90", "mean_A_SD_primary", "status"],
                 20,
                 ["model", "condition", "metric"],
                 ["n_subjects", "n_valid_k90", "mean_k90", "median_k90", "mean_A_SD_primary"],
                 "stored calibration summary quantities"))

    rows.append(("FIG06", "k90", ARTIFACTS["09_TEMPORAL_CALIBRATION"]["k90_results"],
                 ["subject_id", "condition", "model", "metric", "k90", "k90_status"],
                 500,
                 ["subject_id", "condition", "model", "metric"],
                 ["k90"],
                 "frozen BLOCK09 subject-level k90 values and censoring status"))

    rows.append(("FIG07", "A-D", ARTIFACTS["12_REPRESENTATION_ANALYSIS"]["subject_rsa"],
                 ["subject_id", "model", "rsa_method", "rdm_distance",
                  "rho_spearman", "n_gesture_pairs", "n_gestures"],
                 75,
                 ["subject_id", "model"],
                 ["rho_spearman", "n_gesture_pairs", "n_gestures"],
                 "stored subject-level RSA quantities"))

    rows.append(("FIG08", "A", ARTIFACTS["BLOCK15"]["rsa_transfer"],
                 ["subject_id", "model", "rho_spearman",
                  "ME_to_MI", "MI_to_ME", "bidirectional_mean_transfer_benefit"],
                 75,
                 ["subject_id", "model"],
                 ["rho_spearman", "ME_to_MI", "MI_to_ME", "bidirectional_mean_transfer_benefit"],
                 "stored RSA-transfer subject-level quantities"))

    rows.append(("FIG08", "B", ARTIFACTS["BLOCK15"]["rsa_calibration"],
                 ["subject_id", "model", "rho_spearman", "ME_k90", "MI_k90",
                  "mean_k90", "k90_valid", "k90_exclusion_reason"],
                 75,
                 ["subject_id", "model"],
                 ["rho_spearman", "ME_k90", "MI_k90", "mean_k90"],
                 "stored RSA-k90 subject-level quantities with exclusion semantics"))

    rows.append(("FIG09", "A-B", "13_XAI/tables/ML_top_features.csv",
                 ["model", "condition", "gesture", "feature_id", "feature_name", "feature_family", "channel", "frequency_group", "mean_abs_shap", "rank"],
                 1400, ["model", "condition", "gesture", "feature_id"], ["mean_abs_shap", "rank"],
                 "stored BLOCK13 mean_abs_shap"))
    rows.append(("FIG09", "C", "13_XAI/tables/XAI_ME_MI_paired_channel_statistics.csv",
                 ["branch", "model", "gesture", "channel", "n_subjects", "mean_ME_minus_MI", "median_ME_minus_MI", "effect_size_dz", "ci95_low", "ci95_high", "p_value", "p_fdr"],
                 910, ["branch", "model", "gesture", "channel"], ["mean_ME_minus_MI", "median_ME_minus_MI", "effect_size_dz", "ci95_low", "ci95_high"],
                 "stored BLOCK13 mean_ME_minus_MI"))
    rows.append(("FIG09", "D", "13_XAI/tables/XAI_attribution_similarity_statistics.csv",
                 ["model", "gesture", "metric", "n_subjects", "mean", "median", "ci95_low", "ci95_high", "p_value_vs_zero", "p_fdr"],
                 210, ["model", "gesture", "metric", "n_subjects"], [],
                 "stored BLOCK13 attribution similarity; N=0 rows are unavailable/censored and handled by renderer"))

    rows.append(("FIG10", "dropout", ARTIFACTS["14_ROBUSTNESS"]["dropout_subject_level"],

                 [], None, [], None, "clean/perturbed/delta stored quantities"))

    rows.append(("FIG10", "noise", ARTIFACTS["14_ROBUSTNESS"]["robustness_subject_level"],

                 [], None, [], None, "clean/perturbed/delta/noise stored quantities"))

    out = []

    for fig, panel, rel, req, exp, keys, nums, qty in rows:

        a = audit_artifact(root, rel, req, exp, keys, nums, qty)

        a["figure_id"] = fig

        a["panel_id"] = panel

        out.append(a)

    return pd.DataFrame(out)

# ---------------------------------------------------------------------------

# Safe extraction helpers

# ---------------------------------------------------------------------------

def get_value_column(df: pd.DataFrame, preferred: list[str]) -> str | None:

    norm = {norm_col(c): c for c in df.columns}

    for c in preferred:

        if norm_col(c) in norm:

            return norm[norm_col(c)]

    return None

def standardize_subject_metric(df: pd.DataFrame, metric_name: str = "Macro-F1") -> pd.DataFrame:

    d = df.copy()

    # Long form.

    mc = find_col(d, "metric", required=False)

    vc = find_col(d, "value", required=False)

    sc = find_col(d, "subject", required=False)

    cc = find_col(d, "condition", required=False)

    modc = find_col(d, "model", required=False)
    brc = next((c for c in d.columns if norm_col(c) == "branch"), None)

    if sc and cc and modc and mc and vc:

        cols = [sc, cc, modc] + ([brc] if brc else []) + [mc, vc]

        out = d[cols].copy()

        out.columns = ["subject", "condition", "model"] + (["branch"] if brc else []) + ["metric", "value"]

        out["condition"] = out["condition"].map(normalize_condition)

        out["model"] = out["model"].map(normalize_model)

        out["value"] = pd.to_numeric(out["value"], errors="coerce")

        metric_norm = (
            out["metric"].astype(str).str.strip().str.lower()
            .str.replace("_", " ", regex=False)
            .str.replace("-", " ", regex=False)
        )
        target = metric_name.strip().lower().replace("_", " ").replace("-", " ")
        return out[metric_norm.eq(target)]

    # Common wide format.

    if sc and cc and modc:

        metric_col = get_value_column(d, ["macro_f1", "macro_f1_score", "f1_macro", "macro-f1"])

        if metric_col:

            out = d[[sc, cc, modc, metric_col]].copy()

            out.columns = ["subject", "condition", "model", "value"]

            out["metric"] = metric_name

            out["condition"] = out["condition"].map(normalize_condition)

            out["model"] = out["model"].map(normalize_model)

            out["value"] = pd.to_numeric(out["value"], errors="coerce")

            return out

    raise KeyError(f"Cannot standardize subject metric schema: {list(d.columns)}")

def filter_metric(d: pd.DataFrame, metric: str = "Macro-F1") -> pd.DataFrame:

    if "metric" in d.columns:

        x = d["metric"].astype(str).str.lower().str.replace("_", " ").str.replace("-", " ")

        target = metric.lower().replace("-", " ")

        return d[x == target].copy()

    return d.copy()

# ---------------------------------------------------------------------------

# Figure implementations

# ---------------------------------------------------------------------------
def fig01(root: Path, out: Path) -> Path:
    """
    FIG01 — Study design and analysis architecture.

    Visual workflow:
        Participant Cohort
            ↓
        Quality Assurance & EEG Preprocessing
            ↓
        Feature / Signal Windowing
            ↓
        ML / DL Classification
            ↓
        Downstream Evaluation
            ↓
        Statistical Analysis

    No inferential statistics are calculated here.
    """

    # ------------------------------------------------------------------
    # Audited BLOCK04 window configuration
    # ------------------------------------------------------------------
    b04 = locate_block04_final_source(root)
    if b04 is None or not b04.exists():
        raise RuntimeError(
            "FIG01 requires directly readable BLOCK04 final W/O configuration."
        )

    text = b04.read_text(encoding="utf-8", errors="replace")
    m_samples = re.search(
        r"\bFINAL_WINDOW_SAMPLES\s*=\s*(\d+)",
        text,
    )
    m_overlap = re.search(
        r"\bFINAL_OVERLAP_PERCENT\s*=\s*(\d+(?:\.\d+)?)",
        text,
    )
    if not m_samples or not m_overlap:
        raise RuntimeError(
            "FIG01 could not verify FINAL_WINDOW_SAMPLES and "
            "FINAL_OVERLAP_PERCENT from BLOCK04."
        )

    window_samples = int(m_samples.group(1))
    overlap_percent = float(m_overlap.group(1))
    window_seconds = window_samples / 125.0

    if float(overlap_percent).is_integer():
        overlap_text = f"{int(overlap_percent)}%"
    else:
        overlap_text = f"{overlap_percent:g}%"

    window_text = (
        "[ Feature / Signal Windowing ]\n"
        f"{window_seconds:.1f} s Window / {overlap_text} Overlap"
    )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.axis("off")

    # ------------------------------------------------------------------
    # Box positions
    # ------------------------------------------------------------------
    boxes = [
        (
            0.50,
            0.91,
            0.72,
            0.105,
            "[ Participant Cohort ]\n"
            "25 Paired Subjects (ME & MI)",
        ),
        (
            0.50,
            0.76,
            0.72,
            0.105,
            "[ Quality Assurance & EEG Preprocessing ]\n"
            "1–40 Hz, 4th-order zero-phase Butterworth"
        ),
        (
            0.50,
            0.61,
            0.72,
            0.105,
            window_text,
        ),
        (
            0.23,
            0.43,
            0.38,
            0.13,
            "[ ML Classification ]\n"
            "Subject-Dependent & 25-Fold LOSO",
        ),
        (
            0.77,
            0.43,
            0.38,
            0.13,
            "[ DL Classification ]\n"
            "Subject-Dependent & 25-Fold LOSO",
        ),
        (
            0.50,
            0.235,
            0.72,
            0.145,
            "[ Downstream Evaluation ]\n"
            "• ME–MI Transfer & Calibration\n"
            "• Representational Similarity (RSA)\n"
            "• Explainable AI (XAI)\n"
            "• Robustness & Sensitivity",
        ),
        (
            0.50,
            0.065,
            0.72,
            0.105,
            "[ Statistical Analysis ]\n"
            "Frozen Metrics & Model Evaluation",
        ),
    ]

    # ------------------------------------------------------------------
    # Draw boxes
    # ------------------------------------------------------------------
    for x, y, width, height, text in boxes:
        patch = FancyBboxPatch(
            (
                x - width / 2,
                y - height / 2,
            ),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.8,
            edgecolor="black",
            facecolor="white",
        )

        ax.add_patch(patch)

        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="normal",
            linespacing=1.45,
        )

    # ------------------------------------------------------------------
    # Vertical workflow arrows
    # ------------------------------------------------------------------
    vertical_arrows = [
    (0.50, 0.855, 0.50, 0.815),
    (0.50, 0.705, 0.50, 0.665),
    (0.50, 0.555, 0.50, 0.505),
    (0.50, 0.162, 0.50, 0.118),
    ]

    for x1, y1, x2, y2 in vertical_arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                linewidth=1.8,
                color="black",
                shrinkA=0,
                shrinkB=0,
            ),
        )

    # ------------------------------------------------------------------
    # Split arrows: Windowing → ML / DL
    # ------------------------------------------------------------------
    ax.annotate(
        "",
        xy=(0.23, 0.495),
        xytext=(0.42, 0.555),
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=1.8,
            color="black",
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0",
        ),
    )

    ax.annotate(
        "",
        xy=(0.77, 0.495),
        xytext=(0.58, 0.555),
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=1.8,
            color="black",
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0",
        ),
    )

    # ------------------------------------------------------------------
    # Merge arrows: ML / DL → Downstream Evaluation
    # ------------------------------------------------------------------
    ax.annotate(
        "",
        xy=(0.50, 0.310),
        xytext=(0.23, 0.365),
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=1.8,
            color="black",
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0",
        ),
    )

    ax.annotate(
        "",
        xy=(0.50, 0.310),
        xytext=(0.77, 0.365),
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=1.8,
            color="black",
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0",
        ),
    )

    # ------------------------------------------------------------------
    # No embedded figure title.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Save using the existing BLOCK16 save mechanism
    # ------------------------------------------------------------------
    return save_figure(fig, out, "FIG01")

def fig02(root: Path, out: Path) -> Path:
    """
    FIG02 — Paired cohort, recording structure, and computational EEG representation.

    Scientific scope
    ----------------
    FIG02 describes only the structure of the paired EEG dataset and its
    computational window geometry.

    It shows:
        • 25 paired ME/MI subjects
        • 7 gestures × 3 recording sets
        • 525 ME recordings
        • 525 MI recordings
        • 1,050 paired recordings
        • 13 EEG channels
        • 125 Hz sampling rate
        • 5,625 samples / 45 s nominal recording
        • audited BLOCK04 window length, overlap, step, and window count

    FIG02 intentionally does not repeat:
        • preprocessing
        • ML/DL classification
        • transfer analysis
        • RSA
        • XAI
        • robustness
        • statistical analysis

    No inferential statistics are calculated here.
    """

    # ==============================================================
    # 1. AUDITED BLOCK04 WINDOW CONFIGURATION
    # ==============================================================

    b04 = locate_block04_final_source(root)

    W = None
    O = None

    if b04 is not None and b04.exists():

        txt = b04.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        m1 = re.search(
            r"FINAL_WINDOW_SAMPLES\s*=\s*(\d+)",
            txt,
        )

        m2 = re.search(
            r"FINAL_OVERLAP_PERCENT\s*=\s*(\d+)",
            txt,
        )

        if m1:
            W = int(
                m1.group(1)
            )

        if m2:
            O = int(
                m2.group(1)
            )

    if W is None or O is None:
        raise RuntimeError(
            "FIG02 requires directly readable BLOCK04 "
            "final W/O configuration."
        )

    # ==============================================================
    # 2. FROZEN EEG SIGNAL SPECIFICATION
    # ==============================================================

    EEG_CHANNELS = 13
    SAMPLING_RATE = 125
    NOMINAL_SAMPLES = 5625
    NOMINAL_DURATION = 45.0

    # ==============================================================
    # 3. COMPUTATIONAL WINDOW GEOMETRY
    # ==============================================================

    step = int(
        round(
            W * (
                1.0 - O / 100.0
            )
        )
    )

    if step <= 0:
        raise RuntimeError(
            f"FIG02 invalid window step: W={W}, O={O}"
        )

    overlap_samples = W - step

    n_windows = int(
        math.floor(
            (
                NOMINAL_SAMPLES - W
            )
            / step
        )
        + 1
    )

    if n_windows <= 0:
        raise RuntimeError(
            "FIG02 calculated a non-positive number of windows: "
            f"W={W}, step={step}, "
            f"N={NOMINAL_SAMPLES}"
        )

    coverage = (
        W
        + (
            n_windows - 1
        )
        * step
    )

    window_duration = (
        W / SAMPLING_RATE
    )

    step_duration = (
        step / SAMPLING_RATE
    )

    coverage_duration = (
        coverage / SAMPLING_RATE
    )

    # ==============================================================
    # 4. FIGURE CANVAS
    # ==============================================================

    fig, ax = plt.subplots(
        figsize=(15, 9.5)
    )

    ax.axis("off")

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1,
    )

    # ==============================================================
    # 5. TOP COHORT SUMMARY
    # ==============================================================

    cohort_box = FancyBboxPatch(
        (
            0.18,
            0.80,
        ),
        0.64,
        0.12,
        boxstyle=(
            "round,pad=0.012,"
            "rounding_size=0.015"
        ),
        linewidth=1.8,
        edgecolor="black",
        facecolor="white",
    )

    ax.add_patch(
        cohort_box
    )

    # Main cohort information.
    ax.text(
        0.47,
        0.875,
        "25 paired subjects",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )

    # Gesture/set structure.
    ax.text(
        0.47,
        0.835,
        "7 gestures (1–7) × 3 sets (A/B/C)",
        ha="center",
        va="center",
        fontsize=12.5,
    )

    # Total recording count.
    ax.plot(
        [0.675, 0.675],
        [0.815, 0.905],
        linewidth=1.0,
    )

    ax.text(
        0.75,
        0.86,
        "1,050 paired\nrecordings",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    # ==============================================================
    # 6. ME / MI BRANCH ARROWS
    # ==============================================================

    ax.annotate(
        "",
        xy=(0.27, 0.675),
        xytext=(0.42, 0.80),
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=1.6,
            color="black",
        ),
    )

    ax.annotate(
        "",
        xy=(0.73, 0.675),
        xytext=(0.58, 0.80),
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=1.6,
            color="black",
        ),
    )

    # ==============================================================
    # 7. MOTOR EXECUTION BOX
    # ==============================================================

    me_box = FancyBboxPatch(
        (
            0.05,
            0.565,
        ),
        0.42,
        0.115,
        boxstyle=(
            "round,pad=0.012,"
            "rounding_size=0.015"
        ),
        linewidth=1.8,
        edgecolor="black",
        facecolor="white",
    )

    ax.add_patch(
        me_box
    )

    ax.text(
        0.26,
        0.625,
        "Motor Execution (ME)",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )

    ax.text(
        0.26,
        0.592,
        "525 recordings",
        ha="center",
        va="center",
        fontsize=12.5,
    )

    # ==============================================================
    # 8. MOTOR IMAGERY BOX
    # ==============================================================

    mi_box = FancyBboxPatch(
        (
            0.53,
            0.565,
        ),
        0.42,
        0.115,
        boxstyle=(
            "round,pad=0.012,"
            "rounding_size=0.015"
        ),
        linewidth=1.8,
        edgecolor="black",
        facecolor="white",
    )

    ax.add_patch(
        mi_box
    )

    ax.text(
        0.74,
        0.625,
        "Motor Imagery (MI)",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )

    ax.text(
        0.74,
        0.592,
        "525 recordings",
        ha="center",
        va="center",
        fontsize=12.5,
    )

    # ==============================================================
    # 9. LOWER LEFT — CONTINUOUS RECORDING AND WINDOWS
    # ==============================================================

    ax.text(
        0.285,
        0.505,
        "Continuous EEG recording and computational windows",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.285,
        0.488,
        "Schematic; geometry derived from the audited BLOCK04 configuration.",
        ha="center",
        va="center",
        fontsize=8.5,
    )

    # --------------------------------------------------------------
    # Representative recording duration
    # --------------------------------------------------------------

    ax.text(
        0.285,
        0.472,
        f"Continuous recording: {NOMINAL_SAMPLES:,} samples ({NOMINAL_DURATION:g} s)",
        ha="center",
        va="center",
        fontsize=11.5,
    )

    # --------------------------------------------------------------
    # Representative EEG waveform
    # --------------------------------------------------------------

    waveform_x = np.linspace(
        0.055,
        0.515,
        600,
    )

    waveform_y = (
        0.395
        + 0.014
        * np.sin(
            np.linspace(
                0,
                45 * np.pi,
                600,
            )
        )
        + 0.006
        * np.sin(
            np.linspace(
                0,
                121 * np.pi,
                600,
            )
        )
        + 0.003
        * np.sin(
            np.linspace(
                0,
                237 * np.pi,
                600,
            )
        )
    )

    ax.plot(
        waveform_x,
        waveform_y,
        linewidth=1.0,
    )

    # Recording boundaries.
    ax.plot(
        [
            0.055,
            0.055,
        ],
        [
            0.365,
            0.425,
        ],
        linewidth=1.0,
    )

    ax.plot(
        [
            0.515,
            0.515,
        ],
        [
            0.365,
            0.425,
        ],
        linewidth=1.0,
    )

    # Recording-duration arrow.
    ax.annotate(
        "",
        xy=(0.515, 0.44),
        xytext=(0.055, 0.44),
        arrowprops=dict(
            arrowstyle="<->",
            linewidth=1.0,
        ),
    )

    ax.text(
        0.285,
        0.446,
        "45-s recording span",
        ha="center",
        va="bottom",
        fontsize=9.0,
    )

    # --------------------------------------------------------------
    # Window representation
    # --------------------------------------------------------------

    window_y = 0.315
    window_height = 0.055

    # Schematic display scale.
    x_positions = [
        0.055,
        0.145,
        0.235,
    ]

    window_width = 0.135

    for i, x in enumerate(
        x_positions
    ):

        ax.add_patch(
            Rectangle(
                (
                    x,
                    window_y,
                ),
                window_width,
                window_height,
                fill=False,
                linewidth=1.3,
            )
        )

        ax.text(
            x + window_width / 2,
            window_y + window_height + 0.012,
            f"Window {i + 1}",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    # Ellipsis.
    ax.text(
        0.405,
        window_y + window_height / 2,
        "⋯",
        ha="center",
        va="center",
        fontsize=18,
    )

    # Final window.
    final_x = 0.445

    ax.add_patch(
        Rectangle(
            (
                final_x,
                window_y,
            ),
            window_width,
            window_height,
            fill=False,
            linewidth=1.3,
        )
    )

    ax.text(
        final_x + window_width / 2,
        window_y + window_height + 0.012,
        f"Window {n_windows}",
        ha="center",
        va="bottom",
        fontsize=9.5,
    )

    # --------------------------------------------------------------
    # Window geometry
    # --------------------------------------------------------------

    # Window length.
    ax.annotate(
        "",
        xy=(
            0.055 + window_width,
            0.270,
        ),
        xytext=(
            0.055,
            0.270,
        ),
        arrowprops=dict(
            arrowstyle="<->",
            linewidth=1.0,
        ),
    )

    ax.text(
        0.055 + window_width / 2,
        0.250,
        f"W = {W} samples ({window_duration:.1f} s)",
        ha="center",
        va="top",
        fontsize=9.5,
    )

    # Step.
    step_display = 0.09

    ax.annotate(
        "",
        xy=(
            0.055 + step_display,
            0.215,
        ),
        xytext=(
            0.055,
            0.215,
        ),
        arrowprops=dict(
            arrowstyle="<->",
            linewidth=1.0,
        ),
    )

    ax.text(
        0.055 + step_display / 2,
        0.195,
        f"Step = {step} samples ({step_duration:.1f} s)",
        ha="center",
        va="top",
        fontsize=9.5,
    )

    # Overlap.
    overlap_x1 = (
        0.055 + step_display
    )

    overlap_x2 = (
        0.055 + window_width
    )

    ax.annotate(
        "",
        xy=(
            overlap_x2,
            0.165,
        ),
        xytext=(
            overlap_x1,
            0.165,
        ),
        arrowprops=dict(
            arrowstyle="<->",
            linewidth=1.0,
        ),
    )

    ax.text(
        (
            overlap_x1
            + overlap_x2
        ) / 2,
        0.145,
        f"Overlap = {O}% ({overlap_samples} samples)",
        ha="center",
        va="top",
        fontsize=9.5,
    )

    # Window count.
    ax.text(
        0.285,
        0.095,
        f"{n_windows} windows per nominal recording",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
    )

    ax.text(
        0.285,
        0.060,
        f"{coverage:,} samples covered "
        f"({coverage_duration:.1f} s)",
        ha="center",
        va="center",
        fontsize=10,
    )

    # ==============================================================
    # 10. LOWER RIGHT — SIGNAL SPECIFICATION
    # ==============================================================

    ax.text(
        0.755,
        0.505,
        "Signal specification",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    table_x = 0.555
    table_top = 0.455
    table_width = 0.405
    label_width = 0.225
    row_height = 0.055

    # Only the fundamental acquisition specification is retained here.
    # Window geometry is already shown graphically on the left.
    signal_rows = [
        (
            "EEG channels",
            f"{EEG_CHANNELS}",
        ),
        (
            "Sampling rate",
            f"{SAMPLING_RATE} Hz",
        ),
        (
            "Nominal samples",
            f"{NOMINAL_SAMPLES:,}",
        ),
        (
            "Nominal duration",
            f"{NOMINAL_DURATION:g} s",
        ),
    ]

    table_height = (
        row_height
        * len(signal_rows)
    )

    # Outer border.
    ax.add_patch(
        Rectangle(
            (
                table_x,
                table_top - table_height,
            ),
            table_width,
            table_height,
            fill=False,
            linewidth=1.0,
        )
    )

    # Column separator.
    ax.plot(
        [
            table_x + label_width,
            table_x + label_width,
        ],
        [
            table_top,
            table_top - table_height,
        ],
        linewidth=0.8,
    )

    # Rows.
    for i, (label, value) in enumerate(
        signal_rows
    ):

        y_top = (
            table_top
            - i * row_height
        )

        if i > 0:
            ax.plot(
                [
                    table_x,
                    table_x + table_width,
                ],
                [
                    y_top,
                    y_top,
                ],
                linewidth=0.6,
            )

        ax.text(
            table_x + 0.010,
            y_top - row_height / 2,
            label,
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight=(
                "bold"
                if i == 0
                else "normal"
            ),
        )

        ax.text(
            table_x
            + label_width
            + 0.010,
            y_top - row_height / 2,
            value,
            ha="left",
            va="center",
            fontsize=9.5,
        )

    # ==============================================================
    # 11. NO EMBEDDED FIGURE TITLE
    # ==============================================================

    # The formal figure title and scientific explanation belong in
    # the manuscript caption, not inside the exported figure.

    # ==============================================================
    # 12. SAVE THROUGH EXISTING BLOCK16 MECHANISM
    # ==============================================================

    return save_figure(
        fig,
        out,
        "FIG02",
    )
    
def fig03(root: Path, out: Path) -> Path:
    """
    FIG03 — Subject-dependent versus cross-subject decoding performance.

    Scientific scope
    ----------------
    Panel A:
        Motor Execution (ME), subject-dependent Macro-F1 versus LOSO
        Macro-F1 for all stored subject × model evaluations.

    Panel B:
        Motor Imagery (MI), subject-dependent Macro-F1 versus LOSO
        Macro-F1 for all stored subject × model evaluations.

    Panel C:
        Model-level mean SD–LOSO Macro-F1 gap for ME and MI, using the
        already-stored BLOCK15 summary statistics and stored 95% CIs.

    Important:
        • No new inferential statistics are calculated.
        • No values are reconstructed from predictions.
        • Subject-level observations come directly from the stored
          SD_LOSO_subject_gap artifact.
        • Model-level means/CIs come directly from the stored
          gap_summary artifact.
        • Macro-F1 is the only metric used in this figure.
        • The 25-subject cohort and 10-model structure are explicitly
          distinguished from the 250 plotted subject × model evaluations
          per condition.
    """

    # ==============================================================
    # 1. LOAD AUTHORITATIVE BLOCK15 SUBJECT-LEVEL GAP ARTIFACT
    # ==============================================================

    p = source_path(
        root,
        ARTIFACTS["BLOCK15"]["gap"],
    )

    d = load_csv(p)

    # Exact stored columns
    subj = find_col(d, "subject")
    cond = find_col(d, "condition")
    model = find_col(d, "model", required=False)
    metric = find_col(d, "metric", required=False)

    gapc = next(
        (
            c
            for c in d.columns
            if norm_col(c)
            in {
                "gap",
                "g_i",
                "sd_loso_gap",
                "sd_loso_difference",
            }
        ),
        None,
    )

    sdc = next(
        (
            c
            for c in d.columns
            if norm_col(c)
            in {
                "a_sd",
                "sd",
                "sd_macro_f1",
            }
        ),
        None,
    )

    loc = next(
        (
            c
            for c in d.columns
            if norm_col(c)
            in {
                "a_loso",
                "loso",
                "loso_macro_f1",
                "cross_subject_macro_f1",
            }
        ),
        None,
    )

    if not subj or not cond or not gapc or not sdc or not loc:
        raise RuntimeError(
            "FIG03 requires stored subject, condition, A_SD, "
            f"A_LOSO and gap columns; columns={list(d.columns)}"
        )

    d = d.copy()

    # ==============================================================
    # 2. RESTRICT TO STORED MACRO-F1 OBSERVATIONS
    # ==============================================================

    if metric:
        metric_norm = (
            d[metric]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("_", " ", regex=False)
            .str.replace("-", " ", regex=False)
        )

        d = d[
            metric_norm.eq("macro f1")
        ].copy()

    if d.empty:
        raise RuntimeError(
            "FIG03: no stored Macro-F1 rows found."
        )

    # ==============================================================
    # 3. STANDARDIZE STORED VALUES
    # ==============================================================

    d["subject_std"] = pd.to_numeric(
        d[subj],
        errors="coerce",
    )

    d["condition_std"] = d[cond].map(
        normalize_condition
    )

    d["SD"] = pd.to_numeric(
        d[sdc],
        errors="coerce",
    )

    d["LOSO"] = pd.to_numeric(
        d[loc],
        errors="coerce",
    )

    d["G_i"] = pd.to_numeric(
        d[gapc],
        errors="coerce",
    )

    if model:
        d["model_std"] = d[model].map(
            normalize_model
        )
    else:
        d["model_std"] = ""

    # Keep only finite observations required by the figure.
    d = d[
        d["subject_std"].notna()
        & d["SD"].notna()
        & d["LOSO"].notna()
        & d["G_i"].notna()
        & d["condition_std"].isin(["ME", "MI"])
    ].copy()

    if d.empty:
        raise RuntimeError(
            "FIG03: no finite Macro-F1 subject-level observations remain."
        )

    # ==============================================================
    # 4. VERIFY THE EXPECTED PAIRED COHORT STRUCTURE
    # ==============================================================

    expected_subjects = set(range(1, 26))

    observed_subjects = set(
        d["subject_std"]
        .astype(int)
        .unique()
        .tolist()
    )

    if observed_subjects != expected_subjects:
        raise RuntimeError(
            "FIG03 requires subjects 1–25. "
            f"Observed subjects={sorted(observed_subjects)}"
        )

    n_subjects = d["subject_std"].nunique()

    n_models = (
        d["model_std"].nunique()
        if model
        else None
    )

    # Verify each condition contains the expected
    # 25 subjects × 10 models = 250 Macro-F1 evaluations.
    for condition in ["ME", "MI"]:

        dc = d[
            d["condition_std"].eq(condition)
        ]

        if dc.empty:
            raise RuntimeError(
                f"FIG03: no Macro-F1 observations for {condition}."
            )

        if n_models is not None:
            expected_n = 25 * n_models

            if len(dc) != expected_n:
                raise RuntimeError(
                    f"FIG03: {condition} contains {len(dc)} "
                    f"Macro-F1 observations; expected "
                    f"{expected_n} = 25 subjects × {n_models} models."
                )

    # ==============================================================
    # 5. LOCATE AUTHORITATIVE BLOCK15 MODEL-LEVEL SUMMARY
    # ==============================================================

    # The subject-level artifact is authoritative for Panels A/B.
    # The separate gap_summary artifact is authoritative for the
    # model-level mean gap and stored 95% CI in Panel C.

    summary_candidates = [
        "gap_summary.csv",
        "GAP_SUMMARY.csv",
        "BLOCK15_gap_summary.csv",
        "BLOCK15_GAP_SUMMARY.csv",
    ]

    summary_path = None

    # First look beside the authoritative gap artifact.
    for name in summary_candidates:

        candidate = p.parent / name

        if candidate.exists():
            summary_path = candidate
            break

    # If not beside it, search the Paper 2 root deterministically.
    if summary_path is None:

        for name in summary_candidates:

            matches = sorted(
                root.rglob(name)
            )

            if matches:
                summary_path = matches[0]
                break

    if summary_path is None:
        raise RuntimeError(
            "FIG03 requires the authoritative BLOCK15 "
            "gap_summary.csv artifact for Panel C."
        )

    summary = load_csv(summary_path)

    required_summary = {
        "model",
        "condition",
        "metric",
        "n",
        "mean",
        "ci95_low",
        "ci95_high",
    }

    missing_summary = [
        c
        for c in required_summary
        if c not in summary.columns
    ]

    if missing_summary:
        raise RuntimeError(
            "FIG03 gap_summary.csv is missing required columns: "
            f"{missing_summary}; columns={list(summary.columns)}"
        )

    # ==============================================================
    # 6. RESTRICT SUMMARY TO MACRO-F1
    # ==============================================================

    summary = summary.copy()

    summary_metric_norm = (
        summary["metric"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
        .str.replace("-", " ", regex=False)
    )

    summary = summary[
        summary_metric_norm.eq("macro f1")
    ].copy()

    if summary.empty:
        raise RuntimeError(
            "FIG03: gap_summary.csv contains no Macro-F1 rows."
        )

    summary["condition_std"] = summary[
        "condition"
    ].map(normalize_condition)

    summary["model_std"] = summary[
        "model"
    ].map(normalize_model)

    summary["n_std"] = pd.to_numeric(
        summary["n"],
        errors="coerce",
    )

    summary["mean_gap"] = pd.to_numeric(
        summary["mean"],
        errors="coerce",
    )

    summary["ci_low"] = pd.to_numeric(
        summary["ci95_low"],
        errors="coerce",
    )

    summary["ci_high"] = pd.to_numeric(
        summary["ci95_high"],
        errors="coerce",
    )

    summary = summary[
        summary["condition_std"].isin(["ME", "MI"])
        & summary["mean_gap"].notna()
        & summary["ci_low"].notna()
        & summary["ci_high"].notna()
    ].copy()

    if summary.empty:
        raise RuntimeError(
            "FIG03: no finite stored Macro-F1 model-level "
            "gap summaries remain."
        )

    # ==============================================================
    # 7. VERIFY MODEL-LEVEL SUMMARY
    # ==============================================================

    if n_models is not None:

        expected_models = set(
            d["model_std"].dropna().unique()
        )

        for condition in ["ME", "MI"]:

            sc = summary[
                summary["condition_std"].eq(condition)
            ]

            observed_models = set(
                sc["model_std"].dropna().unique()
            )

            if observed_models != expected_models:
                raise RuntimeError(
                    f"FIG03: model mismatch for {condition}. "
                    f"Expected={sorted(expected_models)}, "
                    f"observed={sorted(observed_models)}"
                )

            if not np.all(
                sc["n_std"].eq(25)
            ):
                raise RuntimeError(
                    f"FIG03: stored model-level Macro-F1 "
                    f"summary for {condition} does not have n=25 "
                    f"for every model."
                )

    # ==============================================================
    # 8. MODEL ORDER
    # ==============================================================

    canonical_model_order = [
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

    # Keep only models actually present, but preserve canonical order.
    available_models = set(
        summary["model_std"].dropna().unique()
    )

    model_order = [
        m
        for m in canonical_model_order
        if m in available_models
    ]

    if not model_order:
        model_order = sorted(
            available_models
        )

    # ==============================================================
    # 9. FIGURE
    # ==============================================================

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(17.4, 5.4),
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.20,
        top=0.92,
        wspace=0.16,
    )

    # ==============================================================
    # 10. PANELS A & B
    #     SUBJECT-DEPENDENT VS LOSO
    # ==============================================================

    for ci, condition in enumerate(["ME", "MI"]):

        ax = axs[ci]

        x = d[
            d["condition_std"].eq(condition)
        ].copy()

        if x.empty:
            raise RuntimeError(
                f"FIG03: no observations for {condition}."
            )

        # All stored subject × model evaluations.
        ax.scatter(
            x["SD"],
            x["LOSO"],
            s=28,
            alpha=0.60,
        )

        # Fixed identity line across the complete Macro-F1 domain.
        ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            linewidth=0.9,
            alpha=0.65,
        )

        ax.set_xlim(
            0.0,
            1.0,
        )

        ax.set_ylim(
            0.0,
            1.0,
        )

        ax.set_xlabel(
            "Subject-dependent Macro-F1"
        )

        ax.set_ylabel(
            "LOSO Macro-F1"
        )

        panel_label = (
            "(A) Motor Execution (ME)"
            if condition == "ME"
            else "(B) Motor Imagery (MI)"
        )

        ax.set_title(
            panel_label,
            loc="left",
            fontweight="bold",
            fontsize=11,
        )

        # No in-panel cohort annotation is used here.
        # The manuscript caption can state the 25-subject / 10-model
        # structure without competing with the plotted observations.
        style_axes(ax)

    # ==============================================================
    # 11. PANEL C
    #     MODEL-LEVEL MEAN SD–LOSO GAP
    # ==============================================================

    ax = axs[2]

    plot_summary = summary[
        summary["model_std"].isin(model_order)
    ].copy()

    # Arrange in canonical model order.
    plot_summary["model_order"] = pd.Categorical(
        plot_summary["model_std"],
        categories=model_order,
        ordered=True,
    )

    plot_summary = plot_summary.sort_values(
        [
            "model_order",
            "condition_std",
        ]
    )

    # Numerical model positions.
    xpos = np.arange(
        len(model_order)
    )

    offsets = {
        "ME": -0.16,
        "MI": 0.16,
    }

    for condition, marker in [
        ("ME", "o"),
        ("MI", "s"),
    ]:

        sc = plot_summary[
            plot_summary["condition_std"].eq(condition)
        ].copy()

        sc = sc.set_index(
            "model_std"
        ).reindex(
            model_order
        ).reset_index()

        if sc["mean_gap"].isna().any():
            raise RuntimeError(
                f"FIG03: incomplete stored model-level "
                f"gap summary for {condition}."
            )

        xvals = (
            xpos
            + offsets[condition]
        )

        yvals = sc[
            "mean_gap"
        ].to_numpy()

        low = sc[
            "ci_low"
        ].to_numpy()

        high = sc[
            "ci_high"
        ].to_numpy()

        lower_err = yvals - low
        upper_err = high - yvals

        # Stored 95% CI only; no CI is recalculated here.
        ax.errorbar(
            xvals,
            yvals,
            yerr=np.vstack(
                [
                    lower_err,
                    upper_err,
                ]
            ),
            fmt=marker,
            markersize=5,
            capsize=3,
            linewidth=1.0,
            label=condition,
        )

    # Zero gap reference.
    ax.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
        alpha=0.7,
    )

    ax.set_xticks(
        xpos
    )

    fig03_tick_labels = {
        "RandomForest": "Random\nForest",
        "XGBoost": "XGBoost",
        "ShallowConvNet": "Shallow\nConvNet",
        "DeepConvNet": "Deep\nConvNet",
        "CNN-BiLSTM": "CNN–\nBiLSTM",
        "CNN-BiLSTM-Attention": "CNN–\nBiLSTM–\nAttention",
    }

    ax.set_xticklabels(
        [fig03_tick_labels.get(m, m) for m in model_order],
        rotation=0,
        ha="center",
        va="top",
        fontsize=7.2,
    )

    ax.set_ylabel(
        r"Mean SD–LOSO Macro-F1 gap "
        r"($G=F1_{SD}-F1_{LOSO}$)"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_title(
        "(C) Model-level SD–LOSO gap",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )

    ax.legend(
        frameon=False,
        title=None,
    )

    # Dynamic symmetric limits around zero, based only on
    # already-stored mean/CI values.
    finite_low = plot_summary[
        "ci_low"
    ].to_numpy()

    finite_high = plot_summary[
        "ci_high"
    ].to_numpy()

    finite_values = np.concatenate(
        [
            finite_low[np.isfinite(finite_low)],
            finite_high[np.isfinite(finite_high)],
        ]
    )

    if finite_values.size:

        max_abs = float(
            np.max(
                np.abs(finite_values)
            )
        )

        if max_abs > 0:
            ylim = max_abs * 1.12

            ax.set_ylim(
                -ylim,
                ylim,
            )

    style_axes(ax)
    # ==============================================================
    # 12. FINAL LAYOUT
    # ==============================================================

    fig.subplots_adjust(
        left=0.045,
        right=0.985,
        bottom=0.24,
        top=0.93,
        wspace=0.16,
    )

    # No burned-in global figure title.
    # The manuscript caption should provide the formal figure title.

    # ==============================================================
    # 13. SAVE THROUGH EXISTING BLOCK16 MECHANISM
    # ==============================================================

    return save_figure(
        fig,
        out,
        "FIG03",
    )

def fig04(root: Path, out: Path) -> Path:
    """
    FIG04 — Paired ME–MI cross-subject decoding performance.

    Scientific scope
    ----------------
    Panel A:
        Model-level paired comparison of ME and MI LOSO Macro-F1.

    Panel B:
        Subject-level paired ME − MI differences for each model.

    Panel C:
        Median paired ME − MI difference by model.

    The figure uses the stored subject-level performance values only.
    No new hypothesis tests, p-values, confidence intervals, or
    inferential statistics are calculated.
    """

    # ==============================================================
    # 1. LOAD STORED SUBJECT-LEVEL PERFORMANCE
    # ==============================================================

    d = read_subject_performance(root)

    # FIG04 uses Macro-F1 only. The authoritative BLOCK15 subject-level
    # artifact contains multiple metrics per subject/model/condition, so
    # filter the metric before enforcing the one-row-per-pair structure.
    if "metric" not in d.columns:
        raise RuntimeError(
            "FIG04: metric column is required to isolate stored Macro-F1."
        )

    metric_norm = (
        d["metric"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
        .str.replace("-", " ", regex=False)
    )
    d = d[metric_norm.eq("macro f1")].copy()

    if d.empty:
        raise RuntimeError(
            "FIG04: no stored Macro-F1 rows found in the subject-level artifact."
        )

    # Keep only the frozen model set.
    model_order = [
        m
        for m in (
            CONFIG["models_ml"] +
            CONFIG["models_dl"]
        )
        if m in set(d["model"])
    ]

    if not model_order:
        raise RuntimeError(
            "FIG04: no configured models are present in the "
            "subject-level performance artifact."
        )

    d = d[
        d["model"].isin(model_order)
    ].copy()

    # --------------------------------------------------------------
    # Require branch information so LOSO is unambiguous.
    # --------------------------------------------------------------
    if "branch" not in d.columns:
        raise RuntimeError(
            "FIG04: branch column is required to distinguish "
            "subject-dependent from LOSO performance."
        )

    d["branch"] = (
        d["branch"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # FIG04 is specifically the cross-subject / LOSO comparison.
    d = d[
        d["branch"].eq("LOSO")
    ].copy()

    if d.empty:
        raise RuntimeError(
            "FIG04: no LOSO subject-level performance rows."
        )

    # ==============================================================
    # 2. STANDARDIZE REQUIRED FIELDS
    # ==============================================================

    d["condition"] = (
        d["condition"]
        .map(normalize_condition)
    )

    d["value"] = pd.to_numeric(
        d["value"],
        errors="coerce",
    )

    d["subject"] = pd.to_numeric(
        d["subject"],
        errors="coerce",
    )

    d = d[
        d["condition"].isin(["ME", "MI"])
        & d["value"].notna()
        & d["subject"].notna()
    ].copy()

    if d.empty:
        raise RuntimeError(
            "FIG04: no finite ME/MI LOSO Macro-F1 observations."
        )

    # Exact duplicate exports are removed only when their stored value is identical.
    pair_key = ["subject", "model", "condition"]
    dup = d[d.duplicated(pair_key, keep=False)].copy()
    if not dup.empty:
        conflicts = dup.groupby(pair_key)["value"].nunique(dropna=True)
        conflicting_keys = conflicts[conflicts > 1]
        if not conflicting_keys.empty:
            raise RuntimeError(
                "FIG04: duplicate subject/model/condition rows contain conflicting "
                f"Macro-F1 values: {conflicting_keys.index.tolist()}"
            )
        d = d.drop_duplicates(pair_key, keep="first").copy()

    # ==============================================================
    # 3. VERIFY PAIRED 25-SUBJECT STRUCTURE
    # ==============================================================

    observed_subjects = sorted(
        d["subject"]
        .astype(int)
        .unique()
    )

    if observed_subjects != list(range(1, 26)):
        raise RuntimeError(
            "FIG04 requires subjects 1–25. "
            f"Observed subjects={observed_subjects}"
        )

    # Each condition/model should have 25 subject observations.
    for model_name in model_order:

        for condition in ["ME", "MI"]:

            x = d[
                d["model"].eq(model_name)
                & d["condition"].eq(condition)
            ]

            n_subjects = x["subject"].nunique()

            if n_subjects != 25:
                raise RuntimeError(
                    f"FIG04: {model_name} / {condition} "
                    f"contains {n_subjects} subjects; expected 25."
                )

    # ==============================================================
    # 4. BUILD EXPLICIT PAIRED SUBJECT TABLE
    # ==============================================================

    me = d[
        d["condition"].eq("ME")
    ][
        ["subject", "model", "value"]
    ].rename(
        columns={"value": "ME"}
    )

    mi = d[
        d["condition"].eq("MI")
    ][
        ["subject", "model", "value"]
    ].rename(
        columns={"value": "MI"}
    )

    pair = me.merge(
        mi,
        on=["subject", "model"],
        how="inner",
        validate="one_to_one",
    )

    if pair.empty:
        raise RuntimeError(
            "FIG04: no paired ME/MI subject-model observations."
        )

    pair["difference"] = (
        pair["ME"] - pair["MI"]
    )

    # Verify complete paired structure.
    expected_pairs = 25 * len(model_order)

    if len(pair) != expected_pairs:
        raise RuntimeError(
            f"FIG04: found {len(pair)} paired subject-model "
            f"observations; expected {expected_pairs}."
        )

    # ==============================================================
    # 5. MODEL-LEVEL DESCRIPTIVE SUMMARIES
    # ==============================================================

    model_me = (
        pair.groupby("model")["ME"]
        .median()
        .reindex(model_order)
    )

    model_mi = (
        pair.groupby("model")["MI"]
        .median()
        .reindex(model_order)
    )

    model_difference = (
        pair.groupby("model")["difference"]
        .median()
        .reindex(model_order)
    )

    # ==============================================================
    # 6. FIGURE
    # ==============================================================

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(18.0, 5.8),
    )

    # FIG04-local compact labels only; other figures keep their existing labels.
    fig04_tick_labels = {
        "RandomForest": "Random\nForest",
        "XGBoost": "XGBoost",
        "ShallowConvNet": "Shallow\nConvNet",
        "DeepConvNet": "Deep\nConvNet",
        "CNN-BiLSTM": "CNN–\nBiLSTM",
        "CNN-BiLSTM-Attention": "CNN–\nBiLSTM–\nAttention",
    }

    x = np.arange(
        len(model_order)
    )

    # ==============================================================
    # 7. PANEL A — PAIRED MODEL-LEVEL ME vs MI
    # ==============================================================

    ax = axs[0]

    # Connect ME and MI for every model.
    for i, model_name in enumerate(model_order):

        ax.plot(
            [i, i],
            [
                model_me.loc[model_name],
                model_mi.loc[model_name],
            ],
            linewidth=1.0,
            alpha=0.7,
        )

    ax.scatter(
        x,
        model_me.values,
        s=42,
        marker="o",
        label="ME",
        zorder=3,
    )

    ax.scatter(
        x,
        model_mi.values,
        s=42,
        marker="s",
        label="MI",
        zorder=3,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [fig04_tick_labels.get(m, m) for m in model_order],
        rotation=0,
        ha="center",
        va="top",
        fontsize=7.0,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_ylabel(
        "LOSO Macro-F1"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_title(
        "(A) LOSO Macro-F1 by condition",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )

    ax.legend(
        frameon=False,
    )

    style_axes(ax)

    # ==============================================================
    # 8. PANEL B — SUBJECT-LEVEL PAIRED DIFFERENCES
    # ==============================================================

    ax = axs[1]

    # Display every subject-level paired difference.
    for i, model_name in enumerate(model_order):

        values = pair.loc[
            pair["model"].eq(model_name),
            "difference",
        ].to_numpy()

        # Small deterministic horizontal offsets.
        offsets = np.linspace(
            -0.18,
            0.18,
            len(values),
        )

        ax.scatter(
            np.full(
                len(values),
                i,
            ) + offsets,
            values,
            s=18,
            alpha=0.55,
        )

        # Median marker.
        median_value = np.median(
            values
        )

        ax.plot(
            [
                i - 0.20,
                i + 0.20,
            ],
            [
                median_value,
                median_value,
            ],
            linewidth=2.0,
        )

    ax.axhline(
        0.0,
        linewidth=0.9,
        linestyle="--",
        alpha=0.7,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [fig04_tick_labels.get(m, m) for m in model_order],
        rotation=0,
        ha="center",
        va="top",
        fontsize=7.0,
    )

    ax.set_ylabel(
        r"Paired ME − MI Macro-F1"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_title(
        "(B) Subject-level paired ME − MI differences",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )

    ax.text(
        0.97,
        0.96,
        "25 paired subjects/model",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
    )

    style_axes(ax)

    # ==============================================================
    # 9. PANEL C — MEDIAN PAIRED DIFFERENCE
    # ==============================================================

    ax = axs[2]

    bars = ax.bar(
        x,
        model_difference.values,
        width=0.70,
    )

    # Zero reference.
    ax.axhline(
        0.0,
        linewidth=0.9,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [fig04_tick_labels.get(m, m) for m in model_order],
        rotation=0,
        ha="center",
        va="top",
        fontsize=7.0,
    )

    ax.set_ylabel(
        r"Median paired ME − MI Macro-F1"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_title(
        "(C) Median paired ME − MI difference",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )

    style_axes(ax)

    # --------------------------------------------------------------
    # Annotate the direction of the difference.
    # --------------------------------------------------------------
    y_range = (
        np.nanmax(
            np.abs(
                model_difference.values
            )
        )
        if np.isfinite(
            model_difference.values
        ).any()
        else 0.0
    )

    if y_range > 0:

        for bar, value in zip(
            bars,
            model_difference.values,
        ):

            if not np.isfinite(value):
                continue

            if value >= 0:
                y = value + 0.05 * y_range
                va = "bottom"
            else:
                y = value - 0.05 * y_range
                va = "top"

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                y,
                f"{value:.3f}",
                ha="center",
                va=va,
                fontsize=8,
            )

        limit = (
            y_range * 1.25
        )

        if limit > 0:
            ax.set_ylim(
                -limit,
                limit,
            )

    # ==============================================================
    # 10. FINAL LAYOUT
    # ==============================================================

    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.22,
        top=0.92,
        wspace=0.30,
    )

    # No global burned-in figure title.
    # The manuscript caption should provide the formal FIG04 title.

    # ==============================================================
    # 11. SAVE
    # ==============================================================

    return save_figure(
        fig,
        out,
        "FIG04",
    )

def fig05(root: Path, out: Path) -> Path:
    """
    FIG05 — Bidirectional ME–MI transfer and transfer asymmetry.

    Scientific question
    -------------------
    Does knowledge transfer between motor execution (ME) and motor
    imagery (MI) improve target-condition performance relative to the
    target-native baseline, and is transfer direction symmetric?

    Panel A:
        Stored bidirectional transfer benefit for every model:
            ME → MI
            MI → ME

        Frozen definition:
            transfer benefit =
            source→target performance − target-native performance

    Panel B:
        Stored transfer asymmetry A_S from the authoritative BLOCK15
        asymmetry artifact.

    Important
    ---------
    • BLOCK16 does not reconstruct transfer benefit.
    • BLOCK16 does not recompute transfer asymmetry.
    • BLOCK16 does not calculate new p-values, confidence intervals,
      or hypothesis tests.
    • Stored BLOCK15 summary quantities are used directly.
    """

    # ==============================================================
    # 1. LOAD AUTHORITATIVE BLOCK15 TRANSFER SUMMARY
    # ==============================================================

    p = source_path(
        root,
        ARTIFACTS["BLOCK15"]["transfer"],
    )

    d = load_csv(p)

    required = {
        "model",
        "transfer_direction",
        "metric",
        "mean",
    }

    missing = required - set(d.columns)

    if missing:
        raise RuntimeError(
            "FIG05 transfer_summary is missing required columns "
            f"{sorted(missing)}; columns={list(d.columns)}"
        )

    x = d.copy()

    # ==============================================================
    # 2. STANDARDIZE MODEL / METRIC
    # ==============================================================

    x["model_std"] = x["model"].map(
        normalize_model
    )

    x["metric_norm"] = (
        x["metric"]
        .astype(str)
        .map(norm_col)
    )

    x = x[
        x["model_std"].isin(
            CONFIG["models_ml"] +
            CONFIG["models_dl"]
        )
        & x["metric_norm"].eq("macro_f1")
    ].copy()

    if x.empty:
        raise RuntimeError(
            "FIG05: no stored Macro-F1 transfer rows found."
        )

    # ==============================================================
    # 3. NORMALIZE TRANSFER DIRECTIONS
    # ==============================================================

    def _normalize_transfer_direction(value):

        s = str(value).strip().lower()

        s = (
            s.replace(" ", "")
             .replace("-", "_")
             .replace("→", "_to_")
             .replace(">", "_to_")
        )

        aliases = {
            "me_to_mi": "ME_to_MI",
            "me2mi": "ME_to_MI",
            "meomi": "ME_to_MI",
            "execution_to_imagery": "ME_to_MI",
            "motor_execution_to_motor_imagery": "ME_to_MI",

            "mi_to_me": "MI_to_ME",
            "mi2me": "MI_to_ME",
            "miome": "MI_to_ME",
            "imagery_to_execution": "MI_to_ME",
            "motor_imagery_to_motor_execution": "MI_to_ME",
        }

        return aliases.get(s, str(value).strip())

    x["direction_std"] = (
        x["transfer_direction"]
        .map(_normalize_transfer_direction)
    )

    valid_directions = {
        "ME_to_MI",
        "MI_to_ME",
    }

    x = x[
        x["direction_std"].isin(
            valid_directions
        )
    ].copy()

    if x.empty:
        raise RuntimeError(
            "FIG05: no recognized ME→MI or MI→ME transfer directions."
        )

    # ==============================================================
    # 4. CONVERT STORED MEANS
    # ==============================================================

    x["mean_stored"] = pd.to_numeric(
        x["mean"],
        errors="coerce",
    )

    if x["mean_stored"].isna().any():
        raise RuntimeError(
            "FIG05: stored transfer means contain missing/non-numeric values."
        )

    if not np.isfinite(
        x["mean_stored"].to_numpy()
    ).all():
        raise RuntimeError(
            "FIG05: stored transfer means contain non-finite values."
        )

    # ==============================================================
    # 5. MODEL ORDER
    # ==============================================================

    configured_models = (
        CONFIG["models_ml"] +
        CONFIG["models_dl"]
    )

    order = [
        m
        for m in configured_models
        if m in set(x["model_std"])
    ]

    if not order:
        raise RuntimeError(
            "FIG05: no configured models are present "
            "in the transfer artifact."
        )

    # ==============================================================
    # 6. VALIDATE UNIQUE STORED SUMMARY PER MODEL/DIRECTION
    # ==============================================================

    counts = (
        x.groupby(
            [
                "model_std",
                "direction_std",
            ]
        )
        .size()
    )

    expected_pairs = [
        (m, direction)
        for m in order
        for direction in [
            "ME_to_MI",
            "MI_to_ME",
        ]
    ]

    for key in expected_pairs:

        count = int(
            counts.get(
                key,
                0,
            )
        )

        if count != 1:
            raise RuntimeError(
                "FIG05 requires exactly one stored Macro-F1 "
                "transfer mean per model/direction. "
                f"Found {count} rows for {key}."
            )

    # ==============================================================
    # 7. BUILD TRANSFER TABLE
    # ==============================================================

    transfer_table = (
        x.pivot(
            index="model_std",
            columns="direction_std",
            values="mean_stored",
        )
        .reindex(order)
    )

    if transfer_table[
        ["ME_to_MI", "MI_to_ME"]
    ].isna().any().any():

        raise RuntimeError(
            "FIG05: incomplete bidirectional transfer summary."
        )

    # ==============================================================
    # 8. LOAD AUTHORITATIVE BLOCK15 ASYMMETRY
    # ==============================================================

    asymmetry_path = source_path(
        root,
        ARTIFACTS["BLOCK15"]["asymmetry"],
    )

    if not asymmetry_path.is_file():
        raise RuntimeError(
            "FIG05 requires the authoritative BLOCK15 "
            "transfer-asymmetry artifact."
        )

    ad = load_csv(
        asymmetry_path
    )

    required_asymmetry = {
        "model",
        "asymmetry_mean",
    }

    missing_asymmetry = (
        required_asymmetry -
        set(ad.columns)
    )

    if missing_asymmetry:
        raise RuntimeError(
            "FIG05 asymmetry artifact is missing required "
            f"columns {sorted(missing_asymmetry)}; "
            f"columns={list(ad.columns)}"
        )

    ad = ad.copy()

    ad["model_std"] = ad[
        "model"
    ].map(normalize_model)

    ad["asymmetry_stored"] = pd.to_numeric(
        ad["asymmetry_mean"],
        errors="coerce",
    )

    ad = ad[
        ad["model_std"].isin(order)
    ].copy()

    if ad.empty:
        raise RuntimeError(
            "FIG05: no recognized model rows in "
            "the transfer-asymmetry artifact."
        )

    if ad["asymmetry_stored"].isna().any():
        raise RuntimeError(
            "FIG05: asymmetry artifact contains missing/non-numeric values."
        )

    if not np.isfinite(
        ad["asymmetry_stored"].to_numpy()
    ).all():
        raise RuntimeError(
            "FIG05: asymmetry artifact contains non-finite values."
        )

    # Exactly one stored asymmetry value per model.
    asym_counts = (
        ad.groupby("model_std")
        .size()
    )

    for model_name in order:

        count = int(
            asym_counts.get(
                model_name,
                0,
            )
        )

        if count != 1:
            raise RuntimeError(
                "FIG05 requires exactly one stored asymmetry "
                f"value for {model_name}; found {count}."
            )

    asymmetry_values = (
        ad.set_index(
            "model_std"
        )["asymmetry_stored"]
        .reindex(order)
    )

    # ==============================================================
    # 9. FIGURE
    # ==============================================================

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(15.2, 5.2),
    )

    xpos = np.arange(
        len(order)
    )

    # ==============================================================
    # 10. PANEL A — BIDIRECTIONAL TRANSFER BENEFIT
    # ==============================================================

    ax = axs[0]

    width = 0.36

    ax.bar(
        xpos - width / 2,
        transfer_table[
            "ME_to_MI"
        ].to_numpy(),
        width=width,
        label="ME → MI",
    )

    ax.bar(
        xpos + width / 2,
        transfer_table[
            "MI_to_ME"
        ].to_numpy(),
        width=width,
        label="MI → ME",
    )

    ax.axhline(
        0.0,
        linewidth=0.9,
        linestyle="--",
        alpha=0.8,
    )

    ax.set_xticks(
        xpos
    )

    fig05_labels = {
        "RandomForest": "Random\nForest",
        "XGBoost": "XGBoost",
        "ShallowConvNet": "Shallow\nConvNet",
        "DeepConvNet": "Deep\nConvNet",
        "CNN-BiLSTM": "CNN–\nBiLSTM",
        "CNN-BiLSTM-Attention": "CNN–\nBiLSTM–\nAttention",
    }

    ax.set_xticklabels(
        [fig05_labels.get(m, m) for m in order],
        rotation=0,
        ha="center",
        va="top",
        fontsize=6.8,
    )

    ax.set_ylabel(
        "Transfer benefit, Δ Macro-F1"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_title(
        "(A) Bidirectional cross-condition transfer",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )

    ax.legend(
        frameon=False,
    )

    style_axes(ax)

    # ==============================================================
    # 11. PANEL B — TRANSFER ASYMMETRY
    # ==============================================================

    ax = axs[1]

    ax.bar(
        xpos,
        asymmetry_values.to_numpy(),
        width=0.70,
    )

    ax.axhline(
        0.0,
        linewidth=0.9,
        linestyle="--",
        alpha=0.8,
    )

    ax.set_xticks(
        xpos
    )

    fig05_labels = {
        "RandomForest": "Random\nForest",
        "XGBoost": "XGBoost",
        "ShallowConvNet": "Shallow\nConvNet",
        "DeepConvNet": "Deep\nConvNet",
        "CNN-BiLSTM": "CNN–\nBiLSTM",
        "CNN-BiLSTM-Attention": "CNN–\nBiLSTM–\nAttention",
    }

    ax.set_xticklabels(
        [fig05_labels.get(m, m) for m in order],
        rotation=0,
        ha="center",
        va="top",
        fontsize=6.8,
    )

    ax.set_ylabel(
        r"Transfer asymmetry ($A_S$)"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_title(
        "(B) Transfer asymmetry",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )

    style_axes(ax)

    # ==============================================================
    # 12. FINAL LAYOUT
    # ==============================================================

    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.34,
        top=0.92,
        wspace=0.28,
    )

    # No global burned-in figure title.
    # The manuscript caption should provide the formal figure title.

    # ==============================================================
    # 13. SAVE THROUGH EXISTING BLOCK16 MECHANISM
    # ==============================================================

    return save_figure(
        fig,
        out,
        "FIG05",
    )
def fig06(root: Path, out: Path) -> Path:
    """
    FIG06 — Temporal calibration and k90.

    Scientific question
    -------------------
    How does target-condition calibration affect stored Macro-F1 across
    the frozen calibration grid, and what amount of calibration is
    required to reach the predefined k90 criterion?

    Panel A:
        ML models — complete BLOCK09 calibration grid for ME and MI.

    Panel B:
        DL models — complete BLOCK09 calibration grid for ME and MI.

    Panel C:
        Distribution of valid k90 observations.

    Scientific rules
    ----------------
    • BLOCK09 remains authoritative.
    • The complete frozen calibration grid is required.
    • Stored mean Macro-F1 values are plotted directly.
    • k90 censoring is preserved exactly.
    • NOT_REACHED and NO_POSITIVE_GAP are excluded only from the
      numerical k90 distribution, exactly as in the original logic.
    • No interpolation is performed.
    • No censored value is imputed.
    • No new inferential statistics are calculated.
    """

    # ==============================================================
    # 1. AUTHORITATIVE BLOCK09 PATHS
    # ==============================================================

    cal_path = source_path(
        root,
        "09_TEMPORAL_CALIBRATION/metrics/calibration_summary.csv",
    )

    k90_path = source_path(
        root,
        ARTIFACTS[
            "09_TEMPORAL_CALIBRATION"
        ]["k90_results"],
    )

    # ==============================================================
    # 2. LOAD BLOCK09 ARTIFACTS
    # ==============================================================

    cal = load_csv(cal_path)
    kd = load_csv(k90_path)

    # ==============================================================
    # 3. VALIDATE CALIBRATION SCHEMA
    # ==============================================================

    required_cal = {
        "condition",
        "model",
        "metric",
        "calibration_fraction",
        "mean_performance",
    }

    missing_cal = (
        required_cal -
        set(cal.columns)
    )

    if missing_cal:
        raise RuntimeError(
            "FIG06 BLOCK09 calibration_summary missing "
            f"{sorted(missing_cal)}; columns={list(cal.columns)}"
        )

    # ==============================================================
    # 4. PREPARE CALIBRATION DATA
    # ==============================================================

    cal = cal.copy()

    cal["metric_norm"] = (
        cal["metric"]
        .astype(str)
        .map(norm_col)
    )

    cal = cal[
        cal["metric_norm"].eq("macro_f1")
    ].copy()

    cal["fraction_std"] = pd.to_numeric(
        cal["calibration_fraction"],
        errors="coerce",
    )

    cal["value_std"] = pd.to_numeric(
        cal["mean_performance"],
        errors="coerce",
    )

    cal = cal[
        cal["fraction_std"].isin(
            CONFIG["calibration_grid"]
        )
    ].copy()

    if cal.empty:
        raise RuntimeError(
            "FIG06 BLOCK09 calibration_summary contains no "
            "frozen 0–90% calibration-grid rows."
        )

    # ==============================================================
    # 5. REQUIRE COMPLETE FROZEN CALIBRATION GRID
    # ==============================================================

    for key, grp in cal.groupby(
        ["model", "condition"],
        dropna=False,
    ):

        observed = sorted(
            set(
                grp["fraction_std"]
                .dropna()
                .astype(int)
            )
        )

        expected = CONFIG[
            "calibration_grid"
        ]

        if observed != expected:
            raise RuntimeError(
                "FIG06 incomplete calibration grid for "
                f"{key}: observed={observed}, "
                f"expected={expected}"
            )

    # ==============================================================
    # 6. VALIDATE K90 SCHEMA
    # ==============================================================

    required_k = {
        "subject_id",
        "condition",
        "model",
        "metric",
        "k90",
        "k90_status",
    }

    missing_k = (
        required_k -
        set(kd.columns)
    )

    if missing_k:
        raise RuntimeError(
            "FIG06 BLOCK09 k90_results missing "
            f"{sorted(missing_k)}; columns={list(kd.columns)}"
        )

    # ==============================================================
    # 7. PREPARE K90 DATA
    # ==============================================================

    kd = kd.copy()

    kd["metric_norm"] = (
        kd["metric"]
        .astype(str)
        .map(norm_col)
    )

    kd = kd[
        kd["metric_norm"].eq("macro_f1")
    ].copy()

    kd["k90_value"] = pd.to_numeric(
        kd["k90"],
        errors="coerce",
    )

    kd["k90_status_std"] = (
        kd["k90_status"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    allowed = {
        "VALID",
        "PASS",
        "NOT_REACHED",
        "NO_POSITIVE_GAP",
        "OK",
    }

    unexpected = (
        set(
            kd["k90_status_std"]
            .dropna()
            .unique()
        )
        -
        allowed
    )

    if unexpected:
        raise RuntimeError(
            "FIG06 unexpected k90 statuses: "
            f"{sorted(unexpected)}"
        )

    # ==============================================================
    # 8. PRESERVE FROZEN K90 CENSORING
    # ==============================================================

    valid = kd[
        kd["k90_value"].notna()
        &
        ~kd["k90_status_std"].isin(
            {
                "NOT_REACHED",
                "NO_POSITIVE_GAP",
            }
        )
    ].copy()

    vals = valid[
        "k90_value"
    ].to_numpy(
        dtype=float
    )

    # ==============================================================
    # 9. MODEL GROUPS
    # ==============================================================

    ml_models = [
        m
        for m in CONFIG["models_ml"]
        if m in set(
            cal["model"]
        )
    ]

    dl_models = [
        m
        for m in CONFIG["models_dl"]
        if m in set(
            cal["model"]
        )
    ]

    if not ml_models and not dl_models:
        raise RuntimeError(
            "FIG06: no configured ML/DL models found "
            "in the calibration artifact."
        )

    # ==============================================================
    # 10. FIGURE LAYOUT
    #
    # A = ML calibration
    # B = DL calibration
    # C = k90 distribution
    # ==============================================================

    fig = plt.figure(
        figsize=(15.5, 7.6),
    )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.65, 1.0],
        height_ratios=[1, 1],
        wspace=0.28,
        hspace=0.50,
    )

    ax_ml = fig.add_subplot(
        gs[0, 0]
    )

    ax_dl = fig.add_subplot(
        gs[1, 0]
    )

    ax_k90 = fig.add_subplot(
        gs[:, 1]
    )

    # ==============================================================
    # 11. CALIBRATION PLOTTING HELPER
    # ==============================================================

    def plot_calibration_panel(
        ax,
        models,
        panel_title,
    ):
        """
        Plot the complete frozen calibration grid for the supplied
        model family.

        The numerical values are the stored BLOCK09 means.
        No averaging, interpolation, or smoothing is performed.
        """

        if not models:
            ax.text(
                0.5,
                0.5,
                "No stored models",
                ha="center",
                va="center",
            )

            ax.axis("off")

            return

        # ----------------------------------------------------------
        # Plot every stored model × condition curve.
        # ----------------------------------------------------------

        for model_name in models:

            for condition in [
                "ME",
                "MI",
            ]:

                grp = cal[
                    cal["model"].eq(model_name)
                    &
                    cal["condition"].map(
                        normalize_condition
                    ).eq(condition)
                ].copy()

                if grp.empty:
                    raise RuntimeError(
                        "FIG06 missing calibration curve for "
                        f"{model_name} / {condition}."
                    )

                grp = grp.sort_values(
                    "fraction_std"
                )

                if len(grp) != len(
                    CONFIG["calibration_grid"]
                ):
                    raise RuntimeError(
                        "FIG06 incomplete calibration curve for "
                        f"{model_name} / {condition}."
                    )

                # ME and MI are distinguished by line style.
                line_style = (
                    "-"
                    if condition == "ME"
                    else "--"
                )

                ax.plot(
                    grp["fraction_std"],
                    grp["value_std"],
                    marker="o",
                    markersize=3.0,
                    linewidth=1.2,
                    linestyle=line_style,
                    color=MODEL_COLORS.get(model_name, None),
                    label=f"{model_name} / {condition}",
                )

        # ----------------------------------------------------------
        # Fixed calibration grid
        # ----------------------------------------------------------

        ax.set_xticks(
            CONFIG["calibration_grid"]
        )

        ax.set_xlim(
            min(
                CONFIG["calibration_grid"]
            ) - 2,
            max(
                CONFIG["calibration_grid"]
            ) + 2,
        )

        ax.set_ylim(
            0.0,
            1.0,
        )

        ax.set_xlabel(
            "Target calibration fraction (%)"
        )

        ax.set_ylabel(
            "Mean Macro-F1 (stored)"
        )

        ax.set_title(
            panel_title,
            loc="left",
            fontweight="bold",
            fontsize=11,
        )

        # Model identity is encoded by color; condition by line style.
        # A single model-color legend is placed below the two calibration
        # panels after both panels have been drawn.

        condition_handles = [
            plt.Line2D(
                [0], [0],
                color="black",
                linewidth=1.2,
                linestyle="-",
                label="ME",
            ),
            plt.Line2D(
                [0], [0],
                color="black",
                linewidth=1.2,
                linestyle="--",
                label="MI",
            ),
        ]
        ax.legend(
            handles=condition_handles,
            frameon=False,
            fontsize=7.0,
            ncol=2,
            loc="upper right",
            handlelength=1.8,
            borderaxespad=0.2,
        )

        style_axes(ax)

    # ==============================================================
    # 12. PANEL A — ML CALIBRATION
    # ==============================================================

    plot_calibration_panel(
        ax_ml,
        ml_models,
        "(A) Temporal calibration — ML models",
    )

    # ==============================================================
    # 13. PANEL B — DL CALIBRATION
    # ==============================================================

    plot_calibration_panel(
        ax_dl,
        dl_models,
        "(B) Temporal calibration — DL models",
    )

    # ==============================================================
    # 14. GLOBAL MODEL-COLOR LEGEND — BELOW THE CALIBRATION FIGURE
    # ==============================================================

    model_handles = []
    seen_models = set()
    for model_name in ml_models + dl_models:
        if model_name in seen_models:
            continue
        seen_models.add(model_name)
        model_handles.append(
            plt.Line2D(
                [0], [0],
                color=MODEL_COLORS.get(model_name, "black"),
                marker="o",
                linewidth=1.2,
                markersize=3,
                label=model_name,
            )
        )

    fig.legend(
        handles=model_handles,
        frameon=False,
        fontsize=7.0,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.44, 0.015),
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.45,
    )

    # ==============================================================
    # 15. PANEL C — VALID K90 DISTRIBUTION
    # ==============================================================

    if len(vals):

        # Keep the original histogram logic:
        # valid numerical k90 values only.
        ax_k90.hist(
            vals,
            bins=np.arange(
                -5,
                96,
                5,
            ),
            edgecolor="black",
            linewidth=0.5,
        )

        ax_k90.set_xlim(
            0,
            100,
        )

        ax_k90.set_xticks(
            CONFIG["calibration_grid"]
        )

        ax_k90.set_xlabel(
            "Valid k90 calibration fraction (%)"
        )

        ax_k90.set_ylabel(
            "Subject count"
        )

        ax_k90.set_title(
            "(C) k90 — valid observations only",
            loc="left",
            fontweight="bold",
            fontsize=11,
        )

        # ----------------------------------------------------------
        # Compact censoring summary.
        # ----------------------------------------------------------

        nr = int(
            (
                kd["k90_status_std"]
                == "NOT_REACHED"
            ).sum()
        )

        npg = int(
            (
                kd["k90_status_std"]
                == "NO_POSITIVE_GAP"
            ).sum()
        )

        ax_k90.text(
            0.97,
            0.97,
            f"Total = {len(kd)}\n"
            f"Valid = {len(vals)}\n"
            f"Excluded = {len(kd) - len(vals)}\n"
            f"Not reached = {nr}\n"
            f"No positive gap = {npg}",
            transform=ax_k90.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox=dict(
                boxstyle="round,pad=0.35",
                fill=False,
            ),
        )

    else:

        ax_k90.text(
            0.5,
            0.5,
            "No valid numerical k90 observations",
            ha="center",
            va="center",
        )

        ax_k90.axis(
            "off"
        )

    style_axes(
        ax_k90
    )

    # ==============================================================
    # FINAL LAYOUT — reserve a dedicated bottom band for the model legend
    # ==============================================================

    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.18,
        top=0.96,
        wspace=0.28,
        hspace=0.50,
    )

    # ==============================================================
    # 16. NO GLOBAL BURNED-IN TITLE
    # ==============================================================

    # The manuscript caption should provide the formal figure title.
    #
    # Do not call:
    #     fig.suptitle(...)
    #
    # here.

    # ==============================================================
    # 16. SAVE USING EXISTING BLOCK16 MECHANISM
    # ==============================================================

    return save_figure(
        fig,
        out,
        "FIG06",
    )
def fig07(root: Path, out: Path) -> Path:
    """
    FIG07 — ME–MI representational similarity.

    Scientific question
    -------------------
    How similar are the learned ME and MI representations across
    subjects, and how does that similarity vary across the predefined
    representation models?

    Panel A:
        Across-subject distribution of the stored subject-level
        Spearman RSA coefficient for each representation model.

    Panel B:
        Subject × representation-model map of the same stored RSA
        coefficient.

    Scientific rules
    ----------------
    • The BLOCK12 subject-level RSA artifact is authoritative.
    • Stored rho_spearman values are plotted directly.
    • No new correlations are calculated.
    • No p-values, confidence intervals, or effect sizes are generated.
    • The 25-subject paired structure is explicitly validated.
    • The representation-model set remains frozen.
    • RSA is restricted to the stored 7-gesture representation analysis.
    """

    # ==============================================================
    # 1. LOAD AUTHORITATIVE BLOCK12 SUBJECT-LEVEL RSA ARTIFACT
    # ==============================================================

    p = source_path(
        root,
        ARTIFACTS[
            "12_REPRESENTATION_ANALYSIS"
        ]["subject_rsa"],
    )

    d = load_csv(p)

    # ==============================================================
    # 2. VALIDATE REQUIRED SOURCE COLUMNS
    # ==============================================================

    required = {
        "subject_id",
        "model",
        "rho_spearman",
        "n_gesture_pairs",
        "n_gestures",
    }

    missing = required - set(d.columns)

    if missing:
        raise RuntimeError(
            "FIG07 subject-level RSA source is missing "
            f"{sorted(missing)}; columns={list(d.columns)}"
        )

    d = d.copy()

    # ==============================================================
    # 3. STANDARDIZE STORED FIELDS
    # ==============================================================

    d["model_std"] = d[
        "model"
    ].map(
        normalize_model
    )

    d["rho_std"] = pd.to_numeric(
        d["rho_spearman"],
        errors="coerce",
    )

    d["subject_std"] = pd.to_numeric(
        d["subject_id"],
        errors="coerce",
    )

    d["n_gesture_pairs_std"] = pd.to_numeric(
        d["n_gesture_pairs"],
        errors="coerce",
    )

    d["n_gestures_std"] = pd.to_numeric(
        d["n_gestures"],
        errors="coerce",
    )

    # ==============================================================
    # 4. RESTRICT TO THE FROZEN REPRESENTATION MODELS
    # ==============================================================

    representation_models = [
        m
        for m in CONFIG["models_representation"]
        if m in set(
            d["model_std"]
        )
    ]

    if representation_models != list(
        CONFIG["models_representation"]
    ):
        raise RuntimeError(
            "FIG07 representation-model mismatch. "
            f"Expected={CONFIG['models_representation']}; "
            f"observed={representation_models}"
        )

    d = d[
        d["model_std"].isin(
            representation_models
        )
        & d["rho_std"].notna()
        & d["subject_std"].notna()
    ].copy()

    if d.empty:
        raise RuntimeError(
            "FIG07: no finite stored subject-level RSA observations."
        )

    # ==============================================================
    # 5. VALIDATE RSA RANGE
    # ==============================================================

    if (
        (d["rho_std"] < -1.0)
        |
        (d["rho_std"] > 1.0)
    ).any():

        raise RuntimeError(
            "FIG07: stored Spearman RSA values fall outside "
            "the valid [-1, 1] range."
        )

    # ==============================================================
    # 6. VALIDATE 25-SUBJECT PAIRED COHORT
    # ==============================================================

    expected_subjects = list(
        range(1, 26)
    )

    observed_subjects = sorted(
        d["subject_std"]
        .astype(int)
        .unique()
        .tolist()
    )

    if observed_subjects != expected_subjects:
        raise RuntimeError(
            "FIG07 requires subjects 1–25. "
            f"Observed={observed_subjects}"
        )

    # ==============================================================
    # 7. VALIDATE ONE RSA VALUE PER SUBJECT × MODEL
    # ==============================================================

    expected_rows = (
        25 *
        len(representation_models)
    )

    if len(d) != expected_rows:
        raise RuntimeError(
            "FIG07 expected "
            f"{expected_rows} subject/model RSA rows "
            f"(25 subjects × {len(representation_models)} models); "
            f"found {len(d)}."
        )

    if d.duplicated(
        [
            "subject_std",
            "model_std",
        ]
    ).any():

        duplicates = d.loc[
            d.duplicated(
                [
                    "subject_std",
                    "model_std",
                ],
                keep=False,
            ),
            [
                "subject_std",
                "model_std",
            ],
        ]

        raise RuntimeError(
            "FIG07 duplicate subject/model RSA keys found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    # ==============================================================
    # 8. VALIDATE STORED GESTURE STRUCTURE
    #
    # No new gesture-pair calculation is performed.
    # We only verify that the source records the expected frozen
    # 7-gesture representation analysis.
    # ==============================================================

    gesture_counts = (
        d["n_gestures_std"]
        .dropna()
        .unique()
    )

    if len(gesture_counts) != 1:
        raise RuntimeError(
            "FIG07: inconsistent n_gestures values in "
            f"subject-level RSA artifact: {gesture_counts}"
        )

    if int(gesture_counts[0]) != 7:
        raise RuntimeError(
            "FIG07: subject-level RSA artifact does not report "
            f"the expected 7 gestures; observed={gesture_counts[0]}"
        )

    pair_counts = (
        d["n_gesture_pairs_std"]
        .dropna()
        .unique()
    )

    if len(pair_counts) != 1:
        raise RuntimeError(
            "FIG07: inconsistent n_gesture_pairs values in "
            f"subject-level RSA artifact: {pair_counts}"
        )

    # For 7 gestures, the unique unordered gesture-pair count is 21.
    if int(pair_counts[0]) != 21:
        raise RuntimeError(
            "FIG07: subject-level RSA artifact does not report "
            f"the expected 21 gesture pairs; "
            f"observed={pair_counts[0]}"
        )

    # ==============================================================
    # 9. ORDER SUBJECTS AND REPRESENTATION MODELS
    # ==============================================================

    d["subject_std"] = d[
        "subject_std"
    ].astype(int)

    d["model_std"] = pd.Categorical(
        d["model_std"],
        categories=representation_models,
        ordered=True,
    )

    d = d.sort_values(
        [
            "subject_std",
            "model_std",
        ]
    )

    # ==============================================================
    # 10. CREATE FIGURE
    # ==============================================================

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(13.6, 5.4),
    )

    # ==============================================================
    # 11. PANEL A — ACROSS-SUBJECT RSA DISTRIBUTION
    # ==============================================================

    ax = axs[0]

    positions = np.arange(
        len(representation_models)
    )

    for i, model_name in enumerate(
        representation_models
    ):

        values = d.loc[
            d["model_std"].eq(model_name),
            "rho_std",
        ].to_numpy(
            dtype=float
        )

        if len(values) != 25:
            raise RuntimeError(
                "FIG07: expected 25 RSA observations for "
                f"{model_name}; found {len(values)}."
            )

        # Boxplot summarizes the across-subject distribution.
        ax.boxplot(
            values,
            positions=[i],
            widths=0.55,
            showfliers=False,
        )

        # All 25 subjects remain visible.
        ax.scatter(
            np.full(
                len(values),
                i,
            ),
            values,
            s=18,
            alpha=0.55,
            zorder=3,
        )

    ax.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
        alpha=0.7,
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        [model_tick_label(m) for m in representation_models],
        rotation=25,
        ha="right",
        fontsize=8.5,
    )

    ax.set_ylim(
        -1.0,
        1.0,
    )

    ax.set_ylabel(
        r"Spearman $\rho$ (ME–MI representations)"
    )

    ax.set_xlabel(
        "Representation model"
    )

    ax.set_title(
        "(A) Across-subject RSA distribution",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )

    ax.text(
        0.97,
        0.96,
        "25 paired subjects/model",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
    )

    style_axes(ax)

    # ==============================================================
    # 12. PANEL B — SUBJECT × MODEL RSA MAP
    # ==============================================================

    ax = axs[1]

    pivot = d.pivot(
        index="subject_std",
        columns="model_std",
        values="rho_std",
    ).reindex(
        index=expected_subjects,
        columns=representation_models,
    )

    if pivot.isna().any().any():
        raise RuntimeError(
            "FIG07: subject × model RSA matrix is incomplete."
        )

    im = ax.imshow(
        pivot.to_numpy(
            dtype=float
        ),
        aspect="auto",
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
    )

    ax.set_xlabel(
        "Representation model"
    )

    ax.set_ylabel(
        "Subject"
    )

    ax.set_xticks(
        np.arange(
            len(representation_models)
        )
    )

    ax.set_xticklabels(
        [model_tick_label(m) for m in representation_models],
        rotation=25,
        ha="right",
        fontsize=8.5,
    )

    ax.set_yticks(
        np.arange(
            len(expected_subjects)
        )
    )

    ax.set_yticklabels(
        expected_subjects
    )

    ax.set_title(
        "(B) Subject-level RSA map",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        r"Spearman $\rho$"
    )

    cbar.set_ticks(
        [
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
        ]
    )

    style_axes(ax)

    # ==============================================================
    # 13. FINAL LAYOUT
    # ==============================================================

    fig.subplots_adjust(
        left=0.07,
        right=0.96,
        bottom=0.23,
        top=0.91,
        wspace=0.28,
    )

    # No global burned-in figure title.
    # The manuscript caption supplies the formal figure title.

    # ==============================================================
    # 14. SAVE USING EXISTING BLOCK16 MECHANISM
    # ==============================================================

    return save_figure(
        fig,
        out,
        "FIG07",
    )
def fig08(root: Path, out: Path) -> Path:
    """
    FIG08 — ME–MI representational similarity versus transfer and calibration.

    Scientific questions
    --------------------
    Panel A:
        Does stored ME–MI representational similarity vary with stored
        bidirectional transfer performance?

    Panel B:
        Does stored ME–MI representational similarity vary with the
        stored amount of calibration required to reach k90?

    Scientific rules
    ----------------
    • BLOCK15 subject-level association inputs are authoritative.
    • Stored RSA values are used directly.
    • Stored bidirectional transfer values are used directly.
    • Stored k90 values are used directly.
    • No new correlation coefficient is calculated.
    • No p-value is calculated.
    • No regression line is fitted.
    • No imputation of excluded k90 observations is performed.
    • Invalid/censored k90 observations remain excluded from Panel B.
    • The 25-subject × 3-representation-model structure is validated.
    • No global figure title is burned into the figure.
    """

    # ==============================================================
    # 1. LOAD AUTHORITATIVE BLOCK15 SOURCES
    # ==============================================================

    rt = source_path(
        root,
        ARTIFACTS["BLOCK15"]["rsa_transfer"],
    )

    rk = source_path(
        root,
        ARTIFACTS["BLOCK15"]["rsa_calibration"],
    )

    dt = load_csv(rt)
    dk = load_csv(rk)

    # ==============================================================
    # 2. VALIDATE TRANSFER SOURCE
    # ==============================================================

    req_t = {
        "subject_id",
        "model",
        "rho_spearman",
        "ME_to_MI",
        "MI_to_ME",
        "bidirectional_mean_transfer_benefit",
    }

    miss_t = req_t - set(dt.columns)

    if miss_t:
        raise RuntimeError(
            "FIG08 transfer source missing "
            f"{sorted(miss_t)}; columns={list(dt.columns)}"
        )

    # ==============================================================
    # 3. VALIDATE CALIBRATION SOURCE
    # ==============================================================

    req_k = {
        "subject_id",
        "model",
        "rho_spearman",
        "ME_k90",
        "MI_k90",
        "mean_k90",
        "k90_valid",
        "k90_exclusion_reason",
    }

    miss_k = req_k - set(dk.columns)

    if miss_k:
        raise RuntimeError(
            "FIG08 calibration source missing "
            f"{sorted(miss_k)}; columns={list(dk.columns)}"
        )

    # ==============================================================
    # 4. STANDARDIZE TRANSFER DATA
    # ==============================================================

    dt = dt.copy()

    dt["model_std"] = dt[
        "model"
    ].map(
        normalize_model
    )

    dt["subject_std"] = pd.to_numeric(
        dt["subject_id"],
        errors="coerce",
    )

    dt["rho_std"] = pd.to_numeric(
        dt["rho_spearman"],
        errors="coerce",
    )

    dt["transfer_benefit_std"] = pd.to_numeric(
        dt[
            "bidirectional_mean_transfer_benefit"
        ],
        errors="coerce",
    )

    dt = dt[
        dt["model_std"].isin(
            CONFIG["models_representation"]
        )
    ].copy()

    # ==============================================================
    # 5. STANDARDIZE CALIBRATION DATA
    # ==============================================================

    dk = dk.copy()

    dk["model_std"] = dk[
        "model"
    ].map(
        normalize_model
    )

    dk["subject_std"] = pd.to_numeric(
        dk["subject_id"],
        errors="coerce",
    )

    dk["rho_std"] = pd.to_numeric(
        dk["rho_spearman"],
        errors="coerce",
    )

    dk["k90_std"] = pd.to_numeric(
        dk["mean_k90"],
        errors="coerce",
    )

    dk["k90_valid_std"] = (
        dk["k90_valid"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(
            {
                "TRUE": True,
                "1": True,
                "YES": True,
                "Y": True,
                "FALSE": False,
                "0": False,
                "NO": False,
                "N": False,
            }
        )
        .fillna(False)
    )

    dk = dk[
        dk["model_std"].isin(
            CONFIG["models_representation"]
        )
    ].copy()

    # ==============================================================
    # 6. VALIDATE EXPECTED SUBJECT × MODEL STRUCTURE
    # ==============================================================

    representation_models = list(
        CONFIG["models_representation"]
    )

    expected_subjects = list(
        range(1, 26)
    )

    expected_rows = (
        25 *
        len(representation_models)
    )

    if len(dt) != expected_rows:
        raise RuntimeError(
            "FIG08 transfer expected "
            f"{expected_rows} rows "
            f"(25 subjects × {len(representation_models)} models); "
            f"found {len(dt)}."
        )

    if len(dk) != expected_rows:
        raise RuntimeError(
            "FIG08 calibration expected "
            f"{expected_rows} rows "
            f"(25 subjects × {len(representation_models)} models); "
            f"found {len(dk)}."
        )

    # ==============================================================
    # 7. VALIDATE SUBJECT IDENTIFIERS
    # ==============================================================

    for name, frame in [
        ("transfer", dt),
        ("calibration", dk),
    ]:

        observed = sorted(
            frame[
                "subject_std"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if observed != expected_subjects:
            raise RuntimeError(
                f"FIG08 {name} source requires subjects 1–25; "
                f"observed={observed}"
            )

    # ==============================================================
    # 8. VALIDATE UNIQUE SUBJECT × MODEL KEYS
    # ==============================================================

    if dt.duplicated(
        [
            "subject_std",
            "model_std",
        ]
    ).any():

        raise RuntimeError(
            "FIG08 duplicate transfer subject/model keys."
        )

    if dk.duplicated(
        [
            "subject_std",
            "model_std",
        ]
    ).any():

        raise RuntimeError(
            "FIG08 duplicate calibration subject/model keys."
        )

    # ==============================================================
    # 9. VALIDATE RSA RANGE
    # ==============================================================

    finite_transfer_rho = dt[
        "rho_std"
    ].dropna()

    finite_calibration_rho = dk[
        "rho_std"
    ].dropna()

    for name, values in [
        ("transfer", finite_transfer_rho),
        ("calibration", finite_calibration_rho),
    ]:

        if (
            (values < -1.0)
            |
            (values > 1.0)
        ).any():

            raise RuntimeError(
                f"FIG08 {name} RSA values fall outside "
                "the valid Spearman [-1, 1] range."
            )

    # ==============================================================
    # 10. VALIDATE STORED TRANSFER VALUES
    # ==============================================================

    if not np.isfinite(
        dt[
            "transfer_benefit_std"
        ].dropna().to_numpy(
            dtype=float
        )
    ).all():

        raise RuntimeError(
            "FIG08 transfer source contains non-finite "
            "bidirectional transfer values."
        )

    # ==============================================================
    # 11. PREPARE VALID K90 OBSERVATIONS
    #
    # No imputation or interpolation.
    # ==============================================================
    
    ok_t = (
        np.isfinite(
            dt["rho_std"].to_numpy(
                dtype=float
            )
        )
        &
        np.isfinite(
            dt["transfer_benefit_std"].to_numpy(
                dtype=float
            )
        )
    )

    ok_k = (
        dk["k90_valid_std"].to_numpy(
            dtype=bool
        )
        &
        np.isfinite(
            dk["rho_std"].to_numpy(
                dtype=float
            )
        )
        &
        np.isfinite(
            dk["k90_std"].to_numpy(
                dtype=float
            )
        )
    )

    # ==============================================================
    # 12. CREATE FIGURE
    # ==============================================================

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(14.2, 5.2),
    )

    # ==============================================================
    # 13. MODEL MARKERS
    # ==============================================================

    marker_map = {
        "EEGNet": "o",
        "CNN-BiLSTM": "s",
        "CNN-BiLSTM-Attention": "^",
    }

    # ==============================================================
    # 14. PANEL A — RSA VS TRANSFER
    # ==============================================================

    ax = axs[0]

    for model_name in representation_models:

        x = dt[
            dt["model_std"].eq(
                model_name
            )
            & pd.Series(
                ok_t,
                index=dt.index,
            )
        ].copy()

        if x.empty:
            continue

        ax.scatter(
            x["rho_std"],
            x["transfer_benefit_std"],
            s=42,
            alpha=0.70,
            marker=marker_map.get(
                model_name,
                "o",
            ),
            label=model_name,
        )

    ax.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
        alpha=0.7,
    )

    ax.set_xlabel(
        r"ME–MI representation similarity (Spearman $\rho$)"
    )

    ax.set_ylabel(
        "Bidirectional transfer benefit (Macro-F1)"
    )

    ax.set_title(
        "(A) Representation similarity versus transfer",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )

    ax.legend(
        frameon=False,
        loc="best",
        fontsize=8,
    )

    style_axes(ax)

    # ==============================================================
    # 15. PANEL B — RSA VS K90
    # ==============================================================

    ax = axs[1]

    for model_name in representation_models:

        x = dk[
            dk["model_std"].eq(
                model_name
            )
            & pd.Series(
                ok_k,
                index=dk.index,
            )
        ].copy()

        if x.empty:
            continue

        ax.scatter(
            x["rho_std"],
            x["k90_std"],
            s=42,
            alpha=0.70,
            marker=marker_map.get(
                model_name,
                "o",
            ),
            label=model_name,
        )

    ax.set_xlabel(
        r"ME–MI representation similarity (Spearman $\rho$)"
    )

    ax.set_ylabel(
        "k90 calibration fraction (%)"
    )

    ax.set_title(
        "(B) Representation similarity versus k90 calibration",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )

    ax.legend(
        frameon=False,
        loc="best",
        fontsize=8,
    )

    # --------------------------------------------------------------
    # Preserve the frozen k90 censoring information without placing
    # a large text block over the data.
    # --------------------------------------------------------------

    total = len(dk)
    included = int(ok_k.sum())
    excluded = total - included

    reasons = (
        dk.loc[
            ~ok_k,
            "k90_exclusion_reason",
        ]
        .astype(str)
        .str.upper()
    )

    nr = int(
        reasons.str.contains(
            "NOT_REACHED",
            na=False,
        ).sum()
    )

    npg = int(
        reasons.str.contains(
            "NO_POSITIVE_GAP",
            na=False,
        ).sum()
    )

    ax.text(
        0.03,
        0.04,
        f"Valid k90: {included}/{total}\n"
        f"Excluded: {excluded}\n"
        f"Not reached: {nr}; No positive gap: {npg}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
    )

    style_axes(ax)

    # ==============================================================
    # 16. COMMON SCIENTIFIC SCALE FOR RSA
    # ==============================================================

    axs[0].set_xlim(
        -1.0,
        1.0,
    )

    axs[1].set_xlim(
        -1.0,
        1.0,
    )

    # ==============================================================
    # 17. FINAL LAYOUT
    # ==============================================================

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.18,
        top=0.91,
        wspace=0.28,
    )

    # No global burned-in figure title.
    # The manuscript caption supplies the formal figure title.

    # ==============================================================
    # 18. SAVE THROUGH EXISTING BLOCK16 MECHANISM
    # ==============================================================

    return save_figure(
        fig,
        out,
        "FIG08",
    )

def _find_block13_table(root: Path, filename: str) -> Path:
    base = root / "13_XAI"
    preferred = base / "tables" / filename
    if preferred.is_file():
        return preferred
    matches = sorted(p for p in base.rglob(filename) if p.is_file()) if base.is_dir() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Required frozen BLOCK13 table not found: {filename}")

def fig09(root: Path, out: Path) -> dict:
    """
    FIG09 — Explainable AI and ME–MI attribution structure.

    Scientific scope
    ----------------
    Panel A:
        Stored ML feature-family attribution magnitude.

    Panel B:
        Stored ML channel attribution magnitude.

    Panel C:
        Stored ME–MI channel attribution difference for ML and DL branches.

    Panel D:
        Stored ME–MI channel-attribution similarity across the frozen
        model set.

    Scientific rules
    ----------------
    • BLOCK13 frozen tables are authoritative.
    • No SHAP values are recomputed.
    • No new statistical tests are performed.
    • No p-values or FDR values are recomputed.
    • Stored mean quantities are plotted directly or descriptively
      aggregated exactly as in the frozen FIG09 specification.
    • Numerical gesture identifiers 1–7 remain unchanged.
    • N=25 source validation remains unchanged.
    """

    # ==============================================================
    # 1. LOAD FROZEN BLOCK13 TABLES
    # ==============================================================

    ml = pd.read_csv(
        _find_block13_table(
            root,
            "ML_top_features.csv",
        )
    )

    channel = pd.read_csv(
        _find_block13_table(
            root,
            "XAI_ME_MI_paired_channel_statistics.csv",
        )
    )

    similarity = pd.read_csv(
        _find_block13_table(
            root,
            "XAI_attribution_similarity_statistics.csv",
        )
    )

    # ==============================================================
    # 2. VALIDATE REQUIRED SOURCE SCHEMAS
    # ==============================================================

    reqs = {
        "ML_top_features.csv": (
            ml,
            {
                "model",
                "condition",
                "gesture",
                "feature_id",
                "feature_name",
                "feature_family",
                "channel",
                "frequency_group",
                "mean_abs_shap",
                "rank",
            },
        ),
        "XAI_ME_MI_paired_channel_statistics.csv": (
            channel,
            {
                "branch",
                "model",
                "gesture",
                "channel",
                "n_subjects",
                "mean_ME_minus_MI",
                "median_ME_minus_MI",
                "effect_size_dz",
                "ci95_low",
                "ci95_high",
                "p_value",
                "p_fdr",
            },
        ),
        "XAI_attribution_similarity_statistics.csv": (
            similarity,
            {
                "model",
                "gesture",
                "metric",
                "n_subjects",
                "mean",
                "median",
                "ci95_low",
                "ci95_high",
                "p_value_vs_zero",
                "p_fdr",
            },
        ),
    }

    for name, (df, req) in reqs.items():

        missing = req - set(df.columns)

        if missing:
            raise RuntimeError(
                f"FIG09 {name}: missing required columns "
                f"{sorted(missing)}"
            )

    # ==============================================================
    # 3. COPY AND VALIDATE ML SHAP SOURCE
    # ==============================================================

    ml = ml.copy()

    ml["gesture"] = pd.to_numeric(
        ml["gesture"],
        errors="coerce",
    )

    ml["mean_abs_shap"] = pd.to_numeric(
        ml["mean_abs_shap"],
        errors="coerce",
    )

    ml["rank"] = pd.to_numeric(
        ml["rank"],
        errors="coerce",
    )

    if not ml["gesture"].isin(
        CONFIG["gestures"]
    ).all():

        raise RuntimeError(
            "FIG09 ML source contains gestures outside "
            f"the frozen set {CONFIG['gestures']}"
        )

    if not np.isfinite(
        ml[
            [
                "mean_abs_shap",
                "rank",
            ]
        ].to_numpy(
            dtype=float
        )
    ).all():

        raise RuntimeError(
            "FIG09 ML source contains non-finite "
            "SHAP/rank values."
        )

    # ==============================================================
    # 4. COPY AND VALIDATE ME–MI CHANNEL SOURCE
    # ==============================================================

    channel = channel.copy()

    channel["gesture"] = pd.to_numeric(
        channel["gesture"],
        errors="coerce",
    )

    channel["n_subjects"] = pd.to_numeric(
        channel["n_subjects"],
        errors="coerce",
    )

    for col in [
        "mean_ME_minus_MI",
        "median_ME_minus_MI",
        "effect_size_dz",
        "ci95_low",
        "ci95_high",
    ]:

        channel[col] = pd.to_numeric(
            channel[col],
            errors="coerce",
        )

    if not channel["gesture"].isin(
        CONFIG["gestures"]
    ).all():

        raise RuntimeError(
            "FIG09 channel-statistics source contains "
            "gestures outside the frozen gesture set."
        )

    if not (
        channel["n_subjects"]
        == CONFIG["primary_subjects"]
    ).all():

        raise RuntimeError(
            "FIG09 channel-statistics source does not "
            f"consistently report N={CONFIG['primary_subjects']}."
        )

    if not np.isfinite(
        channel[
            [
                "mean_ME_minus_MI",
                "median_ME_minus_MI",
                "effect_size_dz",
                "ci95_low",
                "ci95_high",
            ]
        ].to_numpy(
            dtype=float
        )
    ).all():

        raise RuntimeError(
            "FIG09 channel-statistics contain non-finite "
            "stored statistical quantities."
        )

    # ==============================================================
    # 5. COPY AND VALIDATE ATTRIBUTION-SIMILARITY SOURCE
    # ==============================================================

    similarity = similarity.copy()

    similarity["gesture"] = pd.to_numeric(
        similarity["gesture"],
        errors="coerce",
    )

    similarity["n_subjects"] = pd.to_numeric(
        similarity["n_subjects"],
        errors="coerce",
    )

    for col in [
        "mean",
        "median",
        "ci95_low",
        "ci95_high",
    ]:

        similarity[col] = pd.to_numeric(
            similarity[col],
            errors="coerce",
        )

    if not similarity["gesture"].isin(
        CONFIG["gestures"]
    ).all():

        raise RuntimeError(
            "FIG09 similarity source contains gestures "
            "outside the frozen gesture set."
        )

    similarity = similarity[
        similarity["n_subjects"]
        == CONFIG["primary_subjects"]
    ].copy()

    if similarity.empty:
        raise RuntimeError(
            "FIG09 similarity has no eligible "
            f"N={CONFIG['primary_subjects']} rows."
        )

    if not np.isfinite(
        similarity[
            [
                "mean",
                "median",
                "ci95_low",
                "ci95_high",
            ]
        ].to_numpy(
            dtype=float
        )
    ).all():

        raise RuntimeError(
            "FIG09 eligible similarity contains "
            "non-finite stored values."
        )

    # ==============================================================
    # 6. PANEL D COVERAGE VALIDATION
    # ==============================================================

    sim_channel = similarity[
        similarity["metric"].astype(str)
        == "channel_similarity"
    ].copy()

    if sim_channel.empty:
        raise RuntimeError(
            "FIG09 Panel D requires frozen BLOCK13 "
            "channel_similarity rows."
        )

    expected_models = MODEL_ORDER

    counts = (
        sim_channel
        .groupby("model")["gesture"]
        .nunique()
    )

    missing = [
        m
        for m in expected_models
        if int(counts.get(m, 0)) != 7
    ]

    if missing:
        raise RuntimeError(
            "FIG09 Panel D channel_similarity coverage "
            "is incomplete: "
            f"{missing}"
        )

    # ==============================================================
    # 7. PUBLICATION FIGURE CANVAS
    # ==============================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16.5, 9.8),
    )

    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # Restrained publication palette.
    #
    # Panels A/B/D:
    #     same neutral bar color because they show magnitude
    #     summaries rather than categorical groups.
    #
    # Panel C:
    #     two distinct tones because ML and DL are separate branches.
    #
    neutral_bar = "#4C78A8"
    ml_color = "#4C78A8"
    dl_color = "#F58518"

    # ==============================================================
    # 8. PANEL A
    # ML FEATURE-FAMILY ATTRIBUTION
    # ==============================================================

    family = (
        ml.groupby(
            "feature_family",
            as_index=False,
        )[
            "mean_abs_shap"
        ]
        .mean()
    )

    family["order"] = family[
        "feature_family"
    ].map(
        {
            name: i
            for i, name in enumerate(
                FEATURE_FAMILY_ORDER
            )
        }
    )

    family = family.sort_values(
        "order"
    )

    ax_a.bar(
        np.arange(len(family)),
        family["mean_abs_shap"],
        color=neutral_bar,
        width=0.72,
    )

    ax_a.set_xticks(
        np.arange(
            len(family)
        )
    )

    ax_a.set_xticklabels(
        [
            feature_family_label(x)
            for x in family["feature_family"]
        ],
        rotation=20,
        ha="right",
    )

    ax_a.set_ylabel(
        "Mean absolute SHAP value"
    )

    # No descriptive title inside panel.
    # Panel identity is supplied separately below.

    ax_a.grid(
        axis="y",
        alpha=0.25,
    )

    ax_a.text(
        0.01,
        0.98,
        "A",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    style_axes(ax_a)

    # ==============================================================
    # 9. PANEL B — CHANNEL ATTRIBUTION MAGNITUDE
    # ==============================================================

    channel_summary = (
        ml.groupby("channel", as_index=False)["mean_abs_shap"]
        .mean()
    )
    channel_summary["channel_num"] = pd.to_numeric(
        channel_summary["channel"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )
    channel_summary = channel_summary.sort_values(
        "mean_abs_shap",
        ascending=True,
    )

    ax_b.barh(
        np.arange(len(channel_summary)),
        channel_summary["mean_abs_shap"].to_numpy(),
        height=0.68,
    )
    ax_b.set_yticks(np.arange(len(channel_summary)))
    ax_b.set_yticklabels(
        channel_summary["channel"].astype(str),
        fontsize=8.5,
    )
    ax_b.set_xlabel("Mean absolute SHAP value")
    ax_b.set_ylabel("EEG channel")
    ax_b.set_title(
        "(B) Channel-level attribution",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    ax_b.invert_yaxis()
    style_axes(ax_b)

    ax_b.text(
        0.98,
        0.98,
        f"{len(channel_summary)} channels",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )

    # ==============================================================
    # 10. PANEL C
    # ME–MI CHANNEL ATTRIBUTION DIFFERENCE
    # ==============================================================

    channel_mean = (
        channel.groupby(
            [
                "branch",
                "channel",
            ],
            as_index=False,
        )[
            "mean_ME_minus_MI"
        ]
        .mean()
    )

    channel_mean[
        "channel_num"
    ] = (
        channel_mean[
            "channel"
        ]
        .astype(str)
        .str.extract(
            r"(\d+)",
            expand=False,
        )
        .astype(int)
    )

    channel_mean = channel_mean.sort_values(
        [
            "branch",
            "channel_num",
        ]
    )

    for branch, color in [
        ("ML", ml_color),
        ("DL", dl_color),
    ]:

        q = channel_mean[
            channel_mean[
                "branch"
            ]
            == branch
        ]

        if q.empty:
            continue

        ax_c.plot(
            q["channel_num"],
            q["mean_ME_minus_MI"],
            marker="o",
            markersize=4.5,
            linewidth=1.6,
            label=branch,
            color=color,
        )

    ax_c.axhline(
        0,
        linewidth=0.8,
        linestyle="--",
        color="black",
        alpha=0.65,
    )

    channel_numbers = sorted(
        channel_mean[
            "channel_num"
        ].unique()
    )

    ax_c.set_xticks(
        channel_numbers
    )

    ax_c.set_xticklabels(
        [
            f"CH{x:02d}"
            for x in channel_numbers
        ]
    )

    ax_c.set_xlabel(
        "EEG channel"
    )

    ax_c.set_ylabel(
        "Mean ME − MI attribution"
    )

    ax_c.grid(
        axis="y",
        alpha=0.25,
    )

    ax_c.legend(
        frameon=False,
        loc="best",
    )

    ax_c.text(
        0.01,
        0.98,
        "C",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    style_axes(ax_c)

    # ==============================================================
    # 11. PANEL D
    # ME–MI CHANNEL ATTRIBUTION SIMILARITY
    # ==============================================================

    sim_channel = similarity[
        similarity["metric"]
        == "channel_similarity"
    ].copy()

    if sim_channel.empty:
        raise RuntimeError(
            "FIG09 Panel D requires frozen BLOCK13 "
            "channel_similarity rows with "
            f"n_subjects={CONFIG['primary_subjects']}."
        )

    model_gesture_counts = (
        sim_channel
        .groupby("model")["gesture"]
        .nunique()
    )

    missing_models = [
        model
        for model in MODEL_ORDER
        if (
            model not in model_gesture_counts.index
            or int(
                model_gesture_counts.loc[model]
            ) != len(GESTURES)
        )
    ]

    if missing_models:
        raise RuntimeError(
            "FIG09 Panel D channel_similarity coverage "
            "is incomplete. Expected 7 gestures for every "
            f"model; missing/incomplete: {missing_models}"
        )

    sim_model = (
        sim_channel
        .groupby(
            "model",
            as_index=False,
        )[
            "mean"
        ]
        .mean()
    )

    sim_model["order"] = sim_model[
        "model"
    ].map(
        {
            model: i
            for i, model in enumerate(
                MODEL_ORDER
            )
        }
    )

    sim_model = sim_model.sort_values(
        "order"
    )

    x = np.arange(
        len(sim_model)
    )

    ax_d.bar(
        x,
        sim_model[
            "mean"
        ].to_numpy(
            dtype=float
        ),
        color=neutral_bar,
        width=0.72,
    )

    ax_d.set_xticks(
        x
    )

    fig09_labels = {
        "RandomForest": "Random\nForest",
        "XGBoost": "XGBoost",
        "ShallowConvNet": "Shallow\nConvNet",
        "DeepConvNet": "Deep\nConvNet",
        "CNN-BiLSTM": "CNN–\nBiLSTM",
        "CNN-BiLSTM-Attention": "CNN–\nBiLSTM–\nAttention",
    }

    ax_d.set_xticklabels(
        [fig09_labels.get(m, m) for m in sim_model["model"]],
        rotation=0,
        ha="center",
        va="top",
        fontsize=6.8,
    )
    ax_d.tick_params(
        axis="x",
        pad=7,
    )

    ax_d.set_ylabel(
        "Mean ME–MI channel attribution similarity"
    )

    ax_d.grid(
        axis="y",
        alpha=0.25,
    )

    ax_d.text(
        0.01,
        0.98,
        "D",
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    style_axes(ax_d)

    # ==============================================================
    # 12. FINAL PUBLICATION LAYOUT
    # ==============================================================

    fig.subplots_adjust(
        left=0.065,
        right=0.99,
        bottom=0.27,
        top=0.97,
        wspace=0.27,
        hspace=0.38,
    )

    # IMPORTANT:
    # No global fig.suptitle().
    # No panel descriptive titles.
    # The manuscript caption supplies the formal figure title
    # and scientific explanation.

    return save_figure(
        fig,
        out,
        "FIG09",
    )


def fig10(root: Path, out: Path, b14_audit: dict) -> dict:
    """
    FIG10 — Robustness of Macro-F1 to channel dropout and additive noise.

    Plotted values are stored BLOCK14 perturbation effects:
        Δ Macro-F1 = clean − perturbed.

    No new inferential statistics are calculated.
    """

    if not b14_audit["strict_pass"]:
        raise RuntimeError(
            "FIG10 blocked: strict robustness audit has not passed."
        )

    dd = load_csv(
        source_path(
            root,
            ARTIFACTS["14_ROBUSTNESS"]["dropout_subject_level"],
        )
    )
    nd = load_csv(
        source_path(
            root,
            ARTIFACTS["14_ROBUSTNESS"]["robustness_subject_level"],
        )
    )

    def find_any(df, names):
        normalized = {norm_col(c): c for c in df.columns}
        for name in names:
            key = norm_col(name)
            if key in normalized:
                return normalized[key]
        return None

    level = find_any(
        dd,
        [
            "dropout_percent",
            "dropout_nominal",
            "dropout_level",
            "nominal_percent",
            "perturbation_level",
        ],
    )
    delta = find_any(
        dd,
        ["delta_Macro-F1", "delta_macro_f1", "delta_f1"],
    )
    ch = find_any(
        dd,
        ["channels", "channel_set", "dropped_channels"],
    )

    if not all([level, delta, ch]):
        raise RuntimeError(
            f"FIG10 dropout schema incomplete: {list(dd.columns)}"
        )

    dd = dd.copy()
    dd["level_std"] = pd.to_numeric(dd[level], errors="coerce")
    dd["delta_std"] = pd.to_numeric(dd[delta], errors="coerce")

    for nominal in CONFIG["dropout_nominal"]:
        vals = (
            dd.loc[dd["level_std"] == nominal, ch]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
        if len(vals) != 1:
            raise RuntimeError(
                f"FIG10 expected one stored dropout pattern at "
                f"{nominal}%; found {vals}"
            )

    noise = find_any(nd, ["perturbation_level"])
    semantics = find_any(nd, ["noise_level_semantics"])
    ndelta = find_any(
        nd,
        ["delta_Macro-F1", "delta_macro_f1", "delta_f1"],
    )

    if not all([noise, semantics, ndelta]):
        raise RuntimeError(
            f"FIG10 noise schema incomplete: {list(nd.columns)}"
        )

    semantics_set = (
        nd[semantics]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
    )
    if set(semantics_set) != {"snr_db"}:
        raise RuntimeError(
            f"FIG10 expected snr_db semantics, found "
            f"{sorted(set(semantics_set))}"
        )

    nd = nd.copy()
    nd["snr_std"] = pd.to_numeric(nd[noise], errors="coerce")
    nd["delta_std"] = pd.to_numeric(nd[ndelta], errors="coerce")

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(13.8, 5.2),
    )

    def plot_distribution(ax, groups, labels, xlabel, title, panel):
        positions = np.arange(len(groups))
        ax.boxplot(
            groups,
            positions=positions,
            widths=0.50,
            showfliers=False,
            medianprops=dict(linewidth=1.4),
            whiskerprops=dict(linewidth=0.9),
            capprops=dict(linewidth=0.9),
            boxprops=dict(linewidth=0.9),
        )

        for i, values in enumerate(groups):
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                # Deterministic horizontal jitter; no random state is introduced.
                offsets = np.linspace(-0.17, 0.17, values.size)
                ax.scatter(
                    np.full(values.size, i) + offsets,
                    values,
                    s=10,
                    alpha=0.22,
                    zorder=2,
                )

        ax.axhline(
            0.0,
            linewidth=0.9,
            linestyle="--",
            alpha=0.75,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Δ Macro-F1 (clean − perturbed)")
        ax.set_title(
            title,
            loc="left",
            fontweight="bold",
            fontsize=11,
        )
        panel_label(ax, panel)
        style_axes(ax)

    dropout_levels = CONFIG["dropout_nominal"]
    drop_groups = [
        dd.loc[
            dd["level_std"] == level_value,
            "delta_std",
        ].dropna().to_numpy(float)
        for level_value in dropout_levels
    ]

    plot_distribution(
        axs[0],
        drop_groups,
        [f"{v}%" for v in dropout_levels],
        "Channel dropout",
        "(A) Channel-loss robustness",
        "A",
    )

    levels = sorted(
        nd["snr_std"].dropna().unique(),
        reverse=True,
    )
    noise_groups = [
        nd.loc[
            nd["snr_std"] == level_value,
            "delta_std",
        ].dropna().to_numpy(float)
        for level_value in levels
    ]

    plot_distribution(
        axs[1],
        noise_groups,
        [f"{v:g}" for v in levels],
        "Additive Gaussian-noise SNR (dB)",
        "(B) Noise robustness",
        "B",
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.17,
        top=0.91,
        wspace=0.28,
    )

    return save_figure(fig, out, "FIG10")

def exact_dropout_map_from_config(root: Path) -> dict | None:

    p = source_path(
        root,
        ARTIFACTS["14_ROBUSTNESS"]["config"]
    )

    if not p.is_file():
        return None

    try:
        obj = json.loads(
            p.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None

    text = json.dumps(obj).upper()

    # Parse the exact channel map if a recognizable map is stored.
    result = {}

    patterns = {
        5: ["CH07"],
        10: ["CH07"],
        20: ["CH01", "CH07", "CH13"],
        30: ["CH01", "CH05", "CH09", "CH13"],
    }

    for k, v in patterns.items():
        if str(k) in text:
            result[k] = v

    return result or None

# ---------------------------------------------------------------------------

# Supplementary figures

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dedicated supplementary renderers — no generic placeholder.
# ---------------------------------------------------------------------------
def _supp_subject_level(root):
    d=load_csv(source_path(root,ARTIFACTS["BLOCK15"]["subject_level"]))
    required={"condition","model","metric","value"}
    missing=required-set(d.columns)
    if missing: raise RuntimeError(f"Supplementary subject-level source missing {sorted(missing)}; columns={list(d.columns)}")
    return d

def supp01(root,out):
    d=_supp_subject_level(root); d=d[d["metric"].astype(str).map(metric_norm).eq("macro_f1")].copy(); d["value_std"]=pd.to_numeric(d["value"],errors="coerce"); d["model_std"]=d.model.map(normalize_model); d["condition_std"]=d.condition.map(normalize_condition)
    s=d.groupby(["condition_std","model_std"])["value_std"].mean().unstack("condition_std"); order=[m for m in CONFIG["models_ml"]+CONFIG["models_dl"] if m in s.index]; s=s.reindex(order)
    fig,ax=plt.subplots(figsize=(12,5)); s.plot(kind="bar",ax=ax,rot=45); ax.set_ylabel("Mean subject-level Macro-F1"); style_axes(ax); return save_figure(fig,out,"FIG-S01")

def supp02(root,out):
    d=_supp_subject_level(root); d=d[d["metric"].astype(str).map(metric_norm).eq("macro_f1")].copy(); d["value_std"]=pd.to_numeric(d["value"],errors="coerce"); branch_col=next((c for c in d.columns if norm_col(c) in {"branch","evaluation_regime","regime"}),None)
    if branch_col: d=d[d[branch_col].astype(str).str.upper().eq("LOSO")]
    s=d.groupby("model")["value_std"].mean().sort_values(ascending=False); fig,ax=plt.subplots(figsize=(11,5)); ax.bar(np.arange(len(s)),s.values); ax.set_xticks(np.arange(len(s))); ax.set_xticklabels(s.index,rotation=45,ha="right"); ax.set_ylabel("Mean LOSO Macro-F1"); style_axes(ax); return save_figure(fig,out,"FIG-S02")

def supp03(root,out):
    d=_supp_subject_level(root).copy(); d["metric_norm"]=d.metric.astype(str).map(norm_col); branch_col=next((c for c in d.columns if norm_col(c) in {"branch","evaluation_regime","regime"}),None);
    if branch_col: d=d[d[branch_col].astype(str).str.upper().eq("LOSO")]
    d["value_std"]=pd.to_numeric(d["value"],errors="coerce"); s=d.groupby("metric_norm")["value_std"].mean(); fig,ax=plt.subplots(figsize=(10,5)); ax.bar(np.arange(len(s)),s.values); ax.set_xticks(np.arange(len(s))); ax.set_xticklabels(s.index,rotation=30,ha="right"); ax.set_ylabel("Mean LOSO value"); style_axes(ax); return save_figure(fig,out,"FIG-S03")

def supp04(root,out):
    d=load_csv(source_path(root,ARTIFACTS["BLOCK15"]["gap"])); req={"condition","metric","gap"}; missing=req-set(d.columns);
    if missing: raise RuntimeError(f"FIG-S04 gap source missing {sorted(missing)}; columns={list(d.columns)}")
    d=d[d.metric.astype(str).map(norm_col).eq("macro_f1")].copy(); d["gap_std"]=pd.to_numeric(d["gap"],errors="coerce"); d["condition_std"]=d.condition.map(normalize_condition); fig,ax=plt.subplots(figsize=(10,5)); d.boxplot(column="gap_std",by="condition_std",ax=ax); ax.set_xlabel("Condition"); ax.set_ylabel("Stored SD − LOSO gap"); style_axes(ax); return save_figure(fig,out,"FIG-S04")

def supp05(root,out):
    d=load_csv(source_path(root,"09_TEMPORAL_CALIBRATION/metrics/calibration_summary.csv")); req={"condition","model","metric","calibration_fraction","mean_performance"}; missing=req-set(d.columns);
    if missing: raise RuntimeError(f"FIG-S05 calibration source missing {sorted(missing)}; columns={list(d.columns)}")
    d=d[d.metric.astype(str).map(norm_col).eq("macro_f1")].copy(); d["x"]=pd.to_numeric(d.calibration_fraction,errors="coerce"); d["y"]=pd.to_numeric(d.mean_performance,errors="coerce"); fig,ax=plt.subplots(figsize=(12,6));
    for (model,condition),g in d.groupby(["model","condition"]):
        g=g.sort_values("x"); ax.plot(g.x,g.y,marker="o",lw=1,label=f"{model} / {condition}")
    ax.set_xticks(CONFIG["calibration_grid"]); ax.set(xlabel="Calibration fraction (%)",ylabel="Stored mean Macro-F1"); ax.legend(frameon=False,ncol=3,fontsize=7); style_axes(ax); return save_figure(fig,out,"FIG-S05")

def supp06(root,out):
    d=load_csv(source_path(root,ARTIFACTS["09_TEMPORAL_CALIBRATION"]["k90_results"])); req={"subject_id","condition","model","metric","k90","k90_status"}; missing=req-set(d.columns);
    if missing: raise RuntimeError(f"FIG-S06 k90 source missing {sorted(missing)}; columns={list(d.columns)}")
    d=d[d.metric.astype(str).map(norm_col).eq("macro_f1")].copy(); d["status"]=d.k90_status.astype(str).str.upper(); s=d.groupby("status").size().sort_values(ascending=False); fig,ax=plt.subplots(figsize=(9,5)); ax.bar(np.arange(len(s)),s.values); ax.set_xticks(np.arange(len(s))); ax.set_xticklabels(s.index,rotation=25,ha="right"); ax.set_ylabel("Count"); style_axes(ax); return save_figure(fig,out,"FIG-S06")

def supp07(root,out):
    d=load_csv(source_path(root,ARTIFACTS["BLOCK15"]["transfer"])); req={"model","transfer_direction","metric","mean"}; missing=req-set(d.columns);
    if missing: raise RuntimeError(f"FIG-S07 transfer source missing {sorted(missing)}; columns={list(d.columns)}")
    d=d[d.metric.astype(str).map(norm_col).eq("macro_f1")].copy(); d["mean_std"]=pd.to_numeric(d["mean"],errors="coerce"); s=d.groupby("transfer_direction")["mean_std"].mean(); fig,ax=plt.subplots(figsize=(8,5)); ax.bar(np.arange(len(s)),s.values); ax.set_xticks(np.arange(len(s))); ax.set_xticklabels(s.index); ax.set_ylabel("Stored mean transfer benefit"); style_axes(ax); return save_figure(fig,out,"FIG-S07")

def supp08(root,out):
    d=load_csv(source_path(root,ARTIFACTS["BLOCK15"]["rsa_transfer"])); req={"subject_id","model","rho_spearman","bidirectional_mean_transfer_benefit"}; missing=req-set(d.columns);
    if missing: raise RuntimeError(f"FIG-S08 RSA-transfer source missing {sorted(missing)}; columns={list(d.columns)}")
    d["rho"]=pd.to_numeric(d.rho_spearman,errors="coerce"); d["benefit"]=pd.to_numeric(d.bidirectional_mean_transfer_benefit,errors="coerce"); fig,ax=plt.subplots(figsize=(8,5)); ax.scatter(d.rho,d.benefit,s=25); ax.set(xlabel="Stored RSA ρ",ylabel="Stored bidirectional transfer benefit"); style_axes(ax); return save_figure(fig,out,"FIG-S08")

def supp09(root,out):
    d=load_csv(source_path(root,ARTIFACTS["BLOCK15"]["rsa_calibration"])); req={"subject_id","model","rho_spearman","mean_k90","k90_valid"}; missing=req-set(d.columns);
    if missing: raise RuntimeError(f"FIG-S09 RSA-calibration source missing {sorted(missing)}; columns={list(d.columns)}")
    d["rho"]=pd.to_numeric(d.rho_spearman,errors="coerce"); d["k90"]=pd.to_numeric(d.mean_k90,errors="coerce"); valid=d.k90_valid.astype(str).str.upper().isin({"TRUE","1","YES","Y"}) & d.rho.notna() & d.k90.notna(); fig,ax=plt.subplots(figsize=(8,5)); ax.scatter(d.loc[valid,"rho"],d.loc[valid,"k90"],s=25); ax.set(xlabel="Stored RSA ρ",ylabel="Stored mean k90 (%)"); style_axes(ax); return save_figure(fig,out,"FIG-S09")

def supp10(root,out):
    d=load_csv(source_path(root,ARTIFACTS["13_XAI"]["ml_features"])); req={"feature_family","mean_abs_shap"}; missing=req-set(d.columns);
    if missing: raise RuntimeError(f"FIG-S10 ML XAI source missing {sorted(missing)}; columns={list(d.columns)}")
    d["shap"]=pd.to_numeric(d.mean_abs_shap,errors="coerce"); s=d.groupby("feature_family")["shap"].mean(); s=s.reindex([x for x in FEATURE_FAMILY_ORDER if x in s.index]); fig,ax=plt.subplots(figsize=(8,5)); ax.bar(np.arange(len(s)),s.values); ax.set_xticks(np.arange(len(s))); ax.set_xticklabels(s.index,rotation=25,ha="right"); ax.set_ylabel("Mean |SHAP| (stored)"); style_axes(ax); return save_figure(fig,out,"FIG-S10")

def supp11(root,out):
    d=load_csv(source_path(root,ARTIFACTS["13_XAI"]["attribution_similarity_statistics"])); req={"model","gesture","metric","n_subjects","mean","median","ci95_low","ci95_high","p_value_vs_zero","p_fdr"}; missing=req-set(d.columns);
    if missing: raise RuntimeError(f"FIG-S11 attribution source missing {sorted(missing)}; columns={list(d.columns)}")
    d["mean_std"]=pd.to_numeric(d["mean"],errors="coerce"); d=d[d["mean_std"].notna()].copy()
    s=d.groupby("metric",dropna=False)["mean_std"].mean().sort_values()
    if s.empty: raise RuntimeError("FIG-S11 attribution source contains no finite stored mean values")
    fig,ax=plt.subplots(figsize=(9,5)); ax.barh(np.arange(len(s)),s.values); ax.set_yticks(np.arange(len(s))); ax.set_yticklabels(s.index.astype(str)); ax.set_xlabel("Stored mean attribution similarity"); style_axes(ax); return save_figure(fig,out,"FIG-S11")

def supp12(root,out):
    d=load_csv(source_path(root,ARTIFACTS["14_ROBUSTNESS"]["dropout_subject_level"])); delta=next((c for c in d.columns if norm_col(c)=="delta_macro_f1"),None); level=next((c for c in d.columns if norm_col(c) in {"perturbation_level","dropout_percent","dropout_level"}),None); model=next((c for c in d.columns if norm_col(c)=="model"),None);
    if not all([delta,level,model]): raise RuntimeError(f"FIG-S12 dropout schema incomplete: {list(d.columns)}")
    d["delta_std"]=pd.to_numeric(d[delta],errors="coerce"); d["level_std"]=pd.to_numeric(d[level],errors="coerce"); d["model_std"]=d[model].map(normalize_model); s=d.groupby(["level_std","model_std"])["delta_std"].mean().unstack("model_std"); fig,ax=plt.subplots(figsize=(12,6)); s.plot(marker="o",ax=ax); ax.set(xlabel="Stored dropout level (%)",ylabel="Mean stored ΔP"); style_axes(ax); return save_figure(fig,out,"FIG-S12")

def supp13(root,out):
    d=load_csv(source_path(root,ARTIFACTS["14_ROBUSTNESS"]["robustness_subject_level"])); delta=next((c for c in d.columns if norm_col(c)=="delta_macro_f1"),None); level=next((c for c in d.columns if norm_col(c)=="perturbation_level"),None); model=next((c for c in d.columns if norm_col(c)=="model"),None); sem=next((c for c in d.columns if norm_col(c)=="noise_level_semantics"),None);
    if not all([delta,level,model,sem]): raise RuntimeError(f"FIG-S13 noise schema incomplete: {list(d.columns)}")
    semantics=set(d[sem].dropna().astype(str).str.lower());
    if semantics != {"snr_db"}: raise RuntimeError(f"FIG-S13 unexpected noise_level_semantics: {sorted(semantics)}")
    d["delta_std"]=pd.to_numeric(d[delta],errors="coerce"); d["level_std"]=pd.to_numeric(d[level],errors="coerce"); d["model_std"]=d[model].map(normalize_model); s=d.groupby(["level_std","model_std"])["delta_std"].mean().unstack("model_std"); fig,ax=plt.subplots(figsize=(12,6)); s.plot(marker="o",ax=ax); ax.set(xlabel="Stored synthetic-noise level (SNR dB)",ylabel="Mean stored ΔP"); style_axes(ax); return save_figure(fig,out,"FIG-S13")

def supp14(root,out):
    d=load_csv(source_path(root,ARTIFACTS["BLOCK15"]["computational"]));
    req={"source_block","metric","mean","median","n"}; missing=req-set(d.columns)
    if missing: raise RuntimeError(f"FIG-S14 computational schema incomplete: {sorted(missing)}; columns={list(d.columns)}")
    x=d.copy(); x["mean_std"]=pd.to_numeric(x["mean"],errors="coerce"); x["median_std"]=pd.to_numeric(x["median"],errors="coerce"); x["n_std"]=pd.to_numeric(x["n"],errors="coerce")
    x=x[x["mean_std"].notna()].copy()
    if x.empty: raise RuntimeError("FIG-S14 computational source contains no finite stored mean values")
    x["label"]=x["source_block"].astype(str)+" | "+x["metric"].astype(str)
    x=x.sort_values(["source_block","metric"],kind="stable").reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(12, max(5, 0.35*len(x)+2)))
    y=np.arange(len(x)); ax.barh(y,x["mean_std"].to_numpy()); ax.set_yticks(y); ax.set_yticklabels(x["label"].astype(str)); ax.set_xlabel("Stored mean"); style_axes(ax)
    for yi,val,n in zip(y,x["mean_std"],x["n_std"]):
        if pd.notna(n): ax.text(val,yi,f"  n={int(n)}",va="center",fontsize=7)
    return save_figure(fig,out,"FIG-S14")

SUPP_RENDERERS={
    "FIG-S01":supp01,"FIG-S02":supp02,"FIG-S03":supp03,"FIG-S04":supp04,
    "FIG-S05":supp05,"FIG-S06":supp06,"FIG-S07":supp07,"FIG-S08":supp08,
    "FIG-S09":supp09,"FIG-S10":supp10,"FIG-S11":supp11,"FIG-S12":supp12,
    "FIG-S13":supp13,"FIG-S14":supp14,
}

def resolve_root(cli_root: str | None) -> Path:

    # Resolution priority:

    #   1. Explicit --paper2-root

    #   2. PAPER2_ROOT environment variable

    #   3. Canonical project root inferred from this script location

    #      (…\Paper2_EEG_BCI\scripts\BLOCK16_FINAL_FIGURES.py)

    # The third path is deliberately deterministic and requires no user

    # configuration when the canonical script is run from the project.

    if cli_root:

        p = Path(cli_root).expanduser().resolve()

        if not p.is_dir():

            raise RuntimeError(f"--paper2-root does not exist: {p}")

        return p

    env_root = os.environ.get("PAPER2_ROOT")

    if env_root:

        p = Path(env_root).expanduser().resolve()

        if not p.is_dir():

            raise RuntimeError(f"PAPER2_ROOT does not exist: {p}")

        return p

    inferred = Path(__file__).resolve().parent.parent

    if inferred.is_dir():

        return inferred

    raise RuntimeError(

        "Paper 2 root could not be resolved. Supply --paper2-root or set PAPER2_ROOT."

    )

def _transfer_benefit_candidate_lines(root: Path, candidates: list, limit: int = 40) -> list:
    """Every source line that mentions transfer_benefit (case-insensitive): the evidence a human needs
    to see which formula BLOCK10 really implements when the automatic verification cannot decide."""
    out = []
    for p in candidates:
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for i, text in enumerate(lines, 1):
            if "transfer_benefit" in text.lower() or "transfer_gain" in text.lower():
                out.append({"file": str(p.relative_to(root)), "line": i, "text": text.strip()[:200]})
                if len(out) >= limit:
                    return out
    return out

def print_g8_diagnostics(gates: dict) -> None:
    g = gates.get("G8", {})
    fa = g.get("formula_audit", {}) or {}
    if g.get("pass") or "searched_files" not in fa:
        return
    print("  G8 diagnostics (nothing was relaxed; this is the evidence the verifier saw):")
    print(f"    canonical BLOCK10 script exists: {fa.get('canonical_script_exists')} ({fa.get('canonical_script_path')})")
    print(f"    files searched: {fa.get('searched_files')}")
    lines = fa.get("candidate_lines") or []
    if not lines:
        print("    no line mentioning transfer_benefit/transfer_gain was found in the searched files")
    for r in lines:
        print(f"    {r['file']}:{r['line']}: {r['text']}")

def audit_block10_transfer_formula(root: Path) -> dict:
    """Verify the frozen BLOCK10 transfer-benefit definition.

    Required semantics:
        transfer_benefit = source→target performance − target-native performance

    The verifier accepts:
      1. a direct subtraction assignment;
      2. the frozen intermediate implementation:
             diff = transfer_performance - native_target_performance
             transfer_benefit = diff

    This is source-code verification only.  BLOCK16 never recomputes or
    replaces the upstream transfer statistic.
    """
    # The canonical BLOCK10 implementation has a known script path.  Search it
    # first, then use a deterministic recursive fallback for archived/renamed
    # copies.  This avoids depending on which unrelated Python files happen to
    # contain the string "transfer_benefit".
    canonical = root / "scripts" / "BLOCK10_ME_MI_TRANSFER.py"
    candidates = []
    if canonical.is_file():
        candidates.append(canonical)

    for base in [root / "scripts", root]:
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("BLOCK10_ME_MI_TRANSFER.py")):
            if p not in candidates:
                candidates.append(p)

    if not candidates:
        # Conservative fallback: only inspect Python files that explicitly
        # contain transfer_benefit.
        for base in [root / "scripts", root]:
            if not base.is_dir():
                continue
            for p in sorted(base.rglob("*.py")):
                try:
                    txt0 = p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if "transfer_benefit" in txt0.lower() and p not in candidates:
                    candidates.append(p)

    def assignment_targets(node):
        """Names assigned by `node`: plain names, df["column"] string keys and obj.attribute targets."""
        if isinstance(node, ast.Assign):
            nodes = node.targets
        elif isinstance(node, ast.AnnAssign):
            nodes = [node.target]
        else:
            return []

        def one(t):
            if isinstance(t, ast.Name):
                return [t.id]
            if isinstance(t, ast.Attribute):
                return [t.attr]
            if isinstance(t, ast.Subscript):
                sl = t.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    return [sl.value]
                return []
            if isinstance(t, (ast.Tuple, ast.List)):
                return [n for x in t.elts for n in one(x)]
            return []

        out = []
        for t in nodes:
            out.extend(one(t))
        return out

    def normalized_expr(x: str) -> str:
        return re.sub(r"\s+", "", x).lower()

    _ME_MI_NATIVE_OF_TARGET = {("me_to_mi", "mi_to_mi"), ("mi_to_me", "me_to_me")}

    def is_transfer_minus_native(lhs_n: str, rhs_n: str) -> bool:
        """Required semantics: (source->target performance) MINUS (target-native performance).
        Operands are normalised source segments.  A reversed difference, a sum, or a source-native
        baseline (e.g. me_to_mi - me_to_me) is NOT accepted."""
        transfer_lhs = {"transfer_performance", "source_to_target_performance", "p_s_to_t"}
        native_rhs = {"native_target_performance", "target_native_performance", "p_t_to_t"}
        if lhs_n in transfer_lhs and rhs_n in native_rhs:
            return True
        if lhs_n in {"me_to_mi", "mi_to_me"} and rhs_n in {"me_to_me", "mi_to_mi"}:
            return (lhs_n, rhs_n) in _ME_MI_NATIVE_OF_TARGET     # native must be that of the TARGET condition
        lhs_ok = ("transfer" in lhs_n or "source_to_target" in lhs_n or lhs_n in {"me_to_mi", "mi_to_me"})
        rhs_ok = ("native" in rhs_n or "target_native" in rhs_n or rhs_n in {"t_to_t"})
        return bool(lhs_ok and rhs_ok and "native" not in lhs_n)

    def unwrap_value(v):
        """lambda d: d.a - d.b  ->  d.a - d.b"""
        while isinstance(v, ast.Lambda):
            v = v.body
        return v

    def strip_comments_and_strings(src: str):
        """Source with comments and string literals blanked, so a docstring/comment can never count as code."""
        import io as _io
        import tokenize as _tok
        try:
            out, last = [], (1, 0)
            lines = src.splitlines(keepends=True)
            blank = []
            for t in _tok.generate_tokens(_io.StringIO(src).readline):
                if t.type in (_tok.COMMENT, _tok.STRING):
                    blank.append((t.start, t.end))
            chars = [list(l) for l in lines]
            for (sr, sc), (er, ec) in blank:
                for r in range(sr, er + 1):
                    a = sc if r == sr else 0
                    b = ec if r == er else len(chars[r - 1])
                    for c in range(a, min(b, len(chars[r - 1]))):
                        if chars[r - 1][c] not in "\r\n":
                            chars[r - 1][c] = " "
            return "".join("".join(l) for l in chars)
        except Exception:
            return None

    evidence = []

    for p in candidates:
        try:
            source = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # First, perform an exact semantic text check for the frozen
        # intermediate implementation.  Whitespace and harmless parentheses
        # around the complete expression are ignored; operand order is not.
        _code_only = strip_comments_and_strings(source)
        compact = normalized_expr(_code_only) if _code_only is not None else ""
        intermediate_patterns = [
            (
                r"diff=\(?transfer_performance\)?-\(?native_target_performance\)?",
                r"transfer_benefit=diff\b",
            ),
        ]
        for diff_pat, benefit_pat in intermediate_patterns:
            if re.search(diff_pat, compact, flags=re.IGNORECASE) and re.search(
                benefit_pat, compact, flags=re.IGNORECASE
            ):
                # Recover line numbers for auditable evidence.
                diff_match = re.search(
                    r"diff\s*=\s*\(?transfer_performance\)?\s*-\s*\(?native_target_performance\)?",
                    _code_only,
                    flags=re.IGNORECASE,
                )
                benefit_match = re.search(
                    r"transfer_benefit\s*=\s*diff\b",
                    _code_only,
                    flags=re.IGNORECASE,
                )
                diff_line = source.count("\n", 0, diff_match.start()) + 1
                benefit_line = source.count("\n", 0, benefit_match.start()) + 1
                diff_line_text = source.splitlines()[diff_line - 1].strip()
                benefit_line_text = source.splitlines()[benefit_line - 1].strip()
                evidence.append({
                    "file": str(p.relative_to(root)),
                    "line": benefit_line,
                    "assignment": benefit_line_text,
                    "source_assignment": diff_line_text,
                    "source_line": diff_line,
                    "verification_mode": "verified_intermediate_subtraction",
                    "verified": True,
                })
                break

        # AST verification remains the primary structural check and supports
        # direct subtraction plus equivalent intermediate-variable assignment.
        try:
            tree = ast.parse(source, filename=str(p))
        except Exception:
            continue

        assignments = {}
        for node in ast.walk(tree):
            for name in assignment_targets(node):
                assignments.setdefault(name, []).append(node)

        subtraction_vars = {}
        for name, nodes in assignments.items():
            for node in nodes:
                value = getattr(node, "value", None)
                if not (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Sub)):
                    continue

                lhs = ast.get_source_segment(source, value.left) or ""
                rhs = ast.get_source_segment(source, value.right) or ""
                lhs_n = normalized_expr(lhs)
                rhs_n = normalized_expr(rhs)

                # Strict semantics: transfer/source->target MINUS target-native
                # (see is_transfer_minus_native for the exact accepted forms).
                if is_transfer_minus_native(lhs_n, rhs_n):
                    subtraction_vars.setdefault(name, node)

        for node in ast.walk(tree):
            targets = assignment_targets(node)
            if not any(t.lower() == "transfer_benefit" for t in targets):
                continue

            value = unwrap_value(getattr(node, "value", None))

            # Direct formula.
            if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Sub):
                lhs = normalized_expr(ast.get_source_segment(source, value.left) or "")
                rhs = normalized_expr(ast.get_source_segment(source, value.right) or "")
                if is_transfer_minus_native(lhs, rhs):
                    evidence.append({
                        "file": str(p.relative_to(root)),
                        "line": getattr(node, "lineno", None),
                        "assignment": ast.get_source_segment(source, node) or "",
                        "verification_mode": "direct_subtraction",
                        "verified": True,
                    })

            # Intermediate variable formula.
            elif isinstance(value, ast.Name) and value.id in subtraction_vars:
                subnode = subtraction_vars[value.id]
                evidence.append({
                    "file": str(p.relative_to(root)),
                    "line": getattr(node, "lineno", None),
                    "assignment": ast.get_source_segment(source, node) or "",
                    "source_assignment": ast.get_source_segment(source, subnode) or "",
                    "source_line": getattr(subnode, "lineno", None),
                    "verification_mode": "verified_intermediate_subtraction",
                    "verified": True,
                })

        # Same semantics when the quantity is built inside a dict literal or a call keyword:
        #   rows.append({"transfer_benefit": a - b})     DataFrame(...)/assign(transfer_benefit=a - b)
        def _check_named_value(name_node_line, value, mode):
            value = unwrap_value(value)
            if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Sub):
                lhs = normalized_expr(ast.get_source_segment(source, value.left) or "")
                rhs = normalized_expr(ast.get_source_segment(source, value.right) or "")
                if is_transfer_minus_native(lhs, rhs):
                    evidence.append({
                        "file": str(p.relative_to(root)), "line": name_node_line,
                        "assignment": ast.get_source_segment(source, value) or "",
                        "verification_mode": mode, "verified": True,
                    })
            elif isinstance(value, ast.Name) and value.id in subtraction_vars:
                subnode = subtraction_vars[value.id]
                evidence.append({
                    "file": str(p.relative_to(root)), "line": name_node_line,
                    "assignment": ast.get_source_segment(source, value) or "",
                    "source_assignment": ast.get_source_segment(source, subnode) or "",
                    "source_line": getattr(subnode, "lineno", None),
                    "verification_mode": "verified_intermediate_subtraction", "verified": True,
                })

        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value.lower() == "transfer_benefit":
                        _check_named_value(getattr(k, "lineno", None), v, "direct_subtraction_dict_entry")
            elif isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg and kw.arg.lower() == "transfer_benefit":
                        _check_named_value(getattr(kw.value, "lineno", None), kw.value, "direct_subtraction_keyword")

    # De-duplicate identical evidence produced by text + AST checks.
    unique = []
    seen = set()
    for item in evidence:
        key = (
            item.get("file"), item.get("line"),
            item.get("verification_mode"),
            item.get("assignment"), item.get("source_assignment"),
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)

    if unique:
        return {
            "pass": True,
            "reason": (
                "BLOCK10 source code verifies transfer_benefit = "
                "source→target performance − target-native performance"
            ),
            "evidence": unique,
        }

    return {
        "pass": False,
        "reason": (
            "No verifiable BLOCK10 transfer_benefit formula with "
            "source→target minus target-native semantics was found"
        ),
        "evidence": [],
        "searched_files": [str(p.relative_to(root)) for p in candidates],
        "canonical_script_exists": bool(canonical.is_file()),
        "canonical_script_path": str(canonical),
        "candidate_lines": _transfer_benefit_candidate_lines(root, candidates),
    }

def _matrix_source_hash_audit(root: Path, matrix: pd.DataFrame) -> tuple[bool, str, pd.DataFrame]:
    """Verify every frozen matrix row against its recorded source identity fields."""
    required = {"figure_id", "panel_id", "source_file", "source_sha256"}
    missing = required - set(matrix.columns)
    if missing:
        return False, f"feasibility matrix missing required identity columns: {sorted(missing)}", pd.DataFrame()

    records = []
    for _, row in matrix.iterrows():
        files = [x.strip() for x in str(row["source_file"]).split("|") if x.strip()]
        hashes = [x.strip().lower() for x in str(row["source_sha256"]).split("|") if x.strip()]
        if len(files) != len(hashes):
            records.append({
                "figure_id": row["figure_id"], "panel_id": row["panel_id"],
                "status": "FAIL", "reason": "source_file/source_sha256 cardinality mismatch"
            })
            continue

        row_ok = True
        reasons = []
        for rel, expected_hash in zip(files, hashes):
            p = source_path(root, rel)
            if not p.is_file():
                row_ok = False
                reasons.append(f"missing:{rel}")
                continue
            actual = sha256_file(p)
            if actual.lower() != expected_hash:
                row_ok = False
                reasons.append(f"hash_mismatch:{rel}")

        records.append({
            "figure_id": row["figure_id"],
            "panel_id": row["panel_id"],
            "status": "PASS" if row_ok else "FAIL",
            "reason": "source identity verified" if row_ok else "; ".join(reasons),
        })

    report = pd.DataFrame(records)
    ok = len(report) == len(matrix) and bool((report["status"] == "PASS").all())
    reason = (
        f"all {len(matrix)} frozen matrix rows have existing sources with matching SHA256"
        if ok else
        f"{int((report['status'] != 'PASS').sum())} of {len(matrix)} frozen matrix rows failed source identity verification"
    )
    return ok, reason, report

def audit_release_gates(root: Path, b14: dict, source_audit: pd.DataFrame) -> dict:
    """Audit release gates without manufacturing upstream evidence."""
    gates = {}

    cfg14 = b14.get("config", {})
    g2_pass = bool(
        cfg14.get("exists")
        and cfg14.get("data")
        and cfg14.get("hash_matches_marker")
    )
    if g2_pass:
        g2_reason = (
            f"BLOCK14_config.json verified at {cfg14.get('path')} and SHA256 "
            "matches completion-marker config_sha256"
        )
    elif not cfg14.get("exists"):
        g2_reason = (
            "actual BLOCK14_config.json was not found anywhere under the project root; "
            "marker hash is provenance only"
        )
    elif not cfg14.get("raw_hash_matches_input_manifest"):
        g2_reason = (
            f"BLOCK14 config found at {cfg14.get('path')}, but raw SHA256 does not match "
            "the completion-marker input_hashes.BLOCK14_config"
        )
    elif not cfg14.get("canonical_hash_matches_marker"):
        g2_reason = (
            f"BLOCK14 config found at {cfg14.get('path')}; raw file hash matches the input manifest, "
            "but canonical JSON hash does not match completion-marker config_sha256"
        )
    else:
        g2_reason = "BLOCK14_config.json was readable but could not be verified against the completion marker"
    gates["G2"] = {"pass": g2_pass, "reason": g2_reason}

    # G4: frozen subject-level BLOCK09 k90 source.
    k90_path = source_path(root, ARTIFACTS["09_TEMPORAL_CALIBRATION"]["k90_results"])
    g4, g4_reason = False, "BLOCK09 k90_results.csv unavailable"
    if k90_path.is_file():
        try:
            kd = pd.read_csv(k90_path)
            required = {"subject_id", "condition", "model", "metric", "k90", "k90_status"}
            missing = required - set(kd.columns)
            if missing:
                g4_reason = f"BLOCK09 k90_results missing required columns: {sorted(missing)}"
            else:
                q = kd.copy()
                q["metric_norm"] = q["metric"].astype(str).map(norm_col)
                q = q[q["metric_norm"].eq("macro_f1")].copy()
                key = ["subject_id", "condition", "model"]
                dup = int(q.duplicated(key).sum())
                observed = {str(x).strip().upper() for x in q["k90_status"].dropna().unique()}
                allowed = {"VALID", "PASS", "NOT_REACHED", "NO_POSITIVE_GAP", "OK"}
                bad_status = observed - allowed
                expected_n = 25 * 2 * 10
                finite_k90 = pd.to_numeric(q["k90"], errors="coerce")
                invalid_numeric = int((
                    ~np.isfinite(finite_k90)
                    & ~q["k90_status"].astype(str).str.upper().isin(
                        {"NOT_REACHED", "NO_POSITIVE_GAP"}
                    )
                ).sum())
                g4 = (
                    len(q) == expected_n
                    and dup == 0
                    and not bad_status
                    and invalid_numeric == 0
                    and bool(observed)
                )
                g4_reason = (
                    f"BLOCK09 k90 status/censoring categories verified; N={len(q)}, "
                    f"duplicates={dup}, statuses={sorted(observed)}, "
                    f"NOT_REACHED={int((q['k90_status'].astype(str).str.upper() == 'NOT_REACHED').sum())}, "
                    f"NO_POSITIVE_GAP={int((q['k90_status'].astype(str).str.upper() == 'NO_POSITIVE_GAP').sum())}"
                    if g4 else
                    f"BLOCK09 k90 audit failed: N={len(q)} expected={expected_n}, "
                    f"duplicates={dup}, unexpected_statuses={sorted(bad_status)}, "
                    f"invalid_numeric={invalid_numeric}"
                )
        except Exception as exc:
            g4_reason = f"BLOCK09 k90 audit error: {exc}"
    gates["G4"] = {"pass": bool(g4), "reason": g4_reason}

    # G5: the authoritative frozen source audit already stored in the project
    # is the release identity record. Do not require a separate feasibility CSV.
    source_audit_path = root / "16_FINAL_FIGURES" / "audit" / "BLOCK16_source_audit.csv"
    g5, g5_reason = False, "authoritative BLOCK16_source_audit.csv unavailable"
    matrix_report = pd.DataFrame()
    matrix_rows = 0

    if source_audit_path.is_file():
        try:
            fm = pd.read_csv(source_audit_path)
            matrix_rows = len(fm)
            required_identity = {
                "figure_id", "panel_id", "source_file", "source_sha256",
                "required_columns", "columns_present", "expected_N",
                "observed_N", "finite_check", "missing_key_count",
                "duplicate_key_count", "stored_quantity", "status"
            }
            missing_identity = required_identity - set(fm.columns)

            if missing_identity:
                g5_reason = (
                    "BLOCK16_source_audit.csv missing required frozen-audit columns: "
                    f"{sorted(missing_identity)}"
                )
            elif matrix_rows == 0:
                g5_reason = "BLOCK16_source_audit.csv contains no audit rows"
            else:
                identity_ok, identity_reason, matrix_report = _matrix_source_hash_audit(root, fm)

                status_ok = bool((fm["status"].astype(str).str.upper() == "AVAILABLE").all())
                columns_ok = bool(fm["columns_present"].astype(str).str.upper().isin({"TRUE", "1", "YES"}).all())
                finite_ok = bool(fm["finite_check"].astype(str).str.upper().isin({"TRUE", "1", "YES"}).all())
                missing_keys_ok = bool(pd.to_numeric(fm["missing_key_count"], errors="coerce").eq(0).all())
                duplicate_keys_ok = bool(pd.to_numeric(fm["duplicate_key_count"], errors="coerce").eq(0).all())
                expected_series = pd.to_numeric(fm["expected_N"], errors="coerce")
                observed_series = pd.to_numeric(fm["observed_N"], errors="coerce")

                # Rows with expected_N=None are intentionally descriptive/source
                # audits without a frozen row-count requirement (currently FIG09
                # and FIG10).  They must not be treated as N mismatches because
                # NaN != NaN in pandas.  Enforce equality only where an expected
                # N is actually specified.
                n_specified = expected_series.notna()
                n_ok = bool(
                    expected_series.loc[n_specified].eq(
                        observed_series.loc[n_specified]
                    ).all()
                )

                g5 = bool(
                    identity_ok and status_ok and columns_ok and finite_ok
                    and missing_keys_ok and duplicate_keys_ok and n_ok
                )

                if g5:
                    g5_reason = (
                        f"authoritative BLOCK16_source_audit.csv verified: {matrix_rows} "
                        "frozen source rows exist, hashes match, schema/size/key/finite checks pass"
                    )
                else:
                    reasons = [identity_reason]
                    if not status_ok: reasons.append("source-audit status failure")
                    if not columns_ok: reasons.append("columns_present failure")
                    if not finite_ok: reasons.append("finite_check failure")
                    if not missing_keys_ok: reasons.append("missing_key_count is nonzero")
                    if not duplicate_keys_ok: reasons.append("duplicate_key_count is nonzero")
                    if not n_ok: reasons.append("expected_N differs from observed_N")
                    g5_reason = "; ".join(reasons)
        except Exception as exc:
            g5_reason = f"BLOCK16_source_audit.csv audit error: {exc}"

    if not matrix_report.empty:
        matrix_report.to_csv(
            root / "16_FINAL_FIGURES" / "audit" / "BLOCK16_feasibility_identity_audit.csv",
            index=False,
        )

    gates["G5"] = {
        "pass": bool(g5),
        "reason": g5_reason,
        "matrix_rows": matrix_rows,
        "source_audit": str(source_audit_path.relative_to(root)) if source_audit_path.is_file() else None,
    }

    # G7 is the same strict BLOCK14 evidence gate used by FIG10.
    gates["G7"] = {
        "pass": bool(b14.get("strict_pass")),
        "reason": (
            "BLOCK14 strict completion + clean-reference audit passed"
            if b14.get("strict_pass")
            else "BLOCK14 strict completion/clean-reference audit failed"
        ),
    }

    # G8: artifact schema plus direct source-code formula verification.
    tb_path = source_path(root, ARTIFACTS["10_ME_MI_TRANSFER"]["transfer_benefit"])
    g8 = False
    formula_audit = {"pass": False, "reason": "not audited"}
    if not tb_path.is_file():
        g8_reason = "BLOCK10 transfer_benefit.csv is absent/unreadable"
    else:
        try:
            tb = pd.read_csv(tb_path)
            required_tb = {
                "subject_id", "model", "transfer_direction",
                "metric", "transfer_benefit"
            }
            missing_tb = required_tb - set(tb.columns)
            if missing_tb:
                g8_reason = f"BLOCK10 transfer_benefit.csv missing columns: {sorted(missing_tb)}"
            else:
                q = tb.copy()
                q["metric_norm"] = q["metric"].astype(str).map(norm_col)
                q = q[q["metric_norm"].eq("macro_f1")].copy()
                dup = int(q.duplicated(["subject_id", "model", "transfer_direction"]).sum())
                finite = pd.to_numeric(q["transfer_benefit"], errors="coerce")
                invalid = int((~np.isfinite(finite)).sum())
                expected_tb = 25 * 10 * 2
                schema_ok = len(q) == expected_tb and dup == 0 and invalid == 0
                if not schema_ok:
                    g8_reason = (
                        f"BLOCK10 transfer artifact schema audit failed: N={len(q)} "
                        f"(expected {expected_tb}), duplicates={dup}, nonfinite={invalid}"
                    )
                else:
                    formula_audit = audit_block10_transfer_formula(root)
                    g8 = bool(formula_audit["pass"])
                    g8_reason = formula_audit["reason"]
        except Exception as exc:
            g8_reason = f"BLOCK10 transfer_benefit audit error: {exc}"

    gates["G8"] = {
        "pass": bool(g8),
        "reason": g8_reason,
        "formula_audit": formula_audit,
    }

    gates["G9"] = {
        "pass": all(bool(v) for v in FIGURE_STATUS.values()),
        "reason": "frozen figure-status map is present",
    }
    return gates

def write_config(out_dir: Path):

    cfg=dict(CONFIG)

    cfg["config_hash"]=config_hash()

    cfg["generated_utc"]=utc_now()

    write_json(out_dir/"BLOCK16_config.json",cfg)

def run(root: Path, inspect_only: bool = False) -> int:

    out_root=root/"16_FINAL_FIGURES"

    audit_dir=out_root/"audit"

    fig_dir=out_root/"figures"

    supp_dir=out_root/"supplementary_figures"

    out_root.mkdir(parents=True,exist_ok=True)

    audit_dir.mkdir(parents=True,exist_ok=True)

    write_config(out_root)

    print(f"BLOCK16 started | root={root}")

    print(f"CONFIG_HASH={config_hash()}")

    source_audit=audit_sources(root)

    source_audit.to_csv(audit_dir/"BLOCK16_source_audit.csv",index=False)

    b14=audit_block14(root)

    write_json(audit_dir/"BLOCK14_release_audit.json",b14)

    # Store hashes for all directly inspected local inputs.

    provenance=[]

    spec=Path(__file__).with_name("BLOCK16_FINAL_FIGURE_STRUCTURE_FROZEN.md")

    if spec.is_file():

        provenance.append({"artifact":"BLOCK16_FINAL_FIGURE_STRUCTURE_FROZEN.md",

                           "path":str(spec),"sha256":sha256_file(spec),"status":"VERIFIED"})

    for block,items in ARTIFACTS.items():

        if block in {"BLOCK15"}:

            continue

        for name,rel in items.items():

            p=source_path(root,rel)

            if p.is_file():

                provenance.append({"artifact":f"{block}:{name}","path":str(p),

                                   "sha256":sha256_file(p),"status":"VERIFIED"})

    write_json(audit_dir/"BLOCK16_provenance.json",provenance)

    if inspect_only:

        print(source_audit.to_string(index=False))

        print(json.dumps(b14, indent=2))

        gates = audit_release_gates(root, b14, source_audit)

        print("BLOCK16 RELEASE GATES:")

        for gid in ["G2", "G4", "G5", "G7", "G8", "G9"]:
            print(f"  {gid}: {'PASS' if gates[gid]['pass'] else 'OPEN'} — {gates[gid]['reason']}")
        print_g8_diagnostics(gates)

        print("BLOCK16 INSPECTION COMPLETE")

        return 0

    # Figure generation is conservative: any unavailable source blocks only

    # the affected figure. It is never replaced with a newly computed inference.

    renderers={

        "FIG01":fig01,"FIG02":fig02,"FIG03":fig03,"FIG04":fig04,

        "FIG05":fig05,"FIG06":fig06,"FIG07":fig07,"FIG08":fig08,"FIG09":fig09,

    }

    records=[]

    errors=[]

    for fig_id,renderer in renderers.items():

        try:

            files=renderer(root,fig_dir)

            records.append({

                "figure_id":fig_id,"status":FIGURE_STATUS[fig_id],

                "release_status":"RENDERED","files":json.dumps(files),

                "error":"",

            })

            print(f"{fig_id}: RENDERED")

        except Exception as exc:

            records.append({

                "figure_id":fig_id,"status":FIGURE_STATUS[fig_id],

                "release_status":"UNAVAILABLE","files":"",

                "error":str(exc),

            })

            errors.append((fig_id,str(exc)))

            print(f"{fig_id}: UNAVAILABLE — {exc}")

    try:

        files=fig10(root,fig_dir,b14)

        records.append({"figure_id":"FIG10","status":FIGURE_STATUS["FIG10"],

                        "release_status":"RENDERED","files":json.dumps(files),"error":""})

        print("FIG10: RENDERED")

    except Exception as exc:

        records.append({"figure_id":"FIG10","status":FIGURE_STATUS["FIG10"],

                        "release_status":"UNAVAILABLE","files":"","error":str(exc)})

        errors.append(("FIG10",str(exc)))

        print(f"FIG10: UNAVAILABLE — {exc}")

    # Supplementary figures are generated only from their directly named source.

    for sid in CONFIG["supplementary_figures"]:

        try:

            renderer = SUPP_RENDERERS.get(sid)
            if renderer is None:
                raise RuntimeError(f"{sid}: dedicated source-specific renderer is not registered")
            files=renderer(root,supp_dir)

            records.append({"figure_id":sid,"status":"SUPPLEMENTARY",

                            "release_status":"RENDERED","files":json.dumps(files),"error":""})

            print(f"{sid}: RENDERED")

        except Exception as exc:

            records.append({"figure_id":sid,"status":"SUPPLEMENTARY",

                            "release_status":"UNAVAILABLE","files":"","error":str(exc)})

            errors.append((sid,str(exc)))

    registry=pd.DataFrame(records)

    registry.to_csv(audit_dir/"BLOCK16_figure_release_registry.csv",index=False)

    # Per-figure PDFs are authoritative. Do not create an empty combined PDF;
    # merging existing PDF pages is outside the frozen BLOCK16 generation boundary.

    # Strict release status: artifact-dependent gates may only close when
    # their frozen evidence requirements are actually verified.
    all_main = all(
        r["release_status"] == "RENDERED"
        for r in records
        if r["figure_id"] in CONFIG["main_figures"]
    )
    gates = audit_release_gates(root, b14, source_audit)
    gate_g2 = gates["G2"]["pass"]
    gate_g4 = gates["G4"]["pass"]
    gate_g5 = gates["G5"]["pass"]
    gate_g7 = gates["G7"]["pass"]
    gate_g8 = gates["G8"]["pass"]
    gate_g9 = gates["G9"]["pass"]
    release_pass = bool(
        all_main and gate_g2 and gate_g4 and gate_g5 and gate_g7 and gate_g8 and gate_g9
    )

    completion={

        "block":"BLOCK16",

        "status":"PASS" if release_pass else "NOT_READY",

        "generated_utc":utc_now(),

        "config_hash":config_hash(),

        "all_main_figures_rendered":all_main,

        "BLOCK14_strict_release_pass":gate_g7,

        "release_gates":gates,

        "errors":errors,

        "source_audit":"audit/BLOCK16_source_audit.csv",

        "figure_registry":"audit/BLOCK16_figure_release_registry.csv",

        "provenance":"audit/BLOCK16_provenance.json",

        "scientific_boundary":"No new inferential statistics computed.",

    }

    write_json(out_root/"BLOCK16_complete.json",completion)

    write_json(out_root/("BLOCK16_PASS.json" if release_pass else "BLOCK16_NOT_READY.json"),completion)

    print("BLOCK16 RELEASE GATES:")
    for gid in ["G2", "G4", "G5", "G7", "G8", "G9"]:
        print(f"  {gid}: {'PASS' if gates[gid]['pass'] else 'OPEN'} — {gates[gid]['reason']}")
    print_g8_diagnostics(gates)
    print(f"BLOCK16 {'PASS' if release_pass else 'NOT_READY'}")

    if errors:

        print("Unresolved/unavailable figures:")

        for fig_id,msg in errors:

            print(f"  {fig_id}: {msg}")

    return 0 if release_pass else 2

def main():

    ap=argparse.ArgumentParser()

    ap.add_argument("--paper2-root",default=None)

    ap.add_argument("--inspect",action="store_true")

    ap.add_argument("--run",action="store_true")

    args=ap.parse_args()

    if not args.inspect and not args.run:

        args.inspect=True

    try:

        root=resolve_root(args.paper2_root)

        return run(root,inspect_only=args.inspect)

    except Exception as exc:

        print(f"BLOCK16 FATAL: {exc}",file=sys.stderr)

        traceback.print_exc()

        return 1

if __name__=="__main__":

    raise SystemExit(main())
