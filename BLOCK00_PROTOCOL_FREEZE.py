
#!/usr/bin/env python3

"""
BLOCK00_PROTOCOL_FREEZE.py

Paper 2
Shared and Condition-Specific EEG Representations of Hand and Finger Gestures:
Subject-Dependent and Cross-Subject Decoding, Motor Execution–Imagery Transfer,
and Temporal Calibration.

BLOCK 00 — Protocol Freeze

IMPORTANT
---------
This block freezes the analysis protocol only.

It DOES NOT:
- inspect raw EEG
- modify raw EEG
- preprocess EEG
- generate computational EEG windows
- extract features
- train models
- evaluate models
- perform statistical testing
- generate scientific figures

Actual subject/file discovery starts in BLOCK 01.
"""


from __future__ import annotations

import argparse
import hashlib
import json

from datetime import datetime, timezone

from pathlib import Path


# ============================================================
# BLOCK IDENTITY
# ============================================================

BLOCK_ID = "BLOCK00"

PROTOCOL_VERSION = (
    "PAPER2-BLOCK00-FREEZE-1.0"
)

PAPER_TITLE = (
    "Shared and Condition-Specific EEG Representations of Hand and Finger "
    "Gestures: Subject-Dependent and Cross-Subject Decoding, "
    "Motor Execution–Imagery Transfer, and Temporal Calibration"
)

DEFAULT_RAW_DRIVE_URL = (
    "https://drive.google.com/drive/folders/"
    "1JKIIqykbHm2l3q7gTTUBf5CHhI4EQwyl"
)


# ============================================================
# PRIMARY COHORT
# ============================================================

PRIMARY_SUBJECT_COUNT = 25

CONDITIONS = [
    "ME",
    "MI",
]

SETS = [
    "A",
    "B",
    "C",
]

GESTURES = {

    1: "HOC",

    2: "IFM",

    3: "LFM",

    4: "MFM",

    5: "PH",

    6: "RFM",

    7: "TFM",
}


# ============================================================
# EEG ACQUISITION
# ============================================================

EEG_CHANNEL_COUNT = 13

SAMPLING_RATE_HZ = 125

NOMINAL_DURATION_SECONDS = 45.0

NOMINAL_SAMPLE_COUNT = 5625


# ============================================================
# PREPROCESSING
# ============================================================

PREPROCESSING = {

    "bandpass_hz": [
        1.0,
        40.0
    ],

    "filter_type":
        "Butterworth band-pass",

    "filter_order":
        4,

    "phase":
        "zero-phase offline",

    "sampling_rate_hz":
        125,

    "resampling":
        False,

    "preserve_channel_count":
        True,

    "preserve_sample_count":
        True,

    "extra_notch_filter_default":
        False,

    "arbitrary_permanent_rereference":
        False,

    "automatic_bad_channel_interpolation":
        False,

    "ica_default":
        False,

    "automatic_artifact_rejection":
        False,

    "global_normalization":
        False,

    "feature_selection":
        False,

    "dimensionality_reduction":
        False,

    "class_balancing":
        False,

    "smote":
        False,

    "trial_reconstruction":
        False,
}


# ============================================================
# WINDOWING
# ============================================================

WINDOW_LENGTHS_SAMPLES = [

    200,

    256,

    300,

    400,

    500,
]

WINDOW_OVERLAPS_PERCENT = [

    0,

    25,

    50,

    75,
]

HISTORICAL_WINDOW = {

    "length_samples":
        250,

    "overlap_percent":
        50,

    "role":
        "diagnostic_compatibility_only",

    "primary_candidate":
        False,
}


# ============================================================
# MACHINE LEARNING
# ============================================================

ML_MODELS = [

    "LDA",

    "KNN",

    "SVM",

    "RandomForest",

    "XGBoost",
]

ML_FEATURES = {

    "total":
        341,

    "channel_level_feature_families":
        24,

    "channel_level_features":
        312,

    "channel_mean_features":
        13,

    "channel_sd_features":
        13,

    "global_features":
        3,

    "formula":
        "13*24 + 13 + 13 + 3 = 341",

    "exact_24_feature_family_names":
        None,

    "exact_feature_family_status":
        "MUST_BE_FROZEN_IN_BLOCK05",

    "logistic_regression":
        False,

    "smote":
        False,
}


# ============================================================
# DEEP LEARNING
# ============================================================

DL_MODELS = [

    "EEGNet",

    "ShallowConvNet",

    "DeepConvNet",

    "CNN-BiLSTM",

    "CNN-BiLSTM+Self-Attention",
]


# ============================================================
# EVALUATION
# ============================================================

EVALUATION = {

    "primary_metrics": [

        "Accuracy",

        "Balanced Accuracy",

        "Macro-F1",
    ],

    "secondary_metrics": [

        "MCC",

        "Cohen's kappa",
    ],

    "class_level_metrics": [

        "Precision",

        "Recall",

        "F1",
    ],

    "confusion_matrix":
        True,

    "inferential_unit":
        "subject",

    "subject_dependent":
        True,

    "loso": {

        "folds":
            25,

        "training_subjects":
            24,

        "unseen_test_subjects":
            1,

        "test_subject_used_for_tuning":
            False,
    },
}


# ============================================================
# CHANCE LEVEL
# ============================================================

CHANCE_LEVEL = {

    "number_of_classes":
        7,

    "accuracy":
        1.0 / 7.0,

    "accuracy_percent":
        100.0 / 7.0,
}


# ============================================================
# ME–MI PAIRED STATISTICS
# ============================================================

ME_MI_STATISTICS = {

    "inferential_unit":
        "subject",

    "paired_difference":
        "D_i = A_ME,i - A_MI,i",

    "effect_size":
        "paired standardized mean difference d_z",

    "bootstrap": {

        "resamples":
            10000,

        "confidence_interval":
            0.95,

        "unit":
            "subject",

        "paired":
            True,
    },

    "primary_test":
        "two-sided paired permutation/sign-flip",

    "permutation_resamples":
        10000,

    "wilcoxon":
        "optional sensitivity analysis only",

    "multiple_comparison_correction":
        "Benjamini-Hochberg FDR",

    "fdr_q":
        0.05,

    "maximum_planned_tests":
        100,

    "window_level_inference":
        False,

    "pseudoreplication":
        False,
}


# ============================================================
# TEMPORAL CALIBRATION
# ============================================================

CALIBRATION_FRACTIONS_PERCENT = [

    0,

    10,

    20,

    30,

    40,

    50,

    60,

    80,

    90,
]

TEMPORAL_CALIBRATION = {

    "fractions_percent":
        CALIBRATION_FRACTIONS_PERCENT,

    "calibration_region":
        "earliest contiguous target recording portion",

    "evaluation_region":
        "later unseen target recording portion",

    "random_calibration":
        False,

    "physical_sample_embargo":
        True,

    "crossing_windows_excluded":
        True,

    "primary_metric":
        "Macro-F1",

    "secondary_metric":
        "Balanced Accuracy",

    "calibration_efficiency":
        "CE(k) = [A(k) - A(0)] / k",

    "k90":
        (
            "min{k: A_i(k) >= A_i(0) + "
            "0.90*(A_SD,i - A_i(0))}"
        ),

    "k90_interpolation":
        False,

    "k90_smoothing":
        False,

    "k90_no_positive_gap":
        "NO_POSITIVE_GAP",

    "k90_not_reached":
        "NOT_REACHED",

    "adaptation_operator":
        None,

    "adaptation_operator_status":
        "MUST_BE_FROZEN_IN_BLOCK09",
}


# ============================================================
# BIDIRECTIONAL ME–MI TRANSFER
# ============================================================

TRANSFER = {

    "directions": [

        "ME->MI",

        "MI->ME",
    ],

    "pure_transfer_target_label_adaptation":
        False,

    "target_native_reference_required":
        True,

    "transfer_difference":
        "A_source_to_target - A_target_to_target",

    "transfer_loss":
        "A_target_to_target - A_source_to_target",

    "negative_transfer":
        "A_source_to_target < A_target_to_target",

    "asymmetry":
        "L_ME_to_MI - L_MI_to_ME",
}


# ============================================================
# LATENT REPRESENTATION / RSA
# ============================================================

REPRESENTATION = {

    "primary_models": [

        "EEGNet",

        "CNN-BiLSTM",

        "CNN-BiLSTM+Self-Attention",
    ],

    "latent_layer":
        None,

    "latent_layer_status":
        "MUST_BE_FROZEN_BEFORE_BLOCK12",

    "aggregation":
        "gesture-level mean latent vector",

    "rdm_shape": [

        7,

        7,
    ],

    "unique_upper_triangle_pairs":
        21,

    "rdm_dissimilarity":
        "correlation distance = 1 - Pearson r",

    "rdm_similarity":
        "Spearman correlation of 21 upper-triangle entries",

    "subject_similarity":
        "rho_i = Spearman(vec(RDM_ME,i), vec(RDM_MI,i))",

    "no_pooled_window_inference":
        True,
}


# ============================================================
# NEUROPHYSIOLOGY
# ============================================================

NEUROPHYSIOLOGY = {

    "methods": [

        "PSD",

        "band power",

        "relative band power",

        "alpha/beta ratio",
    ],

    "true_erd_ers":
        False,

    "erd_ers_requires_verified_rest_baseline":
        True,

    "anatomical_topography_requires_verified_channel_geometry":
        True,
}


# ============================================================
# XAI
# ============================================================

XAI = {

    "ml_method":
        "SHAP",

    "ml_models":
        ML_MODELS,

    "dl_methods": [

        "Integrated Gradients",

        "occlusion",
    ],

    "dl_models":
        DL_MODELS,

    "frozen_models_only":
        True,

    "causal_neurophysiology_claims":
        False,

    "anatomical_source_claims":
        False,
}


# ============================================================
# ROBUSTNESS
# ============================================================

ROBUSTNESS = {

    "channel_dropout_percent": [

        5,

        10,

        20,

        30,
    ],

    "noise":
        "predefined noise condition",

    "noise_parameters":
        None,

    "noise_parameters_status":
        "MUST_BE_FROZEN_BEFORE_BLOCK14",
}


# ============================================================
# PIPELINE ORDER
# ============================================================

PIPELINE = [

    "00_PROTOCOL",

    "01_SUBJECT_MATCHING",

    "02_SIGNAL_QA",

    "03_PREPROCESSING",

    "04_WINDOWING",

    "05_FEATURE",

    "06A_ML_SD",

    "06B_ML_LOSO",

    "07A_DL_SD",

    "07B_DL_LOSO",

    "08_ME_MI_STATISTICS",

    "09_TEMPORAL_CALIBRATION",

    "10_ME_MI_TRANSFER",

    "11_NEURO",

    "12_REPRESENTATION",

    "13_XAI",

    "14_ROBUSTNESS",

    "15_FINAL_STATISTICS",

    "16_FINAL_FIGURES",

    "17_FINAL_TABLES",

    "18_FINAL_AUDIT",

    "MANUSCRIPT",
]


# ============================================================
# NO-GO RULES
# ============================================================

NO_GO_RULES = [

    "No SMOTE.",

    "No Logistic Regression.",

    "No random calibration.",

    "No reconstruction of undocumented trials.",

    "No artificial rest baseline.",

    "No fabricated repetition count.",

    "No test-data leakage.",

    "No test-set tuning.",

    "No post-hoc model selection.",

    "No post-hoc window/overlap selection.",

    "No window-level pseudoreplication.",

    "No performance-based subject exclusion.",

    "No unsupported bad-channel interpolation.",

    "No unsupported anatomical/electrode localization claims.",

    "No result-driven protocol changes.",
]


# ============================================================
# EXPECTED RECORDINGS
# ============================================================

EXPECTED_RECORDINGS = {

    "per_subject_per_condition":
        7 * 3,

    "per_complete_paired_subject":
        2 * 7 * 3,

    "complete_me_recordings":
        25 * 7 * 3,

    "complete_mi_recordings":
        25 * 7 * 3,

    "complete_paired_recordings":
        25 * 2 * 7 * 3,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def build_reproducibility(seed):

    return {

        "global_seed":
            seed,

        "seed_status": (
            "FROZEN_BY_BLOCK00"
            if seed is not None
            else
            "MUST_BE_FROZEN_BEFORE_STOCHASTIC_DOWNSTREAM_EXECUTION"
        ),

        "record_python_version":
            True,

        "record_package_versions":
            True,

        "record_cuda_pytorch":
            True,

        "record_hardware":
            True,

        "record_configs":
            True,

        "record_fold_assignments":
            True,

        "record_calibration_partitions":
            True,

        "record_transfer_direction":
            True,

        "record_file_hashes":
            True,
    }


# ============================================================
# TIME
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# SHA256
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# JSON WRITER
# ============================================================

def write_json(path, obj):

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    temp.replace(path)


# ============================================================
# PROTOCOL VALIDATION
# ============================================================

def validate_protocol():

    errors = []

    if PRIMARY_SUBJECT_COUNT != 25:

        errors.append(
            "Primary cohort must contain 25 paired subjects."
        )

    if CONDITIONS != [
        "ME",
        "MI"
    ]:

        errors.append(
            "Conditions must be ME and MI."
        )

    if SETS != [
        "A",
        "B",
        "C"
    ]:

        errors.append(
            "Sets must be A, B, C."
        )

    if len(GESTURES) != 7:

        errors.append(
            "Exactly 7 gestures are required."
        )

    if EEG_CHANNEL_COUNT != 13:

        errors.append(
            "EEG channel count must be 13."
        )

    if SAMPLING_RATE_HZ != 125:

        errors.append(
            "Sampling rate must be 125 Hz."
        )

    if NOMINAL_SAMPLE_COUNT != 5625:

        errors.append(
            "Nominal sample count must be 5625."
        )

    if (
        NOMINAL_DURATION_SECONDS
        * SAMPLING_RATE_HZ
        != NOMINAL_SAMPLE_COUNT
    ):

        errors.append(
            "45 s × 125 Hz must equal 5625 samples."
        )

    if WINDOW_LENGTHS_SAMPLES != [
        200,
        256,
        300,
        400,
        500
    ]:

        errors.append(
            "Window candidate lengths changed."
        )

    if WINDOW_OVERLAPS_PERCENT != [
        0,
        25,
        50,
        75
    ]:

        errors.append(
            "Window candidate overlaps changed."
        )

    if len(ML_MODELS) != 5:

        errors.append(
            "Five ML models are required."
        )

    if len(DL_MODELS) != 5:

        errors.append(
            "Five DL models are required."
        )

    if CALIBRATION_FRACTIONS_PERCENT != [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        80,
        90
    ]:

        errors.append(
            "Calibration fractions changed."
        )

    if ML_FEATURES["total"] != 341:

        errors.append(
            "ML feature total must be 341."
        )

    if (
        ML_FEATURES["channel_level_features"]
        != 13 * 24
    ):

        errors.append(
            "Channel-level feature accounting is invalid."
        )

    return errors


# ============================================================
# BUILD COMPLETE CONFIGURATION
# ============================================================

def build_config(
    args,
    created
):

    config = {

        "block":
            BLOCK_ID,

        "protocol_version":
            PROTOCOL_VERSION,

        "status":
            "FROZEN",

        "created_utc":
            created,

        "paper_title":
            PAPER_TITLE,


        # ----------------------------------------------------
        # DATA PROVENANCE
        # ----------------------------------------------------

        "data_provenance": {

            "raw_data_google_drive_folder":
                args.raw_drive_url,

            "raw_data_local_root":
                args.raw_root,

            "block00_downloads_data":
                False,

            "block00_scans_raw_data":
                False,

            "block00_modifies_raw_data":
                False,
        },


        # ----------------------------------------------------
        # COHORT
        # ----------------------------------------------------

        "cohort": {

            "primary_cohort":
                "25 paired subjects with both ME and MI",

            "conditions":
                CONDITIONS,

            "gestures":
                GESTURES,

            "sets":
                SETS,

            "eeg_channels":
                EEG_CHANNEL_COUNT,

            "sampling_rate_hz":
                SAMPLING_RATE_HZ,

            "nominal_duration_seconds":
                NOMINAL_DURATION_SECONDS,

            "nominal_samples":
                NOMINAL_SAMPLE_COUNT,

            "expected_recordings_per_subject_condition":
                EXPECTED_RECORDINGS[
                    "per_subject_per_condition"
                ],

            "expected_recordings_per_complete_paired_subject":
                EXPECTED_RECORDINGS[
                    "per_complete_paired_subject"
                ],

            "expected_complete_me_recordings":
                EXPECTED_RECORDINGS[
                    "complete_me_recordings"
                ],

            "expected_complete_mi_recordings":
                EXPECTED_RECORDINGS[
                    "complete_mi_recordings"
                ],

            "expected_complete_paired_recordings":
                EXPECTED_RECORDINGS[
                    "complete_paired_recordings"
                ],

            "window_term":
                "computational EEG windows",

            "trial_reconstruction":
                False,
        },


        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        "preprocessing":
            PREPROCESSING,


        # ----------------------------------------------------
        # WINDOWING
        # ----------------------------------------------------

        "windowing": {

            "candidate_lengths_samples":
                WINDOW_LENGTHS_SAMPLES,

            "candidate_lengths_seconds": {

                str(L):
                    L / SAMPLING_RATE_HZ

                for L in WINDOW_LENGTHS_SAMPLES
            },

            "candidate_overlap_percent":
                WINDOW_OVERLAPS_PERCENT,

            "candidate_count":
                20,

            "step_formula":
                "S = L*(1-O)",

            "window_count_formula":
                "floor((N-L)/(L*(1-O)))+1",

            "historical_configuration":
                HISTORICAL_WINDOW,

            "cross_recording":
                False,

            "cross_gesture":
                False,

            "cross_condition":
                False,

            "padding":
                False,

            "window_level_artifact_cleaning":
                False,

            "selection_metric":
                "Macro-F1",

            "secondary_selection_metric":
                "Balanced Accuracy",

            "selection_scope":
                "development/internal validation only",

            "one_global_final_configuration":
                True,

            "same_for_ml_and_dl":
                True,

            "same_for_me_and_mi":
                True,

            "same_for_transfer":
                True,

            "same_for_calibration":
                True,

            "per_subject_selection":
                False,

            "tie_break_order": [

                "higher Macro-F1",

                "higher Balanced Accuracy",

                "lower overlap",

                "shorter duration",

                "lower computational window count",
            ],

            "tie_tolerance":
                None,

            "tie_tolerance_status":
                "MUST_BE_FROZEN_IN_BLOCK04",
        },


        # ----------------------------------------------------
        # ML
        # ----------------------------------------------------

        "machine_learning": {

            "models":
                ML_MODELS,

            "features":
                ML_FEATURES,
        },


        # ----------------------------------------------------
        # DL
        # ----------------------------------------------------

        "deep_learning": {

            "models":
                DL_MODELS,
        },


        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        "evaluation":
            EVALUATION,


        # ----------------------------------------------------
        # CHANCE
        # ----------------------------------------------------

        "chance_level":
            CHANCE_LEVEL,


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        "me_mi_statistics":
            ME_MI_STATISTICS,


        # ----------------------------------------------------
        # CALIBRATION
        # ----------------------------------------------------

        "temporal_calibration":
            TEMPORAL_CALIBRATION,


        # ----------------------------------------------------
        # TRANSFER
        # ----------------------------------------------------

        "transfer":
            TRANSFER,


        # ----------------------------------------------------
        # REPRESENTATION
        # ----------------------------------------------------

        "representation":
            REPRESENTATION,


        # ----------------------------------------------------
        # NEUROPHYSIOLOGY
        # ----------------------------------------------------

        "neurophysiology":
            NEUROPHYSIOLOGY,


        # ----------------------------------------------------
        # XAI
        # ----------------------------------------------------

        "xai":
            XAI,


        # ----------------------------------------------------
        # ROBUSTNESS
        # ----------------------------------------------------

        "robustness":
            ROBUSTNESS,


        # ----------------------------------------------------
        # REPRODUCIBILITY
        # ----------------------------------------------------

        "reproducibility":
            build_reproducibility(
                args.seed
            ),


        # ----------------------------------------------------
        # PIPELINE
        # ----------------------------------------------------

        "pipeline":
            PIPELINE,


        # ----------------------------------------------------
        # NO-GO
        # ----------------------------------------------------

        "no_go_rules":
            NO_GO_RULES,


        # ----------------------------------------------------
        # BLOCK 00 SCOPE
        # ----------------------------------------------------

        "block00_scope": {

            "protocol_freeze_only":
                True,

            "signal_processing":
                False,

            "window_generation":
                False,

            "feature_extraction":
                False,

            "model_training":
                False,

            "model_evaluation":
                False,

            "statistical_testing":
                False,

            "figure_generation":
                False,

            "raw_data_modification":
                False,
        },


        # ----------------------------------------------------
        # DOWNSTREAM FREEZES
        # ----------------------------------------------------

        "required_downstream_freezes": {

            "BLOCK01": [

                "discover actual ME subjects",

                "discover actual MI subjects",

                "establish actual paired intersection",

                "verify G1-G7 completeness",

                "verify A/B/C completeness",
            ],

            "BLOCK02": [

                "freeze QA thresholds before final decisions",
            ],

            "BLOCK04": [

                "freeze exact development/internal validation design",

                "freeze tie tolerance",
            ],

            "BLOCK05": [

                "freeze exact 24 ML feature-family definitions",
            ],

            "BLOCK09": [

                "freeze calibration adaptation operator",
            ],

            "BLOCK12": [

                "freeze exact latent layer per primary model",
            ],

            "BLOCK14": [

                "freeze exact predefined noise parameters",
            ],
        },
    }

    return config


# ============================================================
# TIME
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=
        "Create the frozen Paper 2 BLOCK00 protocol."
    )


    parser.add_argument(

        "--root",

        default=r"F:\Faruk\EEG_Motor_Execution_Imagery\Paper2_EEG_BCI",

        help=
        "Paper 2 project root. Defaults to the local Windows project."
    )


    parser.add_argument(

        "--raw-drive-url",

        default=
        DEFAULT_RAW_DRIVE_URL
    )


    parser.add_argument(

        "--raw-root",

        default=r"F:\Faruk\EEG_Motor_Execution_Imagery",

        help=
        "Raw-data parent path for provenance only. "
        "BLOCK00 does not scan it."
    )


    parser.add_argument(

        "--seed",

        type=int,

        default=42
    )


    parser.add_argument(

        "--overwrite",

        action="store_true"
    )


    args = parser.parse_args()


    # --------------------------------------------------------
    # VALIDATE PROTOCOL
    # --------------------------------------------------------

    errors = validate_protocol()

    if errors:

        print(
            "BLOCK00 VALIDATION FAILED"
        )

        for error in errors:

            print(
                " -",
                error
            )

        return 2


    # --------------------------------------------------------
    # ROOT
    # --------------------------------------------------------

    root = Path(
        args.root
    ).expanduser().resolve()


    block_dir = (
        root
        / "00_PROTOCOL"
    )


    # --------------------------------------------------------
    # PROTECT EXISTING FREEZE
    # --------------------------------------------------------

    if (

        block_dir.exists()

        and any(
            block_dir.iterdir()
        )

        and not args.overwrite

    ):

        print(
            f"ERROR: {block_dir} already contains files."
        )

        print(
            "Use --overwrite only when intentionally "
            "replacing the frozen protocol."
        )

        return 3


    # --------------------------------------------------------
    # CREATE DIRECTORIES
    # --------------------------------------------------------

    (
        block_dir
        / "configs"
    ).mkdir(
        parents=True,
        exist_ok=True
    )


    (
        block_dir
        / "audit"
    ).mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    created = utc_now()


    # --------------------------------------------------------
    # BUILD CONFIG
    # --------------------------------------------------------

    config = build_config(
        args,
        created
    )


    # --------------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------------

    config_path = (

        block_dir
        / "configs"
        / "BLOCK00_config.json"
    )


    summary_path = (

        block_dir
        / "configs"
        / "BLOCK00_protocol_summary.json"
    )


    text_path = (

        block_dir
        / "BLOCK00_protocol_summary.txt"
    )


    # --------------------------------------------------------
    # WRITE CONFIG
    # --------------------------------------------------------

    write_json(
        config_path,
        config
    )


    # --------------------------------------------------------
    # SUMMARY JSON
    # --------------------------------------------------------

    summary = {

        "block":
            BLOCK_ID,

        "protocol_version":
            PROTOCOL_VERSION,

        "status":
            "FROZEN",

        "created_utc":
            created,

        "primary_cohort":
            "25 paired ME-MI subjects",

        "conditions":
            CONDITIONS,

        "gestures":
            GESTURES,

        "sets":
            SETS,

        "eeg_channels":
            EEG_CHANNEL_COUNT,

        "sampling_rate_hz":
            SAMPLING_RATE_HZ,

        "nominal_samples":
            NOMINAL_SAMPLE_COUNT,

        "window_candidates":
            20,

        "ml_models":
            ML_MODELS,

        "dl_models":
            DL_MODELS,

        "ml_feature_total":
            341,

        "calibration_fractions_percent":
            CALIBRATION_FRACTIONS_PERCENT,

        "loso_folds":
            25,

        "chance_accuracy_percent":
            100.0 / 7.0,

        "raw_data_scanned":
            False,

        "raw_data_modified":
            False,

        "scientific_analysis_performed":
            False,

        "next_block":
            "BLOCK01",
    }


    write_json(
        summary_path,
        summary
    )


    # --------------------------------------------------------
    # HUMAN-READABLE SUMMARY
    # --------------------------------------------------------

    text = f"""
PAPER 2 — BLOCK 00 PROTOCOL FREEZE
===================================

Protocol version:
{PROTOCOL_VERSION}

Status:
FROZEN

Created UTC:
{created}


TITLE
-----
{PAPER_TITLE}


PRIMARY COHORT
--------------
25 paired subjects with both Motor Execution (ME)
and Motor Imagery (MI).

Conditions:
ME, MI

Gestures:
1 = HOC
2 = IFM
3 = LFM
4 = MFM
5 = PH
6 = RFM
7 = TFM

Sets:
A, B, C

EEG:
13 channels
125 Hz
45 seconds nominal
5625 samples nominal

Expected recordings:
21 per subject per condition
42 per complete paired subject
525 ME recordings
525 MI recordings
1050 paired recordings


PREPROCESSING
-------------
1–40 Hz
4th-order Butterworth
zero-phase offline
no resampling
13 channels preserved
sample count preserved
no default extra notch
no arbitrary permanent re-reference
no automatic bad-channel interpolation
no ICA by default
no automatic artifact rejection
no global normalization
no feature selection
no dimensionality reduction
no class balancing
no SMOTE
no trial reconstruction


WINDOWING
---------
Lengths:
200, 256, 300, 400, 500 samples

Overlaps:
0%, 25%, 50%, 75%

Total candidates:
20

Historical 250-sample/50%:
diagnostic/compatibility only

Selection:
development/internal validation only

Primary selection metric:
Macro-F1

Secondary:
Balanced Accuracy

One global final window configuration is required
for ML, DL, ME, MI, transfer, calibration,
representation, and robustness.

No per-subject window selection.


MACHINE LEARNING
----------------
LDA
KNN
SVM
RandomForest
XGBoost

Total features:
341

Feature accounting:
13*24 + 13 + 13 + 3 = 341

Exact 24 feature-family names:
MUST BE FROZEN IN BLOCK05

Logistic Regression:
PROHIBITED

SMOTE:
PROHIBITED


DEEP LEARNING
-------------
EEGNet
ShallowConvNet
DeepConvNet
CNN-BiLSTM
CNN-BiLSTM+Self-Attention


EVALUATION
----------
Primary:
Accuracy
Balanced Accuracy
Macro-F1

Secondary:
MCC
Cohen's kappa

Class-level:
Precision
Recall
F1

Confusion matrix:
YES

Inferential unit:
subject

LOSO:
25 folds
24 training subjects
1 unseen test subject

Test subject used for tuning:
NO


CHANCE LEVEL
------------
7 classes

Accuracy chance:
1/7

Percentage:
14.285714%


ME–MI STATISTICS
----------------
Paired difference:
D_i = A_ME,i - A_MI,i

Effect size:
paired standardized mean difference d_z

Bootstrap:
10,000 paired subject resamples

Confidence interval:
95%

Primary test:
two-sided paired permutation/sign-flip

Permutation resamples:
10,000

Multiplicity:
Benjamini-Hochberg FDR

q:
0.05


TEMPORAL CALIBRATION
--------------------
Fractions:
0, 10, 20, 30, 40, 50, 60, 80, 90%

Calibration:
earliest contiguous target recording portion

Evaluation:
later unseen target recording portion

Physical-sample embargo:
REQUIRED

Random calibration:
PROHIBITED

Crossing windows:
EXCLUDED

Calibration efficiency:
CE(k) = [A(k) - A(0)] / k

k90:
minimum tested fraction reaching
90% of the gap between zero-shot and
subject-dependent reference.

No interpolation.
No smoothing.

Exact adaptation operator:
MUST BE FROZEN IN BLOCK09.


ME–MI TRANSFER
--------------
ME -> MI
MI -> ME

Pure transfer:
no target-label adaptation

Target-native reference:
REQUIRED


REPRESENTATION / RSA
--------------------
Primary models:
EEGNet
CNN-BiLSTM
CNN-BiLSTM+Self-Attention

Gesture representation:
7 gestures

RDM:
7 x 7

Unique upper-triangle pairs:
21

RDM dissimilarity:
correlation distance = 1 - Pearson r

ME/MI RDM similarity:
Spearman correlation

Exact latent layers:
MUST BE FROZEN BEFORE BLOCK12


NEUROPHYSIOLOGY
---------------
PSD
band power
relative band power
alpha/beta ratio

True ERD/ERS:
NOT PERMITTED without a verified rest baseline.

Anatomical topography:
requires verified channel geometry.


XAI
---
ML:
SHAP

DL:
Integrated Gradients
occlusion

Frozen models only.

XAI does not establish causal neurophysiology
or anatomical source.


ROBUSTNESS
----------
Channel dropout:
5%, 10%, 20%, 30%

Noise:
predefined condition

Exact noise parameters:
MUST BE FROZEN BEFORE BLOCK14.


NO-GO RULES
-----------
"""

    for rule in NO_GO_RULES:

        text += (
            "- "
            + rule
            + "\n"
        )


    text += """

BLOCK 00 DOES NOT SCAN RAW DATA.
BLOCK 00 DOES NOT MODIFY RAW DATA.
BLOCK 00 DOES NOT PRODUCE SCIENTIFIC RESULTS.

NEXT:
BLOCK01 — Subject Matching & Cohort Integrity
"""


    text_path.write_text(
        text,
        encoding="utf-8"
    )


    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    script_path = Path(
        __file__
    ).resolve()


    manifest = {

        "block":
            BLOCK_ID,

        "protocol_version":
            PROTOCOL_VERSION,

        "created_utc":
            created,

        "status":
            "COMPLETE_AND_FROZEN",

        "script":
            script_path.name,

        "script_sha256":
            sha256_file(
                script_path
            ),

        "config_sha256":
            sha256_file(
                config_path
            ),

        "summary_sha256":
            sha256_file(
                summary_path
            ),

        "raw_data_scanned":
            False,

        "raw_data_modified":
            False,

        "scientific_analysis_performed":
            False,

        "next_block":
            "BLOCK01",
    }


    manifest_path = (

        block_dir
        / "audit"
        / "BLOCK00_manifest.json"
    )


    write_json(
        manifest_path,
        manifest
    )


    # --------------------------------------------------------
    # COMPLETION MARKER
    # --------------------------------------------------------

    complete = {

        **manifest,

        "complete_marker":
            True,

        "complete_marker_meaning":
            "protocol successfully frozen",
    }


    complete_path = (

        block_dir
        / "BLOCK00_complete.json"
    )


    write_json(
        complete_path,
        complete
    )


    # --------------------------------------------------------
    # FINAL CONSOLE OUTPUT
    # --------------------------------------------------------

    print("=" * 72)

    print(
        "BLOCK 00 — PROTOCOL FREEZE"
    )

    print("=" * 72)

    print(
        "STATUS              : COMPLETE_AND_FROZEN"
    )

    print(
        f"PROJECT ROOT        : {root}"
    )

    print(
        f"RAW DRIVE URL       : {args.raw_drive_url}"
    )

    print(
        "PRIMARY COHORT      : 25 paired ME–MI subjects"
    )

    print(
        "EEG                 : 13 channels @ 125 Hz"
    )

    print(
        "NOMINAL RECORDING   : 45 s / 5625 samples"
    )

    print(
        "GESTURES            : 7"
    )

    print(
        "SETS                : A, B, C"
    )

    print(
        "WINDOW CANDIDATES   : 20"
    )

    print(
        "ML MODELS           : 5"
    )

    print(
        "DL MODELS           : 5"
    )

    print(
        "ML FEATURES         : 341"
    )

    print(
        "LOSO FOLDS          : 25"
    )

    print(
        "CALIBRATION         : "
        "0,10,20,30,40,50,60,80,90%"
    )

    print()

    print(
        "CREATED:"
    )

    print(
        f"  {config_path}"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {text_path}"
    )

    print(
        f"  {manifest_path}"
    )

    print(
        f"  {complete_path}"
    )

    print()

    print(
        "BLOCK00 did NOT inspect or modify raw EEG."
    )

    print(
        "NEXT: BLOCK01 — Subject Matching & Cohort Integrity"
    )

    print("=" * 72)

    return 0


if __name__ == "__main__":
    main()
