# Execution–Imagery Representational Similarity and Bidirectional Transfer with Temporal Calibration in Cross-Subject EEG Hand Gesture Decoding

## Overview

This repository documents an EEG analysis project investigating how neural representations of hand and finger gestures differ between **motor execution (ME)** and **motor imagery (MI)**, and how those differences relate to cross-subject decoding, cross-condition transfer, and subject-specific temporal calibration.

The project is designed around a paired cohort of 25 participants with EEG recordings from both ME and MI conditions. Seven gesture classes are examined using 13 EEG channels sampled at 125 Hz. The analysis framework combines classical machine learning, deep learning, temporal few-shot calibration, bidirectional cross-condition transfer, neurophysiological analysis, representation comparison, explainability, and statistical evaluation.

> **Status note:** This README describes the study design and analysis framework. Add only verified results and implementation details from the final executed code. Do not interpret planned analyses as completed results unless the corresponding outputs have been checked.

## Research Question

How similar and different are EEG representations of continuous hand and finger gestures during motor execution and motor imagery, and how do these differences affect cross-subject decoding, cross-condition transfer, and subject adaptation?

## Objectives

1. Evaluate motor-execution and motor-imagery gesture decoding independently.
2. Compare decoding performance across ME and MI using the same paired participants and gesture repertoire.
3. Assess transfer in both directions: ME → MI and MI → ME.
4. Examine how performance changes with increasing amounts of early target-subject calibration data.
5. Investigate spectral characteristics and learned representation similarity across conditions.
6. Evaluate model interpretability, robustness, and statistical reliability where supported by the executed pipeline.

## Data and Experimental Configuration

The primary analysis cohort contains **25 participants with both motor-execution and motor-imagery EEG recordings**.

| Property | Configuration |
|---|---|
| Paired participants | 25 |
| Conditions | Motor execution (ME) and motor imagery (MI) |
| Gesture classes | 7 |
| EEG channels | 13 |
| Sampling frequency | 125 Hz |
| Recording duration | Approximately 45 seconds |
| Recording sets | 3 sets per gesture and condition |
| Window length | 250 samples (2 seconds) |
| Window overlap | 50% (125 samples) |
| Expected complete windows | Approximately 44 per recording |

### Gesture classes

1. Hand Open/Close
2. Index Finger Movement
3. Little Finger Movement
4. Middle Finger Movement
5. Pen Holding
6. Ring Finger Movement
7. Thumb Finger Movement

### Recording labels and windowing

The source recordings are continuous, and the gesture label is assigned at the **recording level**. Fixed-length sliding windows are extracted for computational analysis. These windows are not independent experimental trials and should not be described as such.

At 125 Hz, a 250-sample window represents 2 seconds. With a 125-sample step, approximately 44 complete windows can be extracted from a 5,625-sample recording.

## Signal Processing and Feature Engineering

The analysis framework includes EEG quality assessment, missing-value handling, filtering, artifact assessment, and fixed-length window extraction. The final preprocessing implementation should be documented alongside the code, including filter settings, artifact rules, and any exclusions or quality flags.

### Handcrafted features for classical machine learning

The finalized feature library specified for the study contains **341 features** from the following families:

- Time-domain features
- Hjorth parameters
- Absolute spectral power
- Relative spectral power
- Alpha/beta power ratios
- Theta/alpha power ratios
- Channel-level statistics
- Global signal statistics

Any feature selection, imputation, normalization, or scaling must be fitted within the training data of each evaluation fold. The held-out test subject must not influence training-time transformations.

### Input for deep learning

The primary deep-learning input is the windowed EEG signal with dimensions:

```text
13 channels × 250 time samples
```

The deep-learning branch uses EEG windows directly rather than the handcrafted 341-feature representation.

## Classical Machine-Learning Models

The study specifies five classical machine-learning models:

1. Linear Discriminant Analysis (LDA)
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Random Forest
5. XGBoost

The exact model hyperparameters and any tuning strategy should be reported from the final code and experiment configuration.

## Deep-Learning Models

The study specifies five deep-learning architectures:

1. EEGNet
2. ShallowConvNet
3. DeepConvNet
4. CNN-BiLSTM
5. CNN-BiLSTM with Self-Attention

For reproducibility, document the implemented layers, kernel sizes, filters, activations, pooling, dropout, normalization, parameter counts, optimizer, learning rate, batch size, epoch limit, early-stopping criteria, random seeds, and training time where those details are available in the executed implementation.

## Evaluation Protocols

### 1. Leave-One-Subject-Out (LOSO)

The primary cross-subject evaluation uses a 25-fold LOSO design. Each participant is held out once for testing, while the remaining participants provide training data. This evaluates generalization to a participant not seen during training.

All training-time preprocessing and model fitting must be performed without access to the held-out subject's test data, except where a specific experiment explicitly permits target-subject calibration.

### 2. Separate ME and MI Decoding

Motor execution and motor imagery are evaluated separately. Subject-level metrics are then compared using the paired participant cohort.

### 3. Temporal Few-Shot Calibration

The specified calibration levels are:

```text
k ∈ {0, 1, 3, 5, 7, 10}
```

The protocol uses early target-subject data for calibration and later data for testing. Calibration samples must not be selected randomly from across the entire target recording when evaluating temporal adaptation.

The calibration analysis examines the change in performance relative to the zero-calibration condition. Any adaptation mechanism must be described precisely from the executed code.

### 4. Bidirectional Cross-Condition Transfer

The transfer matrix includes four configurations:

| Training condition | Test condition | Purpose |
|---|---|---|
| ME | ME | Within-condition reference |
| MI | MI | Within-condition reference |
| ME | MI | Execution-to-imagery transfer |
| MI | ME | Imagery-to-execution transfer |

Within-condition results provide reference baselines for interpreting cross-condition transfer. Transfer gaps can be calculated from comparable metrics when the evaluation design supports that comparison.

## Evaluation Metrics

The specified metrics include:

- Accuracy
- Balanced accuracy
- Macro-F1
- Multiclass Matthews correlation coefficient (MCC)
- Cohen's kappa
- Per-class precision, recall, and F1 score

For seven balanced classes, the nominal uniform-guess accuracy reference is 1/7, or approximately 14.29%. This reference is not a substitute for a suitable statistical baseline, especially if class distributions or prediction behavior are not uniform.

## Statistical Analysis

The ME-versus-MI comparison should use paired subject-level results. A paired t-test may be used when its assumptions are reasonably supported; otherwise, a Wilcoxon signed-rank test may be appropriate. Report effect sizes and confidence intervals, and account for multiple comparisons when conducting multiple metric or model comparisons.

Statistical tests and claims should be based on verified subject-level outputs. Overlapping windows from the same continuous recording should not be treated as independent participants.

## Neurophysiological and Representation Analyses

The analysis plan includes the following components where supported by the data and executed code:

### Power spectral density

Compare predefined EEG frequency-band characteristics between ME and MI. Document the spectral estimation method, frequency bands, and aggregation strategy used.

### ERD/ERS

Event-related desynchronization/synchronization analysis requires clearly defined baseline and task intervals. Do not report ERD/ERS values unless the acquisition protocol and available annotations support the computation.

### Representation similarity

Compare learned representations using methods such as representational dissimilarity matrices (RDMs) and Spearman correlation. Explain which representations are compared and how observations are paired across conditions.

### Latent-space visualization

PCA and UMAP may be used for exploratory visualization of feature or model representations. Visual separation alone should not be interpreted as proof of statistically reliable class separability.

### Explainability

The planned methods include SHAP for suitable classical machine-learning models and Integrated Gradients, occlusion, and temporal/channel attribution for suitable deep-learning models. Use only methods that were actually implemented and validated.

### Robustness and computational analysis

Selected channel-dropout or noise experiments may assess robustness. Computational reporting may include training time, inference time, model size, and parameter count where measured.

### Channel topography

Anatomical scalp maps should only be generated if reliable electrode-location information is available. If the channel names do not map to verified electrode coordinates, report channel-level comparisons without assigning unsupported anatomical locations.

Topological data analysis is optional and exploratory, not a required core component.

## Reproducibility and Data Governance

For reproducibility, the repository should include the actual code needed to reproduce the published analyses, an environment specification, configuration details, random seeds, evaluation definitions, and instructions for locating outputs.

Before releasing code or data, confirm that sharing is permitted under the applicable participant-consent, ethics, institutional, and data-governance requirements. Do not commit identifiable information, private participant records, credentials, or restricted raw data.

## Suggested Repository Structure

The structure below is a suggested organization. Keep only directories and files that exist in the actual repository; do not create empty folders merely to match this template.

```text
EEG_ME_MI_Gesture_Decoding/
├── README.md
├── requirements.txt
├── LICENSE
├── Code/
│   ├── preprocessing/
│   ├── feature_extraction/
│   ├── machine_learning/
│   ├── deep_learning/
│   ├── loso_evaluation/
│   ├── temporal_calibration/
│   ├── cross_condition_transfer/
│   ├── representation_analysis/
│   ├── explainability/
│   └── statistical_analysis/
├── Documentation/
│   ├── methodology.md
│   ├── evaluation_protocol.md
│   └── reproducibility.md
├── Results/
│   ├── tables/
│   ├── figures/
│   └── statistical_reports/
└── figures/
```

## Reproduction Guide

The following is a general workflow, not a guarantee that commands or entry-point scripts already exist in this repository.

1. Review the methodology and evaluation protocol.
2. Set up the Python environment using the dependencies specified in `requirements.txt`.
3. Configure paths to the authorized data location.
4. Run the preprocessing and feature-extraction steps implemented in the repository.
5. Execute the ME and MI LOSO experiments.
6. Execute the temporal calibration and cross-condition transfer experiments.
7. Generate neurophysiological, representation, explainability, robustness, and statistical outputs that are implemented.
8. Verify the saved tables and figures against the experiment logs before reporting results.

Add exact commands, filenames, operating-system requirements, and expected outputs after confirming them against the actual codebase.

## Results and Reporting

Insert final results only after checking the saved outputs. For each experiment, report the evaluation protocol, sample or subject count, metrics, uncertainty estimates where available, and any exclusions. Do not invent accuracy values, rankings, statistical significance, or scientific conclusions.

## Scope of This Project

This is an **EEG analysis project**, not a dataset-release paper. Its focus is cross-subject generalization, execution–imagery comparison, bidirectional transfer, temporal calibration, and neural representation analysis. It connects biomedical signal processing, machine learning, deep learning, computational neuroscience, and brain–computer interface research, with potential relevance to neural decoding and neuroprosthetic control.

## Project and Manuscript Status

The project owner should update this section to reflect the verified status of the code and manuscript. Do not describe the manuscript as submitted or published unless that is true.

## Keywords

EEG; motor execution; motor imagery; hand gesture decoding; cross-subject generalization; leave-one-subject-out; few-shot calibration; cross-condition transfer; representation similarity; deep learning; explainable AI; computational neuroscience; brain–computer interfaces.
