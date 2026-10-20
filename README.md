# Execution–Imagery Representational Similarity and Bidirectional Transfer with Temporal Calibration in Cross-Subject EEG Hand Gesture Decoding

## Overview

This repository documents a completed EEG analysis project investigating how neural representations of hand and finger gestures differ between **motor execution (ME)** and **motor imagery (MI)**, and how these differences relate to cross-subject decoding, cross-condition transfer, and subject-specific temporal calibration.

The study uses a paired cohort of 25 participants with EEG recordings from both ME and MI conditions. Seven gesture classes are examined using 13 EEG channels sampled at 125 Hz. The experimental framework includes classical machine learning, deep learning, temporal few-shot calibration, bidirectional cross-condition transfer, neurophysiological analysis, representation comparison, explainability, robustness assessment, and statistical evaluation.

The project's central objective is to investigate the relationship between execution-related and imagery-related EEG representations while evaluating their implications for cross-subject generalization and adaptation.

## Research Question

How similar and different are EEG representations of continuous hand and finger gestures during motor execution and motor imagery, and how do these differences affect cross-subject decoding, cross-condition transfer, and subject adaptation?

## Objectives

1. Evaluate motor-execution and motor-imagery gesture decoding independently.
2. Compare ME and MI decoding performance using paired participants and the same gesture repertoire.
3. Evaluate cross-condition transfer in both directions: ME → MI and MI → ME.
4. Examine performance changes with increasing amounts of early target-subject calibration data.
5. Investigate spectral characteristics and learned representation similarity across conditions.
6. Assess model interpretability, robustness, computational characteristics, and statistical reliability.

## Data and Experimental Configuration

The primary analysis cohort consists of **25 participants with both motor-execution and motor-imagery EEG recordings**.

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

### Gesture Classes

1. Hand Open/Close
2. Index Finger Movement
3. Little Finger Movement
4. Middle Finger Movement
5. Pen Holding
6. Ring Finger Movement
7. Thumb Finger Movement

### Recording Labels and Windowing

The source recordings are continuous, with gesture labels assigned at the recording level. Fixed-length sliding windows are extracted for computational analysis. These windows are computational segments rather than independent experimental trials.

At 125 Hz, a 250-sample window represents 2 seconds. With a 125-sample step, approximately 44 complete windows can be extracted from a 5,625-sample recording.

## Signal Processing and Feature Engineering

The EEG analysis workflow includes signal-quality assessment, missing-value handling, filtering, artifact assessment, and fixed-length window extraction.

Preprocessing and feature transformations are handled according to the evaluation protocol. Training-dependent transformations must be fitted using training data only, without information leakage from held-out test participants.

### Handcrafted Features for Classical Machine Learning

The study's feature library contains **341 handcrafted features** from the following feature families:

- Time-domain features
- Hjorth parameters
- Absolute spectral power
- Relative spectral power
- Alpha/beta power ratios
- Theta/alpha power ratios
- Channel-level statistics
- Global signal statistics

Feature extraction provides a structured representation of EEG activity for classical machine-learning models. Any feature selection, imputation, normalization, or scaling must respect the training/test separation established by the evaluation design.

### Deep-Learning Input

The deep-learning branch operates directly on windowed EEG signals with the following dimensions:

```text
13 EEG channels × 250 time samples
```

Unlike the classical machine-learning branch, this branch uses the windowed signal rather than the handcrafted 341-feature representation.

## Classical Machine-Learning Models

The study evaluates five classical machine-learning models:

1. Linear Discriminant Analysis (LDA)
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Random Forest
5. XGBoost

These models provide complementary approaches to classifying EEG windows and evaluating cross-subject decoding performance.

## Deep-Learning Models

The study evaluates five deep-learning architectures:

1. EEGNet
2. ShallowConvNet
3. DeepConvNet
4. CNN-BiLSTM
5. CNN-BiLSTM with Self-Attention

These architectures represent different approaches to learning discriminative temporal and spatial patterns from EEG signals. Their performance is assessed within the study's defined evaluation protocols.

## Evaluation Protocols

### 1. Leave-One-Subject-Out Evaluation

The primary cross-subject evaluation uses a 25-fold leave-one-subject-out (LOSO) design. Each participant is held out once for testing, while the remaining participants provide training data.

This design evaluates the generalization of trained models to participants who were not included in model training.

Training-time preprocessing, feature selection, model fitting, and other learned transformations must remain independent of the held-out participant's test data, except where target-subject calibration is explicitly permitted by a separate experimental protocol.

### 2. Separate ME and MI Decoding

Motor execution and motor imagery are evaluated as separate decoding conditions. Subject-level performance results provide the basis for paired ME-versus-MI comparisons.

This approach allows differences in decoding performance to be examined while accounting for the fact that both conditions were recorded from the same participants.

### 3. Temporal Few-Shot Calibration

The temporal calibration experiment evaluates the following calibration levels:

```text
k ∈ {0%, 10%, 20%, 30%, 40%, 50%, 60%, 80%, 90%} 
```
70% is excluded intentionally; there is no cause 
The calibration protocol uses early target-subject data for calibration and later target-subject data for testing. The temporal evaluation design selects calibration data rather than randomly sampling from the entire target recording.

The zero-shot condition provides the uncalibrated reference. The remaining conditions evaluate performance with increasing amounts of calibration data.

For a given calibration level, the interpretation depends on the implemented adaptation procedure and the exact definition of a calibration shot. Calibration shots should therefore be interpreted according to the experimental protocol and the corresponding code.

### 4. Bidirectional Cross-Condition Transfer

The cross-condition transfer evaluation includes four configurations:

| Training condition | Test condition | Purpose |
|---|---|---|
| ME | ME | Within-condition reference |
| MI | MI | Within-condition reference |
| ME | MI | Execution-to-imagery transfer |
| MI | ME | Imagery-to-execution transfer |

The within-condition experiments establish reference performance for ME and MI. The cross-condition experiments evaluate whether models trained in one condition generalize to the other.

Transfer gaps can be calculated from comparable metrics and evaluation settings to quantify the performance difference between within-condition decoding and cross-condition transfer.

## Evaluation Metrics

The study uses the following classification metrics:

- Accuracy
- Balanced accuracy
- Macro-F1
- Multiclass Matthews correlation coefficient (MCC)
- Cohen's kappa
- Per-class precision, recall, and F1 score

These metrics provide complementary views of overall classification performance, class-balanced performance, agreement, and class-specific behavior.

For seven classes, the nominal accuracy of uniform random guessing is approximately 14.29%. This value is only a reference point and does not replace an appropriate empirical baseline or permutation-based statistical analysis.

## Statistical Analysis

ME-versus-MI decoding performance is compared using paired subject-level results from the same participants. Within-subject differences are evaluated using paired statistical tests.

A paired t-test is appropriate when the distribution of within-subject differences reasonably supports its assumptions. The Wilcoxon signed-rank test provides a nonparametric alternative when appropriate.

Effect sizes and confidence intervals are used to describe the magnitude and uncertainty of observed differences. Multiple-comparison correction is applied where required by the number and structure of the statistical comparisons.

Statistical inference is based on verified subject-level evaluation outputs. Overlapping windows extracted from the same continuous recording are correlated observations and are not treated as independent participants. Window-level predictions contribute to subject-level performance estimates, while the participant remains the inferential unit for paired ME-versus-MI comparisons.

Statistical significance is interpreted alongside effect sizes, confidence intervals, evaluation design, and the limitations of the available data.

## Neurophysiological and Representation Analyses

The analysis framework investigates neurophysiological characteristics and learned representations relevant to ME–MI decoding.

### Power Spectral Density

Power spectral density (PSD) analysis characterizes EEG frequency-domain activity. Comparisons between ME and MI use the frequency bands, spectral estimation procedures, and aggregation methods defined in the implemented analysis.

Interpretations are restricted to the frequency-domain measures actually computed and supported by the data.

### Event-Related Desynchronization and Synchronization

Event-related desynchronization/synchronization (ERD/ERS) can characterize changes in oscillatory power relative to a baseline.

Interpretation requires valid baseline and task intervals. ERD/ERS findings must therefore be reported only when the available acquisition protocol and annotations support the relevant computation.

### Representation Similarity

Representation analysis investigates similarities and differences between EEG representations learned or extracted under ME and MI conditions.

Methods such as representational dissimilarity matrices (RDMs) and Spearman correlation can quantify representational relationships. Their interpretation depends on the representations compared, the observations paired across conditions, and the specific implementation.

### Latent-Space Visualization

Principal component analysis (PCA) and Uniform Manifold Approximation and Projection (UMAP) provide exploratory visualizations of feature or model representations.

These visualizations support qualitative inspection of representation structure. Visual separation alone does not establish statistically reliable class separability or prove a specific neural mechanism.

### Explainability

Model interpretation methods include SHAP for suitable classical machine-learning models and Integrated Gradients, occlusion, and temporal or channel attribution for suitable deep-learning models.

Interpretations must be consistent with the model architecture, the actual implementation, and the limitations of post-hoc explanation methods.

### Robustness and Computational Analysis

Robustness assessment can examine the sensitivity of decoding performance to perturbations such as channel removal or added noise, where implemented.

Computational analysis may include training time, inference time, model size, and parameter count when these quantities have been measured and verified.

### Channel Topography

Scalp topographies require reliable electrode-location information. When verified electrode coordinates are unavailable, channel-level results should be reported without assigning unsupported anatomical locations.

Topological data analysis is an optional exploratory method rather than a necessary component of the core decoding evaluation.

## Reproducibility and Data Governance

Reproducibility depends on documenting the implemented analysis, data organization, preprocessing procedures, feature definitions, model configurations, random seeds, evaluation protocols, and result-generation steps.

The repository should contain the code and supporting documentation needed to understand and reproduce the analyses, subject to the availability and permitted use of the underlying data.

Before releasing code or data, sharing permissions must be checked against participant consent, ethics approval, institutional requirements, and applicable data-governance policies. Identifiable information, private participant records, credentials, and restricted raw data must not be committed to a public repository.

## Suggested Repository Structure

The following structure is a suggested organization, not a claim that every directory or file already exists. Retain only the directories and files present in the actual repository.

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

To reproduce the study, users should:

1. Review the methodology and evaluation protocols.
2. Configure the Python environment using the dependencies specified in `requirements.txt`.
3. Configure paths to the authorized data location.
4. Run the implemented preprocessing and feature-extraction procedures.
5. Execute the ME and MI LOSO experiments.
6. Execute the temporal calibration experiments using the specified shot levels.
7. Execute the within-condition and bidirectional cross-condition transfer experiments.
8. Generate the neurophysiological, representation, explainability, robustness, and statistical outputs supported by the implementation.
9. Verify saved tables and figures against the corresponding experiment logs before reporting results.

Exact commands, script names, operating-system requirements, and expected outputs should match the actual repository and its executed code.

## Results and Reporting

Quantitative findings should be reported using verified experiment outputs. Results should identify the relevant evaluation protocol, participant count, model or configuration, metrics, uncertainty estimates where available, and any exclusions.

Performance rankings, statistical significance, and scientific conclusions must be supported by the corresponding results. No accuracy values or inferential findings should be inferred from the study design alone.

## Scope of the Project

This is an **EEG analysis project**, rather than a dataset-release paper. Its focus is cross-subject generalization, execution–imagery comparison, bidirectional transfer, temporal calibration, and neural representation analysis.

The project connects biomedical signal processing, classical machine learning, deep learning, computational neuroscience, and brain–computer interface research, with potential relevance to neural decoding and neuroprosthetic control.

## Project and Manuscript Status

The computational project has been completed. Manuscript submission or publication status should be reported separately and updated only when confirmed.

## Keywords

EEG; motor execution; motor imagery; hand gesture decoding; cross-subject generalization; leave-one-subject-out; few-shot calibration; temporal adaptation; cross-condition transfer; representation similarity; deep learning; explainable AI; computational neuroscience; brain–computer interfaces.
