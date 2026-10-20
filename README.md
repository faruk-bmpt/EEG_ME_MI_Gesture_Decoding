# Execution–Imagery Representational Similarity and Bidirectional Transfer with Temporal Calibration in Cross-Subject EEG Hand Gesture Decoding
Study: Subject-dependent and cross-subject decoding, motor execution–imagery transfer, and temporal calibration
Modality: EEG
Conditions: Motor Execution (ME) and Motor Imagery (MI)
Primary paired cohort: 25 subjects, subject to verification by the dataset audit
Gesture classes: 7
Sampling frequency: 125 Hz
EEG channels: 13
Overview
This repository contains the analysis pipeline for investigating how EEG representations of hand and finger gestures are shared between motor execution (ME) and motor imagery (MI), and whether their similarity is associated with cross-condition transfer and temporal calibration requirements.

The study compares conventional machine-learning (ML) and deep-learning (DL) approaches under subject-dependent and leave-one-subject-out (LOSO) evaluation. Additional analyses are designed to evaluate bidirectional ME↔MI transfer, temporal calibration, latent representation similarity, explainability, robustness, and computational cost.

The manuscript describes the planned scientific analysis. Numerical results and conclusions must be generated from the executed pipeline; they must not be inferred from this README or fabricated.
Research questions
1. How does subject-dependent decoding compare with cross-subject decoding?
2. Do ME and MI differ in decoding performance?
3. How well does a model trained in one condition transfer to the other condition?
4. How does performance change as progressively more of a target subject's recording is used for temporal calibration?
5. Is subject-level ME–MI representational similarity associated with transfer benefit or calibration requirement?
6. How robust are the models to channel dropout and controlled noise, and what are their computational costs?
Dataset and task
The planned primary analysis uses paired ME and MI recordings from the same participants.
Item	Study specification
Conditions	Motor Execution (ME), Motor Imagery (MI)
Paired cohort	25 subjects, to be confirmed by the audit
Gesture classes	7
Recording sets	3 per gesture and condition, subject to audit
EEG channels	13
Sampling frequency	125 Hz
Nominal recording duration	45 seconds
Nominal samples per recording	5,625
Trigger timing	Not available in the documented study design
Verified resting baseline	Not available in the documented study design


The dataset audit is authoritative. Do not assume that all expected recordings are present or complete until the audit outputs confirm this.
Gesture label mapping
Use the canonical numerical gesture identifiers 1–7 in data, manifests, filenames/keys, and downstream outputs. Keep the mapping consistent throughout the pipeline.
Gesture ID	Gesture
1	Hand Open–Close
2	Index Finger Movement
3	Little Finger Movement
4	Middle Finger Movement
5	Pen Holding
6	Ring Finger Movement
7	Thumb Finger Movement


For seven balanced classes, nominal chance-level accuracy is 1/7 (approximately 14.29%). Actual class balance should be checked rather than assumed.
Analysis workflow
Run the repository's canonical block scripts in their documented dependency order. The manuscript's planned workflow is:
1. BLOCK00_PROTOCOL_FREEZE.py — freeze the protocol and configuration.
2. BLOCK01_SUBJECT_MATCHING.py — identify and verify paired ME/MI subjects.
3. BLOCK02_SIGNAL_QA.py — audit files, labels, channels, samples, and signal quality.
4. BLOCK03_PREPROCESSING_FINAL.py — apply the specified EEG preprocessing.
5. BLOCK04_WINDOWING_FINAL_SCIENTIFIC.py — create computational windows and select/freeze the window configuration without using final held-out subjects.
6. BLOCK05_FEATURE_ENGINEERING.py — extract the deterministic ML feature representation.
7. BLOCK06_CONVENTIONAL_ML_FINAL.py — subject-dependent and LOSO conventional ML.
8. BLOCK07_DEEP_LEARNING_FINAL.py - subject-dependent and LOSO deep learning
9. BLOCK08_ME_MI_STATISTICS.py — paired ME/MI and subject-level comparisons.
10. BLOCK09_TEMPORAL_CALIBRATION.py — evaluate calibration fractions.
11. BLOCK10_ME_MI_TRANSFER.py — evaluate ME→MI and MI→ME transfer.
12. BLOCK11_NEURO.py — spectral and other pre-specified neurophysiological analyses.
13. BLOCK12_REPRESENTATION.py — latent representations and representational similarity analysis (RSA).
14. BLOCK13_XAI.py — explainability analyses.
15. BLOCK14_ROBUSTNESS.py — channel-dropout and controlled-noise analyses
16. BLOCK15_FINAL_STATISTICS.py — final subject-level statistics and inferential summaries.
17. BLOCK16_FINAL_FIGURES.py — generate figures from completed outputs.
18. BLOCK17_FINAL_TABLES.py — generate manuscript tables from completed outputs.

## Methods at a glance

** Preprocessing and windowing ** 
- The manuscript specifies 1–40 Hz zero-phase Butterworth filtering.
- Candidate window lengths: 200, 256, 300, 400, and 500 samples.
- Candidate overlap values: 0%, 25%, 50%, and 75%.
- Window selection must be completed using training/internal validation data and frozen before final LOSO evaluation.
- Because recordings are continuous and verified trigger timing is unavailable, derived segments are described as computational windows, not reconstructed trials.
- Calibration/evaluation partitions must respect temporal ordering. An appropriate physical-sample embargo is needed where overlapping windows could otherwise share samples across the boundary.
- Any data-fitted transformation (including scaling, feature selection, dimensionality reduction, and model fitting) must be learned from training data only.
Models
The manuscript's planned ML models are LDA, KNN, SVM, Random Forest, and XGBoost.

The planned DL architectures are EEGNet, ShallowConvNet, DeepConvNet, CNN-BiLSTM, and CNN-BiLSTM with self-attention.

Use the implementation and frozen configuration in the repository as the source of truth for model parameters. Do not modify architectures or hyperparameters after inspecting final test performance.
Evaluation
The planned evaluation includes:
- Subject-dependent evaluation.
- 25-fold leave-one-subject-out (LOSO) evaluation, subject to confirmation of the final paired cohort.
- Accuracy, balanced accuracy, macro-F1, MCC, Cohen's kappa, class-level metrics, and confusion matrices.
- Subject-level summaries and paired comparisons; windows must not be treated as independent subjects.
- Bidirectional ME→MI and MI→ME transfer, including negative transfer and transfer asymmetry.
- Temporal calibration at 0%, 10%, 20%, 30%, 40%, 50%, 60%, 80%, and 90% of the target recording, where implemented by the pipeline.
- Latent representation similarity/RSA, representation–transfer and representation–calibration associations, explainability, robustness, and computational measurements, where implemented.
The actual output files determine which analyses completed successfully. Do not report a planned analysis as completed unless its outputs and audit checks exist.
Environment and installation
Use a dedicated Python environment. The exact Python and package versions should be taken from the repository's tested environment, lock file, or run manifest where available.

Example setup for Windows PowerShell:
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r REQUIREMENTS.txt
Example setup for Linux/WSL:
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r REQUIREMENTS.txt
These commands are examples; use the Python version supported by the actual code and installed PyTorch/CUDA build. A CUDA-enabled PyTorch installation may require a platform-specific installation command. Follow the official PyTorch selector for the target operating system and CUDA configuration rather than assuming that a generic pip installation enables GPU support.
Configure and run
1. Clone or download this repository.
2. Read the protocol and configuration files before running any block.
3. Set the dataset root and output locations in the configuration expected by the scripts. Do not hard-code another user's local paths.
4. Confirm that the dataset files are accessible and that subject, condition, gesture, set, channel, and sample metadata match the protocol.
5. Run the blocks in dependency order, using each script's documented invocation.
6. Inspect each block's log, status/manifest, QC report, and expected output files before proceeding.
7. If a block fails or produces incomplete outputs, resolve the failure and rerun according to that block's recovery logic. Do not treat a partial output as a completed experiment.
8. Preserve configuration files, seeds, environment details, logs, output manifests, and hashes for the final audit.
No universal command is provided here intentionally: invoke scripts using their actual names and arguments in this repository. Do not execute all scripts blindly if a preceding block failed.
Expected outputs
Exact filenames depend on the committed scripts and their configurations. The pipeline is intended to produce traceable outputs such as:
- Dataset/subject matching inventories and signal-QA reports.
- Preprocessed recordings and window manifests.
- Feature arrays and label/subject metadata.
- Model predictions and evaluation metrics.
- Subject-level and aggregate statistical summaries.
- Calibration and cross-condition transfer results.
- Representation/RSA, explainability, and robustness outputs.
- Figures, tables, execution logs, manifests, and final audit reports.
Consult the individual block scripts and their output manifests for authoritative filenames and schemas.
Reproducibility and data leakage safeguards
- Keep the subject as the primary unit for inferential statistics.
- Keep final LOSO subjects completely held out from preprocessing choices that require fitting, feature selection, hyperparameter selection, model selection, and window-configuration selection.
- Fit data-dependent transformations on training data only.
- Do not randomly mix temporally adjacent overlapping windows across calibration and evaluation partitions.
- Use only verified participant demographics, ethics information, and dataset metadata.
- Preserve random seeds, package versions, hardware/software details, configurations, logs, and output hashes.
- Treat negative transfer and weak or null representation associations as valid findings.
- Interpret feature attribution as model evidence, not proof of a causal or anatomical source.
Current status and interpretation
This repository documents the analysis pipeline; it does not by itself establish the study's results. Complete the execution and final audit before filling the manuscript's Results, Discussion, and Conclusion with numerical claims.

The manuscript currently identifies limitations including the size of the paired cohort, limited channel count, unavailable verified trigger timing and resting baseline, possible dependence between overlapping windows, and limits to generalization. Preserve these qualifications in any publication.
Data, ethics, and citation
- Dataset: Add the verified public dataset DOI or URL when the release is public and its metadata have been checked.
- Code: Add the canonical GitHub repository URL and, if available, an archived release DOI.
- Ethics: Insert the exact approved ethics statement/reference from the source records. Do not invent an approval number or consent statement.
- Citation: Add the final manuscript citation after publication. Until then, cite the preprint or dataset only if one exists and is appropriate.
Repository maintenance
When updating this README:
- Keep the filenames and commands aligned with the actual committed scripts.
- Document any required environment variables, configuration files, and data access permissions.
- Record tested Python/package versions and hardware requirements.
- Do not commit private participant information, credentials, or restricted raw data.
- Ensure all claimed outputs can be traced to a script, configuration, and execution record.
Disclaimer
This repository is intended for research and reproducibility. It is not a clinically validated medical device or a substitute for clinical assessment.
