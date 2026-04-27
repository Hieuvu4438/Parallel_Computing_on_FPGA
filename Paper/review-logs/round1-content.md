# Scientific Content & Methodology Review Report
Round: 1
Target journal: IEEE Internet of Things Journal

## Summary
The manuscript proposes an uncertainty-aware cascaded respiratory sound classifier that combines Random Forest early exits on the ARM processing system with an INT8 MobileNetV2 DPU path on FPGA hardware. The hardware-software co-design concept is relevant to IEEE Internet of Things Journal, but the current scientific evidence is not yet sufficient for a Q1 journal because key methodological details, ablations, energy calculations, and statistically consistent results are missing or internally inconsistent.

## Score Breakdown
- Novelty & Contribution: 16/25
- Technical Soundness: 15/30
- Experimental Design: 12/25
- Presentation Quality: 13/20
- Overall Score: 56/100

## Strengths
S1. The paper addresses an important edge-IoT medical sensing problem: low-power, local respiratory sound diagnosis on heterogeneous FPGA hardware.
S2. The cascaded early-exit formulation is practically motivated and more deployment-aware than a monolithic CNN-only respiratory classifier.
S3. The manuscript recognizes important clinical limitations by stating that the results are retrospective and should not be interpreted as prospective clinical performance.

## Weaknesses
- **ID:** W1
- **Severity:** CRITICAL
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/main.tex:18`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/01_introduction.tex:19`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/04_experiments.tex:30`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/05_conclusion.tex:3`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/tables/table_7.tex:11-15`
- **Issue:** The manuscript reports inconsistent primary accuracy values. The abstract, contribution statement, and results text claim 96.56%, while the conclusion claims a 94.94% peak average across 3-fold cross-validation; Table `table_7.tex` shows 96.56% only for Fold 0 and a mean of 94.94%.
- **Why it matters:** The headline performance claim is central to the paper's scientific contribution. Reporting a best fold as the main result instead of the cross-validation mean substantially overstates generalization and undermines trust in the evaluation.
- **Suggested fix:** Use the cross-validation mean and dispersion as the main result throughout the abstract, contributions, results, and conclusion. Report 94.94% as mean accuracy, include standard deviation or confidence intervals, and identify 96.56% explicitly as Fold 0 rather than the overall result.

- **ID:** W2
- **Severity:** CRITICAL
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/03_methodology.tex:77-85`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/tables/table_6.tex:17`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/04_experiments.tex:23-30`
- **Issue:** The early-exit thresholds `tau_i` are not specified, and the paper does not explain how thresholds were selected, tuned, or validated.
- **Why it matters:** The uncertainty-aware cascade depends directly on these thresholds. Without fixed threshold values and a leakage-free selection protocol, the routing behavior, accuracy-energy tradeoff, and reproducibility cannot be assessed.
- **Suggested fix:** Report all threshold values, the dataset split used for tuning, the optimization criterion, and whether thresholds were selected within each training fold only. Add a sensitivity analysis showing accuracy, macro-F1, exit rate per layer, latency, and energy versus threshold.

- **ID:** W3
- **Severity:** CRITICAL
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/04_experiments.tex:47-48`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/01_introduction.tex:19`
- **Issue:** The claimed 52.5% expected energy reduction is asserted without measured power, per-layer energy, exit probabilities, or the baseline definition needed to reproduce the calculation.
- **Why it matters:** Energy efficiency is a core claimed contribution for an IoT journal. A percentage reduction without measurement methodology is not sufficient evidence of hardware energy savings.
- **Suggested fix:** Provide the complete energy model: per-layer latency and power, DPU activation cost, PS preprocessing/feature extraction cost, data-transfer overhead, measured or estimated `P_i` route probabilities, and the exact static-CNN baseline. Ideally include board-level power measurements with instrumentation details.

- **ID:** W4
- **Severity:** MAJOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/03_methodology.tex:16-20`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/04_experiments.tex:9-12`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/tables/table_4.tex:11-13`
- **Issue:** The merged ICBHI and KAUH dataset protocol is under-specified. It is unclear whether “samples” means recordings, windows, or augmented segments; how three KAUH filter modes are handled; how labels from heterogeneous disease taxonomies are mapped; and whether subject identifiers are harmonized across databases.
- **Why it matters:** Dataset construction choices can strongly affect performance and may introduce leakage or domain shortcuts, especially when combining databases with different devices, demographics, and class definitions.
- **Suggested fix:** Add a reproducible dataset-construction subsection detailing record/window counts, segmentation rules, per-class counts before and after merging, handling of KAUH filter modes, subject IDs, label mapping rules, and any excluded samples.

- **ID:** W5
- **Severity:** MAJOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/03_methodology.tex:20`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/tables/table_5.tex:12-15`
- **Issue:** The Random Forest path uses 5-second segments and 50-2500 Hz filtering, while the CNN path uses 8-second segments and 25-2000 Hz filtering; the manuscript does not justify this mismatch or explain how routing from one representation to the other is implemented for the same sample.
- **Why it matters:** Different temporal windows and frequency bands can change the diagnostic information available to each layer and complicate fair cascade evaluation. The mismatch may also create implementation ambiguity for streaming inference.
- **Suggested fix:** Define the unit of inference precisely and explain how a sample is represented simultaneously in the 5-second RF branch and 8-second CNN branch. Justify the chosen bands and durations empirically or harmonize them.

- **ID:** W6
- **Severity:** MAJOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/04_experiments.tex:28-30`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/tables/table_7.tex:9-15`
- **Issue:** Performance reporting is incomplete for an imbalanced 3-class medical task. The text highlights accuracy despite a class distribution of 65.3% COPD, 23.6% Non-COPD, and 11.1% Healthy, and Table `table_7.tex` shows macro-F1 as low as 86.32% in one fold.
- **Why it matters:** Accuracy can be misleading under class imbalance and may hide poor detection of Healthy or minority Non-COPD samples. Clinical screening requires class-wise sensitivity, specificity, and error analysis.
- **Suggested fix:** Report per-class precision, recall/sensitivity, specificity, F1, confusion matrices for each fold, and aggregate confidence intervals. Discuss clinically relevant errors, especially COPD versus Non-COPD and Healthy false positives/negatives.

- **ID:** W7
- **Severity:** MAJOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/04_experiments.tex:32-37`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/tables/table_perf_cpu_dpu.tex:11-15`
- **Issue:** Hardware benchmarking reports only Layer-4 MobileNetV2 latency, not complete end-to-end cascade latency including filtering, feature extraction, RF inference, spectrogram construction, PS-PL transfer, DPU invocation overhead, and postprocessing.
- **Why it matters:** The real deployment value of UA-HCI depends on end-to-end latency and energy. Reporting only the CNN block can exaggerate practical acceleration and does not validate the full hardware-software co-design.
- **Suggested fix:** Add end-to-end measurements for each route: exit at Layer 1, exit at Layer 2, exit at Layer 3, and full Layer-4 route. Report mean, median, and tail latency under batch size 1 on Ultra96-V2.

- **ID:** W8
- **Severity:** MAJOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/03_methodology.tex:59-63`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/tables/table_6.tex:22-28`
- **Issue:** The knowledge distillation setup is not sufficiently specified. The manuscript mentions an EfficientNet-B0 teacher ensemble, focal loss, `T=4`, and `alpha=0.7`, but does not report teacher performance, number of teachers, training settings, validation procedure, or ablation against a non-distilled MobileNetV2.
- **Why it matters:** Distillation is claimed as part of the technical contribution, but without teacher and ablation evidence it is impossible to know whether it improves accuracy, robustness, or quantization behavior.
- **Suggested fix:** Report teacher architecture details, ensemble size, training hyperparameters, teacher and student fold-wise performance, and ablations for MobileNetV2 trained with and without distillation and focal loss.

- **ID:** W9
- **Severity:** MAJOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/04_experiments.tex:39-45`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/tables/table_resource_util.tex:13-16`
- **Issue:** Resource utilization is presented as synthesized DPU usage, but no DPU configuration, clock frequency, synthesis tool version, timing closure result, or bitstream configuration is reported.
- **Why it matters:** FPGA resource and latency claims are not reproducible without the DPU architecture parameters and implementation conditions. Timing closure is especially important for validating feasible deployment.
- **Suggested fix:** Specify the DPU IP variant/configuration, clock rates, Vitis/Vivado/Vitis AI versions, target bitstream, whether timing was met, and whether resources include only the DPU or also PS-PL interconnect and preprocessing accelerators.

- **ID:** W10
- **Severity:** MAJOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/02_related_work.tex:34-36`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/supplementary/related-papers/`
- **Issue:** The related-papers folder is empty, and the related work mainly compares respiratory classification accuracy rather than the closest literature on early-exit inference, uncertainty-aware cascades, FPGA/DPU edge deployment, and IoT medical hardware acceleration.
- **Why it matters:** The novelty claim is not adequately positioned against existing efficient inference and FPGA medical-AI literature. This weakens the contribution argument for IEEE Internet of Things Journal.
- **Suggested fix:** Expand the related work to include early-exit/cascaded neural inference, uncertainty/calibration for medical classification, FPGA-based audio or biomedical inference, Vitis AI/DPU deployment studies, and edge-IoT respiratory monitoring systems.

- **ID:** W11
- **Severity:** MAJOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/04_experiments.tex:6-21`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/tables/table_7.tex:1-19`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/tables/table_qat.tex:1-18`
- **Issue:** Some important result tables exist in the source tree but are not included in the current manuscript section. In particular, the cross-validation table and QAT evaluation table are not input in `sections/04_experiments.tex`.
- **Why it matters:** Readers cannot verify the reported mean accuracy, fold variability, or QAT claims from the compiled manuscript if the tables are omitted.
- **Suggested fix:** Include the cross-validation and QAT tables in the Results section, or remove unsupported claims. Ensure all headline results are visible in the compiled paper.

- **ID:** W12
- **Severity:** MAJOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/tables/table_2.tex:14-17`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/03_methodology.tex:10-16`
- **Issue:** The ICBHI class distribution is highly skewed by device and diagnosis: COPD dominates and is strongly associated with specific recording equipment in Table `table_2.tex`. The manuscript does not analyze whether the classifier is learning disease acoustics or database/device artifacts.
- **Why it matters:** Device and cohort confounding can produce inflated accuracy in respiratory sound classification. A subject-independent split alone does not eliminate device or database leakage.
- **Suggested fix:** Add cross-device or cross-database validation, report performance when training on one database and testing on the other where labels allow, and analyze results stratified by recording device and chest location.

- **ID:** W13
- **Severity:** MINOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/main.tex:17-18`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/01_introduction.tex:13-20`
- **Issue:** The abstract describes the final classifier as an “Artificial Neural Network” and “CNN via Knowledge Distillation,” while the introduction/methodology specify a MobileNetV2 student on DPU.
- **Why it matters:** Inconsistent terminology obscures the exact model architecture and contribution.
- **Suggested fix:** Use one consistent description: Random Forest early exits plus INT8 QAT MobileNetV2 student on Xilinx DPU.

- **ID:** W14
- **Severity:** MINOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/tables/table_7.tex:1-19`, `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/04_experiments.tex:28-30`
- **Issue:** Only 3-fold cross-validation is reported, with no rationale for the number of folds and no uncertainty interval.
- **Why it matters:** With only three folds and a small subject count, fold composition can substantially affect reported performance.
- **Suggested fix:** Provide fold-level subject and class counts, report mean ± standard deviation or confidence intervals, and consider repeated subject-independent cross-validation if computationally feasible.

## Questions for Authors
Q1. What exact values of `tau_1`, `tau_2`, and `tau_3` were used, and were they tuned only on training data within each cross-validation fold?
Q2. What fraction of samples exits at each RF layer, and what are the per-layer accuracy, macro-F1, latency, and energy values?
Q3. Is the headline 96.56% accuracy a single fold result? If so, why is it presented as the overall result instead of the 94.94% cross-validation mean?
Q4. How were KAUH recordings with multiple filter modes handled, and could multiple transformed versions of the same subject/recording appear in different folds?
Q5. What measurement equipment and protocol were used to obtain the 52.5% energy reduction claim?
Q6. Does the reported DPU latency include input tensor preparation, memory transfer, and invocation overhead, or only accelerator compute time?
Q7. How does the proposed cascade compare against a MobileNetV2-only baseline, an RF-only baseline, and a non-cascaded RF+CNN ensemble under the same subject-independent folds?
Q8. What is the teacher ensemble size and teacher performance for knowledge distillation?
Q9. Were any augmentation, normalization, or preprocessing parameters fitted globally before splitting, or only on each training fold?
Q10. What is the performance under cross-database validation, especially given database/device confounding?

## Missing References
List papers or categories.
- Early-exit and cascaded inference methods for efficient edge AI, including confidence/entropy-based exits and adaptive computation.
- Uncertainty estimation and calibration for medical machine-learning classifiers, including reliability diagrams, expected calibration error, and selective classification.
- FPGA and Xilinx DPU/Vitis AI deployment studies for audio, biomedical signal processing, and edge-IoT inference.
- Prior respiratory sound classification works that report subject-independent or patient-level splits, cross-database validation, and device-confounding analysis.
- Edge/mobile respiratory monitoring and digital stethoscope systems with measured latency, power, and energy.
- Quantization-aware training and knowledge distillation literature specific to resource-constrained medical AI.

## Detailed Comments (Section by Section)

### Abstract
- `main.tex:18`: The abstract claims 96.56% average accuracy, but the available cross-validation table reports 94.94% mean and 96.56% for only Fold 0. This should be corrected because the abstract sets the paper's main evidence claim.
- `main.tex:18`: The abstract states that the system reduces latency and model size via QAT, but the visible QAT table reports CPU latency and comments out model size. If model-size reduction is claimed, the actual FP32 and INT8 model sizes should be reported and included.
- `main.tex:18`: The clinical implication is too strong relative to retrospective validation. The final sentence should be moderated unless prospective or external validation is provided.

### Introduction
- `sections/01_introduction.tex:7-13`: The research gap is plausible and relevant, but it should be tied to specific prior works on adaptive/early-exit edge inference, not only respiratory classification papers.
- `sections/01_introduction.tex:15-20`: The contribution list is clear, but the third contribution overstates the result by using 96.56% and 52.5% without enough supporting methodology.
- `sections/01_introduction.tex:19`: The phrase “subject-independent evaluation” is important; the paper should later provide fold construction details and subject/class counts per fold.

### Related Work
- `sections/02_related_work.tex:3-32`: The related work is mainly a catalogue of classification papers and reported accuracies. It does not critically compare evaluation protocols, subject independence, datasets, class definitions, or hardware settings.
- `sections/02_related_work.tex:34-36`: The FPGA gap is stated but not substantiated by a focused review of FPGA/DPU respiratory or biomedical edge inference literature.
- `tables/table_1.tex:9-18`: Accuracy comparisons across different datasets and splits may be misleading. The table should clearly flag non-comparable protocols and include whether each method used patient-independent evaluation.
- `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/supplementary/related-papers/`: The related-papers directory is empty. This limits the ability to verify the comprehensiveness of the literature base; the review therefore relies only on manuscript text and bibliography.

### Methodology
- `sections/03_methodology.tex:7-20`: The dataset and preprocessing subsection is not detailed enough for reproduction. The merged dataset, class mapping, windowing, and filter-mode handling require more precise definitions.
- `sections/03_methodology.tex:20`: The two branches use different segment durations and filtering ranges. This may be technically justified, but the paper does not provide the justification or implementation detail.
- `sections/03_methodology.tex:49-53`: The Hybrid Spectrogram third channel is the average of two normalized spectrograms. The scientific value of this simple averaging should be supported by ablation versus Mel-only, Gammatone-only, and two-channel/three-channel alternatives.
- `sections/03_methodology.tex:59-63`: The distillation objective is specified mathematically, but the teacher ensemble and training protocol are not sufficiently specified.
- `sections/03_methodology.tex:67-85`: The seven-ensemble RF voting mechanism is described, but the independence of ensembles is unclear. If all ensembles use the same features and training data, explain how diversity is induced.
- `sections/03_methodology.tex:87-91`: The expected cost equation is useful, but the manuscript should instantiate it with measured `C_i` and `P_i` values.
- `sections/03_methodology.tex:124-125`: QAT details are minimal. Report quantization framework, calibration data split, batch size, optimizer, and whether calibration/fine-tuning is fold-specific.

### Experiments / Results
- `sections/04_experiments.tex:9-21`: The experimental setup lists datasets and hardware but lacks training hyperparameters, augmentation details, optimizer settings, number of epochs, fold composition, and statistical testing.
- `sections/04_experiments.tex:28-30`: The diagnostic accuracy paragraph reports only 96.56% and no fold variability, macro-F1, class-wise metrics, or confusion matrix.
- `tables/table_7.tex:11-15`: The fold-level results show meaningful variability, particularly in macro-F1. This table should be included and discussed.
- `sections/04_experiments.tex:32-37`: The hardware acceleration benchmark is limited to MobileNetV2 Layer 4. It should not be presented as full-system acceleration without end-to-end cascade measurements.
- `tables/table_perf_cpu_dpu.tex:11-15`: The CPU and DPU comparisons also change precision from FP32 to INT8, so they conflate hardware acceleration with quantization. Include INT8 CPU or FP32-compatible baseline where possible, or explicitly separate the two factors.
- `sections/04_experiments.tex:47-48`: The energy reduction claim needs derivation and measurement details. This is currently a major unsupported result.
- `sections/04_experiments.tex:56-63`: The discussion appropriately acknowledges retrospective validation limitations, but it should also address dataset/device confounding and lack of external validation.

### Conclusion
- `sections/05_conclusion.tex:3`: The conclusion uses 94.94% mean accuracy and reports QAT accuracy on 198 samples, conflicting with other parts of the manuscript and introducing a sample count not explained in the results.
- `sections/05_conclusion.tex:3`: The conclusion says the system was successfully deployed and reduces classification time, but the paper does not provide end-to-end deployment measurements.
- `sections/05_conclusion.tex:5`: The clinical and real-world deployment claims should be tempered until prospective validation, calibration, and workflow evaluation are available.

## Recommendation
[ ] Strong Accept
[ ] Accept
[ ] Weak Accept
[ ] Borderline
[X] Weak Reject
[ ] Reject
Confidence Level: 4/5
