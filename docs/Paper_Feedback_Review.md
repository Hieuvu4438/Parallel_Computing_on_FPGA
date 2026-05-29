# Paper Feedback Review

Review target: `Paper/` LaTeX manuscript  
Scope: feedback only; no manuscript rewriting or direct edits.

## Editorial Decision

**Major Revision before submission.**

The paper has a promising direction: an uncertainty-aware cascaded FPGA/edge inference system for respiratory sound classification. The writing is mostly coherent, and the hardware/software co-design framing is relevant. However, the current manuscript is not yet submission-ready because the central claims about the full cascade, early exits, and energy efficiency are only partially supported by measured results.

The most important issue is that the paper reports strong CNN/KD accuracy, but it does not yet provide enough evidence that the **complete UA-HCI cascade** improves latency/energy while preserving diagnostic performance.

## Main Strengths

- **Clear high-level motivation.** The introduction identifies a real gap: respiratory sound models often optimize accuracy but ignore edge deployment cost and adaptive inference.
- **Good system concept.** The PS/PL division is logical: lightweight RF layers on ARM PS, CNN/DPU path only for uncertain samples.
- **Appropriate subject-independent evaluation.** The manuscript explicitly uses group-based splitting to reduce patient leakage.
- **Useful dataset description.** ICBHI and KAUH are described with class counts, sources, devices, and demographics.
- **Promising accuracy.** Reported 3-fold results are strong: 94.94% mean accuracy and 90.00% macro F1.

## Critical Issues to Fix First

### 1. Full cascade performance is not clearly separated from CNN/KD performance

The results table is titled “Knowledge distillation performance,” but the experiment text says “the proposed UA-HCI framework achieved” those numbers. This creates ambiguity:

- Are the reported metrics from the **Layer-4 MobileNetV2 student only**?
- Or from the **complete 4-layer cascade** with RF early exits?
- If complete cascade, what percentage exits at Layer 1, 2, 3, and 4?
- What is the accuracy of each exit route?

This is the most serious scientific clarity problem.

**Recommendation:** Add a route-level table:

| Route | Exit % | Route accuracy | Macro F1 | Error contribution | Mean latency |
|---|---:|---:|---:|---:|---:|
| Layer 1 exit | x% | x% | x% | x% | x ms |
| Layer 2 exit | x% | x% | x% | x% | x ms |
| Layer 3 exit | x% | x% | x% | x% | x ms |
| Layer 4 CNN | x% | x% | x% | x% | x ms |
| Full cascade | 100% | 94.94% | 90.00% | — | x ms |

Without this, reviewers may say the paper is mainly a CNN/KD classifier plus a proposed cascade, not a fully validated cascaded system.

### 2. Energy-efficiency claim is not yet demonstrated

The abstract carefully says the paper defines the measurements required to quantify route-level energy savings, but the title and framing emphasize hardware acceleration and energy reduction. The current energy table still contains `TBD` values. The end-to-end latency table also contains `TBD` values.

**Problem:** If route-level energy and full pipeline latency are not measured, reviewers will ask what the actual hardware contribution is.

**Recommendation:** Either:

1. Measure and report route-level latency/energy, or
2. Downgrade the claims and frame the work as an architecture plus partial hardware profiling rather than a proven energy-efficient deployment.

For a strong submission, option 1 is much better.

### 3. Early-exit threshold calibration is under-specified

The manuscript gives thresholds:

- `tau_1 = 0.95`
- `tau_2 = 0.90`
- `tau_3 = 0.85`
- `lambda = 4`

But it does not explain how these thresholds were selected.

Reviewers will ask:

- Were thresholds tuned on validation folds only?
- Were they selected once globally or per fold?
- Was there a latency/accuracy trade-off curve?
- Is vote support really a calibrated uncertainty measure?

The paper calls the method “uncertainty-aware,” but the uncertainty score is currently normalized vote agreement. That is useful, but it is not the same as calibrated uncertainty.

**Recommendation:** Add:

- threshold search procedure;
- accuracy vs. Layer-4 routing percentage curve;
- expected cost curve;
- calibration metric such as ECE, Brier score, or reliability diagram if you keep the term “uncertainty-aware.”

### 4. Segment and record handling need more precision

The methodology says:

- unit of inference is an 8-second segment;
- RF layers operate on 5-second segments;
- dataset contains 1,256 recordings.

This raises questions:

- Are metrics computed per recording or per segment?
- If a recording is longer than 8 seconds and center-cropped, is only one segment used?
- If recordings are shorter, does repeat-padding duplicate acoustic patterns and affect generalization?
- Why do RF layers use 5 s while CNN uses 8 s?
- How are 5-second RF features aligned with 8-second CNN spectrograms?

**Recommendation:** Add a short “Inference unit and label aggregation” paragraph. Be explicit about whether each recording produces exactly one standardized sample.

### 5. The merged dataset may introduce confounding

The paper merges ICBHI and KAUH and maps labels into COPD, Healthy, and Non-COPD. This is reasonable, but it creates potential dataset-source confounding:

- COPD is heavily dominated by ICBHI.
- KAUH has only a small COPD subject count.
- Healthy and Non-COPD may have different source/device distributions.

A model may partially learn dataset/device/filter artifacts instead of disease acoustics.

**Recommendation:** Add source-aware analysis:

- performance by source dataset: ICBHI vs. KAUH;
- confusion matrix per source;
- leave-one-dataset-out or train-on-ICBHI/test-on-KAUH if possible;
- check whether source identity can be predicted from features.

## Methodology Reviewer Comments

### Major

- **Non-COPD class is too heterogeneous.** It combines asthma, heart failure, pneumonia, bronchiectasis, URTI, fibrosis, bronchitis, and pleural effusion. This makes clinical interpretation difficult. The paper should justify why this aggregation is clinically meaningful.
- **Class imbalance is substantial.** COPD dominates the dataset, while Healthy is much smaller. Accuracy alone is not enough. Macro F1 is reported, which is good, but balanced accuracy and per-fold confusion matrices would strengthen the paper.
- **Only 3 folds.** The explanation is reasonable, but 3-fold CV may still be considered weak for a small clinical dataset. Consider repeated GroupKFold or report confidence intervals.
- **RF layer training details are thin.** You give trees, depth, and ensembles, but not class weighting, bootstrap settings, feature scaling, random seeds, fold-specific tuning, or training/validation split inside each fold.

### Minor

- Layer 1 using only RMSE may be too weak for disease classification. If it exits samples, you need route-specific evidence that it is safe.
- Layer 2 using only ZCR has similar risk. It may detect transients, but disease-level classification from ZCR alone needs stronger justification.
- The “seven independent ensembles” design should explain independence: different seeds, bootstrap subsets, feature subsets, or data partitions.

## Hardware Reviewer Comments

### Major

- **DPU latency is reported only for Layer 4.** The CNN latency numbers are useful, but the full system also includes preprocessing, RF feature extraction, spectrogram generation, and transfer overhead.
- **Energy model is not populated.** The energy table is a template. This weakens the FPGA/energy contribution.
- **Resource utilization needs provenance.** The resource table reports LUT/FF/BRAM/DSP usage, but reviewers may ask whether these are measured from Vivado synthesis, implementation, or estimated. Add tool version, DPU config, clock, bitstream source, and whether timing closure was achieved.

### Minor

- PYNQ-Z2 DPU feasibility may need careful explanation. PYNQ-Z2 is resource-constrained, and the reported utilization is high. Clarify whether the same MobileNetV2 model actually runs there or whether this is a baseline/profiling configuration.
- The paper should report batch size consistently. Make sure all latency tables use the same condition.

## Related Work and Novelty

The related work is directionally good, but the novelty argument needs sharper positioning.

Current related work says most systems are static classifiers and do not exploit uncertainty-aware inference. This is a strong angle, but the paper should compare against:

- early-exit neural networks such as BranchyNet or MSDNet;
- cascaded classifiers in biomedical signal processing;
- FPGA/DPU medical AI deployments;
- respiratory sound models with patient-independent evaluation.

The bibliography already contains BranchyNet and MSDNet entries, but they should be discussed in the main text if cascaded inference is central to the contribution.

## Writing and Presentation Issues

### High-priority wording fixes

- The conclusion says the system “achieved the expected accuracy and performance.” This is too vague and slightly too strong because route-level latency/energy are not completed.
- The abstract is honest about defining measurements rather than completing all route-level measurements, but that may make the paper sound incomplete. The paper will be much stronger if the measurements are filled.
- The claim that COPD sensitivity ensures samples are correctly identified and routed to the DPU is logically unclear. Sensitivity describes classification performance, not routing behavior.

### Table/file cleanup

- Placeholder projected tables with `[XX]` values should not be included in a submitted manuscript.
- `TBD` latency/energy tables should be treated as internal planning tables unless the venue accepts measurement-framework papers.

## Citation and Bibliography Concerns

- Some BibTeX entries are incomplete, such as entries using “and others” with limited metadata.
- Verify that the KAUH database citation is truly the correct source and not a duplicated or mismatched reference.
- The manuscript should prioritize respiratory-sound-specific and edge-FPGA-specific citations over generic audio-feature citations.
- Early-exit/cascaded inference literature should be cited if “uncertainty-aware cascaded inference” is central.

## Suggested Revision Roadmap

### Must fix before submission

1. Clarify whether reported accuracy is CNN-only or full cascade.
2. Add route-level exit percentages and route-level accuracy.
3. Fill or remove `TBD` latency/energy tables.
4. Explain threshold calibration and prevent validation leakage.
5. Add source-wise or dataset-wise analysis for ICBHI vs. KAUH.
6. Tighten claims about energy efficiency and clinical utility.

### Should fix

1. Add balanced accuracy and confusion matrices.
2. Add ablation:
   - CNN only;
   - RF cascade only;
   - UA-HCI full cascade;
   - no KD vs. KD;
   - FP32 vs. INT8.
3. Add calibration/reliability analysis for the “uncertainty-aware” claim.
4. Discuss Non-COPD heterogeneity more carefully.
5. Improve related work on early-exit models and FPGA medical AI.

### Nice to have

1. Report model size and memory footprint.
2. Add per-class route distribution.
3. Add Grad-CAM or spectrogram examples for qualitative interpretability.
4. Add reproducibility details: seeds, libraries, Vitis AI version, DPU architecture, exact split protocol.

## Bottom Line

The paper has a strong engineering idea and promising classification results, but it currently reads like a **partially validated cascaded FPGA framework** rather than a fully demonstrated energy-efficient respiratory diagnosis system. If you add full-cascade route metrics, threshold calibration, and actual end-to-end energy/latency measurements, the paper will become much stronger and more defensible.
