# Final QA Report — Submission Readiness
Round: 1
Target journal: IEEE Internet of Things Journal
Overall Readiness: NOT READY — 8 critical issues remain

## Unresolved Critical Issues from Previous Reviews
- **Source report:** content
- **Issue:** Primary diagnostic accuracy remains inconsistent and overstated: the abstract, contribution list, results, and discussion claim 96.56% as the achieved/average framework result, while the available cross-validation table shows 96.56% only for Fold 0 and 94.94% as the mean.
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:18; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\01_introduction.tex:19; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\04_experiments.tex:30,63; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\05_conclusion.tex:3; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_7.tex:11-15
- **Status:** Unfixed
- **Evidence:** `table_7.tex` reports Fold 0 accuracy 96.56%, Fold 1 accuracy 94.25%, Fold 2 accuracy 94.02%, and Mean accuracy 94.94%, but the active manuscript text still reports 96.56% as the headline result and `table_7.tex` is not included in the active manuscript.

- **Source report:** content
- **Issue:** Early-exit thresholds remain unspecified; the manuscript names $\tau_i$ but does not provide actual values, tuning protocol, fold-specific validation procedure, or sensitivity analysis.
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:77-85; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_6.tex:17; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\04_experiments.tex:24
- **Status:** Unfixed
- **Evidence:** `table_6.tex` lists “Early-exit threshold & $\tau_i$, $i \in \{1,2,3\}$” instead of numeric thresholds; the experiments section repeats the threshold condition but gives no selected values or selection method.

- **Source report:** content
- **Issue:** The 52.5% expected energy-reduction claim remains unsupported by measured power, route probabilities, per-layer energy/latency, or a reproducible baseline definition.
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\01_introduction.tex:19; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\04_experiments.tex:47-48,63
- **Status:** Unfixed
- **Evidence:** The text asserts a 52.5% reduction, while the only cost equation in `sections\03_methodology.tex:87-91` defines symbolic $C_i$ and $P_i$ without instantiating values. No power table or exit-rate table is included.

- **Source report:** references
- **Issue:** The KAUH dataset citation remains incorrect or unverifiable; `kauh_db` still points to Park et al. Scientific Reports pediatric lung-sound model rather than a verified King Abdullah University Hospital database source.
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\references.bib:639-652; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:12
- **Status:** Unfixed
- **Evidence:** `kauh_db` duplicates the Park et al. article and adds only `note = "King Abdullah University Hospital respiratory sound database"`; no URL, DOI, data descriptor, repository, or access date verifies the stated 112-subject KAUH database.

- **Source report:** references
- **Issue:** RDLINet/ILDNet source associations remain inconsistent between the related-work prose and comparison table.
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\02_related_work.tex:26,28; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_2.tex:16-17; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\references.bib:179-204
- **Status:** Unfixed
- **Evidence:** The prose cites `b13` for RDLINet and `b14` for ILDNet, but `table_2.tex` uses `b13` for ILDNet/81.25% and `b14` for RDLINet/96.6%-99.6%.

- **Source report:** structure
- **Issue:** IEEE Internet of Things Journal-specific submission requirements still cannot be verified because the local guideline folder is empty.
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\journal-guidelines\; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:1
- **Status:** Verify manually
- **Evidence:** The manuscript uses IEEEtran, but journal-specific page limits, initial-submission metadata rules, ORCID requirements, graphical abstract/cover-letter requirements, ethics/funding/data-availability order, and figure-resolution rules are not locally available.

## New Issues Found
- **Severity:** CRITICAL
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:18; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\05_conclusion.tex:3; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_qat.tex:11-14
- **Issue:** The abstract claims QAT reduces “latency and model size,” while the active manuscript does not include the QAT table and the only QAT source table comments out model size; the conclusion additionally claims a QAT accuracy drop and 82.7% CPU-latency reduction on 198 samples without an active table, sample-count explanation, or DPU measurement linkage.
- **Why it matters:** QAT and hardware acceleration are central contributions. Unsupported model-size and sample-count claims are likely to be challenged by reviewers and cannot be verified from the compiled paper.
- **Suggested fix:** Include a QAT results table in the active manuscript or remove all unsupported QAT/model-size/sample-count claims. If retained, report FP32/INT8 model sizes, test-set size provenance, CPU and DPU latency, and whether the QAT results are fold-specific or from a separate split.

- **Severity:** CRITICAL
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\04_experiments.tex:28-30; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_7.tex:1-19
- **Issue:** The cross-validation results table is not included in the active manuscript, yet the results section reports a single accuracy value and the conclusion reports a mean value.
- **Why it matters:** Readers cannot verify the headline diagnostic result, fold variability, macro-F1, or whether 96.56% is a fold result rather than an aggregate result.
- **Suggested fix:** Add `\input{tables/table_7.tex}` near the diagnostic-accuracy discussion and revise the text to report mean ± dispersion as the main result, with 96.56% identified only as the best fold.

- **Severity:** CRITICAL
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:7-20; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\04_experiments.tex:9-12; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_kauh.tex:17,40-46
- **Issue:** Methodology and experiments still do not explain how KAUH’s three exported filter modes are handled in the merged dataset or folds.
- **Why it matters:** If multiple filtered versions of the same recording are treated as independent samples or split across folds, the reported subject-independent validation may contain leakage or inflated performance.
- **Suggested fix:** State exactly which KAUH filter mode(s) are used, whether duplicates are removed, how recordings are counted, and how subject IDs/recording variants are grouped during fold assignment.

- **Severity:** MAJOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:97-109
- **Issue:** Figures 2 and 3 are present and labeled but are not explicitly referenced in the surrounding text.
- **Why it matters:** Every figure should be introduced and interpreted in the manuscript. Uncalled figures weaken narrative flow and may be flagged during formatting review.
- **Suggested fix:** Add callouts before the figure environments, e.g., refer to Fig. 2 when explaining septuple RF voting and Fig. 3 when explaining the UA-HCI dataflow.

- **Severity:** MAJOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:31-90
- **Issue:** Seven displayed equations are numbered but none has a `\label{}` or textual equation reference.
- **Why it matters:** Important equations such as the early-exit decision and expected cost should be referable; otherwise automatic numbering adds clutter without improving traceability.
- **Suggested fix:** Label and cite the central equations, especially the routing decision and expected-cost equation, or convert nonessential displays to unnumbered equations.

- **Severity:** MAJOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:20; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_5.tex:12-15
- **Issue:** The RF branch uses 5-s, 50--2500 Hz inputs while the CNN branch uses 8-s, 25--2000 Hz inputs, but the experiments do not define the common unit of inference or how a routed sample is transformed between branches.
- **Why it matters:** The cascade cannot be reproduced without knowing whether routing decisions apply to recordings, cycles, 5-s windows, or 8-s windows and how labels/segments are aligned.
- **Suggested fix:** Define the inference unit and provide a routing/segmentation protocol that maps each sample consistently across Layers 1--4.

- **Severity:** MAJOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\02_related_work.tex:3-36; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:55-63,113-125
- **Issue:** Related Work still covers mostly respiratory-classification accuracy papers and does not substantively cover early-exit inference, uncertainty/calibration, FPGA/DPU edge deployment, Vitis AI/PYNQ, QAT, knowledge distillation, or MobileNetV2/EfficientNet methods that the manuscript uses.
- **Why it matters:** The paper’s novelty is a heterogeneous uncertainty-aware FPGA cascade, so the closest methodological and hardware baselines must be reviewed and cited for IEEE Internet of Things Journal.
- **Suggested fix:** Expand Related Work with focused subsections or paragraphs on adaptive/early-exit inference, uncertainty-aware cascades, edge medical-AI deployment, Xilinx DPU/Vitis AI systems, QAT, and distillation; cite verified sources.

- **Severity:** MAJOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\05_conclusion.tex:3-5
- **Issue:** The conclusion overstates deployment and clinical readiness by claiming successful deployment, reliability, superior parallel processing, reduced waiting times, real-world/clinical use, and enhanced patient care from retrospective/offline evidence.
- **Why it matters:** The manuscript itself states that prospective validation is still required. Strong clinical and deployment claims exceed the demonstrated evidence and may be viewed as unsafe for a medical IoT paper.
- **Suggested fix:** Recast the conclusion as retrospective/offline evidence only, limit deployment claims to the measured CNN/DPU block unless end-to-end measurements are added, and state prospective validation as future work.

- **Severity:** MAJOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_perf_cpu_dpu.tex:11-15; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\04_experiments.tex:32-37
- **Issue:** Best-result bolding and performance interpretation conflate hardware execution with precision change: CPU rows are FP32 and DPU rows are INT8, yet the text attributes latency reduction to DPU execution.
- **Why it matters:** The comparison does not isolate hardware acceleration from quantization. Reviewers may see the benchmark as unfair or incomplete.
- **Suggested fix:** Add matched baselines where possible, such as INT8 CPU and/or FP32 hardware-compatible measurements, or explicitly frame the result as combined DPU+INT8 acceleration rather than pure hardware acceleration.

- **Severity:** MAJOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:16; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_4.tex:11-13; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\04_experiments.tex:28-30
- **Issue:** The manuscript reports accuracy for an imbalanced dataset but does not include class-wise sensitivity, specificity, precision/recall, or confusion matrices in the active results.
- **Why it matters:** With 65.3% COPD, 23.6% Non-COPD, and 11.1% Healthy, accuracy can mask poor minority-class detection in a medical screening context.
- **Suggested fix:** Add fold-wise and aggregate class-wise metrics, confusion matrices, and macro-F1 discussion. Make macro-F1 at least as prominent as accuracy.

- **Severity:** MINOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:18; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\01_introduction.tex:13; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:49-53
- **Issue:** Terminology is inconsistent: the abstract uses “Artificial Neural Network,” “Convolutional Neural Network (CNN),” and “Hybrid Spectrogram,” while the method specifies a MobileNetV2 student and varies capitalization of technical terms.
- **Why it matters:** Inconsistent terminology obscures the exact architecture and makes the contribution harder to follow.
- **Suggested fix:** Use one formulation throughout, such as “Random Forest early exits plus an INT8 QAT MobileNetV2 student using hybrid spectrogram inputs on the Xilinx DPU.”

- **Severity:** MINOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:18; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\01_introduction.tex:13; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:125; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\04_experiments.tex:26
- **Issue:** Acronyms and technical terms are not consistently defined on first use in each context; for example QAT appears in the abstract after expansion but DPU is not in the abstract, and Focal Loss appears without definition or citation.
- **Why it matters:** IEEE readers should be able to understand acronyms from first use, including in the abstract, which often stands alone in indexing systems.
- **Suggested fix:** Expand all acronyms at first use in the abstract and main text where needed, including DPU if mentioned, and standardize capitalization of focal loss, mel, MFCCs, and hardware--software co-design.

- **Severity:** MINOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_8.tex:15-27; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_9.tex:12-15
- **Issue:** Placeholder values remain in unused table source files.
- **Why it matters:** They are not active manuscript inputs, but they could confuse collaborators or accidentally enter the submission package.
- **Suggested fix:** Exclude unused placeholder tables from the submission package or move them to an archive folder; if they are later included, replace all placeholders before submission.

- **Severity:** MINOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:13; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\05_conclusion.tex:11-14
- **Issue:** Submission metadata appears provisional: “Manuscript received 2025” lacks full production metadata, and the acknowledgment thanks editors/referees before peer review.
- **Why it matters:** These phrases may be inappropriate for initial submission and can make the manuscript look like an accepted-production version.
- **Suggested fix:** Verify the journal’s initial-submission template requirements. Remove or revise provisional production metadata and premature referee thanks unless specifically required.

## Pre-Submission Checklist
Use `[✓]`, `[✗]`, `[?]`.

### Document
- [?] Page count OK
- [✓] Required sections present
- [✗] No placeholders remain

### Cross-References
- [✗] Figures referenced and present
- [✓] Tables referenced and present
- [?] Equations referenced and present
- [✓] Sections referenced and present

### Numbers and Claims
- [✗] Abstract numbers match results
- [✗] Conclusion matches evidence
- [✗] Units and precision are consistent

### Submission Package
- [?] PDF builds successfully if checked
- [?] Fonts embedded if checked
- [?] Supplementary files ready if needed
- [?] Cover letter ready if needed

## Final Recommendation
NOT READY for IEEE Internet of Things Journal submission. The exact next action is to fix the critical result-consistency and evidence-support issues before any formatting polish: use the 94.94% cross-validation mean as the headline diagnostic result, include the cross-validation and QAT evidence tables or remove unsupported claims, specify early-exit thresholds and energy calculations, correct the KAUH/RDLINet/ILDNet citations, and rerun final QA after the active manuscript and bibliography are updated.
