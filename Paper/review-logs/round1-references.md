# Reference & Citation Audit Report
Round: 1
Target journal: IEEE Internet of Things Journal

## Summary
- Total references: 47
- Total citation keys used: 14
- Orphan references in `.bib` but not cited: [b04, b05, b06, b17, b18, b19, b20, b21, b22, b23, b24, b25, b26, b27, b28, b29, b30, b31, b32, b33, b34, b35, b36, b37, b38, b39, b40, b41, b42, b43, b44, b45, b46]
- Dangling citations cited but not in `.bib`: []
- Incomplete entries: 43
- Duplicate or suspected duplicate entries: [b46 and kauh_db: same Park et al. Sci. Rep. article; b13 and b14 appear swapped/inconsistent between bibliography metadata and manuscript/table descriptions]
- Recency score: 40.4% from last 3 years

## Critical Issues
- **Severity:** CRITICAL
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/references.bib`:639, citation key `kauh_db`; cited at `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/03_methodology.tex`:12
- **Issue:** The KAUH database citation key points to Park et al., "A machine learning approach to the development and prospective evaluation of a pediatric lung sound classification model," not to the King Abdullah University Hospital respiratory sound database source. It is also a duplicate of `b46` with only a database note added.
- **Why it matters:** The manuscript's dataset provenance is not verifiable. A wrong database citation is a serious reproducibility and citation-integrity problem because readers cannot locate the KAUH data or confirm the claimed 112-subject, 3M Littmann 3200 database details.
- **Suggested fix:** Replace `kauh_db` with the original KAUH database/data-descriptor/publication citation, including accurate authors, title, venue or repository, year, URL/DOI if available, and access date if it is a web dataset. Do not cite the Park pediatric lung-sound article unless it is actually the source of the database.

- **Severity:** CRITICAL
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/02_related_work.tex`:26, 28; `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/tables/table_2.tex`:16, 17; BibTeX entries `b13`, `b14`
- **Issue:** The manuscript text/table appear to swap `b13` and `b14`. The bibliography has `b13` = RDLINet (IEEE Trans. Instrum. Meas., 2023) and `b14` = ILDNet (SPCOM, 2024), but line 26 cites `b13` while describing RDLINet with 99.6% six-class accuracy, and line 28 cites `b14` while describing ILDNet with BRACETS/KAUH and 81.25%. Table 2 reverses the paper names/datasets/accuracies relative to the bibliography.
- **Why it matters:** Reviewers checking prior-work claims will find that cited sources do not support the stated dataset, model, and accuracy claims. This undermines the literature review and comparative table.
- **Suggested fix:** Verify the original RDLINet and ILDNet papers, then align each manuscript paragraph and table row with the correct key, model name, venue, dataset, and reported metrics. If the bibliography entries are correct, edit the table and text associations accordingly.

- **Severity:** MAJOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/references.bib`:all entries; orphan keys listed in Summary
- **Issue:** 33 of 47 bibliography entries are not cited anywhere in the active manuscript inputs.
- **Why it matters:** IEEE reference lists should include only works cited in the manuscript. Orphan entries inflate the bibliography, distort recency statistics, and may signal unintegrated or irrelevant literature.
- **Suggested fix:** Remove unused entries from `references.bib` or cite them in appropriate locations only if they directly support manuscript claims.

- **Severity:** MAJOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/03_methodology.tex`:30-46
- **Issue:** Definitions and use of RMSE, ZCR, MFCC, Mel scale, and MFCC delta statistics are uncited, despite multiple potentially relevant unused references existing in `references.bib` (`b19`, `b20`, `b22`, `b28`).
- **Why it matters:** Methodological feature definitions need authoritative citations, especially when used as core components of the proposed diagnostic pipeline.
- **Suggested fix:** Cite foundational or domain-relevant sources for MFCC/Mel scale and ZCR/audio features at the first mention, and prefer respiratory-sound feature references where available.

- **Severity:** MAJOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/03_methodology.tex`:55-63, 124-125; `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/04_experiments.tex`:20, 26
- **Issue:** MobileNetV2, EfficientNet-B0, knowledge distillation, focal loss, QAT, Xilinx Vitis AI, PYNQ, and DPU deployment are discussed without citations. `b35` cites MobileNetV2 but is currently orphaned; no cited entries support EfficientNet, distillation, focal loss, Vitis AI, PYNQ, or DPU.
- **Why it matters:** The paper's hardware-software contribution depends on these architectures/toolchains/training methods. Unsupported method claims weaken technical credibility and reproducibility.
- **Suggested fix:** Cite MobileNetV2 (`b35`, if complete), add verified citations for EfficientNet, knowledge distillation, focal loss, quantization-aware training, Vitis AI/DPU documentation or papers, and PYNQ/Ultra96 platform sources.

- **Severity:** MAJOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/tables/table_1.tex`:10-19
- **Issue:** The respiratory sound frequency/duration/timing characteristics table has no citations.
- **Why it matters:** The table contains clinical/acoustic factual claims that must be traceable to authoritative pulmonary auscultation or respiratory acoustics sources.
- **Suggested fix:** Add citations in the caption or relevant rows to validated respiratory sound nomenclature/characteristics sources; reuse a corrected `b41` only if it supports these exact values, otherwise add more authoritative clinical references.

- **Severity:** MAJOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/04_experiments.tex`:10-20, 48, 57, 63
- **Issue:** Dataset merging rationale, subject-independent protocol, hardware platform specifications, PYNQ/Vitis AI stack, and energy-reduction claims lack external citations or explicit linkage to measured internal data.
- **Why it matters:** Reviewers need to distinguish original measurements from claims based on platform specifications, prior literature, or tool documentation. Unsupported energy and deployment claims are especially sensitive for IEEE IoT Journal.
- **Suggested fix:** Cite platform/tool documentation for hardware and deployment stack; state whether latency/energy values are measured or estimated; cite any energy model or include equations/data that support the 52.5% claim.

- **Severity:** MAJOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/references.bib`:94, 107, 121, 133, 147, 163, 179, 191, 206, 216, 243, 262, 276, 289, 303, 318, 332, 345, 369, 388, 401, 415, 428, 441, 459, 473, 488, 502, 517, 532, 566, 586, 601, 609, 625, 639
- **Issue:** Most entries are missing DOI and/or complete publication metadata. Examples: cited entries `b07`, `b08`, `b09`, `b10`, `b11`, `b12`, `b13`, `b14`, `b15`, `b16`, and `kauh_db` lack DOI fields; several conference entries lack pages; `b44` uses the nonstandard type `@unknown`.
- **Why it matters:** IEEE bibliographies should be complete enough for indexing and retrieval. Missing DOIs/pages/venues reduce traceability and can trigger editorial corrections.
- **Suggested fix:** For each retained source, verify metadata against IEEE Xplore, Crossref, PubMed, Springer, publisher pages, or official arXiv records; add DOI, pages/article number, venue, publisher, and URL where applicable.

- **Severity:** MINOR
- **Location:** `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/main.tex`:33; `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/preamble.tex`:11
- **Issue:** The manuscript uses `IEEEtran` bibliography style and the `cite` package, which is consistent with numbered IEEE style, but many in-text citations are attached to author-year prose/table cells, e.g., "(Ariyanti et al., 2023) \cite{b07}" in `table_2.tex`.
- **Why it matters:** IEEE style is numbered; author-year labels in tables can be acceptable as descriptive text but may look inconsistent or redundant in a numbered-reference manuscript.
- **Suggested fix:** Prefer author names plus numbered citation only, e.g., "Ariyanti et al. [n]", or keep author-year only if the target journal template permits it.

## Reference-by-Reference Audit
| Key | Status | Issues | Suggested fix |
|---|---|---|---|
| b01 | Cited | Mostly complete; uses nonstandard `elocation-id` instead of standard `pages`/article-number field. | Retain; consider mapping article number 1900164 into IEEE-friendly pages/article field. |
| b02 | Cited | Complete enough; includes DOI, volume, issue, pages/article number. | Retain. |
| b03 | Cited | Missing DOI. Older global-burden projection may be acceptable but should be checked for currentness. | Add DOI if available; consider supplementing with recent WHO/GBD COPD burden source. |
| b04 | Orphan | Uncited; missing DOI. | Remove unless used to support COPD economic burden. |
| b05 | Orphan | Uncited; missing DOI. | Remove unless used to support ED/hospital burden. |
| b06 | Orphan | Uncited. Could support FPGA background but no manuscript citation uses it. | Remove or cite where FPGA architecture is introduced if directly relevant. |
| b07 | Cited | Missing DOI; journal field is conference name; author list appears suspicious/incomplete (`{Yu-Tsao}`). | Verify IEEE EMBC metadata, authors, DOI, and proceedings fields. |
| b08 | Cited | Missing DOI and pages. | Verify IEEE DICCT metadata and add DOI/pages. |
| b09 | Cited | Missing DOI and issue/article number if applicable. | Add DOI for IEEE Access article. |
| b10 | Cited | Missing DOI and pages. | Verify IEEE APCCAS metadata and add DOI/pages. |
| b11 | Cited | Missing DOI. | Add DOI if available. |
| b12 | Cited | Missing DOI. | Add DOI if available. |
| b13 | Cited | Missing DOI/number; manuscript/table likely misalign this entry with ILDNet/RDLINet claims. | Verify RDLINet article details and align citation claims; add DOI. |
| b14 | Cited | Missing DOI; manuscript/table likely misalign this entry with ILDNet/RDLINet claims. | Verify ILDNet conference details and align citation claims; add DOI. |
| b15 | Cited | arXiv preprint only; missing version/access date; lower evidentiary weight than peer-reviewed work. | Use only if necessary; add arXiv version/access date or replace/supplement with peer-reviewed source. |
| b16 | Cited | Missing DOI/URL; database source otherwise plausible for ICBHI. | Add DOI/URL if available and ensure it is the canonical ICBHI 2017 database citation. |
| b17 | Orphan | Uncited; missing DOI. | Remove or cite for respiratory CNN prior work if relevant. |
| b18 | Orphan | Uncited; unrelated EEG feature extraction; missing DOI. | Remove unless explicitly used for energy feature theory. |
| b19 | Orphan | Uncited; missing DOI; could support audio feature extraction. | Cite in feature-extraction section if retained; add DOI. |
| b20 | Orphan | Uncited; missing DOI/pages; music pitch context not respiratory-specific. | Remove or replace with more relevant ZCR/audio source. |
| b21 | Orphan | Uncited; missing DOI; combustion domain not respiratory-specific. | Remove unless used only for general ZCR rationale. |
| b22 | Orphan | Uncited; missing DOI/pages. | Remove or cite for robust ZCR only if relevant. |
| b23 | Orphan | Uncited; missing DOI/pages. | Remove unless supporting auditory scene/audio features. |
| b24 | Orphan | Uncited; no DOI. | Remove unless supporting timbral features. |
| b25 | Orphan | Complete enough but not used. | Remove unless cited for MPEG-7/audio features. |
| b26 | Orphan | Incomplete: no venue/publisher/pages/DOI. | Remove or replace with complete source. |
| b27 | Orphan | Title duplicates journal citation text. | Remove or clean title if retained for wavelet background. |
| b28 | Orphan | Missing DOI; foundational MFCC paper. | Cite at MFCC definition if retained; add DOI if available. |
| b29 | Orphan | Speech age/gender recognition, not directly relevant. | Remove. |
| b30 | Orphan | Random forest vs ANN in building energy, weak relevance. | Remove or replace with RF foundational citation. |
| b31 | Orphan | Colorectal cancer PCA, weak relevance. | Remove. |
| b32 | Orphan | Random forest big data, missing DOI. | Remove or replace with foundational Random Forest citation. |
| b33 | Orphan | MobileNet in pick-and-place, weak relevance. | Remove; cite original MobileNetV2 instead. |
| b34 | Orphan | Tomato seed MobileNet-BiLSTM, weak relevance. | Remove unless directly compared. |
| b35 | Orphan | Uncited despite manuscript using MobileNetV2; missing DOI/pages. | Cite in MobileNetV2 methodology and add DOI/pages. |
| b36 | Orphan | ShuffleNet not discussed. | Remove unless comparison includes ShuffleNet. |
| b37 | Orphan | FPGA PUF paper, not relevant to DPU acceleration. | Remove. |
| b38 | Orphan | Incomplete: no venue/pages/DOI; weak relevance. | Remove or replace with authoritative FPGA embedded processor source. |
| b39 | Orphan | Breast-cancer classifier, weak relevance. | Remove. |
| b40 | Orphan | Fish stock prediction, irrelevant. | Remove. |
| b41 | Orphan | Incomplete metadata but potentially relevant for respiratory sound preprocessing. | Verify and cite if it supports preprocessing/acoustic characteristics; add complete metadata. |
| b42 | Orphan | Speech vocoder, likely unrelated. | Remove. |
| b43 | Orphan | Respiratory CNN/LSTM prior work, missing DOI. | Cite in related work if relevant and add DOI. |
| b44 | Orphan | `@unknown`, no venue/pages/DOI; likely unsuitable. | Remove or replace with peer-reviewed pneumonia respiratory-sound source. |
| b45 | Orphan | Respiratory sound DNN, missing DOI. | Cite in related work if relevant and add DOI. |
| b46 | Orphan / suspected duplicate | Duplicate bibliographic record of `kauh_db`; missing DOI. | Keep one corrected Park entry only if cited; otherwise remove. |
| kauh_db | Cited / incorrect | Duplicate of `b46` and appears not to be the actual KAUH database citation; missing DOI/URL. | Replace with verified KAUH database source. |

## Citation Gap Analysis
- `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/01_introduction.tex`:3: Claim that abnormal respiratory sounds carry clinically meaningful information about airway obstruction and pulmonary function needs a respiratory acoustics/auscultation citation in addition to the COPD GOLD citation.
- `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/tables/table_1.tex`:10-19: All disease-specific respiratory sound frequency, duration, timing, continuity, and type values need source citations.
- `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/01_introduction.tex`:5: Underdiagnosis claim needs a COPD underdiagnosis/public-health citation; current `b02,b03` support burden/projections but not necessarily underdiagnosis.
- `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/01_introduction.tex`:9: Claim that fewer works jointly address uncertainty-aware inference, low-power edge execution, and FPGA co-design needs citations to edge/FPGA respiratory-sound or edge medical-AI literature to establish the gap.
- `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/01_introduction.tex`:19: Claim of 52.5% expected energy reduction needs either an internal equation/table reference or a citation/model explaining the energy calculation.
- `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/03_methodology.tex`:16: The merged class distribution and disease mapping need clearer dataset-source support and/or a reproducible preprocessing citation/data statement.
- `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/03_methodology.tex`:20: Resampling to 4 kHz, mono conversion, normalization, band-pass ranges 50--2500 Hz and 25--2000 Hz need respiratory sound preprocessing citations or a rationale.
- `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/03_methodology.tex`:30-46: RMSE, ZCR, MFCC, Mel scale, and delta-statistics definitions need foundational feature-extraction citations.
- `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/03_methodology.tex`:49-53: Gammatonegram, Mel-spectrogram, hybrid spectrogram, 64-filter/128-Mel choices, and 224x224 resizing need signal-processing/model-input citations or justification.
- `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/03_methodology.tex`:55-63: MobileNetV2, EfficientNet-B0 teacher ensemble, knowledge distillation, KL-temperature loss, and focal loss need citations.
- `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/03_methodology.tex`:87-91: Expected computational cost formulation needs an early-exit/cascaded inference citation if presented as established methodology.
- `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/03_methodology.tex`:113-125: ARM PS/PL allocation, Xilinx DPU, AXI interface, INT8 MAC execution, calibration over 32 batches, and QAT need platform/toolchain citations.
- `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/04_experiments.tex`:10-20: Hardware platform specifications and software stack need citations to board datasheets, PYNQ, Vitis AI, and DPU documentation.
- `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/04_experiments.tex`:48: 52.5% expected energy reduction compared with static CNN-style deployment needs a citation to the energy model or an internal table/equation reference with measured assumptions.
- `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/sections/04_experiments.tex`:60: Grad-CAM mention needs the original Grad-CAM citation if retained.

## Suggested Additional References
- Authoritative KAUH respiratory sound database source: add the original dataset paper/repository citation for the King Abdullah University Hospital respiratory sound database. VERIFY RELEVANCE and exact metadata.
- Respiratory sound nomenclature/acoustics: add authoritative clinical or biomedical engineering references for wheeze, crackle, rhonchi, stridor, frequency ranges, and timing. Prioritize peer-reviewed respiratory acoustics standards/reviews. VERIFY RELEVANCE.
- COPD epidemiology/underdiagnosis: add recent WHO/GBD/GOLD or Lancet Respiratory Medicine/European Respiratory Journal references to support prevalence, mortality, and underdiagnosis claims. VERIFY RELEVANCE.
- Foundational feature extraction: cite Davis and Mermelstein for MFCC (`b28`, after completion), a ZCR/audio-feature reference, and a respiratory sound preprocessing source. VERIFY RELEVANCE.
- Model architecture/training: add original MobileNetV2 (already `b35`, cite and complete), EfficientNet-B0, knowledge distillation, focal loss, and Grad-CAM papers. Do not fabricate DOI values.
- Hardware/toolchain: add Xilinx/Vitis AI/DPU, PYNQ, Ultra96-V2, and PYNQ-Z2 official documentation or citable white papers/datasheets; IEEE IoT Journal reviewers will expect hardware claims to be traceable. VERIFY RELEVANCE.
- Edge/IoT medical-AI comparisons: add recent Q1 journal papers, especially IEEE Internet of Things Journal, IEEE Journal of Biomedical and Health Informatics, IEEE Transactions on Instrumentation and Measurement, and IEEE Access papers on edge respiratory-sound analysis, FPGA/DPU medical inference, or low-power IoT CDSS. VERIFY RELEVANCE.
- Random Forest/cascaded early exit: add foundational Random Forest citation and recent early-exit/cascaded inference references, preferably in edge-AI/IoT contexts. VERIFY RELEVANCE.

## Style Compliance Notes
- The manuscript uses `\bibliographystyle{IEEEtran}` in `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/main.tex`:33 and `\usepackage{cite}` in `D:/PROJECTS/Parallel_Computing_on_FPGA/Paper/Parallel_Computing_on_FPGA/preamble.tex`:11, so the intended bibliography mechanism is IEEE numbered style.
- No dangling citation keys were found among active manuscript inputs.
- Major IEEE compliance deviations are bibliographic completeness, incorrect/duplicate dataset citation, uncited bibliography entries, missing DOI/page metadata, and author-year phrasing embedded in comparison-table labels.
- The bibliography includes many weakly relevant or suspicious sources for an IEEE Internet of Things Journal submission, including unrelated EEG, fish-stock, breast-cancer, tomato-seed, FPGA-PUF, speech-vocoder, and `@unknown` entries. These should be pruned unless explicitly needed and properly cited.
