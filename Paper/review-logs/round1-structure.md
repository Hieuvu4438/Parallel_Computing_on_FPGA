# Structure & Format Compliance Report
Round: 1
Target journal: IEEE Internet of Things Journal
Overall Score: 72/100

## CRITICAL Issues (Must fix before submission)
- **Severity:** CRITICAL
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\journal-guidelines\ (empty); D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:1
- **Issue:** Journal-specific IEEE Internet of Things Journal requirements could not be verified because the required `journal-guidelines/` input contains no guideline files. The manuscript uses `\documentclass[lettersize,journal]{IEEEtran}`, but journal-specific page limits, article type rules, graphical abstract requirements, author metadata requirements, open-access/license statements, and final-submission assets are not locally available.
- **Why it matters:** The review cannot certify compliance with journal-specific requirements without the target journal's author instructions. IEEEtran compliance is not identical to IEEE Internet of Things Journal submission compliance.
- **Suggested fix:** Add the official IEEE Internet of Things Journal author guidelines to `D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\journal-guidelines\` and rerun this audit. Until then, mark all journal-specific requirements as VERIFY WITH JOURNAL.

## MAJOR Issues (Strongly recommended to fix)
- **Severity:** MAJOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:7-13
- **Issue:** The author block does not include ORCID identifiers, and the corresponding-author information is only in a `\thanks{}` note. No author e-mail addresses are provided for non-corresponding authors.
- **Why it matters:** IEEE journal submissions commonly require or strongly encourage ORCID IDs for all authors and complete author metadata in the submission system and/or manuscript. Because the journal guidelines are unavailable, exact IEEE Internet of Things Journal requirements are VERIFY WITH JOURNAL.
- **Suggested fix:** Verify the journal's required author metadata. If required, add ORCID IDs and complete e-mail/affiliation metadata in the IEEE-compatible author block or submission metadata.

- **Severity:** MAJOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\02_related_work.tex:13-18 and 36
- **Issue:** Figure 1 is inserted in the Related Work section before it is first referenced in the text at line 36.
- **Why it matters:** IEEE style convention expects figures to be cited in the text before or near their appearance. A figure appearing before first callout can confuse readers and may be flagged during production review.
- **Suggested fix:** Move the first textual callout for Fig. 1 before the figure environment, or move the figure environment after line 36.

- **Severity:** MAJOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\04_experiments.tex:57; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_3.tex:1-47; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_kauh.tex:1-54
- **Issue:** Tables `tab:icbhi_demographics` and `tab:kauh_db` appear in Methodology through `\input{tables/table_kauh.tex}` at D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:14 and `\input{tables/table_3.tex}` is not directly included in the active section flow, while both are referenced much later in Discussion at line 57. This creates a reference/placement mismatch and, for `table_3.tex`, likely no visible table despite a later reference.
- **Why it matters:** Readers should encounter tables near their first substantive discussion, and all referenced tables must be included in the manuscript. Referencing an omitted table will compile as an unresolved or absent float if not included elsewhere.
- **Suggested fix:** Ensure `table_3.tex` is actually included in the active manuscript, and place or reference both demographic tables where they are first discussed. If the tables belong in Methodology, add callouts there before the inputs.

- **Severity:** MAJOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_1.tex:1-22; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_2.tex:1-20; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_3.tex:1-47; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_kauh.tex:1-54
- **Issue:** Several large `table*` floats are very dense and use `\resizebox{\textwidth}{!}`, `\scriptsize`, or fixed-width columns with extensive content.
- **Why it matters:** IEEE two-column journal layouts can produce readability problems, over-compressed text, delayed float placement, and orphaned captions when wide tables are too large. Excessively scaled tables may violate minimum readable font expectations.
- **Suggested fix:** Split oversized tables, shorten text, move detailed dataset metadata to supplementary material if allowed, or redesign as concise IEEE-style tables without aggressive scaling. VERIFY WITH JOURNAL whether supplementary tables are acceptable.

- **Severity:** MAJOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:31-90
- **Issue:** The manuscript uses multiple displayed equations but does not label or reference any equation numbers.
- **Why it matters:** IEEE style supports numbered equations, but equations that are numbered should usually be referenced if they are important enough to number. Unreferenced numbered equations create clutter and reduce navigability.
- **Suggested fix:** Add `\label{}` and textual references for central equations such as the early-exit decision and expected cost, or use unnumbered display environments for equations that are not referenced.

- **Severity:** MAJOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\references.bib:1-652
- **Issue:** The bibliography file contains 47 entries, but only 14 keys are cited in the active manuscript. Unused entries include `b04`, `b05`, `b06`, and `b17` through `b46`.
- **Why it matters:** While BibTeX will normally print only cited entries, a large unused bibliography file complicates reference auditing and increases the risk of stale or inconsistent references being unintentionally cited later.
- **Suggested fix:** Keep only cited entries in the submission bibliography or move unused references to an archive file outside the active submission package.

## MINOR Issues (Nice to fix)
- **Severity:** MINOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:17-19
- **Issue:** The abstract is approximately 219 words and reads as one long paragraph.
- **Why it matters:** The length is likely acceptable for many IEEE journals, but abstract word limits are journal-specific and cannot be confirmed because guidelines are missing. A long single paragraph can also obscure contribution, method, and results.
- **Suggested fix:** VERIFY WITH JOURNAL the abstract word limit. Consider tightening the abstract while preserving objective, method, key result, and conclusion.

- **Severity:** MINOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:21-23
- **Issue:** Keywords are present but use inconsistent capitalization and one broad term, `Parallel Computing`.
- **Why it matters:** IEEE keyword lists should align with indexing terms and be consistently styled for discoverability.
- **Suggested fix:** Verify IEEE taxonomy terms and normalize capitalization, e.g., use consistent lower-case phrase style unless an acronym/proper noun requires capitalization.

- **Severity:** MINOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:13
- **Issue:** The manuscript note says `Manuscript received 2025.` without month/day or revision/acceptance fields.
- **Why it matters:** IEEE publication metadata is often handled by production and may be omitted or completed according to journal workflow. An incomplete received date may look provisional in a submission PDF.
- **Suggested fix:** VERIFY WITH JOURNAL whether to omit this note for initial submission or provide the required metadata format.

- **Severity:** MINOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\05_conclusion.tex:7-14
- **Issue:** `Data Availability` and `Acknowledgment` are placed after the Conclusion, with Acknowledgment after Data Availability.
- **Why it matters:** IEEE journal ordering for data availability, acknowledgments, conflicts/funding, and references varies by journal and submission system. With no local guidelines, this order is not verifiable.
- **Suggested fix:** VERIFY WITH JOURNAL the required order of data availability, funding/acknowledgment, conflict of interest, references, and biographies.

- **Severity:** MINOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\02_related_work.tex:36; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:97-109
- **Issue:** Figure references use `Fig.~\ref{fig1}` for Fig. 1, but Figures 2 and 3 are not explicitly called out in nearby text after their environments.
- **Why it matters:** Every figure should be discussed in the text, not merely inserted. This is especially important for IEEE reviewers evaluating whether figures support claims.
- **Suggested fix:** Add concise callouts before the Figure 2 and Figure 3 environments or in the surrounding methodology narrative.

- **Severity:** MINOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:99; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:106
- **Issue:** Figure paths are inconsistent: one uses a bare filename with extra braces, while another uses `figures/` despite `\graphicspath{{figures/}}` in D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\preamble.tex:24.
- **Why it matters:** Inconsistent figure paths can cause compilation fragility when build directories or graphics search paths change.
- **Suggested fix:** Standardize figure inclusion paths and avoid filenames with spaces where possible.

- **Severity:** MINOR
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\preamble.tex:12 and D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:97-109
- **Issue:** `placeins` is loaded but no `\FloatBarrier` commands are used, while several wide floats may move far from their references.
- **Why it matters:** Wide floats in IEEE two-column format can drift to later pages, causing section-level placement confusion.
- **Suggested fix:** Either remove the unused package or use barriers judiciously after major sections, subject to IEEE float-placement constraints.

## SUGGESTIONS
- **Severity:** SUGGESTION
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:5
- **Issue:** The title is 12 words and reasonably concise, but it does not explicitly mention uncertainty-aware routing, cascaded inference, or the Ultra96-V2/Xilinx DPU deployment that distinguishes the paper.
- **Why it matters:** A title should clearly identify the central technical contribution and target platform for indexing and reviewer expectations.
- **Suggested fix:** Consider revising the title to include the uncertainty-aware cascaded inference contribution or heterogeneous FPGA/DPU deployment, while keeping it concise.

- **Severity:** SUGGESTION
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_7.tex:1-20; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_8.tex:1-20; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_9.tex:1-20; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_qat.tex:1-20
- **Issue:** These active table files define labeled tables but are not included in `main.tex` or the included section files.
- **Why it matters:** Unused active-source tables can confuse collaborators and reviewers auditing the source package.
- **Suggested fix:** If these tables are superseded, move them to `archive/`; if they are intended for the manuscript, add them at the appropriate first-reference locations.

- **Severity:** SUGGESTION
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\figures\*.png and D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\figures\*.jpg
- **Issue:** All manuscript figures appear to be raster images (`.png`/`.jpg`), including diagrams and flowcharts that may be better as vector graphics.
- **Why it matters:** IEEE production prefers high-resolution figures; diagrams and plots are usually sharper and more scalable as PDF/EPS/SVG-derived assets.
- **Suggested fix:** VERIFY WITH JOURNAL required resolution and accepted formats. Convert diagrams/plots to vector PDF/EPS where possible, and ensure raster images meet minimum DPI.

- **Severity:** SUGGESTION
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\figures\FIG 1 (123).png; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\figures\FIG 3 (123).png; D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\figures\Architectural Configuration of the Septuple Forest Ensembles Within Each Processing Tier.png
- **Issue:** Figure filenames contain spaces, parentheses, and very long names.
- **Why it matters:** Such filenames can be fragile across LaTeX engines, shell scripts, arXiv-like pipelines, and publisher production systems.
- **Suggested fix:** Rename figures to short ASCII names such as `fig_system_overview.png`, `fig_uahci_dataflow.png`, and `fig_rf_ensemble.png`, then update `\includegraphics` calls.

- **Severity:** SUGGESTION
- **Location:** D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:31-90
- **Issue:** Variables are mostly defined near equations, but not all notation is collected or cross-referenced.
- **Why it matters:** For mathematically dense methods, a compact notation table or consistently referenced variable definitions can improve readability.
- **Suggested fix:** If space permits, add a short notation table or ensure each symbol is defined once before use and reused consistently.

## Compliant Items
- The manuscript uses the IEEEtran journal class with letter size: `\documentclass[lettersize,journal]{IEEEtran}` at D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:1.
- The manuscript has a title, author block, affiliation block, corresponding author note, and manuscript note at D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:5-13.
- Abstract and IEEE keywords are present at D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:17-23.
- Main section order is conventional for an IEEE journal article: Introduction, Related Work, Methodology, Experiments and Results, Discussion, Conclusion, Data Availability, Acknowledgment, References, and biographies. Evidence: D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:25-47 and section labels in `sections/`.
- The Introduction includes motivation, research gap, contributions, and paper organization at D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\01_introduction.tex:3-22.
- Captions are correctly placed above tables, as shown in D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_1.tex:1-4 and D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\tables\table_perf_cpu_dpu.tex:1-4.
- Captions are correctly placed below figures, as shown in D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\02_related_work.tex:13-18 and D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\sections\03_methodology.tex:97-109.
- Citation keys used in the active manuscript are all present in `references.bib`; no missing BibTeX keys were found for cited references.
- Cross-reference labels used by `\ref{}` are defined; no undefined LaTeX references were found in the active source scan.
- Bibliography style is IEEE-compatible: `\bibliographystyle{IEEEtran}` and `\bibliography{references}` at D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:33-34.
- Author biographies are included using `IEEEbiography` environments at D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\Parallel_Computing_on_FPGA\main.tex:37-47.

## Guideline Gaps / Manual Verification Needed
- VERIFY WITH JOURNAL: `D:\PROJECTS\Parallel_Computing_on_FPGA\Paper\journal-guidelines\` is empty, so IEEE Internet of Things Journal-specific rules could not be confirmed.
- VERIFY WITH JOURNAL: Abstract word limit, keyword count/controlled vocabulary, and whether structured abstracts are prohibited or required.
- VERIFY WITH JOURNAL: ORCID requirement, corresponding-author format, author e-mail requirements, membership status requirements, and author photo/biography requirements for initial submission.
- VERIFY WITH JOURNAL: Page limits, overlength charges, manuscript category, and whether biographies should be included at initial submission.
- VERIFY WITH JOURNAL: Figure resolution, color/grayscale requirements, accepted graphics formats, and whether vector versions are required for diagrams.
- VERIFY WITH JOURNAL: Required sections and ordering for Data Availability, funding, conflicts of interest, ethics/IRB, consent, code availability, and acknowledgments.
- VERIFY WITH JOURNAL: Whether page numbers, headers, footers, copyright notice, and IEEE publication ID must be omitted or included at initial submission.
- MANUAL CHECK NEEDED: Compile the PDF and visually inspect margins, fonts, float positions, grayscale readability, line breaks, table legibility, and page count; these cannot be fully verified from source alone.
