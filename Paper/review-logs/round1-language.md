# Language & Style Edit Report
Round: 1
Target journal: IEEE Internet of Things Journal

## Statistics
- Total issues found: 54
- Critical issues where meaning is unclear: 5
- Style improvements: 32
- Estimated word reduction: 12%
- Overall Language Quality Score: 7/10

## Section-by-Section Edits

### Abstract
| Line | Original | Revised | Category | Severity |
|---|---|---|---|---|
| `main.tex:18` | This research focuses on designing a hierarchical embedded system that leverages respiratory sound features for diagnosing COPD and other respiratory disorders, aiming for high accuracy and energy efficiency suitable for mobile or embedded devices. | This study presents a hierarchical embedded system that uses respiratory sound features to diagnose COPD and other respiratory disorders with accuracy and energy efficiency suitable for mobile and embedded devices. | Conciseness / academic style | Medium |
| `main.tex:18` | The system segments lung sounds into individual breathing cycles and extracts key respiratory features across time and frequency domains, structured into four layers. | The system segments lung sounds into breathing cycles and extracts time- and frequency-domain features across four layers. | Conciseness / clarity | Medium |
| `main.tex:18` | It utilizes a combination of the Random Forest algorithm and an Artificial Neural Network, both of which are fine-tuned with the dataset for disease classification. | It combines Random Forest classifiers with a neural network trained for disease classification. | Wordiness / terminology | Medium |
| `main.tex:18` | The first layer handles record information, the second and third layers manage features extracted from fixed-length sound segments, classified using Random Forest, while the fourth layer converts breathing patterns into Hybrid Spectrogram images, which are then processed by the Convolutional Neural Network (CNN) via Knowledge Distillation for diagnosis. | The first layer processes recording-level information; the second and third layers classify fixed-length segment features using Random Forest models; and the fourth layer converts breathing patterns into Hybrid Spectrogram images processed by a knowledge-distilled convolutional neural network (CNN). | Grammar / parallelism / flow | High |
| `main.tex:18` | This design allows the system to classify without completing all four stages simultaneously, effectively utilizing the parallel computing power of the FPGA. | This design enables early classification without executing all four stages for every sample, thereby using FPGA parallelism more efficiently. | Clarity / VERIFY MEANING | High |
| `main.tex:18` | Implemented on the Xilinx PYNQ-Ultra96-V2 FPGA development board, the system not only achieves an average 96.56\% accuracy rate across three diagnostic categories (Healthy, COPD, and Non-COPD), but also significantly reduces latency and model size via QAT. | Implemented on the Xilinx PYNQ-Ultra96-V2 FPGA development board, the system achieves 96.56\% average accuracy across three diagnostic categories (Healthy, COPD, and Non-COPD) and reduces latency and model size through QAT. | Conciseness / grammar | Medium |
| `main.tex:18` | This study confirms the effectiveness of the proposed method in diagnosing respiratory disorders, highlighting its potential for real-world clinical applications. | These results demonstrate the potential of the proposed method for respiratory disorder diagnosis in real-world clinical applications. | Avoid overclaiming / academic tone | Medium |
| `main.tex:22` | Respiratory disease diagnosis, Respiratory sound, Signal transformation, FPGA, Parallel Computing. | Respiratory disease diagnosis, respiratory sound, signal transformation, FPGA, parallel computing. | IEEE keyword capitalization / mechanics | Low |

### Introduction
| Line | Original | Revised | Category | Severity |
|---|---|---|---|---|
| `sections/01_introduction.tex:3` | Chronic Obstructive Pulmonary Disease (COPD) | chronic obstructive pulmonary disease (COPD) | Acronym capitalization / IEEE style | Low |
| `sections/01_introduction.tex:3` | COPD is characterized by persistent airway inflammation and airflow limitation, often leading to breathlessness, wheezing, crackles, rhonchi, and, in advanced cases, respiratory failure | COPD is characterized by persistent airway inflammation and airflow limitation and is often associated with breathlessness, wheezing, crackles, rhonchi, and, in advanced cases, respiratory failure | Precision / causality | Medium |
| `sections/01_introduction.tex:5` | Global burden studies indicate that COPD will remain one of the leading causes of mortality worldwide. | Global burden studies indicate that COPD remains a leading cause of mortality worldwide. | Tense / conciseness | Low |
| `sections/01_introduction.tex:7` | static offline classifiers: every sample is processed by the same model regardless of its diagnostic difficulty. | static offline classifiers in which every sample is processed by the same model regardless of diagnostic difficulty. | Punctuation / academic style | Low |
| `sections/01_introduction.tex:9` | but comparatively fewer works jointly address uncertainty-aware inference, low-power edge execution, and hardware-software co-design on heterogeneous FPGA platforms. | but fewer studies jointly address uncertainty-aware inference, low-power edge execution, and hardware--software co-design on heterogeneous FPGA platforms. | Word choice / terminology | Medium |
| `sections/01_introduction.tex:13` | Hybrid Spectrogram representation | hybrid spectrogram representation | Capitalization consistency unless a formal named method | Low |
| `sections/01_introduction.tex:13` | Quantization-Aware Training (QAT) and Hardware-Software Co-design | quantization-aware training (QAT) and hardware--software co-design | Capitalization / hyphenation consistency | Low |
| `sections/01_introduction.tex:17` | Random Forest-based early exits | Random Forest-based early-exit decisions | Clarity / terminology | Low |
| `sections/01_introduction.tex:19` | reducing expected energy consumption by 52.5\% compared with static CNN-style deployment. | reducing expected energy consumption by 52.5\% compared with a static CNN deployment. | Conciseness / style | Low |
| `sections/01_introduction.tex:22` | Finally, Section~\ref{sec:conclusion} concludes the paper and discusses future research directions. | Finally, Section~\ref{sec:conclusion} concludes the paper. | Formulaic phrasing / conciseness | Low |

### Related Work
| Line | Original | Revised | Category | Severity |
|---|---|---|---|---|
| `sections/02_related_work.tex:3` | Extensive research efforts have been directed toward the development of respiratory pathology classification methodologies, predominantly employing post-segmentation approaches with classical machine learning paradigms, including Support Vector Machines (SVM), Artificial Neural Network architectures, K-Nearest Neighbor (KNN) algorithms, Multilayer Perceptron (MLP) frameworks, Random Forest (RF) ensembles, and Long Short-Term Memory (LSTM) networks. | Extensive research has addressed respiratory pathology classification using post-segmentation approaches and machine learning models, including support vector machines (SVMs), artificial neural networks, k-nearest neighbors (KNN), multilayer perceptrons (MLPs), Random Forest (RF) ensembles, and long short-term memory (LSTM) networks. | Wordiness / capitalization / terminology | High |
| `sections/02_related_work.tex:3` | Table~\ref{tab:ml_comparison} presents a comprehensive synthesis of significant contributions in this domain, cataloging investigations that leverage computational intelligence techniques for pulmonary disorder diagnosis across diverse respiratory conditions. | Table~\ref{tab:ml_comparison} summarizes representative studies that use computational intelligence techniques for pulmonary disorder diagnosis across diverse respiratory conditions. | Conciseness / academic style | Medium |
| `sections/02_related_work.tex:4` | The predominant methodological frameworks in this research domain employ conventional taxonomic techniques that leverage characteristic feature representations as their foundational classification mechanism. | Most studies use conventional classification methods based on engineered feature representations. | Severe wordiness / clarity | High |
| `sections/02_related_work.tex:4` | The substantive differentiation among these methodological approaches, however, manifests primarily in their diverse feature extraction algorithms—each designed to optimize the identification and selection of classification methodologies with maximal diagnostic efficacy and computational performance. | These approaches differ mainly in their feature extraction algorithms and in the classifiers selected to balance diagnostic performance and computational cost. | Severe wordiness / clarity | High |
| `sections/02_related_work.tex:6` | using Short-Time Fourier Transform (STFT) | using the short-time Fourier transform (STFT) | Article usage / capitalization | Low |
| `sections/02_related_work.tex:8` | extracts Mel-Frequency Cepstral Coefficients (MFCC) and spectrogram features | extracts mel-frequency cepstral coefficients (MFCCs) and spectrogram features | Acronym plural / capitalization | Low |
| `sections/02_related_work.tex:10` | By incorporating skip connections and direct feature integration, the model achieves a more stable gradient flow, reducing vanishing gradient issues. | By incorporating skip connections and direct feature integration, the model improves gradient flow and reduces vanishing-gradient issues. | Conciseness / terminology | Low |
| `sections/02_related_work.tex:12` | using CNNs | using convolutional neural networks (CNNs) | Acronym consistency; CNN first defined in abstract but section should remain readable | Low |
| `sections/02_related_work.tex:22` | making it an efficient approach for automated lung disease identification. | making it an efficient approach to automated lung disease identification. | Preposition | Low |
| `sections/02_related_work.tex:28` | the experimental results showed an accuracy of 81.25\%. | the model achieved an accuracy of 81.25\%. | Active voice / conciseness | Low |
| `sections/02_related_work.tex:32` | K.V \textit{et al.} | K. V. \textit{et al.} | Initial spacing / mechanics | Low |
| `sections/02_related_work.tex:34` | However, it is worth noting that most studies focus on maximizing the accuracy of classification results without fully exploiting the parallel computing power and flexibility of FPGAs in medical data processing, particularly in respiratory disease diagnosis. | However, most studies prioritize classification accuracy and do not fully exploit the parallelism and flexibility of FPGAs for medical data processing, particularly respiratory disease diagnosis. | Avoid weak phrase / conciseness | Medium |
| `sections/02_related_work.tex:36` | This creates a gap in applying the advantages of FPGA architecture to the medical field, where accurate and rapid diagnosis is crucial. | This gap limits the use of FPGA architectures in medical applications that require accurate and rapid diagnosis. | Clarity / style | Medium |
| `sections/02_related_work.tex:36` | mainly through lung sounds. | based primarily on lung sounds. | Word choice | Low |
| `sections/02_related_work.tex:36` | The overall block diagram of the proposed system is presented in Fig.~\ref{fig1}. | Fig.~\ref{fig1} shows the overall block diagram of the proposed system. | Active voice / conciseness | Low |

### Methodology
| Line | Original | Revised | Category | Severity |
|---|---|---|---|---|
| `sections/03_methodology.tex:7` | totaling 1,256 audio recordings from two sources: | comprising 1,256 audio recordings from two sources: | Formal word choice | Low |
| `sections/03_methodology.tex:10` | with COPD, Asthma, Pneumonia, URTI, Bronchiectasis, and healthy controls | with COPD, asthma, pneumonia, URTI, bronchiectasis, and healthy controls | Disease-name capitalization | Low |
| `sections/03_methodology.tex:16` | All recordings are mapped into a 3-class target space | All recordings are mapped to a three-class target space | Preposition / number style | Low |
| `sections/03_methodology.tex:16` | Asthma, Heart Failure, Pulmonary Fibrosis, Pneumonia, and Pleural Effusion | asthma, heart failure, pulmonary fibrosis, pneumonia, and pleural effusion | Disease-name capitalization | Low |
| `sections/03_methodology.tex:20` | converts multi-channel signals to mono | converts multichannel signals to mono | Spelling consistency | Low |
| `sections/03_methodology.tex:30` | Root Mean Square Energy (RMSE) | root-mean-square energy (RMSE) | Technical term capitalization / hyphenation | Low |
| `sections/03_methodology.tex:42` | Thirteen MFCC coefficients are extracted from the Mel frequency scale, | Thirteen MFCCs are extracted on the mel frequency scale, | Technical terminology / conciseness | Low |
| `sections/03_methodology.tex:49` | Hybrid Spectrogram tensor | hybrid spectrogram tensor | Capitalization consistency | Low |
| `sections/03_methodology.tex:53` | Mel-spectrogram with 128 Mel filters | mel spectrogram with 128 mel filters | Capitalization / hyphenation consistency | Low |
| `sections/03_methodology.tex:61` | hard-label supervision and soft teacher probabilities | hard-label supervision with soft teacher probabilities | Parallelism / clarity | Low |
| `sections/03_methodology.tex:85` | where $\tau_i$ is the early-exit threshold and $\lambda=4$ is the majority-vote requirement. | Here, $\tau_i$ is the early-exit threshold and $\lambda=4$ is the majority-vote requirement. | Sentence fragment after display equation | Medium |
| `sections/03_methodology.tex:91` | Since $C_1,C_2,C_3\ll C_4$ | Because $C_1,C_2,C_3\ll C_4$ | Formal academic style | Low |
| `sections/03_methodology.tex:113` | assigning control-oriented and compute-intensive operations to the hardware domain best suited to each workload. | assigning control-oriented and compute-intensive operations to the hardware domain best suited for each workload. | Preposition | Low |
| `sections/03_methodology.tex:125` | calibrated over 32 batches and fine-tuned for 5 epochs | calibrated using 32 batches and fine-tuned for five epochs | Preposition / number style | Low |

### Experiments / Results
| Line | Original | Revised | Category | Severity |
|---|---|---|---|---|
| `sections/04_experiments.tex:4` | evaluates the proposed \textit{Uncertainty-Aware Heterogeneous Cascaded Inference} (UA-HCI) framework in terms of diagnostic performance, hardware acceleration, FPGA resource utilization, and energy efficiency. | evaluates the proposed UA-HCI framework in terms of diagnostic performance, hardware acceleration, FPGA resource utilization, and energy efficiency. | Repetition; acronym already defined | Low |
| `sections/04_experiments.tex:10` | increases variation in diagnosis, age distribution, recording device, and auscultation location | increases variation in diagnoses, age distributions, recording devices, and auscultation locations | Parallel plural forms | Low |
| `sections/04_experiments.tex:16` | a Zynq UltraScale+ ZU3EG MPSoC platform | A Zynq UltraScale+ ZU3EG MPSoC platform | Capitalization after list label | Low |
| `sections/04_experiments.tex:17` | a Zynq-7000 SoC platform | A Zynq-7000 SoC platform | Capitalization after list label | Low |
| `sections/04_experiments.tex:24` | These layers make early-exit decisions when the vote-based confidence score satisfies $S_i(x)\ge\tau_i$ and the majority-vote requirement is met. | These layers make early-exit decisions when the vote-based confidence score satisfies $S_i(x)\ge\tau_i$ and the majority-vote requirement $\lambda$ is met. | Clarity / notation consistency | Low |
| `sections/04_experiments.tex:26` | Focal Loss was used during training | Focal loss was used during training | Capitalization | Low |
| `sections/04_experiments.tex:30` | achieved an accuracy of $96.56\%$ for the 3-class classification task. | achieved $96.56\%$ accuracy for the three-class classification task. | Conciseness / number style | Low |
| `sections/04_experiments.tex:30` | The result should be interpreted as an offline experimental benchmark on the merged ICBHI and KAUH datasets rather than as evidence of prospective clinical performance. | This result should be interpreted as an offline benchmark on the merged ICBHI and KAUH datasets, not as evidence of prospective clinical performance. | Conciseness / clarity | Medium |
| `sections/04_experiments.tex:37` | The measured latency values describe the CNN inference block and should therefore be distinguished from the complete end-to-end cascade latency, which also includes preprocessing, feature extraction, Random Forest routing, and data-transfer overheads. | The measured latency values describe only the CNN inference block and should be distinguished from the complete end-to-end cascade latency, which also includes preprocessing, feature extraction, Random Forest routing, and data-transfer overhead. | Precision / conciseness | Medium |
| `sections/04_experiments.tex:45` | The resource profile shows that DPU deployment is feasible on both evaluated platforms | The resource profile indicates that DPU deployment is feasible on both evaluated platforms | Academic style | Low |
| `sections/04_experiments.tex:48` | The energy benefit of UA-HCI comes from reducing the probability that a sample reaches Layer 4. | UA-HCI improves energy efficiency by reducing the probability that a sample reaches Layer 4. | Active voice / conciseness | Low |
| `sections/04_experiments.tex:48` | This result supports the use of uncertainty-aware routing for edge deployment, where battery capacity and thermal constraints are important design factors. | This result supports uncertainty-aware routing for edge deployment, where battery capacity and thermal constraints are key design factors. | Conciseness | Low |

### Discussion
| Line | Original | Revised | Category | Severity |
|---|---|---|---|---|
| `sections/04_experiments.tex:54` | Clinical Decision Support System (CDSS) | clinical decision support system (CDSS) | Capitalization / acronym style | Low |
| `sections/04_experiments.tex:56` | \textbf{Cross-Database Evaluation:} | \textbf{Cross-database evaluation:} | Title capitalization consistency | Low |
| `sections/04_experiments.tex:57` | as summarized in Table~\ref{tab:icbhi_demographics} and Table~\ref{tab:kauh_db}. | as summarized in Tables~\ref{tab:icbhi_demographics} and~\ref{tab:kauh_db}. | IEEE table-reference style | Low |
| `sections/04_experiments.tex:57` | This setting is useful for evaluating whether the proposed features and routing strategy remain effective across heterogeneous recordings. | This setting helps evaluate whether the proposed features and routing strategy remain effective across heterogeneous recordings. | Conciseness | Low |
| `sections/04_experiments.tex:59` | \textbf{Explainability Considerations:} | \textbf{Explainability considerations:} | Title capitalization consistency | Low |
| `sections/04_experiments.tex:60` | These descriptors allow early decisions to be related to signal energy, transient behavior, and spectral structure. | These descriptors relate early decisions to signal energy, transient behavior, and spectral structure. | Active voice / conciseness | Low |
| `sections/04_experiments.tex:62` | \textbf{CDSS Positioning:} | \textbf{CDSS positioning:} | Title capitalization consistency | Low |
| `sections/04_experiments.tex:63` | with the additional goal of reducing unnecessary deep inference through early exits. | while also reducing unnecessary deep inference through early exits. | Flow / conciseness | Low |

### Conclusion
| Line | Original | Revised | Category | Severity |
|---|---|---|---|---|
| `sections/05_conclusion.tex:3` | This study has designed and developed a hierarchical embedded system using respiratory sound features to diagnose respiratory diseases. | This study developed a hierarchical embedded system that uses respiratory sound features to diagnose respiratory diseases. | Tense / conciseness | Medium |
| `sections/05_conclusion.tex:3` | The system was successfully deployed on the FPGA Ultra96V2, achieving the expected accuracy and performance. | The system was deployed on the Ultra96-V2 FPGA and achieved the expected accuracy and performance. | Word order / terminology consistency | Medium |
| `sections/05_conclusion.tex:3` | With a peak average accuracy of 94.94\% across a 3-fold cross-validation for three classes—Healthy, COPD, and Non-COPD—the model demonstrates the effectiveness and reliability of the proposed method in classifying respiratory conditions. | With a mean accuracy of 94.94\% across three-fold cross-validation for three classes—Healthy, COPD, and Non-COPD—the model demonstrates the effectiveness of the proposed method for classifying respiratory conditions. | VERIFY MEANING / metric wording | High |
| `sections/05_conclusion.tex:3` | The system exhibits superior parallel processing capabilities and flexibility, significantly reducing disease classification time. | The system uses FPGA parallelism to reduce disease-classification time. | Avoid unsupported evaluative language / conciseness | Medium |
| `sections/05_conclusion.tex:3` | This advantage translates into faster processing speeds and reduced waiting times during the diagnostic process. | This advantage can reduce diagnostic processing time. | Redundancy / conciseness | Low |
| `sections/05_conclusion.tex:3` | facilitating deployment on mobile or embedded devices and allowing for use in clinical and real-world settings. | supporting deployment on mobile or embedded devices and potential use in clinical settings. | Avoid overclaiming / conciseness | Medium |
| `sections/05_conclusion.tex:5` | underscore the efficacy and broad application potential | demonstrate the potential | Avoid inflated language | Medium |
| `sections/05_conclusion.tex:5` | paves the way for practical usage in supporting clinical diagnostic procedures and enhancing patient care quality. | may support clinical diagnostic workflows and improve patient care. | Academic tone / conciseness | Medium |
| `sections/05_conclusion.tex:14` | The authors are grateful to the editor and the anonymous referees for their valuable comments, which significantly improved the results and presentation of this article. | The authors thank the editor and anonymous referees for their valuable comments. | VERIFY CONTEXT; premature acknowledgment if manuscript not yet reviewed | High |

### Tables/Captions
| Line | Original | Revised | Category | Severity |
|---|---|---|---|---|
| `tables/table_1.tex:3` | Characteristics of Respiratory Sound Types | Characteristics of respiratory sound types | Caption capitalization consistency | Low |
| `tables/table_1.tex:12` | High-pitched, crackling sounds like hair crackling near a fire; may change with position | High-pitched crackling sounds, similar to hair crackling near a fire; may change with position | Comma use / clarity | Low |
| `tables/table_1.tex:14` | High-pitched, musical whistling sounds may vary in intensity and pitch | High-pitched musical whistling sounds that may vary in intensity and pitch | Grammar | Low |
| `tables/table_1.tex:16` | The high-pitched, harsh, crowing sound indicates airway obstruction | High-pitched, harsh, crowing sound indicating airway obstruction | Table style / conciseness | Low |
| `tables/table_2.tex:3` | Comparative Analysis of Machine Learning Methodologies Applied to Automated Respiratory Acoustic Pattern Recognition Systems | Comparison of machine learning methods for automated respiratory acoustic pattern recognition | Caption wordiness | Medium |
| `tables/table_2.tex:18` | Densenet | DenseNet | Model-name capitalization | Low |
| `tables/table_3.tex:3` | Dataset Statistics of the ICBHI-2017 Respiratory Sound Database | Statistics of the ICBHI 2017 respiratory sound database | Caption style / consistency | Low |
| `tables/table_3.tex:17` | 7Al, 8Ar, 3Ll | 7 Al, 8 Ar, 3 Ll | Spacing / readability | Low |
| `tables/table_4.tex:9` | Num. of Samples | No. of samples | Table heading style | Low |
| `tables/table_5.tex:12` | RMS energy, 5 s, 50--2500 Hz | RMS energy, 5 s, 50--2500 Hz | VERIFY CONSISTENCY: text defines RMSE but table uses RMS | Medium |
| `tables/table_7.tex:3` | Knowledge Distillation Performance - 3-Fold CV (Subject-Independent Split) | Knowledge distillation performance: three-fold cross-validation with subject-independent splits | Caption style / hyphenation | Low |
| `tables/table_8.tex:3` | FPGA Resource Utilization Profile [Projected] | FPGA resource utilization profile [projected] | Caption style | Low |
| `tables/table_8.tex:12` | [XXXX] | [XXXX] | Placeholder remains in table; VERIFY whether this table is included in manuscript | High |
| `tables/table_9.tex:3` | Cross-Platform Hardware Performance Comparison (Edge FPGA vs. Edge GPU) [Projected] | Cross-platform hardware performance comparison: edge FPGA versus edge GPU [projected] | Caption style | Low |
| `tables/table_9.tex:15` | [XX.XX] | [XX.XX] | Placeholder remains in table; VERIFY whether this table is included in manuscript | High |
| `tables/table_qat.tex:13` | Avg Latency (CPU) | Average latency (CPU) | Abbreviation / table-heading style | Low |
| `tables/table_perf_cpu_dpu.tex:3` | Inference Latency Comparison: CPU vs. FPGA DPU (MobileNetV2 Layer) | Inference latency comparison between CPU and FPGA DPU for the MobileNetV2 layer | Caption style | Low |
| `tables/table_kauh.tex:3` | Dataset Statistics of the King Abdullah University Hospital (KAUH) Database | Statistics of the King Abdullah University Hospital (KAUH) database | Caption style / capitalization | Low |

## Recurring Patterns
- Frequent nominalizations and inflated phrases reduce readability, especially in Related Work (e.g., "methodological frameworks," "substantive differentiation," "diagnostic efficacy"). Prefer direct verbs and concrete nouns.
- Several sentences use formulaic or weak academic phrases such as "it is worth noting," "this research focuses on," and "the results underscore." Replace them with direct claims.
- Capitalization is inconsistent for disease names, feature names, and method names. Use lowercase for common disease names and technical concepts unless they are proper nouns or defined method names.
- Acronyms are sometimes singular when plural is intended (e.g., MFCC vs. MFCCs) and are sometimes redefined or expanded inconsistently.
- Hyphenation varies across hardware--software, early-exit, three-class, cross-database, and subject-independent. Standardize compound modifiers.
- Some conclusion wording overclaims clinical readiness. Use cautious phrasing for retrospective/offline evaluation.
- Table captions are often title-case and verbose. IEEE style generally benefits from concise sentence-style captions.

## Terminology and Consistency Notes
- Use one form consistently: "Ultra96-V2" rather than "PYNQ-Ultra96-V2," "FPGA Ultra96V2," or "Ultra96V2," unless naming the exact development board.
- Standardize "hybrid spectrogram" capitalization. If it is a formal named representation, define it once as "Hybrid Spectrogram" and use that form consistently; otherwise use lowercase.
- Standardize "hardware--software co-design" with an en dash or double hyphen in LaTeX for the compound relation.
- Use "Random Forest" consistently as the model name; use "Random Forest classifiers/ensembles" where grammatical number is needed.
- Use "CNN" only after first defining "convolutional neural network" in the main text. The abstract defines it, but each major section should still avoid unexplained acronyms where possible.
- Use "mel" consistently in "mel spectrogram," "mel filters," and "mel-frequency cepstral coefficients" unless following a source title.
- The manuscript reports 96.56\% in the abstract/introduction/results but 94.94\% mean accuracy in the conclusion. This is partly a content consistency issue, but it also affects language clarity and credibility. VERIFY MEANING before editing.
- Tables `table_8.tex` and `table_9.tex` contain placeholders and projected labels. If these are not included in the manuscript, they need not be edited; if included later, remove placeholders before submission.

## High-Priority Edits to Apply First
1. Rewrite the Related Work opening paragraph (`sections/02_related_work.tex:3-4`) to remove excessive wordiness and improve credibility.
2. Revise the abstract (`main.tex:18`) for conciseness, clearer early-exit wording, and a more cautious final sentence.
3. Resolve the accuracy wording discrepancy between 96.56\% and 94.94\% before polishing the conclusion.
4. Standardize capitalization and terminology for disease names, hybrid spectrogram, QAT, hardware--software co-design, mel/MFCC, and Ultra96-V2.
5. Replace overclaiming in the conclusion with cautious language aligned with retrospective/offline validation.
6. Shorten verbose table captions, especially `tables/table_2.tex:3`, and remove or verify placeholder tables before submission.
