# Prompt: Make KD Benchmark Comparable with ADD-RSC

Use this prompt to modify the KD EfficientNet/CNN6 benchmark code so its evaluation protocol is comparable with the ADD-RSC repository benchmark.

```text
Modify the KD EfficientNet/CNN6 benchmark code so its evaluation protocol is comparable with the ADD-RSC repository benchmark.

Important constraint:
Do NOT change the KD method itself:
- Do not change EfficientNet teacher architecture.
- Do not change CNN6 student architecture.
- Do not change KD loss, focal loss, optimizer, training schedule, augmentation, or preprocessing.
- Do not copy ADD-RSC model components.
- Only modify the dataset split protocol, benchmark/evaluation logic, metric calculation, and result reporting.

Goal:
Make the KD method report results under the same benchmark protocol used by ADD-RSC, so the scores can be compared fairly with ADD-RSC numbers.

Reference ADD-RSC behavior:
ADD-RSC uses the ICBHI 2017 respiratory cycle classification task with these classes:
- 0: Normal
- 1: Crackle
- 2: Wheeze
- 3: Both

The ADD-RSC repository reports:
- Specificity, `Sp`
- Sensitivity, `Se`
- ICBHI Score = `(Sp + Se) / 2`

ADD-RSC's local dataset split logic appears to use a shuffled 60/40 file-level split:
- Collect all unique recording basenames from `.wav` and `.txt`.
- Sort filenames.
- Shuffle indices using `random.Random(1).shuffle(indices)`.
- Use first 60% as train files.
- Use remaining 40% as test files.
- Split is based on recording/file basename, not respiratory cycle and not necessarily patient-wise.
- Then extract respiratory cycles from files belonging to each split.

Implement this same ADD-RSC-compatible split as an explicit benchmark mode in the KD code.

Required changes:

1. Add benchmark protocol argument
   Add an argument such as:
   - `--benchmark_protocol`, choices: `official_icbhi`, `add_rsc`, default `add_rsc`

   Behavior:
   - `official_icbhi`: keep the current KD official patient-wise split.
   - `add_rsc`: use the ADD-RSC-compatible 60/40 shuffled file-level split.

2. Implement ADD-RSC split
   For `--benchmark_protocol add_rsc`:
   - Read all `.wav` and `.txt` basenames from `args.data_dir`.
   - Sort unique basenames.
   - Create indices `[0, 1, ..., N-1]`.
   - Shuffle indices using `random.Random(1).shuffle(indices)`.
   - Use `int(len(indices) * 0.6)` as train size.
   - First 60% filenames are train.
   - Last 40% filenames are test.
   - Build respiratory cycle records only from those files.
   - Do not use patient ID official split in this mode.

3. Validation split
   ADD-RSC primarily evaluates on the held-out test split during training.
   For KD, keep training clean by making validation optional:
   - Add `--add_rsc_use_test_for_selection` default `False`.
   - If `False`, split the ADD-RSC train files into train/val using a deterministic file-level split.
   - If `True`, mimic ADD-RSC more closely by evaluating on the ADD-RSC test split during training.
   - Clearly record which mode was used in the output JSON.

   Preferred fair reporting:
   - Train/val from ADD-RSC train split.
   - Final benchmark only on ADD-RSC test split.
   - But include the option to mimic ADD-RSC exactly if needed.

4. Keep KD preprocessing unchanged
   Do NOT change:
   - sample rate
   - duration
   - STFT/log-mel computation
   - feature normalization
   - augmentation
   - bandpass behavior
   - teacher/student training

   This change is only to make the benchmark split and metrics comparable with ADD-RSC.

5. Match ADD-RSC metric definitions
   Implement/report metrics using the same ICBHI definitions.

   For 4-class labels:
   - 0 = Normal
   - 1 = Crackle
   - 2 = Wheeze
   - 3 = Both

   Specificity, `Sp`:
   - Normal-class accuracy.
   - Formula:
     `Sp = mean(y_pred[y_true == 0] == 0)`

   Sensitivity, `Se`:
   - Abnormal detection accuracy.
   - Treat classes 1, 2, and 3 as abnormal.
   - Formula:
     `Se = mean(y_pred[y_true != 0] != 0)`

   ICBHI Score:
   - `Score = (Sp + Se) / 2`

   Report these in percentage format to match ADD-RSC tables:
   - `Sp (%)`
   - `Se (%)`
   - `Score (%)`

6. Additional metrics
   Also keep/report the KD supporting metrics:
   - accuracy
   - macro F1
   - weighted F1
   - balanced accuracy
   - confusion matrix
   - per-class precision
   - per-class recall
   - per-class F1
   - per-class specificity
   - per-class support

   But the main comparison with ADD-RSC should use:
   - `Sp`
   - `Se`
   - `ICBHI Score`

7. Save benchmark artifacts
   Save outputs under the KD output directory:
   - `splits_add_rsc.json`
   - `metrics/test_add_rsc.json`
   - `metrics/confusion_matrix_test_add_rsc.csv`
   - `metrics/summary_add_rsc.json`

   The split JSON must include:
   - benchmark protocol name
   - train filenames
   - val filenames, if used
   - test filenames
   - number of cycles per split
   - class counts per split
   - random seed used for split
   - whether test split was used for checkpoint selection

8. Console output
   At final evaluation, print:

   ```text
   ADD-RSC-compatible benchmark:
   Sp: xx.xx
   Se: xx.xx
   Score: xx.xx
   Accuracy: xx.xx
   Macro F1: xx.xx
   Balanced Accuracy: xx.xx
   ```

9. Sanity checks
   Add checks that:
   - train and test filenames do not overlap.
   - validation filenames, if used, do not overlap train or test.
   - test split is non-empty.
   - all labels are in `[0, 1, 2, 3]`.
   - class counts are printed for train/val/test.

10. Verification
   After implementation:
   - Run a small debug/max-files run if available.
   - Confirm the split counts match the ADD-RSC split logic.
   - Confirm metric calculation on a small synthetic example.
   - Confirm no KD model, loss, optimizer, or preprocessing logic was changed.

Expected result:
The KD method can now be evaluated in an `add_rsc` benchmark mode, producing Sp/Se/Score numbers that are directly comparable with ADD-RSC repository-style benchmark results.
```

## Short version

```text
Modify the KD benchmark code to add an `--benchmark_protocol add_rsc` mode. In this mode, do not change the KD model, loss, optimizer, preprocessing, or augmentation. Only change the split/evaluation protocol to match ADD-RSC.

Use ADD-RSC's 60/40 shuffled file-level split: collect sorted unique recording basenames, shuffle indices with `random.Random(1)`, first 60% train, last 40% test, then extract respiratory cycles. Report ADD-RSC-compatible metrics: Sp = normal-class accuracy, Se = abnormal detection accuracy where classes 1/2/3 are abnormal, Score = `(Sp + Se) / 2`. Save split JSON, test metrics JSON, confusion matrix CSV, and summary JSON. Keep additional metrics like accuracy, macro F1, weighted F1, balanced accuracy, per-class precision/recall/F1/specificity/support. Ensure train/test filenames do not overlap. Do not modify KD preprocessing or model behavior.
```
