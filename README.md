# Disease_Detection_extended

**A pre-registered, one-time test of a frozen apple-leaf failure-prediction system on a second, independent field dataset (PlantDoc), plus exploratory analyses that explain the results.**

This repository extends the original study *"Source-Only Failure-Head Prediction for Reliable Apple Foliar Disease Classification under Unseen Field-Domain Shift"* (repository `Disease_Detection_`). The original study trained everything on lab images (PlantVillage) and tested once on field images (Plant Pathology 2020, "PP2020"). This extension asks whether its findings hold on new field data that played no part in any decision.

> **Nothing from the original study was retrained, recalibrated or re-thresholded.** The original project is only *read*, and its freeze record (31 hashed files) is verified before every step that uses it.

---

## Contents

1. [Background: the original study in brief](#1-background-the-original-study-in-brief)
2. [Why this extension](#2-why-this-extension)
3. [Protocol and safeguards](#3-protocol-and-safeguards)
4. [Experiment log, step by step](#4-experiment-log-step-by-step)
5. [Results: confirmatory (pre-registered)](#5-results-confirmatory-pre-registered)
6. [Results: exploratory](#6-results-exploratory)
7. [Observations and interpretation](#7-observations-and-interpretation)
8. [Limitations](#8-limitations)
9. [Repository layout](#9-repository-layout)
10. [How to reproduce](#10-how-to-reproduce)
11. [Figure data](#11-figure-data)
12. [Provenance, licences and hashes](#12-provenance-licences-and-hashes)

---

## 1. Background: the original study in brief

| Item | Original study |
|---|---|
| Source (training) | PlantVillage apple, 2,550 lab images (healthy / scab / cedar apple rust) |
| Target (one-time test) | PP2020, 1,730 field images, same three classes |
| Classifier | MobileNetV3-Small (928,739 parameters), 3 training seeds (42, 123, 2026) |
| Failure heads (trained on source images degraded by brightness, contrast, blur and noise) | **P1** proposed head (576 image features + 3 logits); **A1** features-only head; **B5** logits-only head |
| Other baselines | B1 maximum softmax probability (MSP), B2 entropy, B3 margin, B4 temperature-scaled MSP |
| Primary endpoint | ΔAURC = AURC(P1) − AURC(MSP); negative means better than MSP |
| Main PP2020 results | Accuracy 99.7% on lab images → 35–36% on field images (about 98% of images predicted "scab"). P1 did **not** beat MSP (+0.029 [−0.002, +0.058]). The logits-only head B5 **did** (−0.099 [−0.120, −0.078]), but B5 was a comparator, not the proposed method. |

**AURC** (area under the risk-coverage curve) measures how well a score puts wrong predictions last. Lower is better, and a useless score has AURC equal to the error rate.

## 2. Why this extension

The original paper had three weaknesses a reviewer would point to:
1. It used **one** source–target pair.
2. Its one positive finding (B5 better than MSP) came from a comparator, so it could be chance or specific to the dataset.
3. It showed *that* some methods failed or succeeded, but not *why*.

The extension follows an **explain → predict → confirm** design:
1. **Explain** the PP2020 result with exploratory analyses of saved outputs.
2. **Predict:** write the hypotheses down and seal them (hash, commit, push) before any new prediction exists.
3. **Confirm:** run the frozen system **once** on a new, independent field dataset and report whatever happens.

## 3. Protocol and safeguards

| Safeguard | How it is enforced |
|---|---|
| Frozen system unchanged | `verify_frozen()` from the original project runs before feature extraction, inference and evaluation, and checks the SHA-256 of all 31 frozen files. |
| Original project untouched | It is only read. Python bytecode caching is switched off so that nothing is written there. |
| Hypotheses fixed in advance | `prereg/plantdoc_preregistration.md` was committed and pushed (commit `9692e18`) before any PlantDoc logit existed. The inference and evaluation scripts refuse to run if its SHA-256 differs. |
| Blinded inference | Inference reads only a label-free manifest (path, hash, cluster). Files are anonymised (`pd_XXXX.jpg`, numbered in SHA-256 order) because PlantDoc folder names reveal the labels. |
| Predictions before labels | Prediction files were hashed, committed and pushed (`e5027c6`) before any label was joined. The evaluation script checks the hashes and that the files are pushed. |
| Run once | The inference and evaluation scripts refuse to run if outputs exist, and there is no overwrite option. |
| Analysis code before results | The evaluation script was committed (`c296fbd`) before the real run. It was tested only with random labels in a temporary folder. |
| Byte-exact hashes | `.gitattributes` disables line-ending conversion, so recorded hashes verify on any system (this fixes a caveat of the original study). |
| Exploratory is labelled | Anything decided after seeing results lives in `results/exploratory_*` and is marked as exploratory. |

## 4. Experiment log, step by step

The full decision log is in [`notes/study_log.md`](notes/study_log.md).

### Step 0: Dataset choice

| Candidate | Decision | Reason |
|---|---|---|
| **PlantDoc** (apple healthy / scab / rust) | **Chosen** | Independent web-collected field images; CC BY 4.0; direct GitHub download; all three classes present |
| Plant Pathology 2021 | Rejected for this test | Built on PP2020 (same orchards and team), so it would not be an independent replication |
| AppleLeaf9, Zenodo "Apple Leaf Diseases (13 classes)" | Rejected | Compiled from PlantVillage, PP2020/2021 and other sets, so they contain source or target images |
| ATLDSD (healthy / rust), AppleScabLDs (healthy / scab) | Deferred | Only two of the three classes; access and licence not yet verified |

### Step 1: Acquisition (`src/acquire_plantdoc.py`, commit `3a38956`)
- The three apple folders were taken from the original PlantDoc repository at commit `5467f6012d78d1c446145d5f582da6096f852ae8`.
- Some file names contain characters Windows forbids, so the bytes were read from the Git object store and **checked against their Git blob ids**. They are saved under safe names, and the original names are recorded.
- 273 files were taken: healthy 91, scab 93, rust 89 (train and test folders combined).

### Step 2: Audit (`src/audit_plantdoc.py`, commit `aaa7eab`)
- Every file was fully decoded: all 273 decode. Modes are RGB 271, grayscale 1 and CMYK 1, and all are converted to RGB at load time, as in the original study.
- Sizes are very mixed (shorter side 85–4,000 px). 3 files carry an EXIF rotation tag, which is not applied, as in the original study.
- SHA-256 and a 64-bit pHash were computed exactly as in the original audit and compared within PlantDoc and against all 4,371 audited PlantVillage and PP2020 images.
  - **No exact copies**, either within PlantDoc or against PlantVillage/PP2020.
  - **18 pHash pairs at 0–10 bits** went to visual review.

### Step 3: Duplicate review (commit `400ac2f`)
- Rule, identical to the original study: `duplicate`, `different` or `unsure`, with unsure counted as duplicate. The SIFT check was unavailable, so a pixel-difference hint was used.
- Pairs 1–3 are the same photo twice inside PlantDoc. Pairs 4–18 are different images, including all 12 PlantDoc–PlantVillage pairs.
- **No PlantDoc image overlaps PlantVillage or PP2020.**

### Step 4: Exploratory analysis of the saved PP2020 outputs (`src/explore_pp2020.py`, commit `243a935`)
This step generates hypotheses only; it is not evidence.

| Question | Finding |
|---|---|
| Is B5's PP2020 win just the shortcut "predicted scab = wrong"? | No. The shortcut alone gives AUROC-error 0.51–0.52. **Inside** the predicted-scab group, B5 reaches 0.75–0.79 against MSP's 0.66, and ΔAURC is −0.08 to −0.11 with every interval below 0. |
| What did B5 learn? | B5 is most sensitive to the healthy and scab logits. On PP2020 its ranking matches **logit magnitude** scores (energy, max logit; ρ ≈ 0.72–0.90), not the logit gap (margin). |
| Mechanism check | Wrong PP2020 predictions have **smaller logits** (median log-sum-exp 2.5–3.1) than correct ones (3.2–3.8) and than lab images (3.6–4.6). Softmax confidence (MSP) discards this magnitude. |
| Source-only selection rule | Fixed in advance: pick the candidate score most correlated with B5 **on source views**. It picked *margin*, but all candidates were nearly tied on source data (ρ 0.81–0.83), so the rule could not discriminate. The energy hypothesis therefore came from PP2020, and this is disclosed in the pre-registration. |
| Batch alarm (source-only thresholds) | S1 is the share of a batch beyond the source 95th-percentile feature distance; S2 is the shift in predicted-class shares. Thresholds were set at the 99th percentile of source batches. Result: 0% false alarms on held-out source (about 1% for S2 on S-test) and 100% alarms on PP2020 batches. |

### Step 5: Label check and eligible set (`src/prepare_plantdoc_eligible.py`, commit `61a2b2a`)
- PlantDoc is web-scraped, so all 273 images were checked visually on contact sheets. The rule was fixed before looking: exclude only obvious problems (not a photo; no apple leaf, e.g. fruit only; another plant; clearly another class). An unsure image is kept.
- The review was done by the author, and every flagged image was cross-checked at full resolution. **8 images were excluded:**

| # | Folder label | Reason |
|---|---|---|
| 134 | healthy | Leaves in the photo are not apple leaves |
| 32 | scab | Pressed, dried specimen leaves with orange rust-like spots |
| 47 | scab | Only fine white speckles (water or dust); no scab lesions |
| 217 | rust | Collage including fruit, a tree trunk and dead leaves |
| 219 | rust | Grayscale printed plate (book scan); no colour information |
| 231 | rust | Botanical illustration |
| 238 | rust | Captioned composite showing scab on fruit and leaf |
| 239 | rust | Captioned composite including frog-eye leaf spot |

Kept after full-resolution review:
- #20: early rust spots are visible at full size.
- #45: two real scab photos side by side.
- #6, #81, #90, #109, #114: unsure or acceptable.

**Eligible set: 265 images (healthy 90, scab 91, rust 84)**, above the pre-fixed minimum of 60 per class. They form 262 clusters, three of which contain a repeated pair. The label-free manifest has SHA-256 `66c43f19…`.

### Step 6: Feature distance (`src/measure_plantdoc_shift.py`, commit `3caaf1c`)
Only backbone features were computed; no logits or predictions. Images were standardised with the S-calibration mean and standard deviation.

| Seed | Source median | **PlantDoc median** | PP2020 median | PlantDoc beyond source 95th percentile |
|---|---|---|---|---|
| 42 | 0.96 | **2.10** | 1.85 | 97.7% |
| 123 | 0.96 | **2.03** | 1.85 | 96.6% |
| 2026 | 0.97 | **2.04** | 1.81 | 97.7% |

PlantDoc is **even further from the lab images than PP2020**. It is a strong-shift replication, so a "mild-shift" comparison is not possible with it. Because this result was known before sealing, the feature-based alarm S1 was made **descriptive only**.

### Step 7: Pre-registration sealed (`prereg/plantdoc_preregistration.md`, commit `9692e18`, SHA-256 `813e2283…`)

| ID | Hypothesis | Supported if |
|---|---|---|
| **H1 (primary)** | Logits-only head B5 ranks errors better than MSP | Mean ΔAURC(B5) < 0 and its 95% interval lies below 0 |
| H2 | Energy score E = −log Σ exp(logits) beats MSP | Mean ΔAURC(E) < 0 and its interval lies below 0 |
| H3 (mechanism) | B5 follows logit magnitude rather than logit gap | Mean [ρ(B5,E) − ρ(B5,−margin)] > 0 and its interval lies above 0 |
| H4 | Proposed head P1 does **not** beat MSP | Interval of mean ΔAURC(P1) includes 0 or lies above 0 |
| H5 | Source-only batch alarm S2 fires on PlantDoc | At least 95% of random batches (sizes 20 and 50) alarm, in every seed |

The underpowered rule: fewer than 50 errors in a seed. Statistics: 2,000 cluster resamples, seed 2026, with the seeds resampled jointly.

### Step 8: One-time inference (`src/run_plantdoc_inference.py`, commit `e5027c6`)
The prechecks passed: pre-registration hash, freeze record, manifest hash, and all 265 image hashes. Predicted classes, with no labels involved:

| Seed | healthy | scab | rust |
|---|---|---|---|
| 42 | 34 | 192 | 39 |
| 123 | 34 | 183 | 48 |
| 2026 | 36 | 172 | 57 |

### Step 9: One-time evaluation (`src/evaluate_plantdoc.py`, script `c296fbd`, results `b7547b1`)
The labels were joined for the first time here. The results are in sections 5 and 6.

### Step 10: Exploratory error-regime comparison (`src/explore_regimes.py`, commit `1e7f777`)
Decided after the PlantDoc results were known; see section 6.

## 5. Results: confirmatory (pre-registered)

### 5.1 Classification

| Seed | Errors | Accuracy | Macro-F1 | Recall healthy / scab / rust |
|---|---|---|---|---|
| 42 | 140 | 47.2% | 0.434 | 0.34 / 0.87 / 0.18 |
| 123 | 140 | 47.2% | 0.443 | 0.36 / 0.84 / 0.20 |
| 2026 | 138 | 47.9% | 0.458 | 0.36 / 0.80 / 0.26 |

- No seed is underpowered (138–140 errors, against a minimum of 50).
- **Unlike PP2020, the classifier did not collapse into one class.** Most errors are healthy or rust leaves predicted as scab.

### 5.2 Hypotheses

| ID | Result (mean over seeds, 95% cluster interval) | Verdict |
|---|---|---|
| **H1** B5 vs MSP | ΔAURC **+0.027 [−0.034, +0.085]** | **Inconclusive: the PP2020 advantage did not replicate** |
| H2 Energy vs MSP | ΔAURC −0.003 [−0.043, +0.037] | Inconclusive |
| H3 Mechanism | Δρ **+0.44 [+0.35, +0.53]** | **Supported** |
| H4 P1 vs MSP | ΔAURC **+0.077 [+0.006, +0.145]** (P1 is worse) | **Supported** |
| H5 Batch alarm S2 | 99.75–100% of batches alarm (sizes 20 and 50, all seeds) | **Supported** |

### 5.3 All methods (mean over seeds)

| Method | AURC | ΔAURC vs MSP [95% CI] | AUROC-error |
|---|---|---|---|
| B1 MSP | 0.434 | reference | 0.592 |
| B2 Entropy | 0.434 | +0.000 [−0.003, +0.004] | 0.593 |
| B3 Margin | 0.434 | +0.000 [−0.002, +0.003] | 0.592 |
| B4 Temp-scaled MSP | 0.435 | +0.001 [−0.002, +0.004] | 0.592 |
| B5 Logits-only head | 0.461 | +0.027 [−0.034, +0.085] | 0.568 |
| A1 Features-only head | 0.524 | **+0.090 [+0.019, +0.156]** | 0.520 |
| P1 Proposed head | 0.511 | **+0.077 [+0.006, +0.145]** | 0.525 |
| E Energy | 0.431 | −0.003 [−0.043, +0.037] | 0.601 |

The error rate (the AURC of a useless ranking) is 0.52–0.53. **Every method is only slightly better than chance at ranking individual errors.**

### 5.4 Calibration and cut-off transfer

| Quantity | PlantDoc result |
|---|---|
| NLL before → after frozen temperature | 1.87 → 4.41 (seed 42); 1.88 → 3.63; 1.75 → 3.35. **Temperature scaling made it worse.** |
| ECE before → after | 0.33–0.37 → 0.42–0.45 |
| Coverage at frozen tau80 (MSP) | 14–21%, against 80% intended |
| Coverage at frozen tau90 (MSP) | 28–43%, against 90% intended |

### 5.5 Sensitivity check (one image per cluster, 262 images)
The pattern is the same: B5 +0.027 [−0.036, +0.085]; E −0.003 [−0.044, +0.037]; P1 +0.079 [+0.007, +0.148].

## 6. Results: exploratory

### 6.1 Why B5 helped on PP2020 but not on PlantDoc (`results/exploratory_regimes/`)
A score's ability to rank errors has two parts:
- **Between classes:** whether whole predicted classes get higher or lower risk.
- **Within classes:** the order of images that share a predicted class.

| | PP2020 | PlantDoc |
|---|---|---|
| Predictions | 98–99% "scab" | scab 65–72%, rust 15–22%, healthy 13–14% |
| Where the errors are | 99.6–100% inside predicted scab | scab 72–81%, rust 17–25% |
| Error rate by predicted class | scab 65%; healthy and rust almost always correct | healthy 6–11%; **scab about 59%; rust about 62%** |
| Between-class AUROC (all methods) | 0.48–0.52 (uninformative: one class) | **0.38–0.45 (below chance: inverted)** |
| Within-class AUROC | MSP 0.66; **B5 0.75–0.79**; energy 0.71–0.73 | MSP 0.62; B5 0.62–0.65; **energy 0.66–0.69** |
| Median log-sum-exp, correct → wrong | 3.2–3.8 → 2.5–3.1 | 2.0–2.7 → 1.7–2.1 |

**Explanation.**
- Every confidence score inherits from the lab a **per-class trust order**: it trusts "scab" predictions most and "healthy" predictions least.
- In the field this order is **backwards**. Predicted healthy is the most reliable class (about 90% correct), while predicted scab and predicted rust are each about 60% wrong.
- On PP2020 the class collapse removed the between-class part, so the **within-class magnitude signal** decided the ranking, and B5 and energy won.
- On PlantDoc the inverted between-class part dominates and cancels the within-class gain, which still exists for energy.

### 6.2 Other exploratory observations
- The feature distance from the source predicts errors only weakly (AUROC 0.60–0.61, PlantDoc).
- The feature-based alarm S1 fires on 100% of PlantDoc batches. This is descriptive: its outcome was known before sealing.

## 7. Observations and interpretation

1. **Robust negative result, now pre-registered and replicated.** Failure heads that use image features and are trained on synthetic lab failures (P1, A1) rank field errors **no better than MSP on PP2020, and significantly worse on PlantDoc.** Field features lie far outside anything these heads saw in training.
2. **Source calibration does not transfer.** On both field sets, the source-fitted temperature makes calibration worse, and source-selected cut-offs answer far fewer images than intended.
3. **The B5 / energy advantage is regime-dependent.** It appears when errors are confined to one predicted class (class collapse, PP2020) and disappears when errors spread across classes (PlantDoc). The post-hoc PP2020 lead did not survive a sealed test, which is exactly what pre-registration is designed to catch.
4. **The mechanism is consistent.** B5 tracks logit magnitude on both datasets (H3), and wrong predictions have smaller logits on both. Magnitude separates errors *within* a predicted class, but not *across* classes.
5. **Per-image failure prediction is weak under strong field shift.** Every method is close to chance on PlantDoc (AUROC-error at most about 0.60). **Batch-level monitoring works:** a source-only alarm flagged both field datasets as out of scope.
6. **Practical message.** Under lab-to-field shift, do not rely on per-image confidence or source-fitted thresholds. Use a batch-level, source-only check to detect that the incoming images are outside the system's competence.

## 8. Limitations

- **PlantDoc is small** (265 images, 262 clusters), so intervals are wide. It is web-scraped, with mixed cameras, resolutions and some composites, and its labels are not expert-verified (only obvious errors were removed).
- **Both targets are strong-shift sets,** so no mild-shift point exists. The between/within explanation is exploratory and decided after the results.
- **Single architecture (MobileNetV3-Small) and three seeds** sharing the same data.
- **H3 does not depend on labels.** It describes how B5 behaves, not whether B5 helps.
- **The energy hypothesis came from PP2020 data** (disclosed). The source-only selection rule could not discriminate between candidates.
- **Duplicate review** used a pixel-difference hint instead of SIFT. The label check was done by non-pathologists.
- **The S1 alarm outcome** was visible before sealing and is therefore descriptive only.

## 9. Repository layout

```
README.md                      this file
notes/study_log.md             decisions, timeline, results and interpretation
prereg/plantdoc_preregistration.md   sealed pre-registration (SHA-256 813e2283...)
src/
  common.py                    paths, fixed rules, link to the read-only original project
  acquire_plantdoc.py          Step 1  acquisition with Git-blob verification
  audit_plantdoc.py            Step 2  decode, SHA-256, pHash, review sheets
  explore_pp2020.py            Step 4  exploratory analysis of the PP2020 outputs
  prepare_plantdoc_eligible.py Step 5  exclusions, clusters, label-free manifest
  measure_plantdoc_shift.py    Step 6  feature distance (features only)
  run_plantdoc_inference.py    Step 8  one-time frozen inference
  evaluate_plantdoc.py         Step 9  one-time evaluation
  explore_regimes.py           Step 10 exploratory regime comparison
  prepare_figure_data.py       plot-ready tables
manifests/                     acquisition, audit, review decisions, eligible set, label-free manifest
results/
  exploratory_pp2020/          Step 4 (exploratory)
  plantdoc_shift/              Step 6
  plantdoc_inference/          Step 8 predictions + hashes
  plantdoc_evaluation/         Step 9 confirmatory results + hashes
  exploratory_regimes/         Step 10 (exploratory)
figure_data/                   tables for figures (section 11)
data/                          images (not tracked: licence and size)
```

## 10. How to reproduce

Requirements:
- The original project at `D:\Capstone Project\Capstone Project`, with its models and the PlantVillage and PP2020 data.
- Its Python environment (Python 3.11.9, PyTorch 2.11.0+cu128).
- If the original project lives elsewhere, edit `FROZEN_ROOT` in `src/common.py`.

```bash
# 1. acquisition (blobless sparse clone, then blob-verified extraction)
git clone --filter=blob:none --no-checkout --sparse https://github.com/pratikkayal/PlantDoc-Dataset.git data/plantdoc_raw
python src/acquire_plantdoc.py
# 2-3. audit and review files (fill the decision columns as recorded in manifests/)
python src/audit_plantdoc.py
# 4. exploratory PP2020 analysis
python src/explore_pp2020.py
# 5-6. eligible set and feature distance
python src/prepare_plantdoc_eligible.py
python src/measure_plantdoc_shift.py
# 8-9. one-time inference and evaluation (both refuse to run twice)
python src/run_plantdoc_inference.py
python src/evaluate_plantdoc.py
# 10. exploratory regime comparison and figure data
python src/explore_regimes.py
python src/prepare_figure_data.py
```

Every one-time script verifies the freeze record, the sealed pre-registration and the recorded hashes before it runs.

## 11. Figure data

The tables in `figure_data/` are tidy CSVs, ready to plot:

| File | Planned figure |
|---|---|
| `F1_risk_coverage_curves.csv` | Risk-coverage curves, PlantDoc (confirmatory) next to PP2020, per seed and method |
| `F2_delta_aurc_both_targets.csv` | Forest plot of ΔAURC vs MSP on both field datasets, with 95% intervals |
| `F3_confusion_plantdoc.csv` | PlantDoc confusion matrices (counts and row shares) |
| `F4_reliability_bins_plantdoc.csv` | PlantDoc reliability diagrams before and after the frozen temperature |
| `F5_logit_magnitude.csv` | Log-sum-exp of correct vs wrong predictions, both datasets (violin or box plot) |
| `F6_between_within_auroc.csv` | Between-class vs within-class AUROC-error per score (grouped bars), explaining H1 |
| `F7_error_regime.csv` | Share of predictions and errors per predicted class (stacked bars), collapse vs spread |

## 12. Provenance, licences and hashes

| Item | Value |
|---|---|
| PlantDoc | https://github.com/pratikkayal/PlantDoc-Dataset, commit `5467f6012d78d1c446145d5f582da6096f852ae8`, **CC BY 4.0**. Singh et al., *PlantDoc: A Dataset for Visual Plant Disease Detection*, CoDS-COMAD 2020. Images are not redistributed here. |
| Original freeze record | `results/frozen_system_manifest.json`, SHA-256 `eebe9c9b28c7dc670dca0329c305a1154cff4861b31737799bc244225df5ca83` |
| PlantDoc audit manifest | `385ee847b41ee0c9aa616d1e101bddf6228a8a64b69f972a2cf34b1d0ed9300d` |
| Label-free inference manifest | `66c43f19417c3fa39588ef88b13f1988d8a93e3e06d17931991615ea23ffe149` |
| Pre-registration | `813e22832c7b7c814184ea7814a4327d676eb08d4937da0525975efbc2cc8dc7` |
| Prediction and evaluation file hashes | `results/plantdoc_inference/plantdoc_inference_hashes.txt`, `results/plantdoc_evaluation/evaluation_hashes.txt` |

Contact sheets containing PP2020 or PlantVillage thumbnails are kept locally and not committed, because PP2020 images may not be redistributed.
