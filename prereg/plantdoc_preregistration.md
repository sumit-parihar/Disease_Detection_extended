# Pre-registration: one-time test of the frozen system on PlantDoc

Status: written and committed **before any PlantDoc logit, prediction or score exists**. The only model computation run on PlantDoc so far is the backbone feature extraction of Phase 5 (576 numbers per image, used for the feature distance). The classifier's output layer and the failure heads have not been run on any PlantDoc image.

Once this file is committed, its SHA-256 is written into the inference and evaluation scripts, which refuse to run if it changes.

## 1. Frozen system (unchanged; nothing is retrained, recalibrated or re-thresholded)

- Original study: `D:\Capstone Project\Capstone Project`, repository `Disease_Detection_`, frozen at commit `160d92a`.
- Freeze record `results/frozen_system_manifest.json`, SHA-256 `eebe9c9b28c7dc670dca0329c305a1154cff4861b31737799bc244225df5ca83` (31 files). `verify_frozen()` is called before inference and before evaluation, and the run stops if any file differs.
- Components used: classifiers for seeds 42, 123 and 2026; the proposed head (P1), the logits-only head (B5) and the features-only head (A1); the frozen temperatures and the tau80/tau90 cut-offs in `results/frozen_calibration_settings.json`.
- Preprocessing is identical to the original target evaluation: decode, convert to RGB (one grayscale and one CMYK file are converted), resize to 224 x 224 without keeping the aspect ratio, and ImageNet normalisation. No EXIF rotation is applied (3 files carry orientation tag 6) and there is no augmentation.

## 2. Dataset

- PlantDoc (Singh et al., CoDS-COMAD 2020), https://github.com/pratikkayal/PlantDoc-Dataset, commit `5467f6012d78d1c446145d5f582da6096f852ae8`, licence CC BY 4.0. Folders `Apple leaf` (healthy), `Apple Scab Leaf` (scab) and `Apple rust leaf` (rust), with the train and test folders both used.
- Acquisition: 273 files. The bytes were read from the Git objects and checked against their blob ids (`manifests/plantdoc_acquisition.csv`).
- Audit (`manifests/plantdoc_manifest.csv`, SHA-256 `385ee847b41ee0c9aa616d1e101bddf6228a8a64b69f972a2cf34b1d0ed9300d`):
  - All 273 files decode.
  - There are no exact copies within PlantDoc and no exact copies of any PlantVillage or PP2020 image.
  - 18 pHash pairs at 0 to 10 bits were reviewed visually. Pairs 1 to 3 (within PlantDoc) are duplicates. The 12 PlantDoc-PlantVillage pairs and the 3 other within-PlantDoc pairs are different images. No pair involves PP2020.
- Visual label check: rule fixed before review. Only obvious problems are excluded, and an unsure image is kept. 8 images were excluded (`manifests/plantdoc_label_check.csv`):
  - healthy #134: not apple leaves;
  - scab #32: dried specimens; #47: no lesions;
  - rust #217: collage; #219: grayscale plate; #231: illustration; #238 and #239: captioned composites showing other diseases or fruit.
- Eligible set: **265 images (healthy 90, scab 91, rust 84)**. The size rule (at least 60 per class) is passed.
- Clusters: 262. Three clusters contain two images each (the three duplicate pairs).
- Label-free inference manifest: `manifests/plantdoc_inference_manifest.csv` (path, sha256, cluster_id, cluster_size), SHA-256 `66c43f19417c3fa39588ef88b13f1988d8a93e3e06d17931991615ea23ffe149`. Files are anonymised as `pd_XXXX.jpg` in SHA-256 order, and the inference script reads only this file.
- Feature distance (Phase 5, `results/plantdoc_shift/summary.txt`): the median centre distance is 2.10, 2.03 and 2.04 (seeds 42, 123 and 2026), against about 0.96 for source and 1.81 to 1.85 for PP2020. Between 96.6% and 97.7% of images lie beyond the source 95th percentile.
  - **PlantDoc is therefore a strong-shift field set, not a mild-shift set.** The planned mild-shift comparison cannot be made with this dataset.

## 3. How the hypotheses were formed (disclosure)

- H1 (B5 better than MSP) restates the prespecified comparison B5 from the original study, where it was observed on PP2020.
- H2 and H3 come from an **exploratory** analysis of the saved PP2020 outputs (`results/exploratory_pp2020/`, commit `243a935`). That analysis found the following:
  - B5's advantage holds inside the predicted-scab group (AUROC-error 0.75 to 0.79, against 0.51 to 0.52 for the "predicted scab" shortcut).
  - Wrong PP2020 predictions have smaller logits than correct ones.
  - Magnitude-based scores (energy, max logit) beat MSP, whereas the logit margin does not.
- A rule fixed before that analysis chose the candidate with the highest correlation with B5 **on source views**. It selected the logit margin, but all candidates were nearly tied on source data (Spearman 0.81 to 0.83), so that rule could not discriminate between them.
  - **The energy hypothesis was therefore formed from PP2020 data.** It is tested here on independent data for the first time.
- The batch-alarm thresholds were set from source data only (reliability-training originals) in the same exploratory run.

## 4. Hypotheses and endpoints

Definitions:
- An error is a predicted class different from the PlantDoc folder label.
- For every score, higher means less trustworthy.
- AURC, AUROC-error and AUPRC-error are computed with the original `metrics.py`, including its tie averaging.
- ΔAURC(X) = AURC(X) − AURC(MSP) on the same images. It is computed per seed and as the mean over the three seeds.

| ID | Type | Hypothesis | Supported if |
|---|---|---|---|
| **H1** | **Primary** | The logits-only head B5 ranks errors better than MSP | Mean ΔAURC(B5) < 0 and its 95% interval lies entirely below 0 |
| H2 | Secondary | The energy score E = −log Σ exp(logits) (raw logits, no temperature) ranks errors better than MSP | Mean ΔAURC(E) < 0 and its 95% interval lies below 0 |
| H3 | Secondary (mechanism) | B5 follows logit magnitude rather than logit gap | The seed-mean of [Spearman(B5, E) − Spearman(B5, −margin)] > 0 and its 95% interval lies above 0. Margin = top logit − runner-up logit. |
| H4 | Secondary | The proposed head P1 does **not** rank errors better than MSP | The 95% interval of mean ΔAURC(P1) includes 0 or lies above 0 |
| H5 | Secondary | The source-only batch alarm S2 fires on PlantDoc | In every seed, at batch sizes 20 and 50, at least 95% of 2,000 random PlantDoc batches exceed the frozen threshold |

Interpretation of H1 and H2: an interval entirely below 0 means supported, an interval containing 0 means inconclusive, and an interval entirely above 0 means contradicted.

**H5 details.**
- S2 is the total-variation distance between the batch's predicted-class shares and the predicted-class shares of S-calibration for the same seed.
- The thresholds are frozen, copied from `results/exploratory_pp2020/part3_batch_alarm.csv`, and come from the 99th percentile over reliability-training source batches:

| Seed | Batch size 20 | Batch size 50 |
|---|---|---|
| 42 | 0.299606 | 0.169606 |
| 123 | 0.255906 | 0.175906 |
| 2026 | 0.255906 | 0.155906 |

- The feature-based alarm S1 (share beyond the source 95th percentile, thresholds 0.20 at size 20 and 0.14 at size 50) is **not** a confirmatory test. Its outcome was already visible in Phase 5 (97% beyond the 95th percentile), so it is reported descriptively only.

## 5. Also reported (descriptive, no hypothesis)

- **Classification:** accuracy, macro-F1, per-class precision, recall and F1, and confusion matrices, per seed.
- **Calibration:** NLL, ECE (15 bins) and Brier score, before and after the frozen temperature.
- **All seven methods (B1 to B5, A1 and P1) plus E:**
  - AURC, ΔAURC against MSP with intervals, AUROC-error and AUPRC-error;
  - coverage and selective risk at the frozen tau80 and tau90 (E has no frozen cut-off, so it is omitted here);
  - selective risk at fixed coverages of 50%, 70%, 80% and 90%.
- **Exploratory, labelled as such:** within-predicted-class AUROC-error; centre distance as an error predictor; per-class breakdowns.

## 6. Statistics and decision rules

- **Bootstrap:** 2,000 resamples of whole clusters (`cluster_id`), random seed 2026, percentile 95% intervals. The same resamples are shared by all methods and seeds, and the mean over seeds is resampled jointly.
- **Sensitivity check:** one image per cluster (262 images), drawn from the first image in SHA-256 order.
- **Underpowered rule:** a seed with fewer than 50 errors is reported as underpowered. If any seed is underpowered, the H1 to H4 conclusions are labelled underpowered. A seed with no errors makes AUROC and AUPRC undefined, and they are reported as such.
- **One run only:** inference runs once and refuses to run again if outputs exist. The prediction files are hashed, committed and pushed **before** labels are joined.
- **No changes after results:** no dataset swap, threshold change or method change is allowed. Every outcome, including failure of H1, is reported.

## 7. Deviations from the original protocol, declared in advance

- The SIFT consistency check of the duplicate review was not available, so a pixel-difference hint was used instead and the decisions are visual.
- A visual label check was added because PlantDoc is web-scraped. The rule was fixed before the review.
- File names are anonymised for inference because PlantDoc folder names reveal the labels.
- The size threshold of 60 per class was fixed before the audit.
