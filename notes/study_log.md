# Study log: pre-registered PlantDoc test of the frozen system

This log records only things that were actually decided or observed. Labels mark each item as confirmatory (pre-registered) or exploratory. Last updated 2026-11-25.

## 1. Purpose

The aim is to strengthen the original study (source-only failure head, PlantVillage → PP2020) by testing the **frozen** system once on a second, independent field dataset under a sealed pre-registration. The approach is to explain on PP2020, predict in advance, and confirm on new data.

## 2. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | Second target is PlantDoc (apple healthy/scab/rust, CC BY 4.0, GitHub, no access request). PP2021 was rejected because it is built on PP2020 (same orchards). AppleLeaf9 and the Zenodo 13-class set were rejected because they contain PlantVillage and PP data. | Independence from both source and PP2020 |
| D2 | The original project is read-only. Every step that uses the frozen system calls `verify_frozen()` first. | Keeps the freeze intact |
| D3 | Size rule: at least 60 images per class after exclusions; fewer than 50 errors in a seed means underpowered. | PlantDoc has about 90 per class; statistical power depends on the error count |
| D4 | Duplicate review: same rule as the original study (`duplicate`/`different`/`unsure`, with unsure counted as duplicate). No SIFT was available, so a pixel-difference hint was used. | Consistency |
| D5 | Label check: rule fixed before review. Only obvious problems are excluded (not a photo; no apple leaf, e.g. fruit only; another plant; clearly another class), and unsure images are kept. | PlantDoc is web-scraped; avoids selective exclusion |
| D6 | Inference reads anonymised copies (`pd_XXXX.jpg`, ordered by SHA-256). | Folder names reveal labels |
| D7 | Pre-registration option A: H1 primary (B5 better than MSP); H2 energy; H3 mechanism; H4 P1 does not beat MSP; H5 batch alarm S2. The origin of the energy hypothesis in PP2020 is disclosed. | Strongest explain-predict-confirm design |
| D8 | Alarm S1 (feature distance) is descriptive only, because its outcome was visible in Phase 5 before sealing. | Honesty about what was already known |
| D9 | Files are stored byte-exact in Git (`.gitattributes: * -text`). | Hashes verify on any system |

## 3. Timeline (commits in `Disease_Detection_extended`)

| Commit | Step |
|---|---|
| `3a38956` | Phase 1: acquired 273 files from PlantDoc commit `5467f60`, bytes verified against the Git blob ids |
| `aaa7eab` | Phase 3: audit. All files decode; no exact copies; 18 pHash pairs (0–10 bits) prepared for review |
| `400ac2f` | Duplicate review: pairs 1–3 are duplicates within PlantDoc; 4–18 are different (no overlap with PlantVillage or PP2020) |
| `243a935` | Step 2, exploratory on PP2020 outputs: B5 shortcut check, B5 probe, logit magnitude, batch alarm |
| `61a2b2a` | Phase 4: label check excluded 8; 265 eligible (healthy 90 / scab 91 / rust 84); 262 clusters; label-free manifest |
| `3caaf1c` | Phase 5: feature distance of PlantDoc (median 2.03–2.10; PP2020 1.81–1.85; source about 0.96) |
| `9692e18` | **Pre-registration sealed** (SHA-256 `813e2283…`) |
| `e5027c6` | Phase 8: one-time inference; predictions hashed and pushed before labels were read |
| `c296fbd` | Evaluation script committed (tested only on random labels) |
| `b7547b1` | Phase 9: one-time evaluation |
| `1e7f777` | Exploratory: error-regime comparison of PP2020 and PlantDoc |

## 4. Confirmatory results on PlantDoc (as pre-registered)

The classifier reached 47–48% accuracy with 138–140 errors per seed, so no seed is underpowered. The classifier did not collapse into a single class; scab was predicted for 65–72% of images.

| ID | Hypothesis | Result (mean over seeds, 95% cluster interval) | Verdict |
|---|---|---|---|
| H1 | B5 better than MSP | ΔAURC +0.027 [−0.034, +0.085] | Inconclusive (not replicated) |
| H2 | Energy better than MSP | ΔAURC −0.003 [−0.043, +0.037] | Inconclusive |
| H3 | B5 follows magnitude rather than gap | Δρ +0.44 [+0.35, +0.53] | Supported |
| H4 | P1 does not beat MSP | ΔAURC +0.077 [+0.006, +0.145] (worse than MSP) | Supported |
| H5 | Batch alarm S2 fires | 99.8–100% of batches | Supported |

Descriptive results:
- The features-only head A1 is worse than MSP: +0.090 [+0.019, +0.156].
- The frozen temperature worsens calibration (NLL 1.75–1.88 → 3.35–4.41).
- The tau80 cut-off answers only 14–21% of images under MSP (80% intended).
- The best AUROC-error of any method is about 0.60.
- The one-image-per-cluster check gives the same pattern.

## 5. Exploratory explanation: why B5 helped on PP2020 but not on PlantDoc

This analysis was decided after the PlantDoc results were known (`results/exploratory_regimes/`).

1. **Different error regimes.**
   - On PP2020, 98–99% of images are predicted scab, and 99.6–100% of errors lie in that one group.
   - On PlantDoc, predictions are spread out: errors fall in scab (72–81%) and rust (17–25%). Predicted healthy is mostly correct (error rate 6–11%), while predicted scab and predicted rust are each about 60% wrong.
2. **The ranking splits into a between-class part and a within-class part.**
   - On PP2020, the between-class part is uninformative for every method (AUROC 0.48–0.52) because there is effectively one class. Ranking quality therefore comes entirely from *within* the scab group, where B5 (0.75–0.79) and energy (0.71–0.73) beat MSP (0.66).
   - On PlantDoc, the between-class part is **inverted** for every method (AUROC 0.38–0.45, below chance). All scores treat predicted healthy as the least trustworthy class and predicted scab as the most trustworthy, which is the opposite of the field error rates. Confidence learned in the lab ranks whole classes in the wrong order in the field.
3. **The magnitude signal still exists within classes.**
   - Wrong predictions have smaller logits in both datasets (PlantDoc median log-sum-exp 1.74–2.12 for wrong against 2.01–2.68 for correct).
   - Within predicted classes, energy beats MSP on PlantDoc (weighted AUROC 0.66–0.69 against 0.62).
   - However, energy and B5 give predicted scab the lowest risk, so the inverted between-class ordering cancels the within-class gain.
4. **Interpretation (hypothesis, not tested).** The B5 and energy advantage appears when errors are confined to one predicted class, as under class collapse. When errors spread across classes, class-level miscalibration inherited from the source dominates, and no output-based score fixes it. A per-predicted-class correction could address this. It cannot be fitted from source data alone and would need new, pre-registered evaluation.

## 6. What the evidence now supports (for the later write-up)

- **Replicated and pre-registered:** feature-based failure heads trained on synthetic source failures (A1, P1) rank field errors no better than MSP, and worse on PlantDoc.
- **Replicated:** source-fitted temperature scaling and source-selected cut-offs do not transfer to field data.
- **Not replicated:** the PP2020 advantage of the logits-only head B5 and of the energy score. It is regime-dependent, as explained in section 5, which is exploratory.
- **Replicated:** a source-only batch-level alarm flags both field datasets as out of scope.
- **Mechanism (H3):** B5 tracks logit magnitude in both datasets. Magnitude separates errors within a predicted class but not across classes.
