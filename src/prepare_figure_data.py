"""
prepare_figure_data.py  -  Plot-ready tables for the extended study (no plotting, no new analysis).

Everything is computed from saved, already-reported outputs; nothing is fitted or changed.
Files written to figure_data/ (one tidy CSV per planned figure):

  F1_risk_coverage_curves.csv     dataset, seed, method, coverage, selective_risk
                                  (PlantDoc = confirmatory; PP2020 shown for comparison)
  F2_delta_aurc_both_targets.csv  dataset, method, mean_delta_aurc, ci_low, ci_high  (forest plot)
  F3_confusion_plantdoc.csv       seed, true, predicted, count, row_share
  F4_reliability_bins_plantdoc.csv seed, temperature (before/after), bin, confidence, accuracy, images
  F5_logit_magnitude.csv          dataset, seed, path, correct, log_sum_exp  (violin/box plot)
  F6_between_within_auroc.csv     dataset, seed, score, component, auroc_error  (grouped bars)
  F7_error_regime.csv             dataset, seed, predicted class, share of images, share of errors, error rate

RUN (project root):  python src/prepare_figure_data.py
"""

import json

import numpy as np
import pandas as pd

from common import FROZEN_ROOT, MANIFEST_DIR, PROJECT_ROOT, RESULTS_DIR, use_frozen_code

use_frozen_code()
from calibrate_and_thresholds import softmax   # noqa: E402  (original code, read only)
from metrics import risk_coverage_curve        # noqa: E402

SEEDS = (42, 123, 2026)
CLASSES = ["healthy", "scab", "rust"]
OUT = PROJECT_ROOT / "figure_data"
SETTINGS = json.loads((FROZEN_ROOT / "results" / "frozen_calibration_settings.json").read_text(encoding="utf-8"))
METHODS = {"B1 MSP": "msp_score", "B2 Entropy": "entropy_score", "B3 Margin": "margin_score",
           "B4 Temp-scaled MSP": "temp_scaled_msp_score", "B5 Logits-only head": "head_logits_only",
           "A1 Features-only head": "head_features_only", "P1 Proposed head": "head_proposed",
           "E Energy": "energy_score"}


def load(dataset, seed):
    if dataset == "PP2020":
        t = pd.read_csv(FROZEN_ROOT / "results" / f"target_predictions_seed{seed}.csv", keep_default_na=False)
        m = pd.read_csv(FROZEN_ROOT / "manifests" / "manifest.csv", keep_default_na=False)
        labels = m[m["dataset"] == "target"].set_index("path")["label"]
    else:
        t = pd.read_csv(RESULTS_DIR / "plantdoc_inference" / f"plantdoc_predictions_seed{seed}.csv", keep_default_na=False)
        labels = pd.read_csv(MANIFEST_DIR / "plantdoc_eligible.csv", keep_default_na=False).set_index("staged_path")["label"]
    t["true_index"] = [CLASSES.index(labels[p]) for p in t["path"]]
    t["error"] = (t["predicted_index"] != t["true_index"]).astype(int)
    logits = t[["logit_healthy", "logit_scab", "logit_rust"]].to_numpy(dtype=np.float64)
    top = logits.max(axis=1)
    t["log_sum_exp"] = top + np.log(np.exp(logits - top[:, None]).sum(axis=1))
    t["energy_score"] = -t["log_sum_exp"]
    return t


def reliability_bins(logits, labels, temperature, bins=15):
    probs = softmax(logits / temperature)
    confidence, predicted = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for b in range(bins):
        mask = (confidence > edges[b]) & (confidence <= edges[b + 1])
        if mask.any():
            rows.append({"bin": b + 1, "bin_low": edges[b], "bin_high": edges[b + 1], "images": int(mask.sum()),
                         "mean_confidence": confidence[mask].mean(), "accuracy": (predicted[mask] == labels[mask]).mean()})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curves, magnitude = [], []
    for dataset in ("PlantDoc", "PP2020"):
        for seed in SEEDS:
            t = load(dataset, seed)
            for method, col in METHODS.items():
                coverage, risk = risk_coverage_curve(t["error"], t[col])
                curves.append(pd.DataFrame({"dataset": dataset, "seed": seed, "method": method,
                                            "coverage": coverage, "selective_risk": risk}))
            magnitude.append(pd.DataFrame({"dataset": dataset, "seed": seed, "path": t["path"],
                                           "correct": 1 - t["error"], "log_sum_exp": t["log_sum_exp"]}))
    pd.concat(curves).to_csv(OUT / "F1_risk_coverage_curves.csv", index=False, lineterminator="\n")
    pd.concat(magnitude).to_csv(OUT / "F5_logit_magnitude.csv", index=False, lineterminator="\n")

    pp = pd.read_csv(FROZEN_ROOT / "results" / "target_reliability_seed_mean.csv")
    pp = pp.rename(columns={"mean_delta_low": "ci_low", "mean_delta_high": "ci_high"})
    pdoc = pd.read_csv(RESULTS_DIR / "plantdoc_evaluation" / "reliability_seed_mean.csv")
    pdoc = pdoc.rename(columns={"mean_delta_aurc_vs_msp": "mean_delta_aurc"})
    pdoc["method"] = pdoc["method"].replace({"E Energy (pre-registered)": "E Energy"})
    columns = ["dataset", "method", "mean_delta_aurc", "ci_low", "ci_high"]
    forest = pd.concat([pp.assign(dataset="PP2020")[columns], pdoc.assign(dataset="PlantDoc")[columns]])
    forest["note"] = np.where(forest["dataset"] == "PlantDoc", "pre-registered one-time test", "original one-time test")
    forest.to_csv(OUT / "F2_delta_aurc_both_targets.csv", index=False, lineterminator="\n")

    conf = pd.read_csv(RESULTS_DIR / "plantdoc_evaluation" / "confusion_matrices.csv")
    long = conf.melt(id_vars=["seed", "true"], var_name="predicted", value_name="count")
    long["predicted"] = long["predicted"].str.replace("pred_", "", regex=False)
    long["row_share"] = long["count"] / long.groupby(["seed", "true"])["count"].transform("sum")
    long.to_csv(OUT / "F3_confusion_plantdoc.csv", index=False, lineterminator="\n")

    bins = []
    for seed in SEEDS:
        t = load("PlantDoc", seed)
        logits = t[["logit_healthy", "logit_scab", "logit_rust"]].to_numpy(dtype=np.float64)
        labels = t["true_index"].to_numpy()
        for tag, temp in (("before", 1.0), ("after", float(SETTINGS[str(seed)]["temperature"]))):
            bins += [{"seed": seed, "temperature": tag, **row} for row in reliability_bins(logits, labels, temp)]
    pd.DataFrame(bins).to_csv(OUT / "F4_reliability_bins_plantdoc.csv", index=False, lineterminator="\n")

    dec = pd.read_csv(RESULTS_DIR / "exploratory_regimes" / "auroc_decomposition.csv")
    dec.melt(id_vars=["dataset", "seed", "score"],
             value_vars=["AUROC overall", "AUROC between classes only", "AUROC within classes (weighted)"],
             var_name="component", value_name="auroc_error") \
       .to_csv(OUT / "F6_between_within_auroc.csv", index=False, lineterminator="\n")
    reg = pd.read_csv(RESULTS_DIR / "exploratory_regimes" / "error_regime.csv")
    reg[["dataset", "seed", "predicted", "images", "share_of_images", "errors", "share_of_all_errors", "error_rate"]] \
        .to_csv(OUT / "F7_error_regime.csv", index=False, lineterminator="\n")

    for f in sorted(OUT.glob("*.csv")):
        print(f"{f.name:36s} {len(pd.read_csv(f)):>7,} rows")


if __name__ == "__main__":
    main()
