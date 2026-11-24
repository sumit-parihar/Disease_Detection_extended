"""
evaluate_plantdoc.py  -  Phase 9: the ONE-TIME evaluation on PlantDoc, exactly as pre-registered
(prereg/plantdoc_preregistration.md). Labels are joined here for the first time.

Prechecks (any failure stops): outputs do not exist; the pre-registration is the sealed version;
the freeze record verifies; the label-free manifest is unchanged; every prediction file matches
the hash recorded right after inference; the prediction files are committed and pushed.

RUN (project root), once:  python src/evaluate_plantdoc.py
OUTPUT: results/plantdoc_evaluation/
"""

import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support

from common import (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, FROZEN_ROOT, MANIFEST_DIR, MIN_ERRORS_PER_SEED, PREREG_DIR,
                    PROJECT_ROOT, RESULTS_DIR, sha256_file, use_frozen_code)

use_frozen_code()
from freeze_system import verify_frozen                                              # noqa: E402
from metrics import (auprc_error, aurc, auroc_error, calibration_summary,             # noqa: E402
                     classification_summary, risk_coverage_curve, selective_risk_at)

SEEDS = (42, 123, 2026)
CLASSES = ["healthy", "scab", "rust"]
PREREG = PREREG_DIR / "plantdoc_preregistration.md"
PREREG_SHA = "813e22832c7b7c814184ea7814a4327d676eb08d4937da0525975efbc2cc8dc7"
MANIFEST = MANIFEST_DIR / "plantdoc_inference_manifest.csv"
SIDECAR = MANIFEST_DIR / "plantdoc_inference_manifest.csv.sha256"
INF = RESULTS_DIR / "plantdoc_inference"
HASHES = INF / "plantdoc_inference_hashes.txt"
PREDICTIONS = {s: INF / f"plantdoc_predictions_seed{s}.csv" for s in SEEDS}
OUT = RESULTS_DIR / "plantdoc_evaluation"
SETTINGS = json.loads((FROZEN_ROOT / "results" / "frozen_calibration_settings.json").read_text(encoding="utf-8"))

METHODS = {"B1 MSP": "msp_score", "B2 Entropy": "entropy_score", "B3 Margin": "margin_score",
           "B4 Temp-scaled MSP": "temp_scaled_msp_score", "B5 Logits-only head": "head_logits_only",
           "A1 Features-only head": "head_features_only", "P1 Proposed head": "head_proposed",
           "E Energy (pre-registered)": "energy_score"}
S2_THRESHOLDS = {42: {20: 0.299606, 50: 0.169606}, 123: {20: 0.255906, 50: 0.175906}, 2026: {20: 0.255906, 50: 0.155906}}
S1_THRESHOLDS = {20: 0.20, 50: 0.14}
FIXED_COVERAGES = (0.5, 0.7, 0.8, 0.9)


def relative(path):
    return path.relative_to(PROJECT_ROOT).as_posix()


def precheck():
    if OUT.exists() and any(OUT.iterdir()):
        sys.exit("PlantDoc evaluation outputs already exist. The evaluation is run once.")
    if sha256_file(PREREG) != PREREG_SHA:
        sys.exit("The pre-registration differs from the sealed version.")
    verify_frozen()
    if sha256_file(MANIFEST) != SIDECAR.read_text(encoding="utf-8").split()[0]:
        sys.exit("The label-free manifest has changed.")
    recorded = {}
    for line in HASHES.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            recorded[parts[1]] = parts[0]
    for path in PREDICTIONS.values():
        if recorded.get(relative(path)) != sha256_file(path):
            sys.exit(f"{path.name} differs from the hash recorded after inference.")
    tracked = [relative(p) for p in PREDICTIONS.values()] + [relative(HASHES), relative(PREREG)]
    status = subprocess.run(["git", "status", "--porcelain", "--", *tracked], cwd=PROJECT_ROOT, capture_output=True, text=True)
    ahead = subprocess.run(["git", "rev-list", "--count", "@{u}..HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    if status.returncode or status.stdout.strip() or ahead.returncode or ahead.stdout.strip() != "0":
        sys.exit("Prediction files are not committed and pushed. Push them first.")
    print("Prechecks passed: pre-registration, freeze, manifest, prediction hashes, commit and push.")


def cluster_resamples(cluster_ids):
    groups = [np.flatnonzero(cluster_ids == c) for c in pd.unique(cluster_ids)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
            for _ in range(BOOTSTRAP_RESAMPLES)]


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if len(values) else (math.nan, math.nan)


def risk_at_coverage(errors, scores, coverage):
    risks = risk_coverage_curve(errors, scores)[1]
    return float(risks[math.ceil(coverage * len(errors)) - 1])


def tv_distance(predicted, reference_shares):
    shares = np.bincount(predicted, minlength=3) / len(predicted)
    return 0.5 * np.abs(shares - reference_shares).sum()


def verdict_below_zero(low, high):
    return "SUPPORTED" if high < 0 else ("CONTRADICTED" if low > 0 else "INCONCLUSIVE")


def main():
    global OUT
    dry_run = len(sys.argv) == 3 and sys.argv[1] == "--dry-run-random-labels"
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    if dry_run:
        # Code test only: the real labels are NOT read; random labels; output to the given folder.
        from pathlib import Path
        OUT = Path(sys.argv[2])
        OUT.mkdir(parents=True, exist_ok=True)
        fake = np.random.default_rng(0).choice(CLASSES, len(manifest))
        labels = pd.Series(fake, index=manifest["path"])
        print("DRY RUN with random labels (real labels not read).")
    else:
        precheck()
        OUT.mkdir(parents=True, exist_ok=True)
        print("Joining the PlantDoc labels now.")
        labels = pd.read_csv(MANIFEST_DIR / "plantdoc_eligible.csv", keep_default_na=False).set_index("staged_path")["label"]

    tables = {}
    for seed in SEEDS:
        t = pd.read_csv(PREDICTIONS[seed], keep_default_na=False)
        if t["path"].tolist() != manifest["path"].tolist():
            sys.exit("Prediction rows are not in manifest order.")
        t["true_index"] = [CLASSES.index(labels[p]) for p in t["path"]]
        t["error"] = (t["predicted_index"] != t["true_index"]).astype(int)
        logits = t[["logit_healthy", "logit_scab", "logit_rust"]].to_numpy(dtype=np.float64)
        s = np.sort(logits, axis=1)
        t["neg_logit_margin"] = -(s[:, 2] - s[:, 1])
        tables[seed] = t
    clusters = manifest["cluster_id"].to_numpy()
    resamples = cluster_resamples(clusters)

    # ---------------- classification and calibration
    class_rows, confusion_rows, calib_rows = [], [], []
    for seed, t in tables.items():
        summary = classification_summary(t["true_index"], t["predicted_index"], CLASSES)
        p, r, f, n = precision_recall_fscore_support(t["true_index"], t["predicted_index"], labels=[0, 1, 2], zero_division=0)
        class_rows.append({"seed": seed, "images": len(t), "errors": int(t["error"].sum()), "error_rate": t["error"].mean(),
                           "accuracy": summary["accuracy"], "macro_f1": summary["macro_f1"],
                           "underpowered": bool(t["error"].sum() < MIN_ERRORS_PER_SEED),
                           **{f"precision_{c}": p[i] for i, c in enumerate(CLASSES)},
                           **{f"recall_{c}": r[i] for i, c in enumerate(CLASSES)},
                           **{f"f1_{c}": f[i] for i, c in enumerate(CLASSES)}})
        for i, c in enumerate(CLASSES):
            confusion_rows.append({"seed": seed, "true": c, **{f"pred_{d}": int(summary["confusion_matrix"][i, j]) for j, d in enumerate(CLASSES)}})
        logits = t[["logit_healthy", "logit_scab", "logit_rust"]].to_numpy(dtype=np.float64)
        before = calibration_summary(logits, t["true_index"].to_numpy())
        after = calibration_summary(logits, t["true_index"].to_numpy(), SETTINGS[str(seed)]["temperature"])
        calib_rows.append({"seed": seed, **{f"{k}_before": v for k, v in before.items()}, **{f"{k}_after": v for k, v in after.items()}})
    classification = pd.DataFrame(class_rows)
    underpowered = bool(classification["underpowered"].any())

    # ---------------- reliability (per seed) with shared cluster resamples
    rel_rows, deltas = [], {m: np.zeros((len(SEEDS), BOOTSTRAP_RESAMPLES)) for m in METHODS}
    point_delta = {m: [] for m in METHODS}
    for k, (seed, t) in enumerate(tables.items()):
        e = t["error"].to_numpy()
        msp = t["msp_score"].to_numpy()
        for m, col in METHODS.items():
            x = t[col].to_numpy()
            d = aurc(e, x) - aurc(e, msp)
            point_delta[m].append(d)
            deltas[m][k] = [aurc(e[i], x[i]) - aurc(e[i], msp[i]) for i in resamples]
            lo, hi = interval(deltas[m][k])
            row = {"seed": seed, "method": m, "aurc": aurc(e, x), "delta_aurc_vs_msp": d, "ci_low": lo, "ci_high": hi,
                   "auroc_error": auroc_error(e, x) if 0 < e.sum() < len(e) else math.nan,
                   "auprc_error": auprc_error(e, x) if 0 < e.sum() < len(e) else math.nan}
            if m in SETTINGS[str(seed)]["thresholds"]:
                for tag in ("tau80", "tau90"):
                    sr = selective_risk_at(e, x, SETTINGS[str(seed)]["thresholds"][m][tag])
                    row[f"coverage_{tag}"], row[f"risk_{tag}"] = sr["coverage"], sr["selective_risk"]
            for c in FIXED_COVERAGES:
                row[f"risk_at_{int(c * 100)}pct"] = risk_at_coverage(e, x, c)
            rel_rows.append(row)
    reliability = pd.DataFrame(rel_rows)
    mean_rows = []
    for m in METHODS:
        lo, hi = interval(deltas[m].mean(axis=0))
        sub = reliability[reliability["method"] == m]
        mean_rows.append({"method": m, "mean_aurc": sub["aurc"].mean(), "mean_delta_aurc_vs_msp": float(np.mean(point_delta[m])),
                          "ci_low": lo, "ci_high": hi, "mean_auroc_error": sub["auroc_error"].mean(),
                          "mean_auprc_error": sub["auprc_error"].mean()})
    seed_mean = pd.DataFrame(mean_rows).set_index("method")

    # ---------------- H3 mechanism
    h3_point, h3_draws = [], np.zeros((len(SEEDS), BOOTSTRAP_RESAMPLES))
    for k, t in enumerate(tables.values()):
        b5, en, mg = t["head_logits_only"].to_numpy(), t["energy_score"].to_numpy(), t["neg_logit_margin"].to_numpy()
        h3_point.append(spearmanr(b5, en)[0] - spearmanr(b5, mg)[0])
        h3_draws[k] = [spearmanr(b5[i], en[i])[0] - spearmanr(b5[i], mg[i])[0] for i in resamples]
    h3_lo, h3_hi = interval(h3_draws.mean(axis=0))
    h3_mean = float(np.mean(h3_point))

    # ---------------- H5 batch alarm (S2 confirmatory; S1 descriptive)
    distances = pd.read_csv(RESULTS_DIR / "plantdoc_shift" / "centre_distances.csv")
    alarm_rows = []
    for seed, t in tables.items():
        cal = np.load(FROZEN_ROOT / "results" / f"s_calibration_features_seed{seed}.npz")
        cal_x = cal["features"].astype(np.float64)
        mean, std = cal_x.mean(axis=0), cal_x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        q95 = np.percentile(np.linalg.norm((cal_x - mean) / std, axis=1) / np.sqrt(cal_x.shape[1]), 95)
        shares = np.bincount(cal["logits"].argmax(axis=1), minlength=3) / len(cal_x)
        beyond = (distances[distances["seed"] == seed].set_index("path").loc[t["path"], "centre_distance"].to_numpy() > q95)
        predicted = t["predicted_index"].to_numpy()
        rng = np.random.default_rng(BOOTSTRAP_SEED + seed)
        for size in (20, 50):
            draws = [rng.choice(len(t), size, replace=False) for _ in range(BOOTSTRAP_RESAMPLES)]
            s2 = np.array([tv_distance(predicted[i], shares) for i in draws])
            s1 = np.array([beyond[i].mean() for i in draws])
            alarm_rows.append({"seed": seed, "batch_size": size, "S2 threshold": S2_THRESHOLDS[seed][size],
                               "S2 alarm rate": float((s2 > S2_THRESHOLDS[seed][size]).mean()),
                               "S2 median statistic": float(np.median(s2)),
                               "S1 alarm rate (descriptive)": float((s1 > S1_THRESHOLDS[size]).mean())})
    alarms = pd.DataFrame(alarm_rows)
    h5_ok = bool((alarms["S2 alarm rate"] >= 0.95).all())

    # ---------------- sensitivity: one image per cluster
    keep = ~manifest["cluster_id"].duplicated().to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    single_draws = [rng.integers(0, keep.sum(), keep.sum()) for _ in range(BOOTSTRAP_RESAMPLES)]
    sens_rows = []
    for m in ("B5 Logits-only head", "E Energy (pre-registered)", "P1 Proposed head"):
        per_seed = []
        for seed, t in tables.items():
            e, x, msp = t["error"].to_numpy()[keep], t[METHODS[m]].to_numpy()[keep], t["msp_score"].to_numpy()[keep]
            per_seed.append([aurc(e[i], x[i]) - aurc(e[i], msp[i]) for i in single_draws])
            lo, hi = interval(per_seed[-1])
            sens_rows.append({"method": m, "seed": seed, "images": int(keep.sum()),
                              "delta_aurc_vs_msp": aurc(e, x) - aurc(e, msp), "ci_low": lo, "ci_high": hi})
        lo, hi = interval(np.mean(per_seed, axis=0))
        sens_rows.append({"method": m, "seed": "mean", "images": int(keep.sum()),
                          "delta_aurc_vs_msp": float(np.mean([r["delta_aurc_vs_msp"] for r in sens_rows[-3:]])),
                          "ci_low": lo, "ci_high": hi})
    sensitivity = pd.DataFrame(sens_rows)

    # ---------------- exploratory: within the most-predicted class; centre distance as a score
    expl_rows = []
    for seed, t in tables.items():
        top_class = int(t["predicted_index"].mode()[0])
        sub = t[t["predicted_index"] == top_class]
        e = sub["error"].to_numpy()
        dist = distances[distances["seed"] == seed].set_index("path").loc[t["path"], "centre_distance"].to_numpy()
        for m, col in METHODS.items():
            expl_rows.append({"seed": seed, "subset": f"predicted {CLASSES[top_class]} ({len(sub)} images, {int(e.sum())} errors)",
                              "method": m, "auroc_error": auroc_error(e, sub[col]) if 0 < e.sum() < len(e) else math.nan})
        expl_rows.append({"seed": seed, "subset": "all images", "method": "centre distance",
                          "auroc_error": auroc_error(t["error"], dist)})
    exploratory = pd.DataFrame(expl_rows)

    # ---------------- verdicts
    h1 = seed_mean.loc["B5 Logits-only head"]
    h2 = seed_mean.loc["E Energy (pre-registered)"]
    h4 = seed_mean.loc["P1 Proposed head"]
    tag = " (UNDERPOWERED)" if underpowered else ""
    verdicts = [
        f"H1 primary  B5 vs MSP     mean dAURC {h1['mean_delta_aurc_vs_msp']:+.4f} [{h1['ci_low']:+.4f}, {h1['ci_high']:+.4f}]  -> {verdict_below_zero(h1['ci_low'], h1['ci_high'])}{tag}",
        f"H2          Energy vs MSP mean dAURC {h2['mean_delta_aurc_vs_msp']:+.4f} [{h2['ci_low']:+.4f}, {h2['ci_high']:+.4f}]  -> {verdict_below_zero(h2['ci_low'], h2['ci_high'])}{tag}",
        f"H3          rho(B5,E) - rho(B5,-margin), seed mean {h3_mean:+.4f} [{h3_lo:+.4f}, {h3_hi:+.4f}]  -> "
        f"{'SUPPORTED' if h3_lo > 0 else ('CONTRADICTED' if h3_hi < 0 else 'INCONCLUSIVE')}{tag}",
        f"H4          P1 vs MSP     mean dAURC {h4['mean_delta_aurc_vs_msp']:+.4f} [{h4['ci_low']:+.4f}, {h4['ci_high']:+.4f}]  -> "
        f"{'SUPPORTED (P1 does not beat MSP)' if h4['ci_high'] >= 0 else 'NOT SUPPORTED (P1 beats MSP)'}{tag}",
        f"H5          S2 batch alarm rate >= 0.95 in every seed and batch size -> {'SUPPORTED' if h5_ok else 'NOT SUPPORTED'} "
        f"(lowest rate {alarms['S2 alarm rate'].min():.3f})",
    ]

    for frame, name in ((classification, "classification"), (pd.DataFrame(confusion_rows), "confusion_matrices"),
                        (pd.DataFrame(calib_rows), "calibration"), (reliability, "reliability_by_seed"),
                        (seed_mean.reset_index(), "reliability_seed_mean"), (alarms, "batch_alarm"),
                        (sensitivity, "sensitivity_one_per_cluster"), (exploratory, "exploratory")):
        frame.to_csv(OUT / f"{name}.csv", index=False, lineterminator="\n")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    text = ["PLANTDOC EVALUATION OF THE FROZEN SYSTEM (one-time, as pre-registered)", "=" * 90, "",
            "PRE-REGISTERED HYPOTHESES", *verdicts, "",
            "CLASSIFICATION", classification.round(4).to_string(index=False), "",
            "CONFUSION MATRICES (rows true, columns predicted)", pd.DataFrame(confusion_rows).to_string(index=False), "",
            "CALIBRATION (frozen temperature)", pd.DataFrame(calib_rows).round(4).to_string(index=False), "",
            "RELIABILITY, MEAN OVER SEEDS (2,000 cluster resamples, seeds resampled jointly)", seed_mean.round(4).to_string(), "",
            "RELIABILITY BY SEED", reliability.round(4).to_string(index=False), "",
            "BATCH ALARM (S2 confirmatory, S1 descriptive)", alarms.round(4).to_string(index=False), "",
            "SENSITIVITY: ONE IMAGE PER CLUSTER", sensitivity.round(4).to_string(index=False), "",
            "EXPLORATORY (not pre-registered as hypotheses)", exploratory.round(4).to_string(index=False), "",
            "Nothing may be changed in the system, thresholds or analysis in response to these results."]
    (OUT / "summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8", newline="\n")
    hashes = [f"{sha256_file(p)}  {relative(p) if p.is_relative_to(PROJECT_ROOT) else p.name}"
              for p in sorted(OUT.glob("*")) if p.name != "evaluation_hashes.txt"]
    (OUT / "evaluation_hashes.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
