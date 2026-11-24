"""
explore_pp2020.py  -  Step 2: EXPLORATORY analyses on the saved outputs of the original
study. Their only purpose is to generate the hypotheses that will be pre-registered for
the PlantDoc test. Nothing here is confirmatory and nothing in the original folder is
changed (it is only read; the freeze record is verified first).

PART 1  Shortcut check for the logits-only head (B5)
        Almost every PP2020 image was predicted "scab", and every error is in that group.
        Does B5 still rank errors above correct answers INSIDE the predicted-scab group,
        or does it only exploit "predicted scab = risky"?

PART 2  What did B5 learn?  (decided from SOURCE data only)
        B5 is a small network of the 3 logits. It is probed on the source reliability
        views (never target data):
          - sensitivity of B5 to each logit (mean absolute gradient on standardised inputs),
          - Spearman correlation between B5 and a FIXED list of zero-parameter logit scores.
        Selection rule, fixed before running: the candidate with the highest seed-mean
        Spearman correlation with B5 on the source views becomes the "explained rule" to
        pre-register. Target performance is NOT used to choose it; it is only reported.

PART 3  Batch-level shift alarm  (thresholds from SOURCE data only)
        Statistic 1: share of a batch whose feature centre distance exceeds the 95th
                     percentile of S-calibration (as in the feature-shift analysis).
        Statistic 2: total-variation distance between the batch's predicted-class shares
                     and the S-calibration predicted-class shares.
        Threshold  : 99th percentile of the statistic over random batches drawn from the
                     reliability-TRAINING original images (source).
        False-alarm check on held-out source (reliability-validation originals; S-test
        predictions for statistic 2). Alarm rate reported on random PP2020 batches.

RUN (project root):  python src/explore_pp2020.py
OUTPUT: results/exploratory_pp2020/  (every file is labelled EXPLORATORY)
"""

import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from common import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, FROZEN_ROOT, RESULTS_DIR, use_frozen_code

use_frozen_code()
from freeze_system import verify_frozen          # noqa: E402  (original code, read only)
from frozen_pipeline import load_frozen_head     # noqa: E402
from metrics import aurc, auroc_error, delta_aurc  # noqa: E402

SEEDS = (42, 123, 2026)
CLASSES = ["healthy", "scab", "rust"]
OLD = FROZEN_ROOT / "results"
OUT = RESULTS_DIR / "exploratory_pp2020"
BATCH_SIZES = (20, 50)
ALARM_QUANTILE = 99

METHODS = {"B1 MSP": "msp_score", "B2 Entropy": "entropy_score", "B3 Margin": "margin_score",
           "B4 Temp-scaled MSP": "temp_scaled_msp_score", "B5 Logits-only head": "head_logits_only",
           "A1 Features-only head": "head_features_only", "P1 Proposed head": "head_proposed"}


# ------------------------------------------------------------------ candidate rules
def candidate_scores(logits):
    """Zero-parameter scores from the 3 logits. Higher = less trustworthy. Fixed list."""
    z = np.sort(np.asarray(logits, dtype=np.float64), axis=1)          # ascending
    top, second, low = z[:, 2], z[:, 1], z[:, 0]
    logsumexp = top + np.log(np.exp(z - top[:, None]).sum(axis=1))
    probs = np.exp(np.asarray(logits) - logsumexp[:, None])
    return {
        "C0 MSP (reference)": 1.0 - probs.max(axis=1),
        "C1 negative logit margin (top - runner-up)": -(top - second),
        "C2 negative max logit": -top,
        "C3 energy (negative log-sum-exp)": -logsumexp,
        "C4 runner-up logit": second,
        "C5 lowest logit": low,
    }


def b5_scores_and_gradients(seed, logits):
    head, mean, std = load_frozen_head("logits_only", seed, "cpu")
    x = (torch.as_tensor(np.asarray(logits), dtype=torch.float32) - mean) / std
    x.requires_grad_(True)
    out = head(x).squeeze(-1)
    out.sum().backward()
    scores = torch.sigmoid(out).detach().numpy().astype(np.float64)
    gradient = x.grad.abs().mean(dim=0).numpy()          # per standardised logit
    return scores, gradient


# ------------------------------------------------------------------ data
def target_table(seed, labels):
    table = pd.read_csv(OLD / f"target_predictions_seed{seed}.csv", keep_default_na=False)
    true = np.array([CLASSES.index(labels[p]) for p in table["path"]])
    table["error"] = (table["predicted_index"].to_numpy() != true).astype(int)
    return table


def centre_distance(x, mean, std):
    return np.linalg.norm((x - mean) / std, axis=1) / np.sqrt(x.shape[1])


def batch_values(statistic, n_items, size, rng, draws=BOOTSTRAP_RESAMPLES):
    return np.array([statistic(rng.choice(n_items, size, replace=False)) for _ in range(draws)])


def tv_distance(predicted, reference_shares):
    shares = np.bincount(predicted, minlength=3) / len(predicted)
    return 0.5 * np.abs(shares - reference_shares).sum()


# ------------------------------------------------------------------ main
def main():
    verify_frozen()
    print("Freeze record verified (original system unchanged).")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(FROZEN_ROOT / "manifests" / "manifest.csv", keep_default_na=False)
    labels = manifest[manifest["dataset"] == "target"].set_index("path")["label"]

    part1, part2_corr, part2_grad, part2_target, part3, part2_magnitude = [], [], [], [], [], []
    for seed in SEEDS:
        table = target_table(seed, labels)
        errors = table["error"].to_numpy()
        clusters = table["cluster_id"].to_numpy()

        # sanity: the reloaded B5 reproduces the saved target scores
        b5_target, _ = b5_scores_and_gradients(seed, table[["logit_healthy", "logit_scab", "logit_rust"]].to_numpy())
        if not np.allclose(b5_target, table["head_logits_only"], atol=1e-5):
            sys.exit(f"Seed {seed}: reloaded B5 does not reproduce the saved target scores.")

        # ---------- PART 1
        predicted_scab = table["predicted_index"].to_numpy() == 1
        shortcut = predicted_scab.astype(float)
        part1.append({"seed": seed, "subset": "all images", "images": len(table), "errors": int(errors.sum()),
                      "method": "Shortcut: predicted scab = 1", "auroc_error": auroc_error(errors, shortcut),
                      "aurc": aurc(errors, shortcut), "delta_aurc_vs_msp": np.nan, "ci_low": np.nan, "ci_high": np.nan})
        sub = table[predicted_scab]
        e_sub, c_sub = sub["error"].to_numpy(), sub["cluster_id"].to_numpy()
        for name, column in METHODS.items():
            d, lo, hi = delta_aurc(e_sub, sub[column].to_numpy(), sub["msp_score"].to_numpy(), c_sub,
                                   BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
            part1.append({"seed": seed, "subset": "predicted scab only", "images": len(sub), "errors": int(e_sub.sum()),
                          "method": name, "auroc_error": auroc_error(e_sub, sub[column]), "aurc": aurc(e_sub, sub[column]),
                          "delta_aurc_vs_msp": d, "ci_low": lo, "ci_high": hi})

        # ---------- PART 2 (source views decide; target only reported)
        views = np.load(OLD / f"reliability_v2_features_seed{seed}.npz")
        source_logits = views["logits"].astype(np.float64)
        b5_source, gradient = b5_scores_and_gradients(seed, source_logits)
        part2_grad.append({"seed": seed, **{f"mean |dB5/d logit_{c}|": g for c, g in zip(CLASSES, gradient)}})
        for name, values in candidate_scores(source_logits).items():
            part2_corr.append({"seed": seed, "candidate": name, "spearman_with_B5_source_views": spearmanr(b5_source, values)[0]})
        target_logits = table[["logit_healthy", "logit_scab", "logit_rust"]].to_numpy()
        # mechanism check: logit magnitude (log-sum-exp) and margin, source vs target correct vs target wrong
        cal_logits = np.load(OLD / f"s_calibration_features_seed{seed}.npz")["logits"].astype(np.float64)
        for group, lg in (("S-calibration (source)", cal_logits),
                          ("PP2020 correct", target_logits[errors == 0]), ("PP2020 wrong", target_logits[errors == 1])):
            s = np.sort(lg, axis=1)
            lse = s[:, 2] + np.log(np.exp(s - s[:, 2:3]).sum(axis=1))
            part2_magnitude.append({"seed": seed, "group": group, "images": len(lg),
                                    "median log-sum-exp": np.median(lse), "median max logit": np.median(s[:, 2]),
                                    "median margin (top - runner-up)": np.median(s[:, 2] - s[:, 1])})
        for name, values in candidate_scores(target_logits).items():
            d, lo, hi = delta_aurc(errors, values, table["msp_score"].to_numpy(), clusters, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
            vs = values[predicted_scab]
            part2_target.append({"seed": seed, "candidate": name,
                                 "spearman_with_B5_target": spearmanr(table["head_logits_only"], values)[0],
                                 "aurc": aurc(errors, values), "delta_aurc_vs_msp": d, "ci_low": lo, "ci_high": hi,
                                 "auroc_error": auroc_error(errors, values),
                                 "auroc_error_within_predicted_scab": auroc_error(e_sub, vs)})

        # ---------- PART 3 (batch alarm)
        calib = np.load(OLD / f"s_calibration_features_seed{seed}.npz")
        cal_x = calib["features"].astype(np.float64)
        mean, std = cal_x.mean(axis=0), cal_x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        q95 = np.percentile(centre_distance(cal_x, mean, std), 95)
        cal_pred = calib["logits"].argmax(axis=1)
        cal_shares = np.bincount(cal_pred, minlength=3) / len(cal_pred)

        originals = views["view_types"] == "original"
        split = views["reliability_split"]
        null_train = originals & (split == "reliability_train")
        null_val = originals & (split == "reliability_val")
        feats = views["features"].astype(np.float64)
        beyond_train = centre_distance(feats[null_train], mean, std) > q95
        beyond_val = centre_distance(feats[null_val], mean, std) > q95
        pred_train = views["predicted_label_index"][null_train]
        pred_val = views["predicted_label_index"][null_val]
        stest_pred = pd.read_csv(OLD / f"s_test_outputs_seed{seed}.csv")["predicted_index"].to_numpy()

        tgt = np.load(OLD / f"target_features_seed{seed}.npz")
        beyond_tgt = centre_distance(tgt["features"].astype(np.float64), mean, std) > q95
        pred_tgt = table["predicted_index"].to_numpy()

        rng = np.random.default_rng(BOOTSTRAP_SEED + seed)
        for size in BATCH_SIZES:
            thr_dist = np.percentile(batch_values(lambda i: beyond_train[i].mean(), len(beyond_train), size, rng), ALARM_QUANTILE)
            thr_tv = np.percentile(batch_values(lambda i: tv_distance(pred_train[i], cal_shares), len(pred_train), size, rng), ALARM_QUANTILE)
            def rate(values, threshold):
                return float((values > threshold).mean())
            part3.append({
                "seed": seed, "batch_size": size,
                "S1 threshold (share beyond source p95)": thr_dist,
                "S1 false alarms, held-out source (rel-val originals)": rate(batch_values(lambda i: beyond_val[i].mean(), len(beyond_val), size, rng), thr_dist),
                "S1 alarm rate, PP2020 batches": rate(batch_values(lambda i: beyond_tgt[i].mean(), len(beyond_tgt), size, rng), thr_dist),
                "S2 threshold (TV distance of predicted shares)": thr_tv,
                "S2 false alarms, held-out source (rel-val originals)": rate(batch_values(lambda i: tv_distance(pred_val[i], cal_shares), len(pred_val), size, rng), thr_tv),
                "S2 false alarms, S-test": rate(batch_values(lambda i: tv_distance(stest_pred[i], cal_shares), len(stest_pred), size, rng), thr_tv),
                "S2 alarm rate, PP2020 batches": rate(batch_values(lambda i: tv_distance(pred_tgt[i], cal_shares), len(pred_tgt), size, rng), thr_tv),
                "share beyond p95: rel-val / PP2020": f"{beyond_val.mean():.3f} / {beyond_tgt.mean():.3f}",
            })

    # ------------------------------------------------------------------ write
    p1, corr, grad, p2t, p3, mag = map(pd.DataFrame, (part1, part2_corr, part2_grad, part2_target, part3, part2_magnitude))
    for frame, name in ((p1, "part1_shortcut_check"), (corr, "part2_source_correlations"), (grad, "part2_b5_sensitivity"),
                        (mag, "part2_logit_magnitude"),
                        (p2t, "part2_candidates_on_pp2020"), (p3, "part3_batch_alarm")):
        frame.to_csv(OUT / f"{name}.csv", index=False, lineterminator="\n")

    ranking = corr.groupby("candidate")["spearman_with_B5_source_views"].mean().sort_values(ascending=False)
    ranking_no_ref = ranking.drop("C0 MSP (reference)")
    selected = ranking_no_ref.index[0]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    text = [
        "EXPLORATORY ANALYSES ON THE ORIGINAL PP2020 OUTPUTS (not confirmatory; hypothesis generation only)",
        "=" * 100, "",
        "PART 1  Shortcut check: B5 inside the predicted-scab group",
        p1.round(4).to_string(index=False), "",
        "PART 2  What B5 learned (source reliability views decide)",
        "Mean absolute sensitivity of B5 to each standardised logit:",
        grad.round(4).to_string(index=False), "",
        "Seed-mean Spearman correlation with B5 on source views (selection criterion):",
        ranking.round(4).to_string(), "",
        f"SELECTED RULE (highest correlation with B5 on source, excluding the MSP reference): {selected}", "",
        "Logit magnitude and margin, source vs PP2020 (mechanism check):",
        mag.round(3).to_string(index=False), "",
        "All candidates on PP2020 (reported, NOT used for selection):",
        p2t.round(4).to_string(index=False), "",
        "PART 3  Batch alarm (thresholds from source only; 99th percentile of reliability-training batches)",
        p3.round(4).to_string(index=False), "",
    ]
    (OUT / "summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
