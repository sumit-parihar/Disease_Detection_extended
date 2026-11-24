"""
explore_regimes.py  -  EXPLORATORY (after the PlantDoc results were known).
Question: why did the logits-only head B5 beat MSP on PP2020 but not on PlantDoc?

Nothing is trained, tuned or changed; only saved outputs of both one-time evaluations are read.

A. Error regime: how predictions and errors are spread over the predicted classes.
B. Where the ranking comes from: a method's AUROC-error can come from
     - BETWEEN predicted classes (it gives whole predicted classes higher or lower scores), and
     - WITHIN a predicted class (it orders images of the same predicted class).
   "between" = AUROC-error of the score replaced by its predicted-class mean;
   "within"  = AUROC-error inside each predicted class with >= 10 errors and >= 10 correct.
C. Logit magnitude (log-sum-exp) of correct vs wrong predictions in both datasets.

RUN (project root):  python src/explore_regimes.py      OUTPUT: results/exploratory_regimes/
"""

import numpy as np
import pandas as pd

from common import FROZEN_ROOT, MANIFEST_DIR, RESULTS_DIR, use_frozen_code

use_frozen_code()
from metrics import auroc_error  # noqa: E402  (original code, read only)

SEEDS = (42, 123, 2026)
CLASSES = ["healthy", "scab", "rust"]
OUT = RESULTS_DIR / "exploratory_regimes"
SCORES = {"MSP": "msp_score", "B5": "head_logits_only", "Energy": "energy", "P1": "head_proposed"}


def load(dataset, seed):
    if dataset == "PP2020":
        table = pd.read_csv(FROZEN_ROOT / "results" / f"target_predictions_seed{seed}.csv", keep_default_na=False)
        manifest = pd.read_csv(FROZEN_ROOT / "manifests" / "manifest.csv", keep_default_na=False)
        labels = manifest[manifest["dataset"] == "target"].set_index("path")["label"]
    else:
        table = pd.read_csv(RESULTS_DIR / "plantdoc_inference" / f"plantdoc_predictions_seed{seed}.csv", keep_default_na=False)
        labels = pd.read_csv(MANIFEST_DIR / "plantdoc_eligible.csv", keep_default_na=False).set_index("staged_path")["label"]
    table["true_index"] = [CLASSES.index(labels[p]) for p in table["path"]]
    table["error"] = (table["predicted_index"] != table["true_index"]).astype(int)
    logits = table[["logit_healthy", "logit_scab", "logit_rust"]].to_numpy(dtype=np.float64)
    top = logits.max(axis=1)
    table["lse"] = top + np.log(np.exp(logits - top[:, None]).sum(axis=1))
    table["energy"] = -table["lse"]
    return table


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    regime, decomposition, magnitude = [], [], []
    for dataset in ("PP2020", "PlantDoc"):
        for seed in SEEDS:
            t = load(dataset, seed)
            e = t["error"].to_numpy()
            for k, c in enumerate(CLASSES):
                sub = t[t["predicted_index"] == k]
                regime.append({"dataset": dataset, "seed": seed, "predicted": c, "images": len(sub),
                               "share_of_images": len(sub) / len(t), "errors": int(sub["error"].sum()),
                               "share_of_all_errors": sub["error"].sum() / e.sum(),
                               "error_rate": sub["error"].mean() if len(sub) else np.nan,
                               **{f"mean {s}": sub[col].mean() if len(sub) else np.nan for s, col in SCORES.items()}})
            for s, col in SCORES.items():
                x = t[col].to_numpy()
                class_mean = t.groupby("predicted_index")[col].transform("mean").to_numpy()
                within = []
                for k in range(3):
                    mask = t["predicted_index"].to_numpy() == k
                    if e[mask].sum() >= 10 and (1 - e[mask]).sum() >= 10:
                        within.append((CLASSES[k], auroc_error(e[mask], x[mask]), int(mask.sum())))
                decomposition.append({"dataset": dataset, "seed": seed, "score": s,
                                      "AUROC overall": auroc_error(e, x),
                                      "AUROC between classes only": auroc_error(e, class_mean),
                                      "AUROC within classes (weighted)": (sum(a * n for _, a, n in within) / sum(n for *_, n in within)) if within else np.nan,
                                      "within detail": "; ".join(f"{c}:{a:.3f}(n={n})" for c, a, n in within)})
            for label, mask in (("correct", e == 0), ("wrong", e == 1)):
                magnitude.append({"dataset": dataset, "seed": seed, "group": label, "images": int(mask.sum()),
                                  "median log-sum-exp": float(np.median(t["lse"][mask]))})

    reg, dec, mag = map(pd.DataFrame, (regime, decomposition, magnitude))
    for frame, name in ((reg, "error_regime"), (dec, "auroc_decomposition"), (mag, "logit_magnitude")):
        frame.to_csv(OUT / f"{name}.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 80)
    text = ["EXPLORATORY: WHY B5 HELPED ON PP2020 BUT NOT ON PLANTDOC (decided after the PlantDoc results)", "=" * 95, "",
            "A. ERROR REGIME BY PREDICTED CLASS", reg.round(3).to_string(index=False), "",
            "B. WHERE THE RANKING COMES FROM (AUROC-error; 0.5 = chance)", dec.round(3).to_string(index=False), "",
            "C. LOGIT MAGNITUDE OF CORRECT VS WRONG PREDICTIONS", mag.round(3).to_string(index=False), ""]
    (OUT / "summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
