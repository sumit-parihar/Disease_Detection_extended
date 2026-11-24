"""
measure_plantdoc_shift.py  -  Phase 5: how far PlantDoc lies from the source, BEFORE inference.

Only the frozen backbone's 576 image numbers are computed (classifier.extract_features).
The output layer is never run, so no logits, predictions or labels exist after this step.
Each image is standardised with the S-calibration mean and SD of the same classifier seed,
exactly as in the original feature-shift analysis, and its centre distance is
||z|| / sqrt(576)  (about 1 for a typical source image).

Images are prepared as in the original evaluation: convert to RGB, resize to 224 x 224,
ImageNet normalisation, no EXIF rotation, no augmentation.

OUTPUT (results/plantdoc_shift/): centre_distances.csv (path, seed, centre_distance), summary.txt
RUN (project root):  python src/measure_plantdoc_shift.py
"""

import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import FROZEN_ROOT, MANIFEST_DIR, PROJECT_ROOT, RESULTS_DIR, sha256_file, use_frozen_code

use_frozen_code()
from dataset import build_eval_transform               # noqa: E402  (original code, read only)
from freeze_system import verify_frozen                # noqa: E402
from frozen_pipeline import load_frozen_classifier     # noqa: E402

SEEDS = (42, 123, 2026)
MANIFEST = MANIFEST_DIR / "plantdoc_inference_manifest.csv"
SIDECAR = MANIFEST_DIR / "plantdoc_inference_manifest.csv.sha256"
OUT = RESULTS_DIR / "plantdoc_shift"
PP2020_MEDIAN = {42: 1.85, 123: 1.85, 2026: 1.81}      # Table 25 of the manuscript
BATCH = 32


@torch.inference_mode()
def main():
    verify_frozen()
    print("Freeze record verified.")
    if sha256_file(MANIFEST) != SIDECAR.read_text(encoding="utf-8").split()[0]:
        sys.exit("The label-free manifest does not match its recorded hash.")
    manifest = pd.read_csv(MANIFEST)
    for path, sha in zip(manifest["path"], manifest["sha256"]):
        if sha256_file(PROJECT_ROOT / path) != sha:
            sys.exit(f"Image changed: {path}")
    OUT.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = build_eval_transform()
    rows, summary = [], []
    for seed in SEEDS:
        model = load_frozen_classifier(seed, device)
        features = []
        for start in range(0, len(manifest), BATCH):
            batch = torch.stack([transform(Image.open(PROJECT_ROOT / p).convert("RGB"))
                                 for p in manifest["path"].iloc[start:start + BATCH]])
            features.append(model.extract_features(batch.to(device)).cpu().numpy().astype(np.float64))
        features = np.concatenate(features)

        source = np.load(FROZEN_ROOT / "results" / f"s_calibration_features_seed{seed}.npz")["features"].astype(np.float64)
        mean, std = source.mean(axis=0), source.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        dim = source.shape[1]
        source_distance = np.linalg.norm((source - mean) / std, axis=1) / np.sqrt(dim)
        distance = np.linalg.norm((features - mean) / std, axis=1) / np.sqrt(dim)
        rows += [{"path": p, "seed": seed, "centre_distance": d} for p, d in zip(manifest["path"], distance)]
        summary.append({"seed": seed, "images": len(distance),
                        "source median": np.median(source_distance),
                        "PlantDoc median": np.median(distance),
                        "PP2020 median (manuscript)": PP2020_MEDIAN[seed],
                        "PlantDoc beyond source p95": float((distance > np.percentile(source_distance, 95)).mean())})
        del features

    pd.DataFrame(rows).to_csv(OUT / "centre_distances.csv", index=False, lineterminator="\n")
    table = pd.DataFrame(summary)
    text = ["PLANTDOC FEATURE SHIFT (backbone features only; no logits or predictions were computed)",
            table.round(3).to_string(index=False), "",
            "Interpretation is fixed in advance: PlantDoc is a field set; the distance is reported for information",
            "and to place PlantDoc relative to the source (about 1.0) and PP2020 (about 1.8)."]
    (OUT / "summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
