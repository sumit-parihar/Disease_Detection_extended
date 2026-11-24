"""
run_plantdoc_inference.py  -  Phase 8: the ONE-TIME frozen inference on PlantDoc.
It never reads a label.

Checks before running (any failure stops the script):
  1. no output exists yet (inference is run once; there is no --overwrite option);
  2. the sealed pre-registration has its recorded SHA-256;
  3. the original freeze record verifies (verify_frozen);
  4. the label-free manifest matches its sidecar hash and every image matches its hash.

Then, for seeds 42, 123 and 2026, every eligible image goes through the frozen classifier
and the three frozen heads with the original preparation. The score table is built with the
ORIGINAL build_table() (MSP, entropy, margin, temperature-scaled MSP with the frozen
temperature, the three heads) and one pre-registered addition: energy_score = -logsumexp(logits).

OUTPUT (results/plantdoc_inference/):
  plantdoc_predictions_seed{seed}.csv   one row per image, no labels
  plantdoc_features_seed{seed}.npz      576 features + logits (not tracked by Git)
  plantdoc_inference_hashes.txt         SHA-256 of inputs and outputs
NEXT: commit and push these files BEFORE any label is joined.
"""

import json
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import FROZEN_ROOT, MANIFEST_DIR, PREREG_DIR, PROJECT_ROOT, RESULTS_DIR, sha256_file, use_frozen_code

use_frozen_code()
from dataset import build_eval_transform                                    # noqa: E402  (original, read only)
from freeze_system import FREEZE_FILE, verify_frozen                       # noqa: E402
from frozen_pipeline import HEAD_VARIANTS, load_frozen_classifier, load_frozen_head, run_images  # noqa: E402
from run_target_inference import build_table                               # noqa: E402

SEEDS = (42, 123, 2026)
PREREG = PREREG_DIR / "plantdoc_preregistration.md"
PREREG_SHA = "813e22832c7b7c814184ea7814a4327d676eb08d4937da0525975efbc2cc8dc7"
MANIFEST = MANIFEST_DIR / "plantdoc_inference_manifest.csv"
SIDECAR = MANIFEST_DIR / "plantdoc_inference_manifest.csv.sha256"
SETTINGS = FROZEN_ROOT / "results" / "frozen_calibration_settings.json"
EXPECTED_COLUMNS = ["path", "sha256", "cluster_id", "cluster_size"]
EXPECTED_IMAGES = 265
BATCH = 32

OUT = RESULTS_DIR / "plantdoc_inference"
OUT_PREDICTIONS = {s: OUT / f"plantdoc_predictions_seed{s}.csv" for s in SEEDS}
OUT_FEATURES = {s: OUT / f"plantdoc_features_seed{s}.npz" for s in SEEDS}
OUT_HASHES = OUT / "plantdoc_inference_hashes.txt"


class LabelFreeImages(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.transform = build_eval_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(PROJECT_ROOT / self.paths[index]) as image:
            return self.transform(image.convert("RGB")), self.paths[index]


def relative(path):
    return path.relative_to(PROJECT_ROOT).as_posix()


def main():
    outputs = list(OUT_PREDICTIONS.values()) + list(OUT_FEATURES.values()) + [OUT_HASHES]
    if any(p.exists() for p in outputs):
        sys.exit("PlantDoc inference outputs already exist. Inference is run once.")
    if sha256_file(PREREG) != PREREG_SHA:
        sys.exit("The pre-registration differs from the sealed version.")
    verify_frozen()
    print("Pre-registration and freeze record verified.")
    if sha256_file(MANIFEST) != SIDECAR.read_text(encoding="utf-8").split()[0]:
        sys.exit("The label-free manifest has changed.")
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    if list(manifest.columns) != EXPECTED_COLUMNS or len(manifest) != EXPECTED_IMAGES or manifest["path"].duplicated().any():
        sys.exit("The manifest does not have the expected columns or number of distinct images.")
    for path, recorded in zip(manifest["path"], manifest["sha256"]):
        if sha256_file(PROJECT_ROOT / path) != recorded:
            sys.exit(f"Image differs from the manifest: {path}")
    print(f"All {len(manifest)} image hashes verified. No label is read.")

    OUT.mkdir(parents=True, exist_ok=True)
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(LabelFreeImages(manifest["path"]), batch_size=BATCH, shuffle=False, num_workers=0)

    tables = {}
    for seed in SEEDS:
        classifier = load_frozen_classifier(seed, device)
        heads = {variant: load_frozen_head(variant, seed, device) for variant in HEAD_VARIANTS}
        out = run_images(loader, classifier, heads, device)
        for name, array in [("logits", out["logits"])] + list(out["head_scores"].items()):
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite values in {name} for seed {seed}.")
        table = build_table(out, manifest, float(settings[str(seed)]["temperature"]))
        logits = out["logits"].astype(np.float64)
        top = logits.max(axis=1)
        table["energy_score"] = -(top + np.log(np.exp(logits - top[:, None]).sum(axis=1)))
        tables[seed] = table
        np.savez(OUT_FEATURES[seed], features=out["features"], logits=out["logits"], paths=out["paths"])

    for seed in SEEDS:
        tables[seed].to_csv(OUT_PREDICTIONS[seed], index=False, lineterminator="\n")

    lines = ["SHA-256 checksums for the one-time PlantDoc inference (no labels were read)", "", "INPUTS:"]
    lines += [f"{sha256_file(p)}  {label}" for p, label in [(PREREG, relative(PREREG)), (MANIFEST, relative(MANIFEST)),
                                                             (FREEZE_FILE, "original: results/frozen_system_manifest.json")]]
    lines += ["", "OUTPUT FILES:"] + [f"{sha256_file(p)}  {relative(p)}" for p in outputs if p != OUT_HASHES]
    OUT_HASHES.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    print("\nPREDICTED CLASS COUNTS (no labels involved)")
    for seed in SEEDS:
        print(f"  seed {seed}: {tables[seed]['predicted_class'].value_counts().to_dict()}")
    print("NEXT: commit and push the prediction files and the hash file BEFORE any label is joined.")


if __name__ == "__main__":
    main()
