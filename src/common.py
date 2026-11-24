"""
common.py  -  Paths and fixed values for the extended (second-target) study.

The frozen system lives in the ORIGINAL project folder and is only READ from there:
classifiers, failure heads, temperatures, cut-offs, metrics code. Nothing in that
folder is ever written by this project. Every step that uses the frozen system first
calls verify_frozen() from the original freeze_system.py.
"""

import hashlib
import sys
from pathlib import Path

# ---------------------------------------------------------------- folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # D:\capstone-paper-extended
FROZEN_ROOT = Path(r"D:\Capstone Project\Capstone Project")    # original study (read-only)
FROZEN_SRC = FROZEN_ROOT / "src"

DATA_DIR = PROJECT_ROOT / "data"                     # not tracked by Git
PLANTDOC_GIT_DIR = DATA_DIR / "plantdoc_raw"         # sparse clone of the original repo (no checkout)
PLANTDOC_FILES_DIR = DATA_DIR / "plantdoc_files"     # exact blob bytes, never edited
PLANTDOC_STAGED_DIR = DATA_DIR / "plantdoc_staged"   # label-free copies for inference (pd_0001.jpg ...)

MANIFEST_DIR = PROJECT_ROOT / "manifests"
RESULTS_DIR = PROJECT_ROOT / "results"
PREREG_DIR = PROJECT_ROOT / "prereg"

# ---------------------------------------------------------------- PlantDoc source
PLANTDOC_URL = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
PLANTDOC_LICENCE = "CC BY 4.0 (LICENSE.txt in the original repository)"
# PlantDoc folder name -> class name of the frozen classifier (order: healthy 0, scab 1, rust 2)
PLANTDOC_FOLDERS = {
    "Apple leaf": "healthy",
    "Apple Scab Leaf": "scab",
    "Apple rust leaf": "rust",
}
PLANTDOC_SPLITS = ("train", "test")      # both original folders are used; the split is recorded

# ---------------------------------------------------------------- fixed study rules
MIN_IMAGES_PER_CLASS = 60     # after all exclusions; fixed before the audit
MIN_ERRORS_PER_SEED = 50      # fewer errors in a seed = "underpowered" (reported as such)
PHASH_MAX_REVIEW_DISTANCE = 10
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 2026


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def use_frozen_code():
    """Make the original project's src/ importable (read-only use)."""
    if str(FROZEN_SRC) not in sys.path:
        sys.path.insert(0, str(FROZEN_SRC))


def make_folders():
    for folder in (DATA_DIR, MANIFEST_DIR, RESULTS_DIR, PREREG_DIR):
        folder.mkdir(parents=True, exist_ok=True)
