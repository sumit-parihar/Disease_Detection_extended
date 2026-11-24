"""
acquire_plantdoc.py  -  Phase 1: take the three apple folders of PlantDoc from the ORIGINAL
GitHub repository and store their exact bytes, with a provenance record.

WHY NOT A NORMAL CHECKOUT
-------------------------
Some PlantDoc file names contain characters that Windows forbids (for example '?'), so a
normal checkout fails. Instead, each file is read from the Git object store (git cat-file)
and written under a safe name. The bytes are identical: the script checks that the Git blob
hash of every written file equals the blob id in the repository tree.

PREREQUISITE (done once):
    git clone --filter=blob:none --no-checkout --sparse <PLANTDOC_URL> data/plantdoc_raw

OUTPUT
------
  data/plantdoc_files/<split>/<class folder>/<safe name>     exact bytes (not tracked by Git)
  manifests/plantdoc_acquisition.csv   original path, blob id, saved path, SHA-256, size, folder label
  manifests/plantdoc_acquisition.txt   repository URL, commit, commit date, download date, licence

RUN (project root):  python src/acquire_plantdoc.py
It refuses to overwrite an existing acquisition record.
"""

import datetime
import hashlib
import re
import subprocess
import sys
from urllib.parse import unquote

import pandas as pd

from common import (MANIFEST_DIR, PLANTDOC_FILES_DIR, PLANTDOC_FOLDERS, PLANTDOC_GIT_DIR,
                    PLANTDOC_LICENCE, PLANTDOC_SPLITS, PLANTDOC_URL, PROJECT_ROOT, make_folders, sha256_file)

OUT_CSV = MANIFEST_DIR / "plantdoc_acquisition.csv"
OUT_TXT = MANIFEST_DIR / "plantdoc_acquisition.txt"


def git(*args, binary=False):
    result = subprocess.run(["git", "-C", str(PLANTDOC_GIT_DIR), *args], capture_output=True, check=True)
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def git_blob_id(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def safe_name(original, index):
    """Readable, Windows-safe, unique file name. The original name is kept in the record."""
    name = unquote(original)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f%&=]', "_", name)
    stem, dot, ext = name.rpartition(".")
    stem = stem[:80] if dot else name[:80]
    return f"{index:04d}_{stem}.{ext.lower() if dot else 'jpg'}"


def main():
    if OUT_CSV.exists():
        sys.exit(f"{OUT_CSV.name} already exists. The acquisition is done once.")
    make_folders()
    commit = git("rev-parse", "HEAD")
    commit_date = git("log", "-1", "--format=%cI")
    paths = [f"{split}/{folder}" for split in PLANTDOC_SPLITS for folder in PLANTDOC_FOLDERS]
    tree = git("ls-tree", "-r", "--full-tree", "HEAD", "--", *paths).splitlines()

    rows = []
    for index, line in enumerate(sorted(tree, key=lambda l: l.split("\t", 1)[1]), start=1):
        meta, original_path = line.split("\t", 1)
        _, kind, blob = meta.split()
        assert kind == "blob"
        split, folder, original_name = original_path.split("/", 2)
        data = git("cat-file", "blob", blob, binary=True)
        if git_blob_id(data) != blob:
            sys.exit(f"Blob check failed for {original_path}")
        target = PLANTDOC_FILES_DIR / split / folder / safe_name(original_name, index)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            sys.exit(f"{target} already exists.")
        target.write_bytes(data)
        if git_blob_id(target.read_bytes()) != blob:
            sys.exit(f"Written file differs from the blob: {target}")
        rows.append({
            "original_path": original_path,
            "git_blob": blob,
            "file": target.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(target),
            "bytes": len(data),
            "original_split": split,
            "original_folder": folder,
            "label": PLANTDOC_FOLDERS[folder],
        })

    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False, lineterminator="\n")
    counts = table.groupby(["original_split", "label"]).size().unstack(fill_value=0)
    OUT_TXT.write_text(
        "PLANTDOC ACQUISITION RECORD\n"
        f"repository      : {PLANTDOC_URL}\n"
        f"commit          : {commit}\n"
        f"commit date     : {commit_date}\n"
        f"downloaded (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n"
        f"licence         : {PLANTDOC_LICENCE}\n"
        f"folders taken   : {', '.join(paths)}\n"
        "method          : sparse blobless clone; bytes read with 'git cat-file' and checked against the blob ids\n"
        f"files           : {len(table)}\n"
        f"record SHA-256  : {sha256_file(OUT_CSV)}  ({OUT_CSV.name})\n\n"
        f"counts by original split and label:\n{counts.to_string()}\n"
        f"totals: {table['label'].value_counts().to_dict()}\n",
        encoding="utf-8", newline="\n")
    print(OUT_TXT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
