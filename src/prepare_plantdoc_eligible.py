"""
prepare_plantdoc_eligible.py  -  Phase 4: eligible set, duplicate clusters, label-free copies.

1. Applies the exclusions (in this order, each counted):
     a. visual label check: decision "exclude" in plantdoc_label_check.csv
     b. overlap with PlantVillage / PP2020: exact SHA-256 copy, or a reviewed pHash pair
        marked duplicate/unsure (none were found, but the rule is applied)
2. Size rule: at least MIN_IMAGES_PER_CLASS eligible images per class, otherwise stop.
3. Duplicate clusters inside PlantDoc: pairs marked duplicate/unsure and exact copies are
   joined with union-find; every image gets cluster_id and cluster_size.
4. Label-free copies: each eligible file is copied to data/plantdoc_staged/pd_XXXX.jpg.
   Numbers follow the SHA-256 order (not the folder order), so a name reveals nothing about
   the class. Every copy is checked byte for byte.
5. Writes
     manifests/plantdoc_eligible.csv                 staged name -> original file, label, cluster
     manifests/plantdoc_inference_manifest.csv       path, sha256, cluster_id, cluster_size (NO labels)
     manifests/plantdoc_inference_manifest.csv.sha256
     manifests/plantdoc_eligible_summary.txt

The inference script will read ONLY the label-free manifest.
RUN (project root):  python src/prepare_plantdoc_eligible.py      (refuses to overwrite)
"""

import shutil
import sys

import pandas as pd

from common import MANIFEST_DIR, MIN_IMAGES_PER_CLASS, PLANTDOC_STAGED_DIR, PROJECT_ROOT, sha256_file

DUPLICATE_DECISIONS = {"duplicate", "unsure"}
OUT_ELIGIBLE = MANIFEST_DIR / "plantdoc_eligible.csv"
OUT_MANIFEST = MANIFEST_DIR / "plantdoc_inference_manifest.csv"
OUT_SIDECAR = MANIFEST_DIR / "plantdoc_inference_manifest.csv.sha256"
OUT_SUMMARY = MANIFEST_DIR / "plantdoc_eligible_summary.txt"


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def main():
    if OUT_MANIFEST.exists() or OUT_ELIGIBLE.exists():
        sys.exit("Eligible set already written. It is created once.")
    manifest = pd.read_csv(MANIFEST_DIR / "plantdoc_manifest.csv", keep_default_na=False)
    labels_check = pd.read_csv(MANIFEST_DIR / "plantdoc_label_check.csv", keep_default_na=False)
    review = pd.read_csv(MANIFEST_DIR / "plantdoc_pair_review.csv", keep_default_na=False)
    pairs = pd.read_csv(MANIFEST_DIR / "plantdoc_phash_pairs.csv", keep_default_na=False)
    exact = pd.read_csv(MANIFEST_DIR / "plantdoc_exact_duplicates.csv", keep_default_na=False)

    if (labels_check["decision"] == "").any() or not set(labels_check["decision"]) <= {"ok", "exclude"}:
        sys.exit("The label check is incomplete or has invalid decisions.")
    if (review["decision"] == "").any() or not set(review["decision"]) <= {"duplicate", "different", "unsure"}:
        sys.exit("The pair review is incomplete or has invalid decisions.")
    if not manifest["decode_ok"].astype(str).eq("True").all():
        sys.exit("Undecodable files are present; handle them first.")

    pairs = pairs.merge(review[["pair_id", "decision"]], on="pair_id")
    excluded = {}
    for file in labels_check.loc[labels_check["decision"] == "exclude", "file"]:
        excluded[file] = "visual label check"
    overlap = pairs[(pairs["scope"] != "within_plantdoc") & pairs["decision"].isin(DUPLICATE_DECISIONS)]
    for file in overlap["path_a"]:
        excluded.setdefault(file, "overlap with PlantVillage/PP2020 (reviewed pair)")
    for file in exact.loc[exact["scope"] == "plantdoc_vs_original", "files"]:
        excluded.setdefault(file, "exact copy of a PlantVillage/PP2020 image")

    eligible = manifest[~manifest["file"].isin(excluded)].copy()
    counts = eligible["label"].value_counts()
    if (counts.reindex(["healthy", "scab", "rust"]).fillna(0) < MIN_IMAGES_PER_CLASS).any():
        sys.exit(f"Size rule failed: {counts.to_dict()} (minimum {MIN_IMAGES_PER_CLASS} per class).")

    # clusters inside PlantDoc
    uf = UnionFind(eligible["file"])
    within = pairs[(pairs["scope"] == "within_plantdoc") & pairs["decision"].isin(DUPLICATE_DECISIONS)]
    for a, b in zip(within["path_a"], within["path_b"]):
        if a in uf.parent and b in uf.parent:
            uf.union(a, b)
    for files in exact.loc[exact["scope"] == "within_plantdoc", "files"]:
        members = [f for f in files.split(";") if f in uf.parent]
        for other in members[1:]:
            uf.union(members[0], other)

    # label-free names in SHA-256 order
    eligible = eligible.sort_values("sha256").reset_index(drop=True)
    eligible["staged_path"] = [f"data/plantdoc_staged/pd_{i:04d}.jpg" for i in range(1, len(eligible) + 1)]
    root_to_cluster, cluster_ids = {}, []
    for file in eligible["file"]:
        root = uf.find(file)
        root_to_cluster.setdefault(root, f"p{len(root_to_cluster) + 1:04d}")
        cluster_ids.append(root_to_cluster[root])
    eligible["cluster_id"] = cluster_ids
    eligible["cluster_size"] = eligible.groupby("cluster_id")["file"].transform("size")

    PLANTDOC_STAGED_DIR.mkdir(parents=True, exist_ok=True)
    for source, target, sha in zip(eligible["file"], eligible["staged_path"], eligible["sha256"]):
        destination = PROJECT_ROOT / target
        if destination.exists():
            sys.exit(f"{destination} already exists.")
        shutil.copyfile(PROJECT_ROOT / source, destination)
        if sha256_file(destination) != sha:
            sys.exit(f"Copy differs from the original: {source}")

    eligible[["staged_path", "image_no", "file", "label", "original_split", "sha256", "cluster_id", "cluster_size"]] \
        .to_csv(OUT_ELIGIBLE, index=False, lineterminator="\n")
    label_free = eligible.rename(columns={"staged_path": "path"})[["path", "sha256", "cluster_id", "cluster_size"]]
    label_free.to_csv(OUT_MANIFEST, index=False, lineterminator="\n")
    OUT_SIDECAR.write_text(f"{sha256_file(OUT_MANIFEST)}  {OUT_MANIFEST.name}\n", encoding="utf-8", newline="\n")

    removed = pd.Series(excluded).value_counts()
    removed_by_class = manifest[manifest["file"].isin(excluded)].groupby("label").size()
    multi = eligible[eligible["cluster_size"] > 1]
    text = [
        "PLANTDOC ELIGIBLE SET",
        f"audited images                 : {len(manifest)}  {manifest['label'].value_counts().to_dict()}",
        f"removed, by rule               : {removed.to_dict() if len(removed) else 'none'}",
        f"removed, by class              : {removed_by_class.to_dict()}",
        f"eligible images                : {len(eligible)}  {eligible['label'].value_counts().to_dict()}",
        f"size rule (>= {MIN_IMAGES_PER_CLASS} per class)       : passed",
        f"clusters                       : {eligible['cluster_id'].nunique()} "
        f"({multi['cluster_id'].nunique()} clusters of size > 1 containing {len(multi)} images)",
        f"label-free manifest SHA-256    : {sha256_file(OUT_MANIFEST)}",
        "staged names follow SHA-256 order; every copy verified byte for byte",
    ]
    OUT_SUMMARY.write_text("\n".join(text) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
