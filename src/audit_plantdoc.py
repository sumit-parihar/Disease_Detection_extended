"""
audit_plantdoc.py  -  Phase 3 (audit) and preparation of the duplicate review.

WHAT IT DOES
------------
1. Fully decodes every acquired PlantDoc image and records size, colour mode, channels,
   file format, EXIF orientation tag, SHA-256 and 64-bit pHash. The pHash is computed
   exactly as in the original audit (imagehash.phash on the decoded image, no rotation),
   so distances to the original manifest are directly comparable.
2. Writes a sorted manifest with a SHA-256 sidecar (refuses to overwrite).
3. Finds
     - exact copies inside PlantDoc (same SHA-256),
     - exact copies of a PlantVillage or PP2020 image (SHA-256 found in the original
       manifest.csv, which is only READ),
     - every pHash pair at distance 0..10 bits inside PlantDoc and between PlantDoc and
       the 4,371 audited PlantVillage/PP2020 images.
4. Draws contact sheets for every pHash pair and writes a review file with an empty
   "decision" column (duplicate / different / unsure; unsure counts as duplicate).
5. Draws label sheets showing all PlantDoc images per class, for the visual label check.

Contact sheets contain PP2020 thumbnails, which may not be redistributed, so the sheet
folders are not tracked by Git.

RUN (project root):  python src/audit_plantdoc.py
"""

import sys

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from common import (FROZEN_ROOT, MANIFEST_DIR, PHASH_MAX_REVIEW_DISTANCE, PROJECT_ROOT, make_folders,
                    sha256_file)

ACQUISITION = MANIFEST_DIR / "plantdoc_acquisition.csv"
ORIGINAL_MANIFEST = FROZEN_ROOT / "manifests" / "manifest.csv"
ORIGINAL_MANIFEST_SHA = "d0aabb245d6993e840d43ab2cad4f798a3d3360825a7b53e8a69c420b7268b73"

OUT_MANIFEST = MANIFEST_DIR / "plantdoc_manifest.csv"
OUT_SIDECAR = MANIFEST_DIR / "plantdoc_manifest.csv.sha256"
OUT_EXACT = MANIFEST_DIR / "plantdoc_exact_duplicates.csv"
OUT_PAIRS = MANIFEST_DIR / "plantdoc_phash_pairs.csv"
OUT_REVIEW = MANIFEST_DIR / "plantdoc_pair_review.csv"
OUT_SUMMARY = MANIFEST_DIR / "plantdoc_audit_summary.txt"
PAIR_SHEETS = MANIFEST_DIR / "plantdoc_review"
LABEL_SHEETS = MANIFEST_DIR / "plantdoc_label_sheets"

CHANNELS_BY_MODE = {"1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4, "YCbCr": 3, "LA": 2, "I": 1, "F": 1, "I;16": 1}
THUMB = 256
PAIRS_PER_SHEET = 6
LABEL_GRID = (6, 5)          # columns x rows per label sheet
LABEL_THUMB = 200


# ------------------------------------------------------------------ inspection
def inspect_image(path):
    info = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path), "width": None, "height": None,
            "mode": None, "channels": None, "format": None, "exif_orientation": None,
            "phash": None, "decode_ok": False, "error": ""}
    try:
        with Image.open(path) as image:
            info["format"] = image.format
            image.load()
            info["width"], info["height"] = image.size
            info["mode"] = image.mode
            info["channels"] = CHANNELS_BY_MODE.get(image.mode)
            try:
                info["exif_orientation"] = image.getexif().get(0x0112)
            except Exception:
                info["exif_orientation"] = None
            info["phash"] = str(imagehash.phash(image))
            info["decode_ok"] = True
    except Exception as error:
        info["error"] = type(error).__name__
    return info


def popcount(values):
    return np.unpackbits(np.atleast_1d(values).view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def hashes_to_int(series):
    return np.array([int(h, 16) for h in series], dtype=np.uint64)


# ------------------------------------------------------------------ drawing
def open_any(dataset, relative_path):
    root = PROJECT_ROOT if dataset == "plantdoc" else FROZEN_ROOT
    with Image.open(root / relative_path) as image:
        return image.convert("RGB")


def pixel_difference(a, b):
    small_a = np.asarray(a.resize((64, 64)), dtype=np.float32)
    small_b = np.asarray(b.resize((64, 64)), dtype=np.float32)
    return float(np.abs(small_a - small_b).mean())


def thumbnail(image, size):
    copy = image.copy()
    copy.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(copy, ((size - copy.width) // 2, (size - copy.height) // 2))
    return canvas


def draw_pair_sheet(rows, path):
    text_w = 360
    sheet = Image.new("RGB", (2 * THUMB + text_w + 24, len(rows) * (THUMB + 8) + 8), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        y = 8 + i * (THUMB + 8)
        sheet.paste(thumbnail(row["_img_a"], THUMB), (8, y))
        sheet.paste(thumbnail(row["_img_b"], THUMB), (16 + THUMB, y))
        lines = [f"pair {row['pair_id']}   pHash distance {row['phash_distance']}",
                 f"scope: {row['scope']}",
                 f"A: {row['dataset_a']} / {row['label_a']}",
                 f"B: {row['dataset_b']} / {row['label_b']}",
                 f"pixel_diff: {row['pixel_diff']:.1f}",
                 f"A size {row['size_a']}", f"B size {row['size_b']}"]
        for j, line in enumerate(lines):
            draw.text((24 + 2 * THUMB, y + 6 + 16 * j), line, fill="black")
    sheet.save(path)


def draw_label_sheets(manifest):
    LABEL_SHEETS.mkdir(parents=True, exist_ok=True)
    cols, rows = LABEL_GRID
    per_sheet = cols * rows
    listing = []
    for label, group in manifest.groupby("label"):
        group = group.reset_index(drop=True)
        for start in range(0, len(group), per_sheet):
            chunk = group.iloc[start:start + per_sheet]
            sheet = Image.new("RGB", (cols * (LABEL_THUMB + 8) + 8, rows * (LABEL_THUMB + 24) + 8), "white")
            draw = ImageDraw.Draw(sheet)
            for k, (_, row) in enumerate(chunk.iterrows()):
                x = 8 + (k % cols) * (LABEL_THUMB + 8)
                y = 8 + (k // cols) * (LABEL_THUMB + 24)
                if row["decode_ok"]:
                    sheet.paste(thumbnail(open_any("plantdoc", row["file"]), LABEL_THUMB), (x, y))
                draw.text((x, y + LABEL_THUMB + 4), f"#{row['image_no']} {label}", fill="black")
            name = f"{label}_{start // per_sheet + 1:02d}.png"
            sheet.save(LABEL_SHEETS / name)
            listing += [{"sheet": name, "image_no": r["image_no"], "file": r["file"], "label": label,
                         "decision": "", "note": ""} for _, r in chunk.iterrows()]
    return pd.DataFrame(listing)


# ------------------------------------------------------------------ main
def main():
    if OUT_MANIFEST.exists():
        sys.exit(f"{OUT_MANIFEST.name} already exists. The audit manifest is written once.")
    make_folders()
    if sha256_file(ORIGINAL_MANIFEST) != ORIGINAL_MANIFEST_SHA:
        sys.exit("The original manifest.csv does not match its recorded SHA-256.")

    acquisition = pd.read_csv(ACQUISITION)
    records = []
    for _, row in acquisition.sort_values("file").iterrows():
        info = inspect_image(PROJECT_ROOT / row["file"])
        if info["sha256"] != row["sha256"]:
            sys.exit(f"File changed since acquisition: {row['file']}")
        records.append({"file": row["file"], "original_path": row["original_path"], "label": row["label"],
                        "original_split": row["original_split"], **info})
    manifest = pd.DataFrame(records)
    manifest.insert(0, "image_no", range(1, len(manifest) + 1))
    manifest.to_csv(OUT_MANIFEST, index=False, lineterminator="\n")
    OUT_SIDECAR.write_text(f"{sha256_file(OUT_MANIFEST)}  {OUT_MANIFEST.name}\n", encoding="utf-8", newline="\n")

    ok = manifest[manifest["decode_ok"]].copy()
    original = pd.read_csv(ORIGINAL_MANIFEST)
    original = original[original["decode_ok"]].copy()

    # ---- exact copies
    exact_rows = []
    for sha, group in ok.groupby("sha256"):
        if len(group) > 1:
            exact_rows.append({"scope": "within_plantdoc", "sha256": sha, "count": len(group),
                               "labels": ";".join(sorted(group["label"].unique())),
                               "files": ";".join(group["file"]), "matches": ""})
    original_by_sha = original.groupby("sha256")
    for _, row in ok.iterrows():
        if row["sha256"] in original_by_sha.groups:
            match = original_by_sha.get_group(row["sha256"])
            exact_rows.append({"scope": "plantdoc_vs_original", "sha256": row["sha256"], "count": 1 + len(match),
                               "labels": ";".join([row["label"]] + [f"{d}:{l}" for d, l in zip(match["dataset"], match["label"])]),
                               "files": row["file"], "matches": ";".join(match["path"])})
    exact = pd.DataFrame(exact_rows, columns=["scope", "sha256", "count", "labels", "files", "matches"])
    exact.to_csv(OUT_EXACT, index=False, lineterminator="\n")

    # ---- pHash pairs (exact SHA copies inside PlantDoc are still listed, flagged)
    pd_hash = hashes_to_int(ok["phash"])
    og_hash = hashes_to_int(original["phash"])
    pairs = []
    ok_rows = ok.reset_index(drop=True)
    og_rows = original.reset_index(drop=True)
    for i in range(len(ok_rows)):
        if i + 1 < len(ok_rows):
            within = popcount(np.bitwise_xor(pd_hash[i], pd_hash[i + 1:]))
            for offset in np.flatnonzero(within <= PHASH_MAX_REVIEW_DISTANCE):
                j = i + 1 + offset
                pairs.append(("within_plantdoc", ok_rows.loc[i], "plantdoc", ok_rows.loc[j], "plantdoc", int(within[offset])))
        cross = popcount(np.bitwise_xor(pd_hash[i], og_hash))
        for j in np.flatnonzero(cross <= PHASH_MAX_REVIEW_DISTANCE):
            other = og_rows.loc[j]
            pairs.append((f"plantdoc_vs_{other['dataset']}", ok_rows.loc[i], "plantdoc", other, other["dataset"], int(cross[j])))

    pair_rows = []
    for pair_id, (scope, a, da, b, db, distance) in enumerate(sorted(pairs, key=lambda p: (p[5], p[0])), start=1):
        path_a = a["file"]
        path_b = b["file"] if db == "plantdoc" else b["path"]
        img_a, img_b = open_any(da, path_a), open_any(db, path_b)
        pair_rows.append({"pair_id": pair_id, "scope": scope, "phash_distance": distance,
                          "same_sha256": a["sha256"] == b["sha256"],
                          "dataset_a": da, "label_a": a["label"], "path_a": path_a,
                          "dataset_b": db, "label_b": b["label"], "path_b": path_b,
                          "pixel_diff": round(pixel_difference(img_a, img_b), 2),
                          "size_a": f"{img_a.width}x{img_a.height}", "size_b": f"{img_b.width}x{img_b.height}",
                          "_img_a": img_a, "_img_b": img_b})
    columns = ["pair_id", "scope", "phash_distance", "same_sha256", "dataset_a", "label_a", "path_a",
               "dataset_b", "label_b", "path_b", "pixel_diff", "size_a", "size_b"]
    pair_table = pd.DataFrame([{k: r[k] for k in columns} for r in pair_rows], columns=columns)
    pair_table.to_csv(OUT_PAIRS, index=False, lineterminator="\n")

    PAIR_SHEETS.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pair_rows), PAIRS_PER_SHEET):
        draw_pair_sheet(pair_rows[start:start + PAIRS_PER_SHEET], PAIR_SHEETS / f"pairs_{start // PAIRS_PER_SHEET + 1:02d}.png")
    if not OUT_REVIEW.exists():
        review = pair_table[["pair_id", "scope", "phash_distance", "same_sha256", "label_a", "label_b", "pixel_diff"]].copy()
        review["decision"] = np.where(review["same_sha256"], "duplicate", "")
        review["note"] = np.where(review["same_sha256"], "identical bytes", "")
        review.to_csv(OUT_REVIEW, index=False, lineterminator="\n")

    label_listing = draw_label_sheets(manifest)
    label_file = MANIFEST_DIR / "plantdoc_label_check.csv"
    if not label_file.exists():
        label_listing.to_csv(label_file, index=False, lineterminator="\n")

    # ---- summary
    sizes = ok.assign(short=ok[["width", "height"]].min(axis=1), long=ok[["width", "height"]].max(axis=1))
    dist_counts = pair_table.groupby(["phash_distance", "scope"]).size().unstack(fill_value=0) if len(pair_table) else "none"
    text = [
        "PLANTDOC AUDIT",
        f"images audited      : {len(manifest)}   decoded: {int(manifest['decode_ok'].sum())}   failed: {int((~manifest['decode_ok']).sum())}",
        f"failed files        : {manifest.loc[~manifest['decode_ok'], 'file'].tolist()}",
        f"class counts        : {manifest['label'].value_counts().to_dict()}  (published folder counts 91/93/89)",
        f"colour modes        : {manifest['mode'].value_counts(dropna=False).to_dict()}",
        f"file formats        : {manifest['format'].value_counts(dropna=False).to_dict()}",
        f"EXIF orientation    : {manifest['exif_orientation'].value_counts(dropna=False).to_dict()}",
        f"shorter side (px)   : min {sizes['short'].min()}, median {sizes['short'].median():.0f}, max {sizes['short'].max()}",
        f"longer side (px)    : min {sizes['long'].min()}, median {sizes['long'].median():.0f}, max {sizes['long'].max()}",
        f"manifest SHA-256    : {sha256_file(OUT_MANIFEST)}",
        f"original manifest   : {ORIGINAL_MANIFEST_SHA} (verified, read only)",
        "",
        f"exact copies inside PlantDoc          : {int((exact['scope'] == 'within_plantdoc').sum())} groups",
        f"exact copies of PlantVillage / PP2020 : {int((exact['scope'] == 'plantdoc_vs_original').sum())} images",
        f"pHash pairs 0..{PHASH_MAX_REVIEW_DISTANCE} bits (to review)         : {len(pair_table)}",
        f"{dist_counts}",
        "",
        "pHash is computed as in the original audit; consistency check is the pixel-difference hint",
        "(SIFT matching was not available in this environment).",
    ]
    OUT_SUMMARY.write_text("\n".join(map(str, text)) + "\n", encoding="utf-8", newline="\n")
    print(OUT_SUMMARY.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
