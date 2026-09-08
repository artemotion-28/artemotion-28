#!/usr/bin/env python3
"""
Integrity checks for the ArtEmotion-28 public data release.

Run from the repository root:

    python3 scripts/verify_dataset.py

Checks:
  1. every image referenced by the annotations exists on disk
  2. every image on disk is referenced by the annotations
  3. splits are disjoint at the painting level -- a painting belongs to
     exactly one split, so its annotations in other languages cannot leak
     across the boundary
  4. emotion labels come from the documented vocabulary
  5. reports per-language and per-split counts

Exits non-zero if any check fails, so it can be used in CI.
"""
import argparse
import collections
import csv
import os
import sys

POSITIVE = {"amusement", "awe", "contentment", "excitement"}
NEGATIVE = {"anger", "disgust", "fear", "sadness"}
AMBIGUOUS = {"something else", "other"}
KNOWN_EMOTIONS = POSITIVE | NEGATIVE | AMBIGUOUS

DEFAULT_CSV = os.path.join("annotations", "artelingo28_train_val.csv")
DEFAULT_IMAGES = "images"


def load(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def check_image_mapping(rows, images_dir, report):
    referenced = {row["image_name"].strip() for row in rows}

    on_disk = set()
    for dirpath, _dirnames, filenames in os.walk(images_dir):
        for filename in filenames:
            if filename.startswith("."):
                continue
            on_disk.add(
                os.path.relpath(os.path.join(dirpath, filename), images_dir)
            )

    missing = referenced - on_disk
    orphans = on_disk - referenced

    report.append(("images referenced by annotations", len(referenced)))
    report.append(("image files on disk", len(on_disk)))

    ok = True
    if missing:
        print("FAIL: {} referenced images are missing from {}/".format(
            len(missing), images_dir))
        for name in sorted(missing)[:10]:
            print("        {}".format(name))
        ok = False
    else:
        print("PASS: every referenced image is present")

    if orphans:
        print("WARN: {} images on disk are never referenced".format(len(orphans)))
        for name in sorted(orphans)[:5]:
            print("        {}".format(name))
    else:
        print("PASS: no unreferenced images")

    return ok


def check_split_disjointness(rows):
    """
    Splits must be disjoint at the painting level: every annotation of a
    painting, in every language, belongs to the same split. Otherwise a
    painting's annotations in one language leak information about the same
    painting in another.

    Group on `painting`, which is 1:1 with the image file. Grouping on
    `image_id` would not detect this, since that column is not unique across
    paintings.
    """
    splits_by_painting = collections.defaultdict(set)
    rows_by_painting = collections.defaultdict(list)
    for row in rows:
        painting = row["painting"].strip()
        splits_by_painting[painting].add(row["split"].strip())
        rows_by_painting[painting].append(row)

    straddling = sorted(
        painting for painting, splits in splits_by_painting.items() if len(splits) > 1
    )

    if straddling:
        affected = sum(len(rows_by_painting[p]) for p in straddling)
        languages = collections.Counter(
            row["language"].strip() for p in straddling for row in rows_by_painting[p]
        )
        print(
            "FAIL: {} paintings appear in more than one split "
            "({} annotations involved)".format(len(straddling), affected)
        )
        print("        languages involved: {}".format(dict(languages)))
        for painting in straddling[:5]:
            print("        {}".format(painting))
        return False

    print("PASS: splits are disjoint at the painting level")
    return True


def check_emotions(rows):
    seen = collections.Counter(row["emotion"].strip().lower() for row in rows)
    unknown = set(seen) - KNOWN_EMOTIONS
    if unknown:
        print("FAIL: unexpected emotion labels: {}".format(sorted(unknown)))
        return False
    print("PASS: all emotion labels are in the documented vocabulary")
    return True


def summarize(rows):
    by_split = collections.Counter(row["split"].strip() for row in rows)
    by_language = collections.Counter(row["language"].strip() for row in rows)
    binary = sum(
        1
        for row in rows
        if row["emotion"].strip().lower() in POSITIVE | NEGATIVE
    )

    print("\nSummary")
    print("  total annotations      : {:,}".format(len(rows)))
    print("  binary-mappable        : {:,}".format(binary))
    print("  excluded as ambiguous  : {:,}".format(len(rows) - binary))
    print("  languages              : {}".format(len(by_language)))
    for split, count in sorted(by_split.items()):
        print("  split {:<16} : {:,}".format(split, count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit("Could not find {} (run from the repository root)".format(args.csv))

    rows = load(args.csv)
    print("Loaded {:,} annotations from {}\n".format(len(rows), args.csv))

    report = []
    results = [
        check_image_mapping(rows, args.images, report),
        check_split_disjointness(rows),
        check_emotions(rows),
    ]

    summarize(rows)

    if not all(results):
        print("\nOne or more checks FAILED.")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
