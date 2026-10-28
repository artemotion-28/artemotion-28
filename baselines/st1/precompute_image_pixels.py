"""Precompute ViT-processed pixel tensors for every unique painting, once.

Stores raw processed pixel tensors (not frozen embeddings), since ViT here is
fine-tuned. Only 1,658 unique images -> avoids re-decoding and re-normalizing
the same image on every caption row during training.

Reads the published annotations directly and derives the image path the same way
finetune_st1.py does ("images/" + image_name), so it does not depend on any
intermediate CSV.

Usage (from baselines/st1/, after the images are in place):
    python3 precompute_image_pixels.py
    python3 precompute_image_pixels.py --img-root /scratch/artemotion
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor

from model import IMAGE_MODEL_NAME

DEFAULT_DATA = "../../annotations/artelingo28_train_val.csv"
# image_path values look like "images/<art_style>/<painting>.jpg", so the root
# is whatever directory contains images/: the repository root, two levels up.
DEFAULT_IMG_ROOT = "../.."
DEFAULT_OUT_DIR = "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--img-root", default=DEFAULT_IMG_ROOT,
                        help="Directory containing the images/ folder")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    unique_paths = sorted(("images/" + df["image_name"]).unique())
    print(f"{len(unique_paths):,} unique images to preprocess")

    missing = [p for p in unique_paths[:50]
               if not os.path.isfile(os.path.join(args.img_root, p))]
    if missing:
        raise SystemExit(
            "Cannot find images under {}/ -- e.g. {}\n"
            "Run from baselines/st1/ in the repository, or pass --img-root\n"
            "pointing at the directory that contains images/.".format(args.img_root, missing[0]))

    processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_NAME)

    all_pixels = []
    for i, path in enumerate(unique_paths):
        image = Image.open(os.path.join(args.img_root, path)).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"][0]
        all_pixels.append(pixel_values.to(torch.float16))
        if (i + 1) % 200 == 0 or (i + 1) == len(unique_paths):
            print(f"  {i + 1}/{len(unique_paths)}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_pixels = os.path.join(args.out_dir, "image_pixel_values.pt")
    out_paths = os.path.join(args.out_dir, "image_pixel_paths.npy")

    pixel_tensor = torch.stack(all_pixels)  # [N, 3, 224, 224], float16
    torch.save(pixel_tensor, out_pixels)
    np.save(out_paths, np.array(unique_paths))
    size_gb = pixel_tensor.element_size() * pixel_tensor.nelement() / 1e9
    print(f"Saved {tuple(pixel_tensor.shape)} to {out_pixels} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
