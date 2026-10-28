"""Predict a phase input file with the trained ST1 baseline and write a
Codabench submission.

Loads the best checkpoint of the text_only and multimodal runs (the ones
finetune_st1.py selected on the val split) and predicts every row of
../../phase_inputs/<phase>/st1_test.csv, decoys included.

Usage (from baselines/st1/, after the training runs):
    python3 predict_phase.py --phase dev
    python3 predict_phase.py --phase test

Output: results/<phase>/predictions_text.csv and predictions_multimodal.csv,
and results/st1_<phase>_submission.zip containing both, ready to upload.
"""
import argparse
import json
import os
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoImageProcessor, AutoTokenizer

from finetune_st1 import EMOTIONS, MAX_TEXT_LENGTH
from model import IMAGE_MODEL_NAME, TEXT_MODEL_NAME, MultimodalSentimentModel

IMG_ROOT = "../.."
SUBMISSION_FILES = {"text_only": "predictions_text.csv",
                    "multimodal": "predictions_multimodal.csv"}


def best_checkpoint(mode):
    """The checkpoint the Trainer marked best (highest val F1-Macro)."""
    root = f"checkpoints/{mode}"
    if not os.path.isdir(root):
        raise SystemExit(f"No {root}/ -- train first: ./run_all.sh {mode}")
    latest = max((d for d in os.listdir(root) if d.startswith("checkpoint-")),
                 key=lambda d: int(d.split("-")[1]))
    with open(os.path.join(root, latest, "trainer_state.json")) as handle:
        best = json.load(handle)["best_model_checkpoint"]
    return os.path.join(root, os.path.basename(best))


def load_model(mode, device):
    path = best_checkpoint(mode)
    model = MultimodalSentimentModel(mode=mode, num_labels=len(EMOTIONS))
    model.load_state_dict(load_file(os.path.join(path, "model.safetensors")))
    print(f"{mode}: loaded {path}")
    return model.to(device).eval()


def image_tensors(paths):
    """ViT pixel tensors for each unique image, in the order of `paths`."""
    processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_NAME)
    pixels = []
    for path in paths:
        image = Image.open(os.path.join(IMG_ROOT, "images", path)).convert("RGB")
        pixels.append(processor(images=image, return_tensors="pt")["pixel_values"][0])
    return torch.stack(pixels)


@torch.no_grad()
def predict(model, df, tokenizer, pixels, image_index, device, batch_size=64):
    labels = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        inputs = {}
        if model.text_encoder is not None:
            enc = tokenizer(batch["caption"].astype(str).tolist(), truncation=True,
                            max_length=MAX_TEXT_LENGTH, padding="max_length",
                            return_tensors="pt")
            inputs["input_ids"] = enc["input_ids"].to(device)
            inputs["attention_mask"] = enc["attention_mask"].to(device)
        if model.image_encoder is not None:
            idx = torch.tensor(batch["image"].map(image_index).tolist())
            inputs["pixel_values"] = pixels[idx].to(device)
        logits = model(**inputs)["logits"]
        labels.extend(EMOTIONS[i] for i in logits.argmax(-1).tolist())
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="dev", help="dev or test")
    args = parser.parse_args()

    input_path = f"../../phase_inputs/{args.phase}/st1_test.csv"
    df = pd.read_csv(input_path)
    print(f"{len(df):,} rows from {input_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    unique_images = sorted(df["image"].unique())
    pixels = image_tensors(unique_images)
    image_index = {path: i for i, path in enumerate(unique_images)}

    out_dir = f"results/{args.phase}"
    os.makedirs(out_dir, exist_ok=True)
    zip_path = f"results/st1_{args.phase}_submission.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for mode, filename in SUBMISSION_FILES.items():
            model = load_model(mode, device)
            out = pd.DataFrame({"id": df["id"],
                                "label": predict(model, df, tokenizer, pixels, image_index, device)})
            path = os.path.join(out_dir, filename)
            out.to_csv(path, index=False)
            archive.write(path, arcname=filename)
            del model
    print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()
