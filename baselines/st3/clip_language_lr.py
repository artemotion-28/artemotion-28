"""ST3 baseline: logistic regression on frozen CLIP image features, with
language-specific image weights.

Each painting is embedded once with CLIP ViT-B/32. The features for a
(painting, language) pair are the image embedding, a one-hot language vector
and their outer product, so the same visual feature can push different
languages in different directions. Each training pair contributes its positive
and negative annotators as weighted examples, so the model predicts the
language's positive share.

Usage (from baselines/st3/):
    python3 clip_language_lr.py --phase dev
    python3 clip_language_lr.py --phase test

Output: results/<phase>/predictions.csv and results/st3_<phase>_submission.zip,
ready to upload to Codabench. CLIP embeddings are cached in
results/clip_image_features.npz.
"""
import argparse
import os
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import CLIPModel, CLIPProcessor

POSITIVE = {"amusement", "awe", "contentment", "excitement"}
NEGATIVE = {"anger", "disgust", "fear", "sadness"}
MIN_ANNOTATORS = 5
CLIP_MODEL = "openai/clip-vit-base-patch32"
C = 0.3
FEATURES = "results/clip_image_features.npz"


def image_features(names):
    """L2-normalised CLIP embeddings, keyed by image path."""
    if os.path.exists(FEATURES):
        cached = np.load(FEATURES, allow_pickle=True)
        return dict(zip(cached["names"], cached["features"]))
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    names = sorted(names)
    batches = []
    for start in range(0, len(names), 32):
        images = [Image.open(f"../../images/{n}").convert("RGB") for n in names[start:start + 32]]
        with torch.no_grad():
            features = model.get_image_features(**processor(images=images, return_tensors="pt").to(device))
        batches.append(torch.nn.functional.normalize(features, dim=-1).cpu().numpy())
    features = np.concatenate(batches)
    os.makedirs("results", exist_ok=True)
    np.savez(FEATURES, names=np.array(names), features=features)
    return dict(zip(names, features))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="dev", help="dev or test")
    args = parser.parse_args()

    df = pd.read_csv("../../annotations/artelingo28_train_val.csv")
    df["language"] = df["language"].replace({"arabic": "Arabic", "chinese": "Chinese", "english": "English"})
    image_of = df.groupby("painting")["image_name"].first().str.strip()

    train = df[(df["split"] == "train") & df["emotion"].isin(POSITIVE | NEGATIVE)].copy()
    train["positive"] = train["emotion"].isin(POSITIVE).astype(int)
    shares = train.groupby(["painting", "language"])["positive"].agg(k="sum", n="size").reset_index()
    shares = shares[shares["n"] >= MIN_ANNOTATORS]

    input_path = f"../../phase_inputs/{args.phase}/st3_test.csv"
    test = pd.read_csv(input_path)
    print(f"{len(shares):,} training pairs; {len(test):,} rows from {input_path}")

    embedding = image_features(set(image_of) | set(test["image"].str.strip()))
    languages = sorted(shares["language"].unique())
    language_index = {language: i for i, language in enumerate(languages)}

    def features(images, langs):
        img = np.stack([embedding[i] for i in images])
        onehot = np.zeros((len(img), len(languages)))
        for row, language in enumerate(langs):
            if language in language_index:
                onehot[row, language_index[language]] = 1
        interaction = (onehot[:, :, None] * img[:, None, :]).reshape(len(img), -1)
        return np.hstack([img, onehot, interaction])

    X = features(shares["painting"].map(image_of), shares["language"])
    model = LogisticRegression(C=C, max_iter=3000).fit(
        np.vstack([X, X]),
        np.r_[np.ones(len(X)), np.zeros(len(X))],
        sample_weight=np.r_[shares["k"], shares["n"] - shares["k"]])

    test["p_positive"] = model.predict_proba(features(test["image"].str.strip(), test["language"]))[:, 1].round(6)

    out_dir = f"results/{args.phase}"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "predictions.csv")
    test[["painting", "language", "p_positive"]].to_csv(path, index=False)
    zip_path = f"results/st3_{args.phase}_submission.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(path, arcname="predictions.csv")
    print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()
