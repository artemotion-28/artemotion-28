"""ST2 baseline: blend the source share with a target-language prior.

For each (painting, language) pair with at least 5 annotators, the positive
share is the fraction of annotators who chose a positive emotion. The
prediction for a target language is

    q = w * source_share + (1 - w) * prior[target_language]

where prior is the language's pooled positive share on the training paintings
and w is chosen on the training pairs to minimise log loss.

Usage (from baselines/st2/):
    python3 blend.py --phase dev
    python3 blend.py --phase test

Output: results/<phase>/predictions.csv and results/st2_<phase>_submission.zip,
ready to upload to Codabench.
"""
import argparse
import os
import zipfile

import numpy as np
import pandas as pd

POSITIVE = {"amusement", "awe", "contentment", "excitement"}
NEGATIVE = {"anger", "disgust", "fear", "sadness"}
MIN_ANNOTATORS = 5


def training_shares():
    """Positive count and annotator count per (painting, language) on the train split."""
    df = pd.read_csv("../../annotations/artelingo28_train_val.csv")
    df["language"] = df["language"].replace({"arabic": "Arabic", "chinese": "Chinese", "english": "English"})
    df = df[(df["split"] == "train") & df["emotion"].isin(POSITIVE | NEGATIVE)]
    df["positive"] = df["emotion"].isin(POSITIVE).astype(int)
    shares = df.groupby(["painting", "language"])["positive"].agg(k="sum", n="size").reset_index()
    shares = shares[shares["n"] >= MIN_ANNOTATORS]
    return shares.assign(p=shares["k"] / shares["n"])


def log_loss(q, k, n):
    q = np.clip(q, 0.01, 0.99)
    return np.mean(-(k * np.log2(q) + (n - k) * np.log2(1 - q)) / n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="dev", help="dev or test")
    args = parser.parse_args()

    shares = training_shares()
    prior = shares.groupby("language")["k"].sum() / shares.groupby("language")["n"].sum()
    overall = shares["k"].sum() / shares["n"].sum()

    pairs = shares.merge(shares, on="painting", suffixes=("_source", "_target"))
    pairs = pairs[pairs["language_source"] != pairs["language_target"]]
    target_prior = pairs["language_target"].map(prior)
    weights = np.linspace(0, 1, 21)
    losses = [log_loss(w * pairs["p_source"] + (1 - w) * target_prior,
                       pairs["k_target"], pairs["n_target"]) for w in weights]
    w = weights[int(np.argmin(losses))]
    print(f"blend weight w = {w:.2f} (training log loss {min(losses):.4f})")

    input_path = f"../../phase_inputs/{args.phase}/st2_test.csv"
    test = pd.read_csv(input_path)
    print(f"{len(test):,} rows from {input_path}")
    test["p_positive"] = (w * test["source_p_positive"]
                          + (1 - w) * test["target_language"].map(prior).fillna(overall)).round(6)

    out_dir = f"results/{args.phase}"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "predictions.csv")
    test[["painting", "source_language", "target_language", "p_positive"]].to_csv(path, index=False)
    zip_path = f"results/st2_{args.phase}_submission.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(path, arcname="predictions.csv")
    print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()
