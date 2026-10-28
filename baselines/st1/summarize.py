"""Score the ST1 predictions the way the competition scorer does.

Per language: macro-F1 over the emotions present in that language's gold labels.
Headline: the unweighted mean of those per-language scores across languages.

Usage (from baselines/st1/, after the training runs):
    python3 summarize.py

Writes results/summary_per_language.csv and prints the headline numbers.
"""
import os

import pandas as pd

RESULTS = "results"
N_CLASSES = 8


def macro_f1(gold, pred):
    """Same rule as the Codabench ST1 scorer: average over classes present in gold."""
    scores = []
    for target in range(N_CLASSES):
        if not (gold == target).any():
            continue
        tp = ((gold == target) & (pred == target)).sum()
        fp = ((gold != target) & (pred == target)).sum()
        fn = ((gold == target) & (pred != target)).sum()
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores)


def load(mode):
    path = os.path.join(RESULTS, f"predictions_{mode}.csv")
    return pd.read_csv(path) if os.path.isfile(path) else None


def main():
    runs = {mode: load(mode) for mode in ("text_only", "multimodal", "image_only")}
    if runs["text_only"] is None or runs["multimodal"] is None:
        raise SystemExit("Need predictions_text_only.csv and predictions_multimodal.csv in results")

    columns = {"f1_text": ("text_only", "pred"), "f1_multi": ("multimodal", "pred"),
               "f1_multi_shuffled_images": ("multimodal", "pred_shuffled_images")}
    if runs["image_only"] is not None:
        columns["f1_image"] = ("image_only", "pred")

    rows = []
    for language, group in runs["multimodal"].groupby("language"):
        row = {"language": language, "n_val": len(group)}
        for name, (mode, field) in columns.items():
            df = runs[mode][runs[mode]["language"] == language]
            row[name] = macro_f1(df["label"].to_numpy(), df[field].to_numpy())
        row["delta_fusion"] = row["f1_multi"] - row["f1_text"]
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("delta_fusion", ascending=False)
    out = os.path.join(RESULTS, "summary_per_language.csv")
    table.round(4).to_csv(out, index=False)

    print(table.round(3).to_string(index=False))
    print()
    print(f"Languages: {len(table)}   val instances: {table['n_val'].sum():,}")
    for name in columns:
        print(f"  {name:26s} macro-avg across languages: {table[name].mean():.4f}")
    positive = (table["delta_fusion"] > 0).sum()
    print(f"  fusion-positive languages (multi > text): {positive} of {len(table)}")
    print(f"  F1 lost when images are shuffled: "
          f"{table['f1_multi'].mean() - table['f1_multi_shuffled_images'].mean():.4f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
