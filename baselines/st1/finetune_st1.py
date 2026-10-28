"""Fine-tune the ArtEmotion-28 ST1 baseline: eight-way emotion classification.

XLM-R (text) + ViT (image), concatenation fusion, both encoders fine-tuned.
Trains on the release's train split and predicts its val split.

Usage (from baselines/st1/, after precompute_image_pixels.py):
    python3 finetune_st1.py --mode text_only
    python3 finetune_st1.py --mode multimodal
    python3 finetune_st1.py --mode image_only        # optional

Each mode is an independent job; give each its own GPU to run them in parallel.
Re-running a mode resumes from its last checkpoint.

Output: results/predictions_<mode>.csv, one row per val instance. For the
multimodal run it also contains predictions with the images shuffled among
val rows, which shows how much the model actually uses the image.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from model import MultimodalSentimentModel, TEXT_MODEL_NAME

DATA = "../../annotations/artelingo28_train_val.csv"
PIXELS_PATH = "results/image_pixel_values.pt"
PIXEL_PATHS = "results/image_pixel_paths.npy"
MAX_TEXT_LENGTH = 96

# Fixed order: label index i means EMOTIONS[i]. "something else" and "other"
# are not ST1 classes and are dropped.
EMOTIONS = ["amusement", "anger", "awe", "contentment",
            "disgust", "excitement", "fear", "sadness"]
EMOTION_INDEX = {name: i for i, name in enumerate(EMOTIONS)}


class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, mode, image_pixels=None, image_path_to_idx=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mode = mode
        self.image_pixels = image_pixels
        self.image_path_to_idx = image_path_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {"labels": torch.tensor(int(row["label"]), dtype=torch.long)}

        if self.mode in ("text_only", "multimodal"):
            enc = self.tokenizer(
                str(row["caption"]),
                truncation=True,
                max_length=MAX_TEXT_LENGTH,
                padding="max_length",
            )
            item["input_ids"] = torch.tensor(enc["input_ids"], dtype=torch.long)
            item["attention_mask"] = torch.tensor(enc["attention_mask"], dtype=torch.long)

        if self.mode in ("image_only", "multimodal"):
            img_idx = self.image_path_to_idx[row["image_path"]]
            item["pixel_values"] = self.image_pixels[img_idx].to(torch.float32)

        return item


def load_frame(data_path):
    df = pd.read_csv(data_path)
    df["emotion"] = df["emotion"].str.strip().str.lower()
    df = df[df["emotion"].isin(EMOTION_INDEX)].copy()
    df["label"] = df["emotion"].map(EMOTION_INDEX)
    df["language"] = df["language"].str.strip()
    df["image_path"] = "images/" + df["image_name"].str.strip()
    df["row_id"] = np.arange(len(df))
    return df


def build_datasets(mode, tokenizer, data_path, max_train_samples=None, max_val_samples=None):
    df = load_frame(data_path)

    image_pixels, image_path_to_idx = None, None
    if mode in ("image_only", "multimodal"):
        image_pixels = torch.load(PIXELS_PATH)
        image_paths = np.load(PIXEL_PATHS, allow_pickle=True)
        image_path_to_idx = {p: i for i, p in enumerate(image_paths)}

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    if max_train_samples:
        train_df = train_df.sample(n=min(max_train_samples, len(train_df)), random_state=0)
    if max_val_samples:
        val_df = val_df.sample(n=min(max_val_samples, len(val_df)), random_state=0)

    val_df = val_df.reset_index(drop=True)
    train_ds = EmotionDataset(train_df, tokenizer, mode, image_pixels, image_path_to_idx)
    val_ds = EmotionDataset(val_df, tokenizer, mode, image_pixels, image_path_to_idx)
    return train_ds, val_ds, val_df, (image_pixels, image_path_to_idx)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, preds, average="macro")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["text_only", "image_only", "multimodal"])
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--out", default=None,
                        help="Prediction CSV. Defaults to results/predictions_<mode>.csv")
    parser.add_argument("--num_train_epochs", type=float, default=8.0,
                        help="Max epochs -- training stops early via --early_stopping_patience")
    parser.add_argument("--early_stopping_patience", type=int, default=2,
                        help="Stop if val F1-Macro doesn't improve for this many epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Defaults to checkpoints/<mode>")
    parser.add_argument("--max_train_samples", type=int, default=None, help="For smoke-testing on a subset")
    parser.add_argument("--max_val_samples", type=int, default=None, help="For smoke-testing on a subset")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    train_ds, val_ds, val_df, (image_pixels, image_path_to_idx) = build_datasets(
        args.mode, tokenizer, args.data, args.max_train_samples, args.max_val_samples
    )
    print(f"mode={args.mode}  train={len(train_ds)}  val={len(val_ds)}  "
          f"classes={len(EMOTIONS)}  seed={args.seed}  epochs={args.num_train_epochs}")

    model = MultimodalSentimentModel(mode=args.mode, num_labels=len(EMOTIONS))

    use_cuda = torch.cuda.is_available()
    checkpoint_dir = args.checkpoint_dir or f"checkpoints/{args.mode}"
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        fp16=use_cuda,
        report_to=[],
        dataloader_num_workers=2 if use_cuda else 0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    resume_checkpoint = get_last_checkpoint(checkpoint_dir) if os.path.isdir(checkpoint_dir) else None
    if resume_checkpoint:
        print(f"Resuming from existing checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    print("Running final prediction on val set...")
    logits = trainer.predict(val_ds).predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    out_df = val_df[["row_id", "painting", "language", "emotion", "label"]].copy()
    out_df["pred"] = np.argmax(logits, axis=-1)
    out_df["confidence"] = probs.max(axis=1)

    if args.mode == "multimodal":
        # Same captions, images permuted among val rows. If predictions barely
        # change, the multimodal model is not really using the image.
        shuffled = val_df.copy()
        rng = np.random.default_rng(args.seed)
        shuffled["image_path"] = rng.permutation(shuffled["image_path"].to_numpy())
        shuffled_ds = EmotionDataset(shuffled, tokenizer, args.mode, image_pixels, image_path_to_idx)
        out_df["pred_shuffled_images"] = np.argmax(trainer.predict(shuffled_ds).predictions, axis=-1)

    out_path = args.out or f"results/predictions_{args.mode}.csv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(f"Overall val F1-Macro ({args.mode}): "
          f"{f1_score(out_df['label'], out_df['pred'], average='macro'):.4f}")


if __name__ == "__main__":
    main()
