# ST3 baseline: CLIP features with language-specific weights

The reference baseline for Subtask 3. Each painting is embedded once with a
frozen [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32).
A logistic regression predicts a language's positive share from the image
embedding, the language, and their interaction, so the same visual feature can
push different languages in different directions. Each training pair
contributes its positive and negative annotators as weighted examples.

## Results (development data)

Within-painting cultural contrast is the ranking metric (higher is better);
log loss is reported alongside.

| System | Contrast | Log loss |
|---|---|---|
| Predict 0.5 everywhere | 0.000 | 1.000 |
| Target-language prior alone | 0.364 | 0.868 |
| CLIP features, same image weights for every language | 0.370 | 0.766 |
| **CLIP features with language-specific weights (this baseline)** | **0.448** | **0.750** |

## Run

From this folder, using `../../annotations/`, `../../images/` and
`../../phase_inputs/`. Embedding the 1,658 paintings takes a few minutes on a
CPU and is cached in `results/`.

```bash
pip install -r requirements.txt
python3 clip_language_lr.py --phase dev      # or --phase test once released
```

It writes `results/<phase>/predictions.csv` and
`results/st3_<phase>_submission.zip`, ready to upload to Codabench.

## Settings

Logistic regression with L2 regularisation `C = 0.3`, trained on the 16,019
(painting, language) pairs with at least 5 annotators in the `train` split.
