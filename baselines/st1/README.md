# ST1 baseline: XLM-R + ViT

The reference baseline for Subtask 1 (eight-way emotion prediction). XLM-R
encodes the caption, ViT encodes the painting; their `[CLS]` vectors are
concatenated and fed to one linear layer. Both encoders are fine-tuned.

It is trained twice, as ST1 requires: **text-only** (XLM-R alone) and
**multimodal** (XLM-R + ViT). An optional **image-only** run (ViT alone) shows
how much the painting says on its own.

## Results (development data)

F1-Macro computed per language, then averaged over the 28 languages — the
official ST1 metric.

| Run | F1-Macro |
|---|---|
| Text-only | 0.550 |
| Multimodal | 0.535 |
| Multimodal, on the decoy rows of the dev input file | 0.522 |
| Multimodal, val images shuffled at random (`summarize.py`) | 0.524 |
| Image-only | 0.161 |

The two shuffled-image rows differ because the decoys always pair a caption
with a *different* painting's image, as Codabench's image check does, while a
random shuffle sometimes leaves an image with its own painting. Multimodal
beats text-only in 9 of the 28 languages. On the dev input file the
text-only run labels every caption and its decoy identically, and the
multimodal run scores lower on decoys than on real rows, so both pass the
checks described in the main README.

## Requirements

One GPU. On an NVIDIA GB10 the three trainings took about 4.8 hours in total
(text-only 1.2 h, multimodal 2.6 h, image-only 1 h); early stopping makes this
vary. About 6 GB of disk for image tensors and
checkpoints.

```bash
pip install -r requirements.txt
```

## Run

Everything runs from this folder, using `../../annotations/` and
`../../images/` in the repository.

```bash
cd baselines/st1
./run_all.sh                      # text-only, then multimodal, then the summary
./run_all.sh image_only           # optional
python3 predict_phase.py --phase dev
```

`run_all.sh` preprocesses the images once, trains on the `train` split with
early stopping on `val`, and writes `results/summary_per_language.csv`. It is
safe to interrupt and rerun: finished steps are skipped and an interrupted
training resumes from its last checkpoint.

With two GPUs, train the two runs in parallel, one GPU each, then summarise:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_all.sh text_only
CUDA_VISIBLE_DEVICES=1 ./run_all.sh multimodal
./run_all.sh summary
```

`predict_phase.py` loads each run's best checkpoint and predicts
`../../phase_inputs/<phase>/st1_test.csv`, decoys included. It writes
`results/st1_<phase>_submission.zip`, ready to upload to Codabench. Use
`--phase test` once the evaluation-phase file is released.

## Settings

Up to 8 epochs, early stopping after 2 epochs without improvement in val
F1-Macro, batch size 32, learning rate 2e-5, captions truncated to 96 tokens,
seed 42. The best checkpoint was epoch 4 for both text-only and multimodal.

| File | Purpose |
|---|---|
| `model.py` | the classifier (text-only, image-only or multimodal) |
| `precompute_image_pixels.py` | ViT preprocessing of all 1,658 images, once |
| `finetune_st1.py` | training and val prediction for one mode |
| `summarize.py` | per-language scores, as the competition scores them |
| `predict_phase.py` | Codabench submission for a phase input file |
| `run_all.sh` | runs the steps above in order |
