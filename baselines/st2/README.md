# ST2 baseline: source / target-prior blend

The reference baseline for Subtask 2. It predicts the target language's
positive share as a blend of the source language's share and the target
language's average share on the training paintings:

```
q = w · source_share + (1 − w) · prior[target_language]
```

`w` is chosen on the training pairs to minimise log loss (`w = 0.55`).

## Results (development data)

Log loss in bits per annotator, aggregated as the competition scores it
(lower is better).

| System | Log loss |
|---|---|
| Copy the source share | 1.054 |
| Predict 0.5 everywhere | 1.000 |
| Target-language prior alone | 0.880 |
| **Blend (this baseline)** | **0.795** |

## Run

From this folder, using `../../annotations/` and `../../phase_inputs/`. Runs in
seconds on a CPU.

```bash
pip install -r requirements.txt
python3 blend.py --phase dev      # or --phase test once released
```

It writes `results/<phase>/predictions.csv` and
`results/st2_<phase>_submission.zip`, ready to upload to Codabench.
