# Starter kit

`starter_kit.ipynb` is a minimal starting point for SemEval-2027 Task 12
(ArtEmotion-28). For each subtask it loads the data, trains a very simple
baseline, checks it on the `val` split, and writes a submission file in the
Codabench format.

```bash
pip install pandas scikit-learn pillow torch transformers jupyter
jupyter notebook starter_kit.ipynb
```

Run it from this folder. It predicts for the Development-phase input files in
`../phase_inputs/dev/` and writes one upload-ready zip per subtask.
