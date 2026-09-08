# ArtEmotion-28 — public data

Public data release for **[ArtEmotion-28 @ SemEval-2027](https://artemotion-28.github.io/)**:
Cross-Cultural Multimodal Sentiment Analysis Across 28 Languages.

Built on [ArtELingo-28](https://www.artelingo.org) (Mohamed et al., EMNLP 2024):
emotional responses to WikiArt paintings from native speakers of 28 languages.

## Contents

```
annotations/artelingo28_train_val.csv   173,745 annotations (train + val)
images/                                 1,658 WikiArt paintings
scripts/verify_dataset.py               integrity checks
```

| Split | Annotations | Binary-mappable |
|---|---|---|
| `train` | 165,013 | 152,447 |
| `val` | 8,732 | 8,057 |
| **Total** | **173,745** | **160,504** |

The test split is released separately at the start of the evaluation phase —
see the [task timeline](https://artemotion-28.github.io/dates.html).

### Annotation columns

| Column | Description |
|---|---|
| `painting` | painting identifier, e.g. `mark-rothko_no-17-1961(1)` |
| `image_name` | path to the image, relative to `images/` |
| `image_id` | numeric id (see note below) |
| `art_style` | WikiArt style, e.g. `Color_Field_Painting` |
| `genre` | WikiArt genre, e.g. `abstract_painting` |
| `language` | annotator's language (28 values) |
| `caption` | free-form caption describing the emotional response |
| `emotion` | one of nine emotion labels (see below) |
| `split` | `train` or `val` |

To locate an image: `images/<image_name>`.

### Emotion labels and binary sentiment

Nine emotion categories. For the shared task these map to binary sentiment
following emotion valence theory (Russell, 1980):

| Sentiment | Emotions |
|---|---|
| **positive** | amusement, awe, contentment, excitement |
| **negative** | anger, disgust, fear, sadness |
| *excluded* | something else, other |

Excluding the ambiguous categories leaves **160,504** binary-mappable
annotations of the 173,745 total.

## Working with the data

**Identify a painting by `painting` or `image_name`, not `image_id`.** Those two
are 1:1 with the image file; `image_id` is not unique across paintings, so
grouping on it will merge unrelated works. This matters whenever you group,
deduplicate, or build your own cross-validation folds — the released
`train`/`val` split is disjoint at the painting level, and grouping on
`image_id` would not preserve that.

**Normalise language names before grouping.** `arabic`, `chinese` and `english`
are lowercase; the other 25 are capitalised.

## Verifying the data

```bash
python3 scripts/verify_dataset.py
```

Checks that every referenced image is present, that splits are disjoint at the
painting level, and that all emotion labels are in the documented vocabulary.
Exits non-zero on failure, so it can be used in CI.

## Licensing

**This repository contains two kinds of material under different terms.**

### Annotations — CC BY 4.0

Everything in `annotations/` (captions, emotion labels, splits, and metadata)
is released by the ArtEmotion-28 organizers under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
You may share and adapt it, including commercially, with attribution.

### Images — not relicensed

The paintings in `images/` are works by third parties, sourced from
[WikiArt](https://www.wikiart.org). **They are not covered by the CC BY 4.0
grant above and are not relicensed by us.** They are mirrored here purely as a
convenience so that participants do not each have to re-scrape them, and all
rights remain with the original rights holders. If you intend to use the images
beyond research on this task, check their individual terms at the source.

See [LICENSE](LICENSE) for the full statement.

## Citation

Please cite the underlying dataset:

```bibtex
@inproceedings{mohamed-etal-2024-artelingo28,
    title     = {No Culture Left Behind: ArtELingo-28, a Benchmark of WikiArt with Captions in 28 Languages},
    author    = {Mohamed, Youssef and Li, Runjia and Ahmad, Ibrahim Said and Haydarov, Kilichbek and Torr, Philip and Church, Kenneth and Elhoseiny, Mohamed},
    booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
    year      = {2024},
    pages     = {20939--20962},
    address   = {Miami, Florida, USA},
    publisher = {Association for Computational Linguistics}
}
```

## Ethics

The annotations contain no personally identifiable information — annotator
identifiers have been removed. The data must not be used to identify or profile
individuals.
