# ArtEmotion-28 @ SemEval-2027

### Cross-Cultural Multimodal Sentiment Analysis Across 28 Languages

[Task Website](https://artemotion-28.github.io/) | [Subtasks](https://artemotion-28.github.io/tasks.html) | [How to Participate](https://artemotion-28.github.io/participation.html) | [Important Dates](https://artemotion-28.github.io/dates.html) | [Ask a Question](https://github.com/artemotion-28/data-public/issues)

---

Multimodal AI systems that combine vision and language are increasingly deployed
for emotion understanding worldwide. These systems typically assume that visual
features for emotion recognition transfer universally across cultures — an
assumption rooted in predominantly Western training data. Cultural psychology
research has long established that emotional expression and perception vary
significantly between cultures.

**ArtEmotion-28 asks participants to build systems that model how the emotional
interpretation of visual content varies across languages and cultures.**

This repository holds the public data release. It is built on
[ArtELingo-28](https://www.artelingo.org) (Mohamed et al., EMNLP 2024):
emotional responses to WikiArt paintings from native speakers of 28 languages.

---

# Contents

- [Subtasks](#subtasks)
  - [ST1: Cross-Cultural Multimodal Sentiment Classification](#st1-cross-cultural-multimodal-sentiment-classification)
  - [ST2: Modality Contribution Prediction](#st2-modality-contribution-prediction)
  - [ST3: Cross-Cultural Sentiment Divergence Prediction](#st3-cross-cultural-sentiment-divergence-prediction)
- [Languages](#languages)
- [Getting the Data](#getting-the-data)
- [Dataset Description](#dataset-description)
- [Class Distribution Per Language](#class-distribution-per-language)
- [Working With the Data](#working-with-the-data)
- [Evaluation](#evaluation)
- [Baselines](#baselines)
- [Important Dates](#important-dates)
- [How to Participate](#how-to-participate)
- [Competition Rules and Terms](#competition-rules-and-terms)
- [FAQs](#faqs)
- [Communication](#communication)
- [Citation](#citation)
- [Organizers](#organizers)
- [Ethical Statement](#ethical-statement)
- [License](#license)

---

# Subtasks

Three subtasks. **You may enter one, two, or all three**, and within a subtask
you may work on any subset of the 28 languages.

## ST1: Cross-Cultural Multimodal Sentiment Classification

Given an **artwork image** and a **native-speaker caption** in a target
language, predict the binary sentiment (**positive** / **negative**) of the
caption.

**You must submit two runs: a text-only run and a multimodal (vision + text)
run.** Requiring both is what makes the task measure the actual contribution of
visual features in each language, rather than assuming multimodal is always
better. Our baseline shows fusion can *underperform* text-only in many
languages.

> **Your multimodal run must genuinely condition on the image.** Runs that do
> not measurably do so are excluded from the multimodal leaderboard.

### Example

A real instance from the training data:

| Field | Value |
|---|---|
| `painting` | `mark-rothko_no-17-1961(1)` |
| `image_name` | `Color_Field_Painting/mark-rothko_no-17-1961(1).jpg` |
| `art_style` | `Color_Field_Painting` |
| `language` | English |
| `caption` | "The light and dark shades together remind me of the ups and downs of life and finding the balance" |
| `emotion` | contentment |
| → **sentiment** | **positive** |

The same painting carries captions from native speakers of all 28 languages,
and they do not always agree — that disagreement is the subject of ST3.

### Submission format

A `.zip` containing **both** files at the top level:

```
predictions_text.csv          predictions_multimodal.csv
```

Each with the header `id,label`, covering every id in the test file:

```csv
id,label
st1_000000,positive
st1_000001,negative
```

`label` accepts `positive`/`negative` or `1`/`0`. A submission missing either
file is rejected — both runs are required.

## ST2: Modality Contribution Prediction

Given a **target language** and an **artwork image**, predict whether
multimodal fusion will *improve* sentiment classification over a text-only
baseline for that language.

This operationalizes a question that matters for real deployments: **when should
a multimodal system trust its vision encoder?**

The gold label comes from the per-language fusion delta

```
ΔFusion(ℓ) = F1-Macro_multi(ℓ) − F1-Macro_text(ℓ)
fusion_positive(ℓ) = 1 if ΔFusion(ℓ) > 0 else 0
```

computed with a **fixed reference architecture**: the released fine-tuned
XLM-R (text) + ViT (image) concatenation-fusion baseline. The label is anchored
to that one architecture; a different baseline could move languages across the
boundary.

> **ST2's test split holds out a disjoint set of languages entirely.** Unlike
> ST1 and ST3, which hold out instances within languages you have seen, ST2
> evaluates on languages absent from training — otherwise a system could
> memorize each training language's label instead of predicting it. The held-out
> language set is announced with the training data; only their labels stay
> confidential until evaluation.

### Example

**One prediction per language, not per instance.** There are only 28 labels in
the whole subtask, so think of it as 28 data points, not 160,000.

*At training time* you are given the announced training languages **with their
ΔFusion already computed**, plus all their images and captions. These are real
values from the released baseline:

| Language | Train rows | ΔFusion | `fusion_positive` |
|---|---|---|---|
| Tagalog | 6,506 | **+0.0358** | **1** |
| Hindi | 6,497 | **+0.0273** | **1** |
| English | 5,481 | **+0.0143** | **1** |
| Thai | 4,077 | −0.0114 | 0 |
| IsiNdebele | 2,015 | **−0.0976** | **0** |
| IsiXhosa | 1,850 | −0.0666 | 0 |
| Emakhuwa | 1,734 | −0.0703 | 0 |

`Train rows` is the language's binary-mappable training count, which you can
compute yourself from the annotations. `ΔFusion` and `fusion_positive` come from
the reference baseline — see
[`baselines/st2_delta_fusion_reference.csv`](baselines/st2_delta_fusion_reference.csv)
for all 28 languages with confidence intervals.

A pattern is already visible: the languages where vision *helps* tend to have
more training data, and the worst ΔFusion values sit with the smallest
languages. In our baseline, ΔFusion correlates with per-language data volume
(r = 0.63) and text-only strength (r = 0.55), but **not** with vision quality
(r = 0.014 against an image-only classifier). Learning that relationship — from
whatever signals you can measure — is the task.

*At test time* you get a **held-out language you have never seen a label for**,
along with its images and captions, and you predict whether fusion helps:

```
Given:   language = Kazakh, plus its artworks and captions
Predict: by how much does multimodal fusion beat (or lose to) text-only?
Answer:  delta_fusion = -0.03     <- ranked on this
         fusion_positive = 0      <- its sign, scored secondarily
```

Predict the **magnitude**, not just the direction. Estimating ΔFusion ≈ −0.03
and ΔFusion ≈ −0.09 both give `fusion_positive = 0`, but they are very
different answers, and the ranking metric can tell them apart.

### Submission format

A `.zip` containing `predictions.csv`, one row per held-out language:

```csv
language,fusion_positive,delta_fusion
Kazakh,0,-0.0282
Urdu,0,-0.0265
Tagalog,1,0.0358
```

**`delta_fusion` is required.** MAE on it is the primary ranking metric: with
only a handful of held-out languages, the binary sign metric admits very few
distinct scores — one flipped language moves macro-F1 by more than 0.13 — so
systems are ranked on the continuous magnitude instead. The sign is still
scored, as a secondary metric.

### Reference ΔFusion values

`baselines/st2_delta_fusion_reference.csv` gives the per-language ΔFusion from
the reference architecture, with 95% bootstrap confidence intervals:

```csv
language,n_val,f1_text,f1_multi,delta_fusion,ci_low,ci_high,sign_determined,fusion_positive
Tagalog,284,0.840683,0.876522,0.035839,0.000494,0.072260,1,1
Hindi,287,0.859563,0.886846,0.027283,-0.005706,0.060051,0,1
```

`sign_determined` is 1 when the 95% interval excludes zero. **Read this column
before trusting a label.** Many languages sit close to zero relative to their
interval, which is precisely why MAE on the magnitude — not the sign — decides
the ranking. At release, this file covers the announced training languages
only; the held-out test languages are withheld until evaluation.

## ST3: Cross-Cultural Sentiment Divergence Prediction

Given an **artwork image** and the **source-language sentiment distribution
across annotators**, predict the **target-language sentiment distribution** for
the same artwork.

The input is a *distribution over annotators*, not a single label. Predictions
are made per **(image, source language, target language)** triple. This is the
most ambitious subtask: it asks systems to model cultural distance in emotional
interpretation rather than classify sentiment within one language.

**Eligibility:** restricted to (image, language) pairs with **≥ 5 annotators**,
so every evaluated distribution is statistically meaningful rather than a single
noisy vote. This yields **16,868 eligible pairs** across 25 languages — see the
per-language counts [below](#class-distribution-per-language).

### Example

Take Vasily Perov's *Children Sleeping* (1870), a real painting in the dataset.
Here is how native speakers of five languages actually labelled it:

| Language | Annotators | Emotions given | `p_positive` |
|---|---|---|---|
| English | 5 | awe, awe, awe, awe, contentment | **1.00** |
| Hausa | 5 | contentment, contentment, excitement, excitement, sadness | **0.80** |
| Igbo | 5 | contentment, contentment, awe, excitement, sadness | **0.80** |
| Korean | 5 | contentment, sadness, sadness, sadness, disgust | **0.20** |
| Swahili | 5 | sadness, sadness, sadness, sadness, fear | **0.00** |

The same image, read as *peaceful* by English annotators and as *distressing*
by Swahili annotators. Their captions show why:

> **English** (awe): "The total relaxation on the child's face show just how peaceful her sleep is and how safe she feels…"
>
> **Swahili** (fear): "wamelala kwa sakafu, na baridi kali wako hali hatari ya kuwa wagonjwa"
> — *approximate gloss: they are sleeping on the floor, and in the bitter cold they are in danger of falling ill*

One group sees a child safe at rest; the other sees children on a cold floor at
risk of illness. **Predicting that shift is the whole subtask.**

*What you are given* — the source distribution and the target language, but not
the target distribution:

```
painting        = vasily-perov_children-sleeping-1870
source_language = English      source_p_positive = 1.00
target_language = Swahili      target_p_positive = ???
```

*What you predict* — the proportion of **target-language** annotators who
responded positively:

```
p_positive = 0.15     (gold for this triple is 0.00)
```

Note the target is a **proportion, not a label**. `0.80` means "4 of 5
annotators responded positively" — Hausa and Igbo above are genuine partial
splits, not noise. Predicting `1.0` or `0.0` everywhere throws away most of the
available credit, since scoring compares full distributions.

Copying the source distribution unchanged is the "no cultural shift" baseline.
It scores **JSD 0.099** on this data — a legitimate starting point, and the
number to beat.

### Submission format

A `.zip` containing `predictions.csv`, one row per triple in the test file:

```csv
painting,source_language,target_language,p_positive
vasily-perov_children-sleeping-1870,English,Swahili,0.15
vasily-perov_children-sleeping-1870,English,Hausa,0.72
mark-rothko_no-17-1961(1),English,Tamil,0.64
```

`p_positive` is a float in `[0, 1]`; `p_negative` is implied as `1 − p_positive`.
Every triple in the test file must appear exactly once.

---

# Languages

**28 languages**, concentrated in Africa (11) and Asia (16), with English as the
sole representative outside those regions.

| Region | Languages |
|---|---|
| West Africa (3) | Hausa, Igbo, Yoruba |
| East Africa (3) | Swahili, Kinyarwanda, Emakhuwa |
| South Africa (4) | IsiZulu, IsiXhosa, Setswana, IsiNdebele |
| North Africa (1) | Darija |
| Middle East / West Asia (2) | Arabic, Turkish |
| South Asia (3) | Hindi, Tamil, Urdu |
| Central Asia (3) | Kazakh, Kyrgyz, Uzbek |
| Southeast Asia (6) | Burmese, Indonesian, Malay, Tagalog, Thai, Vietnamese |
| East Asia (2) | Chinese, Korean |
| Other (1) | English |

---

# Getting the Data

```bash
git clone https://github.com/artemotion-28/data-public.git
cd data-public
python3 scripts/verify_dataset.py     # optional integrity check
```

The repository is ~130 MB and needs no Git LFS.

```
annotations/artelingo28_train_val.csv   173,745 annotations (train + val)
images/                                 1,658 WikiArt paintings
baselines/st2_delta_fusion_reference.csv  per-language ΔFusion + 95% CIs
scripts/verify_dataset.py               integrity checks
```

**Test data** is released separately at the start of the evaluation phase
(10 January 2027). The 342 remaining paintings are held back with it.

**Competition registration** will be hosted on Codabench. The link is posted on
the [task website](https://artemotion-28.github.io/) and announced here when
registration opens.

---

# Dataset Description

Each row is one annotator's response to one painting, in one language.

| Column | Description |
|---|---|
| `painting` | painting identifier, e.g. `mark-rothko_no-17-1961(1)` |
| `image_name` | path to the image, relative to `images/` |
| `image_id` | numeric id — **not unique across paintings**, see [Working With the Data](#working-with-the-data) |
| `art_style` | WikiArt style, e.g. `Color_Field_Painting` |
| `genre` | WikiArt genre, e.g. `abstract_painting` |
| `language` | annotator's language (28 values) |
| `caption` | free-form caption describing the emotional response |
| `emotion` | one of nine emotion labels |
| `split` | `train` or `val` |

To load an image: `images/<image_name>`.

### Emotion labels → binary sentiment

Annotators chose from nine emotion categories. For this task they map to binary
sentiment following emotion valence theory (Russell, 1980):

| Sentiment | Emotions |
|---|---|
| **positive** | amusement, awe, contentment, excitement |
| **negative** | anger, disgust, fear, sadness |
| *excluded* | something else, other |

Excluding the two ambiguous categories leaves **160,504** binary-mappable
annotations of the 173,745 total. Overall class balance is **69.0% positive /
31.0% negative**, which is why **F1-Macro computed per language** is the primary
ST1 metric rather than accuracy.

### Splits

Splits are made at the **image level**: a painting belongs to exactly one split,
and all of its annotations across all 28 languages follow it. The same artwork
therefore never appears on both sides of the boundary in a different language.

---

# Class Distribution Per Language

Binary-mappable annotations only. `ST3-eligible pairs` counts (image, language)
pairs with ≥ 5 annotators — the subset ST3 evaluates on.

| Language | Region | Train (pos/neg) | Val (pos/neg) | Train | Val | % positive | ST3-eligible pairs |
|---|---|---|---|---|---|---|---|
| Arabic | Middle East / West Asia | 4,227 / 828 | 246 / 34 | 5,055 | 280 | 83.8% | 564 |
| Burmese | Southeast Asia | 3,937 / 1,530 | 202 / 92 | 5,467 | 294 | 71.8% | 208 |
| Chinese | East Asia | 4,360 / 2,035 | 251 / 96 | 6,395 | 347 | 68.4% | 1,174 |
| Darija | North Africa | 4,067 / 1,939 | 211 / 105 | 6,006 | 316 | 67.7% | 17 |
| Emakhuwa | East Africa | 1,338 / 396 | 68 / 13 | 1,734 | 81 | 77.5% | 5 |
| English | Other | 3,925 / 1,556 | 240 / 68 | 5,481 | 308 | 71.9% | 684 |
| Hausa | West Africa | 5,390 / 1,585 | 283 / 80 | 6,975 | 363 | 77.3% | 979 |
| Hindi | South Asia | 4,900 / 1,597 | 246 / 73 | 6,497 | 319 | 75.5% | 952 |
| Igbo | West Africa | 5,526 / 1,798 | 298 / 92 | 7,324 | 390 | 75.5% | 1,340 |
| Indonesian | Southeast Asia | 3,050 / 1,182 | 158 / 70 | 4,232 | 228 | 71.9% | 308 |
| IsiNdebele | South Africa | 1,373 / 642 | 70 / 32 | 2,015 | 102 | 68.2% | 0 |
| IsiXhosa | South Africa | 1,354 / 496 | 72 / 23 | 1,850 | 95 | 73.3% | 0 |
| IsiZulu | South Africa | 4,588 / 2,185 | 241 / 116 | 6,773 | 357 | 67.7% | 798 |
| Kazakh | Central Asia | 1,430 / 818 | 75 / 32 | 2,248 | 107 | 63.9% | 0 |
| Kinyarwanda | East Africa | 3,959 / 1,425 | 209 / 81 | 5,384 | 290 | 73.5% | 9 |
| Korean | East Asia | 3,841 / 3,136 | 206 / 163 | 6,977 | 369 | 55.1% | 1,108 |
| Kyrgyz | Central Asia | 2,882 / 3,228 | 144 / 162 | 6,110 | 306 | 47.2% | 670 |
| Malay | Southeast Asia | 4,274 / 2,009 | 233 / 111 | 6,283 | 344 | 68.0% | 765 |
| Setswana | South Africa | 4,332 / 1,410 | 219 / 69 | 5,742 | 288 | 75.5% | 345 |
| Swahili | East Africa | 4,283 / 2,659 | 224 / 148 | 6,942 | 372 | 61.6% | 1,048 |
| Tagalog | Southeast Asia | 4,222 / 2,284 | 211 / 135 | 6,506 | 346 | 64.7% | 699 |
| Tamil | South Asia | 4,056 / 1,548 | 215 / 91 | 5,604 | 306 | 72.3% | 699 |
| Thai | Southeast Asia | 2,559 / 1,518 | 146 / 72 | 4,077 | 218 | 63.0% | 254 |
| Turkish | Middle East / West Asia | 4,587 / 1,926 | 247 / 98 | 6,513 | 345 | 70.5% | 601 |
| Urdu | South Asia | 2,283 / 661 | 128 / 32 | 2,944 | 160 | 77.7% | 19 |
| Uzbek | Central Asia | 5,293 / 1,677 | 276 / 88 | 6,970 | 364 | 75.9% | 1,093 |
| Vietnamese | Southeast Asia | 4,723 / 2,587 | 255 / 139 | 7,310 | 394 | 64.6% | 1,298 |
| Yoruba | West Africa | 4,399 / 2,634 | 228 / 140 | 7,033 | 368 | 62.5% | 1,231 |
| **Total** | | | | **152,447** | **8,057** | | **16,868** |

Class balance varies widely — from 47.2% positive (Kyrgyz) to 83.8% (Arabic).
Three languages (IsiNdebele, IsiXhosa, Kazakh) have no ST3-eligible pairs
because no painting reaches 5 annotators in them; they remain fully available
for ST1 and ST2.

---

# Working With the Data

**Identify a painting by `painting` or `image_name`, not `image_id`.** Those two
are 1:1 with the image file; `image_id` is not unique across paintings, so
grouping on it merges unrelated works. This matters whenever you group,
deduplicate, or build your own cross-validation folds — the released split is
disjoint at the painting level, and grouping on `image_id` would not preserve
that.

**Language names are capitalised consistently** (`Arabic`, `Chinese`,
`English`, …) and match exactly the names used in the competition input files
and expected in submissions, so no normalisation is needed.

**Exclude the ambiguous emotions.** Filter out `something else` and `other`
before mapping to binary sentiment, or your class counts will not match the
table above.

---

# Evaluation

| Subtask | Primary metric | Also reported |
|---|---|---|
| **ST1** | **F1-Macro**, computed per language then macro-averaged across all 28 | F1-Weighted, Accuracy |
| **ST2** | **MAE** of the predicted ΔFusion magnitude (lower is better) | F1-Macro over the two outcome classes |
| **ST3** | **Jensen–Shannon Divergence** (lower is better) | Pearson correlation of cross-lingual sentiment shift |

**ST1** rankings use the **multimodal** run. Every language contributes equally
to the macro-average regardless of how many instances it has, so high-resource
languages cannot dominate the ranking — consistent with the task's
cross-cultural-equity motivation.

**ST3** aggregates JSD hierarchically: mean JSD per (source, target) language
pair → averaged across each target's eligible source partners → macro-averaged
across target languages.

---

# Baselines

Baselines for all three subtasks are implemented and released with the training
data.

| Subtask | Baseline | Result |
|---|---|---|
| **ST1** | Fine-tuned XLM-R (text) + ViT (image), concatenation fusion, trained to convergence with early stopping | macro-F1 **0.87** text-only, **0.85** multimodal |
| **ST2** | Predicts fusion contribution from per-language training-set size and text-model confidence | F1-Macro **0.67** vs. **0.39** majority-class (leave-one-language-out) |
| **ST3** | Blends the source distribution with a target-language prior, tuned to minimize JSD | mean JSD **0.08** vs. **0.10** for single-signal baselines |

Per-language ΔFusion ranges from **−0.098** (IsiNdebele) to **+0.036**
(Tagalog), with **10 of 28** languages fusion-positive. Fusion gain correlates
with per-language data volume (r = 0.63) and text-only strength (r = 0.55)
rather than with vision quality (r = 0.014 against an image-only classifier) —
which is what makes ST2 a non-trivial prediction problem.

---

# Important Dates

All deadlines are **11:59 PM AoE (Anywhere on Earth)**. These follow the
official [SemEval-2027 timeline](https://semeval.github.io/SemEval2027/).

| Event | Date |
|---|---|
| Sample data ready | 8 August 2026 |
| Training data ready | 8 September 2026 |
| Evaluation period starts | 10 January 2027 |
| Evaluation period ends | 31 January 2027 |
| System paper submission *(tentative)* | February 2027 |
| Notification to authors *(tentative)* | March 2027 |
| Camera-ready due *(tentative)* | April 2027 |
| SemEval-2027 workshop | Summer 2027 |

---

# How to Participate

1. **Register** on Codabench when registration opens (link posted on the
   [task website](https://artemotion-28.github.io/)).
2. **Choose your subtasks** — one, two, or all three.
3. **Choose your languages** — any subset of the 28. Note that ST2's test
   partition holds out languages entirely.
4. **Download the data** from this repository.
5. **Build your system.** For ST1, remember that *both* a text-only and a
   multimodal run are required.
6. **Submit predictions** during the evaluation period (10–31 January 2027).
7. **Write a system description paper** — required to appear in the official
   ranking.

---

# Competition Rules and Terms

**Scores.** By submitting results you consent to the public release of your
scores on the competition website, at the SemEval-2027 workshop, and in the
proceedings. Organizers have discretion over the release and choice of metrics,
and may withhold scores for incomplete, erroneous, deceptive, or rule-violating
submissions. Inclusion of a submission's scores does not constitute endorsement.

**Teams.** Participants may be involved in only one team; exceptions require
prior approval, requested before the evaluation period begins. Each team uses
exactly one account, and team membership cannot change once evaluation starts.

**Submissions.** During development, results are visible only to your team.
During evaluation, only your final submission counts for the ranking.

**Post-competition.** Gold labels are released after the competition. Teams are
encouraged to report all their system variants in their paper, clearly marking
which was the official submission.

**Papers.** A system description paper is required for inclusion in the official
ranking. Each participating team is expected to review another team's paper.

**Data use.** The annotations are released under CC BY 4.0 for scientific and
research purposes. They contain no personally identifiable information and must
not be used to identify or profile individuals. Please direct interested parties
to this repository rather than redistributing modified copies.

**No warranty.** The organizers and their institutions provide no warranties on
dataset correctness or completeness and accept no liability arising from its
access or use.

---

# FAQs

**Do I have to enter all three subtasks?**
No. One, two, or all three.

**Do I have to cover all 28 languages?**
No, any subset. For ST2, note that the test partition holds out languages
entirely, so systems there need to generalize to unseen languages.

**What exactly do I submit for ST1?**
Two runs — text-only and multimodal — in a single archive. Both are required.
The primary ranking uses the multimodal run.

**My multimodal system barely differs from my text-only one. Is that a problem?**
Yes, potentially. Runs that do not measurably condition on the image are
excluded from the multimodal leaderboard. Our own baseline shows fusion
underperforming text-only in 18 of 28 languages, so a system that quietly
ignores the image is a real failure mode we check for.

**Can I use LLMs or additional data?**
Guidance will be published with the competition platform details. Any external
resources must be cited in your system description paper.

**Do I get the artwork images?**
Yes, mirrored here for convenience. They are third-party works and are not
relicensed — see [License](#license). The released data also includes each
image's source URL.

**Why do some languages have no ST3 pairs?**
ST3 needs ≥ 5 annotators per (image, language) pair. IsiNdebele, IsiXhosa and
Kazakh do not reach that threshold for any painting. They are fully available
for ST1 and ST2.

**When are gold labels released?**
After the competition ends.

**Do I need to attend the workshop for my paper to be published?**
No. Publication does not require attendance.

**Our system did poorly. Should we still write a paper?**
Yes. In a task specifically about when visual features fail to transfer across
cultures, negative results are arguably the point.

---

# Communication

- **Questions about the data or task:** [open an issue](https://github.com/artemotion-28/data-public/issues) in this repository.
- **Announcements:** posted on the [task website](https://artemotion-28.github.io/).

---

# Citation

Until the task description paper is published, please cite the underlying
dataset:

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

---

# Organizers

| Organizer | Affiliation | Role |
|---|---|---|
| Youssef Mohamed | KAUST | Lead Organizer |
| Ibrahim Said Ahmad | UW–Stevens Point | Lead Organizer |
| Yunusa Haruna | CMVS, University of Oulu | Co-Organizer |
| Adamu Lawan | Beihang Univ. & Beijing GoerTek Alpha Labs | Co-Organizer |
| Faizan Farooq Khan | KAUST | Co-Organizer |
| Usman Naseem | Macquarie University | Co-Organizer |
| Shamsuddeen Hassan Muhammad | Imperial College London | Co-Organizer |
| Mohamed Elhoseiny | KAUST | Advisory Organizer |

Three organizers are co-authors of the ArtELingo-28 dataset paper, with direct
knowledge of the collection process. Several have prior shared-task organization
experience across AfriSenti-SemEval 2023, SemRel 2024, SemEval-2025 Task 11, and
SemEval-2026 Tasks 3 and 9.

---

# Ethical Statement

The dataset consists of publicly available artworks and anonymized multilingual
annotations. Annotator identifiers have been removed and no personally
identifiable information is retained. The task does not involve identifying or
profiling individuals, and any such use is outside its intended scope. The
organizers adhere to the [ACL Code of Ethics](https://www.aclweb.org/portal/content/acl-code-ethics).

---

# License

**This repository contains two kinds of material under different terms.**

**Annotations — CC BY 4.0.** Everything in `annotations/` (captions, emotion
labels, splits, metadata) is released by the ArtEmotion-28 organizers under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
Share and adapt freely, including commercially, with attribution.

**Images — not relicensed.** The paintings in `images/` are third-party works
sourced from [WikiArt](https://www.wikiart.org). They are **not** covered by the
CC BY 4.0 grant and are not relicensed by us. They are mirrored here purely so
participants need not re-scrape them; all rights remain with the original rights
holders. If you intend to use the images beyond research on this task, consult
the terms applying to each work at its source.

See [LICENSE](LICENSE) for the full statement.
