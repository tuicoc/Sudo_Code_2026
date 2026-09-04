# Deep Learning for NLP — a first neural network

| | |
|---|---|
| **Goal** | Does a simple neural network beat TF-IDF + a linear model, on a task with a published baseline? |
| **Dataset** | VNTC 10Topics — 33,759 train / 50,373 test Vietnamese news articles ([GitHub](https://github.com/duyvuleo/VNTC)) |
| **Result** | **accuracy 0.9266, macro-F1 0.9087** — beats LinearSVC by 0.5 points, costs 12× the training time |

---

## 1. How to run

### What is in here

| File | What it does |
|---|---|
| `config/config.yaml` | Encoding, hyperparameters, file paths, published baselines |
| `src/config.py` | Reads `config.yaml` and turns its paths into absolute paths |
| `src/dataloader.py` | `DataLoader` — read the corpus and every intermediate artifact |
| `src/preprocessing.py` | `Preprocessor` — UTF-16 → NFC → word-segmented UTF-8 (parallel, resumable) |
| `src/feature_extraction.py` | `FeatureExtractor` — the segmentation-aware tokenizer and TF-IDF |
| `src/model.py` | `TopicClassifier` — build, train, save, predict |
| `src/evaluation.py` | `Evaluator` — metrics, floors, confusion matrix, learning curves |
| `src/comparison.py` | `ModelComparison` — NN vs SVM vs NB, segmented vs unsegmented |
| `main.py` | 5 stages: `prepare` → `features` → `train` → `evaluate` → `compare` |
| `notebooks/01..05` | The experiment, one notebook per step |

### Run

VNTC is not downloadable by script — the archives are `.rar` inside the repo:

```bash
git clone https://github.com/duyvuleo/VNTC.git /tmp/VNTC
unar -o data/raw /tmp/VNTC/Data/10Topics/Ver1.1/Train_Full.rar
unar -o data/raw /tmp/VNTC/Data/10Topics/Ver1.1/Test_Full.rar
```

```bash
pip install -r requirements.txt
python main.py                   # all 5 stages
python main.py --stage prepare   # UTF-16 → NFC → segmented UTF-8 (~28 min, resumable)
python main.py --stage features  # TF-IDF + label encoding, cached
python main.py --stage train     # fit the network (~70 s)
python main.py --stage evaluate  # metrics, confusion matrix, learning curves
python main.py --stage compare   # NN vs SVM vs NB, segmented vs unsegmented
```

> This project has its **own `.venv`**. The machine is an Intel Mac, and TensorFlow stopped
> publishing macOS x86_64 wheels after 2.16.2, which pins `numpy<2.0` — installing it into the
> shared venv would downgrade numpy for every other project.

---

## 2. Results

| Model | Input | Accuracy | Macro-F1 | Train |
|---|---|---|---|---|
| **Neural network** | segmented | **0.9266** | **0.9087** | 70 s |
| LinearSVC | segmented | 0.9219 | 0.9015 | 6 s |
| MultinomialNB | segmented | 0.8907 | 0.8677 | 0.1 s |
| Neural network | unsegmented | 0.9174 | 0.8972 | 80 s |
| LinearSVC | unsegmented | 0.9154 | 0.8945 | 7 s |
| MultinomialNB | unsegmented | 0.8544 | 0.8277 | 0.1 s |
| *floor: always largest class* | — | 0.1502 | 0.0261 | — |
| *floor: uniform random* | — | 0.1016 | 0.0977 | — |
| *published RIVF'07 SVM Multi* | — | 0.9340 | — | — |
| *published RIVF'07 NGRAM* | — | 0.9710 | — | — |

**Did the network win? Yes — narrowly.** +0.47 accuracy, +0.72 macro-F1 over LinearSVC on identical
features, for **12× the training time**. "The network won" and "the network was worth it" are two
different claims and only the first one is supported. On 10,000-dimensional bag-of-words features,
topic classification is nearly linearly separable, so a hidden layer has very little structure left
to find.

**Where the errors are.** `Đời sống` (lifestyle) is the one class the model cannot hold — F1 0.7347
against 0.9823 for `Thể thao`. It leaks 11.6% to `Văn hóa` and 10.1% to `Chính trị Xã hội`,
categories that genuinely overlap in the source.

**Caveat on every number in the table** (ours and the published ones): 5.0% of the test set is
byte-identical to a training file, so all scores are inflated by memorization. 219 test files are
duplicated across *different* classes, which puts the accuracy ceiling at 99.57%. The official split
was kept anyway — silently deduplicating would make the score incomparable with everyone else's.

---

## 3. Experiments

| Tried | Result | Kept? |
|---|---|---|
| Read the corpus as UTF-8 | Does **not** raise — returns text with a null byte after every character | No — pin `utf-16` |
| Skip `underthesea` (syllables only) | Costs 0.9 accuracy for the NN, 3.6 for NB | No — segmentation is worth 28 min |
| `Embedding` → `GlobalAveragePooling1D` (the tutorial) | Averages the sequence and discards word order anyway | No — switched to TF-IDF |
| Bigrams on top of segmentation | Segmentation already produces `kinh_doanh`; bigrams pay twice | No |
| Keras `validation_split=0.2` | Corpus is ordered by class, so the tail is a handful of classes | No — stratified split |
| `max_features` 10k vs 20k, `min_df` 3 vs 5 | Whole sweep spans **0.001 macro-F1** | Keep the defaults |
| `class_weight="balanced"` alone | 0.9176 — *worse* than baseline 0.9198 | No |
| `sublinear_tf` + `class_weight` together | 0.9226, best variant, +0.28 on validation only | No — inside single-seed noise |
| Bigger network (512, or 512→256) | Both scored **below** 256 units | No |
| Pretrained Word2Vec embeddings from `260106_Word2Vec` | 93% token coverage after matching preprocessing — viable | Sketched as future work |

**The segmentation table is the most informative result:**

| Model | Δ accuracy | Δ macro-F1 |
|---|---|---|
| Neural network | +0.92 | +1.15 |
| LinearSVC | +0.65 | +0.70 |
| MultinomialNB | +3.63 | +4.00 |

The weaker the model, the more it depends on the features being right. NB treats every feature as
independent evidence, so `kinh` + `doanh` as two ambiguous signals hurt it badly. Stronger models
partly compensate by weighting co-occurring fragments, so they gain less.

---

## 4. What I learned

**Silent failure is worse than a crash.** Reading UTF-16 files as UTF-8 raises nothing. It returns
`b'\xff\xfe \x00T\x00h...'`, a vectorizer builds a vocabulary out of it, training runs, loss falls,
and the model has learned nothing. Notebook 01 opens by pointing TensorFlow at the raw corpus and
watching it fail, because that is the part worth seeing.

**Choose the representation before the architecture.** I first argued for skipping `underthesea` —
and each argument was true, but they all assumed a bag-of-embeddings model where averaging blurs the
input. Once the representation became TF-IDF, where the vocabulary *is* the feature space, the
conclusion flipped. The measurement then confirmed it. Being wrong in a traceable way was the useful
part.

**Always compute the floor.** "0.9266 accuracy" means nothing until you know that always guessing
the largest class scores 0.1502. Now `Evaluator.baseline_scores()` prints both floors before the
real number.

**Check the split, not just the score.** Validation accuracy came out implausibly high once, which
is what caught Keras's `validation_split` taking the *last* 20% of a class-ordered corpus. Stratify.

**Macro-F1, not accuracy, when train and test have different class proportions.** `Đời sống` is 9.4%
of train and 4.0% of test. A model that learns the training prior gets rewarded or punished by that
shift, which is exactly where plain accuracy misleads.

**Early stopping earned its place.** Final epoch: train accuracy 0.9882 vs validation 0.9163 — a
7-point gap. Best validation loss was at epoch 4, training ran to 7, and `restore_best_weights=True`
rewound. A fixed 10 epochs would have shipped a visibly worse model.

**Know when tuning is finished.** Everything reachable from this representation lands within ~0.3
points. Getting past ~0.92 means changing the representation (pretrained embeddings, a transformer),
not turning knobs. The sweep is the evidence for that, not an assumption.

**Check "junk" before calling it junk.** High-IDF tokens I wrote off as crawler noise turned out to
be `kcb` (khám chữa bệnh), `larry` (a name in a quoted interview), and `value` (inside a code snippet
in a real `Vi tinh` article). The corpus was far cleaner than assumed — 41 URLs and zero HTML tags in
a 3,000-article sample — so the repo's usual cleaning regexes were not applied wholesale.

---

## 5. Restructure (2026-09-01)

Mentor feedback: a notebook is not a project. Added `src/`, `config/`, `README.md` and pinned
`requirements.txt`; `data/` now holds only the empty folder skeleton plus download instructions.

`vn_tokenizer` was defined **twice** — in `02_text_vectorization.ipynb` and again in
`05_compare.ipynb` — and the vectorizers around them had drifted: the second dropped
`strip_accents=None`. I checked before writing this down, and it changes nothing, since that *is*
sklearn's default. What was lost is the intent: in notebook 02 the argument is written out with an
explanation that accents must never be stripped from Vietnamese; in notebook 05 the protection is
invisible, resting on a default nobody stated. It is now a config key with the reason beside it —
the version that survives someone tidying up "redundant" arguments.

Re-ran `evaluate` and the segmented half of `compare` afterwards: accuracy 0.9266, macro-F1 0.9087,
LinearSVC 0.9219/0.9015, MultinomialNB 0.8907/0.8677 — all identical to the notebooks.
