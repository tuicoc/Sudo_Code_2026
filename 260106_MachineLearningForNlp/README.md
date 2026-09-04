# Machine Learning for NLP — Naive Bayes vs SVM

Classic scikit-learn classifiers on Vietnamese product-review sentiment, compared
properly rather than picking one and assuming it is good.

Three combinations, not two: once it was clear the *vectorizer* choice mattered as much as
the model choice, all of NB+Bag-of-Words, NB+TF-IDF and SVM+TF-IDF had to be measured.

## Results (5-fold stratified CV, 3,040 reviews)

|  | NB + Bag-of-Words | NB + TF-IDF | SVM + TF-IDF |
|---|---|---|---|
| accuracy | 0.7895 ± 0.0159 | 0.7648 ± 0.0132 | **0.8112 ± 0.0055** |
| precision (macro) | 0.7807 ± 0.0165 | 0.7679 ± 0.0202 | **0.8040 ± 0.0059** |
| recall (macro) | 0.7789 ± 0.0153 | 0.7440 ± 0.0126 | **0.8017 ± 0.0059** |
| F1 (macro) | 0.7775 ± 0.0152 | 0.7329 ± 0.0126 | **0.8014 ± 0.0062** |

The `±` is the point of the table. A single train/test split on 3,040 reviews would put
the whole comparison at the mercy of which reviews happened to land where; five folds give
a mean *and* a spread, and a difference only counts if it is bigger than that spread.
SVM+TF-IDF wins on all four, and its folds are the most consistent.

## Two decisions that look like omissions

**Stopwords are not removed.** The Vietnamese stopword list contains "không" (not),
"chưa" (not yet), "rất" (very) and "quá" (too) — exactly the negation and intensifier
words that flip or scale a review's sentiment. Removing them turns "không tốt" (not good)
into "tốt" (good). `Preprocessor.stopwords_that_would_be_lost()` prints the four words the
decision is protecting, so it is evidence rather than an assertion.

**Run-together words are left alone.** "sảnphẩm" with a missing space is real in this
corpus — but so is "iPhone", and no rule separates the typo from the brand name without
breaking the brand name too.

## Layout

```
config/config.yaml        dataset id, regexes, hyperparameters, and the experiment list
src/config.py             loads config.yaml, resolves its paths against the project root
src/dataloader.py         DataLoader       -- download the reviews, read/write every data file
src/preprocessing.py      Preprocessor     -- NFC -> clean -> lowercase -> segment -> teencode
src/feature_extraction.py FeatureExtractor -- the Bag-of-Words / TF-IDF vectorizers compared
src/cross_validation.py   CrossValidator   -- stratified k-fold, per-fold metrics, summary
src/reporting.py          ResultsReporter  -- the comparison table and the bar chart
main.py                   the three stages, runnable separately or all at once
notebooks/                the experiment, narrated over four notebooks
data/                     downloads, cleaned corpus, metrics, chart (never committed)
note/                     learning log, and its PDF export
```

Adding a fourth method means adding an entry to `experiments:` in the config — no code
change. Each entry names a vectorizer, a model, a metrics-file key and a chart colour.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Nothing in `data/` is committed; the folders ship empty and `main.py` fills them.

[`tuannguyenvananh/vietnamese-text-classification-dataset`](https://www.kaggle.com/datasets/tuannguyenvananh/vietnamese-text-classification-dataset)
— 3,040 labelled Vietnamese product reviews, one CSV, downloaded by `kagglehub` on the
first run and cached to `data/raw/train.csv` so later runs do not depend on the network.
The Vietnamese stopword list comes from
[`heeraldedhia/stop-words-in-28-languages`](https://www.kaggle.com/datasets/heeraldedhia/stop-words-in-28-languages).

Both need Kaggle credentials once: `~/.kaggle/kaggle.json`, from your Kaggle account's
*Settings → API → Create New Token*. To download by hand instead, put the CSV at
`data/raw/train.csv` with a `label,comment` header.

## Run

```bash
python main.py                   # prepare -> train -> compare
python main.py --stage prepare   # download and clean the reviews
python main.py --stage train     # cross-validate every experiment in the config
python main.py --stage compare   # table + chart from the saved metrics
```

`prepare` is the slow stage: `underthesea` word segmentation over the whole corpus.

## Use the pieces directly

```python
from src.config import load_config
from src.dataloader import DataLoader
from src.preprocessing import Preprocessor

config = load_config()
preprocessor = Preprocessor(config, stopwords=DataLoader(config).load_stopwords())

preprocessor.process_text("Sản phẩm này KHÔNG tốt!!! Mua vs giá 250.000đ")
# ['sản_phẩm', 'này', 'không', 'tốt', 'mua', 'với', 'giá', 'đ']
#   compounds joined, "vs" expanded to "với", "không" kept
```

`notebooks/` holds the four-notebook experiment this package was extracted from:
`01_download_and_preprocess`, `02_naive_bayes`, `03_svm`, `04_compare`.
