# Scikit-learn Text Feature Extraction

Turning the preprocessed Vietnamese news corpus into TF-IDF features, and reading the
numbers rather than just producing them.

This project does **no text processing of its own** — no cleaning, no tokenizing, no
stopword removal. All of that is
[`260106_TextPreprocessingwithNLP`](../260106_TextPreprocessingwithNLP)'s job; this one
consumes its finished output and applies vectorization on top.

## What it does

Bag-of-Words and TF-IDF are kept as two explicit steps (`CountVectorizer`, then
`TfidfTransformer`) instead of the single `TfidfVectorizer` that does both, because the raw
count is what makes the weight readable: a TF-IDF score only means something next to how
often the term actually occurred and how rare it is across the corpus.

`FeatureExtractor.breakdown_table` puts those three numbers side by side for one document:

```
Term  BoW Count      IDF   TF-IDF
 huế          9 4.047525 0.372629
  an         10 2.150932 0.220024
 chợ          5 3.874015 0.198142
```

Two n-gram ranges are fitted and compared — unigrams alone, and unigrams+bigrams — to show
what the second one costs in vocabulary size and sparsity.

## Layout

```
config/config.yaml       min_df, the n-gram ranges to compare, which document to break down
src/config.py            loads config.yaml, resolves its paths against the project root
src/dataloader.py        DataLoader       -- read the corpus the preprocessing project wrote
src/feature_extraction.py FeatureExtractor -- BoW -> TF-IDF, plus the breakdown and summary tables
main.py                  fit every n-gram range and print the reports
notebooks/               the experiment, narrated
data/raw/                the input parquet (never committed -- see below)
Personal Note.md         learning log
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Nothing in `data/` is committed; the folder ships empty.

The input, `data/raw/processed_news.parquet`, is produced by the preprocessing project:

```bash
cd ../260106_TextPreprocessingwithNLP
pip install -r requirements.txt
python main.py            # downloads the news corpus and writes the parquet here
```

That project downloads its own source data from Kaggle, so there is nothing to fetch by
hand. `DataLoader.load` says exactly this if the file is missing.

## Run

```bash
python main.py                # the whole corpus: 184,539 documents, a few minutes
python main.py --limit 2000   # the first N documents, for a quick look
```

## Use the pieces directly

```python
from src.config import load_config
from src.dataloader import DataLoader
from src.feature_extraction import FeatureExtractor

config = load_config()
corpus = DataLoader(config).load_corpus()

extractor = FeatureExtractor(config)
fitted = extractor.fit_all(corpus)          # {"unigram": Features, "unigram_bigram": Features}

extractor.summary(fitted)                   # size and sparsity per n-gram range
extractor.breakdown_table(fitted["unigram"]) # one document, term by term
```

`notebooks/feature_extraction.ipynb` is the experiment this package was extracted from:
it works through why word segmentation is not the same thing as an n-gram, and what the
sparsity number is really saying.
