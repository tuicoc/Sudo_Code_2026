# Text Preprocessing with NLP

A Vietnamese news preprocessing pipeline, worked through as the NLTK book's
"Processing Raw Text" chapter applied to a real corpus rather than to toy strings.

Raw article text goes in; a clean, tokenized, stopword-free corpus comes out, saved as a
parquet file that [`260106_Scikit-learnTextFeatureExtraction`](../260106_Scikit-learnTextFeatureExtraction)
reads as its input.

## Pipeline

| Stage | Method | What it does |
|---|---|---|
| 0 | `Preprocessor.scan_noise` | counts emails, URLs, phone numbers, CDATA blocks, bare domains and press-license codes *before* any rule is written |
| 1 | `Preprocessor.normalize_unicode` | NFC, so precomposed and decomposed Vietnamese diacritics compare equal |
| 2 | `Preprocessor.clean_text` | removes the noise found in stage 0, plus digits and punctuation |
| 3 | `Preprocessor.fold_case` | lowercases, then expands the abbreviations in the config |
| 4 | `Preprocessor.tokenize` | NLTK word tokenizer (syllable-level on Vietnamese, by choice) |
| 5 | `Preprocessor.remove_stopwords` | drops tokens with little topical information |

## Layout

```
config/config.yaml   every default and hardcoded value: dataset ids, regexes, export columns
src/config.py        loads config.yaml, resolves its paths against the project root
src/dataloader.py    DataLoader     -- download the corpus and stopword list, read/write files
src/preprocessing.py Preprocessor   -- the five stages above
main.py              runs the whole pipeline
notebooks/           the experiment: the same pipeline, narrated step by step
data/                downloaded and generated files (never committed -- see below)
note/                learning log, and the book chapter it follows
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Nothing in `data/` is committed; the folders ship empty and the code fills them.

Both datasets are pulled from Kaggle on the first run by `kagglehub`, which needs Kaggle
credentials once (`~/.kaggle/kaggle.json`, from your Kaggle account's *Settings → API →
Create New Token*):

| Dataset | Used for | Lands in |
|---|---|---|
| [`haitranquangofficial/vietnamese-online-news-dataset`](https://www.kaggle.com/datasets/haitranquangofficial/vietnamese-online-news-dataset) (`news_dataset.json`) | the corpus | `data/raw/` |
| [`heeraldedhia/stop-words-in-28-languages`](https://www.kaggle.com/datasets/heeraldedhia/stop-words-in-28-languages) (`vietnamese.txt`) | stage 5 | the kagglehub cache |

To download by hand instead, put `news_dataset.json` in `data/raw/` and the run will use it.

## Run

```bash
python main.py                # download -> preprocess -> export
python main.py --scan-noise   # also print the stage-0 noise survey first
```

Output: `../260106_Scikit-learnTextFeatureExtraction/data/raw/processed_news.parquet`,
the input to the feature-extraction project. Cross-project data always moves as a file
on disk, never as an import.

## Use the pieces directly

```python
from src.config import load_config
from src.dataloader import DataLoader
from src.preprocessing import Preprocessor

config = load_config()
loader = DataLoader(config)
preprocessor = Preprocessor(config, stopwords=loader.load_stopwords())

preprocessor.process_text("Liên hệ: toasoan@vnexpress.net hoặc 0913940742 -- Xem tại http://vnexpress.net")
# ['liên', 'hệ']   -- noise gone, 'hoặc'/'xem'/'tại' dropped as stopwords
```

`notebooks/text_preprocessing.ipynb` is the experiment this package was extracted from:
it narrates each decision, shows the before/after of every stage on one article, and keeps
the "watch it fail" cells. `src/` is the same pipeline as code you can import, test and
deploy.
