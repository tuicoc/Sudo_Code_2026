# Deep Learning for NLP — a first neural network for text classification

A one-hidden-layer Keras network over TF-IDF features, classifying Vietnamese news into
10 topics on **VNTC** (33,759 training / 50,373 test articles).

The question the project exists to answer: *does a simple neural network beat the classic
TF-IDF + linear model from [`260106_MachineLearningForNlp`](../260106_MachineLearningForNlp),
on a task where a published baseline exists to check both against?*

## Results (VNTC official test set, 50,373 articles)

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

Yes — but by 0.5 points of accuracy over LinearSVC, for **twelve times the training
time**, and still short of the 2007 published SVM. Word segmentation is worth about the
same 0.9 points to every model, which is why both axes are measured: without the
unsegmented column, the network's score could just as easily be credit owed to
`underthesea`.

The model's real weakness is visible in the confusion matrix, not the headline number:
*Đời sống* (lifestyle) is the one topic it cannot hold, leaking 11.6% to *Văn hóa* and
10.1% to *Chính trị Xã hội* — categories that genuinely overlap in the source.

## The trap this corpus sets

The raw files are **UTF-16LE**, and reading them as UTF-8 does not raise. It returns

```
b'\xff\xfe \x00T\x00h\x00\xe0\x00n\x00h\x00 \x00l\x00\xad\x1ep\x00 ...'
```

— the text with a null byte after every character. A vectorizer builds a vocabulary out of
that quite happily, training runs, the loss falls, and the model has learned nothing.
Nothing warns you. That silence is why the encoding is pinned in `config/config.yaml`
rather than left to a default, and why `notebooks/01_prepare_data.ipynb` opens by pointing
TensorFlow at the raw corpus and watching it fail.

## Layout

```
config/config.yaml        encoding, hyperparameters, file paths, published baselines
src/config.py             loads config.yaml, resolves its paths against the project root
src/dataloader.py         DataLoader       -- read the corpus and every intermediate artifact
src/preprocessing.py      Preprocessor     -- UTF-16 -> NFC -> segmented UTF-8 (parallel, resumable)
src/feature_extraction.py FeatureExtractor -- the segmentation-aware tokenizer and TF-IDF
src/model.py              TopicClassifier  -- build, train, save, predict
src/evaluation.py         Evaluator        -- metrics, floors, confusion matrix, learning curves
src/comparison.py         ModelComparison  -- network vs SVM vs NB, segmented vs unsegmented
main.py                   the five stages, runnable separately or all at once
notebooks/                the experiment, narrated over five notebooks
data/                     corpus, features, model, metrics, figures (never committed)
note/                     learning log, and the course PDF it follows
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Nothing in `data/` is committed; the folders ship empty.

**VNTC** — [duyvuleo/VNTC](https://github.com/duyvuleo/VNTC) (MIT), the corpus from Vu et
al., *A Comparative Study on Vietnamese Text Classification Methods*, RIVF 2007. It is not
downloadable by script: the archives live in the repo as `.rar`, so fetch them once by
hand.

```bash
git clone https://github.com/duyvuleo/VNTC.git /tmp/VNTC
# 10Topics Ver1.1 -- the level with published baselines to compare against
unar -o data/raw /tmp/VNTC/Data/10Topics/Ver1.1/Train_Full.rar
unar -o data/raw /tmp/VNTC/Data/10Topics/Ver1.1/Test_Full.rar
```

`unar` comes from `brew install unar`. The result must look like:

```
data/raw/Train_Full/<class name>/<article>.txt      33,759 files, 10 classes
data/raw/Test_Full/<class name>/<article>.txt       50,373 files, 10 classes
```

`main.py --stage prepare` then converts those into `data/processed/`, mirroring the same
folder layout as word-segmented UTF-8. That stage is the slow one (`underthesea` over
84,132 articles) — it runs in parallel and is **resumable**: an article whose output
already exists is skipped, so an interrupted run continues where it stopped.

## Run

```bash
python main.py                   # all five stages
python main.py --stage prepare   # UTF-16 -> NFC -> word-segmented UTF-8
python main.py --stage features  # TF-IDF matrices + label encoding, cached to data/processed/
python main.py --stage train     # fit the network, save model + history + metadata
python main.py --stage evaluate  # test metrics, confusion matrix, learning curves
python main.py --stage compare   # network vs SVM vs NB, segmented vs unsegmented
```

## Use the pieces directly

```python
from src.config import load_config
from src.dataloader import DataLoader
from src.evaluation import Evaluator
from src.model import TopicClassifier

config = load_config()
loader = DataLoader(config)

X_test, y_test = loader.load_features("test")
classifier = TopicClassifier(config)
classifier.load(config.path("paths.model_file"))

evaluator = Evaluator(config, loader.load_class_names())
evaluator.score(y_test, classifier.predict(X_test))["macro_f1"]   # 0.9087
```

`notebooks/` holds the five-notebook experiment this package was extracted from:
`01_prepare_data`, `02_text_vectorization`, `03_train_neural_network`, `04_evaluate`,
`05_compare`.
