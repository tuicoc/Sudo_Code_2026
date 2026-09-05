# Sudo_Code_2026

Personal NLP / ML learning repo. Each `260106_*` folder is one self-contained learning
project — a "nộp bài" (submission) — worked through step by step, with a running log of
what was learned along the way.

Every project ships as a **runnable repository**, not just a notebook: the pipeline lives
in `src/` as importable classes, every constant lives in `config/`, and the project's own
`README.md` says how to install it, where to get the data, and how to re-run it.

## Reference structure

Layout follows
[honestsoul/generative_ai_project](https://github.com/honestsoul/generative_ai_project):

```
config/       # every default and hardcoded value, as yaml
src/          # the implementation -- one small file per pipeline stage, one class each
data/         # inputs and generated artifacts; never committed, downloaded on first run
notebooks/    # the experiments, narrated
README.md     # install, data, run
requirements.txt  # pinned versions
```

## Project convention

Every `260106_*` project folder:

```
260106_TopicName/
├── README.md              # install, where the data comes from, how to re-run
├── requirements.txt       # pinned versions -- pandas==3.0.5, not pandas
├── main.py                # entry point; multi-stage projects take --stage
├── config/
│   └── config.yaml        # every default and hardcoded value used by src/
├── src/
│   ├── config.py          # loads config.yaml, resolves its paths against the project root
│   ├── dataloader.py      # DataLoader     -- downloads, reads, writes
│   ├── preprocessing.py   # Preprocessor   -- the cleaning/tokenizing pipeline
│   └── ...                # feature_extraction.py, model.py, evaluation.py, as the project needs
├── data/
│   ├── raw/               # input as downloaded -- nothing this project generated
│   ├── processed/         # cleaned/intermediate data this project's own logic produced
│   └── outputs/           # final artifacts: trained models, metrics, plots
├── notebooks/
│   └── *.ipynb            # the experiment, narrated
└── Personal Note.md       # learning log -- plan first, findings after
                           #   (2+ note files → a note/ folder instead, see below)
```

- **Name**: starts with `260106_`, then a short CamelCase topic, e.g. `260106_Word2Vec`.
- **`src/`**: the part that matters. Small files, one class each, named after the pipeline
  stage they own. A class holds the smaller functions of that stage. Written so that a
  developer who knows nothing about the project can open one file and read it — which is
  also what makes a bug land in one function instead of somewhere in a notebook.
- **`config/config.yaml`**: dataset ids, regexes, hyperparameters, file paths, figure
  settings. If a value is hardcoded in `src/`, it is in the wrong place. `src/config.py` is
  the same small loader in every project (`load_config()`, `.require()`, `.path()`), so
  code runs identically from `main.py`, a notebook, or a shell.
- **`requirements.txt`**: pinned — `pip list --format=freeze` against the venv the project
  actually ran in. A flat list, not a lockfile.
- **`main.py`**: reproduces the whole project. Multi-stage projects expose `--stage`, and
  each stage skips work whose output already exists, so an interrupted run resumes.
- **`data/`**: **never committed.** The folders ship as empty skeletons (`.gitkeep`), and
  the project README says how to obtain the data — a `kagglehub` call the code makes
  itself, or manual unpack instructions when the source is not scriptable. Ship the *way
  to get* the data, not the data.
- **`notebooks/*.ipynb`**: where the work is narrated and the experiments live. Single
  technique → one descriptively-named notebook (`word2vec_training.ipynb`). Multi-stage
  pipeline → one numbered notebook per stage (`01_prepare_data.ipynb`, ...), so execution
  order is explicit. Notebooks are for experiments; `src/` is what deploys.
- **`Personal Note.md`**: the learning log, and **every project uses the same five sections**, so
  anyone can open any project and find the same thing in the same place:

  | Section | What goes in it |
  |---|---|
  | header table | Goal, dataset, headline result — three lines |
  | `1. How to run` | What each file does, then the commands to run it |
  | `2. Results` | The real numbers, and a few sentences on how to read them |
  | `3. Experiments` | A table: what was tried, what happened, was it kept |
  | `4. What I learned` | The lessons worth carrying to the next project |

  Write the plan before starting, then keep sections 2–4 updated with what actually happened —
  confusions, mistakes, and why a fix was right. Plain language and tables, not essays.
- **`note/`**: while the learning log is the only note file, it sits at the project root as
  `Personal Note.md`. **As soon as there is a second note file** — an annotated course PDF,
  a marked-up book chapter, a slide deck — both move into a `note/` folder. Give the
  reading material a descriptive name, not whatever the download was called.
- **`data/{raw,processed,outputs}`**: not every project needs all three, but when a
  subfolder exists it means what it says. A file that arrives already preprocessed from an
  earlier project (e.g. `processed_news.parquet` from `260106_TextPreprocessingwithNLP`)
  still counts as `raw/` from the *consuming* project's point of view — it is external
  input that project did not generate, whatever its filename claims.

### Cross-project data flow

A project that consumes another project's output reads straight from that project's
`data/` folder — **no package imports across `260106_*` folders, ever**. When a project
reuses a *technique* rather than the data, check that project's `Personal Note.md` and
`src/` for the established approach and stay consistent with it rather than reinventing
(e.g. the same NFC-normalize → clean → lowercase → tokenize → stopword-removal shape for
any new Vietnamese text preprocessing).

### Shared tools already established

- **Dataset download**: `kagglehub.dataset_download(...)`, into the project's own
  `data/raw/`. `heeraldedhia/stop-words-in-28-languages` (`vietnamese.txt`) is the
  Vietnamese stopword list already in use — reuse it rather than sourcing a new one.
- **Vietnamese tokenization**: `nltk.word_tokenize` by default (fast, syllable-level). Real
  word segmentation (`underthesea`) is only worth its cost when the task needs whole
  compound words as single units — Word2Vec does, and the VNTC classifier gains about 0.9
  points from it; Bag-of-Words / TF-IDF topic classification tolerated the cheaper option.
  See `260106_Word2Vec/README.md` and
  `260106_Scikit-learnTextFeatureExtraction/Personal Note.md`.

## Projects

| Folder | Topic | Headline result |
|---|---|---|
| [`260106_TextPreprocessingwithNLP`](260106_TextPreprocessingwithNLP) | Vietnamese news preprocessing pipeline (NLTK book ch. 3) | feeds the next project |
| [`260106_Scikit-learnTextFeatureExtraction`](260106_Scikit-learnTextFeatureExtraction) | TF-IDF feature extraction with scikit-learn | BoW → TF-IDF, read term by term |
| [`260106_Word2Vec`](260106_Word2Vec) | Word2Vec on Vietnamese Wikipedia (viwik18) with gensim | 6.6 M sentences, 414 K vocabulary |
| [`260106_MachineLearningForNlp`](260106_MachineLearningForNlp) | Naive Bayes vs. SVM on Vietnamese review sentiment | SVM+TF-IDF, macro-F1 0.801 |
| [`260106_DeepLearningForNlp`](260106_DeepLearningForNlp) | A first neural network (Keras) for 10-class VNTC topic classification | accuracy 0.9266, macro-F1 0.9087 |
| [`260106_SequentialModel`](260106_SequentialModel) | A word-level LSTM language model generating Vietnamese | val perplexity 220 vs 20,000 uniform |
| [`260106_AttentionIsAllYouNeed`](260106_AttentionIsAllYouNeed) | Attention written by hand, on Vietnamese news summarization (VNDS) | attention +1.05 ROUGE-1; Lead-1 baseline still wins |
| [`260106_TransformerModel`](260106_TransformerModel) | Encoder–decoder Transformer for EN→VI translation (EVBCorpus) | BLEU 16.57; cross-attention matches human word alignment 40.9% vs 5.1% chance |
