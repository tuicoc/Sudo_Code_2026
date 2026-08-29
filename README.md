# Sudo_Code_2026

Personal NLP / ML learning repo. Each `260106_*` folder is one self-contained learning project — a
"nộp bài" (submission) — worked through step by step, with a running log of what was learned along
the way.

## Reference structure

Project layout is inspired by
[honestsoul/generative_ai_project](https://github.com/honestsoul/generative_ai_project):

```
config/       # settings, prompt templates, etc. (yaml)
src/          # source code, grouped by concern (llm/, utils/, handlers/, ...)
data/         # cache/, prompts/, outputs/, embeddings/
examples/     # small runnable scripts demonstrating one thing each
notebooks/    # exploratory / narrated notebooks
```

That template is built for a reusable LLM app (API clients, prompt templates, rate limiting) —
more machinery than a single-topic learning exercise needs, so it isn't copied wholesale. What
*is* borrowed, directly:

- **`data/` split by role**, not dumped flat — inputs kept separate from cache kept separate from
  generated output.
- **`notebooks/`** as where the actual work lives and gets narrated.
- **One thing per unit** — each notebook (their `examples/*.py`) does one job end to end.

## Project convention

Every `260106_*` project folder:

```
260106_TopicName/
├── requirements.txt      # only what this project needs, into the repo's shared .venv
├── Personal Note.md       # learning log — plan first, findings after
│                         #   (2+ note files → a note/ folder instead, see below)
├── data/
│   ├── raw/               # input as downloaded/scraped — nothing this project generated
│   ├── processed/         # cleaned/intermediate data this project's own logic produced
│   └── outputs/           # final artifacts: trained models, metrics, plots
└── notebooks/
    └── *.ipynb            # narrated work — no plain .py scripts
```

- **Name**: starts with `260106_`, then a short CamelCase topic, e.g. `260106_Word2Vec`.
- **`requirements.txt`**: a flat list, not a lockfile — `pip install -r requirements.txt`.
- **`Personal Note.md`**: write the step-by-step plan *before* starting, so the work can be
  followed one step at a time while learning. Keep it updated afterward with what actually
  happened — confusions, mistakes and why the fix was right, decisions and the reasoning. A
  journal, not a spec; prose, not a checklist restating the code.
- **`note/`**: as long as the learning log is the only note file, it sits at the project root as
  `Personal Note.md`. **As soon as there is a second note file** — an annotated course PDF, a marked-up
  book chapter, a slide deck read alongside the work — both move into a `note/` folder so the project
  root stays down to `requirements.txt`, `data/`, `notebooks/` and one folder of notes. Existing
  examples: `260106_MachineLearningForNlp/note/` and `260106_DeepLearningForNlp/note/`. Give the
  reading material a descriptive name, not whatever the download was called.
- **`data/{raw,processed,outputs}`**: not every project needs all three (a project that only
  consumes another project's finished output may have only `raw/`), but when a subfolder exists it
  means what it says. A file that arrives already preprocessed from an earlier project (e.g.
  `processed_news.parquet` from `260106_TextPreprocessingwithNLP`) still counts as `raw/` from the
  *consuming* project's own point of view — it's external input that project didn't generate
  itself, whatever its filename claims.
- **`notebooks/*.ipynb`**: single-technique project → one descriptively-named notebook
  (`word2vec_training.ipynb`). Multi-stage pipeline → one numbered notebook per stage
  (`01_download_and_preprocess.ipynb`, `02_naive_bayes.ipynb`, ...) so execution order is explicit.
  Always notebooks, never standalone `.py` scripts — keep this uniform across every project.

### Cross-project data flow

A project that consumes another project's output reads straight from that project's `data/`
folder — no package imports across `260106_*` folders, ever. When a project only reuses a
*technique* (not the actual data) from an earlier one, check that project's `Personal Note.md` and
notebook for the established approach and stay consistent with it rather than reinventing (e.g.
the same NFC-normalize → clean → lowercase → tokenize → stopword-removal shape for any new
Vietnamese text preprocessing).

### Shared tools already established

- **Dataset download**: `kagglehub.dataset_download(...)`, saved into the project's own
  `data/raw/`. `heeraldedhia/stop-words-in-28-languages` (`vietnamese.txt`) is the Vietnamese
  stopword list already in use — reuse it rather than sourcing a new one.
- **Vietnamese tokenization**: `nltk.word_tokenize` by default (fast, syllable-level). Real word
  segmentation (`underthesea`) is only worth its cost when the task needs whole compound words as
  single units (Word2Vec does; Bag-of-Words/TF-IDF topic classification tolerated the cheaper
  option) — see `260106_Word2Vec/Personal Note.md` and
  `260106_Scikit-learnTextFeatureExtraction/Personal Note.md` for the tradeoff.

## Projects

| Folder | Topic |
|---|---|
| [`260106_TextPreprocessingwithNLP`](260106_TextPreprocessingwithNLP) | Vietnamese news text preprocessing pipeline (NLTK book ch. 3 walkthrough) |
| [`260106_Scikit-learnTextFeatureExtraction`](260106_Scikit-learnTextFeatureExtraction) | TF-IDF feature extraction with scikit-learn |
| [`260106_Word2Vec`](260106_Word2Vec) | Word2Vec trained on Vietnamese Wikipedia (viwik18) with gensim |
| [`260106_MachineLearningForNlp`](260106_MachineLearningForNlp) | Naive Bayes vs. SVM on Vietnamese product-review sentiment classification |
| [`260106_DeepLearningForNlp`](260106_DeepLearningForNlp) | A first neural network (Keras) for 10-class Vietnamese news topic classification on VNTC |
| [`260106_SequentialModel`](260106_SequentialModel) | A word-level LSTM language model that generates Vietnamese, trained on a sample of 10,000 books |
