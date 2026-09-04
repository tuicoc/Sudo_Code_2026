# Scikit-learn Text Feature Extraction

| | |
|---|---|
| **Goal** | Turn the preprocessed news corpus into TF-IDF features, and be able to *read* the numbers |
| **Dataset** | 184,539 articles, produced by [`260106_TextPreprocessingwithNLP`](../260106_TextPreprocessingwithNLP) |
| **Result** | unigram: **32,701 features**, 99.58% sparse · +bigram: **886,893 features**, 99.96% sparse |

This project does **no text processing of its own** — no cleaning, no tokenizing, no stopword
removal. All of that belongs to the preprocessing project; this one only consumes its output.

---

## 1. How to run

### What is in here

| File | What it does |
|---|---|
| `config/config.yaml` | `min_df`, the n-gram ranges to compare, which document to break down |
| `src/config.py` | Reads `config.yaml` and turns its paths into absolute paths |
| `src/dataloader.py` | `DataLoader` — read the parquet the preprocessing project writes |
| `src/feature_extraction.py` | `FeatureExtractor` — BoW → TF-IDF, plus the breakdown and summary tables |
| `main.py` | Fit every n-gram range and print the reports |
| `notebooks/feature_extraction.ipynb` | The experiment, narrated |

### Run

The input comes from the previous project:

```bash
cd ../260106_TextPreprocessingwithNLP && python main.py    # writes processed_news.parquet here
```

```bash
pip install -r requirements.txt
python main.py                # full corpus: 184,539 documents, a few minutes
python main.py --limit 2000   # first N documents, for a quick look
```

If the input file is missing, `DataLoader.load` says exactly which project produces it and what to
run — not just `FileNotFoundError`.

---

## 2. Results

| ngram_range | documents | vocabulary | sparsity |
|---|---|---|---|
| unigram (1,1) | 184,539 | 32,701 | 99.5783% |
| unigram + bigram (1,2) | 184,539 | 886,893 | 99.9617% |

Adding bigrams multiplies the vocabulary **27×** and makes the matrix sparser still. That cost is
only worth paying if the downstream task needs word adjacency (see the two "jobs" below).

**Reading one document term by term** — this is the actual output of `breakdown_table`, and the
reason BoW and TF-IDF are kept as two separate steps (`CountVectorizer`, then `TfidfTransformer`)
instead of one `TfidfVectorizer`:

| Term | BoW Count | IDF | TF-IDF |
|---|---|---|---|
| huế | 9 | 5.108 | 0.238 |
| công an | 9 | 3.211 | 0.150 |
| tp huế | 4 | 7.147 | 0.148 |
| chợ đông | 3 | 8.250 | 0.128 |
| an | 10 | 2.225 | 0.115 |

`an` occurs **most often** (10) and scores nearly lowest, because its IDF is 2.2 — it is everywhere,
so it distinguishes nothing. `chợ đông` occurs 3 times and beats it, because its IDF is 8.25. The
TF-IDF number only means something next to the two numbers it came from.

---

## 3. Experiments

| Tried | Result | Kept? |
|---|---|---|
| Tokenize with `underthesea` in this notebook | 44 ms/article → ~2 hours for 184,539 articles | No |
| Speed it up with `ProcessPoolExecutor` in a notebook | Fails — macOS `spawn` needs the worker importable from a real module | No |
| Swap to `nltk.word_tokenize` here instead | Fast enough, but still the wrong project for the decision | No — see below |
| Take whatever the preprocessing project hands over | One tokenizer decision, made once, inherited here | **Yes** |
| `TfidfVectorizer` (one step) | Hides the counts the weight came from | No — `CountVectorizer` + `TfidfTransformer` |
| Bigrams only | Throws away meaningful standalone words like `cao` | No — `(1,2)` mixes both |

**The tokenizer story is really an architecture story.** My first fix for underthesea being slow was
to swap in NLTK and keep tokenizing here. That solved the speed problem and missed the bigger one:
tokenization is a *text processing* decision, and text processing already has its own project. Doing
it here duplicated logic and made two projects responsible for the same choice. The real fix was to
delete it from this project entirely. If underthesea is ever worth paying for, that change happens in
the other project and this one does not change at all.

---

## 4. What I learned

**Word segmentation is not the same thing as an n-gram.** I confused these two because they share the
word "gram". They happen at different stages and mean different things:

| | What it does | Example |
|---|---|---|
| Segmentation | Decides where one *word* ends. A property of the language, not a setting I pick. | `sản phẩm` → `sản_phẩm` (still **1 token**) |
| N-gram | Joins two *already-complete words* that sit next to each other. A vectorizer setting. | `chất_lượng` + `cao` → `chất_lượng_cao` |

`vô_tuyến_truyền_hình` is four syllables and **one** token. Syllable count is not "n".

**A bigram has two separate jobs, and segmentation only removes one of them.**

- *Job 1 — patching a broken segmentation.* With syllable-level tokenizing, `kinh tế` becomes two
  meaningless pieces, and the bigram `kinh_tế` reconstructs the word by brute force. This job
  **does** disappear once a real segmenter is in the pipeline.
- *Job 2 — capturing a relationship between two complete words.* `không` and `thích` are each a real,
  separate word; a segmenter should never merge them. But a unigram model sees `thích` and reads it
  as positive, losing the negation. **This job never disappears.**

Which one you need depends on the task: topic classification leans on keyword presence, so unigrams
are already strong; sentiment needs Job 2, so bigrams stay worth it. (That is exactly the choice
`260106_MachineLearningForNlp` had to make.)

**Keep BoW and TF-IDF as two steps while you are still learning to read them.** The count and the IDF
are what make the weight interpretable. `an` scoring low with the highest count is the whole lesson.

**Scope belongs to one project.** The most useful change here was *deleting* work, not adding it.

---

## 5. Restructure (2026-09-01)

Mentor feedback: a notebook is not a project. Added `src/`, `config/`, `README.md` and pinned
`requirements.txt`; `data/` now holds only the empty folder skeleton plus download instructions.

Small project, so `src/` stayed small: `DataLoader` and `FeatureExtractor`. Two things came out of
the extraction:

- The notebook's `build_tfidf` returned a bare 4-tuple every caller had to unpack positionally. It
  is now a `Features` dataclass, so nothing depends on argument order.
- `sparsity` was the same `1 - nnz / (rows * cols)` written out **four** times — once per n-gram
  range, and twice more inline in the summary table. I only counted them after moving the code into
  one file; spread across cells, the repetition did not read as repetition. It is one property now.
