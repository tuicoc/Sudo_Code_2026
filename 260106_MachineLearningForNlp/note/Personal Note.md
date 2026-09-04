# Machine Learning for NLP — Naive Bayes vs SVM

| | |
|---|---|
| **Goal** | Compare classic classifiers properly, instead of picking one and assuming it is good |
| **Dataset** | 3,040 Vietnamese product reviews, 3 classes ([Kaggle](https://www.kaggle.com/datasets/tuannguyenvananh/vietnamese-text-classification-dataset)) |
| **Result** | **SVM + TF-IDF wins: macro-F1 0.8014 ± 0.0062** (5-fold CV) |

---

## 1. How to run

### What is in here

| File | What it does |
|---|---|
| `config/config.yaml` | Dataset id, regexes, hyperparameters, and the list of experiments to run |
| `src/config.py` | Reads `config.yaml` and turns its paths into absolute paths |
| `src/dataloader.py` | `DataLoader` — download the reviews, read/write every data file |
| `src/preprocessing.py` | `Preprocessor` — NFC → clean → lowercase → segment → expand teencode |
| `src/feature_extraction.py` | `FeatureExtractor` — builds the Bag-of-Words / TF-IDF vectorizers |
| `src/cross_validation.py` | `CrossValidator` — stratified 5-fold, per-fold metrics, mean ± std |
| `src/reporting.py` | `ResultsReporter` — the comparison table and the bar chart |
| `main.py` | 3 stages: `prepare` → `train` → `compare` |
| `notebooks/01..04` | The experiment, one notebook per step |

### Run

```bash
pip install -r requirements.txt
python main.py                   # prepare → train → compare
python main.py --stage prepare   # download + clean the reviews
python main.py --stage train     # cross-validate every experiment in the config
python main.py --stage compare   # table + chart from the saved metrics
```

Needs a Kaggle token the first time. `prepare` is the slow stage (`underthesea` over the corpus,
~13 s). Everything runs in about a minute.

**Adding a 4th method is a config entry, not new code** — `experiments:` in `config.yaml` lists
(vectorizer, model, output key, chart colour) per row and `main.py` loops over it.

---

## 2. Results

5-fold stratified CV, mean ± std across folds:

| | accuracy | precision (macro) | recall (macro) | **F1 (macro)** |
|---|---|---|---|---|
| NB + Bag-of-Words | 0.7895 ± 0.0159 | 0.7807 ± 0.0165 | 0.7789 ± 0.0153 | 0.7775 ± 0.0152 |
| NB + TF-IDF | 0.7648 ± 0.0132 | 0.7679 ± 0.0202 | 0.7440 ± 0.0126 | 0.7329 ± 0.0126 |
| **SVM + TF-IDF** | **0.8112 ± 0.0055** | **0.8040 ± 0.0059** | **0.8017 ± 0.0059** | **0.8014 ± 0.0062** |

Three things this table says:

1. **SVM + TF-IDF is best on every metric**, and its fold-to-fold spread is about a third of
   NB+BoW's — more consistent, not just higher on average.
2. **The textbook claim "Naive Bayes wants counts, not TF-IDF" held up.** BoW beat TF-IDF for
   `MultinomialNB` by ~4.5 points of macro-F1, which is bigger than either one's noise. Worth having
   actually measured rather than repeating on faith.
3. **The `±` is the point.** With only 3,040 rows, one 80/20 split would have made this comparison
   mostly luck about which reviews landed where. A difference only counts if it beats the spread.

Chart: `data/outputs/comparison.png`.

---

## 3. Experiments

| Tried | Result | Kept? |
|---|---|---|
| Remove stopwords (as the earlier projects do) | The list contains `không`, `chưa`, `rất`, `quá` — the words that flip sentiment | **No** — kept them |
| Unigrams only | `không thích` becomes `không` + `thích`; `thích` alone still reads positive | No — use `ngram_range=(1,2)` |
| Negation tagging (proper scope detection) | Needs a clause-boundary rule, but punctuation is already stripped by then | No — bigrams get most of it |
| One 80/20 train/test split | On 3,040 rows the result is mostly noise | No — `StratifiedKFold(5)` |
| Fit the vectorizer once on the whole corpus | Leaks validation vocabulary into training | No — fit inside each fold |
| Teencode: `k`, `ko`, `dc`, `đc`, `sp`, `mn` | All at or near zero in this corpus — it is standard Vietnamese | Only `vs` → `với` |
| Fix glued words (`hayXài`) | 2.4% of rows, but identical at character level to real brand casing (`tikiNow`) | No — documented as a limit |

**Dataset took 3 tries.** A single job posting (not enough documents), then AIVIVN 2019 (real, fully
built against — dropped, but the *method* of surveying noise before writing rules carried over),
then UIT-ViSFD (real, but multi-label multi-aspect — too broad a problem for a first comparison).

---

## 4. What I learned

**The right preprocessing depends on the task, not on habit.** Removing stopwords is correct for
topic classification and destructive for sentiment. Same corpus language, same tool, opposite answer.
This is now `Preprocessor.stopwords_that_would_be_lost()`, which prints the four words the decision
protects, so the reasoning shows itself on every run instead of living in a comment.

**Report a spread, not just a mean.** Three methods on 3,040 rows: without the std, I could not have
told a real 4.5-point gap from fold noise.

**Fit the vectorizer inside the fold.** Vocabulary and document frequencies are *learned* from data,
so fitting them on everything before splitting leaks. Cleaning and tokenizing are per-row rules with
nothing learned, so those can stay outside.

**A hand-written Unicode range is not "Vietnamese letters".** My first glued-word detector used
`[a-zà-ỹ]` and matched 96% of rows, because `à` (U+00E0) to `ỹ` (U+1EF9) spans several unrelated
blocks. Using Python's own `str.islower()`/`isupper()` per character gave the real number: 2.4%.

**81% of the stopword list is multi-word phrases** (`"nói chung"`), which `underthesea` joins into
`nói_chung`. Without adding the underscore form to the set, those would have silently stopped being
filtered the moment the tokenizer changed — no crash, just a quiet regression.

**`LinearSVC` handles 3 classes with no extra code.** It defaults to One-vs-Rest, so `.fit()` trains
3 hyperplanes. Verified rather than assumed: `coef_.shape` came out `(3, 10929)` and
`decision_function` returned 3 scores per sample. (`SVC` is different — always One-vs-One.)

**Macro averaging, not accuracy.** It scores each class separately then averages unweighted, so the
smallest class (Neutral, ~29%) cannot be swamped by the largest (Negative, ~36%).

---

## 5. Restructure (2026-09-01)

Mentor feedback: a notebook is not a project. Added `src/`, `config/`, `README.md` and pinned
`requirements.txt`; `data/` now holds only the empty folder skeleton plus download instructions.

The extraction found a real problem: **`run_cv` and `summarize` were duplicated in
`02_naive_bayes.ipynb` and `03_svm.ipynb`, and had already drifted** — the SVM copy took a `model`
argument that the NB copy hardcoded. Nothing broke, because each notebook only used its own copy.
That is exactly the failure the mentor described: a bug would have had two places to hide. There is
one `CrossValidator` now.

Re-ran `train` + `compare` after the extraction to confirm nothing moved: SVM+TF-IDF macro-F1
0.8014 ± 0.0062, identical to the notebook.
