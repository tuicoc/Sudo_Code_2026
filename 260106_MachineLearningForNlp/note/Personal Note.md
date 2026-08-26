# Learning Log: Machine Learning for NLP — Naive Bayes vs SVM

Goal: train classic scikit-learn classifiers on a real Vietnamese text classification task and
compare them properly, instead of just picking one and assuming it's good. Grew from a straight
NB-vs-SVM comparison into a 3-way one — NB+Bag-of-Words, NB+TF-IDF, SVM+TF-IDF — once it became
clear the vectorizer choice itself was worth testing, not just the model choice (see "Vectorization
and evaluation methodology" below).

## Dataset decision (three tries to get here)

1. **`xcrotek.com/careers/...`** — a mistake caught before writing any code: it's a single job
   posting (one page), and Naive Bayes / SVM need many labeled documents across classes to mean
   anything. The career site as a whole only has 10 postings, 9 of which are the same Data/Cloud
   Engineering category — not enough signal or balance for a real train/test comparison.
2. **[AIVIVN 2019](https://www.kaggle.com/datasets/mcocoz/aivivn-2019)** — a real, working binary
   sentiment dataset (12,870 train / 3,217 test Vietnamese e-commerce reviews), and Step 1 was
   fully built and run against it (noise investigation, cleaning, `nltk` tokenization, export). Then
   dropped in favor of dataset 3 below — kept here because the investigative *method* built for it
   (check what noise a corpus actually has before writing cleaning rules, with real before/after
   samples per category) carried over directly and is what Step 1 below is built on.
3. **[tuannguyenvananh/vietnamese-text-classification-dataset](https://www.kaggle.com/datasets/tuannguyenvananh/vietnamese-text-classification-dataset)**
   — the actual dataset this project uses. Also considered and dropped in between:
   **UIT-ViSFD** (`/Users/lumos/Downloads/UIT-ViSFD`, a smartphone-review Aspect-Based Sentiment
   Analysis dataset — `{ASPECT#Sentiment}` labels, ~3.3 aspects/review across 10 categories). Real
   dataset, but "too broad a problem" for a first Naive Bayes vs SVM comparison: it has no single
   classification target, only a multi-label, multi-aspect one, which would have meant deciding a
   whole separate labeling scheme (general-aspect sentiment only? drop rows without it? predict
   aspects instead of sentiment?) before any modeling could start.

**The dataset actually used**: 3,040 Vietnamese product reviews, single CSV (`label,comment`, no
header row — the first data row was silently getting read as a column header until caught), no
train/test split of its own (built one, see Step 1). 3-class sentiment, confirmed by reading
samples of each class rather than trusting the numbers alone: **0 = Negative (1,105), 1 = Neutral
(887), 2 = Positive (1,048)** — reasonably balanced.

This is a 3-class sentiment task, not the binary task AIVIVN would have been or the topic
classification the earlier 3 projects did, but the Vietnamese-preprocessing groundwork from
`260106_TextPreprocessingwithNLP` and the TF-IDF groundwork from
`260106_Scikit-learnTextFeatureExtraction` both still carry over — see `README.md` at the repo root
for the conventions this project follows.

This project uses narrated `notebooks/*.ipynb`, one per step, numbered so the execution order is
explicit (`01_`, `02_`, ...) — matching the other 3 projects' `notebooks/` convention instead of
standing out as the one project using plain scripts. (First draft of Step 1 was written as
`src/01_download_and_preprocess.py`; converted to `notebooks/01_download_and_preprocess.ipynb`
and `src/` deleted once I realized it broke the repo's established pattern — see `README.md`.)

The whole repo's `data/` folders were also retrofitted to one convention at the same time
(`data/raw/`, `data/processed/`, `data/outputs/` in every project, not just this one) — done as a
plain file move + notebook path edit in the 3 earlier projects, without re-running them (Word2Vec's
full training run takes ~2.5h, not worth repeating for a rename). See `README.md` for the final
convention.

## Vectorization And Evaluation Methodology (revised after Step 1)

Several things changed after Step 1 was already done and run, worth recording as decisions rather
than silently editing history:

**1. Naive Bayes doesn't actually want TF-IDF.** `MultinomialNB` models each feature as a count
drawn from a multinomial distribution — its whole math assumes the input is raw term frequencies.
TF-IDF feeds it real-valued, re-weighted numbers instead, which isn't the assumption the model was
built for; `MultinomialNB` is conventionally paired with plain `CountVectorizer` (Bag-of-Words)
counts. This project doesn't just take that as received wisdom, though — it's exactly the kind of
claim worth checking against real numbers, which is why the comparison grew a third leg: **NB +
Bag-of-Words, NB + TF-IDF, and SVM + TF-IDF** (SVM doesn't have this issue — it works directly on
whatever real-valued feature space it's given, TF-IDF's weighting is if anything a *help* there by
scaling down ubiquitous tokens, so only one vectorizer is worth testing for it).

**2. Stopword removal is skipped entirely — checked, not assumed.** `260106_TextPreprocessingwithNLP`
and `260106_Scikit-learnTextFeatureExtraction` both remove stopwords, for topic classification,
where function words genuinely carry no topic signal. This project is sentiment classification, a
different task with a different answer: checked the actual stopword list built for Step 1 and
`không` (not), `chưa` (not yet), `rất` (very), `quá` (too/very) are all in it — exactly the
negation and intensifier words that flip or scale sentiment. Removing them would turn "không tốt"
(not good) into just "tốt" (good), the opposite of its real meaning. Step 1's notebook keeps the
investigation (the stopword list needs its multi-word entries underscore-normalized to match
`underthesea`'s compound tokens — a real technical finding, worth keeping on record) but does not
call `remove_stopwords()` on the exported text.

**3. Bigrams (`ngram_range=(1, 2)`) on every vectorizer, so keeping `không` actually pays off.**
Keeping negation words in the vocabulary (point 2) only half-solves the negation problem if the
vectorizer still only looks at unigrams: on unigrams alone, "không thích" (don't like) contributes
the separate features `không` and `thích`, and `thích` on its own still reads as positive — the
model never sees that the two words are adjacent, only that both happened to appear somewhere in
the review. This is exactly "Job 2" from
`260106_Scikit-learnTextFeatureExtraction/Personal Note.md`'s bigram discussion (capturing the
relationship between two already-complete words sitting next to each other) — not re-derived here,
just applied: `ngram_range=(1, 2)` lets `không_thích` exist as its own feature, distinct from
`không` and `thích` in isolation. "Job 1" from that same discussion (bigrams patching a *broken*
segmentation) doesn't apply here at all, since `underthesea` already segments real words correctly
— the only reason for bigrams in this project is Job 2, negation/intensifier scope.

**4. Negation tagging was considered and deliberately not built — too complex for what this
project needs, and there's a concrete reason it wouldn't even work cleanly here.** The more
thorough technique for this problem, common in sentiment-analysis literature, is negation
tagging/scoping: detect a trigger word (`không`, `chưa`, ...) and mark every word until the next
clause boundary as negated (e.g. `không_thích` AND `không_ổn_định` from "không thích và không ổn
định" all get a negation marker), rather than only catching two-word adjacency the way a bigram
does. Not built here, for two reasons:

- It needs a real scope-detection rule (how far past the trigger word does "negated" extend? bound
  it at the next punctuation mark, a fixed word window, a POS-tag boundary?), and handling that
  correctly — including double negation, "không những... mà còn" (not only... but also) not being
  a true negation, etc. — is a real sub-project on its own, not a line or two of code.
- **Concretely, for this pipeline, it can't even anchor on punctuation**: Step 1 already strips all
  punctuation in the cleaning step, before this would run, so the usual clause-boundary signal
  scope detection relies on doesn't exist anymore by the time it would be needed. Fixing that would
  mean either moving negation tagging earlier in the pipeline (before punctuation stripping,
  reordering steps that currently don't need to care about order) or falling back to an arbitrary
  fixed-word window — a hyperparameter with no principled way to pick it on a corpus this size
  (3,040 reviews).

Bigrams get most of the practical benefit (adjacent negation+content pairs preserved as one
feature) with none of this — no scope rule, no new hyperparameter, just one vectorizer argument.
Worth remembering as a real limitation, not a silently-skipped one: multi-word negation scope
("không thích và cũng không ổn định" — two separate negated ideas) is still invisible to a bigram
model the way it wouldn't be to real negation tagging.

**5. Vectorization moves out of Step 1, into each model notebook, fit inside each CV fold.** Step 1
originally fit a shared vectorizer once on the whole corpus before any train/test split existed.
That's backwards: fitting `CountVectorizer`/`TfidfVectorizer` (vocabulary, document frequencies)
on data that includes what will later be evaluation data leaks information across the split.
`underthesea` tokenization, normalization, cleaning, and teencode expansion all stay in Step 1
(they're deterministic per-row rules, nothing "fit" on the corpus, so there's no leakage risk) —
only Bag-of-Words/TF-IDF moves to per-fold fitting in Steps 2-3.

**6. `StratifiedKFold` cross-validation replaces the single train/test split.** With 3,040 rows and
3 methods to compare, one 80/20 split risks a comparison that's really just noise from which rows
happened to land in the test set. 5-fold stratified CV (`random_state=42`, re-created with the same
seed in every notebook that needs it, so no fold indices need to be persisted to disk — same data,
same seed, same folds) means every row gets evaluated once, out-of-fold, across all 5 folds, and
each of the 3 methods is compared on the exact same folds. No separate held-out test set beyond
that — the whole point of K-fold CV here is that a fixed holdout isn't needed for a fair comparison.

## The Plan

Working through this sequentially, one notebook at a time, running and understanding each before
moving to the next. Model/metric artifacts are saved to `data/outputs/` (not a separate top-level
`results/` folder) — the repo-wide `data/{raw,processed,outputs}` convention, see `README.md`.

### Step 1 — `notebooks/01_download_and_preprocess.ipynb`
- Download via `kagglehub.dataset_download("tuannguyenvananh/vietnamese-text-classification-dataset")`
  into `data/raw/`.
- **Investigate each noise category on the real corpus before writing any cleaning rule for it** —
  abbreviations, missing diacritics, emoji/icons, glued-together words — with real counts and real
  before/after examples per category, not assumptions carried over from a different corpus. See
  "Step 1 done" below for what was actually found and what got fixed vs. documented as a known
  limitation.
- Clean the `comment` text: Unicode NFC normalize, strip punctuation/digits, lowercase.
- Tokenize with **`underthesea.word_tokenize`**, not `nltk` — real word segmentation instead of
  syllable splitting (the earlier two projects used `nltk` for speed on much larger corpora;
  `underthesea` is slow enough to need a sample-first speed check, done below, before committing to
  a full run — same caution as `260106_Word2Vec`). Expand the one confirmed teencode abbreviation
  (`vs` → `với`). Stopwords are investigated but **not** removed — see "Vectorization and
  evaluation methodology" above for why.
- Export the cleaned, tokenized corpus as one file, `data/processed/reviews.parquet` — no
  train/test split here; that's what `StratifiedKFold` in Steps 2-3 is for.

### Step 2 — `notebooks/02_naive_bayes.ipynb`
- Load `data/processed/reviews.parquet`.
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. For each fold: fit
  `CountVectorizer(ngram_range=(1, 2))` on the training portion only, transform both portions,
  train `MultinomialNB`, evaluate on the held-out portion (accuracy, macro precision/recall/F1 —
  macro because there are 3 classes now, not 2). Repeat the same loop with
  `TfidfVectorizer(ngram_range=(1, 2))` instead, same folds (same seed), same model.
- Report both variants' cross-validated metrics (mean ± std across the 5 folds) side by side —
  this notebook is where the "NB wants BoW, not TF-IDF" claim actually gets checked against real
  numbers, not just asserted.
- Save both sets of metrics to `data/outputs/` (`nb_bow_metrics.json`, `nb_tfidf_metrics.json`).

### Step 3 — `notebooks/03_svm.ipynb`
- Same `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` — identical folds to Step 2,
  same seed, same data order.
- `TfidfVectorizer(ngram_range=(1, 2))` (fit per fold, same as Step 2) + `LinearSVC`, the standard
  scikit-learn choice for high-dimensional sparse TF-IDF text features.
- Same metrics as Step 2, saved to `data/outputs/svm_tfidf_metrics.json`.

### Step 4 — `notebooks/04_compare.ipynb`
- Load all three metrics JSON files from `data/outputs/`.
- Build one comparison table (accuracy/macro-precision/recall/F1, mean ± std across folds) and a
  bar chart, saved to `data/outputs/comparison.png`.
- Write a short conclusion: does NB actually do better with BoW than TF-IDF here, as the theory
  predicts, and how does the better NB variant compare to SVM+TF-IDF — checked against the real
  cross-validated numbers, not assumed from the modeling theory alone.

## Step 1 done — what actually happened

Ran `notebooks/01_download_and_preprocess.ipynb` (`jupyter nbconvert --execute`, output cells
cleared afterward — same convention as the other 3 projects' notebooks: execution_count kept,
outputs not committed). `kagglehub.dataset_download` worked anonymously again, no credentials
needed. Raw CSV saved to `data/raw/`, cleaned/tokenized text exported to
`data/processed/reviews.parquet` (all 3,040 rows, no split — see "Vectorization and evaluation
methodology" above for why the train/test split originally planned here was replaced by
`StratifiedKFold` in Steps 2-3 instead, and stopwords are investigated but not removed).

**The noise investigation, category by category** (each checked on the real corpus with actual
counts, not assumed from the AIVIVN findings — this corpus turned out much cleaner):

| Category | Found | Fixed? | Why |
|---|---|---|---|
| URLs / HTML | 0 rows | n/a | not present |
| Emoji / icons | 0.2% of rows (6/3,040) | No | too rare to matter, and wouldn't become a TF-IDF feature anyway (default token pattern ignores emoji) |
| Glued words (`hayXài`) | 2.4% of rows (72/3,040) | No | indistinguishable at the character level from real brand-name casing (`tikiNow` matches the exact same lower→upper pattern) |
| Missing diacritics | 0.2% of rows (5/3,040) | No | would need a dedicated diacritic-restoration model; rare enough here it barely matters |
| Abbreviations | 1 confirmed (`vs` → `với`, context-checked) | Yes | everything else checked (`k`, `ko`, `dc`, `đc`, `sp`, `mn`...) was at or near zero — this corpus is written in standard Vietnamese, not AIVIVN-style teencode. `ok`/`oke`/`ship` are loanwords, not abbreviations, left untranslated |
| Syllable-split compounds (`uy` + `tín` instead of `uy_tín`) | pervasive — any multi-syllable word, with `nltk` | Yes | switched tokenizer to `underthesea` |

Two mechanical bugs caught and fixed while building this, worth remembering:

- **A hand-written Vietnamese character range is not the same as "Vietnamese letters."** First
  attempt at detecting glued words used a regex range `[a-zà-ỹ]`, which matched 96% of rows —
  because the Unicode range between `à` (U+00E0) and `ỹ` (U+1EF9) spans several unrelated blocks,
  not just Vietnamese diacritics. Fixed by checking `str.islower()`/`str.isupper()` per character
  instead (Python's own Unicode-aware case logic), which brought it down to the real number: 2.4%.
- **81% of the stopword list is multi-word phrases** (`"nói chung"`, `"bao giờ"`, space-separated),
  which `underthesea` joins into single underscore tokens (`nói_chung`). Without normalizing the
  stopword set to also include the underscore-joined form of every multi-word entry, these phrases
  would have silently stopped being filtered the moment the tokenizer changed — not a crash, just a
  quiet accuracy regression that would've been hard to trace back to this cause later. Confirmed the
  fix actually works (not just that it runs) by checking `nói_chung` was gone from a real example
  row's tokens after stopword removal, not just trusting the code.

`underthesea` benchmark on a 200-row sample: 4.26 ms/item → ~13s for the full 3,040-row corpus.
Nothing like the ~2h it took on the 184K-article news corpus — reviews are short, so committing to
the full run was safe once actually measured, not just assumed from the earlier project's
benchmark at a very different scale.

**Next up: Step 2 (`notebooks/02_naive_bayes.ipynb`)** — TF-IDF + `MultinomialNB` on `data/processed/`.

## Step 2 done — the "NB wants BoW, not TF-IDF" claim, checked

Ran `notebooks/02_naive_bayes.ipynb`: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`,
`CountVectorizer(ngram_range=(1,2), min_df=2)` and `TfidfVectorizer(ngram_range=(1,2), min_df=2)`
each fit fresh inside every fold, `MultinomialNB` on top of both. Real numbers, mean ± std across
the 5 folds:

| | accuracy | precision (macro) | recall (macro) | F1 (macro) |
|---|---|---|---|---|
| **Bag-of-Words** | 0.7895 ± 0.0159 | 0.7807 ± 0.0165 | 0.7789 ± 0.0153 | 0.7775 ± 0.0152 |
| **TF-IDF** | 0.7648 ± 0.0132 | 0.7679 ± 0.0202 | 0.7440 ± 0.0126 | 0.7329 ± 0.0126 |

The theory holds up on this dataset: Bag-of-Words beats TF-IDF for `MultinomialNB` on every metric,
by a margin (~2.5 points accuracy, ~4.5 points macro-F1) that's larger than the fold-to-fold std —
not just noise. Worth having actually run this instead of trusting the textbook claim on faith:
now there's a real number to cite (macro-F1 0.777 vs 0.733) instead of just "NB prefers counts."

Saved to `data/outputs/nb_bow_metrics.json` and `data/outputs/nb_tfidf_metrics.json` (per-fold
numbers plus the mean/std summary above, so Step 4 can rebuild this table without re-running CV).

**Next up: Step 3 (`notebooks/03_svm.ipynb`)** — same folds, `TfidfVectorizer(ngram_range=(1,2))` +
`LinearSVC`.

## Step 3 done

Ran `notebooks/03_svm.ipynb`: same `StratifiedKFold(random_state=42)` folds as Step 2,
`TfidfVectorizer(ngram_range=(1,2), min_df=2)` + `LinearSVC(random_state=42)`, same `run_cv`/
`summarize` shape as Step 2 (redefined in this notebook rather than imported — no cross-notebook
imports in this repo, see `README.md`). No `dual=...` warning at this scikit-learn version
(1.9.0) — `LinearSVC`'s `dual` default is now `"auto"`, so nothing to configure by hand.

| | accuracy | precision (macro) | recall (macro) | F1 (macro) |
|---|---|---|---|---|
| **SVM + TF-IDF** | 0.8112 ± 0.0055 | 0.8040 ± 0.0059 | 0.8017 ± 0.0059 | 0.8014 ± 0.0062 |

Beats both NB variants on every metric, and its fold-to-fold std is roughly a third of NB+BoW's —
more consistent, not just higher on average. Saved to `data/outputs/svm_tfidf_metrics.json`.

**How `LinearSVC` actually handles 3 classes, since this project never explicitly coded multi-class
anything** — asked myself this after the fact and it was worth answering with real evidence, not
just "sklearn handles it": `LinearSVC`'s `multi_class` parameter defaults to `"ovr"`
(One-vs-Rest). With `label` having 3 values, `.fit(X, y)` trains 3 binary hyperplanes internally
without any extra code from this project — Negative vs {Neutral, Positive}, Neutral vs
{Negative, Positive}, Positive vs {Negative, Neutral}. At predict time each of the 3 gives a
decision score (signed distance to its own hyperplane) and the sample goes to whichever is
highest. Verified in `03_svm.ipynb`, not just asserted: fit once on the full corpus outside the CV
loop (a demo only, not used for any reported metric) and checked `coef_.shape` came out
`(3, 10929)` — literally 3 separate hyperplanes over the fold's ~11K-term vocabulary — and
`decision_function` on one sample returned 3 scores, exactly as the mechanism predicts.

**Not the same mechanism as `sklearn.svm.SVC`** (kernel-based), which always trains One-vs-One
internally regardless of settings — n×(n-1)/2 = 3 classifiers here too, coincidentally the same
count exactly at 3 classes, but a strategy that scales differently as classes grow (6 vs. 4 at 4
classes, and the gap widens from there). `LinearSVC` defaults to OvR specifically because it's
cheaper for linear/sparse data like TF-IDF: n classifiers instead of n×(n-1)/2, each a plain linear
fit.

This is also the concrete reason macro-averaging (`precision_recall_fscore_support(...,
average="macro")`) has been used for evaluation since Step 2, not accuracy alone or a micro
average: it scores each of the 3 classes independently, then averages unweighted, so the smallest
class (Neutral, ~29% of the data) can't get quietly swamped by the largest (Negative, ~36%) the
way accuracy or micro-averaging would let happen.

**Next up: Step 4 (`notebooks/04_compare.ipynb`)** — pull all three metrics files together.

## Step 4 done — project complete

Ran `notebooks/04_compare.ipynb`: loaded all three `data/outputs/*_metrics.json` files, built a
comparison table, and a grouped bar chart (`data/outputs/comparison.png` — one shared 0-1 y-axis,
error bars from the fold std already computed in Steps 2-3, methods in a fixed color order rather
than an arbitrary/cycled one, following the repo's `dataviz` skill guidance even though this is a
static matplotlib chart in a notebook rather than an interactive web artifact — the parts that
actually apply to a static PNG: fixed categorical color order, one axis, no rainbow, thin clean
bars, always-present legend for 3 series).

**Final result**: SVM + TF-IDF (bigrams) is the best of the three methods tested, on both average
performance (macro-F1 0.801 vs NB+BoW's 0.777 and NB+TF-IDF's 0.733) and consistency across folds.
The "NB wants BoW, not TF-IDF" textbook claim held up empirically too — BoW beat TF-IDF for
`MultinomialNB` by a margin bigger than either's fold-to-fold noise. NB+TF-IDF, the pairing theory
says shouldn't work well, was in fact the weakest of the three — a genuinely useful negative
result, not just a formality run to fill out the comparison.

All 4 planned notebooks done: `01_download_and_preprocess.ipynb` → `02_naive_bayes.ipynb` →
`03_svm.ipynb` → `04_compare.ipynb`, each runnable independently, artifacts in `data/{raw,
processed,outputs}` per the repo convention.
