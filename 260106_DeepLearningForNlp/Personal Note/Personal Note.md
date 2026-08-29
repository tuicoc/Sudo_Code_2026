# Learning Log: Deep Learning for NLP — a first neural network for text classification

Goal: follow the TensorFlow [basic text classification
tutorial](https://www.tensorflow.org/tutorials/keras/text_classification) but on a real Vietnamese
corpus instead of IMDB, and actually understand each layer rather than copying the notebook. The
question the project has to answer at the end: **does a simple neural network beat the classic
TF-IDF + linear model from `260106_MachineLearningForNlp`, on a task where a published baseline
exists to check both against?**

This is a learn-by-building project, so the notebooks are written step by step and each step gets
explained before it gets coded — see "How this project is being worked through" at the bottom.

## The dataset: VNTC

[duyvuleo/VNTC](https://github.com/duyvuleo/VNTC) (MIT) — "A Large-scale Vietnamese News Text
Classification Corpus", the dataset from Vu et al., *A Comparative Study on Vietnamese Text
Classification Methods*, RIVF 2007. Two levels ship in the repo, both as `.rar`:

| Level | Train | Test | Classes |
|---|---|---|---|
| `Data/10Topics/Ver1.1` | 33,759 | 50,373 | 10 top-level topics |
| `Data/27Topics/Ver1.1` | 14,375 | 12,076 | 27 sub-topics |

**Using 10Topics**, for two reasons: it is the level with published baselines to compare against,
and 10 balanced-ish classes is the right size for a first neural network — 27 fine-grained classes
on 14k documents would make a weak result ambiguous (bad model, or just too little data per class?).

### What the raw data actually is (checked, not assumed)

- One `.txt` per article, in a folder named after its class — `Train_Full/The thao/TT_ VNE_ (100).txt`.
  That folder-per-class shape is exactly what `tf.keras.utils.text_dataset_from_directory` expects,
  which is a large part of why this corpus suits the tutorial.
- **Encoding is UTF-16LE with BOM, CRLF line endings** — see the section below, it is the first
  real obstacle in the project.
- Line 0 is the headline, the rest are body paragraphs. Text is clean: no HTML, no tags, no
  scraped boilerplate.
- Length: median **361 words** per article (p25 228, p75 614, p95 1197, max 5891). Much longer than
  the IMDB reviews the tutorial is tuned for — the tutorial's `output_sequence_length=250` would
  truncate ~80% of these documents, so that number has to be re-derived here rather than copied.
- Vocabulary: on a 3,000-document sample, 51,515 distinct whitespace tokens; the top 10,000 cover
  95.7% of all token occurrences, the top 20,000 cover 97.6%.
- Sources are VnExpress / Tuổi Trẻ / Thanh Niên / Người Lao Động, roughly 2004–2006, encoded in the
  filename prefix (`_VNE_`, `_TT_`, `_TN_`, `_NLD_`).

### Encoding: UTF-16LE in, UTF-8 out — and what happens if you skip it

Sampled 1,200 files across both splits: **100% are UTF-16LE with a `\xff\xfe` BOM and CRLF line
endings**, zero decode failures. Fully consistent, which is a relief — no per-file encoding sniffing
needed.

It matters because `text_dataset_from_directory` reads files as UTF-8 and does not sniff. Pointed
it straight at `data/raw/Train_Full` to see the failure rather than assume it, and it does not
raise — it silently hands back garbage:

```
Found 33759 files belonging to 10 classes.
b'\xff\xfe \x00T\x00h\x00\xe0\x00n\x00h\x00 \x00l\x00\xad\x1ep\x00 \x00d\x00\xf1\x1e ...'
```

That is `" Thành lập dự án POLICY..."` with a null byte after every character — the BOM survives as
literal bytes and every ASCII character is followed by `\x00`, which is UTF-16 read one byte at a
time. **The silence is the danger**: no exception, no warning, and `TextVectorization` downstream
would happily build a vocabulary out of null-byte-riddled junk. Training would run, produce a
plausible-looking loss curve, and be meaningless. Convert-to-UTF-8 into `data/processed/` is
mandatory, not a nicety.

### Line endings: CRLF → LF, and the reason is *not* the one I assumed

The corpus is CRLF (`\r\n`), this machine is macOS (LF). The obvious worry is that `\r` gets glued
onto tokens — `"bao\r"` instead of `"bao"` — quietly doubling the vocabulary with invisible
variants. Checked it instead of guessing:

```python
tf.strings.split(tf.constant(["Tieu de bai bao\r\nDoan than bai."]))
# [[b'Tieu' b'de' b'bai' b'bao' b'Doan' b'than' b'bai.']]
```

**No `\r` anywhere.** `TextVectorization`'s default `split="whitespace"` calls `tf.strings.split`
with `sep=None`, which splits on *any* whitespace run — `\r`, `\n`, `\t`, space alike — so CRLF is
absorbed as a separator and never reaches the vocabulary. So: not a correctness bug, and the
project would train fine either way.

Normalizing to `\n` anyway, for two reasons that are real even though the tokenizer one is not:

1. **Line 0 is the headline.** Any later step that wants title separately from body does
   `text.split("\n")` and gets `"Roddick gặp đối thủ khó tại vòng 1 Australia Mở rộng\r"` — a
   trailing `\r` on every single headline, invisible when printed, which then survives into
   whatever it feeds.
2. **It only works by accident of the default.** The safety comes from `split="whitespace"`. Swap
   in a custom `split` callable or a regex standardizer — a live option for Vietnamese, where the
   default `lower_and_strip_punctuation` is not obviously right — and `\r` is suddenly back in
   play. `lower_and_strip_punctuation` strips `!"#$%&()*+,-./:;<=>?@[\]^_{|}~'` and **does not
   touch `\r`**, so the regex path would not save it.

The macOS detail worth remembering: Python does **not** fix this for you on write. `open(p, "w")`
translates `\n` to `os.linesep`, and on POSIX `os.linesep` *is* `\n`, so a string already
containing `\r\n` is written back out verbatim, CRLF and all. The fix belongs on the **read** side,
where one keyword does both jobs at once:

```python
open(p, encoding="utf-16").read()          # -> 'Mở rộng\nHạt giống'   (universal newlines)
open(p, "rb").read().decode("utf-16")      # -> 'Mở rộng\r\nHạt giống' (CRLF preserved)
```

Text-mode `open` defaults to `newline=None`, i.e. universal-newline mode, which converts `\r\n` and
lone `\r` to `\n` during decoding. Using `encoding="utf-16"` (not `"utf-16-le"`) also makes Python
consume the BOM to determine endianness instead of leaving it as a stray `﻿` character at the
start of the string. One call, both problems, no manual `.replace()`.

### Class distribution — train and test are not drawn the same way

| Class | Train | | Test | |
|---|---|---|---|---|
| Chinh tri Xa hoi | 5,219 | 15.46% | 7,567 | 15.02% |
| Doi song | 3,159 | 9.36% | 2,036 | 4.04% |
| Khoa hoc | 1,820 | 5.39% | 2,096 | 4.16% |
| Kinh doanh | 2,552 | 7.56% | 5,276 | 10.47% |
| Phap luat | 3,868 | 11.46% | 3,788 | 7.52% |
| Suc khoe | 3,384 | 10.02% | 5,417 | 10.75% |
| The gioi | 2,898 | 8.58% | 6,716 | 13.33% |
| The thao | 5,298 | 15.69% | 6,667 | 13.24% |
| Van hoa | 3,080 | 9.12% | 6,250 | 12.41% |
| Vi tinh | 2,481 | 7.35% | 4,560 | 9.05% |

Two things to carry forward. First, the test set is *bigger than* the train set (50k vs 34k) — the
opposite of the usual 80/20 habit, so no re-splitting: the official split is the split. Second, the
class proportions differ noticeably between the two (Doi song is 9.4% of train but 4.0% of test,
The gioi is 8.6% vs 13.3%). A model that quietly learns the training prior gets rewarded or
punished by that shift, which is exactly the situation where plain accuracy misleads — so
**macro-F1 is the headline metric here, with per-class F1 and a confusion matrix next to it.**

### Data-quality finding: ~5% of the test set also appears in train

Hashed all 84,132 documents (whitespace-normalized MD5) before writing any model code:

- **2,531 test files (5.0% of the test set) are byte-identical in content to a file in train.**
  119 of those carry a *different* label in train than in test.
- Test also has 854 redundant copies internally (1.7%), in 850 duplicate groups, 219 of which span
  more than one class.
- Train is nearly clean by comparison: 50 redundant copies (0.1%).

**What those 219 groups are — and are not.** First written up here as evidence that the classes
overlap and that "the task itself is contradictory", which was an overstatement worth correcting
rather than quietly editing. Opened them and read the actual articles: they are ordinary, full-length
news pieces (median 2,158 characters) — e.g. *"Gắn chíp điện tử cho 2.800 con gấu nuôi nhốt"* filed
under both `Chinh tri Xa hoi` and `Khoa hoc`, *"TP HCM sẽ lắp camera giám sát hoạt động hải quan"*
under both `Chinh tri Xa hoi` and `Kinh doanh`.

That is a **duplicated file**, not a multi-labelled document. Every file in VNTC sits in exactly one
folder and carries exactly one label; nobody annotated anything with a label *set*. The corpus is
single-label, and the same article simply got packaged twice. So the task formulation is
unambiguous — and the quantitative impact is small: 219 files out of 50,373 are guaranteed wrong
whatever the model predicts, which puts the **accuracy ceiling at 99.57%**, a loss of 0.43 points.
Worth knowing, not worth building a framing around.

**Decision: keep the official split untouched.** The corpus is a published benchmark and the point
of using it is comparability with the numbers below; silently deduplicating would make our score
incomparable with everyone else's. The leakage is recorded here instead, and gets re-stated when
the test score is reported, so the number is read with the right caveat — a reported accuracy on
this benchmark is inflated by up to ~5 points of memorization, ours and the published baselines'
alike.

### Baselines to check ourselves against

From [NLP-Vietnamese-progress](https://github.com/undertheseanlp/NLP-Vietnamese-progress/blob/master/tasks/text_classification.md),
sourced from the original RIVF'07 paper:

| Level | Model | Score |
|---|---|---|
| 10 topics | NGRAM | 97.1 |
| 10 topics | SVM Multi | 93.4 |
| 27 topics | SVM Multi | 96.21 |

Plus an in-house baseline: the TF-IDF + linear-model approach from
`260106_MachineLearningForNlp`, re-run on *this* corpus. That one matters more than the published
numbers, because it is the only comparison where every other variable is held fixed — same data,
same split, same metric, same machine — so a difference is attributable to the model and nothing
else. The published baselines are a sanity band ("are we in the right neighbourhood at all?"), not
a like-for-like comparison, since 2007 preprocessing is not documented in enough detail to
reproduce.

## What kind of problem this is: sigmoid vs. softmax

The single most useful thing learned so far, and the thing that actually separates this project from
the tutorial it follows. Three problem shapes that look similar and are not:

| Shape | Meaning | Output layer | Loss |
|---|---|---|---|
| Binary | 2 classes, pick 1 | `Dense(1, "sigmoid")` | `BinaryCrossentropy` |
| **Multi-class, single-label** | **N classes, pick exactly 1** | **`Dense(10, "softmax")`** | **`SparseCategoricalCrossentropy`** |
| Multi-label | N classes, pick 0 to N | `Dense(10, "sigmoid")` | `BinaryCrossentropy` |

The TensorFlow tutorial is row 1 (IMDB, positive/negative). **VNTC is row 2.** Four things change
and nothing else does: `Dense(1)`→`Dense(10)`, `sigmoid`→`softmax`,
`BinaryCrossentropy`→`SparseCategoricalCrossentropy`, `BinaryAccuracy`→`SparseCategoricalAccuracy`.

**Why softmax and not ten sigmoids**, which is the part worth actually understanding rather than
memorizing. Both produce ten numbers between 0 and 1; the difference is whether those numbers know
about each other.

- `softmax` exponentiates all ten logits and divides by their sum, so **the outputs are forced to
  add up to exactly 1**. The classes compete: raising confidence in `The thao` mathematically
  requires taking it away from the other nine. That is the right inductive bias when the classes are
  mutually exclusive, and it means the network is trained to answer "*which one* of these ten".
- Ten independent `sigmoid` units each squash their own logit separately. Nothing ties them
  together, so all ten can be 0.9, or all ten 0.1. That expresses "*which of* these ten, possibly
  several, possibly none" — the right choice for tagging an article as *both* Business *and* World,
  and the wrong choice here.

VNTC files each live in exactly one folder, so the labels are mutually exclusive by construction and
softmax is correct. (Checked whether the 219 duplicate groups above undermine that — they do not;
see that section. Single-label stands.)

The `Sparse` prefix is a separate axis and easy to conflate with the above: it refers only to how
the *label* is formatted, not to the problem shape. `SparseCategoricalCrossentropy` takes an integer
(`3`); `CategoricalCrossentropy` takes a one-hot vector (`[0,0,0,1,0,0,0,0,0,0]`). Identical maths,
different input format. `text_dataset_from_directory` hands back integers, so `Sparse` is the one
that needs no conversion step.

**The floor to measure against.** Before training anything, know what "bad" looks like on this data:

| Trivial strategy | Accuracy | Macro-F1 |
|---|---|---|
| Uniform random guess over 10 classes | ~10% | ~10% |
| Always predict the largest class (`Chinh tri Xa hoi`) | **15.0%** | **2.6%** |

That second row is the argument for macro-F1 as the headline metric, in one line: a model that has
learned *nothing* except the class prior scores 15% accuracy — high enough to look like partial
learning — while its macro-F1 is 2.6%, which correctly reports that it is useless. Accuracy hides
this failure, macro-F1 does not. The target band for this project is therefore 15 (floor) to 93.4
(RIVF'07 SVM).

## Do we need `underthesea`? — measured, deferred, then reversed

`260106_Word2Vec` paid for real word segmentation and `260106_Scikit-learnTextFeatureExtraction`
did not, so this is a live question here rather than a settled one. Three options, all measured on
real VNTC documents before deciding:

**A. No segmentation.** `TextVectorization` splits on whitespace, which for Vietnamese is
*syllable*-level: `kinh doanh` becomes two unrelated tokens, `công nghệ thông tin` becomes four.
Cost: 0 minutes.

**B. `underthesea` segmentation, embeddings learned from scratch.** Benchmarked
`word_tokenize(..., format="text")` on 40 real VNTC articles (585 whitespace tokens each on
average — these are much longer than the ~39-word wiki segments benchmarked in
`260106_Word2Vec`): **56 ms/document**. Scaled up that is **32 minutes for train alone, 79 minutes
for train + test**. It converts 25.1% of output tokens into underscore-joined compounds and drops
the total token count 12.4%.

**C. `underthesea` + the Word2Vec vectors from `260106_Word2Vec` as a pretrained `Embedding`.**
This is the option that would tie the two projects together, so it got the closest look. The
`viwik18` model is 414,646 tokens at `vector_size=100`. Coverage measured on 120 VNTC articles:

| Tokenization | type coverage | **token coverage** |
|---|---|---|
| underthesea, everything kept | 74.9% | 40.6% |
| whitespace only | 40.6% | 46.0% |
| underthesea + strip punctuation + strip stopwords | 83.3% | **93.0%** |

The first two rows look alarming until you read the OOV list: `và`, `của`, `các`, `có`, `trong`,
`là`, `,`, `.`. Those are missing because **the Word2Vec model never saw them** — that project
removed stopwords and punctuation before training. So the honest number is the third row: matched
against the pipeline that actually produced it, the vectors cover **93% of content-word
occurrences**. The genuine misses are 2004–2006 news proper nouns absent from Wikipedia — `s-fone`,
`cityphone`, `scomi`, `kđtmtt` — a real but narrow domain gap, and small enough that C is
technically viable.

**Decision: A for Steps 1–5. C becomes Step 6, as an experiment, not a prerequisite.** Reasons, in
order of weight:

1. **The thing being learned is TensorFlow, not Vietnamese tokenization.** Steps 1–5 exist to
   understand `TextVectorization`, `Embedding`, `fit()`, and a learning curve. Bolting on an
   external segmenter adds 79 minutes of preprocessing and a second uncontrolled variable to every
   result, right where the point is to see one mechanism clearly.
2. **`TextVectorization` can already recover part of what segmentation buys, for free.** Verified
   that `ngrams=2` works with `output_mode="int"` (not only with the `count`/`tf_idf` modes), so
   `kinh doanh` can be a single vocabulary entry without underthesea. There is a catch worth
   knowing before using it — see Step 2.
3. **This model may not be able to exploit better tokens anyway.** `GlobalAveragePooling1D`
   averages the sequence and discards word order entirely; it is a bag-of-embeddings model. Feeding
   a bag cleaner units helps some, but the ceiling is set by the architecture, not the tokenizer.
   Spending 79 minutes to improve inputs to a model that throws away structure is the wrong order
   of operations.
4. **A is also the honest baseline for C.** Without A's number first, C has nothing to be compared
   against, and "pretrained embeddings helped" would be unfalsifiable.

Consequence to state plainly when reporting the final score: this project's tokens are Vietnamese
*syllables*, not words, and that is a known handicap versus the published RIVF'07 baselines, which
used real word segmentation.

Also considered and rejected for now: `pyvi` (faster than underthesea, but a second segmenter with
different conventions is not worth introducing when the repo already standardized on underthesea)
and PhoBERT (a pretrained transformer would very likely win outright, and that is exactly why it is
wrong here — it would answer a different question than "what does a simple neural network do?").

### Reversed: `underthesea` moves to Step 1, and the features become TF-IDF

The four arguments above are still each true, but they were answering the wrong question. They
assumed the model would be the tutorial's `Embedding → GlobalAveragePooling1D` bag-of-embeddings.
Reframing the task settles it differently:

**This is keyword detection, not sequence understanding.** Telling `The thao` from `Kinh doanh` is
about *which words are present* — `cầu thủ`, `vô địch`, `bàn thắng` versus `cổ phiếu`, `xuất khẩu`,
`doanh nghiệp`. Word order carries almost nothing. Once that is stated plainly, the natural
representation is not a sequence at all: it is **TF-IDF over the whole document**, one fixed-length
vector per article, order discarded by construction rather than discarded accidentally by an
average-pooling layer.

And under *that* representation the cost/benefit of segmentation inverts. A bag-of-embeddings
averages vectors, so it blurs whatever it is given and the tokenizer matters less. TF-IDF weights
each vocabulary entry independently, so the vocabulary **is** the feature space — and `kinh_doanh`
as one weighted feature is strictly more informative than `kinh` and `doanh` as two, since `kinh`
alone is nearly meaningless and appears across unrelated topics. The 79 minutes buys something real
here in a way it did not before.

**No n-grams.** Two reasons, and the second is the stronger one:
1. Order does not matter for this task, which is the whole premise above — bigrams exist to recover
   local order, and we are not paying for what we do not need.
2. **Segmentation already does the job bigrams were being considered for.** The only reason
   `ngrams=2` came up was to glue `kinh doanh` back together; `underthesea` returns `kinh_doanh`
   directly. Running both would mean paying twice for the same thing and multiplying the feature
   space for nothing. This also sidesteps the unigrams-then-bigrams ordering trap documented in
   Step 2 entirely.

Consequence for the architecture: with TF-IDF as input there is no `Embedding` layer and no
sequence dimension, so the model becomes `Dense → Dense(10, softmax)` on a fixed-width vector. This
is a real departure from the tutorial and the reason has to be understood rather than waved through
— but it also produces a **strictly better comparison** against `260106_MachineLearningForNlp`:
same TF-IDF features, same split, same metric, only the classifier swapped from `LinearSVC` to a
neural network. That is a controlled experiment, where the tutorial's architecture would have
confounded representation and model together.

## Environment: why this project has its own `.venv`

Every earlier project installs into the repo's shared `.venv`. This one does not, and the reason is
a hard platform constraint rather than a preference.

The machine is an **Intel Mac (x86_64), Python 3.12**. TensorFlow stopped publishing macOS x86_64
wheels after **2.16.2** — 2.17 onwards is Apple-Silicon-only on macOS. And 2.16.2 pins
`numpy<2.0`, so installing it into the shared environment would drag `numpy` from 2.5.2 down to
1.26.4 for all four existing projects. Their declared constraints would technically still be
satisfied (pandas 3.0.5 wants `>=1.26.0`, scikit-learn 1.9 `>=1.24.1`, gensim 4.4 `>=1.18.5`,
matplotlib 3.11 `>=1.25`), but risking four working projects to avoid one extra environment is a
bad trade.

So: `260106_DeepLearningForNlp/.venv`, registered as its own Jupyter kernel. Consequence to
remember — **this project cannot import anything installed only in the shared `.venv`**
(`underthesea`, `gensim`, `nltk` are not here). That is fine and arguably a feature: it forces the
preprocessing to stay inside TensorFlow's own `TextVectorization`, which is the thing being learned.

TensorFlow 2.16 also means **Keras 3** is the default frontend, so tutorial code written for
`tf.keras` 2.x may differ in places; where it does, the difference gets noted rather than
worked around silently.

## The plan

**Step 1 — `01_prepare_data.ipynb`: get the corpus into a shape TensorFlow can read.**
Point `text_dataset_from_directory` at `data/raw/` first and *look at the garbage it returns* — the
failure is silent, and seeing it once is worth more than being told. Then one pass over all 84,132
files producing `data/processed/`:

1. read with `open(p, encoding="utf-16")` — handles the BOM and converts CRLF to LF in one call
2. `unicodedata.normalize("NFC", text)` — cheap insurance, 299/300 files are already NFC
3. `underthesea.word_tokenize(text, format="text")` — underscore-joined compounds (**~79 min**,
   benchmarked, so this runs once and is written to disk rather than repeated per experiment)
4. write UTF-8, folder-per-class layout preserved, because that is where the labels come from

Then reload with `text_dataset_from_directory` and carve a validation split off the *training* set
only — the test set is not touched until Step 4. Verify by decoding a few batches back to text and
checking the diacritics and the `_` compounds both survived.
*Concepts to actually understand here:* what a `tf.data.Dataset` is and why it is streamed rather
than loaded into a list; what `batch_size`, `shuffle`, `seed`, `validation_split` do; why
`.cache()` and `.prefetch()` exist.
*Why segmentation lives here and not in Step 2:* `TextVectorization` runs inside the TensorFlow
graph, and `underthesea` is ordinary Python — it cannot be called from inside a `tf.data` pipeline
without `tf.py_function` and losing the graph's parallelism. Segmenting to disk once, up front, is
both simpler and the only way to avoid paying the 79 minutes on every epoch.

**Step 2 — `02_text_vectorization.ipynb`: turn Vietnamese words into TF-IDF features.**
Build a `TextVectorization` layer with `output_mode="tf_idf"`, `max_tokens`, and a standardizer that
is safe for the underscore-joined compounds Step 1 produced — the default
`lower_and_strip_punctuation` strips `_`, which would undo the whole segmentation step, so this has
to be checked and not assumed. `adapt()` on the training set **only** (adapting on test is leakage),
then read the vocabulary and the learned IDF weights to see which words the corpus considers
informative.
*Concepts:* what TF-IDF actually computes and why a word common in *this* document but rare across
the corpus is the informative one; why `adapt()` is a separate fit-like step; what `max_tokens`
trades off (measured earlier: top 10k tokens cover 95.7% of occurrences, top 20k cover 97.6%).
Note there is **no `output_sequence_length`** here — TF-IDF emits one fixed-width vector per
document regardless of its length, which is exactly why the "the tutorial truncates 80% of our
articles" problem disappears instead of needing to be solved.

*No n-grams* — see the reversal section above. Recorded here because the finding is real and cost
time to establish: `ngrams=2` *does* work with `output_mode="int"` (not only with the count/tf_idf
modes, as the docs' phrasing suggests), but it emits **all unigrams first, then all bigrams
appended** — `['kinh','doanh',...,'nganh','kinh doanh','doanh quoc',...]` — so the sequence roughly
doubles in length and is no longer in reading order at all. Harmless for order-blind models,
actively wrong for a Conv1D or LSTM. Not used here because segmentation already supplies the
compounds bigrams were wanted for.

**Step 3 — `03_train_neural_network.ipynb`: the model, and what each layer is for.**
`Dense(64, relu) → Dropout → Dense(10, softmax)` on the TF-IDF vector — a plain feed-forward net,
no `Embedding` and no sequence dimension, for the reasons in the reversal section.
*Concepts:* what a `Dense` layer is computing (`Wx + b`) and why stacking two with a non-linearity
between them is more expressive than one; what `relu` is for; why the output layer has 10 units and
softmax rather than 1 and sigmoid (see the sigmoid/softmax section — this is where that knowledge
gets used); what `SparseCategoricalCrossentropy` is and how it differs from the categorical version;
what Adam is doing. Then `fit()`, watching train vs validation loss per epoch.
*Expect overfitting to be the main event:* the input is a ~20,000-dimensional sparse vector and
there are only 33,759 training documents, so the first `Dense` layer alone has more parameters than
the dataset has examples. That is the situation `Dropout` exists for, and watching it fail or work
is the lesson.

**Step 4 — `04_evaluate.ipynb`: read the training curves, then score it honestly.**
Plot loss and accuracy per epoch and diagnose what they show — the tutorial's own run overfits
after ~7 epochs and that is the lesson, not a bug. Decide on early stopping from evidence rather
than habit. Then, and only then, run once on the official test set: macro-F1 (the headline),
accuracy, per-class F1, confusion matrix. Re-state the 5% leakage caveat next to the number.

**Step 5 — `05_compare_with_classical.ipynb`: was the neural network worth it?**
`LinearSVC` on the *same* TF-IDF features, same split, same metric. Because Step 2 fixed the
representation for both, this is a genuinely controlled comparison — the only variable that moves is
the classifier — which is what makes the answer mean something. Compare on score, on training time,
and on what each model can and cannot represent. An honest "the classical model won" is a real
result, not a failure: on bag-of-words features with tens of thousands of dimensions and only 34k
examples, a linear SVM is a strong and well-matched baseline, and understanding *why* a neural
network does not automatically beat it is worth more than a better number.

**Step 6 (optional, only after 1–5 have a number) — `06_pretrained_embeddings.ipynb`: do the
Word2Vec vectors help?**
Now cheap to attempt, since Step 1 already produced the segmented corpus these vectors need. Replace
TF-IDF with an `Embedding` layer initialised from `260106_Word2Vec`'s `viwik18` vectors and re-run.
93% content-token coverage measured, so the experiment is worth running rather than assumed. Two
sub-questions that make it interesting either way: does *freezing* the pretrained embedding beat
fine-tuning it on 34k articles, and does a Wikipedia-trained embedding transfer to 2004–2006 news at
all given the proper-noun gap? This is also the first cross-project data dependency in the repo
where the input is a *model*, not a dataset — read by file path from
`260106_Word2Vec/data/outputs/`, no imports, per the repo's cross-project rule.

## How this project is being worked through

Learn-by-building, not copy-the-tutorial. For each step: the concept and the decision get discussed
first, the code gets written second, the output gets read and explained third. The notebooks are
the record of that, so they should keep the reasoning in markdown cells, not just working code.

## Findings

*(filled in as each step actually runs — the plan above is what was intended, this section is what
happened)*

## Findings — Step 1 (run 2026-08-28)

Ran end to end, all 84,132 files converted: `Train_Full` 33,759/33,759 and `Test_Full`
50,373/50,373, 418 MB in `data/processed/`. Verified on samples: UTF-8, diacritics intact, `_`
compounds present, no `\r` left, NFC. The notebook executed clean — 8/8 code cells with output, no
errors.

**Unplanned consequence: segmentation destroyed the line structure.** `file` reports the processed
documents as "with no line terminators" — `underthesea.word_tokenize(..., format="text")` returns
one flat string, so the paragraph breaks and, more importantly, the **line-0-is-the-headline
structure are gone**. Slightly ironic, since the argument for normalising CRLF to LF was precisely
that a later step might want to split title from body with `text.split("\n")`; that option no longer
exists in `data/processed/` at all.

It costs nothing for the planned work — TF-IDF is a bag of words and would have discarded the
structure anyway. But it does close a door worth knowing about: anything that wants the headline
separately (weighting the title higher, a title-only baseline, a headline-vs-body ablation) has to
go back to `data/raw/` and re-segment line by line, not read `data/processed/`. If that experiment
ever looks interesting, the fix is to segment each line and rejoin with `\n` rather than segmenting
the whole document at once.

The CRLF→LF normalisation was still the right call, just for a narrower reason than the one written
up above: it means `data/processed/` contains no stray `\r` characters for a future line-based step
to trip over, not that the line structure survived.

## Pivot: `TextVectorization` → sklearn's `TfidfVectorizer`

Step 2 was planned as `layers.TextVectorization(output_mode="tf_idf")`, staying inside TensorFlow
end to end. Reversed after building it and measuring. The reasoning is worth keeping because the
tool is not *broken* — it is being used outside what it was designed for.

**What `TextVectorization` is for.** Its whole purpose is to put preprocessing *inside the model*,
so a deployed model takes raw strings and there is no separate Python step that can drift out of
sync with training. For English that is a genuine and valuable guarantee.

**Why that guarantee is unreachable here.** `underthesea` is a Python CRF model; it cannot run
inside a TensorFlow graph. A Python preprocessing step outside the model is therefore mandatory,
and the model can never be self-contained no matter what Step 2 does. So the project was paying
every cost of `TextVectorization` for a benefit it structurally cannot collect:

| Cost | Measured |
|---|---|
| dense output | 33,759 × 10,000 × 4B = **1,350 MB** if cached |
| `lower_and_strip_punctuation` deletes `_` | `xuất_khẩu` → `xuấtkhẩu` |
| no `min_df` / `max_df` | rare junk keeps a vocabulary slot |
| vocabulary hard to read back | `côngnghiệp` instead of `công_nghiệp` |

On the underscore: the prediction was that the default standardizer would *split* the compounds and
undo the segmentation. It does not — it deletes the `_` and glues the halves, so `xuất_khẩu` stays
**one** token, just an unreadable one. Functionally survivable, which is worth recording because the
initial alarm was wrong; the real cost is that Step 4's "which words drove this prediction" analysis
becomes unreadable.

**The measurements that decided it.** Same corpus, `max_features=10000`, `min_df=3`:

| | |
|---|---|
| non-zero cells | 5,780,117 of 337,590,000 → **98.3% sparse**, 171 distinct words/article |
| dense float32 | 1,350 MB |
| scipy CSR | **46 MB — 29× smaller** |
| `fit_transform` on 33,759 docs | 10 s |
| Keras `fit()` on the CSR matrix directly | works, 4 s/epoch |

Keras accepts a scipy sparse matrix as long as the input layer declares
`Input(shape=(n_features,), sparse=True)`. So the RAM problem was never inherent to the task — it
was created by choosing a representation that forces dense output, and it disappears rather than
needing to be worked around. `max_tokens=10000` stays, not to save memory but because the tail is
single-occurrence proper nouns that cannot generalise anyway.

Keras is unaffected: still `Dense → Dense(10, softmax)`, still
`SparseCategoricalCrossentropy`. Only the source of X changes. And Step 5's comparison against
`260106_MachineLearningForNlp` gets *closer*, since both projects now use the same
`TfidfVectorizer`.

### Two findings from the pivot

**`underthesea` glues punctuation to the following word.** `word_tokenize("... vời ! Anh ấy ...")`
returns `'! Anh'` as a single token, which `format="text"` writes as `!_Anh`. Real, but **0.07% of
tokens** (113 in 165,426) — spotted because such tokens sort to the top of
`get_feature_names_out()`, which made a rounding error look like a defect. Recorded to stop it being
re-investigated later.

The mundane problem is bigger: **13.5% of tokens are bare punctuation** (`.`, `,`, `"`). Passing
`tokenizer=str.split` bypasses sklearn's default `token_pattern`, which would have dropped them, so
a custom tokenizer has to strip punctuation while preserving `_`. Note `string.punctuation`
**contains `_`** — a first attempt at measuring this flagged `Việt_Nam` and `cửa_hàng` as corrupt and
reported 27% pollution. The set has to be `set(string.punctuation) - {"_"}`.

### Rejected: moving to Colab

Considered when the RAM number first appeared, and dropped. It buys a bigger box for a problem that
should not exist, and costs re-uploading (or re-running) the 28-minute segmentation, plus session
timeouts and slower debugging. A GPU is not the bottleneck for a two-layer MLP on 27k examples —
4 s/epoch on CPU, measured. Colab becomes correct at PhoBERT or a large sequence model, not here.

### What the fitted vectorizer actually shows

`max_features=10000`, `min_df=3`, custom tokenizer, on the 33,759 segmented training articles:

**59.5% of features are compounds** (5,948 of 10,000) — `nhập_viện`, `nhẹ_nhàng`, up to
`ban_chấp_hành_trung_ương`. Worth being precise about what that proves: segmentation *changes* 60%
of the feature space, it does not prove the model gets better. That is now cheap to test, since
`data/raw/` is untouched — TF-IDF on the unsegmented corpus is a one-line variant and belongs in
Step 5 rather than being assumed either way.

**IDF confirms stopword removal is unnecessary, and reveals the opposite problem.**

| lowest IDF | | highest IDF | |
|---|---|---|---|
| và | 1.060 | contactus | 10.041 |
| của | 1.105 | document | 9.635 |
| trong | 1.143 | đột_quị | 9.481 |
| có | 1.163 | hoc_sinh | 9.481 |
| là | 1.168 | larry | 9.348 |

The low end is pure function words, weighted ~10× below the high end — TF-IDF suppresses them
automatically, so the explicit stopword-removal step the earlier three projects needed is genuinely
redundant here. But the high end is the more useful finding: **high IDF means rare, not
informative.** `contactus`, `document`, `value` are leftover HTML boilerplate; `g00`, `đkhk`, `kcb`
are fragments. TF-IDF hands its largest weights to rare junk, and `min_df=3` did not filter it.
Raising `min_df` is the lever, and Step 3 should check whether it matters.

Two entries also expose a corpus property: `hoc_sinh` (undiacriticised) and `đột_quị` (variant
spelling) exist as features separate from `học_sinh` / `đột_quỵ`. Some articles are missing
diacritics or misspelled. Small, but real.

**Underscores and diacritics survive the sklearn path**: 5,948 features contain `_`, 7,830 (78.3%)
carry Vietnamese diacritics, and 0 contain punctuation. Python's `.lower()` is diacritic-safe
(`XUẤT_KHẨU` → `xuất_khẩu`).

One landmine: `TfidfVectorizer(strip_accents=...)` defaults to `None` and **must stay there**.
Setting `strip_accents="unicode"` turns `Việt_Nam xuất_khẩu` into `viet_nam xuat_khau`, collapsing
`ma/má/mà/mã/mạ` into one token. That parameter exists for French and Spanish; for Vietnamese it is
destructive.

### The `validation_split` trap — caught by an implausible number

The first trial run reported `accuracy: 0.8208` / `val_accuracy: 0.1719` after a single epoch. Not
overfitting — one epoch cannot produce that gap. `tf.keras` `validation_split` takes the **last** x%
of the arrays **before shuffling**, and `sorted(Path.rglob(...))` groups files by class folder:

```
80% train : Chinh tri Xa hoi, Doi song, Khoa hoc, Kinh doanh,
            Phap luat, Suc khoe, The gioi, The thao (4107)
20% val   : The thao (1191), Van hoa (3080), Vi tinh (2481)
```

Training never saw `Van hoa` or `Vi tinh` at all, and validation held only three classes. The model
got `The thao` right and everything else wrong: `1191/6752 = 17.6%` against the 17.19% measured.
The arithmetic confirms the diagnosis exactly.

Fix: `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)` — `stratify` additionally
preserves the 10-class proportions in both halves, which a plain shuffle only approximates.

Worth recording as a *class* of bug rather than an incident: no exception, no warning, and it only
became visible because the consequence was extreme. On a dataset that happened to arrive
pre-shuffled, the same mistake would silently inflate every number instead.

## Findings — final results (run 2026-08-28)

All numbers on the **official VNTC test set** (50,373 articles), single run, `random_state=42`.
Features: TF-IDF, `max_features=10000`, `min_df=3`, identical for every row. Model:
`Dense(256, relu) → Dropout(0.5) → Dense(10, softmax)`, Adam, early stopping on validation loss.

| Model | Accuracy | Macro-F1 | Train time |
|---|---|---|---|
| **Neural network (segmented)** | **0.9266** | **0.9087** | 72 s |
| LinearSVC (segmented) | 0.9219 | 0.9015 | 6 s |
| Neural network (unsegmented) | 0.9174 | 0.8972 | 79 s |
| LinearSVC (unsegmented) | 0.9154 | 0.8945 | 8 s |
| MultinomialNB (segmented) | 0.8907 | 0.8677 | <1 s |
| MultinomialNB (unsegmented) | 0.8544 | 0.8277 | <1 s |
| *Floor: always largest class* | *0.1502* | *0.0261* | — |
| *Floor: uniform random* | *0.1016* | *0.0977* | — |
| *Published: RIVF'07 SVM Multi* | *0.934* | *n/a* | — |
| *Published: RIVF'07 NGRAM* | *0.971* | *n/a* | — |

### Did the neural network beat the classical model? Yes — narrowly

+0.47 points accuracy, +0.72 macro-F1 over `LinearSVC` on identical features. Real and consistent,
but small, and bought with **12× the training time**. The two claims "the neural network won" and
"the neural network was worth it" are not the same, and only the first is supported.

The reason is worth stating rather than treated as a disappointment: on 10,000-dimensional
bag-of-words features, topic classification is close to linearly separable, which leaves a hidden
layer very little structure left to discover. A `Dense(256)` layer can represent word *interactions*
that a linear model cannot — but for "does this article contain `cầu_thủ` and `bàn_thắng`", there
are barely any interactions to represent. This is a property of the task and the representation, not
a failure of the model.

### Was `underthesea` worth 28 minutes? Yes, and the size of the gain is itself informative

Every model improves on segmented text, but by very different amounts:

| Model | Δ accuracy | Δ macro-F1 |
|---|---|---|
| Neural network | +0.92 | +1.15 |
| LinearSVC | +0.65 | +0.70 |
| MultinomialNB | +3.63 | +4.00 |

**The weaker the model, the more it depends on the features being right.** `MultinomialNB` treats
every feature as independent evidence, so `kinh` and `doanh` as two weak, ambiguous signals hurt it
badly; `kinh_doanh` as one strong signal helps it a lot. The stronger models can partly compensate
for bad tokenization by weighting co-occurring fragments, so they gain less. One table, one general
principle about the preprocessing/model trade-off.

Recorded because the earlier reasoning was wrong in an instructive way: the argument for deferring
`underthesea` assumed a `GlobalAveragePooling1D` bag-of-embeddings model, where averaging blurs the
input and tokenization matters less. Once the representation became TF-IDF — where the vocabulary
*is* the feature space — the conclusion flipped, and the measurement confirms it.

### Against the published baselines

0.9266 lands just under RIVF'07's SVM Multi (0.934) and well below NGRAM (0.971). Two honest
qualifications: the 2007 paper does not document its preprocessing in enough detail to reproduce, so
this is a sanity band rather than a like-for-like comparison; and every number in the table — ours
and theirs — is inflated by the 5% train/test overlap in the published corpus. Landing within a
point of a 2007 SVM using syllable-free segmentation and a two-layer network is roughly where this
should land.

### Per-class results — where the macro-F1 actually goes

The 0.9087 macro-F1 is not evenly earned. Per-class F1, worst to best:

| Class | F1 | | Class | F1 |
|---|---|---|---|---|
| **Doi song** | **0.7347** | | Van hoa | 0.9423 |
| Khoa hoc | 0.8479 | | Suc khoe | 0.9442 |
| Chinh tri Xa hoi | 0.8909 | | The gioi | 0.9503 |
| Phap luat | 0.9190 | | Vi tinh | 0.9535 |
| Kinh doanh | 0.9217 | | The thao | **0.9823** |

The spread is the whole story: `The thao` at 0.98 and `Doi song` at 0.73. Sport has a closed,
unmistakable vocabulary (`cầu_thủ`, `bàn_thắng`, `vô_địch`, `hlv`) — exactly what TF-IDF is built
to exploit. `Doi song` ("daily life") has no vocabulary of its own; it is defined by not being one
of the other nine.

This is also where the accuracy/macro-F1 gap comes from: 0.9266 vs 0.9087. `Doi song` is only 4.0%
of the test set, so failing on it barely dents accuracy, while macro-F1 weights it equally with
`The thao`. The metric choice made in Step 0 turned out to matter by ~1.8 points.

**Top confusions** (row-normalised):

```
11.6%  Doi song   -> Van hoa
10.1%  Doi song   -> Chinh tri Xa hoi
 6.1%  Phap luat  -> Chinh tri Xa hoi
 5.1%  Kinh doanh -> Chinh tri Xa hoi
 4.3%  Khoa hoc   -> Suc khoe
 4.1%  Khoa hoc   -> Chinh tri Xa hoi
```

The prediction written here before the matrix was computed — that `Chinh tri Xa hoi` would absorb
errors from almost every class as the broadest category — holds: it is the destination in four of
the six largest confusions. The part not predicted is that the single largest is
`Doi song → Van hoa`, two categories whose boundary is genuinely a matter of editorial convention
rather than content. `Khoa hoc → Suc khoe` is the same kind of overlap: a medical-research article
is legitimately both.

None of these are the model failing to read the text. They are the same ambiguity that produced the
219 duplicate-but-differently-labelled articles found in Step 0, showing up as errors because the
task forces a single answer.

### Overfitting, and early stopping doing its job

Final epoch: **train accuracy 0.9882 against validation 0.9163** — a 7-point gap, on 2,562,826
parameters and 27,007 training examples. Best validation loss came at **epoch 4**; training ran to
7 and `restore_best_weights=True` rewound. The tutorial's fixed 10 epochs would have shipped a
visibly worse model, and its own advice to "consider `EarlyStopping`" turns out to be the load-
bearing part rather than an aside.

### Hyperparameters did not matter

The validation sweep (`max_features` ∈ {10k, 20k} × `min_df` ∈ {3, 5}) spanned **0.001 macro-F1**.
Doubling the vocabulary added nothing. The signal is concentrated in common vocabulary and the rare
tail is noise either way — which also retires the earlier worry that high-IDF junk (`contactus`,
`g00`) was hurting: raising `min_df` to cut it changed nothing measurable.

Related correction: those high-IDF tokens were initially written off as crawler junk. Checking them
against the raw corpus showed `kcb` is the abbreviation for *khám chữa bệnh*, `larry` is a person's
name in a quoted interview, and `value` comes from a code snippet inside a genuine `Vi tinh`
article. Real content, not noise. The corpus turned out to be far cleaner than assumed — 41 URLs
and **zero** HTML tags in a 3,000-article sample — so the repo's established `clean_text` regex
suite was not applied wholesale. The one part that mattered was digit removal (43,648 numbers in
that sample), which the tokenizer handles directly, and no re-segmentation was needed.

## Tuning that was tried and deliberately not adopted

After the results above were in, one round of improvement was tested on the **validation set only**,
to answer "is this as good as it gets, or is a point being left on the table?"

| Variant | val accuracy | val macro-F1 |
|---|---|---|
| NN 256 (the shipped model) | 0.9208 | 0.9198 |
| NN 256 + `class_weight="balanced"` | 0.9194 | 0.9176 |
| NN 256 + `sublinear_tf` | 0.9206 | 0.9196 |
| **NN 256 + `sublinear_tf` + `class_weight`** | **0.9234** | **0.9226** |
| NN 512 + both | 0.9215 | 0.9203 |
| NN 512→256 + both | 0.9217 | 0.9201 |
| NN 256, dropout 0.3 + both | 0.9224 | 0.9213 |
| LinearSVC + `class_weight` + `sublinear_tf` | 0.9191 | 0.9176 |

**Decision: keep the original model.** The best variant is +0.28 macro-F1 on validation — inside the
noise of a single seed, and never confirmed on test. Three things are worth keeping from the
exercise anyway:

1. **More capacity actively hurt.** 512 units and 512→256 both scored *below* 256. The model is not
   capacity-limited; it is limited by what bag-of-words features can express.
2. **`class_weight` only helped in combination.** Alone it made things worse (0.9176), and only
   turned positive alongside `sublinear_tf` (0.9226). Interpreting either in isolation would have
   been wrong.
3. **The two knobs that would matter are architectural, not hyperparameter.** Everything reachable
   from this representation lands within ~0.3 points of everything else. Getting meaningfully past
   ~0.92 means changing the representation — pretrained embeddings (the Step 6 sketch) or a
   transformer — not tuning this one.

That third point is the real answer to "can this be improved?": not from here, and the sweep is the
evidence rather than an assumption.
