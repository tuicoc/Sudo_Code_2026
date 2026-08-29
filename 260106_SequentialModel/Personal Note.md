# Learning Log: Sequential Models — an LSTM that writes Vietnamese

Goal: build a character-of-the-book language model — an LSTM trained to predict the next token — on
a corpus of Vietnamese books, and use it to generate text. The reference for the mechanism is
Christopher Olah's [Understanding LSTM
Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/); the point of the project is
to understand *why* an LSTM has a cell state and three gates, not just to call `layers.LSTM`.

This is the first **generative** project in the repo. Everything before it —
`260106_TextPreprocessingwithNLP`, `260106_Scikit-learnTextFeatureExtraction`,
`260106_MachineLearningForNlp`, `260106_DeepLearningForNlp` — was discriminative: given a document,
pick a label. Here there is no label at all, and the training signal is the text predicting itself.

## The dataset, and the first thing that had to change

[iambestfeeder/10000-vietnamese-books](https://www.kaggle.com/datasets/iambestfeeder/10000-vietnamese-books)
— 10,415 `.txt` files, **1.73 GB**, all UTF-8, 37/40 sampled files already NFC.

The project started as a classification task and had to be re-scoped after the survey, which is
worth recording rather than quietly fixing. Two things the survey found:

**There are no labels.** No CSV, no JSON, no metadata of any kind — 10,415 `.txt` and one
`.complete` marker, nothing else. The only derivable label is the **author**, from a filename
pattern (`Title - Author.txt`) that matches all 10,415 files. That is a real option (429 authors
have ≥5 books, 80 have ≥20) but it needs the junk entries dropped first: the two largest "authors"
are `nhiều tác giả` (203 books) and `Khuyết Danh` (181) — "various" and "anonymous".

**These are whole books, not blurbs.** Median 22 KB, p95 842 KB, largest 20 MB. The initial plan
assumed short descriptions ("a 200-word description becomes ~340 tokens"); the actual unit here is a
novel.

Re-scoped to **language modelling / text generation**, which needs no labels and uses the text
directly. Author classification stays available as a later variant.

## The measurement that reversed the plan's premise

The stated reason for segmenting with `underthesea` was that it is a free sequence compression:
Vietnamese words average ~1.7 syllables, so segmentation should roughly halve the sequence length,
and LSTM cost is linear in length. Both halves of that turned out to need correcting.

**The 1.7 figure is a lexical statistic, not a corpus one.** Most Vietnamese *dictionary entries*
are disyllabic, but in running text the most *frequent* tokens are monosyllabic function words.
Measured over 25 books, counting by token frequency and excluding punctuation:

| word length | share of tokens |
|---|---|
| 1 syllable | **77.3%** |
| 2 syllables | 21.4% |
| 3+ syllables | 1.2% |

That averages **1.226 syllables per word**, not 1.7. Segmentation shortens sequences by **18%**, not
by roughly half.

(First attempt at this measurement returned 1.09, which looked wrong and was: `underthesea` splits
punctuation into separate tokens, inflating the denominator. Filtering punctuation on both sides
gives the 1.226 above. Worth recording because the same mistake would silently understate the effect
anywhere else it is measured.)

**And for a language model the compression is not free — it is paid for at the softmax.** On a
60-book sample:

| | tokens | vocabulary |
|---|---|---|
| unsegmented | 145,896 | 7,882 |
| segmented | 121,882 (**84%**) | 13,921 (**177%**) |

Joining `kinh` + `doanh` into `kinh_doanh` removes a timestep but *creates a new vocabulary entry*.
A language model's output layer is a softmax over the whole vocabulary at **every** timestep, so
shorter sequences and a bigger vocabulary pull in opposite directions — unlike the previous project,
where TF-IDF features had no per-timestep softmax and segmentation was a one-sided win.

Measured on this machine (Intel i9, CPU only), `Embedding(128) → LSTM(256) → Dense(vocab)`,
sequence 100, batch 64:

| vocabulary | throughput | 2M tokens |
|---|---|---|
| 8,000 | 8,212 tok/s | 4.1 min/epoch |
| 14,000 | 6,713 tok/s | 5.0 min/epoch |

So the 77% larger vocabulary costs **18% throughput** — almost exactly cancelling the 18% saved on
sequence length. The naive estimate (cost ∝ tokens × vocab → 1.48× worse) overstates it, because the
LSTM's own recurrent cost is independent of vocabulary size and is a large share of the total.

**Net: for this task the two effects cancel, and segmentation is neither the free win the plan
assumed nor a loss.** Which makes running both variants the point rather than a formality — the
interesting question is no longer speed but whether word-level tokens produce better *text*.

## Padding and masking

Raised as a concern before any code was written, and correct — but it applies to only one of the two
ways of building language-model training data, so which one this project uses is the decision that
settles it.

- **Fixed-length windows over concatenated text**: chop the whole corpus into equal `SEQ_LEN` spans.
  Every sequence has identical length, so **there is no padding and nothing to mask.** This is the
  standard construction for language modelling.
- **Per-sentence or per-paragraph sequences**: lengths vary, so short ones get padded with 0 — and
  then masking is mandatory.

Verified what actually happens in Keras 3 rather than assuming, by training a small model and then
changing the labels *only at padded positions*. If the mask works, the loss must not move:

| | labels at pad = 0 | labels at pad = 7 | |
|---|---|---|---|
| `mask_zero=False` | 3.3587 | **3.8899** | loss moves → padding is being trained on |
| `mask_zero=True` | 3.8941 | **3.8941** | loss identical → mask reaches the loss |

So `Embedding(vocab, 128, mask_zero=True)` genuinely propagates through `LSTM` into the loss. Without
it, a generative model is explicitly taught that the token following real text is padding — which is
exactly what it will then generate.

Two attached gotchas:

1. **`mask_zero=True` reserves index 0.** The vocabulary must start at 1; any real word assigned
   index 0 is silently treated as padding and disappears from its sequence.
2. **`sample_weight=0` at padded positions also works** and is the explicit alternative, but it
   normalises differently — Keras divides by the element count rather than by the weight sum, so the
   reported loss value is scaled (1.4677 vs 3.8941 for the same masked positions). The gradients
   point the same way; only the printed number differs. Worth knowing before comparing two runs that
   used different masking styles.

A first attempt at this test compared masked and unmasked loss on an *untrained* model and found no
difference (3.9093 vs 3.9127) — meaningless, because an untrained model predicts near-uniformly and
every position has loss ≈ log(vocab) whether it is padding or not. The test only became informative
after training the model first.

## Corpus size — what a CPU can actually do

The full 1.73 GB is roughly 430M syllables; at ~8,000 tokens/s that is **15 hours per epoch**. Not
possible here, and not necessary: the project needs a corpus big enough to learn Vietnamese
structure, not the whole library.

Target **~2M tokens** (≈0.5% of the corpus, a few dozen books): 4–5 minutes per epoch, so ~10 epochs
per variant and two variants fits in a couple of hours. Selection is a fixed, seeded sample so the
two tokenization variants see exactly the same books.

## No stopword removal, no n-grams

Continuing the convention from `260106_DeepLearningForNlp`, and for sharper reasons here.

**N-grams are meaningless for this model.** They exist to smuggle local word order into
order-blind representations; an LSTM consumes the sequence in order by construction. There is
nothing to add.

**Stopword removal would be actively destructive.** It was skipped in the previous project because
TF-IDF already down-weights function words. Here the reason is stronger: 77.3% of tokens are
monosyllabic function words, and they *are* the grammar. A generative model with `và`, `của`, `là`,
`đã` deleted cannot produce a well-formed Vietnamese sentence — it would emit content words in a
row. The thing being modelled is precisely the glue that stopword removal throws away.

## The plan

**Step 1 — `01_prepare_corpus.ipynb`.** Survey, then build the two corpora. Fixed seeded sample of
books to ~2M tokens; clean (NFC, punctuation policy, lowercase decision); produce an unsegmented
version and an `underthesea`-segmented version of *the same books*; save both as token id arrays plus
their vocabularies. Decide `SEQ_LEN` from measured sentence/paragraph structure rather than by habit.

**Step 2 — `02_lstm_language_model.ipynb`.** The model, and what each part of an LSTM is for —
cell state as the conveyor belt, forget/input/output gates, why this survives long dependencies where
a plain RNN's gradients vanish (following Olah). Build
`Embedding(mask_zero=True) → LSTM → Dense(vocab)`, train on the segmented corpus, watch loss and
perplexity.

**Step 3 — `03_generate.ipynb`.** Sampling: greedy vs temperature vs top-k, and why greedy decoding
loops. Generate from seed prompts, read the output honestly.

**Step 4 — `04_compare_tokenization.ipynb`.** The same model on the unsegmented corpus. Compare
perplexity (noting that perplexity across *different vocabularies is not directly comparable* — this
needs a normalisation, probably per-syllable, and getting that right is part of the step), wall-clock
training time, and generated-text quality side by side.

## What actually ran — one notebook, not four

The four-notebook plan above is not what shipped, and the difference is worth stating plainly rather
than leaving the file listing to imply otherwise.

`01_prepare_corpus.ipynb` ran locally and is the only one of the three that has outputs: it produced
`data/processed/` (651 MB `train_tokens.npy`, 17 MB `val_tokens.npy`, `vocab.json`) from the full
1.73 GB corpus. `02_lstm_language_model.ipynb` and `03_generate.ipynb` were written against those
files — a CPU subset locally, Drive paths for a Colab run — and **neither was ever executed**. Their
cells have no outputs, and their contents were superseded before they could be.

**Why the training left this machine at all: there is no GPU on it.** The throughput measured here
was 8,212 tokens/s, and that was on a *smaller* model than the one that ended up training
(`Embedding(128) → LSTM(256)`, vocabulary 8,000 — against 256 / 512 / 20,000). At that rate:

| | on this CPU | Kaggle T4 x2 |
|---|---|---|
| throughput | 8,212 tok/s (smaller model) | 53,128 tok/s (the real model) |
| 17M-token demo slice, 1 epoch | ~35 min | 5.3 min |
| the same 15 epochs | ~8.6 h | **80 min** |
| full 334M-token corpus, 1 epoch | ~11 h | ~2 h |

So the GPU is not a convenience here, it is the difference between a run measured in minutes and one
measured in days — and the real gap is wider than the table, because the CPU number was measured on a
model roughly four times cheaper at the output layer. Worth contrasting with
`260106_DeepLearningForNlp`, which reached the opposite conclusion and was right to: an MLP over
TF-IDF features trained in 4 seconds an epoch on this same CPU. The distinction is not "neural
network or not" — it is a few large matrix multiplications versus millions of small sequential steps.

What replaced them is `lstm_vietnamese_books.ipynb`: preprocessing, training and generation in a
single self-contained file that detects Kaggle, Colab or local and sets its paths accordingly. Colab
was the first attempt and is where the out-of-memory below happened; **Kaggle won on three counts** —
the corpus is *already mounted* read-only under `/kaggle/input/`, so nothing has to be uploaded
(Colab needed 668 MB of `.npy` pushed to Drive first); the session has ~29 GB of RAM against Colab's
~12.7 GB, which is what the crash below was about; and the accelerator offered is **T4 x2** rather
than one. A single notebook is also the only form that survives being re-run from scratch on a fresh
session, which is what a hosted runtime forces.

The two dead notebooks stay in the repo as the record of the local path. Nothing below came from
them.

## Findings — the Colab out-of-memory, and what actually caused it

The combined Colab notebook crashed on its first real run, partway through preprocessing: about
book 5,000 of 10,415, roughly four minutes in. The natural reading was that the batch size was too
large. It was not — and the timing is what rules it out. **The crash happened before `BATCH_SIZE`,
`train_ds` or the model existed**; nothing had been batched yet.

The cause was a shortcut taken while merging the three local notebooks into one file. To avoid
reading each book twice, `build_corpus_from_scratch()` accumulated every book's token list:

```python
toks_per_book.append(toks)          # <- holds the whole corpus as Python strings
counts.update(toks)
```

Measured on a 30-book sample, a token held as a Python `str` costs **59.3 bytes** — the object
header dominates, the characters are almost incidental. For 334,149,238 tokens:

| representation | size |
|---|---|
| `list[str]` (what the code held) | **19.8 GB** |
| `numpy.uint16` (what it becomes) | 0.67 GB |

Free-tier Colab has ~12.7 GB, so it dies somewhere past the halfway mark. That matches the observed
failure point almost exactly, which is what confirms the diagnosis rather than merely being
consistent with it.

Worth noting the local `01_prepare_corpus.ipynb` never had this bug: it runs two `ProcessPoolExecutor`
passes whose workers return a `Counter` and then a `uint16` array, so tokens are never all resident.
The bug was introduced by *simplifying* correct code into a single pass — the classic shape of this
mistake.

**Fix:** two streaming passes. Pass 1 counts and discards; pass 2 re-reads and encodes straight into
`uint16` via `np.fromiter`. Costs one extra read of the corpus — about 17 minutes on Colab's 2 vCPUs —
and never holds more than a `Counter` plus a list of small integer arrays. The faster route is to run
Step 1 locally (4 minutes with 8 processes) and upload the 668 MB of `.npy` to Drive; the notebook
detects them and skips preprocessing.

### A second memory trap found while fixing the first

`tf.data.Dataset.from_tensor_slices(windows)` — the obvious way to build the pipeline, and what the
first draft used — copies the entire array into a **graph constant**. That is another 0.65 GB on top
of the numpy array, and if TF upcasts `uint16` to `int32` it becomes 1.3 GB, close enough to the 2 GB
`GraphDef` protobuf limit to be worth avoiding.

Replaced with `from_generator` yielding whole batches, gathering rows from a reshaped **view** of the
token array. Verified by measuring peak RSS across dataset creation: **1.00 GB before, 1.04 GB
after** — the pipeline adds one batch, not a copy of the corpus. Data-only throughput is 6.46M
tokens/s, roughly three orders of magnitude above what the GPU will consume, so the Python generator
is nowhere near being the bottleneck.

## The run that stands — a demo, deliberately small

One decision governs everything below and is worth stating before the numbers rather than after:
**this run was scoped as a demo.** 500 books out of 10,415, 15 epochs, 80 minutes — chosen so the
whole pipeline could be shown working end to end in one sitting, not so the model would be good. The
full corpus is 334M tokens at roughly 2 h/epoch even on the T4 pair — a couple of days of GPU time
for a meaningful number of epochs, against a free-tier session that is capped at 12 hours and a
weekly GPU allowance that is finite. Nothing here was ever going to approach it, and pretending the
configuration was tuned for quality would misread what was being tested.

What the run had to prove was mechanical: that preprocessing survives 10,415 files without running
out of memory, that the input pipeline feeds a GPU, that the loss goes down, and that the sampling
code produces Vietnamese rather than `<unk>` soup. It proved all four. The perplexity is a
by-product, not the goal.

Kaggle T4 x2, TF 2.20, `mixed_float16`, `MirroredStrategy` across both GPUs.

| | |
|---|---|
| corpus | 500 books, seeded sample (rng 42) of the 10,415 — 0.09 GB, **17.2M tokens**, 41,501 distinct |
| vocabulary | 20,000 (the rest → `<unk>`), split by book, `<eob>` at each boundary |
| train / val | 17,009,016 / 208,852 tokens |
| model | `Embedding(20000, 256) → LSTM(512, return_sequences) → Dense(20000, float32)` — 16,954,912 params |
| optimiser | Adam 5e-4, `clipnorm=1.0`, global batch 256 (128 × 2 replicas) |
| training | 15 epochs, 657 steps each, ~5.3 min/epoch, **80.0 min total**, 53,128 tokens/s |

| | epoch 1 | epoch 15 |
|---|---|---|
| train loss | 6.5401 | 4.9008 |
| val loss | 6.1458 | **5.3953** |

**Val perplexity 220.4**, against the 20,000 of uniform guessing. So the model is choosing between
roughly 220 tokens where an untrained one chooses between 20,000 — real learning, and nowhere near a
model that has understood Vietnamese.

Three things that run says:

**It was stopped by the demo budget, not by convergence.** Validation loss fell at *every* one of the
15 epochs and `EarlyStopping(patience=2)` never fired — the run ended because 15 epochs was what had
been allotted. 220 is simply where a short run on 5% of the corpus lands. It is not a ceiling this
architecture reached, and no conclusion about LSTM capacity should be drawn from it.

**fp16 did not show up in the throughput** — and at this scale that was left alone rather than
chased. The benchmark measured 3.5 TFLOP/s and printed "running
at fp32 speed", which understates the problem: that figure is computed from the *global* batch, so it
is the total across both T4s — about 1.75 each, below even single-GPU fp32. (The benchmark's
thresholds were written for one GPU and are wrong by 2× under `MirroredStrategy`; worth fixing before
they are trusted again.) Not diagnosed, because 80 minutes was affordable either way; it would have
to be, before any run on the full corpus. The two candidates are that the LSTM is 100 sequential small
steps — latency-bound work that tensor cores cannot help — and that the 10.3M-parameter output layer
is all-reduced every step. The FLOP split says the output layer is 87% of the arithmetic, so the
second is the one to measure first.

**The arithmetic estimate was right about where the time goes, and that was the useful part.** 1814
GFLOP per step, 87% of it in `Dense(512 → 20000)`, is what made mixed precision the lever to reach
for rather than cuDNN — and it is why the LSTM being 13% of the cost was worth knowing before
spending an hour on it.

## Reading the generated text honestly

Seed: *ông ấy nhìn ra ngoài cửa sổ và*

**Greedy collapses**, exactly as predicted: `- anh không biết. - anh không biết.` repeating to the
token limit. Not a bug — it is what "always take the most probable token" means once the model
reaches a state whose likeliest continuation returns it to that state.

**Temperature 0.5** produces fluent, dialogue-shaped Vietnamese: *thấy một người đã trả lời: - anh tự
thông, em không thể nói...* Punctuation lands in sentence-shaped places, the dialogue dash convention
of the corpus is reproduced, clause openings are well-formed. **0.8** drifts mid-sentence; **1.2** is
word salad with rare tokens surfacing (`pessoa`, `hì hả`). **Top-k 40 at 0.9** is the best of them,
and it is still nowhere near coherent past a clause.

Against the four-level ladder in the notebook: real words ✓, local grammar ✓, sentence-level
coherence partially, cross-sentence coherence no. For 80 minutes on 5% of the books that is about
what should be expected — the first two levels are cheap, and the demo was never going to reach the
last. That last one is the honest limit of a single
512-unit LSTM carrying the whole past in one fixed-size vector — and it is precisely the failure that
attention was invented to fix. The corpus being dialogue-heavy fiction shows through too: the model's
favourite output is a line of dialogue, because that is what 500 novels mostly are.

## Where this stops

This is the end of the project, and it stops at the demo. The pipeline runs, the model learns, the
sampler produces Vietnamese — that was the goal, and going further would have meant paying for GPU
hours to improve a number nothing depends on. What is *not* here, so it is not mistaken for an
oversight:

- **No full-corpus run, and none intended.** 334M tokens is ~2 h/epoch even on a T4. Everything above
  is 5% of the books, chosen so that 15 epochs cost 80 minutes instead of a day.
- **Nothing was tuned.** One configuration, one seed, one run. No sweep over `LSTM_UNITS`, embedding
  size, learning rate or `SEQ_LEN`, no second layer, no dropout. Every number is a first attempt.
- **Step 4, the tokenization comparison, was never run.** Its premise had already been measured away
  before any model existed: segmentation shortens sequences 18% and grows the vocabulary 77%, and the
  two cancel. That left only "does word-level tokenization produce better *text*", which is a real
  question and stays unanswered — and answering it properly needs two runs long enough to tell apart,
  which is exactly what a demo-scale budget cannot buy.
- **The artefacts live on Kaggle, not in this repo.** `lstm_lm.keras`, `lm_metrics.json`,
  `learning_curve.png` and the token arrays are in `/kaggle/working` from that session;
  `data/` is gitignored here, and the local `data/outputs/training_log.csv` is a 0-byte leftover from
  a local start that was abandoned in favour of Kaggle.
- **`02_lstm_language_model.ipynb` and `03_generate.ipynb` have no outputs and will not run as-is**
  against a fresh checkout — they expect `data/processed/` and a Drive folder. `lstm_vietnamese_books.ipynb`
  is the notebook to run.
