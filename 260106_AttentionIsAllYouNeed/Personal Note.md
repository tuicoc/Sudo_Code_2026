# Attention Is All You Need — what is attention actually worth?

| | |
|---|---|
| **Goal** | Implement an attention mechanism by hand and measure what it adds to Vietnamese news summarization |
| **Dataset** | VNDS / vietnews — 143,816 news articles with human abstracts, word-segmented ([HuggingFace](https://huggingface.co/datasets/nam194/vietnews)); 15,000 used |
| **Result** | Attention **+1.05 ROUGE-1** over the identical model without it — and **both lose to copying the article's first sentence** (26.28) |

The project is a controlled comparison, not a summarization system. One network, trained three
times, differing only in how the decoder reads the article.

---

## 1. How to run

### What is in here

| File | What it does |
|---|---|
| `config/config.yaml` | Truncation lengths, vocabulary size, hyperparameters, decoding settings |
| `src/config.py` | Reads `config.yaml` and turns its paths into absolute paths |
| `src/dataloader.py` | `DataLoader` — download the parquet corpus, read/write every artifact |
| `src/preprocessing.py` | `Preprocessor` — truncate, split sentences, measure what truncation costs |
| `src/vocabulary.py` | `Vocabulary` — the shared 20k vocabulary, coverage table, encode/decode |
| `src/attention.py` | `AdditiveAttention`, `DotProductAttention` — the exercise, written out |
| `src/model.py` | `Seq2Seq` — GRU encoder-decoder, with attention or without |
| `src/decoding.py` | `Decoder` — greedy decoding, n-gram blocking, attention weights |
| `src/evaluation.py` | `Evaluator` — ROUGE-1/2/L and the Lead-n baselines |
| `main.py` | 4 stages: `prepare` → `train` → `evaluate` → `summarize` |
| `notebooks/attention_summarization_vietnews.ipynb` | The whole project as one file, with the run's outputs |

### Run

```bash
pip install -r requirements.txt
python main.py                    # prepare -> train -> evaluate -> summarize
python main.py --stage prepare    # download, measure, build the vocabulary, encode
python main.py --stage train      # fit every variant in training.variants
python main.py --stage evaluate   # decode the test set, ROUGE against Lead-n, plots
python main.py --stage summarize  # print a few summaries side by side
```

No login needed — the corpus downloads from HuggingFace. On a CPU this takes hours; the notebook
is the practical path and produced everything below in ~40 minutes on one Kaggle T4.

---

## 2. Results

15,000 training articles of 99,134, truncated to 256 tokens; 1,000 test articles. One T4,
float32, one seed. TensorFlow 2.20.0 on Kaggle (the code also runs on 2.16.2 locally).

| | ROUGE-1 | ROUGE-2 | ROUGE-L | tokens | best val loss | train |
|---|---|---|---|---|---|---|
| Lead-1 baseline | **26.28** | 8.64 | **21.18** | 29 | — | — |
| Lead-2 baseline | **26.76** | **9.53** | 20.10 | 59 | — | — |
| Lead-3 baseline | 24.69 | 9.17 | 18.07 | 86 | — | — |
| no attention | 15.95 | 1.14 | 13.53 | 24 | 5.2628 | 7.3 min, 11 ep |
| additive attention | **17.00** | **1.66** | **14.20** | 27 | **5.2374** | 10.5 min, 14 ep |
| dot-product attention | 16.56 | 1.25 | 13.87 | 24 | 5.2514 | 9.4 min, 14 ep |

**Attention won, consistently but small.** +1.05 ROUGE-1, +0.52 ROUGE-2, +0.67 ROUGE-L, and a
lower validation loss — four measurements agreeing, with dot-product between the other two on
every one of them. It is one seed, so the ordering is what this supports, not the exact gap.

**The three models converge to the same validation loss** (5.24–5.26). That is the finding I did
not expect. Attention did not lower the floor; it changed what the decoder emits at that floor.
The bottleneck story assumes the model is limited by the single vector between encoder and
decoder — at 15,000 examples it is limited by not having read enough Vietnamese.

**Everything loses to the first sentence of the article**, by about ten ROUGE-1 points, and the
ROUGE-2 column says why: 1.14–1.66 against 8.64–9.53 means the models nearly never get two words
in a row right. They write fluent generic news prose with invented facts. For an article about
China at the Shangri-La Dialogue, the best model wrote *"Bộ Ngoại_giao Mỹ cho_biết ông sẽ có cuộc
tập_trận chung với Mỹ và Trung_Quốc."* — grammatical Vietnamese, correct register, wrong country,
wrong event.

**The attention weights are mostly flat.** Uniform attention over 256 positions is 0.0039; this
model sits at 0.005–0.02 with occasional spikes to 0.21. Averaged over output steps it peaks
around positions 74 and 145, not at the lead. Implementing attention correctly and having the
model learn to align turn out to be two different achievements.

---

## 3. Experiments

| Tried | Result | Kept? |
|---|---|---|
| Attention vs no attention, everything else identical | +1.05 ROUGE-1, +0.52 ROUGE-2, lower val loss | **Yes** — this is the project |
| Dot-product scoring instead of additive | Lands between the two, 12% faster per epoch, 132× less memory | Yes, as the third comparison |
| Truncate articles at 256 tokens | Measured first: keeps 73.4% of the abstract's words vs 80.1% for the whole article | Yes — 6.7 points for a 40% shorter sequence |
| Vocabulary 20k | Covers 97.24% of tokens; only 2.00% `<unk>` in the summaries the model must produce | Yes |
| Lead-3 as *the* baseline, as papers usually report | Lead-1 (26.28) and Lead-2 (26.76) both beat Lead-3 (24.69) | No — report all three |
| 3-gram blocking at decode time | Without it greedy decoding repeats a phrase to the length limit | Yes |
| `mixed_float16` | The `-1e9` attention mask is below fp16's -65504 → `-inf` → NaN loss | No — fp32, runs are minutes anyway |
| `MirroredStrategy` on T4 ×2 | Previous project measured 1.6–1.9×, not 2×; these runs are 7–10 minutes | No — one GPU |
| Function metric for masked accuracy | Crashes at step 1: Keras hands the propagated mask to it as `sample_weight` | No — `Metric` subclass |
| Feed the context back into the GRU (true Bahdanau) | Needs a Python loop over output steps, ~10× slower to train | No — noted as a real limitation |

**The measurement that set the input length.** `max_article` is the most expensive number in the
project: attention memory grows with `article_length × summary_length`. Instead of picking 256 by
feel, I measured what truncation throws away — the share of the abstract's words still present in
the first N article tokens:

| first N tokens | abstract words found |
|---|---|
| 64 | 46.7% |
| 128 | 61.4% |
| **256** | **73.4%** |
| 400 | 77.4% |
| whole article | 80.1% |

Past 256 the curve flattens: the last ~160 tokens of an average article contribute 2.7 points.
The same table is the first evidence of the lead bias that makes Lead-1 so hard to beat.

---

## 4. What I learned

**A correct implementation and a working mechanism are different things.** The attention layer
does what the paper says — valid distributions, occasional sharp spikes at 0.21 — and the model
still mostly averages the article instead of pointing at it. Having written the mechanism, I can
now say what "attention learned to align" would look like, and this is not it.

**Read ROUGE-2 first.** ROUGE-1 of 17 looks like partial success. ROUGE-2 of 1.66 says the model
produces almost no correct word pairs, which is the real state of it. A single-word overlap
metric rewards a model for knowing what Vietnamese news vocabulary sounds like.

**F1 makes the baseline shorter, not longer.** I expected Lead-3 to be the strongest baseline
because it contains more of the abstract's words. It scores lowest of the three: ROUGE is F1, and
86 tokens of prediction against a 29-token reference is charged for the 57 extra tokens. The
precision term is doing the work.

**The fp16 trap in attention is arithmetic, not framework-specific.** Masking padded positions
means adding a large negative number before the softmax. fp16's most negative value is -65504, so
`-1e9` becomes `-inf`, and `softmax([-inf, ...])` is NaN. The symptom is a loss that goes NaN with
nothing visibly wrong in the model. Computing scores in float32 costs nothing and removes it.

**Masking lives in three places and I only got warned about one.** The encoder GRU must skip
padded steps, the attention softmax must not spend weight on them, and the loss must not be graded
on them. The one that actually crashed was none of those — it was the *metric*: with
`mask_zero=True` Keras hands the propagated mask to a function metric as `sample_weight` and tries
to broadcast a `(batch, time)` array onto the scalar it returned. The error message
(`rank 2 into rank 0`) names a Keras internal and not the mask. A `Metric` subclass fixes it, and
is more correct anyway — totals accumulate instead of averaging per-batch means.

**Attention costs a slow start, not just time per epoch.** Both attention variants sat near their
initial loss for the first two to three epochs before learning anything, while the plain model was
already down to 5.66 by epoch 3. With the 15-epoch budget the plain model early-stopped at 11 and
the attention ones were still improving at 14 — the comparison is fair on budget, not on
convergence.

**The paper's efficiency argument is checkable with a shape.** Additive scoring, vectorised over
all output steps, builds a `(batch, summary_len, article_len, attn_units)` tensor — here
32 × 63 × 256 × 64 = 33M floats, 132 MB, before the backward pass. Dot-product never leaves
`(batch, summary_len, article_len)`, 132× smaller. That is exactly why *Attention Is All You Need*
picks it, and it is visible in one line of arithmetic rather than a benchmark.

**Where this stops.** One seed, one configuration, 15% of the corpus, no tuning. The claim that
data rather than architecture is the binding constraint is an inference from three coincident
val-loss curves, not a measurement — the run that would settle it is the same three models on
50,000 articles, which was outside the compute budget for this exercise.
