# Transformer Model — English → Vietnamese, and does cross-attention really align?

| | |
|---|---|
| **Goal** | Build the encoder–decoder Transformer for translation and report the translation quality |
| **Dataset** | EVBCorpus / EVBNews — 1,000 news documents, 45,308 sentence pairs, hand-aligned at word level ([GitHub](https://github.com/qhungngo/EVBCorpus)) |
| **Result** | **BLEU 16.57** EN→VI — and cross-attention agrees with the human word alignment **40.9% of the time against 5.1% chance** |

Second Transformer project. The previous one wrote the attention mechanism by hand on a summarizer;
this one uses `keras.layers.MultiHeadAttention` and is about the whole architecture, and about
whether the alignment story people tell about cross-attention is true.

---

## 1. How to run

### What is in here

| File | What it does |
|---|---|
| `config/config.yaml` | Lengths, vocabulary sizes, hyperparameters, decoding settings |
| `src/config.py` | Reads `config.yaml` and turns its paths into absolute paths |
| `src/dataloader.py` | `DataLoader` — download and unpack the `.rar`, parse SGML, split by document |
| `src/preprocessing.py` | `Preprocessor` — tokenise, length filter, the statistics behind the config |
| `src/vocabulary.py` | `Vocabulary` — word ↔ id, one per language |
| `src/transformer.py` | `Transformer` — positional embedding, encoder/decoder, the three attention uses |
| `src/training.py` | `Trainer` — warmup schedule, label-smoothed loss, distributed fit |
| `src/translator.py` | `Translator` — greedy and beam-search decoding |
| `src/evaluation.py` | `Evaluator` — BLEU, dictionary baseline, alignment agreement |
| `main.py` | 4 stages: `prepare` → `train` → `translate` → `evaluate` |
| `notebooks/transformer_en_vi_evbcorpus.ipynb` | The whole project as one file, with the run's outputs |

### Run

```bash
pip install -r requirements.txt
python main.py                     # prepare -> train -> translate -> evaluate
python main.py --stage prepare     # download, unpack, split by document, build vocabularies
python main.py --stage train       # fit, save weights and history
python main.py --stage translate   # a few sentences, greedy and beam side by side
python main.py --stage evaluate    # BLEU, dictionary baseline, alignment, plots
python main.py --stage train --benchmark   # also measure one GPU against all of them
```

The corpus is a `.rar` and needs an external tool; `DataLoader.extract` tries five and installs
`unar` if none is there. `dataset.max_train_docs` caps training documents for a local CPU run.

---

## 2. Results

Kaggle **T4 ×2**, MirroredStrategy, float32, 8.5 minutes over 13 epochs, one seed.
38,523 training pairs / 2,791 validation / 2,793 test, split by whole document.

| | BLEU | 1-gram | 4-gram | length ratio |
|---|---|---|---|---|
| transformer, greedy (2,793 sentences) | **16.57** | 49.58 | **6.22** | 0.969 |
| word-for-word dictionary | 14.78 | **56.08** | 4.13 | 0.989 |
| transformer, beam 4 (first 300) | **15.94** | 50.27 | 5.57 | 0.973 |
| transformer, greedy (same 300) | 13.93 | 48.12 | 4.59 | 0.971 |

**Beam search is worth +2.01 BLEU.** The last row exists because without it the comparison would
have been beam's 15.94 against greedy's 16.57 on a *different* set of sentences, which says beam is
worse. The 300-sentence subset is just harder.

**The dictionary baseline is 1.8 BLEU behind a Transformer**, which is much closer than expected,
and the two precision columns say why: word-for-word wins on single words (56.08 vs 49.58) and
loses on 4-grams (4.13 vs 6.22). It knows the vocabulary and nothing about order; the model
produces fluent Vietnamese that drifts from the source. Reading the outputs, they are not close at
all — BLEU nearly cannot tell them apart.

**Cross-attention against the human alignment**, 5,694 aligned tokens over 256 test sentences:

| decoder layer | agrees with the human | chance |
|---|---|---|
| 1 | 25.3% | 5.1% |
| **2** | **40.9%** | 5.1% |
| 3 | 28.3% | 5.1% |
| 4 | 27.6% | 5.1% |

Eight times chance, with no alignment supervision anywhere — the model only ever saw sentence
pairs. And alignment lives in the **middle** of the stack, not the end. The attention map for
*"Looking at your newborn - What 's normal"* reads cleanly: `trẻ sơ sinh` → **newborn**,
`bình thường` → **normal** (0.44, the brightest cell), `quan sát` → **looking** — including the
reordering, since Vietnamese puts `bình thường` where English puts *normal* at the end.

**The model overfits from epoch 8**: train token accuracy climbs 0.11 → 0.77 while validation
stalls at 0.4755. 38k sentence pairs against 6M parameters, so this is the expected shape.

---

## 3. Experiments

| Tried | Result | Kept? |
|---|---|---|
| Encoder–decoder Transformer, 4 layers, d_model 128 | BLEU 16.57 EN→VI | **Yes** |
| Beam 4 vs greedy | +2.01 BLEU on the same 300 sentences | Yes — reported next to greedy on the full set |
| Word-for-word dictionary from the corpus's gold alignments | 14.78 BLEU, only 1.8 behind the model | Yes — as the floor, and as a comment on BLEU |
| Split by sentence | Sentences in one article share names, dates, phrasing | **No** — whole documents held out |
| Word-level vocabulary instead of subword | Vietnamese: 8k covers 99.35%; English needs 16k for 97.50% | Yes — no sentencepiece needed |
| Keras `mask_zero` propagation | Silently dropped before reaching any attention layer | **No** — explicit mask arguments |
| Label smoothing 0.1 | Helps, but makes the loss value unreadable | Kept, with early stopping moved off the loss |
| Early stopping on `val_loss` | Restored a checkpoint 3 epochs before the best one | **No** — monitor `val_masked_accuracy` |
| `MirroredStrategy` on T4 ×2 | Measured **1.56×**, not 2× | Yes |
| Eager decode loop | 175 s for 2,793 sentences; traced, 31 s | **No** — `tf.function` with a pinned signature |
| `categorical_crossentropy(label_smoothing=...)` | Needs a one-hot target: 128 × 63 × 8000 = 258 MB | No — wrote the two-term formula out |
| `mixed_float16` | Runs are minutes at fp32 | No |

**The measurement that removed a dependency.** Translation projects reach for subword
tokenisation to avoid `<unk>`. Counting first said it was not needed here:

| | distinct types | vocabulary chosen | coverage |
|---|---|---|---|
| English | 31,364 | 16,000 | 97.50% |
| Vietnamese | 14,071 | 8,000 | **99.35%** |

Vietnamese has **less than half** as many distinct types as English while being 37% *longer* in
tokens, because it writes syllables and its syllable inventory is close to a closed set. The
language that looks harder is the easier one to cover.

---

## 4. What I learned

**A mask you cannot see is a mask you have to test.** Keras propagates an embedding mask only
through layers that declare `supports_masking`, and drops it *silently* at the first one that does
not — which is every custom layer in a hand-built Transformer. The model attended over padding,
trained fine, and produced plausible numbers. Nothing warns you. The fix was to stop using implicit
masking at all and pass masks as arguments.

**Designing the test was harder than fixing the bug.** My first mask test filled the padded slots
with garbage and reported a leak — but the mask is derived from the ids, so garbage in a padded
slot *becomes a real token* and the output is supposed to change. The test was wrong, not the
model. What masking actually promises is that padding is inert, so the test is to add *more*
padding and require the logits not to move. That version gave 0.00e+00.

**Cross-attention really does learn alignment.** 40.9% against 5.1% chance, from sentence pairs
alone. This is the one result here that does not depend on the corpus being small — the model
either points where a human pointed or it does not.

**A word-for-word dictionary gets within 1.8 BLEU of a Transformer at this scale.** That says
something about 38k training pairs, and something about BLEU. A metric that nearly cannot separate
"fluent Vietnamese that drifts" from "correct words in English order" is not measuring what a
reader is measuring. It is why the unigram and 4-gram columns are reported next to the score.

**Beam search has to be compared on the same sentences.** Beam ran on 300 sentences and greedy on
2,793, and the raw numbers said beam was worse by 0.6 BLEU. Re-scoring greedy on the same 300
turned that into beam winning by 2.01. The subset was harder, not the decoder.

**Label smoothing makes the loss stop being a quality signal.** The smoothing term is
`ε·mean(-log p_v)`, which *grows* as the model becomes confident, so the loss has a high floor —
this run ended at 4.44 with 77% training accuracy — and validation loss rises while the model is
still improving. Early stopping on `val_loss` picked a checkpoint three epochs before the best one.

**Two GPUs are worth 1.56×, and only for training.** Close to the 1.6–1.9× the LSTM project
measured, slightly lower because this model is smaller so the all-reduce is a larger share of each
step. Decoding uses one device no matter what: `MirroredStrategy` distributes training steps, and a
generation loop is sequential.

**Eager dispatch, not arithmetic, dominated decoding.** 175 s → 31 s from a `tf.function` with a
pinned `input_signature`. For a 6M-parameter model the Python overhead of ~1,400 decoder calls is
bigger than the maths in them, and beam search made it ~19,000 calls.

**Encoder–decoder earns its shape on this task.** The source is fully known before translating, so
the encoder reads it in both directions; a decoder-only model would force source token *i* not to
see token *i+1*. It also pays `S² + T² + S·T` in attention instead of `(S+T)²`. Decoder-only won
for general-purpose LLMs, not for translation — NLLB, MADLAD-400 and mBART are all still
encoder–decoder.

**Where this stops.** 38k sentence pairs is tiny for translation; published systems use millions.
One seed, one configuration, no tuning, and a test set of 50 news documents — a narrow domain that
these numbers would not survive leaving.
