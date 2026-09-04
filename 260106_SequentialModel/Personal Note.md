# Sequential Model — an LSTM that writes Vietnamese

| | |
|---|---|
| **Goal** | Build a word-level LSTM language model and generate Vietnamese with it |
| **Dataset** | 10,415 Vietnamese books, 1.73 GB, 334M tokens ([Kaggle](https://www.kaggle.com/datasets/iambestfeeder/10000-vietnamese-books)) |
| **Result** | **Validation perplexity 220** vs 20,000 for uniform guessing — a deliberate 80-minute demo, not a full run |

First **generative** project in this repo. Everything before it was discriminative — given a
document, pick a label. Here there is no label at all: the text predicts itself, shifted one token.

---

## 1. How to run

### What is in here

| File | What it does |
|---|---|
| `config/config.yaml` | Regexes, vocabulary size, hyperparameters, sampling settings |
| `src/config.py` | Reads `config.yaml` and turns its paths into absolute paths |
| `src/dataloader.py` | `DataLoader` — find the books, read/write every artifact |
| `src/preprocessing.py` | `Preprocessor` — survey noise, clean, tokenize (punctuation kept) |
| `src/vocabulary.py` | `Vocabulary` — the 20k vocabulary, encode/decode, coverage table |
| `src/corpus.py` | `CorpusBuilder` — the two passes, and the split-by-book |
| `src/model.py` | `LanguageModel` — windowed dataset, Embedding → LSTM → logits |
| `src/generator.py` | `TextGenerator` — greedy / temperature / top-k sampling |
| `main.py` | 3 stages: `prepare` → `train` → `generate` |
| `notebooks/01..03` | The experiment, staged |
| `notebooks/lstm_vietnamese_books.ipynb` | The whole project as one file, for a Kaggle/Colab GPU |

### Run

```bash
pip install -r requirements.txt
python main.py                   # prepare → train → generate
python main.py --stage prepare   # books → vocabulary → uint16 token streams
python main.py --stage train     # fit the LSTM, save model + metrics
python main.py --stage generate  # greedy, a temperature sweep, and top-k
```

Needs a Kaggle token the first time. `training.subset_tokens: 4000000` in the config keeps a local
CPU run to minutes; set it to `null` for the full corpus (~11 h/epoch on CPU, ~2 h on a T4 — use a
GPU notebook for that).

---

## 2. Results

The reference run is a **deliberate demo**: 500 books (17.2M tokens, 5% of the corpus), 15 epochs,
80 minutes on Kaggle T4 ×2.

| | epoch 1 | epoch 15 |
|---|---|---|
| train loss | 6.5401 | 4.9008 |
| val loss | 6.1458 | **5.3953** |

**Validation perplexity 220.4**, against 20,000 for uniform guessing. The model is choosing between
about 220 words where an untrained one chooses between 20,000 — real learning, and nowhere near
understanding Vietnamese.

Validation loss fell at **every one** of the 15 epochs and early stopping never fired: the run ended
because the budget ran out, not because it converged. 220 is where a short run on 5% of the corpus
lands, not a ceiling of this architecture.

**Generated text**, seed *"ông ấy nhìn ra ngoài cửa sổ và"*:

| Sampling | Output |
|---|---|
| greedy | Collapses: `- anh không biết. - anh không biết.` repeating |
| temperature 0.5 | Fluent, dialogue-shaped: *thấy một người đã trả lời: - anh tự thông, em không thể nói...* |
| temperature 0.8 | Drifts mid-sentence |
| temperature 1.2 | Word salad, rare tokens surfacing (`pessoa`, `hì hả`) |
| top-k 40 @ 0.9 | Best of them — and still not coherent past a clause |

Against a four-level ladder: real words ✓, local grammar ✓, sentence coherence partly, cross-sentence
coherence no. That is the honest limit of one 512-unit LSTM carrying the whole past in one
fixed-size vector.

---

## 3. Experiments

| Tried | Result | Kept? |
|---|---|---|
| Strip punctuation (as the earlier projects do) | A generative model that cannot emit a comma emits a word list | **No** — punctuation kept as tokens |
| Remove stopwords | 77.3% of tokens are monosyllabic function words — they *are* the grammar | **No** |
| N-grams | They exist to smuggle order into order-blind models; an LSTM reads in order | No |
| `underthesea` segmentation for "free" compression | Shortens sequences 18%, but grows vocabulary 77% — the two cancel | No — regex tokenizer |
| One pass, keeping tokens as Python strings | 334M tokens as `str` = **19.8 GB**; dies at ~book 5,000 | No — two passes, `uint16` (0.67 GB) |
| Split train/val by token offset | A book split down the middle puts its own style on both sides | No — split by whole book |
| Vocabulary 5k / 10k / 30k / 50k | Coverage table printed for each; softmax cost is linear in size | 20k |
| `recurrent_dropout > 0` | Silently loses the cuDNN fast path | No — must stay `0.0` |

**The measurement that reversed the plan.** The stated reason for segmenting was that Vietnamese
words average 1.7 syllables, so segmentation should roughly halve sequence length. But 1.7 is a
*dictionary* statistic, not a *corpus* one — in running text the most frequent tokens are
monosyllabic function words:

| word length | share of tokens |
|---|---|
| 1 syllable | **77.3%** |
| 2 syllables | 21.4% |
| 3+ syllables | 1.2% |

That averages **1.226**, not 1.7 — an 18% compression, not 50%. And it is not free: the vocabulary
grows 77%, and the softmax cost is linear in vocabulary size. The two cancel, so the whole premise
was gone before any model existed.

*(First attempt at this measurement gave 1.09, which looked wrong and was: `underthesea` splits
punctuation into separate tokens, inflating the denominator.)*

---

## 4. What I learned

**A generative task inverts almost every preprocessing rule.** Punctuation, digits and stopwords are
all noise for classification and all *essential* here. Same language, same tools, opposite answers —
which is the same lesson as the sentiment project, one step further.

**Python objects are the memory problem, not the data.** 334M tokens as `str` is 19.8 GB, because a
`str` costs ~59 bytes of object header and the characters are almost incidental. As `uint16` the
same data is 0.67 GB. Reading every book *twice* is far cheaper than holding the strings once. The
crash from getting this wrong looks like a batch-size problem, which is why it is worth knowing.

**Measure the premise before building on it.** The 1.7-syllable figure was the reason for a whole
planned pipeline stage. Measuring it on the actual corpus killed the stage before it cost anything.

**Test masking on a *trained* model.** My first padding test compared masked vs unmasked loss on an
untrained model and found no difference — meaningless, because an untrained model predicts
near-uniformly and every position has loss ≈ log(vocab). The test only became informative after
training first. (This project ends up using fixed-length windows, so there is no padding at all —
but knowing which construction you are using is the decision that settles it.)

**Greedy decoding collapsing is not a bug.** It is what "always take the most probable token" means
once the model reaches a state whose likeliest continuation returns it to that state.

**Reserve index 0 and never emit it.** Then a padded batch can never be confused with real text.

---

## 5. Restructure (2026-09-01)

Mentor feedback: a notebook is not a project. Added `src/`, `config/`, `README.md` and pinned
`requirements.txt`; `data/` now holds only the empty folder skeleton plus download instructions.

The extraction found two real things:

**The tokenizer existed in three places.** `01_prepare_corpus`, `03_generate` and
`lstm_vietnamese_books` each had their own regexes and `clean_text`, and the generation copy had
quietly diverged: it lowercased and normalized but skipped URL, email, domain and phone stripping.
So prompt text and training text were tokenized by two different functions — the kind of bug that is
impossible to find later. `TextGenerator.encode` now runs the same `Preprocessor` the corpus went
through.

**`evaluate()` broke on a Keras return type**, and only because I actually ran the extracted code.
`02_lstm_language_model.ipynb` does `val_loss = model.evaluate(val_ds, verbose=0)` and formats it
straight into an f-string; under Keras 3.15.1 that call returns a *list*, so the same line raises.
The notebook's numbers came off Kaggle under a different TensorFlow, which is exactly how a
version-dependent line survives unnoticed.

Verified end to end on a small subset (200k tokens, 1 epoch): loss falls, checkpoint and CSV log are
written, `evaluate` returns a perplexity, and both samplers work — greedy collapsing into
`tôi tôi tôi`, the right failure. The reference numbers above are still the Kaggle run; nothing about
it was re-run.

### Where this stops

- **No full-corpus run, and none intended.** Everything above is 5% of the books.
- **Nothing was tuned.** One configuration, one seed, one run.
- **The trained model lives on Kaggle, not in this repo.** `data/` is gitignored, so
  `main.py --stage generate` will ask you to train first.
