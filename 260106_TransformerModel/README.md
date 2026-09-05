# A Transformer for English → Vietnamese translation — EVBCorpus

The encoder–decoder Transformer of
[Attention Is All You Need](https://arxiv.org/abs/1706.03762), following the structure of the
[TensorFlow transformer tutorial](https://www.tensorflow.org/text/tutorials/transformer), trained
on [EVBCorpus / EVBNews](https://github.com/qhungngo/EVBCorpus) — 1,000 parallel news documents,
45,308 sentence pairs, sentence-aligned **and hand-aligned at word level**.

Those hand-written word alignments are what makes this corpus worth using here. Cross-attention is
supposed to learn alignment; this corpus ships alignments a person wrote down, so the claim can be
measured instead of admired.

## Results

38,523 training pairs, one Kaggle **T4 ×2** run, 8.5 minutes of training, one seed.

| | BLEU | 1-gram | 4-gram | length ratio |
|---|---|---|---|---|
| **transformer, greedy** (2,793 sentences) | **16.57** | 49.58 | **6.22** | 0.969 |
| word-for-word dictionary baseline | 14.78 | **56.08** | 4.13 | 0.989 |
| transformer, beam 4 (first 300) | **15.94** | 50.27 | 5.57 | 0.973 |
| transformer, greedy (same 300) | 13.93 | 48.12 | 4.59 | 0.971 |

**Beam search is worth +2.01 BLEU** over greedy on the same 300 sentences. The row above it exists
because comparing beam's 15.94 against greedy's full-set 16.57 would have said the opposite — the
300-sentence subset is simply harder.

**The dictionary baseline is only 1.8 BLEU behind the Transformer**, and it wins on unigrams
(56.08 vs 49.58) while losing badly on 4-grams (4.13 vs 6.22). That is the two systems failing in
opposite directions: word-for-word translation knows the vocabulary and nothing about order; the
Transformer produces fluent Vietnamese that drifts from the source. A reader would not rate these
two systems as nearly equal, and BLEU nearly does.

### Cross-attention against the human alignment

For every Vietnamese token in a teacher-forced pass, is the English word the model attended to most
the one a person linked it to? Over 5,694 aligned tokens in 256 test sentences:

| decoder layer | agrees with the human alignment | chance |
|---|---|---|
| 1 | 25.3% | 5.1% |
| **2** | **40.9%** | 5.1% |
| 3 | 28.3% | 5.1% |
| 4 | 27.6% | 5.1% |

**Eight times chance, with no alignment supervision anywhere in training** — the model only ever
saw sentence pairs. The layer profile is the other half of the result: alignment concentrates in
the middle of the stack, not at the end.

### The second GPU

Measured, not quoted:

```
1 replica :  97.6 ms/step at batch 64
2 replicas: 125.5 ms/step at batch 128
speedup on samples/second: 1.56x   (perfect scaling would be 2.00x)
```

The gap is the per-step gradient all-reduce. Decoding does **not** use both GPUs:
`MirroredStrategy` distributes training steps, and a generation loop is sequential by nature.

## Layout

```
config/config.yaml     dataset ids, lengths, vocabulary sizes, hyperparameters, decoding settings
src/config.py          loads config.yaml, resolves its paths against the project root
src/dataloader.py      DataLoader   -- download and unpack EVBNews, parse SGML, split by document
src/preprocessing.py   Preprocessor -- tokenise, length filter, the statistics behind the config
src/vocabulary.py      Vocabulary   -- word <-> id, one per language
src/transformer.py     Transformer  -- positional embedding, encoder/decoder, three attention uses
src/training.py        Trainer      -- warmup schedule, label-smoothed loss, distributed fit
src/translator.py      Translator   -- greedy and beam-search decoding
src/evaluation.py      Evaluator    -- BLEU, dictionary baseline, alignment agreement
main.py                four stages, runnable separately or all at once
notebooks/             the experiment as one self-contained notebook, with the run's outputs
data/                  corpus, vocabularies, weights, metrics, plots (never committed)
Personal Note.md       learning log
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Nothing in `data/` is committed; `main.py --stage prepare` fills it.

EVBNews ships as a **`.rar`**, and there is no pure-Python reader — `rarfile` is a wrapper that
still needs a backend. `DataLoader.extract` tries `tar` (macOS `tar` is bsdtar and reads RAR
through libarchive), `bsdtar`, `unar`, `7z` and `unrar` in that order, and if none is present
runs `apt-get install -y unar` and tries again. On a machine where that is not possible:

```bash
apt-get install -y unar        # or libarchive-tools, or p7zip
```

## Run

```bash
python main.py                     # prepare -> train -> translate -> evaluate
python main.py --stage prepare     # download, unpack, split by document, build vocabularies
python main.py --stage train       # fit the Transformer, save weights and history
python main.py --stage translate   # decode a few test sentences, greedy and beam side by side
python main.py --stage evaluate    # BLEU, the dictionary baseline, alignment, plots
python main.py --stage train --benchmark   # also measure one GPU against all of them
```

`dataset.max_train_docs` caps the training documents so a local CPU run finishes in minutes; it is
`null` (all 900) for a real run.

## Three decisions that shape the code

**Padding masks are explicit arguments, not Keras's `mask_zero` propagation.** That propagation
stops at the first layer that does not declare `supports_masking` — which is every custom layer in
`src/transformer.py` — and Keras drops the mask *silently*. The first version of this project
attended over padding while everything still ran and produced plausible numbers.
`verify_masking()` is the check that caught it, and `main.py --stage train` runs it before fitting.

**Early stopping watches accuracy, not loss.** Label smoothing makes the loss
`(1-ε)·(-log p_true) + ε·mean(-log p_v)`, and the second term *grows* as the model gets confident.
Validation loss therefore rises while the model is still improving. On the first run of this
project monitoring `val_loss` restored a checkpoint three epochs before the best one.

**The decode loop is traced.** A test set is ~1,400 decoder calls and beam search ~19,000; eager
mode pays full Python dispatch on each, which for a 6M-parameter model costs more than the
arithmetic. `tf.function` with a pinned `input_signature` took greedy decoding of 2,793 sentences
from **175 s to 31 s**.

## Use the pieces directly

```python
from src.config import load_config
from src.dataloader import DataLoader
from src.training import Trainer
from src.translator import Translator
from src.vocabulary import Vocabulary

config = load_config()
loader = DataLoader(config)
payload = loader.load_json("paths.vocabulary_file")
english, vietnamese = Vocabulary.from_payload(payload["en"]), Vocabulary.from_payload(payload["vi"])

data = loader.load_encoded()
trainer = Trainer(config)
model = trainer.build(len(english), len(vietnamese),
                      sample=(data["test_source"][:2], data["test_target"][:2, :-1]),
                      compile_model=False)
model.load_weights(config.path("paths.weights_file"))

translator = Translator(config, model, vietnamese)
translator.beam(data["test_source"][0])
```
