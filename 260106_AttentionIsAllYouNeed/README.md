# Attention for Vietnamese text summarization — VNDS

An attention mechanism written out by hand, wired into a GRU encoder–decoder, and measured on
Vietnamese news summarization. Background reading:
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) and
[Attention Is All You Need](https://arxiv.org/abs/1706.03762).

The project is a **controlled comparison**, not a summarization system. The same network is
trained three times and differs only in how the decoder reads the article:

| Variant | The decoder sees |
|---|---|
| `none` | one fixed vector — the encoder's final state |
| `additive` | every encoder position, re-weighted at each output step (Bahdanau scoring) |
| `dot` | the same, with scaled dot-product scoring |

## Results

15,000 of the 99,134 training articles, articles truncated to 256 tokens, 1,000 test articles.
One Kaggle T4, float32, one seed. Roughly 27 minutes of training for all three.

| | ROUGE-1 | ROUGE-2 | ROUGE-L | tokens | best val loss | train |
|---|---|---|---|---|---|---|
| **Lead-1 baseline** | **26.28** | 8.64 | **21.18** | 29 | — | — |
| **Lead-2 baseline** | **26.76** | **9.53** | 20.10 | 59 | — | — |
| Lead-3 baseline | 24.69 | 9.17 | 18.07 | 86 | — | — |
| no attention | 15.95 | 1.14 | 13.53 | 24 | 5.2628 | 7.3 min, 11 ep |
| **additive attention** | **17.00** | **1.66** | **14.20** | 27 | **5.2374** | 10.5 min, 14 ep |
| dot-product attention | 16.56 | 1.25 | 13.87 | 24 | 5.2514 | 9.4 min, 14 ep |

**Attention is worth +1.05 ROUGE-1**, and the gain is consistent: it also wins ROUGE-2 (+0.52),
ROUGE-L (+0.67) and validation loss, with dot-product landing between the two on every one of
them. One seed, so the size of the gap is not a precise number — the ordering is what the run
supports.

**All three lose to copying the first sentence of the article**, by about ten ROUGE-1 points.
The ROUGE-2 column is where it hurts: 1.14–1.66 against 8.64–9.53 means the models almost never
get two words in a row right. They write fluent, generic Vietnamese news prose that is factually
invented — for an article about China at the Shangri-La Dialogue, the best model produced *"Bộ
Ngoại_giao Mỹ cho_biết ông sẽ có cuộc tập_trận chung với Mỹ và Trung_Quốc."*

**The three models converge to the same validation loss** (5.24–5.26). Attention does not lower
the floor here; it changes what comes out of the decoder. At 15,000 examples the training data is
the binding constraint, not the architecture — see the note for what that does and does not prove.

## Layout

```
config/config.yaml     dataset ids, truncation lengths, hyperparameters, decoding settings
src/config.py          loads config.yaml, resolves its paths against the project root
src/dataloader.py      DataLoader   -- download the corpus, read/write every artifact
src/preprocessing.py   Preprocessor -- truncate, split sentences, measure the truncation cost
src/vocabulary.py      Vocabulary   -- the shared 20k vocabulary, coverage table, encode/decode
src/attention.py       AdditiveAttention / DotProductAttention  -- the exercise, written out
src/model.py           Seq2Seq      -- GRU encoder-decoder, with attention or without
src/decoding.py        Decoder      -- greedy decoding, n-gram blocking, attention weights
src/evaluation.py      Evaluator    -- ROUGE-1/2/L and the Lead-n baselines
main.py                four stages, runnable separately or all at once
notebooks/             the experiment as one self-contained notebook, with the run's outputs
data/                  corpus, vocabulary, weights, metrics, plots (never committed)
Personal Note.md       learning log
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Nothing in `data/` is committed; the folders ship empty and `main.py` fills them.

[VNDS / vietnews](https://huggingface.co/datasets/nam194/vietnews) — 143,816 Vietnamese news
articles with human-written abstracts, from *VNDS: A Vietnamese Dataset for Summarization*
(Nguyen et al., 2019). The original release is 150k separate `.txt.seg` files on GitHub; this
mirror is the same corpus as three parquet files, **needs no login**, and keeps the original word
segmentation (`khởi_tố`, `ma_tuý`). `DataLoader.fetch` downloads them on the first run.

`dataset.subset` in the config decides how much is used. Set the entries to `null` for the full
corpus.

## Run

```bash
python main.py                    # prepare -> train -> evaluate -> summarize
python main.py --stage prepare    # download, measure, build the vocabulary, encode
python main.py --stage train      # fit every variant in training.variants
python main.py --stage evaluate   # decode the test set, ROUGE against Lead-n, plots
python main.py --stage summarize  # print a few summaries side by side
```

On a CPU this takes hours. The notebook is the practical path: it is self-contained, detects
Kaggle/Colab, and produced the numbers above in about 40 minutes on one T4.

## Use the pieces directly

```python
from src.config import load_config
from src.dataloader import DataLoader
from src.decoding import Decoder
from src.model import Seq2Seq
from src.vocabulary import Vocabulary

config = load_config()
loader = DataLoader(config)
vocabulary = Vocabulary.from_payload(loader.load_json("paths.vocabulary_file"))

model = Seq2Seq.create(config, len(vocabulary), "additive",
                       sample_batch=None, compile_model=False)
model.load_weights(loader.weights_path("additive"))

decoder = Decoder(config, model, vocabulary)
weights, tokens = decoder.attention_for(loader.load_encoded()["test_articles"][0])
```

## Two decisions worth knowing before reading the code

**Attention is applied after the decoder GRU, not fed back into it.** Bahdanau's original
computes the context from `s_{t-1}` and feeds it into the GRU at step `t`, which needs a Python
loop over output steps. Scoring every output step against every input position in one batched
call trains roughly ten times faster; the cost is that the GRU never learns what it attended to
at the previous step. That is a real limitation of these numbers, not a detail.

**One GPU, float32.** The previous project in this repo measured `MirroredStrategy` on T4 ×2 at
1.6–1.9×, not 2×. These models train in minutes, so the second GPU is not the constraint. fp16
is also a specific hazard for attention: masking padded positions adds `-1e9` before the softmax,
and `-1e9` is `-inf` in fp16 (maximum 65504), which turns the softmax into NaN. `src/attention.py`
computes its scores in float32 so `training.mixed_precision: true` is safe — it just buys little.
