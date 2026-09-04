# Sequential Model — an LSTM that writes Vietnamese

A word-level LSTM language model trained on
[10,415 Vietnamese books](https://www.kaggle.com/datasets/iambestfeeder/10000-vietnamese-books)
(1.73 GB, 334 M tokens), following the mechanism in
[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).

The first **generative** project in this repo. Everything before it was discriminative —
given a document, pick a label. Here there is no label at all: the training signal is the
text predicting itself, one token shifted.

## Results

The reference run is a **deliberate demo**: 500 books (17.2 M tokens, 5% of the corpus),
15 epochs, 80 minutes on Kaggle T4 ×2.

| | epoch 1 | epoch 15 |
|---|---|---|
| train loss | 6.5401 | 4.9008 |
| val loss | 6.1458 | **5.3953** |

**Validation perplexity 220.4**, against 20,000 for uniform guessing — the model is
choosing between roughly 220 words where an untrained one chooses between 20,000. Real
learning, and nowhere near a model that has understood Vietnamese.

Validation loss fell at *every one* of the 15 epochs and early stopping never fired: the
run ended because the budget ran out, not because it converged. 220 is where a short run
on 5% of the corpus lands, not a ceiling this architecture reached.

Reading the samples honestly (seed: *ông ấy nhìn ra ngoài cửa sổ và*):

| Sampling | What comes out |
|---|---|
| greedy | collapses — `- anh không biết. - anh không biết.` repeating. Not a bug: it is what "always take the most probable token" means once a state's likeliest continuation returns to that state. |
| temperature 0.5 | fluent, dialogue-shaped Vietnamese; punctuation in sentence-shaped places |
| temperature 0.8 | drifts mid-sentence |
| temperature 1.2 | word salad, rare tokens surfacing |
| top-k 40 @ 0.9 | the best of them — and still not coherent past a clause |

Real words ✓, local grammar ✓, sentence-level coherence partially, cross-sentence
coherence no. Which is the honest limit of a single 512-unit LSTM carrying the whole past
in one fixed-size vector.

## Two decisions that shape the whole codebase

**Punctuation and digits are kept**, as their own tokens, and stopwords stay. Every earlier
project in this repo strips all three. A generative model that cannot emit a comma does not
generate text, it generates a word list — and "và", "là", "của" are most of what fluent
Vietnamese is made of.

**The corpus is built in two passes.** The obvious single-pass version keeps each book's
tokens so the files are read once — but 334 M tokens held as Python strings is **19.8 GB**
(a `str` costs ~59 bytes; the object header dominates). The same data as `uint16` is
0.67 GB. Pass one counts tokens to build the vocabulary, pass two re-reads and encodes
straight to `uint16`. Reading every book twice is far cheaper than holding the strings, and
the single-pass version dies around book 5,000 on a 12.7 GB machine — with a crash that
looks like a batch-size problem even though nothing about the model is involved.

## Layout

```
config/config.yaml     regexes, vocabulary size, hyperparameters, sampling settings
src/config.py          loads config.yaml, resolves its paths against the project root
src/dataloader.py      DataLoader    -- find the books, read/write every artifact
src/preprocessing.py   Preprocessor  -- survey noise, clean, tokenize (punctuation kept)
src/vocabulary.py      Vocabulary    -- the 20k vocabulary, encode/decode, coverage table
src/corpus.py          CorpusBuilder -- the two passes, and the split-by-book
src/model.py           LanguageModel -- windowed dataset, embedding -> LSTM -> logits
src/generator.py       TextGenerator -- greedy / temperature / top-k sampling
main.py                the three stages, runnable separately or all at once
notebooks/             the experiment: three staged notebooks, plus one self-contained
                       Kaggle/Colab notebook that runs the whole thing on a GPU
data/                  tokens, vocabulary, model, metrics (never committed)
Personal Note.md       learning log
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Nothing in `data/` is committed; the folders ship empty and `main.py` fills them.

[`iambestfeeder/10000-vietnamese-books`](https://www.kaggle.com/datasets/iambestfeeder/10000-vietnamese-books)
— 10,415 plain `.txt` books, 1.73 GB, downloaded by `kagglehub` on the first run. It needs
Kaggle credentials once: `~/.kaggle/kaggle.json`, from *Settings → API → Create New Token*.

`DataLoader.find_books` searches the download for the folder holding the most `.txt` files
rather than hardcoding a path, because kagglehub's version folder naming has changed
between releases.

## Run

```bash
python main.py                   # prepare -> train -> generate
python main.py --stage prepare   # books -> vocabulary -> uint16 token streams
python main.py --stage train     # fit the LSTM, save model + metrics
python main.py --stage generate  # greedy, a temperature sweep, and top-k
```

`training.subset_tokens: 4000000` in the config keeps a local CPU run to minutes. Set it to
`null` for the full ~334 M-token corpus — which is roughly 11 h/epoch on a CPU and ~2 h on
a T4 pair, so use a GPU notebook for that.

## Use the pieces directly

```python
from src.config import load_config
from src.dataloader import DataLoader
from src.generator import TextGenerator
from src.model import LanguageModel
from src.vocabulary import Vocabulary

config = load_config()
vocabulary = Vocabulary.from_payload(DataLoader(config).load_json("paths.vocabulary_file"))

model = LanguageModel(config, vocab_size=len(vocabulary))
generator = TextGenerator(config, model.load(), vocabulary)

generator.generate("ngày hôm đó trời mưa rất to", n_tokens=40, top_k=40)
```

`TextGenerator` transfers the trained weights into an architecturally identical model built
with an *unconstrained* input length — the trained one is fixed at `seq_len=100`, which
generation cannot use, since a prompt is however long it is and grows by one token per step.

## Notebooks

`01_prepare_corpus`, `02_lstm_language_model` and `03_generate` are the staged experiment.
`lstm_vietnamese_books.ipynb` is the same project as one self-contained notebook that
detects Kaggle/Colab and runs the whole thing on a GPU — that is where the reference
results above came from.
