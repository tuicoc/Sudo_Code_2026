# Word2Vec on Vietnamese Wikipedia

A skip-gram Word2Vec model trained with gensim on **viwik18**, a cleaned Vietnamese
Wikipedia dump (~940 MB across ten shards, 6.6 M sentences, 414 K word vocabulary).

## The one thing that makes this project different

The earlier projects in this repo tokenize Vietnamese at the *syllable* level, which is a
speed tradeoff they can afford. Word2Vec cannot. It learns exactly one vector per distinct
token, so if "kinh" and "tế" arrive as two separate tokens, the model learns a vector for
each — and **"kinh_tế", the word actually being asked about, never exists as a token at
all**. That is not weaker signal, it is a missing target. So this project pays for real
word segmentation with `underthesea`.

That choice has a mechanical consequence, which is why `format="text"` appears in
`Preprocessor.segment_to_tokens`: underthesea's default output returns "tổ chức" as one
*space-containing* string, which silently re-splits into two tokens the moment it is
written to a space-joined line. `format="text"` joins compounds with an underscore
("tổ_chức"), which survives the round trip through the sentences file.

## The corpus has no punctuation

viwik18 ships already lowercased with all punctuation stripped, so there is no period to
split sentences on. Runs of **two or more spaces** are where a title or paragraph break
used to be, and that is the only sentence boundary the corpus offers — the
`preprocessing.segment_boundary` regex in the config.

## Layout

```
config/config.yaml      shard URLs, hyperparameters, seed words, t-SNE and figure settings
src/config.py           loads config.yaml, resolves its paths against the project root
src/dataloader.py       DataLoader          -- download shards (resumable) and the stopword list
src/preprocessing.py    Preprocessor        -- segments -> whole words -> sentences.txt
src/trainer.py          Word2VecTrainer     -- train, save, load, nearest neighbours
src/visualization.py    EmbeddingVisualizer -- t-SNE scatter of seed words and neighbours
main.py                 the three stages, runnable separately or all at once
notebooks/              the experiment, narrated
data/                   downloads, the sentences file, the trained model (never committed)
Personal Note.md        learning log
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Nothing in `data/` is committed; the folders ship empty and `main.py` fills them.

| What | Where from | Lands in | Size |
|---|---|---|---|
| viwik18 shards `viwik18_aa` … `viwik18_aj` | [NTT123/viwik18](https://github.com/NTT123/viwik18) over HTTPS, no credentials needed | `data/raw/` | ~940 MB |
| Vietnamese stopword list | [`heeraldedhia/stop-words-in-28-languages`](https://www.kaggle.com/datasets/heeraldedhia/stop-words-in-28-languages) via `kagglehub` | kagglehub cache | tiny |

Shard downloads are skipped when the file is already on disk, so an interrupted download
is resumed by running the same command again. The Kaggle list needs credentials once
(`~/.kaggle/kaggle.json`, from *Settings → API → Create New Token*).

To sample the corpus without downloading a full shard, `DataLoader.fetch_sample` pulls the
first 200 KB with an HTTP Range request.

## Run

```bash
python main.py                   # all three stages; finished stages are skipped
python main.py --stage corpus    # download shards, word-segment, write data/processed/sentences.txt
python main.py --stage train     # train Word2Vec on that file, save to data/outputs/
python main.py --stage evaluate  # nearest neighbours + the t-SNE plot
```

The corpus stage is the expensive one — downloading and word-segmenting ~940 MB. Both it
and training refuse to redo finished work: delete the output file to force a rebuild.

## Use the pieces directly

```python
from src.config import load_config
from src.trainer import Word2VecTrainer

trainer = Word2VecTrainer(load_config())
trainer.load()
trainer.describe()                      # 'trained on 6,564,854 sentences, vocabulary 414,646'
trainer.most_similar("việt_nam", topn=3)
```

`notebooks/word2vec_training.ipynb` is the experiment this package was extracted from.
