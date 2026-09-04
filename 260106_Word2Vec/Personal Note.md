# Word2Vec on Vietnamese Wikipedia

| | |
|---|---|
| **Goal** | Train Word2Vec with gensim and check the vectors actually learned meaning |
| **Dataset** | viwik18 — cleaned Vietnamese Wikipedia, ~940 MB, 10 shards ([GitHub](https://github.com/NTT123/viwik18)) |
| **Result** | **6,564,854 sentences → 414,646-word vocabulary**, ~3.1 hours end to end |

---

## 1. How to run

### What is in here

| File | What it does |
|---|---|
| `config/config.yaml` | Shard URLs, hyperparameters, seed words, t-SNE settings |
| `src/config.py` | Reads `config.yaml` and turns its paths into absolute paths |
| `src/dataloader.py` | `DataLoader` — downloads the 10 shards (resumable) and the stopword list |
| `src/preprocessing.py` | `Preprocessor` — split into sentences, segment into words, drop stopwords |
| `src/trainer.py` | `Word2VecTrainer` — train, save, load, nearest neighbours |
| `src/visualization.py` | `EmbeddingVisualizer` — t-SNE scatter plot |
| `main.py` | 3 stages: `corpus` → `train` → `evaluate` |
| `notebooks/word2vec_training.ipynb` | The experiment, narrated |

### Run

```bash
pip install -r requirements.txt
python main.py                   # all 3 stages; finished stages are skipped
python main.py --stage corpus    # download + segment → data/processed/sentences.txt
python main.py --stage train     # train Word2Vec → data/outputs/
python main.py --stage evaluate  # nearest neighbours + t-SNE plot
```

The `corpus` stage is the expensive one: ~940 MB to download plus `underthesea` over all of it,
about 2.5 hours. It skips shards already on disk, so an interrupted run just continues.

---

## 2. Results

| | |
|---|---|
| Sentences | 6,564,854 |
| Vocabulary (`min_count=5`) | 414,646 |
| Total time | ~3.1 hours |
| Settings | `vector_size=100`, `window=5`, `sg=1` (skip-gram) |

Nearest neighbours for the 5 seed words:

| Seed | Neighbours | How to read it |
|---|---|---|
| `hà_nội` | `hcm`, `tp`, `hải_phòng`, `tphcm`, `nam_định` | Clean — other Vietnamese cities |
| `internet` | `irc`, `intranet`, `lulzsec`, `kwangmyong` | Clean — networking terms |
| `chính_phủ` | `chính_quyền`, `thủ_tướng`, `yingluck_shinawatra` | Clean — government terms |
| `khoa_học` | `amblystegiaceae`, `entodontaceae` (moss families) | Honest but odd — see below |
| `việt_nam` | `nguyễn_nhật`, `nguyễn_kỳ_nam` (people's names) | Honest but odd — see below |

The last two are not failures. Wikipedia's science coverage is heavily weighted toward exhaustive
species-taxonomy pages, so moss families genuinely *are* what "khoa_học" sits next to in this corpus;
"việt_nam" appears constantly in historical and political articles, so it lands near people's names.
**An embedding reflects what the corpus talks about, not what a word "should" mean.** Worth reporting
instead of only showing the three clean clusters.

---

## 3. Experiments

| Tried | Result | Kept? |
|---|---|---|
| Split sentences on `.` | viwik18 has **no punctuation at all** | Split on `\s{2,}` instead |
| `underthesea.word_tokenize()` default output | Returns `"tổ chức"` with a space — silently re-splits in the file | Use `format="text"` → `tổ_chức` |
| Keep all sentences in memory | ~2.3M sentences as Python strings would be several GB | Stream to a file, read with `LineSentence` |
| Remove stopwords before segmenting | Stopword list is word-level, tokens would be syllable-level | Segment first, then filter |
| Run the full corpus straight away | 2.5 h — too long to discover a bug at the end | Test on a 200 KB HTTP-Range sample first |

**Why syllable-level tokenizing is not acceptable here** (unlike the earlier projects): Word2Vec
learns exactly **one vector per token**. If `kinh` and `tế` arrive as two separate tokens, the model
learns a vector for `kinh` (pulled toward wherever that standalone syllable appears) and another for
`tế`. Neither means "economy" — and `kinh_tế`, the word I actually care about, never exists as a
token for the model to learn at all. That is not weaker signal, it is a **missing target**. So this
project pays for `underthesea` where the others did not.

---

## 4. What I learned

**Test on a small real sample before an expensive run.** The 200 KB sample caught the
`format="text"` bug. Without it, every compound would have silently split back into syllables — the
exact thing this project exists to prevent — and I would only have found out by staring at bad
vectors 3 hours later.

**Look at the data instead of assuming its shape.** I expected to split sentences on periods. There
are none. Reading a real sample showed that runs of 2+ spaces are where paragraph breaks used to be.

**Order matters between segmentation and stopword removal.** The stopword list is written at word
level (`và`, `của`). Filtering before segmenting compares against the wrong tokenization; segmenting
then filtering by syllable would break compounds back apart. Segment first, filter second.

**Stream, do not accumulate.** Writing one sentence per line to a file and letting gensim's
`LineSentence` read it back keeps memory flat regardless of corpus size.

**Report the clusters that look bad too.** `khoa_học` → moss families was the most informative result
in the project, because it explains what an embedding actually is.

---

## 5. Restructure (2026-09-01)

Mentor feedback: a notebook is not a project. Added `src/`, `config/`, `README.md` and pinned
`requirements.txt`; `data/` now holds only the empty folder skeleton plus download instructions.

One thing the extraction improved: in the notebook, `process_corpus` called `download_file` directly,
so downloading and segmenting were welded together and neither could be tested without the other. In
`src/`, `build_sentences_file` takes a `read_shard` callable, so `Preprocessor` never knows where its
text comes from — which is what let me test it against a 20 KB slice already on disk instead of
waiting on a 94 MB download.

`main.py` is staged and each stage refuses to redo finished work. In the notebook, re-running the
cell meant re-running the whole 2.5 hours.
