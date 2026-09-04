# Text Preprocessing with NLP

| | |
|---|---|
| **Goal** | Clean raw Vietnamese news text into tokens, following NLTK book chapter 3 |
| **Dataset** | 184,539 Vietnamese news articles ([Kaggle](https://www.kaggle.com/datasets/haitranquangofficial/vietnamese-online-news-dataset)) |
| **Result** | 184,539 articles cleaned → **111,477-token vocabulary**, exported for the next project |

---

## 1. How to run

### What is in here

| File | What it does |
|---|---|
| `config/config.yaml` | All settings: dataset ids, every regex, the teencode map, export columns |
| `src/config.py` | Reads `config.yaml` and turns its paths into absolute paths |
| `src/dataloader.py` | `DataLoader` — downloads the corpus and the stopword list, writes the output file |
| `src/preprocessing.py` | `Preprocessor` — the 5 cleaning steps |
| `main.py` | Runs everything: download → clean → export |
| `notebooks/text_preprocessing.ipynb` | The experiment, with before/after shown at every step |

### Run

```bash
pip install -r requirements.txt
python main.py                # download → preprocess → export
python main.py --scan-noise   # also print how much noise the corpus has, first
```

Needs a Kaggle token at `~/.kaggle/kaggle.json` the first time. Takes a few minutes.
Output: `../260106_Scikit-learnTextFeatureExtraction/data/raw/processed_news.parquet`.

### Pipeline

| Step | Method | What it does |
|---|---|---|
| 0 | `scan_noise` | Count what noise the corpus actually has, before writing any rule |
| 1 | `normalize_unicode` | NFC, so `é` written two different ways compares equal |
| 2 | `clean_text` | Remove URLs, emails, phones, domains, licence codes, digits, punctuation |
| 3 | `fold_case` | Lowercase, then expand abbreviations |
| 4 | `tokenize` | NLTK `word_tokenize` |
| 5 | `remove_stopwords` | Drop words that carry no topic signal |

---

## 2. Results

| | |
|---|---|
| Articles in | 184,539 |
| Vocabulary out | 111,477 unique tokens |
| CDATA blocks found | 258 articles (0.14%) |
| Bare domains (`docbao.vn`) | ~1 in 7 articles |
| Licence codes (`GP-STTTT`) | ~1 in 12 articles |

Example (article 79), raw → final:

```
raw:   Theo Kienthuc.net.vn, chiều 12/9 ... GP-STTTT số 1234
final: chiều thông tin ...
```

The output feeds `260106_Scikit-learnTextFeatureExtraction` as a file on disk. No imports between
projects.

---

## 3. Experiments

| Tried | Result | Kept? |
|---|---|---|
| Teencode map with `k`, `ko`, `vs`, `dc`, `đc` | Checked all 5 against the real corpus — only `đc` is real teencode | Only `đc` |
| URL regex alone for source credits | Misses `Kienthuc.net.vn` (no `http://`), which then shatters into `net` + `vn` | Added a bare-domain pattern |
| Licence-code pattern `[A-Z]{1,4}-[A-Z0-9]{2,8}` | Catches 32,477 matches — mostly boilerplate, but also `A-10`, `ABS-CBN` | Kept, accepting the loss |
| Stemming / lemmatization (Porter, WordNet) | English-only tools; Vietnamese words do not inflect | Not used |
| Replacing noise with `""` instead of `" "` | Glues neighbouring words together | Always replace with a space |

**The teencode check in detail** — this was the most useful experiment. I did not trust my own
guess that a news corpus has little slang, so I checked each candidate against the real text:

| Token | Articles | What it actually is |
|---|---|---|
| `k` | 2,430 | Initials — "Ảnh: H.K" |
| `ko` | 174 | "Ko Samae San", a Thai island |
| `vs` | 2,340 | Sports "versus" — "Liverpool vs Strasbourg" |
| `dc` | 287 | "Washington, DC" |
| `đc` | 28 | **Real teencode** — "chưa cảm nhận đc" |

Mapping `vs` → `với` would have flipped "against" into "with" across thousands of sports headlines.

---

## 4. What I learned

**Check the corpus before writing a cleaning rule.** Every rule that turned out to matter came from
looking at real text first (`scan_noise`), and every rule I guessed at was wrong. The teencode table
above is the clearest case.

**One example is not enough — check the whole corpus.** Running the pipeline on article 79 found the
bare-domain bug. But only counting it across 20,000 articles showed it was 1-in-7, worth a dedicated
pattern, and only counting the licence-code pattern's 32,477 matches showed it also eats real content
like `A-10`. Both directions needed the full count.

**Replace noise with a space, never with nothing.** Deleting a URL outright glues the words on either
side into one fake token.

**`str.replace()` has no word boundaries.** That is why the teencode keys are `" đc "` with spaces on
both sides — otherwise `"k"` matches inside `"kim"` and `"khách"`. And because a word at the very
start of a string has no space before it, `fold_case` pads the whole string with spaces first, then
strips them.

**NFC vs NFD.** The same visible character can be one code point or two. Two files can look identical
and still fail `==`. Normalizing first makes everything after it comparable.

**Vietnamese does not need stemming.** It is an analytic language — words do not change form for
tense or number, so there is no suffix to strip. Porter/Lancaster would just chop letters off.

---

## 5. Restructure (2026-09-01)

Mentor feedback: a notebook is not a project. Added `src/`, `config/`, `README.md` and pinned
`requirements.txt`; `data/` now holds only the empty folder skeleton plus download instructions.

Two things the extraction exposed:

- Writing the regexes out as named config entries showed that `url` (the detection pattern) and
  `url_loose` (the one `clean_text` actually uses) were **two different regexes** I had never
  noticed were different — in the notebook they were two `re.compile` calls twenty cells apart.
- `scan_noise` produces no file, so it existed only as notebook cells and would have been lost. It
  is now a method, and `main.py --scan-noise` runs it.

This project now has two note files (this log and the chapter-3 PDF), so both moved into `note/`.
