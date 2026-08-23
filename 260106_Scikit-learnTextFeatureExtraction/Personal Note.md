# Learning Log: Scikit Learn Text Feature Extraction

Notes to myself while turning the fully processed Vietnamese news text (from the
`260106_TextPreprocessingwithNLP` project) into TF IDF features with scikit-learn.

This project does no text processing of its own: no cleaning, no tokenizing, no stopword removal.
All of that lives in the preprocessing project, which is what "text processing" actually means; this
one only consumes its finished output and applies vectorization techniques on top. I originally had
this notebook redo its own tokenization and stopword removal (first with `underthesea`, then with
NLTK once `underthesea` proved too slow), which was a scope mistake: those are text processing
steps, they belong in the other project, not duplicated here.

## Word Segmentation Is Not The Same Thing As N Grams

While planning this project I confused two concepts that only coincidentally share the word
"gram". Writing out the difference clearly here so I do not make the same mistake again.

### Syllable Count In A Compound Word Is Not An N Gram

When a Vietnamese word segmenter such as underthesea splits "sản phẩm" into "sản_phẩm", it does not
create a bigram in the unigram/bigram sense used by Bag of Words or TF IDF. That is still one word,
one single token, it just happens to be built from two syllables joined together, the same way
"butterfly" in English is one word even though it reads as two syllables. Whether a Vietnamese word
has one, two, three, or four syllables is simply a natural property of that word, not a setting I
choose:

| Word | Syllable count | Counted as |
|---|---|---|
| cao | 1 | 1 token |
| sản_phẩm | 2 | 1 token |
| công_ty | 2 | 1 token |
| vô_tuyến_truyền_hình | 4 | 1 token |

A segmenter decides each word boundary automatically, based on the model it was trained on, not on
any "how many grams" setting I would supply. Inside the same sentence, a one syllable word can sit
right next to a three or four syllable word, and that is completely normal, nothing needs to be made
uniform at this step.

### Real N Grams Happen Afterward, On Words That Are Already Segmented

N grams (unigram, bigram, trigram) in Bag of Words or TF IDF operate on tokens that already went
through segmentation, not on syllables. Take the sentence "công ty sản xuất sản phẩm chất lượng
cao" (a company producing high quality products). After segmentation it becomes 5 tokens, each one
already a complete word: `công_ty`, `sản_xuất`, `sản_phẩm`, `chất_lượng`, `cao`. Only now does the
n gram choice for the vectorizer come into play:

* Unigram (n = 1): `công_ty`, `sản_xuất`, `sản_phẩm`, `chất_lượng`, `cao`, five separate features.
* Bigram (n = 2): `công_ty_sản_xuất`, `sản_xuất_sản_phẩm`, `sản_phẩm_chất_lượng`,
  `chất_lượng_cao`, four features, each one joining two adjacent already segmented words.

That is the distinction I was missing: a bigram at this stage merges two whole words that already
went through segmentation, and has nothing to do with how many syllables either of those words is
built from.

## An Option Worth Testing: Mixing N Values

scikit-learn's `ngram_range` lets a single vectorizer mix several values of n at once, so this was
never really an either/or choice:

```python
TfidfVectorizer(ngram_range=(1, 2))   # vocabulary mixes unigrams and bigrams together
TfidfVectorizer(ngram_range=(1, 3))   # unigrams, bigrams, and trigrams, all in one vocabulary
```

With `ngram_range=(1, 2)` on the example above, the vocabulary ends up with all 5 unigrams plus all
4 bigrams combined, 9 columns total, not a choice between the two. This turns out to be the most
common setting in practice: nobody restricts a vectorizer to pure bigrams only, since that would
throw away every meaningful standalone word such as `cao`.

### But Bigram Actually Has Two Separate Jobs, And Segmentation Only Replaces One Of Them

This is the part I initially got wrong. A bigram at the vectorization stage serves two genuinely
different purposes, and word segmentation only makes one of them unnecessary.

**Job 1: patching a word that segmentation broke.** Before I had underthesea, splitting only on
whitespace (the way `nltk.word_tokenize` still does in the preprocessing notebook) breaks
"kinh tế" (economy) into two meaningless pieces, `kinh` and `tế`. In that situation, forming the
bigram `kinh_tế` was a way to patch the damage after the fact, reconstructing the whole word by
brute force. This job genuinely disappears once a real segmenter is in the pipeline, since
underthesea already returns `kinh_tế` as one complete unigram. My original instinct was right about
this part.

**Job 2: capturing the relationship between two words that are already complete on their own. This
job does not disappear.** A bigram is also used to capture how two independent, already meaningful
words relate when they sit next to each other, information that segmentation never touches, since
segmentation only decides syllable boundaries within a single word. After segmentation:

| Phrase | Unigrams (separate) | Bigram (keeps the relationship) |
|---|---|---|
| "không thích" (do not like) | không, thích | không_thích |
| "rất tệ" (very bad) | rất, tệ | rất_tệ |
| "chất_lượng cao" (high quality) | chất_lượng, cao | chất_lượng_cao |

Here `không` and `thích` are each already a complete, separate word. A segmenter will not, and
should not, merge them into one word, since grammatically they are two distinct words, not a
compound. But a model that only sees unigrams would notice `thích` on its own and might read it as
positive, losing the negation signal carried by the preceding `không`. The bigram is what preserves
that pair.

### So What Should I Actually Use After Segmentation?

It depends on the downstream task, not on a fixed rule:

* Topic classification, the kind of task this Vietnamese news dataset is built for (predicting a
  topic such as economy, sports, or entertainment): the main signal comes from the presence of
  characteristic keywords (`bóng_đá` for football, `chứng_khoán` for stocks, `tuyển_sinh` for
  admissions), word order and negation matter much less. Unigrams alone are usually already a
  strong baseline here, since segmentation already hands the model complete, meaningful words.
* Sentiment analysis (positive versus negative): negation and degree, such as `không tốt` (not
  good) or `rất tệ` (very bad), directly changes the label, so bigrams are still very much worth
  adding, even on top of good segmentation.

### The Practical Plan

Do not lock into just one of the two ahead of time. Treat `ngram_range` as a hyperparameter to try,
the same way I would try different values of C for an SVM:

```python
TfidfVectorizer(ngram_range=(1, 1))   # baseline: unigrams only
TfidfVectorizer(ngram_range=(1, 2))   # add bigrams, check whether accuracy actually improves
```

For a topic classification task like this dataset, my guess is that unigrams will already be a
strong baseline and bigrams will only add a small improvement, smaller than before segmentation
existed at all, but the only reliable way to know is to run both and compare accuracy on a held out
validation set rather than guessing in advance.

## Underthesea Is Slow, And Also The Wrong Project For It

I benchmarked `underthesea.word_tokenize` on this machine: about 44 ms per article single
threaded, which works out to roughly 2 hours for the full corpus of 184,539 articles. I tried
speeding this up with `concurrent.futures.ProcessPoolExecutor`, but it failed outright when the
worker function lives in a Jupyter cell (macOS's default "spawn" start method needs the worker
function to be importable from an actual module on disk, an interactive cell does not count).

My first fix was to swap `underthesea` for `nltk.word_tokenize` here and keep tokenizing inside this
notebook, since NLTK handles the full corpus in a few minutes instead of hours. That fixed the speed
problem but missed a bigger one: tokenization is a text processing decision, and text processing
already has its own project. Redoing it here, with either tokenizer, duplicated logic that belongs
in `260106_TextPreprocessingwithNLP` and made two projects responsible for the same choice.

The actual fix was architectural, not a tokenizer swap: this notebook now takes whatever tokenized,
stopword filtered text the preprocessing project hands it (currently `nltk.word_tokenize`, splitting
on syllables, not real Vietnamese words) and does not touch tokenization at all. I know that is not
the linguistically ideal choice: a compound word like "kinh tế" arrives here as two unrelated tokens,
`kinh` and `tế`, instead of one, which reopens Job 1 from the section above (bigrams patching a
segmentation that was never actually done). But that tradeoff is now made once, in the preprocessing
project, and inherited here, rather than being a second decision this notebook also has to make. If
it is ever worth paying for `underthesea`'s accuracy, that change happens in the other project's
tokenization step, and this notebook does not need to change at all.
