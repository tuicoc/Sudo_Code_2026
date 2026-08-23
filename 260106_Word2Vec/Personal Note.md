# Learning Log: Word2Vec On Vietnamese Wikipedia

Notes to myself while training a Word2Vec model with gensim on viwik18, a cleaned Vietnamese
Wikipedia dump. This is a different dataset from the other two projects, downloaded and processed
entirely inside this project.

## Why Kinh, Tế, And Kinh_Tế Are Not Interchangeable Here

This is the one thing I want to remember above everything else in this project. In the other two
projects, using NLTK's syllable level tokenizer instead of real Vietnamese word segmentation was a
tradeoff I accepted for speed: a Bag of Words or TF IDF model still gets some signal out of "kinh"
and "tế" sitting near each other as separate tokens, just weaker signal than "kinh_tế" would give.

Word2Vec does not work that way. It learns exactly one vector per distinct token, from that token's
surrounding context. If "kinh" and "tế" show up as two separate tokens, the model learns a vector
for "kinh" (pulled toward whatever contexts the standalone syllable "kinh" happens to appear in
across the corpus) and a separate, unrelated vector for "tế" (same story). Neither of those vectors
represents "economy" the concept. The word "kinh_tế" that I actually care about simply never exists
as a token for the model to learn a vector for at all. This is not noisier signal, it is a missing
target: the exact thing I am asking the model to represent was never handed to it as a unit. That is
why this project pays the real cost of `underthesea`, unlike the other two.

## The Raw Text Has No Punctuation, So I Had To Find A Different Sentence Boundary

I expected to split the corpus into sentences the usual way, on periods. There are none: viwik18 is
already cleaned down to lowercase words separated by single spaces, with no punctuation left at all.
Looking at a real sample instead of guessing, I found that runs of two or more spaces mark where a
title or paragraph break used to be, for example "trang chính  internet society  internet society
hay isoc là một tổ chức..." (the repeated title is how WikiExtractor represents an article heading).
Splitting on `\s{2,}` gave segments averaging 40 to 65 words in my sample, a reasonable stand in for
a sentence, so that is what I used as the unit fed to Word2Vec instead of true sentences.

## Underthesea Has To Run Before Stopword Removal, Not After

This one I already learned the hard way in the feature extraction project, but it matters even more
here. A Vietnamese stopword list is written at the word level, entries like "và" or "của", not at
the syllable level. If I filtered stopwords before segmenting, I would be comparing the segmenter's
eventual compound tokens against a list built for a different tokenization scheme, and if I
segmented and then still filtered against syllable boundaries I would risk breaking a compound back
apart. The only order that is actually consistent is: segment into real words first with
`underthesea`, then check each resulting word against the stopword list.

## Streaming To A File Instead Of Holding The Corpus In Memory

viwik18 is about 875 MB of raw text across 10 files, which turns into roughly 2.3 million segments
once split. Keeping every tokenized sentence as a Python list of strings in memory the whole time
would very plausibly balloon into multiple gigabytes, given how much overhead a few million small
Python string objects carry. Instead, `process_corpus` writes each processed sentence straight to a
text file, one sentence per line with tokens space joined, which is exactly the format gensim's
`LineSentence` class expects. Training then streams sentences off disk through `LineSentence` rather
than needing them all resident in memory, which is the pattern gensim's own documentation recommends
for a corpus that does not fit in RAM.

## The Realistic Time Cost

I benchmarked `underthesea.word_tokenize` on a real sample of these shorter wiki segments (about 39
words each on average): roughly 4 ms per segment. Scaled up to the full corpus of about 2.3 million
segments, that comes out to roughly 2.5 hours just for segmentation, before Word2Vec training even
starts. That is long enough that I ran a small sample first (fetching just the first 200 KB of one
file with an HTTP Range request, not the full 94 MB) to check the pipeline actually produces
sensible tokens, before committing to a run that long on the full dataset.

That sample run caught a real bug, not a theoretical one: `word_tokenize()` without
`format="text"` returns compounds as space containing strings like `"tổ chức"`, not underscore
joined. Writing that straight into a space joined `LineSentence` file would have silently split
every compound back into separate syllables at exactly the step this whole project exists to avoid,
and I would not have noticed until looking at the trained vectors much later. Testing on a real,
if small, slice of the actual data before the expensive run is what caught it.

## What The Full Run Actually Produced

The full corpus took about 3.1 hours end to end (matching the benchmark) and produced 6,564,854
sentences and a vocabulary of 414,646 tokens with `min_count=5`. The visualization backs up that the
model learned real structure, not noise: `internet`, `chính_phủ`, and `hà_nội` each form a tight,
clearly separated cluster of genuinely related terms (`hà_nội` neighbors with `hải_phòng`, `tphcm`,
`hcm`, other Vietnamese cities; `internet` neighbors with `botnet`, `irc`, `intranet`, `usenet`).

Two of the five seed words told a more honest story than a clean success would have. `khoa_học`
(science) clustered with `amblystegiaceae`, `entodontaceae`, `plagiotheciaceae`, all moss family
taxonomy terms, not because the model is wrong but because Wikipedia's science coverage is heavily
weighted toward exhaustive species taxonomy pages, so that is genuinely the term "khoa_học" appears
next to most in this corpus. `việt_nam` clustered mostly with Vietnamese historical and political
figures' names (`nguyễn_nhật`, `nguyễn_kỳ_nam`, `lưu_văn_lợi`) rather than geography, which reflects
the same thing: the word shows up constantly in historical and political article context. Both are
a reminder that a word embedding reflects what the training corpus actually talks about around that
word, not some external notion of what the word "should" mean, and that is worth reporting honestly
rather than only picking the clusters that look clean.
