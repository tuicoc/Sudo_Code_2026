# Learning Log: NLP Text Preprocessing

Notes to myself while working through the "Processing Raw Text" chapter and applying it to a real
Vietnamese news dataset.

## Accessing Text From The Web And From Files

* I learned about a new concept: RSS feeds (Really Simple Syndication). A Python library can pull
  blog post content directly from a feed instead of me having to scrape and clean raw HTML by hand.
* I realized that search engines such as Google are not a reliable corpus tool for language research.
  Results depend on geography, are inconsistent across queries, and get noisy because of duplicate
  content across sites.
* I learned that when working with data such as electronic books or HTML pages, there is often no
  reliable rule for where the real content starts and ends. I usually have to inspect the raw text
  manually and find a pattern (a header phrase, a footer phrase) to trim it myself.

## Python Strings And Unicode

* I learned that the `rU` file mode does not exist anymore in Python 3. Universal newline handling
  (treating `\r\n`, `\r`, and `\n` all as a line break) is simply the default now.
* I learned that `rb` mode is for reading binary files, such as images, video, or audio, or any time
  I want to control the decoding myself instead of letting Python guess the text encoding.
* I learned that `f.read()` returns the entire file as a single string, not a list of lines.
* The `string.join()` method still feels backwards to me: it is called on the separator, and the
  list of pieces is passed in as the argument, not the other way around.
* I learned what Unicode actually is: a giant table mapping characters to numbers (code points). This
  finally explained why byte streams keep showing up everywhere, not only when I work with PDFs or
  images.
* I learned that NFC (the form commonly produced on Windows) and NFD (the form commonly produced on
  macOS) are two different Unicode normalization forms. The same visible character, such as `é`, can
  be stored either as one precomposed code point or as two code points (a plain `e` followed by a
  combining acute accent). Two files can look identical on screen and still fail a string comparison
  if they were normalized differently.
* I learned why encodings like UTF8, UTF16, and UTF32 exist: text always has to be turned into bytes
  before it can be stored or sent anywhere. UTF8 won out as the common default because it is variable
  length: common characters, including plain ASCII, take a single byte, while rarer characters take
  more, so it stays backward compatible with ASCII and does not waste space by always spending 4
  bytes per character.
* Disemvoweling: a small regex exercise from the book where vowels are stripped from a word but it
  stays mostly readable, for example turning "declaration" into "dclrtn".

## Where Word Level Text Processing Is Actually Used

Text processing and word level tokenization are usually building blocks for tasks such as:

* Text classification: topic classification, sentiment analysis, spam detection.
* Information retrieval: search, measuring how similar two documents are.
* Clustering and topic modeling: grouping articles by topic automatically.
* Feature input for traditional ML models: Naive Bayes, SVM, Logistic Regression, KNN.

I also realized that generative AI (GPT style LLMs) is solving a different problem entirely:
predicting and generating the next token autoregressively to produce new text. Technically it does
not need a separate bag of words or TF IDF step at all. The subword tokenizer and the embeddings are
learned together, inside the Transformer network itself, instead of being a separate preprocessing
stage.

* That kind of tokenization is subword tokenization, for example BPE, WordPiece, or Unigram.
* That kind of embedding is contextual embedding, for example ELMo, BERT, or GPT.

### Traditional ML Versus Generative AI: Word2Vec Versus Transformer Embeddings

This got much clearer once I laid the two approaches side by side. Word2Vec is the traditional ML
style of turning text into vectors: a separate embedding step whose output feeds into a downstream
model. The Transformer style used by generative AI learns its embeddings end to end, as part of the
very same model that goes on to generate text.

| Criterion | Word2Vec (traditional, static embedding) | Transformer (generative, contextual embedding) |
|---|---|---|
| Input unit | Whole word | Subword/token (BPE, WordPiece, and similar) |
| How the vector is built | Fixed lookup table (1 word → 1 vector) | Lookup table first → then many layers of self-attention |
| Depends on context? | No | Yes |
| Same word always gets the same vector? | Yes, always the same regardless of the sentence | No, the same word in different sentences gets different vectors |
| Example: the word "bank" | "river bank" and "money bank" → same vector | "river bank" and "money bank" → two different vectors |
| Unseen word (OOV) | No vector at all (skipped or assigned randomly) | Still handled, since it splits by subword instead of depending on the whole word |
| Training objective | Skip-gram / CBOW (predicting surrounding words) | Masked LM (BERT) or next-token prediction (GPT) |
| What the vector is used for | A feature fed into a separate downstream model (classification, clustering...) | The model's own internal representation, used to understand or generate text |
| Example models | Word2Vec, GloVe, Doc2Vec | BERT, GPT, RoBERTa, T5... |

## Preprocessing The Vietnamese News Dataset

Applying all of the above to a real corpus (184,539 Vietnamese news articles) surfaced a few things
the textbook examples never really force you to confront.

### The Pipeline, Step By Step

I ended up structuring the actual notebook to mirror the book's own pipeline, going from raw text
all the way to a finished vocabulary. Writing out what each step does, in order, made it much easier
to reason about where something like a wrong teencode mapping could quietly sneak in.

* Step 0, detecting noise patterns: before writing any cleaner, I scanned the raw corpus with regular
  expressions to see what kinds of noise it actually contains: email addresses, URLs, Vietnamese
  phone numbers, and leftover CDATA blocks (explained just below).
* Step 1, Unicode normalization: canonicalize every article to NFC, so the same visible character is
  never silently treated as two different strings later on.
* Step 2, removing unwanted material: apply the patterns found in step 0, replacing every match with
  a single space rather than deleting it outright, so that words on either side never get glued
  together into one token.
* Step 3, normalizing text: lowercase everything, then fold a small set of teencode abbreviations.
  This is the step where I actually had to decide which abbreviations were safe to fold. Candidates
  like `k`, `ko`, `vs`, `dc`, and `đc` looked reasonable enough on paper, but I did not fully trust my
  own guess that a formal news dataset would barely contain any teencode at all, so instead of just
  assuming it, I asked an AI assistant (Claude) to check the real data for me. What that check found
  is explained in full in the two sections right after this one.
* Step 4, tokenization: split the normalized text into word tokens using NLTK's `word_tokenize`.
* Step 5, stopword removal and vocabulary: drop tokens that carry little topical information, then
  collect everything that survives into the corpus vocabulary, the final output of the pipeline.

### What CDATA Actually Is

I learned that `<![CDATA[ ... ]]>` is an XML and HTML construct, short for Character Data. It tells a
parser to treat everything inside as literal text and not attempt to interpret it as markup. It is
the standard way to embed JavaScript, or any text containing `<`, `>`, or `&`, inside an XML document
without escaping every special character, which is why it shows up so often in RSS or Atom feeds and
inside `<script>` tags.

I did not just take this on faith, I pulled a real example out of my own dataset:

```
//<![CDATA[ window.addEventListener("load", function(){if(document.getElementById("left") &&
document.getElementById("left").offsetWidth < 760){$("#upload_chum_anh_...").justifiedGallery({...
```

It turned out to be leftover JavaScript from an image gallery widget that the original crawler never
fully stripped out of the article body. It affects 258 of 184,539 articles, roughly 0.14 percent, so
rare but real, and worth handling in the cleaning step.

### Fixing Leftover Domain And Reference Code Fragments

Running the pipeline on a real article (source index 79) caught a real bug in Step 2, not just a
theoretical one. The article credited itself as "Kienthuc.net.vn" and "Docbao.vn", written without
an `http://` prefix, so my URL pattern (which only looks for `http` or `www.`) missed both of them
completely. A few lines later, once punctuation got stripped, each domain fell apart into
meaningless leftover pieces: `net`, `vn`, `docbao`. The same thing happened to a press license
reference written as `GP-STTTT`, which split into `gp` and `stttt` once the hyphen was treated as
generic punctuation.

Before treating this as worth fixing, I checked how common it actually is, rather than assuming one
example generalizes: across a 20,000 article sample, roughly 1 in 7 articles contains a bare domain
mention like this, and roughly 1 in 12 contains an administrative reference code. Both were common
enough to deserve a dedicated pattern instead of being left as noise tokens.

I added two more patterns to Step 2, both running before the generic punctuation stripping step so
the whole identifier gets removed as one unit instead of being shredded:

* A bare domain pattern: one or more `label.` groups followed by a known top level domain such as
  `vn`, `com`, `net`, or `org`, catching self references like `docbao.vn` even without a URL scheme.
* A reference code pattern: a short run of uppercase letters, a hyphen, then more uppercase letters
  or digits, catching administrative codes like `GP-STTTT`.

**This second pattern is a real precision tradeoff, not a clean fix.** Checking its full corpus
matches (32,477 total), most are genuinely boilerplate codes, but it also strips things that are
actual content: weapon or aircraft model numbers in world news (`A-10`, `A-135`), and organization
or brand acronyms written with a hyphen (`ABS-CBN`, `AFL-CIO`). I decided this was still worth
keeping, since for a task like topic classification, losing one isolated token like `A-10` barely
matters (the surrounding words still carry the topic), while the boilerplate codes it also removes
are pure noise repeated across many articles. I did not try to make the pattern smarter, since
telling an administrative code apart from a weapon designation with a regex alone is not really
possible without a lookup list, and that felt like solving a much bigger problem than this cleaning
step needed.

The lesson that stuck with me: a regex cleaner that only handles the cases I thought of in advance
will always leak on real data. Running it against an actual article and reading the output line by
line caught two systematic bugs that a purely theoretical review of the code never would have, and
checking the fix against the full corpus (not just one sample) surfaced a tradeoff I would have
missed otherwise.

### The Teencode Padding Trick

I learned that Python's `str.replace()` has no concept of a word boundary, it just matches a literal
substring wherever it appears. That is why my teencode dictionary keys look like `" k "`, with a
space on each side: without those spaces, the key `"k"` would also match inside completely unrelated
words such as `"kim"` or `"khách"`.

That still leaves one gap: a slang word sitting at the very start or the very end of the string has
no neighboring space on that side, so it would never match the padded key. That is why `fold_case()`
temporarily wraps the whole string in an extra leading and trailing space before running the
replacements, then strips it back off afterward. Because the cleaning step earlier in the pipeline
already turns every punctuation mark into a space, by the time this function runs, every real word
boundary in the text is already a plain space, so this trick is enough.

### Verifying Teencode Cases With An AI Assistant

I was not convinced abbreviations like `k`, `ko`, `vs`, `dc`, and `đc` actually show up as real
teencode in a formal news dataset, since news writing tends to avoid chat slang. Rather than guess, I
asked Claude to scan the raw corpus for word boundary matches of each candidate and pull out a real
example sentence for each one. The result was not what I expected going in:

* `k`, 2,430 articles: almost all of them turned out to be initials, for example the fragment
  "Ảnh: H.K" where K is a person's initial, not slang for "không".
* `ko`, 174 articles: mostly the proper noun "Ko Samae San", a Thai island, not slang either.
* `vs`, 2,340 articles: overwhelmingly the sports word "versus", as in "Liverpool vs Strasbourg", not
  the Vietnamese word "với".
* `dc`, 287 articles: mostly "Washington, DC".
* `đc` (with the Vietnamese `đ`), 28 articles: the only one that was genuinely teencode, for example
  "chưa đủ cảm nhận đc cái cô quạnh".

So I trimmed my teencode dictionary down to just `đc`. This was a good lesson in not trusting a slang
dictionary just because it looks reasonable on paper: I had to check it against the actual data.
Mapping `vs` to `với` would have been the worst offender, since it silently flips the meaning of
"against" into "with" across thousands of sports headlines.

### Why I Am Not Using Stemming Or Lemmatization For Vietnamese

I learned that NLTK's built in stemmers (Porter, Lancaster) and its `WordNetLemmatizer` are English
specific tools. Porter and Lancaster strip English inflectional suffixes such as `ing`, `ed`, or `s`.
`WordNetLemmatizer` goes a step further and checks whether the stripped form actually exists inside
the English WordNet dictionary.

Vietnamese is an analytic language: words essentially do not change form for tense, number, or
person the way English words do, so there is no inflectional suffix to strip in the first place.
There is also no NLTK WordNet resource for Vietnamese. Running an English stemmer over Vietnamese
text would not normalize anything meaningful, it would just chop letters off words that were never
inflected to begin with. If I ever need Vietnamese specific normalization beyond simple case folding,
the right tool is a dedicated Vietnamese word segmentation or lemmatization library, not NLTK's
English oriented ones.

Word segmentation and n grams for the actual Bag of Words / TF IDF feature extraction work are
covered in their own note, in `260106_Scikit-learnTextFeatureExtraction/Personal Note.md`.
