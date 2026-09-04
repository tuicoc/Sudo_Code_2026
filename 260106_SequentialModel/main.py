"""A word-level LSTM that writes Vietnamese, trained on 10,415 books.

    python main.py                   # prepare -> train -> generate
    python main.py --stage prepare   # books -> vocabulary -> uint16 token streams
    python main.py --stage train     # fit the LSTM, save model + metrics
    python main.py --stage generate  # sample text: greedy, temperature sweep, top-k

`training.subset_tokens` in the config caps how much of the corpus training uses -- 4 M
tokens keeps a local CPU run to minutes. Set it to `null` for the whole ~334 M-token corpus.
"""

from __future__ import annotations

import argparse

from src.config import load_config
from src.corpus import CorpusBuilder
from src.dataloader import DataLoader
from src.generator import TextGenerator
from src.model import LanguageModel
from src.preprocessing import Preprocessor
from src.vocabulary import Vocabulary

STAGES = ("prepare", "train", "generate")


def prepare(config) -> None:
    loader = DataLoader(config)
    books = loader.find_books()
    print(f"  {len(books):,} books, {sum(f.stat().st_size for f in books) / 1e9:.2f} GB")

    builder = CorpusBuilder(config)

    print("  pass 1: counting tokens ...")
    counts = builder.count_tokens(books)

    print("  vocabulary coverage by size:")
    for row in Vocabulary.coverage_table(counts, config):
        print(f"    {row['vocab']:>7,}  covers {row['coverage']:6.2%}  <unk> {row['unk_rate']:5.2%}")

    vocabulary = Vocabulary.build(counts, config)
    print(f"  vocabulary {len(vocabulary):,}; most common: {vocabulary.itos[3:15]}")

    print("  pass 2: encoding books ...")
    train_ids, val_ids = builder.split(builder.encode_books(books, vocabulary))
    print(f"  {builder.describe(train_ids, val_ids)}")

    loader.save_tokens(train_ids, val_ids)
    loader.save_json("paths.vocabulary_file", vocabulary.to_payload({
        "seq_len": config.require("model.seq_len"),
        "total_tokens": int(sum(counts.values())),
        "true_vocab": len(counts),
    }))
    print(f"  saved -> {loader.processed_dir}")


def train(config) -> None:
    loader = DataLoader(config)
    vocabulary = Vocabulary.from_payload(loader.load_json("paths.vocabulary_file"))
    train_tokens, val_tokens = loader.load_tokens()

    subset = config.get("training.subset_tokens")
    if subset:
        train_tokens, val_tokens = train_tokens[:subset], val_tokens[: subset // 20]
        print(f"  using a {subset:,}-token subset (set training.subset_tokens: null for all)")
    print(f"  vocab {len(vocabulary):,} | train {train_tokens.size:,} | val {val_tokens.size:,}")

    language_model = LanguageModel(config, vocab_size=len(vocabulary))
    language_model.build(seq_len=config.require("model.seq_len")).summary()
    language_model.train(train_tokens, val_tokens)

    metrics = language_model.evaluate(val_tokens)
    print(f"\n  val loss       {metrics['val_loss']:.4f} nats")
    print(f"  val perplexity {metrics['val_perplexity']:,.1f}")
    print(f"  uniform guess  {metrics['uniform_perplexity']:,}   (what 'learned nothing' looks like)")
    print(f"  saved -> {loader.save_json('paths.metrics_file', metrics)}")


def generate(config) -> None:
    loader = DataLoader(config)
    vocabulary = Vocabulary.from_payload(loader.load_json("paths.vocabulary_file"))

    language_model = LanguageModel(config, vocab_size=len(vocabulary))
    generator = TextGenerator(config, language_model.load(), vocabulary)

    prompt = config.require("generation.seed_text")
    top_k = config.require("generation.top_k")

    print(f"  prompt: {prompt!r}\n")
    print(f"  GREEDY\n    {generator.generate(prompt, greedy=True)}\n")
    for temperature in config.require("generation.sweep.temperatures"):
        print(f"  TEMPERATURE {temperature}\n    {generator.generate(prompt, temperature=temperature)}\n")
    print(f"  TOP-K {top_k}\n    {generator.generate(prompt, top_k=top_k)}\n")

    for other_prompt in config.require("generation.prompts"):
        print(f"  [{other_prompt}]\n    {generator.generate(other_prompt, top_k=top_k)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, help="run one stage instead of all of them")
    args = parser.parse_args()

    config = load_config()
    for stage in [args.stage] if args.stage else list(STAGES):
        print(f"\n=== {stage} ===")
        {"prepare": prepare, "train": train, "generate": generate}[stage](config)


if __name__ == "__main__":
    main()
