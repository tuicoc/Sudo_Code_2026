"""Train Word2Vec on viwik18, end to end.

    python main.py                  # corpus -> train -> evaluate (skips finished stages)
    python main.py --stage corpus   # download the shards and build data/processed/sentences.txt
    python main.py --stage train    # train the model on that file and save it
    python main.py --stage evaluate # nearest neighbours + the t-SNE plot, from the saved model

The corpus stage is the expensive one: ten ~94 MB shards to download and word-segment.
Both it and the training stage are skipped when their output already exists, so a re-run
picks up where the last one stopped.
"""

from __future__ import annotations

import argparse

from src.config import load_config
from src.dataloader import DataLoader
from src.preprocessing import Preprocessor
from src.trainer import Word2VecTrainer
from src.visualization import EmbeddingVisualizer

STAGES = ("corpus", "train", "evaluate")


def build_corpus(config) -> None:
    loader = DataLoader(config)
    preprocessor = Preprocessor(config, stopwords=loader.load_stopwords())
    sentences_path = config.path("paths.sentences_file")
    if sentences_path.exists():
        print(f"Corpus already built: {sentences_path} (delete it to rebuild)")
        return
    print("Building the sentences file (downloads each shard as it goes) ...")
    preprocessor.build_sentences_file(loader.shard_names, loader.read_shard)
    print(f"Wrote {sentences_path}")


def train(config) -> None:
    trainer = Word2VecTrainer(config)
    model_path = config.path("paths.model_file")
    if model_path.exists():
        print(f"Model already trained: {model_path} (delete it to retrain)")
        return
    print("Training Word2Vec ...")
    trainer.train()
    print(" ", trainer.describe())
    print(f"Saved {trainer.save()}")


def evaluate(config) -> None:
    trainer = Word2VecTrainer(config)
    model = trainer.load()
    print("Nearest neighbours:")
    for word in config.require("evaluation.seed_words"):
        neighbors = trainer.most_similar(word)
        print(f"  {word} -> {neighbors if neighbors else 'not in vocabulary'}")
    print(f"Saved {EmbeddingVisualizer(config).plot(model)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, help="run one stage instead of all of them")
    args = parser.parse_args()

    config = load_config()
    stages = [args.stage] if args.stage else list(STAGES)
    for stage in stages:
        print(f"\n=== {stage} ===")
        {"corpus": build_corpus, "train": train, "evaluate": evaluate}[stage](config)


if __name__ == "__main__":
    main()
