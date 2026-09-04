"""Naive Bayes vs SVM on Vietnamese product reviews, end to end.

    python main.py                  # prepare -> train -> compare
    python main.py --stage prepare  # download and clean the reviews
    python main.py --stage train    # cross-validate every experiment in the config
    python main.py --stage compare  # table + chart from the saved metrics
"""

from __future__ import annotations

import argparse

from src.config import load_config
from src.cross_validation import CrossValidator
from src.dataloader import DataLoader
from src.feature_extraction import FeatureExtractor
from src.preprocessing import Preprocessor
from src.reporting import ResultsReporter

STAGES = ("prepare", "train", "compare")


def prepare(config) -> None:
    loader = DataLoader(config)
    df = loader.load_raw()
    print(f"  {len(df):,} reviews")
    print(f"  class balance:\n{df['label'].value_counts(normalize=True).sort_index().to_string()}")

    preprocessor = Preprocessor(config, stopwords=loader.load_stopwords())
    lost = preprocessor.stopwords_that_would_be_lost()
    print(f"  stopword removal is off; it would have deleted: {lost}")

    df = preprocessor.process_dataframe(df)
    path = loader.save_processed(df[["clean_comment", "label"]].reset_index(drop=True))
    print(f"  saved -> {path}")


def train(config) -> None:
    loader = DataLoader(config)
    df = loader.load_processed()
    X = df["clean_comment"].values
    y = df["label"].values

    extractor = FeatureExtractor(config)
    validator = CrossValidator(config)

    for experiment in config.require("experiments"):
        print(f"\n  {experiment['label']}")
        folds = validator.run(
            vectorizer_factory=extractor.factory(experiment["vectorizer"]),
            model_factory=validator.model_factory(experiment["model"]),
            X=X,
            y=y,
        )
        results = validator.summarize(
            folds, experiment, extractor.describe(experiment["vectorizer"])
        )
        for metric, value in results["mean"].items():
            print(f"    {metric:<16} {value:.4f} ± {results['std'][metric]:.4f}")
        print(f"    saved -> {loader.save_metrics(experiment['key'], results)}")


def compare(config) -> None:
    loader = DataLoader(config)
    results = {
        experiment["label"]: loader.load_metrics(experiment["key"])
        for experiment in config.require("experiments")
    }

    reporter = ResultsReporter(config)
    print(reporter.table(results).to_string())
    print(f"\n  saved -> {reporter.bar_chart(results)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, help="run one stage instead of all of them")
    args = parser.parse_args()

    config = load_config()
    for stage in [args.stage] if args.stage else list(STAGES):
        print(f"\n=== {stage} ===")
        {"prepare": prepare, "train": train, "compare": compare}[stage](config)


if __name__ == "__main__":
    main()
