"""Run the whole preprocessing pipeline end to end.

    python main.py

Downloads the news corpus, runs the five cleaning stages, reports the vocabulary size,
and writes the result where `260106_Scikit-learnTextFeatureExtraction` reads it from.
"""

from __future__ import annotations

import argparse

from src.config import load_config
from src.dataloader import DataLoader
from src.preprocessing import Preprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-noise",
        action="store_true",
        help="report how much of each noise pattern the corpus contains, then continue",
    )
    args = parser.parse_args()

    config = load_config()
    loader = DataLoader(config)

    print("Loading articles ...")
    df = loader.load_articles()
    print(f"  {len(df):,} articles")

    preprocessor = Preprocessor(config, stopwords=loader.load_stopwords())

    if args.scan_noise:
        corpus = "\n".join(df["title"] + " " + df["content"])
        print(f"Scanning {len(corpus):,} characters for noise ...")
        for kind, count in preprocessor.scan_noise(corpus).items():
            print(f"  {kind:<16} {count:>8,}")

    print("Preprocessing ...")
    df = preprocessor.process_dataframe(df)

    vocabulary = preprocessor.build_vocabulary(df["content_tokens"])
    print(f"  vocabulary: {len(vocabulary):,} unique tokens")

    demo_index = config.get("demo_index")
    if demo_index in df.index:
        print(f"\nArticle {demo_index} after preprocessing:")
        print(" ", df.loc[demo_index, "content_no_stopwords"][:300])

    export_path = loader.save_processed(df)
    print(f"\nSaved {len(df):,} rows -> {export_path}")


if __name__ == "__main__":
    main()
