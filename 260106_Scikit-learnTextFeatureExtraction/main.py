"""Vectorize the preprocessed news corpus and report on the result.

    python main.py                # the whole corpus (184,539 documents -- minutes)
    python main.py --limit 2000   # the first N documents, for a quick look

Fits every n-gram range in `config/config.yaml`, prints the size and sparsity of each,
and breaks one document down term by term so the TF-IDF numbers can be read against the
raw counts they came from.
"""

from __future__ import annotations

import argparse

from src.config import load_config
from src.dataloader import DataLoader
from src.feature_extraction import FeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        help="vectorize only the first N documents (the full corpus takes minutes)",
    )
    args = parser.parse_args()

    config = load_config()

    corpus = DataLoader(config).load_corpus()
    if args.limit:
        corpus = corpus[: args.limit]
    print(f"Loaded {len(corpus):,} documents")

    extractor = FeatureExtractor(config)
    fitted = extractor.fit_all(corpus)

    for name, features in fitted.items():
        print(
            f"\n{name}: documents {features.n_documents:,}  "
            f"vocabulary {features.vocabulary_size:,}  "
            f"sparsity {features.sparsity:.4%}"
        )
        print(extractor.breakdown_table(features).to_string(index=False))

    print("\nSummary")
    print(extractor.summary(fitted).to_string())


if __name__ == "__main__":
    main()
