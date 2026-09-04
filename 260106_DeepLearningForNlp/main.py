"""A neural network for 10-class Vietnamese news topic classification (VNTC).

    python main.py                   # prepare -> features -> train -> evaluate -> compare
    python main.py --stage prepare   # UTF-16 -> NFC -> word-segmented UTF-8 (slow, resumable)
    python main.py --stage features  # TF-IDF matrices + label encoding, cached
    python main.py --stage train     # fit the network, save model/history/metadata
    python main.py --stage evaluate  # test metrics, confusion matrix, learning curves
    python main.py --stage compare   # network vs SVM vs NB, segmented vs unsegmented

The corpus itself is not downloadable by script -- see the README's Data section for how to
unpack VNTC into `data/raw/` first.
"""

from __future__ import annotations

import argparse

from src.comparison import ModelComparison
from src.config import load_config
from src.dataloader import DataLoader
from src.evaluation import Evaluator
from src.feature_extraction import FeatureExtractor
from src.model import TopicClassifier
from src.preprocessing import Preprocessor

STAGES = ("prepare", "features", "train", "evaluate", "compare")


def prepare(config) -> None:
    Preprocessor(config).run()

    loader = DataLoader(config)
    for split in loader.splits:
        raw = loader.count_articles(split, processed=False)
        done = loader.count_articles(split, processed=True)
        status = "OK" if raw == done else "MISMATCH"
        print(f"  {split:<11} raw={raw:>6,}  processed={done:>6,}  {status}")


def features(config) -> None:
    loader = DataLoader(config)
    train_texts, train_labels = loader.load_split("Train_Full")
    test_texts, test_labels = loader.load_split("Test_Full")
    print(f"  train {len(train_texts):,}   test {len(test_texts):,}")

    extractor = FeatureExtractor(config)
    X_train, X_test = extractor.fit_transform(train_texts, test_texts)
    y_train, y_test, class_names = extractor.encode_labels(train_labels, test_labels)

    compounds, total = extractor.compound_feature_share()
    print(f"  classes: {class_names}")
    print(f"  {extractor.describe_matrix(X_train)}")
    print(f"  {compounds:,}/{total:,} features are compound words ({100 * compounds / total:.1f}%)")

    loader.save_features(X_train, X_test, y_train, y_test,
                         extractor.vocabulary_payload(class_names))
    print(f"  saved -> {loader.processed_dir}")


def train(config) -> None:
    loader = DataLoader(config)
    X_train, y_train = loader.load_features("train")

    classifier = TopicClassifier(config)
    classifier.build(n_features=X_train.shape[1]).summary()
    history = classifier.train(X_train, y_train)

    best_epoch = min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i])
    print(f"\n  best val_loss {history['val_loss'][best_epoch]:.4f} at epoch {best_epoch + 1}")
    print(f"  best val_accuracy {max(history['val_accuracy']):.4f}")

    print(f"  saved -> {classifier.save(config.path('paths.model_file'))}")
    loader.save_json("paths.history_file", history)
    loader.save_json("paths.train_meta_file", classifier.metadata(history))


def evaluate(config) -> None:
    loader = DataLoader(config)
    X_test, y_test = loader.load_features("test")
    class_names = loader.load_class_names()

    classifier = TopicClassifier(config)
    classifier.load(config.path("paths.model_file"))
    y_pred = classifier.predict(X_test)

    evaluator = Evaluator(config, class_names)

    print("  floors -- what 'learned nothing' scores:")
    for name, scores in evaluator.baseline_scores(y_test).items():
        print(f"    {name:<22} accuracy={scores['accuracy']:.4f}  macro-F1={scores['macro_f1']:.4f}")

    metrics = evaluator.score(y_test, y_pred)
    print(f"\n  accuracy    {metrics['accuracy']:.4f}")
    print(f"  macro-F1    {metrics['macro_f1']:.4f}")
    print(f"  weighted-F1 {metrics['weighted_f1']:.4f}")
    print(f"\n{evaluator.report(y_test, y_pred)}")

    print("  largest confusions:")
    for share, true_class, predicted_class in evaluator.largest_confusions(y_test, y_pred):
        print(f"    {share:6.1%}  {true_class}  ->  {predicted_class}")

    print("\n  published baselines on this test set:")
    for name, accuracy in config.require("evaluation.baselines").items():
        print(f"    {name:<22} {accuracy:.4f}")

    loader.save_json("paths.test_metrics_file", metrics)
    evaluator.plot_confusion_matrix(y_test, y_pred, config.path("paths.confusion_matrix_figure"))
    evaluator.plot_learning_curves(loader.load_json("paths.history_file"),
                                   config.path("paths.learning_curves_figure"))
    print(f"  saved -> {loader.outputs_dir}")


def compare(config) -> None:
    comparison = ModelComparison(config)
    comparison.run_segmented()
    comparison.run_unsegmented()
    comparison.run_floors()

    print()
    print(comparison.format_table())

    DataLoader(config).save_json("paths.comparison_file", comparison.results)
    print(f"\n  saved -> {comparison.plot()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, help="run one stage instead of all of them")
    args = parser.parse_args()

    config = load_config()
    for stage in [args.stage] if args.stage else list(STAGES):
        print(f"\n=== {stage} ===")
        {
            "prepare": prepare,
            "features": features,
            "train": train,
            "evaluate": evaluate,
            "compare": compare,
        }[stage](config)


if __name__ == "__main__":
    main()
