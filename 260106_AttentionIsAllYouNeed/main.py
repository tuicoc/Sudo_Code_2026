"""Attention for Vietnamese text summarization, on VNDS.

    python main.py                    # prepare -> train -> evaluate -> summarize
    python main.py --stage prepare    # download, build the vocabulary, encode
    python main.py --stage train      # fit every variant in training.variants
    python main.py --stage evaluate   # decode the test set, ROUGE against Lead-n, plots
    python main.py --stage summarize  # print a few summaries side by side

The point of the project is the comparison: `training.variants: [none, additive, dot]` trains the
same network three times, differing only in the attention layer, so the difference in ROUGE is
what attention is worth and nothing else.

`dataset.subset` in the config caps how much of the corpus is used -- 15,000 of the 99,134
training articles keeps the whole run to about half an hour on one GPU. Set the entries to `null`
for the full corpus.
"""

from __future__ import annotations

import argparse

import numpy as np

from src.config import load_config
from src.dataloader import DataLoader
from src.decoding import Decoder
from src.evaluation import Evaluator
from src.model import Seq2Seq
from src.preprocessing import Preprocessor
from src.vocabulary import Vocabulary

STAGES = ("prepare", "train", "evaluate", "summarize")


def prepare(config) -> None:
    loader, preprocessor = DataLoader(config), Preprocessor(config)
    frames = {split: loader.load_split(split) for split in ("train", "validation", "test")}
    print(f"  {' | '.join(f'{k} {len(v):,}' for k, v in frames.items())}")

    stats = preprocessor.length_stats(frames["train"])
    for name, s in stats.items():
        print(f"  {name:9} mean {s['mean']:5.0f}  median {s['median']:4.0f}  p95 {s['p95']:4.0f}"
              f"  max {s['max']:5d}  within limit {s['within_limit']:.1%}")

    coverage = preprocessor.lead_coverage(frames["train"], (64, 128, 256, 400, 10 ** 6))
    print("  share of abstract words inside the first N article tokens:")
    for n, share in coverage.items():
        print(f"    first {'all' if n > 10 ** 5 else n:>4}: {share:6.1%}")

    counts = Vocabulary.count(frames["train"], preprocessor)
    print(f"  {len(counts):,} distinct words over {sum(counts.values()):,} tokens")
    for row in Vocabulary.coverage_table(counts, config):
        print(f"    vocab {row['vocab']:>6,}  covers {row['coverage']:6.2%}"
              f"  <unk> {row['unk_rate']:5.2%}")

    vocabulary = Vocabulary.build(counts, config)
    encoded = {}
    for split, frame in frames.items():
        articles, summaries = vocabulary.encode_frame(frame, preprocessor)
        encoded[f"{split}_articles"], encoded[f"{split}_summaries"] = articles, summaries
    print(f"  vocabulary {len(vocabulary):,}; most common: {vocabulary.itos[4:16]}")

    loader.save_encoded(encoded)
    loader.save_json("paths.vocabulary_file", vocabulary.to_payload({
        "true_vocab": len(counts),
        "length_stats": stats,
        "lead_coverage": {str(k): v for k, v in coverage.items()},
        "coverage_table": Vocabulary.coverage_table(counts, config),
    }))
    print(f"  saved -> {loader.processed_dir}")


def train(config) -> None:
    loader = DataLoader(config)
    vocabulary = Vocabulary.from_payload(loader.load_json("paths.vocabulary_file"))
    data = loader.load_encoded()

    train_ds = Seq2Seq.make_dataset(config, data["train_articles"], data["train_summaries"], True)
    val_ds = Seq2Seq.make_dataset(config, data["validation_articles"],
                                  data["validation_summaries"], False)
    sample_batch = next(iter(train_ds))[0]

    metrics = _load_metrics(loader)
    for variant in config.require("training.variants"):
        print(f"\n  --- {variant} ---")
        model = Seq2Seq.create(config, len(vocabulary), variant, sample_batch)
        print(f"  {model.count_params():,} parameters")
        history = model.fit_on(train_ds, val_ds)
        model.save_weights(loader.weights_path(variant))

        metrics.setdefault("runs", {})[variant] = {
            "history": history,
            "params": int(model.count_params()),
            "train_secs": round(model.train_seconds, 1),
            "epochs_run": len(history["loss"]),
            "best_val_loss": min(history["val_loss"]),
        }
        print(f"  {model.train_seconds / 60:.1f} min, best val_loss "
              f"{min(history['val_loss']):.4f} -> {loader.weights_path(variant).name}")
    loader.save_json("paths.metrics_file", metrics)


def evaluate(config) -> None:
    loader, preprocessor = DataLoader(config), Preprocessor(config)
    vocabulary = Vocabulary.from_payload(loader.load_json("paths.vocabulary_file"))
    evaluator = Evaluator(config, preprocessor)
    data = loader.load_encoded()
    test_frame = loader.load_split("test")
    references = [s.split() for s in test_frame.abstract]

    metrics = _load_metrics(loader)
    scores = evaluator.lead_baselines(test_frame)

    for variant, model, decoder in _each_model(config, loader, vocabulary, data):
        predictions = decoder.generate(data["test_articles"])
        scores[variant] = {
            **evaluator.rouge(predictions, references),
            "mean_length": float(np.mean([len(p) for p in predictions])),
        }
        metrics.setdefault("runs", {}).setdefault(variant, {})["predictions"] = [
            " ".join(p) for p in predictions[:20]]

    header = f"\n  {'':24} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'tokens':>8}"
    print(header + "\n  " + "-" * (len(header) - 3))
    for name, s in scores.items():
        print(f"  {name:24} {s['rouge1']:8.2f} {s['rouge2']:8.2f} {s['rougeL']:8.2f}"
              f" {s['mean_length']:8.0f}")

    if "additive" in scores and "none" in scores:
        gap = scores["additive"]["rouge1"] - scores["none"]["rouge1"]
        print(f"\n  attention is worth {gap:+.2f} ROUGE-1 against the identical model without it")

    metrics["scores"] = scores
    loader.save_json("paths.metrics_file", metrics)
    _plot(config, loader, metrics, data, vocabulary)


def summarize(config) -> None:
    loader, preprocessor = DataLoader(config), Preprocessor(config)
    vocabulary = Vocabulary.from_payload(loader.load_json("paths.vocabulary_file"))
    data = loader.load_encoded()
    test_frame = loader.load_split("test")
    n = config.require("decoding.n_examples")

    outputs = {}
    for variant, model, decoder in _each_model(config, loader, vocabulary, data):
        outputs[variant] = decoder.generate(data["test_articles"][:n])

    for i in range(n):
        print("\n  " + "=" * 96)
        print("  ARTICLE  ", " ".join(test_frame.article.iloc[i].split()[:40]), "...")
        print("  REFERENCE", test_frame.abstract.iloc[i])
        print("  LEAD-1   ", " ".join(preprocessor.sentences(test_frame.article.iloc[i])[:1]))
        for variant, summaries in outputs.items():
            print(f"  {variant:9}", " ".join(summaries[i]))


# -- shared helpers ---------------------------------------------------------------------

def _load_metrics(loader: DataLoader) -> dict:
    try:
        return loader.load_json("paths.metrics_file")
    except FileNotFoundError:
        return {}


def _each_model(config, loader, vocabulary, data):
    """Yield (variant, model, decoder) for every trained variant, weights loaded."""
    sample = (data["test_articles"][:2], data["test_summaries"][:2, :-1])
    for variant in config.require("training.variants"):
        path = loader.weights_path(variant)
        if not path.exists():
            print(f"  {variant}: no weights at {path.name}, skipping "
                  f"(run: python main.py --stage train)")
            continue
        model = Seq2Seq.create(config, len(vocabulary), variant, sample, compile_model=False)
        model.load_weights(path)
        yield variant, model, Decoder(config, model, vocabulary)


def _plot(config, loader, metrics, data, vocabulary) -> None:
    import matplotlib.pyplot as plt

    runs = metrics.get("runs", {})
    if runs:
        plt.figure(figsize=(11, 3.6))
        for i, key in enumerate(("loss", "val_loss")):
            plt.subplot(1, 2, i + 1)
            for variant, run in runs.items():
                values = run.get("history", {}).get(key, [])
                plt.plot(range(1, len(values) + 1), values, "o-", label=variant, markersize=3)
            plt.title(key); plt.xlabel("epoch"); plt.ylabel("masked cross-entropy")
            plt.grid(alpha=.3); plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(config.path("paths.learning_curves"), dpi=110)
        plt.close()

    variants = [v for v in config.require("training.variants") if v != "none"]
    for variant in variants:
        if not loader.weights_path(variant).exists():
            continue
        sample = (data["test_articles"][:2], data["test_summaries"][:2, :-1])
        model = Seq2Seq.create(config, len(vocabulary), variant, sample, compile_model=False)
        model.load_weights(loader.weights_path(variant))
        decoder = Decoder(config, model, vocabulary)
        weights, tokens = decoder.attention_for(
            data["test_articles"][config.require("evaluation.attention_example")])
        if not tokens:
            print("  the model produced an empty summary for this article -- nothing to plot")
            break

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                       gridspec_kw={"height_ratios": [3, 1]})
        image = ax1.imshow(weights[:len(tokens)], aspect="auto", cmap="viridis")
        ax1.set_yticks(range(len(tokens))); ax1.set_yticklabels(tokens, fontsize=7)
        ax1.set_xlabel("article position"); ax1.set_title(f"attention weights -- {variant}")
        fig.colorbar(image, ax=ax1)
        ax2.plot(weights[:len(tokens)].mean(0))
        ax2.set_xlabel("article position"); ax2.set_ylabel("mean weight")
        ax2.set_title("averaged over output steps"); ax2.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(config.path("paths.attention_plot"), dpi=110)
        plt.close()
        break

    print(f"  plots -> {loader.outputs_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, help="run one stage instead of all of them")
    args = parser.parse_args()

    config = load_config()
    for stage in [args.stage] if args.stage else list(STAGES):
        print(f"\n=== {stage} ===")
        {"prepare": prepare, "train": train, "evaluate": evaluate,
         "summarize": summarize}[stage](config)


if __name__ == "__main__":
    main()
