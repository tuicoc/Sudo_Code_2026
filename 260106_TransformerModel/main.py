"""An encoder-decoder Transformer for English -> Vietnamese translation, on EVBCorpus.

    python main.py                     # prepare -> train -> translate -> evaluate
    python main.py --stage prepare     # download, unpack, split by document, build vocabularies
    python main.py --stage train       # fit the Transformer, save weights and history
    python main.py --stage translate   # decode the test set, print examples
    python main.py --stage evaluate    # BLEU against the dictionary baseline, alignment, plots
    python main.py --stage train --benchmark    # also measure 1 GPU against all of them

Both GPUs are used for training when `training.use_multi_gpu` is set and more than one is
present. Decoding runs on one device -- MirroredStrategy distributes training steps, and a
generation loop is sequential by nature.
"""

from __future__ import annotations

import argparse

import numpy as np

from src.config import load_config
from src.dataloader import DataLoader
from src.evaluation import Evaluator
from src.preprocessing import Preprocessor
from src.training import Trainer
from src.transformer import verify_masking
from src.translator import Translator
from src.vocabulary import Vocabulary

STAGES = ("prepare", "train", "translate", "evaluate")


def _rows(loader: DataLoader, preprocessor: Preprocessor) -> dict[str, list]:
    """The same filtered splits every stage works from."""
    return {name: preprocessor.filter_rows(rows)
            for name, rows in loader.load_splits().items()}


def _vocabularies(loader: DataLoader) -> tuple[Vocabulary, Vocabulary]:
    payload = loader.load_json("paths.vocabulary_file")
    return Vocabulary.from_payload(payload["en"]), Vocabulary.from_payload(payload["vi"])


def _load_metrics(loader: DataLoader) -> dict:
    try:
        return loader.load_json("paths.metrics_file")
    except FileNotFoundError:
        return {}


def prepare(config) -> None:
    loader, preprocessor = DataLoader(config), Preprocessor(config)
    print(f"  {loader.extract(loader.download())}")
    print(f"  {len(list(loader.corpus_dir.glob('*.sgml'))):,} documents")

    raw = loader.load_splits()
    rows = {name: preprocessor.filter_rows(r) for name, r in raw.items()}
    for name in ("train", "validation", "test"):
        print(f"  {name:11} {len(raw[name]):>6,} pairs -> {len(rows[name]):>6,} within "
              f"{preprocessor.max_len} tokens ({len(rows[name]) / len(raw[name]):.1%})")

    for name, stat in preprocessor.length_stats(rows["train"]).items():
        print(f"  {name}: mean {stat['mean']:5.1f}  median {stat['median']:4.0f}"
              f"  p95 {stat['p95']:4.0f}  max {stat['max']}")

    vocabularies = {}
    for language, index, key in (("EN", 0, "preprocessing.en_vocab"),
                                 ("VI", 1, "preprocessing.vi_vocab")):
        counter = preprocessor.counts(rows["train"], index)
        chosen = config.require(key)
        print(f"  {language}: {len(counter):,} distinct over {sum(counter.values()):,} tokens")
        for row in preprocessor.coverage_table(counter):
            mark = "   <-" if row["vocab"] == chosen else ""
            print(f"      vocab {row['vocab']:>6,} covers {row['coverage']:6.2%}{mark}")
        vocabularies[language] = Vocabulary.build(counter, chosen, config)

    english, vietnamese = vocabularies["EN"], vocabularies["VI"]
    encoded = {}
    for name, split in rows.items():
        encoded[f"{name}_source"] = english.encode_many(
            [r[0] for r in split], preprocessor, preprocessor.max_len, with_markers=False)
        encoded[f"{name}_target"] = vietnamese.encode_many(
            [r[1] for r in split], preprocessor, preprocessor.max_len, with_markers=True)

    loader.save_encoded(encoded)
    loader.save_json("paths.vocabulary_file",
                     {"en": english.to_payload(), "vi": vietnamese.to_payload()})
    print(f"  EN vocabulary {len(english):,} | VI vocabulary {len(vietnamese):,}")
    print(f"  saved -> {loader.processed_dir}")


def train(config, benchmark: bool = False) -> None:
    loader = DataLoader(config)
    english, vietnamese = _vocabularies(loader)
    data = loader.load_encoded()
    trainer = Trainer(config)
    print(f"  {type(trainer.strategy).__name__} over {trainer.replicas} replica(s); "
          f"global batch {trainer.global_batch}")

    train_ds = trainer.make_dataset(data["train_source"], data["train_target"], shuffle=True)
    val_ds = trainer.make_dataset(data["validation_source"], data["validation_target"],
                                  shuffle=False)
    sample = next(iter(train_ds))[0]

    metrics = _load_metrics(loader)
    if benchmark:
        result = trainer.benchmark(len(english), len(vietnamese),
                                   data["train_source"], data["train_target"], sample)
        if result is None:
            print("  benchmark skipped: only one replica")
        else:
            print(f"  1 replica  {result['one_replica_ms']:6.1f} ms/step")
            print(f"  {result['replicas']} replicas {result['all_replicas_ms']:6.1f} ms/step")
            print(f"  speedup on samples/second: {result['speedup']:.2f}x "
                  f"(perfect would be {result['replicas']}.00x)")
            metrics["gpu_benchmark"] = result

    model = trainer.build(len(english), len(vietnamese), sample)
    print(f"  {model.count_params():,} parameters")

    drift = verify_masking(model, sample[0].numpy(), sample[1].numpy())
    print(f"  padding mask check: largest logit change {drift:.2e} "
          f"({'ok' if drift < 1e-4 else 'LEAKING -- results below are suspect'})")

    history = trainer.fit(model, train_ds, val_ds)
    model.save_weights(config.path("paths.weights_file"))

    best_accuracy = int(np.argmax(history["val_masked_accuracy"])) + 1
    best_loss = int(np.argmin(history["val_loss"])) + 1
    print(f"\n  {trainer.train_seconds / 60:.1f} min over {len(history['loss'])} epochs")
    print(f"  best val accuracy {max(history['val_masked_accuracy']):.4f} at epoch "
          f"{best_accuracy}  (val loss bottomed at epoch {best_loss})")

    metrics.update({"history": history, "train_secs": round(trainer.train_seconds, 1),
                    "epochs_run": len(history["loss"]), "params": int(model.count_params()),
                    "best_epoch_by_accuracy": best_accuracy, "best_epoch_by_loss": best_loss,
                    "mask_drift": drift, "replicas": trainer.replicas})
    loader.save_json("paths.metrics_file", metrics)


def _restore(config, loader):
    """The trained model, its vocabularies and a translator."""
    english, vietnamese = _vocabularies(loader)
    data = loader.load_encoded()
    trainer = Trainer(config)
    sample = (data["test_source"][:2], data["test_target"][:2, :-1])
    model = trainer.build(len(english), len(vietnamese), sample, compile_model=False)
    weights = config.path("paths.weights_file")
    if not weights.exists():
        raise FileNotFoundError(f"{weights} is missing. Run: python main.py --stage train")
    model.load_weights(weights)
    return model, english, vietnamese, data, Translator(config, model, vietnamese)


def translate(config) -> None:
    loader, preprocessor = DataLoader(config), Preprocessor(config)
    model, _, _, data, translator = _restore(config, loader)
    rows = _rows(loader, preprocessor)["test"]

    n = config.require("decoding.n_examples")
    greedy = translator.greedy(data["test_source"][:n])
    for i in range(min(n, len(rows))):
        print("\n  " + "=" * 96)
        print("  EN       ", rows[i][0])
        print("  REFERENCE", rows[i][1])
        print("  GREEDY   ", " ".join(greedy[i]))
        print(f"  BEAM {translator.beam_width}   ",
              " ".join(translator.beam(data["test_source"][i])))


def evaluate(config) -> None:
    loader, preprocessor = DataLoader(config), Preprocessor(config)
    evaluator = Evaluator(config, preprocessor)
    model, _, _, data, translator = _restore(config, loader)
    rows = _rows(loader, preprocessor)
    test = rows["test"]
    references = [preprocessor.tokens(r[1]) for r in test]

    scores = {}
    greedy = translator.greedy(data["test_source"])
    scores["transformer (greedy)"] = evaluator.bleu(greedy, references)

    dictionary = evaluator.build_dictionary(rows["train"])
    print(f"  {len(dictionary):,} english words have a most-frequent Vietnamese alignment")
    scores["word-for-word dictionary"] = evaluator.bleu(
        evaluator.translate_by_dictionary(test, dictionary), references)

    n = min(config.require("decoding.beam_sentences"), len(test))
    beam = [translator.beam(data["test_source"][i]) for i in range(n)]
    scores[f"transformer (beam {translator.beam_width}, first {n})"] = evaluator.bleu(
        beam, references[:n])
    scores[f"transformer (greedy, same {n})"] = evaluator.bleu(greedy[:n], references[:n])

    header = f"\n  {'':38} {'BLEU':>7} {'1-gram':>8} {'4-gram':>8} {'len ratio':>10}"
    print(header + "\n  " + "-" * (len(header) - 3))
    for name, s in scores.items():
        print(f"  {name:38} {s['bleu']:7.2f} {s['precisions'][0]:8.2f}"
              f" {s['precisions'][3]:8.2f} {s['length_ratio']:10.3f}")

    check = evaluator.cross_check(greedy, references)
    if check is not None:
        print(f"\n  cross-check: sacrebleu {check:.2f} vs this project "
              f"{scores['transformer (greedy)']['bleu']:.2f}")

    agreement = _alignment(config, evaluator, model, translator, data, test)
    metrics = _load_metrics(loader)
    metrics.update({"scores": scores, "sacrebleu": check, "alignment": agreement})
    loader.save_json("paths.metrics_file", metrics)
    _plot(config, loader, metrics)


def _alignment(config, evaluator, model, translator, data, test) -> dict:
    """Teacher-force the references, then compare cross-attention with the human alignment."""
    import tensorflow as tf

    indexed = [(i, row) for i, row in enumerate(test) if row[2]]
    indexed = indexed[: config.require("evaluation.alignment_sentences")]
    if not indexed:
        return {}
    source = np.array([data["test_source"][i] for i, _ in indexed], np.int32)
    target = np.array([data["test_target"][i] for i, _ in indexed], np.int32)

    model((tf.constant(source), tf.constant(target[:, :-1])))
    agreement = evaluator.alignment_agreement(model.attention_scores(), [r for _, r in indexed])

    print(f"\n  {'layer':>6} {'agreement with the human alignment':>36} {'random':>9}")
    for depth, row in agreement.items():
        print(f"  {depth:>6} {row['agreement']:>35.1%} {row['random']:>8.1%}")
    print(f"\n  {next(iter(agreement.values()))['tokens']:,} aligned tokens over "
          f"{len(indexed)} sentences")
    return {str(k): v for k, v in agreement.items()}


def _plot(config, loader, metrics) -> None:
    import matplotlib.pyplot as plt

    history = metrics.get("history")
    if not history:
        return
    plt.figure(figsize=(11, 3.6))
    for i, (key, label) in enumerate((("loss", "label-smoothed cross-entropy"),
                                      ("masked_accuracy", "token accuracy"))):
        plt.subplot(1, 2, i + 1)
        for prefix, style, name in (("", "o-", "train"), ("val_", "s-", "validation")):
            values = history.get(prefix + key, [])
            plt.plot(range(1, len(values) + 1), values, style, label=name, markersize=3)
        plt.title(key); plt.xlabel("epoch"); plt.ylabel(label); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout()
    plt.savefig(config.path("paths.learning_curves"), dpi=110)
    plt.close()
    print(f"  plots -> {loader.outputs_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, help="run one stage instead of all of them")
    parser.add_argument("--benchmark", action="store_true",
                        help="during train, measure one GPU against all of them")
    args = parser.parse_args()

    config = load_config()
    for stage in [args.stage] if args.stage else list(STAGES):
        print(f"\n=== {stage} ===")
        if stage == "train":
            train(config, benchmark=args.benchmark)
        else:
            {"prepare": prepare, "translate": translate, "evaluate": evaluate}[stage](config)


if __name__ == "__main__":
    main()
