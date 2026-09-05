"""ROUGE, and the baseline that has to be beaten.

ROUGE is implemented here rather than imported. `rouge-score`'s default tokenizer lowercases and
strips everything non-alphanumeric, which destroys Vietnamese diacritics and the `_` that marks a
segmented compound -- `khởi_tố` would become two English-looking fragments. VNDS is already
tokenized, so splitting on whitespace is both correct and the whole job.

All three variants are reported as F1, which matters more than it sounds: F1 punishes a long
summary that happens to contain the right words. That single fact is why Lead-1 beats Lead-3 on
this corpus.
"""

from __future__ import annotations

import collections

import numpy as np

from src.config import Config
from src.preprocessing import Preprocessor


class Evaluator:
    """ROUGE-1/2/L over a set of summaries, and the Lead-n extractive baselines."""

    def __init__(self, config: Config, preprocessor: Preprocessor) -> None:
        self.config = config
        self.preprocessor = preprocessor

    # -- the metric ---------------------------------------------------------------------

    @staticmethod
    def _f1(overlap: int, n_pred: int, n_ref: int) -> float:
        if overlap == 0:
            return 0.0
        precision, recall = overlap / n_pred, overlap / n_ref
        return 2 * precision * recall / (precision + recall)

    @classmethod
    def rouge_n(cls, pred: list[str], ref: list[str], n: int) -> float:
        def ngrams(tokens):
            return collections.Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

        p, r = ngrams(pred), ngrams(ref)
        return cls._f1(sum((p & r).values()), max(sum(p.values()), 1), max(sum(r.values()), 1))

    @classmethod
    def rouge_l(cls, pred: list[str], ref: list[str]) -> float:
        """Longest common subsequence, one row of the table at a time -- O(len(ref)) memory."""
        row = [0] * (len(ref) + 1)
        for x in pred:
            previous = 0
            for j, y in enumerate(ref, 1):
                previous, row[j] = row[j], previous + 1 if x == y else max(row[j], row[j - 1])
        return cls._f1(row[-1], max(len(pred), 1), max(len(ref), 1))

    def rouge(self, predictions: list[list[str]], references: list[list[str]]) -> dict:
        """Mean ROUGE-1/2/L as percentages."""
        scores = np.array([[self.rouge_n(p, r, 1), self.rouge_n(p, r, 2), self.rouge_l(p, r)]
                           for p, r in zip(predictions, references)])
        return dict(zip(("rouge1", "rouge2", "rougeL"), (scores.mean(0) * 100).tolist()))

    # -- the baseline -------------------------------------------------------------------

    def lead(self, frame, n: int) -> list[list[str]]:
        """Copy the first n sentences of the article. On news this is a hard baseline."""
        return [" ".join(self.preprocessor.sentences(a)[:n]).split() for a in frame.article]

    def lead_baselines(self, frame) -> dict[str, dict]:
        references = [s.split() for s in frame.abstract]
        results = {}
        for n in self.config.require("evaluation.lead_baselines"):
            predictions = self.lead(frame, n)
            results[f"Lead-{n}"] = {
                **self.rouge(predictions, references),
                "mean_length": float(np.mean([len(p) for p in predictions])),
            }
        return results
