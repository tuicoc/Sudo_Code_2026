"""BLEU, a dictionary baseline, and the check that makes this corpus special.

BLEU is implemented here rather than imported for the same reason ROUGE was in the previous
project: EVBNews is already tokenised, and a scorer that re-tokenises it would be counting
something else. `sacrebleu` is used as a cross-check when it is installed, with `tokenize="none"`
so both count the same tokens.

The alignment check is the part that could not be done on another corpus. EVBNews ships word
alignments a person wrote down, so the usual claim that "cross-attention learns alignment" can be
measured instead of admired.
"""

from __future__ import annotations

import collections
import math

import numpy as np

from src.config import Config
from src.preprocessing import Preprocessor


class Evaluator:
    """Corpus BLEU, the word-for-word baseline, and cross-attention vs human alignment."""

    def __init__(self, config: Config, preprocessor: Preprocessor) -> None:
        self.config = config
        self.preprocessor = preprocessor
        self.max_ngram: int = config.require("evaluation.max_ngram")

    # -- BLEU ---------------------------------------------------------------------------

    @staticmethod
    def ngram_counts(sequence: list[str], n: int) -> collections.Counter:
        return collections.Counter(tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1))

    def bleu(self, hypotheses: list[list[str]], references: list[list[str]]) -> dict:
        """Corpus BLEU: clipped n-gram precision, geometric mean, brevity penalty.

        Clipping stops "the the the the" from buying precision; the brevity penalty stops a model
        from scoring well by saying almost nothing. The mean is geometric, so a zero in any one
        precision takes the whole score to zero.

        Read `length_ratio` next to the score: a ratio far from 1.00 says the model is
        systematically short or long, which the single number hides.
        """
        matched, total = [0] * self.max_ngram, [0] * self.max_ngram
        hyp_len = ref_len = 0
        for hypothesis, reference in zip(hypotheses, references):
            hyp_len += len(hypothesis)
            ref_len += len(reference)
            for n in range(1, self.max_ngram + 1):
                h = self.ngram_counts(hypothesis, n)
                r = self.ngram_counts(reference, n)
                matched[n - 1] += sum((h & r).values())
                total[n - 1] += sum(h.values())

        precisions = [m / t if t else 0.0 for m, t in zip(matched, total)]
        geometric = (0.0 if min(precisions) == 0 else
                     math.exp(sum(math.log(p) for p in precisions) / self.max_ngram))
        ratio = hyp_len / max(ref_len, 1)
        penalty = 1.0 if ratio > 1 else math.exp(1 - 1 / max(ratio, 1e-9))
        return {"bleu": 100 * penalty * geometric,
                "precisions": [round(100 * p, 2) for p in precisions],
                "brevity_penalty": round(penalty, 4),
                "length_ratio": round(ratio, 4)}

    def cross_check(self, hypotheses: list[list[str]], references: list[list[str]]) -> float | None:
        """The same corpus BLEU from sacrebleu, when it is installed. None otherwise."""
        try:
            import sacrebleu
        except ImportError:
            return None
        return sacrebleu.corpus_bleu([" ".join(h) for h in hypotheses],
                                     [[" ".join(r) for r in references]],
                                     tokenize="none").score

    # -- the baseline -------------------------------------------------------------------

    @staticmethod
    def parse_alignment(alignment: str, n_source: int, n_target: int) -> dict[int, set[int]]:
        """`1-1;4-5,6;` -> {target index: {source indices}}, 0-based and bounds-checked.

        The corpus writes links as `english-vietnamese[,vietnamese...]`, 1-based. This inverts
        them, because what gets asked later is "which English word did this Vietnamese word come
        from".
        """
        mapping: dict[int, set[int]] = collections.defaultdict(set)
        for link in alignment.strip(";").split(";"):
            if "-" not in link:
                continue
            left, right = link.split("-", 1)
            try:
                source = int(left) - 1
                targets = [int(i) - 1 for i in right.split(",") if i]
            except ValueError:
                continue
            if 0 <= source < n_source:
                for target in targets:
                    if 0 <= target < n_target:
                        mapping[target].add(source)
        return mapping

    def build_dictionary(self, rows) -> dict[str, str]:
        """source word -> the target phrase a human most often aligned it to.

        Roughly what statistical MT looked like before phrase tables, and a real floor: it gets
        the vocabulary right and the grammar entirely wrong, the opposite failure to the model's.
        """
        pairs: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for english, vietnamese, alignment in rows:
            if not alignment:
                continue
            source_tokens = self.preprocessor.tokens(english)
            target_tokens = self.preprocessor.tokens(vietnamese)
            for target, sources in self.parse_alignment(alignment, len(source_tokens),
                                                        len(target_tokens)).items():
                for source in sources:
                    pairs[source_tokens[source]][target_tokens[target]] += 1
        return {word: counter.most_common(1)[0][0] for word, counter in pairs.items()}

    def translate_by_dictionary(self, rows, dictionary: dict[str, str]) -> list[list[str]]:
        return [[dictionary[w] for w in self.preprocessor.tokens(english) if w in dictionary]
                for english, _, _ in rows]

    # -- cross-attention vs the human alignment -----------------------------------------

    def alignment_agreement(self, layer_scores, rows) -> dict[int, dict]:
        """How often the most-attended source word is one a human linked to that target word.

        `layer_scores` is one (batch, heads, target_len, source_len) array per decoder layer, from
        a teacher-forced pass over `rows`. Heads are averaged. Reported per layer because they do
        not behave alike -- alignment tends to be sharpest in the middle of the stack.

        The random column is the share a uniform guess would get, which is not 1/len(source):
        a target word often has several valid sources, and all of them count as correct.
        """
        results = {}
        for depth, scores in enumerate(layer_scores, start=1):
            averaged = np.array(scores).mean(axis=1)            # (batch, target_len, source_len)
            hits = total = 0
            chance = 0.0
            for b, (english, vietnamese, alignment) in enumerate(rows):
                source_tokens = self.preprocessor.tokens(english)
                target_tokens = self.preprocessor.tokens(vietnamese)
                gold = self.parse_alignment(alignment, len(source_tokens), len(target_tokens))
                for target, sources in gold.items():
                    # +1: decoder position 0 is <bos>, so target token j sits at position j+1
                    if target + 1 >= averaged.shape[1] or not sources:
                        continue
                    predicted = int(averaged[b, target + 1, :len(source_tokens)].argmax())
                    hits += predicted in sources
                    chance += len(sources) / len(source_tokens)
                    total += 1
            results[depth] = {"agreement": hits / max(total, 1),
                              "random": chance / max(total, 1),
                              "tokens": total}
        return results
