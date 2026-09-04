"""Projects the embeddings to 2-D so they can be looked at.

t-SNE preserves neighbourhoods, not distances: points close together were close in the
original 100-D space, but the axes and the gaps between clusters mean nothing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

from src.config import Config


class EmbeddingVisualizer:
    """Plots a handful of seed words and their nearest neighbours, coloured by seed."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def collect_neighborhoods(self, model: Word2Vec) -> tuple[list[str], list[str]]:
        """Each seed word plus its nearest neighbours. Returns (words, which seed each belongs to)."""
        neighbors_per_seed = self.config.require("evaluation.neighbors_per_seed")
        words: list[str] = []
        seed_of: list[str] = []
        for seed in self.config.require("evaluation.seed_words"):
            if seed not in model.wv:
                continue
            words.append(seed)
            seed_of.append(seed)
            for neighbor, _score in model.wv.most_similar(seed, topn=neighbors_per_seed):
                words.append(neighbor)
                seed_of.append(seed)
        return words, seed_of

    def plot(self, model: Word2Vec, out_path: Path | None = None) -> Path:
        """Run t-SNE over those neighbourhoods and save the scatter plot."""
        words, seed_of = self.collect_neighborhoods(model)
        if not words:
            raise ValueError("None of the configured seed words are in the model's vocabulary.")

        tsne_config = self.config.require("evaluation.tsne")
        # t-SNE requires perplexity < n_samples; clamp so a small vocabulary still plots.
        perplexity = min(tsne_config["max_perplexity"], max(2, len(words) - 1))
        coords = TSNE(
            n_components=2,
            random_state=tsne_config["random_state"],
            perplexity=perplexity,
            init=tsne_config["init"],
        ).fit_transform(np.array([model.wv[word] for word in words]))

        figure_config = self.config.require("evaluation.figure")
        plt.figure(figsize=(figure_config["width"], figure_config["height"]))
        for seed in dict.fromkeys(seed_of):
            indices = [i for i, s in enumerate(seed_of) if s == seed]
            plt.scatter(coords[indices, 0], coords[indices, 1], label=seed, s=60)
            for i in indices:
                plt.annotate(words[i], (coords[i, 0], coords[i, 1]), fontsize=9)

        plt.legend()
        plt.title("t-SNE of Word2Vec embeddings: seed words and their nearest neighbors")
        plt.tight_layout()

        out_path = out_path or self.config.path("paths.tsne_figure")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=figure_config["dpi"])
        plt.close()
        return out_path
