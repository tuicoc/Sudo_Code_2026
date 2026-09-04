"""Training the gensim Word2Vec model and querying the embeddings it learned."""

from __future__ import annotations

import os
from pathlib import Path

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from src.config import Config


class Word2VecTrainer:
    """Trains a skip-gram Word2Vec model on the sentences file, and loads it back.

    Sentences stream off disk via `LineSentence` -- the corpus does not fit comfortably in RAM.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model: Word2Vec | None = None

    def _hyperparameters(self) -> dict:
        params = dict(self.config.require("word2vec"))
        # `workers: null` in the config means "every core this machine has".
        params["workers"] = params.get("workers") or os.cpu_count() or 4
        return params

    def train(self, sentences_path: Path | None = None) -> Word2Vec:
        """Train on the sentences file and keep the model on the instance."""
        sentences_path = sentences_path or self.config.path("paths.sentences_file")
        if not sentences_path.exists():
            raise FileNotFoundError(
                f"{sentences_path} is missing. Run the corpus stage first: python main.py --stage corpus"
            )
        self.model = Word2Vec(sentences=LineSentence(str(sentences_path)), **self._hyperparameters())
        return self.model

    def save(self, path: Path | None = None) -> Path:
        """Save the trained model (gensim writes several sidecar `.npy` files next to it)."""
        if self.model is None:
            raise RuntimeError("Nothing to save: call train() or load() first.")
        path = path or self.config.path("paths.model_file")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        return path

    def load(self, path: Path | None = None) -> Word2Vec:
        """Load a previously trained model."""
        path = path or self.config.path("paths.model_file")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Train the model first: python main.py --stage train"
            )
        self.model = Word2Vec.load(str(path))
        return self.model

    def most_similar(self, word: str, topn: int = 5) -> list[tuple[str, float]]:
        """Nearest neighbours of a word, or an empty list if it is out of vocabulary."""
        model = self._require_model()
        return model.wv.most_similar(word, topn=topn) if word in model.wv else []

    def describe(self) -> str:
        """A one-line summary of what was trained."""
        model = self._require_model()
        return f"trained on {model.corpus_count:,} sentences, vocabulary {len(model.wv):,}"

    def _require_model(self) -> Word2Vec:
        if self.model is None:
            raise RuntimeError("No model yet: call train() or load() first.")
        return self.model
