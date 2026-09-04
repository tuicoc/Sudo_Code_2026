"""Word2Vec trained on viwik18, a cleaned Vietnamese Wikipedia dump.

DataLoader          - download the corpus shards and the stopword list
Preprocessor        - split into segments, segment into words, drop stopwords
Word2VecTrainer     - train and query the gensim model
EmbeddingVisualizer - project the embeddings to 2-D with t-SNE
"""

from src.config import Config, load_config
from src.dataloader import DataLoader
from src.preprocessing import Preprocessor
from src.trainer import Word2VecTrainer
from src.visualization import EmbeddingVisualizer

__all__ = [
    "Config", "load_config", "DataLoader", "Preprocessor",
    "Word2VecTrainer", "EmbeddingVisualizer",
]
