"""A neural network for 10-class Vietnamese news topic classification (VNTC).

DataLoader       - read the corpus and every artifact the stages exchange
Preprocessor     - UTF-16 -> NFC -> word-segmented UTF-8, cached to data/processed/
FeatureExtractor - the segmentation-aware tokenizer and the TF-IDF matrices
TopicClassifier  - the Keras model: build, train, save, predict
Evaluator        - test metrics, confusion matrix, and the baselines to check against
ModelComparison  - network vs SVM vs NB, segmented vs unsegmented
"""

from src.comparison import ModelComparison
from src.config import Config, load_config
from src.dataloader import DataLoader
from src.evaluation import Evaluator
from src.feature_extraction import FeatureExtractor
from src.model import TopicClassifier
from src.preprocessing import Preprocessor

__all__ = [
    "Config", "load_config", "DataLoader", "Preprocessor", "FeatureExtractor",
    "TopicClassifier", "Evaluator", "ModelComparison",
]
