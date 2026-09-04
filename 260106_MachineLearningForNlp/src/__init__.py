"""Naive Bayes vs SVM on Vietnamese product-review sentiment.

DataLoader       - download the reviews, read and write the corpus files
Preprocessor     - NFC -> clean -> lowercase -> word-segment -> expand teencode
FeatureExtractor - the Bag-of-Words / TF-IDF vectorizers being compared
CrossValidator   - stratified 5-fold CV, one summary dict per experiment
ResultsReporter  - the comparison table and the bar chart
"""

from src.config import Config, load_config
from src.cross_validation import CrossValidator
from src.dataloader import DataLoader
from src.feature_extraction import FeatureExtractor
from src.preprocessing import Preprocessor
from src.reporting import ResultsReporter

__all__ = [
    "Config", "load_config", "DataLoader", "Preprocessor",
    "FeatureExtractor", "CrossValidator", "ResultsReporter",
]
