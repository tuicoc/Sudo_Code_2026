"""Vietnamese news text preprocessing.

DataLoader   - download the corpus and stopword list, read and write files
Preprocessor - NFC -> clean -> fold case -> tokenize -> drop stopwords
"""

from src.config import Config, load_config
from src.dataloader import DataLoader
from src.preprocessing import Preprocessor

__all__ = ["Config", "load_config", "DataLoader", "Preprocessor"]
