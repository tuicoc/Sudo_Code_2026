"""An encoder-decoder Transformer for English -> Vietnamese translation (EVBCorpus).

DataLoader   - download and unpack EVBNews, parse its SGML, split by document
Preprocessor - tokenise, filter by length, and the statistics behind the config
Vocabulary   - word <-> id, one per language
Transformer  - the model: positional embedding, encoder/decoder stacks, three attention uses
Trainer      - the warmup schedule, the label-smoothed loss, and the distributed fit
Translator   - greedy and beam-search decoding
Evaluator    - BLEU, the dictionary baseline, cross-attention vs the human word alignment
"""

from src.config import Config, load_config
from src.dataloader import DataLoader
from src.evaluation import Evaluator
from src.preprocessing import Preprocessor
from src.training import MaskedAccuracy, Trainer, WarmupSchedule
from src.transformer import Transformer, verify_masking
from src.translator import Translator
from src.vocabulary import Vocabulary

__all__ = [
    "Config", "load_config", "DataLoader", "Preprocessor", "Vocabulary",
    "Transformer", "verify_masking", "Trainer", "WarmupSchedule", "MaskedAccuracy",
    "Translator", "Evaluator",
]
