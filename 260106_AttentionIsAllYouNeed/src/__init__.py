"""Attention for Vietnamese text summarization (VNDS).

DataLoader   - download the corpus, read and write every artifact
Preprocessor - truncate, split sentences, and measure what the truncation costs
Vocabulary   - the shared 20k vocabulary, coverage table, encode/decode
attention    - AdditiveAttention and DotProductAttention, written out by hand
Seq2Seq      - the GRU encoder-decoder, with attention or without it
Decoder      - greedy decoding with n-gram blocking, and the attention weights
Evaluator    - ROUGE-1/2/L and the Lead-n extractive baselines
"""

from src.attention import AdditiveAttention, DotProductAttention, build_attention
from src.config import Config, load_config
from src.dataloader import DataLoader
from src.decoding import Decoder
from src.evaluation import Evaluator
from src.model import MaskedAccuracy, Seq2Seq, masked_loss
from src.preprocessing import Preprocessor
from src.vocabulary import Vocabulary

__all__ = [
    "Config", "load_config", "DataLoader", "Preprocessor", "Vocabulary",
    "AdditiveAttention", "DotProductAttention", "build_attention",
    "Seq2Seq", "masked_loss", "MaskedAccuracy", "Decoder", "Evaluator",
]
