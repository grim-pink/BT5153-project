from __future__ import annotations

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.utils.config import settings

_MODEL = None
_TOKENIZER = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device():
    return _DEVICE


def get_model():
    return _MODEL


def get_tokenizer():
    return _TOKENIZER


def load_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = DistilBertTokenizerFast.from_pretrained(settings.model_path)
    return _TOKENIZER


def load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = DistilBertForSequenceClassification.from_pretrained(settings.model_path)
        _MODEL.to(_DEVICE)
        _MODEL.eval()
    return _MODEL
