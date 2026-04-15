from __future__ import annotations

from typing import Dict, List

import torch

from src.inference.load_spam_model import get_device, get_model, get_tokenizer, load_model, load_tokenizer
from src.preprocessing.clean_text import light_clean_text
from src.utils.config import settings


def _ensure_loaded():
    tokenizer = get_tokenizer()
    model = get_model()

    if tokenizer is None:
        tokenizer = load_tokenizer()
    if model is None:
        model = load_model()

    return tokenizer, model, get_device()


def predict_spam_single(text: str) -> Dict:
    results = predict_spam_batch([text])
    return results[0]


def predict_spam_batch(texts: List[str]) -> List[Dict]:
    tokenizer, model, device = _ensure_loaded()

    cleaned_texts = [light_clean_text(t) for t in texts]

    inputs = tokenizer(
        cleaned_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=settings.max_length,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    predicted_labels = preds.cpu().tolist()
    confidences = probs.max(dim=1).values.cpu().tolist()
    all_probabilities = probs.cpu().tolist()

    results = []
    for original_text, cleaned_text, pred, conf, prob in zip(
        texts, cleaned_texts, predicted_labels, confidences, all_probabilities
    ):
        label = "spam" if pred == 1 else "ham"

        results.append({
            "text": original_text,
            "cleaned_text": cleaned_text,
            "prediction": label,
            "confidence": round(conf, 4),
            "probabilities": {
                "ham": round(prob[0], 4),
                "spam": round(prob[1], 4),
            },
        })

    return results
