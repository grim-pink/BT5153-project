from __future__ import annotations

from typing import Dict, List

from src.inference.spam_predictor import predict_spam_batch, predict_spam_single
from src.intent.classify_intent import classify_intent
from src.utils.config import settings


def full_pipeline_single(text: str) -> Dict:
    result = predict_spam_single(text)

    if result["prediction"] == "spam":
        try:
            result["intent"] = classify_intent(result["cleaned_text"])
        except Exception:
            result["intent"] = "Unknown"
    else:
        result["intent"] = None

    result["model_version"] = settings.model_version
    return result


def full_pipeline_batch(texts: List[str]) -> List[Dict]:
    results = predict_spam_batch(texts)

    for item in results:
        if item["prediction"] == "spam":
            try:
                item["intent"] = classify_intent(item["cleaned_text"])
            except Exception:
                item["intent"] = "Unknown"
        else:
            item["intent"] = None

        item["model_version"] = settings.model_version

    return results