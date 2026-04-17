from __future__ import annotations

from typing import Dict, List

from src.inference.spam_predictor import predict_spam_batch, predict_spam_single
from src.intent.classify_intent import classify_intent
from src.utils.config import settings

# Delivery failed: your package is on hold. Pay the shipping fee now to reschedule delivery.
def full_pipeline_single(text: str) -> Dict:
    result = predict_spam_single(text)

    if result["prediction"] == "spam":
        try:
            result["intent"] = classify_intent(result["cleaned_text"])
            result["intent_error"] = None
        except Exception as e:
            result["intent"] = "Unknown"
            result["intent_error"] = str(e)
    else:
        result["intent"] = None
        result["intent_error"] = None

    result["model_version"] = settings.model_version
    return result


def full_pipeline_batch(texts: List[str]) -> List[Dict]:
    results = predict_spam_batch(texts)

    for item in results:
        if item["prediction"] == "spam":
            try:
                item["intent"] = classify_intent(item["cleaned_text"])
                item["intent_error"] = None
            except Exception:
                item["intent"] = "Unknown"
                item["intent_error"] = str(e)
        else:
            item["intent"] = None
            item["intent_error"] = None

        item["model_version"] = settings.model_version

    return results