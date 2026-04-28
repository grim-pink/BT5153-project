from __future__ import annotations

import time
from typing import Dict, List

from src.inference.spam_predictor import predict_spam_batch, predict_spam_single
from src.intent.classify_intent import classify_intent
from src.utils.config import settings
from src.utils.audit_logger import log_inference_event


def full_pipeline_single(text: str) -> Dict:
    total_start = time.perf_counter()

    spam_start = time.perf_counter()
    result = predict_spam_single(text)
    spam_latency_ms = round((time.perf_counter() - spam_start) * 1000, 2)

    intent_latency_ms = 0.0

    if result["prediction"] == "spam":
        try:
            intent_start = time.perf_counter()
            result["intent"] = classify_intent(result["cleaned_text"])
            intent_latency_ms = round((time.perf_counter() - intent_start) * 1000, 2)
            result["intent_error"] = None
        except Exception as e:
            intent_latency_ms = round((time.perf_counter() - intent_start) * 1000, 2)
            result["intent"] = "Unknown"
            result["intent_error"] = str(e)
    else:
        result["intent"] = None
        result["intent_error"] = None

    total_latency_ms = round((time.perf_counter() - total_start) * 1000, 2)

    result["model_version"] = settings.model_version
    result["spam_latency_ms"] = spam_latency_ms
    result["intent_latency_ms"] = intent_latency_ms
    result["total_latency_ms"] = total_latency_ms

    log_inference_event(result)
    return result


def full_pipeline_batch(texts: List[str]) -> List[Dict]:
    results = []
    for text in texts:
        results.append(full_pipeline_single(text))
    return results