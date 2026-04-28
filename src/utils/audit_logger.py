from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

CSV_LOG_PATH = LOG_DIR / "inference_audit.csv"
JSONL_LOG_PATH = LOG_DIR / "inference_audit.jsonl"


CSV_HEADERS = [
    "timestamp_utc",
    "input_hash",
    "text_preview",
    "prediction",
    "confidence",
    "intent",
    "intent_error",
    "model_version",
    "spam_latency_ms",
    "intent_latency_ms",
    "total_latency_ms",
]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _preview(text: str, n: int = 80) -> str:
    return text[:n].replace("\n", " ")


def log_inference_event(event: Dict[str, Any]) -> None:
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_hash": _hash_text(event.get("text", "")),
        "text_preview": _preview(event.get("text", "")),
        "prediction": event.get("prediction"),
        "confidence": event.get("confidence"),
        "intent": event.get("intent"),
        "intent_error": event.get("intent_error"),
        "model_version": event.get("model_version"),
        "spam_latency_ms": event.get("spam_latency_ms"),
        "intent_latency_ms": event.get("intent_latency_ms"),
        "total_latency_ms": event.get("total_latency_ms"),
    }

    # CSV
    write_header = not CSV_LOG_PATH.exists()
    with open(CSV_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    # JSONL
    with open(JSONL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")