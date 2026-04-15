from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    model_version: str = os.getenv("MODEL_VERSION", "v1.0")
    model_path: str = os.getenv("MODEL_PATH", "./artifacts/distilbert_spam_model")
    max_length: int = int(os.getenv("MAX_LENGTH", "64"))


settings = Settings()
