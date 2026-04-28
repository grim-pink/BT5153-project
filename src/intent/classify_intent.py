from __future__ import annotations

import os

from langchain_ollama import ChatOllama

from src.intent.prompts import FEW_SHOT_EXAMPLES, LABELS, SYSTEM_PROMPT

_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
_OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
#http://host.docker.internal:11434
#http://localhost:11434 - for training
_LLM = None


def get_llm():
    global _LLM
    if _LLM is None:
        _LLM = ChatOllama(
            model=_OLLAMA_MODEL,
            temperature=_OLLAMA_TEMPERATURE,
            base_url=_OLLAMA_BASE_URL
        )
    return _LLM


def _build_messages(text: str):
    messages = [("system", SYSTEM_PROMPT)]

    for example_text, example_label in FEW_SHOT_EXAMPLES:
        messages.append(("human", f"Text: {example_text}"))
        messages.append(("assistant", example_label))

    messages.append(("human", f"Text: {text}"))
    return messages


def _normalize_label(response_text: str) -> str:
    cleaned = response_text.strip()

    if cleaned in LABELS:
        return cleaned

    for label in LABELS:
        if label.lower() in cleaned.lower():
            return label

    return "Benign"


def classify_intent(text: str, mode: str = "few") -> str:
    llm = get_llm()

    if mode == "zero":
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", f"Text: {text}")
        ]
    else:
        messages = _build_messages(text)  # few-shot

    response = llm.invoke(messages).content
    return _normalize_label(response)
