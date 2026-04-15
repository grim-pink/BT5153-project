from __future__ import annotations

import html
import re
import unicodedata

from bs4 import BeautifulSoup
from ftfy import fix_text


def light_clean_text(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text)
    text = fix_text(text)
    text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text