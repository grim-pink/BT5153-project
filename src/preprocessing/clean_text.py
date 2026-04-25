import re
import html
import unicodedata

from bs4 import BeautifulSoup
from ftfy import fix_text


def light_clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = fix_text(text)
    text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\bect\b", "", text)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text
