# ================================
# FILE: src/preprocessing/text_cleaner.py
# ================================
from __future__ import annotations
import unicodedata
import regex as re
from dataclasses import dataclass
from typing import List

try:
    import emoji as _emoji
except Exception:
    _emoji = None

# Patterns
URL_PATTERN = re.compile(r"((?:https?://|www\.)\S+)")
EMAIL_PATTERN = re.compile(r"\b[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}\b")
MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#\w+")
HTML_LIKE = re.compile(r"<[^>]+>")
REPEAT_CHAR = re.compile(r"(\w)\1{2,}")  # char len>2 -> 2


def _casefold(s: str) -> str:
    try:
        return s.casefold()
    except Exception:
        return s.lower()


def _squash(token: str) -> str:
    return REPEAT_CHAR.sub(r"\1\1", token)


def _demojize(text: str) -> str:
    if _emoji is None:
        return text
    return _emoji.demojize(text, language="en", delimiters=(" ", " "))


@dataclass
class NormalizationConfig:
    lower: bool = True
    unicode_nfkc: bool = True
    collapse_ws: bool = True
    strip_emails: bool = True
    strip_urls: bool = True
    strip_hashtags: bool = False  # if False, remove '#', keep word
    strip_mentions: bool = True
    strip_html_like: bool = True
    drop_digits: bool = False
    drop_punct: bool = True   # handled by pattern below
    emoji_policy: str = "demojize"  # remove|demojize|keep


class TextCleaner:
    """Rule-based normalizer/cleaner before tokenization."""
    def __init__(self, cfg: NormalizationConfig = NormalizationConfig()):
        self.cfg = cfg

    def clean(self, s: str) -> str:
        if not isinstance(s, str):
            return ""
        x = s
        if self.cfg.unicode_nfkc:
            x = unicodedata.normalize("NFKC", x)
        if self.cfg.strip_html_like:
            x = HTML_LIKE.sub(" ", x)
        if self.cfg.strip_urls:
            x = URL_PATTERN.sub(" ", x)
        if self.cfg.strip_emails:
            x = EMAIL_PATTERN.sub(" ", x)
        if self.cfg.strip_mentions:
            x = MENTION_PATTERN.sub(" ", x)
        # hashtags
        if self.cfg.strip_hashtags:
            x = HASHTAG_PATTERN.sub(" ", x)
        else:
            x = HASHTAG_PATTERN.sub(lambda m: m.group(0)[1:], x)
        # emoji
        if self.cfg.emoji_policy == "remove" and _emoji:
            x = _emoji.replace_emoji(x, replace=" ")
        elif self.cfg.emoji_policy == "demojize":
            x = _demojize(x)
        # keep letters, numbers, spaces, '.', '-' and apostrophes
        x = re.sub(r"[^\p{L}\p{N}\s'\.\-]", " ", x)
        if self.cfg.drop_digits:
            x = re.sub(r"\d+", " ", x)
        if self.cfg.lower:
            x = _casefold(x)
        if self.cfg.collapse_ws:
            x = re.sub(r"\s+", " ", x).strip()
        return x

    def clean_many(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]