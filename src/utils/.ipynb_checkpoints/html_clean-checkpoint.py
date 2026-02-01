# src/utils/html_clean.py
import re
from collections import Counter

CHROME_SUBSTRINGS = [
    "skip to main content",
    "skip to main navigation",
    "top of form",
    "bottom of form",
    "cookie",
    "privacy notice",
    "terms of use",
    "sign in",
    "log in",
    "subscribe",
    "you are being connected to our automated chatbot",
]

# These are NOT site-specific; common on investor relations / press release pages
PRESS_RELEASE_CHROME = [
    "view all press releases",
    "download this press release",
    "(opens in new window)",
    "view original content",
]

FOOTER_SUBSTRINGS = [
    "©", "(c)", "all rights reserved",
    "privacy", "terms", "help", "careers",
]


def _is_mostly_menu_tokens(line: str) -> bool:
    """
    Heuristic: menu lines are often short-ish and look like category lists.
    """
    low = line.lower().strip()
    if not low:
        return False

    # If it contains lots of separators typical of menus
    if any(sep in line for sep in ["|", "•", "  "]):
        tokens = re.findall(r"[a-z0-9]+", low)
        if 3 <= len(tokens) <= 18:
            return True

    # If it's a short list of "category-like" tokens (few verbs, mostly nouns)
    tokens = re.findall(r"[a-z0-9]+", low)
    if 3 <= len(tokens) <= 12:
        common_verbs = {
            "is", "are", "was", "were", "be", "been", "being",
            "announce", "announces", "said", "says", "today", "will"
        }
        if not any(v in tokens for v in common_verbs):
            if not re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", low):
                avg_len = sum(len(t) for t in tokens) / len(tokens)
                if avg_len <= 9:
                    return True

    return False


def strip_html_boilerplate(text: str, repeat_threshold: int = 3) -> str:
    """
    Remove repeated nav/header/footer lines and obvious chrome.
    IMPORTANT: only remove 'chrome' phrases when line is short-ish, otherwise
    we might delete the entire article if extraction collapses it into one long line.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    lowered = [ln.lower().strip() for ln in lines if ln.strip()]
    counts = Counter(lowered)

    cleaned = []
    for ln in lines:
        low = ln.lower().strip()

        if not low:
            cleaned.append("")
            continue

        word_count = len(low.split())
        char_count = len(low)
        is_shortish = (word_count <= 12) or (char_count <= 90)

        # A) remove lines that repeat a lot (nav/menu repeats)
        if counts[low] >= repeat_threshold:
            continue

        # B) remove obvious chrome phrases (ONLY if shortish)
        if is_shortish and any(s in low for s in CHROME_SUBSTRINGS):
            continue

        # C) remove common press-release page chrome (ONLY if shortish)
        if is_shortish and any(s in low for s in PRESS_RELEASE_CHROME):
            continue

        # D) remove SOURCE lines safely (ONLY if shortish and startswith)
        if is_shortish and (low.startswith("source") or low.startswith("source:")):
            continue

        # E) remove footer-ish single lines
        if any(s in low for s in FOOTER_SUBSTRINGS) and word_count <= 8:
            continue

        # F) remove menu-like lines
        if _is_mostly_menu_tokens(ln):
            continue

        cleaned.append(ln)

    return "\n".join(cleaned)


def dedupe_repeated_paragraphs(text: str, min_len: int = 200) -> str:
    """
    Remove duplicated long paragraphs (often content appears twice).
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return text

    counts = Counter(paras)
    out = []
    seen = set()

    for p in paras:
        if len(p) >= min_len and counts[p] > 1:
            if p in seen:
                continue
            seen.add(p)
        out.append(p)

    return "\n\n".join(out)


def normalize_text_for_embedding(text: str) -> str:
    text = strip_html_boilerplate(text)
    text = dedupe_repeated_paragraphs(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
