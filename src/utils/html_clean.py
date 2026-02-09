# src/utils/html_clean.py
import re
from collections import Counter

# ----------------------------
# Common chrome / boilerplate
# ----------------------------
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

# ----------------------------
# Step 4 additions: catch short menu headings + more anchors
# ----------------------------

# Very short, high-signal nav headings that appear at the top of IR pages.
# This addresses cases like: "Top Menu", "Overview", "Investors", "Press Releases"
# which your old _is_mostly_menu_tokens() missed due to 1–2 tokens.
SHORT_MENU_HEADINGS = {
    "top menu",
    "menu",
    "overview",
    "investors",
    "investor relations",
    "investor resources",
    "news",
    "newsroom",
    "press releases",
    "press release",
    "events",
    "events & presentations",
    "events and presentations",
    "presentations",
    "financials",
    "quarterly results",
    "annual reports",
    "sec filings",
    "governance",
    "stock info",
    "stock information",
    "contact",
    "resources",
}

# Expand NAV_ANCHORS with common IR/menu tokens that show up in your samples.
NAV_ANCHORS = [
    "top of form",
    "bottom of form",
    "site search",
    "search query",
    "email alerts",
    "investor alert options",
    "powered by q4",
    "view all news",
    "open item",
    "subscribe",
    # Step 4 additions
    "top menu",
    "investor relations",
    "press releases",
    "events & presentations",
    "events and presentations",
    "ir overview",
    "financials",
    "quarterly results",
    "annual reports",
    "sec filings",
]

# ----------------------------
# Menu-line heuristics
# ----------------------------

def _is_mostly_menu_tokens(line: str) -> bool:
    """
    Heuristic: menu lines are often short-ish and look like category lists.
    Step 4: catch very short menu headings (1–2 tokens) that were previously missed.
    """
    low = line.lower().strip()
    if not low:
        return False

    # A) Special-case: very short nav headings (1–2 tokens) and common IR headers
    # Normalize bullet prefixes like "* Overview" or "o News"
    low_norm = re.sub(r"^\s*[\*\-•o]+\s+", "", low).strip()
    if low_norm in SHORT_MENU_HEADINGS:
        return True

    # B) If it contains lots of separators typical of menus
    if any(sep in line for sep in ["|", "•", "  "]):
        tokens = re.findall(r"[a-z0-9]+", low)
        if 3 <= len(tokens) <= 18:
            return True

    # C) If it's a short list of "category-like" tokens (few verbs, mostly nouns)
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

        # F) remove menu-like lines (Step 4 enhanced)
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


# ----------------------------
# Press release body cutter
# ----------------------------
PRESS_WIRE_MARKERS = [
    r"\(business wire\)\s*--",
    r"/prnewswire/\s*--",
    r"\(globe newswire\)\s*--",
    r"\(newsfile\)\s*--",
]

IMMEDIATE_RELEASE_MARKERS = [
    r"\bfor immediate release\b",
    r"\bnews release\b",
    r"\bpress release\b",
]

def cut_to_press_release_body(text: str, max_scan_lines: int = 300) -> str:
    """
    If this looks like an IR/PR page with navigation at the top,
    cut everything before the first likely PR-body marker.
    """
    lines = text.splitlines()
    scan = lines[:max_scan_lines]
    joined = "\n".join(scan).lower()

    # 1) Strong wire markers
    for pat in PRESS_WIRE_MARKERS:
        m = re.search(pat, joined, flags=re.IGNORECASE)
        if m:
            upto = joined[:m.start()]
            cut_line = upto.count("\n")
            return "\n".join(lines[cut_line:]).strip()

    # 2) "FOR IMMEDIATE RELEASE" / "News Release" / "Press Release"
    for pat in IMMEDIATE_RELEASE_MARKERS:
        m = re.search(pat, joined, flags=re.IGNORECASE)
        if m:
            upto = joined[:m.start()]
            cut_line = upto.count("\n")
            return "\n".join(lines[cut_line:]).strip()

    # 3) Dateline style: "CITY, Jan. 15, 2026 --"
    dateline_re = re.compile(
        r"^[A-Z][A-Za-z .'-]{2,40},\s*(?:[A-Z][a-z]{2,9}\.?\s+\d{1,2},\s+\d{4})\s*--",
        re.MULTILINE
    )
    m = dateline_re.search("\n".join(scan))
    if m:
        cut_line = "\n".join(scan)[:m.start()].count("\n")
        return "\n".join(lines[cut_line:]).strip()

    return text.strip()


# ----------------------------
# Block-level nav remover
# ----------------------------
def _looks_like_menu_block(lines):
    if len(lines) < 8:
        return False

    nonempty = [l for l in lines if l.strip()]
    if len(nonempty) < 8:
        return False

    short = sum(1 for l in nonempty if len(l.split()) <= 6) / len(nonempty)
    bullets = sum(1 for l in nonempty if re.match(r"^\s*[\*\-•o]\s+", l)) / len(nonempty)
    punct = sum(1 for l in nonempty if re.search(r"[\.!?]", l)) / len(nonempty)

    if (short > 0.70 and punct < 0.15) or (bullets > 0.30 and punct < 0.20):
        return True

    titlecase = sum(1 for l in nonempty if l[:1].isupper() and len(l.split()) <= 5) / len(nonempty)
    if titlecase > 0.70 and punct < 0.20:
        return True

    # Step 4-ish: blocks dominated by known nav headings
    navish = 0
    for l in nonempty:
        low = re.sub(r"^\s*[\*\-•o]+\s+", "", l.lower()).strip()
        if low in SHORT_MENU_HEADINGS:
            navish += 1
    if navish / len(nonempty) > 0.35 and punct < 0.25:
        return True

    return False


def drop_nav_blocks(text: str, window: int = 25, stride: int = 10) -> str:
    lines = text.splitlines()
    n = len(lines)
    to_drop = [False] * n

    # mark lines containing strong nav anchors
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(a in low for a in NAV_ANCHORS):
            for j in range(max(0, i - 10), min(n, i + 40)):
                to_drop[j] = True

    # sliding window menu detector
    for start in range(0, n, stride):
        end = min(n, start + window)
        block = lines[start:end]
        if _looks_like_menu_block(block):
            for i in range(start, end):
                to_drop[i] = True

    kept = [ln for i, ln in enumerate(lines) if not to_drop[i]]
    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


# ----------------------------
# Main entry
# ----------------------------
def normalize_text_for_embedding(text: str) -> str:
    """
    Order matters:
      1) cut to likely PR body (kills massive nav headers)
      2) drop nav blocks (window detector + anchor-based)
      3) line-level boilerplate removal (repeat + chrome + menu-line heuristic)
      4) dedupe paragraphs
      5) final whitespace normalize
    """
    text = cut_to_press_release_body(text)
    text = drop_nav_blocks(text)
    text = strip_html_boilerplate(text)
    text = dedupe_repeated_paragraphs(text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
