import os
import uuid
import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

import yaml
from nltk.tokenize import sent_tokenize

from classifier_knn import KNNClassifier
from mf_utils.process import load_file

# Keep your existing text normalizer for general whitespace cleanup
from mf_utils.text_clean import normalize_text

from mf_utils.html_clean import normalize_text_for_embedding

# ----------------------------
# CONFIG
# ----------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
CHUNK_TOKEN_LIMIT = 256

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        CONFIG = yaml.safe_load(f) or {}
    CHUNK_TOKEN_LIMIT = CONFIG.get("agent", {}).get("chunk_token_limit", CHUNK_TOKEN_LIMIT)

# ----------------------------
# CLASSIFIER (lazy init)
# ----------------------------
_classifier_instance: Optional[KNNClassifier] = None

def get_classifier() -> KNNClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        ref_dir = os.path.join(os.path.dirname(__file__), "..", "reference_docs_clean")
        print(f"⚡ Initializing KNN Classifier from {ref_dir}...")
        _classifier_instance = KNNClassifier(ref_dir)
    return _classifier_instance

# ----------------------------
# DATA MODELS
# ----------------------------
@dataclass
class Section:
    role: str
    text: str

@dataclass
class DocumentMF:
    id: str
    source: str
    document_type: str
    file_path: str
    raw_text: str
    chunks: List[str]
    sections: List[Section] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

# ----------------------------
# HELPERS
# ----------------------------
def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def to_full_text(loaded_file) -> str:
    """
    load_file() returns either:
      - list[Document] (langchain) or
      - single Document-like object
      - or string
    """
    if isinstance(loaded_file, list):
        return "\n".join(getattr(doc, "page_content", str(doc)) for doc in loaded_file)
    return getattr(loaded_file, "page_content", str(loaded_file))

def get_title_from_loaded(loaded_file) -> str:
    if isinstance(loaded_file, list) and loaded_file:
        md = getattr(loaded_file[0], "metadata", {}) or {}
        return (md.get("title") or "").strip()
    md = getattr(loaded_file, "metadata", {}) or {}
    return (md.get("title") or "").strip()

def is_html(path: str) -> bool:
    return path.lower().endswith((".html", ".htm"))

def chunk_str(text: str) -> List[str]:
    sentences = sent_tokenize(text)
    chunks: List[str] = []
    current: List[str] = []
    length = 0

    for s in sentences:
        n = len(s.split())
        if length + n <= CHUNK_TOKEN_LIMIT:
            current.append(s)
            length += n
        else:
            if current:
                chunks.append(" ".join(current).strip())
            current = [s]
            length = n

    if current:
        chunks.append(" ".join(current).strip())

    # Avoid empty chunk lists
    return [c for c in chunks if c]

def _safe_compact(text: str) -> str:
    """
    Lightweight cleanup that should never erase content.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = normalize_text(text)
    # normalize_text may already collapse spaces; ensure newlines aren't destroyed
    text = "\n".join(line.rstrip() for line in text.splitlines())
    # collapse excessive blank lines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()

def clean_html_text(raw: str) -> str:
    """
    HTML cleanup pipeline:
      1) lightweight safe normalize
      2) strip boilerplate once
      3) dedupe repeated paragraphs
      4) final safe normalize
    Includes a guardrail: if cleaning nukes content, revert.
    """
    original = _safe_compact(raw)
    print("orig len:", len(original))

    if not original:
        return ""

    # Step 1: strip obvious boilerplate & menus
    cleaned = strip_html_boilerplate(original, repeat_threshold=3)
    cleaned = _safe_compact(cleaned)

    # Step 2: dedupe long repeated paragraphs
    cleaned2 = dedupe_repeated_paragraphs(cleaned, min_len=200)
    cleaned2 = _safe_compact(cleaned2)

    # Guardrail: if we removed too much, revert to a milder version
    # (Your current failure is basically "cleaned text becomes tiny".)
    if len(cleaned2) < max(400, int(0.10 * len(original))):
        # Try a less aggressive boilerplate filter
        cleaned_soft = strip_html_boilerplate(original, repeat_threshold=5)
        cleaned_soft = _safe_compact(cleaned_soft)
        cleaned_soft = dedupe_repeated_paragraphs(cleaned_soft, min_len=200)
        cleaned_soft = _safe_compact(cleaned_soft)

        # Pick the best candidate by length (proxy for "didn't nuke content")
        best = max([cleaned2, cleaned_soft, original], key=len)
        return best
    print("clean len:", len(cleaned2))
    return cleaned2

def make_knn_sample(chunks: List[str], title: str = "") -> str:
    """
    A better representative sample than grabbing arbitrary positions.
    We want:
      - headline/title if available
      - first 1-2 chunks (lead)
      - last chunk (often has About/Forward-Looking)
    """
    if not chunks:
        return title.strip()

    parts: List[str] = []
    if title:
        parts.append(title.strip())

    parts.extend(chunks[:2])

    if len(chunks) >= 3:
        parts.append(chunks[-1])

    sample = " ".join(p.strip() for p in parts if p and p.strip())
    # Keep sample from getting too long for embedding
    return sample[:6000]

# ----------------------------
# CLASSIFICATION
# ----------------------------
def classify_document_type(sample_text: str) -> str:
    low = sample_text.lower()

    SEC_FILE_NO_RE = re.compile(r"commission\s*file\s*number", re.IGNORECASE)

    def is_likely_sec_filing(text: str) -> bool:
        compact = re.sub(r"\s+", "", text.lower())
        has_header = "unitedstatessecuritiesandexchangecommission" in compact
        has_file_no = bool(SEC_FILE_NO_RE.search(text))
        has_xbrl = "xbrli:shares" in compact or "us-gaap:" in compact or "ix:header" in compact
        return has_header or has_file_no or has_xbrl

    # LEVEL 1: explicit rules
    if ("operator:" in low and "question-and-answer" in low) or ("operator:" in low and "q&a" in low):
        return "EARNINGS_CALL_TRANSCRIPT"

    if is_likely_sec_filing(sample_text):
        return "SEC_FILING"
        
    # LEVEL 2: KNN semantic
    clf = get_classifier()
    label, score = clf.classify(sample_text)
    print(f"DEBUG: KNN matched '{label}' (Confidence: {score:.2f})")

    # IMPORTANT: don't force INTERNAL_MEMO too easily
    # If KNN says press_release, accept at a slightly lower threshold
    # (HTML noise makes similarity lower.)
    # if label == "press_release" and score >= 0.35:
    #     return "PRESS_RELEASE"

    if score >= 0.35:
        return label.upper()

    # LEVEL 3: weak heuristics (optional)
    # if "business wire" in low or "prnewswire" in low or "forward-looking statements" in low:
    #     return "PRESS_RELEASE"

    return "INTERNAL_MEMO"

# ----------------------------
# SECTIONING
# ----------------------------
def sectionize_earnings_call(text: str) -> List[Section]:
    lines = [l for l in text.splitlines() if l.strip()]
    sections: List[Section] = []
    current_role = "intro"
    current_text: List[str] = []

    def flush(role: str, buf: List[str]):
        if buf:
            sections.append(Section(role=role, text="\n".join(buf).strip()))

    for line in lines:
        lower = line.lower()
        if "question-and-answer" in lower or "q&a" in lower:
            flush(current_role, current_text)
            current_role, current_text = "qa", []
        elif any(k in lower for k in ["revenue", "margin", "growth"]) and current_role == "intro":
            flush(current_role, current_text)
            current_role, current_text = "kpi_commentary", []
        elif any(k in lower for k in ["guidance", "outlook"]) and current_role != "qa":
            flush(current_role, current_text)
            current_role, current_text = "guidance", []

        current_text.append(line)

    flush(current_role, current_text)
    return sections

def sectionize_default(text: str) -> List[Section]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return []
    out = [Section("summary", paras[0])]
    if len(paras) > 1:
        out.append(Section("body", "\n\n".join(paras[1:])))
    return out

def sectionize(text: str, doc_type: str) -> List[Section]:
    if doc_type == "EARNINGS_CALL_TRANSCRIPT":
        return sectionize_earnings_call(text)
    return sectionize_default(text)

# ----------------------------
# MAIN INGESTION
# ----------------------------
def ingest_and_classify(file_path: str) -> DocumentMF:
    print("ABS PATH:", os.path.abspath(file_path))
    print("SIZE:", os.path.getsize(file_path))
    print("SHA1:", sha1_file(file_path))

    print("   ...Loading file...")
    loaded = load_file(file_path)

    # Debug: show first doc's meta/content
    if isinstance(loaded, list) and loaded:
        print("loaded_file type:", type(loaded))
        print("loaded_file len:", len(loaded))
        md0 = getattr(loaded[0], "metadata", {}) or {}
        print("doc0 keys:", md0.keys())
        # print("doc0 page_content[:1000]:", repr(getattr(loaded[0], "page_content", "")[:1000]))

    title = get_title_from_loaded(loaded)
    raw_text = to_full_text(loaded)

    # Clean
    if is_html(file_path):
        raw_text = _safe_compact(raw_text)
        raw_text = normalize_text_for_embedding(raw_text)
    else:
        raw_text = _safe_compact(raw_text)


    raw0 = to_full_text(loaded)
    print("before_clean[:200]:", raw0[:200])
    
    if is_html(file_path):
        raw_text = _safe_compact(raw0)
        raw_text = normalize_text_for_embedding(raw_text)
    else:
        raw_text = _safe_compact(raw0)
    
    print("after_clean[:200]:", raw_text[:200])


    print("   ...Chunking text...")
    chunks = chunk_str(raw_text)

    # If chunking returns a single tiny chunk for HTML, something is still wrong:
    # keep a non-empty fallback to avoid empty embeddings/classification.
    if not chunks and raw_text:
        chunks = [raw_text[:2000]]

    sample_text = make_knn_sample(chunks, title=title)

    print("   ...Classifying document type...")
    doc_type = classify_document_type(sample_text)

    secs = sectionize(raw_text, doc_type)

    return DocumentMF(
        id=str(uuid.uuid4()),
        source="upload",
        document_type=doc_type,
        file_path=file_path,
        raw_text=raw_text,
        chunks=chunks,
        sections=secs,
        metadata={
            "original_path": file_path,
            "title": title,
        },
    )
