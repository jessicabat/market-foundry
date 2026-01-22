import os
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any
import re
import yaml
from transformers import pipeline
classifier_pipeline = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-small")
from nltk.tokenize import sent_tokenize

from utils.process import load_file


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)
    CHUNK_TOKEN_LIMIT = CONFIG.get("agent", {}).get("chunk_token_limit", 256)
else:
    CHUNK_TOKEN_LIMIT = 256


@dataclass
class Section:
    role: str
    text: str


@dataclass
class DocumentMF:
    id: str
    source: str          # "upload"
    document_type: str   # EARNINGS_CALL_TRANSCRIPT, PERFORMANCE_REPORT, etc.
    file_path: str
    raw_text: str
    chunks: List[str]
    sections: List[Section] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def to_full_text(loaded_file) -> str:
    """Flatten LangChain docs into one string."""
    if isinstance(loaded_file, list):
        return "".join(doc.page_content for doc in loaded_file)
    return getattr(loaded_file, "page_content", str(loaded_file))


def chunk_str(text: str) -> List[str]:
    sentences = sent_tokenize(text)
    chunks, current, length = [], [], 0
    for s in sentences:
        n = len(s.split())
        if length + n <= CHUNK_TOKEN_LIMIT:
            current.append(s)
            length += n
        else:
            if current:
                chunks.append(" ".join(current))
            current, length = [s], n
    if current:
        chunks.append(" ".join(current))
    return chunks


# def classify_document_type(sample_text: str, filename: str = "") -> str:
#     """
#     Robust hierarchical classifier. 
#     1. Checks explicit headers (SEC, Earnings).
#     2. Checks layout signals (Press Release).
#     3. Filters 'Safe Harbor' noise before checking Internal types.
#     """
#     text = sample_text.lower()
    
#     # --- LEVEL 1: STRUCTURED / PUBLIC DOCS (High Confidence) ---
    
#     # 1. SEC Filings (Regex is safer than keywords)
#     # Matches "United States Securities... Form 8-K" or "Form 10-K" headers
#     if re.search(r"form\s+(8-k|10-k|10-q)", text) or \
#        ("securities and exchange commission" in text and "washington" in text):
#         return "SEC_FILING"

#     # 2. Earnings Call Transcripts
#     # Look for "Operator" dialogue or Q&A headers
#     if "operator:" in text and ("question-and-answer" in text or "q&a" in text):
#         return "EARNINGS_CALL_TRANSCRIPT"

#     # 3. Press Releases
#     # Look for PR specific contact info or release lines
#     if any(k in text for k in ["for immediate release", "contact investors:", "contact media:"]):
#         return "PRESS_RELEASE"
        
#     # --- LEVEL 2: NOISE FILTERING ---
    
#     # Many decks/reports start with "Forward Looking Statements". 
#     # If we see this, we should be careful about matching "Risk" keywords immediately.
#     is_disclaimer_heavy = "forward-looking statements" in text or "safe harbor" in text

#     # --- LEVEL 3: INTERNAL / SPECIFIC DOCS ---

#     # 4. Strategy Decks
#     # Often PPT PDFs. Look for "Vision", "Roadmap" BUT also check file extension if possible
#     # We prioritize this over 'Policy' to avoid misclassifying decks as risk docs.
#     if any(k in text for k in ["strategic plan", "roadmap", "growth pillar", "long-term targets"]):
#         return "STRATEGY_DECK"

#     # 5. Technical Reports (distinct vocabulary)
#     if any(k in text for k in ["model validation", "hyperparameter", "inference latency", "backtest"]):
#         return "TECHNICAL_REPORT"

#     # 6. Performance Reports (FP&A vocabulary)
#     if any(k in text for k in ["variance analysis", "actual vs budget", "kpi dashboard", "ebitda bridge"]):
#         return "PERFORMANCE_REPORT"

#     # 7. Policy / Risk Docs
#     # CRITICAL FIX: Only classify as Policy if it's NOT just a disclaimer.
#     # We check if 'risk' appearing is substantive (e.g., 'Risk Management Framework') 
#     # rather than just 'risk factors' in a disclaimer.
#     if "risk management policy" in text or "compliance framework" in text or "internal audit" in text:
#         return "POLICY_OR_RISK_DOC"
        
#     # Fallback for "Risk" keywords only if it's NOT a disclaimer-heavy text
#     if not is_disclaimer_heavy and "risk appetite" in text:
#         return "POLICY_OR_RISK_DOC"

#     # --- LEVEL 4: FALLBACKS ---
    
#     # If it's HTML but didn't match above, it's likely a generic News Article
#     if filename.endswith(".html") or filename.endswith(".htm"):
#         return "NEWS_ARTICLE"

#     return "INTERNAL_MEMO"

def classify_document_type(sample_text: str) -> str:
    """
    Semantic classification using a Zero-Shot Transformer.
    This understands meaning, not just keywords.
    """
    # 1. Define your candidate labels (the list from your table)
    candidate_labels = [
        "Earnings Call Transcript",
        "Press Release", 
        "News Article",
        "SEC Filing", 
        "Internal Memo", 
        "Strategy Deck", 
        "Technical Report",
        "Performance Report",
        "Risk Policy Document"
    ]

    # 2. Run inference
    # We only need the first ~500-1000 characters to usually tell the type.
    # Passing too much text slows it down and confuses the model.
    truncated_text = sample_text[:2000] 
    
    result = classifier_pipeline(truncated_text, candidate_labels)
    
    # 3. Get the top predicted label
    top_label = result['labels'][0]
    score = result['scores'][0]

    # 4. Map the nice label back to your ENUM strings
    label_map = {
        "Earnings Call Transcript": "EARNINGS_CALL_TRANSCRIPT",
        "Press Release": "PRESS_RELEASE",
        "News Article": "NEWS_ARTICLE",
        "SEC Filing": "SEC_FILING",
        "Internal Memo": "INTERNAL_MEMO",
        "Strategy Deck": "STRATEGY_DECK",
        "Technical Report": "TECHNICAL_REPORT",
        "Performance Report": "PERFORMANCE_REPORT",
        "Risk Policy Document": "POLICY_OR_RISK_DOC"
    }
    
    print(f"DEBUG: Classified as {top_label} (Confidence: {score:.2f})")
    
    # Optional: Threshold check. If not confident, mark as UNKNOWN or MEMO
    if score < 0.4:
        return "INTERNAL_MEMO"
        
    return label_map.get(top_label, "INTERNAL_MEMO")

def sectionize_earnings_call(text: str) -> List[Section]:
    # Very rough: split by lines, tag intro vs KPI vs Q&A
    lines = [l for l in text.splitlines() if l.strip()]
    sections: List[Section] = []
    current_role = "intro"
    current_text = []

    def flush(role, buf):
        if buf:
            sections.append(Section(role, "\n".join(buf)))

    for line in lines:
        lower = line.lower()
        if "question-and-answer" in lower or "q&a" in lower:
            flush(current_role, current_text)
            current_role, current_text = "qa", []
        elif any(k in lower for k in ["revenue", "margin", "eps", "growth"]) and current_role == "intro":
            flush(current_role, current_text)
            current_role, current_text = "kpi_commentary", []
        elif any(k in lower for k in ["guidance", "outlook", "we expect", "we anticipate"]) and current_role != "qa":
            flush(current_role, current_text)
            current_role, current_text = "guidance", []
        current_text.append(line)

    flush(current_role, current_text)
    return sections


def sectionize_performance_report(text: str) -> List[Section]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    sections: List[Section] = []
    for p in paras:
        lower = p.lower()
        if any(k in lower for k in ["executive summary", "overview"]):
            role = "executive_summary"
        elif any(k in lower for k in ["kpi", "metric", "dashboard"]):
            role = "metrics"
        elif any(k in lower for k in ["driver", "root cause", "due to", "because"]):
            role = "drivers"
        elif any(k in lower for k in ["action item", "next steps"]):
            role = "action_items"
        else:
            role = "body"
        sections.append(Section(role, p))
    return sections


def sectionize_internal_memo(text: str) -> List[Section]:
    # Simple: first paragraph summary, rest body
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return []
    sections = [Section("summary", paras[0])]
    if len(paras) > 1:
        sections.append(Section("body", "\n\n".join(paras[1:])))
    return sections


def sectionize_strategy_deck(text: str) -> List[Section]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    sections: List[Section] = []
    for p in paras:
        lower = p.lower()
        if any(k in lower for k in ["vision", "mission", "strategy"]):
            role = "vision"
        elif any(k in lower for k in ["initiative", "pillar", "roadmap"]):
            role = "initiatives"
        elif any(k in lower for k in ["financials", "targets", "kpi"]):
            role = "financials"
        else:
            role = "body"
        sections.append(Section(role, p))
    return sections


def sectionize_technical_report(text: str) -> List[Section]:
    lines = text.splitlines()
    sections: List[Section] = []
    current_role = "body"
    current_text = []

    def flush(role, buf):
        if buf:
            sections.append(Section(role, "\n".join(buf)))

    for line in lines:
        lower = line.lower()
        if "methodology" in lower or "methods" in lower:
            flush(current_role, current_text)
            current_role, current_text = "methodology", []
        elif "results" in lower:
            flush(current_role, current_text)
            current_role, current_text = "results", []
        elif "validation" in lower or "model performance" in lower:
            flush(current_role, current_text)
            current_role, current_text = "validation", []
        elif "business impact" in lower or "impact on" in lower:
            flush(current_role, current_text)
            current_role, current_text = "impact", []
        current_text.append(line)

    flush(current_role, current_text)
    return sections


def sectionize(text: str, doc_type: str) -> List[Section]:
    if doc_type == "EARNINGS_CALL_TRANSCRIPT":
        return sectionize_earnings_call(text)
    if doc_type == "PERFORMANCE_REPORT":
        return sectionize_performance_report(text)
    if doc_type == "STRATEGY_DECK":
        return sectionize_strategy_deck(text)
    if doc_type == "TECHNICAL_REPORT":
        return sectionize_technical_report(text)
    # INTERNAL_MEMO, POLICY_OR_RISK_DOC, fallback
    return sectionize_internal_memo(text)

def ingest_and_classify(file_path: str) -> DocumentMF:
    loaded = load_file(file_path)
    raw_text = to_full_text(loaded)
    chunks = chunk_str(raw_text)
    sample_text = " ".join(chunks[:3])

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
        metadata={"original_path": file_path},
    )
