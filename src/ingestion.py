import os
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml
from nltk.tokenize import sent_tokenize

# Imports
from classifier_knn import KNNClassifier
from utils.process import load_file

# --- CONFIG ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)
    CHUNK_TOKEN_LIMIT = CONFIG.get("agent", {}).get("chunk_token_limit", 256)
else:
    CHUNK_TOKEN_LIMIT = 256

# --- GLOBAL VARS (LAZY) ---
_classifier_instance = None

def get_classifier():
    """
    Only load the classifier when we specifically ask for it.
    This prevents the script from hanging during 'import'.
    """
    global _classifier_instance
    if _classifier_instance is None:
        ref_dir = os.path.join(os.path.dirname(__file__), "..", "reference_docs")
        print(f"⚡ Initializing KNN Classifier from {ref_dir}...")
        _classifier_instance = KNNClassifier(ref_dir)
    return _classifier_instance

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

# --- UTILS ---
def to_full_text(loaded_file) -> str:
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

# --- CLASSIFICATION ---
def classify_document_type(sample_text: str, filename: str = "") -> str:
    text_lower = sample_text.lower()
    
    # LEVEL 1: Explicit Rules
    if "operator:" in text_lower and "question-and-answer" in text_lower:
        return "EARNINGS_CALL_TRANSCRIPT"
    
    # LEVEL 2: Semantic Similarity (KNN)
    # CALL THE LAZY LOADER HERE
    clf = get_classifier() 
    label, score = clf.classify(sample_text)
    
    print(f"DEBUG: KNN matched '{label}' (Confidence: {score:.2f})")
    
    if score > 0.45:
        return label
        
    # LEVEL 3: Fallback
    if filename.endswith(".html"):
        return "NEWS_ARTICLE"
    return "INTERNAL_MEMO"

# --- SECTIONING ---
def sectionize_earnings_call(text: str) -> List[Section]:
    lines = [l for l in text.splitlines() if l.strip()]
    sections, current_role, current_text = [], "intro", []
    def flush(role, buf):
        if buf: sections.append(Section(role, "\n".join(buf)))

    for line in lines:
        lower = line.lower()
        if "question-and-answer" in lower or "q&a" in lower:
            flush(current_role, current_text); current_role, current_text = "qa", []
        elif any(k in lower for k in ["revenue", "margin", "growth"]) and current_role == "intro":
            flush(current_role, current_text); current_role, current_text = "kpi_commentary", []
        elif any(k in lower for k in ["guidance", "outlook"]) and current_role != "qa":
            flush(current_role, current_text); current_role, current_text = "guidance", []
        current_text.append(line)
    flush(current_role, current_text)
    return sections

def sectionize_internal_memo(text: str) -> List[Section]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras: return []
    sections = [Section("summary", paras[0])]
    if len(paras) > 1: sections.append(Section("body", "\n\n".join(paras[1:])))
    return sections

def sectionize(text: str, doc_type: str) -> List[Section]:
    if doc_type == "EARNINGS_CALL_TRANSCRIPT": return sectionize_earnings_call(text)
    return sectionize_internal_memo(text)

# --- MAIN INGESTION FUNCTION ---
def ingest_and_classify(file_path: str) -> DocumentMF:
    print("   ...Loading file...")
    loaded = load_file(file_path)
    raw_text = to_full_text(loaded)
    
    print("   ...Chunking text...")
    chunks = chunk_str(raw_text)
    
    # Pass filename correctly
    sample_text = " ".join(chunks[:4])
    
    print("   ...Classifying document type...")
    doc_type = classify_document_type(sample_text, filename=file_path)
    
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