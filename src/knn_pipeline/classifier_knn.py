from __future__ import annotations
import os
import re
import pickle
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import Optional, Tuple

MODEL_NAME = 'all-MiniLM-L12-v2'
_embedder = None

HEAD_SNIPPET_CHARS = 1400
TAIL_SNIPPET_CHARS = 900
MAX_SNIPPET_CHARS = 2600

def read_text_robust(path: str | Path, max_bytes: Optional[int] = None) -> Tuple[str, str]:
    """Returns (text, encoding_used). Never raises UnicodeDecodeError."""
    p = Path(path)
    data = p.read_bytes()
    if max_bytes is not None:
        data = data[:max_bytes]

    for enc in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            pass

    try:
        from charset_normalizer import from_bytes  
        best = from_bytes(data).best()
        if best and best.encoding:
            return str(best), best.encoding
    except Exception:
        pass

    for enc in ("cp1252", "latin-1"):
        try:
            return data.decode(enc), enc
        except Exception:
            pass

    return data.decode("utf-8", errors="replace"), "utf-8(replace)"

def _clean_text(text: str) -> str:
    if not text: return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def build_managed_snippet(
    text: str,
    head_chars: int = HEAD_SNIPPET_CHARS,
    tail_chars: int = TAIL_SNIPPET_CHARS,
    max_chars: int = MAX_SNIPPET_CHARS,
) -> str:
    clean_text = _clean_text(text)
    if len(clean_text) <= max_chars:
        return clean_text

    head = clean_text[:head_chars].rstrip()
    tail = clean_text[-tail_chars:].lstrip()

    if not tail or head.endswith(tail[:120]):
        return clean_text[:max_chars]

    snippet = f"{head}\n\n[...]\n\n{tail}"
    return snippet[: max_chars + len("\n\n[...]\n\n")]

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(
            MODEL_NAME,
            tokenizer_kwargs={'clean_up_tokenization_spaces': True}
            )
    return _embedder

class KNNClassifier:
    def __init__(self, reference_dir: str):
        self.reference_dir = os.path.abspath(reference_dir) 
        self.embeddings = []
        self.labels = []
        self.label_to_indices = defaultdict(list)
        self.cache_path = os.path.join(self.reference_dir, "embeddings_cache.pkl")
        
        # loading from cache first (INSTANT START)
        if self._load_from_cache():
            print(f"⚡ Loaded {len(self.labels)} reference docs from cache.")
        else:
            self._build_references()

    def _load_from_cache(self) -> bool:
        if not os.path.exists(self.cache_path): return False
        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
                self.embeddings = data["embeddings"]
                self.labels = data["labels"]
            self._rebuild_label_index()
            return True
        except Exception:
            return False

    def _rebuild_label_index(self):
        self.label_to_indices = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.label_to_indices[label].append(index)

    def _build_references(self):
        if not os.path.exists(self.reference_dir):
            print(f"⚠️ WARNING: Reference directory '{self.reference_dir}' not found.")
            return

        print(f"📂 Building reference cache from {self.reference_dir}...")
        embedder = get_embedder()
        count = 0

        for label in os.listdir(self.reference_dir):
            label_dir = os.path.join(self.reference_dir, label)
            if not os.path.isdir(label_dir): continue

            for fname in os.listdir(label_dir):
                if not fname.lower().endswith(".txt"): continue

                fpath = os.path.join(label_dir, fname)
                try:
                    text, enc = read_text_robust(fpath, max_bytes=100000)
                    text = build_managed_snippet(text)
                    if len(text) < 50: continue

                    vector = embedder.encode(text, normalize_embeddings=True)
                    
                    self.embeddings.append(vector)
                    self.labels.append(label)
                    count += 1
                except Exception as e:
                    print(f"   Skipping {fname}: {e}")

        if self.embeddings:
            self.embeddings = np.array(self.embeddings)
            self._rebuild_label_index()
            with open(self.cache_path, "wb") as f:
                pickle.dump({"embeddings": self.embeddings, "labels": self.labels}, f)
            print(f"✅ Classifier ready with {count} reference examples. Cache saved.")
        else:
            print("⚠️ No reference documents found! Defaulting to INTERNAL_MEMO.")


    def _score_from_query_vectors(self, query_vectors, per_label_top_k: int = 3) -> list[dict[str, float]]:
        if len(self.embeddings) == 0:
            return [{} for _ in range(len(query_vectors))]

        all_label_scores = []
        similarities_matrix = np.matmul(query_vectors, self.embeddings.T)
        for similarities in similarities_matrix:
            label_scores = {}
            for label, indices in self.label_to_indices.items():
                per_label_scores = similarities[indices]
                top_k = min(per_label_top_k, len(per_label_scores))
                if top_k == 0:
                    continue
                top_scores = np.partition(per_label_scores, -top_k)[-top_k:]
                label_scores[label] = float(np.mean(top_scores))
            all_label_scores.append(label_scores)
        return all_label_scores

    def score_labels_batch(self, texts: list[str], per_label_top_k: int = 3) -> list[dict[str, float]]:
        if len(self.embeddings) == 0:
            return [{} for _ in texts]

        embedder = get_embedder()
        snippets = [build_managed_snippet(text) for text in texts]
        query_vectors = embedder.encode(snippets, normalize_embeddings=True, batch_size=32)
        return self._score_from_query_vectors(query_vectors, per_label_top_k=per_label_top_k)

    def score_labels(self, text: str, per_label_top_k: int = 3) -> dict[str, float]:
        return self.score_labels_batch([text], per_label_top_k=per_label_top_k)[0]


    def classify(self, text: str) -> Tuple[str, float]:
        label_scores = self.score_labels(text)
        if not label_scores:
            return "INTERNAL_MEMO", 0.0

        best_label = max(label_scores, key=label_scores.get)
        return best_label, float(label_scores[best_label])