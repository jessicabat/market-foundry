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

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder

class KNNClassifier:
    def __init__(self, reference_dir: str):
        self.reference_dir = os.path.abspath(reference_dir) 
        self.embeddings = []
        self.labels = []
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
            return True
        except Exception:
            return False

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
                    text = _clean_text(text)
                    text = text[:1000]
                    if len(text) < 50: continue

                    vector = embedder.encode(text, normalize_embeddings=True)
                    
                    self.embeddings.append(vector)
                    self.labels.append(label)
                    count += 1
                except Exception as e:
                    print(f"   Skipping {fname}: {e}")

        if self.embeddings:
            self.embeddings = np.array(self.embeddings)
            with open(self.cache_path, "wb") as f:
                pickle.dump({"embeddings": self.embeddings, "labels": self.labels}, f)
            print(f"✅ Classifier ready with {count} reference examples. Cache saved.")
        else:
            print("⚠️ No reference documents found! Defaulting to INTERNAL_MEMO.")


    def classify(self, text: str) -> Tuple[str, float]:
        if len(self.embeddings) == 0:
            return "INTERNAL_MEMO", 0.0

        embedder = get_embedder()
        clean_input = _clean_text(text[:1500])
        query_vec = embedder.encode(clean_input[:1000], normalize_embeddings=True)
        
        # norm_query = np.linalg.norm(query_vec)
        # norm_refs = np.linalg.norm(self.embeddings, axis=1)
        # if norm_query == 0: return "INTERNAL_MEMO", 0.0
        # similarities = np.dot(self.embeddings, query_vec) / (norm_refs * norm_query)

        similarities = np.dot(self.embeddings, query_vec)
        k = min(5, len(self.labels))
        top_k_idx = np.argsort(similarities)[-k:]

        weighted_votes = defaultdict(float)

        for i in top_k_idx:
            label = self.labels[i]
            weighted_votes[label] += similarities[i]

        # label w/ highest accumulated similarity points
        best_label = max(weighted_votes, key=weighted_votes.get)
        winning_scores = [similarities[i] for i in top_k_idx if self.labels[i] == best_label]
        best_score = float(np.mean(winning_scores))

        return best_label, best_score
        # best_idx = np.argmax(similarities)
        # return self.labels[best_idx], float(similarities[best_idx])