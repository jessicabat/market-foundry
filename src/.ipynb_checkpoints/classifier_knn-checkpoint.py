import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Tuple

from utils.text_io import read_text_robust  # <-- ADD

MODEL_NAME = 'all-MiniLM-L6-v2'
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        print(f"⏳ Loading embedding model '{MODEL_NAME}'... (This may take a moment first time)")
        _embedder = SentenceTransformer(MODEL_NAME)
        print("✅ Model loaded.")
    return _embedder

class KNNClassifier:
    def __init__(self, reference_dir: str):
        self.reference_dir = os.path.abspath(reference_dir)  # normalize
        self.embeddings = []
        self.labels = []
        self._load_references()

    def _load_references(self):
        if not os.path.exists(self.reference_dir):
            print(f"⚠️ WARNING: Reference directory '{self.reference_dir}' not found.")
            return

        print(f"📂 Loading reference docs from {self.reference_dir}...")
        embedder = get_embedder()

        candidates = 0
        count = 0

        for label in os.listdir(self.reference_dir):
            label_dir = os.path.join(self.reference_dir, label)
            if not os.path.isdir(label_dir):
                continue

            for fname in os.listdir(label_dir):
                if not fname.lower().endswith(".txt"):   # <-- make robust
                    continue

                candidates += 1
                fpath = os.path.join(label_dir, fname)
                try:
                    text, enc = read_text_robust(fpath, max_bytes=100000)
                    text = text[:1000]
                    if not text.strip():
                        continue

                    vector = embedder.encode(text)
                    self.embeddings.append(vector)
                    self.labels.append(label)
                    count += 1

                except Exception as e:
                    print(f"   Skipping {fname}: {e}")

        if self.embeddings:
            self.embeddings = np.array(self.embeddings)
            print(f"✅ Classifier ready with {count}/{candidates} reference examples.")
        else:
            print(f"⚠️ No reference documents found! (0/{candidates} usable) Defaulting to INTERNAL_MEMO.")


    def classify(self, text: str) -> Tuple[str, float]:
        if len(self.embeddings) == 0:
            return "INTERNAL_MEMO", 0.0

        embedder = get_embedder()
        # 1. Vectorize input
        query_vec = embedder.encode(text[:1000])
        
        # 2. Cosine Similarity
        norm_query = np.linalg.norm(query_vec)
        norm_refs = np.linalg.norm(self.embeddings, axis=1)
        
        # Avoid division by zero
        if norm_query == 0: return "INTERNAL_MEMO", 0.0
        
        similarities = np.dot(self.embeddings, query_vec) / (norm_refs * norm_query)
        
        # 3. Best Match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_label = self.labels[best_idx]

        return best_label, float(best_score)