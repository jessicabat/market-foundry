import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Tuple

# Global constant, but we WON'T load the model here to prevent import-time hangs
MODEL_NAME = 'all-MiniLM-L6-v2'
_embedder = None

def get_embedder():
    """Singleton to load the model only when needed (Lazy Loading)."""
    global _embedder
    if _embedder is None:
        print(f"⏳ Loading embedding model '{MODEL_NAME}'... (This may take a moment first time)")
        _embedder = SentenceTransformer(MODEL_NAME)
        print("✅ Model loaded.")
    return _embedder

class KNNClassifier:
    def __init__(self, reference_dir: str):
        self.reference_dir = reference_dir
        self.embeddings = []
        self.labels = []
        # Load references immediately upon class creation
        self._load_references()

    def _load_references(self):
        """
        Reads .txt files from reference_docs/ folder to build the 'Gold Standard' memory.
        """
        if not os.path.exists(self.reference_dir):
            print(f"⚠️ WARNING: Reference directory '{self.reference_dir}' not found.")
            return

        print(f"📂 Loading reference docs from {self.reference_dir}...")
        embedder = get_embedder()
        
        count = 0
        # Walk through subfolders
        for label in os.listdir(self.reference_dir):
            label_dir = os.path.join(self.reference_dir, label)
            if not os.path.isdir(label_dir):
                continue
            
            # Read each .txt file
            for fname in os.listdir(label_dir):
                if not fname.endswith(".txt"): 
                    continue
                    
                fpath = os.path.join(label_dir, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        # Optimization: Read only first 1000 chars directly
                        text = f.read(1000) 
                        if not text.strip(): continue
                        
                        # Convert text -> Vector
                        vector = embedder.encode(text)
                        self.embeddings.append(vector)
                        self.labels.append(label)
                        count += 1
                except Exception as e:
                    print(f"   Skipping {fname}: {e}")
        
        # Convert to numpy for speed
        if self.embeddings:
            self.embeddings = np.array(self.embeddings)
            print(f"✅ Classifier ready with {count} reference examples.")
        else:
            print("⚠️ No reference documents found! Classifier will default to INTERNAL_MEMO.")

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