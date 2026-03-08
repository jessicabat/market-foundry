import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

src_root = os.path.join(project_root, "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from src.utils.document_classification import (
    load_files, extract_text, clean_texts, 
    load_tfidf_model, load_tfidf_vectorizer, classify_document_types
)

CLASSES = {"press_release", "earnings_call", "sec_filing", "news_article", "research_paper"}

def normalize_label(s: str) -> str:
    return s.strip().lower()

def canonical_pred(label: str) -> str:
    """Maps both folder names and model output strings to standard snake_case."""
    l = normalize_label(label)
    if l in {"sec_filing", "sec filing", "sec filing"}: return "sec_filing"
    if l in {"press_release", "press release"}: return "press_release"
    if l in {"earnings_call", "earnings call", "earnings_call_transcript", "earnings call transcript"}: return "earnings_call"
    if l in {"news_article", "news article", "news", "article"}: return "news_article"
    if l in {"research_paper", "research paper", "research", "paper"}: return "research_paper"
    return "other"

def main():
    eval_root = Path("Papers/eval")  # Change if your eval folder is named differently
    if not eval_root.exists():
        raise SystemExit(f"Eval folder not found: {eval_root.resolve()}")

    # 1. Gather all file paths from the class folders
    file_paths = []
    for cls in CLASSES:
        cls_dir = eval_root / cls
        if cls_dir.exists():
            file_paths.extend([str(p) for p in cls_dir.rglob("*") if p.is_file()])

    if not file_paths:
        raise SystemExit("No eval files found under Papers/eval/<class_name>/")

    print(f"🔍 Found {len(file_paths)} files for evaluation. Starting batch processing...")

    # 2. Run your pipeline on the entire batch
    loaded_files = load_files(file_paths)
    texts = extract_text(loaded_files)
    cleaned_texts = clean_texts(texts)

    # Load models
    model_path = os.path.join(src_root, "models", "Document_Classifier.joblib")
    vectorizer_path = os.path.join(src_root, "models", "TFIDF_Vectorizer.joblib")
    
    model = load_tfidf_model(model_path)
    vectorizer = load_tfidf_vectorizer(vectorizer_path)

    # Classify the batch
    print("🧠 Classifying documents...")
    classifications = classify_document_types(model, vectorizer, cleaned_texts)

    # 3. Grade the results
    y_true = []
    y_pred = []
    per_class_counts = Counter()
    confusion = defaultdict(Counter)

    for source_path, pred_str in classifications.items():
        # The true label is the name of the folder the file was inside
        true_label = canonical_pred(Path(source_path).parent.name)
        pred_label = canonical_pred(pred_str)

        if true_label not in CLASSES:
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)

        per_class_counts[true_label] += 1
        confusion[true_label][pred_label] += 1

    # --- Accuracy ---
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total = len(y_true)
    acc = correct / total if total else 0.0

    # --- Precision/Recall/F1 per class ---
    metrics = {}
    for cls in sorted(CLASSES):
        tp = sum((t == cls and p == cls) for t, p in zip(y_true, y_pred))
        fp = sum((t != cls and p == cls) for t, p in zip(y_true, y_pred))
        fn = sum((t == cls and p != cls) for t, p in zip(y_true, y_pred))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        metrics[cls] = (precision, recall, f1)

    macro_f1 = sum(m[2] for m in metrics.values()) / len(CLASSES)

    weighted_f1 = 0.0
    for cls in CLASSES:
        support = per_class_counts[cls]
        weighted_f1 += support * metrics[cls][2]
    weighted_f1 /= total if total else 1

    # --- Print results ---
    print("\n" + "=" * 76)
    print("EVALUATION RESULTS")
    print("=" * 76)
    print(f"Total samples: {total}")
    print(f"Accuracy:      {acc:.3f}")
    print(f"Macro F1:      {macro_f1:.3f}")
    print(f"Weighted F1:   {weighted_f1:.3f}")
    print("-" * 76)
    print(f"{'Class':<24} {'Support':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    for cls in sorted(CLASSES):
        prec, rec, f1 = metrics[cls]
        print(f"{cls:<24} {per_class_counts[cls]:>8} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f}")

    # --- Confusion matrix ---
    print("\nCONFUSION MATRIX (rows=true, cols=pred)")
    header = ["true\\pred"] + sorted(CLASSES)
    print("".join(f"{h:>15}" for h in header))
    for tcls in sorted(CLASSES):
        row = [tcls]
        for pcl in sorted(CLASSES):
            row.append(str(confusion[tcls][pcl]))
        print("".join(f"{c:>15}" for c in row))
    print("=" * 76)

if __name__ == "__main__":
    main()