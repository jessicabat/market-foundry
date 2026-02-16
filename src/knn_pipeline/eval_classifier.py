import os
from pathlib import Path
from collections import Counter, defaultdict

from ingestion import ingest_and_classify

CLASSES = {"press_release", "earnings_call", "sec_filing"}

def normalize_label(s: str) -> str:
    return s.strip().lower()

def canonical_pred(label: str) -> str:
    l = normalize_label(label)
    if l in {"sec_filing", "sec filing", "sec_filing"}: return "sec_filing"
    if l in {"press_release", "press release", "press_release"}: return "press_release"
    if l in {"earnings_call", "earnings call", "earnings_call_transcript"}: return "earnings_call"
    return "other"


def main():
    eval_root = Path("FinancialPapers/eval")  # change if needed
    if not eval_root.exists():
        raise SystemExit(f"Eval folder not found: {eval_root.resolve()}")

    y_true = []
    y_pred = []
    per_class_counts = Counter()
    confusion = defaultdict(Counter)

    files = []
    for cls in CLASSES:
        cls_dir = eval_root / cls
        if cls_dir.exists():
            files.extend(list(cls_dir.rglob("*")))
    files = [p for p in files if p.is_file()]

    if not files:
        raise SystemExit("No eval files found under FinancialPapers/eval/<class_name>/")

    for p in files:
        true_label = normalize_label(p.parent.name)
        if true_label not in CLASSES:
            continue

        doc = ingest_and_classify(str(p))
        pred_label = canonical_pred(doc.document_type)

        y_true.append(true_label)
        y_pred.append(pred_label)

        per_class_counts[true_label] += 1
        confusion[true_label][pred_label] += 1

    # --- Accuracy ---
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total = len(y_true)
    acc = correct / total if total else 0.0

    # --- Precision/Recall/F1 per class ---
    # tp, fp, fn for each class
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
    weighted_f1 /= total

    # --- Print results ---
    print("\n" + "=" * 72)
    print("EVALUATION RESULTS")
    print("=" * 72)
    print(f"Total samples: {total}")
    print(f"Accuracy:      {acc:.3f}")
    print(f"Macro F1:      {macro_f1:.3f}")
    print(f"Weighted F1:   {weighted_f1:.3f}")
    print("-" * 72)
    print(f"{'Class':<24} {'Support':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    for cls in sorted(CLASSES):
        prec, rec, f1 = metrics[cls]
        print(f"{cls:<24} {per_class_counts[cls]:>8} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f}")

    # --- Confusion matrix ---
    print("\nCONFUSION MATRIX (rows=true, cols=pred)")
    header = ["true\\pred"] + sorted(CLASSES)
    print("".join(f"{h:>20}" for h in header))
    for tcls in sorted(CLASSES):
        row = [tcls]
        for pcl in sorted(CLASSES):
            row.append(str(confusion[tcls][pcl]))
        print("".join(f"{c:>20}" for c in row))
    print("=" * 72)

if __name__ == "__main__":
    main()
# python src/eval_classifier.py
