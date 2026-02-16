import argparse
import os
import sys
from pathlib import Path
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion import ingest_and_classify


SUPPORTED_EXTS = {".pdf", ".html", ".htm", ".docx", ".txt", ".json"}


def iter_supported_files(folder: Path):
    """Recursively yield supported files from a folder."""
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def print_summary(counter: Counter, total: int, failed: int):
    """Pretty-print a summary table of doc types."""
    print("\n" + "=" * 60)
    print("📊 PIPELINE SUMMARY")
    print("=" * 60)

    if total == 0:
        print("No files processed.")
        print("=" * 60 + "\n")
        return

    # Table header
    print(f"{'Document Type':<28} {'Count':>8} {'Pct':>8}")
    print("-" * 60)

    # Sort by count desc then name
    for doc_type, count in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        pct = (count / total) * 100
        print(f"{doc_type:<28} {count:>8} {pct:>7.1f}%")

    print("-" * 60)
    print(f"{'TOTAL':<28} {total:>8} {100:>7.1f}%")
    if failed:
        print(f"{'FAILED':<28} {failed:>8}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="MarketFoundry: Batch Ingestion & Classification (Folder Mode)"
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to a folder containing documents (processed recursively).",
    )

    args = parser.parse_args()
    folder = Path(args.folder)

    if not folder.exists() or not folder.is_dir():
        print(f"❌ Error: Folder not found or not a directory: {folder}")
        sys.exit(1)

    files = list(iter_supported_files(folder))
    if not files:
        print(f"⚠️ No supported files found in: {folder}")
        print(f"   Supported extensions: {', '.join(sorted(SUPPORTED_EXTS))}")
        sys.exit(0)

    print(f"📁 Folder: {folder.resolve()}")
    print(f"📄 Found {len(files)} supported file(s) (recursive).")
    print("-" * 60)

    doc_type_counts = Counter()
    failed = 0

    for i, path in enumerate(files, start=1):
        rel = path.relative_to(folder)
        print(f"\n[{i}/{len(files)}] 🚀 Ingesting: {rel}")

        try:
            doc = ingest_and_classify(str(path))
            doc_type_counts[doc.document_type] += 1

            # Minimal per-file report (keep logs manageable)
            print(f"   ✅ Type: {doc.document_type}")
            print(f"   🧩 Chunks: {len(doc.chunks)}")
            print(f"   📑 Sections: {len(doc.sections)}")

        except Exception as e:
            failed += 1
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    total_success = sum(doc_type_counts.values())
    print_summary(doc_type_counts, total_success, failed)


if __name__ == "__main__":
    main()
