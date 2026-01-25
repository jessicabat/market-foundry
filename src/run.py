import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion import ingest_and_classify

def main():
    parser = argparse.ArgumentParser(description="MarketFoundry: Ingestion & Classification Pipeline")
    parser.add_argument("--file", type=str, required=True, help="Path to the input file")
    
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ Error: File not found at {args.file}")
        sys.exit(1)

    print(f"🚀 Starting ingestion for: {args.file}...")

    try:
        doc = ingest_and_classify(args.file)

        print("\n" + "="*50)
        print(f"📄 DOCUMENT REPORT")
        print("="*50)
        print(f"🆔  ID:            {doc.id}")
        print(f"🏷️   Type:          {doc.document_type}")
        print(f"🧩  Total Chunks:  {len(doc.chunks)}") # FIXED: was doc.raw_chunks
        print("-" * 50)
        
        print(f"📑  DETECTED SECTIONS ({len(doc.sections)})")
        for i, sec in enumerate(doc.sections):
            preview = sec.text[:80].replace("\n", " ") + "..."
            print(f"   {i+1}. [{sec.role.upper()}] - {len(sec.text)} chars")
            print(f"      \"{preview}\"")
        print("="*50 + "\n")

    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()