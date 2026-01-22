# import argparse
# import os
# import yaml
# from nltk.tokenize import sent_tokenize
# from utils import *
# from utils.process import load_file

# def main():
#     # Create command-line argument parser
#     parser = argparse.ArgumentParser(description='Run the extraction framefork.')
#     parser.add_argument('--file', type=str, required=True,
#                         help='Path to the input file.')

#     # Parse command-line arguments
#     args = parser.parse_args()
#     # Load and process the file
#     loaded_file = load_file(args.file)
    
#     print(loaded_file)


# if __name__ == "__main__":
#     main()

# src/run.py
import argparse
import os
from ingestion import ingest_and_classify

def main():
    parser = argparse.ArgumentParser(description="Run MarketFoundry ingestion + classification.")
    parser.add_argument("--file", type=str, required=True, help="Path to the input file.")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"File not found: {args.file}")

    doc = ingest_and_classify(args.file)

    print("Document ID   :", doc.id)
    print("File path     :", doc.file_path)
    print("Doc type      :", doc.document_type)
    print("Num chunks    :", len(doc.chunks))
    print("Sections:")
    for s in doc.sections:
        print(f"  - role={s.role}, len={len(s.text)}")

if __name__ == "__main__":
    main()
