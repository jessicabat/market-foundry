# src/pipeline.py
import argparse
import os
import yaml
import json
from ingest import ingest_and_classify  # your existing classifier
from oneke_runner import run_oneke_extraction  # wrapper around OneKE
from graph_builder import push_to_neo4j      # your neo4j pusher

# Map doc types to YAML config paths
SCHEMA_MAP = {
    "EARNINGS_CALL_TRANSCRIPT": "schemas/earnings_call.yaml",
    "SEC_FILING": "schemas/sec_filing.yaml",
    "PRESS_RELEASE": "schemas/press_release.yaml",
    "STRATEGY_DECK": "schemas/strategy_deck.yaml",  # fallbacks for now
    "PERFORMANCE_REPORT": "schemas/performance_report.yaml",
    "INTERNAL_MEMO": "schemas/internal_memo.yaml",
    "TECHNICAL_REPORT": "schemas/technical_report.yaml",
    "UNKNOWN": "schemas/fallback_general.yaml"  # low confidence fallback
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    args = parser.parse_args()

    for filename in os.listdir(args.input_dir):
        if filename.startswith("."): continue # skip hidden
        filepath = os.path.join(args.input_dir, filename)
        
        # 1. Ingest
        print(f"Processing {filename}...")
        doc = ingest_and_classify(filepath)
        print(f"  -> Type: {doc.document_type}")

        # 2. Pick Schema
        schema_path = SCHEMA_MAP.get(doc.document_type, SCHEMA_MAP["UNKNOWN"])
        
        # 3. Extract (Passing raw text directly)
        try:
            triples = run_oneke_extraction(doc.raw_text, schema_path)
            print(f"  -> Extracted {len(triples)} triples")
            
            # 4. Push to Graph
            if triples:
                # Add doc metadata to each triple for lineage
                for t in triples:
                    t['source_doc'] = filename
                    t['doc_type'] = doc.document_type
                
                push_to_neo4j(triples)
                print("  -> Pushed to Neo4j")
                
        except Exception as e:
            print(f"  -> Extraction failed: {e}")

if __name__ == "__main__":
    main()