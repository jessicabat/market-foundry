# src/pipeline.py
import argparse
import os
import yaml
import json
from datetime import datetime  # <--- Added missing import

# Import your modules (adjust if necessary)
try:
    from ingestion import ingest_and_classify   # Renamed from 'ingest' based on previous convo
except ImportError:
    from ingest import ingest_and_classify      # Fallback to 'ingest' if filename differs

from oneke_runner import run_oneke_extraction
from graph_builder import push_to_neo4j

# ... [SCHEMA_MAP remains same] ...

def save_triples_to_file(triples, filename, output_dir="output/raw_triples"):
    # Ensure output directory is relative to where script is run (usually project root)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{filename}_triples.json")
    
    # Handle non-serializable objects just in case
    def json_serial(obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError (f"Type {type(obj)} not serializable")

    with open(out_path, 'w') as f:
        json.dump(triples, f, indent=2, default=json_serial)
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    args = parser.parse_args()
    
    all_extracted_data = [] 

    # Ensure input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    for filename in os.listdir(args.input_dir):
        if filename.startswith("."): continue 
        filepath = os.path.join(args.input_dir, filename)
        
        if not os.path.isfile(filepath): continue # Skip subdirectories

        # 1. Ingest
        print(f"Processing {filename}...")
        try:
            doc = ingest_and_classify(filepath)
            print(f"  -> Type: {doc.document_type}")
        except Exception as e:
            print(f"  -> Ingestion failed: {e}")
            continue

        # 2. Pick Schema
        schema_path = SCHEMA_MAP.get(doc.document_type, SCHEMA_MAP["UNKNOWN"])
        
        # 3. Extract
        try:
            triples = run_oneke_extraction(doc.raw_text, schema_path)
            print(f"  -> Extracted {len(triples)} triples")
            
            # 4. Save & Push
            if triples:
                # Save individual JSON
                save_triples_to_file(triples, filename)
                
                # Prepare Record
                doc_record = {
                    "doc_id": doc.id,
                    "filename": filename,
                    "doc_type": doc.document_type,
                    "triples": triples,
                    "processed_at": datetime.now().isoformat()
                }
                all_extracted_data.append(doc_record)
                
                # Push to Neo4j
                # Ensure metadata is serializable if push_to_neo4j expects dict
                push_to_neo4j(triples, doc_metadata=doc.metadata)
                print("  -> Pushed to Neo4j")
                
        except Exception as e:
            print(f"  -> Extraction/Push failed: {e}")

    # 5. Save Master JSON
    master_path = os.path.join(os.getcwd(), "output/knowledge_graph/master_graph_snapshot.json")
    os.makedirs(os.path.dirname(master_path), exist_ok=True)
    
    with open(master_path, 'w') as f:
        json.dump(all_extracted_data, f, indent=2)
    
    print(f"\n[Done] Processed {len(all_extracted_data)} documents.")
    print(f"       Master JSON saved to: {master_path}")

if __name__ == "__main__":
    main()
