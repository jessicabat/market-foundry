# tools/auto_convert_refs.py
import os
import sys
from pathlib import Path

# Add src to path so we can import your existing tools
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src/knn_pipeline'))

from utils.process import load_file
from utils.cleaning import safe_compact

# Define where you will drop your raw downloads, and where the .txt files should go
RAW_DIR = Path("raw_references")
REF_DIR = Path("reference_docs_clean")

def to_full_text(loaded_file) -> str:
    """Flatten LangChain docs into one string."""
    if isinstance(loaded_file, list):
        return "\n".join(getattr(doc, "page_content", str(doc)) for doc in loaded_file)
    return getattr(loaded_file, "page_content", str(loaded_file))

def main():
    if not RAW_DIR.exists():
        print(f"Creating {RAW_DIR}. Drop your raw PDFs, HTML, and DOCX files in subfolders here.")
        RAW_DIR.mkdir()
        return

    # Walk through the raw_references folder
    for category_dir in RAW_DIR.iterdir():
        if not category_dir.is_dir():
            continue
            
        category_name = category_dir.name # e.g., "RESEARCH_PAPER"
        output_dir = REF_DIR / category_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📂 Processing category: {category_name}")
        
        for file_path in category_dir.iterdir():
            if file_path.suffix.lower() not in ['.pdf', '.html', '.htm', '.docx', '.json']:
                continue
                
            out_filename = file_path.stem + ".txt"
            out_filepath = output_dir / out_filename
            
            # Skip if we already converted it
            if out_filepath.exists():
                continue
                
            print(f"   ⚙️ Converting: {file_path.name}...")
            try:
                # 1. Load using your existing OneKE/LangChain loader
                loaded = load_file(str(file_path))
                raw_text = to_full_text(loaded)
                
                # 2. Clean it using your new standard cleaner
                clean_text = safe_compact(raw_text)
                
                # 3. Save it as a .txt file
                with open(out_filepath, "w", encoding="utf-8") as f:
                    f.write(clean_text)
                    
                print(f"   ✅ Saved to: {out_filepath}")
            except Exception as e:
                print(f"   ❌ Failed to convert {file_path.name}: {e}")

if __name__ == "__main__":
    main()