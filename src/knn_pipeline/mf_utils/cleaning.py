import re

def safe_compact(text: str) -> str:
    """
    Standardizes text for BOTH Ingestion and Classification.
    1. Fixes windows newlines
    2. Removes non-printable characters 
    3. Collapses excessive whitespace
    """
    if not text: return ""
    
    # 1. Standardize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # 2. Remove non-printable characters (except newlines/tabs)
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)

    # 3. Collapse multiple spaces/tabs to one
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 4. Collapse excessive newlines (max 2)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()