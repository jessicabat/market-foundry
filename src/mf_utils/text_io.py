from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

def read_text_robust(path: str | Path, max_bytes: Optional[int] = None) -> Tuple[str, str]:
    """
    Returns (text, encoding_used). Never raises UnicodeDecodeError.
    """
    p = Path(path)
    data = p.read_bytes()
    if max_bytes is not None:
        data = data[:max_bytes]

    # Fast path
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            pass

    # Best-effort detector if available
    try:
        from charset_normalizer import from_bytes  # pip install charset-normalizer
        best = from_bytes(data).best()
        if best and best.encoding:
            return str(best), best.encoding
    except Exception:
        pass

    # Practical fallbacks
    for enc in ("cp1252", "latin-1"):
        try:
            return data.decode(enc), enc
        except Exception:
            pass

    return data.decode("utf-8", errors="replace"), "utf-8(replace)"
