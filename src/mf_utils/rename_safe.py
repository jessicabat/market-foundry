from __future__ import annotations
from pathlib import Path
import re

def safe_filename(name: str) -> str:
    # keep letters, numbers, dot, dash, underscore; replace everything else with underscore
    name = name.strip()
    name = re.sub(r"\s+", "_", name)                 # spaces -> _
    name = re.sub(r"[^\w\.\-]+", "_", name)          # other junk -> _
    name = re.sub(r"_+", "_", name)                  # collapse ____
    return name

def rename_to_safe(path: str | Path) -> Path:
    p = Path(path)
    safe = safe_filename(p.name)
    new_path = p.with_name(safe)

    if new_path == p:
        return p

    # Avoid overwriting: if exists, append _1, _2, ...
    if new_path.exists():
        stem, suf = new_path.stem, new_path.suffix
        i = 1
        while True:
            candidate = new_path.with_name(f"{stem}_{i}{suf}")
            if not candidate.exists():
                new_path = candidate
                break
            i += 1

    p.rename(new_path)
    return new_path
