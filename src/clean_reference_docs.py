import os
import argparse
from pathlib import Path

from utils.text_io import read_text_robust
from utils.text_clean import normalize_text
from utils.html_clean import drop_nav_blocks

# Reuse your existing cleaners
from utils.html_clean import (
    strip_html_boilerplate,
    dedupe_repeated_paragraphs,
)
# If you added drop_nav_blocks earlier, import it too:
# from utils.html_clean import drop_nav_blocks

def safe_compact(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = normalize_text(text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()

def clean_reference_text(raw: str) -> str:
    """
    Reference docs are .txt, but they may contain:
    - pasted IR menus
    - boilerplate
    - duplicated paragraphs
    """
    original = safe_compact(raw)
    if not original:
        return ""

    cleaned = strip_html_boilerplate(original, repeat_threshold=3)
    cleaned = safe_compact(cleaned)

    # If you have block-level nav removal, enable it:
    cleaned = drop_nav_blocks(cleaned)
    cleaned = safe_compact(cleaned)

    cleaned = dedupe_repeated_paragraphs(cleaned, min_len=200)
    cleaned = safe_compact(cleaned)

    # Guardrail: if cleaning nuked too much, back off
    if len(cleaned) < max(200, int(0.10 * len(original))):
        soft = strip_html_boilerplate(original, repeat_threshold=5)
        soft = safe_compact(soft)
        soft = dedupe_repeated_paragraphs(soft, min_len=200)
        soft = safe_compact(soft)
        cleaned = max([cleaned, soft, original], key=len)

    return cleaned

def main():
    parser = argparse.ArgumentParser(description="One-time cleaner for reference_docs/*.txt")
    parser.add_argument("--in_dir", type=str, default="reference_docs", help="Input reference docs folder.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="reference_docs_clean",
        help="Output folder for cleaned references (recommended).",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Overwrite files in-place instead of writing to out_dir (dangerous).",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=200,
        help="Skip writing if cleaned text is shorter than this.",
    )

    args = parser.parse_args()
    in_dir = Path(args.in_dir).resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"❌ Input dir not found: {in_dir}")

    if args.in_place:
        out_dir = in_dir
    else:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0

    print(f"📂 Input:  {in_dir}")
    print(f"📁 Output: {out_dir} {'(in-place)' if args.in_place else '(mirror)'}")
    print("-" * 60)

    # Walk label subfolders
    for label_dir in sorted([p for p in in_dir.iterdir() if p.is_dir()]):
        label = label_dir.name
        txt_files = sorted(label_dir.glob("*.txt"))
        if not txt_files:
            continue

        # Mirror label dir
        target_label_dir = out_dir / label
        target_label_dir.mkdir(parents=True, exist_ok=True)

        for fpath in txt_files:
            try:
                raw, _enc = read_text_robust(str(fpath), max_bytes=2_000_000)
                raw = raw or ""
                cleaned = clean_reference_text(raw)

                if len(cleaned) < args.min_chars:
                    skipped += 1
                    print(f"⚠️  SKIP (too short) {label}/{fpath.name}  (cleaned={len(cleaned)})")
                    continue

                out_path = (target_label_dir / fpath.name) if not args.in_place else fpath

                # Write only if changed (optional speed)
                if out_path.exists() and out_path.read_text(errors="ignore").strip() == cleaned.strip():
                    skipped += 1
                    continue

                out_path.write_text(cleaned, encoding="utf-8")
                processed += 1

                delta = len(cleaned) - len(raw)
                print(f"✅ {label}/{fpath.name}  raw={len(raw):>7}  clean={len(cleaned):>7}  Δ={delta:>7}")

            except Exception as e:
                failed += 1
                print(f"❌ FAIL {label}/{fpath.name}: {e}")

    print("-" * 60)
    print(f"Done. processed={processed} skipped={skipped} failed={failed}")

if __name__ == "__main__":
    main()
