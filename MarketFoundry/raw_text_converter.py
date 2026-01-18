import os
import subprocess
import tempfile
import pdfplumber

from langchain_community.document_loaders import TextLoader, Docx2txtLoader, BSHTMLLoader, JSONLoader
from langchain_core.documents import Document


def file_name_simplifier(file_path):
    base_name = os.path.basename(file_path)
    name, _ = os.path.splitext(base_name)
    return name


def _pdfplumber_extract_text(pdf_path: str) -> tuple[str, list[str]]:
    """Extract text using pdfplumber; returns (raw_text, per_page_texts)."""
    per_page = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            per_page.append(page.extract_text() or "")
    raw_text = "\n".join(per_page).strip()
    return raw_text, per_page


def _looks_scanned(per_page_texts: list[str], min_avg_chars_per_page: int = 40, sample_pages: int = 3) -> bool:
    """
    Heuristic: if extracted text is near-empty for the first few pages, it's probably scanned.
    """
    if not per_page_texts:
        return True
    sample = per_page_texts[: min(sample_pages, len(per_page_texts))]
    avg_chars = sum(len(t.strip()) for t in sample) / max(1, len(sample))
    return avg_chars < min_avg_chars_per_page


def _run_ocrmypdf(input_pdf: str, output_pdf: str, force_ocr: bool = False) -> None:
    """
    Requires: ocrmypdf + system deps (tesseract, ghostscript).
    """
    cmd = [
        "ocrmypdf",
        "--deskew",
        "--rotate-pages",
        "--clean",
    ]
    if force_ocr:
        cmd.append("--force-ocr")
    cmd += [input_pdf, output_pdf]
    subprocess.run(cmd, check=True)


def pdf_to_raw_text(pdf_path: str, *, ocr_if_needed: bool = True) -> tuple[str, dict]:
    """
    Returns (raw_text, meta). If scan-based and ocrmypdf is available, OCRs then extracts again.
    """
    raw_text, per_page = _pdfplumber_extract_text(pdf_path)

    scanned = _looks_scanned(per_page)
    meta = {
        "source_path": pdf_path,
        "ocr_used": False,
        "scanned_detected": scanned,
    }

    if not (ocr_if_needed and scanned):
        return raw_text, meta

    # If scanned, OCR to a searchable PDF then re-extract.
    # We keep OCR output in a temp dir; if you want to persist it, write to your own path instead.
    with tempfile.TemporaryDirectory() as td:
        ocr_pdf = os.path.join(td, "searchable.pdf")

        try:
            _run_ocrmypdf(pdf_path, ocr_pdf, force_ocr=False)
        except FileNotFoundError:
            raise RuntimeError(
                "ocrmypdf is not installed/available on PATH. Install it (and tesseract + ghostscript) "
                "or disable ocr_if_needed."
            )
        except subprocess.CalledProcessError:
            # If OCR fails (rare), try forcing OCR (handles some weird PDFs with partial text layers)
            _run_ocrmypdf(pdf_path, ocr_pdf, force_ocr=True)

        raw_text2, per_page2 = _pdfplumber_extract_text(ocr_pdf)
        meta["ocr_used"] = True
        meta["scanned_detected_after_ocr"] = _looks_scanned(per_page2)

        return raw_text2, meta


def load_file(file_path: str):
    """
    For PDFs: uses pdfplumber + OCR fallback, returns a single LangChain Document with full raw text.
    For other types: uses standard LangChain loaders.
    """
    if file_path.lower().endswith(".pdf"):
        raw_text, meta = pdf_to_raw_text(file_path, ocr_if_needed=True)
        return [Document(page_content=raw_text, metadata=meta)]

    elif file_path.lower().endswith(".txt"):
        return TextLoader(file_path).load()

    elif file_path.lower().endswith(".docx"):
        return Docx2txtLoader(file_path).load()

    elif file_path.lower().endswith(".html"):
        return BSHTMLLoader(file_path).load()

    elif file_path.lower().endswith(".json"):
        return JSONLoader(file_path).load()

    else:
        raise ValueError("Unsupported file format")
