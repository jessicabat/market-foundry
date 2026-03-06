""" Document Sectioning for src/run.py
Functions to run the document sectioning pipeline.
"""

import os
import pandas as pd
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, BSHTMLLoader, JSONLoader
from utils import *
from utils.document_classification import *

SECTION_HEADERS = {
    "overview": [
        "overview", "business overview", "company overview",
        "about the company", "item 1.", "item 1 ", "corporate overview",
        "company description", "business description"
    ],
    "financials": [
        "financial statements", "consolidated statements",
        "balance sheet", "income statement",
        "statement of operations", "cash flow", "item 8.",
        "statement of financial position", "statement of comprehensive income",
        "statements of cash flows", "financial position"
    ],
    "mdna": [
        "management's discussion", "md&a", "results of operations", 
        "item 7.", "management discussion and analysis",
        "operating performance", "financial performance", "performance analysis"
    ],
    "risk_factors": [
        "risk factors", "risks and uncertainties",
        "forward-looking statements", "item 1a.", "risk management",
        "potential risks", "business risks", "market risks"
    ],
    "notes": [
        "notes to financial statements", "accounting policies",
        "significant accounting", "footnotes", "note disclosures",
        "accounting standards", "financial note"
    ],
    "outlook": [
        "outlook", "guidance", "future outlook",
        "expectations", "forecast", "future prospects", "forward guidance"
    ],
    "legal": [
        "legal proceedings", "regulatory matters",
        "compliance", "litigation", "legal notices", "regulatory compliance",
        "legal issues", "court proceedings"
    ],
    "introduction": [
        "introduction", "executive summary", "summary",
        "background", "item 1.", "related work", "preface", "prologue", "abstract"
    ],
    "methodology": [
        "methodology", "approach", "data and methods",
        "research design", "experimental setup", "experiments",
        "methods and materials", "research methodology", "techniques"
    ],
    "results": [
        "results", "findings", "outcomes", "item 2.", "empirical results", "key findings"
    ],
    "discussion": [
        "discussion", "interpretation", "implications", "item 3.",
        "analysis and discussion", "interpretation of results"
    ],
    "conclusion": [
        "conclusion", "conclusions", "summary of findings",
        "final thoughts", "item 4.", "concluding remarks", "summary"
    ],
}

# Heuristic to determine if a line is likely a section header based on formatting and content cues
def is_header_like(line):
    stripped = line.strip()
    return (
        5 < len(stripped) < 80 and
        not stripped.endswith(".") and
        (
            stripped.isupper() or
            stripped[0].isdigit() or
            stripped.istitle()
        )
    )

# Section a single document based on the defined headers and heuristics
def section_document(text):
    sections = {key: [] for key in SECTION_HEADERS.keys()}
    current_section = None

    for line in text.splitlines():
        line_lower = line.lower()

        for section, headers in SECTION_HEADERS.items():
            if any(h in line_lower for h in headers):
                current_section = section
                break

        if current_section:
            sections[current_section].append(line)

    return {k: "\n".join(v) for k, v in sections.items()}

# Section documents based on their classifications
def section_documents(texts):
    sectioned_documents = {}
    # For each document, run the sectioning algorithm to identify sections based on the defined headers and heuristics. Store the identified sections in a dictionary for each document.
    for file, text in texts:
        sections = section_document(text)
        # keep only filled sections
        filled_sections_only = {
            section: content
            for section, content in sections.items()
            if content.strip()
        }
        # List the identified sections for each document
        print(
            f"Sections for {get_basename(file)}:\n",
            list(filled_sections_only.keys())
        )
        sectioned_documents[file] = filled_sections_only
    return sectioned_documents