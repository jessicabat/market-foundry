"""
Functions to process text and files for data extraction tasks. Supported file formats include .pdf, .txt, .docx, .html, and .json.
"""

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, BSHTMLLoader, JSONLoader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".html", ".json"}

# Load file based on its extension
def load_file(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".html"):
        loader = BSHTMLLoader(file_path)
    elif file_path.endswith(".json"):
        loader = JSONLoader(file_path)
    else:
        raise ValueError("Unsupported file format")

    return loader.load()

# Basename extraction
def get_basename(file_path):
    return os.path.basename(file_path)

# Expand a given path if it is a directory to process all supported files
def expand_path(path):
    expanded = []

    if os.path.isfile(path):
        expanded.append(path)

    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS:
                    expanded.append(os.path.join(root, file))
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    return expanded