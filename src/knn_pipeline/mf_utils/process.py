"""
Functions to process text and files for data extraction tasks. Supported file formats include .pdf, .txt, .docx, .html, and .json.
"""

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, BSHTMLLoader, JSONLoader

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