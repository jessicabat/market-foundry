""" Document Classification 
Functions to process text and files for data extraction tasks. Supported file formats include .pdf, .txt, .docx, .html, and .json.
"""

import os
import joblib
import pandas as pd
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, BSHTMLLoader, JSONLoader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".html", ".json"}

# Basename extraction
def get_basename(file_path):
    return os.path.basename(file_path)

# Load file based on its extension
def load_file(file_path):
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".html"):
            loader = BSHTMLLoader(file_path)
        elif file_path.endswith(".json"):
            loader = JSONLoader(file_path, jq_schema=".text")
        else:
            raise ValueError("Unsupported file format")
        return loader.load()
    except Exception as e:
        print(f"Skipping file due to load error: {file_path} ({e})")
        return []

# Load multiple files given a list of file paths
def load_files(file_paths):
    loaded_files = []
    for file_path in file_paths:
        loaded_file = load_file(file_path)
        loaded_files.append(loaded_file)
    print(f"Loaded {len(loaded_files)} files.")
    return loaded_files

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

# Extract raw text from loaded files
def extract_text(loaded_files):
    texts = []
    for loaded_file in loaded_files:
        combined_text = " ".join(page.page_content for page in loaded_file)
        texts.append((loaded_file[0].metadata.get('source'), combined_text))
    return texts

# Load the trained document classification model
def load_document_classification_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Ensure there is a model to load in the specified path.")
    return joblib.load(model_path)

# Load the TF-IDF vectorizer
def load_tfidf_vectorizer(vectorizer_path):
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {vectorizer_path}. Ensure there is a model to load in the specified path.")
    return joblib.load(vectorizer_path)

# Classify documents using the provided model and vectorizer
def classify_document_types(model, vectorizer, texts):
    vectorized_texts = vectorizer.transform([text[1] for text in texts])
    classifications = {}
    for file, text in zip([text[0] for text in texts], vectorized_texts):
        classification = model.predict(text)
        classifications[file] = classification[0]
    return classifications

# Output document classifications
def output_classifications(classifications):
    df_classifications = pd.DataFrame(list(classifications.items()), columns=["File", "Document Type"])
    df_classifications["File"] = df_classifications["File"].apply(get_basename)
    print("Document Classifications:\n", df_classifications)
    
""" Document Sectioning for src/run.py
Functions to run the document sectioning pipeline.
"""

SECTION_HEADERS = {
    "introduction": ["introduction", "intro"],
    "methodology": ["methodology", "methods", "method"],
    "analysis": ["analysis", "results"],
    "discussion": ["discussion"],
    "conclusion": ["conclusion", "concluding remarks"],
}

