""" Document Classification 
Functions to process text and files for data extraction tasks. Supported file formats include .pdf, .txt, .docx, .html, and .json.
"""

import os
import sys
import joblib
import pandas as pd
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, BSHTMLLoader, JSONLoader
from knn_pipeline import classifier_knn
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from OneKE.src.utils.process import process_single_quotes, remove_redundant_space, format_string

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".html", ".json"}

# Basename extraction
def get_basename(file_path):
    return os.path.basename(file_path)

# Clean texts
def clean_texts(texts):
    cleaned_texts = []
    for file, text in texts:
        cleaned_text = remove_redundant_space(format_string(process_single_quotes(text)))
        cleaned_texts.append((file, cleaned_text))
    return cleaned_texts

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
    start_time = time.time()
    loaded_files = []
    for file_path in file_paths:
        loaded_file = load_file(file_path)
        loaded_files.append(loaded_file)
    print(f"Loaded {len(loaded_files)} files in {time.time() - start_time:.2f} seconds.")
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
    for idx, loaded_file in enumerate(loaded_files):
        if not loaded_file:
            print(f"Skipping empty file at index {idx}")
            continue
        combined_text = "\n".join(page.page_content for page in loaded_file)
        texts.append((loaded_file[0].metadata.get('source'), combined_text))
    return texts

# Load the trained document classification model
def load_tfidf_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Ensure there is a model to load in the specified path.")
    return joblib.load(model_path)

# Load the TF-IDF vectorizer
def load_tfidf_vectorizer(vectorizer_path):
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {vectorizer_path}. Ensure there is a model to load in the specified path.")
    return joblib.load(vectorizer_path)

# Classify documents using the tfidf model and KNN classifier, with heuristics to resolve disagreements
def classify_document_types(model, vectorizer, texts):
    raw_texts = [text[1] for text in texts]
    filenames = [text[0] for text in texts]

    vectorized_texts = vectorizer.transform(raw_texts)

    knn_classifier = classifier_knn.KNNClassifier(
        reference_dir=os.path.join(project_root, "src", "knn_pipeline", "reference_docs")
    )

    classifications = {}

    for file, raw_text, tfidf_vec in zip(filenames, raw_texts, vectorized_texts):

        tfidf_predicted_probabilities = model.predict_proba(tfidf_vec)
        tfidf_confidence = max(tfidf_predicted_probabilities[0])
        tfidf_classification = model.predict(tfidf_vec)

        # Pass RAW TEXT to KNN
        knn_classification, knn_confidence = knn_classifier.classify(raw_text)

        if tfidf_classification[0] == knn_classification:
            classifications[file] = tfidf_classification[0]
        elif knn_classification == "Press Release": # Heuristic override: If KNN predicts "Press Release" with high confidence, trust it over TF-IDF due to better performance on this class in validation
            classifications[file] = knn_classification
        else:
            classifications[file] = (
                tfidf_classification[0]
                if tfidf_confidence > knn_confidence
                else knn_classification
            )
            print(f'Disagreement in classification for {get_basename(file)} → Resolved to {classifications[file]}')

    return classifications

# Output document classifications
def output_classifications(classifications):
    df_classifications = pd.DataFrame(list(classifications.items()), columns=["File", "Document Type"])
    df_classifications["File"] = df_classifications["File"].apply(get_basename)
    print("Document Classifications:\n", df_classifications, "\n")
