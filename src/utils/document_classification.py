""" Document Classification 
Functions to process text and files for data extraction tasks. Supported file formats include .pdf, .txt, .docx, .html, and .json.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, BSHTMLLoader, JSONLoader
from scipy.sparse import csr_matrix, hstack
import time

try:
    from knn_pipeline import classifier_knn
except ModuleNotFoundError:
    from src.knn_pipeline import classifier_knn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from OneKE.src.utils.process import process_single_quotes, remove_redundant_space, format_string
except ModuleNotFoundError:
    def process_single_quotes(text):
        return text

    def remove_redundant_space(text):
        return " ".join(text.split())

    def format_string(text):
        return text

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".html", ".json"}
CLASS_LABELS = [
    "Earnings Call Transcript",
    "News Article",
    "Press Release",
    "Research Paper",
    "SEC Filing",
]
PRESS_RELEASE_LABEL = "Press Release"
NEWS_ARTICLE_LABEL = "News Article"
STACKER_BUNDLE_FILENAME = "Document_Stacker.joblib"
PRESS_RELEASE_NEWS_BUNDLE_FILENAME = "PressReleaseNews_Classifier.joblib"
PRESS_RELEASE_SIGNAL_PATTERNS = {
    "wire_service": (
        "business wire",
        "globenewswire",
        "pr newswire",
        "newswire",
    ),
    "contact_block": (
        "media contact",
        "press contact",
        "investor relations",
        "ir contact",
    ),
    "about_company": (
        "about the company",
        "about us",
        "about ",
    ),
    "announcement": (
        "today announced",
        "announces",
        "declares",
        "launches",
        "to acquire",
        "agreement",
    ),
    "event_language": (
        "conference call",
        "webcast",
        "quarterly dividend",
        "debt offering",
        "earnings release",
    ),
    "newsroom_shell": (
        "press center",
        "newsroom",
        "media kit",
    ),
    "news_byline": (
        "associated press",
        "reuters",
        "bloomberg",
        "cnbc",
        "author",
        " by ",
    ),
}


def _default_model_dir():
    return os.path.join(project_root, "src", "models")


def _load_optional_bundle(filename, model_dir=None):
    bundle_path = os.path.join(model_dir or _default_model_dir(), filename)
    if not os.path.exists(bundle_path):
        return None
    return joblib.load(bundle_path)


def load_document_stacker_bundle(model_dir=None):
    return _load_optional_bundle(STACKER_BUNDLE_FILENAME, model_dir=model_dir)


def load_press_release_news_bundle(model_dir=None):
    return _load_optional_bundle(PRESS_RELEASE_NEWS_BUNDLE_FILENAME, model_dir=model_dir)


def _prepare_text_for_model(text):
    return remove_redundant_space(format_string(process_single_quotes(text)))


def prepare_text_for_model(text):
    return _prepare_text_for_model(text)


def _align_probabilities(labels, probabilities, class_order):
    label_to_probability = {label: prob for label, prob in zip(labels, probabilities)}
    return np.array([label_to_probability.get(label, 0.0) for label in class_order], dtype=float)


def _align_knn_scores(score_map, class_order):
    return np.array([score_map.get(label, -1.0) for label in class_order], dtype=float)


def _top_gap(values):
    if len(values) == 0:
        return 0.0
    sorted_values = np.sort(values)[::-1]
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    return float(sorted_values[0] - sorted_values[1])


def _softmax(values):
    if len(values) == 0:
        return values
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = exp_values.sum()
    if total == 0:
        return np.full_like(values, 1.0 / len(values), dtype=float)
    return exp_values / total


def extract_press_release_signal_features(text):
    snippet = classifier_knn.build_managed_snippet(text)
    lower_text = snippet.lower()
    line_count = max(snippet.count("\n") + 1, 1)
    feature_values = []
    for phrases in PRESS_RELEASE_SIGNAL_PATTERNS.values():
        feature_values.append(float(any(phrase in lower_text for phrase in phrases)))
    feature_values.extend(
        [
            min(len(snippet), 5000) / 5000.0,
            line_count / 200.0,
        ]
    )
    return np.array(feature_values, dtype=float)


def _build_stacker_features(tfidf_labels, tfidf_probabilities, knn_score_map, class_order, text):
    aligned_tfidf = _align_probabilities(tfidf_labels, tfidf_probabilities, class_order)
    aligned_knn = _align_knn_scores(knn_score_map, class_order)
    scalar_features = np.concatenate(
        [
            np.array(
                [
                    aligned_tfidf.max(initial=0.0),
                    _top_gap(aligned_tfidf),
                    aligned_knn.max(initial=-1.0),
                    _top_gap(aligned_knn),
                ],
                dtype=float,
            ),
            extract_press_release_signal_features(text),
        ]
    )
    return np.concatenate([aligned_tfidf, aligned_knn, scalar_features])


def _build_press_release_news_features(bundle, texts):
    snippets = [classifier_knn.build_managed_snippet(text) for text in texts]
    text_features = bundle["vectorizer"].transform(snippets)
    scalar_features = np.vstack([extract_press_release_signal_features(text) for text in texts])
    return hstack([text_features, csr_matrix(scalar_features)], format="csr")


def _fallback_ensemble_probabilities(tfidf_labels, tfidf_probabilities, knn_score_map, class_order):
    aligned_tfidf = _align_probabilities(tfidf_labels, tfidf_probabilities, class_order)
    aligned_knn = _align_knn_scores(knn_score_map, class_order)
    aligned_knn_probs = _softmax(aligned_knn * 6.0)
    return 0.7 * aligned_tfidf + 0.3 * aligned_knn_probs


def _should_rerank_press_release_news(probabilities, class_order):
    if PRESS_RELEASE_LABEL not in class_order or NEWS_ARTICLE_LABEL not in class_order:
        return False

    press_release_index = class_order.index(PRESS_RELEASE_LABEL)
    news_article_index = class_order.index(NEWS_ARTICLE_LABEL)
    combined_mass = probabilities[press_release_index] + probabilities[news_article_index]
    top_two = set(np.argsort(probabilities)[-2:])
    return combined_mass >= 0.55 or top_two == {press_release_index, news_article_index}


def _apply_press_release_news_reranker(probabilities, class_order, text, bundle):
    reranker_features = _build_press_release_news_features(bundle, [text])
    reranker_probabilities = bundle["model"].predict_proba(reranker_features)[0]
    reranker_classes = list(bundle.get("class_order", bundle["model"].classes_))

    press_release_index = class_order.index(PRESS_RELEASE_LABEL)
    news_article_index = class_order.index(NEWS_ARTICLE_LABEL)
    combined_mass = probabilities[press_release_index] + probabilities[news_article_index]

    reranker_map = {
        label: prob for label, prob in zip(reranker_classes, reranker_probabilities)
    }
    updated_probabilities = probabilities.copy()
    updated_probabilities[press_release_index] = combined_mass * reranker_map.get(PRESS_RELEASE_LABEL, 0.0)
    updated_probabilities[news_article_index] = combined_mass * reranker_map.get(NEWS_ARTICLE_LABEL, 0.0)

    total_probability = updated_probabilities.sum()
    if total_probability > 0:
        updated_probabilities = updated_probabilities / total_probability
    return updated_probabilities

# Basename extraction
def get_basename(file_path):
    return os.path.basename(file_path)

# Clean texts
def clean_texts(texts):
    cleaned_texts = []
    for file, text in texts:
        cleaned_text = _prepare_text_for_model(text)
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

# Classify documents using a stacked tf-idf + KNN ensemble, with a targeted reranker for press releases vs news articles.
def classify_document_types(model, vectorizer, texts, stacker_bundle=None, press_release_news_bundle=None):
    raw_texts = [text[1] for text in texts]
    filenames = [text[0] for text in texts]

    vectorized_texts = vectorizer.transform(raw_texts)
    stacker_bundle = stacker_bundle or load_document_stacker_bundle()
    press_release_news_bundle = press_release_news_bundle or load_press_release_news_bundle()

    knn_classifier = classifier_knn.KNNClassifier(
        reference_dir=os.path.join(project_root, "src", "knn_pipeline", "reference_docs")
    )

    classifications = {}
    tfidf_labels = list(model.classes_)
    class_order = list(stacker_bundle.get("class_order", tfidf_labels)) if stacker_bundle else tfidf_labels

    for file, raw_text, tfidf_vec in zip(filenames, raw_texts, vectorized_texts):
        tfidf_predicted_probabilities = model.predict_proba(tfidf_vec)[0]
        knn_score_map = knn_classifier.score_labels(raw_text)

        if stacker_bundle:
            stacker_features = _build_stacker_features(
                tfidf_labels,
                tfidf_predicted_probabilities,
                knn_score_map,
                class_order,
                raw_text,
            ).reshape(1, -1)
            combined_probabilities = stacker_bundle["model"].predict_proba(stacker_features)[0]
        else:
            combined_probabilities = _fallback_ensemble_probabilities(
                tfidf_labels,
                tfidf_predicted_probabilities,
                knn_score_map,
                class_order,
            )

        if press_release_news_bundle and _should_rerank_press_release_news(combined_probabilities, class_order):
            combined_probabilities = _apply_press_release_news_reranker(
                combined_probabilities,
                class_order,
                raw_text,
                press_release_news_bundle,
            )

        classifications[file] = class_order[int(np.argmax(combined_probabilities))]

    return classifications

# Output document classifications
def output_classifications(classifications):
    df_classifications = pd.DataFrame(list(classifications.items()), columns=["File", "Document Type"])
    df_classifications["File"] = df_classifications["File"].apply(get_basename)
    print("Document Classifications:\n", df_classifications, "\n")