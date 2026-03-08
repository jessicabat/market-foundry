from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.knn_pipeline import classifier_knn
from src.utils.document_classification import (
    CLASS_LABELS,
    NEWS_ARTICLE_LABEL,
    PRESS_RELEASE_LABEL,
    PRESS_RELEASE_NEWS_BUNDLE_FILENAME,
    STACKER_BUNDLE_FILENAME,
    extract_press_release_signal_features,
    load_file,
    prepare_text_for_model,
)


MULTICLASS_RANDOM_STATE = 42
BOUNDARY_NEGATIVE_RATIO = 8


def build_multiclass_vectorizer():
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    min_df=2,
                    max_df=0.9,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    min_df=2,
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def build_boundary_vectorizer():
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    min_df=1,
                    max_df=0.95,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    min_df=1,
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def build_multiclass_model():
    return LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
    )


def build_boundary_model():
    return CalibratedClassifierCV(
        estimator=LogisticRegression(
            C=2.5,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
        ),
        method="sigmoid",
        cv=3,
    )


def build_stacker_model():
    return CalibratedClassifierCV(
        estimator=LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
        ),
        method="sigmoid",
        cv=3,
    )


def read_training_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open() as handle:
            payload = json.load(handle)
        thread = payload.get("thread", {}) if isinstance(payload, dict) else {}
        parts = [
            payload.get("title") if isinstance(payload, dict) else None,
            thread.get("title") if isinstance(thread, dict) else None,
            thread.get("site_full") if isinstance(thread, dict) else None,
            payload.get("author") if isinstance(payload, dict) else None,
            payload.get("text") if isinstance(payload, dict) else None,
        ]
        return prepare_text_for_model("\n\n".join(str(part) for part in parts if part))

    if suffix == ".txt":
        text, _ = classifier_knn.read_text_robust(path, max_bytes=250000)
        return prepare_text_for_model(text)

    loaded = load_file(str(path))
    if not loaded:
        return ""
    combined_text = "\n".join(page.page_content for page in loaded)
    return prepare_text_for_model(combined_text)


def collect_examples(*roots: Path):
    examples = []
    for root in roots:
        if not root.exists():
            continue
        for class_name in CLASS_LABELS:
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for path in sorted(p for p in class_dir.rglob("*") if p.is_file()):
                text = read_training_text(path)
                if len(text) < 50:
                    continue
                examples.append((str(path), text, class_name))
    return examples


def collect_boundary_examples(data_root: Path, reference_root: Path):
    positives = []
    negatives = []
    for source_root in (data_root, reference_root):
        for _, text, label in collect_examples(source_root):
            if label == PRESS_RELEASE_LABEL:
                positives.append(text)
            elif label == NEWS_ARTICLE_LABEL:
                negatives.append(text)

    rng = np.random.default_rng(MULTICLASS_RANDOM_STATE)
    max_negative_count = max(len(positives) * BOUNDARY_NEGATIVE_RATIO, len(positives))
    if len(negatives) > max_negative_count:
        selected_indices = rng.choice(len(negatives), size=max_negative_count, replace=False)
        negatives = [negatives[index] for index in np.sort(selected_indices)]

    texts = positives + negatives
    labels = [PRESS_RELEASE_LABEL] * len(positives) + [NEWS_ARTICLE_LABEL] * len(negatives)
    return texts, labels


def build_stacker_features(texts, vectorizer, model, knn_model, class_order):
    tfidf_vectors = vectorizer.transform(texts)
    tfidf_probabilities = model.predict_proba(tfidf_vectors)
    tfidf_labels = list(model.classes_)
    knn_score_maps = knn_model.score_labels_batch(texts)

    features = []
    for text, probabilities, knn_scores in zip(texts, tfidf_probabilities, knn_score_maps):
        aligned_tfidf = np.array(
            [{label: prob for label, prob in zip(tfidf_labels, probabilities)}.get(label, 0.0) for label in class_order],
            dtype=float,
        )
        aligned_knn = np.array([knn_scores.get(label, -1.0) for label in class_order], dtype=float)
        sorted_tfidf = np.sort(aligned_tfidf)[::-1]
        sorted_knn = np.sort(aligned_knn)[::-1]
        scalar = np.concatenate(
            [
                np.array(
                    [
                        aligned_tfidf.max() if len(aligned_tfidf) else 0.0,
                        (sorted_tfidf[0] - sorted_tfidf[1]) if len(sorted_tfidf) > 1 else (sorted_tfidf[0] if len(sorted_tfidf) else 0.0),
                        aligned_knn.max() if len(aligned_knn) else -1.0,
                        (sorted_knn[0] - sorted_knn[1]) if len(sorted_knn) > 1 else (sorted_knn[0] if len(sorted_knn) else 0.0),
                    ],
                    dtype=float,
                ),
                extract_press_release_signal_features(text),
            ]
        )
        features.append(np.concatenate([aligned_tfidf, aligned_knn, scalar]))
    return np.vstack(features)


def build_boundary_features(texts, vectorizer):
    snippets = [classifier_knn.build_managed_snippet(text) for text in texts]
    text_features = vectorizer.transform(snippets)
    scalar_features = np.vstack([extract_press_release_signal_features(text) for text in texts])
    return hstack([text_features, csr_matrix(scalar_features)], format="csr")


def train_multiclass_models(train_texts, train_labels, reference_root: Path):
    print("[1/6] Fitting multiclass TF-IDF vectorizer...", flush=True)
    vectorizer = build_multiclass_vectorizer()
    train_vectors = vectorizer.fit_transform(train_texts)

    print("[2/6] Training multiclass logistic regression...", flush=True)
    model = build_multiclass_model()
    model.fit(train_vectors, train_labels)

    class_order = list(model.classes_)
    print("[3/6] Loading KNN reference embeddings...", flush=True)
    knn_model = classifier_knn.KNNClassifier(str(reference_root))

    min_class_support = min(Counter(train_labels).values())
    fold_count = max(3, min(5, min_class_support))
    skf = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=MULTICLASS_RANDOM_STATE)
    oof_features = None

    train_texts_array = np.array(train_texts)
    train_labels_array = np.array(train_labels)
    print(f"[4/6] Building out-of-fold stacker features across {fold_count} folds...", flush=True)
    for fold_number, (fold_train_idx, fold_val_idx) in enumerate(skf.split(train_texts_array, train_labels_array), start=1):
        print(f"  - Fold {fold_number}/{fold_count}: fitting base model and scoring {len(fold_val_idx)} validation documents", flush=True)
        fold_vectorizer = build_multiclass_vectorizer()
        fold_model = build_multiclass_model()
        fold_train_texts = train_texts_array[fold_train_idx].tolist()
        fold_val_texts = train_texts_array[fold_val_idx].tolist()
        fold_train_labels = train_labels_array[fold_train_idx].tolist()

        fold_train_vectors = fold_vectorizer.fit_transform(fold_train_texts)
        fold_model.fit(fold_train_vectors, fold_train_labels)

        fold_features = build_stacker_features(
            fold_val_texts,
            fold_vectorizer,
            fold_model,
            knn_model,
            class_order,
        )
        if oof_features is None:
            oof_features = np.zeros((len(train_texts), fold_features.shape[1]), dtype=float)
        oof_features[fold_val_idx] = fold_features

    print("[5/6] Training calibrated stacker...", flush=True)
    stacker = build_stacker_model()
    stacker.fit(oof_features, train_labels)
    return vectorizer, model, stacker, knn_model


def train_boundary_model(train_texts, train_labels):
    print(f"[6/6] Training press-release vs news reranker on {len(train_texts)} documents...", flush=True)
    vectorizer = build_boundary_vectorizer()
    train_vectors = vectorizer.fit_transform(
        [classifier_knn.build_managed_snippet(text) for text in train_texts]
    )
    scalar_features = np.vstack([extract_press_release_signal_features(text) for text in train_texts])
    combined_features = hstack([train_vectors, csr_matrix(scalar_features)], format="csr")

    model = build_boundary_model()
    model.fit(combined_features, train_labels)
    return vectorizer, model


def evaluate_models(test_texts, test_labels, vectorizer, model, stacker, knn_model, boundary_bundle):
    base_predictions = model.predict(vectorizer.transform(test_texts))
    stacker_features = build_stacker_features(test_texts, vectorizer, model, knn_model, list(model.classes_))
    stacked_probabilities = stacker.predict_proba(stacker_features)
    stacked_classes = list(stacker.classes_)

    final_predictions = []
    press_release_index = stacked_classes.index(PRESS_RELEASE_LABEL)
    news_article_index = stacked_classes.index(NEWS_ARTICLE_LABEL)
    for text, probabilities in zip(test_texts, stacked_probabilities):
        updated_probabilities = probabilities.copy()
        combined_mass = updated_probabilities[press_release_index] + updated_probabilities[news_article_index]
        top_two = set(np.argsort(updated_probabilities)[-2:])
        if combined_mass >= 0.55 or top_two == {press_release_index, news_article_index}:
            reranker_features = build_boundary_features([text], boundary_bundle["vectorizer"])
            reranker_probabilities = boundary_bundle["model"].predict_proba(reranker_features)[0]
            reranker_map = {
                label: prob
                for label, prob in zip(boundary_bundle["class_order"], reranker_probabilities)
            }
            updated_probabilities[press_release_index] = combined_mass * reranker_map.get(PRESS_RELEASE_LABEL, 0.0)
            updated_probabilities[news_article_index] = combined_mass * reranker_map.get(NEWS_ARTICLE_LABEL, 0.0)
            updated_probabilities /= updated_probabilities.sum()
        final_predictions.append(stacked_classes[int(np.argmax(updated_probabilities))])

    print("Base multiclass accuracy:", f"{accuracy_score(test_labels, base_predictions):.3f}")
    print("Stacked pipeline accuracy:", f"{accuracy_score(test_labels, final_predictions):.3f}")
    print(classification_report(test_labels, final_predictions, labels=CLASS_LABELS, zero_division=0))


def save_artifacts(models_dir: Path, vectorizer, model, stacker, boundary_vectorizer, boundary_model):
    joblib.dump(model, models_dir / "Document_Classifier.joblib")
    joblib.dump(vectorizer, models_dir / "TFIDF_Vectorizer.joblib")
    joblib.dump(
        {
            "model": stacker,
            "class_order": list(stacker.classes_),
        },
        models_dir / STACKER_BUNDLE_FILENAME,
    )
    joblib.dump(
        {
            "model": boundary_model,
            "vectorizer": boundary_vectorizer,
            "class_order": list(boundary_model.classes_),
        },
        models_dir / PRESS_RELEASE_NEWS_BUNDLE_FILENAME,
    )


def main():
    parser = argparse.ArgumentParser(description="Train the stacked Market Foundry document classifier.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=project_root / "src" / "models",
        help="Directory where trained artifacts will be written.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "src" / "models" / "data",
        help="Directory containing labeled training data.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=project_root / "src" / "knn_pipeline" / "reference_docs",
        help="Reference documents used by the embedding KNN classifier.",
    )
    args = parser.parse_args()

    examples = collect_examples(args.data_dir, args.reference_dir)
    if not examples:
        raise SystemExit("No training documents found.")

    texts = [text for _, text, _ in examples]
    labels = [label for _, _, label in examples]
    print("Training set distribution:")
    for label, count in sorted(Counter(labels).items()):
        print(f"  {label}: {count}")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=MULTICLASS_RANDOM_STATE,
        stratify=labels,
    )
    print(f"Train/test split: {len(train_texts)} train, {len(test_texts)} test", flush=True)

    vectorizer, model, stacker, knn_model = train_multiclass_models(
        train_texts,
        train_labels,
        args.reference_dir,
    )

    boundary_texts, boundary_labels = collect_boundary_examples(args.data_dir, args.reference_dir)
    boundary_vectorizer, boundary_model = train_boundary_model(boundary_texts, boundary_labels)
    boundary_bundle = {
        "model": boundary_model,
        "vectorizer": boundary_vectorizer,
        "class_order": list(boundary_model.classes_),
    }

    evaluate_models(
        test_texts,
        test_labels,
        vectorizer,
        model,
        stacker,
        knn_model,
        boundary_bundle,
    )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    print("Saving updated model artifacts...", flush=True)
    save_artifacts(
        args.models_dir,
        vectorizer,
        model,
        stacker,
        boundary_vectorizer,
        boundary_model,
    )


if __name__ == "__main__":
    main()