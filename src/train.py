"""Train the TF-IDF + logistic regression classifier.

Run as a module::

    python -m src.train

The trained pipeline (vectorizer + classifier) is serialized to
``models/news_classifier.joblib`` for reuse by the CLI and web app.
"""

from __future__ import annotations

import argparse
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .dataset import load_dataset
from .preprocess import clean_text

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "news_classifier.joblib")


def build_pipeline() -> Pipeline:
    """Construct the TF-IDF -> LogisticRegression pipeline.

    ``preprocessor`` wires our deterministic :func:`clean_text` into the
    vectorizer so training and inference share the exact same text handling.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                    max_features=50_000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=4.0,
                    class_weight="balanced",
                    n_jobs=None,
                ),
            ),
        ]
    )


def train(test_size: float = 0.2, seed: int = 42, model_path: str = MODEL_PATH) -> Pipeline:
    """Load data, fit the pipeline, report metrics, and persist the model."""
    data = load_dataset()
    print(f"Loaded {len(data)} labeled articles "
          f"({int((data['label'] == 1).sum())} reliable, "
          f"{int((data['label'] == 0).sum())} unreliable).")

    # Stratify so both classes appear in train and test even for tiny datasets.
    stratify = data["label"] if data["label"].nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    pipeline = build_pipeline()
    print("Training TF-IDF + Logistic Regression...")
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    print("\nConfusion matrix (rows=true, cols=pred) [Unreliable, Reliable]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, target_names=["Unreliable", "Reliable"], zero_division=0
    ))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the news classifier.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data held out for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out", default=MODEL_PATH, help="Output model path.")
    args = parser.parse_args()
    train(test_size=args.test_size, seed=args.seed, model_path=args.out)


if __name__ == "__main__":
    main()
