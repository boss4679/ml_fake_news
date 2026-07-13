"""Export the trained sklearn pipeline to a browser-loadable ``model.json``.

Because the pipeline is **linear** (TF-IDF features + logistic regression), the
entire model is just a vocabulary, per-term IDF weights, per-term classifier
coefficients, and a bias. That is all we need to reproduce predictions in pure
JavaScript inside a Chrome extension — no ML runtime required.

Run::

    python -m src.export_model

Writes ``extension/model.json``.
"""

from __future__ import annotations

import json
import os

import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .train import MODEL_PATH

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "extension")
OUT_PATH = os.path.join(OUT_DIR, "model.json")


def export(model_path: str = MODEL_PATH, out_path: str = OUT_PATH) -> str:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `python -m src.train` first."
        )

    pipeline = joblib.load(model_path)
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    # vocabulary_ maps an n-gram string -> feature column index.
    vocabulary = {term: int(idx) for term, idx in tfidf.vocabulary_.items()}
    idf = [float(v) for v in tfidf.idf_]
    # Binary logistic regression: one coefficient row, aligned with class 1.
    coef = [float(v) for v in clf.coef_[0]]
    intercept = float(clf.intercept_[0])
    classes = [int(c) for c in clf.classes_]

    model = {
        "format": "tfidf-logreg-v1",
        "config": {
            # Everything JS needs to reproduce the exact feature transform.
            "sublinear_tf": bool(tfidf.sublinear_tf),
            "ngram_range": list(tfidf.ngram_range),
            "norm": tfidf.norm,           # "l2"
            "min_token_len": 2,           # from the default token_pattern \b\w\w+\b
            "lowercase": True,
        },
        # class 1 == Reliable, class 0 == Unreliable (see dataset.py).
        "classes": classes,
        "reliable_class_index": classes.index(1),
        "intercept": intercept,
        "stop_words": sorted(ENGLISH_STOP_WORDS),
        "vocabulary": vocabulary,
        "idf": idf,
        "coef": coef,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(model, fh, ensure_ascii=False)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Exported {len(vocabulary)} features to {out_path} ({size_kb:.0f} KB)")
    return out_path


if __name__ == "__main__":
    export()
