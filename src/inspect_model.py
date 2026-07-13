"""Inspect what the model actually learned.

Because the classifier is linear, each TF-IDF feature has a single coefficient:
positive weights push a prediction toward **Reliable**, negative toward
**Unreliable**. Printing the strongest ones is a quick sanity check on whether
the model learned genuine signal or a dataset artifact (e.g. always predicting
"reliable" from a wire-service byline).

Run::

    python -m src.inspect_model            # top 20 terms per class
    python -m src.inspect_model --top 30
"""

from __future__ import annotations

import argparse

import joblib

from .train import MODEL_PATH


def top_terms(model_path: str = MODEL_PATH, top: int = 20):
    """Return ``(reliable, unreliable)`` lists of ``(term, weight)`` tuples."""
    pipeline = joblib.load(model_path)
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    features = tfidf.get_feature_names_out()
    weights = clf.coef_[0]
    order = weights.argsort()  # ascending: most negative first

    unreliable = [(features[i], float(weights[i])) for i in order[:top]]
    reliable = [(features[i], float(weights[i])) for i in order[-top:][::-1]]
    return reliable, unreliable


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the model's most predictive terms.")
    parser.add_argument("--top", type=int, default=20, help="Terms to show per class.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the trained model.")
    args = parser.parse_args()

    reliable, unreliable = top_terms(args.model, args.top)

    print(f"\nTop {args.top} terms pushing toward RELIABLE:")
    for term, weight in reliable:
        print(f"  {weight:+.3f}  {term}")

    print(f"\nTop {args.top} terms pushing toward UNRELIABLE:")
    for term, weight in unreliable:
        print(f"  {weight:+.3f}  {term}")


if __name__ == "__main__":
    main()
