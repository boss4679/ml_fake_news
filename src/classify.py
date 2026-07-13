"""Classify news articles (raw text or scraped from a URL).

Load the trained model once and reuse it for many predictions::

    from src.classify import Classifier
    clf = Classifier()
    print(clf.classify_text("Some article body..."))
    print(clf.classify_url("https://example.com/some-news-story"))

As a CLI::

    python -m src.classify --text "Breaking: ..."
    python -m src.classify --url https://example.com/story
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import joblib

from .scraper import scrape_article
from .train import MODEL_PATH


@dataclass
class Prediction:
    """Result of classifying one article."""

    label: str            # "Reliable" or "Unreliable"
    confidence: float     # probability of the predicted label, 0..1
    source: str           # "text" or the scraped URL

    def __str__(self) -> str:
        return f"{self.label} ({self.confidence:.1%} confidence)"


class Classifier:
    """Thin wrapper around the persisted TF-IDF + logistic regression pipeline."""

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `python -m src.train` first."
            )
        self.pipeline = joblib.load(model_path)
        # Locate the "Reliable" (label 1) column rather than assuming its
        # position, mirroring export_model.py.
        classes = list(self.pipeline.named_steps["clf"].classes_)
        self._reliable_col = classes.index(1)

    def _predict(self, text: str, source: str) -> Prediction:
        proba = self.pipeline.predict_proba([text])[0]
        reliable_p = float(proba[self._reliable_col])
        if reliable_p >= 0.5:
            return Prediction("Reliable", reliable_p, source)
        return Prediction("Unreliable", 1.0 - reliable_p, source)

    def classify_text(self, text: str) -> Prediction:
        """Classify a raw article string."""
        if not text or not text.strip():
            raise ValueError("Cannot classify empty text.")
        return self._predict(text, source="text")

    def classify_url(self, url: str) -> Prediction:
        """Scrape ``url`` with BeautifulSoup and classify the extracted text."""
        article = scrape_article(url)
        return self._predict(article.combined, source=url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a news article.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Raw article text to classify.")
    group.add_argument("--url", help="URL of an article to scrape and classify.")
    args = parser.parse_args()

    clf = Classifier()
    if args.text:
        result = clf.classify_text(args.text)
    else:
        result = clf.classify_url(args.url)
        print(f"Scraped: {result.source}")
    print(f"Prediction: {result}")


if __name__ == "__main__":
    main()
