"""Step 1 of the parity test: dump the Python model's predictions to JSON.

Run from the repo root:  python -m tests.gen_expected
Then:                    node tests/parity_test.mjs

Together these prove the browser (JS) classifier reproduces the trained
scikit-learn pipeline to floating-point precision.
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd

from src.train import MODEL_PATH

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected.json")

# A mix of synthetic and (if available) real dataset articles.
TEXTS = [
    "The central bank announced on Wednesday it would keep interest rates "
    "unchanged, according to officials, citing stable inflation data.",
    "Doctors HATE this one weird miracle trick that big pharma does not want "
    "you to know. Share before they delete this shocking secret cure forever!",
    "a",  # below min token length -> no features
    "According to the report, officials said the measure passed after a vote.",
]


def _maybe_add_real(texts: list[str]) -> None:
    for name in ("True.csv", "Fake.csv"):
        path = os.path.join(BASE, "data", name)
        if os.path.exists(path):
            df = pd.read_csv(path, nrows=5)
            for _, row in df.iterrows():
                texts.append(f"{row.get('title', '')}. {row.get('text', '')}"[:2000])


def main() -> None:
    pipeline = joblib.load(MODEL_PATH)
    texts = list(TEXTS)
    _maybe_add_real(texts)

    rows = []
    for text in texts:
        proba = pipeline.predict_proba([text])[0]
        rows.append({"text": text, "prob_reliable": float(proba[1])})

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    print(f"Wrote {len(rows)} expected predictions to {OUT}")


if __name__ == "__main__":
    main()
