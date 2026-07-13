"""Load labeled news data for training.

Supports two common layouts so the pipeline works with the popular public
datasets out of the box:

1. **Kaggle "Fake and Real News"** — two files, ``True.csv`` and ``Fake.csv``,
   each with a ``title`` and ``text`` column. Place them under ``data/``.
2. **Single CSV** — one file with a ``text`` column and a ``label`` column,
   where the label is either the strings ``reliable``/``unreliable`` (or
   ``real``/``fake``) or the integers ``1``/``0``.

If neither is present, a small bundled sample (``data/sample_news.csv``) is used
so you can run the whole pipeline end to end immediately.

Label convention throughout the project: **1 = Reliable, 0 = Unreliable.**
"""

from __future__ import annotations

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

RELIABLE = 1
UNRELIABLE = 0

_LABEL_MAP = {
    "reliable": RELIABLE,
    "real": RELIABLE,
    "true": RELIABLE,
    "1": RELIABLE,
    "unreliable": UNRELIABLE,
    "fake": UNRELIABLE,
    "false": UNRELIABLE,
    "0": UNRELIABLE,
}


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return ``frame[name]`` as strings, or an empty column if it is absent."""
    if name in frame.columns:
        return frame[name].fillna("").astype(str)
    return pd.Series([""] * len(frame), index=frame.index)


def _combine(frame: pd.DataFrame) -> pd.Series:
    """Merge title + text columns into a single string per row."""
    return (_column(frame, "title") + ". " + _column(frame, "text")).str.strip()


def _normalize_label(value) -> int:
    key = str(value).strip().lower()
    if key not in _LABEL_MAP:
        raise ValueError(f"Unrecognized label {value!r}; expected reliable/unreliable")
    return _LABEL_MAP[key]


def load_dataset(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Return a DataFrame with two columns: ``text`` (str) and ``label`` (int)."""
    true_path = os.path.join(data_dir, "True.csv")
    fake_path = os.path.join(data_dir, "Fake.csv")

    if os.path.exists(true_path) and os.path.exists(fake_path):
        true_df = pd.read_csv(true_path)
        fake_df = pd.read_csv(fake_path)
        true_df["label"] = RELIABLE
        fake_df["label"] = UNRELIABLE
        combined = pd.concat([true_df, fake_df], ignore_index=True)
        return pd.DataFrame(
            {"text": _combine(combined), "label": combined["label"].astype(int)}
        )

    # Fall back to a single labeled CSV, or the bundled sample.
    single = os.path.join(data_dir, "news.csv")
    sample = os.path.join(data_dir, "sample_news.csv")
    path = single if os.path.exists(single) else sample

    frame = pd.read_csv(path)
    if "label" not in frame.columns:
        raise ValueError(f"{path} must contain a 'label' column")

    text = _combine(frame)
    labels = frame["label"].map(_normalize_label)
    return pd.DataFrame({"text": text, "label": labels.astype(int)})
