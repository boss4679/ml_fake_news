"""Scrape readable article text from a URL using BeautifulSoup.

This module powers real-time classification: given a news URL, it fetches the
page and extracts the main body text so the trained model can score it.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

# A desktop User-Agent avoids the trivial bot blocks that some news sites use.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}

# Tags whose text is almost always chrome (navigation, ads, scripts) rather
# than article content.
_NOISE_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form", "noscript")


@dataclass
class Article:
    """A scraped article."""

    url: str
    title: str
    text: str

    @property
    def combined(self) -> str:
        """Title + body, matching how the model is trained on labeled data."""
        return f"{self.title}. {self.text}".strip()


def scrape_article(url: str, timeout: int = 15) -> Article:
    """Download ``url`` and return the extracted :class:`Article`.

    Raises ``requests.RequestException`` on network errors and ``ValueError`` if
    no usable text could be extracted from the page.
    """
    response = requests.get(url, headers=_HEADERS, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    # Drop obvious non-content nodes before extracting text.
    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""

    # Prefer paragraph text — it is the most reliable signal of article body
    # across the huge variety of news site layouts.
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(p for p in paragraphs if p)

    # Fall back to the whole document if the page has no <p> tags.
    if not text:
        text = soup.get_text(" ", strip=True)

    if not text.strip():
        raise ValueError(f"Could not extract any article text from {url!r}")

    return Article(url=url, title=title, text=text)
