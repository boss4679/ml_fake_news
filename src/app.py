"""Minimal Flask web app for real-time news classification.

Run::

    python -m src.app

Then open http://127.0.0.1:5000 and paste article text or a URL.
"""

from __future__ import annotations

import requests
from flask import Flask, jsonify, request

from .classify import Classifier

app = Flask(__name__)

# Load the model once at startup; fail loudly if it is missing.
try:
    _classifier: Classifier | None = Classifier()
except FileNotFoundError as exc:  # pragma: no cover - startup guard
    print(f"WARNING: {exc}")
    _classifier = None

_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>News Reliability Classifier</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: system-ui, sans-serif; max-width: 680px; margin: 3rem auto;
           padding: 0 1rem; line-height: 1.5; }
    h1 { font-size: 1.5rem; }
    textarea, input { width: 100%; box-sizing: border-box; padding: .6rem;
                      font: inherit; margin: .4rem 0; }
    textarea { min-height: 140px; }
    button { padding: .6rem 1.2rem; font: inherit; cursor: pointer; }
    .tabs { display: flex; gap: .5rem; margin-bottom: 1rem; }
    .tab { padding: .4rem .9rem; border: 1px solid #8888; border-radius: 6px;
           cursor: pointer; background: transparent; }
    .tab.active { background: #8883; font-weight: 600; }
    .result { margin-top: 1.2rem; padding: 1rem; border-radius: 8px; border: 1px solid #8888; }
    .Reliable { border-left: 6px solid #2e8b57; }
    .Unreliable { border-left: 6px solid #c0392b; }
    .muted { opacity: .7; font-size: .9rem; }
    .hidden { display: none; }
  </style>
</head>
<body>
  <h1>📰 News Reliability Classifier</h1>
  <p class="muted">TF-IDF + Logistic Regression. Paste article text, or give a URL to scrape.</p>

  <div class="tabs">
    <button class="tab active" data-mode="text" type="button">Paste text</button>
    <button class="tab" data-mode="url" type="button">From URL</button>
  </div>

  <form id="form">
    <div id="text-input">
      <textarea name="text" placeholder="Paste the article text here..."></textarea>
    </div>
    <div id="url-input" class="hidden">
      <input name="url" type="url" placeholder="https://example.com/news-story">
    </div>
    <button type="submit">Classify</button>
  </form>

  <div id="output"></div>

  <script>
    let mode = "text";
    const tabs = document.querySelectorAll(".tab");
    tabs.forEach(t => t.addEventListener("click", () => {
      mode = t.dataset.mode;
      tabs.forEach(x => x.classList.toggle("active", x === t));
      document.getElementById("text-input").classList.toggle("hidden", mode !== "text");
      document.getElementById("url-input").classList.toggle("hidden", mode !== "url");
    }));

    document.getElementById("form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const out = document.getElementById("output");
      out.innerHTML = '<p class="muted">Classifying…</p>';
      const body = mode === "text"
        ? { text: e.target.text.value }
        : { url: e.target.url.value };
      try {
        const res = await fetch("/api/classify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Request failed");
        // Build with textContent so a user-supplied URL/source can't inject markup.
        out.textContent = "";
        const box = document.createElement("div");
        box.className = "result " + data.label;
        const head = document.createElement("strong");
        head.textContent = data.label;
        const conf = document.createTextNode(
          ` — ${(data.confidence * 100).toFixed(1)}% confidence`);
        const src = document.createElement("div");
        src.className = "muted";
        src.textContent = "Source: " + data.source;
        box.append(head, conf, src);
        out.appendChild(box);
      } catch (err) {
        out.textContent = "";
        const box = document.createElement("div");
        box.className = "result Unreliable";
        box.textContent = "Error: " + err.message;
        out.appendChild(box);
      }
    });
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    # A plain static page — no template variables, so return it directly rather
    # than running it through Jinja.
    return _PAGE


@app.post("/api/classify")
def api_classify():
    if _classifier is None:
        return jsonify(error="Model not trained. Run `python -m src.train`."), 503

    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    url = (payload.get("url") or "").strip()

    try:
        if url:
            result = _classifier.classify_url(url)
        elif text:
            result = _classifier.classify_text(text)
        else:
            return jsonify(error="Provide 'text' or 'url'."), 400
    except requests.RequestException as exc:
        return jsonify(error=f"Could not fetch URL: {exc}"), 502
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    return jsonify(label=result.label, confidence=result.confidence, source=result.source)


def main() -> None:
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
