/*
 * Content script: runs on every page you open. It extracts the main article
 * text with Mozilla's Readability, then asks the background service worker to
 * classify it. The heavy lifting (loading the 3 MB model, running the math)
 * lives in the background so the model is loaded once and reused, not re-parsed
 * on every page.
 *
 * Readability is loaded before this file (see manifest content_scripts).
 */
(function () {
  "use strict";

  const MIN_ARTICLE_CHARS = 400; // below this it's probably not an article

  function extractArticle() {
    try {
      // Readability mutates the document, so hand it a clone.
      const clone = document.cloneNode(true);
      const parsed = new Readability(clone).parse();
      if (parsed && parsed.textContent && parsed.textContent.trim().length >= MIN_ARTICLE_CHARS) {
        return { title: parsed.title || document.title || "", text: parsed.textContent };
      }
    } catch (e) {
      // Readability throws on some odd DOMs; fall through to the heuristic.
    }

    // Fallback: concatenate paragraph text, mirroring the Python scraper.
    const paras = Array.from(document.querySelectorAll("p"))
      .map((p) => p.innerText.trim())
      .filter(Boolean);
    const text = paras.join(" ");
    if (text.length >= MIN_ARTICLE_CHARS) {
      return { title: document.title || "", text };
    }
    return null;
  }

  const article = extractArticle();

  if (!article) {
    chrome.runtime.sendMessage({ type: "noArticle", url: location.href });
    return;
  }

  chrome.runtime.sendMessage({
    type: "classify",
    url: location.href,
    title: article.title,
    // Cap payload size — the tail of a long article rarely changes the verdict.
    text: article.text.slice(0, 20000),
  });
})();
