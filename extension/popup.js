/*
 * Popup: read this tab's result from chrome.storage.session and render it.
 *
 * The result is written asynchronously by the background worker after it
 * fetches and runs the model, so if the popup opens first we show "Analyzing…"
 * and subscribe to storage changes to update the moment the result lands.
 */
(function () {
  "use strict";
  const el = document.getElementById("content");

  // Build the result view with textContent for any page-derived strings
  // (title/url), so a hostile page title can't inject markup into the popup.
  function render(result) {
    el.textContent = "";

    if (!result) {
      const p = document.createElement("div");
      p.className = "empty";
      p.textContent = "No news article detected on this page.";
      el.appendChild(p);
      return;
    }

    const pct = Math.round(result.confidence * 100);
    const reliablePct = Math.round((result.probReliable ?? 0) * 100);
    const color = result.label === "Reliable" ? "#2e8b57" : "#c0392b";

    const verdict = document.createElement("div");
    verdict.className = `verdict ${result.label}`;
    verdict.textContent = result.label;

    const conf = document.createElement("div");
    conf.className = "conf";
    conf.textContent = `${pct}% confidence`;

    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("span");
    fill.style.width = `${reliablePct}%`;
    fill.style.background = color;
    bar.appendChild(fill);

    const prob = document.createElement("div");
    prob.className = "conf";
    prob.textContent = `P(reliable) = ${reliablePct}%`;

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = (result.title || result.url || "").slice(0, 120);

    el.append(verdict, conf, bar, prob, meta);
  }

  function showAnalyzing() {
    el.textContent = "";
    const p = document.createElement("div");
    p.className = "empty";
    p.textContent = "Analyzing…";
    el.appendChild(p);
  }

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tab = tabs[0];
    if (!tab) {
      render(null);
      return;
    }
    const key = `tab_${tab.id}`;

    chrome.storage.session.get(key, (data) => {
      if (chrome.runtime.lastError || data[key] === undefined) {
        showAnalyzing();
      } else {
        render(data[key]);
      }
    });

    // Update live if the result arrives (or changes) while the popup is open.
    chrome.storage.session.onChanged.addListener((changes) => {
      if (changes[key]) render(changes[key].newValue ?? null);
    });
  });
})();
