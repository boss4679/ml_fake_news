/*
 * Background service worker: owns the model and does the classification.
 *
 * Loading the ~3 MB model.json is the one expensive step, so we do it lazily and
 * cache it in memory for as long as the service worker stays alive (Chrome may
 * suspend it when idle; the next request simply reloads it).
 *
 * Per-tab results are stored in chrome.storage.session rather than a plain
 * object, so they survive the worker being suspended — otherwise the popup and
 * the toolbar badge could disagree after an idle period.
 *
 * Flow: content.js sends extracted article text -> we classify -> paint the
 * badge -> persist the result for the popup.
 */
importScripts("classifier.js"); // defines self.NewsClassifier

let classifierPromise = null; // Promise<NewsClassifier>, memoized

function getClassifier() {
  if (!classifierPromise) {
    classifierPromise = fetch(chrome.runtime.getURL("model.json"))
      .then((r) => r.json())
      .then((model) => new self.NewsClassifier(model))
      .catch((err) => {
        // Don't cache a rejected promise, or every later call fails forever.
        classifierPromise = null;
        throw err;
      });
  }
  return classifierPromise;
}

const GREEN = "#2e8b57";
const RED = "#c0392b";
const tabKey = (tabId) => `tab_${tabId}`;

function paintBadge(tabId, result) {
  if (!result) {
    chrome.action.setBadgeText({ tabId, text: "" });
    chrome.action.setTitle({ tabId, title: "No article detected on this page" });
    return;
  }
  const reliable = result.label === "Reliable";
  const pct = Math.round(result.confidence * 100);
  chrome.action.setBadgeText({ tabId, text: String(pct) });
  chrome.action.setBadgeBackgroundColor({ tabId, color: reliable ? GREEN : RED });
  chrome.action.setTitle({ tabId, title: `${result.label} — ${pct}% confidence` });
}

chrome.runtime.onMessage.addListener((msg, sender) => {
  const tabId = sender.tab && sender.tab.id;
  if (tabId == null) return;

  if (msg.type === "noArticle") {
    paintBadge(tabId, null);
    chrome.storage.session.remove(tabKey(tabId));
    return;
  }

  if (msg.type === "classify") {
    getClassifier()
      .then((clf) => {
        const combined = `${msg.title || ""}. ${msg.text || ""}`.trim();
        const out = clf.classify(combined);
        const result = {
          label: out.label,
          confidence: out.confidence,
          probReliable: out.probReliable,
          url: msg.url,
          title: msg.title,
        };
        paintBadge(tabId, result);
        chrome.storage.session.set({ [tabKey(tabId)]: result });
      })
      .catch((err) => console.error("Classification failed:", err));
  }
});

// Drop stored results when tabs close.
chrome.tabs.onRemoved.addListener((tabId) => {
  chrome.storage.session.remove(tabKey(tabId));
});
