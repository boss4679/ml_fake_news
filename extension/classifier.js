/*
 * Pure-JavaScript reimplementation of the trained scikit-learn pipeline
 * (TfidfVectorizer + LogisticRegression), so classification runs entirely in
 * the browser with no server and no ML runtime.
 *
 * It must reproduce sklearn's feature transform EXACTLY, or predictions will
 * drift from the Python model. The steps below mirror, in order:
 *   1. our preprocess.clean_text()
 *   2. TfidfVectorizer's default tokenizer (token_pattern \b\w\w+\b)
 *   3. English stop-word removal (applied BEFORE building n-grams, like sklearn)
 *   4. n-gram generation
 *   5. sublinear TF, multiply by IDF, L2-normalize
 *   6. logistic regression: dot(features, coef) + intercept -> sigmoid
 *
 * Works both in the browser (attaches window.NewsClassifier) and in Node
 * (module.exports) so the same file can be parity-tested against Python.
 */
(function (root) {
  "use strict";

  // --- mirror of src/preprocess.py: clean_text -------------------------------
  const URL_RE = /https?:\/\/\S+|www\.\S+/g;
  const EMAIL_RE = /\S+@\S+/g;
  const NON_ALPHA_RE = /[^a-z\s]/g;
  const MULTISPACE_RE = /\s+/g;

  function cleanText(text) {
    if (typeof text !== "string") return "";
    text = text.toLowerCase();
    text = text.replace(URL_RE, " ");
    text = text.replace(EMAIL_RE, " ");
    text = text.replace(NON_ALPHA_RE, " ");
    text = text.replace(MULTISPACE_RE, " ");
    return text.trim();
  }

  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  class NewsClassifier {
    constructor(model) {
      this.model = model;
      this.stopWords = new Set(model.stop_words);
      this.minLen = model.config.min_token_len || 2;
      const [lo, hi] = model.config.ngram_range || [1, 1];
      this.ngramLo = lo;
      this.ngramHi = hi;
      this.sublinear = !!model.config.sublinear_tf;
      this.l2 = model.config.norm === "l2";
      this.reliableIdx = model.reliable_class_index;
    }

    // Tokenize like sklearn's default analyzer, then drop stop words. After
    // clean_text the string is only [a-z ] so whitespace-splitting with a
    // min-length filter is equivalent to the \b\w\w+\b token pattern.
    _tokens(text) {
      const raw = cleanText(text).split(" ");
      const out = [];
      for (const tok of raw) {
        if (tok.length >= this.minLen && !this.stopWords.has(tok)) out.push(tok);
      }
      return out;
    }

    // Build 1..n-grams from the (already stop-word-filtered) token list,
    // exactly as sklearn's _word_ngrams does.
    _ngrams(tokens) {
      const grams = [];
      for (let n = this.ngramLo; n <= this.ngramHi; n++) {
        if (n === 1) {
          for (const t of tokens) grams.push(t);
        } else {
          for (let i = 0; i + n <= tokens.length; i++) {
            grams.push(tokens.slice(i, i + n).join(" "));
          }
        }
      }
      return grams;
    }

    // Return { label, confidence, probReliable } for a raw article string.
    classify(text) {
      const { vocabulary, idf, coef, intercept } = this.model;

      // Count only n-grams that exist in the trained vocabulary.
      const counts = new Map(); // feature index -> raw count
      for (const gram of this._ngrams(this._tokens(text))) {
        const idx = vocabulary[gram];
        if (idx !== undefined) counts.set(idx, (counts.get(idx) || 0) + 1);
      }

      // tf-idf values, then L2 norm — the norm is over vocab features only,
      // matching sklearn's sparse transform.
      const values = new Map(); // index -> tf-idf value
      let sumSq = 0;
      for (const [idx, count] of counts) {
        const tf = this.sublinear ? 1 + Math.log(count) : count;
        const v = tf * idf[idx];
        values.set(idx, v);
        sumSq += v * v;
      }
      const norm = this.l2 && sumSq > 0 ? Math.sqrt(sumSq) : 1;

      // decision = intercept + sum(normalized_value * coef)
      let score = intercept;
      for (const [idx, v] of values) {
        score += (v / norm) * coef[idx];
      }

      // coef row corresponds to class 1; if class 1 is "Reliable", sigmoid of
      // the score is P(Reliable) directly.
      let probReliable = sigmoid(score);
      if (this.reliableIdx === 0) probReliable = 1 - probReliable;

      const label = probReliable >= 0.5 ? "Reliable" : "Unreliable";
      const confidence = label === "Reliable" ? probReliable : 1 - probReliable;
      return { label, confidence, probReliable };
    }
  }

  const api = { NewsClassifier, cleanText };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api; // Node (parity test)
  } else {
    root.NewsClassifier = NewsClassifier; // browser
    root.NewsClassifierUtils = api;
  }
})(typeof self !== "undefined" ? self : this);
