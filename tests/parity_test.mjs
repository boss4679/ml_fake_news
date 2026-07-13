/*
 * Step 2 of the parity test: run the browser JS classifier on the same texts
 * and assert it matches the Python model within floating-point tolerance.
 *
 *   python -m tests.gen_expected   # writes tests/expected.json
 *   node tests/parity_test.mjs
 *
 * Exits non-zero if any prediction diverges by more than 1e-6.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { NewsClassifier } = require(path.join(__dirname, "..", "extension", "classifier.js"));

const model = JSON.parse(
  fs.readFileSync(path.join(__dirname, "..", "extension", "model.json"), "utf8")
);
const expectedPath = path.join(__dirname, "expected.json");
if (!fs.existsSync(expectedPath)) {
  console.error("Missing tests/expected.json — run `python -m tests.gen_expected` first.");
  process.exit(2);
}
const expected = JSON.parse(fs.readFileSync(expectedPath, "utf8"));

const clf = new NewsClassifier(model);
const TOL = 1e-6;
let maxDiff = 0;
let failures = 0;

for (const row of expected) {
  const js = clf.classify(row.text).probReliable;
  const diff = Math.abs(js - row.prob_reliable);
  maxDiff = Math.max(maxDiff, diff);
  if (diff > TOL) failures++;
  const flag = diff > TOL ? "  <-- MISMATCH" : "";
  console.log(
    `py=${row.prob_reliable.toFixed(8)}  js=${js.toFixed(8)}  ` +
      `diff=${diff.toExponential(2)}${flag}  | ${row.text.slice(0, 40).replace(/\s+/g, " ")}`
  );
}

console.log(`\nMax abs diff: ${maxDiff.toExponential(3)}   mismatches(>${TOL}): ${failures}`);
if (failures > 0) {
  console.error("PARITY FAILED");
  process.exit(1);
}
console.log("PARITY OK — JS classifier matches the Python model.");
