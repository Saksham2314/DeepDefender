/**
 * Real (not mocked-away) test of extension/background/service-worker.js's
 * message handling, using a minimal mock `chrome` API and mocked fetch.
 * No test framework dependency -- plain Node assertions, run with:
 *
 *   node tests/extension/test_service_worker.js
 *
 * Exits non-zero if any assertion fails, so it can be wired into CI later.
 */

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const SCRIPT_PATH = path.join(__dirname, "..", "..", "extension", "background", "service-worker.js");

let results = [];
let failures = 0;

function test(name, fn) {
  results.push({ name, fn });
}

async function runAll() {
  for (const { name, fn } of results) {
    try {
      await fn();
      console.log(`PASS: ${name}`);
    } catch (e) {
      failures += 1;
      console.error(`FAIL: ${name}`);
      console.error(`      ${e && e.stack ? e.stack : e}`);
    }
  }
  console.log(`\n${results.length - failures}/${results.length} passed`);
  process.exit(failures > 0 ? 1 : 0);
}

// --- sandbox / mock chrome setup ------------------------------------------

function buildSandbox(fetchImpl) {
  const storage = {};
  let messageListener = null;

  const chrome = {
    storage: {
      local: {
        get: (key) => Promise.resolve({ [key]: storage[key] }),
        set: (obj) => {
          Object.assign(storage, obj);
          return Promise.resolve();
        },
      },
    },
    runtime: {
      onMessage: {
        addListener: (fn) => {
          messageListener = fn;
        },
      },
    },
    tabs: {
      onRemoved: { addListener: () => {} },
    },
    webNavigation: {
      onCommitted: { addListener: () => {} },
    },
  };

  const sandbox = {
    chrome,
    fetch: fetchImpl,
    AbortController: global.AbortController,
    setTimeout: global.setTimeout,
    clearTimeout: global.clearTimeout,
    console,
    Map,
    Set,
    Promise,
    JSON,
    Date,
    String,
    Object,
    Array,
  };
  vm.createContext(sandbox);
  const code = fs.readFileSync(SCRIPT_PATH, "utf-8");
  vm.runInContext(code, sandbox, { filename: SCRIPT_PATH });

  function callListener(message, sender = {}) {
    return new Promise((resolve) => {
      const keepAlive = sandbox.chrome.runtime.onMessage._invoke(message, sender, resolve);
    });
  }

  // Wrap addListener so we can invoke the captured listener with a
  // promise-friendly sendResponse.
  chrome.runtime.onMessage._invoke = (message, sender, resolve) => {
    return messageListener(message, sender, resolve);
  };

  return { sandbox, storage, callListener };
}

function jsonResponse(status, body) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  });
}

// --- tests -------------------------------------------------------------

test("GET_SETTINGS returns defaults when nothing stored", async () => {
  const { callListener } = buildSandbox(() => Promise.reject(new Error("should not fetch")));
  const settings = await callListener({ type: "GET_SETTINGS" });
  assert.strictEqual(settings.enabled, true);
  assert.strictEqual(settings.confidenceThreshold, 0.7);
  assert.strictEqual(settings.apiBaseUrl, "http://localhost:8000");
});

test("SET_SETTINGS persists and GET_SETTINGS reflects the change", async () => {
  const { callListener } = buildSandbox(() => Promise.reject(new Error("should not fetch")));
  await callListener({ type: "SET_SETTINGS", payload: { confidenceThreshold: 0.9 } });
  const settings = await callListener({ type: "GET_SETTINGS" });
  assert.strictEqual(settings.confidenceThreshold, 0.9);
});

test("ANALYZE_TEXT returns a valid prediction and caches it (no second fetch for same hash)", async () => {
  let fetchCalls = 0;
  const fetchImpl = (url) => {
    fetchCalls += 1;
    if (url.endsWith("/predict")) {
      return jsonResponse(200, { label: "OAG", label_name: "Overtly Aggressive", probabilities: { NAG: 0.05, CAG: 0.1, OAG: 0.85 }, confidence: 0.85 });
    }
    return jsonResponse(404, {});
  };
  const { callListener } = buildSandbox(fetchImpl);

  const first = await callListener({ type: "ANALYZE_TEXT", text: "tum bahut bakwas ho", textHash: "abc123" }, { tab: { id: 1 } });
  assert.strictEqual(first.label, "OAG");
  assert.strictEqual(fetchCalls, 1);

  const second = await callListener({ type: "ANALYZE_TEXT", text: "tum bahut bakwas ho", textHash: "abc123" }, { tab: { id: 1 } });
  assert.strictEqual(second.label, "OAG");
  assert.strictEqual(second.cached, true);
  assert.strictEqual(fetchCalls, 1, "second call with same hash must not re-fetch");
});

test("ANALYZE_TEXT maps HTTP 503 to error 'model_unavailable'", async () => {
  const fetchImpl = () => jsonResponse(503, { detail: "loading" });
  const { callListener } = buildSandbox(fetchImpl);
  const result = await callListener({ type: "ANALYZE_TEXT", text: "hello world text", textHash: "h1" });
  assert.strictEqual(result.error, "model_unavailable");
});

test("ANALYZE_TEXT maps a network failure to error 'network_error'", async () => {
  const fetchImpl = () => Promise.reject(new Error("ECONNREFUSED"));
  const { callListener } = buildSandbox(fetchImpl);
  const result = await callListener({ type: "ANALYZE_TEXT", text: "hello world text", textHash: "h2" });
  assert.strictEqual(result.error, "network_error");
});

test("ANALYZE_TEXT rejects a malformed API response instead of crashing", async () => {
  const fetchImpl = () => jsonResponse(200, { not_a_label: true });
  const { callListener } = buildSandbox(fetchImpl);
  const result = await callListener({ type: "ANALYZE_TEXT", text: "hello world text", textHash: "h3" });
  assert.strictEqual(result.error, "malformed_response");
});

test("GET_TAB_STATS starts at zero and increments after a successful ANALYZE_TEXT", async () => {
  const fetchImpl = (url) => {
    if (url.endsWith("/predict")) {
      return jsonResponse(200, { label: "CAG", label_name: "Covertly Aggressive", probabilities: { NAG: 0.2, CAG: 0.6, OAG: 0.2 }, confidence: 0.6 });
    }
    return jsonResponse(404, {});
  };
  const { callListener } = buildSandbox(fetchImpl);

  const before = await callListener({ type: "GET_TAB_STATS", tabId: 42 });
  assert.strictEqual(before.analyzed, 0);

  await callListener({ type: "ANALYZE_TEXT", text: "some covertly aggressive text", textHash: "h4" }, { tab: { id: 42 } });

  const after = await callListener({ type: "GET_TAB_STATS", tabId: 42 });
  assert.strictEqual(after.analyzed, 1);
  assert.strictEqual(after.CAG, 1);
});

test("Unknown message type returns an error instead of hanging or throwing", async () => {
  const { callListener } = buildSandbox(() => Promise.reject(new Error("should not fetch")));
  const result = await callListener({ type: "SOMETHING_MADE_UP" });
  assert.strictEqual(result.error, "unknown_message_type");
});

runAll();
