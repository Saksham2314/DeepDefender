/**
 * DeepDefender background service worker (Manifest V3).
 *
 * Centralizes everything the content script shouldn't do itself:
 *   - the actual fetch() calls to the local inference API (host_permissions
 *     are granted here, and centralizing avoids each content script
 *     instance managing its own CORS/network state),
 *   - a request queue with bounded concurrency + a minimum gap between
 *     requests (task section 20: debouncing/throttling, don't hammer the
 *     API),
 *   - a text-hash -> result cache so the same normalized text is never
 *     re-sent (task section 20),
 *   - per-tab analyzed/NAG/CAG/OAG counts for the popup,
 *   - settings storage (chrome.storage.local; per docs/PRIVACY.md this
 *     never leaves the device).
 *
 * Service workers in MV3 can be killed and restarted by the browser at any
 * time, so nothing here is assumed to survive forever -- the cache and
 * per-tab stats are best-effort, in-memory only, and rebuilding them is
 * cheap (a fresh /predict call).
 */

const DEFAULT_SETTINGS = {
  enabled: true,
  showHighlights: true,
  showExplanations: true,
  analyzeDynamic: true,
  confidenceThreshold: 0.7,
  apiBaseUrl: "http://localhost:8000",
};

const MAX_CACHE_ENTRIES = 1000;
const MAX_CONCURRENT_REQUESTS = 2;
const MIN_REQUEST_GAP_MS = 120;
const FETCH_TIMEOUT_MS = 8000;
const HEALTH_CACHE_MS = 5000;

/** @type {Map<string, object>} textHash -> prediction result */
const resultCache = new Map();

/** @type {Map<number, {analyzed:number, NAG:number, CAG:number, OAG:number}>} tabId -> stats */
const tabStats = new Map();

let settingsCache = null;
let lastHealth = { status: "unknown", checkedAt: 0 };

// --- simple bounded-concurrency queue -------------------------------------
let inFlight = 0;
let lastRequestAt = 0;
const queue = [];

function runQueue() {
  if (inFlight >= MAX_CONCURRENT_REQUESTS || queue.length === 0) return;
  const now = Date.now();
  const wait = Math.max(0, lastRequestAt + MIN_REQUEST_GAP_MS - now);
  setTimeout(() => {
    if (inFlight >= MAX_CONCURRENT_REQUESTS || queue.length === 0) return;
    const job = queue.shift();
    inFlight += 1;
    lastRequestAt = Date.now();
    job().finally(() => {
      inFlight -= 1;
      runQueue();
    });
    runQueue();
  }, wait);
}

function enqueue(job) {
  return new Promise((resolve, reject) => {
    queue.push(() => job().then(resolve).catch(reject));
    runQueue();
  });
}

async function fetchWithTimeout(url, options) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    return response;
  } finally {
    clearTimeout(id);
  }
}

// --- settings ---------------------------------------------------------------
async function getSettings() {
  if (settingsCache) return settingsCache;
  try {
    const stored = await chrome.storage.local.get("deepdefender_settings");
    settingsCache = { ...DEFAULT_SETTINGS, ...(stored.deepdefender_settings || {}) };
  } catch (e) {
    settingsCache = { ...DEFAULT_SETTINGS };
  }
  return settingsCache;
}

async function setSettings(partial) {
  const current = await getSettings();
  settingsCache = { ...current, ...partial };
  try {
    await chrome.storage.local.set({ deepdefender_settings: settingsCache });
  } catch (e) {
    // storage unavailable (e.g. quota / permission issue) -- keep in-memory
    // settings working regardless, never throw up to the caller.
  }
  return settingsCache;
}

// --- health -------------------------------------------------------------
async function checkHealth(force) {
  const now = Date.now();
  if (!force && now - lastHealth.checkedAt < HEALTH_CACHE_MS) {
    return lastHealth;
  }
  const settings = await getSettings();
  try {
    const response = await fetchWithTimeout(`${settings.apiBaseUrl}/health`, { method: "GET" });
    if (!response.ok) {
      lastHealth = { status: "error", checkedAt: now, httpStatus: response.status };
      return lastHealth;
    }
    const body = await response.json();
    lastHealth = {
      status: body.model_loaded ? "online" : "loading",
      checkedAt: now,
      raw: body,
    };
  } catch (e) {
    lastHealth = { status: "offline", checkedAt: now, error: String(e && e.message || e) };
  }
  return lastHealth;
}

// --- prediction -----------------------------------------------------------
async function predict(text, textHash) {
  if (resultCache.has(textHash)) {
    return { ...resultCache.get(textHash), cached: true };
  }

  const settings = await getSettings();
  const result = await enqueue(async () => {
    try {
      const response = await fetchWithTimeout(`${settings.apiBaseUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (response.status === 503) {
        return { error: "model_unavailable" };
      }
      if (!response.ok) {
        let detail = "";
        try {
          const body = await response.json();
          detail = body && body.detail ? body.detail : "";
        } catch (_) {
          // malformed/non-JSON error body -- ignore, use generic message
        }
        return { error: "api_error", detail };
      }
      const body = await response.json();
      if (!body || typeof body.label !== "string" || typeof body.confidence !== "number") {
        // malformed API response shape -- don't crash the content script on it
        return { error: "malformed_response" };
      }
      return body;
    } catch (e) {
      return { error: "network_error", detail: String(e && e.message || e) };
    }
  });

  if (!result.error) {
    if (resultCache.size >= MAX_CACHE_ENTRIES) {
      const oldestKey = resultCache.keys().next().value;
      resultCache.delete(oldestKey);
    }
    resultCache.set(textHash, result);
  }
  return result;
}

async function explain(text, method, topK) {
  const settings = await getSettings();
  try {
    const response = await fetchWithTimeout(`${settings.apiBaseUrl}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, method: method || "occlusion", top_k: topK || 5 }),
    });
    if (!response.ok) {
      return { error: "api_error", status: response.status };
    }
    return await response.json();
  } catch (e) {
    return { error: "network_error", detail: String(e && e.message || e) };
  }
}

// --- per-tab stats ----------------------------------------------------------
function getTabStats(tabId) {
  if (!tabStats.has(tabId)) {
    tabStats.set(tabId, { analyzed: 0, NAG: 0, CAG: 0, OAG: 0 });
  }
  return tabStats.get(tabId);
}

function recordPrediction(tabId, label) {
  const stats = getTabStats(tabId);
  stats.analyzed += 1;
  if (label in stats) stats[label] += 1;
}

chrome.tabs.onRemoved.addListener((tabId) => tabStats.delete(tabId));
chrome.webNavigation?.onCommitted?.addListener((details) => {
  if (details.frameId === 0) {
    // New top-level navigation (including SPA history navigations where
    // supported) -- reset this tab's stats for the "new page".
    tabStats.set(details.tabId, { analyzed: 0, NAG: 0, CAG: 0, OAG: 0 });
  }
});

// --- message handling ---------------------------------------------------
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const tabId = sender.tab ? sender.tab.id : null;

  (async () => {
    try {
      switch (message.type) {
        case "GET_SETTINGS": {
          sendResponse(await getSettings());
          break;
        }
        case "SET_SETTINGS": {
          sendResponse(await setSettings(message.payload || {}));
          break;
        }
        case "GET_HEALTH": {
          sendResponse(await checkHealth(!!message.force));
          break;
        }
        case "ANALYZE_TEXT": {
          const settings = await getSettings();
          if (!settings.enabled) {
            sendResponse({ error: "disabled" });
            break;
          }
          const result = await predict(message.text, message.textHash);
          if (tabId !== null && !result.error) {
            recordPrediction(tabId, result.label);
          }
          sendResponse(result);
          break;
        }
        case "EXPLAIN_TEXT": {
          const result = await explain(message.text, message.method, message.topK);
          sendResponse(result);
          break;
        }
        case "GET_TAB_STATS": {
          const targetTabId = message.tabId !== undefined ? message.tabId : tabId;
          sendResponse(getTabStats(targetTabId));
          break;
        }
        default: {
          sendResponse({ error: "unknown_message_type" });
        }
      }
    } catch (e) {
      // Absolute last resort: never leave the content script/popup hanging
      // without a response, and never throw an uncaught error out of the
      // service worker.
      sendResponse({ error: "internal_error", detail: String(e && e.message || e) });
    }
  })();

  return true; // keep the message channel open for the async response
});
