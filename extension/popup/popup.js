function sendMessage(message) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(message, (response) => {
      if (chrome.runtime.lastError) {
        resolve({ error: "extension_context_error" });
        return;
      }
      resolve(response || {});
    });
  });
}

function setStatus(status) {
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  dot.className = "status-dot";
  if (status === "online") {
    dot.classList.add("status-online");
    text.textContent = "Model Online";
  } else if (status === "loading") {
    dot.classList.add("status-loading");
    text.textContent = "Model Loading...";
  } else if (status === "offline") {
    dot.classList.add("status-offline");
    text.textContent = "API Offline";
  } else {
    dot.classList.add("status-unknown");
    text.textContent = "Unknown";
  }
}

async function refreshHealth() {
  const health = await sendMessage({ type: "GET_HEALTH", force: true });
  setStatus(health.status || "unknown");
}

async function refreshStats() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;
  const stats = await sendMessage({ type: "GET_TAB_STATS", tabId: tab.id });
  document.getElementById("stat-analyzed").textContent = stats.analyzed || 0;
  document.getElementById("stat-nag").textContent = stats.NAG || 0;
  document.getElementById("stat-cag").textContent = stats.CAG || 0;
  document.getElementById("stat-oag").textContent = stats.OAG || 0;
}

async function loadSettings() {
  const settings = await sendMessage({ type: "GET_SETTINGS" });
  document.getElementById("setting-enabled").checked = !!settings.enabled;
  document.getElementById("setting-highlights").checked = !!settings.showHighlights;
  document.getElementById("setting-explanations").checked = !!settings.showExplanations;
  document.getElementById("setting-dynamic").checked = !!settings.analyzeDynamic;
  const thresholdPct = Math.round((settings.confidenceThreshold || 0.7) * 100);
  document.getElementById("setting-threshold").value = thresholdPct;
  document.getElementById("threshold-value").textContent = `${thresholdPct}%`;
  document.getElementById("api-url-label").textContent = `API: ${settings.apiBaseUrl}`;
}

function wireSettingControls() {
  const enabled = document.getElementById("setting-enabled");
  const highlights = document.getElementById("setting-highlights");
  const explanations = document.getElementById("setting-explanations");
  const dynamic = document.getElementById("setting-dynamic");
  const threshold = document.getElementById("setting-threshold");
  const thresholdValue = document.getElementById("threshold-value");

  enabled.addEventListener("change", () => sendMessage({ type: "SET_SETTINGS", payload: { enabled: enabled.checked } }));
  highlights.addEventListener("change", () => sendMessage({ type: "SET_SETTINGS", payload: { showHighlights: highlights.checked } }));
  explanations.addEventListener("change", () => sendMessage({ type: "SET_SETTINGS", payload: { showExplanations: explanations.checked } }));
  dynamic.addEventListener("change", () => sendMessage({ type: "SET_SETTINGS", payload: { analyzeDynamic: dynamic.checked } }));
  threshold.addEventListener("input", () => {
    thresholdValue.textContent = `${threshold.value}%`;
  });
  threshold.addEventListener("change", () => {
    sendMessage({ type: "SET_SETTINGS", payload: { confidenceThreshold: Number(threshold.value) / 100 } });
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  wireSettingControls();
  await loadSettings();
  await refreshHealth();
  await refreshStats();
});
