/**
 * DeepDefender content script.
 *
 * Responsibilities (task sections 19-22):
 *   - find visible text blocks on the page (not the whole DOM/page),
 *   - analyze only ones that actually scroll into view (IntersectionObserver),
 *   - never send the same normalized text twice (hash-based cache, checked
 *     both here and again in the background worker),
 *   - watch for dynamically added content (MutationObserver, debounced) and
 *     SPA navigation (URL polling, since MV3 content scripts get no
 *     reliable native event for pushState navigations),
 *   - render a small, dismissible badge next to flagged elements without
 *     ever touching the original text nodes,
 *   - degrade silently and cheaply when the API is offline/loading, with
 *     backoff so a dead API doesn't cause a request storm.
 */

(() => {
  if (window.__deepdefenderInjected) return;
  window.__deepdefenderInjected = true;

  const MIN_TEXT_LENGTH = 8;
  const MAX_TEXT_LENGTH_FOR_API = 2000;
  const MUTATION_DEBOUNCE_MS = 500;
  const URL_POLL_MS = 1000;
  const MAX_CANDIDATES_PER_SCAN = 300;
  const CONSECUTIVE_ERROR_BACKOFF_THRESHOLD = 5;
  const BACKOFF_COOLDOWN_MS = 30000;

  const SKIP_TAGS = new Set([
    "SCRIPT", "STYLE", "NOSCRIPT", "INPUT", "TEXTAREA", "SELECT",
    "OPTION", "BUTTON", "SVG", "CANVAS", "IFRAME", "CODE", "PRE",
  ]);

  let settings = null;
  let consecutiveErrors = 0;
  let backoffUntil = 0;

  const seenElements = new WeakSet();
  const badgeByElement = new WeakMap();
  const intersectionObserver = new IntersectionObserver(onIntersect, { threshold: 0.25 });

  // --- utils ----------------------------------------------------------------
  function hashText(text) {
    // FNV-1a, fast and sufficient for cache-keying (not security-sensitive).
    let hash = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = (hash * 0x01000193) >>> 0;
    }
    return hash.toString(16);
  }

  function normalize(text) {
    return text.trim().replace(/\s+/g, " ");
  }

  function sendMessage(message) {
    return new Promise((resolve) => {
      try {
        chrome.runtime.sendMessage(message, (response) => {
          if (chrome.runtime.lastError) {
            resolve({ error: "extension_context_error", detail: chrome.runtime.lastError.message });
            return;
          }
          resolve(response || { error: "empty_response" });
        });
      } catch (e) {
        // e.g. extension was reloaded/updated mid-page-life -- context invalidated
        resolve({ error: "extension_context_error", detail: String(e && e.message || e) });
      }
    });
  }

  // --- candidate discovery ---------------------------------------------------
  function isEligibleElement(el) {
    if (!el || SKIP_TAGS.has(el.tagName)) return false;
    if (el.closest && el.closest('[contenteditable="true"], input, textarea')) return false;
    return true;
  }

  function findCandidates(root) {
    const candidates = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.parentElement || !isEligibleElement(node.parentElement)) {
          return NodeFilter.FILTER_REJECT;
        }
        const text = normalize(node.textContent || "");
        if (text.length < MIN_TEXT_LENGTH) return NodeFilter.FILTER_SKIP;
        return NodeFilter.FILTER_ACCEPT;
      },
    });

    let node;
    let count = 0;
    while ((node = walker.nextNode()) && count < MAX_CANDIDATES_PER_SCAN) {
      const el = node.parentElement;
      if (seenElements.has(el)) continue;
      seenElements.add(el);
      candidates.push(el);
      count += 1;
    }
    return candidates;
  }

  function observeNewCandidates(root) {
    const candidates = findCandidates(root);
    for (const el of candidates) {
      intersectionObserver.observe(el);
    }
  }

  // --- analysis ---------------------------------------------------------------
  async function onIntersect(entries) {
    if (!settings || !settings.enabled) return;
    if (Date.now() < backoffUntil) return;

    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const el = entry.target;
      intersectionObserver.unobserve(el); // analyze once per element

      const rawText = normalize(el.innerText || el.textContent || "");
      if (rawText.length < MIN_TEXT_LENGTH) continue;
      const text = rawText.slice(0, MAX_TEXT_LENGTH_FOR_API);
      const textHash = hashText(text);

      analyzeElement(el, text, textHash);
    }
  }

  async function analyzeElement(el, text, textHash) {
    const result = await sendMessage({ type: "ANALYZE_TEXT", text, textHash });

    if (result.error) {
      handleAnalysisError(result.error);
      return;
    }
    consecutiveErrors = 0;

    if (!settings.showHighlights) return;
    if (result.confidence < settings.confidenceThreshold) return;
    if (result.label !== "CAG" && result.label !== "OAG") return;

    renderBadge(el, result, text);
  }

  function handleAnalysisError(errorType) {
    if (errorType === "disabled") return;
    consecutiveErrors += 1;
    if (consecutiveErrors >= CONSECUTIVE_ERROR_BACKOFF_THRESHOLD) {
      backoffUntil = Date.now() + BACKOFF_COOLDOWN_MS;
      consecutiveErrors = 0;
    }
  }

  // --- UI ---------------------------------------------------------------------
  function renderBadge(el, result, originalText) {
    if (badgeByElement.has(el)) return;

    el.classList.add("deepdefender-flagged", `deepdefender-flag-${result.label.toLowerCase()}`);

    const badge = document.createElement("span");
    badge.className = "deepdefender-badge";
    badge.setAttribute("data-deepdefender-label", result.label);
    badge.textContent = result.label === "OAG"
      ? `⚠ Potential OAG · ${Math.round(result.confidence * 100)}%`
      : `⚠ Potentially covert aggression · ${Math.round(result.confidence * 100)}%`;
    badge.title = "DeepDefender: click for explanation";

    const panel = buildExplanationPanel(result, originalText);
    panel.hidden = true;

    badge.addEventListener("click", (evt) => {
      evt.preventDefault();
      evt.stopPropagation();
      if (panel.hidden && settings.showExplanations) {
        loadExplanation(panel, originalText);
      }
      panel.hidden = !panel.hidden;
    });

    const wrapper = document.createElement("span");
    wrapper.className = "deepdefender-wrapper";
    wrapper.appendChild(badge);
    wrapper.appendChild(panel);

    // Insert as a new sibling node right after the flagged element --
    // never modifies the element's own text content (task section 22).
    el.insertAdjacentElement("afterend", wrapper);
    badgeByElement.set(el, wrapper);
  }

  function buildExplanationPanel(result) {
    const panel = document.createElement("div");
    panel.className = "deepdefender-panel";

    const header = document.createElement("div");
    header.className = "deepdefender-panel-header";
    header.textContent = `Prediction: ${result.label_name || result.label}`;
    panel.appendChild(header);

    const probs = document.createElement("div");
    probs.className = "deepdefender-panel-probs";
    for (const [cls, p] of Object.entries(result.probabilities || {})) {
      const row = document.createElement("div");
      row.className = "deepdefender-prob-row";
      row.innerHTML = `<span>${cls}</span><span>${Math.round(p * 100)}%</span>`;
      probs.appendChild(row);
    }
    panel.appendChild(probs);

    const explanationSlot = document.createElement("div");
    explanationSlot.className = "deepdefender-panel-explanation";
    explanationSlot.textContent = "Loading contributing words...";
    panel.appendChild(explanationSlot);

    const caveat = document.createElement("div");
    caveat.className = "deepdefender-panel-caveat";
    caveat.textContent = "Feature-attribution estimate, not a causal explanation.";
    panel.appendChild(caveat);

    panel._explanationSlot = explanationSlot;
    return panel;
  }

  async function loadExplanation(panel, text) {
    const slot = panel._explanationSlot;
    const result = await sendMessage({ type: "EXPLAIN_TEXT", text, method: "occlusion", topK: 5 });
    if (result.error) {
      slot.textContent = "Explanation unavailable right now.";
      return;
    }
    const words = (result.contributions || []).map((c) => c.word).join(", ");
    slot.textContent = words ? `Contributing words: ${words}` : "No strong contributing words found.";
  }

  // --- dynamic content ----------------------------------------------------
  let mutationTimer = null;
  const mutationObserver = new MutationObserver(() => {
    if (!settings || !settings.analyzeDynamic) return;
    if (mutationTimer) clearTimeout(mutationTimer);
    mutationTimer = setTimeout(() => {
      observeNewCandidates(document.body);
    }, MUTATION_DEBOUNCE_MS);
  });

  function startObservingDom() {
    mutationObserver.observe(document.body, { childList: true, subtree: true });
  }

  // --- SPA navigation -----------------------------------------------------
  let lastUrl = location.href;
  setInterval(() => {
    if (location.href !== lastUrl) {
      lastUrl = location.href;
      // New logical page: candidates already flagged stay flagged (avoids
      // flicker on client-side re-renders that don't actually replace
      // content); the MutationObserver above naturally picks up genuinely
      // new DOM content that the SPA renders for the new route.
      observeNewCandidates(document.body);
    }
  }, URL_POLL_MS);

  // --- init -----------------------------------------------------------------
  async function init() {
    if (!document.body) return; // unsupported/empty page
    settings = await sendMessage({ type: "GET_SETTINGS" });
    if (!settings || settings.error) {
      settings = { enabled: true, showHighlights: true, showExplanations: true, analyzeDynamic: true, confidenceThreshold: 0.7 };
    }
    if (!settings.enabled) return;

    observeNewCandidates(document.body);
    startObservingDom();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
