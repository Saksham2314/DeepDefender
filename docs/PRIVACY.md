# Privacy

DeepDefender's browser extension and API are designed around a simple rule:
**send only what's needed to classify a piece of visible text, and keep
everything else local.**

## What is processed, and where

| Data | Sent where | Stored where | Retention |
|---|---|---|---|
| Visible text blocks the extension decides to analyze | To the configured inference API (`http://localhost:8000` by default) as the `text` field of a `POST /predict` (and, on click-to-inspect, `POST /explain`) request | Nowhere on the server by default — the API is stateless per-request; see `api/main.py` | Not persisted server-side. The extension's own in-memory/local cache (a hash of normalized text -> last prediction) lives only in the browser and is cleared on navigation/extension reload. |
| Page URLs | **Not sent** to the API. Requests carry only extracted text strings. | N/A | N/A |
| Browsing history | **Never collected or stored** by the extension. | N/A | N/A |
| Full page HTML/DOM | **Never sent.** The content script extracts short visible text strings from specific elements; it does not serialize or transmit the page. | N/A | N/A |

## What the extension does NOT do

- It does not read or transmit form inputs, passwords, or any input-field
  content — only text rendered as page content (e.g. post/comment bodies).
- It does not send data to any third-party service. The only network target
  is the inference API URL configured in the extension's settings, which
  defaults to a `localhost` address you control.
- It does not persist analyzed text beyond an in-memory, per-tab cache used
  to avoid re-analyzing the same text block repeatedly (task section 20).
  That cache is not written to disk and does not survive a tab reload.
- It does not use analytics, telemetry, or tracking scripts.

## Local-first by design

The default and intended deployment is: DeepDefender's FastAPI server
(`api/main.py`) runs on the same machine as the browser (`localhost`), so
no text ever leaves your machine. If you choose to point the extension at a
remote API instance instead (by changing the API URL in extension
settings), you are responsible for that endpoint's own data handling —
DeepDefender does not do this by default and does not encourage it.

## Content sensitivity (task section 24)

Because this is an aggression-detection tool, the text it processes can
include insults, threats, harassment, hate speech, profanity, and other
disturbing content. Server-side logging (`src/*` and `api/main.py`) is
written to avoid printing raw analyzed text to logs — inference latency and
predicted labels are logged, not the input text itself — and any example
text that does appear in documentation or test fixtures is either
synthetic/mild or truncated.

## Your responsibilities as an operator

If you deploy the API somewhere other than `localhost` (e.g. a shared
server), you take on responsibility for that deployment's own privacy and
security posture (transport encryption, access control, log retention)
— none of that is provided by this codebase, which is built for local,
single-user research use.
