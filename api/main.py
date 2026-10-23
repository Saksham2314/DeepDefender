"""DeepDefender inference API (task section 17).

    uvicorn api.main:app --reload

The StackingPredictor (3 base transformers + meta-classifier) is loaded
once at startup, not per-request -- loading it takes real time (multiple
model downloads/loads) and per-request loading would make every request
slow and eventually exhaust VRAM. If loading fails (e.g. models not trained
yet), the API still starts, but /predict and /explain return 503 rather
than crashing the whole server or exposing a stack trace to the client
(task sections 17, 25: no stack traces to clients, handle "model
unavailable" gracefully).
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepdefender.api")

MAX_TEXT_LENGTH = 2000  # characters; task section 17 "handle very long text"

_predictor = None
_predictor_load_error: Optional[str] = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _predictor, _predictor_load_error
    try:
        from src.stacking import StackingPredictor
        _predictor = StackingPredictor()
        logger.info("StackingPredictor loaded successfully.")
    except Exception as e:  # noqa: BLE001
        # Deliberately broad: any failure here (missing trained models,
        # missing meta-classifier, CUDA error, ...) must not crash the API
        # process -- it must leave the API able to report "model
        # unavailable" via /health and 503s, per task section 17/25.
        _predictor_load_error = str(e)
        logger.error(f"Failed to load StackingPredictor at startup: {e}")
    yield


app = FastAPI(title="DeepDefender API", version="0.1.0", lifespan=_lifespan)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=0, max_length=MAX_TEXT_LENGTH)


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=0, max_length=MAX_TEXT_LENGTH)
    method: str = Field("occlusion", pattern="^(occlusion|shap)$")
    top_k: int = Field(5, ge=1, le=20)


@app.get("/health")
def health():
    return {
        "status": "ok" if _predictor is not None else "model_unavailable",
        "model_loaded": _predictor is not None,
        "load_error": _predictor_load_error if _predictor is None else None,
    }


def _require_predictor():
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. It may still be loading, or "
                   "training/stacking may not have completed yet. Check /health.",
        )


@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="`text` must not be empty.")

    _require_predictor()

    start = time.time()
    try:
        result = _predictor.predict(text)
    except ValueError as e:
        # e.g. text became empty after preprocessing (all URLs/mentions/etc.)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:  # noqa: BLE001
        logger.exception("Inference error in /predict (text redacted from log)")
        raise HTTPException(status_code=500, detail="Internal inference error.")

    latency_ms = round((time.time() - start) * 1000, 1)
    logger.info(f"/predict latency_ms={latency_ms} label={result['label']}")

    return {
        "label": result["label"],
        "label_name": result["label_name"],
        "probabilities": result["probabilities"],
        "confidence": result["confidence"],
        "latency_ms": latency_ms,
    }


@app.post("/explain")
def explain_endpoint(req: ExplainRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="`text` must not be empty.")

    _require_predictor()

    from src.explainability import explain

    start = time.time()
    try:
        result = explain(_predictor, text, method=req.method, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:  # noqa: BLE001
        logger.exception("Inference error in /explain (text redacted from log)")
        raise HTTPException(status_code=500, detail="Internal explanation error.")

    latency_ms = round((time.time() - start) * 1000, 1)
    result["latency_ms"] = latency_ms
    return result


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # noqa: ANN001
    # Last-resort catch-all: never leak a raw traceback to the client
    # (task section 17: "Do not expose stack traces to the client").
    logger.exception(f"Unhandled exception on {request.url.path}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
