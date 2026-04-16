"""POST /api/ai-insights — Generate local AI recommendations from model outputs."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib import error, request

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatTurn(BaseModel):
    role: str
    content: str


class AIInsightsRequest(BaseModel):
    churn_results: dict[str, Any] = Field(default_factory=dict)
    demand_results: dict[str, Any] = Field(default_factory=dict)
    pricing_results: dict[str, Any] = Field(default_factory=dict)
    basket_results: dict[str, Any] = Field(default_factory=dict)
    user_question: str
    chat_history: list[ChatTurn] = Field(default_factory=list)


def _compact(value: Any, max_items: int = 20) -> Any:
    if isinstance(value, list):
        if len(value) <= max_items:
            return [_compact(item, max_items=max_items) for item in value]
        return {
            "preview": [_compact(item, max_items=max_items) for item in value[:max_items]],
            "truncated": len(value) - max_items,
        }
    if isinstance(value, dict):
        return {k: _compact(v, max_items=max_items) for k, v in value.items()}
    return value


def _build_prompt(payload: AIInsightsRequest) -> str:
    chat_context = "\n".join(
        f"{turn.role.strip().upper()}: {turn.content.strip()}" for turn in payload.chat_history[-6:]
        if turn.role.strip() and turn.content.strip()
    )

    context = {
        "churn_results": _compact(payload.churn_results),
        "demand_results": _compact(payload.demand_results),
        "pricing_results": _compact(payload.pricing_results),
        "basket_results": _compact(payload.basket_results),
    }

    return (
        "You are a retail analytics assistant. Provide concise, actionable business insights "
        "based only on the model outputs below. If data is missing, say so clearly.\n\n"
        f"Model Outputs (JSON):\n{json.dumps(context, indent=2, default=str)}\n\n"
        f"{'Recent Chat:\n' + chat_context + '\n\n' if chat_context else ''}"
        f"User Question: {payload.user_question.strip()}\n\n"
        "Respond in plain text with practical recommendations."
    )


@router.post("/ai-insights")
async def ai_insights(payload: AIInsightsRequest):
    question = payload.user_question.strip()
    if not question:
        return JSONResponse(
            status_code=422,
            content={"status": "error", "message": "user_question cannot be empty."},
        )

    body = {
        "model": OLLAMA_MODEL,
        "prompt": _build_prompt(payload),
        "stream": False,
    }
    ollama_url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    req = request.Request(
        ollama_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as resp:
            resp_body = resp.read().decode("utf-8")
        parsed = json.loads(resp_body) if resp_body else {}
        response_text = str(parsed.get("response", "")).strip()
        if not response_text:
            logger.warning("Ollama returned empty response payload: %s", parsed)
            return JSONResponse(
                status_code=502,
                content={"status": "error", "message": "AI service returned an empty response."},
            )
        return {"response": response_text, "model_used": OLLAMA_MODEL}
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        logger.exception("Ollama HTTP error %s: %s", exc.code, detail)
        return JSONResponse(
            status_code=502,
            content={"status": "error", "message": "Failed to get response from local AI service."},
        )
    except (error.URLError, TimeoutError) as exc:
        logger.exception("Could not connect to Ollama: %s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": (
                    "Could not connect to local Ollama service. "
                    "Please ensure Ollama is running at the configured endpoint."
                ),
            },
        )
    except json.JSONDecodeError as exc:
        logger.exception("Invalid JSON from Ollama: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"status": "error", "message": "AI service returned invalid response format."},
        )
    except Exception as exc:
        logger.exception("Unexpected AI insights error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Failed to generate AI insights."},
        )
