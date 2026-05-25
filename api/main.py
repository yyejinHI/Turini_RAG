"""FastAPI 앱 — POST /chat 엔드포인트.

실행 (프로젝트 루트에서):
  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

호출 예시:
  curl -X POST http://localhost:8000/chat \\
       -H "Content-Type: application/json" \\
       -H "X-API-Key: $CHATBOT_API_KEY" \\
       -d '{
         "message": "내 포트폴리오 위험도 알려줘",
         "conversationId": "8c4f5a72-1f3e-4b2a-9c8e-7a1b3d4e5f6a",
         "user": {
           "userId": 1,
           "profile": {
             "age": 55, "gender": "F",
             "riskTolerance": "보수적",
             "investmentGoal": "노후대비",
             "investmentHorizon": "장기 (3년 이상)",
             "investmentExperience": "중급",
             "hasFundExperience": true,
             "hasEtfExperience": true
           }
         }
       }'
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.auth import verify_api_key
from api.profile import profile_to_user_info
from api.schemas import ChatRequest, ChatResponse, ErrorResponse
from api.session import SessionStore
from core.pipeline import ChatPipeline

logger = logging.getLogger("finrag.chatbot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)


# =============================================================================
# Lifespan — 시작 시 retriever + reranker 로드 (1회)
# =============================================================================
pipeline: ChatPipeline | None = None
session_store: SessionStore | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pipeline, session_store
    logger.info("starting up — loading retriever + reranker...")
    pipeline = ChatPipeline()
    session_store = SessionStore()
    logger.info("ready.")
    yield
    logger.info("shutting down.")


app = FastAPI(
    title="Finance RAG 챗봇 서버",
    description="금융 포트폴리오 / 상품 정보 RAG + LLM 답변 생성 API",
    version="0.1.0-baseline",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def chat(
    req: ChatRequest,
    _: str = Depends(verify_api_key),
) -> ChatResponse:
    if pipeline is None or session_store is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "OVERLOADED", "message": "service initializing"},
        )

    state = session_store.get(req.conversationId)
    user_info = profile_to_user_info(req.user.profile)

    try:
        reply, portfolio_ids, new_state, debug = pipeline.run(
            message=req.message,
            user_info=user_info,
            state=state,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("pipeline failure: conv=%s", req.conversationId)
        raise HTTPException(
            status_code=500,
            detail={"error": "INTERNAL_ERROR", "message": str(e)},
        ) from e

    session_store.set(req.conversationId, new_state)

    logger.info(
        "userId=%s conv=%s msg=%r intent=%s → reply_len=%d portfolios=%d top1=%.3f",
        req.user.userId, req.conversationId, req.message[:60],
        debug.get("intent", "PORTFOLIO"),
        len(reply), len(portfolio_ids),
        debug["top1_rerank_score"],
    )

    return ChatResponse(
        reply=reply,
        conversationId=req.conversationId,
        recommendedPortfolioIds=portfolio_ids,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "INVALID_REQUEST", "message": str(exc.detail)},
    )
