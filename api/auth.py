"""X-API-Key 헤더 검증."""

from __future__ import annotations

from typing import Optional

from fastapi import Header, HTTPException

from config import CHATBOT_API_KEY


async def verify_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> str:
    if not x_api_key or x_api_key != CHATBOT_API_KEY:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "INVALID_API_KEY",
                "message": "X-API-Key header is missing or invalid",
            },
        )
    return x_api_key
