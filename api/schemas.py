"""Pydantic Request/Response 모델.

투자자 프로필 (InvestorProfile) 기반.
원본 MOZI 의 노인 복지 프로필 → 금융 도메인 투자자 정보로 치환.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Request
# =============================================================================
class InvestorProfile(BaseModel):
    """투자자 프로필. 모든 필드 optional — 사용자가 입력 안 했으면 None."""
    age: Optional[int] = None
    gender: Optional[Literal["M", "F", "NONE"]] = None

    # 위험성향 (예: "보수적", "중립", "적극적", "공격적")
    riskTolerance: Optional[str] = None

    # 투자 목적 (예: "노후대비", "자산증식", "단기수익", "원금보존")
    investmentGoal: Optional[str] = None

    # 투자 기간 (예: "단기 (1년 미만)", "중기 (1~3년)", "장기 (3년 이상)")
    investmentHorizon: Optional[str] = None

    # 투자 경험 (예: "초보", "중급", "고급")
    investmentExperience: Optional[str] = None

    # 자산·소득 정보 (한글 라벨 또는 None)
    capitalRange: Optional[str] = None        # 예: "1억 이상", "3천만원 ~ 1억"
    monthlyInvestRange: Optional[str] = None  # 예: "월 50만원 ~ 100만원"

    # 보유 경험 플래그 (boolean)
    hasStockExperience: bool = False
    hasFundExperience: bool = False
    hasBondExperience: bool = False
    hasEtfExperience: bool = False
    hasDerivativeExperience: bool = False


class UserContext(BaseModel):
    userId: int
    profile: Optional[InvestorProfile] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    conversationId: str
    user: UserContext


# =============================================================================
# Response
# =============================================================================
class ChatResponse(BaseModel):
    reply: str
    conversationId: str
    recommendedPortfolioIds: list[str]


# =============================================================================
# Error
# =============================================================================
class ErrorResponse(BaseModel):
    error: str
    message: str
