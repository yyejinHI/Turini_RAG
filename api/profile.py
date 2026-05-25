"""InvestorProfile (wire format) → generator.py 의 user_info 자연어 문자열 변환.

LLM 프롬프트에 자연스럽게 끼워 넣기 위해 한국어 한 줄로 변환한다.
예: "55세 여성, 위험성향 보수적, 투자기간 중기 (1~3년), 펀드 경험 있음"
"""

from __future__ import annotations

from typing import Optional

from api.schemas import InvestorProfile


def profile_to_user_info(profile: Optional[InvestorProfile]) -> Optional[str]:
    """InvestorProfile → 한국어 자연어 한 줄.

    None / 빈 정보면 None 반환 → generator.py 가 "정보 없음" 으로 처리.
    """
    if profile is None:
        return None

    parts: list[str] = []

    # 나이 + 성별
    age_gender: list[str] = []
    if profile.age is not None:
        age_gender.append(f"{profile.age}세")
    if profile.gender == "F":
        age_gender.append("여성")
    elif profile.gender == "M":
        age_gender.append("남성")
    if age_gender:
        parts.append(" ".join(age_gender))

    # 위험성향·투자목적·기간·경험
    if profile.riskTolerance:
        parts.append(f"위험성향 {profile.riskTolerance}")
    if profile.investmentGoal:
        parts.append(f"투자목적 {profile.investmentGoal}")
    if profile.investmentHorizon:
        parts.append(f"투자기간 {profile.investmentHorizon}")
    if profile.investmentExperience:
        parts.append(f"투자경험 {profile.investmentExperience}")

    # 금액 범위
    if profile.capitalRange:
        parts.append(f"투자가능금액 {profile.capitalRange}")
    if profile.monthlyInvestRange:
        parts.append(f"월투자금액 {profile.monthlyInvestRange}")

    # 상품별 경험 플래그
    experiences: list[str] = []
    if profile.hasStockExperience:
        experiences.append("주식")
    if profile.hasFundExperience:
        experiences.append("펀드")
    if profile.hasBondExperience:
        experiences.append("채권")
    if profile.hasEtfExperience:
        experiences.append("ETF")
    if profile.hasDerivativeExperience:
        experiences.append("파생상품")
    if experiences:
        parts.append(f"{'/'.join(experiences)} 경험 있음")

    return ", ".join(parts) if parts else None
