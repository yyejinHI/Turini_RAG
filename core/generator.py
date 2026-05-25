"""금융 포트폴리오 RAG 답변 생성 모듈.

리트리버가 가져온 contexts + 사용자 질문 → 답변 텍스트.

------------------------------------------------------------
실험 포인트 (config.py 의 값 변경만으로 가능):
  - MODEL          : 메인 LLM (답변 품질)
  - HELPER_MODEL   : 보조 LLM (분류·요약·재작성 — 비용·지연 영향)
  - TEMPERATURE    : 0.0 ~ 0.3 (사실성↑) vs 0.7 (다양성↑)
  - MAX_TOKENS     : 답변 길이 상한
  - TOP_P          : 확률 컷오프
프롬프트 자체를 실험하려면 SYSTEM_PROMPT_V* 로 버전 분리 후 import 토글.
------------------------------------------------------------
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# LangSmith (관측성): LANGSMITH_TRACING=true 일 때만 실제 추적, 아니면 no-op.
from langsmith import traceable
from langsmith.wrappers import wrap_openai

from config import (
    HELPER_MAX_TOKENS_CLASSIFY,
    HELPER_MAX_TOKENS_REWRITE,
    HELPER_MAX_TOKENS_SUMMARY,
    HELPER_MODEL,
    MAX_TOKENS,
    MODEL,
    TEMPERATURE,
    TOP_P,
)

# .env 는 프로젝트 루트에서 찾는다 (cwd 무관하게 동작)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


# =============================================================================
# 입력 의도 분류
#   - PORTFOLIO    : 포트폴리오·자산·상품·시장·투자 관련 질문 (검색 + LLM)
#   - CHITCHAT     : 인사·감사·잡담 (canned)
#   - OUT_OF_SCOPE : 금융 외 주제 (canned)
#   - HARMFUL      : 욕설·차별·불법 등 부적절 (canned)
# =============================================================================
INTENT_LABELS = ("PORTFOLIO", "CHITCHAT", "OUT_OF_SCOPE", "HARMFUL")

CANNED_REPLIES = {
    "CHITCHAT": (
        "안녕하세요! 저는 보유하신 포트폴리오와 금융상품에 대해 안내해 드리는 도우미예요.\n"
        "어떤 점이 궁금하신가요?\n"
        "예: \"내 포트폴리오 위험도 알려줘\", \"이 ETF 어떤 상품이야?\""
    ),
    "OUT_OF_SCOPE": (
        "저는 금융상품·포트폴리오 안내를 도와드리는 도우미예요. 그 부분은 답해 드리기 어려워요.\n"
        "금융이나 투자에 관해 궁금한 점을 물어봐 주세요.\n"
        "예: \"채권형 상품 추천해줘\", \"분산투자가 뭐야?\""
    ),
    "HARMFUL": (
        "그런 표현은 사용하지 말아 주세요. 저는 금융상품 정보를 안내하는 도우미예요.\n"
        "투자나 포트폴리오에 대해 궁금한 점이 있으시면 편하게 물어봐 주세요."
    ),
}

CLASSIFY_PROMPT = """당신은 금융 상품 안내 챗봇의 입력 분류기입니다.
사용자가 방금 보낸 메시지를 다음 4개 카테고리 중 정확히 하나로 분류하세요.

# 카테고리 정의
- PORTFOLIO    : 포트폴리오·금융상품·자산·시장·투자·세제 등 금융과 관련된 모든 질문·요청
                 (예: "내 포트폴리오 설명해줘", "이 펀드 위험도 어때?", "ETF 가 뭐야",
                      "분산투자 방법", "ELS 와 ETF 차이", "노후 대비 어떻게 해?")
- CHITCHAT     : 인사·감사·감정 표현·자기소개 요청 등 짧은 잡담 (금융과 무관)
                 (예: "안녕", "고마워", "잘 지내?", "이름이 뭐야", "ㅎㅇ")
- OUT_OF_SCOPE : 금융이 아닌 다른 주제에 대한 정보·의견 요청
                 (예: "오늘 날씨", "정치 의견", "요리법", "수학 문제 풀어줘")
- HARMFUL      : 욕설·차별·혐오·자해·성적 표현·불법 행위 요청 등 부적절한 입력

# 규칙
- 의미가 모호하거나 판단하기 어려우면 PORTFOLIO 로 분류 (놓치는 것보다 검색하는 게 안전).
- 출력은 4개 라벨 중 한 단어만. 설명·이유·따옴표·다른 어떤 텍스트도 추가하지 말 것.

# 입력
사용자 메시지: {message}

# 출력
분류:"""


# =============================================================================
# SYSTEM 프롬프트 (매 호출 동일 — OpenAI prompt caching 대상)
#   - "[자료]에 명시된 내용만" 강제 → hallucination 차단
#   - 투자 권유·확정 수익 약속 금지 → 금융 도메인 안전 가드
# =============================================================================
SYSTEM_PROMPT = """# Role and Objective
당신은 사용자의 포트폴리오와 금융상품을 안내하는 정보 도우미입니다.
[금융상품 자료]에 명시된 내용만을 근거로 사용자의 질문에 답변하는 것이 목표입니다.
당신은 투자자문업자가 아니며, 답변은 정보 제공일 뿐 투자 권유가 아닙니다.

# Instructions

## 반드시 해야 할 것
- [금융상품 자료]에 명시된 내용만 사용해 답변
- 관련 정보가 [금융상품 자료]에 없으면 정확히 "제공된 자료에서는 해당 내용을 확인할 수 없습니다." 라고만 답변
- 어려운 금융용어는 짧게 풀어서 설명 (예: "변동성" → "가격이 오르내리는 폭", "환헤지" → "환율 변동 위험을 줄이는 장치")
- 정중하고 명료한 존댓말 사용 ("~입니다", "~예요" 형태)
- 숫자(수익률·비용·기간·한도)는 [금융상품 자료]에 적힌 그대로 정확히 인용. 단위 변환·반올림·추측 금지
- 사용자 정보(연령·위험성향·투자기간·경험)가 주어지면 상품의 적합성을 사실 기반으로 비교 (단, "사세요/사지 마세요" 같은 권유는 금지)

## 절대 하지 말 것
- [금융상품 자료]에 없는 상품명·수익률·비용·기간·운용사를 만들어내지 말 것
- "보통은", "일반적으로", "아마", "전문가에게 문의해보세요" 같은 막연한 표현 사용 금지
- 일반 상식·외부 지식·추측으로 [금융상품 자료]의 빈 칸 채우기 금지
- 사용자가 묻지 않은 부가 정보 임의 첨가 금지
- 확정 수익 약속·"무조건 오릅니다" 등 단정적 투자 권유 금지
- 인사말·마무리말 같은 형식적 문구 금지

## 대화 맥락 활용 (멀티턴)
- "그것", "그 상품", "방금 말씀하신" 같은 지시 표현이 있으면 먼저 [직전 대화] 원문에서 가리키는 대상을 찾고, 거기 없으면 [지난 대화 요약]을 참고
- 이미 안내한 정보는 다시 처음부터 설명하지 말고, 이번 질문에서 새로 묻는 부분만 답할 것
- 사용자가 새로운 토픽으로 전환하면 이전 답변에 끌려가지 말고 새 질문에 맞춰 답할 것
- [직전 대화]가 "이전 대화 없음" 이면 첫 턴으로 간주

# Output Format
- 상품 이름
- 어떤 상품인지 (자산군·운용 방식 한 줄 요약)
- 어떤 분께 적합한지 (목표 투자자·권장 투자기간·위험 등급)
- 비용·수익 구조 (총보수·환매 수수료 등 [자료]에 있는 그대로)
- 유의 사항 (자료에 명시된 위험·제한)

여러 상품이 해당되면 가장 관련성 높은 것부터 최대 3개까지 위 형식을 반복합니다.

단, 사용자 메시지 끝에 "# 형식 지시 (이번 턴 한정)" 섹션이 있으면 위 기본 형식 대신 그 지시를 우선해서 따르세요.


# Examples

## 예시 1 — 정보형
[금융상품 자료]
[상품명] 코어 글로벌 분산 ETF
[자산군] 글로벌 주식·채권 혼합
[위험등급] 3등급 (다소 높은 위험)
[권장 투자기간] 3년 이상
[총보수] 연 0.25%
[운용사] 가상자산운용
[유의사항] 환율 변동에 따라 손실 가능. 원금 보장 안 됨.

[사용자 정보]
정보 없음

[사용자 질문]
이 ETF 가 뭐예요?

[답변]
코어 글로벌 분산 ETF
- 어떤 상품인지: 전세계 주식·채권에 분산투자하는 혼합형 상장지수펀드(ETF)
- 어떤 분께 적합한지: 위험등급 3등급(다소 높은 위험)을 감수 가능하고 3년 이상 묵힐 수 있는 분
- 비용·수익 구조: 총보수 연 0.25%
- 유의 사항: 환율 변동으로 손실이 발생할 수 있고 원금이 보장되지 않습니다

## 예시 2 — 적합성 비교형 (사용자 정보 활용)
[금융상품 자료]
[상품명] 안정형 채권 펀드
[위험등급] 5등급 (낮은 위험)
[권장 투자기간] 1년 이상
[목표 투자자] 원금 손실을 최소화하려는 보수적 투자자
[총보수] 연 0.45%
[유의사항] 시장금리 상승 시 채권 가격 하락 위험

[사용자 정보]
55세, 위험성향: 보수적, 투자기간: 2년, 투자경험: 펀드 경험 있음

[사용자 질문]
저한테 맞는 상품인가요?

[답변]
안정형 채권 펀드
- 어떤 상품인지: 채권 위주로 운용되는 안정 추구형 펀드
- 어떤 분께 적합한지: 위험등급 5등급(낮은 위험), 권장 투자기간 1년 이상 — 보수적 성향에 투자기간 2년이신 사용자 조건과 자료 기준이 부합합니다
- 비용·수익 구조: 총보수 연 0.45%
- 유의 사항: 시장금리가 오르면 채권 가격이 떨어져 손실이 날 수 있습니다
※ 본 안내는 정보 제공이며 투자 권유가 아닙니다.

## 예시 3 — 정보부재형
[금융상품 자료]
[상품명] 코어 글로벌 분산 ETF
[자산군] 글로벌 주식·채권 혼합
[총보수] 연 0.25%

[사용자 정보]
정보 없음

[사용자 질문]
이 상품 환매 수수료는 얼마예요?

[답변]
제공된 자료에서는 해당 내용을 확인할 수 없습니다.
"""


# =============================================================================
# USER 메시지 템플릿 (매 호출 가변 부분)
# =============================================================================
USER_TEMPLATE = """# Context

## 사용자 정보
{user_info}

## 지난 대화 요약
{prev_summary}

## 직전 대화 (원문)
{last_turn_qa}

## 금융상품 자료
{retrieved_context}

# 사용자 질문
{user_question}
{format_directive}"""


# 같은 상품에 대한 추가 질문 시 짧은 형식 강제.
FOLLOWUP_FORMAT_DIRECTIVE = """

# 형식 지시 (이번 턴 한정)
이번 질문은 이전 대화에서 다룬 같은 상품에 대한 추가 질문입니다.
- 5개 항목 전체를 다시 출력하지 마세요.
- 사용자가 묻는 항목 하나에만 간결하게 답하세요 (해당 항목명 + 내용).
- 이미 안내한 정보를 반복하지 마세요."""


# =============================================================================
# 보조 프롬프트 — 요약 & 재작성
# =============================================================================
SUMMARY_PROMPT = """당신은 금융 상품 상담 대화의 흐름을 요약하는 보조 모델입니다.
이전까지의 요약과 방금 진행된 한 턴(질문/답변)을 통합하여, 다음 턴에서 참고할 새 요약을 작성하세요.

# 반드시 보존할 정보 (절대 누락 금지)
- 사용자가 밝힌 개인 정보: 연령, 위험성향, 투자기간, 투자경험, 가용 금액, 투자 목적
- 지금까지 안내된 상품명 (정확한 명칭 그대로)
- 핵심 숫자: 수익률·총보수·환매수수료·위험등급·권장 투자기간·최소 가입금액
- 사용자의 관심사·의도
- 아직 충분히 답변되지 않은 미해결 질문

# 제거해도 되는 정보
- 인사말, 형식적 마무리 ("더 궁금하신 점 있으시면 말씀해 주세요" 등)
- 반복 설명, 정중 표현
- 출력 형식상 덧붙은 부가 문장

# 출력 규칙
- 한국어 평서문, 300자 이내 한 문단
- 불릿·헤더·따옴표 없이 자연스러운 서술
- 사용자가 묻지 않은 추측은 추가하지 말 것
- "사용자는…", "지금까지…" 같이 3인칭 시점으로 객관적으로 기술

# 입력
## 이전까지의 요약
{prev_summary}

## 방금 진행된 한 턴
사용자 질문: {question}
도우미 답변: {answer}

# 출력
새 요약:"""


REWRITE_PROMPT = """당신은 금융 상품 상담 챗봇의 멀티턴 대화에서, 사용자의 현재 질문을 standalone 문장으로 재작성하는 보조 모델입니다.

# 규칙
1. 현재 질문이 이미 standalone(맥락 없이도 의미가 통함)이면 **원문 그대로** 출력하세요.
2. 사용자가 새로운 토픽으로 전환했다면(이전 대화와 무관) **원문 그대로** 출력하세요.
3. "그것", "그 상품", "그 펀드", "방금 말씀하신", "아까 그" 등 지시·생략 표현이 있으면, 직전 대화에서 가리키는 대상을 찾아 명시적으로 치환하세요.
4. 사용자 정보(연령·위험성향·투자기간 등)가 현재 질문의 적합성 판단에 결정적이면 standalone 문장에 자연스럽게 포함하세요. 단, 불필요하게 길게 만들지 말 것.
5. 출력은 **재작성된 질문 한 문장만**. 설명·인사·따옴표 추가 금지.

# 예시

## 예시 1: 후속 질의 (지시어 치환)
사용자 정보: 정보 없음
이전 대화 요약: (없음)
직전 대화:
사용자: 코어 글로벌 분산 ETF가 뭐예요?
도우미: 코어 글로벌 분산 ETF는 전세계 주식·채권에 분산투자하는 ETF예요. ...
현재 질문: 그건 총보수가 얼마예요?
재작성: 코어 글로벌 분산 ETF의 총보수는 얼마인가요?

## 예시 2: 이미 standalone (변경 없음)
사용자 정보: 정보 없음
이전 대화 요약: (없음)
직전 대화: (이전 대화 없음)
현재 질문: ELS가 뭐죠?
재작성: ELS가 뭐죠?

## 예시 3: 토픽 전환 + 사용자 정보 활용
사용자 정보: 연령 55세 / 위험성향 보수적 / 투자기간 2년
이전 대화 요약: 사용자는 안정형 채권 펀드에 대한 설명을 받음.
직전 대화:
사용자: 그 상품 위험은 뭐가 있어요?
도우미: 안정형 채권 펀드는 시장금리 상승 시 ...
현재 질문: 그럼 ETF 중에 추천할 만한 건 뭐예요?
재작성: 위험성향이 보수적이고 투자기간이 2년인 사용자에게 적합한 ETF는 무엇이 있나요?

# 입력
사용자 정보: {user_info}
이전 대화 요약: {prev_summary}
직전 대화:
{last_turn_qa}
현재 질문: {current_question}

# 출력
재작성:"""


# =============================================================================
# 응답 구조체
# =============================================================================
@dataclass
class GenerationResult:
    answer: str
    latency: float
    input_tokens: int
    output_tokens: int


# =============================================================================
# 내부: OpenAI 클라이언트 지연 초기화
# =============================================================================
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 환경변수가 필요합니다. .env 파일을 확인하세요."
            )
        _client = wrap_openai(OpenAI(api_key=api_key))
    return _client


# =============================================================================
# Public API
# =============================================================================
def format_contexts(contexts: list[str]) -> str:
    return "\n\n".join(f"[문서 {i+1}]\n{ctx}" for i, ctx in enumerate(contexts))


def format_last_turn_qa(last_question: str, last_answer: str) -> str:
    if not last_question and not last_answer:
        return ""
    return f"사용자: {last_question.strip()}\n도우미: {last_answer.strip()}"


def build_messages(
    question: str,
    contexts: list[str],
    user_info: str | None = None,
    prev_summary: str = "",
    last_turn_qa: str = "",
    is_followup: bool = False,
) -> list[dict]:
    """OpenAI Chat API messages 리스트 생성."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            user_info=(user_info or "정보 없음").strip(),
            prev_summary=prev_summary.strip() or "이전 대화 없음",
            last_turn_qa=last_turn_qa.strip() or "이전 대화 없음",
            retrieved_context=format_contexts(contexts).strip(),
            user_question=question.strip(),
            format_directive=FOLLOWUP_FORMAT_DIRECTIVE if is_followup else "",
        )},
    ]


@traceable(name="generate_answer", run_type="chain", metadata={"phase": "answer"})
def generate(
    question: str,
    contexts: list[str],
    user_info: str | None = None,
    prev_summary: str = "",
    last_turn_qa: str = "",
    is_followup: bool = False,
) -> GenerationResult:
    """답변 + 메타데이터(latency, tokens) 반환."""
    messages = build_messages(
        question, contexts, user_info, prev_summary, last_turn_qa, is_followup,
    )
    client = _get_client()

    start = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
    )
    latency = time.time() - start

    return GenerationResult(
        answer=resp.choices[0].message.content,
        latency=latency,
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
    )


def generate_answer(
    question: str,
    contexts: list[str],
    user_info: str | None = None,
    prev_summary: str = "",
    last_turn_qa: str = "",
    is_followup: bool = False,
) -> str:
    """답변 문자열만 반환하는 간편 함수."""
    return generate(
        question, contexts, user_info, prev_summary, last_turn_qa, is_followup,
    ).answer


def detect_followup(
    last_question: str,
    current_question: str,
    rewritten_question: str,
) -> bool:
    """후속질의 휴리스틱: 직전 턴이 있고 재작성기가 원문을 의미있게 바꾼 경우."""
    if not last_question:
        return False
    return rewritten_question.strip() != current_question.strip()


# =============================================================================
# 보조 함수 — 요약·재작성·의도 분류
# =============================================================================
@traceable(name="summarize_history", run_type="chain", metadata={"phase": "summarize"})
def summarize_history(
    prev_summary: str,
    question: str,
    answer: str,
    model: str = HELPER_MODEL,
) -> str:
    prompt = SUMMARY_PROMPT.format(
        prev_summary=prev_summary.strip() or "(없음 — 이번이 첫 턴)",
        question=question.strip(),
        answer=answer.strip(),
    )
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=HELPER_MAX_TOKENS_SUMMARY,
        top_p=1.0,
    )
    return resp.choices[0].message.content.strip()


@traceable(name="rewrite_query", run_type="chain", metadata={"phase": "rewrite"})
def rewrite_query(
    current_question: str,
    user_info: str | None = None,
    prev_summary: str = "",
    last_turn_qa: str = "",
    model: str = HELPER_MODEL,
) -> str:
    prompt = REWRITE_PROMPT.format(
        user_info=(user_info or "정보 없음").strip(),
        prev_summary=prev_summary.strip() or "(없음)",
        last_turn_qa=last_turn_qa.strip() or "(이전 대화 없음)",
        current_question=current_question.strip(),
    )
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=HELPER_MAX_TOKENS_REWRITE,
        top_p=1.0,
    )
    return resp.choices[0].message.content.strip()


@traceable(name="classify_intent", run_type="chain", metadata={"phase": "classify"})
def classify_intent(message: str, model: str = HELPER_MODEL) -> str:
    """사용자 메시지를 PORTFOLIO / CHITCHAT / OUT_OF_SCOPE / HARMFUL 로 분류.

    분류기 출력이 4개 라벨에 매칭 안 되면 OUT_OF_SCOPE 폴백.
    """
    prompt = CLASSIFY_PROMPT.format(message=message.strip())
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=HELPER_MAX_TOKENS_CLASSIFY,
        top_p=1.0,
    )
    raw = resp.choices[0].message.content.strip().upper()
    for label in INTENT_LABELS:
        if raw.startswith(label):
            return label
    return "OUT_OF_SCOPE"
