"""한국어 형태소 분석 — BM25 인덱스/쿼리 토큰화.

kiwipiepy 로 형태소 분석 후 의미 있는 품사(명사·외국어·숫자·한자) 만 추출.
용언(VV/VA)은 어간 변형이 까다로워 제외해도 검색 신호에 큰 영향 없음.

BM25 인덱스 빌드 시점과 쿼리 시점 모두 동일한 함수를 사용해야 일관성 확보.

------------------------------------------------------------
실험 포인트:
  - KEEP_TAGS 조정:
      금융 도메인에서는 영문 약자(ETF, NAV, ESG)·숫자(연 5%, KOSPI 200)가 많아
      SL(외국어), SN(숫자)는 반드시 유지. 분석 결과에 따라 VV/VA 어간 추가 실험.
  - MIN_TOKEN_LEN: 1 → 2 로 올리면 1글자 노이즈 제거 (단, "주", "채" 등 단어 손실)
  - 사용자 사전 추가: kiwi.add_user_word("ELS", "NNP") 같이 도메인 단어 강제 인식.
"""

from __future__ import annotations

from typing import Callable, List

from kiwipiepy import Kiwi

# ----------------------------------------------------------------------
# 토큰화 대상 품사
#   NNG: 일반명사   NNP: 고유명사   SL: 외국어(ETF, NASDAQ 등)
#   SN: 숫자        SH: 한자
# ----------------------------------------------------------------------
KEEP_TAGS = {"NNG", "NNP", "SL", "SN", "SH"}
MIN_TOKEN_LEN = 1

# 금융 도메인 사용자 사전 — 분석기가 분리해 버리기 쉬운 단어 보강.
# 필요 시 자유롭게 추가/실험 (실험 ID: tokenizer-userdict-v1).
USER_DICT: list[tuple[str, str]] = [
    ("ETF", "NNP"),
    ("ELS", "NNP"),
    ("ELB", "NNP"),
    ("DLS", "NNP"),
    ("DLB", "NNP"),
    ("리츠", "NNG"),
    ("코스피", "NNP"),
    ("코스닥", "NNP"),
    ("나스닥", "NNP"),
    ("연금저축", "NNG"),
]


def make_tokenizer() -> Callable[[str], List[str]]:
    """kiwipiepy 인스턴스를 클로저에 가둔 토큰화 함수 반환.

    Kiwi 객체는 thread-safe 하지 않으므로 호출자가 worker 당 1개씩 관리하는 게 안전.
    여기서는 한 인스턴스 단일 보유 (FastAPI lifespan 안에서 1회 생성).
    """
    kiwi = Kiwi()
    for word, tag in USER_DICT:
        kiwi.add_user_word(word, tag)

    def _tok(text: str) -> List[str]:
        return [
            t.form
            for t in kiwi.tokenize(text)
            if t.tag in KEEP_TAGS and len(t.form) >= MIN_TOKEN_LEN
        ]

    return _tok
