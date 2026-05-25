"""환경 설정 + 하이퍼파라미터 — RAG 실험의 단일 진입점.

이 파일 하나만 바꾸면 파이프라인의 거의 모든 동작이 바뀜.
실험 매트릭스를 돌리려면 환경변수 또는 이 파일의 상수만 조정하면 됨.

환경변수 (.env 또는 OS env):
  - OPENAI_API_KEY  (필수) — LLM + Embedding
  - CHATBOT_API_KEY (필수) — 본 서버 X-API-Key 인증
  - INDEX_DIR       (선택) — 인덱스 디렉토리. 기본: ./vectorstores
  - LANGSMITH_TRACING (선택) — true 면 LangSmith 추적

================================================================
실험 가능한 축 (EXPERIMENTS.md 참조):
  - EMBEDDING_MODEL          : 임베딩 모델 교체
  - DENSE_K / BM25_K / TOP_N : 검색 후보 개수
  - RRF_K_CONST              : RRF 가중치 파라미터
  - RERANKER_MODEL           : 리랭커 교체
  - TOP_K_GEN                : LLM 입력 문서 수
  - MODEL / HELPER_MODEL     : 생성·보조 LLM
  - TEMPERATURE / TOP_P      : 생성 다양성
  - USE_RERANKER             : 리랭커 on/off (ablation)
  - USE_QUERY_REWRITE        : 쿼리 재작성 on/off (ablation)
  - USE_INTENT_CLASSIFIER    : 의도 분류 on/off (ablation)
================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트 = 이 파일이 있는 디렉토리
PROJECT_ROOT = Path(__file__).resolve().parent

# .env 로드 (OS 환경변수가 항상 우선)
load_dotenv(PROJECT_ROOT / ".env")


# =============================================================================
# 환경변수 (필수 키 + 외부 설정)
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHATBOT_API_KEY = os.getenv("CHATBOT_API_KEY", "dev-key-change-me")


# =============================================================================
# 인덱스 디렉토리
#   기본: <PROJECT_ROOT>/vectorstores/portfolio/{chunk_record, bm25_record}/
#   외부 경로 사용 시 INDEX_DIR 환경변수 지정 (e.g. /var/lib/finrag/vectorstores).
# =============================================================================
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(PROJECT_ROOT / "vectorstores")))

# 도메인 이름 — 한 서비스에서 여러 도메인을 다룰 때 디렉토리 분리용
DOMAIN = "portfolio"


# =============================================================================
# 임베딩 모델
#   주의: FAISS 인덱스 빌드 시점과 런타임이 동일해야 함.
#         모델 바꾸면 반드시 인덱스 재빌드.
#   실험 후보:
#     - "text-embedding-3-large" (3072d, 정확도↑)
#     - "text-embedding-3-small" (1536d, 속도↑/비용↓)
#     - "BAAI/bge-m3" (HF 오픈모델, 다국어 강함 — 임베더 클래스 교체 필요)
#     - "jhgan/ko-sroberta-multitask" (한국어 특화 sentence-BERT)
# =============================================================================
EMBEDDING_MODEL = "text-embedding-3-large"


# =============================================================================
# Hybrid retrieval (Dense + BM25 + RRF)
#   실험으로 결정해야 할 핵심 파라미터.
# =============================================================================
DENSE_K = 50          # FAISS top-K
BM25_K = 50           # BM25 top-K
TOP_N = 50            # RRF 통합 후 자를 후보 수
RRF_K_CONST = 60      # RRF 의 k 상수 (rank + k 의 분모). 큰 값일수록 순위차 완화


# =============================================================================
# Reranker (CrossEncoder)
#   실험 후보:
#     - "BAAI/bge-reranker-v2-m3" (다국어 강함, 567MB)
#     - "BAAI/bge-reranker-large" (영어 위주, 더 큼)
#     - "Dongjin-kr/ko-reranker" (한국어 특화)
#     - "jinaai/jina-reranker-v2-base-multilingual"
# =============================================================================
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_MAX_LENGTH = 512
TOP_K_GEN = 3         # LLM 에 넘기는 최종 후보 수 (3~5 권장)


# =============================================================================
# LLM (Generator)
#   메인은 답변 생성, helper 는 분류/요약/재작성 (저렴한 모델 권장).
#   실험 후보 (메인):
#     - "gpt-4.1-mini" (균형)
#     - "gpt-4.1"       (최고 품질, 비쌈)
#     - "gpt-4o-mini"   (속도↑)
#     - "gpt-4o"
# =============================================================================
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.1
MAX_TOKENS = 768
TOP_P = 0.9

HELPER_MODEL = "gpt-4.1-nano"
HELPER_MAX_TOKENS_SUMMARY = 400
HELPER_MAX_TOKENS_REWRITE = 200
HELPER_MAX_TOKENS_CLASSIFY = 10


# =============================================================================
# Session
# =============================================================================
SESSION_TTL_SECONDS = 1800   # 30분


# =============================================================================
# Ablation 토글 — True/False 만 바꾸면 해당 단계 우회
#   실험 시 한 번에 하나씩 끄고 성능 변화 측정 (controlled ablation).
# =============================================================================
USE_RERANKER = True           # False → RRF top-K 그대로 LLM 에 전달
USE_QUERY_REWRITE = True      # False → 사용자 원문을 그대로 검색에 사용
USE_INTENT_CLASSIFIER = True  # False → 모든 입력을 PORTFOLIO 로 처리
