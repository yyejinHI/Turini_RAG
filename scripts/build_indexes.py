"""인덱스 빌드 — FAISS (Dense) + BM25.

첫 실행 시 1회 (OpenAI 임베딩 API 호출 비용 발생).
이후 vectorstores/ 디렉토리 그대로 두면 챗봇 서버는 로드만 함.

원본 데이터 형식:
  - JSON 배열, 각 원소 = 한 금융상품 dict
  - 필드: portfolio_id, name, asset_class, risk_level, expected_return,
          investment_strategy, target_investor, fees, ... (CONTENT_FIELDS 참조)

산출:
  - <INDEX_DIR>/portfolio/chunk_record/{index.faiss, index.pkl}   ← FAISS
  - <INDEX_DIR>/portfolio/bm25_record/{bm25.pkl, docs.json}        ← BM25

실행:
  python scripts/build_indexes.py

  # 데이터 경로 override
  PORTFOLIO_DATA_PATH=/path/to/data.json python scripts/build_indexes.py

------------------------------------------------------------
실험 포인트:
  - CONTENT_FIELDS  : 어떤 필드를 임베딩에 넣을지 (검색 신호 디자인의 핵심)
  - METADATA_FIELDS : 어떤 필드를 메타로 보존할지 (응답 표시용)
  - 청킹 전략: 현재 1행=1청크. 긴 설명문은 sentence splitter 로 추가 청크 분리 실험.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

# 프로젝트 루트를 import path 에 추가 (scripts/ 하위에서 실행 가능하도록)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from config import DOMAIN, EMBEDDING_MODEL, INDEX_DIR, OPENAI_API_KEY, PROJECT_ROOT
from core.tokenizer import make_tokenizer

# 원본 데이터 경로 (기본: <PROJECT_ROOT>/data/portfolios.json)
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "portfolios.json"
DATA_PATH = Path(os.getenv("PORTFOLIO_DATA_PATH", str(DEFAULT_DATA_PATH)))


# ============================================================
# 임베딩에 들어가는 본문 필드 (검색 신호 강한 자유 텍스트 위주)
#   ※ 실험 시 이 리스트를 늘리거나 줄여 검색 정확도 영향 측정.
# ============================================================
CONTENT_FIELDS = [
    "name",
    "description",
    "asset_class",
    "risk_level",
    "expected_return",
    "investment_strategy",
    "target_investor",
    "recommended_horizon",
    "fees",
    "holdings_summary",
    "benchmark",
    "key_features",
    "tax_treatment",
]


# ============================================================
# 메타데이터로 보존할 필드 (검색 결과 표시·식별·식별 ID 용)
# ============================================================
METADATA_FIELDS = [
    "portfolio_id",
    "name",
    "asset_class",
    "risk_level",
    "manager",
    "inception_date",
    "detail_url",
    "ticker",
    "currency",
]


def _row_to_text(row: dict) -> str:
    """금융상품 dict → 검색용 단일 텍스트.

    필드 앞에 [필드명] 라벨을 붙여 임베딩 모델이 정보 종류를 구분할 수 있게.
    빈 값은 건너뜀.
    """
    parts: List[str] = []
    for f in CONTENT_FIELDS:
        v = row.get(f)
        if v in (None, "", []):
            continue
        # list 필드는 콤마 결합
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        parts.append(f"[{f}] {v}")
    return "\n".join(parts)


def load_source_documents() -> List[Document]:
    if not DATA_PATH.exists():
        raise SystemExit(
            f"원본 데이터를 찾을 수 없습니다: {DATA_PATH}\n"
            f"PORTFOLIO_DATA_PATH 환경변수로 위치를 지정하거나, "
            f"data/portfolios.json 에 파일을 두세요.\n"
            f"샘플 파일: data/portfolios_sample.json"
        )
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
    docs: List[Document] = []
    for row in rows:
        text = _row_to_text(row)
        if not text.strip():
            continue
        metadata = {k: row.get(k) for k in METADATA_FIELDS}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


# ============================================================
# FAISS (Dense)
# ============================================================
def build_faiss() -> None:
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY 가 .env 에 없습니다.")
    print(f"[faiss] source: {DATA_PATH}")
    docs = load_source_documents()
    print(f"[faiss] documents: {len(docs)}")
    print(f"[faiss] embedding ({EMBEDDING_MODEL}) — OpenAI API 호출 시작...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    vs = FAISS.from_documents(docs, embeddings)
    out_dir = INDEX_DIR / DOMAIN / "chunk_record"
    out_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(out_dir))
    print(f"[faiss] saved → {out_dir}")


# ============================================================
# BM25 (Sparse)
# ============================================================
def build_bm25() -> None:
    print(f"[bm25] source: {DATA_PATH}")
    docs = load_source_documents()
    serial_docs: List[Dict[str, Any]] = [
        {"page_content": d.page_content, "metadata": dict(d.metadata)}
        for d in docs
    ]
    print(f"[bm25] documents: {len(serial_docs)}")
    print("[bm25] tokenizing (kiwipiepy)...")
    tokenize = make_tokenizer()
    corpus_tokens: List[List[str]] = []
    for i, d in enumerate(serial_docs):
        corpus_tokens.append(tokenize(d["page_content"]))
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(serial_docs)}")
    avg = sum(len(t) for t in corpus_tokens) / max(len(corpus_tokens), 1)
    print(f"[bm25] avg tokens/doc: {avg:.1f}")

    print("[bm25] building BM25Okapi...")
    bm25 = BM25Okapi(corpus_tokens)

    out_dir = INDEX_DIR / DOMAIN / "bm25_record"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(out_dir / "docs.json", "w", encoding="utf-8") as f:
        json.dump(serial_docs, f, ensure_ascii=False)
    print(f"[bm25] saved → {out_dir}")


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"== building indexes into: {INDEX_DIR} ==")
    build_faiss()
    print()
    build_bm25()
    print("\n== done ==")
    print("이제 다음 명령으로 서버 기동:")
    print("  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
