"""인덱스 빌드 — FAISS (Dense) + BM25.

첫 실행 시 1회 (OpenAI 임베딩 API 호출 비용 발생).
이후 vectorstores/ 디렉토리 그대로 두면 챗봇 서버는 로드만 함.

원본 데이터 형식 (JSONL, 1줄 = 1청크):
  - 공통 필드: chunk_id, doc_id, source_name, source_url, title, section_title,
              topic, published_date, text, embedding_text, chunk_index, total_chunks,
              prev_chunk_id, next_chunk_id, source_file
  - 파일별 추가 필드:
      05_tax_deduction_fund         : subsection_type
      fss_financial_companies_merged: sector_code, sector_name, company_name,
                                      fin_co_no, homepage_url

산출:
  - <INDEX_DIR>/portfolio/chunk_record/{index.faiss, index.pkl}   ← FAISS
  - <INDEX_DIR>/portfolio/bm25_record/{bm25.pkl, docs.json}        ← BM25

실행:
  python scripts/build_indexes.py

  # 데이터 경로 / 파일 패턴 override
  CORPUS_GLOB="data/01_*.jsonl,data/02_*.jsonl" python scripts/build_indexes.py

------------------------------------------------------------
실험 포인트:
  - PAGE_CONTENT_SOURCE : 'embedding_text' vs 'text' vs 'title+text' 조합
  - METADATA_FIELDS     : 어떤 필드를 메타로 보존할지 (응답 표시용)
  - CORPUS_GLOB         : 어떤 JSONL 들을 인덱싱 대상에 포함할지
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

# ============================================================
# 코퍼스 선택
#   기본: data/ 하위 모든 .jsonl
#   CORPUS_GLOB 환경변수로 콤마 구분 패턴(루트 기준 상대경로) 지정 가능.
# ============================================================
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
CORPUS_GLOB = os.getenv("CORPUS_GLOB", "")


def _resolve_corpus_files() -> List[Path]:
    if CORPUS_GLOB:
        files: List[Path] = []
        for pat in [p.strip() for p in CORPUS_GLOB.split(",") if p.strip()]:
            files.extend(sorted(PROJECT_ROOT.glob(pat)))
        return files
    return sorted(DEFAULT_DATA_DIR.glob("*.jsonl"))


# ============================================================
# page_content 결정 — 검색에 들어가는 단일 텍스트.
#   각 JSONL 줄에 이미 embedding_text 가 정제돼 있으므로 그대로 사용.
#   비어 있으면 title + text 로 fallback.
# ============================================================
def _row_to_text(row: dict) -> str:
    et = (row.get("embedding_text") or "").strip()
    if et:
        return et
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    if title and text:
        return f"{title}\n\n{text}"
    return title or text


# ============================================================
# 메타데이터로 보존할 필드.
#   - downstream 호환: core/hybrid_search.py 는 metadata.portfolio_id 를 doc_id 로,
#     metadata.name 을 표시용으로 씀. 새 스키마의 chunk_id / title 을 이 자리에 매핑.
# ============================================================
COMMON_META_FIELDS = [
    "doc_id",
    "chunk_id",
    "chunk_index",
    "total_chunks",
    "prev_chunk_id",
    "next_chunk_id",
    "source_name",
    "source_url",
    "source_file",
    "title",
    "section_title",
    "topic",
    "published_date",
]

# FSS(금융사) 전용 — 있을 때만 보존
FSS_META_FIELDS = [
    "company_name",
    "sector_name",
    "sector_code",
    "fin_co_no",
    "homepage_url",
]

# tax_deduction 전용
EXTRA_META_FIELDS = [
    "subsection_type",
]


def _row_to_metadata(row: dict) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for f in COMMON_META_FIELDS + FSS_META_FIELDS + EXTRA_META_FIELDS:
        v = row.get(f)
        if v not in (None, "", []):
            meta[f] = v
    # downstream(core/hybrid_search.py) 호환용 별칭
    meta["portfolio_id"] = row.get("chunk_id") or row.get("doc_id") or ""
    # 표시용 name — FSS 는 회사명, 그 외엔 title
    meta["name"] = row.get("company_name") or row.get("title") or ""
    return meta


# ============================================================
# JSONL 로더 (UTF-8 BOM 허용)
# ============================================================
def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] {path.name}:{ln} JSON parse error: {e}")
    return rows


def load_source_documents() -> List[Document]:
    files = _resolve_corpus_files()
    if not files:
        raise SystemExit(
            f"인덱싱 대상 JSONL 을 찾을 수 없습니다.\n"
            f"기본 경로: {DEFAULT_DATA_DIR}/*.jsonl\n"
            f"또는 CORPUS_GLOB 환경변수로 패턴 지정 (콤마 구분, 루트 기준 상대경로)."
        )

    docs: List[Document] = []
    per_file_counts: List[str] = []
    for path in files:
        rows = _load_jsonl(path)
        n_added = 0
        for row in rows:
            text = _row_to_text(row)
            if not text.strip():
                continue
            docs.append(Document(page_content=text, metadata=_row_to_metadata(row)))
            n_added += 1
        per_file_counts.append(f"  - {path.name}: {n_added} chunks")

    print(f"[load] {len(files)} files → {len(docs)} chunks total")
    for line in per_file_counts:
        print(line)
    return docs


# ============================================================
# FAISS (Dense)
# ============================================================
def build_faiss(docs: List[Document]) -> None:
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY 가 .env 에 없습니다.")
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
def build_bm25(docs: List[Document]) -> None:
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
    docs = load_source_documents()
    print()
    build_faiss(docs)
    print()
    build_bm25(docs)
    print("\n== done ==")
    print("이제 다음 명령으로 서버 기동:")
    print("  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
