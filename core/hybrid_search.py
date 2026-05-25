"""Dense (FAISS) + BM25 + RRF 통합 검색.

흐름:
  query → kiwipiepy 토큰화
       ├─ FAISS similarity_search_with_score(query, k=DENSE_K) → [(doc, dense_dist)]
       └─ BM25 get_scores → argsort top-K BM25_K                → [(doc, bm25_score)]
  RRF 통합 (k=RRF_K_CONST):
      rrf_score(doc) = Σ_i 1/(RRF_K_CONST + rank_i)
  중복 제거 (doc_id 기준) 후 rrf_score 내림차순으로 top-N 반환

------------------------------------------------------------
실험 포인트:
  - DENSE_K / BM25_K / TOP_N 스윕 (config.py)
  - RRF_K_CONST: 60 이 표준 — 작게(10) 하면 상위 순위 가중↑, 크게(120) 하면 평탄화
  - Sparse 알고리즘 교체: BM25Okapi → BM25Plus / BM25L (rank_bm25 라이브러리)
  - 점수 정규화 후 weighted sum (α·dense_norm + (1-α)·bm25_norm) — RRF 와 비교 실험
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from config import DOMAIN, INDEX_DIR
from core.tokenizer import make_tokenizer
from core.vectorstore import load_faiss_vectorstore

BM25_DIR = INDEX_DIR / DOMAIN / "bm25_record"


# ---------------- BM25 로드 ----------------
class BM25Searcher:
    def __init__(self, index_dir: Path = BM25_DIR):
        bm25_path = index_dir / "bm25.pkl"
        docs_path = index_dir / "docs.json"
        if not bm25_path.exists() or not docs_path.exists():
            raise FileNotFoundError(
                f"BM25 인덱스가 없습니다: {index_dir}. "
                f"`python scripts/build_indexes.py` 로 빌드하세요."
            )
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(docs_path, "r", encoding="utf-8") as f:
            self.docs: List[Dict[str, Any]] = json.load(f)
        self.tokenize = make_tokenizer()

    def search(self, query: str, k: int) -> List[Tuple[Dict[str, Any], float]]:
        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.docs[i], float(scores[i])) for i in top_idx]


# ---------------- Dense (FAISS) ----------------
def dense_search(vs, query: str, k: int) -> List[Tuple[Dict[str, Any], float]]:
    """FAISS similarity_search_with_score. score = L2 distance (낮을수록 가까움)."""
    pairs = vs.similarity_search_with_score(query, k=k)
    out: List[Tuple[Dict[str, Any], float]] = []
    for doc, dist in pairs:
        meta = doc.metadata or {}
        out.append((
            {
                "page_content": doc.page_content,
                "metadata": dict(meta),
            },
            float(dist),
        ))
    return out


# ---------------- RRF 통합 ----------------
def rrf_fuse(
    dense: List[Tuple[Dict[str, Any], float]],
    bm25: List[Tuple[Dict[str, Any], float]],
    k_const: int = 60,
    top_n: int = 50,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion. doc_id 기준 중복 제거 후 top-N 반환."""
    table: Dict[str, Dict[str, Any]] = {}

    def _ensure(entry: Dict[str, Any]) -> str:
        # portfolio 도메인은 metadata.portfolio_id 를 doc_id 로 사용 (없으면 content prefix fallback).
        doc_id = entry["metadata"].get("portfolio_id") or entry["page_content"][:80]
        if doc_id not in table:
            table[doc_id] = {
                "doc_id": doc_id,
                "name": entry["metadata"].get("name", ""),
                "asset_class": entry["metadata"].get("asset_class", ""),
                "page_content": entry["page_content"],
                "metadata": entry["metadata"],
                "rrf_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score": None,
                "bm25_score": None,
            }
        return doc_id

    for rank, (doc, score) in enumerate(dense, start=1):
        doc_id = _ensure(doc)
        table[doc_id]["dense_rank"] = rank
        table[doc_id]["dense_score"] = score
        table[doc_id]["rrf_score"] += 1.0 / (k_const + rank)

    for rank, (doc, score) in enumerate(bm25, start=1):
        doc_id = _ensure(doc)
        table[doc_id]["bm25_rank"] = rank
        table[doc_id]["bm25_score"] = score
        table[doc_id]["rrf_score"] += 1.0 / (k_const + rank)

    fused = sorted(table.values(), key=lambda r: -r["rrf_score"])[:top_n]
    for i, item in enumerate(fused, start=1):
        item["rank"] = i
    return fused


# ---------------- 통합 검색 ----------------
class HybridSearcher:
    def __init__(self, rrf_k_const: int = 60, bm25_dir: Path = BM25_DIR):
        self.vs = load_faiss_vectorstore("chunk_record")
        self.bm25 = BM25Searcher(bm25_dir)
        self.rrf_k_const = rrf_k_const

    def search(
        self,
        query: str,
        dense_k: int = 50,
        bm25_k: int = 50,
        top_n: int = 50,
    ) -> List[Dict[str, Any]]:
        dense_res = dense_search(self.vs, query, k=dense_k)
        bm25_res = self.bm25.search(query, k=bm25_k)
        return rrf_fuse(dense_res, bm25_res, k_const=self.rrf_k_const, top_n=top_n)
