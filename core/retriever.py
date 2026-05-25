"""Hybrid 검색 (Dense + BM25 + RRF) 싱글톤 래퍼.

HybridSearcher 를 한 번만 로드해 FastAPI lifespan 동안 재사용.
"""

from __future__ import annotations

from typing import Any, Dict, List

from config import BM25_K, DENSE_K, RRF_K_CONST, TOP_N
from core.hybrid_search import HybridSearcher


class HybridRetriever:
    """앱 lifespan 동안 유지되는 single instance.

    config.py 의 DENSE_K / BM25_K / TOP_N 을 그대로 사용. 실험 시 config 만 수정.
    """

    def __init__(self) -> None:
        self._searcher = HybridSearcher(rrf_k_const=RRF_K_CONST)

    def search(self, query: str) -> List[Dict[str, Any]]:
        return self._searcher.search(
            query,
            dense_k=DENSE_K,
            bm25_k=BM25_K,
            top_n=TOP_N,
        )
