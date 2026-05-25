"""CrossEncoder 기반 reranker 싱글톤 래퍼.

기본: BAAI/bge-reranker-v2-m3 (다국어 강함).
앱 시작 시 1회 로드 (GPU 메모리 상주) — 매 요청마다 모델 로드 안 함.

------------------------------------------------------------
실험 포인트:
  - RERANKER_MODEL 교체 (config.py)
  - RERANKER_MAX_LENGTH: 길이 늘리면 long context 처리 ↑, 메모리·속도 ↓
  - sigmoid 스코어 정규화 사용 여부 (현재 사용)
  - 점수 컷오프 도입: top1_score < threshold 시 "관련 자료 없음" 응답
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
from sentence_transformers import CrossEncoder

from config import RERANKER_MAX_LENGTH, RERANKER_MODEL


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model = CrossEncoder(
            RERANKER_MODEL,
            device=device,
            max_length=RERANKER_MAX_LENGTH,
        )
        # GPU warmup — 첫 forward 의 cuDNN/JIT 초기화 흡수
        _ = self._model.predict(
            [("warmup query", "warmup doc")] * 4,
            show_progress_bar=False,
        )
        if device == "cuda":
            torch.cuda.synchronize()

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """후보 전체를 점수화 → 내림차순 정렬 → top_k 반환.

        Returns:
            (top_k 후보 리스트, top-1 sigmoid score)
            top-1 score 는 로그·디버깅용 (실험 시 컷오프 도입 후보).
        """
        if not candidates:
            return [], 0.0
        pairs = [(query, c["page_content"]) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        scores_sig = [_sigmoid(float(s)) for s in scores]
        ranked = sorted(
            zip(scores_sig, candidates),
            key=lambda x: -x[0],
        )
        top1_score = ranked[0][0] if ranked else 0.0
        out: List[Dict[str, Any]] = []
        for new_rank, (score, c) in enumerate(ranked[:top_k], start=1):
            item = dict(c)
            item["reranker_score"] = float(score)
            item["reranker_rank"] = new_rank
            out.append(item)
        return out, float(top1_score)
