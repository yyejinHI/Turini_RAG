"""챗봇 파이프라인 — 한 요청을 처리하는 단일 진입점.

흐름:
  ① classify_intent       — PORTFOLIO 외면 canned 응답 (config.USE_INTENT_CLASSIFIER)
  ② rewrite_query         — 멀티턴 지시어 치환     (config.USE_QUERY_REWRITE)
  ③ hybrid retrieval      — Dense + BM25 + RRF
  ④ rerank                — bge-reranker-v2-m3    (config.USE_RERANKER)
  ⑤ generate_answer       — LLM 답변
  ⑥ summarize_history     — 다음 턴용 요약
  ⑦ new_state 반환
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from config import TOP_K_GEN, TOP_N, USE_INTENT_CLASSIFIER, USE_QUERY_REWRITE, USE_RERANKER
from core.generator import (
    CANNED_REPLIES,
    classify_intent,
    detect_followup,
    format_last_turn_qa,
    generate_answer,
    rewrite_query,
    summarize_history,
)
from core.reranker import Reranker
from core.retriever import HybridRetriever


def _build_search_query(rewritten_question: str, user_info: str | None) -> str:
    """검색용 쿼리. user_info 가 있으면 자격 키워드를 prefix 로 붙여 검색 신호 강화.

    예: "55세 보수적 2년 조건의 사용자 질문: 채권형 상품 추천해줘"
    """
    if user_info:
        return f"{user_info} 조건의 사용자 질문: {rewritten_question}"
    return rewritten_question


class ChatPipeline:
    """앱 lifespan 동안 유지되는 single instance.

    retriever 와 reranker 는 한 번만 로드 (FAISS / GPU 메모리 상주).
    """

    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.reranker = Reranker() if USE_RERANKER else None

    def run(
        self,
        message: str,
        user_info: str | None,
        state: Dict[str, str],
    ) -> Tuple[str, List[str], Dict[str, str], Dict[str, Any]]:
        """한 요청 처리.

        Args:
            message: 사용자 메시지 (원문)
            user_info: profile_to_user_info() 변환 결과 또는 None
            state: 이전 세션 상태 {"summary", "last_q", "last_a"}

        Returns:
            (reply, recommended_portfolio_ids, new_state, debug_meta)
        """
        last_qa_str = format_last_turn_qa(state["last_q"], state["last_a"])

        # ① 쿼리 재작성 (멀티턴 지시어 치환)
        if USE_QUERY_REWRITE:
            rewritten = rewrite_query(
                current_question=message,
                user_info=user_info,
                prev_summary=state["summary"],
                last_turn_qa=last_qa_str,
            )
        else:
            rewritten = message
        is_followup = detect_followup(state["last_q"], message, rewritten)

        # ② 의도 분류 (후속 질의는 자동 PORTFOLIO 처리)
        if not USE_INTENT_CLASSIFIER or is_followup:
            intent = "PORTFOLIO"
        else:
            intent = classify_intent(message)

        # PORTFOLIO 외는 검색·LLM 모두 스킵하고 canned 응답
        # 단, state 는 유지 (다음 턴에서 "그 상품"이 가리킬 수 있게)
        if intent != "PORTFOLIO":
            canned = CANNED_REPLIES[intent]
            debug_meta = {
                "rewritten_query": rewritten,
                "is_followup": is_followup,
                "intent": intent,
                "n_candidates": 0,
                "top1_rerank_score": 0.0,
                "top_k_doc_ids": [],
            }
            return canned, [], state, debug_meta

        # ③ Hybrid retrieval
        search_query = _build_search_query(rewritten, user_info)
        candidates = self.retriever.search(search_query)

        # ④ Rerank → top-K (ablation: off 면 RRF top-K 그대로)
        if self.reranker is not None:
            top_k, top1_score = self.reranker.rerank(
                query=search_query,
                candidates=candidates,
                top_k=TOP_K_GEN,
            )
        else:
            top_k = candidates[:TOP_K_GEN]
            top1_score = 0.0

        contexts = [c["page_content"] for c in top_k]
        portfolio_ids = [c["doc_id"] for c in top_k if c.get("doc_id")]

        # ⑤ Generate
        answer = generate_answer(
            question=message,
            contexts=contexts,
            user_info=user_info,
            prev_summary=state["summary"],
            last_turn_qa=last_qa_str,
            is_followup=is_followup,
        )

        # ⑥ Summarize (직전 1턴 있을 때만)
        new_summary = state["summary"]
        if state["last_q"] or state["last_a"]:
            new_summary = summarize_history(
                prev_summary=state["summary"],
                question=state["last_q"],
                answer=state["last_a"],
            )

        new_state = {
            "summary": new_summary,
            "last_q": message,
            "last_a": answer,
        }
        debug_meta = {
            "rewritten_query": rewritten,
            "is_followup": is_followup,
            "intent": intent,
            "n_candidates": len(candidates),
            "top1_rerank_score": round(top1_score, 4),
            "top_k_doc_ids": [c.get("doc_id") for c in top_k],
        }
        return answer, portfolio_ids, new_state, debug_meta
