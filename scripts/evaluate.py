"""RAG 평가 베이스라인 — Retrieval + Generation 메트릭.

실험 매트릭스를 돌리기 위한 단일 진입점. config.py 의 값만 바꾸고
이 스크립트를 재실행하면 새 설정으로 평가됨.

평가 데이터 (goldset) 형식 — data/goldset.json :
  [
    {
      "query_id": "Q001",
      "question": "보수적 투자자에게 맞는 채권 ETF 알려줘",
      "user_info": "55세, 위험성향 보수적, 투자기간 장기",   // optional
      "relevant_portfolio_ids": ["P012", "P037"],            // 정답 ID 들
      "reference_answer": "안정형 채권 ETF는 ...",            // optional (생성 평가용)
      "turn_type": "initial"                                  // initial | followup
    },
    ...
  ]

산출:
  results/eval_<timestamp>.json  — 메트릭 + 모든 후보 / 답변 로그
  콘솔에 요약 메트릭 출력

------------------------------------------------------------
실험 워크플로 (controlled experiment):
  1) 베이스라인 측정 (현재 config 그대로)
       python scripts/evaluate.py --tag baseline
  2) 하나의 변수만 변경 후 재측정
       # config.py 에서 DENSE_K=50 → 30 변경
       python scripts/evaluate.py --tag k_dense_30
  3) results/ 폴더의 두 결과 비교 (메트릭 + 정성 검토)

------------------------------------------------------------
메트릭 정의:
  Retrieval (after RRF 직후, 리랭킹 직후 둘 다 측정):
    - Hit@K : top-K 안에 정답 ID 가 하나라도 있나
    - MRR@K : 1 / (정답이 처음 등장한 rank) — 클수록 좋음
    - nDCG@K: 다중 정답일 때 권장
  Generation:
    - 평균 latency (s), 평균 input/output tokens
    - faithfulness, answer_relevance (별도 LLM-as-judge — 추후 추가 가능)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    BM25_K,
    DENSE_K,
    MODEL,
    PROJECT_ROOT,
    RERANKER_MODEL,
    RRF_K_CONST,
    TOP_K_GEN,
    TOP_N,
    USE_INTENT_CLASSIFIER,
    USE_QUERY_REWRITE,
    USE_RERANKER,
)
from core.generator import rewrite_query
from core.pipeline import ChatPipeline

# ----------------------------------------------------------------------
# 평가 헬퍼
# ----------------------------------------------------------------------
def hit_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    topk = retrieved_ids[:k]
    return 1.0 if any(g in topk for g in gold_ids) else 0.0


def mrr_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Binary relevance nDCG (정답 ID 셋 ∩ top-K 의 위치 가중합 / 이상적 DCG)."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold_ids), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


# ----------------------------------------------------------------------
# 평가 루프
# ----------------------------------------------------------------------
def run_eval(
    goldset_path: Path,
    output_dir: Path,
    tag: str,
    k_values: list[int],
) -> dict:
    with open(goldset_path, "r", encoding="utf-8") as f:
        goldset = json.load(f)

    pipeline = ChatPipeline()

    # 디버그를 위해 retrieval / rerank 단계 결과를 둘 다 보고 싶음
    # 일단 ChatPipeline.run 은 최종 결과만 반환하므로,
    # 평가용으로 retriever / reranker 를 직접 호출해 단계별 ID 리스트를 얻는다.
    per_item: list[dict] = []
    latencies: list[float] = []

    for item in goldset:
        question = item["question"]
        user_info = item.get("user_info")
        gold_ids = item.get("relevant_portfolio_ids", [])
        state = {"summary": "", "last_q": "", "last_a": ""}

        # 직접 호출 — 단계별 후보 ID 를 수집하기 위해 retriever/reranker 분리 호출
        rewritten = rewrite_query(question, user_info, "", "") if USE_QUERY_REWRITE else question
        search_query = (
            f"{user_info} 조건의 사용자 질문: {rewritten}" if user_info else rewritten
        )

        t0 = time.time()
        candidates = pipeline.retriever.search(search_query)
        retrieval_ids = [c["doc_id"] for c in candidates]

        if pipeline.reranker is not None:
            reranked, top1 = pipeline.reranker.rerank(search_query, candidates, top_k=TOP_K_GEN)
            rerank_ids = [c["doc_id"] for c in reranked]
        else:
            reranked = candidates[:TOP_K_GEN]
            rerank_ids = [c["doc_id"] for c in reranked]
            top1 = 0.0
        latency = time.time() - t0
        latencies.append(latency)

        record = {
            "query_id": item.get("query_id"),
            "question": question,
            "rewritten": rewritten,
            "gold_ids": gold_ids,
            "retrieval_top_ids": retrieval_ids[:max(k_values)],
            "rerank_top_ids": rerank_ids,
            "top1_rerank_score": top1,
            "latency_s": round(latency, 3),
        }
        per_item.append(record)

    # 집계 메트릭
    def agg(name_fn, ids_field, k):
        return mean(name_fn(r[ids_field], r["gold_ids"], k) for r in per_item)

    metrics = {}
    for k in k_values:
        metrics[f"retrieval_hit@{k}"]  = round(agg(hit_at_k,  "retrieval_top_ids", k), 4)
        metrics[f"retrieval_mrr@{k}"]  = round(agg(mrr_at_k,  "retrieval_top_ids", k), 4)
        metrics[f"retrieval_ndcg@{k}"] = round(agg(ndcg_at_k, "retrieval_top_ids", k), 4)
    # 리랭커 결과는 TOP_K_GEN 까지만 (대개 3)
    metrics[f"rerank_hit@{TOP_K_GEN}"]  = round(mean(hit_at_k(r["rerank_top_ids"], r["gold_ids"], TOP_K_GEN) for r in per_item), 4)
    metrics[f"rerank_mrr@{TOP_K_GEN}"]  = round(mean(mrr_at_k(r["rerank_top_ids"], r["gold_ids"], TOP_K_GEN) for r in per_item), 4)
    metrics[f"rerank_ndcg@{TOP_K_GEN}"] = round(mean(ndcg_at_k(r["rerank_top_ids"], r["gold_ids"], TOP_K_GEN) for r in per_item), 4)
    metrics["avg_latency_s"] = round(mean(latencies), 3)

    # 결과 직렬화
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"eval_{tag}_{ts}.json"
    payload = {
        "tag": tag,
        "timestamp": ts,
        "config_snapshot": {
            "DENSE_K": DENSE_K, "BM25_K": BM25_K, "TOP_N": TOP_N,
            "RRF_K_CONST": RRF_K_CONST,
            "RERANKER_MODEL": RERANKER_MODEL, "TOP_K_GEN": TOP_K_GEN,
            "MODEL": MODEL,
            "USE_RERANKER": USE_RERANKER,
            "USE_QUERY_REWRITE": USE_QUERY_REWRITE,
            "USE_INTENT_CLASSIFIER": USE_INTENT_CLASSIFIER,
        },
        "metrics": metrics,
        "items": per_item,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n== eval: {tag} ==")
    for k, v in metrics.items():
        print(f"  {k:30s}  {v}")
    print(f"\nsaved → {out_file}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goldset", default=str(PROJECT_ROOT / "data" / "goldset.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--tag", default="baseline", help="실험 식별자 (결과 파일명에 사용)")
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5, 10, 20, 50],
                        help="retrieval 메트릭 계산할 K 값들")
    args = parser.parse_args()

    run_eval(
        goldset_path=Path(args.goldset),
        output_dir=Path(args.output),
        tag=args.tag,
        k_values=args.k,
    )


if __name__ == "__main__":
    main()
