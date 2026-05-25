# 실험 가이드 — RAG 성능 개선을 위해 무엇을 바꿀 것인가

베이스라인 측정 후 **한 번에 한 변수만** 바꿔 비교합니다 (controlled ablation).
각 실험은 `config.py` 의 값을 바꾸고 `python scripts/evaluate.py --tag <이름>` 으로 재측정.

---

## 실험 우선순위 (효과 큰 순)

| 순위 | 실험 축 | 기대 효과 | 비용 |
|---|---|---|---|
| 1 | **임베딩 모델 교체** | Recall 큰 폭 변동 | 인덱스 재빌드 + API 호출 비용 |
| 2 | **K_dense / K_bm25 / TOP_N 스윕** | Hit@K, MRR 직접 영향 | 인덱스 재빌드 불필요 |
| 3 | **리랭커 on/off & 모델 교체** | NDCG, Top-3 정확도 | 모델 다운로드만 |
| 4 | **청킹 전략** | 긴 설명문 검색 정확도 | 인덱스 재빌드 |
| 5 | **프롬프트 버전** | 답변 충실도·환각 감소 | 비용 없음 |
| 6 | **LLM 모델 / temperature** | 답변 품질·비용 trade-off | 호출 비용 |
| 7 | **Query rewrite on/off** | 멀티턴 후속 질의 검색 | 비용 없음 |
| 8 | **Intent classifier on/off** | OOS 입력 처리 정확도 | 비용 없음 |
| 9 | **Tokenizer 사용자 사전·KEEP_TAGS** | 도메인 용어 검색 신호 | 인덱스 재빌드 |
| 10 | **RRF k 상수** | 상위 가중 vs 평탄화 | 비용 없음 |

---

## 1. 임베딩 모델 교체 (`config.EMBEDDING_MODEL`)

| 후보 | 차원 | 특징 |
|---|---|---|
| `text-embedding-3-large` (기본) | 3072 | OpenAI 최고 정확도 |
| `text-embedding-3-small` | 1536 | 1/5 비용, 약간 낮은 정확도 |
| `BAAI/bge-m3` | 1024 | HF 오픈 모델, 다국어 강함, 자가 호스팅 가능 |
| `jhgan/ko-sroberta-multitask` | 768 | 한국어 특화, 짧은 쿼리에 강함 |

**작업**:
1. `config.py` 의 `EMBEDDING_MODEL` 변경
2. HF 모델인 경우 `core/vectorstore.py` 의 `OpenAIEmbeddings` 를 `HuggingFaceEmbeddings` 로 교체
3. **인덱스 재빌드 필수** (`python scripts/build_indexes.py`)
4. `python scripts/evaluate.py --tag embed_<모델>`

**관찰 포인트**: `retrieval_hit@10`, `retrieval_mrr@10` 의 변화.

---

## 2. Retrieval K 값 스윕 (`DENSE_K`, `BM25_K`, `TOP_N`)

베이스라인: 모두 50.

**스윕 그리드**:
```
DENSE_K ∈ {20, 30, 50, 80, 100}
BM25_K  ∈ {20, 30, 50, 80, 100}
TOP_N   ∈ {20, 30, 50, 80}      (TOP_N ≤ DENSE_K + BM25_K)
```

**관찰 포인트**:
- 너무 작으면 정답 누락 (낮은 hit@TOP_K_GEN)
- 너무 크면 리랭커 부하 ↑ + 노이즈 후보로 인한 rerank 혼란
- 보통 30~50 사이 sweet spot

**자동 스윕 예** (수동 루프):
```bash
for k in 20 30 50 80; do
  sed -i "s/^DENSE_K = .*/DENSE_K = $k/" config.py
  sed -i "s/^BM25_K = .*/BM25_K = $k/" config.py
  python scripts/evaluate.py --tag k_${k}
done
```

---

## 3. 리랭커 (`USE_RERANKER`, `RERANKER_MODEL`)

**On/Off 비교** (`config.USE_RERANKER = False`):
- 리랭커 없을 때 rerank_hit@3 가 얼마나 하락하는지 측정
- 베이스라인 대비 NDCG 하락폭이 곧 리랭커의 가치

**모델 교체 후보**:
- `BAAI/bge-reranker-v2-m3` (기본, 다국어, 567MB)
- `Dongjin-kr/ko-reranker` (한국어 특화)
- `jinaai/jina-reranker-v2-base-multilingual`
- `BAAI/bge-reranker-large` (영어 강함)
- **API**: Cohere Rerank v3.5 (`cohere.client.rerank()`) — 자가 호스팅 안 하려면

**관찰 포인트**: `rerank_hit@3`, `rerank_mrr@3`, latency.

---

## 4. 청킹 전략

베이스라인: 1행=1청크 (`chunk_record`).

**대안**:
- **Sentence chunk**: `investment_strategy`, `key_features` 같은 긴 필드를 문장 단위로 분리
- **Window chunk**: 인접 필드를 묶어 의미 보존 + 중복 약간 감수
- **Hierarchical**: 상품 전체 요약 + 세부 청크 둘 다 색인 → 검색 후 부모 문서 fetch

**작업**:
1. `scripts/build_indexes.py` 에 새 청킹 함수 추가 후 `chunk_record_sentence/` 하위에 저장
2. `core/vectorstore.py` 호출 시 `name="chunk_record_sentence"` 로 변경
3. 평가 비교

---

## 5. 프롬프트 버전 관리

`core/generator.py` 의 `SYSTEM_PROMPT` 를 버전별로 분리:

```python
SYSTEM_PROMPT_V1 = """..."""   # 베이스라인 (현재 SYSTEM_PROMPT)
SYSTEM_PROMPT_V2 = """..."""   # 예: Chain-of-Thought 강화
SYSTEM_PROMPT_V3 = """..."""   # 예: 예시 더 추가
SYSTEM_PROMPT = SYSTEM_PROMPT_V1
```

**실험 축**:
- 예시(few-shot) 개수: 2개 vs 5개
- 부정 규칙 강도 ("절대 하지 말 것" 명시 vs 생략)
- Output Format 항목 수: 5개 vs 3개 vs 자유
- 투자 권유 면책 위치: SYSTEM 상단 vs 답변 말미

**측정**: 자동 메트릭으로 안 잡힘 → **LLM-as-judge** 또는 수작업 채점.
간단한 LLM-as-judge 스크립트는 `scripts/evaluate.py` 에 `--llm-judge` 플래그 형태로 추후 추가 가능.

---

## 6. LLM 모델 / 생성 파라미터

| `MODEL` | 품질 | 속도 | 비용 |
|---|---|---|---|
| `gpt-4.1-mini` (기본) | ★★★★ | ★★★ | $$ |
| `gpt-4.1` | ★★★★★ | ★★ | $$$$ |
| `gpt-4o-mini` | ★★★ | ★★★★★ | $ |
| `gpt-4o` | ★★★★ | ★★★★ | $$$ |

**HELPER_MODEL** (분류·요약·재작성): `gpt-4.1-nano` 베이스라인. 더 가벼운 호출이라 비용 영향 크지 않으나 자주 호출되므로 latency 영향은 있음.

**Temperature**:
- `0.1` (기본): 사실 답변에 안전
- `0.0`: 결정적 출력, 디버깅 쉬움 — A/B 비교 시 권장
- `0.3+`: 다양성 ↑ 대신 환각 위험

---

## 7. Query Rewrite on/off (`config.USE_QUERY_REWRITE`)

**Off** 시 사용자 원문을 그대로 검색.
- 싱글턴 질의에서는 차이 거의 없음
- **멀티턴 후속 질의 ("그건 어떻게 사요?")** 에서 차이 큼

**측정**: goldset 의 `turn_type: "followup"` 항목만 분리해 hit@3 비교.

---

## 8. Intent Classifier on/off (`config.USE_INTENT_CLASSIFIER`)

**Off** 시 모든 입력을 PORTFOLIO 로 처리 → OOS / HARMFUL 입력에 LLM 이 직접 노출.
**측정**: OOS 입력 셋을 따로 만들어 적절한 거절 응답 비율 측정 (정성 평가 + LLM-as-judge).

---

## 9. Tokenizer (`core/tokenizer.py`)

- `KEEP_TAGS`: VV/VA(동사/형용사) 추가 시 행위형 쿼리 ("매수하고 싶다") 검색 신호 ↑
- `USER_DICT`: ETF, ELS 같은 약자 추가. 도메인 데이터를 봐가며 늘려야 함.
- `MIN_TOKEN_LEN`: 1 → 2 로 변경 시 1글자 노이즈 제거되나 "주", "채" 같은 핵심어 손실

**작업**: BM25 인덱스 재빌드 필수 (FAISS 는 영향 없음).

---

## 10. RRF k 상수 (`config.RRF_K_CONST`)

베이스라인: 60 (RRF 논문 표준).

- 작게 (예: 10): 1~2위 가중치가 매우 높아짐 → 어느 한 신호가 강하게 동의하는 후보가 부상
- 크게 (예: 120): 순위차 평탄화 → top-K 안에 들기만 하면 다 비슷한 점수

**관찰**: dense 와 bm25 신호가 자주 어긋나는 데이터셋이라면 큰 k가 유리.

---

## 추가 실험 (베이스라인 외 발전 방향)

- **HyDE** (Hypothetical Document Embeddings): 쿼리를 LLM으로 가상의 문서로 확장 후 검색
- **Multi-query**: LLM이 쿼리를 3개로 분해 → 각각 검색 → 결과 통합
- **Self-RAG / CRAG**: 검색 결과 품질을 LLM이 판정해 재검색 트리거
- **Function calling**: 사용자 프로필 자동 추출 (대화에서 "저 50살이에요" → age=50 으로 자동 갱신)
- **Caching**: 동일 쿼리 재요청 시 Redis 캐시 (latency·비용 절감)

---

## 실험 기록 템플릿

`results/` 디렉토리에 누적되는 JSON 외에 별도로 표 정리:

| 실험 ID | 변경 변수 | hit@3 | mrr@3 | ndcg@3 | latency | 비고 |
|---|---|---|---|---|---|---|
| baseline | - | 0.84 | 0.71 | 0.78 | 1.2s | 기준 |
| k_dense_30 | DENSE_K 50→30 | 0.82 | 0.70 | 0.77 | 1.0s | -0.02 hit, -0.2s latency |
| embed_small | embedding 3-large→3-small | 0.79 | 0.66 | 0.72 | 1.1s | 비용 1/5, 정확도 하락 |
| no_reranker | USE_RERANKER=False | 0.74 | 0.58 | 0.65 | 0.6s | 리랭커 가치 명확 |
| ... | | | | | | |
