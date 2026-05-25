# Finance RAG 챗봇 — 베이스라인

금융 포트폴리오·상품 설명과 후속 Q&A를 위한 RAG 챗봇 베이스라인.
Hybrid 검색(Dense + BM25 + RRF) → CrossEncoder 리랭킹 → LLM 답변 생성에 멀티턴 컨텍스트 관리를 결합한 구조입니다.

핵심 파이프라인:

```
사용자 메시지
     │
     ▼
[1] 의도 분류 (classify_intent)        ── PORTFOLIO 외면 canned 응답 후 종료
     │
     ▼
[2] 쿼리 재작성 (rewrite_query)         ── "그것 어떻게 사요?" → "TDF 2045 어떻게 매수해요?"
     │
     ▼
[3] Hybrid 검색
     ├─ Dense (FAISS, text-embedding-3-large)  top-50
     └─ BM25  (kiwipiepy 토큰화)                top-50
     ├─ RRF 통합 (k=60)                          top-50
     │
     ▼
[4] CrossEncoder Rerank (bge-reranker-v2-m3)    top-3
     │
     ▼
[5] LLM 답변 생성 (gpt-4.1-mini, temp=0.1)
     │
     ▼
[6] 다음 턴용 요약 (summarize_history)
```

---

## 폴더 구조

```
finance_rag_chatbot/
├── README.md                ← 본 문서
├── EXPERIMENTS.md           ← 무엇을 바꿔서 테스트할지 정리 (별도 파일)
├── requirements.txt
├── .env.example
│
├── config.py                ← 모든 하이퍼파라미터 중앙 관리 (실험 진입점)
│
├── api/                     ← FastAPI 레이어
│   ├── __init__.py
│   ├── main.py              ← /chat, /health 엔드포인트
│   ├── auth.py              ← X-API-Key 검증
│   ├── schemas.py           ← Pydantic 요청/응답 모델 (InvestorProfile)
│   ├── profile.py           ← 프로필 → LLM 프롬프트용 문자열 변환
│   └── session.py           ← 대화별 멀티턴 상태 (메모리 + TTL)
│
├── core/                    ← RAG 핵심 — 도메인 무관, 재사용 가능
│   ├── __init__.py
│   ├── tokenizer.py         ← Kiwi 형태소 분석 (BM25 용)
│   ├── vectorstore.py       ← FAISS 로드
│   ├── hybrid_search.py     ← Dense + BM25 + RRF
│   ├── retriever.py         ← HybridSearcher 싱글톤
│   ├── reranker.py          ← CrossEncoder 싱글톤
│   ├── generator.py         ← 프롬프트 + OpenAI 호출 + 의도 분류
│   └── pipeline.py          ← 오케스트레이션 (ablation 토글 반영)
│
├── scripts/                 ← 오프라인 작업
│   ├── build_indexes.py     ← 포트폴리오 데이터 → FAISS + BM25 빌드
│   └── evaluate.py          ← goldset 기반 메트릭 측정 (실험 베이스)
│
├── data/                    ← 원본/평가 데이터
│   ├── portfolios_sample.json   ← 포트폴리오 5건 예시 (필드 구조 템플릿)
│   └── goldset_sample.json      ← 평가 정답셋 예시
│
├── vectorstores/            ← (생성됨) 인덱스 산출물
│   └── portfolio/
│       ├── chunk_record/    ← FAISS
│       └── bm25_record/     ← BM25
│
└── results/                 ← (생성됨) 평가 결과 JSON
```

---

## 빠른 시작

### 1. 설치

```bash
git clone <this-repo> finance_rag_chatbot && cd finance_rag_chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # OPENAI_API_KEY, CHATBOT_API_KEY 입력
```

### 2. 데이터 준비

`data/portfolios_sample.json` 의 필드 구조를 따라 실제 데이터를 `data/portfolios.json` 으로 저장합니다 (필수 필드: `portfolio_id`, `name` + `scripts/build_indexes.py` 의 `CONTENT_FIELDS`).

### 3. 인덱스 빌드 (1회, 비용 발생)

```bash
python scripts/build_indexes.py
# 산출: vectorstores/portfolio/{chunk_record, bm25_record}/
```

### 4. 서버 실행

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 호출 테스트

본 챗봇은 **개념·정보 안내용**입니다 — 특정 종목·상품을 매수/매도하라는 권유는 하지 않습니다. 질문도 개념·용어·운용 원리 위주로 보내야 의미 있는 답이 옵니다.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $CHATBOT_API_KEY" \
  -d '{
    "message": "글로벌 분산투자가 뭐고 어떤 특징이 있나요?",
    "conversationId": "test-001",
    "user": {
      "userId": 42,
      "profile": {
        "age": 40, "gender": "M",
        "riskTolerance": "중립",
        "investmentGoal": "자산증식",
        "investmentHorizon": "장기 (3년 이상)",
        "investmentExperience": "중급",
        "hasFundExperience": true,
        "hasEtfExperience": true
      }
    }
  }'
```

응답 (개념 설명 — 검색된 상품 자료를 근거로 인용):
```json
{
  "reply": "글로벌 분산투자는 여러 국가·자산군에 나눠 투자해 특정 시장의 위험을 줄이는 방식입니다. 예를 들어 [자료]의 코어 글로벌 분산 ETF는 선진국 주식 60% / 글로벌 채권 30% / 대체자산 10% 비중으로 운용됩니다 ...",
  "conversationId": "test-001",
  "recommendedPortfolioIds": ["P001"]
}
```

추가 예시 — 개념·용어·운용 원리형 질문:
```bash
# message 예시
"ETF 와 펀드는 뭐가 달라요?"
"TDF 는 어떤 원리로 운용되나요?"
"환헤지가 적용되면 어떤 차이가 있어요?"
"위험등급은 어떻게 매겨지나요?"
"총보수와 환매수수료의 차이가 뭐예요?"
```

> `recommendedPortfolioIds` 는 답변 근거로 *참조된* 상품 ID 일 뿐, 매수 권유가 아닙니다. 프론트엔드에서 "이 답변은 다음 상품 정보를 참고했습니다" 같은 출처 표시 용도로 쓰는 걸 권장.

---

## 데이터 스키마

### `data/portfolios.json` (인덱스 빌드 입력)

배열의 각 원소는 한 개의 금융상품. 인덱스에 들어가는 필드는 `scripts/build_indexes.py` 의 두 상수로 결정됩니다:

- `CONTENT_FIELDS` — **임베딩에 들어가는 본문 필드** (검색 신호)
  현재: `name, description, asset_class, risk_level, expected_return, investment_strategy, target_investor, recommended_horizon, fees, holdings_summary, benchmark, key_features, tax_treatment`
- `METADATA_FIELDS` — **메타데이터로 보존되는 필드** (검색 결과 표시·식별용)
  현재: `portfolio_id, name, asset_class, risk_level, manager, inception_date, detail_url, ticker, currency`

각 필드명 앞에 `[필드명]` 라벨을 붙여 한 줄씩 연결한 텍스트가 임베딩 대상이 됩니다.

### `InvestorProfile` (API 요청에 포함)

`api/schemas.py` 에 정의. 모든 필드 optional — 사용자가 입력 안 했으면 `None`. profile_to_user_info() 가 한국어 한 줄로 변환해 LLM 프롬프트에 주입됩니다.

| 필드 | 예시 값 |
|---|---|
| age, gender | 55, "F" |
| riskTolerance | "보수적", "중립", "적극적", "공격적" |
| investmentGoal | "노후대비", "자산증식", "단기수익", "원금보존" |
| investmentHorizon | "단기 (1년 미만)", "중기 (1~3년)", "장기 (3년 이상)" |
| investmentExperience | "초보", "중급", "고급" |
| hasStockExperience, hasFundExperience, hasBondExperience, hasEtfExperience, hasDerivativeExperience | `true` / `false` |

---

## API 명세

### `POST /chat`

**Headers**: `X-API-Key: <CHATBOT_API_KEY>`

**Request**:
```json
{
  "message": "string (1~1000)",
  "conversationId": "string",
  "user": {
    "userId": 1,
    "profile": { ... InvestorProfile ... }    // optional
  }
}
```

**Response**:
```json
{
  "reply": "string",
  "conversationId": "string",
  "recommendedPortfolioIds": ["P001", "P004", ...]
}
```

**Error**: `{ "error": "INVALID_API_KEY" | "INTERNAL_ERROR" | ..., "message": "..." }`

### `GET /health`

`{ "status": "ok" }`

---

## 평가 (성능 측정)

`data/goldset.json` 을 작성한 뒤:

```bash
python scripts/evaluate.py --tag baseline
```

산출: `results/eval_baseline_<timestamp>.json` (메트릭 + 항목별 후보 ID 로그).
메트릭: `retrieval_hit@K`, `retrieval_mrr@K`, `retrieval_ndcg@K`, `rerank_hit@TOP_K_GEN`, `avg_latency_s`.

실험 변형은 `config.py` 의 값만 바꾸고 `--tag` 만 다르게 줘서 비교합니다.

---

## 다음 단계 — 실험 가이드

**`EXPERIMENTS.md` 참조.** 베이스라인 대비 어떤 축을 변경해 비교해야 하는지, 우선순위와 예상 효과를 정리해 두었습니다.

요약: 임베딩 모델 → K 값 스윕 → 리랭커 → 청킹 전략 → 프롬프트 → LLM 모델 순으로 ablation 추천.
