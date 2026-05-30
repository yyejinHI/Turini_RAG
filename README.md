# Finance RAG 챗봇 — 베이스라인

금융 **개념·용어·운용 원리** 안내용 RAG 챗봇 베이스라인. 종목·상품의 매수/매도 권유는 하지 않으며,
검색된 자료에 명시된 내용만 인용해 답합니다.

**구성**: Hybrid 검색(FAISS Dense + BM25 + RRF) → CrossEncoder 리랭킹 → GPT-4.1-mini 답변 생성 + 멀티턴 컨텍스트(쿼리 재작성·요약·의도 분류).

---

## 핵심 파이프라인

```
사용자 메시지
     │
     ▼
[1] 의도 분류 (classify_intent)        ── PORTFOLIO 외면 canned 응답 후 종료
     │                                     (CHITCHAT / OUT_OF_SCOPE / HARMFUL)
     ▼
[2] 쿼리 재작성 (rewrite_query)         ── "그건 어떻게 운용돼요?" → "TDF 의 글라이드패스 운용 원리는?"
     │                                     (싱글턴/토픽 전환은 원문 유지)
     ▼
[3] Hybrid 검색
     ├─ Dense (FAISS, text-embedding-3-large)  top-50
     └─ BM25  (kiwipiepy 형태소)                top-50
     └→ RRF 통합 (k=60)                          top-50
     │
     ▼
[4] CrossEncoder Rerank (BAAI/bge-reranker-v2-m3)  top-3
     │
     ▼
[5] LLM 답변 생성 (gpt-4.1-mini, temp=0.1)
     │   — 시스템 프롬프트로 hallucination·투자권유 차단
     ▼
[6] 다음 턴용 요약 (summarize_history)
```

각 단계는 `config.py` 의 `USE_INTENT_CLASSIFIER` / `USE_QUERY_REWRITE` / `USE_RERANKER` 토글로 끌 수 있어 ablation 실험이 가능합니다.

---

## 폴더 구조

```
Turini_RAG/
├── README.md                     ← 본 문서
├── EXPERIMENTS.md                ← 실험 가이드 (어떤 축을 바꿔 비교할지)
├── requirements.txt
├── .env.example
├── .gitignore
│
├── config.py                     ← 하이퍼파라미터 중앙 관리 — 실험 진입점
│
├── api/                          ← FastAPI 레이어 (얇은 wrapper)
│   ├── main.py                   ← POST /chat · GET /health
│   ├── auth.py                   ← X-API-Key 검증
│   ├── schemas.py                ← Pydantic 요청/응답 + InvestorProfile
│   ├── profile.py                ← InvestorProfile → 한국어 한 줄 변환
│   └── session.py                ← conversationId 별 멀티턴 상태 (인메모리 + TTL)
│
├── core/                         ← RAG 핵심 (도메인 무관)
│   ├── tokenizer.py              ← kiwipiepy 형태소 (BM25 인덱스/쿼리 공용)
│   ├── vectorstore.py            ← FAISS 로드 (런타임 전용)
│   ├── hybrid_search.py          ← Dense + BM25 + RRF 통합
│   ├── retriever.py              ← HybridSearcher 싱글톤 래퍼
│   ├── reranker.py               ← CrossEncoder 싱글톤 (GPU warmup 포함)
│   ├── generator.py              ← SYSTEM/USER 프롬프트 + LLM + 분류·재작성·요약
│   └── pipeline.py               ← ChatPipeline — 한 요청 처리 오케스트레이션
│
├── scripts/
│   ├── build_indexes.py          ← 원본 데이터 → FAISS + BM25 빌드
│   └── evaluate.py               ← goldset 기반 retrieval/latency 메트릭
│
├── data/                         ← 원본 지식 베이스 (JSONL, 청크 단위)
│   ├── 01_fund_pre_info.jsonl
│   ├── 02_fund_subscription.jsonl
│   ├── 03_fund_post_management.jsonl
│   ├── 04_pension_savings.jsonl
│   ├── 05_tax_deduction_fund.jsonl
│   ├── chunks_kb_kdi.jsonl
│   ├── chunks_market_data.jsonl
│   └── fss_financial_companies_merged.jsonl
│
├── vectorstores.zip              ← 미리 빌드된 인덱스 (압축 해제하면 vectorstores/ 생성)
│
├── vectorstores/                 ← (압축 해제 또는 build 후 생성)
│   └── portfolio/
│       ├── chunk_record/         ← FAISS  (index.faiss · index.pkl)
│       └── bm25_record/          ← BM25   (bm25.pkl · docs.json)
│
└── results/                      ← (평가 실행 시 생성) eval_*.json
```

> `vectorstores/` · `results/` · `data/*.jsonl` 본 데이터는 `.gitignore` 에서 제외 대상.
> 인덱스 산출물은 [vectorstores.zip](vectorstores.zip) 으로 동봉돼 있어 즉시 서버를 띄울 수 있습니다.

---

## 데이터

### 출처

총 **686 청크** / 8 파일, 모두 공개 자료 기반.

| 파일 | 출처 | topic | 내용 |
|---|---|---|---|
| `01_fund_pre_info.jsonl` | 금융투자협회 펀드다모아 | `fund_pre_subscription` | 펀드 가입 전 알아야 할 사항 |
| `02_fund_subscription.jsonl` | 금융투자협회 펀드다모아 | `fund_subscription` | 펀드 가입 절차·서류 |
| `03_fund_post_management.jsonl` | 금융투자협회 펀드다모아 | `fund_post_subscription` | 가입 후 운용·환매·관리 |
| `04_pension_savings.jsonl` | 금융투자협회 | `pension_savings` | 연금저축 제도·세제 |
| `05_tax_deduction_fund.jsonl` | 금융투자협회 | `tax_deduction_fund` | 소득공제 장기펀드(소장펀드) |
| `chunks_kb_kdi.jsonl` | KDI (한국개발연구원) | `stock` 외 | 경제·금융 일반 해설 |
| `chunks_market_data.jsonl` | 시장 데이터 | — | 시장 지표·통계 |
| `fss_financial_companies_merged.jsonl` | 금융감독원 금융상품통합비교공시 | `bank` 외 | 금융사·상품 비교 정보 |

### 청크 스키마 (JSONL 한 줄)

모든 파일이 공통적으로 갖는 필드:

| 필드 | 설명 |
|---|---|
| `chunk_id` | 청크 고유 ID — 검색 결과의 `recommendedPortfolioIds` 로 노출됨 |
| `doc_id` | 원문 문서 ID |
| `chunk_index` / `total_chunks` | 한 문서 안에서의 청크 순번 |
| `prev_chunk_id` / `next_chunk_id` | 인접 청크 링크 (확장형 표시용) |
| `source_name` / `source_url` | 출처 표시용 |
| `title` / `section_title` | 청크가 속한 섹션 제목 |
| `topic` | 의미 단위 카테고리 |
| `text` | 원문 텍스트 |
| `embedding_text` | 임베딩에 실제로 입력되는 텍스트 (전처리·요약된 형태) |
| `char_count` | 청크 길이 |

`fss_financial_companies_merged.jsonl` 만 추가로 `sector_code`, `sector_name`, `company_name`, `fin_co_no`, `homepage_url` 보유.

### ⚠️ 현재 코드 vs 데이터 정합성

- **인덱스가 이미 빌드돼 있어** ([vectorstores.zip](vectorstores.zip)) 압축만 풀면 서버는 바로 동작합니다.
- 다만 [scripts/build_indexes.py](scripts/build_indexes.py) 는 *옛 단일 `portfolios.json` 스키마* (필드 `name`·`description`·`asset_class` 등) 기반으로 작성돼 있어, 위 JSONL 데이터를 그대로 재빌드하려면 `CONTENT_FIELDS` / `load_source_documents()` 를 JSONL·`embedding_text` 기반으로 수정해야 합니다. 재빌드가 필요할 때 손봐야 할 위치는 build_indexes.py 상단의 두 상수와 로더 함수입니다.

---

## 빠른 시작

### 1. 의존성 설치

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows PowerShell
# source .venv/bin/activate           # macOS / Linux
pip install -r requirements.txt
```

### 2. 환경 변수

```bash
copy .env.example .env                 # Windows
# cp .env.example .env                 # macOS / Linux
```

`.env` 안에 채워야 할 키:

| 키 | 필수 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | LLM + 임베딩 호출용 |
| `CHATBOT_API_KEY` | ✅ | 본 서버 `X-API-Key` 인증값 |
| `INDEX_DIR` | – | 인덱스 디렉토리 (기본: `./vectorstores`) |
| `LANGSMITH_TRACING` | – | `true` 면 LangSmith 추적 활성 |
| `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | – | LangSmith 사용 시 |

### 3. 인덱스 준비

**A. 동봉된 zip 사용 (권장 — 즉시 사용 가능)**:
```powershell
Expand-Archive vectorstores.zip -DestinationPath .          # Windows
# unzip vectorstores.zip                                    # macOS / Linux
```
풀고 나면 `vectorstores/portfolio/{chunk_record, bm25_record}/` 생성.

**B. 처음부터 빌드** (데이터 변경 시 — 단, 현재 build_indexes.py 는 위 ⚠️ 참고하여 JSONL 로더로 수정 후 사용):
```bash
python scripts/build_indexes.py
```

### 4. 서버 기동

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

서버 시작 시 FAISS · BM25 · CrossEncoder(`bge-reranker-v2-m3`) 가 한 번 로드됩니다 (GPU 가용 시 자동 사용).

### 5. 호출 테스트

본 챗봇은 **개념·정보 안내용**입니다 — 특정 종목·상품 매수/매도 권유는 하지 않습니다.
질문은 개념·용어·운용 원리 위주로 보내야 의미 있는 답이 옵니다.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $env:CHATBOT_API_KEY" \
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

응답 (개념 설명 — 검색된 자료를 근거로 인용):
```json
{
  "reply": "글로벌 분산투자는 여러 국가·자산군에 나눠 투자해 특정 시장의 위험을 줄이는 방식입니다. 자료의 ...",
  "conversationId": "test-001",
  "recommendedPortfolioIds": ["doc_fund_pre_info_c003", "doc_68272b18050f_c000", "..."]
}
```

추가 예시 — 개념·용어·운용 원리·제도형 질문:
```text
"ETF 와 펀드는 뭐가 달라요?"
"TDF 는 어떤 원리로 운용되나요?"
"환헤지가 적용되면 어떤 차이가 있어요?"
"위험등급은 어떻게 매겨지나요?"
"총보수와 환매수수료의 차이가 뭐예요?"
"연금저축 세제혜택 한도가 어떻게 되나요?"
"펀드 가입 전에 확인해야 할 서류는?"
"소장펀드 가입 자격이 어떻게 되나요?"
```

> `recommendedPortfolioIds` 는 답변의 **근거로 인용된 청크 ID** 입니다 (매수 권유 아님).
> 프론트엔드에서 "이 답변은 다음 자료를 참고했습니다" 형태의 출처 표시 용도로 사용 권장.

멀티턴 예시 — 두 번째 호출 시 같은 `conversationId` 사용:
```bash
# 1턴: "TDF 는 어떤 원리로 운용되나요?"
# 2턴: "그거 연금저축 계좌에서도 가입할 수 있어요?"
#  → rewrite_query 가 "TDF 는 연금저축 계좌에서 가입 가능한가요?" 로 변환 후 검색
```

---

## API 명세

### `POST /chat`

**Headers**: `X-API-Key: <CHATBOT_API_KEY>` (필수)

**Request**:
```json
{
  "message": "string (1~1000자)",
  "conversationId": "string",
  "user": {
    "userId": 1,
    "profile": { /* InvestorProfile, optional */ }
  }
}
```

**Response (200)**:
```json
{
  "reply": "string",
  "conversationId": "string",
  "recommendedPortfolioIds": ["chunk_id_1", "chunk_id_2", "chunk_id_3"]
}
```

**에러 응답**:
| HTTP | error | 의미 |
|---|---|---|
| 400 | `INVALID_REQUEST` | Pydantic 검증 실패 |
| 401 | `INVALID_API_KEY` | `X-API-Key` 누락/오인 |
| 500 | `INTERNAL_ERROR` | 파이프라인 내부 예외 |
| 503 | `OVERLOADED` | 시작 중 (lifespan 미완) |

### `GET /health`

`{ "status": "ok" }` — 헬스체크용 (인증 불필요).

### InvestorProfile (요청에 포함, 전 필드 optional)

[api/schemas.py](api/schemas.py) 에 정의. `api/profile.py` 의 `profile_to_user_info()` 가 한국어 한 줄로 변환해 LLM 프롬프트에 주입됩니다 (예: "55세 여성, 위험성향 보수적, 투자기간 장기 (3년 이상), 펀드/ETF 경험 있음").

| 필드 | 타입 | 예시 |
|---|---|---|
| `age` | int? | `55` |
| `gender` | `"M" \| "F" \| "NONE"`? | `"F"` |
| `riskTolerance` | str? | `"보수적"`, `"중립"`, `"적극적"`, `"공격적"` |
| `investmentGoal` | str? | `"자산증식"`, `"단기수익"`, `"원금보존"`, `"노후대비"` |
| `investmentHorizon` | str? | `"단기 (1년 미만)"`, `"중기 (1~3년)"`, `"장기 (3년 이상)"` |
| `investmentExperience` | str? | `"초보"`, `"중급"`, `"고급"` |
| `capitalRange` / `monthlyInvestRange` | str? | `"1억 이상"`, `"월 50만원 ~ 100만원"` |
| `hasStockExperience` / `hasFundExperience` / `hasBondExperience` / `hasEtfExperience` / `hasDerivativeExperience` | bool | 기본 `false` |

---

## 설정 (config.py)

`config.py` 한 파일이 실험의 단일 진입점입니다. 자주 건드릴 값:

| 그룹 | 키 | 기본값 |
|---|---|---|
| 임베딩 | `EMBEDDING_MODEL` | `text-embedding-3-large` |
| Retrieval | `DENSE_K` · `BM25_K` · `TOP_N` · `RRF_K_CONST` | 50 · 50 · 50 · 60 |
| Reranker | `RERANKER_MODEL` · `RERANKER_MAX_LENGTH` · `TOP_K_GEN` | `BAAI/bge-reranker-v2-m3` · 512 · 3 |
| 생성 LLM | `MODEL` · `TEMPERATURE` · `MAX_TOKENS` · `TOP_P` | `gpt-4.1-mini` · 0.1 · 768 · 0.9 |
| 보조 LLM | `HELPER_MODEL` | `gpt-4.1-nano` |
| 세션 | `SESSION_TTL_SECONDS` | 1800 (30분) |
| Ablation | `USE_RERANKER` · `USE_QUERY_REWRITE` · `USE_INTENT_CLASSIFIER` | True · True · True |

---

## 평가

`data/goldset.json` 작성 후:

```bash
python scripts/evaluate.py --tag baseline
python scripts/evaluate.py --tag k_dense_30 --k 3 5 10 20 50
```

산출: `results/eval_<tag>_<timestamp>.json` (메트릭 + 항목별 후보 ID 로그).

**메트릭**:
- Retrieval: `retrieval_hit@K`, `retrieval_mrr@K`, `retrieval_ndcg@K`
- Rerank: `rerank_hit@TOP_K_GEN`, `rerank_mrr@TOP_K_GEN`, `rerank_ndcg@TOP_K_GEN`
- `avg_latency_s`

**goldset 항목 형식** (한 줄 = 한 평가 케이스):
```json
{
  "query_id": "Q001",
  "question": "ETF 와 펀드는 뭐가 달라요?",
  "user_info": "40세, 위험성향 중립, 투자기간 장기",
  "relevant_portfolio_ids": ["doc_fund_pre_info_c012", "..."],
  "turn_type": "initial"
}
```

`relevant_portfolio_ids` 는 정답 청크 ID(s) — 본 데이터의 `chunk_id` 또는 검색 결과에서 노출되는 doc_id 와 일치해야 합니다.

실험 설계는 [EXPERIMENTS.md](EXPERIMENTS.md) 참고 — 임베딩 모델, K 값 스윕, 리랭커 on/off, 청킹 전략, 프롬프트 버전 등 우선순위가 정리돼 있습니다.

---

## 안전 가드 (시스템 프롬프트)

[core/generator.py](core/generator.py) 의 `SYSTEM_PROMPT` 가 다음을 강제합니다:

- **자료 외 사실 금지** — [금융상품 자료] 에 없으면 "제공된 자료에서는 해당 내용을 확인할 수 없습니다." 로 답
- **숫자는 자료 그대로** — 수익률·비용·기간 단위 변환·반올림·추측 금지
- **권유 금지** — "사세요/사지 마세요", "무조건 오릅니다" 류 단정 표현 금지
- **면책** — 답변은 정보 제공이며 투자 권유가 아님 (자료 기반 적합성 비교는 허용)

의도 분류기가 `OUT_OF_SCOPE` / `HARMFUL` / `CHITCHAT` 으로 판단하면 검색·LLM 호출 없이 canned 응답을 보냅니다.

---

## 관측성 (옵션)

`.env` 에 `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` 를 넣으면 generator.py 의 주요 함수(`generate_answer`, `rewrite_query`, `summarize_history`, `classify_intent`) 가 자동으로 LangSmith 에 추적됩니다 (`@traceable` 데코레이터).
