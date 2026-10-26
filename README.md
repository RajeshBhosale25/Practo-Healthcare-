# Practo Domain Support Agent

**Completed: Practo (Healthcare) track.**

This repository implements the phase-wise project in [orchestration.md](./orchestration.md), using fabricated appointment and policy data only. It does not provide clinical advice and must not be supplied with real patient information. All deliverables are code or text in this repository; no images, screenshots, diagrams, PDFs, slides, video, or audio are required or accepted.

## Part 1 dataset design

The seeded generator in `src/practo_support/dataset.py` uses **seed `271828`** and generates **50 records**. It assigns category and status counts by weighted largest-remainder quotas and shuffles those lists with the seeded random generator. Follow-up values are independent seeded random draws at the configured probability:

| Choice | Weights | Generated counts |
|---|---|---|
| General Medicine | 0.40 | 20 |
| Cardiology | 0.20 | 10 |
| Dermatology | 0.16 | 8 |
| Pediatrics | 0.14 | 7 |
| Orthopedics | 0.10 | 5 |
| Scheduled | 0.36 | 18 |
| Completed | 0.24 | 12 |
| Cancelled | 0.14 | 7 |
| No-Show | 0.10 | 5 |
| Rescheduled | 0.16 | 8 |

For each record, `follow_up_required` is drawn with the seeded Bernoulli probability **0.20** (not assigned by editing selected rows); seed `271828` produces **12/50 = 24%**, within the required band. The generator validates the result and rejects a seed whose draw falls outside 10%–30%. Consultation fees are generated in INR, in ₹50 increments, within these category bands: General Medicine ₹300–₹800, Cardiology ₹700–₹2,000, Dermatology ₹500–₹1,500, Pediatrics ₹400–₹1,200, and Orthopedics ₹600–₹1,800. These fabricated ranges make specialty costs distinguishable while keeping the overall dataset within ₹300–₹2,000. The output also assigns `days_since_created` uniformly from integer values 0–30.

Run `python dataset.py` or `practo-dataset` to print the counts and write `data/fixtures/appointments.json`; the root-level entry point delegates to the package generator. Every generated record contains the required `record_id`, `category`, `status`, `consultation_fee_inr`, `days_since_created`, and `follow_up_required` fields.

## Part 1 knowledge base and RAG

The 12 standalone JSON files in `data/policies/` cover appointment booking, cancellation/rescheduling, consultation fees, insurance claims, prescription refills, lab turnaround, telemedicine, emergency visits, privacy, follow-up discounts, second opinions, and home visits. Each document contains 2–5 synthetic policy sentences and metadata. These are capstone examples, not actual Practo policies.

The RAG core embeds all chunks with the local `sentence-transformers/all-MiniLM-L6-v2` model (including each policy title and topic as embedding context) and indexes two separate ChromaDB collections:

- `policy_fixed_overlap_v1`: fixed 180-character chunks with 40-character overlap; currently 24 indexed chunks.
- `policy_sentence_v1`: sentence chunks; currently 36 indexed chunks.

For generation, retrieval results are filtered by an empirically calibrated top-1 cosine threshold. The retrieved parent documents are reconstructed from their indexed chunks, so generation sees complete source passages rather than truncated chunk fragments. Unsupported or low-similarity queries return exactly **“I don't know.”**

The local measurements used three in-scope and two out-of-scope queries:

| Strategy | In-scope top-1 cosine similarities | Out-of-scope top-1 cosine similarities | Chosen threshold |
|---|---|---|---:|
| Fixed overlap | 0.7248, 0.5148, 0.5925 | 0.2567, 0.1099 | **0.3857** |
| Sentence | 0.7970, 0.5033, 0.6652 | 0.2749, 0.1339 | **0.3891** |

Each threshold is the midpoint between the lowest measured in-scope score and highest measured out-of-scope score for that strategy. The two groups were separated in the measured samples; the threshold is not a preset tutorial value.

Precision and recall are measured at the parent-document level on the same five queries, deduplicating retrieved chunks before scoring. Results:

| Strategy | Per-query precision | Per-query recall | Macro precision | Macro recall |
|---|---|---|---:|---:|
| Fixed overlap | 0.500, 0.500, 0.500, 0.500, 0.500 | 1.000, 1.000, 1.000, 1.000, 1.000 | **0.500** | **1.000** |
| Sentence | 1.000, 1.000, 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000, 1.000, 1.000 | **1.000** | **1.000** |

Sentence-based chunking is selected for deployment because it retrieved the relevant document for all five tested queries (1.000 macro recall) and had higher macro precision than fixed overlap (1.000 versus 0.500). Both strategies had 1.000 macro recall on this sample; sentence chunks returned fewer irrelevant parent documents. The exact per-query document sets and arithmetic are in `data/eval/rag_comparison.json`.

## Parts 2–4 orchestration, API, and governance

- CrewAI runs a Retrieval Agent, a Lookup Agent, and a Response Composer through `.kickoff()`. Its deterministic `MockCrewLLM` extends CrewAI's `BaseLLM`; it dispatches tools by declared argument schema. `check_appointment_status` is wired only to the Lookup Agent.
- Lookup returns a designed score: `0.6 * int(follow_up_required) + 0.4 * (days_since_created / 30)`. The escalation threshold is **0.3227**, the recency contribution corresponding to the dataset's 80th percentile of `days_since_created` (**24.2 days**); it incorporates both required follow-up and high recency rather than acting as a boolean lookup.
- LangChain `InMemoryChatMessageHistory` stores turns per conversation ID in process. The demonstrations include a multi-turn repeat request and an independent fresh conversation with no prior state.
- Crew outputs, tool results, public API requests/responses, and AutoGen review verdicts use declared Pydantic models. An input guardrail masks fixed-format contact numbers, blocks prompt injection, and refuses clinical diagnosis/prescription requests or another patient's appointment actions. The output guardrail accepts only answers supported by retrieved passages.
- FastAPI exposes `GET /health`, `POST /ask`, `POST /add-document`, and `WS /ws/chat`. A WebSocket disconnect is handled without taking down subsequent requests.
- HTTP requests and WebSocket messages each emit one JSON Lines log event containing a trace ID, path, outcome, status, and duration. Request bodies are not logged, and contact numbers are masked before the request reaches the service.
- The AutoGen `RoundRobinGroupChat` review uses two agents and `max_turns=2`, returning the Pydantic `VerdictModel`. The demos show an unchanged approval and a revision.
- Runtime requests use a simulated `$0.0005` cost cap at `$0.002 / 1,000` estimated tokens. A full-length request is rejected before crew execution. Reviewed policy responses are cached in memory by normalized query and corpus version; a hit skips both CrewAI and AutoGen calls. Appointment and identity-dependent results are not cached.
- The system is classified **High risk** because the healthcare domain may involve sensitive medical information and affect access to care, despite the fabricated data and refusal controls in this prototype.

**Least-autonomy guard.** The lookup tool is created inside crew construction and assigned only to the Lookup Agent; neither the Retrieval Agent nor the Response Composer receives it, and delegation is disabled. This is an allow-by-construction boundary rather than a prompt instruction, and `artifacts/demonstrations.json` records each agent's tool list with an assertion that only Lookup can access the tool.

**Risk assessment — High.** The system is classified High because the healthcare setting may involve sensitive medical information and its appointment guidance can affect a person's access to care. Synthetic fixtures, contact masking, least-privilege tool wiring, and a refusal boundary reduce exposure in this prototype but do not remove the domain's inherent healthcare risk.

The four governance layers are applied as follows:

| Layer | Enforcement |
|---|---|
| Data | Only fabricated appointments and synthetic policies are present; contact numbers are masked before service processing, and no request text is logged. |
| Model | Deterministic local `MOCK_LLM`, calibrated retrieval threshold, output groundedness guardrail, and Pydantic response validation. |
| Application | Least autonomy: only the Lookup Agent receives `check_appointment_status`; Retrieval and Composer cannot call it. |
| Runtime | Per-request cost cap, trace/timing logs, disconnect-safe WebSocket handling, TTL response cache, and no external model calls. |

## Offline `MOCK_LLM` setup and execution

The acceptance run was validated with Python 3.12.10, needs no paid account, credit card, or API key, and performs no network calls. It expects the SentenceTransformers model to already be in the local Hugging Face cache; embedding initialization is `local_files_only=True`. Before running the acceptance commands, set these Windows PowerShell environment variables:

```powershell
$env:CREWAI_DISABLE_TELEMETRY = "true"
$env:OTEL_SDK_DISABLED = "true"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

The CrewAI telemetry flag is set before CrewAI is imported, avoiding its default outbound telemetry attempt. All agent/reviewer generation and evaluation use deterministic local `MOCK_LLM` implementations.

```powershell
# Either command installs the project dependencies and the test extras.
python -m pip install -r requirements.txt
# Equivalent canonical project install:
# python -m pip install -e ".[dev]"
practo-dataset
practo-ingest
python scripts/evaluate_chunking.py
python scripts/calibrate_rag.py
practo-evaluate
python scripts/run_demos.py
python -m pytest
uvicorn practo_support.api:app --host 127.0.0.1 --port 8000
```

`practo-ingest` builds or updates both local ChromaDB collections. `evaluate_chunking.py` writes both strategies' per-query precision/recall arithmetic to `data/eval/rag_comparison.json`; `calibrate_rag.py` measures the similarity clusters and selects the strategy with the best macro recall (using macro precision to break ties). `practo-evaluate` runs the complete 15-query suite and writes every answer and all four per-query scores plus the averages to `artifacts/evaluation.json`. `run_demos.py` writes memory, guardrail, review, least-autonomy, budget, and cache evidence to `artifacts/demonstrations.json`.

The 15-query evaluation contains at least one query for each required knowledge-base topic and three deliberate out-of-scope cases. Its deterministic `MOCK_LLM` judge reports Accuracy, Grounding, Completeness, and Safety on a 1–5 scale for each query and the average of each metric across all 15. The current report averages are **Accuracy 5.00, Grounding 5.00, Completeness 4.80, Safety 5.00**.

## Run the API

With the local index and calibration data prepared, start the API:

```powershell
uvicorn practo_support.api:app --host 127.0.0.1 --port 8000
```

Example request body for `POST /ask`:

```json
{
  "query": "What is the cancellation window?",
  "conversation_id": "demo-session"
}
```

The WebSocket endpoint is `/ws/chat` and accepts the same fields as `SupportRequest`. The `/add-document` endpoint accepts a `PolicyDocument` marked `synthetic: true`, validates a safe source ID, indexes both collections, and clears the response cache.

## Acceptance evidence

Reproducible evidence is generated in the repository:

- `data/eval/rag_comparison.json` — both strategies, five queries, retrieved parent documents, and visible arithmetic.
- `data/eval/rag_calibration.json` — measured similarity values and selected thresholds.
- `artifacts/evaluation.json` — 15 answers, per-query four-metric scores, and metric averages.
- `artifacts/demonstrations.json` — memory/fresh-session, guardrails, AutoGen review, least autonomy, risk, cost rejection, and cache hit evidence.
- `tests/test_acceptance.py` — focused deterministic coverage including both CrewAI tool invocations, API/WebSocket survival, Pydantic outputs, guardrails, and cache behavior.
