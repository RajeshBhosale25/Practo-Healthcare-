# Phase-Wise Implementation Plan

## Practo Domain Support Agent

This plan translates the project rubric into sequenced implementation phases
for a Practo (Healthcare) track prototype. It includes a deterministic
appointment dataset, JSON policy knowledge base, dual-collection ChromaDB RAG,
CrewAI agents, AutoGen review, guardrails, session memory, FastAPI HTTP and
WebSocket APIs, structured observability, evaluation, and response caching.

## Delivery principles

- Use synthetic or explicitly authorized de-identified data only.
- At the top of `README.md`, state that the **Practo (Healthcare)** track was
  completed and document the exact dataset seed, category/status weights,
  consultation-fee range, and the reason for that range so generation is
  reproducible.
- All deliverables are code or text within the repository. Do not create or
  submit images, screenshots, diagrams, PDFs, slide decks, presentations,
  video, or audio.
- The full acceptance path must work with no paid account, credit card, API
  key, or runtime network access. Use local SentenceTransformers embeddings and local
  ChromaDB, plus deterministic `MOCK_LLM` for generation, evaluation judging,
  CrewAI, and AutoGen. A real LLM provider may be optional behind an
  environment-variable flag, but is not part of acceptance.
- Make the required SentenceTransformers model available locally before the
  offline acceptance run; document its identifier and local-cache setup.
  Acceptance execution itself must not fetch models or contact a provider.
- Set `CREWAI_DISABLE_TELEMETRY=true` (or `OTEL_SDK_DISABLED=true`) before
  CrewAI runs, and confirm this in `README.md`.
- Keep clinical advice, appointment changes, and access to another person's
  appointment out of scope.
- FastAPI owns request validation, authenticated identity context, and response
  delivery. Ordinary application code owns authorization, permissions, limits,
  and release gates. CrewAI sequences agent tasks but does not grant
  permissions.
- CrewAI provides the Retrieval, Lookup, and Composer agents. AutoGen provides
  a separate review stage. Application code—not either agent framework—
  enforces authorization, governance, and the final release decision.
- Treat user input, retrieved documents, memory, and tool results as untrusted
  data. Only approved policy evidence and authorized appointment-tool results
  can support their respective factual claims.
- Make the minimum viable version work with typed local tools. Add MCP only if
  it has a clear need and a documented, allowlisted boundary.
- Keep retries, agent iterations, tool calls, retrieved context, and output
  size bounded. Fail closed when authorization, review, or validation cannot
  complete.
- Do not log raw conversations or appointment details by default. Do not claim
  production readiness or clinical validation.

## Phase overview

| Phase | Depends on | Main outcome |
|---|---|---|
| 1. Requirements, contracts, and safety design | — | Scope, contracts, threat model, test cases, and dependency choices defined |
| 2. Application foundation and API boundary | Phase 1 | FastAPI HTTP/WebSocket service with trusted request boundary |
| 3. Synthetic data and policy knowledge base | Phases 1–2 | Deterministic appointment fixtures and at least 12 versioned JSON policy documents |
| 4. RAG and grounded generation | Phase 3 | Dual chunking, approved-source retrieval, and cited grounded answers |
| 5. Appointment access, memory, and isolation | Phases 1–3 | Authorized lookup, scoped conversation memory, and deterministic tests |
| 6. CrewAI workflow and AutoGen review gate | Phases 2–5 | Retrieval/Lookup/Composer workflow with independent review |
| 7. Guardrails, resilience, cache, and observability | Phases 2–6 | Enforced controls, safe cache demonstration, structured logs, and metrics |
| 8. Evaluation, integration, and delivery | Phases 2–7 | Acceptance-tested, reproducible, documented capstone |

Phases are ordered by implementation dependency, not by a calendar. Work
within a phase may proceed in parallel when its prerequisites are met. A phase
is complete only when its exit criteria pass.

## Phase 1 — Scope, contracts, and safety design

**Goal:** Resolve the system's boundaries and define the contracts that later
phases will implement.

### Work

1. Confirm in-scope intents: policy question, appointment status, mixed,
   unsupported, and unclear.
2. Document out-of-scope behavior, especially medical diagnosis or treatment,
   appointment writes, unauthorized lookups, and requests to reveal internal
   prompts or data.
3. Define the request and response models, including status and intent enums,
   optional source and appointment fields, follow-up behavior, and
   status-dependent validation.
4. Define the trusted identity/session boundary. For a local demo, specify a
   clearly labelled mock identity mechanism; never infer authorization from
   chat text.
5. Write the governance checklist and threat model, covering prompt injection,
   data isolation, least privilege, sensitive logging, and fail-closed paths.
6. Prepare synthetic policy documents with stable IDs, versions, effective
   dates, and active/superseded/withdrawn states.
7. Prepare synthetic appointment fixtures and a labelled evaluation-case
   inventory for success, ambiguity, denial, not-found, conflict, injection,
   and outage scenarios.
8. Select dependency-compatible Python, FastAPI, Pydantic, CrewAI, AutoGen,
   embedding/vector-store, persistence, and test-library versions. Capture
   configuration needs without adding real credentials.
9. Freeze the Part 1 Task 1 design: seed, weighted category and status
   distributions, consultation-fee range and rationale, and follow-up
   generation method. Record exact values at the top of `README.md`; run the
   generator and adjust the seed/weights and regenerate until all stated
   count/status/follow-up constraints pass. Never hand-edit individual rows.
10. Confirm local/offline operation for the embedding model, ChromaDB,
    `MOCK_LLM`, CrewAI, and AutoGen. Record any one-time model-download
    prerequisite separately from runtime so acceptance runs require no
    network.

### Deliverables

- Request/response and internal evidence/tool-result contract definitions.
- Governance rules and threat-model outline.
- Synthetic data and evaluation case outline.
- Initial architecture and dependency decisions.
- Dependency compatibility record for CrewAI and AutoGen, including their
  distinct orchestration and review responsibilities.
- README opening Practo track statement and exact reproducible dataset design.
- README documentation of local embedding model identifier/cache, offline
  `MOCK_LLM` configuration, zero API-key requirement, and disabled CrewAI
  telemetry.

### Exit criteria

- Every intent has an explicit route or safe response.
- Identity and appointment authorization are owned by trusted application
  code, not an agent.
- Public response fields cannot expose unauthorized appointment data or
  internal reasoning.
- Required safety cases and acceptance criteria are mapped to tests.
- No acceptance test depends on paid credentials, network access, or a real
  patient's data.

## Phase 2 — API foundation and typed contracts

**Goal:** Create a runnable service boundary before integrating retrieval or
agents, with both HTTP and WebSocket interfaces.

### Work

1. Establish the modular application layout for API, models, configuration,
   orchestration, tools, retrieval, governance, memory, persistence, and
   evaluation.
2. Implement FastAPI application startup, configuration loading, request
   correlation IDs, `POST /ask`, and `POST /add-document`, both with Pydantic
   request/response models, plus a health route. Limit `/add-document` to
   schema-validated synthetic or explicitly approved local policy content; it
   must not allow arbitrary content to become authoritative policy.
3. Add a WebSocket support endpoint for interactive conversations. Define
   message envelopes, conversation/user scoping, validation, disconnect and
   timeout behavior, and safe error messages. Handle client disconnects
   by catching `WebSocketDisconnect` without an unhandled exception or
   process crash. Use the same application workflow and governance checks as
   HTTP.
4. Implement typed request/response models and validation for supported
   locales/preferences, input limits, identifiers, enums, and cross-field
   constraints, including structured outputs for agent/tool/reviewer
   boundaries.
5. Define error mapping so invalid requests, unavailable dependencies, and
   unexpected internal failures do not become misleading successful answers.
6. Add a trusted identity dependency. Keep any mock authentication explicitly
   limited to local development.
7. Add baseline unit tests for schemas, health checks, at least two HTTP
   endpoints, WebSocket validation/identity scoping, client-disconnect
   resilience, correlation IDs, and error responses.

### Deliverables

- Runnable FastAPI service with at least two working HTTP endpoints, health,
  modeled `/ask` and `/add-document`, and a WebSocket support endpoint.
- Validated public and internal data contracts.
- Configuration and safe error-handling foundation.
- HTTP/WebSocket API-level tests.
- WebSocket disconnect test demonstrating the server remains healthy and
  continues serving requests afterward.

### Exit criteria

- The service starts with documented local configuration and no embedded
  secrets.
- Invalid input is rejected with predictable HTTP behavior.
- The endpoint cannot accept user-supplied identity as proof of authorization.
- Response-model validation rejects invalid enum or status/field combinations.
- At least two HTTP endpoints work, and a WebSocket client disconnect does
  not crash the endpoint or application.

## Phase 3 — Synthetic data and policy knowledge base

**Goal:** Prepare reproducible appointment fixtures and a sufficiently broad,
versioned policy corpus for retrieval and evaluation.

### Work

1. Implement `dataset.py` (for example,
   `src/practo_support/dataset.py`) with a seeded deterministic generator that
   exports an `APPOINTMENTS` list of at least 40 synthetic records. Every
   record must contain exactly the required core fields: `record_id`,
   `category`, `status`, `consultation_fee_inr`, `days_since_created`
   (integer 0–30), and `follow_up_required` (boolean).
2. Use all five required categories at least once and at least three records
   per category: `General Medicine`, `Cardiology`, `Dermatology`,
   `Pediatrics`, and `Orthopedics`. Use all five required statuses at least
   once: `Scheduled`, `Completed`, `Cancelled`, `No-Show`, and `Rescheduled`.
   Additional categories/statuses are not needed for this rubric.
3. Select and document the exact seed, category weights, status weights, and
   consultation-fee range in the opening section of `README.md`. State the
   fee-range rationale in one sentence. Choose realistic INR fees; keep the
   generator weighted and deterministic, and regenerate with a changed seed
   or weights until constraints pass rather than hand-editing records.
4. Generate `follow_up_required` deterministically so between 10% and 30%
   (inclusive) of records are `True`; compute the percentage from the final
   list. Print and report the count per category, count per status, total
   records, and follow-up percentage from `dataset.py` or its documented
   command.
5. Add tests for record fields/types/ranges, minimum total, every category
   count, required status coverage, follow-up band, exact repeatability for
   the chosen seed/configuration, and lookup scenarios.
6. Create at least 12 separate `.json` files, one policy document per file,
   with 2–5 sentences each, in the author's own words. Cover every required
   topic: appointment booking policy; cancellation/rescheduling window;
   consultation-fee structure by specialty; insurance-claim process;
   prescription-refill policy; lab-test turnaround times; telemedicine
   eligibility; emergency-visit protocol; patient-data privacy policy;
   follow-up-visit discount policy; second-opinion process; and home-visit
   eligibility. Clearly mark synthetic content as synthetic and do not
   represent it as official Practo policy.
7. Give each JSON document a stable ID and the metadata required for
   ingestion (including title, version, effective/review dates, status, and
   source). Define and validate one consistent document schema. Ensure
   specialty-fee JSON content agrees with the generator's documented fee
   range/design choices; keep policy values clearly synthetic.
8. Implement repeatable, idempotent ingestion and verify the stored document
   count and metadata.
9. Keep appointment records out of the general policy knowledge base.

### Deliverables

- `dataset.py` exposing `APPOINTMENTS`, deterministic seeded generation, and
  the required printed counts/percentage.
- README opening section stating Practo (Healthcare) track completion plus
  exact seed, category/status weights, INR fee range, one-sentence rationale,
  and reproducible generation command.
- At least 12 separate, schema-valid `.json` policy files, 2–5 sentences per
  document, covering all 12 specified topics.
- Idempotent local knowledge-base ingestion command and source store.
- Data-generation and ingestion tests, including all rubric thresholds.
- README documentation of the exact structural thresholds and design choices
  used to meet them.

### Exit criteria

- `APPOINTMENTS` contains at least 40 valid records with every required field,
  all five required categories represented at least three times, all five
  required statuses represented, and `days_since_created` in [0, 30].
- Re-running with the exact README configuration produces identical records;
  follow-up `True` is within 10%–30%, and the generator reports category and
  status counts plus that percentage.
- The source corpus contains at least 12 separate valid JSON files, each
  containing one 2–5 sentence document and collectively covering all 12
  required topics, with required metadata.
- JSON policy text is original to this brief, clearly synthetic when
  applicable, and not represented as official Practo policy.
- Ingestion validates each JSON file against the document schema and reports
  malformed files explicitly rather than silently skipping them.
- Re-running generation/ingestion is reproducible and does not create
  unintended duplicates.
- Synthetic materials are visibly identified and are not represented as
  official Practo policy.

## Phase 4 — RAG core and grounded generation

**Goal:** Retrieve traceable, current policy evidence using two chunking
strategies and generate answers constrained to that evidence.

### Work

1. Extract text from the policy JSON files and implement two chunking
   strategies: fixed-size chunks with overlap and sentence-based chunks.
   Preserve document/chunk IDs and source metadata for both.
2. Embed every chunk locally with SentenceTransformers and index each strategy
   into its own separate ChromaDB collection using `collection.upsert()`.
   Keep collection names distinct and document chunking/model configuration.
3. Implement retrieval from either collection, returning top-k chunks with
   relevance/similarity values and metadata. Filter to eligible documents and
   enforce context-size limits.
4. Return typed evidence records with stable document/chunk IDs, title,
   version, status, and supporting excerpts.
5. Implement grounded generation using only retrieved context. Under
   `MOCK_LLM`, return a deterministic response assembled from the supporting
   retrieved text or the exact "I don't know" fallback; do not use model
   memory or unrelated policy content.
6. Demonstrate grounded generation on at least five in-scope queries and one
   out-of-scope query. The out-of-scope query must produce the defined "I
   don't know" fallback rather than unsupported content.
7. Empirically calibrate the fallback threshold before choosing it. Compute
   top-1 cosine similarity (not raw ChromaDB distance) for at least three
   in-scope and two deliberately out-of-scope queries, document the observed
   values/clusters and chosen threshold between them in `README.md`, and test
   it. Do not use an untested preset threshold.
8. Define explicit weak, empty, stale, and conflicting-evidence outcomes:
   clarify, state uncertainty, or escalate rather than fabricate.
9. Run the same sample query set against both collections and verify each
   collection returns sensible, source-traceable results.
10. Compute document-level precision and recall for each collection using the
   same set of at least five queries. Map retrieved chunks back to parent
   document IDs and deduplicate before scoring. Include per-query
   relevant/retrieved counts and visible arithmetic (`precision = relevant
   retrieved documents / total retrieved documents`; `recall = relevant
   retrieved documents / total relevant documents`), using documented
   relevance judgments. Then provide a 2–3-sentence recommendation that cites
   the measured numbers.
11. Add tests comparing both chunking strategies and checking metadata
   preservation, citation validity, filtering, and unsupported-answer
   behavior.

### Deliverables

- Fixed-size-with-overlap and sentence-based chunking modules.
- Two separately indexed ChromaDB collections, one per chunking strategy,
  populated using local SentenceTransformers embeddings and
  `collection.upsert()`.
- Top-k retriever for either collection with source/chunk provenance.
- Grounded generation using only retrieved context and deterministic
  `MOCK_LLM` fallback behavior.
- README evidence for calibrated top-1 cosine threshold measurements and
  chosen threshold.
- RAG tests for current, stale, withdrawn, conflicting, and absent evidence.
- Sample-query retrieval comparison for both collections.
- At least five in-scope grounded-generation examples and one out-of-scope
  "I don't know" fallback example.
- Per-query precision/recall arithmetic for both collections on the same
  minimum five queries, with a numbers-cited recommendation.

### Exit criteria

- Both chunking strategies are runnable and covered by tests.
- Each strategy is embedded and indexed in a separate ChromaDB collection;
  both return sensible results for the same sample query set.
- Both collections use local SentenceTransformers embeddings and are indexed
  through `collection.upsert()`.
- Grounded generation is demonstrated on at least five in-scope queries and
  one out-of-scope query that triggers the "I don't know" fallback.
- The fallback threshold is based on measured top-1 cosine similarities from
  at least three in-scope and two out-of-scope queries, with measurements and
  threshold documented in README.
- Precision and recall are calculated for both collections on the same
  minimum five queries, with visible per-query arithmetic and a
  numbers-supported recommendation.
- Retrieved chunks retain their source/version provenance.
- Withdrawn or superseded documents are excluded from ordinary current-policy
  answers.
- Every policy citation resolves to a retrieved source/chunk; unsupported
  content produces uncertainty or escalation.

## Phase 5 — Appointment access, memory, and isolation

**Goal:** Add deterministic, read-only appointment lookup behind a trusted
authorization boundary.

### Work

1. Implement the typed tool
   `check_appointment_status(record_id: str) -> dict` against the deterministic
   `APPOINTMENTS` list. For a found record, return `status`,
   `consultation_fee_inr`, and `escalation_score`; for an unknown ID, return a
   defined not-found result without inventing details.
2. Compute an `escalation_score` in `[0, 1]` with this documented formula:
   `0.6 * int(follow_up_required) + 0.4 * (days_since_created / 30)`.
   The follow-up flag has greater weight because it is an explicit support
   signal; elapsed appointment age is normalized to [0, 1] as a secondary
   signal.
   Set the escalation recommendation threshold to the 80th percentile of
   `escalation_score` for the generated `APPOINTMENTS`; calculate and record
   its actual value, percentile method, and score distribution in README,
   then test records above and below it. This is a graded combination, not a
   boolean OR.
3. Add explicit tool tests for known/unknown IDs, returned fields, score
   boundaries, and representative cases above/below the threshold.
4. Add explicit timeouts and ensure private appointment results are never
   cached across users.
5. Integrate the deterministic records generated by `dataset.py` as the local
   lookup fixture.
6. Implement in-process, session-based memory with LangChain
   `InMemoryChatMessageHistory` and `RunnableWithMessageHistory`, keyed by
   conversation ID. A `LangChainDeprecationWarning` about LangGraph
   persistence is expected for this rubric and need not be suppressed. Memory
   need not survive process restart.
7. Define clear/reset behavior and minimization. Do not store secrets or
   unnecessary patient information. Allow memory to resolve a prior reference
   only when it is unambiguous;
   memory must never act as policy evidence or authorization evidence.
8. Add tests for authorized and unauthorized lookup, not-found and outage
   cases, field allowlisting, retention, reset, ambiguous references, and
   cross-user isolation. Demonstrate a successful multi-turn reference in one
   conversation and a separate fresh-conversation transcript where that
   context is absent.

### Deliverables

- Appointment lookup adapter and typed results.
- `check_appointment_status` tool with a documented and tested
  `escalation_score` calculation.
- In-process LangChain session memory with reset behavior and isolation tests.
- Authorization and tool tests using the deterministic dataset.
- Multi-turn memory transcript and separate fresh-conversation transcript
  proving context isolation.

### Exit criteria

- Authorization failures reveal no appointment fields.
- Appointment details are sourced only from a successful authorized lookup.
- The tool returns status, fee, and an `escalation_score` in `[0, 1]`; its
  exact formula combines follow-up with normalized recency, and its
  data-derived escalation threshold and justification are in README.
- A lookup outage or malformed result cannot be converted into invented
  status.
- Context can be retrieved only within the correct user/session and
  conversation scope; expired or cleared context is not used.
- Memory cannot authorize an appointment lookup or support a policy claim.
- The fresh-conversation transcript demonstrates that prior conversation
  details are not implicitly available.

## Phase 6 — CrewAI workflow and AutoGen review gate

**Goal:** Connect the API, router, evidence/tools, writer, and independent
reviewer into a bounded workflow.

### Work

1. Define CrewAI agents/tasks with explicit typed inputs/outputs and
   least-privilege tools:
   - **Retrieval agent:** call the approved RAG retriever and return evidence
     with provenance; no appointment access.
   - **Lookup agent:** call only the authorized, read-only appointment lookup;
     no policy-store or arbitrary database access.
   - **Composer agent:** draft a response from verified retrieval/lookup
     results; no direct unrestricted tools.
2. Implement deterministic CrewAI `MOCK_LLM` by extending
   `crewai.llms.base_llm.BaseLLM`; do not externally intercept or redirect
   provider calls. Test that its parser inspects the generated model text,
   never the ReAct system prompt's example `"Observation:"` text. Dispatch
   tool arguments from each tool's declared argument schema, never by
   substring matching tool names (for example, `rag_lookup` is a RAG tool,
   not an appointment lookup).
3. Set `CREWAI_DISABLE_TELEMETRY=true` or `OTEL_SDK_DISABLED=true` before
   importing/running CrewAI and verify no telemetry network call is made.
4. Keep intent routing and hard capability selection in application code;
   agents cannot grant themselves tool access.
5. Route only the capabilities required for the request. Clarify ambiguous
   requests instead of invoking risky tools.
6. Supply the Composer with bounded evidence and typed tool results; do not give
   it arbitrary database or filesystem access.
7. Implement a distinct **AutoGen review stage** after composition. Give the
   reviewer the draft, supporting evidence/tool results, and governance
   checklist. Require it to return a structured approve/reject result and
   reason codes. It must not have lookup or retrieval permissions and cannot
   modify governance or grant access.
8. Implement the review team as a two-agent AutoGen `RoundRobinGroupChat`
   (Policy-Compliance-Reviewer and Final-Editor), bounded with
   `max_turns=2`. If using `MaxMessageTermination` instead, account for the
   initiating task message and use `MaxMessageTermination(3)` so both agents
   can speak. Give Final-Editor Pydantic output via `output_content_type` and
   register the corresponding `StructuredMessage[VerdictModel]` in the team's
   `custom_message_types`. Use a deterministic AutoGen-compatible mock model
   client under `MOCK_LLM`, with no API key or network call.
9. Use a verdict model with `approved: bool`, `final_answer: str`, and
   `reason: str`. On at least two sample queries, demonstrate approval of one
   draft unchanged and revision of a deliberately deficient draft containing
   an ungrounded claim; validate both verdicts and show both released answers.
10. Implement the application-owned release gate: approve, allow one bounded
   revision, or choose a safe escalation/refusal. Do not return an unreviewed
   draft.
11. Ensure every crew run uses a declared Pydantic `response_format` schema;
   validate every response and reject invalid outputs before release.
12. Enforce least autonomy in application wiring: only the Lookup Agent may
   invoke `check_appointment_status`; demonstrate that Retrieval and Composer
   cannot call it.
13. Classify the system risk using the rubric's scheme: Low for
   summarization/transcription, Medium for code generation/customer-support
   tickets, and High for medical data, hiring, or financial data. Classify
   this healthcare appointment-support system as High because it processes
   health-related appointment context, despite using synthetic fixtures in
   this prototype; include this paragraph and its justification in README.
   Risk classification may restrict the workflow but must not replace
   authorization.
14. Implement a runtime token/cost budget cap that rejects an oversized
   simulated request before unbounded agent/tool execution; test and show the
   rejection.
15. Demonstrate agent/tool execution via CrewAI `.kickoff()` on separate
   policy-retrieval and appointment-status queries. Capture evidence that the
   RAG tool is invoked for the former and `check_appointment_status` for the
   latter.
16. Validate the final response with the public schema and verify all cited
   sources and appointment fields against their originating results.
17. Keep direct typed Python tools as the minimum implementation. If MCP is
   added, expose only documented, authenticated, narrowly scoped operations;
   do not expose arbitrary queries, filesystem access, or appointment writes.
18. Add workflow tests for policy-only, appointment-only, mixed, unclear,
   unsupported, reviewer rejection, revision, and escalation paths.

### Deliverables

- CrewAI Retrieval, Lookup, and Composer agents with explicit tool
  permissions.
- Two-agent AutoGen `RoundRobinGroupChat`, bounded to two turns, with
  registered Pydantic structured message type.
- Verdict model containing `approved`, `final_answer`, and `reason`.
- AutoGen demonstrations showing unchanged approval and actual revision of
  an intentionally ungrounded draft, with both verdicts validated.
- Application-owned workflow coordinator and release gate.
- Declared Pydantic output schemas for all crew responses.
- Least-autonomy test showing only Lookup can invoke
  `check_appointment_status`.
- Low/Medium/High risk classification with the required paragraph rationale.
- Runtime token/cost budget cap and oversized-request rejection evidence.
- `.kickoff()` execution evidence showing separate RAG-tool and appointment
  tool invocations on their corresponding queries.
- End-to-end orchestration tests with deterministic tool fixtures.
- CrewAI `MOCK_LLM` adapter extending `crewai.llms.base_llm.BaseLLM`.
- Tests for ReAct-output parsing and schema-driven tool-argument dispatch.
- README confirmation of disabled CrewAI telemetry and zero-network mock run.

### Exit criteria

- Only the relevant permitted tool is invoked for each request.
- Reviewer rejection cannot cause an unbounded loop or release an unsafe
  answer.
- Reviewer or schema-validation failure selects a defined safe outcome.
- Every crew response validates against its declared Pydantic schema.
- AutoGen demonstrates both unchanged approval and revision with structured
  Pydantic verdicts.
- The AutoGen team uses two agents and is bounded to two turns, with
  structured messages registered as required by its runtime.
- Only Lookup can invoke the appointment tool; the restriction is tested.
- The system risk level is selected from Low/Medium/High and justified using
  the rubric's definitions.
- An oversized simulated request is rejected by the runtime budget cap.
- `.kickoff()` demonstrations prove that the RAG tool and
  `check_appointment_status` are invoked on different, appropriate requests.
- CrewAI runs offline under `MOCK_LLM` using the documented `BaseLLM`
  extension; no outbound telemetry request occurs.
- `MOCK_LLM` correctly parses generated text rather than ReAct template
  examples and dispatches tool inputs by declared argument schema.
- All public claims and citations are traceable to the workflow's verified
  evidence.

## Phase 7 — Governance, resilience, and observability

**Goal:** Enforce hard safety rules, make downstream failures predictable,
demonstrate safe response caching, and record operational metadata without
unnecessary sensitive data.

### Work

1. Implement deterministic governance checks for scope, authorization,
   appointment field disclosure, source/citation validity, status-dependent
   response fields, and reviewer approval. Put these checks, tool allowlists,
   request budgets, iteration limits, and release checks in ordinary
   application code rather than relying on prompts.
2. Document and demonstrate the four governance layers:
   - **Data:** synthetic-only appointment/patient examples, approved-source
     policy metadata, contact-number masking, and minimized memory/logging.
   - **Model:** local SentenceTransformers embeddings, deterministic
     `MOCK_LLM` for generation/judging/agents, and schema/grounding checks.
   - **Application:** least-autonomy tool wiring, risk classification,
     authorization, AutoGen review, and the release gate.
   - **Runtime:** per-request token/cost cap, call/iteration/time limits,
     offline execution, and fail-closed handling.
3. Apply input and output limits and governance pre-checks. Treat user text,
   retrieved documents, memory, and tool results as untrusted.
4. Implement input-side PII masking and prompt-injection detection plus an
   output-side groundedness guardrail. Mask the brief's fixed-format contact
   number in both the model input and logged request text, and demonstrate the
   mask firing. Name, diagnosis/condition, and insurance ID have no universal
   format for the keyless mock masker: explicitly state these fields are out
   of masking scope and use fabricated examples only. Demonstrate injection
   detection and groundedness refusal on deliberate cases.
5. Add an in-memory response cache for the grounded-generation step, keyed by
   normalized query text and the corpus/version fingerprint to prevent stale
   policy reuse. Store the generated draft together with its supporting
   evidence so a hit can be reviewed without another retrieval/generation
   call. Demonstrate a first miss and repeated-query hit with retrieval,
   `MOCK_LLM` generation, and tool call counters or timing evidence showing
   the hit avoids redundant RAG/tool/generation work. Never cache appointment
   results or identity/conversation-dependent
   responses. Continue to run the AutoGen review and application
   governance/release checks on a cached draft, then validate its response
   schema.
6. Add privacy-conscious JSON Lines structured logging with correlation IDs
   and timing. Emit exactly one parseable JSON object per HTTP request and
   per inbound WebSocket message, each containing a trace ID and safe
   operational metadata. Apply contact-number
   masking before logging and before model input so the raw value never
   reaches disk or the model. Use fabricated names, diagnoses/conditions,
   and insurance IDs only.
7. Define safe handling for LLM timeout, vector-store outage, empty retrieval,
   appointment-service failure, rate limit, invalid tool result, reviewer
   failure, malformed output, and exhausted budget.
8. Ensure human escalation language does not claim a ticket was created
   unless a configured ticketing integration confirms it.
9. Emit evaluation/operational metrics for response status, retrieval quality,
   valid citations, tool usage/errors, reviewer approvals/rejections,
   latency, token usage, and estimated cost. Keep aggregate metrics separate
   from sensitive payloads.
10. Report Accuracy, Grounding, Completeness, and Safety for every one of 15
   evaluation queries under `MOCK_LLM`, then calculate and report the four
   corresponding averages. Preserve per-query scores in the evaluation
   output.
11. Test cache isolation/invalidation, structured log redaction, metrics
   emission, and fail-closed behavior for authorization, privacy, validation,
   and required review. Verify that cache hits cannot skip the reviewer or
   governance gate.

### Deliverables

- Enforced governance and redaction components.
- Explicit, user-safe failure and escalation outcomes.
- Structured logs, correlation IDs, and privacy-conscious audit fields.
- Evaluation/operational metrics for quality, tools, review, latency, and cost.
- One JSONL log entry per request with a trace ID.
- Per-query Accuracy, Grounding, Completeness, and Safety scores for all 15
  `MOCK_LLM` queries, plus the four averages.
- Demonstrations that input PII/injection and output groundedness guardrails
  each fire on deliberate test cases.
- Demonstrable safe response caching with tests proving appointment results
  are not cached or shared.
- Repeated-query cache demonstration showing an initial miss and subsequent
  hit with before/after evidence.
- Governance and failure-path regression tests.

### Exit criteria

- Injection attempts cannot expand permissions, bypass review, or change
  system governance in the test suite.
- Authorization, review, and validation failures do not release protected or
  unsupported details.
- Every known dependency failure maps to a predictable safe status and
  next step.
- Logs and audit records exclude unnecessary raw sensitive payloads.
- Every request produces exactly one structured JSONL log entry with a trace
  ID.
- All 15 `MOCK_LLM` evaluation queries have all four required scores, and the
  four averages are calculated from those per-query results.
- Both guardrails are shown firing on their respective deliberate cases.
- Cache tests demonstrate safe policy-response reuse and no cross-user or
  appointment-result caching; cache hits still pass review and governance.
- A repeated query produces observable miss-then-hit behavior with recorded
  evidence, while still passing review/governance on the hit.

## Phase 8 — Evaluation, integration, and delivery

**Goal:** Validate the complete system against the project acceptance
criteria, measure quality and resource use, and make delivery reproducible.

### Work

1. Complete a labelled synthetic evaluation set covering supported policy
   questions, no-evidence questions, stale/conflicting sources, appointment
   success/not-found/unauthorized/outage, ambiguous references, prompt
   injection, medical advice, reviewer rejection, HTTP/WebSocket behavior,
   cache hits/misses, and cache isolation.
2. Implement a repeatable evaluation runner with per-case outcomes and
   categorized failures.
3. Measure correctness/groundedness, citation validity, appointment lookup
   correctness, schema validity, privacy/guardrail behavior, reviewer
   decisions, latency, tool errors, and estimated token/cost use.
4. Enforce configurable limits for LLM calls, agent iterations, retries,
   retrieved chunks, context/output tokens, timeouts, and estimated
   per-request cost.
5. Build exactly 15 evaluation queries for the required report: include at
   least one touching each of the 12 required policy topics and at least two
   deliberately out-of-scope or edge-case queries; use the remaining query
   for an additional in-scope or boundary case.
6. Define an LLM-as-judge prompt for Accuracy, Grounding, Completeness, and
   Safety, and execute it through the deterministic `MOCK_LLM` judge adapter.
   Report each of the four scores for every query and calculate the
   arithmetic mean (sum of each query's score divided by 15) for each
   dimension. Include per-query scores and all four averages in the evaluation
   artifact. Grader runs must not call a real provider.

### Deliverables

- Evaluation dataset and repeatable runner.
- Integrated API, retrieval, tool, memory, orchestration, and release workflow.
- Configured request budgets and a summarized evaluation report.
- Per-query metric report for all 15 `MOCK_LLM` cases and four calculated
  averages.
- Exactly 15 evaluation cases covering all 12 policy topics and at least two
  out-of-scope/edge cases, evaluated with deterministic mock judging.
- RAG comparison report with per-query precision/recall arithmetic for both
  ChromaDB collections and a recommendation supported by measured values.
- Setup, architecture, API, governance, threat-model, and run documentation.

### Exit criteria

- Evaluation is repeatable and reports per-case failures.
- Exactly 15 evaluation queries run under `MOCK_LLM`; at least one covers
  every required policy topic and at least two are deliberately out of scope
  or edge cases. Every query reports all
  four scores (Accuracy, Grounding, Completeness, Safety) and each metric's
  average is calculated across the 15 queries.
- The retrieval comparison uses the same minimum five queries for both
  collections and includes visible per-query precision/recall arithmetic
  plus a numbers-cited recommendation.
- Dataset tests verify `dataset.py` deterministically produces at least 40
  appointment records, and corpus checks verify at least 12 policy documents
  covering the required operational topics.
- HTTP and WebSocket endpoints are exercised through representative success,
  validation, authorization, and failure cases.
- RAG evaluation exercises both chunking strategies and citation grounding.
- CrewAI Retrieval/Lookup/Composer flow and the separate AutoGen review stage
  are tested, including reviewer rejection and bounded revision.
- Unauthorized appointment disclosure and tested prompt-injection policy
  bypasses are zero in the suite.
- Resource ceilings are enforced, including on failure/retry paths.
- Cache demonstration verifies policy response hit/miss behavior and excludes
  private appointment results.
- The report includes known limitations and does not claim clinical
  validation.

### Integration and delivery work

1. Run unit, integration, security, and regression tests; fix critical
   failures and rerun affected suites.
2. Verify local setup from a clean environment using documented dependency,
   configuration, ingestion, seed-data, and run commands.
3. Complete architecture, API examples, environment-variable, ingestion,
   governance, threat-model, evaluation, known-limitations, and run
   documentation.
4. Containerize only if it supports the intended demonstration or deployment
   path; packaging is optional and must not displace required safety and test
   coverage.
5. Demonstrate grounded policy answers, authorized appointment lookup,
   clarification, unauthorized lookup, conflicting/unsupported policy, and
   dependency failure.
6. Verify health checks, response schemas, source traceability, safe audit
   records, and escalation wording.
7. Present evaluation results and distinguish prototype behavior from
   production-grade authentication, integrations, and compliance.

### Final deliverables

- Runnable, documented service and complete source code.
- Passing automated test suites and evaluation report.
- README with exact dataset threshold choices, generation rationale, run
  instructions, and required demonstrations.
- RAG retrieval comparison with per-query precision/recall arithmetic and
  numbers-based collection recommendation.
- Grounded-generation examples and out-of-scope fallback.
- Crew `.kickoff()` tool-invocation evidence, memory/fresh-conversation
  transcripts, guardrail demonstrations, AutoGen review demonstrations,
  risk/budget evidence, cache miss/hit evidence, and structured logs.
- Text-only README walkthrough including successful and safely rejected/
  failed cases.
- Known-limitations and production-readiness distinction.

## Required deliverables checklist

The completed project must include each concrete deliverable below, in
addition to the phase-specific outputs:

| Requirement | Required deliverable | Verification |
|---|---|---|
| Deterministic appointment dataset | `dataset.py` (for example, `src/practo_support/dataset.py`) generating at least 40 synthetic appointment records | Repeat generation with the same configuration and compare records; validate count and schemas |
| Healthcare policy knowledge base | At least 12 separate, schema-valid `.json` files, one versioned policy document per file, covering the Phase 3 topics | Validate JSON/schema, file and document counts, topic coverage, metadata, and synthetic/approved status |
| Dual-strategy RAG | Two selectable chunking strategies, retrieval, provenance, and grounded generation | Run retrieval and citation tests against both chunking strategies |
| CrewAI orchestration | Retrieval, Lookup, and Composer agents with least-privilege tool access | Exercise policy, appointment, mixed, ambiguous, and failure workflows |
| Guardrails, structured outputs, and memory | Application-enforced governance, validated Pydantic contracts, and scoped memory with retention/reset behavior | Test invalid outputs, unauthorized access, injection, retention, and cross-user isolation |
| FastAPI deployment surface | Health check, HTTP chat endpoint, and interactive WebSocket endpoint | Run endpoint tests for valid requests, validation errors, authorization, disconnects, and downstream failures |
| Structured logging and evaluation metrics | Correlated, redacted operational logs and metrics for quality, tools, reviewer, latency, tokens, and cost | Assert required metadata is emitted and sensitive payloads are not logged |
| AutoGen review and governance enforcement | Separate structured review stage followed by an application-owned release gate | Verify approve/reject, bounded revision, escalation, and fail-closed behavior |
| Response-cache demonstration | Safe cache for eligible non-personal policy responses with expiry/invalidation and visible hit/miss behavior | Verify cache reuse and invalidation; appointment and identity-dependent results are never shared/cached; every hit still passes AutoGen review and governance |

## Rubric-aligned acceptance checks

### Part 1 — Data, knowledge base, and RAG

- [ ] The top of `README.md` states completion of the **Practo (Healthcare)**
  track and gives the actual generator seed, category/status weights, INR
  amount range and one-sentence rationale, plus reproducible run instructions.
- [ ] `dataset.py` exports a deterministic `APPOINTMENTS` list of at least 40
  records with exactly the required fields; each of the five given
  categories appears at least three times, each given status at least once,
  and `days_since_created` is an integer from 0 through 30.
- [ ] The generator prints total count, per-category counts, per-status
  counts, and `follow_up_required=True` percentage between 10% and 30%;
  re-running with README choices is identical and records were not hand-edited
  to meet the percentage.
- [ ] At least 12 separate, original `.json` policy documents contain 2–5
  sentences each and cover all 12 required topics.
- [ ] Fixed-size-with-overlap and sentence-based chunks are both embedded
  locally with SentenceTransformers and indexed with `collection.upsert()` in
  separate ChromaDB collections.
- [ ] Both collections return sensible results for the same sample query set.
  Top-1 cosine values for at least three in-scope and two out-of-scope queries
  empirically justify the "I don't know" threshold documented in README.
- [ ] Grounded generation uses only retrieved context and demonstrates at
  least five in-scope queries plus one out-of-scope query that returns "I
  don't know".
- [ ] For the same minimum five queries, document-level precision and recall
  are computed separately for both collections after chunk-to-document
  mapping and deduplication; per-query arithmetic and a numbers-based
  2–3-sentence recommendation are included.

### Part 2 — Appointment tools, agents, memory, and guardrails

- [ ] `check_appointment_status(record_id: str) -> dict` retrieves the correct
  record and returns status, consultation fee, and
  `0.6 * int(follow_up_required) + 0.4 * (days_since_created / 30)` as its
  `[0,1]` escalation score; README records the data-derived 80th-percentile
  threshold and actual value.
- [ ] The CrewAI crew has at least Retrieval, Lookup, and Composer agents.
  Separate `.kickoff()` demonstrations show the RAG tool being invoked for a
  policy query and `check_appointment_status` for an appointment query.
- [ ] Multi-turn memory works within its conversation scope; a separate fresh
  conversation transcript shows prior context is absent.
- [ ] Every crew response validates against its declared Pydantic
  `response_format` schema.
- [ ] LangChain in-process session memory is shown working across multiple
  turns, with a separate fresh-session transcript showing no prior context.
- [ ] Input-side fixed-format contact-number masking and prompt-injection
  detection, and output-side groundedness refusal, each fire in deliberate
  tests. Logs and model inputs contain the masked contact value; all
  free-text PII examples are fabricated and their masking limitation is
  documented.

### Part 3 — API, logging, and evaluation

- [ ] FastAPI exposes Pydantic-modeled `/ask` and `/add-document` HTTP
  endpoints plus a WebSocket endpoint; `WebSocketDisconnect` is caught, and
  the service remains available after a client disconnect.
- [ ] Every HTTP request and inbound WebSocket message produces exactly one
  JSONL log entry containing a trace ID and timing; no raw fixed-format
  contact number is logged.
- [ ] Exactly 15 queries run under `MOCK_LLM`, including all 12 required
  topics and at least two out-of-scope/edge cases. A deterministic
  LLM-as-judge prompt reports Accuracy, Grounding, Completeness, and Safety
  per query and each of the four averages.

### Part 4 — Review, governance, budgets, and cache

- [ ] Two-agent AutoGen `RoundRobinGroupChat` is bounded with
  `max_turns=2` (or the specified equivalent), registers structured
  `VerdictModel` messages, and demonstrates unchanged approval and actual
  revision of an intentionally ungrounded draft.
- [ ] Application wiring makes the Lookup Agent the only agent able to call
  `check_appointment_status`; risk is classified High and justified using
  the rubric scheme.
- [ ] Runtime token/cost budget cap rejects an oversized simulated request.
- [ ] Repeating a normalized query shows before/after cache evidence and
  avoids redundant retrieval, `MOCK_LLM` generation, and tool work; no
  appointment result is cached, and every cached draft still passes
  review/governance.
- [ ] All acceptance demonstrations execute offline with no API keys under
  `MOCK_LLM`; CrewAI telemetry is disabled. Deliverables are repository code
  or text only—no visual/media artifacts.

### Final acceptance criteria

- A developer can reproduce setup and run the service using the documentation.
- Acceptance criteria in the problem statement are covered by tests or
  explicitly documented as optional extensions.
- End-to-end behavior matches the validated API contract and safe-failure
  requirements.
- No unsupported claim is made that the prototype is clinically validated or
  production-certified.

## Cross-phase verification checklist

Use this checklist at phase reviews and before the final demonstration:

- [ ] Identity is established by the API trust boundary, not inferred from
  user text.
- [ ] Policy claims cite approved, current evidence with valid source
  metadata.
- [ ] Appointment details come only from an authorized successful lookup and
  use only allowlisted fields.
- [ ] Memory is scoped to both conversation and user/session and follows
  retention/reset rules.
- [ ] Agents have only the tools required for their role.
- [ ] CrewAI Retrieval, Lookup, and Composer agents use least-privilege tools;
  the separate AutoGen reviewer and final schema/governance checks are
  required before release.
- [ ] HTTP and WebSocket requests share identity, governance, and response
  validation rules.
- [ ] At least 40 deterministic synthetic appointment records and 12
  separate, versioned `.json` policy documents covering the required topics
  are available.
- [ ] Both RAG chunking strategies preserve source provenance and produce
  grounded citations.
- [ ] Structured logs and evaluation metrics avoid unnecessary sensitive
  payloads.
- [ ] The response-cache demonstration excludes appointment results and
  proves appropriate cache scope/invalidation; cache hits still pass the
  reviewer and governance release gate.
- [ ] Retries, review cycles, tool calls, context, output, and cost are
  bounded.
- [ ] Injection, unsupported medical requests, conflicts, ambiguity, and
  outages result in the documented safe behavior.
- [ ] Operational records are correlated and useful without storing
  unnecessary sensitive payloads.
- [ ] Evaluation results and known limitations are documented; no clinical or
  production certification is implied.
