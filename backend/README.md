# NetValor backend

FastAPI service for NetValor's lessons, deterministic finance calculations,
user profiles and memory, and the source-grounded co-pilot. It runs entirely
locally with SQLite, ChromaDB and Ollama.

> NetValor is educational software, not a regulated financial adviser.
> Calculations are estimates with documented assumptions.

## Quick start

Requires Python 3.12+ and, for search and chat, [Ollama](https://ollama.com/).
PowerShell, from the repository root:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env          # then edit as needed
python -m pytest                     # offline: no Ollama needed
python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs> (development only).

### Local demo account

In `.env`, set `SEED_DEMO_USER=true` and a `DEMO_USER_PASSWORD` of at least 12
characters. The `alex` demo profile is created on start-up. Production refuses
this setting. Or register your own account through `POST /api/auth/register`.

### Lesson index and co-pilot

```powershell
ollama pull embeddinggemma-2:270m
ollama pull qwen3.5:9b-q4_K_M
python -m app.knowledge.scripts.ingest   # rebuilds data/chroma
```

The starter lessons are **drafts that haven't been fact-reviewed**. Chat,
search and the lesson list only use approved lessons unless `ALLOW_DRAFT=true`
(local testing only). Answers that use draft material carry a visible notice.

### Calling the API

```powershell
$login = Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/auth/login `
  -ContentType "application/json" `
  -Body '{"user_id":"alex","password":"<your demo password>"}'
$headers = @{ Authorization = "Bearer $($login.access_token)" }
Invoke-RestMethod http://127.0.0.1:8000/api/profile/alex -Headers $headers

curl.exe -N -X POST http://127.0.0.1:8000/api/chat/alex `
  -H "Authorization: Bearer $($login.access_token)" -H "Content-Type: application/json" `
  -d '{\"message\":\"Why does my credit card balance go down so slowly?\"}'
```

The chat stream sends `token` events, then `done` (conversation ID, citations,
verified calculations, grounding checks and notice codes) or `error`
(`{"code", "message"}`).

## Quality checks

```powershell
python -m ruff format --check app tests evals
python -m ruff check app tests evals
python -m mypy app evals tests          # strict mode
python -m pytest --cov                  # about 90% branch coverage
python -m evals                         # offline AI evaluation
python -m evals --live                  # same cases against Ollama and Chroma
```

GitHub Actions (`.github/workflows/backend.yml`) runs all of these, apart from
the live evaluation, on Python 3.12 and 3.13. Two tests that need a running
Ollama skip automatically.

## Architecture

```text
app/
  main.py            app factory, lifespan, middleware, wiring
  config.py          typed settings and production safety checks
  api/               HTTP layer: one router per area, plus deps.py (auth,
                     ownership, rate limits) and errors.py (error handling,
                     body limits, security headers)
  finance.py         Decimal calculation engine (no I/O)
  calc_intent.py     interprets calculation requests in chat; clarifies when unsure
  chat.py            co-pilot pipeline (ChatService)
  grounding.py       evidence filtering, citation filter, grounding checks, safety signals
  retrieval.py       async Chroma retrieval
  llm.py             streaming Ollama client
  knowledge/         lesson parsing, chunking, embeddings, ingestion script
  db.py, migrations/ SQLite connections and numbered SQL migrations
  users.py           accounts and hashed session tokens
  memory.py          profile and memory-note models and storage
  conversations.py   conversation storage (atomic, append-only turns)
  security.py        password hashing, tokens, encryption at rest
  ratelimit.py       rate limiter and concurrency gate
  cli.py             admin commands
evals/               evaluation cases and harness
tests/               pytest suite (no network)
```

Routes depend only on a `Services` container built in the lifespan.
`create_app(settings, llm=..., retriever=...)` lets tests inject fakes, so no
dependency-injection framework is needed.

## Security model

* **Authentication.** `POST /api/auth/register` and `/login` issue a random
  256-bit bearer token. The database stores only its SHA-256 hash, and the token
  expires after `SESSION_TTL_HOURS`. Passwords use scrypt (stdlib) with a
  per-password salt. Unknown users and wrong passwords get the same response and
  take a similar time. `POST /api/auth/delete-account` (password required)
  deletes the account and, through cascades, all of its data.
* **Authorisation.** Every `/{user_id}` route requires a session for that same
  user and returns 403 otherwise. Note and conversation queries are also
  scoped by `user_id` in SQL, so ownership doesn't rely on the routes alone.
* **Personal data at rest.** Profiles, memory notes and messages are encrypted
  with Fernet when `DATA_ENCRYPTION_KEYS` is set, which production requires.
  The keys support rotation. On POSIX the database file is created with mode
  0600. Logs never record message content or financial values.
* **Abuse limits.** Request bodies are capped at 64 KB, including chunked
  bodies. Field lengths and list sizes are bounded, monetary values are
  limited to £10m, and calculations stop after 600 months. Logins,
  registration, chat and search are rate-limited. Concurrent LLM streams are
  capped overall and per user (`429` with `Retry-After`).
* **Errors.** Validation errors don't echo submitted values. Unexpected errors
  return a generic message with a request ID, and the details go only to the
  server log. Dependency failures (Ollama, index) are reported without
  internal paths.
* **CORS and headers.** CORS allows only the listed origins and sends no
  credentials. Responses include `nosniff`, `no-store`, `DENY` framing and
  `X-Request-ID` headers.
* **Production guard.** `ENVIRONMENT=production` refuses to start with draft
  lessons, the demo account, `*` or `http://` CORS origins, enabled docs, or no
  encryption key.

Limits are per process. A multi-worker or multi-host deployment also needs
rate limiting at the proxy. Run uvicorn with `--proxy-headers` behind a trusted
proxy so client IPs are correct. Uvicorn's access log records request paths,
including `/api/search?q=...` queries, so configure log retention to match.

## Calculations

All money is `Decimal`, rounded to the penny with ROUND_HALF_UP. The model
never calculates; it only explains the verified results it is given.
Assumptions (also returned by the API):

* Monthly interest is balance × APR ÷ 12. Real providers usually charge daily,
  and an effective APR includes compounding and fees. This estimate slightly
  overstates the cost rather than understating it.
* Payments are fixed, with no new spending, fees or rate changes.
* APR is a fraction from 0 to 1 (0.249 = 24.9%). "24.9" is rejected rather
  than read as 2,490%.
* Payoff that would take more than 600 months is reported as `exceeds_max_term`.

### Interpreting chat requests (`calc_intent.py`)

Each amount gets a role (balance, regular payment, extra payment, new total
payment, lump sum, savings or essentials) from the nearest cue word in its
clause, and the search never reaches past another amount. For example, in "I
owe **£2,000** on my credit card. What happens if I pay **£50** extra?", £2,000
is the balance and £50 is the extra payment. Missing values come from the
user's saved debt only when exactly one saved debt matches. Each input records
whether it came from the message or the profile.

NetValor asks a clarifying question, without calling the model, when:

* a role is uncertain ("pay £50 on my card");
* two values compete for one role;
* the request is a lump sum or a percentage, which aren't modelled;
* there are several debts and none is named;
* figures are missing.

The follow-up message is merged with the original question.

## Co-pilot pipeline and AI safety

1. Load the profile, notes and history. `use_saved_context: false` skips all
   saved data, and general questions never receive any personal context.
2. Plan calculations, or clarify (see above).
3. Retrieve 6 chunks, then keep at most 4. Chunks are dropped if they are
   unapproved, below `MIN_RETRIEVAL_SCORE`, duplicates, or contain
   instruction-like text. Short follow-ups are expanded with the previous
   question.
4. If there's no evidence and no calculation, give a fixed "lessons don't
   cover this" answer instead of calling the model.
5. Send the model instructions in the system prompt, plus `<sources>`,
   `<saved_context>` and `<verified_calculations>` as escaped, delimited data.
   History is trimmed by message count and characters, and interrupted turns
   are skipped.
6. Stream the answer and remove citations to unknown sources, including
   citations split across tokens and `[S1, S2]` groups.
7. Run post-checks: figures that don't match any calculation, source or user
   text; citations whose sentence shares little with the passage; uncited
   sentences. Unverified figures add a visible caution. Draft content, product
   recommendation requests, money crises and wellbeing risks add fixed
   notices and signposting (MoneyHelper, StepChange, National Debtline,
   Citizens Advice, Samaritans 116 123).
8. Save the turn atomically. If the model fails part-way or the client
   disconnects, the partial answer is saved as `incomplete` and left out of
   future history.

These checks are heuristics. **They reduce, but cannot rule out, misleading
answers.** A valid citation ID doesn't prove the cited passage supports the
claim.

### Evaluation (`evals/`)

`cases.yaml` covers the following categories, with 23 cases in total:

* retrieval relevance and UK terminology;
* calculation interpretation;
* ambiguity;
* missing evidence;
* unsupported figures;
* citation validity and grounding;
* sensitive circumstances;
* prompt injection through user messages, lessons and memory notes;
* privacy.

Offline runs use TF-IDF retrieval over the real lessons and scripted answers,
so they're deterministic and run in CI. `--live` uses the real models. Treat
its output as a report to review, because model output varies.

## Data and migrations

The database lives at `DATABASE_PATH` (default `data/netvalor.db`, which git
ignores). Migrations in `app/migrations/NNNN_*.sql` are applied automatically
at start-up, each one in a transaction. A database from newer code is refused
rather than modified.

**Upgrading from the JSON store.** Older versions kept `data/profiles/*.json`
and `data/conversations/<user>/*.json`. Import them without changing the
originals:

```powershell
python -m app.cli import-json --dry-run
python -m app.cli import-json
python -m app.cli set-password alex     # imported accounts can't log in until this
```

The import can be re-run safely, because existing rows are kept. Other commands:
`python -m app.cli generate-key` and `python -m app.cli create-user <id>`.

## Breaking API changes (v0.2)

* All profile, memory, chat and conversation routes need
  `Authorization: Bearer <token>` and only serve the token's own user.
* `PUT /api/profile/{user_id}` no longer creates accounts; register first.
  Memory notes no longer need an existing profile.
* The demo user is no longer seeded automatically.
* Chat `done` events have a new calculation format (`type`, `inputs`,
  `input_sources`, results, `assumptions`) and new fields: `clarification_needed`,
  `evidence`, `grounding` and `notices`. Citations gain `lesson_id`, `heading`
  and `review_status`. `error` events gain a `code`.
* `GET /api/lessons` and `/api/search` follow the draft policy. Lessons
  include `review_status` and `last_reviewed`.
* `apr`, money fields and list sizes have upper bounds, and unknown JSON
  fields are rejected.

New: `/api/auth/*`, `DELETE /api/memory/{user_id}`,
`DELETE /api/conversations/{user_id}/{id}`, `/api/calc/goal-timeline` and
`/api/health/ready`.
