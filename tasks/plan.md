# ContextManager — Plan

> Offline project. Goal: let a low-context local model run an agentic session
> **almost forever without losing context**, and kill the Hermes compaction loop.

## Problem

Running Qwen3.6-35B-A3B (IQ4_XS, `-c 75776`) under **Hermes Agent** for an agentic
build (Spaceshooter demo). After a point Hermes compacts repeatedly — each compaction
re-triggers another — destroying performance. Raising the context window only delays
the lockout; it never removes it.

## Root cause (grounded in Hermes Agent internals)

- Compaction fires when `prompt_tokens >= threshold * context_length` (default
  `threshold: 0.50`).
- Protected tail keeps `max(protect_last_n msgs, threshold_tokens * target_ratio)` —
  *"whichever protects more"* (defaults: 20 msgs **or** ~20% of threshold).
- **Documented gap:** *"if the tail itself exceeds the threshold after compression,
  there is no mechanism to re-compress."*
- Our workload puts **big tool outputs (file reads, code, game-state) into recent
  messages** → protected tail alone exceeds the 50% line → compaction can't reduce it →
  fires fruitlessly every turn = livelock.
- **Silent-loss trap:** if the summarizer (`auxiliary.compression`) model context <
  main model context, Hermes drops the middle with no summary (`_generate_summary()`
  returns `None`). Keep them equal.

Full mechanics + sources: [[hermes-compaction]] (wiki).

## Goal / success criteria

- [x] No fruitless re-compaction loop, ever (mathematically bounded). — PROVEN in the core
      engine: `budget.py` rejects any config where `low_water >= high_water` (incl. the
      integer-truncation collapse case), and `compactor.py` raises a real (`-O`-safe) exception
      if `load_after > low_water`; since `low_water < high_water` by construction,
      `load_after <= low_water < high_water` ⇒ no re-fire. Hypothesis property test passes.
      Live: the shipped proxy ran a 138-msg/73-tool Hermes session with no livelock.
- [x] Retained context ~O(1) vs session length — by construction: the budget is a fixed
      fraction of `n_ctx`; full content is externalized to the `DurableStore` and the oldest
      window messages are paged out (mechanical fallback) so the in-context size stays bounded.
- [x] No silent information loss — `page_out` always persists to the store BEFORE a message
      leaves the window; `page_in` / `context_rehydrate` page it back on demand. (Contrast
      Hermes, which drops the middle with no summary when the summarizer ctx < main.)
- [x] Universal — Surface A proxy works with any OpenAI-compatible CLI (no cooperation);
      Surface B MCP works with any MCP CLI. Both validated/registered live (Hermes + OpenCode).

## Review (2026-06-23, full audit)

Verdict: **the implementation faithfully realizes this plan and exceeds it in robustness.**
Architecture is 1:1 (tokenizer with all four endpoints + fallback; tiered budget; hysteresis
compaction; sealing; mechanical fallback; proxy handle-ization; the six MCP tools; state.json +
notes + lexical retriever). Beyond spec: defensive cap enforcement, integer-water-collapse
guard, truncation-overshoot handling, `/metrics`, `/v1/props`, console scripts, offline
`HeuristicTokenCounter`, cross-surface shared-store test. 224 tests, 0 skips, no TODO/stubs.
Honest deltas: (a) the `HysteresisCompactor` engine is built + tested but NOT on either shipped
surface's critical path — the proxy delegates actual compaction to the host CLI and just keeps
it rarely-needed via handle-ization (matches the Surface-A design here); wiring a
governor-owned compaction mode is a ready future step. (b) Phase 5's formal A/B/C measurement
table is still pending (proxy side proven at ~60% wire reduction; baseline/Tier-0 not run).

---

## Tier 0 — Immediate relief: Hermes `config.yaml` patch (no code)

Stops the catastrophic loop today. `compression.*` and `model.context_length`
hot-reload on a running gateway.

```yaml
model:
  context_length: 75776        # match llama-server -c

compression:
  enabled: true
  threshold: 0.72              # was 0.50 — fire later, leave real headroom
  target_ratio: 0.12           # was 0.20 — smaller protected-tail token budget
  protect_last_n: 8            # was 20 — fewer protected tail messages
  protect_first_n: 3
  hygiene_hard_message_limit: 5000

auxiliary:
  compression:
    model: ""                  # "" = same local model (ctx == main, OK)
    provider: "auto"
    # NEVER point this at a smaller-context model: if summarizer ctx < main,
    # Hermes drops the middle SILENTLY (no summary) instead of compressing.
```

- [ ] Locate the active `config.yaml` and apply the patch.
- [ ] Verify the context-pressure bar stabilizes (no per-turn compaction).

**Limit of Tier 0:** it bounds tail *count* and *token budget*, but NOT individual
message size. One 40K-token file dump in the protected tail still breaks it. That
residual is what the build below removes.

---

## Architecture — the Context Governor

```
Hermes CLI ──/v1/chat/completions──▶ [ Context Governor PROXY ] ──▶ llama-server (Qwen3.6 IQ4_XS)
                                          │  ▲              │
                              externalize │  │ retrieve     │ /tokenize · /props (exact counts)
                                          ▼  │              ▼
                                 [ STORE: state.json · wiki/*.md · vector index ]
                                          ▲
  any MCP-capable CLI ──MCP tools──▶ [ Governor MCP SERVER ] ──┘
```

**Core engine** (shared)
- Exact token accounting via `llama-server` `/tokenize` + `/props` (n_ctx).
- Tiered budget: pinned spec · state snapshot · sealed distilled memory · rolling
  window + retrieved slices · headroom.
- Hysteresis controller: high-water trigger, low-water target; floor (pinned + state +
  distilled cap) <= low-water *by construction*; summary capped by **measurement +
  truncation**, never by trusting the model. Post-compaction load < trigger ⇒ no
  re-fire. Mechanical fallback (drop oldest) guarantees forward progress.

**Surface A — endpoint proxy (universal, no cooperation)**
- Transparent OpenAI-compatible proxy; set Hermes `model.base_url` → proxy.
- Replaces bulky/stable messages (big tool outputs, file dumps) with **handles +
  short slices**; stores full content. Because Hermes measures *API-reported* prompt
  tokens, shrinking the wire lowers Hermes' own pressure → its native compaction
  rarely fires. Preserve prefix stability for KV/prompt caching.

**Surface B — MCP server (cooperative, precise)**
- Tools: `store.save`, `store.search`, `state.snapshot`, `state.load`,
  `context.checkpoint`, `context.rehydrate`. Driven by a system-prompt / skill
  protocol so the agent deliberately externalizes (e.g. game-state JSON each turn).

**Store — the Wiki Layer**
- `state.json` (authoritative world/project state, rewritten in place).
- `wiki/*.md` atomic notes + frontmatter + `[[wikilinks]]` (human-auditable, also the
  sterilization surface before any GitHub push).
- Vector index for semantic page-in. Obsidian = optional human lens over the same
  files (graph view doubles as a debugging/sterilization aid). Not a runtime dep.

---

## Milestones

- [x] Phase 0 — Research Hermes compaction; init offline git; draft this plan.
- [x] Phase 1 — Core engine: tokenizer client (`/tokenize`,`/props`,`/apply-template`,
      `/v1/chat/completions/input_tokens`), budget math, hysteresis compactor, sealing,
      mechanical fallback. 53 unit tests proving the no-re-fire invariant (incl.
      hypothesis property test + floor-under-stress + water-collapse + giant-tail
      page-out). Contract: `tasks/phase1-spec.md`. Language locked: **Python**.
- [x] Phase 2 — Store + retriever: `state.json` (atomic rewrite), markdown notes
      (frontmatter + newline-stable, byte-exact bodies), lexical retriever (pure-Python
      BM25 over stdlib sqlite), `DurableStore` facade behind the engine's `Store` Protocol,
      page-out/page-in (demand paging with exact token budgeting). `Embedder` Protocol +
      `LlamaServerEmbedder` ready for a future vector backend. 57 tests (110 total).
      Contract: `tasks/phase2-spec.md`. **Decision:** lexical retriever ships now (zero heavy
      deps, offline); vector index (sqlite-vec/faiss) deferred behind the `Retriever` interface.
- [x] Phase 3 — Surface A proxy: transparent OpenAI-compatible reverse proxy
      (`src/contextmanager/proxy/`) applying the engine on the wire. `PromptRewriter`
      handle-izes bulky messages (>= `handle_threshold_tokens`) into parseable
      `[[cm:stored …]]` stubs, stores full content in the Phase 2 `DurableStore`, and
      auto-rehydrates explicit `[[cm:stored handle=…]]` references under a token budget.
      Idempotent / prefix-stable by construction (per-message `stable_id`), so KV-cache
      and Hermes' own pressure stay low. FastAPI app: `POST /v1/chat/completions`
      (stream + non-stream SSE passthrough), `/v1/models`, `/props`, `/healthz`;
      `UpstreamError` → 502 on BOTH branches (streaming primes the first chunk).
      Round-2 review fixed a CRITICAL idempotency break (rehydrated messages were
      re-expanded every turn = the livelock reborn) — now `is_rehydrated` passthrough +
      Pass-2 dedup guarantee no per-turn growth. **39 proxy tests (149 total)** incl.
      idempotency-with-explicit-reference, multiturn-no-growth, stream-error-502, and
      real-lifespan wiring. Contract: `tasks/phase3-spec.md` (§9 = Round-2 corrections).
      Run: `python -m contextmanager.proxy` (config via `CM_*` env vars).
- [x] Phase 4 — Surface B MCP server (`src/contextmanager/mcp/`). Cooperative surface:
      six MCP tools — `store_save`/`store_search`, `state_snapshot`/`state_load`,
      `context_checkpoint`/`context_rehydrate` — over a pure `GovernorService` wrapping the
      SAME `DurableStore` (never forked). stdio `FastMCP` server (official `mcp` SDK);
      offline `HeuristicTokenCounter` (~1 tok/4 chars, measurable) means the budgeted
      page-in works with NO llama-server, or wire `CM_UPSTREAM_BASE_URL` for exact counts.
      All logic in the pure service (no MCP types) so it's tested without an MCP runtime;
      `server.py` is a thin adapter. **65 proxy/mcp tests added (214 total)**: service ops,
      counter measurability invariant, config validation, and live FastMCP `list_tools` +
      `call_tool` round-trips. Contract: `tasks/phase4-spec.md`. Run:
      `python -m contextmanager.mcp` (stdio; `CM_*` env vars). NOTE: the system-prompt/skill
      protocol that drives an agent to call these tools each turn is folded into Phase 5
      (it only matters once wired to a real agent on the proving ground).
- [~] Phase 5 — Hermes integration + proving ground. **Offline scaffolding DONE:** proxy
      `/metrics` observability (`src/contextmanager/proxy/metrics.py` — zero-tokenizer-cost
      counter: requests, handle-izations, rehydrations, `chars_in/out/saved`; +4 tests, 218
      total), and the `integration/` artifacts — `hermes-config.tier0.yaml` (copy-paste
      Tier-0 patch), `README.md` (wiring topology + run order), `surface-b-protocol.md`
      (system-prompt block driving the MCP tools), `measurement.md` (A/B/C before-after
      runbook). Contract: `tasks/phase5-spec.md`.
      **First live run DONE (2026-06-23, Hermes + Qwen3.6, 138-msg / 73-tool-output
      session):** proxy validated — `/metrics` 72 requests, 405 messages handle-ized,
      **chars 7.03M → 2.82M = 4.21M saved (~60% wire reduction)**, all 66 chat completions
      200, session completed, no livelock. `messages_rehydrated: 0` (agent didn't recall →
      motivated wiring Surface B). Fixed `/v1/props` 404; registered the MCP server in both
      OpenCode + Hermes (shared store via `--store-root`); added
      `integration/surface-b-systemprompt.md`. Raw: `tests/live_tests/`. **Remaining:** the
      formal A/B/C measurement table (baseline vs tier-0 vs proxy) in `measurement.md`.
- [x] Phase 6 — Sterilize for public release (MIT). Secret/PII/personal-path scan came back
      clean; runtime store, session transcripts, and live captures are gitignored; machine
      paths genericized; personal working notes removed; added `LICENSE`, `.gitattributes`,
      public `README`, and pyproject metadata + console scripts. Published from a fresh,
      single clean commit (no PII in history) under a GitHub no-reply author.

## Tech stack (recommended)

- **Python** (FastAPI + uvicorn proxy; official `mcp` SDK; `httpx`).
  Rationale: exact tokenization via `llama-server` HTTP endpoints, Windows-friendly,
  strongest local-LLM ecosystem. Alternative: TypeScript (smoother `npx` MCP distro).
  Decide at Phase 1 start.

## Open decisions

- [x] Language: **Python** (locked at Phase 1).
- [x] Primary surface to ship first: **proxy** (shipped Phase 3) — universal, fixes dumb
      CLIs too, needs no agent cooperation. MCP (cooperative) follows in Phase 4.
- [x] Vector index: **Decided (Phase 2)** — ship pure-Python lexical (BM25/sqlite) now;
      defer a vector backend behind the `Retriever` interface until a local embedding
      model is confirmed.
- [ ] Phase 5 measurement: how to quantify "compaction frequency before/after" on the
      Spaceshooter build (instrument Hermes logs vs proxy-side counter). Decide at Phase 5.

---

## Phase 7 — Store & Retrieval Overhaul ("the tax-free hot store")

> Goal: make the `DurableStore` self-maintaining and fast as the corpus grows —
> sublinear search, bounded disk **and** working set, hotness-aware recall — without
> adding any cost on the `/v1/chat/completions` hot path, any token cost, or any new
> runtime process. Committed scope: **full overhaul** (Stages 1–3 below).

### Grounding facts (2026-06-24)
- The retrieval path is **unproven live**: the first run logged `messages_rehydrated: 0`;
  the ~60% wire reduction was all handle-ization, not recall. So Stage 1 instrumentation
  precedes/justifies the heavier stages even under the full-overhaul commit.
- Engine fork resolved **in our env**: SQLite 3.45.1 → FTS5 **and** `contentless_delete=1`
  both available. But the **published** package cannot assume FTS5 is compiled in, so FTS5
  ships **behind the existing `Retriever` Protocol with the pure-Python `LexicalRetriever`
  as the always-present fallback** — never a hard swap.

### Invariants to keep green (cross-cutting)
- **Cross-surface global resolution:** a handle minted by either surface resolves in both;
  `page_in(handle)`/GET stays global. Scope/hotness affect **ranking only**, never resolution.
- **Idempotency / prefix-stability** of the proxy rewriter is untouched (this is store-layer,
  below the rewriter — its tests stay green).
- **Tax-free:** zero work on the request hot path; maintenance is amortized / off-path /
  at-shutdown; no tokenizer cost; no new process.
- **Portability:** FTS5 optional; the pure-Python path always works.
- All disk writes atomic (temp + `os.replace`); byte-exact note bodies preserved; full type
  hints; `from __future__ import annotations`.

### Stage 1 — Learn + free wins (low risk, partly owed)  ✅ code done 2026-06-24
- [ ] Close the Phase 5 **A/B/C measurement table** (baseline vs Tier-0 vs proxy) in
      `integration/measurement.md`. **BLOCKED on live infra** (needs a llama-server + Hermes
      run) — cannot be produced offline; do it on the next proving-ground session.
- [x] **Retrieval-path metrics** — added to `DurableStore.stats()` (the shared layer, reused by
      both surfaces — better than `proxy/metrics.py`, which the proxy never feeds since it does
      not search): `corpus_size`, `search_calls`, `search_hits`/`search_empty`, `recall_hit_rate`,
      `results_returned`, `avg_search_ms`, `page_in_calls`/`page_in_slices`. Surfaced under a
      `"retrieval"` block in the proxy `/metrics`. Zero tokenizer cost.
- [x] **Incremental auto-vacuum** in `LexicalRetriever`: `PRAGMA auto_vacuum=INCREMENTAL` set
      BEFORE any write (incl. `journal_mode=WAL`, which else bakes in mode 0); one-time `VACUUM`
      converts a pre-existing non-auto-vacuum `index.db`; bounded reclaim after `remove()` via
      `executescript("PRAGMA incremental_vacuum(64);")` (a plain `execute` only steps it once).
- [x] Tests: auto_vacuum enabled/converted, per-remove reclaim drains the freelist, `:memory:`
      left untouched; stats counters (hits/misses/page_in) + `/metrics` retrieval block. +10 tests.

### Stage 2 — FTS5 engine + tax-free GC  ✅ done 2026-06-24
- [x] `Fts5Retriever` (`retriever.py`) implementing the `Retriever` Protocol: **contentless**
      FTS5 (`content=''`, `contentless_delete=1`) + a `handle_map(rid, handle)` bridge,
      `bm25()` ranking returned positive/descending (handle-asc tie-break, matching the
      fallback). Bodies stay canonical in `notes/*.md` — **no second text copy** (verified: the
      indexed column reads back NULL). Same WAL + incremental-auto-vacuum maintenance.
- [x] **Capability detection + graceful fallback**: `fts5_available()` + `make_retriever()`
      pick FTS5 when compiled in (it is here), else the portable `LexicalRetriever`.
      `DurableStore` now defaults to `make_retriever` — all 250 prior tests pass unchanged
      through FTS5 (real parity).
- [x] **Index-rooted mark-and-sweep GC** (`DurableStore.gc()`): Sweep 1 drops index entries
      whose note is gone; Sweep 2 removes note files NOT in the index (evicted from search),
      honoring `protect`, a conservative state-reference check (false positives only KEEP —
      never delete), and `min_age_seconds`. A note still in the index is never touched, so
      anything search-reachable / rehydratable stays safe. Idempotent + re-entrant.
- [~] Config knobs: `min_age_seconds` + `protect` are `gc()` params; vacuum page budget is a
      fixed 64. A per-call sweep budget `K` and ProxyConfig wiring are deferred to **Stage 3**,
      where the eviction *decision* (what to drop from the index) and `gc()` *scheduling*
      (shutdown / amortized) actually become meaningful — today nothing un-indexes a note, so
      auto-running `gc()` would be a no-op.
- [x] Tests: FTS5 ranking/remove/upsert-no-dup/persistence/handles/contentless/close/CM,
      `make_retriever` selection, **parity** (FTS5 vs pure-Python first-hit agreement), and 8
      GC cases (reconcile, evict-then-sweep, protect, state-ref, min-age, indexed-note-safe,
      idempotent). **+23 tests → 273 passing.** Ad-hoc benchmark: **~185× faster search**
      (0.24 ms vs 44.5 ms/query at 4k docs; gap widens with corpus size).

### Stage 3 — Hot-scope intelligence + cold tier  ✅ core done 2026-06-24
- [x] **Decay-on-read hotness** (`hotness.py` `HotnessTracker`): per-handle `score` +
      `last_seen` in its own WAL db; on access `score = score*0.5**((now-last)/half_life) + 1`
      (lazy decay, O(1), persisted, no sweep; injectable clock). Bumped on GET, page_in_by_id,
      and the returned search set. Denning working-set made concrete.
- [x] **Hotness re-rank** (`DurableStore._rerank`): the index returns a candidate POOL
      (`rerank_pool`, default 50); blend normalized `relevance` × `hotness` over just that pool
      (O(pool), never O(corpus)); `rerank_weight=0` ⇒ pure relevance. Returned `score` is the
      blend, so results stay sorted-desc.
- [x] **Generational eviction + ghost feedback** (`DurableStore.evict_cold` + ghost ring):
      while corpus > `target_size`, drop the COLDEST (never-accessed score 0 first) from index
      AND notes, never touching `protect`/state-referenced; record each as a ghost. A later
      `get()` of a ghost increments `ghost_hits` (the cascade-miss signal) and re-raises
      (callers already degrade gracefully).
- [x] **Cold-tier body compression** (`NoteStore.compress`/`decompress` + `DurableStore.
      compress_cold`): gzip the coldest notes at rest (`<h>.md.gz`); GET/read_meta decompress
      transparently and BYTE-EXACT; the index is untouched so a cold note is still findable.
      LOSSLESS (vs `evict_cold`'s lossy delete). Atomic temp+replace.
- [~] **Scope-column partition + cascade predicate-widening (local→group→global)** and the
      optional **co-access Markov** prefetch are DEFERRED: they need a scope key threaded from
      the surfaces (the proxy/MCP don't supply one today) and are best tuned against Stage-1
      live metrics. The ghost ring already captures the cascade-miss *signal* the widening would
      consume. Tracked as the remaining Phase 7 work.
- [~] **Maintenance scheduling** (run `gc`/`evict_cold`/`compress_cold` at shutdown / amortized
      per N page-outs, + ProxyConfig knobs) deferred with it — destructive policy should be
      tuned on the proving ground, not auto-enabled blind. The mechanisms are ready to wire.
- [x] Tests: decay half-life math + persistence; re-rank promotes a warm doc / weight-0 stays
      pure relevance; eviction removes coldest, honors protect+state, no-op under target;
      ghost-hit increments `ghost_hits`; compressed body round-trips byte-exact and stays
      searchable; compress/decompress/remove on gzipped notes. **+21 tests → 294 passing.**

### Phase 7 — Results (2026-06-24)

Built Stages 1–3 of the full overhaul; **224 → 294 tests, 0 failures, 0 skips** (FTS5 present).
Commits: `9c97b5c` (S1), `65913cb` (S2), `6eb0f6f` (S3). Highlights:
- **Search is now sublinear and duplication-free.** Ad-hoc benchmark at 4k docs:
  **0.24 ms/query (FTS5) vs 44.5 ms (pure-Python) — ~185×**, gap widening with corpus size;
  the text is no longer stored twice (contentless index + canonical `notes/*.md`).
- **The store self-maintains, tax-free:** per-delete incremental auto-vacuum; index-rooted GC;
  decay-on-read hotness driving a re-rank toward the working set and generational `evict_cold`;
  gzip cold-tier (lossless, still searchable). Nothing runs on the `/v1/chat/completions` hot
  path; no token cost; portable fallback keeps FTS5 optional.
- **Backward compatible (added after review):** the FTS5 backend uses its own table
  (`fts_idx`, not `docs`) so it never collides with a legacy pure-Python `index.db`; and
  `DurableStore._reconcile_index()` reindexes any note missing from the index on open — so an
  existing `contextstore` (verified: 70 notes + a legacy `docs` table) migrates to FTS5 on the
  next run with zero user action. `stats()`/`/metrics` now report the live `backend` so the
  active retriever (FTS5 vs the pure-Python fallback) is observable. **297 tests.**
- **Invariants held:** all 250 pre-existing tests pass unchanged through the new FTS5 default
  (real parity); rewriter idempotency untouched (store-layer only); GC/eviction never delete an
  indexed/protected/state-referenced note.
- **Honest deltas / remaining:** (a) the A/B/C measurement table needs a live llama-server +
  Hermes run; (b) scope-column partition + cascade predicate-widening + co-access Markov are
  deferred (need a surface-supplied scope key + live tuning) — the ghost ring already records
  the cascade-miss signal they'd consume; (c) auto-scheduling of gc/evict/compress is wired-ready
  but intentionally left off until thresholds are tuned on the proving ground (destructive ops).
- **Reusable gotchas found:** `PRAGMA auto_vacuum` must be set before the first write (WAL init
  bakes mode 0); `PRAGMA incremental_vacuum(N)` via `execute()` steps only ONCE — use
  `executescript`. Contentless FTS5 columns read back `NULL`; `bm25()` is negative (negate).

---

## Phase 8 — Central launcher + multi-provider front-end + token-aware metrics

> Goal: ONE un-scattered entrypoint — `run-governor` with llama-server-style flags — that
> presents the right wire protocol to a CLI, forwards to any provider, and auto-wires the CLI
> in/out safely. Motivation: not cheaper $/token — **slow the rate a capped context/usage budget
> drains** (e.g. Claude's 5-hour / weekly / 1M-window limits). "Anything that fills the context
> is the enemy." Just-discussed; gated items below are NOT green-lit to build yet.
>
> **Active scope (decided 2026-06-25):** OpenAI (API-key) + Hermes + local llama/ollama + OpenCode.
> **Claude/Anthropic DEFERRED** to a later round — its subscription/OAuth proxying feasibility + ToS
> are unresolved, and dropping it removes the only uncertain piece. Revisit Claude separately.

### Design model (two independent axes — keep them named separately)
- **`--cli claude|opencode|hermes`** = the *wiring profile*: which inbound protocol to PRESENT,
  where the CLI's config lives, and how to insert/revert. (Claude Code speaks Anthropic Messages;
  OpenCode/Hermes speak OpenAI.) The CLI dictates the inbound protocol.
- **`--provider openai|anthropic|llama|ollama`** = the *outbound adapter*: where to FORWARD.
- Usually equal (transparent same-protocol proxy = present what the CLI expects, forward to the
  same place, shrink the middle). That transparent case is the safe sweet spot.

### Stage 8.0 — Token-aware `/metrics`  ✅ done 2026-06-25
- [x] Token-estimate fields on the proxy stats (≈ chars/4 → ZERO tokenizer cost): `tokens_in_est`,
      `tokens_out_est`, `tokens_saved_est`, `pct_saved`, plus a human `summary` ("saved ~9.7M
      tokens (~86%) over N requests; peak prompt ~X tokens").
- [x] `peak_prompt_tokens_est` gauge — the single-turn high-water the cumulative counters can't
      show. (`proxy/metrics.py`; +2 tests.)

### Stage 8.1 — Central config + unified launcher  ✅ done 2026-06-25
- [x] `run-governor` console script (`launcher.py`): llama-server-style flags; loads a CENTRAL
      **TOML** config via stdlib `tomllib` (no new dep) — `[proxy]` table + `[providers.<name>]`.
      Precedence: defaults < config file < `--provider` profile < flags. Consolidates the `CM_*`
      env sprawl behind one door. Console entry added to pyproject; `integration/governor.example.toml`.
- [x] Flags: `--config`, `--provider {llama,ollama,openai}` (anthropic/claude deferred → clean
      error), `--cli {opencode,hermes}` (prints wiring hint; claude deferred), all ProxyConfig
      knobs, `--dry-run` (prints resolved config, **API key redacted**), `--revert` (stub → 8.3).
      **API keys come ONLY from the provider's `api_key_env` — never flags or the file body.**
      No engine behavior change. +15 tests → 320 passing.

### Stage 8.2 — Provider adapters (outbound) behind one interface  ✅ active set done 2026-06-25
- [x] **Active:** `llama` · `ollama` · `openai` — all OpenAI-compatible, and the existing
      `UpstreamClient` already forwards `Authorization: Bearer <key>` (the launcher pulls the key
      from the provider's `api_key_env`). No new adapter code needed; real end-to-end is the
      "test at the end". (OpenAI has no `/props`, so it uses the fixed `handle_threshold_tokens`
      rather than n_ctx anchoring — expected.)
- [ ] **DEFERRED — `anthropic`** (revisit later): Messages API ≠ OpenAI shape, and the
      subscription/OAuth proxying question is unresolved. When un-deferred, prefer a **LiteLLM
      passthrough** (governor stays OpenAI-shaped; LiteLLM does Anthropic translation + caching)
      over a hand-rolled adapter, plus the **Anthropic inbound** protocol (Claude Code speaks
      Messages). Compose, don't rebuild LiteLLM/OpenRouter.

### Stage 8.3 — Safe CLI auto-wiring (backup / insert / revert)  ✅ done 2026-06-25
- [x] `wiring.py`: per-CLI insert (OpenCode JSON → a `context-governor` provider's `options.
      baseURL`; Hermes YAML → repoint `model.base_url` via **lazy** PyYAML, optional `[wiring]`
      extra). **Timestamped backup** + **atomic** temp-replace + **idempotent** (already-wired is a
      no-op; already-wired-without-state refuses rather than back up the wired copy) + a JSON
      **wiring-state** file (under `CM_WIRING_DIR` or `~/.context-governor/wiring`).
- [x] Graceful revert: launcher's `finally` reverts on exit/CTRL+C; `run-governor --revert --cli X`
      restores the backup **verbatim** after a crash/BSOD. `--dry-run --cli X` previews the plan
      (writes nothing); `--cli-config PATH` overrides the location.
- [x] CLEAN targets done: `opencode`, `hermes`. Tested against SYNTHETIC configs only (never a real
      one): insert/backup, idempotency, verbatim revert, missing-file + already-wired guards,
      dry-run-no-write, and the launcher `--revert`/`--dry-run` paths. **+13 tests → 333 passing.**
      NOTE: OpenCode's full provider schema (models/default) may need a manual touch — the insert is
      best-effort + always reverted from backup; preview with `--dry-run --cli opencode`.
- [x] **Enhanced 2026-06-25 (portability):** (a) **discovers** the config path — OpenCode via
      `opencode debug paths`, Hermes via platform default (its config is edited directly, so the
      broken hermes binary doesn't matter); (b) wires the **MCP server too** (Surface B), pointing
      at `sys.executable -m contextmanager.mcp --store-root <abs>` so it works after a fresh
      `git pull` on any OS/layout; (c) wiring is now **PERSISTENT** (no revert-on-exit) — one-time
      setup, idempotent re-run, undo with `--revert --cli X`. Validated against the real OpenCode
      config (provider `contextgovernor` w/ full model + `mcp.context-governor`) and Hermes shapes
      (`model.base_url` + `mcp_servers.context-governor`). **334 tests.**
- [ ] **DEFERRED — Claude/Anthropic wiring** (revisit later): blocked on the open question — does
      Claude Code honor a custom endpoint under **subscription/OAuth** (the 5h/weekly case), and is
      proxying that session through third-party middleware supported AND within ToS? Not in active
      scope; the clean targets above are.

### Stage 8.4 — Cache-aware rewriting (prereq for the OpenAI remote path)
- [ ] Only stub/rewrite BELOW the provider's prompt-cache breakpoint; never touch the cached prefix
      — else cache miss = full re-process = drains the budget FASTER (opposite of the goal). Needed
      before pointing at OpenAI for real; the same rule applies to Anthropic when it's un-deferred.

### Carry-through risks
- External-config edits: backup-first, atomic, idempotent, revert-after-crash.
- Subscription/OAuth proxying is the uncertain + ToS-sensitive path → gated, not assumed.
- Don't reinvent LiteLLM routing/translation/caching — compose with it.

---

## Phase 9 — Lossless Saving: the archive tier ("no byte ever dies")

> Goal: close the LAST two lossy paths in the store, so externalized content is
> **never destroyed** — the working-set bound stays exactly as tight (index, hotness,
> `notes/`, `corpus_size` all unchanged), but eviction becomes a *demotion* to a
> compressed archive tier instead of a delete, and a later request for an evicted
> handle **resurrects** it (restore + re-index + hotness bump) instead of raising.
> Disk is cheap; context — and content — is precious.

### The two lossy paths being closed (grounded 2026-07-01)
- `DurableStore.evict_cold` — self-documented as "the deliberate, LOSSY working-set
  bound": unlinks note bodies once cold.
- `DurableStore.gc` Sweep 2 — unlinks note files that fell out of the index.
- (Everything else already round-trips byte-exact: page-out persists before the window
  drops a message; `compress_cold` is lossless-at-rest; stubs keep full bodies stored.)

### Design
- [x] `NoteStore` archive tier under `<root>/archive/<handle>.md.gz` (sibling of
      `notes/` — invisible to `list_handles()`/`corpus_size` by construction):
      `archive(handle)` (plain → gzip atomically then unlink; already-gz → pure
      `os.replace` move), `restore(handle)` (`os.replace` back into `notes/` as
      `.md.gz` — reads already decompress transparently), `is_archived()`,
      `list_archived()`. Byte-exact bodies through the round-trip (same
      `newline=""` discipline as `compress`).
- [x] `DurableStore.evict_cold(..., archive=True)` — default LOSSLESS: index +
      hotness dropped, ghost recorded (the cascade-miss signal stays), body archived.
      `archive=False` keeps the old hard-delete for when disk itself must be reclaimed.
- [x] `DurableStore.gc(..., archive=True)` — Sweep 2 archives instead of unlinks.
      Return shape unchanged (`removed_notes` = notes taken out of the live tier).
- [x] **Resurrection**: `DurableStore.get()` on a missing note tries the archive —
      on hit: restore, re-index (meta from frontmatter), hotness bump, count
      `resurrections` (+ `ghost_hits` when in the ghost ring), return the body.
      `page_in_by_id` delegates to `get()` so the explicit rehydration path
      resurrects too. Only a note that never existed (or was hard-deleted) raises.
- [x] `stats()` gains `archived_count` + `resurrections` (flows to `/metrics` free).

### Invariants
- Working set stays bounded: archived notes are OUT of the index, hotness, and
  `corpus_size`; `_reconcile_index` never resurrects (it only walks `notes/`), so an
  archived note stays evicted across restarts until actually requested.
- Cross-surface: the archive lives in the shared store root → a handle archived by
  either surface resurrects in both.
- Tax-free: archive/restore only run inside evict/gc/get-miss — nothing new on the
  `/v1/chat/completions` hot path; no tokenizer cost; no new process.
- Atomic writes (temp + `os.replace`); byte-exact bodies; restore is a pure rename.
- Existing 334 tests stay green (one repointed: the ghost-hit test now exercises the
  explicit `archive=False` lossy path; the default path gets new resurrection tests).

### Phase 9 — Results (2026-07-01)

Shipped in one pass; **334 → 346 tests, 0 failures, 0 skips.** The store is now lossless
END TO END by default: page-out persists before the window drops a message (Phase 3),
`compress_cold` gzips at rest (Phase 7), and now eviction/GC *demote to the archive tier*
instead of deleting — `get()`/`page_in_by_id` on an evicted handle transparently
resurrects (restore + re-index + hotness bump), byte-exact (CRLF/CR/LF all covered),
across restarts, on both surfaces. Hard delete still exists but only as the explicit
opt-in `archive=False`. New tests: 8 durable (evict-archives-by-default, byte-exact
resurrection incl. re-searchability + counters, hard-delete path, reopen-resurrection
proving persistence beyond the in-memory ghost ring, page_in_by_id resurrect, gc
archive/hard-delete, compressed-note archive = pure rename) + 4 note_store primitives.
Docs: README lossless story, wiki/durable-store.md refreshed (was stale at Phase 2) with
the 4-tier lifecycle (live → compressed → archived → deleted-only-by-opt-in).
Deliberate non-goals: no auto-scheduling (still proving-ground work), no archive size
bound (unbounded by design — prune manually or use `archive=False`).

---

## Phase 10 — Auto-recall: the read path becomes intelligent

> Driver fact: the live run logged `messages_rehydrated: 0` — the governor is a perfect
> writer and a mute reader. A local model never learns to ask for `[[cm:stored …]]` by
> itself, and it cannot page-fault on memory it cannot see. Phase 7 built the working-set
> half of a virtual-memory system (eviction); this builds the other half: **anticipatory
> demand paging** — locality-driven recall, zero cooperation needed (the proxy ethos).

### Design (grounded in rewriter.py, 2026-07-01)
- [x] `proxy/recall.py` (pure, deterministic, stdlib-only):
      `extract_query(messages, tail_messages, max_terms)` — derive the implicit query
      from the live tail: last N messages' text, stub/marker lines stripped, `\w{3,}`
      terms lowercased, stopwords dropped, term frequency weighted by message recency,
      top-K terms joined. `\w`-only terms are inherently FTS5-safe.
      `select_diverse(slices, max_similarity, cap_chars)` — near-duplicate suppression
      over the candidate slices (agent sessions re-read the same file; don't spend the
      recall budget on five copies). O(k²) difflib over char-capped prefixes.
- [x] Rewriter **Pass 4 — auto-recall** (after Pass 3 windowing), marker `[[cm:recall]]`:
      - **Strip-on-entry**: drop any incoming `[[cm:recall]]` message BEFORE Pass 1 —
        at most ONE recall block ever exists on the wire, so `rewrite(rewrite(x))`
        cannot grow (the §9.1 no-growth invariant extends by construction).
      - Query from the tail; `store.search(query, pool)`; keep only **off-wire** slices
        (skip any handle already present as a stored-/diff-stub — incl. diff `base=` —
        or rehydrated marker); `select_diverse`; assemble under `recall_budget_tokens`
        with the same truncate-marker-never discipline as Pass 2.
      - Inject ONE `{"role":"system"}` message `[[cm:recall]]\n[[cm:recalled handle=H]]\n
        <slice>…\n[[/cm:recall]]` immediately BEFORE the final message (KV impact
        bounded to the tail; templates keep user/tool last). Empty store, trivial query
        (<2 terms), no survivors, or `auto_recall_k=0` ⇒ no injection at all.
      - `[[cm:recalled` does NOT match `_HANDLE_RE` (`[[cm:stored`) ⇒ Pass 2 never
        re-expands recalled slices; `is_recall` guards Pass 1/3 defensively.
- [x] Config: `auto_recall_k` (default 3; 0 = off) + `recall_budget_tokens` (default
      1500) — ProxyConfig, `CM_AUTO_RECALL_K`/`CM_RECALL_BUDGET_TOKENS`, launcher flags,
      `governor.example.toml`, wiki/configuration.md. Wire bound becomes
      `ratio*n_ctx + recall_budget` (documented; ~2% of a 75K window at defaults).
- [x] Observability: `RewriteResult.recalled_handles`; `messages_recalled` in
      ProxyStats/`/metrics`. Recall flows through `store.search()` so the Stage-1
      retrieval counters AND hotness warming come free — recall feeds the working-set
      signal, which feeds eviction, which feeds the archive, which resurrection feeds
      back: the organs interlock.
- [x] Tests: recall.py units (salience, recency weighting, stopwords, marker stripping,
      FTS5-safe output, diversity suppression); rewriter Pass-4 (injects relevant
      off-wire content; skips on-wire handles; budget respected; double-rewrite does not
      grow; k=0 ⇒ byte-identical to Phase-9 behavior; empty store no-op; deterministic).

### Non-goals (this phase)
- Structure-aware stub previews (would touch the normative §3.1 stub format — separate
  decision), co-access prefetch (needs live recall traffic first — this phase GENERATES
  that traffic), embeddings retriever (still gated on a confirmed local embedding model).

---

## Phase 11 — Hysteresis windowing: optimize COMPUTE, not just tokens

> Driver: Pass 3 pages down to a single budget line EVERY turn, so at steady state the
> stub frontier advances every turn — invalidating the llama-server KV prefix from that
> point and re-processing ~10s-of-K tokens per turn. The engine's hysteresis math
> (Phase 1, proven, never wired to a shipped surface) finally lands on the wire: trigger
> at HIGH water, page down to LOW water in one big bite, then hold the prefix
> byte-stable for many turns. With TurboQuant KV compression the win is multiplicative:
> compressed cache × actually-reused cache.

### The trap the naive design falls into (grounded 2026-07-02)
The CLI resends the ORIGINAL transcript every turn, so a stateless two-water check would
re-trigger every turn (input total never shrinks) and re-advance the frontier anyway.
The rewriter therefore needs a **sticky frontier**: remember what it already windowed
out and re-apply it verbatim each turn.

### Design
- [x] `context_target_ratio` (default **0.35**; 0 = legacy nibble-to-ceiling): LOW water.
      `context_budget_ratio` (0.50) becomes the HIGH water / trigger and remains the wire
      ceiling. Validation: `0 <= target < 1` and `target < context_budget_ratio` when > 0;
      runtime integer-collapse guard (`low >= high` ⇒ legacy).
- [x] **Sticky frontier** on the rewriter: `_windowed: dict[stable_id -> token_count]`.
      Pass 3a re-applies it to non-pinned middles every turn using CACHED counts —
      byte-identical minimal stubs, zero tokenizer cost (a `notes.has` stat guards the
      rare re-page_out). Pass 3b: cheap char-estimate gate vs high water → precise counts
      → if `total > high_water`, advance oldest-first until `total <= low_water`,
      recording each id+count into the frontier. Frontier is in-memory (proxy lifetime —
      matches the KV cache's own lifetime; restart = one-time re-derivation).
- [x] Observability: `RewriteResult.windowing_triggered` → `windowing_triggers` counter in
      `/metrics` (the live proof of "rare big bites": triggers ≪ requests).
- [x] Config plumbing: env `CM_CONTEXT_TARGET_RATIO`, flag `--context-target-ratio`,
      example TOML, wiki §3 matrix + §4c, README.
- [x] Tests: between-waters no-op (over low, under high ⇒ ZERO paging); trigger pages to
      <= low water; **prefix stability across turns** (turn t+1 with modest growth reuses
      turn t's frontier — shared wire prefix byte-identical, no new stubs); re-trigger
      advances when high is crossed again; `target=0` restores legacy per-turn nibble;
      idempotency (double-rewrite no-op); pinned head/tail untouched. Existing
      TestBudgetWindowing behavior tests must pass unchanged.

### Invariants
- Ceiling unchanged: wire never exceeds `context_budget_ratio × n_ctx` (trigger cuts it).
- Losslessness untouched (same page_out-before-stub path); idempotency preserved (post-
  rewrite total <= low < high ⇒ second pass never triggers).
- Between triggers the wire prefix up to the tail is BYTE-IDENTICAL across turns ⇒
  llama-server prefix cache reuses it; only tail + recall block re-process.

### Phase 11 — Results (2026-07-02)

Shipped; **368 → 375 tests, 0 failures, 0 skips.** `context_target_ratio` (default 0.35,
0 = legacy) + sticky in-memory frontier (`_windowed: id → original count`) on the
rewriter: Pass 3a re-applies the frontier byte-identically each turn with ZERO tokenizer
cost (cached counts; `notes.has` stat guards the rare re-page); Pass 3b triggers only
above the high water and cuts to the low water in one bite. `windowing_triggers` counter
on `/metrics` — the live proof is triggers ≪ requests. Seven new tests incl. the crown:
turn t+1 with modest growth reuses turn t's frontier and the ENTIRE turn-t wire is a
byte-exact prefix of turn t+1's (KV-reuse claim, pinned). All pre-existing windowing
behavior tests passed unchanged. The engine's Phase-1 hysteresis theorem finally serves
a shipped surface — the last built-but-unwired organ is wired.

## Phase 12 — Sticky auto-recall: the last per-turn prefix breaker (2026-07-17)

### Diagnosis (llama-server logs, D:\CLAUDE_GLP\Project_9_test; hybrid-SSM Qwen3.6-35B-A3B)
- Hybrid-SSM prompt cache needs a BYTE-EXACT pure prefix extension (`f_keep=1.000`);
  any divergence forces a full re-prefill (12-35k tokens ≈ 12-35 s **every turn**).
- Divergence sat at a near-constant **~1502 tokens before the previous prompt end**
  (11500→9998, 11819→10317, 12156→10654). Suspect list:
  - Pass 3 windowing? NO — Phase 11 made it sticky; live `/metrics`:
    `windowing_triggers: 0` over 26 requests. History already byte-stable.
  - **Pass 4 auto-recall? YES** — `recall_budget_tokens=1500` ≈ the 1502 offset, and
    the block is recomputed FRESH and re-inserted before the FINAL message on
    **every request** (live `/metrics`: `search_calls: 26` == `requests: 26`).
    A fresh block at a tail-tracking insertion point = guaranteed per-turn miss.

### Design — give recall the same hysteresis windowing got in Phase 11
- Freeze `(anchor_mid, block_text)` after building a block: anchor = stable_id of the
  message the block was inserted BEFORE (the then-final message).
- Every later turn, re-inject the SAME block bytes at the SAME anchor (first message
  matching anchor_mid — verbatim stable_id or its Pass-1/3 stub handle). The wire
  becomes a pure extension: `[history, block, anchor]` + new tail.
- Refresh (recompute + re-anchor at the new tail, one jump) ONLY when:
  (a) est. growth after the anchor ≥ `recall_refresh_tokens` (chars/4, no tokenizer);
  (b) the anchor vanished (host CLI compacted/edited); or
  (c) Pass 3 windowing triggered this turn (prefix already broken — refresh is free).
- `recall_refresh_tokens = 4000` default; **0 = legacy per-turn recompute** (the
  current behavior, kept behind the setting).

### Checklist
- [x] `ProxyConfig.recall_refresh_tokens` (+ validation, docstring)
- [x] Env `CM_RECALL_REFRESH_TOKENS`, flag `--recall-refresh-tokens` (TOML is automatic)
- [x] Rewriter: `_recall_frozen` state + sticky/refresh logic in Pass 4
- [x] Tests: sticky byte-identical re-injection + pure-prefix assertion; refresh on
      growth / anchor loss / windowing trigger; legacy mode unchanged; double-rewrite
      idempotency updated (2nd pass reuses the frozen block)
- [x] Docs: wiki/configuration.md, governor.example.toml

### Invariants
- Between refreshes the ENTIRE previous wire (incl. recall block) is a byte-exact
  prefix of the next wire ⇒ hybrid-SSM cache can pure-extend on non-refresh turns.
- Refresh turns pay ONE full re-prefill — rare and in one jump, like Phase 11.
- Recall staleness bounded by `recall_refresh_tokens` of conversation growth.

### Phase 12 — Results (2026-07-17)

Shipped; **375 → 380 tests, 0 failures.** `recall_refresh_tokens` (default 4000, 0 = legacy)
+ frozen `(anchor_mid, block)` on the rewriter. Five new tests incl. the crown (mirroring
Phase 11's): turn t+1 with modest growth re-injects turn t's block byte-identically at the
same anchor and the ENTIRE turn-t wire is a byte-exact prefix of turn t+1's. Refresh proven
for all three triggers (growth, anchor loss, windowing). Verified end-to-end: `--dry-run`
resolves the flag, `CM_RECALL_REFRESH_TOKENS` env read, negative value rejected. NOTE: the
long-running proxy on :8900 must be restarted to pick this up. Live health signature after:
`search_calls ≪ requests` in `/metrics` retrieval stats (was 26/26 — every turn — before).

---

## Phase 13 — Loop guard: a mechanical breaker for degenerate agent cycles

> Driver (real incident, 2026-07-19, logs in `D:\CLAUDE_GLP\Project_9_test`): an
> opencode + Qwen3.6 session livelocked for ~40 turns on an UNVERIFIABLE checklist item
> ("check if the render is visible" — no tool can see pixels). Signature: >= 9
> consecutive turns growing context by exactly +329 tokens (byte-identical tool call +
> result + reply), 13 turns of llama-server draft acceptance = 1.00000 (the whole output
> drafted from existing context = verbatim recycling), 530 reads of the same file. The
> harness auto-continue prompt never stopped it. Anatomy:
> `C:\tools\llama-cpp-turboquant\wiki\hybrid-ssm-prompt-cache-misses.md`.

### Design
- [x] `src/contextmanager/loop_guard.py` — self-contained core module (BudgetConfig
      style): `LoopGuardConfig` (frozen, validated), `turn_fingerprint()` (trailing
      segment from the last assistant message; whitespace-collapsed, volatile substrings
      stripped — timestamps/UUIDs/hex-runs via regex, tool-call `id`/`tool_call_id`
      structurally — then sha1), `LoopGuard` state machine, breaker texts.
- [x] Signal 1 (CONTENT, primary, model-agnostic): `loop_repeat_k` (default 3)
      consecutive equal fingerprints of NEW turns (anchor = (last-assistant idx, len) —
      an unchanged anchor is a client retry, not a repeat).
- [x] Signal 2 (TIMINGS, opportunistic corroboration, never required):
      `timings.draft_n_accepted / draft_n >= loop_accept_threshold` (0.99) with
      `draft_n > loop_draft_n_min` (200) for `loop_timings_m` (3) consecutive responses
      lowers the effective k by ONE (never below 2). Non-stream JSON observed directly;
      SSE chunks scanned best-effort (cheap substring gate, stream forwarded verbatim).
- [x] Action: breaker notice APPENDED AT THE TAIL of the next outbound request as one
      `user`-role message (strict templates reject mid-wire `system`; same lesson as
      Pass 2/4). Append-only by construction — earlier history byte-stable, f_keep = 1.0
      for the hybrid-SSM prompt cache (a mid-history edit = 30–60 s full re-prefill).
- [x] Escalation ladder: fire 1 = notice; cooldown `loop_cooldown_turns` (3); a cycle
      that survives it = ONE escalated FINAL notice; further fires repeat the final
      notice — unless `loop_hard_stop` (default OFF) is set, then the proxy returns a
      synthetic final response (`finish_reason: stop`, JSON or SSE) WITHOUT calling the
      upstream, ending the harness auto-continue loop cleanly. A broken cycle (new
      fingerprint) fully re-arms.
- [x] Config: 7 `loop_*` fields on ProxyConfig (validated in `__post_init__`), env
      `CM_LOOP_*`, launcher flags (`--no-loop-guard`, `--loop-repeat-k`,
      `--loop-timings-m`, `--loop-accept-threshold`, `--loop-draft-n-min`,
      `--loop-cooldown-turns`, `--loop-hard-stop`), example TOML,
      wiki/configuration.md §3 matrix + §4g.
- [x] Observability: `loop_injections` / `loop_hard_stops` in ProxyStats + `/metrics`.
- [x] Surface A wired (proxy request path, pre-rewrite observation, post-rewrite
      tail-append). MCP surface = optional follow-up, not this phase.

### Phase 13 — Results (2026-07-19)

Shipped; **375 → 413 tests, 0 failures, 0 skips** (29 loop_guard core + 9 app-wiring).
`tests/test_loop_guard.py`: fingerprint units (whitespace/volatile/tool-call-id
normalization, trailing-segment scope, retry-vs-repeat anchors), full state-machine
ladder (fires exactly at k, cooldown suppresses exactly c, escalation caps at the final
notice, hard-stop on the third fire only when enabled, changed fingerprint re-arms),
timings corroboration (accelerates by one turn; never required; resets on
low-acceptance/small-draft/absent), and THE property test in the test_compactor.py
mold: hypothesis drives random turn sequences against an independent reference model —
a decision is produced **iff** the trailing run of identical turns >= k AND the
cooldown window is clear (subsumes "never fires below k" + "always fires at k").
`tests/proxy/test_app_loop_guard.py` pins the append-only invariant on the wire
(payload[:-1] byte-identical to the rewrite, breaker last), the synthetic hard-stop
(JSON + SSE, upstream NOT called), disabled mode (payload untouched), and the /metrics
counters. The livelock class from the 2026-07-19 incident is now mechanically bounded:
worst case k + c + 1 + c + 1 turns (default 11) to the final notice, and with hard-stop
on, a guaranteed clean end-of-run report instead of an unbounded burn.

---

## Phase 14 — Closed-loop governor ("dancing with the harness")

Spec: `tasks/phase14-spec.md` (the contract — diagnosis, checklists, acceptance criteria).
Built on branch `claude/closed-loop-governor`, on top of Phase 12. 14d (economic layer)
left specced, not implemented.

### Phase 14 — Results (2026-07-19)

Shipped 14a+14b+14c; **380 → 439 tests, 0 failures.**

**14a — Sensing (zero wire change).** New `proxy/sensing.py`: canonical content
serialization (str OR content-parts; str passes through so historical ids are unchanged),
per-message boundary hashes incl. `tool_calls` (assistant tool-call messages carry
`content: null` — without the payload the classifier is blind where agent wires are
densest), request-diff classifier {new-conversation | pure-append | tail-edit |
mid-wire-edit | head-rewrite(pos)} × cause {harness-edit | own-mutation | multimodal |
new}, per-conversation LRU ledger (`max_conversations`, default 32), and a PURE response
tee: `usage`/`timings` parsed from non-stream bodies AND the final SSE chunk via a bounded
tail buffer (forwarded bytes byte-identical — pinned by test; parse failures swallowed;
TTFT measured as the no-`timings` fallback). `/metrics` gains `closed_loop`:
`real_prompt_tokens` last/peak, `real_reuse_ratio` (1 − prompt_n/prompt_tokens),
`breaks_by_cause`, `native_compaction_observed`, learned ceilings, per-conversation
summary. Acceptance met: a 2026-07-19-class session needs zero server-log forensics.

**14b — Break-riding.** `_recall_frozen` and `_windowed` keyed PER CONVERSATION
(conversation key = first message hash + first non-system message hash — separates the
interleaved main/title/summary calls that thrashed the old single slot, 14/25 rebuilds).
Non-str (content-parts) anchors freeze via `stable_id_any` — image turns no longer force
per-turn breaks. Refresh policy is now flush-epoch: rebuild ONLY when the prefix is
already broken (classifier verdict `prefix_broken`, passed into the rewriter; or a Pass-3
trigger) or at the HARD bound — `recall_refresh_tokens` demoted to
`recall_max_stale_tokens` (a bound, not a cadence; 0 = legacy). Acceptance met (pinned):
steady-state synthetic session ⇒ zero voluntary breaks, every turn a byte-exact prefix
extension; `prefix_broken=True` flushes below the bound (the refresh rides the break).

**14c — Self-calibrating pressure.** Windowing trigger driven by REAL pressure
(`usage.prompt_tokens` + completion + estimated growth) when observed — the chars/4
estimate saw ~12% of the real wire (47K est vs 420K real over 25 requests). Learned
native-compaction ceiling: an observed harness head-rewrite records the preceding real
prompt size (minimum sample wins; cross-key detection handles compactions that replace
the first user message); effective high water =
`min(context_budget_ratio × n_ctx, ceiling_safety × learned_ceiling)` (`ceiling_safety`
default 0.8, 0 disables), low water scales proportionally so the hysteresis gap survives.
Profiles persist in the contextstore `state.json` keyed by harness fingerprint
(system-prompt head hash) — restart keeps calibration (pinned by round-trip test).

**Invariants kept green (pinned):** rewrite idempotency
(`test_idempotency_with_explicit_reference`, `test_multiturn_no_growth`), byte-identical
stream passthrough, recall/sensing failures never fail a request (controller calls are
try/except-wrapped enrichment on every path), rewriter = pure function of
(messages, setpoints, frozen state); the controller moves setpoints only at flush epochs.

**Config:** `recall_max_stale_tokens` (renamed; env `CM_RECALL_MAX_STALE_TOKENS`, flag
`--recall-max-stale-tokens`), `ceiling_safety` (`CM_CEILING_SAFETY`, `--ceiling-safety`),
`max_conversations` (`CM_MAX_CONVERSATIONS`, `--max-conversations`); TOML automatic;
example TOML + wiki/configuration.md + wiki/surface-a-proxy.md updated. Rename is safe:
the public repo (Phases 0–11) never shipped the old knob.

**NOTE (operational):** the long-running governor proxy on 127.0.0.1:8900 predates
Phase 14 — it must be restarted to pick this up. **Remaining verification gate:** the
one-time LIVE shakedown (llama-server + Hermes) cross-checking `/metrics` reuse numbers
against the server log — calibrating the sensor itself — has NOT been run yet; do it at
the first restart. 14d (measured-cost economics) remains specced in
`tasks/phase14-spec.md` for its own phase.

---

### Merge record (2026-07-19)

Phases 12 (sticky recall, `75e0192`), 14 (closed loop, `be3d353`), and 13 (loop guard,
renumbered from its working title "Phase 12" at merge, `fdfd982`) are all on `main`;
combined suite **477 tests, 0 failures**. The two features compose: the guard observes
the pre-rewrite transcript and hard-stops BEFORE sensing ever runs; breaker tails are
deliberately excluded from the sensing ledger's `sent_sig` (metered by `loop_injections`
instead) so 14b's own-mutation acceptance signal stays clean. Still owed: restart the
:8900 proxy + the one-time live sensor-calibration shakedown. NOT pushed anywhere —
local `main` carries the real-email history; publish only via the orphan `commit-tree`
recipe (see the re-publish notes).
