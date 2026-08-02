# Context Governor

**Run your local-LLM agent almost forever.** The Context Governor is a drop-in
layer between any OpenAI-compatible agent CLI (Hermes Agent, OpenCode, pi, …) and
your `llama-server` that keeps the prompt **approximately constant over an
arbitrarily long session** — instead of letting the transcript grow until the
CLI's own compaction eats it alive.

Not a demo: this is a long-term, daily-driven, self-proven setup. A governor
instance running right now reports, live from `GET /metrics`:

> **~1.6M tokens saved (~72%) over 115 requests · peak prompt ~9.6K tokens ·
> memory recall hit rate 100%**

> Python package: `contextmanager`. License: MIT. 576 tests, all offline.

## ⚡ Quick start

```powershell
git clone https://github.com/gbgh1/context-governor.git
cd context-governor
python -m venv .venv
.venv\Scripts\python -m pip install -e .    # Linux/macOS: .venv/bin/python -m pip install -e .
```

**1. Start your model** — `llama-server` (llama.cpp) on the default
`http://127.0.0.1:8080`.

**2. Start the governor:**

```powershell
.\integration\run-governor.ps1
```

If your harness already carries the governor entries, that's the whole ritual —
the bare call is enough. First time, let it wire your CLI for you
(provider + MCP entries, timestamped backup, fully reversible):

```powershell
.\integration\run-governor.ps1 --cli opencode      # or --cli hermes; undo any time with --revert
.\integration\run-governor.ps1 --provider ollama   # upstream flavor: llama (default) / ollama / openai
.\integration\run-governor.ps1 --config governor.toml --dry-run   # central TOML config; prints, doesn't run
```

(The wrapper runs on the repo venv and forwards every flag to the Python
launcher — on Linux/macOS, activate the venv and use the identical
`run-governor` console script. A commented starter config lives at
[`integration/governor.example.toml`](integration/governor.example.toml).)

**3. Point your agent at the governor** — set its OpenAI-compatible base URL to:

```
http://127.0.0.1:8900/v1
```

Done. Now open **http://127.0.0.1:8900/metrics** and watch the horizon: tokens
saved, messages handle-ized, recall hits, windowing triggers, real cache-reuse
ratios — live and cumulative, while your agent works.

## The problem it solves

Long agentic sessions on a small local model fall into a **compaction livelock**:
the CLI's protected message tail (full of big tool outputs and file dumps) grows
past the compaction threshold, there's no mechanism to re-compress it, so
compaction fires fruitlessly *every turn* and the session grinds to a halt.
Raising the context window only delays it.

The Context Governor fixes this structurally by keeping bulky, stable content
**off the wire** and in a durable store, reconstructing a bounded prompt each
turn.

## How it works — two surfaces over one engine

- **Core engine** — tiered token budget, hysteresis compaction with a *no-re-fire
  invariant* (a compaction can never trigger another), bounded summaries, exact token
  counts from `llama-server` (`/tokenize`, `/props`), and a durable store (authoritative
  `state.json` + human-auditable markdown notes + an FTS5/lexical retriever; a vector
  backend is pluggable behind the `Retriever` interface). The store is **lossless end to
  end**: content is persisted *before* it ever leaves the window, cold notes are gzipped
  (still searchable), and evicted notes are archived — never deleted — so a later request
  for any handle transparently *resurrects* it, byte-exact, even across restarts.
- **Surface A — endpoint proxy** (universal, no cooperation needed): a transparent
  OpenAI-compatible reverse proxy between *any* CLI and `llama-server`. It replaces bulky
  messages with short, parseable stubs (full content stored), so the CLI's
  API-reported prompt stays small and its native compaction rarely fires. Idempotent and
  prefix-stable, so KV-cache reuse is preserved. And it reads as well as it writes:
  **auto-recall** derives a query from the live conversation tail each turn and injects the
  most relevant off-wire memory back as one small, budgeted block — the model gets its
  memory back without ever knowing to ask.
- **Surface B — MCP server** (cooperative, precise): six MCP tools
  (`store_save`/`store_search`, `state_snapshot`/`state_load`,
  `context_checkpoint`/`context_rehydrate`) so an MCP-capable agent can *deliberately*
  externalize and recall state. Shares the same store as the proxy.

Both surfaces share one on-disk store, so a handle minted by either resolves in both.

On top of that, the proxy runs **closed-loop**: it observes the *real* prompt
tokens and cache-reuse ratios the server reports, classifies every prompt-prefix
break by cause, learns each harness's native compaction ceiling, and keeps its
own edits byte-stable between windowing triggers — so the KV cache actually gets
reused turn after turn. A built-in **loop guard** detects degenerate
repeated-turn spirals and breaks them mechanically.

## Proven in the field

- **First live run** (Hermes Agent + a 35B-A3B local model, a 138-message /
  73-tool-output build session): the proxy cut the wire prompt by **~60%**
  (7.03M → 2.82M characters, 405 messages handle-ized), every chat completion
  returned 200, and the long tool-heavy session completed with no livelock.
- **Months of daily use later**, the numbers hold: the live instance quoted at
  the top is at **~72% saved** with a peak prompt under 10K tokens — on sessions
  that would otherwise have blown far past the model's window.

## Install & verify

```bash
git clone https://github.com/gbgh1/context-governor.git
cd context-governor
python -m venv .venv
# Linux/macOS:           source .venv/bin/activate
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q          # 576 tests, no network / llama-server needed
```

See it work end-to-end with **no model required** (uses the offline heuristic
counter):

```bash
python examples/demo.py
```

The surfaces can also be run directly (the launcher wrapper above is the
recommended front door):

```bash
# Surface A — OpenAI-compatible proxy (listens on :8900, forwards to llama-server)
CM_UPSTREAM_BASE_URL=http://127.0.0.1:8080 python -m contextmanager.proxy

# Surface B — MCP stdio server (usually spawned by your CLI; --store-root shares the proxy's store)
python -m contextmanager.mcp --store-root ./contextstore
```

## Wire it to your agent CLI

[`integration/`](integration/README.md) has copy-paste setup for **Hermes Agent** and
**OpenCode** — provider entries, MCP registration, an optional system-prompt directive that
makes the agent use the cooperative tools, an immediate-relief Hermes config patch, and a
before/after measurement runbook. (`.\integration\run-governor.ps1 --cli <name>` automates
the same wiring.)

## Configuration

Everything is layered: built-in defaults < TOML config < provider profile < CLI
flags, with `CM_*` env vars honored throughout (the MCP server also accepts
`--store-root` / `--upstream-base-url` / `--transport`). Key knobs:
`CM_UPSTREAM_BASE_URL`, `CM_STORE_ROOT`, `CM_LISTEN_PORT`, `CM_MODEL_ALIAS`
(name the proxy advertises in `/v1/models`; default `context-governor`, `""` to
pass through), and `CM_DIFF_MIN_SIMILARITY` (lossless delta-compression of
near-duplicate content; `0` disables). See `src/contextmanager/proxy/config.py`
and `.../mcp/config.py`.

**llama-server is the source of truth for context size.** At startup the proxy reads the
real `n_ctx` from `/props` and:
- **(a) anchors** the per-message handle-ization threshold to it — `CM_HANDLE_THRESHOLD_RATIO`
  (default `0.02` = 2% of the true window), so the governor self-tunes to whatever `-c` you
  launch (fixed `CM_HANDLE_THRESHOLD_TOKENS` is the fallback when the server is unreachable);
- **(b) bounds the *total* wire** below `CM_CONTEXT_BUDGET_RATIO` (default `0.50` = 50% of the
  window) via **lossless budget-windowing** — paging out the oldest non-pinned middle messages
  to retrievable stubs (pinned head + recent tail kept verbatim). This pre-empts the CLI's own
  *lossy* compaction so it rarely needs to fire. Windowing uses **two-water hysteresis**: it
  triggers at the high water, cuts deep to `CM_CONTEXT_TARGET_RATIO` (default `0.35`) in one
  bite, then holds the stub frontier byte-stable between triggers — so the upstream's KV/prefix
  cache is actually reused turn after turn (a *compute* saving on top of the token saving) —
  plus a `CM_CONTEXT_EMERGENCY_RATIO` hard ceiling for when a harness floods the wire faster
  than the normal tiers can drain it;
- **(c) propagates** the true `n_ctx` into `/v1/models`, so CLIs read the real window instead
  of guessing.

Deeper concept notes (the surfaces, the store, the no-re-fire invariant, the
compaction mechanics): [wiki/index.md](wiki/index.md).

## License

[MIT](LICENSE).
