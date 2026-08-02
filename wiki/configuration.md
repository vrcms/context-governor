---
title: Configuration — every parameter, safely
updated: 2026-07-19
tags: [configuration, operations, reference, tuning]
---

# Configuring the Context Governor

The single reference for **every** knob, where to set it, what it does, and which direction
is safe to turn it. Part of the [[context-governor]]. If you only remember one thing:

> **The two levers are [[surface-a-proxy]]'s `handle_threshold_ratio` (which messages leave
> the wire) and `context_budget_ratio` (how small the wire is held). Everything else is a
> seatbelt or a default you rarely touch.**

The dormant [[no-re-fire-invariant]] engine (`budget.py`) and the Hermes-side YAML are
separate layers, documented further down — don't confuse them with the live proxy.

---

## 1. The mental model — what is even being configured?

There is **one** config object that matters at runtime: `ProxyConfig`
(`src/contextmanager/proxy/config.py`). It is a frozen, self-validating dataclass — the single
source of truth. Every entrypoint just builds a `ProxyConfig` and hands it to the proxy app.

Three layers exist; only the **proxy** is on the live critical path:

| Layer | Lives in | Status | Configured by |
|---|---|---|---|
| **Proxy** (`ProxyConfig`) | `proxy/config.py` | **LIVE** | TOML / env / `run-governor` flags |
| **Hysteresis engine** (`BudgetConfig`) | `budget.py` | dormant — built+tested, not wired | code only (no env/flag yet) |
| **Hermes Tier-0** (`compression.*`) | the *CLI's own* `config.yaml` | the host's setting, not ours | YAML, hot-reloads |

---

## 2. Which file does what

| File | Role |
|---|---|
| `src/contextmanager/proxy/config.py` | **`ProxyConfig`** — defines every proxy parameter, its default, and validation. **Add a parameter here or nowhere.** |
| `src/contextmanager/launcher.py` | **`run-governor`** — the recommended front door. Layered resolver (defaults → TOML → provider → flags), provider profiles, `--dry-run`, `--cli` auto-wiring. |
| `src/contextmanager/proxy/__main__.py` | **`context-governor-proxy`** — the bare env-var entrypoint. Maps `CM_*` → `ProxyConfig`. Use when you don't want the launcher. |
| `src/contextmanager/proxy/rewriter.py` | Where the params **take effect**: handle-ization, budget-windowing, diff-encoding, rehydration. |
| `src/contextmanager/budget.py` | `BudgetConfig` for the dormant hysteresis engine (`trigger_ratio`/`target_ratio`/floor). |
| `src/contextmanager/compactor.py` | `HysteresisCompactor` — enforces the floor-≤-low_water invariant at construction. |
| `integration/governor.example.toml` | The example central config. **Copy it, edit it, point `--config` at it.** |
| `integration/hermes-config.tier0.yaml` | The **Hermes-side** patch (the CLI's own config, not ours). |
| `integration/README.md` | Copy-paste wiring steps (llama-server → proxy → CLI). |

---

## 3. How to set a parameter (and precedence)

There are **two entrypoints**, each with its own surface. Prefer `run-governor`.

### `run-governor` (launcher) — recommended
Precedence, low → high:

```
built-in defaults  <  config-file [proxy] table  <  selected --provider  <  CLI flags
```

```bash
run-governor --config governor.toml --provider llama        # normal run
run-governor --provider ollama --dry-run                    # print resolved config, don't run
run-governor --provider llama --cli hermes                  # also auto-wire Hermes (persistent)
run-governor --revert --cli hermes                          # undo that wiring
```

### `context-governor-proxy` (env-only)
Reads `CM_*` env vars, falls back to dataclass defaults. No TOML, no provider logic.

```bash
CM_UPSTREAM_BASE_URL=http://127.0.0.1:8080 CM_CONTEXT_BUDGET_RATIO=0.50 \
  python -m contextmanager.proxy
```

### The reachability matrix (important)
Not every parameter is reachable from every surface. The **TOML `[proxy]` table is the only
surface that can set every field** (the launcher copies any `[proxy]` key whose name matches a
`ProxyConfig` field).

| Parameter | TOML `[proxy]` | `run-governor` flag | `CM_*` env |
|---|:--:|:--:|:--:|
| `upstream_base_url` | ✅ | `--upstream-url` | `CM_UPSTREAM_BASE_URL` |
| `upstream_api_key` | ❌ *(never)* | ❌ *(never)* | `CM_UPSTREAM_API_KEY` |
| `store_root` | ✅ | `--store-root` | `CM_STORE_ROOT` |
| `listen_host` / `listen_port` | ✅ | `--listen-host` / `--listen-port` | `CM_LISTEN_HOST` / `CM_LISTEN_PORT` |
| `handle_threshold_ratio` | ✅ | `--handle-threshold-ratio` | `CM_HANDLE_THRESHOLD_RATIO` |
| `handle_threshold_tokens` | ✅ | `--handle-threshold-tokens` | `CM_HANDLE_THRESHOLD_TOKENS` |
| `context_budget_ratio` | ✅ | `--context-budget-ratio` | `CM_CONTEXT_BUDGET_RATIO` |
| `context_target_ratio` | ✅ | `--context-target-ratio` | `CM_CONTEXT_TARGET_RATIO` |
| `protect_first_n` / `protect_last_n` | ✅ | ❌ | ❌ |
| `rehydrate_budget_tokens` | ✅ | `--rehydrate-budget-tokens` | `CM_REHYDRATE_BUDGET_TOKENS` |
| `auto_recall_k` | ✅ | `--auto-recall-k` | `CM_AUTO_RECALL_K` |
| `recall_budget_tokens` | ✅ | `--recall-budget-tokens` | `CM_RECALL_BUDGET_TOKENS` |
| `recall_max_stale_tokens` | ✅ | `--recall-max-stale-tokens` | `CM_RECALL_MAX_STALE_TOKENS` |
| `hotness_half_life_seconds` | ✅ | `--hotness-half-life-seconds` | `CM_HOTNESS_HALF_LIFE_SECONDS` |
| `ceiling_safety` | ✅ | `--ceiling-safety` | `CM_CEILING_SAFETY` |
| `max_conversations` | ✅ | `--max-conversations` | `CM_MAX_CONVERSATIONS` |
| `stub_preview_chars` | ✅ | `--stub-preview-chars` | `CM_STUB_PREVIEW_CHARS` |
| `diff_min_similarity` | ✅ | `--diff-min-similarity` | `CM_DIFF_MIN_SIMILARITY` |
| `diff_lookback` | ✅ | `--diff-lookback` | `CM_DIFF_LOOKBACK` |
| `diff_max_chars` | ✅ | `--diff-max-chars` | ❌ |
| `tokenize_max_chars` | ✅ | `--tokenize-max-chars` | ❌ |
| `request_timeout` | ✅ | `--request-timeout` | `CM_REQUEST_TIMEOUT` |
| `model_alias` | ✅ | `--model-alias` | `CM_MODEL_ALIAS` |
| `loop_guard_enabled` | ✅ | `--no-loop-guard` *(disable)* | `CM_LOOP_GUARD_ENABLED` |
| `loop_repeat_k` | ✅ | `--loop-repeat-k` | `CM_LOOP_REPEAT_K` |
| `loop_timings_m` | ✅ | `--loop-timings-m` | `CM_LOOP_TIMINGS_M` |
| `loop_accept_threshold` | ✅ | `--loop-accept-threshold` | `CM_LOOP_ACCEPT_THRESHOLD` |
| `loop_draft_n_min` | ✅ | `--loop-draft-n-min` | `CM_LOOP_DRAFT_N_MIN` |
| `loop_cooldown_turns` | ✅ | `--loop-cooldown-turns` | `CM_LOOP_COOLDOWN_TURNS` |
| `loop_hard_stop` | ✅ | `--loop-hard-stop` *(enable)* | `CM_LOOP_HARD_STOP` |

> **`protect_first_n` / `protect_last_n` are TOML-only.** If you need to change them and aren't
> using a TOML config, you must add one. They have no flag and no env var by design — they
> rarely change.

---

## 4. Parameter reference (the live proxy)

Grouped by concern. **Default** and **valid range** are enforced by `__post_init__`; an invalid
value raises `ValueError` at startup (the launcher wraps it as `LauncherError` and exits).

### 4a. Connection & serving
| Parameter | Default | Range | What it does |
|---|---|---|---|
| `upstream_base_url` | *(required)* | URL | The real llama-server / provider. The proxy needs `/tokenize` + `/props` reachable for exact token accounting. |
| `store_root` | `./contextstore` | path | Root of the [[durable-store]] (`state.json`, `notes/`, `index.db`). Both surfaces share one store. |
| `upstream_api_key` | `None` | str | Forwarded to the upstream. **Only ever read from the env var named by the provider's `api_key_env`** — never from TOML or a flag. |
| `listen_host` | `127.0.0.1` | str | Interface the proxy binds. Keep loopback unless you mean to expose it. |
| `listen_port` | `8900` | 1..65535 | Port the proxy listens on. |
| `model_alias` | `context-governor` | str / `None` | Name presented in `/v1/models`. Set to `""`/`None` to pass the upstream's real model name through unchanged. Chat requests forward verbatim regardless. |
| `request_timeout` | `300.0` | float s | Upstream HTTP timeout. Generation can be long — don't set this small. |

### 4b. Lever 1 — handle-ization (*which* messages leave the wire) → cadence
A message that tokenizes `>=` the threshold is paged out to the store and replaced by a tiny
stub. Lowering the threshold means **more (smaller) messages get stubbed** — i.e. the governor
"bites more often."

| Parameter | Default | Range | What it does |
|---|---|---|---|
| `handle_threshold_ratio` | `0.02` | 0.0..1.0 | When `> 0` **and** the true `n_ctx` is known (from `/props`), the threshold is anchored to `ratio × n_ctx` — so it self-tunes to whatever `-c` llama-server runs. `0` disables anchoring → the fixed token count is used. |
| `handle_threshold_tokens` | `2000` | **> 0** | Fixed fallback threshold, used when `n_ctx` is unknown (server unreachable) or `ratio = 0`. Must be `> 0`. |

### 4c. Lever 2 — budget windowing (*how small* the wire is held) → depth
Bounds the **total** wire below `context_budget_ratio × n_ctx` by paging out the **oldest
non-pinned middle** messages (lossless → retrievable stubs). The head (`protect_first_n`) and
recent tail (`protect_last_n`) are **never** touched. This pre-empts the host CLI's own lossy
compaction so it rarely fires.

**Two-water hysteresis (Phase 11):** `context_budget_ratio` is the HIGH water — trigger and
ceiling. On crossing it, windowing cuts deep to `context_target_ratio` (LOW water) in one bite,
then a **sticky frontier** re-stubs the same messages byte-identically every turn WITHOUT
advancing — so between triggers the wire prefix is byte-stable and the upstream's KV/prefix
cache is actually reused (a compute saving on top of the token saving). Watch
`windowing_triggers` in `/metrics`: triggers ≪ requests is the win.

| Parameter | Default | Range | What it does |
|---|---|---|---|
| `context_budget_ratio` | `0.50` | 0.0..1.0 | HIGH water: wire ceiling + windowing trigger. `0` disables windowing entirely. Needs `n_ctx` known to act. |
| `context_target_ratio` | `0.35` | 0.0..<budget | LOW water: on trigger, page down to here in one bite. Must be `< context_budget_ratio`. `0` = legacy per-turn nibble to the ceiling. |
| `ceiling_safety` | `0.8` | 0.0..1.0 | **Phase 14c.** When sensing observes the host harness compact its own transcript, the effective high water becomes `min(context_budget_ratio × n_ctx, ceiling_safety × learned_ceiling)` — the governor windows (controlled, lossless) *before* the harness floods. `0` disables the learned ceiling. Learned profiles persist in the contextstore (`state.json`, keyed by harness fingerprint). |
| `protect_first_n` | `2` | ≥ 0 | Head messages (system/spec) pinned — never paged out. |
| `protect_last_n` | `6` | ≥ 0 | Recent-tail messages pinned — never paged out. |

Since **Phase 14c** the windowing *trigger* uses REAL pressure when available —
`usage.prompt_tokens` observed from the upstream's responses plus estimated growth — instead of
the string-only chars/4 estimate (which was blind to the `tools` array, content-parts, and
template overhead, ~88% of a real agent wire in the 2026-07-19 shakedown).

### 4d. Rehydration (paging content *back* in)
Two read paths. **Explicit** (Pass 2): the model references `[[cm:stored handle=H]]` and the
full content is appended back. **Implicit** (Pass 4, auto-recall): the proxy derives a query
from the live tail, searches the store, and injects the top *off-wire* slices as one marked
`[[cm:recall]]` user-role message before the final message — the model can't page-fault on memory
it can't see, so the governor recalls *for* it. The block is stripped on entry (never
accumulates) and, since Phase 12, **frozen between refreshes**: the same bytes are re-injected
before the same anchor message every turn, so each wire is a byte-exact prefix extension of the
previous one and the upstream's prefix/SSM cache can extend instead of re-prefilling. Since
**Phase 14b** the freeze is keyed **per conversation** (interleaved conversations no longer
thrash one slot), non-str (content-parts) anchors freeze too, and the refresh policy is
**break-riding**: the block is rebuilt — in one jump, at the new tail — when the prefix is
*already broken* this turn (a harness edit, a new conversation, a multimodal turn, or a Pass-3
windowing trigger — the refresh rides the break for free), or when a HARD bound is hit: growth
past `recall_max_stale_tokens` since the freeze, or the anchor message vanishing. In steady
state voluntary prefix breaks are zero. The total wire bound stays
`context_budget_ratio × n_ctx + recall_budget_tokens`.

| Parameter | Default | Range | What it does |
|---|---|---|---|
| `rehydrate_budget_tokens` | `4000` | ≥ 0 | Max tokens paged back per request when a message contains an explicit `[[cm:stored handle=H]]` reference. Stubs themselves do **not** auto-expand (keeps the common case bounded). `0` disables. |
| `auto_recall_k` | `3` | ≥ 0 | Max slices auto-recalled per request (Pass 4). `0` disables auto-recall entirely. |
| `recall_budget_tokens` | `1500` | ≥ 0 | Max tokens the recall block may occupy (~2% of a 75K window at the default). `0` disables. |
| `recall_max_stale_tokens` | `4000` | ≥ 0 | Sticky-recall staleness BOUND (14b; the Phase-12 cadence demoted): refreshes ride already-broken turns; only this much growth since the freeze (or anchor loss) *forces* a break. `0` = legacy recompute-and-move every turn — a guaranteed per-turn prefix break. |
| `hotness_half_life_seconds` | `86400` | > 0 | Decay half-life of access scores: recall re-ranking (and eviction, when run) favor recently-touched memory. Shorter = faster-cooling memory; match it to project cadence (6-12h for fast-moving 1-2 week projects). |
| `stub_preview_chars` | `200` | ≥ 0 | Head/tail characters kept as a preview inside each stub. Bigger = more readable stubs but a larger wire. |

### 4e. Diff-encoding (lossless delta compression)
When a bulky message is a near-duplicate of a recent same-role stored note, it's replaced by a
unified diff instead of a full stub (often ~90% smaller on a file re-read with one line changed).

| Parameter | Default | Range | What it does |
|---|---|---|---|
| `diff_min_similarity` | `0.5` | 0.0..1.0 | `difflib` ratio required to diff against a base. **`0` disables** diff-encoding. Higher = stricter (only very similar content diffs). |
| `diff_lookback` | `6` | ≥ 0 | How many recent stubs to scan for a diff base. |

### 4f. Safety seatbelts — leave these on
| Parameter | Default | Range | What it does |
|---|---|---|---|
| `diff_max_chars` | `20000` | ≥ 0 | Above this size a bulky message becomes a plain stub instead of diffing (`difflib` is O(n·m) and pathological on big repetitive logs). **`0` = no cap = unsafe** (can freeze the proxy for minutes). |
| `tokenize_max_chars` | `100000` | ≥ 0 | Content larger than this is handle-ized with a cheap char-based token **estimate** instead of being POSTed to `/tokenize` (slow + a DoS vector). **`0` = no cap = unsafe.** |
| `max_conversations` | `32` | ≥ 1 | LRU cap on per-conversation governor state (the Phase-14 sensing ledger, frozen recall blocks, windowing frontiers). Bounds pathological many-conversation churn; each entry is tiny. |

### 4g. Loop guard — the mechanical loop-breaker (Phase 13)
Detects a **degenerate cycle** (the agent repeating the same turn — same tool call, same
result, same reply — with fresh ids/timestamps stripped before comparison) and **appends** a
breaker notice at the **tail** of the next outbound request. Tail-append only, never a
mid-history edit: the upstream's (hybrid-SSM) prompt cache needs a byte-stable prefix.
First fire = notice; a cycle that survives the cooldown gets ONE escalated final notice;
`loop_hard_stop` (off by default) ends the run with a synthetic final response after that.
llama-server speculative-decoding timings (`draft_n_accepted/draft_n ≈ 1.0` = verbatim
recycling) *accelerate* the trigger by one turn but are never required. Watch
`loop_injections` / `loop_hard_stops` in `/metrics`.

| Parameter | Default | Range | What it does |
|---|---|---|---|
| `loop_guard_enabled` | `true` | bool | Master switch for the whole feature. |
| `loop_repeat_k` | `3` | ≥ 2 | Consecutive near-identical turns that trigger the breaker. |
| `loop_timings_m` | `3` | ≥ 1 | Consecutive verbatim-recycling timings observations that lower the effective `k` by one (never below 2). |
| `loop_accept_threshold` | `0.99` | 0..1 | Draft-acceptance ratio counting as verbatim recycling. |
| `loop_draft_n_min` | `200` | ≥ 0 | Minimum `draft_n` for a timings observation to count (tiny drafts accept fully all the time). |
| `loop_cooldown_turns` | `3` | ≥ 0 | Turns to suppress re-injection after a fire. |
| `loop_hard_stop` | `false` | bool | After two ignored notices, the proxy answers with a synthetic final response (`finish_reason: stop`) instead of calling the model — ends a harness auto-continue loop. |

---

## 5. Tuning safely — the playbook

Think in two axes (this is the whole game):

- **Cadence / "how often it bites"** → `handle_threshold_ratio` (and `…_tokens`). Lower ⇒ more,
  smaller messages get stubbed, more often.
- **Depth / "how big each bite is"** → `context_budget_ratio`. Lower ⇒ the wire is held smaller,
  more of the old middle externalized **per pass, without changing cadence**.

| Goal | Turn | Direction | Safe-ish band |
|---|---|---|---|
| Governor takes more burden off the CLI (bigger bite, same rhythm) | `context_budget_ratio` | ↓ | `0.50 – 0.75` |
| Stub even small messages (more aggressive externalization) | `handle_threshold_ratio` | ↓ | `0.01 – 0.04` |
| Keep more recent turns verbatim | `protect_last_n` | ↑ | `4 – 12` |
| Fewer rehydration pop-ins | `rehydrate_budget_tokens` | ↓ | `0 – 6000` |

### Guardrails — do **not**
- **Don't starve the protected tail.** `context_budget_ratio × n_ctx` must comfortably exceed the
  tokens held by `protect_first_n + protect_last_n` messages, or windowing can't reach budget and
  the wire stays large. With the defaults (8 pinned messages on a ~75k window) anything `≥ 0.50`
  is safe; below ~0.40 you're risking it.
- **Don't set `handle_threshold_tokens ≤ 0`** — it raises at startup.
- **Don't disable the seatbelts** (`diff_max_chars=0`, `tokenize_max_chars=0`) on real workloads.
- **Don't put API keys in the TOML or a flag.** Only the env var named by `api_key_env` is read.
- **Verify with `--dry-run`** before a real run — it prints the fully-resolved config (key
  redacted) so you can confirm precedence did what you expected.

---

## 6. The dormant hysteresis engine (`BudgetConfig`)

Built and tested, but **not on either shipped surface's critical path** — the proxy delegates
actual compaction to the host CLI and merely keeps it rarely-needed. Documented here so the knobs
are understood for when it gets wired. It is configurable in **code only** today.

| Parameter | Default | Constraint | Role |
|---|---|---|---|
| `n_ctx` | *(required)* | — | True context size (from `/props`). |
| `reserved_headroom_tokens` | *(required)* | `0 < H < n_ctx` | Reserved for generation + margin. Usable budget `B = n_ctx − H`. |
| `state_cap_tokens` | *(required)* | ≥ 0 | Hard cap on the `state_snapshot` tier. |
| `distilled_cap_tokens` | *(required)* | ≥ 0 | Hard cap on the `distilled_memory` tier. |
| `trigger_ratio` | `0.75` | `target < trigger < 1` | **High-water** — compaction *fires* at `int(trigger × B)`. This is the **cadence** knob. |
| `target_ratio` | `0.50` | `0 < target < trigger` | **Low-water** — compaction *runs down to* `int(target × B)`. This is the **depth** knob. |
| `protect_first_n` | `3` | ≥ 0 | Pinned head. |
| `protect_last_n` | `8` | ≥ 0 | Pinned tail. |

**The floor invariant (this is the dangerous one).** The non-compactable floor
`F = pinned_head + state_cap_tokens + distilled_cap_tokens` must be `≤ low_water`, or
`HysteresisCompactor.__init__` raises `FloorExceedsTargetError`. So you **cannot lower
`target_ratio` freely** — drop it below `F / B` and construction fails. Keep
`target_ratio × B ≥ F`. See [[no-re-fire-invariant]] for why `low_water < high_water` (the gap)
is what guarantees a compaction can never trigger another. To deepen each compaction without
firing more often: **lower `target_ratio`, keep `trigger_ratio`** (same metaphor as the proxy's
`context_budget_ratio`).

---

## 7. The Hermes-side layer (Tier-0 YAML)

This is the **host CLI's own** config, not the governor's — it lives in Hermes' `config.yaml`,
and `compression.*` + `model.context_length` hot-reload on a running gateway. Template:
[`integration/hermes-config.tier0.yaml`](../integration/hermes-config.tier0.yaml). Background:
[[hermes-compaction]].

| Key | Tier-0 value | Note |
|---|---|---|
| `model.context_length` | `75776` | **MUST equal llama-server `-c`.** |
| `compression.enabled` | `true` | — |
| `compression.threshold` | `0.72` | Fire later, leave real headroom. |
| `compression.target_ratio` | `0.12` | Smaller protected-tail token budget. |
| `compression.protect_last_n` | `8` | Fewer protected tail messages. |
| `compression.protect_first_n` | `3` | — |
| `compression.hygiene_hard_message_limit` | `5000` | Per-message hard cap (a single 40k dump still breaks this — that residual is exactly what the proxy's handle-ization removes). |
| `auxiliary.compression.model` | `""` | `""` = same local model. |

> **The silent-drop gotcha.** `model.context_length`, llama-server `-c`, and the summarizer's
> context must all be **equal**. If the summarizer's context is *smaller* than the main model's,
> Hermes drops the middle of the conversation **silently** (no summary, no error). Keep them in
> lockstep.

---

## 8. Worked examples

**A. Central TOML (recommended)** — copy `integration/governor.example.toml`, edit, run:
```toml
[proxy]
listen_port = 8900
store_root = "./contextstore"
context_budget_ratio = 0.50     # depth — lower = governor takes a bigger bite
handle_threshold_ratio = 0.02   # cadence — lower = stub smaller messages too
protect_last_n = 6              # TOML-only; keep recent turns verbatim

[providers.llama]
upstream_base_url = "http://127.0.0.1:8080"
```
```bash
run-governor --config governor.toml --provider llama --dry-run   # verify first
run-governor --config governor.toml --provider llama             # then run
```

**B. Env-only**, no launcher:
```bash
CM_UPSTREAM_BASE_URL=http://127.0.0.1:8080 \
CM_CONTEXT_BUDGET_RATIO=0.60 \
CM_HANDLE_THRESHOLD_TOKENS=2000 \
  python -m contextmanager.proxy
```

**C. A flag overrides everything** (highest precedence) for a one-off:
```bash
run-governor --config governor.toml --provider llama --context-budget-ratio 0.55
```

---

## 9. Failure modes (what raises, and where)

| Symptom | Cause | Fix |
|---|---|---|
| `LauncherError: invalid configuration: …` | A `ProxyConfig` validation failed (out-of-range ratio, `port`, `handle_threshold_tokens ≤ 0`, …). | Fix the value; the message names the field. |
| `LauncherError: config file not found / malformed TOML` | Bad `--config` path or TOML syntax. | Check the path / `--dry-run`. |
| `LauncherError: provider 'anthropic' is deferred` | Anthropic/Claude provider chosen. | Use `llama`/`ollama`/`openai`. |
| `FloorExceedsTargetError` | Engine only — floor `>` low_water. | Raise `target_ratio` or lower the caps. |
| Wire not shrinking | `context_budget_ratio = 0`, or `n_ctx` unknown (upstream `/props` unreachable). | Set the ratio `> 0`; ensure llama-server is reachable. |
| Hermes silently loses the middle | Summarizer context `<` `model.context_length`. | Make all contexts equal. |

---

## See also
- [[context-governor]] — architecture overview · [[surface-a-proxy]] — where the params act
- [[no-re-fire-invariant]] — why the high/low-water gap matters · [[durable-store]] — the store
- [`integration/README.md`](../integration/README.md) — copy-paste wiring
