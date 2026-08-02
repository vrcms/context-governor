# Problem 2: residual prefix invalidations, and long "dead" waits

*Written 2026-07-28, after `problem-1-FIXED.md` was resolved (commits `a4290b9`,
`2367da1`). Self-contained for handoff. Everything below is measured from a live
16-turn session unless explicitly marked HYPOTHESIS.*

> **STATUS 2026-07-28 (second sitting, synthetic repro against the live stack):**
> - **C — VALIDATED.** Dead waits gone: ≤1.0 s governor CPU/turn where the old
>   build burned ~57 s. Details in the Problem C section.
> - **A — hypothesis CONFIRMED.** Windowing triggers on pre-stub estimated
>   pressure (37,666 est vs 21,259 real on the trigger turn). Fix 1 (correct
>   the estimate) is the right one; fix 2 (raise the ratio) rejected.
> - **B — server side measured healthy** (reasoning streams live through the
>   governor, 40 ms median). Remaining: one live opencode turn to confirm the
>   TUI renders it; the volume question (1.9k-token generations) is unchanged.

---

## Read this first: the priority is not what it looks like

Full wall-clock accounting of one live session (872 s span, 16 completed turns,
timestamps parsed from the llama-server log):

```
client-side gap   575 s   66%   <- llama-server IDLE between turns
decode            234 s   27%
prefill            62 s    7%
```

**Two thirds of the wall clock is spent with llama-server doing nothing.**
Measured as the interval from `update_slots: all slots are idle` to the next
`params_from_`. Those gaps grow monotonically with session size:

```
0.8  0.9  1.4  1.1  1.4  1.3  2.9  21.4  25.9  58.6  68.5  76.3  79.6  86.8  97.6 s
```

~1 s early, **97.6 s** by turn 16. That growth pattern rules out tool execution
(which would be random, not monotonic) and points at something scaling with
conversation size.

**CORRECTION — the governor is the prime suspect.** An earlier revision of this
document claimed "not the governor, `last_ttft_ms = 4429.6` covers rewrite +
prefill". **That was wrong.** `t0` for TTFT is set at `app.py:544`, while the
rewrite runs at `app.py:455` — so TTFT starts *after* the rewrite and measures
only the upstream call. It says nothing about how long the governor spent
building the request, which happens inside exactly the window measured as idle.

Priorities follow from this, and they are NOT what earlier drafts of this
document said:

1. **Problem C (client-side gap)** — 66% of wall clock. Largest by far.
2. **Problem B (decode)** — 27%. Real but secondary.
3. **Problem A (prefix invalidations)** — ~20 s of 872 s. Effectively noise now.

---

## Stack

- **Pipeline (the session measured here — ALL LOCAL, no LiteLLM, no VPS):**
  `opencode (local) -> context-governor (local :8900) -> llama-server (local :8080)`
  Network latency is therefore NOT a candidate explanation for anything below.
  (A separate pipeline exists for Claude Code, which does route
  `claude (VPS) -> litellm (VPS) -> governor -> llama-server`. LiteLLM is only
  needed for Claude Code's Anthropic protocol; opencode speaks OpenAI directly.
  `problem-1`'s billing-nonce finding came from the Claude Code path.)
- **Model**: Kwaipilot KAT-Coder-V2.5-Dev, `MTP-UD-Q5_K_XL` (25.5 GiB), arch
  `qwen35moe`, 40 trunk layers + 1 MTP head, MoE 256/8, **hybrid SSM + attention**.
- **Server**: `--ctx-size 53248 --cache-type-k q8_0 --cache-type-v turbo2
  --flash-attn on --n-cpu-moe 35 --parallel 1 --ubatch-size 2048 --cache-ram 2048
  --spec-type draft-mtp,ngram-mod --reasoning-format auto`
- **Governor**: built-in defaults — `governor.toml` was deleted and the
  `--config` auto-injection in `run-governor.ps1` is commented out, so:
  `context_budget_ratio = 0.50`, `context_target_ratio = 0.35`,
  `ceiling_safety = 0.8`, `auto_recall_k = 3`, `handle_threshold_ratio = 0.02`,
  `protect_first_n = 2`.
- **Hardware**: RTX 3070 Ti Laptop (8 GB) — **227 MiB free**, below the project's
  own ">= 300 MiB always" rule. 31.7 GB RAM.

---

## Problem A — residual prefix invalidations  [FIXED 2026-07-28]

> ### STATUS: FIXED — the pressure estimate now models the SENT wire
>
> **Confirmed cause** (measured on the trigger turn, 60 KB read):
> `est_pressure 37,666` vs `real prompt 21,259`. The estimate ran a flat
> `(chars - prev_chars) / 4` over the **incoming** wire, but Pass 1 pages out
> anything at or above the handle threshold and sends a ~126-token stub instead.
> The read contributed ~15,000 phantom tokens, crossed the high water, and Pass 3
> windowed — shedding from `protect_first_n` forward and breaking the prefix for
> nothing. Plus a systematic double-count of `last_completion_tokens`, which
> already arrives inside the appended region.
>
> **Fix (candidate 1, per the confirmation):** `sensing._growth_estimate()`
> estimates each appended message at what it will COST ON THE WIRE — `stub_est`
> if it will be paged out, `chars/4` otherwise; `tool_calls` always in full since
> they are never stubbed. `last_completion_tokens` removed. Stub size comes from
> `PromptRewriter.stub_tokens_estimate()`, which renders a real stub through
> `make_stub` so it cannot drift from the format.
>
> Verified against the live case: **legacy growth 15,000 tokens → 126**, removing
> 14,874 of the 16,407 measured overshoot; the rest was the double-count.
>
> Still a cheap PRE-GATE — no tokenizer calls. Falls back to the legacy aggregate
> on non-append turns and when the rewriter's settings are unknown, so existing
> callers are unaffected. Residual error is now tens of tokens per turn.
>
> Candidate fix 2 (raising `context_budget_ratio`) stays **rejected** — the
> overshoot scaled with tool-result size, so raising the line only moves it.
>
> Pinned by `tests/proxy/test_pressure_estimate.py` (11 tests). One existing test
> in `test_sensing.py` had pinned the double-count and was corrected.
>
> **DEFERRED (agreed, not done):** permanent rewrite-latency instrumentation — a
> `perf_counter` around `rewrite_outgoing` surfaced in `/metrics`. The proxy still
> has no visibility into its own latency, which is why Problem C hid behind
> `last_ttft_ms` for a whole session and why this estimate drift went unmeasured.
> Worth adding the next time this area is touched.
>
> **Not yet validated in a live session** — expect `windowing_triggers` to stay
> at 0 through a normal session, with `own-mutation` breaks gone.
>
> ---
>
> ### FOLLOW-UP HARDENING (2026-07-28, same day, two more commits)
>
> The pressure-estimate fix above stops OVER-estimating growth. It does not
> stop windowing from having nothing left to shed when the mass really is
> unsheddable. Two further changes, both reversible (each is its own commit,
> tagged checkpoint `pre-toolcalls-threewater` before either landed):
>
> **1. `tool_calls` handle-ization (Pass 1 extension).** A live wire capture
> showed `tool_calls` at 53% of one request's total chars (116,736 of
> 222,177) — an agentic turn dominated by a large write_file/edit_file/shell
> argument. Neither Pass 1 nor Pass 3 had ever looked at `tool_calls`, only
> `content`, so that mass was completely invisible and unsheddable — the
> root cause the pressure fix couldn't reach on its own.
> `rewriter._handleize_tool_calls()` pages out large STRING VALUES inside
> `function.arguments`, editing the already-JSON-decoded object and
> re-encoding it (never replacing `arguments` with a non-JSON stub — the
> chat template parses it into a mapping via `arguments|items`, so that
> would be a hard request failure, not just a quality regression). Verified
> end-to-end: JSON stays valid, small values (a path, a flag) pass through
> untouched, large values get stubbed, deterministic and idempotent, input
> never mutated. 16 tests in `test_toolcalls_handleize.py`.
>
> **2. Third windowing tier — `context_emergency_ratio` (HIGH water), opt-in,
> default 0 (disabled = today's exact behavior).** The hysteresis latch
> exists so a failed shed doesn't keep breaking the prefix for nothing — but
> that's also its failure mode: on the live session, pressure climbed to 96%
> of n_ctx while latched, because the gap-based re-arm didn't fire fast
> enough against unsheddable mass. Crossing the new HIGH water (set BELOW the
> host harness's own compaction line, e.g. opencode's ~0.75) overrides the
> latch and forces a retry every request regardless of the gap — costs at
> most one more prefix break with no benefit (no worse than a normal first
> trigger), but can recover mass a latched conversation would otherwise never
> re-attempt. `RewriteResult.windowing_emergency` / `/metrics
> windowing_emergencies` make it directly observable whether the tier is
> doing anything, not just whether it's configured — continuing the session's
> pattern of measuring rather than assuming a mechanism works. Verified
> end-to-end with a genuinely-unsheddable message forced into the eligible
> range (non-str content, so `_window_out` always returns 0): the latch holds
> exactly as before when the tier is disabled, and overrides exactly once
> pressure crosses the HIGH water when enabled — 12 tests in
> `test_three_water.py`, including the config-validation boundary
> (`context_emergency_ratio` must be `0` or strictly `> context_budget_ratio`).
>
> **Neither change is validated in a live session yet.** Config surfaces for
> the new ratio: `--context-emergency-ratio` (launcher), `CM_CONTEXT_EMERGENCY_RATIO`
> (env), `context_emergency_ratio` (`governor.toml`). Recommended first live
> value if opencode's compaction line is confirmed at 0.75: something in the
> 0.60-0.70 range, comfortably below that line and above `context_budget_ratio`.

### Measured (original investigation)

Whole-session log scan (16 completed turns):

```
f_keep = 1.000                : 13
f_keep < 1.0                  :  3   (0.697, 0.685, +1)
forcing full re-processing    :  1
context checkpoint RESTORES   :  2
```

Governor `/metrics` at the same point:

```
breaks_by_cause   : {"new-conversation": 2, "own-mutation": 2}
windowing_triggers: 2          slices_recalled: 0
learned_ceilings  : {}
real_prompt_tokens: {"last": 17830, "peak": 17830}
real_reuse_ratio  : 0.964      avg_reuse_ratio: 0.8526
main conv         : turns = 16
```

### What this rules out

- **NOT recall.** `slices_recalled = 0` means the sticky recall block was
  re-injected byte-identically every turn and never rebuilt — working as designed.
  (`rewriter.py:95`: `recalled_handles` is *"Empty on sticky turns that
  re-injected the frozen block"*, so this counter is an exact flush-epoch detector.)
- **NOT a learned ceiling.** `learned_ceilings = {}` is empty, so the
  `ceiling_safety * learned_ceiling` term cannot be binding.
- **NOT identity.** `turns = 16` on one conversation; the `problem-1` fix holds.

### What it IS

`own-mutation: 2` maps 1:1 onto `windowing_triggers: 2`. Pass-3 windowing
(`rewriter._window_out`) sheds messages starting at `protect_first_n = 2` and
walking **forward**, so every trigger stubs messages at the same early index and
breaks the prefix there. The proxy invalidates its own cache.

### The part that doesn't add up

Windowing should not have fired at all:

```
high water   = 0.50 x 53248 = 26,624 tokens
peak REAL prompt            = 17,830 tokens
```

~~HYPOTHESIS (unverified)~~ **CONFIRMED 2026-07-28 (synthetic repro against
the live stack).** The trigger runs on an *estimate*, not on real tokens. From
`sensing.py`:

```python
growth_est = max(0, (chars - entry.incoming_chars) // 4)   # _EST_CHARS_PER_TOKEN
pressure   = last_prompt_tokens + last_completion_tokens + growth_est
```

A single large tool result (reading a ~50 KB file) adds ~13,000 *estimated*
tokens in one turn, which can push `pressure` over 26,624 while the real prompt
is barely half that. Windowing then sheds content it did not need to shed, and
breaks the prefix doing it — cost with no benefit.

**Repro** (12-turn synthetic session, ~19.5 KB near-duplicate tool results per
turn, ~54 KB of tools-schema mass so real prompt rides at 18-22k like the live
session, one 60 KB read at turn 11). Metrics came out with the live session's
exact signature: `windowing_triggers: 2`, `breaks_by_cause: own-mutation: 2`.

The deciding numbers (est_pressure computed with sensing.py's own formula):

```
turn   est_pressure   real prompt_tokens   prompt_n   outcome
 7        26,655            21,783            2,770    TRIGGER 1 (barely over; real 4.8k UNDER the line)
 8        26,747            22,049              284    latch suppressed
 9        26,977            22,292              261    latch suppressed
10        27,210            22,529              241    latch suppressed
11        37,666            21,259            2,246    TRIGGER 2 (16.4k overshoot; real DROPPED 1.3k from shedding)
12        26,174            21,498              258    no trigger
```

Two compounding estimate errors, both in the predicted direction:

1. **`growth_est` counts content that Pass 1 stubs away.** The 60 KB read
   contributed +15,000 estimated tokens to pressure; in the sent wire it landed
   as a ~50-token stub. Real prompt never came near the budget — the shed was
   pure cost (2 own-mutation prefix breaks, ~5,000 tokens reprocessed).
2. **The completion is double-counted** — once as `last_completion_tokens`,
   once more inside `growth_est` (the assistant reply is part of the incoming
   wire's chars delta). Small at these sizes (~120 tokens), but systematic.

The latch (Phase 14b) worked exactly as designed: 3 of 5 spurious crossings
were absorbed without a break. Without it this repro would have produced 5
own-mutation breaks instead of 2.

This kills candidate fix 2 (raise `context_budget_ratio`): it only moves the
line, and the overshoot scales with tool-result size, so any line is crossable
while the estimate runs on pre-stub chars. Candidate fix 1 is the right one —
see updated wording below.

### Candidate fixes (updated after confirmation)

1. **Correct the pressure estimate** — gate windowing on what the wire will
   actually contain, not pre-stub chars. Shape: keep `pressure` as a cheap
   pre-gate, but before shedding verify against the POST-Pass-1 wire: the
   constant mass (tools array + template, learnable as
   `last_real_prompt − tokens(sent messages)` per conversation) plus a precise
   count of the rewritten message list. After Pass 1 the big content is already
   stubbed, so the precise count is cheap. This preserves the cheap common
   case and eliminates both error terms (stub-invisible growth + completion
   double-count). Needs a test pinning the turn-11 case above.
2. ~~**Raise `context_budget_ratio`**~~ — **rejected by the repro:** the
   overshoot scales with tool-result size (15k on one 60 KB read), so any fixed
   line is crossable while the estimate runs on pre-stub chars. It would have
   masked THIS session, not fixed it.
3. **Do nothing.** Defensible in the meantime: the latch capped the damage at
   2 breaks per session here and in the live run, both absorbed by checkpoint
   restores. `avg_reuse_ratio = 0.8526`.

### Note: checkpoints now work

Before the `problem-1` fix, checkpoints were created and immediately erased as
invalidated every turn, because the divergence point always preceded them. With
a stable prefix they land in useful positions and self-organise: one restore hit
`pos_min = 6437` against a divergence at `6443` — **6 tokens of slack**. This is
emergent from turns now being small and incremental. Do not "improve" checkpoint
placement before re-measuring; it may already be solved.

---

## Problem B — long "dead" waits (the real cost)

### Symptom

30-70 s where nothing streams and the model appears hung. Reported by the user as
*"it's like the LLM is dead during this period."*

### Measured — it is DECODE, not prefill

Per-turn breakdown from the live log:

```
  task  prefill_s   ptok  decode_s   dtok  total_s  decode%
   405        0.4     27       4.1    116      4.5      91%
   965        2.6    220      27.1    485     29.6      91%
  1611       11.3   4649      40.3   1163     51.6      78%
  2404        2.7    395      67.4   1911     70.1      96%
  3598        3.4    641      20.8    527     24.2      86%
```

The worst case, task 2404, had `f_keep = 1.000` — a *perfect* cache hit. Its
2,695 ms of prefill was fine; the 67,402 ms was the model generating **1,911
tokens** at 28.35 t/s.

Generation length is climbing over the session: 95 -> 176 -> 283 -> 485 -> 1163
-> **1911** tokens. Decode throughput is *healthy* and stable at 25-31 t/s. The
problem is volume, not speed.

### Why it looks dead rather than slow

**Server side: MEASURED 2026-07-28 — the pipe is live.** A streaming probe
through the governor (thinking-heavy prompt): first delta at 0.41 s, then
`delta.reasoning_content` chunks arrive continuously (123 chunks, median
inter-delta gap **40 ms**, max 137 ms), then `content` streams. llama-server is
NOT silent during thinking and the governor forwards reasoning deltas promptly
— so during a 67 s / 1,911-token generation the wire delivers live tokens the
whole time.

**Client side: opencode SHOULD render it.** The user's config
(`~/.config/opencode/config.json`) declares the governor model with
`"reasoning": true` via `@ai-sdk/openai-compatible`, whose stream parser
extracts `delta.reasoning_content` into reasoning parts, and opencode renders
reasoning parts in the TUI. So the "dead" look during the observed session is
best explained by Problem C's governor stalls (nothing had reached the server
— nothing TO stream), not by invisible thinking. Residual check: watch ONE
thinking turn in opencode live; if thinking text streams, the visibility
hypothesis is dead and what remains is purely the volume question below.

(If a client that does NOT render `reasoning_content` is ever put in front of
this stack — e.g. Claude Code through LiteLLM, unverified — the dead-terminal
look returns with zero server-side cause. `--reasoning-format none` leaves
`<think>` inline in `content` and streams for any client.)

### Candidate directions

1. **Make thinking visible.** Try `--reasoning-format none`, which leaves
   `<think>` inline in `message.content` so it streams. Changes perception
   immediately at zero throughput cost. Caveat recorded in the launch script:
   this interacts with `preserve_thinking` and prompt-prefix stability — verify
   `f_keep` stays at 1.000 afterwards.
2. **Cap thinking length.** `--reasoning-budget`, or the model card's
   non-thinking mode (temp 0.7 / top_p 0.8 / top_k 20). Directly cuts the 1,900
   token generations. Quality trade-off is the user's call.
3. **Recover decode throughput.** VRAM free is **227 MiB**, below the project's
   ">= 300 MiB always" rule. 25-31 t/s vs the ~35-40 t/s this config was tuned
   for. `$CpuMoeLayers` 35 -> 36 buys VRAM headroom at the cost of RAM.
   Worth ~20-30%, not the 4x that direction 2 could give.
4. **Check governor overhead.** `messages_handle_ized = 188` implies significant
   store I/O per request, and one earlier sample showed `last_ttft_ms = 30367`.
   If wall-clock wait materially exceeds llama-server's reported `total time` for
   the same turn, the gap is proxy overhead and belongs in this list. Not yet
   measured.

---

## Problem C — governor CPU stall between turns  [FIXED 2026-07-28]

> ### STATUS: FIXED — `rewriter._similarity` now uses rapidfuzz
>
> **Confirmed cause:** `difflib.SequenceMatcher` in the diff-encoding pass.
> Measured on real ~20 KB store notes: **5-17 s per comparison**, x
> `diff_lookback = 6` = ~60 s per eligible message. That matched the 56.6 s of
> governor CPU measured during a single 97 s dead wait.
>
> **Fix:** swap the metric to `rapidfuzz.distance.Indel.normalized_similarity`,
> keeping an exact difflib fallback when rapidfuzz is absent.
>
> | case | difflib | rapidfuzz | speedup |
> |---|---|---|---|
> | real store notes, 4 pairs ~20 KB | 39.99 s | 0.041 s | **981x** |
> | 19 KB repetitive (difflib worst case) | 29,840 ms | 0.03 ms | **~893,000x** |
>
> **No recalibration needed.** On near-duplicates — the only regime where
> `diff_min_similarity` decides anything — the metrics agree to 3-4 decimals
> (1.0000/1.0000, 0.9996/0.9996, 0.9968/0.9969, 0.9895/0.9896, 0.9408/0.9432,
> 0.7587/0.7692), and every threshold decision matches. Pinned by
> `tests/proxy/test_diff_similarity.py`.
>
> `diff_max_chars = 20000` and `diff_lookback = 6` were left ALONE — bounding
> them was the earlier proposal and is no longer necessary. Full-fidelity diffs
> on large content are retained.
>
> **VALIDATED 2026-07-28 against the live stack.** Two synthetic sessions
> driven through the running governor (PID on :8900, rapidfuzz build):
>
> - **Run 1** (14 turns, near-dup ~19.5 KB tool results, 255 KB/request by the
>   end, 91 handle-izations — hundreds of rapidfuzz similarity calls on ~19.5 KB
>   pairs): governor CPU delta per turn **0.05 → 1.0 s**, proxy overhead
>   (wall − upstream timings) **≤ 0.9 s/turn**. Same regime that burned
>   **56.6 s of CPU in one gap** before the fix.
> - **Run 2** (12 turns, real prompt 18-22k, one 60 KB read, 256 KB/request):
>   governor CPU delta **≤ 0.64 s/turn**, overhead ≤ 0.9 s.
>
> Metrics after run 1: `breaks_by_cause: {new-conversation: 1}` only,
> `real_reuse_ratio 0.94` — no collateral damage. The residual CPU growth
> (0.05→1.0 s across 14 turns) is linear in wire size (hashing/canonicalizing
> ~255 KB), ~0.5% of the old cost. The dead waits are gone; confirm subjectively
> in one real opencode session.

## Problem C — original investigation (kept for the method)

### Measured

```
log span                          872 s
llama-server idle between turns   575 s  (66%)
n gaps 16, mean 35.9 s, max 97.6 s
gap sequence: 0.8 0.9 1.4 1.1 1.4 1.3 2.9 21.4 25.9 58.6 68.5 76.3 79.6 86.8 97.6
governor last_ttft_ms             4429.6   (at last_prompt_tokens = 23259)
governor chars_in 5,066,424 -> chars_out 180,713 (pct_saved 96.4)
retrieval avg_search_ms 20.16, page_in_calls 0, corpus 33
```

Measurement method: parse the llama-server log, take each
`update_slots: all slots are idle` and find the next `params_from_`.

### What this narrows to

- **Not network** — every hop is on one machine in this session.
- **Not llama-server** — by definition it is idle for the whole interval.
- **Not tool execution** — OBSERVED: the client renders the tool's output first,
  *then* sits dead on `Build · [LOCAL] Context-Governor` with no tool running.
  The gap is between "client decides to call the model" and "llama-server
  receives a request".
- **That window contains the governor's whole rewrite path**, which TTFT does not
  measure (see correction above).

### CONFIRMED: the gap is the governor burning CPU

Test 1 below was run 2026-07-28. Sampling `Get-Process python` for the governor
PID across one dead-wait period and into the next active period:

```
DURING the dead wait  (CPU seconds, consecutive samples)
  935.8  940.9  948.2  950.7  954.8  957.7  960.3  962.2  964.2  967.2
  970.3  973.3  975.7  978.0  980.7  983.6  987.0  990.4  992.4
  -> +56.6 s of CPU, climbing ~2-7 s per sample = ONE CORE PINNED AT 100%

THE MOMENT the wait ends and the client starts working
  992.41  992.44  992.45  992.47  992.48  992.53  992.55
  -> +0.14 s across six samples = FLAT
```

Climbing resumes on the next dead-wait period. The correlation is exact and
inverted: governor CPU climbs while the client is idle, and flattens while the
client works.

**This eliminates every non-governor explanation.** Not the client (idle while
the governor burns), not network, not llama-server, and not I/O — I/O wait does
not accumulate CPU time. The dead wait is CPU-bound single-threaded work inside
the proxy's own request path.

What remains open is *which* CPU-bound work. The candidates, ranked:

### PRIME HYPOTHESIS — `difflib` in the diff-encoding pass

`rewriter.py:465`:

```python
candidates = [h for (h, r) in recent_stubs if r == role][-self.config.diff_lookback:]
ratio = difflib.SequenceMatcher(None, base, content, autojunk=False).ratio()
```

with defaults `diff_max_chars = 20000`, `diff_lookback = 6`,
`diff_min_similarity = 0.5`.

`SequenceMatcher` is O(n*m) — the code's own comment calls it "pathological" —
and `autojunk=False` **disables the popular-element heuristic that normally keeps
it tractable on large sequences**. Two 20 KB strings is ~4x10^8 character
comparisons in pure Python, per candidate pair, up to 6 pairs per eligible
message. As history accumulates more stubs, more candidates qualify, so cost
grows with conversation length — matching the observed 1 s -> 97.6 s ramp.

Corroborating: this session pushes large tool outputs (news-scrape JSON dumps,
mixed-script text), `chars_in = 5,066,424` over 19 requests (~267 KB/request),
`messages_handle_ized = 290`. Exactly the shape that lands in the diff path.

### Other candidates in the same window (untested)

- **Remote tokenizer round-trips.** `LlamaServerTokenCounter` POSTs to
  llama-server `/tokenize`. Those calls DO hit llama-server but do not emit
  `params_from_`, so they are invisible to the idle measurement above and would
  be silently counted as "idle".
- **Store write path.** Handle-ization writes notes; `page_in_calls = 0` rules
  out read-path work but the write path was never measured.
- **Client-side work in opencode** scaling with history size. Still possible,
  but demoted: the observation above shows the client is already past tool
  execution and waiting on the model call.

### Other CPU-bound candidates in the same path (if the A/B comes back negative)

- `recall.extract_query` / `recall.select_diverse` — cost never measured.
- `canonical_content` — `json.dumps` over content-parts for every message, every
  turn. Cheap per call but O(total chars) and called constantly.
- `wire_signature` — sha1 per message. Should be milliseconds at 267 KB; listed
  only for completeness.

Ruled out as the *mechanism* by the CPU evidence: the remote `/tokenize`
round-trips and the store write path. Both are I/O, and I/O wait does not
accumulate CPU time.

### How to decide (cheapest first, no code changes)

1. ~~**Watch the governor process's CPU during a gap.**~~ **DONE — confirmed
   above.** One core pinned for the whole wait, flat the instant it ends.
2. **A/B the diff pass with one launcher flag** — `diff_min_similarity <= 0`
   short-circuits the entire diff path (`rewriter.py:436`):

   ```
   run-governor.ps1 --diff-min-similarity 0
   ```

   If the gaps collapse, the hypothesis is confirmed and the fix is a tuning
   question (lower `diff_max_chars`, reduce `diff_lookback`, or re-enable
   `autojunk`). Costs only delta-encoding efficiency, which shows up as a lower
   `pct_saved`.
3. **Only if both come back negative**, instrument the rewrite directly: wrap
   `rewriter.rewrite_outgoing` at `app.py:455` with a `perf_counter` and record
   it next to `ttft_ms`. Arguably worth doing permanently — the proxy currently
   has no visibility into its own latency, which is why this went unnoticed.

---

## Still open from problem-1 (unchanged)

**Prompt-cache token budget.** With `--parallel 1`, an interleaved title-gen
side-call (~333 tokens) can evict the main conversation's cached state. The
server's prompt cache is over its **token** budget, not its MiB budget:

```
cache state: 1 prompts, 556.162 MiB (limits: 2048.000 MiB, 53248 tokens, 104520 est)
```

`104520 est` exceeds the `53248` limit, so only one prompt is retained. Raising
`--cache-ram` does not help — MiB was never the binding constraint. This did not
bite in the observed session (est stayed within budget at these sizes) but will
resurface as context grows.

---

## Method note for whoever picks this up

This investigation burned hours on confident hypotheses that the data later
killed — cache-ram sizing, the recall block, a key-only identity fix. Every one
of them was resolved by *capturing bytes*, not by reasoning:

- `GET /metrics` -> `closed_loop.breaks_by_cause` separates
  `own-mutation` (ours to fix) from `harness-edit` (not ours).
- `--wire-capture-dir <path>` (see `diagnostics.py`) dumps incoming and outgoing
  payloads plus headers; diffing two consecutive `-out` files shows the exact
  divergence.
- The server log's `prompt eval time` vs `eval time` split settles
  prefill-vs-decode in one line — it is what proved Problem B is decode.

Measure first. The two hypotheses flagged above (`pressure_tokens` overshoot,
`reasoning_content` not rendered) are each one measurement away from certainty.
