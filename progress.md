# Progress log — volatile-stamp normalizer (Pass -1)

**Status: IMPLEMENTED, tests green, VALIDATED against live traffic 2026-07-28.**

> **RESULT: fixed.** A 13-turn live Claude Code session (server log, ctx 53248)
> produced `f_keep = 1.000` on **10 of 11 post-warmup turns**. The message
> `forcing full prompt re-processing` appears exactly twice, both cold starts,
> and never again. Prompt eval per turn fell from 25,000-28,000 tokens
> (35-45 s of dead time) to **27-1,387 tokens (0.4-4 s)**.
>
> The single partial break (task 675, `f_keep = 0.697`) was **rescued by a
> context checkpoint** — `restored context checkpoint (n_tokens = 6438,
> n_past = 6438)` — costing 2,337 tokens instead of ~9,000. This is itself new:
> before the fix, checkpoints were always created and then erased as invalidated,
> because the divergence point always preceded them. With a stable prefix they
> finally land in useful positions.
>
> Observed `f_keep = 1.000` at sim_best values from 0.888 to 0.997, i.e. reuse
> now survives normal turn-to-turn wire churn, not just identical prompts.
Written 2026-07-28 by Claude (Opus), continuing the investigation logged in
`HARNESS.txt` (Command Code / kimi-k3, which hit its 5-hour limit mid-implementation).
Context: `problem.md`.

---

## What was implemented

The plan from `HARNESS.txt`, essentially unchanged — its analysis was correct.

### 1. `src/contextmanager/proxy/rewriter.py` — new module-level pass

```python
normalize_volatile_stamps(messages) -> list        # public
_normalize_volatile_content(content)               # str OR content-parts list
_normalize_volatile_text(text)                     # window-bounded regex sub
_BILLING_HEADER_MARKER = "x-anthropic-billing-header:"
_BILLING_VOLATILE_RE   = re.compile(r"\b(cc_version|cch)=[^;\s]*")
_VOLATILE_WINDOW       = 512
```

Behaviour:
- **Scope**: `role == "system"` messages only, and only when the marker appears in
  the first 512 chars. Replacement is bounded to that leading window, so a `cch=`
  occurring later in genuine prompt content can never be clipped.
- **Blanks** `cc_version=<v>` and `cch=<hex>` → `cc_version=`, `cch=`.
- **Keeps `cc_entrypoint`'s value** deliberately — it separates cli / sdk / vscode
  sessions and is a legitimate identity discriminator.
- Pure (returns a new list only when something changed, input never mutated),
  idempotent, never raises. Unknown shapes pass through unchanged — it fails
  **open**, so a bug re-introduces the old churn (visible in `/metrics`) rather
  than dropping a request.

Placed in `rewriter.py` rather than `sensing.py` because it is a wire transform,
and `sensing.py`'s module docstring explicitly promises to be "side-effect-free
with respect to the wire".

### 2. `src/contextmanager/proxy/app.py` — wired at ingress

Import extended, plus a `try/except`-guarded call. **The ordering is load-bearing:**

```
397  capture.record_in()        <- RAW request kept as forensic truth
408  normalize_volatile_stamps()  <- Pass -1
416  guard.observe_request()    ┐
436  controller.observe_request() ├─ all see ONE normalized wire
455  rewriter.rewrite_outgoing() ┘
483  capture.record_out()       <- what was actually forwarded
```

Normalization must precede `controller.observe_request()`, because that is where
`conversation_key()` runs. Captures keep the raw bytes, so an in/out diff still
documents exactly what the pass did.

### 3. `tests/proxy/test_volatile_normalize.py` — 14 tests

Shapes taken from the live capture, not invented. Covers: nonce+version blanked
/ entrypoint preserved; other parts untouched; string-content shape; non-system
messages untouched; system without the marker untouched; occurrences beyond the
window untouched; purity; idempotency; identity-returned-when-unchanged; junk
inputs; **key stable across the three real nonces (9b25f/cdb12/88fab) plus a
build-suffix flip**; a pin that the *unnormalized* wire still churns (documents
why the pass exists); side-call separation preserved; different entrypoints stay
separate; forwarded system bytes byte-identical across turns.

---

## Test results

```
508 passed   <- baseline before this change (includes the wire-capture tests)
522 passed   <- after (+14)
```

No existing test modified. Nothing else in the repo touched.

---

## Why this fixes both failure modes

The nonce broke the pipeline twice, and only fixing the wire fixes both:

1. **Identity** — `conversation_key()` hashes `messages[0]`, so the key churned
   every turn → every request filed as `new-conversation` → `prefix_broken=True`
   → the rewriter FLUSHED its frozen recall block and rebuilt it at the tail each
   turn → the ~16,752-token divergence point in the server log.
2. **Wire** — even with a perfect key, the nonce was still *forwarded*. On a
   hybrid SSM/recurrent model, reuse needs a byte-exact prefix, so a differing
   char 94 forces a full re-prefill from token ~24 regardless of identity.

A prior attempt (since reverted) normalized only for *hashing* and still saw full
re-processing, because failure 2 was untouched. Normalizing the wire at ingress
means identity is derived from exactly the bytes that get forwarded — the two
cannot drift apart.

Per `HARNESS.txt`'s reading of `rewriter.py`, **no recall changes are needed**:
the Phase-12 freeze/latch mechanism is already correct and only ever failed
because `flush = prefix_broken or windowing_shed` was `True` every turn. With a
stable key, turns classify as pure-append, the block freezes and is re-injected
byte-identically at the same anchor index.

---

## Validation — how to tell if it worked

Restart the governor (llama-server does **not** need restarting), then run 3+
turns and check:

```bash
curl -s http://127.0.0.1:8900/metrics | python -c "import sys,json;c=json.load(sys.stdin)['closed_loop'];print('breaks:',c['breaks_by_cause']);print('reuse:',c['real_reuse_ratio']);[print(' ',k,'turns=',v['turns']) for k,v in c['conversations'].items()]"
```

**PASS** — one conversation with `turns` incrementing past 1; `breaks_by_cause`
showing a single `new-conversation` (the genuine first turn) and nothing after;
`real_reuse_ratio` climbing above 0. In the server log: `f_keep` → 1.0 and prompt
processing shrinking to the new tail only.

**Expected on the FIRST turn after restart: one more full re-prefill.** The
llama-server slot still holds a cached prompt containing a raw nonce, so the
first normalized request necessarily diverges from it. Judge from turn 2-3
onward, not turn 1. The frozen recall block also starts empty after a proxy
restart, so full warm behaviour lands around turn 2-3.

**If `f_keep` still does not reach 1.0** once `turns` is incrementing correctly:
identity is fixed and the next suspect is the recall anchor. `HARNESS.txt`
reasoned by inspection that re-injection at the frozen index yields a pure
extension — that reasoning is plausible but **unverified**. Use the wire capture
(`--wire-capture-dir`) and diff two consecutive `-out` payloads; that shows the
divergence directly.

---

## Still open (not addressed by this change)

1. **Side-call eviction.** With `--parallel 1`, an interleaved title-gen call
   (~333 tokens) evicts the ~28k main-conversation state. The server's prompt
   cache is over its **token** budget, not its MiB budget:
   `cache state: 1 prompts, 556.162 MiB (limits: 2048.000 MiB, 53248 tokens, 104520 est)`
   — `104520 est` exceeds the `53248` limit, so only one prompt is retained. This
   will still cause periodic full re-prefills. Raising `--cache-ram` does not
   help (MiB was never the binding constraint).
2. **Harness generality.** The pass is *safe* for every harness (gated on the
   marker; opencode/hermes/pi wires come out byte-identical), but it only
   neutralizes the one measured pattern. A different harness with its own
   per-request nonce would need one more entry in `_BILLING_VOLATILE_RE`.
   Symptom to watch for: `breaks_by_cause` dominated by `new-conversation` for
   that harness. opencode/hermes stamp a *date* at ~char 4,500 — daily, not
   per-request — so they were never affected by this bug.
3. **Date rollover.** Deliberately NOT normalized: the date is semantic content
   the model should see. Costs at most one re-prefill per day.

---

## Repo state

Uncommitted, all from this investigation:
- `rewriter.py`, `app.py` — this change.
- `diagnostics.py`, `config.py`, `launcher.py`, `__main__.py`,
  `governor.example.toml`, `tests/proxy/test_wire_capture.py`,
  `tests/proxy/test_app_diagnostics.py`, `raw/*.py` — the wire capture, from the
  `HARNESS.txt` session.
- `problem.md`, `HARNESS.txt`, `progress.md` — documentation.

`problem.md` is now partly stale. Two of its conclusions were superseded by the
wire capture: HTTP-header identity is **dead** (LiteLLM strips everything except
`user-agent: litellm/1.93.0`), and a key-only fix is **insufficient** — the wire
must be normalized too.
