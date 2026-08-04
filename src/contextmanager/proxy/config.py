"""ProxyConfig — configuration for the Phase 3 endpoint proxy.

Normative per Phase 3 spec §2. Frozen dataclass with post-init validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProxyConfig:
    """Immutable configuration for the endpoint proxy.

    Attributes:
        upstream_base_url: llama-server base URL, e.g. "http://127.0.0.1:8080".
        store_root: filesystem root for the DurableStore.
        upstream_api_key: optional API key forwarded to the upstream.
        listen_host: host the proxy listens on.
        listen_port: port the proxy listens on (1..65535).
        handle_threshold_tokens: messages whose token count is >= this get
            handle-ized (their full content paged out to the DurableStore).
        stub_preview_chars: number of head/tail characters kept in the stub
            preview emitted in place of the handle-ized content.
        rehydrate_budget_tokens: maximum tokens paged back in per request via
            auto-rehydration of explicit handle references.
        request_timeout: upstream HTTP request timeout in seconds (generation
            can be long).
        handle_threshold_ratio: when > 0 (default 0.02) AND the upstream's true
            context size (llama-server /props n_ctx) is known at startup, the
            per-message handle-ization threshold is ANCHORED to the real window
            (ratio * n_ctx, floored), making the governor self-tune to whatever
            `-c` the server runs. 0 disables anchoring -> the fixed
            handle_threshold_tokens is always used. llama-server is the source of
            truth for context size, not the CLI.
        handleize_toolcall_args: master switch for stubbing values inside
            tool_call arguments. **Default False, and the default is the
            finding.** Unlike message content, `arguments` is the model's own
            prior output, so a stub there is a template the model imitates in
            its next tool call — observed live on 2026-08-03, with the markers
            reaching /bin/bash as literal input and corrupting the task. Setting
            this True re-enables the (correct, tested) machinery below; do not,
            until that imitation problem has an answer.
        toolcall_threshold_ratio: the same anchoring for tool_call ARGUMENT
            values (default 0.004), which need their own, much lower setpoint.
            The message threshold reaches almost none of that mass: measured
            2026-08-03, argument sizes are mid-tail (p50 94 chars, p90 1,630),
            so 2% of n_ctx fires on 0 of 333 verbatim values while the mass
            itself is 43% of the peak prompt and UNSHEDDABLE by Pass 3.
        toolcall_min_shrink_ratio: hard floor under that ratio (default 2.0),
            as a multiple of the stub's own rendered size. A value smaller than
            the stub that replaces it makes the wire BIGGER, so this bound is
            physical and holds at every context size — the same reasoning as
            window_min_shrink_ratio, which exists because break-even was the
            wrong bar for an operation with unpriced costs. 0 = no floor.
        context_budget_ratio: when > 0 (default 0.50) AND n_ctx is known, bound the
            TOTAL wire to this fraction of the real window by paging out the oldest
            non-pinned middle messages (lossless — they become retrievable stubs).
            This pre-empts the CLI's own (lossy) compaction so it rarely fires. 0
            disables windowing. With hysteresis (below) this is the HIGH water:
            the trigger and the ceiling.
        context_target_ratio: the LOW water (default 0.35). When windowing triggers
            at the high water it pages down TO this fraction in one big bite, then
            holds the stub frontier still — so between triggers the wire prefix is
            byte-stable and the upstream's KV/prefix cache is actually reused
            (compute saving, not just tokens). Must be < context_budget_ratio.
            0 = legacy behavior (page to the ceiling every turn).
        context_emergency_ratio: the THIRD (HIGH) water, ABOVE
            context_budget_ratio, added 2026-07-28. Together the three form
            LOW (context_target_ratio) / MID (context_budget_ratio, the
            existing trigger+ceiling) / HIGH (this field) — the field keeps its
            original name for backward compatibility, but functions as the MID
            water once this is set. 0 (default) disables the emergency tier
            entirely: behavior is UNCHANGED from before this field existed.
            RATIONALE (measured 2026-07-28): the normal trigger can LATCH — if
            a shed pass cannot reach the low water (unsheddable mass: a raw
            multimodal content-parts message, or before the 2026-07-28 fix,
            large tool_calls arguments), it holds the wire steady rather than
            re-breaking the prefix for nothing, and waits for pressure to grow
            by the hysteresis gap before trying again. On one live session that
            let real pressure climb to 96% of n_ctx while latched, because the
            gap-based re-arm didn't fire fast enough relative to growth. When
            pressure crosses `context_emergency_ratio * n_ctx`, the governor
            OVERRIDES the latch and forces a shed attempt every request
            regardless — a hard "you must keep trying" floor that costs at
            most an extra prefix break with no benefit (the same cost a normal
            first trigger already pays) but can recover newly-eligible mass a
            latched conversation would otherwise never re-attempt. Set this
            BELOW the host harness's own compaction line (opencode's default
            is ~0.75 * n_ctx) so the emergency shed always gets a chance to act
            before the harness's own lossy compaction does.
        protect_first_n / protect_last_n: messages at the head (system/spec) and the
            recent tail that budget-windowing never pages out (pinned + recent window).
        model_alias: if set (default "context-governor"), the proxy presents the
            upstream's model under THIS name in /v1/models (and the Ollama discovery
            aliases), inheriting all other fields. Set to None/"" to pass the real
            model name through unchanged. Chat requests forward verbatim — llama-server
            serves the loaded model regardless of the requested name.
        auto_recall_k: max slices auto-recalled per request (Pass 4 anticipatory
            demand paging: an implicit query from the live tail searches the store
            and injects relevant OFF-wire memory as one marked system message).
            0 disables auto-recall entirely.
        recall_budget_tokens: max tokens the Pass-4 recall block may occupy. The
            total wire bound becomes context_budget_ratio*n_ctx + this (~2% of a
            75K window at the default).
        recall_max_stale_tokens: sticky-recall staleness BOUND (Phase 14b: the
            Phase-12 refresh cadence demoted to a hard bound). Once a recall
            block is built it is FROZEN — same bytes, re-injected before the
            same anchor message every turn. It is recomputed (in one jump, at
            the new tail) ONLY when the prefix is already broken this turn
            (harness edit / new conversation / multimodal turn / Pass-3
            windowing trigger — the refresh rides the break for free) or when
            this hard bound is hit: the conversation has grown past this many
            estimated tokens since the freeze, or the anchor message vanished.
            Between refreshes the whole previous wire is a byte-exact prefix of
            the next one, so upstream prefix caches (KV or hybrid-SSM state)
            can extend instead of re-prefilling. 0 = legacy behavior: recompute
            and move the block every turn (a guaranteed per-turn prefix break).
        recall_max_stale_ratio: anchors that bound to the real window (default
            0.25) instead of holding it fixed. A constant 4000 tokens is three
            different policies on three servers — 2% of a 200K window, 6% of
            65K, 50% of 8K — and only the middle one was ever measured. Above
            the windowing band (budget - target = 0.15), so the block refreshes
            at most once per windowing cycle and tends to land on a flush epoch
            where the break is already paid for. 0 = use the fixed token value.
            Ignored when recall_max_stale_tokens is 0, since that explicitly
            selects legacy per-turn recompute and a ratio must not re-enable
            stickiness the operator turned off.
        hotness_half_life_seconds: exponential-decay half-life of the store's
            access scores (default 86400 = 24h). Shorter = memory "cools" faster,
            so recall ranking and (when run) eviction favor the recent working
            set — suit this to project cadence (e.g. 6-12h for 1-2 week projects
            with fast-moving focus).
        ceiling_safety: fraction of the LEARNED native-compaction ceiling used
            as the effective windowing high water (Phase 14c). When the sensing
            layer observes the host harness compact its own transcript (a
            head-rewrite), it records the real prompt size that preceded it;
            thereafter the effective high water is
            ``min(context_budget_ratio * n_ctx, ceiling_safety * learned_ceiling)``
            so the governor windows (controlled, lossless) BEFORE the harness
            floods. 0 disables the learned ceiling (ratio-only, today's
            behavior); until a first observation the ratio applies regardless.
        max_conversations: LRU cap on per-conversation governor state (the
            sensing ledger, sticky recall blocks, windowing frontiers). Each
            entry is small (hashes + counters); the cap only bounds pathological
            many-conversation churn.
        loop_guard_enabled: mechanical loop-breaker (Phase 13): fingerprint each
            completed turn and, when the agent repeats the same turn
            loop_repeat_k times, APPEND a breaker notice at the tail of the next
            outbound request (tail-only — the prefix stays byte-stable for the
            upstream's prompt cache). False disables the guard entirely.
        loop_repeat_k: consecutive near-identical turns that trigger the breaker.
        loop_timings_m: consecutive llama-server verbatim-recycling timings
            observations (draft acceptance >= loop_accept_threshold with
            draft_n > loop_draft_n_min) that ACCELERATE the content trigger by
            one turn. Corroboration only — never required (spec decoding off =
            signal simply absent).
        loop_accept_threshold: draft acceptance ratio counting as recycling.
        loop_draft_n_min: minimum draft_n for a timings observation to count.
        loop_cooldown_turns: after an injection, suppress re-injection for this
            many turns; a cycle that survives a second (escalated) notice keeps
            getting the escalated notice each cooldown.
        loop_hard_stop: when True, a cycle that survives TWO injected notices is
            ended with a synthetic final response (the proxy answers instead of
            the model). OFF by default.
        diag_enabled: measure the outgoing wire's composition per component
            (tools / system / string content / content-parts / tool_calls) and
            pair it with the real usage.prompt_tokens of the SAME request; read
            it at /diagnostics. Char counts only, so the cost is len() plus one
            compact serialization of the non-string parts. ON by default —
            without it, every claim about where the prompt mass lives (and
            therefore every decision about what to shed) is inference.
        diag_tokenize: additionally issue exact /tokenize calls per component
            (6 per sampled request) so the UNACCOUNTED chat-template residual
            can be computed rather than estimated. OFF by default — it adds
            upstream round-trips to the request path.
        diag_max_samples: ring size for retained samples.
        wire_capture_dir: when set, dump every request as received AND the exact
            payload forwarded upstream to req-<seq>-{in,out}.json files in this
            directory (headers redacted), so consecutive turns can be diffed
            offline to attribute prefix breaks — the measured answer to
            own-mutation vs harness-edit. None (default) = capture off.
    """

    upstream_base_url: str
    store_root: str = "./contextstore"
    upstream_api_key: Optional[str] = None
    listen_host: str = "127.0.0.1"
    listen_port: int = 8900
    handle_threshold_tokens: int = 2000
    stub_preview_chars: int = 200
    rehydrate_budget_tokens: int = 4000
    request_timeout: float = 300.0
    model_alias: Optional[str] = "context-governor"
    diff_min_similarity: float = 0.5
    diff_lookback: int = 6
    # Upper size bound (chars) for diff-encoding. difflib.SequenceMatcher is O(n*m) and
    # pathological on large, repetitive content (log files!), so above this size a bulky
    # message becomes a normal stub instead of freezing the proxy for minutes. 0 = no cap
    # (unsafe; restores the old unbounded behaviour). ~20 KB covers typical file re-reads.
    diff_max_chars: int = 20000
    # Upper size bound (chars) for calling the tokenizer. Content larger than this is
    # definitely bulky AND too big to POST to llama-server /tokenize (slow + a DoS risk),
    # so it is handle-ized using a cheap char-based token ESTIMATE instead. 0 = no cap.
    tokenize_max_chars: int = 100000
    # Explicit context window, when the upstream advertises none (Tier 3 of the
    # window resolver). The OpenAI standard carries no context size anywhere, so
    # a server that exposes no vendor extension leaves the governor with only
    # observed evidence to work from; this is the operator saying what they know.
    upstream_n_ctx: Optional[int] = None
    handle_threshold_ratio: float = 0.02
    # tool_call ARGUMENT threshold, anchored to the real window like every other
    # setpoint. Measured 2026-08-03 across three harnesses: argument mass is
    # mid-tail, not a few giants, so the message threshold (2% of n_ctx) reaches
    # almost none of it -- 0 of 333 verbatim values on the opencode capture.
    # 0.004 comes from the sweep's optimum (256 tok at n_ctx 65536); the net
    # saving curve peaks around there and TURNS OVER below it, because each stub
    # costs ~137 tokens of its own.
    # OFF by default since 2026-08-03. Stubbing a tool_call argument is not the
    # same operation as stubbing message content, and the difference is not a
    # matter of degree: `content` is context the model READS, while
    # `tool_calls[].function.arguments` is the model's OWN PRIOR OUTPUT. Editing
    # it puts text in the exact slot an autoregressive model imitates when it
    # produces the next tool call — and it does. Measured on a live opencode run
    # (aborted, `_runs/wire-ABORTED-toolcall-imitation-111732`): the model copied
    # the stub markers verbatim, CM's real handles included, into new `bash`
    # commands, which the shell then received as literal input:
    #     /bin/bash: line 1: [[cm:stored: command not found
    # The task produced wrong output, not just slow output. The threshold below
    # is correct and tested; the OPERATION is what is unsafe, so the machinery
    # stays behind this flag rather than being deleted.
    handleize_toolcall_args: bool = False
    toolcall_threshold_ratio: float = 0.004
    # Hard floor under that ratio, as a multiple of the stub's own rendered size.
    # This is physics, not preference: a value smaller than the stub replacing it
    # BLOATS the wire, so the floor must hold at every context size (0.004 * 8192
    # would be 33 tokens against a ~137-token stub -- a net loss on every fire).
    # Expressed as a ratio so it tracks stub_preview_chars instead of drifting
    # from it, exactly like window_min_shrink_ratio.
    toolcall_min_shrink_ratio: float = 2.0
    # Handle-ize oversized TEXT parts inside content-parts lists (2026-08-02).
    # Pass 1 and Pass 3 both bail on `not isinstance(content, str)`, so a harness
    # that sends tool results as content-parts is INVISIBLE to the governor: an
    # opencode run measured structured_content at 66.3% of the wire against
    # 12.9% string_content, i.e. two thirds of the payload had no code path.
    # OFF by default: it changes the bytes of a message shape the rewriter has
    # never touched before, so it is opt-in until proven on a live harness.
    handleize_content_parts: bool = False
    context_budget_ratio: float = 0.50
    context_target_ratio: float = 0.35
    context_emergency_ratio: float = 0.0
    # Minimum SHRINK a windowing page-out must achieve, as a multiple of the
    # stub's own size: page only if orig_tokens >= ratio * stub_tokens (4.0 =
    # the message must shrink to <=25% of itself). 0 = legacy break-even.
    #
    # WHY (measured 2026-08-02). _window_out's only guard was
    # `if stub_tokens >= orig_tokens: return 0` — a BREAK-EVEN test. Break-even
    # is the wrong threshold for an operation with costs the test does not
    # model: paging a message that was already sent verbatim breaks the upstream
    # prefix (a full re-prefill) and hides the content from the model. At
    # break-even those costs are pure loss. A live hermes run paged 27 messages
    # under 100 tokens, netting 39 tokens each after the ~25-token stub — and
    # 5 of them were assistant turns, i.e. the model's own prior reasoning.
    #
    # Expressed as a RATIO, not an absolute floor, so it tracks the stub format
    # instead of drifting when that format changes — same reasoning as
    # stub_tokens_estimate, which renders a sample rather than hardcoding a size.
    window_min_shrink_ratio: float = 4.0
    protect_first_n: int = 2
    protect_last_n: int = 6
    auto_recall_k: int = 3
    recall_budget_tokens: int = 1500
    recall_max_stale_tokens: int = 4000
    # Anchor the staleness bound to the real window. A fixed 4000 tokens means
    # three different policies on three servers -- 2% of a 200K window (rebuild
    # constantly), 6% of 65K (the ~12 forced rebuilds measured on the opencode
    # run, 6 of which cost a prefix break), 50% of 8K (never rebuild). Every
    # other setpoint here is a fraction of n_ctx; this one was not.
    # 0.25 sits above the windowing band (context_budget_ratio -
    # context_target_ratio = 0.15), so the block refreshes at most once per
    # windowing cycle and tends to land ON a flush epoch, where the prefix is
    # already broken and the refresh rides for free.
    # 0 = use the fixed recall_max_stale_tokens (the pre-2026-08-03 behaviour).
    recall_max_stale_ratio: float = 0.25
    hotness_half_life_seconds: float = 86400.0
    ceiling_safety: float = 0.8
    max_conversations: int = 32
    loop_guard_enabled: bool = True
    loop_repeat_k: int = 3
    loop_timings_m: int = 3
    loop_accept_threshold: float = 0.99
    loop_draft_n_min: int = 200
    loop_cooldown_turns: int = 3
    loop_hard_stop: bool = False
    diag_enabled: bool = True
    diag_tokenize: bool = False
    diag_max_samples: int = 64
    wire_capture_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.handle_threshold_tokens <= 0:
            raise ValueError(
                f"handle_threshold_tokens must be > 0, got {self.handle_threshold_tokens}"
            )
        if self.stub_preview_chars < 0:
            raise ValueError(
                f"stub_preview_chars must be >= 0, got {self.stub_preview_chars}"
            )
        if self.rehydrate_budget_tokens < 0:
            raise ValueError(
                f"rehydrate_budget_tokens must be >= 0, got {self.rehydrate_budget_tokens}"
            )
        if not (1 <= self.listen_port <= 65535):
            raise ValueError(
                f"listen_port must be in 1..65535, got {self.listen_port}"
            )
        if not (0.0 <= self.diff_min_similarity <= 1.0):
            raise ValueError(
                f"diff_min_similarity must be in 0.0..1.0 (0 disables), got {self.diff_min_similarity}"
            )
        if self.diff_lookback < 0:
            raise ValueError(
                f"diff_lookback must be >= 0, got {self.diff_lookback}"
            )
        if self.diff_max_chars < 0:
            raise ValueError(
                f"diff_max_chars must be >= 0 (0 disables the cap), got {self.diff_max_chars}"
            )
        if self.tokenize_max_chars < 0:
            raise ValueError(
                f"tokenize_max_chars must be >= 0 (0 disables the cap), got "
                f"{self.tokenize_max_chars}"
            )
        if not (0.0 <= self.handle_threshold_ratio <= 1.0):
            raise ValueError(
                f"handle_threshold_ratio must be in 0.0..1.0 (0 = use fixed "
                f"handle_threshold_tokens), got {self.handle_threshold_ratio}"
            )
        if not (0.0 <= self.toolcall_threshold_ratio <= 1.0):
            raise ValueError(
                f"toolcall_threshold_ratio must be in 0.0..1.0 (0 = floor only), "
                f"got {self.toolcall_threshold_ratio}"
            )
        if self.toolcall_min_shrink_ratio < 0:
            raise ValueError(
                "toolcall_min_shrink_ratio must be >= 0 (0 = no floor; a value "
                "smaller than its own stub then bloats the wire), got "
                f"{self.toolcall_min_shrink_ratio}"
            )
        if not (0.0 <= self.recall_max_stale_ratio <= 1.0):
            raise ValueError(
                f"recall_max_stale_ratio must be in 0.0..1.0 (0 = use fixed "
                f"recall_max_stale_tokens), got {self.recall_max_stale_ratio}"
            )
        if not (0.0 <= self.context_budget_ratio <= 1.0):
            raise ValueError(
                f"context_budget_ratio must be in 0.0..1.0 (0 disables windowing), "
                f"got {self.context_budget_ratio}"
            )
        if not (0.0 <= self.context_target_ratio < 1.0):
            raise ValueError(
                f"context_target_ratio must be in 0.0..<1.0 (0 = legacy per-turn "
                f"windowing), got {self.context_target_ratio}"
            )
        if (self.context_target_ratio > 0.0 and self.context_budget_ratio > 0.0
                and self.context_target_ratio >= self.context_budget_ratio):
            raise ValueError(
                f"context_target_ratio ({self.context_target_ratio}) must be < "
                f"context_budget_ratio ({self.context_budget_ratio}) — the hysteresis "
                f"gap is what keeps the wire prefix stable between triggers"
            )
        if not (0.0 <= self.context_emergency_ratio <= 1.0):
            raise ValueError(
                f"context_emergency_ratio must be in 0.0..1.0 (0 disables the "
                f"emergency latch-override tier), got {self.context_emergency_ratio}"
            )
        if (self.context_emergency_ratio > 0.0 and self.context_budget_ratio > 0.0
                and self.context_emergency_ratio <= self.context_budget_ratio):
            raise ValueError(
                f"context_emergency_ratio ({self.context_emergency_ratio}) must be > "
                f"context_budget_ratio ({self.context_budget_ratio}) — it is the HIGH "
                f"water, above the existing MID-water trigger"
            )
        if self.window_min_shrink_ratio < 0:
            raise ValueError(
                "window_min_shrink_ratio must be >= 0 (0 = legacy break-even), "
                f"got {self.window_min_shrink_ratio}"
            )
        if self.protect_first_n < 0:
            raise ValueError(f"protect_first_n must be >= 0, got {self.protect_first_n}")
        if self.protect_last_n < 0:
            raise ValueError(f"protect_last_n must be >= 0, got {self.protect_last_n}")
        if self.diag_max_samples <= 0:
            raise ValueError(
                f"diag_max_samples must be > 0, got {self.diag_max_samples}"
            )
        if self.auto_recall_k < 0:
            raise ValueError(f"auto_recall_k must be >= 0 (0 disables), got {self.auto_recall_k}")
        if self.recall_budget_tokens < 0:
            raise ValueError(
                f"recall_budget_tokens must be >= 0 (0 disables), got {self.recall_budget_tokens}"
            )
        if self.recall_max_stale_tokens < 0:
            raise ValueError(
                f"recall_max_stale_tokens must be >= 0 (0 = legacy per-turn recompute), "
                f"got {self.recall_max_stale_tokens}"
            )
        if self.hotness_half_life_seconds <= 0:
            raise ValueError(
                f"hotness_half_life_seconds must be > 0, got {self.hotness_half_life_seconds}"
            )
        if not (0.0 <= self.ceiling_safety <= 1.0):
            raise ValueError(
                f"ceiling_safety must be in 0.0..1.0 (0 disables the learned ceiling), "
                f"got {self.ceiling_safety}"
            )
        if self.max_conversations < 1:
            raise ValueError(
                f"max_conversations must be >= 1, got {self.max_conversations}"
            )
        if self.loop_repeat_k < 2:
            raise ValueError(
                f"loop_repeat_k must be >= 2 (one occurrence is never a repeat), "
                f"got {self.loop_repeat_k}"
            )
        if self.loop_timings_m < 1:
            raise ValueError(f"loop_timings_m must be >= 1, got {self.loop_timings_m}")
        if not (0.0 < self.loop_accept_threshold <= 1.0):
            raise ValueError(
                f"loop_accept_threshold must be in (0.0, 1.0], got {self.loop_accept_threshold}"
            )
        if self.loop_draft_n_min < 0:
            raise ValueError(
                f"loop_draft_n_min must be >= 0, got {self.loop_draft_n_min}"
            )
        if self.loop_cooldown_turns < 0:
            raise ValueError(
                f"loop_cooldown_turns must be >= 0, got {self.loop_cooldown_turns}"
            )
