"""Diff consecutive wire captures (in vs in, out vs out, in vs out).

Approximates the templated prompt stream as:
    for each message: "<|im_start|>{role}\n" + content_text + "<|im_end|>\n"
so first-divergence char offsets can be compared against the server's LCP token
count (~4 chars/token). Run with the venv python from the repo root:
    .venv\\Scripts\\python raw\\analyze_wirecap.py <capture_dir>
"""
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from contextmanager.proxy.sensing import (  # noqa: E402
    canonical_content, conversation_key, harness_fingerprint, message_hash,
)

if len(sys.argv) < 2:
    sys.exit("usage: analyze_wirecap.py <capture_dir>  (the proxy's --wire-capture-dir)")
CAP = sys.argv[1]


def load(tag):
    out = []
    for p in sorted(glob.glob(os.path.join(CAP, f"req-*-{tag}.json"))):
        with open(p, encoding="utf-8") as fh:
            out.append((os.path.basename(p), json.load(fh)))
    return out


def msg_text(m):
    """Content as template-visible text: string, or concatenated part texts,
    plus tool_calls payload (assistant tool calls render into the stream)."""
    if not isinstance(m, dict):
        return repr(m)
    c = m.get("content")
    if isinstance(c, str):
        text = c
    elif isinstance(c, list):
        text = "".join(p.get("text", "") if isinstance(p, dict) else repr(p)
                       for p in c)
    else:
        text = ""
    tc = m.get("tool_calls")
    if tc is not None:
        text += canonical_content(tc)
    return text


def template_stream(messages):
    parts = []
    for m in messages:
        role = m.get("role", "?") if isinstance(m, dict) else "?"
        parts.append(f"<|im_start|>{role}\n{msg_text(m)}<|im_end|>\n")
    return "".join(parts)


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def window(s, i, w=160):
    lo = max(0, i - 40)
    return s[lo:i + w].replace("\n", "\\n")


def diff_pair(name, s1, s2):
    i = first_diff(s1, s2)
    if i == -1:
        print(f"  {name}: IDENTICAL ({len(s1)} chars)")
        return
    print(f"  {name}: first diff at char {i} (~token {i // 4}), "
          f"len {len(s1)} vs {len(s2)}")
    print(f"    A: ...{window(s1, i)}")
    print(f"    B: ...{window(s2, i)}")


def main():
    ins = load("in")
    outs = load("out")
    print(f"loaded {len(ins)} in, {len(outs)} out from {CAP}\n")

    # --- headers: stable vs varying ---
    print("== HEADERS (in) ==")
    if ins:
        names = sorted({k for _, r in ins for k in r.get("headers", {})})
        for n in names:
            vals = [r.get("headers", {}).get(n, "<absent>") for _, r in ins]
            stable = len(set(vals)) == 1
            show = vals[0] if stable else vals
            sv = str(show)
            print(f"  {'STABLE ' if stable else 'VARIES '}{n}: {sv[:160]}")

    # --- body top-level keys (non messages/tools) ---
    print("\n== BODY TOP-LEVEL (in) ==")
    for name, r in ins:
        body = r.get("body", {})
        keys = {k: v for k, v in body.items() if k not in ("messages", "tools")}
        small = {k: (v if isinstance(v, (str, int, float, bool, type(None)))
                     else f"<{type(v).__name__}>") for k, v in keys.items()}
        print(f"  {name}: {json.dumps(small, default=str)[:400]}")

    # --- conversation identity ---
    print("\n== IDENTITY (in) ==")
    for name, r in ins:
        msgs = r.get("body", {}).get("messages", [])
        print(f"  {name}: key={conversation_key(msgs)} "
              f"fp={harness_fingerprint(msgs)} n_msgs={len(msgs)}")

    # --- message-level divergence (incoming) ---
    print("\n== MESSAGE-LEVEL DIFF (in, consecutive) ==")
    for (n1, r1), (n2, r2) in zip(ins, ins[1:]):
        m1 = r1["body"]["messages"]
        m2 = r2["body"]["messages"]
        h1 = [message_hash(m) for m in m1]
        h2 = [message_hash(m) for m in m2]
        i = next((k for k, (a, b) in enumerate(zip(h1, h2)) if a != b),
                 min(len(h1), len(h2)))
        print(f"  {n1} -> {n2}: {len(m1)} vs {len(m2)} msgs, "
              f"first differing message index = {i}")
        if i < min(len(m1), len(m2)):
            t1, t2 = msg_text(m1[i]), msg_text(m2[i])
            ci = first_diff(t1, t2)
            print(f"    role={m1[i].get('role')} text lens {len(t1)} vs {len(t2)},"
                  f" char diff at {ci}")
            print(f"    A: ...{window(t1, ci)}")
            print(f"    B: ...{window(t2, ci)}")

    # --- template-stream divergence, in vs in and out vs out ---
    print("\n== TEMPLATE-STREAM DIFF (approx) ==")
    for tag, rows in (("IN ", ins), ("OUT", outs)):
        streams = [(n, template_stream(r.get("body", r.get("payload", {}))
                                      .get("messages", [])))
                   for n, r in rows]
        for (n1, s1), (n2, s2) in zip(streams, streams[1:]):
            print(f" {tag} {n1} -> {n2}:")
            diff_pair("stream", s1, s2)

    # --- in vs out of the SAME request: what did the proxy change? ---
    print("\n== PROXY MUTATION (in vs out, same request) ==")
    out_by_seq = {n.split("-")[1]: r for n, r in outs}
    for n, r in ins:
        seq = n.split("-")[1]
        o = out_by_seq.get(seq)
        if o is None:
            continue
        mi = r["body"]["messages"]
        mo = o["payload"]["messages"]
        si, so = template_stream(mi), template_stream(mo)
        i = first_diff(si, so)
        print(f"  req-{seq}: {len(mi)} -> {len(mo)} msgs, "
              f"chars {len(si)} -> {len(so)}", end="")
        if i == -1:
            print("  IDENTICAL")
        else:
            print(f"  first diff at char {i} (~token {i // 4})")
            print(f"    IN : ...{window(si, i)}")
            print(f"    OUT: ...{window(so, i)}")


if __name__ == "__main__":
    main()
