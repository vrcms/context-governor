"""Verify: after masking the cch nonce, are outgoing wires pure prefix-extensions?
Also: locate the proxy-injected (+1) message, check tools-array stability,
and scan for other date/version stamps in the system prompt."""
import glob
import json
import os
import re
import sys

if len(sys.argv) < 2:
    sys.exit("usage: verify_prefix.py <capture_dir>  (the proxy's --wire-capture-dir)")
CAP = sys.argv[1]

outs = []
for p in sorted(glob.glob(os.path.join(CAP, "req-*-out.json"))):
    with open(p, encoding="utf-8") as fh:
        outs.append((os.path.basename(p), json.load(fh)["payload"]))

NONCE = re.compile(r"cch=[0-9a-f]+")


def canon(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def masked_stream(payload):
    parts = []
    for m in payload["messages"]:
        role = m.get("role", "?")
        c = m.get("content")
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            text = "".join(x.get("text", "") if isinstance(x, dict) else repr(x)
                           for x in c)
        else:
            text = ""
        tc = m.get("tool_calls")
        if tc is not None:
            text += canon(tc)
        parts.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
    return NONCE.sub("cch=", "".join(parts))


print("== 1) OUT streams masked for cch: prefix-extension check ==")
streams = [(n, masked_stream(p)) for n, p in outs]
ok = True
for (n1, s1), (n2, s2) in zip(streams, streams[1:]):
    is_prefix = s2.startswith(s1)
    print(f"  {n1} prefix-of {n2}: {is_prefix} (len {len(s1)} -> {len(s2)})")
    ok = ok and is_prefix
print(f"  => ALL pure prefix extensions: {ok}")

print("\n== 2) tools array stability across turns ==")
tc = [canon(p.get("tools")) for _, p in outs]
base = tc[0]
for i, t in enumerate(tc[1:], 2):
    print(f"  req-0001 tools == req-{i:04d} tools: {base == t} (len {len(t)})")

print("\n== 3) the +1 injected message (position, role, stability) ==")
for name, p in outs:
    msgs = p["messages"]
    for i, m in enumerate(msgs):
        c = m.get("content")
        txt = c if isinstance(c, str) else ""
        if "cm:stored" in txt or "recall" in txt.lower()[:200] or m.get("role") == "system" and i > 0:
            head = txt[:120].replace("\n", "\\n")
            print(f"  {name} msg[{i}] role={m.get('role')} len={len(txt)}: {head}")

print("\n== 4) volatile-stamp scan in system message (incoming) ==")
ins = []
for p in sorted(glob.glob(os.path.join(CAP, "req-*-in.json"))):
    with open(p, encoding="utf-8") as fh:
        ins.append((os.path.basename(p), json.load(fh)["body"]))
sys0 = ins[0][1]["messages"][0]["content"]
if not isinstance(sys0, str):
    sys0 = canon(sys0)
for pat, label in [
    (r"cc_version=[^;\s]*", "cc_version"),
    (r"cc_entrypoint=[^;\s]*", "cc_entrypoint"),
    (r"cch=[0-9a-f]*", "cch"),
    (r"Today's date[^\n]*", "today"),
    (r"20\d\d[-/]\d\d[-/]\d\d", "iso-date"),
    (r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]* (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d+", "long-date"),
]:
    hits = NONCE.findall(pat, sys0) if pat == r"cch=[0-9a-f]*" else re.findall(pat, sys0)
    print(f"  {label}: {hits[:4]}{' ...' if len(hits) > 4 else ''} (n={len(hits)})")

print("\n== 5) full system-message masked equality (all turns) ==")
masked = [NONCE.sub("cch=", (m0 if isinstance((m0 := r[1]["messages"][0]["content"]), str) else canon(m0))) for r in ins]
all_same = all(m == masked[0] for m in masked)
print(f"  all {len(masked)} system messages identical after cch mask: {all_same}")
if not all_same:
    for i, m in enumerate(masked[1:], 2):
        if m != masked[0]:
            j = next(k for k, (a, b) in enumerate(zip(masked[0], m)) if a != b)
            print(f"  req-0001 vs req-{i:04d} differ at char {j}")
            print(f"    A: ...{masked[0][max(0,j-40):j+120]!r}")
            print(f"    B: ...{m[max(0,j-40):j+120]!r}")
