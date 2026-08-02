import json
import os
import re
import sys

if len(sys.argv) < 2:
    sys.exit("usage: inspect_msg0.py <capture_dir>  (the proxy's --wire-capture-dir)")
CAP = sys.argv[1]

body = json.load(open(os.path.join(CAP, "req-0001-in.json"), encoding="utf-8"))["body"]
m0 = body["messages"][0]
print("role:", m0.get("role"))
c = m0.get("content")
print("content type:", type(c).__name__)
if isinstance(c, str):
    print("len:", len(c))
    print("first 160:", repr(c[:160]))
    text = c
else:
    print("parts:", len(c))
    for i, part in enumerate(c[:4]):
        t = part.get("text", "") if isinstance(part, dict) else ""
        print(f"  part{i} type={part.get('type')} len={len(t)} head={t[:100]!r}")
    text = json.dumps(c, ensure_ascii=False)
for pat in (r"[Tt]oday[^\n]{0,60}", r"20\d\d-\d\d-\d\d", r"currentDate[^\n]{0,60}"):
    print(pat, "->", re.findall(pat, text)[:3])
