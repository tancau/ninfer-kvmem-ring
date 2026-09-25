#!/usr/bin/env python3
"""Long-prefill crash reproducer for the ring build.

Observed failure (4 occurrences, 2026-09-25):
    ninfer-serve.exe dies with 0xc0000409 in ucrtbase.dll at offset 0x7286e
    (a fail-fast, most likely abort() reached through std::terminate) during a
    ~65K-token prefill. Two of the four deaths share that exact signature; the
    other two left no Windows Error Reporting event at all. No stderr output.

Why a unique prefix matters
    A repeated identical prompt is answered from the shared-prefix cache in
    ~0.3 s and exercises no prefill. Measured on this build: 1 real prefill
    (253 s) + 7 cache hits (0.3 s each). Each request therefore prepends a
    unique marker so the full 64K prefill actually runs every iteration.

Usage
    python long_prefill_crash.py [iterations] [blocks] [max_tokens]

    blocks: 100 -> ~64.6K prompt tokens (calibrated against this model's
    tokenizer; the earlier 129,376-token probe used 200 blocks, i.e. ~647
    tokens per block).

What to look for
    * client:  "iter N FAILED: ..." followed by "ENGINE APPEARS DOWN"
    * server:  a line "NINFER-CRASH kind=... detail=..." in the engine's stderr
      (added by the crash handlers in apps/serve/main.cpp), plus exit code
      42 (std::terminate), 43 (CRT invalid parameter), or a WER LocalDump.
    * Windows: Application log Event ID 1000 for ninfer-serve.exe.

A healthy run is iterations/N == 1.0 with no NINFER-CRASH line.
"""

import json
import sys
import time
import urllib.error
import urllib.request

ENGINE = "http://127.0.0.1:8081"
MODEL = "bonsai2-27b"

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BLOCKS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 4

SENTENCE = "The riverbank at dawn is quiet except for the fox and the hounds. "
BLOCK = SENTENCE * 40 + "\n"
BODY = "".join("Paragraph %d. %s" % (i, BLOCK) for i in range(BLOCKS))

ok = 0
for i in range(1, N + 1):
    unique = "Session %d token %d. " % (i, 1000000 + i * 7919)
    content = unique + BODY + "\nAnswer with the single word: OK"
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": MAXTOK,
        "stream": False,
        "extra_body": {"enable_thinking": False},
    }).encode()
    request = urllib.request.Request(ENGINE + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            payload = json.loads(response.read().decode("utf-8"))
        usage = payload.get("usage", {})
        elapsed = time.time() - started
        print("[%s] iter %d OK prompt=%s out=%s %.1fs prefill~%.0f t/s"
              % (time.strftime("%H:%M:%S"), i, usage.get("prompt_tokens"),
                 usage.get("completion_tokens"), elapsed,
                 (usage.get("prompt_tokens") or 0) / max(0.1, elapsed)), flush=True)
        ok += 1
    except urllib.error.HTTPError as error:
        print("[%s] iter %d HTTP %d %s" % (time.strftime("%H:%M:%S"), i, error.code,
                                           error.read()[:200]), flush=True)
    except Exception as error:  # noqa: BLE001 - reproducer, report and stop
        print("[%s] iter %d FAILED: %s: %s" % (time.strftime("%H:%M:%S"), i,
                                               type(error).__name__, str(error)[:160]), flush=True)
        print("ENGINE APPEARS DOWN after %d successful iterations" % ok, flush=True)
        break

print("=== repro end: %d/%d ok ===" % (ok, N), flush=True)
