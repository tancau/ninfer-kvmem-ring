"""Abort a long prefill mid-flight and see whether the engine survives.

Three of the four observed engine deaths followed a request that the engine
logged as `cancelled` (two of them immediately after a ~65K prefill started).
This reproducer connects at the socket level, sends a ~64K-token chat request,
holds the connection for a while (mid-prefill; a full prefill takes ~260 s),
then drops it - optionally with SO_LINGER 0 so the close is an RST rather than
a FIN. After each abort it waits and probes the engine with a short request.

Usage:
    python abort_prefill_crash.py [iterations] [hold_seconds] [mode] [blocks]
      mode: close (FIN) | rst (RST via SO_LINGER 0) | pair (two concurrent)

A healthy run ends with every probe OK and no NINFER-CRASH line in the engine
stderr.
"""

import json
import socket
import struct
import sys
import time
import urllib.request

HOST, PORT = "127.0.0.1", 8081
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
HOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
MODE = sys.argv[3] if len(sys.argv) > 3 else "close"
BLOCKS = int(sys.argv[4]) if len(sys.argv) > 4 else 100

SENTENCE = "The riverbank at dawn is quiet except for the fox and the hounds. "
BLOCK = SENTENCE * 40 + "\n"
BODY_TEXT = "".join("Paragraph %d. %s" % (i, BLOCK) for i in range(BLOCKS))


def body_for(tag):
    unique = "Session %s. " % tag
    content = unique + BODY_TEXT + "\nAnswer with the single word: OK"
    return json.dumps({
        "model": "bonsai2-27b",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4,
        "stream": False,
        "extra_body": {"enable_thinking": False},
    }).encode()


def open_request(tag, rst=False):
    sock = socket.create_connection((HOST, PORT), timeout=10)
    if rst:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    payload = body_for(tag)
    head = (b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1:8081\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n"
            b"Connection: close\r\n\r\n")
    sock.sendall(head + payload)
    return sock


def probe():
    """Short request; returns True when the engine is still serving."""
    try:
        payload = json.dumps({
            "model": "bonsai2-27b",
            "messages": [{"role": "user", "content": "1+1?"}],
            "max_tokens": 4, "stream": False,
            "extra_body": {"enable_thinking": False},
        }).encode()
        request = urllib.request.Request("http://%s:%d/v1/chat/completions" % (HOST, PORT),
                                         data=payload,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            response.read()
        return True
    except Exception as error:  # noqa: BLE001
        print("   probe FAILED: %s: %s" % (type(error).__name__, str(error)[:120]), flush=True)
        return False


print("=== abort repro: N=%d hold=%.0fs mode=%s blocks=%d ===" % (N, HOLD, MODE, BLOCKS),
      flush=True)
alive = True
for i in range(1, N + 1):
    tag = "%d-%d" % (i, int(time.time()))
    try:
        if MODE == "pair":
            a = open_request(tag + "-a")
            b = open_request(tag + "-b", rst=True)
            print("[%s] iter %d: two concurrent 64K requests, holding %.0fs"
                  % (time.strftime("%H:%M:%S"), i, HOLD), flush=True)
            time.sleep(HOLD)
            a.close()
            b.close()
        else:
            sock = open_request(tag, rst=(MODE == "rst"))
            print("[%s] iter %d: 64K request sent, holding %.0fs then %s"
                  % (time.strftime("%H:%M:%S"), i, HOLD, MODE), flush=True)
            time.sleep(HOLD)
            sock.close()
    except Exception as error:  # noqa: BLE001
        print("[%s] iter %d: send failed: %s" % (time.strftime("%H:%M:%S"), i, error),
              flush=True)
        alive = False
        break

    # give the engine time to observe the disconnect and unwind the request
    time.sleep(20)
    if not probe():
        print("=== ENGINE APPEARS DOWN after iter %d ===" % i, flush=True)
        alive = False
        break
    print("   probe OK", flush=True)

print("=== abort repro end: engine_alive=%s ===" % alive, flush=True)
