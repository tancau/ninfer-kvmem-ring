# Serving stack: engine + SSE keepalive proxy

Recorded 2026-10-13. Two things were learned while putting the RC3 engine in front of a
real client for a long session: the client can break the prefix (every turn then pays a
full re-prefill), and even when it does not, a cold prefill can outlast the client's
stream idle timeout. The second one has a fix that belongs to neither the client nor the
engine, so it lives in front of both.

## Symptom

```
重试延迟：462毫秒
失败原因：pi-ai stream idle timeout after 300000ms
```

The client kills a stream after 5 minutes with no bytes, then retries.

## What the engine actually did

From the request log, the turn before the failure:

```
 msgs  prompt   ttft    cache_hit  finish
 112   133152     1.2s   133095    stop_token     reuse fine
 114   134156    93.2s   133235    stop_token     reuse, but slow
 116   135597     0.2s        0    cancelled      NO reuse -> full rebuild
```

`msgs=116` had `cache_hit = 0`: the client's prefix changed (most likely its own context
compaction), so the engine rebuilt from scratch. That path is not just a prefill: a cold
60 K-token request through the proxy took **311 s** end to end (193 tok/s) where a clean
cold prefill measures 809 tok/s, i.e. the un-cached path costs ~4x more than prefill
alone. Past 300 s of silence, the client gives up.

So: an occasional prefix break is a slow turn, and a slow turn is a dead stream.

## Why not fix the client's timeout

Its effective configuration is not in `~/.dsh/settings.yaml` (which does carry
`streamIdleTimeoutMs: 3600000` and is what was edited) nor in
`dsh-desktop/harness/settings.yaml.imported` (no such key). The app resolves providers
from its own store -- visible in that the file says `baseURL: http://localhost:8080/v1`
while the live traffic goes to 8081. Editing the wrong file silently does nothing, which
is exactly the failure mode this project keeps running into.

## The fix that works regardless of client

`keepalive_proxy.py` sits on the port clients use and forwards to the engine. While the
engine is silent it writes an SSE comment line:

```
: keepalive
```

Comments are part of the SSE specification and conforming parsers must ignore them, so
the model output is untouched and no client needs changing.

Framing detail that matters: the engine answers with
`Content-Type: text/event-stream` **and `Transfer-Encoding: chunked`**. Injecting raw
bytes into a chunked body corrupts it, so the proxy de-chunks the response and delimits
it with `Connection: close` instead. Verified both directions after the change:

```
non-stream : content = 'NONSTREAM-OK'
stream     : 4 chunks, content = 'STREAM-OK'
```

Measured effect (60 K-token cold request, 5 s keepalive):

```
total 311.2 s of engine silence
client-visible gaps: all 5.0 s
MAX GAP 5.0 s   -> a 300 s idle timeout can never fire
```

## Layout

```
client -> :8081 keepalive_proxy.py -> :8082 infer-kvmem-server.exe
```

```
start : powershell -File start_stack.ps1
status: powershell -File start_stack.ps1 -Status
stop  : powershell -File rc3_stop.ps1
rollback: C:\Bonsai-App\Start-Bonsai-WebUI-256K.bat   (franken, untouched)
```

## Not a DSH-only problem

The same client also produced `cache_hit = 0` on a normal turn (the table above), so the
keepalive proxy is not papering over a single incident: any client that occasionally
breaks its prefix will occasionally need a multi-minute prefill, and this makes that
survivable instead of fatal.

## Still open

- The un-cached path costing ~4x a clean prefill is the same host-side rebuild found in
  `live-serving-stall-2026-10-13.md`. It is the real cost driver and it is not fixed.
- A client-side fix (append-only history, and a longer idle allowance) remains the right
  answer; this proxy only removes the fatality.
