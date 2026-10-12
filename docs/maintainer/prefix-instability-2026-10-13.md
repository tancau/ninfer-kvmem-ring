# Why every live turn re-prefilled the whole conversation

Recorded 2026-10-13. Finding: the client's prompt is **not append-only**, so the engine's
prefix-identity check fails on every turn and each turn pays a full re-prefill.
This is the cause of the 194-218 s TTFT the user saw in normal use, and it is the same
phenomenon as the "90-second GPU-idle windows" in
`live-serving-stall-2026-10-13.md` (the engine rebuilds, then prefills).

## Evidence 1 — the engine says it never reused anything

Same engine instance, requests from the real client versus from our own harness:

```
 rid  msgs  prompt   cache_hit  reuse_path            ttft
 --- client (ZCode) ---
 1    55    58039    0          root                  194.3
 2    57    59674    0          root                  206.4
 3    59    60239    0          root                  213.1
 4    61    60840    0          root                  212.7
 5    63    61156    0          root                  214.5
 6    65    61727    0          root                  218.3
 --- our harness, byte-identical prefix each turn ---
 8    87    57763    0          root                  166.3   (cold)
 9    88    57784    57756      private_turn_closure    1.2
10    90    57848    57818      private_endpoint        1.1
```

`message_count` grows cleanly (+2 per turn) for the client, so the conversation *is*
being continued. But `cache_hit = 0` and `reuse_path = root`: nothing is inherited.

## Evidence 2 — the request bodies differ in the middle, not at the tail

A logging reverse proxy (`capture_proxy.py`, listening on 8081 and forwarding to the
engine on 8082) captured two consecutive request bodies from the real client. Diffing
them (`body_diff.py`):

```
common prefix: 22,002 chars
first difference at char 22,002

A: ...</system-reminder>"}]},{"role":"user","content":[{"type":"input_text",
   "text":"<system-reminder>\nPlan mode is active. ...

B: ...</system-reminder>"}]},{"role":"user","content":[{"type":"input_text",
   "text":"Reply with exactly: CAP-A"}]},
   {"role":"assistant","content":[{"type":"output_text","text":"CAP-A"}]},
   {"role":"user","content":[{"type":"input_text",
   "text":"<system-reminder>\nPlan mode is active. ...
```

The turn's new messages are **inserted before the per-turn system-reminder block**
rather than appended after the previous turn's content. So the reminder's position in
the history moves every turn, and everything from that point on differs from the
previous request's bytes. A prefix cache can only reuse up to that point, and this
engine's session inheritance is keyed on prefix identity, so it reuses nothing.

Reproduced independently: inserting a changing timestamp into the system prompt of our
own harness produces exactly the same signature (`cache_hit 0`, `reuse_path root`,
turn 1 wall time 88 s instead of 1.6 s).

## Consequences

- Every turn costs a full prefill of the whole conversation: ~58-62 K tokens, 194-218 s
  for the client, measured at ~700-720 tok/s for a cold 49 K-token prefill.
- The engine's own `timings_seconds.prefill` reports **0.00** for these requests (the
  time is attributed to `device_wait_exposed_seconds` instead), so any diagnosis that
  trusts that field will draw the wrong conclusion. This is an engine-side accounting
  gap worth reporting to upstream in its own right.
- `history_hits` stays 0 and `history_misses` climbs: the retained-session machinery
  never engages.

## What can be done

Client side (the real fix, not ours to make): keep the conversation strictly
append-only. Put fixed blocks like the per-turn system-reminder at a stable position
(after the system prompt, before the conversation), or omit the redundant copy instead
of re-inserting it each turn.

Engine side (mitigation only): make the forced full re-prefill as fast as possible.
That is what `prefill_ab.ps1` measures — prefill-chunk size and
`--fast-prefill-kernel`, on a cold 49 K-token prompt, since reuse can never help.

## Harness added

- `capture_proxy.py` (+ `--diff`), `capture_diff.py`, `body_diff.py` — capture request
  bodies and locate the first differing byte.
- `wl.py` — append-only versus unstable-prefix workload (`UNSTABLE=1`).
- `prefix_probe.py` — per-request cache_hit / reuse_path from the engine log.
- `one_prefill.py`, `prefill_ab.ps1` — cold-prefill throughput A/B.

## Mitigation measured (prefill A/B)

Since reuse can never help this client, the only engine-side lever is making the forced
full prefill faster. Cold 48,900-token prefills, two runs each, fresh nonce per run so
the prefix cache can never contribute (`prefill_ab.ps1`, `one_prefill.py`):

| arm | run 1 | run 2 | vs baseline |
|---|---|---|---|
| `--prefill-chunk` 256 (upstream default) | 722.5 tok/s | 720.3 | baseline |
| `--prefill-chunk` 1024 (our franken used this) | 717.9 | 707.8 | -1% |
| `--fast-prefill-kernel` + chunk 1024 | 811.9 | 799.0 | **+12.5%** |
| `--fast-prefill-kernel` + chunk 2048 | 806.8 | 810.0 | +12.4% |

Conclusions:

- **`--fast-prefill-kernel` is worth taking: ~+12.5%**, and chunk size is irrelevant
  (256/1024/2048 identical, matching what the ring work found for 512/1024/4096).
- Prefill is at the hardware ceiling: GPU utilisation 100%, power 168-170 W of a 170 W
  cap during the measurement. There is no large win to be had by tuning; the large win
  is not having to prefill (client-side fix, ~180x).
- Production config adopted: upstream defaults + `--fast-prefill-kernel`, LRFU off,
  `--max-context 262144`, port 8081, request log on.

## How the numbers in this file were obtained

Every figure here comes from the engine's own records or from a measured run; nothing is
illustrative. Reproduction order:

1. `prefix_probe.py` — per-request `cache_hit` / `reuse_path` from the request log,
   scoped to one `server_instance_id`.
2. `capture_proxy.py` (8081 -> engine on 8082) — dump request bodies for two consecutive
   turns; `body_diff.py` locates the first differing byte.
3. `wl.py` with `UNSTABLE=1` — independent reproduction without the client.
4. `one_prefill.py` + `prefill_ab.ps1` — cold-prefill throughput per configuration.

## The lesson, again

The request log is shared by every engine that has run on this port. An unscoped read
reported 69 simultaneous requests and queue waits of 588 s — all of it old franken test
traffic. Every analysis in this file filters on `server_instance_id`; that filter is
what separated "the client's prefix moves" from "the engine is slow".


