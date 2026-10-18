# KV capacity: the ledger, the ActKV mapping, and what 64 GB of RAM buys (2026-10-18)

Desk study; no GPU spent. Sources: live `server_start` records from the production
log, the oneclick launcher, `工作汇总-2026-10-14.md`, and the public KVMem docs.

## 1. The ledger (measured)

- **Device KV: ≈35,905 B per token** (int8, Qwen3.8-27B, includes MTP KV and
  scales). Production: `kv_payload_bytes` 2,647,265,280 ÷ 73,728 tokens
  (budget 57,344 + reserve 16,384) = 35,905 B. Swift iq2xs artifact:
  2,206,076,928 ÷ 61,440 = 35,906 B. The two independent artifacts agree.
- **Host pool:** `--kvmem-host-mib 12288` (12 GiB). The oneclick launcher scales
  it by system RAM: ≥30 GB → 12,288; ≥22 GB → 10,240; ≥14 GB → 6,144; else 4,096
  (`oneclick/setup.ps1`). The 34 GB machine runs the top tier.
- **One full 262,144-token session ≈ 8.8 GiB of host KV** (262,144 × 35,905 B =
  9.41 GB) plus a 147 MiB state image → the 12 GiB pool holds **exactly one**
  258K session. Confirmed empirically on 2026-10-11: with `--kvmem-sessions 4`
  the second session was pushed out (overflow write 1.60 GB; returning to it cost
  one SSD read, 4.3 s against 71.2 s cold compute, effective ~430 MB/s,
  `工作汇总-2026-10-14.md` §二).
- **Whole process, full load at 100K: ~10 GB physical, ~19 GB commit peak**
  (oneclick README, measured on a 32 GB machine). The commit number, not the
  physical number, is what pressures 16–24 GB systems.
- Production today keeps `--kvmem-sessions 1` and no disk tier: capacity beyond
  the pool is not retained, it is re-prefilled. The SSD tier
  (`-DiskPath`/`-DiskMiB`) is a documented option we validated but did not ship.

## 2. ActKV mapping (P6 Phase 1 deliverable)

ActKV (arXiv 2609.31395) is lossy in-place eviction with three parts: (i)
action-region scoring plus attention-aware LRFU, (ii) a confidence-driven
adaptive budget, (iii) page-aware compression at ORA (observation-reasoning-action)
boundaries. Reported: 98.53% of FullKV accuracy at 25.98% of peak KV memory.

Mapping against what the KVMem production engine exposes:

| ActKV part | Counterpart in the stack | Verdict |
|---|---|---|
| Action-region scoring + attention-aware LRFU | KVMem retrieval scores (mean-K key similarity) are internal and opaque; our own LRFU attribution arm measured **~0** (prefill 285.3 vs 288.9, decode 41.02 vs 42.92, within noise) | No reachable signal; would be upstream work |
| Confidence-driven adaptive budget | KVMem's H budget is static config; RAM tiers pick it, nothing adjusts it at runtime | Upstream work; also our budget is not the binding constraint today |
| Page-aware compression (the part that frees RAM) | **No counterpart** — KVMem moves pages at full fidelity and never compresses | The only genuinely new mechanism; upstream-sized project |

Reachability: production consumes KVMem as a released binary (public sources and
patch, but no released build path for us). Same constraint as P1 — a fork-side
implementation would demonstrate the mechanism but could never ship to
production.

Size at stake at our shape: ~25% of the retained host KV ≈ **2.2–2.4 GiB** of the
8.8 GiB a 258K session occupies — at the cost of the stack's first lossy trade
(everything today is lossless: recall gates and greedy equivalence). Verdict:
**stays parked**. It reopens only if host RAM is still the binding limit after a
64 GB upgrade, or if upstream adopts ActKV-style compression. Phase 2 gate remains
hole-region recall 3/3 plus early markers at the 258K shape.

## 3. The 64 GB evaluation

Current machine: MS-Terminator B550M, **2 DIMM slots**, 2×16 GB DDR4-2667
(Aigo `UCB416B C3200/16G`), 34.3 GB usable. A 64 GB upgrade means a 2×32 GB kit
(replacing the existing sticks, not adding).

What 64 GB buys:

1. **Host pool: 12 → 24 GiB** (manual `--kvmem-host-mib`, or the launcher's
   `-HostMiB`; the auto tier table tops out at 12 GiB, so this is a launch-script
   change on our side). 24 GiB holds **2–3 resident 258K sessions** instead of
   one — precisely the observed pain: two long-lived prefixes (coding agent plus a
   second client) currently contend for a single hot slot, and the loser pays a
   cold or SSD restore.
2. **Commit headroom.** Full-load commit already reaches ~19 GB at 100K. Two
   long sessions on a 32 GB machine go straight into the pagefile; 64 GB keeps
   the whole stack physical.
3. **Room to keep experimenting** with weight-offload tiers (Strata-style; the
   author's own floor is 32 GB, comfort 64) without evicting KV to make room.
4. The SSD tier stays useful for cold/archived sessions, but its 430 MB/s
   effective restore and ~5% first-touch cost make it a fallback, not a substitute
   for live capacity.

What it does **not** buy: decode speed (per-round time is weight-bandwidth and
device-bound; RAM size is not on that path — see the Stage 0 report), the
262,144-token artifact ceiling, or prefill speed.

Purchase notes: single kit, matched pair, DDR4-3200 CL16-class is enough — the
board runs JEDEC 2667 today and this workload is not latency-sensitive. Two slots
means the old pair comes out; no mixing questions. Roughly the cheapest remaining
capacity lever on the table (vs ActKV = upstream code, vs SSD = latency).
