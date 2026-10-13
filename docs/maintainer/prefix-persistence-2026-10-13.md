# Prefix persistence across an engine restart (plan + pre-registered predictions)

Written before measurement. The claim under test: a long-context session can survive
an engine restart, converting the ~772 s cold prefill we measured at 258K into
something on the order of a disk read.

## What prompted it

`/props` on the production engine reports three zeros that are actually switches:

```
disk_budget_bytes : 0        ssd_enabled : false       disk_restore : false
```

The worker help has two separate families:

```
--kvmem-disk-path PATH     Ephemeral active/idle KV spill directory
--kvmem-disk-mib N         Shared SSD payload quota D, MiB
--disk-kv-path DIR         disk tier: evicted continuations write their KV and
                           states there, per artifact and profile, and survive restarts
--disk-kv-gib N            disk tier budget (default 64)
--disk-kv-restore          seed a new request's matching prefix from the disk tier
--slot-save-path DIR       enable POST /slots/{id}?action=save|restore|erase, which
                           writes a retained session to a file in DIR or restores
                           one from it
--auto-save-evicted        write a retained session back to its slot file before an
                           involuntary eviction destroys it (needs --slot-save-path)
```

## Two mechanisms, and they are not the same

| | `--kvmem-disk-path` | `--disk-kv-*` | `--slot-save-path` |
|---|---|---|---|
| belongs to | the KVMem block (our production path) | the other context-cache block | context cache, but HTTP-driven |
| accepted with our recipe | **yes** (verified: engine starts) | **no** (verified: exit 1, generic KVMem rejection) | to be tested |
| survives a restart | `ssd_persistence: process` says no | help says yes | yes, if the file is written |

So the autonomous SSD tier we can actually turn on is *process-local*. The thing that
claims "survive restarts" is in the block our recipe is rejected from. That leaves
`--slot-save-path`, which is explicit and API-driven, as the only path to persistence
we control.

## Pre-registered predictions

1. **The engine starts with `--kvmem-disk-path` + `--kvmem-disk-mib`.** VERIFIED
   (`ssd_enabled: true`, `disk_budget_bytes: 68719476736`, `retained_payload_budget_bytes`
   79.7 GB). `ssd_persistence` reads `process`.
2. **`--disk-kv-path` combined with the KVMem recipe is rejected.** VERIFIED — the
   engine exits 1 with the same generic KVMem rejection we saw for
   `--mtp-attention-window` and `--vision-residency overlay`. This is now the fourth
   instance of "help lists it, the KVMem runtime refuses it".
3. **`--slot-save-path` + `--auto-save-evicted` starts with the KVMem recipe.** To be
   tested. If it is refused too, the persistence idea is dead on this build and the
   write-up is the deliverable.
4. **`POST /slots/0?action=save` writes a file** under the slot directory, sized roughly
   like the KV payload for the context (hundreds of MB at 16K, several GB at 258K).
5. **After a hard kill and a restart, `action=restore` makes the identical prefix fast**:
   seconds, against ~772 s cold at 258K, with recall intact. If it restores the KV but
   *not* the GDN state, the answer may be *fast but wrong*, which is the failure mode
   worth looking for hardest.
6. **The honest caveat I expect to survive:** slot save/restore is a client action. It
   is not transparent the way in-process prefix reuse is. So even if it works, the
   deliverable is "restart costs one save call plus a disk read", not "restarts stopped
   mattering".

## Gate

- restores at all (a file is written, restore returns OK)
- the resend after restore answers in seconds, not hundreds
- recall intact on the resend, checked against the same markers the acceptance matrix
  uses (not just "it returned fast")
- a clean production restore afterwards

## Method

`slot_cycle.ps1`: start (slot-save-path + auto-save-evicted + KVMem SSD tier) -> cold
prefix -> decoy prefix (forces the cold one to become evictable) -> save -> **hard kill
(Stop-Process, no graceful shutdown)** -> restart with the same flags -> restore ->
resend the identical prefix -> slot status. Then production is restored via
`start_stack.ps1`.

Mechanism is validated at 16K first (cheap) before the 258K headline number, on the
principle that spending 13 minutes to learn the API does not work is worse than
spending 2.

## Result: the flag matrix (measured 2026-10-13, 23:15)

Seven combinations, run directly against the worker on a scratch port with the exact
production argv. Engine rejects in ~2 s, so each variant is cheap.

| variant | outcome |
|---|---|
| `--kvmem-disk-path` + `--kvmem-disk-mib 65536` | **STARTS** — `ssd_enabled: true`, `disk_budget_bytes: 68719476736` |
| `--kvmem-sessions 4` | **STARTS** — `retained_sessions: 4` |
| `--slot-save-path DIR` | REJECTED (generic KVMem refusal) |
| `--slot-save-path DIR --auto-save-evicted` | REJECTED |
| `--kvmem-disk-path` + `--slot-save-path` + auto-save | REJECTED |
| `--disk-kv-path` + `--disk-kv-gib` + `--disk-kv-restore` | REJECTED |

So: **every persistence mechanism that claims "survive restarts" is refused when
combined with the KVMem recipe** — `--slot-save-path`, `--auto-save-evicted` and the
whole `--disk-kv-*` family. This is the fifth instance of "the help lists it, the KVMem
runtime refuses it" (after `--mtp-attention-window`, `--vision-residency overlay`, and
two on-device profiles). Predictions 1-3 answered.

What survives is the KVMem family's own SSD tier, and its `ssd_persistence` reads
`process`: it does **not** cross a restart. So the original goal — a restart costing a
disk read instead of 772 s — is **not reachable on this build**.

## But the SSD tier does something else, and it is the useful thing

It lets **retained sessions exceed the pinned host budget**, spilling idle ones to SSD
and restoring on demand. First measured at 48K with the host budget deliberately
crushed to 2 GiB so spill is forced cheaply (instead of paying 2 x 772 s to learn
whether the mechanism works):

```
host_mib 2048, sessions 4, target 48K

A cold      71.2 s   prefill 746
B cold      75.0 s   prefill 699        <-- B's arrival pushes A out of the 2 GiB budget
   after B: disk_writes 698, disk_written_bytes 1.60 GB
A again      4.3 s   prefill 0.4        <-- A came back from SSD
   after:   disk_hits 815, disk_read_bytes 1.87 GB,
            history_restored_bytes 1.87 GB

16.6x for the revisit, and the spill cost ~4 s (75.0 vs 71.2) i.e. ~5% on the
prefill that caused it. Effective restore rate ~430 MB/s.
```

## What the claim becomes, stated honestly

Not "a restart is cheap" — that is refuted. The correct claim is:

> With `--kvmem-sessions N` plus the SSD tier, more long prefixes stay warm than the
> pinned host budget can hold. Returning to a spilled prefix costs an SSD read
> instead of a cold prefill. On this box the host budget holds **one** 258K session
> (~7.8 GiB of a 12 GiB budget); the SSD tier is what allows the second and later
> ones to stay reachable.

That maps directly onto how this machine is actually used: two different long
prefixes (a coding agent and a second client) currently fight over one warm slot.

## Second measurement: production-scale

The same test at 258K with the production host budget (12288 MiB) and
`--kvmem-sessions 4`: A cold, B cold (forcing A to spill), then A again. Expected from
the 48K numbers: restore ~7.8 GiB at ~430 MB/s ≈ 18 s against 772 s cold ≈ **42x**.
In flight at the time of writing; results appended below when it lands.

## Prediction 6, revisited

I predicted the caveat would be "this is a client action, not transparent". Wrong:
with the SSD tier there is no client action at all — it is automatic. The honest
caveat is the opposite one: the capacity it buys is **host RAM**, not context window,
and the context ceiling is still the artifact's 262144 with RoPE scaling off.

## Scope note

Even a full success does not raise the context ceiling: the artifact's window is
262144 and RoPE scaling stays off. This is about *time to first token after a
restart*, not about fitting more context.
