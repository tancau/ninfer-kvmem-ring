# Production launcher migration to RC3 — plan (not executed)

Prepared 2026-10-13. **No production file has been modified.** The current
production launcher (`C:\Bonsai-App\Start-Bonsai-WebUI-256K.bat`) is untouched.

## Current arrangement (decided 2026-10-13, in effect)

Decision: run RC3 on 8081 for real use, and do **not** touch production files.

```
start   powershell -NoProfile -ExecutionPolicy Bypass -File <temp>\rc3_prod.ps1
        (launcher exits once the engine is serving; the worker is its own process)
stop    Stop-Process -Name ninfer-kvmem-server -Force
probe   Invoke-RestMethod http://127.0.0.1:8081/props      # full kvmem telemetry
        Invoke-RestMethod http://127.0.0.1:8081/v1/models
logs    <temp>\rc3-prod.log / .err         engine stdout/stderr
        C:\Bonsai-App\logs\requests-8081.jsonl   per-request forensics (rotation 512 MiB x4)
rollback C:\Bonsai-App\Start-Bonsai-WebUI-256K.bat   # franken, unchanged, one command
```

Serving configuration is the upstream default recipe with exactly two deviations:
`--max-context 262144` (upstream default 204800) and the forensic log flags. The
argv is copied verbatim from the package's own
`start-ninfer.ps1 -DryRun` output rather than from its launcher, because
`-WorkerArgs` in `-File` mode binds only its first token (two quoting forms tried,
both fail), and the request log is worth more than the launcher convenience.

## Why this is small (kept from the earlier plan)

RC3's worker is a **superset** of the flags our production launcher passes. Verified
by reading `infer-kvmem-server.exe --help` from the RC3 package:

| our production flag | RC3 |
|---|---|
| `--max-context` | present |
| `--kv-capacity N\|auto` | present (plus a separate `--kvmem-budget` for retrieval) |
| `--kv-dtype nvfp4` | present (bf16/int8/fp8/rk8v4/nvfp4/k8v4/rk4v4/rk4v4-e8/rk2v4-e8) |
| `--gdn-state-fp16` | present |
| `--spec mtp --draft-tokens 3 --lm-head-draft` | present, plus `--adaptive-mtp` |
| `--prefill-cublas`, `--prefill-chunk` | present |
| `--default-thinking-budget` | present |
| `--request-log-jsonl` | present, plus `--request-log-max-mib` / `--request-log-keep` rotation |
| `--vision --vision-residency overlay --vision-max-merged` | present (overlay still selectable; RC3's own default is `cpu`) |

New in RC3, not in our launcher: `--kvmem-budget`, `--kvmem-gen-reserve`,
`--kvmem-sink-tokens`, `--kvmem-recent-tokens`, `--kvmem-lruf*`, `--kvmem-score-agg`,
`--kvmem-sessions`, `--kvmem-host-mib`, `--kvmem-disk-path`, `--ngram-draft-tokens`,
`--ngram-min-match`, `--lookup-ngram`, `--device-profile auto|off|calibrate`.

## Target configuration

Everything that made ZCode work today is already on the right port, so the client
side does not change:

```
port                8081 (ZCode's provider_config.json points here directly)
model-id            bonsai2-27b (unchanged)
context             -Context 262144 (our production context; RC3's own default is 204800)
thinking budget     default 8192 (same value our launcher hard-codes)
vision              keep on; residency CPU is RC3's default and saves VRAM, overlay is
                    what we ran. Decide after measuring image turns.
prefill chunk       256 (RC3 default; ours was 1024 -- the RC3 measurements in this
                    repo were taken at 256, so keep 256 for comparability)
KV                  dual INT8 (RC3 default). Ours was nvfp4. Same VRAM at 10.3 GiB
                    measured; not yet compared for quality beyond recall 3/3.
retrieval budget    36864 + reserve 16384 + sink/recent 5120 each (RC3 defaults)
request log         re-add --request-log-jsonl C:\Bonsai-App\logs\requests-8081.jsonl
                    (RC3's launcher does not pass it; our forensics depend on it)
```

Launcher to use: the package's own `scripts\windows\start-ninfer.ps1`, invoked with
`-Model <artifact> -Port 8081 -Context 262144`, plus `-WorkerArgs` for the request
log. A thin `.bat` wrapper next to the existing one is the cleanest production entry
(it keeps the existing filename free for rollback).

## Rollback

```
Stop-Process -Name ninfer-kvmem-server -Force
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'start-ninfer' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
C:\Bonsai-App\Start-Bonsai-WebUI-256K.bat
```

The old launcher, the old binary (`C:\ninfer-build\franken\...`), and the old argv are
all left in place. Rollback is one command and does not depend on the new package.

## Evidence this is safe to try (from `rc3-acceptance-2026-10-12.md`)

- Acceptance matrix, four cells, all green: OpenAI 173K/262K, Anthropic 50K/262K.
- Recall 3/3 exact in every cell.
- Survival: 46 min worst case replaced by 15.3 min; continuations 4-6 s.
- Multi-turn: six successive commits, all survived, cache grows incrementally.
- Restart cycles: verified separately (see the restart-cycle run).
- Real client: ZCode drove a full agent loop against this engine (read tool, file
  content returned verbatim). The request also shows up engine-side
  (`history_misses` 2->4, `checkpoint_tokens` 25->14968).

## Behaviour differences to keep in mind

1. `reasoning_effort: none` is honoured; `low/medium/xhigh` are not (template).
2. `reasoning_budget_tokens` is a llama.cpp-route field; ignored here. Budget is the
   engine-level `--default-thinking-budget`.
3. Tool calls are emitted under a strict-schema default; `anyOf` degraded gracefully
   instead of erroring.
4. Context beyond native is not needed (native is 262144); RC3 also carries YaRN
   support, unused here.

## Monitoring surface (new, free)

`GET /props` returns a full kvmem block: capabilities (`selected_tokens`,
`reserve_tokens`, `sink_tokens`, `recent_tokens`, `mtp_drafts`, `ngram_drafts`,
`max_active_requests`, `retained_sessions`), per-process usage counters
(`history_hits`, `history_misses`, `checkpoint_tokens`, `checkpoint_bytes`,
`resident_pages`, `disk_*`), and `usage.active_payload_reservation_bytes`. It also
exposes `/slots` and `/health`. This is the data source for the "degradation panel"
idea -- no engine change needed to build it.

## Open questions before flipping

- Does `--kv-capacity 98304` (our ring pool) combine with `--kvmem-budget 36864`
  (retrieval) in RC3, and is that combination stable? Untested. The validated
  configuration is RC3's own recipe (no `--kv-capacity`), so the first production
  attempt should use the validated one, not the combination.
- Vision residency: cpu vs overlay on real image turns.
- Does the ZCode GUI (as opposed to the CLI) behave the same? Same code path, but the
  GUI adds the Electron app server.
