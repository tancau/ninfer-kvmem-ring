# P1 (windowed MTP draft attention) — status after the source dig (2026-10-18)

Candidate P1 of `post-ring-optimization-candidates.md`. Question: can the
production stack run `--mtp-attention-window` (the Windowed-MTP lever,
arXiv 2607.21535)?

**Closed 2026-10-19: the mechanism exists upstream, KVMem refuses the combination
by construction — and where the flag *is* allowed (classic mode) its measured
value is ≈0 (T1 ≤1%, byte-identical outputs). Kill criterion met; no port.**

## 1. The flag exists and is implemented as a draft-only page rebind

`docs/serving.md` in `iamwavecut/ninfer-all` (upstream master, 2026-10-18):

> `--mtp-attention-window N` — MTP only: the draft head attends to the first 64
> keys and the newest `N` before its query; verification keeps full attention.
> […] The Engine rebinds the draft's page table to those pages and shifts its
> positions, so the attention kernels are the unwindowed ones.

Validation (`src/product/speculative_options.h`):

```cpp
if (options.mtp_attention_window != 0 &&
    (options.backend != SpeculativeBackend::Mtp ||
     options.mtp_attention_window < options.draft_tokens + 1)) {
    throw std::invalid_argument(
        "--mtp-attention-window requires --spec mtp and at least --draft-tokens + 1 keys");
}
```

Implementation reach: `src/models/qwen3_5/program/speculative/mtp.cpp` calls
`card.set_mtp_attention_window(state.execution.mtp_attention_window)` on the draft
passes. Verification is untouched, so greedy output is unchanged — the draft only
changes speed.

## 2. KVMem rejects it, by construction

The KVMem patch is public: `kvmem/kvmem` → `backends/patches/ninfer-kvmem.patch`
(1.2 MB; master, fetched 2026-10-18). In the option normalizer:

```cpp
const bool memory_spec_supported = (ordinary || fixed_mtp || adaptive_mtp || memory_dflash2) &&
    memory_ngram_supported && options.speculative.mtp_attention_window == 0;
```

and the reject path is `if (...) !memory_spec_supported || ...` → the engine
refuses to start. The patch also carries a unit test that asserts the rejection:
`o.speculative.mtp_attention_window = 128` sits in the `invalid` case list.

Consequences:

- This is the fifth "help text exists, runtime rejects" instance in this
  project — and the first one with a public, source-level reason.
- The reason is ownership: KVMem manages the draft KV's page mapping itself
  (`ensure_sequence_kv_mapped`, cache-position shifts when the device window
  slides, the MTP-KV mirror of the resident pages), while the window feature is a
  second owner of the same page-table rows. The combination is unimplemented, not
  merely untested.
- It also answers the flag-sweep note: `--mtp-attention-window` could never have
  worked on the production recipe, at any value.

## 3. What the win would be on our stack (first-order estimate)

- Under KVMem the draft reads the **resident window** (B = 57,344 tokens in
  production), not the whole logical history — the MTP KV mirrors the device
  window. Per draft step that is ~1 full-attention layer of KV reads against 16
  in a verification pass; at 4 drafts/round the draft is ~25% of the per-round
  KV-read traffic at most.
- Measured long-context tax: 38.6 ms/round at ≤5k vs 53.4 ms/round at 130K+
  (`decode-host-stage0-2026-10-18.md`). Whether that +15 ms sits in KV-read
  bandwidth, page movement, or retrieval compute is exactly the open Stage 0b
  question.
- The paper's +28–44% per decode step at 1M does **not** transfer: that regime is
  whole-cache-on-device where the draft scan grows with 1M tokens. Here the scan
  is capped by the device window.
- Honest expectation: **low single digits to ~10% of decode at long context**,
  unknown until measured. Worth one measurement; not worth engine surgery before
  that.

## 4. Fork-side alternative (reference only)

The fork lineage can express a draft-only window with the envelope mechanism
already analysed: `mtp_causal_attention_envelopes` sets a bounded draft interval
(`min_visible_keys` is a start offset; the kernel skips key windows outside
`[min, max]`), while `target_verify` keeps full attention, so greedy output is
unchanged by construction. A single interval expresses "newest N" but not
"64-key sink + newest N" (two ranges) — upstream solved that needing a rebind
instead. This stays a reference implementation: production runs KVMem.

## 5. Results (2026-10-19, exclusive GPU window) — kill criterion met

**KVMem rejection, literal text** (engine exits 1 at option validation, before
any device allocation):

```
ninfer-serve: KVMem requires ordinary/fixed/adaptive MTP1..15, ngram1..63 (automatically capped at 15 with 2..8 lanes), single-GPU or layer-pipeline BF16/INT8/FP8/NVFP4/K8V4/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8 generation, 1..8 text or resident/CPU-vision lanes, INT8/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8 fast prefill, and exclusive ownership of history caching
```

Anything but "ordinary/fixed/adaptive" MTP is refused — exactly the guard.

**T1 — classic non-KVMem A/B (the flag where it is allowed).** Engine in classic
mode, `--kv-capacity 49152 --kv-dtype int8 --spec mtp --draft-tokens 4`, ngram
off, prompt 42,909 tokens, 1,024 greedy tokens, same seed:

| arm | rounds | decode | s/round | tok/s | accepted | host µs/round |
|---|---:|---:|---:|---:|---:|---:|
| window 0 (default) | 513 | 23.47 s | 45.8 ms | 43.6 | 510/2050 | 84 |
| window 4096 | 518 | 23.94 s | 46.2 ms | 42.8 | 505/2070 | 80 |

Greedy outputs are **byte-identical**; acceptance nearly unchanged. Delta
+0.4 ms/round (≤1%, wrong sign) — **below the 3% kill threshold**. The draft's
KV scan is simply not a cost in this regime, so the window has nothing to win.

**T2 — draft-width A/B under the production KVMem recipe** (43K prompt, ngram
on as in production; `--draft-tokens 4` vs `1`, adaptive live):

| arm | rounds | decode | s/round | tok/s | accepted/round | drafted/round |
|---|---:|---:|---:|---:|---:|---:|
| draft 4 | 490 | 21.54 s | 44.0 ms | 47.5 | 1.09 | 3.06 |
| draft 1 | 613 | 20.39 s | 33.3 ms | 50.2 | 0.67 | 1.00 |

The marginal cost of ~1 → ~3 draft steps per round is 10.7 ms — **≈5.2 ms per
extra step (including the wider verify batch), ~12% of a round**. That cost sits
in the draft head's own forward execution, not in its KV scan (T1), so a
scan-side window cannot remove it. On this synthetic filler prompt draft 1 even
wins end-to-end (+5.7%) because acceptance there is poor (35.5%). It is **not**
a production recommendation: the real traffic log (`requests-8081.jsonl`, 1,448
requests, 0–20k prompts) shows acceptance 55.5% mean / 53.9% median and 3.14
accepted tokens/round — production drafting pays for itself, and adaptive
already sizes the width.

**Kill:** met (T1 ≤1%). P1 is closed: the flag was unusable under KVMem by
construction, and where usable it buys nothing. T3 (upstream PR) is abandoned.
The one fact worth keeping: the per-draft-step cost (~5 ms) is a real,
quantified ceiling for speculative-decoding tuning, and it lives in the MTP
head's forward, not in its cache reads.

## 6. Test plan as originally specified (all executed or abandoned)

- T1 was run at 49,152 capacity (not 64K) — the effect was already <1% and
  capacity changes nothing about the mechanism. T2 was run at the 43K prompt
  shape (the 173K shape costs ~6 min of prefill per arm for no extra signal:
  the marginal step cost is a per-step property, not a per-context one).
- T3 (upstream PR): abandoned by the kill rule.

## 6. Nearby findings from the same dig (kept for the record)

- Upstream master still carries the guard (checked 2026-10-18); the old-GPU line
  `v0.18.1-ninfer-rc1` pins a newer ninfer (`1383a50e`) with no change here.
- The sm86 production engine binary is **unchanged** between RC3 and the stable
  v0.18.0 (same binary; launcher defaults and metadata differ) — no upgrade path
  to chase for this lever.
