# P1 (windowed MTP draft attention) — status after the source dig (2026-10-18)

Candidate P1 of `post-ring-optimization-candidates.md`. Question: can the
production stack run `--mtp-attention-window` (the Windowed-MTP lever,
arXiv 2607.21535)?

**Answer: the mechanism exists upstream, KVMem refuses the combination by
construction, and the value under KVMem has never been measured — the cheap
measurements come first, not a port.**

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

## 5. Test plan (needs the exclusive GPU window; ordered by cost)

- **T1 — the flag's true effect, no code.** RC3 binary in **classic non-KVMem
  mode** (the flag is allowed there), A/B `--mtp-attention-window 0` vs `4096` at
  ~48K and ~64K context (`--kv-capacity` those sizes, `--kv-dtype int8`,
  `--spec mtp --draft-tokens 4`, ngram off to isolate MTP). Measure s/round and
  tok/s from the request log; confirm greedy outputs are identical. ~40 min.
- **T2 — the draft tax under KVMem, no code.** Production recipe:
  `--draft-tokens 4` vs `--draft-tokens 1` at the 173K anchor shape; the delta
  bounds what windowing could recover at most. ~30 min.
- **T3 — only if T1/T2 justify.** Upstream issue/PR on `kvmem/kvmem` (public,
  accepts PRs) to support the window under KVMem, or a guarded self-build to
  prove the combination works before asking. The sources are public; the build
  chain is not part of this plan.
- **Kill:** T1 <3% at 64K, or T2 <5% → close P1 with these numbers written down.

## 6. Nearby findings from the same dig (kept for the record)

- Upstream master still carries the guard (checked 2026-10-18); the old-GPU line
  `v0.18.1-ninfer-rc1` pins a newer ninfer (`1383a50e`) with no change here.
- The sm86 production engine binary is **unchanged** between RC3 and the stable
  v0.18.0 (same binary; launcher defaults and metadata differ) — no upgrade path
  to chase for this lever.
