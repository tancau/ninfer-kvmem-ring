# ninfer-ring — NInfer ternary engine with KVMem-style ring KV

NInfer (`franken/v0.11` branch family) ported so the **device KV pool may be
smaller than the logical context**, KVMem-style: the device pool is the
attention window, evicted pages spill **losslessly to host RAM**, and a
retrieval scorer brings relevant pages back (H2D). Verified on
**RTX 3060 12GB / sm_86 / Windows 10**: 96K device pool serves 152K-token
sequences (1.6x) at ~10.2 GB VRAM; dense 160K config stays bit-identical
(PPL 5.639521, 62/62 sample points, delta 0).

Model: Ternary-Bonsai-2-27B (`bonsai2-27b`), `--kv-dtype nvfp4`.

## Layout

- Engine source (this directory, `src/` …): build with
  `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build.ps1`
  (requires CUDA 12.8 + MSVC; sm_86 compatible).
- `bonsai-app/`: one-click launchers, WebUI shim, validation docs.
  - `Start-Bonsai-WebUI.bat` — production dense 160K + WebUI on :8080.
  - `BENCH-RESULTS.md` — all measured numbers.
  - `KVMEM-PORT-PLAN.md` — port design, 8 constraint layers, defect log,
    A/B validation, capacity math.

## Ring mode (experimental, opt-in)

```bat
set NINFER_KV_RING=1
set NINFER_KV_WINDOW=32768
set NINFER_KV_RETRIEVE=4096
set NINFER_HOST_PAGEABLE=1
ninfer-serve.exe <model.ninfer> --model-id bonsai2-27b --max-context 131072 --kv-capacity 32768 ...
```

Without `NINFER_KV_RING=1`, `--kv-capacity < --max-context` is rejected and
the engine behaves exactly as upstream (all prototype paths are gated).
