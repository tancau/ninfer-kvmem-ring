# Vision residency: the default costs 300 s per image request

Recorded 2026-10-13. Found the same way as the earlier stall: the user reported that the
machine was busy in the wrong place.

## Symptom

CPU pegged at ~100 % (engine at 1000-1060 %, i.e. ~10 logical cores), GPU at 18 % / 17 W,
for minutes. Any request containing an image behaved this way.

## What the engine reports

```
 id  media  prompt   vision  prepare  prefill  decode   total
 19    0     44790     0.0     0.07     1.4      2.0      3.7
 20    3     48839     0.0     0.31     0.0      0.0    308.7
```

A three-image request took **308.7 s** while the engine's own breakdown attributed
**0.0 s to vision**, 0.0 to prefill and 0.0 to decode. The entire cost is outside the
timers -- the same accounting gap seen in `live-serving-stall-2026-10-13.md`, and again a
case where trusting the engine's own fields would mislead.

## Cause

The RC3 package's own launcher defaults to `--vision-residency cpu`, and that default was
copied into our production command. CPU vision runs the vision tower on the host: it uses
every core and leaves the GPU idle.

## The fix, and the trap on the way to it

GPU vision is `--vision-residency resident`. Note that the worker's `--help` lists
`resident|overlay|cpu`, but **`overlay` is rejected whenever KVMem is active**:

```
ninfer-serve: KVMem requires ordinary/fixed/adaptive MTP1..15, ...,
1..8 text or resident/CPU-vision lanes, ...
```

Passing `overlay` makes the engine print its usage text and exit 1, with the reason on the
first line of stderr. The help text also misleads on meaning: `overlay` is not GPU vision,
it keeps the vision tower in host memory with no device residency.

## Measured

Same three-image request, through the client-facing port:

```
--vision-residency cpu       3 images   308.7 s   CPU ~1000 %, GPU idle
--vision-residency resident  3 images     1.8 s   GPU vision, VRAM 10,054 -> 10,345 MiB
```

About **170x**, for 291 MiB of VRAM. `input_tokens` grows with the image count
(210 -> 502 for 1 -> 3 images), so the images are genuinely encoded into the prompt.

## Adopted configuration

```
--vision --vision-residency resident --vision-max-merged 1024
```

Everything else unchanged: upstream defaults + `--fast-prefill-kernel`, LRFU off,
`--max-context 262144`, port 8082, keepalive proxy on 8081, request log on.

## Reproduce

```
py vision_test.py 8081 3 gpu      # 3 synthetic images, printed wall time
```

## Still open

- `timings_seconds.vision` stays 0.0 for the CPU path and (to be confirmed) for the GPU
  path: the field is not usable for diagnosis.
- The engine's un-attributed time is now three for three (stall, prefix miss, vision).
  Any future investigation should treat these fields as unreliable and measure from the
  outside.
