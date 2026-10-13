# Thinking on this engine: levels, preservation, and the budget

Measured 2026-10-13 on the production configuration
(RC3 defaults + `--fast-prefill-kernel`, LRFU off, GPU vision, ctx 262144).
Question from real use: with thinking on, every agent step takes minutes; is the
thinking worth it, can its cost be cut, and do the "levels" actually exist?

## 1. How many thinking levels are there

`reasoning_effort` sweep, same prompt, `max_tokens` 10000 (above budget), two rounds:

```
level     round 1 reasoning   round 2 reasoning
none      0                   0                    finish=stop, full answer
low       8217                8217                 finish=length (hit the cap)
medium    5328                3892 / 2832 *        finish=length/length/stop
xhigh     8217                8217                 finish=length (hit the cap)
default   8217                8217                 finish=length (hit the cap)
* third medium sample (2832) run to confirm the pattern
```

- `none` is a real off switch: reasoning 0, complete answer, ~3x faster wall time.
- `low` / `xhigh` / `default` are indistinguishable: all run to the engine budget
  (8217 = 8192 + ~25 of template closing) six times out of six.
- `medium` stopped below the cap three times out of three (5328, 3892, 2832), with
  widely varying values. That is a consistent tendency, not a reliable grade: it
  comes from a template instruction toward briefer reasoning, and with sampling
  temperature 1.0 you cannot count on any particular number.

So: **two deterministic levels (off / capped-at-budget) plus a medium that usually
thinks less but by an unpredictable amount.** There is no low/medium/high ladder you
can set expectations on. The budget (`--default-thinking-budget`, exact: 8192->8217,
2048->2073) is the only precise control, and it is engine-wide, not per request
(`reasoning_budget_tokens` is ignored on this route).

## 2. `--preserve-thinking` has no observable effect on the chat path

The flag exists (`retain closed-turn assistant reasoning in later prompts`) and reaches
the worker's argv (verified). Test, both phases with the identical script:

- Ask the model to quote the first sentence of its own prior-turn reasoning: refused
  in both phases ("I can't share private reasoning" / "I can't share that reasoning").
  The refusal policy masks availability, so this test alone cannot discriminate.
- Digit test (designed to bypass the refusal: pick a digit in thinking, never write it
  in the answer, then ask for it next turn):
  **0/3 with the flag, 0/3 without it.** T2's own reasoning states it is stateless
  ("As AI stateless... Choose maybe 7") and T2's prompt (~103-190 tokens) has no room
  for hundreds of tokens of prior reasoning either way.
- T2 prompt sizes are identical across phases: the flag changes nothing in what the
  next turn receives.

Verdict: **no-op on this route** (possibly live on another route, e.g. Responses API
sessions — not tested, and not the route in production use). Production stays without
it. The caveat is recorded rather than deleted: if a future build wires it into the
chat path, this test (`think_digit.py`, self-scoring via `PICKED-DIGIT: N`) reruns
unchanged.

## 3. What thinking costs, measured

- Every agent step with thinking on generates budget-scale output (8-14 K tokens at
  ~40-60 tok/s: 3-5 min per step), because the budget is per turn, not per task.
- Thinking consumes the output budget first: answers regularly arrive truncated
  (`finish=length` with a few thousand chars of content vs 15006 complete chars with
  thinking off on the same shape).
- History inflation follows: each turn appends thousands of unique reasoning tokens,
  so the context fills faster and any client-side compaction triggers sooner, which
  breaks prefix reuse (see `prefix-instability-2026-10-13.md`). The quality/speed/hit
  triangle is real and all three corners were measured.

## 4. Operating recommendation

- Routine agent work: thinking off (`reasoning_effort: "none"` — verified effective).
  Faster (~50x per step in the measured case), complete answers, slow history growth,
  stable 99%+ prefix hits.
- Hard reasoning tasks: thinking on, accepting per-step minutes. No middle grade
  exists that reliably buys "a little thinking".
- Do not use `--preserve-thinking` on this route: measured no-op.
- Keep the engine budget at 8192: exact, fuse-tested (stops at ~8217 and releases the
  answer; earlier `==` vs `>=` class of bug verified absent here).

## Reproduce

- Levels: `py think_test.py effort [levels]` (non-streaming, server timings).
- Preservation: `py think_digit.py <tag> 3` (self-scoring; HIT needs `picked == got`).
- Quote probe: `py think_test.py conv <tag>` (expect refusal either way; the
  discriminating evidence is in turn-2's reasoning text, saved as `reasoning_tail`).
