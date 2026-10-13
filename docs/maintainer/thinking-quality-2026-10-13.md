# Thinking on/off quality: measured answer (2026-10-13)

Question: does thinking produce better answers, and by how much? All prior thinking
measurements on this engine covered speed, recall and hit rate — never quality. This
closes that gap with auto-scored sets, run on the production configuration
(RC3 defaults + fast-prefill, LRFU off, GPU vision, ctx 262144).

## Method (fixed before running)

- Temperature 0, fixed seed 20261014, both arms. Only `reasoning_effort` differs.
- Every reply must end with `ANSWER: <value>`; only that field is graded, so thinking
  verbosity never affects the score.
- Numbers normalised (commas/currency/decimals); missing ANSWER line recorded as
  NO-ANSWER, separately from WRONG.
- Round 1: 12 custom questions (8 reasoning: arithmetic, GCD, modpow, probability,
  knights, sequence; 4 factual as control).
- Round 2: GSM8K test set, first 100, canonical `####` answers.

## Results

```
round        thinking ON   thinking OFF
custom 12    12/12         12/12
GSM8K-100    94/100        95/100
```

Miss breakdown (GSM8K):

```
Q12   both arms  12.0 vs gold 13.0   same wrong answer with and without thinking
Q37   off only   0.0 vs 2.0
Q93   off only   36.36 vs 36.0       grading artefact: right value, wrong format
rest  both arms  NO-ANSWER, all finish=length (2000-token cap hit)
```

Timing (temp 0, short questions): thinking mean 6.0 s vs off 4.6 s per question.
(The production gap is far larger — 62 s vs ~150 s measured earlier — because
production runs temperature 1.0 on hard tasks, where thinking expands to budget.)

## Reading

1. **On these benchmarks thinking adds nothing measurable.** 12/12 ties, 94-vs-95
   ties, and the single shared mistake (Q12, identical wrong answer both arms) shows
   thinking does not even take a different path on the one question both arms miss.
2. **The NO-ANSWERs are harness artefacts, not reasoning failures.** All seven are
   `finish=length` against the 2000-token ceiling, split across both arms. If anything
   they confirm the earlier finding that thinking consumes output budget first.
3. **This does not contradict the observation that thinking helps on real work.**
   Neither round measures agentic coding work (multi-step tool use, ambiguous specs,
   debugging). GSM8K is grade-school arithmetic; the custom set is single-step. The
   honest scope of this result: *on short, well-specified questions with exact answers,
   thinking changes nothing.*
4. The open question it leaves: at what difficulty does thinking start paying? The
   back half of GSM8K (Q50-99: ON 47/50, OFF 48/50) shows no separation either, so the
   threshold — if it exists in this family — is beyond GSM8K. MATH-level problems would
   be the next probe, not more GSM8K.

## Operating recommendation (unchanged, now with quality evidence)

- Routine agent work: thinking off. Faster, complete answers, slow history growth,
  stable 99%+ prefix hits — and no measured quality loss on exact-answer tasks.
- Hard reasoning: thinking on. Its value is attested by use, not by these benchmarks;
  the cost (minutes per step, history inflation) is measured and real.
- `medium` is not a reliable middle grade (2832-5328 across runs); the budget is the
  only precise dial. See `thinking-levels-2026-10-13.md`.

## Reproduce

- `py quality_test.py [port]` (custom 12) and `py gsm8k_test.py [n] [port]` (standard).
  GSM8K set caches to `gsm8k-test.jsonl` on first run.
- Results: `quality-results.json`, `gsm8k-results.json` (kept alongside the scripts).
