# Thinking on/off quality: the three-set answer (2026-10-13 night)

Closes the question the first two rounds left open. Round 1 (custom 12) and round 2
(GSM8K-100) tied; the open question was *at what difficulty does thinking start paying*.
MATH-500 (stratified 60 of 500, 7 subjects, levels 1-5, seed 7) is the probe.

**Answer: it still does not pay, up to level 5.** 49/60 vs 49/60 on strict string
match, 55 vs 58 after manual LaTeX review — thinking *loses*. The mechanism is
visible in the misses: thinking spends the output budget before the answer lands.

Method unchanged from the earlier rounds: temperature 0, fixed seed 20261014, only
`reasoning_effort` differs, `max_tokens` 6000, every reply must end in `\boxed{}`.

## Recovering a crashed run

`math_test.py` summed `correct` values that are `None` (runs with no `\boxed{}`) and
died on the summary line — **after** the loop, **before** `json.dump`. A multi-hour run
was lost from the structured record; the per-question verdict lines in `math-run.out`
survived. `math_rebuild.py` reconstructs `math-results.json` from those lines (sample
regenerated deterministically, verdicts exact, boxed values truncated to 24 chars).
Not recoverable: per-question wall time and reasoning-token counts. Not guessed.

Script fixed: `score()` counts only `correct is True`, None counted as a miss.

## Strict score (as the string matcher sees it)

```
set            thinking ON    thinking OFF
custom 12      12/12          12/12
GSM8K-100      94/100         95/100
MATH-60        49/60          49/60      <-- tie again, at level 5
  L1-2         14/16          12/16
  L3           9/12           10/12
  L4-5         26/32          27/32
```

Split by difficulty shows nothing: the hard half (L4-5) ties 26/32 vs 27/32. The
"thinking pays on hard problems" hypothesis has no support anywhere in this family.

## Manual review of the misses (the string matcher lies a lot)

MATH-60 had 23 miss-records. Of those, 15 are **LaTeX-equivalent to gold**:

```
Q19  58500        vs 58,500        both arms      thousands separator
Q28  even         vs \text{even}   both arms      same token
Q43  0.0000672    vs .0000672      both arms      leading zero
Q49  \frac{17}{50} vs \dfrac{17}{50} both arms   \frac vs \dfrac
Q34  -2,1         vs 1,-2          both arms      set order irrelevant
Q24  2\sqrt{3}+1  vs 1+2\sqrt{3}   OFF            same value, terms swapped
Q39  2516         vs 2516_8        OFF            right number, missing base subscript
Q54  0.09         vs \frac{9}{100}  OFF            decimal vs fraction
Q56  11(1+\sqrt5) vs 11\sqrt{5}+11  ON             factored vs expanded
Q58  -3-2\sqrt{2},2\sqrt{2}-3,... vs 3\pm2\sqrt{2} OFF   four roots vs ± set
```

Adjusted score: **ON 55/60, OFF 58/60**. Thinking *loses by three* — and every one of
those three is a case where the model got the value right and the format wrong.

## The mechanism: thinking eats the answer

`NO-BOX` runs (no `\boxed{}` emitted at all, i.e. `finish=length` against the 6000
ceiling or an unwrapped answer):

```
arm        NO-BOX / 60     rate
thinking ON    5           8.3%      Q23 Q24 Q35 Q57 Q58
thinking OFF   1           1.7%      Q23
```

Same pattern as GSM8K (7 NO-ANSWERs, both arms, all `finish=length`) and the greedy
pass (`toolshape`/`format` at 400 tokens: both arms empty). Three independent runs,
same signature: **the thinking block is generated first and the answer gets whatever
is left**. On hard problems the block expands to the budget and the answer is the
casualty.

The genuinely wrong answers (not fixable by grading) are few and symmetric:

```
Q22 OFF  \frac{3}{4} vs 3            (geometry)
Q23 both  no answer                   (geometry, \frac{\sqrt{21}}{5})
Q24 ON   no answer                    (geometry)
Q35 ON   no answer                    (intermediate algebra)
Q57 ON   no answer                    (precalculus matrix)
Q58 ON   no answer                    (precalculus)
```

## The honest scope

Three rounds, 172 questions, single-step, exact-answer, temperature 0:

```
custom 12  : tie
GSM8K-100  : tie
MATH-60    : tie strictly, loses by 3 after review
```

* On single-step exact-answer work, thinking on this artifact (a 1.58-bit model at
  27B, RC3 engine) has **no measurable accuracy benefit at any difficulty tested**,
  costs 1.3-3x wall clock, and *raises the rate of malformed/empty answers by ~5x*.
* Where thinking demonstrably pays is not in this data: the pelican A/B (thinking-on
  shipped a dead `requestAnimationFrame` loop after 13 min 40 s; thinking-off shipped
  a clean file in 30 s), and DSH's own experience on hard agentic turns.
* The two conclusions do not conflict: they measure different things. These
  benchmarks measure answer correctness. The pelican measures deliverability.

## Operating line (unchanged, now with the strongest evidence yet)

- Routine work: thinking **off**. Faster, fewer format failures (1.7% vs 8.3%),
  shorter history, stable prefix hits.
- Hard reasoning: thinking **on** only when the task is not verifiable by the caller
  (multi-step tool use, ambiguous specs, debugging) — and then verify the artifact.
- `medium` is not a usable middle grade; the budget is the only precise dial.
- For web artifacts add an execution gate (headless browser: console errors,
  interaction, animation frames). Benchmark scores do not catch a dead script.

## Reproduce

- `py math_test.py 60 8082` (fixed, now writes json), `py math_rebuild.py` (from log),
  `py quality_test.py`, `py gsm8k_test.py 100`.
- Data: `math-results.json` (recovered), `gsm8k-results.json`, `quality-results.json`.
