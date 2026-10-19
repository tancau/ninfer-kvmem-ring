"""MMLU-Pro on our engine, aligned with the published reference slice.

Reference conditions (WaveCut's ternary Bonsai 2 quality table): greedy decoding,
no sampling penalties, thinking off. 280 items.

The published note says only "a fixed 1,179-item slice" and does not name the
items, so this is the same suite and item count, not the same items. Scores are
comparable in aggregate, not item-for-item.

  py mmlupro_test.py <port> [n_items]

Writes mmlupro-results.json.
"""
import json
import os
import random
import re
import sys
import time
import urllib.request

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8082
N = int(sys.argv[2]) if len(sys.argv) > 2 else 280
SEED = 20261016
CACHE = r'C:\Users\Administrator\AppData\Local\Temp\opencode\mmlupro-test.jsonl'
LETTERS = 'ABCDEFGHIJ'


def fetch_rows():
    """Fetch all rows, appending to CACHE as we go so a 429 is resumable.

    The datasets-server rate-limits bursts, so pages are paced and retried with
    a growing backoff. A re-run continues from the last row already on disk.
    """
    base = ('https://datasets-server.huggingface.co/rows?dataset=TIGER-Lab%2FMMLU-Pro'
            '&config=default&split=test&length=100&offset=')
    have = 0
    if os.path.isfile(CACHE):
        with open(CACHE, encoding='utf-8') as fh:
            have = sum(1 for line in fh if line.strip())
    mode = 'a' if have else 'w'
    total = None
    with open(CACHE, mode, encoding='utf-8') as out:
        off = have
        while True:
            d = None
            for attempt in range(8):
                try:
                    with urllib.request.urlopen(base + str(off), timeout=60) as h:
                        d = json.loads(h.read().decode())
                    break
                except Exception as e:
                    wait = 5 * (attempt + 1)
                    print('    retry in %ds (%r)' % (wait, e), flush=True)
                    time.sleep(wait)
            if d is None:
                raise RuntimeError('gave up at offset %d (re-run to resume)' % off)
            rows = d['rows']
            for r in rows:
                out.write(json.dumps(r['row'], ensure_ascii=False) + '\n')
            out.flush()
            off += len(rows)
            total = d['num_rows_total']
            if not rows or off >= total:
                break
            if off % 1000 == 0:
                print('  fetched %d/%d' % (off, total), flush=True)
            time.sleep(2.0)
    return total


if not os.path.isfile(CACHE):
    print('downloading MMLU-Pro test split ...', flush=True)
    total = fetch_rows()
    print('cached %s items' % total, flush=True)

items = []
with open(CACHE, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            items.append(json.loads(line))

random.seed(SEED)
sample = random.sample(items, min(N, len(items)))
print('loaded %d questions, sampling %d (seed %d)' % (len(items), len(sample), SEED),
      flush=True)

try:
    _m = json.loads(urllib.request.urlopen(
        'http://127.0.0.1:%d/v1/models' % PORT, timeout=20).read().decode())
    MODEL = _m['data'][0]['id']
except Exception:
    MODEL = 'bonsai2-27b'
print('model id: %s' % MODEL, flush=True)

# ---- resume: continue an interrupted run from its per-item checkpoint ----
RES_PATH = r'C:\Users\Administrator\AppData\Local\Temp\opencode\mmlupro-results.json'
results = []
if os.path.isfile(RES_PATH):
    try:
        prev = json.load(open(RES_PATH, encoding='utf-8'))
        pr = prev.get('results') or []
        if (prev.get('partial') and prev.get('seed') == SEED and
                str(prev.get('model', '')) == MODEL and 0 < len(pr) < len(sample)):
            results = pr
            print('resuming: %d/%d items already done' % (len(results), len(sample)),
                  flush=True)
    except Exception as e:
        print('resume skipped: %r' % e, flush=True)

INSTR = ('Answer the following multiple choice question. Think step by step, then end your '
         'reply with a line of exactly the form "ANSWER: $LETTER" (no quotes), where LETTER '
         'is one of %s.' % 'ABCDEFGHIJ')


def build(q):
    opts = '\n'.join('%s. %s' % (LETTERS[i], o) for i, o in enumerate(q['options']))
    return 'Question: %s\n\nOptions:\n%s\n\n%s' % (q['question'], opts, INSTR)


def ask(text):
    body = {"model": MODEL, "max_tokens": 3072, "temperature": 0.0, "seed": SEED,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions' % PORT,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as h:
        o = json.loads(h.read().decode('utf-8', 'replace'))
    ch = (o.get('choices') or [{}])[0]
    return (ch.get('message') or {}).get('content') or '', o.get('timings') or {}


def parse(txt):
    m = re.findall(r'ANSWER:\s*\(?([A-J])\)?', txt or '', re.I)
    if m:
        return m[-1].upper()
    m = re.findall(r'\b([A-J])\b', txt or '')
    return m[-1].upper() if m else None


t0 = time.time()
for i, q in enumerate(sample):
    if i < len(results):
        continue
    gold = q['answer'].strip().upper()
    try:
        txt, tim = ask(build(q))
        got = parse(txt)
        ok = (got == gold)
    except Exception as e:
        txt, tim, got, ok = '', {}, None, False
        print('  item %d error: %r' % (i, e), flush=True)
    results.append({'i': i, 'category': q.get('category'), 'gold': gold, 'got': got,
                    'correct': ok, 'capped': (tim.get('predicted_n') or 0) >= 3072,
                    'timings': tim})
    # write after every item so a partial run is still usable
    json.dump({'n': len(results), 'seed': SEED, 'model': MODEL,
               'correct': sum(1 for r in results if r['correct']),
               'acc': 100.0 * sum(1 for r in results if r['correct']) / len(results),
               'partial': True, 'results': results},
              open(r'C:\Users\Administrator\AppData\Local\Temp\opencode\mmlupro-results.json',
                   'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    if (i + 1) % 20 == 0:
        acc = sum(1 for r in results if r['correct'])
        print('  %d/%d  acc=%d/%d = %.1f%%  (%.0fs elapsed)'
              % (i + 1, len(sample), acc, i + 1, 100.0 * acc / (i + 1),
                 time.time() - t0), flush=True)

ok = sum(1 for r in results if r['correct'])
print('MMLU-Pro %d/%d = %.1f%%' % (ok, len(results), 100.0 * ok / len(results)))
by_cat = {}
for r in results:
    c = r['category'] or '?'
    a, b = by_cat.get(c, (0, 0))
    by_cat[c] = (a + (1 if r['correct'] else 0), b + 1)
print('by category:')
for c in sorted(by_cat):
    a, b = by_cat[c]
    print('  %-28s %d/%d = %.1f%%' % (c, a, b, 100.0 * a / b))

out = r'C:\Users\Administrator\AppData\Local\Temp\opencode\mmlupro-results.json'
json.dump({'n': len(results), 'seed': SEED, 'model': MODEL, 'correct': ok,
           'acc': 100.0 * ok / len(results), 'partial': False, 'results': results},
          open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', out)
