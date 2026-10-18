"""Re-run only the MMLU-Pro items whose answer was cut off by the token cap.

The main run records `capped` per item. Those items score 26% against 83% for
the rest, so leaving them in understates the model. This re-asks exactly those
items with a much larger budget and rewrites the result file.

  py mmlupro_fix.py <port>
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
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
SEED = 20261016
CACHE = r'C:\Users\Administrator\AppData\Local\Temp\opencode\mmlupro-test.jsonl'
RES = r'C:\Users\Administrator\AppData\Local\Temp\opencode\mmlupro-results.json'
LETTERS = 'ABCDEFGHIJ'

items = []
with open(CACHE, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            items.append(json.loads(line))
random.seed(SEED)
sample = random.sample(items, 280)

d = json.load(open(RES, encoding='utf-8'))
res = d['results']
todo = [r for r in res if r.get('capped')]
print('capped items to re-run: %d' % len(todo), flush=True)

try:
    _m = json.loads(urllib.request.urlopen(
        'http://127.0.0.1:%d/v1/models' % PORT, timeout=20).read().decode())
    MODEL = _m['data'][0]['id']
except Exception:
    MODEL = 'bonsai2-27b'
print('model id: %s' % MODEL, flush=True)

INSTR = ('Answer the following multiple choice question. Think step by step, then end your '
         'reply with a line of exactly the form "ANSWER: $LETTER" (no quotes), where LETTER '
         'is one of %s.' % 'ABCDEFGHIJ')


def build(q):
    opts = '\n'.join('%s. %s' % (LETTERS[i], o) for i, o in enumerate(q['options']))
    return 'Question: %s\n\nOptions:\n%s\n\n%s' % (q['question'], opts, INSTR)


def parse(txt):
    m = re.findall(r'ANSWER:\s*\(?([A-J])\)?', txt or '', re.I)
    if m:
        return m[-1].upper()
    m = re.findall(r'\b([A-J])\b', txt or '')
    return m[-1].upper() if m else None


def ask(text):
    body = {"model": MODEL, "max_tokens": BUDGET, "temperature": 0.0, "seed": SEED,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions' % PORT,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as h:
        o = json.loads(h.read().decode('utf-8', 'replace'))
    ch = (o.get('choices') or [{}])[0]
    return (ch.get('message') or {}).get('content') or '', o.get('timings') or {}


t0 = time.time()
for k, r in enumerate(todo):
    q = sample[r['i']]
    gold = q['answer'].strip().upper()
    try:
        txt, tim = ask(build(q))
        got = parse(txt)
        ok = (got == gold)
    except Exception as e:
        txt, tim, got, ok = '', {}, None, False
        print('  item %d error: %r' % (r['i'], e), flush=True)
    r.update(got=got, correct=ok, capped=(tim.get('predicted_n') or 0) >= BUDGET,
             retried=True, timings=tim)
    if (k + 1) % 10 == 0:
        print('  %d/%d done (%.0fs)' % (k + 1, len(todo), time.time() - t0), flush=True)

ok = sum(1 for r in res if r['correct'])
cap = sum(1 for r in res if r.get('capped'))
json.dump({'n': len(res), 'seed': SEED, 'model': MODEL, 'correct': ok,
           'acc': 100.0 * ok / len(res), 'partial': False,
           'capped_left': cap, 'budget': BUDGET, 'results': res},
          open(RES, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('AFTER FIX: %d/%d = %.1f%%   (still capped: %d)'
      % (ok, len(res), 100.0 * ok / len(res), cap))
