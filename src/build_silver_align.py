"""Batch-align the 5159 silver (audio, transcript) pairs via the local
align-words service (port 8082). Produces char-level timings + score per item.

Output: out_align/silver_align.jsonl — one line per item:
  {id, wav_path, ref, score, duration_sec, chars:[{char,start,end}], ok}

Filtering (word-span dataset builder, next step) drops: score<0.4, low speech
coverage, char-count mismatch, service failures.
Auth: bearer key from /root/workspace/llama.cpp/.api_key (service convention).
"""
import json
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import urllib.request

KEY = open('/root/workspace/llama.cpp/.api_key').read().strip()
URL = 'http://127.0.0.1:8082/align/'
TRAIN_SPLIT = '/data/workspace/asr-model-training/thai-understanding/ptt_train_split.json'
OUT = '/data/workspace/asr-model-training/thai-understanding/out_align/silver_align.jsonl'


def align_one(item):
    import uuid
    boundary = uuid.uuid4().hex
    with open(item['wav_path'], 'rb') as f:
        audio_bytes = f.read()
    body = (f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="audio"; filename="a.wav"\r\n'
            f'Content-Type: audio/wav\r\n\r\n').encode() + audio_bytes + \
           (f'\r\n--{boundary}\r\n'
            f'Content-Disposition: form-data; name="transcript"\r\n\r\n{item["ref"]}\r\n'
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="language"\r\n\r\nth\r\n'
            f'--{boundary}--\r\n').encode()
    req = urllib.request.Request(URL, data=body, method='POST')
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    req.add_header('Authorization', f'Bearer {KEY}')
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            r = json.loads(resp.read())
        chars = r.get('chars', [])
        span = (chars[-1]['end'] - chars[0]['start']) if chars else 0.0
        cover = span / max(r.get('duration_sec', 1e-6), 1e-6)
        score = r['words'][0].get('score', -1) if r.get('words') else -1
        ok = (len(chars) == len(item['ref'])) and cover > 0.3 and (score >= 0.4)
        return {'id': item['id'], 'wav_path': item['wav_path'], 'ref': item['ref'],
                'score': score, 'duration_sec': r.get('duration_sec'),
                'speech_coverage': round(cover, 3), 'chars': chars, 'ok': bool(ok)}
    except Exception as e:
        return {'id': item['id'], 'wav_path': item['wav_path'], 'ref': item['ref'],
                'score': -1, 'duration_sec': None, 'speech_coverage': 0.0,
                'chars': [], 'ok': False, 'error': str(e)[:200]}


def main():
    data = json.load(open(TRAIN_SPLIT))['samples']
    done_ids = set()
    outf = Path(OUT)
    outf.parent.mkdir(exist_ok=True)
    if outf.exists():
        for line in open(OUT):
            try:
                done_ids.add(json.loads(line)['id'])
            except Exception:
                pass
    todo = [s for s in data if s['id'] not in done_ids]
    print(f"total={len(data)} done={len(done_ids)} todo={len(todo)}", flush=True)
    t0 = time.time()
    n_ok = 0
    with ThreadPoolExecutor(max_workers=6) as ex, open(OUT, 'a') as f:
        for k, rec in enumerate(ex.map(align_one, todo)):
            n_ok += rec['ok']
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if (k + 1) % 200 == 0:
                el = time.time() - t0
                print(f"  {k+1}/{len(todo)} ok={n_ok} ({el:.0f}s, ETA {el/(k+1)*(len(todo)-k-1):.0f}s)", flush=True)
    print(f"DONE: {n_ok}/{len(todo)} aligned ok")


if __name__ == '__main__':
    main()
