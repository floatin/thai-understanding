"""Remote F2LLM client (llama.cpp GGUF server at 175.27.138.38:8087).

Local reference implementation (rescore_history_f2llm.py) uses
SentenceTransformer normalize_embeddings=True + dot product. This client
mirrors that: L2-normalize remote embeddings, cosine = dot.

Calibration mode: re-score a stored eval_history line that already carries
local sims, report per-item agreement + SER delta. Run this BEFORE using the
remote endpoint for any conclusion (AGENTS.md metric-implementation gate).
"""
import argparse
import json
import urllib.request

import numpy as np

ENDPOINT = 'http://175.27.138.38:8087/v1/embeddings'


def _post(payload, timeout=120):
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed_remote(texts, batch_size=16, retries=3):
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = [t if t.strip() else ' ' for t in texts[i:i + batch_size]]
        for attempt in range(retries):
            try:
                resp = _post({'input': chunk, 'model': 'f2llm'})
                break
            except Exception:
                if attempt == retries - 1:
                    raise
        data = sorted(resp['data'], key=lambda d: d['index'])
        out.extend([np.asarray(d['embedding'], dtype=np.float64) for d in data])
    embs = np.stack(out)
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default='/data/workspace/asr-model-training/thai-understanding/u_align_stage2_v7/eval_history.jsonl')
    ap.add_argument('--step', type=int, default=None, help='defaults to last line')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--rescore', action='store_true',
                    help='score all lines lacking sims (mirrors rescore_history_f2llm.py)')
    args = ap.parse_args()

    if args.rescore:
        lines = [json.loads(l) for l in open(args.file)]
        with open(args.file, 'w') as f:
            for rec in lines:
                if not rec.get('sims') and rec.get('hyps'):
                    refs, hyps = rec['refs'], rec['hyps']
                    e_ref = embed_remote(refs, args.batch)
                    e_hyp = embed_remote(hyps, args.batch)
                    sims = [round(float(a @ b), 4) for a, b in zip(e_ref, e_hyp)]
                    rec['sims'] = sims
                    if all(len(h) > 0 for h in hyps):
                        rec['ser'] = round(1.0 - float(np.mean(sims)), 4)
                    print(f"  step={rec.get('step')} subset={rec.get('subset')} "
                          f"n={len(sims)} SER={rec.get('ser')}", flush=True)
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print("rescore done")
        return

    lines = [json.loads(l) for l in open(args.file)]
    rec = lines[-1] if args.step is None else next(r for r in lines if r['step'] == args.step)
    refs, hyps, local = rec['refs'], rec['hyps'], rec['sims']

    e_ref = embed_remote(refs, args.batch)
    e_hyp = embed_remote(hyps, args.batch)
    remote = [round(float(a @ b), 4) for a, b in zip(e_ref, e_hyp)]
    rser = 1.0 - float(np.mean(remote))
    lser = 1.0 - float(np.mean(local))

    diff = np.array(remote) - np.array(local)
    print(f"step={rec['step']} subset={rec['subset']} n={len(refs)} emb_dim={e_ref.shape[1]}")
    print(f"local  SER={lser:.4f}")
    print(f"remote SER={rser:.4f}   delta={rser - lser:+.4f}")
    print(f"per-item |diff| mean={np.abs(diff).mean():.4f} max={np.abs(diff).max():.4f}")
    print(f"pearson r={np.corrcoef(remote, local)[0, 1]:.4f}")
    # 方向性 sanity：identical 必须接近 1
    e_id = embed_remote(['สวัสดีครับ', 'สวัสดีครับ'], 2)
    print(f"identical-pair cosine={float(e_id[0] @ e_id[1]):.4f}")


if __name__ == '__main__':
    main()
