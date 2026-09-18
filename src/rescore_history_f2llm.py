"""Post-training F2LLM scoring for eval_history.jsonl lines.

v5 training-time evals deferred F2LLM (GPU shared with the align service);
hyps+refs are already saved per item — score them here (AGENTS.md: saved raw
predictions must be re-scorable under any metric). Idempotent: skips lines
already carrying sims. F2LLM batch needs ~9GB GPU free.

Usage: python rescore_history_f2llm.py [--file u_align_stage2_v5/eval_history.jsonl]
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

EMBED_MODEL = '/data/workspace/models/F2LLM-v2-4B'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default='/data/workspace/asr-model-training/thai-understanding/u_align_stage2_v5/eval_history.jsonl')
    args = ap.parse_args()
    lines = [json.loads(l) for l in open(args.file)]
    todo = [r for r in lines if not r.get('sims')]
    print(f"{len(lines)} lines, {len(todo)} to rescore")
    if not todo:
        return

    model = SentenceTransformer(EMBED_MODEL, device='cuda', model_kwargs={'torch_dtype': torch.bfloat16})
    with open(args.file, 'w') as f:
        for rec in lines:
            if not rec.get('sims'):
                sims = []
                for a, b in zip(rec['refs'], rec['hyps']):
                    e = model.encode([a, b], normalize_embeddings=True, convert_to_numpy=True)
                    sims.append(round(float(e[0] @ e[1]), 4))
                rec['sims'] = sims
                if all(len(h) > 0 for h in rec['hyps']):
                    rec['ser'] = round(1.0 - float(np.mean(sims)), 4)
                print(f"  step={rec['step']} subset={rec['subset']} n={len(sims)} SER={rec.get('ser')}", flush=True)
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print("done")


if __name__ == '__main__':
    main()
