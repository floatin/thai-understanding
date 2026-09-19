"""Offline decoder tuning on the cached h173 logprobs (no GPU).

Grid: beam width x alpha (LM) x LM variants. CER via cer_no_space.
Multiprocess over samples (embarrassingly parallel).
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from ctc_beam import beam_search_decode, UnitLM, build_unit_lm, LM_PATH
from eval_cer_pretrained import cer_no_space

ROOT = Path('/data/workspace/asr-model-training/thai-understanding')
CACHE = ROOT / 'out_metric_gate' / 'h173_logp_v10best.pt'

_STATE = {}


def _init(lm_path, unseen):
    _STATE['lm'] = UnitLM(lm_path, unseen=unseen) if lm_path else None
    d = torch.load(CACHE, map_location='cpu', weights_only=False)
    _STATE['inv'] = {v: k for k, v in d['vocab'].items()}
    _STATE['logps'] = d['logps']
    _STATE['refs'] = d['refs']


def _decode_one(args):
    i, alpha, width = args
    lp = _STATE['logps'][i].numpy().astype(np.float32)
    hyp = beam_search_decode(lp, _STATE['inv'], lm=_STATE['lm'], alpha=alpha, beam_width=width)
    return cer_no_space(_STATE['refs'][i], hyp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--widths', type=int, nargs='*', default=[30, 60, 100])
    ap.add_argument('--alphas', type=float, nargs='*', default=[0.0])
    ap.add_argument('--lm', default=str(LM_PATH))
    ap.add_argument('--tag', default='')
    ap.add_argument('--unseen', type=float, default=-23.0)
    args = ap.parse_args()

    jobs = [(i, a, w) for w in args.widths for a in args.alphas for i in range(173)]
    with Pool(12, initializer=_init, initargs=(args.lm if args.lm != 'none' else None, args.unseen)) as pool:
        cers = pool.map(_decode_one, jobs, chunksize=4)
    cers = np.array(cers).reshape(len(args.widths), len(args.alphas), 173)
    print(f"--- {args.tag or args.lm}")
    for wi, w in enumerate(args.widths):
        for ai, a in enumerate(args.alphas):
            print(f"width={w:3d} alpha={a:<4}: h173 CTC-CER(ns)={cers[wi, ai].mean():.4f}")


if __name__ == '__main__':
    main()
