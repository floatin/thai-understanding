"""Dump h173 CTC logprobs for offline decoder iteration.

One GPU pass over h173 with a front-end checkpoint; everything downstream
(beam width / alpha / LM variants) then iterates on CPU without re-running
the encoder. Artifact: h173_logp_<tag>.pt
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import train_u_align_stage2_v5 as v5
from thai_ctc_units import build_vocab, text_to_ids
from xlsr_thai import load_xlsr_thai, extract_features

ROOT = Path('/data/workspace/asr-model-training/thai-understanding')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(ROOT / 'u_align_ctc_only_v10' / 'best' / 'ctc_only_v10.pt'))
    ap.add_argument('--out', default=str(ROOT / 'out_metric_gate' / 'h173_logp_v10best.pt'))
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    vocab = ck['vocab']
    inv = {v: k for k, v in vocab.items()}
    encoder, _ = load_xlsr_thai(v5.XLSR_CKPT, device='cuda')
    encoder.eval()
    from train_u_align_v2 import UAlignAdapter
    adapter = UAlignAdapter(encoder_dim=1024, llm_dim=2560, downsample=2).cuda().to(torch.bfloat16)
    adapter.load_state_dict(ck['adapter_state_dict'])
    head = torch.nn.Linear(2560, len(vocab)).cuda().float()
    head.load_state_dict(ck['ctc_head_state_dict'])
    adapter.eval()
    head.eval()

    evals = json.load(open(ROOT / 'ptt_eval_split.json'))['samples']
    h173 = [s for s in evals if s['subset'] == 'GT_human']

    logps, refs, ids = [], [], []
    with torch.no_grad():
        for k, s in enumerate(h173):
            audio = v5.load_audio(s['wav_path'])
            t = torch.from_numpy(audio).unsqueeze(0).cuda()
            feats, _ = extract_features(encoder, t, torch.tensor([t.shape[1]]).cuda())
            lp = F.log_softmax(head(adapter(feats.to(torch.bfloat16)).float().squeeze(0)), dim=-1)
            logps.append(lp.cpu())
            refs.append(s['ref'])
            ids.append(text_to_ids(s['ref'], vocab))
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(h173)}", flush=True)
    Path(args.out).parent.mkdir(exist_ok=True)
    torch.save({'logps': logps, 'refs': refs, 'ctc_ids': ids, 'vocab': vocab,
                'ckpt': args.ckpt}, args.out)
    print(f"saved {args.out} ({len(logps)} samples, vocab {len(vocab)})")


if __name__ == '__main__':
    main()
