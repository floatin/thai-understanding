"""v10b: unfreeze the top-N XLSR encoder layers for CTC training.

Motivation (2026-09-19): v10 full recipe (expanded pool + augmentation +
schedule, 18 epochs) plateaued at greedy CER ~0.515 — statistically
indistinguishable from the 10-minute baseline 0.4999. Two very different
recipes landing at the same place points at the FROZEN encoder as the binding
constraint, not optimization. Standard next step: unfreeze top transformer
layers with a small lr.

Changes vs v10 (nothing else):
  - top-N encoder layers requires_grad=True, lr group enc_lr (default 1e-5)
  - adapter + CTC head keep lr 1e-4 (warm start from v10 best)
  - TF32 matmul enabled for the fp32 encoder (2x GEMM throughput)
  - best checkpoint saves the FULL encoder state (downstream drafts need it)
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
import train_ctc_only_v10 as v10
import train_u_align_stage2_v5 as v5
from thai_ctc_units import text_to_ids
from xlsr_thai import load_xlsr_thai

ROOT = Path('/data/workspace/asr-model-training/thai-understanding')
V10_BEST = ROOT / 'u_align_ctc_only_v10' / 'best' / 'ctc_only_v10.pt'
OUT_DIR = ROOT / 'u_align_ctc_only_v10b'
EVAL_HISTORY = OUT_DIR / 'eval_history.jsonl'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--unfreeze_top', type=int, default=12)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--enc_lr', type=float, default=1e-5)
    ap.add_argument('--warmup_frac', type=float, default=0.05)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--batch_budget', type=float, default=100.0)
    ap.add_argument('--max_bs', type=int, default=32)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    OUT_DIR.mkdir(exist_ok=True)

    train_data = json.load(open(v10.TRAIN_SPLIT))['samples']
    eval_full = json.load(open(v10.EVAL_SPLIT))['samples']
    h173 = [s for s in eval_full if s['subset'] == 'GT_human']
    silver = [s for s in eval_full if s['subset'] == 'Silver_top20']
    if args.smoke:
        train_data = train_data[:300]
        h173 = h173[:30]
        silver = silver[:30]
        args.epochs = 1

    ck = torch.load(V10_BEST, map_location='cpu', weights_only=False)
    vocab = ck['vocab']
    inv = {v: k for k, v in vocab.items()}
    for s in train_data + h173 + silver:
        s['ctc_ids'] = text_to_ids(s['ref'], vocab)
    print(f"train={len(train_data)} vocab={len(vocab)} (from v10 best ckpt)", flush=True)

    encoder, _ = load_xlsr_thai(v5.XLSR_CKPT, device='cuda')
    encoder.eval()
    n_layers = len(encoder.encoder.transformer.layers)
    assert args.unfreeze_top <= n_layers
    enc_params = []
    for i, layer in enumerate(encoder.encoder.transformer.layers):
        top = i >= n_layers - args.unfreeze_top
        for p in layer.parameters():
            p.requires_grad = top
            if top:
                enc_params.append(p)
    print(f"unfroze encoder layers {n_layers-args.unfreeze_top}..{n_layers-1} "
          f"({sum(p.numel() for p in enc_params)/1e6:.0f}M params)", flush=True)

    from train_u_align_v2 import UAlignAdapter
    adapter = UAlignAdapter(encoder_dim=1024, llm_dim=2560, downsample=2).cuda().to(torch.bfloat16)
    adapter.load_state_dict(ck['adapter_state_dict'])
    ctc_head = torch.nn.Linear(2560, len(vocab)).cuda().float()
    ctc_head.load_state_dict(ck['ctc_head_state_dict'])

    adapter_params = list(adapter.parameters()) + list(ctc_head.parameters())
    opt = torch.optim.AdamW([
        {'params': adapter_params, 'base_lr': args.lr},
        {'params': enc_params, 'base_lr': args.enc_lr},
    ], lr=args.lr, weight_decay=0.01)

    ds = v10.CTCDataset(train_data)
    durations = [len(a) / 16000.0 for a in ds.cache]
    batch_sampler = v10.DurationBatchSampler(durations, budget_sec=args.batch_budget, max_bs=args.max_bs)
    total_steps = len(batch_sampler) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_frac))

    def mult(step):
        if step < warmup_steps:
            return step / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    loader = torch.utils.data.DataLoader(ds, batch_sampler=batch_sampler,
                                         collate_fn=v10.make_collate(encoder, True), num_workers=0)
    params = adapter_params + enc_params

    print("step-0 baseline (v10 best weights):", flush=True)
    v10.ctc_eval(h173, 'h173 step0', adapter, ctc_head, inv, encoder)
    best, best_ep, bad, gstep = 1e9, -1, 0, 0
    for ep in range(args.epochs):
        adapter.train()
        ctc_head.train()
        for layer in encoder.encoder.transformer.layers[n_layers - args.unfreeze_top:]:
            layer.train()
        t0, tot, nzero = time.time(), 0.0, 0
        for bi, b in enumerate(loader):
            m = mult(gstep)
            for g in opt.param_groups:
                g['lr'] = g['base_lr'] * m
            logp = F.log_softmax(ctc_head(adapter(b['feats']).float()), dim=-1).transpose(0, 1)
            l = F.ctc_loss(logp, b['targets'], b['feat_lens'].clamp(min=1), b['target_lens'],
                           blank=0, zero_infinity=True)
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            gstep += 1
            tot += l.item()
            if l.item() == 0.0:
                nzero += 1
            if (bi + 1) % 100 == 0:
                print(f"  ep{ep} batch {bi+1}/{len(loader)} ctc={tot/(bi+1):.4f} "
                      f"mult={m:.3f} {time.time()-t0:.0f}s", flush=True)
        print(f"== epoch {ep} done, mean ctc={tot/len(loader):.4f}, {time.time()-t0:.0f}s", flush=True)

        cer, bfrac = v10.ctc_eval(h173, f'h173 ep{ep}', adapter, ctc_head, inv, encoder)
        rec = {'epoch': ep, 'ctc_cer_ns': cer, 'blank_frac': round(bfrac, 3), 'zeroed_batches': nzero}
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            scer, _ = v10.ctc_eval(silver, f'silver ep{ep}', adapter, ctc_head, inv, encoder)
            rec['silver_ctc_cer_ns'] = scer
        with open(EVAL_HISTORY, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        if cer < best:
            best, best_ep, bad = cer, ep, 0
            Path(OUT_DIR, 'best').mkdir(exist_ok=True)
            torch.save({'adapter_state_dict': adapter.state_dict(),
                        'ctc_head_state_dict': ctc_head.state_dict(),
                        'encoder_state_dict': encoder.state_dict(),
                        'vocab': vocab, 'epoch': ep, 'cer': cer},
                       f'{OUT_DIR}/best/ctc_only_v10b.pt')
            print(f"  saved best (ep{ep}, CER={cer:.4f})", flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at ep{ep}", flush=True)
                break
    print(f"DONE. best h173 CTC-CER={best:.4f} @ep{best_ep}")


if __name__ == '__main__':
    main()
