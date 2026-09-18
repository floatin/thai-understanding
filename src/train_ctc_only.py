"""Stage 2a' (CTC-only warmup): train adapter + CTC head as a standalone ASR
front-end — NO LLM, NO answer-CE. Removes the objective competition that
killed v5's CTC head (blank collapse under joint loss).

Success criterion: h173 CTC-CER(ns) approaching the reference ZikXewen-XLSR53
CTC level on this domain (~0.65). If the adapter cannot carry frame-level
content even with a clean CTC objective, the adapter architecture itself is
the wall — and the whole U-Align route needs re-examination.
"""
import warnings
warnings.filterwarnings("ignore")
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import train_u_align_stage2_v5 as v5
from thai_ctc_units import build_vocab, text_to_ids, ids_to_text
from xlsr_thai import load_xlsr_thai, extract_features
from eval_cer_pretrained import cer_no_space

OUT_DIR = '/data/workspace/asr-model-training/thai-understanding/u_align_ctc_only'
EVAL_HISTORY = f'{OUT_DIR}/eval_history.jsonl'


class CTCDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {'wav_path': s['wav_path'], 'ref': s['ref'], 'ctc_ids': s['ctc_ids']}


def collate(batch):
    audios = [v5.load_audio(b['wav_path']) for b in batch]
    max_len = max(len(a) for a in audios)
    feats_list, lens = [], []
    for a in audios:
        t = torch.from_numpy(a).unsqueeze(0).cuda()
        with torch.no_grad():
            f, _ = extract_features(encoder, t, torch.tensor([t.shape[1]]).cuda())
        feats_list.append(f.squeeze(0))
        lens.append(f.shape[1])
    max_f = max(f.shape[0] for f in feats_list)
    feats_b = torch.zeros(len(batch), max_f, 1024, dtype=torch.bfloat16, device='cuda')
    for i, f in enumerate(feats_list):
        feats_b[i, :f.shape[0]] = f
    targets = torch.tensor([t for b in batch for t in b['ctc_ids']], dtype=torch.long)
    tlens = torch.tensor([len(b['ctc_ids']) for b in batch], dtype=torch.long)
    # CNNSubsampler: Conv1d(k=2, s=2, no pad) -> out = (in-2)//2 + 1
    alens = torch.tensor([max(1, (int(l) - 2) // 2 + 1) for l in lens], dtype=torch.long)
    T_out = (max_f - 2) // 2 + 1
    alens = alens.clamp(min=1, max=T_out)
    return {'feats': feats_b, 'feat_lens': alens,
            'targets': targets, 'target_lens': tlens, 'refs': [b['ref'] for b in batch]}


def ctc_eval(split_items, tag):
    adapter.eval()
    cers, blanks, zeroed = [], 0, 0
    tot_frames = 0
    with torch.no_grad():
        for s in split_items:
            audio = v5.load_audio(s['wav_path'])
            t = torch.from_numpy(audio).unsqueeze(0).cuda()
            feats, _ = extract_features(encoder, t, torch.tensor([t.shape[1]]).cuda())
            logp = F.log_softmax(ctc_head(adapter(feats.to(torch.bfloat16)).float().squeeze(0)), dim=-1)
            T = logp.shape[0]
            if T < len(s['ctc_ids']):
                zeroed += 1
                continue
            am = logp.argmax(-1).tolist()
            tot_frames += T
            blanks += sum(1 for i in am if i == 0)
            prev, out = -1, []
            for i in am:
                if i != prev and i != 0:
                    out.append(i)
                prev = i
            cers.append(cer_no_space(s['ref'], ids_to_text(out, inv)))
    adapter.train()
    cer = float(np.mean(cers)) if cers else float('nan')
    bfrac = blanks / max(1, tot_frames)
    print(f"  [{tag}] CTC-CER(ns)={cer:.4f}  blank_frac={bfrac:.2f}  T<S skipped={zeroed}/{len(split_items)}", flush=True)
    return cer, bfrac


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    args = ap.parse_args()

    Path(OUT_DIR).mkdir(exist_ok=True)
    train_data = json.load(open(v5.TRAIN_SPLIT))['samples']
    eval_full = json.load(open(v5.EVAL_SPLIT))['samples']
    h173 = [s for s in eval_full if s['subset'] == 'GT_human']
    assert not ({Path(s['wav_path']).name for s in train_data} &
                {Path(s['wav_path']).name for s in h173}), "LEAK"

    vocab, _ = build_vocab([s['ref'] for s in train_data], min_freq=5)
    inv = {v: k for k, v in vocab.items()}
    for s in train_data + h173:
        s['ctc_ids'] = text_to_ids(s['ref'], vocab)
    print(f"vocab={len(vocab)}")

    encoder, _ = load_xlsr_thai(v5.XLSR_CKPT, device='cuda')
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    from train_u_align_v2 import UAlignAdapter
    global adapter, ctc_head
    adapter = UAlignAdapter(encoder_dim=1024, llm_dim=2560, downsample=2).cuda().to(torch.bfloat16)
    ck = torch.load(v5.ADAPTER_B, map_location='cpu', weights_only=False)
    adapter.load_state_dict(ck['adapter_state_dict'])
    ctc_head = torch.nn.Linear(2560, len(vocab)).cuda().float()

    params = list(adapter.parameters()) + list(ctc_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    loader = DataLoader(CTCDataset(train_data), batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0)

    print(f"step-0 baseline:", flush=True)
    ctc_eval(h173[:173], 'h173 step0')
    best = 1e9
    for ep in range(args.epochs):
        adapter.train()
        ctc_head.train()
        t0, tot, nzero = time.time(), 0.0, 0
        for bi, b in enumerate(loader):
            logp = F.log_softmax(ctc_head(adapter(b['feats']).float()), dim=-1).transpose(0, 1)
            l = F.ctc_loss(logp, b['targets'], b['feat_lens'].clamp(min=1), b['target_lens'],
                           blank=0, zero_infinity=True)
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += l.item()
            if l.item() == 0.0:
                nzero += 1
            if (bi + 1) % 100 == 0:
                print(f"  ep{ep} batch {bi+1}/{len(loader)} ctc={tot/(bi+1):.4f} zeroed_batches={nzero} {time.time()-t0:.0f}s", flush=True)
        print(f"== epoch {ep} done, mean ctc={tot/len(loader):.4f}, {time.time()-t0:.0f}s", flush=True)
        cer, bfrac = ctc_eval(h173, f'h173 ep{ep}')
        with open(EVAL_HISTORY, 'a') as f:
            f.write(json.dumps({'epoch': ep, 'ctc_cer_ns': cer, 'blank_frac': round(bfrac, 3)}) + '\n')
        if cer < best:
            best = cer
            Path(OUT_DIR, 'best').mkdir(exist_ok=True)
            torch.save({'adapter_state_dict': adapter.state_dict(),
                        'ctc_head_state_dict': ctc_head.state_dict(),
                        'epoch': ep, 'cer': cer}, f'{OUT_DIR}/best/ctc_only.pt')
            print(f"  saved best (ep{ep}, CER={cer:.4f})", flush=True)
    print(f"DONE. best h173 CTC-CER={best:.4f}")
