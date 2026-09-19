"""v10: CTC front-end retraining on the expanded pool with a proper recipe.

Baseline (train_ctc_only.py) stopped at ep4 with CER still descending
(0.6079->0.4999), constant lr, no augmentation, greedy decode — 10 minutes
total. v10 changes (per TRAINING_METHODOLOGY.md §11.8):
  1. data: ptt_train_split_v10.json (11,924 = 5,159 strong_silver + 6,765
     teacher-accepted tiers, eval wavs excluded, leak-asserted)
  2. on-the-fly speed perturbation {0.9, 1.0, 1.1} (uniform 1/3)
  3. on-the-fly SpecAugment on XLSR features (2 time + 2 channel masks)
  4. linear warmup 5% + cosine decay schedule
  5. up to 30 epochs, h173 early stopping (patience 8), best-checkpoint by
     h173 CTC-CER(ns); silver1284 probed every 5 epochs
  6. vocab is saved inside the checkpoint (downstream draft/beam needs it)

Frozen: XLSR encoder. Trainable: adapter (warm start from ctc_only best) +
fresh CTC head (vocab may grow with the pool). No LLM, no CE.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))
import train_u_align_stage2_v5 as v5
from thai_ctc_units import build_vocab, text_to_ids, ids_to_text
from xlsr_thai import load_xlsr_thai, extract_features
from eval_cer_pretrained import cer_no_space

ROOT = Path('/data/workspace/asr-model-training/thai-understanding')
TRAIN_SPLIT = ROOT / 'ptt_train_split_v10.json'
EVAL_SPLIT = ROOT / 'ptt_eval_split.json'
CTC_ONLY_BEST = ROOT / 'u_align_ctc_only' / 'best' / 'ctc_only.pt'
OUT_DIR = ROOT / 'u_align_ctc_only_v10'
EVAL_HISTORY = OUT_DIR / 'eval_history.jsonl'
MAX_SAMPLES = 10 * 16000
SPEEDS = (0.9, 1.0, 1.1)


def speed_perturb(audio, speed):
    if speed == 1.0:
        return audio
    n = int(len(audio) / speed)
    out = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    return out.astype(np.float32)


def spec_augment(feat, n_time=2, time_w=20, n_chan=2, chan_w=20):
    T, C = feat.shape
    for _ in range(n_time):
        if T > time_w:
            w = torch.randint(1, time_w + 1, (1,)).item()
            t0 = torch.randint(0, T - w, (1,)).item()
            feat[t0:t0 + w, :] = 0
    for _ in range(n_chan):
        w = torch.randint(1, chan_w + 1, (1,)).item()
        c0 = torch.randint(0, max(1, C - w), (1,)).item()
        feat[:, c0:c0 + w] = 0
    return feat


class CTCDataset(Dataset):
    """All audio preloaded to RAM (~2.5GB) — collate does GPU work in the main
    process (num_workers=0 required: fork+CUDA), so disk I/O must not sit on
    the training critical path."""

    def __init__(self, samples):
        self.samples = samples
        t0 = time.time()
        self.cache = [v5.load_audio(s['wav_path']) for s in samples]
        print(f"  audio RAM cache: {len(self.cache)} clips "
              f"({sum(a.nbytes for a in self.cache)/1e9:.2f} GB, {time.time()-t0:.0f}s)", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {'audio': self.cache[idx], 'ref': s['ref'], 'ctc_ids': s['ctc_ids']}


class DurationBatchSampler(torch.utils.data.Sampler):
    """Duration-budget batching: sort by length, accumulate while
    total audio seconds <= budget (cap max_bs). Equalizes per-batch compute
    across long/short clips; composition reshuffled every epoch."""

    def __init__(self, durations, budget_sec=150, max_bs=48, seed=42):
        self.durations = durations
        self.budget = budget_sec
        self.max_bs = max_bs
        self.seed = seed
        self.epoch = 0
        self._n = None

    def __iter__(self):
        g = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.durations)))
        g.shuffle(idx)
        idx.sort(key=lambda i: self.durations[i])
        batches, cur, cur_sum = [], [], 0.0
        for i in idx:
            d = max(self.durations[i], 0.3)
            if cur and (cur_sum + d > self.budget or len(cur) >= self.max_bs):
                batches.append(cur)
                cur, cur_sum = [], 0.0
            cur.append(i)
            cur_sum += d
        if cur:
            batches.append(cur)
        g.shuffle(batches)
        self.epoch += 1
        return iter(batches)

    def __len__(self):
        return math.ceil(sum(self.durations) / self.budget)


def make_collate(encoder, train):
    """Batched feature extraction: ONE frozen-encoder forward per batch
    (pad waveforms to batch max; torchaudio masks group-norm via lengths).
    Verified numerically equivalent to per-sample extraction (max|diff|<=0.017)."""
    def collate(batch):
        audios = []
        for b in batch:
            audio = b['audio']
            if train:
                audio = speed_perturb(audio, random.choice(SPEEDS))
                if len(audio) > MAX_SAMPLES:
                    audio = audio[:MAX_SAMPLES]
                if len(audio) < 3200:
                    audio = np.pad(audio, (0, 3200 - len(audio)))
            audios.append(audio)
        maxlen = max(len(a) for a in audios)
        wav = torch.zeros(len(audios), maxlen, device='cuda')
        for i, a in enumerate(audios):
            wav[i, :len(a)] = torch.from_numpy(np.ascontiguousarray(a))
        with torch.no_grad():
            feats, lens = extract_features(encoder, wav,
                                           torch.tensor([len(a) for a in audios]).cuda())
        feats = feats.to(torch.bfloat16)
        if train:
            for i in range(feats.shape[0]):
                feats[i] = spec_augment(feats[i])
        targets = torch.tensor([t for b in batch for t in b['ctc_ids']], dtype=torch.long)
        tlens = torch.tensor([len(b['ctc_ids']) for b in batch], dtype=torch.long)
        # CNNSubsampler: Conv1d(k=2, s=2, no pad) -> out = (in-2)//2 + 1
        alens = torch.tensor([max(1, (int(l) - 2) // 2 + 1) for l in lens], dtype=torch.long)
        T_out = (feats.shape[1] - 2) // 2 + 1
        alens = alens.clamp(min=1, max=T_out)
        return {'feats': feats, 'feat_lens': alens,
                'targets': targets, 'target_lens': tlens, 'refs': [b['ref'] for b in batch]}
    return collate


@torch.no_grad()
def ctc_eval(split_items, tag, adapter, ctc_head, inv, encoder, batch=8):
    adapter.eval()
    ctc_head.eval()
    cers, blanks, zeroed, tot_frames = [], 0, 0, 0
    for i0 in range(0, len(split_items), batch):
        chunk = split_items[i0:i0 + batch]
        audios = [v5.load_audio(s['wav_path']) for s in chunk]
        maxlen = max(len(a) for a in audios)
        wav = torch.zeros(len(chunk), maxlen, device='cuda')
        for i, a in enumerate(audios):
            wav[i, :len(a)] = torch.from_numpy(np.ascontiguousarray(a))
        feats, lens = extract_features(encoder, wav,
                                       torch.tensor([len(a) for a in audios]).cuda())
        logp = F.log_softmax(ctc_head(adapter(feats.to(torch.bfloat16)).float()), dim=-1)
        for i, s in enumerate(chunk):
            T = min(int(lens[i]), logp.shape[1])
            if T < len(s['ctc_ids']):
                zeroed += 1
                continue
            am = logp[i, :T].argmax(-1).tolist()
            tot_frames += T
            blanks += sum(1 for x in am if x == 0)
            prev, out = -1, []
            for x in am:
                if x != prev and x != 0:
                    out.append(x)
                prev = x
            cers.append(cer_no_space(s['ref'], ids_to_text(out, inv)))
    adapter.train()
    ctc_head.train()
    cer = float(np.mean(cers)) if cers else float('nan')
    bfrac = blanks / max(1, tot_frames)
    print(f"  [{tag}] CTC-CER(ns)={cer:.4f}  blank_frac={bfrac:.2f}  T<S skipped={zeroed}/{len(split_items)}", flush=True)
    return cer, bfrac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_budget', type=float, default=150, help='max audio seconds per batch')
    ap.add_argument('--max_bs', type=int, default=48)
    ap.add_argument('--batch_size', type=int, default=16)  # used by ctc_eval chunks only
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup_frac', type=float, default=0.05)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--init', choices=['ctc_only', 'adapter_b'], default='ctc_only')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    OUT_DIR.mkdir(exist_ok=True)
    train_data = json.load(open(TRAIN_SPLIT))['samples']
    eval_full = json.load(open(EVAL_SPLIT))['samples']
    h173 = [s for s in eval_full if s['subset'] == 'GT_human']
    silver = [s for s in eval_full if s['subset'] == 'Silver_top20']
    if args.smoke:
        train_data = train_data[:300]
        h173 = h173[:30]
        silver = silver[:30]
        args.epochs = 2
    assert not ({Path(s['wav_path']).name for s in train_data} &
                {Path(s['wav_path']).name for s in h173 + silver}), "LEAK"

    vocab, _ = build_vocab([s['ref'] for s in train_data], min_freq=5)
    inv = {v: k for k, v in vocab.items()}
    for s in train_data + h173 + silver:
        s['ctc_ids'] = text_to_ids(s['ref'], vocab)
    print(f"train={len(train_data)} h173={len(h173)} silver={len(silver)} vocab={len(vocab)}", flush=True)

    encoder, _ = load_xlsr_thai(v5.XLSR_CKPT, device='cuda')
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    from train_u_align_v2 import UAlignAdapter
    adapter = UAlignAdapter(encoder_dim=1024, llm_dim=2560, downsample=2).cuda().to(torch.bfloat16)
    if args.init == 'ctc_only':
        ck = torch.load(CTC_ONLY_BEST, map_location='cpu', weights_only=False)
        adapter.load_state_dict(ck['adapter_state_dict'])
        print(f"adapter init: ctc_only best (ep{ck.get('epoch')}, CER={ck.get('cer'):.4f})", flush=True)
    else:
        ck = torch.load(v5.ADAPTER_B, map_location='cpu', weights_only=False)
        adapter.load_state_dict(ck['adapter_state_dict'])
        print("adapter init: Stage1 adapter_best", flush=True)
    ctc_head = torch.nn.Linear(2560, len(vocab)).cuda().float()

    params = list(adapter.parameters()) + list(ctc_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    ds = CTCDataset(train_data)
    durations = [len(a) / 16000.0 for a in ds.cache]
    batch_sampler = DurationBatchSampler(durations, budget_sec=args.batch_budget, max_bs=args.max_bs)
    total_steps = len(batch_sampler) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_frac))

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * step / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    loader = DataLoader(ds, batch_sampler=batch_sampler,
                        collate_fn=make_collate(encoder, True), num_workers=0)

    print("step-0 baseline:", flush=True)
    ctc_eval(h173, 'h173 step0', adapter, ctc_head, inv, encoder)
    best, best_ep, bad = 1e9, -1, 0
    gstep = 0
    for ep in range(args.epochs):
        adapter.train()
        ctc_head.train()
        t0, tot, nzero = time.time(), 0.0, 0
        for bi, b in enumerate(loader):
            for g in opt.param_groups:
                g['lr'] = lr_at(gstep)
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
            if (bi + 1) % 200 == 0:
                print(f"  ep{ep} batch {bi+1}/{len(loader)} ctc={tot/(bi+1):.4f} "
                      f"lr={lr_at(gstep):.2e} zeroed_batches={nzero} {time.time()-t0:.0f}s", flush=True)
        print(f"== epoch {ep} done, mean ctc={tot/len(loader):.4f}, {time.time()-t0:.0f}s", flush=True)

        cer, bfrac = ctc_eval(h173, f'h173 ep{ep}', adapter, ctc_head, inv, encoder)
        rec = {'epoch': ep, 'ctc_cer_ns': cer, 'blank_frac': round(bfrac, 3),
               'zeroed_batches': nzero, 'lr': round(lr_at(gstep), 8)}
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            scer, _ = ctc_eval(silver, f'silver ep{ep}', adapter, ctc_head, inv, encoder)
            rec['silver_ctc_cer_ns'] = scer
        with open(EVAL_HISTORY, 'a') as f:
            f.write(json.dumps(rec) + '\n')

        Path(OUT_DIR, 'last').mkdir(exist_ok=True)
        torch.save({'adapter_state_dict': adapter.state_dict(),
                    'ctc_head_state_dict': ctc_head.state_dict(),
                    'vocab': vocab, 'epoch': ep, 'cer': cer}, f'{OUT_DIR}/last/ctc_only_v10.pt')
        if cer < best:
            best, best_ep, bad = cer, ep, 0
            Path(OUT_DIR, 'best').mkdir(exist_ok=True)
            torch.save({'adapter_state_dict': adapter.state_dict(),
                        'ctc_head_state_dict': ctc_head.state_dict(),
                        'vocab': vocab, 'epoch': ep, 'cer': cer}, f'{OUT_DIR}/best/ctc_only_v10.pt')
            print(f"  saved best (ep{ep}, CER={cer:.4f})", flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at ep{ep} (no improvement for {args.patience} epochs)", flush=True)
                break
    print(f"DONE. best h173 CTC-CER={best:.4f} @ep{best_ep}")


if __name__ == '__main__':
    main()
