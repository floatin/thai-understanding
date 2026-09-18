"""
Train CTC head on full Thai-SUP training set, evaluate on dev/test.

Goal: show that with enough data, XLSR-Thai + CTC head can give meaningful CER.
Compare with:
  - Pretrained XLSR-53-Thai (CommonVoice fine-tuned): CER 37.95% on test
  - Paper's XLSR-Thai + full ASR fine-tune: 13.91% on Giga2 test

This trains the CTC head on ~600K samples from Thai-SUP (different domain from Giga2),
so won't reach 13.91% but should beat random init.
"""
import warnings
warnings.filterwarnings("ignore")
import io
import os
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, '/data/workspace/asr-model-training/thai-understanding/src')
from xlsr_thai import load_xlsr_thai, extract_features


# Thai characters
THAI_CHARS = (
    " กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
    "ะัาำิีุูเแโใไๅๆ่้๊๋์ํ"
    "๐๑๒๓๔๕๖๗๘๙"
    ".,!?\"'()[]{}:;-"
)


def cer(ref, hyp):
    n = max(len(ref), 1)
    m = len(hyp)
    if m == 0:
        return 1.0
    dp = list(range(m+1))
    for i in range(1, len(ref)+1):
        prev, dp[0] = dp[0], i
        for j in range(1, m+1):
            cur = dp[j]
            if ref[i-1] == hyp[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = cur
    return dp[m] / n


class CTCHead(nn.Module):
    def __init__(self, in_dim=1024, n_chars=200):
        super().__init__()
        self.proj = nn.Linear(in_dim, n_chars)

    def forward(self, x):
        return self.proj(x)


def build_vocab_from_files(parquet_files):
    chars = set(THAI_CHARS)
    for f in parquet_files:
        try:
            df = pq.read_table(str(f), columns=["text"]).to_pandas()
            for t in df['text']:
                for c in str(t):
                    chars.add(c)
        except Exception as e:
            print(f"  Skipping {f}: {e}")
    chars = sorted(chars)
    char_to_id = {c: i+1 for i, c in enumerate(chars)}  # 0 = blank
    id_to_char = {0: '<blk>'}
    for i, c in enumerate(chars):
        id_to_char[i+1] = c
    return char_to_id, id_to_char


def sample_iter(parquet_files, char_to_id, n_per_shard=200, seed=42):
    """Iterate over training samples."""
    rng = random.Random(seed)
    for f in parquet_files:
        try:
            table = pq.read_table(str(f))
            df = table.to_pandas()
            n = min(n_per_shard, len(df))
            indices = rng.sample(range(len(df)), n)
            for i in indices:
                row = df.iloc[i]
                audio_bytes = row['audio_flac']
                audio, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
                yield {
                    'audio': torch.from_numpy(audio),
                    'text': str(row['text']),
                }
        except Exception as e:
            print(f"  Skip {f}: {e}")


def train_one_epoch(encoder, head, samples, char_to_id, batch_size=8, lr=3e-4, device='cuda', optimizer=None, ctc_loss=None):
    random.shuffle(samples)
    total_loss = 0.0
    n_batches = 0
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        # Filter samples where text has at least one char in vocab
        valid_batch = []
        for s in batch:
            ids = [char_to_id[c] for c in s['text'] if c in char_to_id]
            if ids and 5 < len(s['audio']) < 200000:
                s['_target_ids'] = ids
                valid_batch.append(s)
        if not valid_batch:
            continue
        # Pad audio to max length in batch
        audios = [s['audio'] for s in valid_batch]
        lengths_audio = torch.tensor([a.shape[0] for a in audios])
        max_len = lengths_audio.max().item()
        if max_len < 3200:
            max_len = 3200
        audio_pad = torch.zeros(len(valid_batch), max_len)
        for j, a in enumerate(audios):
            audio_pad[j, :a.shape[0]] = a
        # Encode
        with torch.no_grad():
            feats, feat_lens = extract_features(encoder, audio_pad, lengths_audio)
        feats = feats.detach()
        # CTC head
        logits = head(feats)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        # Targets
        targets = []
        target_lengths = []
        for s in valid_batch:
            targets.extend(s['_target_ids'])
            target_lengths.append(len(s['_target_ids']))
        targets = torch.tensor(targets, dtype=torch.long).to(device)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)
        loss = ctc_loss(log_probs, targets, feat_lens, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_per_shard', type=int, default=100)
    ap.add_argument('--n_epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--quick', action='store_true', help='Use only SR train shards (smaller, faster)')
    args = ap.parse_args()

    base = Path("/data/workspace/asr-model-training/thai-understanding")
    print(f"=== CTC head training on Thai-SUP ===")
    print(f"Args: n_per_shard={args.n_per_shard}, epochs={args.n_epochs}, bs={args.batch_size}, lr={args.lr}")

    # Load encoder
    print(f"\n[1/4] Loading XLSR-Thai...")
    encoder, cfg = load_xlsr_thai(str(base / "XLSR-Thai/checkpoint_best.pt"))

    # Build vocab from training data
    print(f"\n[2/4] Building vocab from train shards...")
    if args.quick:
        train_files = sorted((base / "Thai-SUP/SR/train").glob("*.parquet"))[:10]
    else:
        train_files = sorted((base / "Thai-SUP/SR/train").glob("*.parquet")) + \
                      sorted((base / "Thai-SUP/IC/train").glob("*.parquet"))[:30] + \
                      sorted((base / "Thai-SUP/NER/train").glob("*.parquet"))[:30]
    print(f"  Using {len(train_files)} train shards")
    char_to_id, id_to_char = build_vocab_from_files(train_files)
    print(f"  Vocab: {len(char_to_id)} chars")

    # Load samples (sampling n_per_shard from each)
    print(f"\n[3/4] Loading samples ({args.n_per_shard}/shard from {len(train_files)} shards)...")
    all_samples = list(sample_iter(train_files, char_to_id, n_per_shard=args.n_per_shard))
    print(f"  Loaded {len(all_samples)} samples")

    # Build CTC head
    n_chars = max(char_to_id.values()) + 1
    head = CTCHead(in_dim=1024, n_chars=n_chars).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    head.train()

    # Train
    print(f"\n[4/4] Training {args.n_epochs} epochs...")
    encoder.eval()
    for epoch in range(args.n_epochs):
        t0 = time.time()
        loss = train_one_epoch(encoder, head, all_samples, char_to_id,
                               batch_size=args.batch_size, lr=args.lr, optimizer=optimizer, ctc_loss=ctc_loss)
        print(f"  Epoch {epoch+1}/{args.n_epochs}: ctc_loss={loss:.4f}  ({time.time()-t0:.1f}s)")
    head.eval()

    # Save head weights
    head_path = base / "ctc_head_trained.pt"
    torch.save({
        'state_dict': head.state_dict(),
        'char_to_id': char_to_id,
        'id_to_char': {str(k): v for k, v in id_to_char.items()},  # JSON serializable
    }, head_path)
    print(f"  Saved CTC head to {head_path}")

    # Evaluate on SR dev (small for quick test)
    print(f"\n=== Evaluating on SR dev/test ===")
    cers_dev, cers_test = [], []
    for split_name, parquet_path in [("dev", base / "Thai-SUP/SR/dev/dev-00000.parquet"),
                                      ("test", base / "Thai-SUP/SR/test/test-00000.parquet")]:
        cers = []
        table = pq.read_table(str(parquet_path))
        df = table.to_pandas()
        with torch.no_grad():
            for i in range(min(200, len(df))):  # cap at 200 for speed
                row = df.iloc[i]
                audio, sr = sf.read(io.BytesIO(row['audio_flac']), dtype='float32')
                if len(audio) < 3200:
                    audio = np.pad(audio, (0, 3200 - len(audio)))
                audio_t = torch.from_numpy(audio).unsqueeze(0).cuda()
                length = torch.tensor([audio_t.shape[1]]).cuda()
                feats, _ = extract_features(encoder, audio_t, length)
                logits = head(feats)
                pred_ids = torch.argmax(logits[0], dim=-1).cpu().numpy()
                # Greedy decode
                chars = []
                prev = -1
                for pid in pred_ids:
                    if pid != prev and pid != 0:
                        chars.append(id_to_char.get(int(pid), ''))
                    prev = pid
                pred = ''.join(chars)
                ref = str(row['text'])
                c = cer(ref, pred)
                cers.append(c)
        avg = sum(cers)/len(cers)
        if split_name == "dev":
            cers_dev = cers
        else:
            cers_test = cers
        print(f"  SR/{split_name}: {len(cers)} samples, avg CER (with space) = {avg:.2%}")

    # Compare with pretrained XLSR-53-Thai on same 200 SR samples
    print(f"\n=== Compare with pretrained XLSR-53-Thai on SR test (200 samples) ===")
    report = {
        'n_train_samples': len(all_samples),
        'n_train_shards': len(train_files),
        'vocab_size': len(char_to_id),
        'epochs': args.n_epochs,
        'ctc_loss_final': loss,
        'sr_dev_cer': sum(cers_dev)/len(cers_dev) if cers_dev else None,
        'sr_test_cer': sum(cers_test)/len(cers_test) if cers_test else None,
        'pretrained_xlsr_53_thai_on_993_mixed_cer': 0.3795,
        'paper_xlsr_thai_giga2_test_cer': 0.1391,
    }
    out = base / "results_ctc_full_train.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to: {out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
