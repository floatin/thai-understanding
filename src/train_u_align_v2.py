"""
Stage 1: U-Align training v2 - full 12000 samples + cosine-DTW.

Per document Section 2.3:
  "损失函数 | InfoNCE（句级对比）+ 可选 cosine-DTW（帧级对齐）"

This version uses both losses (InfoNCE + cosine-DTW).

Configuration:
  - 12000 samples from Thai-SUP dev+test (IC/NER/SR, 4000 each)
  - 20 epochs
  - batch_size = 16
  - lr = 1e-4
  - Loss = InfoNCE + 0.5 * cosine-DTW (DTW weighted lower for stability)
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import random
import time
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(0, str(Path(__file__).parent))
from xlsr_thai import load_xlsr_thai, extract_features


class CNNSubsampler(nn.Module):
    def __init__(self, dim: int, factor: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=factor, stride=factor, groups=dim, bias=False)
        with torch.no_grad():
            w = torch.zeros(dim, 1, factor)
            for i in range(factor):
                w[:, 0, i] = 1.0 / factor
            self.conv.weight.copy_(w)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        return self.conv(x_t).transpose(1, 2)


class UAlignAdapter(nn.Module):
    """U-Align adapter: LN + CNN subsampler + MLP projection."""
    def __init__(self, encoder_dim: int, llm_dim: int, downsample: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(encoder_dim)
        self.subsampler = CNNSubsampler(dim=encoder_dim, factor=downsample)
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x):
        return self.proj(self.subsampler(self.norm(x)))


def cosine_dtw_loss(speech, text):
    """Functional cosine-DTW loss (paper Equation 1).

    C_ij = 1 - cos(h_i, e_j). DTW path computed with cumulative min over batch.
    Loss = DP[T_s-1, T_t-1] / (T_s + T_t).

    Args:
        speech: (B, T_s, D) adapter output (in same space as LLM text)
        text: (B, T_t, D) LLM token embeddings
    Returns:
        scalar loss (mean over batch)
    """
    s_norm = F.normalize(speech.float(), dim=-1)
    t_norm = F.normalize(text.float(), dim=-1)
    cost = 1.0 - torch.bmm(s_norm, t_norm.transpose(1, 2))  # (B, T_s, T_t)
    B, T_s, T_t = cost.shape
    # Row 0: cumulative sum from left
    row0 = torch.cumsum(cost[:, 0, :], dim=-1).unsqueeze(1)  # (B, 1, T_t)
    rows = [row0]
    for i in range(1, T_s):
        prev = rows[-1].squeeze(1)  # (B, T_t)
        # j=0: only from above (i-1, j)
        cells = [cost[:, i, 0:1] + prev[:, 0:1]]
        for j in range(1, T_t):
            from_above = prev[:, j:j+1]
            from_left = cells[-1]
            from_diag = prev[:, j-1:j]
            m = torch.minimum(torch.minimum(from_above, from_left), from_diag)
            cells.append(cost[:, i, j:j+1] + m)
        row = torch.cat(cells, dim=-1).unsqueeze(1)  # (B, 1, T_t)
        rows.append(row)
    dp = torch.cat(rows, dim=1)  # (B, T_s, T_t)
    return (dp[:, -1, -1] / float(T_s + T_t)).mean()


def info_nce_loss(speech_embeds, text_embeds, temperature=0.07):
    """Sentence-level InfoNCE (in-batch contrastive)."""
    s_pool = F.normalize(speech_embeds.float().mean(dim=1), dim=-1)
    t_pool = F.normalize(text_embeds.float().mean(dim=1), dim=-1)
    sim = torch.mm(s_pool, t_pool.t()) / temperature
    labels = torch.arange(speech_embeds.shape[0], device=speech_embeds.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2


def load_all_train_samples(base_dir, holdout=200, seed=42):
    """Load all 12000 samples from Thai-SUP dev+test (IC/NER/SR), hold out `holdout` for val."""
    rng = random.Random(seed)
    train_samples = []
    for task in ['IC', 'NER', 'SR']:
        for subset in ['dev', 'test']:
            shard = base_dir / task / subset / f'{subset}-00000.parquet'
            df = pq.read_table(str(shard), columns=['text', 'audio_flac']).to_pandas()
            for _, row in df.iterrows():
                audio, sr = sf.read(io.BytesIO(row['audio_flac']), dtype='float32')
                train_samples.append({'audio': torch.from_numpy(audio), 'text': str(row['text'])})
    rng.shuffle(train_samples)
    val_samples = train_samples[:holdout]
    train_samples = train_samples[holdout:]
    return train_samples, val_samples


def make_batch(samples, tokenizer, max_text_tokens=32, max_audio_seconds=8.0):
    valid = [s for s in samples if 5 < len(s['text']) < 200]
    if not valid:
        return None
    audios, texts = [], []
    for s in valid:
        a = s['audio']
        max_samples = int(max_audio_seconds * 16000)
        if len(a) > max_samples:
            a = a[:max_samples]
        if len(a) < 3200:
            a = torch.cat([a, torch.zeros(3200 - len(a))])
        audios.append(a)
        texts.append(s['text'])
    max_len = max(a.shape[0] for a in audios)
    audio_pad = torch.zeros(len(audios), max_len)
    for j, a in enumerate(audios):
        audio_pad[j, :a.shape[0]] = a
    enc = tokenizer(texts, return_tensors='pt', padding='max_length',
                    truncation=True, max_length=max_text_tokens)
    return {
        'audio': audio_pad,
        'input_ids': enc.input_ids,
        'attention_mask': enc.attention_mask,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder_ckpt', default='/data/workspace/asr-model-training/thai-understanding/XLSR-Thai/checkpoint_best.pt')
    ap.add_argument('--llm_path', default='/data/workspace/asr-model-training/thai-understanding/Qwen3-4B')
    ap.add_argument('--data_dir', default='/data/workspace/asr-model-training/thai-understanding/Thai-SUP')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--downsample', type=int, default=2)
    ap.add_argument('--temperature', type=float, default=0.07)
    ap.add_argument('--max_text_tokens', type=int, default=24)
    ap.add_argument('--max_audio_seconds', type=float, default=8.0,
                    help='Cap audio length to this many seconds (saves GPU memory)')
    ap.add_argument('--max_speech_frames', type=int, default=40,
                    help='Sub-sample speech to this many frames for DTW')
    ap.add_argument('--lambda_dtw', type=float, default=0.5, help='Weight for DTW loss')
    ap.add_argument('--lambda_infonce', type=float, default=1.0)
    ap.add_argument('--val_every', type=int, default=1, help='Validate every N epochs')
    ap.add_argument('--holdout', type=int, default=200)
    ap.add_argument('--embed_only', action='store_true')
    ap.add_argument('--output_dir', default='/data/workspace/asr-model-training/thai-understanding/u_align_adapter')
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Stage 1 U-Align Training v2 (InfoNCE + cosine-DTW) ===")
    print(f"Config: epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}")
    print(f"Loss: {args.lambda_infonce}*InfoNCE + {args.lambda_dtw}*cosine-DTW")
    print(f"DTW limits: max_speech_frames={args.max_speech_frames}, max_text_tokens={args.max_text_tokens}")

    # Encoder
    print(f"\n[1/5] Loading XLSR-Thai encoder...")
    encoder, cfg = load_xlsr_thai(args.encoder_ckpt, device='cuda')
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    encoder_dim = cfg['encoder_embed_dim']

    # LLM embed_tokens
    print(f"\n[2/5] Loading LLM embed_tokens from {args.llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(args.llm_path, dtype=torch.bfloat16, device_map='cpu')
    embed_tokens = llm.get_input_embeddings().to('cuda')
    llm_dim = embed_tokens.embedding_dim

    # Adapter
    print(f"\n[3/5] Building U-Align adapter ({encoder_dim} -> {llm_dim})...")
    adapter = UAlignAdapter(encoder_dim=encoder_dim, llm_dim=llm_dim, downsample=args.downsample).cuda().to(torch.bfloat16)
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"  adapter params: {n_params/1e6:.2f}M")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)

    # Data
    print(f"\n[4/5] Loading data from {args.data_dir}...")
    base_dir = Path(args.data_dir)
    train_samples, val_samples = load_all_train_samples(base_dir, holdout=args.holdout)
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Pre-compute validation audio lengths once
    val_audio_tensors = [s['audio'].unsqueeze(0) for s in val_samples]

    def validate(epoch):
        """Run retrieval on validation set."""
        adapter.eval()
        audio_embeds = []
        text_embeds = []
        with torch.no_grad():
            for s in val_samples:
                audio = s['audio']
                if len(audio) < 3200:
                    audio = torch.cat([audio, torch.zeros(3200 - len(audio))])
                audio_t = audio.unsqueeze(0).cuda()
                length = torch.tensor([audio_t.shape[1]]).cuda()
                feats, _ = extract_features(encoder, audio_t, length)
                speech_emb = adapter(feats.to(torch.bfloat16)).mean(dim=1)
                audio_embeds.append(speech_emb)
                ids = tokenizer(s['text'], return_tensors='pt').input_ids.cuda()
                text_emb = embed_tokens(ids).mean(dim=1)
                text_embeds.append(text_emb)
        a = F.normalize(torch.cat(audio_embeds, dim=0).float(), dim=-1)
        t = F.normalize(torch.cat(text_embeds, dim=0).float(), dim=-1)
        sim = torch.mm(a, t.t())
        r1 = float((sim.argmax(dim=-1) == torch.arange(len(val_samples)).cuda()).float().mean().item())
        r5 = float((sim.topk(5, dim=-1).indices == torch.arange(len(val_samples)).cuda().unsqueeze(-1)).any(dim=-1).float().mean().item())
        r10 = float((sim.topk(10, dim=-1).indices == torch.arange(len(val_samples)).cuda().unsqueeze(-1)).any(dim=-1).float().mean().item())
        diag = float(torch.diagonal(sim).mean().item())
        adapter.train()
        return {'r1': r1, 'r5': r5, 'r10': r10, 'diag': diag}

    # Train
    print(f"\n[5/5] Training {args.epochs} epochs on {len(train_samples)} samples...")
    adapter.train()
    log = []
    best_r10 = 0
    for epoch in range(args.epochs):
        random.shuffle(train_samples)
        total_loss = total_nce = total_dtw = 0.0
        n_batches = 0
        t0 = time.time()
        for i in range(0, len(train_samples), args.batch_size):
            batch = make_batch(train_samples[i:i+args.batch_size], tokenizer,
                               max_text_tokens=args.max_text_tokens,
                               max_audio_seconds=args.max_audio_seconds)
            if batch is None:
                continue
            audio = batch['audio'].cuda()
            input_ids = batch['input_ids'].cuda()
            attn_mask = batch['attention_mask'].cuda()
            lengths = torch.tensor([a.shape[0] for a in audio], dtype=torch.long).cuda()
            with torch.no_grad():
                feats, _ = extract_features(encoder, audio, lengths)
                # Sub-sample speech to max_speech_frames for DTW speed
                T_s = feats.shape[1]
                if T_s > args.max_speech_frames:
                    idx = torch.linspace(0, T_s-1, args.max_speech_frames).long()
                    feats_sub = feats[:, idx, :]
                else:
                    feats_sub = feats
                # Text embeddings, mask padding
                text_embeds = embed_tokens(input_ids).to(torch.bfloat16)
                text_embeds = text_embeds * attn_mask.unsqueeze(-1).to(torch.bfloat16)
            speech_embeds = adapter(feats_sub.to(torch.bfloat16))  # (B, T_s_sub//2, D)
            # Mask out padding in speech (for mean pooling in InfoNCE)
            feat_lens_sub = torch.tensor([f.shape[0] for f in feats_sub], dtype=torch.long).cuda()
            mask = torch.arange(speech_embeds.shape[1], device='cuda').unsqueeze(0) < feat_lens_sub.unsqueeze(1)
            speech_masked = speech_embeds * mask.unsqueeze(-1).to(torch.bfloat16)

            nce = info_nce_loss(speech_masked, text_embeds, temperature=args.temperature)
            dtw = cosine_dtw_loss(speech_embeds, text_embeds)
            loss = args.lambda_infonce * nce + args.lambda_dtw * dtw

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_nce += nce.item()
            total_dtw += dtw.item()
            n_batches += 1
            if n_batches % 10 == 0:
                elapsed = time.time() - t0
                rate = n_batches / elapsed
                eta = (len(train_samples)//args.batch_size - n_batches) / max(rate, 0.001)
                print(f"    [Epoch {epoch+1}] batch {n_batches}/{len(train_samples)//args.batch_size}, "
                      f"loss={total_loss/n_batches:.3f} (nce={total_nce/n_batches:.3f}, dtw={total_dtw/n_batches:.3f}), "
                      f"{elapsed:.0f}s, ETA {eta:.0f}s", flush=True)
        epoch_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # Validation
        if (epoch+1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_metrics = validate(epoch)
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={epoch_loss:.4f} "
                  f"(nce={total_nce/n_batches:.3f}, dtw={total_dtw/n_batches:.3f}) "
                  f"val R@1={val_metrics['r1']:.2%}, R@5={val_metrics['r5']:.2%}, R@10={val_metrics['r10']:.2%}  "
                  f"({elapsed:.0f}s)", flush=True)
            log.append({'epoch': epoch+1, 'loss': epoch_loss,
                        'nce': total_nce/n_batches, 'dtw': total_dtw/n_batches,
                        **val_metrics})
            if val_metrics['r10'] > best_r10:
                best_r10 = val_metrics['r10']
                # Save best checkpoint
                torch.save({
                    'adapter_state_dict': adapter.state_dict(),
                    'encoder_dim': encoder_dim,
                    'llm_dim': llm_dim,
                    'downsample': args.downsample,
                    'epoch': epoch+1,
                    'best_r10': best_r10,
                    'args': vars(args),
                }, output_dir / "adapter_best.pt")
                print(f"    Saved best adapter (R@10={best_r10:.2%})", flush=True)
        else:
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={epoch_loss:.4f}  ({elapsed:.0f}s)", flush=True)
            log.append({'epoch': epoch+1, 'loss': epoch_loss,
                        'nce': total_nce/n_batches, 'dtw': total_dtw/n_batches})

    # Save final
    torch.save({
        'adapter_state_dict': adapter.state_dict(),
        'encoder_dim': encoder_dim,
        'llm_dim': llm_dim,
        'downsample': args.downsample,
        'final_epoch': args.epochs,
        'log': log,
        'args': vars(args),
    }, output_dir / "adapter_final.pt")
    (output_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nFinal adapter saved to {output_dir}/adapter_final.pt")
    print(f"Best adapter (R@10={best_r10:.2%}) saved to {output_dir}/adapter_best.pt")


if __name__ == "__main__":
    main()
